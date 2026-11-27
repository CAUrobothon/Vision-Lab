"""/api/datasets/{id}/gt, /api/gt/{id} (ground-truth sets and their links) and /api/media/{id}/gt."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool

from lab import gtsets
from lab.api.media import check_dataset
from lab.config import MAX_GT_BYTES, DataPaths
from lab.db import connect
from lab.deps import Conn, Paths
from lab.schemas import Dataset, GtLinksConfirm, GtOrganize, GtSetDetail, GtSetInfo, GtSetPatch, MediaGtLink, Ok
from lab.storage import guard_disk
from lab.upload import Upload, receive_upload

router = APIRouter(prefix="/api", tags=["ground truth"])


def _import(paths: DataPaths, dataset_id: int, up: Upload) -> GtSetInfo:
    conn = connect(paths.db)
    try:
        return gtsets.import_set(conn, paths, dataset_id, up)
    finally:
        conn.close()


@router.post("/datasets/{dataset_id}/gt", openapi_extra={"requestBody": {"content": {"multipart/form-data": {"schema": {
    "type": "object", "required": ["file"], "properties": {
        "file": {"type": "string", "format": "binary", "description": "ground-truth JSON (docs/SCORING.md §1)"},
        "name": {"type": "string", "description": "default: the file name"},
    }}}}}})
async def upload_gt(dataset_id: int, request: Request) -> GtSetInfo:
    """Multipart ``file`` (the JSON, kept byte for byte) and optional ``name``. 400 with the parser's
    message for a file it refuses, 409 when this dataset has the same file already. The answer
    carries the link report (images linked by sha256 right away)."""
    paths: DataPaths = request.app.state.paths
    guard_disk(paths)
    await run_in_threadpool(check_dataset, paths, dataset_id)  # 404 before the body is read
    up = await receive_upload(request, paths.tmp, max_bytes=MAX_GT_BYTES)
    return await run_in_threadpool(_import, paths, dataset_id, up)


@router.get("/datasets/{dataset_id}/gt")
def list_gt(dataset_id: int, conn: Conn) -> list[GtSetInfo]:
    """The dataset's ground-truth sets, newest first, each with its link counts."""
    return gtsets.list_sets(conn, dataset_id)


@router.get("/gt/{gt_set_id}")
def get_gt(gt_set_id: int, conn: Conn) -> GtSetDetail:
    """Links, unlinked GT images with their candidates (same file name), images without GT."""
    return gtsets.detail(conn, gt_set_id)


@router.patch("/gt/{gt_set_id}")
def patch_gt(gt_set_id: int, patch: GtSetPatch, conn: Conn) -> GtSetInfo:
    return gtsets.rename(conn, gt_set_id, patch.name)


@router.delete("/gt/{gt_set_id}")
def delete_gt(gt_set_id: int, conn: Conn, paths: Paths) -> Ok:
    """The set, its confirmed links and its file; media and runs stay."""
    gtsets.delete(conn, paths, gt_set_id)
    return Ok()


@router.post("/gt/{gt_set_id}/links")
def confirm_links(gt_set_id: int, req: GtLinksConfirm, conn: Conn) -> GtSetDetail:
    """Confirm links (normally candidates), all or nothing."""
    return gtsets.confirm_links(conn, gt_set_id, req.links)


@router.delete("/gt/{gt_set_id}/links/{gt_index}")
def remove_link(gt_set_id: int, gt_index: int, conn: Conn) -> GtSetDetail:
    """Removes a confirmed link; a sha256 link cannot be removed (400)."""
    return gtsets.remove_link(conn, gt_set_id, gt_index)


@router.post("/gt/{gt_set_id}/organize")
def organize(gt_set_id: int, req: GtOrganize, conn: Conn) -> Dataset:
    """Moves every linked media into the folder "<root>/<variant>" (made if missing)."""
    return gtsets.organize(conn, gt_set_id, req.root)


@router.get("/media/{media_id}/gt")
def media_gt(media_id: int, conn: Conn) -> list[MediaGtLink]:
    """The ground-truth sets (of any dataset) that link this media, newest first."""
    return gtsets.media_links(conn, media_id)

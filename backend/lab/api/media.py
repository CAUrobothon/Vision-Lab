"""/api/media: upload, list, edit, delete, and the files behind a media item."""

from __future__ import annotations

import sqlite3
from typing import Annotated, Any

import cv2
from fastapi import APIRouter, Query, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from lab import frames
from lab import datasets as datasets_mod
from lab import media as media_mod
from lab.config import MAX_MEDIA_BYTES, DataPaths
from lab.db import connect, dumps
from lab.deps import Conn, Paths
from lab.errors import LabError
from lab.schemas import Media, MediaKind, MediaPatch, MediaUploadResult, Ok
from lab.storage import guard_disk
from lab.upload import Upload, receive_upload

router = APIRouter(prefix="/api/media", tags=["media"])

# Frames and thumbnails of a media file never change, but ids restart at 1 in a new data dir:
# only a URL that carries the file's version (?v=, see media.url_version) may be cached.
IMMUTABLE = {"Cache-Control": "private, max-age=86400"}


def _cache_headers(row: sqlite3.Row, v: str | None) -> dict[str, str] | None:
    return IMMUTABLE if v is not None and v == media_mod.url_version(row) else None


@router.get("")
def list_media(
    conn: Conn,
    kind: MediaKind | None = None,
    q: Annotated[str | None, Query(max_length=200)] = None,
    starred: bool | None = None,
    tag: str | None = None,
    dataset_id: int | None = None,
    folder: Annotated[str | None, Query(max_length=1000)] = None,
) -> list[Media]:
    """``dataset_id`` lists one dataset, ``folder`` one of its folders with everything below it;
    each media then carries its ``folder`` and the list is ordered by folder."""
    join, folder_col, order, args = "", "", "m.id DESC", []
    if dataset_id is not None:
        cond, fargs = datasets_mod.in_folder("dm.folder", datasets_mod.clean_folder(folder))
        join = f" JOIN dataset_media dm ON dm.media_id = m.id AND dm.dataset_id = ? AND {cond}"
        folder_col, order = ", dm.folder AS ds_folder", "dm.folder, m.id DESC"
        args += [dataset_id, *fargs]
    elif folder:
        raise LabError(400, "폴더는 데이터셋과 함께 고르세요 (dataset_id)")
    where: list[str] = []
    if kind:
        where.append("m.kind = ?")
        args.append(kind)
    if q:
        where.append("(m.name LIKE ? OR m.note LIKE ?)")
        args += [f"%{q}%", f"%{q}%"]
    if starred is not None:
        where.append("m.starred = ?")
        args.append(int(starred))
    if tag:
        where.append("EXISTS (SELECT 1 FROM json_each(m.tags) WHERE value = ?)")
        args.append(tag)
    clause = (" WHERE " + " AND ".join(where)) if where else ""
    rows = conn.execute(f"SELECT m.*, {media_mod.EXTRA_COLUMNS}{folder_col} FROM media m{join}{clause} ORDER BY {order}", args)
    return [media_mod.to_api(r) for r in rows]


def check_dataset(paths: DataPaths, dataset_id: int) -> None:
    """404 for an unknown dataset, on a connection of its own (for async routes, before an upload is read)."""
    conn = connect(paths.db)
    try:
        datasets_mod.get_row(conn, dataset_id)
    finally:
        conn.close()


def _ingest_upload(paths: DataPaths, up: Upload, dataset_id: int | None, folder: str) -> MediaUploadResult:
    conn = connect(paths.db)
    try:
        row, duplicate = media_mod.ingest(conn, paths, up.path, name=up.filename, sha256=up.sha256, size=up.bytes)
        if dataset_id is None:
            return MediaUploadResult(media=media_mod.to_api(row), duplicate=duplicate)
        where, already = datasets_mod.put_uploaded(conn, dataset_id, row["id"], folder)
        row = media_mod.get_row(conn, row["id"])
        return MediaUploadResult(media=media_mod.to_api(row), duplicate=duplicate, folder=where, already_in_dataset=already)
    finally:
        conn.close()


@router.post("", openapi_extra={"requestBody": {"content": {"multipart/form-data": {"schema": {
    "type": "object", "required": ["file"], "properties": {
        "file": {"type": "string", "format": "binary"},
        "folder": {"type": "string", "description": "folder in the dataset of ?dataset_id, e.g. 증강1/어둡게"},
    }}}}}})
async def upload_media(request: Request, dataset_id: int | None = None) -> MediaUploadResult:
    """Multipart field ``file``. Videos are answered after their proxy has been built.
    ``dataset_id`` also puts the media (new or already stored) into that dataset, in the folder
    given by the form field ``folder`` ('' = the top level; made if missing). A media that is in
    the dataset already stays in its folder and comes back with ``already_in_dataset``."""
    paths: DataPaths = request.app.state.paths
    guard_disk(paths)
    if dataset_id is not None:
        await run_in_threadpool(check_dataset, paths, dataset_id)  # 404 before the body is read
    up = await receive_upload(request, paths.tmp, max_bytes=MAX_MEDIA_BYTES)
    try:
        folder = datasets_mod.clean_folder(up.fields.get("folder"))
        if folder and dataset_id is None:
            raise LabError(400, "폴더는 데이터셋 안에만 있습니다. dataset_id 도 보내세요")
    except LabError:
        up.discard()
        raise
    return await run_in_threadpool(_ingest_upload, paths, up, dataset_id, folder)


@router.get("/{media_id}")
def get_media(media_id: int, conn: Conn) -> Media:
    return media_mod.to_api(media_mod.get_row(conn, media_id))


@router.patch("/{media_id}")
def patch_media(media_id: int, patch: MediaPatch, conn: Conn) -> Media:
    media_mod.get_row(conn, media_id)
    changes: dict[str, Any] = {}
    if patch.name is not None:
        name = patch.name.strip()
        if not name:
            raise LabError(400, "이름을 비울 수 없습니다")
        changes["name"] = name[:200]
    if patch.tags is not None:
        changes["tags"] = dumps(media_mod.clean_tags(patch.tags))
    if patch.note is not None:
        changes["note"] = patch.note
    if patch.starred is not None:
        changes["starred"] = int(patch.starred)
    if changes:
        conn.execute(
            f"UPDATE media SET {', '.join(f'{k} = ?' for k in changes)} WHERE id = ?", [*changes.values(), media_id]
        )
    return media_mod.to_api(media_mod.get_row(conn, media_id))


@router.delete("/{media_id}")
def delete_media(media_id: int, conn: Conn, paths: Paths) -> Ok:
    media_mod.delete(conn, paths, media_id)
    return Ok()


@router.get("/{media_id}/file")
def media_file(media_id: int, conn: Conn, paths: Paths) -> FileResponse:
    """The image, or the H.264 proxy of a video (HTTP Range supported)."""
    path = media_mod.display_path(paths, media_mod.get_row(conn, media_id))
    if not path.exists():
        raise LabError(404, "파일이 없습니다")
    return FileResponse(path)


@router.get("/{media_id}/original")
def media_original(media_id: int, conn: Conn, paths: Paths) -> FileResponse:
    row = media_mod.get_row(conn, media_id)
    path = media_mod.original_path(paths, row)
    if not path.exists():
        raise LabError(404, "원본 파일이 없습니다")
    return FileResponse(path, filename=row["name"], content_disposition_type="attachment")


@router.get("/{media_id}/thumb")
def media_thumb(media_id: int, conn: Conn, paths: Paths, v: str | None = None) -> FileResponse:
    row = media_mod.get_row(conn, media_id)
    path = paths.media_thumb(media_id)
    if not path.exists():
        raise LabError(404, "썸네일이 없습니다")
    return FileResponse(path, headers=_cache_headers(row, v))


@router.get("/{media_id}/frame/{n}", response_class=Response,
            responses={200: {"content": {"image/jpeg": {}}}})
def media_frame(media_id: int, n: int, conn: Conn, paths: Paths, v: str | None = None) -> Response:
    row = media_mod.get_row(conn, media_id)
    try:
        frame = frames.get_frame(media_mod.frame_source(paths, row), n)
    except IndexError:
        raise LabError(400, f"프레임 번호가 범위를 벗어났습니다 (0 ~ {(row['frame_count'] or 1) - 1})") from None
    except frames.FrameError as exc:
        raise LabError(500, str(exc)) from exc
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise LabError(500, "JPEG 로 인코딩하지 못했습니다")
    return Response(content=buf.tobytes(), media_type="image/jpeg", headers=_cache_headers(row, v))

"""/api/runs: queue, list, edit, cancel, delete runs; their frames, thumbnail and export."""

from __future__ import annotations

from typing import Annotated, Any, get_args

from fastapi import APIRouter, Query
from fastapi.responses import FileResponse

from lab import export as export_mod
from lab import media as media_mod
from lab import runs as runs_mod
from lab.db import dumps, now_iso
from lab.deps import Conn, Paths
from lab.errors import LabError
from lab.schemas import ExportInfo, Ok, Run, RunCreate, RunFrames, RunList, RunPatch, RunStatus
from lab.storage import guard_disk

router = APIRouter(prefix="/api/runs", tags=["runs"])


@router.get("")
def list_runs(
    conn: Conn,
    paths: Paths,
    media_id: int | None = None,
    model_key: str | None = None,
    status: Annotated[str | None, Query(description="one status or a comma-separated list")] = None,
    starred: bool | None = None,
    tag: str | None = None,
    q: Annotated[str | None, Query(max_length=200, description="search note, model, media name or #id")] = None,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> RunList:
    if status:
        allowed = set(get_args(RunStatus))
        bad = [s for s in status.split(",") if s.strip() and s.strip() not in allowed]
        if bad:
            raise LabError(400, f"알 수 없는 상태입니다: {', '.join(bad)}")
    return runs_mod.list_runs(conn, paths, media_id=media_id, model_key=model_key, status=status,
                              starred=starred, tag=tag, q=q, limit=limit, offset=offset)


@router.post("")
def create_run(req: RunCreate, conn: Conn, paths: Paths) -> Run:
    guard_disk(paths)
    return runs_mod.create(conn, paths, req)


@router.get("/{run_id}")
def get_run(run_id: int, conn: Conn, paths: Paths) -> Run:
    return runs_mod.to_api(runs_mod.get_row(conn, run_id), paths)


@router.patch("/{run_id}")
def patch_run(run_id: int, patch: RunPatch, conn: Conn, paths: Paths) -> Run:
    runs_mod.get_row(conn, run_id)
    changes: dict[str, Any] = {}
    if patch.note is not None:
        changes["note"] = patch.note
    if patch.tags is not None:
        changes["tags"] = dumps(media_mod.clean_tags(patch.tags))
    if patch.starred is not None:
        changes["starred"] = int(patch.starred)
    if changes:
        conn.execute(
            f"UPDATE runs SET {', '.join(f'{k} = ?' for k in changes)} WHERE id = ?", [*changes.values(), run_id]
        )
    return runs_mod.to_api(runs_mod.get_row(conn, run_id), paths)


@router.delete("/{run_id}")
def delete_run(run_id: int, conn: Conn, paths: Paths) -> Ok:
    runs_mod.delete(conn, paths, run_id)
    return Ok()


@router.post("/{run_id}/cancel")
def cancel_run(run_id: int, conn: Conn, paths: Paths) -> Run:
    """Queued runs are cancelled at once; a running one stops at the next frame boundary."""
    runs_mod.get_row(conn, run_id)
    conn.execute(
        "UPDATE runs SET status = 'cancelled', cancel_requested = 1, finished_at = ? WHERE id = ? AND status = 'queued'",
        (now_iso(), run_id),
    )
    conn.execute("UPDATE runs SET cancel_requested = 1 WHERE id = ? AND status = 'running'", (run_id,))
    return runs_mod.to_api(runs_mod.get_row(conn, run_id), paths)


@router.get("/{run_id}/frames")
def run_frames(run_id: int, conn: Conn) -> RunFrames:
    return runs_mod.frames_of(conn, run_id)


@router.get("/{run_id}/thumb")
def run_thumb(run_id: int, conn: Conn, paths: Paths) -> FileResponse:
    runs_mod.get_row(conn, run_id)
    path = paths.run_thumb(run_id)
    if not path.exists():
        raise LabError(404, "아직 썸네일이 없습니다")
    return FileResponse(path)


@router.post("/{run_id}/export")
def create_export(run_id: int, conn: Conn, paths: Paths) -> ExportInfo:
    """Synchronous: a video export answers once the MP4 is written."""
    return export_mod.export_run(conn, paths, run_id)


@router.get("/{run_id}/download")
def download_result(run_id: int, conn: Conn, paths: Paths) -> FileResponse:
    """The drawn result (PNG / MP4), exported first if it never was. Named after the media."""
    run = runs_mod.get_row(conn, run_id)
    path = runs_mod.export_file(paths, run_id)
    if path is None:
        export_mod.export_run(conn, paths, run_id)
        path = runs_mod.export_file(paths, run_id)
    assert path is not None
    stem = run["media_name"].rsplit(".", 1)[0] if "." in run["media_name"] else run["media_name"]
    return FileResponse(path, filename=f"{stem}_result{path.suffix}", content_disposition_type="attachment")


@router.get("/{run_id}/export")
def download_export(run_id: int, conn: Conn, paths: Paths) -> FileResponse:
    run = runs_mod.get_row(conn, run_id)
    path = runs_mod.export_file(paths, run_id)
    if path is None:
        raise LabError(404, "내보낸 파일이 없습니다. 먼저 내보내기를 실행하세요")
    stem = run["media_name"].rsplit(".", 1)[0]
    return FileResponse(path, filename=f"run-{run_id}-{stem}{path.suffix}", content_disposition_type="attachment")

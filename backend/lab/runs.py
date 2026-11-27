"""Run records: create, list, convert to API objects, and the timing / summary maths."""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from cubevision.types import COLORS

from lab import media as media_mod
from lab import models
from lab.config import DataPaths
from lab.db import dumps, loads, now_iso, transaction
from lab.errors import LabError
from lab.schemas import FrameRecord, Run, RunCreate, RunFrames, RunList, RunSummary, Timing

_SELECT = """
    SELECT r.*, m.name AS media_name, m.kind AS media_kind
    FROM runs r JOIN media m ON m.id = r.media_id
"""


def get_row(conn: sqlite3.Connection, run_id: int) -> sqlite3.Row:
    row = conn.execute(_SELECT + " WHERE r.id = ?", (run_id,)).fetchone()
    if row is None:
        raise LabError(404, "실행 기록을 찾을 수 없습니다")
    return row


def export_file(paths: DataPaths, run_id: int) -> Path | None:
    for p in paths.run_exports(run_id):
        if p.exists():
            return p
    return None


def _mtime_ns(path: Path) -> int | None:
    try:
        return path.stat().st_mtime_ns
    except FileNotFoundError:
        return None


def to_api(row: sqlite3.Row, paths: DataPaths) -> Run:
    rid = row["id"]
    thumb_v = _mtime_ns(paths.run_thumb(rid))
    export = export_file(paths, rid)
    timing = loads(row["timing"])
    summary = loads(row["summary"])
    return Run(
        id=rid,
        media_id=row["media_id"],
        media_name=row["media_name"],
        media_kind=row["media_kind"],
        model_key=row["model_key"],
        model_name=row["model_name"],
        params=loads(row["params"], {}),
        frame_start=row["frame_start"],
        frame_end=row["frame_end"],
        stride=row["stride"],
        frames_total=row["frames_total"],
        frames_done=row["frames_done"],
        status=row["status"],
        device=row["device"],
        timing=Timing(**timing) if timing else None,
        summary=RunSummary(**summary) if summary else None,
        error=row["error"],
        parent_run_id=row["parent_run_id"],
        batch_id=row["batch_id"],
        tags=loads(row["tags"], []),
        note=row["note"],
        starred=bool(row["starred"]),
        created_at=row["created_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        # The version suffix makes browsers refetch a file that was rewritten (re-export), and
        # keeps run 1 of another data dir (ids restart at 1) out of the browser cache.
        thumb_url=f"/api/runs/{rid}/thumb?v={thumb_v}" if thumb_v else None,
        export_url=f"/api/runs/{rid}/export?v={export.stat().st_mtime_ns}" if export else None,
    )


def create(conn: sqlite3.Connection, paths: DataPaths, req: RunCreate, *, batch_id: int | None = None) -> Run:
    m = media_mod.get_row(conn, req.media_id)
    model = models.resolve(conn, req.model_key, req.params)
    if m["kind"] == "image":
        start, end, stride = 0, 1, 1
    else:
        count = int(m["frame_count"] or 0)
        start = req.frame_start
        end = count if req.frame_end is None else min(req.frame_end, count)
        stride = req.stride
        if not 0 <= start < count:
            raise LabError(400, f"시작 프레임은 0 ~ {count - 1} 사이여야 합니다")
        if end <= start:
            raise LabError(400, "끝 프레임은 시작 프레임보다 커야 합니다")
    if req.parent_run_id is not None and conn.execute(
        "SELECT 1 FROM runs WHERE id = ?", (req.parent_run_id,)
    ).fetchone() is None:
        raise LabError(400, "원래 실행 기록을 찾을 수 없습니다")
    with transaction(conn):
        cur = conn.execute(
            """INSERT INTO runs (media_id, model_key, model_name, params, frame_start, frame_end, stride,
                                 frames_total, status, parent_run_id, note, created_at, batch_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?, ?, ?)""",
            (m["id"], model.key, model.name, dumps(model.params), start, end, stride,
             math.ceil((end - start) / stride), req.parent_run_id, req.note, now_iso(), batch_id),
        )
    return to_api(get_row(conn, int(cur.lastrowid)), paths)


def list_runs(
    conn: sqlite3.Connection,
    paths: DataPaths,
    *,
    media_id: int | None = None,
    model_key: str | None = None,
    status: str | None = None,
    starred: bool | None = None,
    tag: str | None = None,
    q: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> RunList:
    where: list[str] = []
    args: list[Any] = []
    if media_id is not None:
        where.append("r.media_id = ?")
        args.append(media_id)
    if model_key:
        where.append("r.model_key = ?")
        args.append(model_key)
    if status:
        states = [s.strip() for s in status.split(",") if s.strip()]
        where.append(f"r.status IN ({','.join('?' * len(states))})")
        args += states
    if starred is not None:
        where.append("r.starred = ?")
        args.append(int(starred))
    if tag:
        where.append("EXISTS (SELECT 1 FROM json_each(r.tags) WHERE value = ?)")
        args.append(tag)
    if q:
        where.append("(r.note LIKE ? OR r.model_name LIKE ? OR m.name LIKE ? OR CAST(r.id AS TEXT) = ?)")
        like = f"%{q}%"
        args += [like, like, like, q.lstrip("#")]
    clause = (" WHERE " + " AND ".join(where)) if where else ""
    total = conn.execute(
        "SELECT COUNT(*) FROM runs r JOIN media m ON m.id = r.media_id" + clause, args
    ).fetchone()[0]
    rows = conn.execute(_SELECT + clause + " ORDER BY r.id DESC LIMIT ? OFFSET ?", [*args, limit, offset])
    return RunList(items=[to_api(r, paths) for r in rows], total=total)


def frames_of(conn: sqlite3.Connection, run_id: int) -> RunFrames:
    row = get_row(conn, run_id)
    fps = conn.execute("SELECT fps FROM media WHERE id = ?", (row["media_id"],)).fetchone()["fps"]
    frames = [
        FrameRecord(frame=f["frame"], ms=f["ms"], cubes=loads(f["cubes"], []))
        for f in conn.execute("SELECT frame, ms, cubes FROM run_frames WHERE run_id = ? ORDER BY frame", (run_id,))
    ]
    return RunFrames(run_id=run_id, fps=fps, stride=row["stride"], frames=frames)


def delete(conn: sqlite3.Connection, paths: DataPaths, run_id: int) -> None:
    get_row(conn, run_id)
    # A running run is deleted too: the worker notices the row is gone and stops.
    with transaction(conn):
        conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
    for p in [paths.run_thumb(run_id), *paths.run_exports(run_id)]:
        p.unlink(missing_ok=True)


def compute_timing(ms: list[float]) -> dict[str, float] | None:
    """First frame (model warm-up) separately; statistics over the rest, or the single frame."""
    if not ms:
        return None
    rest = np.asarray(ms[1:] if len(ms) > 1 else ms, np.float64)
    return {
        "frames": len(ms),
        "first_ms": round(ms[0], 3),
        "mean_ms": round(float(rest.mean()), 3),
        "p50_ms": round(float(np.percentile(rest, 50)), 3),
        "p95_ms": round(float(np.percentile(rest, 95)), 3),
        "max_ms": round(float(rest.max()), 3),
    }


def count_colours(cubes: list[dict[str, Any]]) -> dict[str, int]:
    """{colour: number of cubes} of one frame: compute_summary takes one per frame."""
    counts: dict[str, int] = {}
    for c in cubes:
        counts[c["color"]] = counts.get(c["color"], 0) + 1
    return counts


def close_interrupted(conn: sqlite3.Connection, reason: str) -> None:
    """End the runs a stopped worker left 'running', keeping the frames it had stored.

    A run with a pending cancel becomes 'cancelled', any other 'failed' with ``reason``. Like a
    run that ended normally, it gets frames_done, timing and summary from its stored frames.
    """
    now = now_iso()
    for run in conn.execute("SELECT id, cancel_requested FROM runs WHERE status = 'running'").fetchall():
        ms: list[float] = []
        counts: list[dict[str, int]] = []
        for f in conn.execute("SELECT ms, cubes FROM run_frames WHERE run_id = ? ORDER BY frame", (run["id"],)):
            ms.append(f["ms"])
            counts.append(count_colours(loads(f["cubes"], [])))
        status, error = ("cancelled", None) if run["cancel_requested"] else ("failed", reason)
        conn.execute(
            """UPDATE runs SET status = ?, error = ?, finished_at = ?, frames_done = ?, timing = ?, summary = ?
               WHERE id = ? AND status = 'running'""",
            (status, error, now, len(ms),
             dumps(compute_timing(ms)) if ms else None,
             dumps(compute_summary(counts)) if counts else None,
             run["id"]),
        )


def compute_summary(colour_counts: list[dict[str, int]]) -> dict[str, Any] | None:
    """``colour_counts`` has one {colour: count} dict per processed frame."""
    if not colour_counts:
        return None
    totals = [sum(c.values()) for c in colour_counts]
    n = len(colour_counts)
    return {
        "frames_with_cubes": sum(1 for t in totals if t > 0),
        "cubes_mean": round(sum(totals) / n, 3),
        "cubes_min": min(totals),
        "cubes_max": max(totals),
        "color_mean": {c: round(sum(fc.get(c, 0) for fc in colour_counts) / n, 3) for c in COLORS},
    }

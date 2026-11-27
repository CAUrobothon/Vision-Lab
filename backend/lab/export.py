"""Bake a run's cubes into a PNG (image run) or an H.264 MP4 (video run) on demand.

The video is streamed: proxy frames are decoded one by one, the overlay of the nearest
processed frame <= the current frame is drawn, and raw BGR is piped into ffmpeg
(cv2.VideoWriter's mp4v would not play in browsers).
"""

from __future__ import annotations

import sqlite3
import subprocess
import tempfile
import uuid

import cv2

from cubevision.draw import draw_cubes

from lab import media as media_mod
from lab import runs as runs_mod
from lab.config import DataPaths
from lab.db import loads
from lab.errors import LabError
from lab.frames import iter_frames
from lab.schemas import ExportInfo
from lab.storage import guard_disk


def export_run(conn: sqlite3.Connection, paths: DataPaths, run_id: int) -> ExportInfo:
    run = runs_mod.get_row(conn, run_id)
    if run["status"] in ("queued", "running"):
        raise LabError(409, "실행이 끝난 뒤에 내보낼 수 있습니다")
    count, last = conn.execute(
        "SELECT COUNT(*), MAX(frame) FROM run_frames WHERE run_id = ?", (run_id,)
    ).fetchone()
    if count == 0:
        raise LabError(409, "저장된 결과 프레임이 없어 내보낼 수 없습니다")
    guard_disk(paths)
    m = media_mod.get_row(conn, run["media_id"])
    src = media_mod.frame_source(paths, m)
    rows = conn.execute("SELECT frame, cubes FROM run_frames WHERE run_id = ? ORDER BY frame", (run_id,))

    if m["kind"] == "image":
        dst = paths.exports / f"run-{run_id}.png"
        frame = next(iter_frames(src, 0, 1))[1]
        tmp = paths.tmp / f"export-{uuid.uuid4().hex}.png"
        if not cv2.imwrite(str(tmp), draw_cubes(frame, loads(rows.fetchone()["cubes"], [])),
                           [cv2.IMWRITE_PNG_COMPRESSION, 3]):
            raise LabError(500, "PNG 를 저장하지 못했습니다")
        tmp.replace(dst)
        return ExportInfo(url=f"/api/runs/{run_id}/export?v={dst.stat().st_mtime_ns}", bytes=dst.stat().st_size,
                          kind="image")

    dst = paths.exports / f"run-{run_id}.mp4"
    tmp = paths.tmp / f"export-{uuid.uuid4().hex}.mp4"
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{m['width']}x{m['height']}", "-r", f"{m['fps']:g}", "-i", "-",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(tmp),
    ]
    with tempfile.TemporaryFile() as errlog:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=errlog)
        try:
            pending = rows.fetchone()  # next stored result not yet reached by the playhead
            overlay: list = []
            for index, frame in iter_frames(src, run["frame_start"], last + 1):
                while pending is not None and pending["frame"] <= index:
                    overlay = loads(pending["cubes"], [])
                    pending = rows.fetchone()
                proc.stdin.write((draw_cubes(frame, overlay) if overlay else frame).tobytes())
            proc.stdin.close()
            code = proc.wait()
        except BrokenPipeError:
            code = proc.wait()
        except BaseException:
            proc.kill()
            proc.wait()
            tmp.unlink(missing_ok=True)
            raise
        if code != 0:
            errlog.seek(0)
            tail = errlog.read().decode("utf-8", "replace").strip().splitlines()[-3:]
            tmp.unlink(missing_ok=True)
            raise LabError(500, "영상 내보내기에 실패했습니다: " + " / ".join(tail))
    tmp.replace(dst)
    return ExportInfo(url=f"/api/runs/{run_id}/export?v={dst.stat().st_mtime_ns}", bytes=dst.stat().st_size,
                      kind="video")

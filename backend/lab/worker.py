"""The worker process: takes queued runs from SQLite, one at a time, oldest first.

It is started with multiprocessing's "spawn" method by the API (see supervisor.py) and shares
nothing with it but the database. torch / Ultralytics are imported here only when a run uses
uploaded weights, and at most one checkpoint stays loaded.
"""

from __future__ import annotations

import os
import signal
import sqlite3
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from cubevision.draw import draw_cubes
from cubevision.pipelines.base import Pipeline
from cubevision.registry import get_pipeline

from lab import media as media_mod
from lab import models, ultra
from lab.config import DataPaths
from lab.db import connect, dumps, loads, now_iso, transaction
from lab.frames import iter_frames
from lab.runs import compute_summary, compute_timing, count_colours

POLL_S = 0.1  # idle wait between queue checks
HEARTBEAT_S = 1.0
FLUSH_FRAMES = 10  # store frames in batches: every 10 frames ...
FLUSH_S = 0.5  # ... or every 0.5 s, whichever comes first
CANCEL_CHECK_S = 0.2


def log(msg: str) -> None:
    print(f"[worker {os.getpid()}] {msg}", file=sys.stderr, flush=True)


def register_pipelines(extra: Sequence[Pipeline]) -> None:
    """Add pipelines that are not in cubevision's registry (tests use this)."""
    from cubevision.pipelines import ALL

    known = {p.key for p in ALL}
    ALL.extend(p for p in extra if p.key not in known)


class RunGone(Exception):
    """The run is no longer this worker's: deleted (or its media), or ended by another process
    (a server that started after this worker's API died, or a forced restart)."""


class RunFailed(Exception):
    """A failure with a ready-made Korean message."""


class _Progress:
    """Collects per-frame results and writes them to run_frames in batches."""

    def __init__(self, conn: sqlite3.Connection, run_id: int):
        self.conn = conn
        self.run_id = run_id
        self.pending: list[tuple[int, int, float, str]] = []
        self.ms: list[float] = []
        self.colour_counts: list[dict[str, int]] = []
        self.last_flush = time.perf_counter()

    def add(self, frame: int, ms: float, cubes: list[dict[str, Any]]) -> None:
        self.pending.append((self.run_id, frame, round(ms, 3), dumps(cubes)))
        self.ms.append(ms)
        self.colour_counts.append(count_colours(cubes))

    def due(self) -> bool:
        return len(self.pending) >= FLUSH_FRAMES or time.perf_counter() - self.last_flush >= FLUSH_S

    def flush(self) -> None:
        with transaction(self.conn):
            cur = self.conn.execute(
                "UPDATE runs SET frames_done = ? WHERE id = ? AND status = 'running'", (len(self.ms), self.run_id)
            )
            if cur.rowcount == 0:
                raise RunGone
            self.conn.executemany(
                "INSERT OR REPLACE INTO run_frames (run_id, frame, ms, cubes) VALUES (?, ?, ?, ?)", self.pending
            )
        self.pending.clear()
        self.last_flush = time.perf_counter()


class Worker:
    def __init__(self, paths: DataPaths):
        self.paths = paths
        self.conn = connect(paths.db)
        self.parent_pid = os.getppid()
        self.model: ultra.UltraModel | None = None
        self.loaded: tuple[str, str] | None = None  # (model key, device) of self.model
        self.loaded_label: str | None = None
        self.mps: bool | None = None  # known once torch has been imported
        self.ultra_ready = False
        self.last_beat = 0.0

    # Heartbeat -------------------------------------------------------------------------------
    def beat(self, busy_run_id: int | None = None, *, force: bool = False) -> None:
        now = time.perf_counter()
        if not force and now - self.last_beat < HEARTBEAT_S:
            return
        self.last_beat = now
        self.conn.execute(
            """INSERT INTO worker (id, pid, busy_run_id, loaded_model, mps_available, updated_at)
               VALUES (1, ?, ?, ?, ?, ?)
               ON CONFLICT (id) DO UPDATE SET pid = excluded.pid, busy_run_id = excluded.busy_run_id,
                   loaded_model = excluded.loaded_model,
                   -- a fresh worker has not imported torch yet: keep the last known answer
                   mps_available = COALESCE(excluded.mps_available, worker.mps_available),
                   updated_at = excluded.updated_at""",
            (os.getpid(), busy_run_id, self.loaded_label, None if self.mps is None else int(self.mps), now_iso()),
        )

    # Queue -----------------------------------------------------------------------------------
    def claim(self) -> int | None:
        row = self.conn.execute("SELECT id FROM runs WHERE status = 'queued' ORDER BY id LIMIT 1").fetchone()
        if row is None:
            return None
        cur = self.conn.execute(
            "UPDATE runs SET status = 'running', started_at = ? WHERE id = ? AND status = 'queued'",
            (now_iso(), row["id"]),
        )
        return row["id"] if cur.rowcount == 1 else None

    def loop(self) -> None:
        log("started")
        self.beat(force=True)
        while os.getppid() == self.parent_pid:  # exit if the API process went away
            run_id = self.claim()
            if run_id is None:
                self.beat()
                time.sleep(POLL_S)
                continue
            self.beat(run_id, force=True)
            self.process(run_id)
            self.beat(force=True)
        log("parent exited; stopping")

    # Models ----------------------------------------------------------------------------------
    def unload(self) -> None:
        if self.model is not None:
            self.model.close()
        self.model = None
        self.loaded = None
        self.loaded_label = None

    def predictor(self, key: str, params: dict[str, Any]) -> tuple[Callable[[np.ndarray], list], str]:
        """Return (frame -> cubes, device label) for a model key."""
        if models.weights_id(key) is None:
            try:
                pipeline = get_pipeline(key)
            except KeyError:
                raise RunFailed(f"모델을 찾을 수 없습니다: {key}") from None
            return (lambda frame: pipeline(frame, params).cubes), "opencv"

        row = self.conn.execute("SELECT * FROM weights WHERE id = ?", (models.weights_id(key),)).fetchone()
        if row is None:
            raise RunFailed("가중치가 삭제되어 실행할 수 없습니다")
        if row["state"] != "ready":
            raise RunFailed(f"사용할 수 없는 가중치입니다: {row['error']}")
        if not self.ultra_ready:
            ultra.configure(self.paths.root)
            self.ultra_ready = True
            self.mps = ultra.mps_available()
        device = ultra.resolve_device(params.get("device", "auto"))
        if self.loaded != (key, device):
            self.unload()
            try:
                self.model = ultra.UltraModel(models.weights_path(self.paths, row), device)
            except Exception as exc:  # noqa: BLE001 - report any load failure on the run
                raise RunFailed(f"가중치를 불러오지 못했습니다: {type(exc).__name__}: {exc}") from exc
            self.loaded = (key, device)
            self.loaded_label = f"{row['name']} ({device})"
        model = self.model
        return (lambda frame: model.predict(frame, params)), device

    # One run ---------------------------------------------------------------------------------
    def _cancelled(self, run_id: int) -> bool:
        row = self.conn.execute("SELECT status, cancel_requested FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None or row["status"] != "running":
            raise RunGone
        return bool(row["cancel_requested"])

    def _finish(self, run_id: int, progress: _Progress, status: str, error: str | None = None) -> None:
        # Only a run still 'running' is ours to end: never overwrite what another process decided.
        cur = self.conn.execute(
            """UPDATE runs SET status = ?, error = ?, finished_at = ?, frames_done = ?, timing = ?, summary = ?
               WHERE id = ? AND status = 'running'""",
            (status, error, now_iso(), len(progress.ms),
             dumps(compute_timing(progress.ms)) if progress.ms else None,
             dumps(compute_summary(progress.colour_counts)) if progress.colour_counts else None,
             run_id),
        )
        if cur.rowcount == 0:
            raise RunGone

    def _gone(self, run_id: int) -> None:
        if self.conn.execute("SELECT 1 FROM runs WHERE id = ?", (run_id,)).fetchone() is None:
            # Deleted during its first frame, the API found no thumbnail to remove; it may exist now.
            self.paths.run_thumb(run_id).unlink(missing_ok=True)
            log(f"run {run_id} was deleted while running")
        else:
            log(f"run {run_id} was ended by another process; stopping")

    def _write_thumb(self, run_id: int, frame: np.ndarray, cubes: list[dict[str, Any]]) -> None:
        try:
            media_mod.write_thumb(draw_cubes(frame, cubes), self.paths.run_thumb(run_id))
        except Exception as exc:  # noqa: BLE001 - a missing thumbnail must not fail the run
            log(f"run {run_id}: thumbnail failed: {exc}")

    def process(self, run_id: int) -> None:
        run = self.conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        media = self.conn.execute("SELECT * FROM media WHERE id = ?", (run["media_id"],)).fetchone() if run else None
        if run is None or media is None:
            return
        progress = _Progress(self.conn, run_id)
        started = time.perf_counter()
        try:
            params = loads(run["params"], {})
            predict, device = self.predictor(run["model_key"], params)
            self.conn.execute("UPDATE runs SET device = ? WHERE id = ?", (device, run_id))
            self.beat(run_id, force=True)
            last_cancel_check = time.perf_counter()
            src = media_mod.frame_source(self.paths, media)
            for index, frame in iter_frames(src, run["frame_start"], run["frame_end"], run["stride"]):
                t0 = time.perf_counter()
                cubes = [c.to_dict() for c in predict(frame)]
                progress.add(index, (time.perf_counter() - t0) * 1000.0, cubes)
                if len(progress.ms) == 1:
                    self._write_thumb(run_id, frame, cubes)
                if progress.due():
                    progress.flush()
                    self.beat(run_id)
                if time.perf_counter() - last_cancel_check >= CANCEL_CHECK_S:
                    last_cancel_check = time.perf_counter()
                    if os.getppid() != self.parent_pid:
                        # The API died: the next server ends this run from its stored frames.
                        log(f"run {run_id}: parent exited; stopping")
                        return
                    if self._cancelled(run_id):
                        progress.flush()
                        self._finish(run_id, progress, "cancelled")
                        log(f"run {run_id} cancelled after {len(progress.ms)} frames")
                        return
            progress.flush()
            if not progress.ms:
                raise RunFailed("처리한 프레임이 없습니다 (영상을 읽지 못했습니다)")
            self._finish(run_id, progress, "done")
            log(f"run {run_id} done: {len(progress.ms)} frames in {time.perf_counter() - started:.1f} s on {device}")
        except RunGone:
            self._gone(run_id)
        except Exception as exc:  # noqa: BLE001 - every failure ends up on the run record
            message = str(exc) if isinstance(exc, RunFailed) else f"실행 중 오류가 났습니다: {type(exc).__name__}: {exc}"
            log(f"run {run_id} failed: {message}")
            try:
                if progress.pending:
                    progress.flush()
                self._finish(run_id, progress, "failed", message[:1000])
            except RunGone:
                self._gone(run_id)


def worker_main(data_dir: str, extra_pipelines: Sequence[Pipeline] = ()) -> None:
    """Entry point of the spawned worker process."""
    # Ctrl+C in the terminal reaches the whole process group; the API stops the worker itself.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    register_pipelines(extra_pipelines)
    Worker(DataPaths(Path(data_dir))).loop()

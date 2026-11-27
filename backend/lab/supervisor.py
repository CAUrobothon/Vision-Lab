"""API-side control of the worker process: start, stop, restart, and a watchdog thread.

The watchdog respawns a worker that died (marking its run failed) and force-restarts one that
ignores a cancel request for more than CANCEL_GRACE_S (e.g. stuck inside a long MPS call).
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import threading
import time
from collections.abc import Sequence
from multiprocessing.process import BaseProcess

from cubevision.pipelines.base import Pipeline

from lab.config import DataPaths
from lab.db import connect
from lab.runs import close_interrupted
from lab.schemas import WorkerInfo
from lab.worker import worker_main

log = logging.getLogger("lab")

WATCH_S = 0.5
CANCEL_GRACE_S = 5.0
STOP_TIMEOUT_S = 3.0
# A worker that dies right after starting (e.g. an import error) is respawned with a growing delay.
CRASH_WINDOW_S = 10.0
MAX_BACKOFF_S = 30.0


class WorkerSupervisor:
    def __init__(self, paths: DataPaths, extra_pipelines: Sequence[Pipeline] = ()):
        self.paths = paths
        self.extra_pipelines = list(extra_pipelines)
        self.proc: BaseProcess | None = None
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._watch: threading.Thread | None = None
        self._cancel_seen: dict[int, float] = {}
        self._started_at = 0.0
        self._backoff = 0.0
        self._retry_at = 0.0

    # Lifecycle -------------------------------------------------------------------------------
    def start(self) -> None:
        with self._lock:
            self._spawn()
        self._stop.clear()
        self._watch = threading.Thread(target=self._watchdog, name="lab-worker-watchdog", daemon=True)
        self._watch.start()

    def shutdown(self) -> None:
        self._stop.set()
        if self._watch is not None:
            self._watch.join(timeout=2)
        with self._lock:
            self._terminate()

    def restart(self, reason: str = "작업자를 다시 시작해서 중단됨") -> WorkerInfo:
        with self._lock:
            self._terminate()
            self._mark_interrupted(reason)
            self._spawn()
        return self.info()

    def _spawn(self) -> None:
        self._started_at = time.monotonic()
        self._retry_at = 0.0
        ctx = mp.get_context("spawn")
        self.proc = ctx.Process(
            target=worker_main, args=(str(self.paths.root), self.extra_pipelines), name="lab-worker", daemon=True
        )
        self.proc.start()

    def _terminate(self) -> None:
        proc, self.proc = self.proc, None
        if proc is None:
            return
        if proc.is_alive():
            proc.terminate()
            proc.join(STOP_TIMEOUT_S)
            if proc.is_alive():
                proc.kill()
                proc.join(STOP_TIMEOUT_S)
        proc.close()

    def _mark_interrupted(self, reason: str) -> None:
        """Runs the old worker left 'running': cancelled if a cancel was pending, else failed."""
        conn = connect(self.paths.db)
        try:
            close_interrupted(conn, reason)
            conn.execute("UPDATE worker SET busy_run_id = NULL, loaded_model = NULL WHERE id = 1")
        finally:
            conn.close()

    # Watchdog --------------------------------------------------------------------------------
    def _watchdog(self) -> None:
        while not self._stop.wait(WATCH_S):
            try:
                self._check()
            except Exception:  # noqa: BLE001 - the watchdog must keep running
                log.exception("worker watchdog error")

    def _check(self) -> None:
        with self._lock:
            if self._stop.is_set():
                return
            if self.proc is not None and not self.proc.is_alive():
                now = time.monotonic()
                if self._retry_at == 0.0:
                    quick = now - self._started_at < CRASH_WINDOW_S
                    self._backoff = min(MAX_BACKOFF_S, max(1.0, self._backoff * 2)) if quick else 0.0
                    self._retry_at = now + self._backoff
                    log.warning("worker exited (code %s); restarting in %.0f s", self.proc.exitcode, self._backoff)
                if now >= self._retry_at:
                    self.restart("작업자가 비정상 종료되어 중단됨")
                return
        conn = connect(self.paths.db)
        try:
            stuck = [r["id"] for r in conn.execute(
                "SELECT id FROM runs WHERE status = 'running' AND cancel_requested = 1"
            )]
        finally:
            conn.close()
        now = time.monotonic()
        self._cancel_seen = {rid: self._cancel_seen.get(rid, now) for rid in stuck}
        if any(now - t > CANCEL_GRACE_S for t in self._cancel_seen.values()):
            log.warning("worker ignored a cancel for %.0f s; restarting it", CANCEL_GRACE_S)
            self._cancel_seen.clear()
            self.restart()

    # Status ----------------------------------------------------------------------------------
    def info(self) -> WorkerInfo:
        proc = self.proc
        alive = bool(proc is not None and proc.is_alive())
        pid = proc.pid if proc is not None else None
        busy = loaded = None
        conn = connect(self.paths.db)
        try:
            row = conn.execute("SELECT * FROM worker WHERE id = 1").fetchone()
        finally:
            conn.close()
        if row is not None and alive and row["pid"] == pid:
            busy, loaded = row["busy_run_id"], row["loaded_model"]
        return WorkerInfo(alive=alive, pid=pid, busy_run_id=busy, loaded_model=loaded)

    def mps_available(self) -> bool | None:
        conn = connect(self.paths.db)
        try:
            row = conn.execute("SELECT mps_available FROM worker WHERE id = 1").fetchone()
        finally:
            conn.close()
        return None if row is None or row["mps_available"] is None else bool(row["mps_available"])

"""The lab's FastAPI app and the ``lab`` command.

    uv run lab [--host 127.0.0.1] [--port 8010] [--data PATH]

Serves the API under /api and, when frontend/dist exists, the built web app at / (any
non-API page such as /runs or /compare falls back to index.html for client-side routing).
"""

from __future__ import annotations

import argparse
import fcntl
import logging
import os
import sys
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TextIO

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

from cubevision.pipelines.base import Pipeline

from lab import errors
from lab.api import ROUTERS
from lab.config import FRONTEND_DIST, DataPaths, resolve_data_dir
from lab.db import connect, migrate
from lab.errors import LabError
from lab.runs import close_interrupted
from lab.storage import clear_tmp
from lab.supervisor import WorkerSupervisor
from lab.worker import register_pipelines

NO_FRONTEND_PAGE = """<!doctype html><html lang="ko"><meta charset="utf-8"><title>Vision Lab</title>
<body style="font-family:sans-serif;padding:2rem"><h1>Vision Lab API</h1>
<p>프론트엔드가 아직 빌드되지 않았습니다. 프로젝트 폴더에서 <code>./run.sh</code> 로 실행하면 빌드 후 화면이 열립니다.</p>
<p>API 문서: <a href="/docs">/docs</a></p></body></html>"""


class DataDirBusy(RuntimeError):
    """Another server already uses the data dir."""


def lock_data_dir(paths: DataPaths) -> TextIO:
    """Lock the data dir for this process until the returned file is closed.

    A second server on the same folder would mark the first one's running run failed and start
    a second worker (two torch processes on a 16 GB Mac). The OS drops the lock when the process
    ends, even on kill -9, so a crash never leaves a stale lock; the worker does not inherit it.
    """
    f = open(paths.lock, "a+", encoding="utf-8")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        f.seek(0)
        holder = f.read().strip() or "?"
        f.close()
        raise DataDirBusy(
            f"다른 Vision Lab 서버(pid {holder})가 이미 이 데이터 폴더를 쓰고 있습니다: {paths.root}"
        ) from None
    except OSError as exc:  # a file system without locks: run unguarded rather than not at all
        logging.getLogger("lab").warning("could not lock %s: %s", paths.lock, exc)
        return f
    f.seek(0)
    f.truncate()
    f.write(str(os.getpid()))
    f.flush()
    return f


def recover_interrupted(paths: DataPaths) -> None:
    """Runs left 'running' by a previous server process can never finish: end them."""
    conn = connect(paths.db)
    try:
        close_interrupted(conn, "서버가 다시 시작되어 중단됨")
        conn.execute("DELETE FROM worker")
    finally:
        conn.close()


def create_app(
    data_dir: Path | None = None,
    *,
    frontend_dir: Path | None = None,
    extra_pipelines: Sequence[Pipeline] = (),
) -> FastAPI:
    """Build the app. ``extra_pipelines`` are registered in the API and worker processes (tests)."""
    paths = DataPaths(resolve_data_dir(data_dir))
    paths.ensure()
    migrate(paths.db)
    register_pipelines(extra_pipelines)
    supervisor = WorkerSupervisor(paths, extra_pipelines)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        lock = lock_data_dir(paths)  # first: recovery must not touch another server's runs
        try:
            recover_interrupted(paths)
            clear_tmp(paths)
            supervisor.start()
            yield
        finally:
            supervisor.shutdown()
            lock.close()

    app = FastAPI(title="Vision Lab", lifespan=lifespan)
    app.state.paths = paths
    app.state.supervisor = supervisor
    errors.install(app)
    for router in ROUTERS:
        app.include_router(router)

    @app.api_route("/api/{rest:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"], include_in_schema=False)
    def unknown_api(rest: str) -> None:
        raise LabError(404, f"없는 API 경로입니다: /api/{rest}")

    dist = frontend_dir or FRONTEND_DIST
    app.state.frontend_dist = dist if (dist / "index.html").is_file() else None

    @app.middleware("http")
    async def cache_headers(request: Request, call_next):  # type: ignore[no-untyped-def]
        """Pages are revalidated on every load (a rebuild must reach the browser at once); the
        content-hashed build files never change, so they are cached for good."""
        response = await call_next(request)
        path = request.url.path
        if not path.startswith("/api/"):
            if path.startswith("/assets/"):
                response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
            elif response.headers.get("content-type", "").startswith("text/html"):
                response.headers["Cache-Control"] = "no-cache"
        return response

    if (dist / "index.html").is_file():
        app.frontend("/", directory=dist)
    else:
        @app.get("/", include_in_schema=False)
        def no_frontend() -> HTMLResponse:
            return HTMLResponse(NO_FRONTEND_PAGE)

    return app


def cli() -> None:
    ap = argparse.ArgumentParser(prog="lab", description="Vision Lab 서버")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--data", type=Path, default=None, help="데이터 폴더 (기본: $LAB_DATA_DIR 또는 <프로젝트>/data)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(name)s: %(message)s")
    paths = DataPaths(resolve_data_dir(args.data))
    paths.ensure()
    try:  # before create_app migrates the database: a plain message, the app takes the lock itself
        lock_data_dir(paths).close()
    except DataDirBusy as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from None
    app = create_app(paths.root)
    print(f"Vision Lab: http://{args.host}:{args.port}  (데이터: {app.state.paths.root})", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

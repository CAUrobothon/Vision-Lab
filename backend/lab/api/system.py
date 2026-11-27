"""/api/health, /api/system, /api/storage, /api/worker/restart and /api/samples."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import shutil
import uuid

from fastapi import APIRouter, Request

from lab import media as media_mod
from lab.deps import Conn, Paths, Supervisor
from lab.errors import LabError
from lab.schemas import CleanupRequest, CleanupResult, Health, Media, StorageInfo, SystemInfo, WorkerInfo
from lab.storage import cleanup, guard_disk, storage_info

router = APIRouter(prefix="/api", tags=["system"])

SAMPLE_TAGS = ["sample", "합성"]
FEATURES = ["folders", "gt", "score"]  # route groups a page may check for (Health.features)


def app_version() -> str:
    try:
        return importlib.metadata.version("vision-lab")
    except importlib.metadata.PackageNotFoundError:
        return "dev"


@router.get("/health")
def health(request: Request) -> Health:
    dist = getattr(request.app.state, "frontend_dist", None)
    build = None
    if dist is not None:
        try:
            st = (dist / "index.html").stat()
            build = f"{st.st_mtime_ns:x}-{st.st_size:x}"
        except FileNotFoundError:
            pass
    return Health(build=build, features=FEATURES)


@router.get("/system")
def system(paths: Paths, supervisor: Supervisor) -> SystemInfo:
    return SystemInfo(
        version=app_version(),
        # find_spec only looks for the package; the API process never imports torch.
        torch_available=importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("ultralytics") is not None,
        mps_available=supervisor.mps_available(),
        worker=supervisor.info(),
        storage=storage_info(paths),
    )


@router.get("/storage")
def storage(paths: Paths) -> StorageInfo:
    return storage_info(paths)


@router.post("/storage/cleanup")
def storage_cleanup(req: CleanupRequest, conn: Conn, paths: Paths) -> CleanupResult:
    return CleanupResult(freed_bytes=cleanup(conn, paths, req))


@router.post("/worker/restart")
def restart_worker(supervisor: Supervisor) -> WorkerInfo:
    """Stops the worker (its running run becomes failed) and starts a fresh one; frees all model memory."""
    return supervisor.restart()


@router.post("/samples")
def add_samples(conn: Conn, paths: Paths) -> list[Media]:
    """Generate synthetic sample images and a video with cubevision.synth and ingest them."""
    guard_disk(paths)
    try:
        from cubevision.synth import sample_assets
    except ImportError as exc:
        raise LabError(503, "합성 샘플 생성기(cubevision.synth)가 아직 준비되지 않았습니다") from exc
    out_dir = paths.tmp / f"samples-{uuid.uuid4().hex}"
    out_dir.mkdir()
    try:
        files = sample_assets(out_dir)
        result = []
        for f in files:
            tmp = paths.tmp / f"upload-{uuid.uuid4().hex}{f.suffix.lower()}"
            shutil.move(f, tmp)
            row, _ = media_mod.ingest(conn, paths, tmp, name=f.name, sha256=media_mod.file_sha256(tmp),
                                      size=tmp.stat().st_size, tags=SAMPLE_TAGS)
            result.append(media_mod.to_api(row))
        return result
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)

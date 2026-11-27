"""Disk usage, the free-space guard, and cleanup of old runs and exports."""

from __future__ import annotations

import os
import shutil
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

from lab.config import MIN_FREE_BYTES, DataPaths
from lab.db import transaction
from lab.errors import LabError
from lab.schemas import CleanupRequest, StorageInfo, StorageUsage

CACHE_S = 5.0
_cache: dict[Path, tuple[float, StorageInfo]] = {}
_cache_lock = threading.Lock()


def dir_bytes(path: Path) -> int:
    total = 0
    try:
        with os.scandir(path) as it:
            for entry in it:
                try:
                    if entry.is_file(follow_symlinks=False):
                        total += entry.stat(follow_symlinks=False).st_size
                    elif entry.is_dir(follow_symlinks=False):
                        total += dir_bytes(Path(entry.path))
                except FileNotFoundError:
                    continue
    except FileNotFoundError:
        return 0
    return total


def db_bytes(paths: DataPaths) -> int:
    return sum(p.stat().st_size for p in (paths.db, Path(f"{paths.db}-wal"), Path(f"{paths.db}-shm")) if p.exists())


def storage_info(paths: DataPaths, *, fresh: bool = False) -> StorageInfo:
    """Walking the data dirs costs a few ms per thousand files; cache the answer briefly."""
    with _cache_lock:
        hit = _cache.get(paths.root)
        if hit and not fresh and time.monotonic() - hit[0] < CACHE_S:
            return hit[1]
    usage = shutil.disk_usage(paths.root)
    info = StorageInfo(
        data_dir=str(paths.root),
        free_bytes=usage.free,
        total_bytes=usage.total,
        usage=StorageUsage(
            media_bytes=dir_bytes(paths.media),
            proxy_bytes=dir_bytes(paths.proxy),
            thumbs_bytes=dir_bytes(paths.thumbs),
            weights_bytes=dir_bytes(paths.weights),
            exports_bytes=dir_bytes(paths.exports),
            db_bytes=db_bytes(paths),
        ),
    )
    with _cache_lock:
        _cache[paths.root] = (time.monotonic(), info)
    return info


def guard_disk(paths: DataPaths) -> None:
    free = shutil.disk_usage(paths.root).free
    if free < MIN_FREE_BYTES:
        raise LabError(
            507,
            f"디스크 여유 공간이 부족합니다 (남은 공간 {free / 1024**3:.1f} GB, 최소 2 GB 필요). "
            "모델·저장공간 화면에서 정리하세요",
        )


def _unlink(path: Path) -> int:
    try:
        size = path.stat().st_size
        path.unlink()
        return size
    except FileNotFoundError:
        return 0


def cleanup(conn: sqlite3.Connection, paths: DataPaths, req: CleanupRequest) -> int:
    """Delete what the request names; returns the bytes freed (files plus the shrunken database)."""
    freed = 0
    if req.delete_exports:
        for p in paths.exports.iterdir():
            if p.is_file():
                freed += _unlink(p)
    if req.runs_older_than_days is not None:
        cutoff = datetime.now().astimezone() - timedelta(days=req.runs_older_than_days)
        sql = "SELECT id, created_at FROM runs WHERE status NOT IN ('queued', 'running')"
        if req.unstarred_only:
            sql += " AND starred = 0"
        old = [r["id"] for r in conn.execute(sql) if datetime.fromisoformat(r["created_at"]) <= cutoff]
        if old:
            before = db_bytes(paths)
            with transaction(conn):
                conn.executemany("DELETE FROM runs WHERE id = ?", [(rid,) for rid in old])
            for rid in old:
                for p in [paths.run_thumb(rid), *paths.run_exports(rid)]:
                    freed += _unlink(p)
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            conn.execute("VACUUM")
            freed += max(0, before - db_bytes(paths))
    with _cache_lock:
        _cache.pop(paths.root, None)
    return freed


def clear_tmp(paths: DataPaths) -> None:
    """Leftovers of interrupted uploads/exports; nothing can be in flight before the app starts."""
    for p in paths.tmp.iterdir():
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        else:
            p.unlink(missing_ok=True)

"""Storage usage, cleanup and the low-disk guard."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from lab import storage
from lab.db import connect
from lab_testkit import wait_run


def test_storage_info(client: TestClient, image: dict, video: dict, data_dir: Path) -> None:
    info = client.get("/api/storage").json()
    assert info["data_dir"] == str(data_dir.resolve())
    assert 0 < info["free_bytes"] <= info["total_bytes"]
    usage = info["usage"]
    assert usage["media_bytes"] > 0 and usage["proxy_bytes"] > 0 and usage["thumbs_bytes"] > 0
    assert usage["db_bytes"] > 0


def test_cleanup_keeps_starred_runs_unless_asked(client: TestClient, image: dict, data_dir: Path) -> None:
    def run() -> dict:
        r = client.post("/api/runs", json={"media_id": image["id"], "model_key": "test.index"}).json()
        return wait_run(client, r["id"])

    keep, drop = run(), run()
    client.patch(f"/api/runs/{keep['id']}", json={"starred": True})
    client.post(f"/api/runs/{drop['id']}/export")
    conn = connect(data_dir / "lab.sqlite3")
    conn.execute("UPDATE runs SET created_at = '2026-01-01T00:00:00.000+09:00' WHERE id IN (?, ?)",
                 (keep["id"], drop["id"]))
    conn.close()

    r = client.post("/api/storage/cleanup", json={"runs_older_than_days": 30})
    assert r.status_code == 200 and r.json()["freed_bytes"] > 0
    assert client.get(f"/api/runs/{drop['id']}").status_code == 404
    assert client.get(f"/api/runs/{keep['id']}").status_code == 200, "starred runs survive by default"
    assert not (data_dir / "exports" / f"run-{drop['id']}.png").exists()

    client.post(f"/api/runs/{keep['id']}/export")
    r = client.post("/api/storage/cleanup", json={"delete_exports": True})
    assert r.json()["freed_bytes"] > 0 and not any((data_dir / "exports").iterdir())
    assert client.get(f"/api/runs/{keep['id']}").json()["export_url"] is None

    r = client.post("/api/storage/cleanup", json={"runs_older_than_days": 30, "unstarred_only": False})
    assert client.get(f"/api/runs/{keep['id']}").status_code == 404
    assert client.post("/api/storage/cleanup", json={"runs_older_than_days": -1}).status_code == 400


def test_low_disk_is_refused_with_507(client: TestClient, image: dict, monkeypatch: pytest.MonkeyPatch,
                                      media_dir: Path) -> None:
    monkeypatch.setattr(storage, "MIN_FREE_BYTES", 1 << 60)
    with open(media_dir / "cubes.png", "rb") as f:
        r = client.post("/api/media", files={"file": ("a.png", f, "image/png")})
    assert r.status_code == 507 and "디스크" in r.json()["detail"]
    r = client.post("/api/runs", json={"media_id": image["id"], "model_key": "test.index"})
    assert r.status_code == 507

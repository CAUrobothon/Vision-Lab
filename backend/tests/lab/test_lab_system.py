"""System info, presets, error format, SPA serving and the sample button."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from lab.main import create_app


def test_health_and_system(client: TestClient) -> None:
    health = client.get("/api/health").json()
    assert health["ok"] is True and set(health) == {"ok", "build", "features"}
    assert health["features"] == ["folders", "gt", "score"], "what a page may gate on instead of 404 texts"
    info = client.get("/api/system").json()
    assert info["version"] == "0.1.0"
    assert info["torch_available"] is True
    assert info["worker"]["alive"] is True and info["worker"]["pid"]
    assert info["storage"]["free_bytes"] > 0


def test_models_list_has_pipelines_first(client: TestClient) -> None:
    models = client.get("/api/models").json()
    test_model = next(m for m in models if m["key"] == "test.index")
    assert test_model["kind"] == "pipeline" and test_model["state"] == "ready"
    assert [p["name"] for p in test_model["params"]] == ["delay_ms", "proto_red", "roi"]
    kinds = [m["kind"] for m in models]
    assert kinds == sorted(kinds, key=lambda k: k != "pipeline")
    r = client.patch("/api/models/test.index", json={"name": "x"})
    assert r.status_code == 400 and r.json()["detail"] == "내장 파이프라인은 수정하거나 지울 수 없습니다"


def test_presets(client: TestClient) -> None:
    body = {"model_key": "test.index", "name": "대회장 조명", "params": {"delay_ms": 3, "junk": 1}}
    p = client.post("/api/presets", json=body).json()
    assert p["params"] == {"delay_ms": 3, "proto_red": "#c8323c", "roi": []}
    again = client.post("/api/presets", json={**body, "params": {"delay_ms": 4}}).json()
    assert again["id"] == p["id"] and again["params"]["delay_ms"] == 4, "same name overwrites"
    assert [x["name"] for x in client.get("/api/presets", params={"model_key": "test.index"}).json()] == ["대회장 조명"]
    assert client.post("/api/presets", json={**body, "model_key": "cv.none"}).status_code == 404
    assert client.post("/api/presets", json={**body, "name": ""}).status_code == 400
    assert client.delete(f"/api/presets/{p['id']}").json() == {"ok": True}
    assert client.delete(f"/api/presets/{p['id']}").status_code == 404


def test_errors_are_korean_json(client: TestClient) -> None:
    for r in (client.get("/api/does-not-exist"), client.post("/api/runs", content=b"{not json",
                                                              headers={"content-type": "application/json"}),
              client.get("/api/runs/abc")):
        assert r.status_code in (400, 404)
        detail = r.json()["detail"]
        assert isinstance(detail, str) and any("가" <= ch <= "힣" for ch in detail), detail


def test_spa_fallback(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>lab</title>")
    (dist / "assets" / "app.js").write_text("console.log(1)")
    app = create_app(tmp_path / "data", frontend_dir=dist)
    c = TestClient(app)  # no lifespan: static routes need no worker
    html = {"accept": "text/html"}
    for path in ("/", "/runs", "/compare", "/models"):
        r = c.get(path, headers=html)
        assert r.status_code == 200 and "<title>lab</title>" in r.text, path
    assert c.get("/assets/app.js").text == "console.log(1)"
    r = c.get("/api/nope", headers=html)
    assert r.status_code == 404 and r.json()["detail"].startswith("없는 API 경로")


def test_no_frontend_page(tmp_path: Path) -> None:
    c = TestClient(create_app(tmp_path / "data", frontend_dir=tmp_path / "missing"))
    r = c.get("/")
    assert r.status_code == 200 and "run.sh" in r.text


def test_samples(client: TestClient) -> None:
    try:
        from cubevision.synth import sample_assets  # noqa: F401
    except ImportError:
        pytest.xfail("cubevision.synth.sample_assets does not exist yet (written by the cubevision agent)")
    r = client.post("/api/samples")
    assert r.status_code == 200, r.text
    media = r.json()
    assert {m["kind"] for m in media} == {"image", "video"}
    assert all(m["tags"] == ["sample", "합성"] for m in media)
    again = client.post("/api/samples").json()
    assert [m["id"] for m in again] == [m["id"] for m in media], "deterministic samples are not stored twice"


def test_pages_are_revalidated_and_build_files_cached(client: TestClient) -> None:
    """A rebuilt frontend must reach the browser on the next load; hashed build files never change."""
    page = client.get("/")
    if page.status_code != 200 or "text/html" not in page.headers.get("content-type", ""):
        return  # no built frontend in this checkout
    assert page.headers["cache-control"] == "no-cache"
    routed = client.get("/runs", headers={"accept": "text/html"})  # what a browser sends
    assert routed.status_code == 200 and routed.headers["cache-control"] == "no-cache", "client-side routes too"
    asset = next(iter(__import__("re").findall(r'/assets/[^"]+\.js', page.text)), None)
    if asset:
        assert "immutable" in client.get(asset).headers["cache-control"]
    assert client.get("/api/health").json()["build"], "the build id comes from index.html"

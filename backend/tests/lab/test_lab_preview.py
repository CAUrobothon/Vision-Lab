"""POST /api/preview: cubes, resolved params, debug layers, errors, and the round-trip time."""

from __future__ import annotations

import base64
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from cubevision.registry import list_pipelines
from lab_testkit import INDEX_PIPELINE, upload

REAL_PIPELINES = [p for p in list_pipelines() if p.key != INDEX_PIPELINE.key]


def _png(url: str) -> np.ndarray:
    assert url.startswith("data:image/png;base64,")
    return cv2.imdecode(np.frombuffer(base64.b64decode(url.split(",", 1)[1]), np.uint8), cv2.IMREAD_UNCHANGED)


def test_preview_video_frame_with_debug_layers(client: TestClient, video: dict) -> None:
    r = client.post("/api/preview", json={"media_id": video["id"], "frame": 33, "model_key": "test.index",
                                          "params": {"delay_ms": "0", "unknown": 1}, "debug": True})
    assert r.status_code == 200, r.text
    res = r.json()
    assert [c["label"] for c in res["cubes"]] == ["33"]
    assert res["params"] == {"delay_ms": 0, "proto_red": "#c8323c", "roi": []}, "resolved, unknown keys dropped"
    assert res["ms"] > 0 and "decode" in res["stages_ms"] and res["notes"] == ["frame 33"]
    layers = {d["name"]: d for d in res["debug_layers"]}
    assert layers["bits"]["label"] == "번호 칸" and layers["gray"]["label"] == "회색조"
    mask, gray = _png(layers["bits"]["url"]), _png(layers["gray"]["url"])
    assert mask.shape == (240, 320), "masks are 1-channel PNGs at frame size"
    assert gray.shape == (240, 320, 3)


def test_preview_without_debug_has_no_layers(client: TestClient, image: dict) -> None:
    res = client.post("/api/preview", json={"media_id": image["id"], "model_key": "test.index"}).json()
    assert res["debug_layers"] == [] and res["frame"] == 0


def test_preview_roi_is_applied_by_the_pipeline(client: TestClient, image: dict) -> None:
    roi = [{"mode": "exclude", "points": [[100, 60], [220, 60], [220, 180], [100, 180]]}]
    res = client.post("/api/preview", json={"media_id": image["id"], "model_key": "test.index",
                                            "params": {"roi": roi}}).json()
    assert res["cubes"] == []


def test_preview_errors(client: TestClient, image: dict, video: dict) -> None:
    def post(**body):
        return client.post("/api/preview", json={"media_id": image["id"], "model_key": "test.index", **body})

    r = post(model_key="w.1")
    assert r.status_code == 400 and r.json()["detail"] == "가중치 모델은 미리보기 대신 실행을 사용하세요"
    assert post(model_key="cv.none").status_code == 404
    assert post(media_id=99999).status_code == 404
    assert post(frame=1).status_code == 400, "images only have frame 0"
    r = post(params={"proto_red": "red"})
    assert r.status_code == 400 and "proto_red" in r.json()["detail"]
    r = client.post("/api/preview", json={"media_id": video["id"], "frame": 10_000, "model_key": "test.index"})
    assert r.status_code == 400


def _roundtrip_ms(client: TestClient, body: dict, n: int = 25) -> float:
    for _ in range(3):  # warm the frame cache and the pipeline
        assert client.post("/api/preview", json=body).status_code == 200
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        r = client.post("/api/preview", json=body)
        times.append((time.perf_counter() - t0) * 1000)
        assert r.status_code == 200
    return statistics.median(times)


def test_preview_overhead_is_small(client: TestClient, media_dir: Path) -> None:
    """The lab's own share of a 1280x720 preview (decode cache, JSON, threadpool) stays well under 60 ms."""
    m = upload(client, media_dir / "scene-1280.png")["media"]
    ms = _roundtrip_ms(client, {"media_id": m["id"], "model_key": "test.index"})
    print(f"\npreview round trip, test pipeline, 1280x720: median {ms:.1f} ms")
    assert ms < 60


@pytest.mark.skipif(not REAL_PIPELINES, reason="cubevision has no registered pipelines yet")
@pytest.mark.parametrize("key", [p.key for p in REAL_PIPELINES])
def test_every_real_pipeline_previews(client: TestClient, media_dir: Path, key: str) -> None:
    m = upload(client, media_dir / "scene-1280.png")["media"]
    r = client.post("/api/preview", json={"media_id": m["id"], "model_key": key, "debug": True})
    assert r.status_code == 200, r.text
    res = r.json()
    for layer in res["debug_layers"]:
        assert _png(layer["url"]).shape[:2] == (720, 1280)
    ms = _roundtrip_ms(client, {"media_id": m["id"], "model_key": key}, n=10)
    print(f"\npreview round trip, {key}, 1280x720: median {ms:.1f} ms")

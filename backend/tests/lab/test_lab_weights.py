"""Uploaded Ultralytics weights: inspection in a subprocess, runs in the worker, edit/delete.

Uses random-weight checkpoints built offline from the bundled YOLO26 yaml files, so the
detections are meaningless; the plumbing (task handling, conversion to cubes, devices,
memory hand-over between models) is what is tested.

Ordering matters for memory on a 16 GB machine: every checkpoint is uploaded (= inspected in
a short-lived subprocess) before the first run makes the worker import torch, so two torch
processes never overlap. This module is named so that it runs last.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from lab.db import connect, now_iso
from lab_testkit import upload, wait_run

FILES = ["seg.pt", "obb.pt", "detect.pt", "cls.pt", "garbage.pt", "plain.pt"]
CUBE_KEYS = {"color", "score", "color_score", "center", "quad", "angle", "area", "polygon", "box", "label", "flags"}


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("checkpoints")
    script = Path(__file__).with_name("build_checkpoints.py")
    proc = subprocess.run([sys.executable, str(script), str(out)], capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        pytest.skip(f"could not build checkpoints offline: {proc.stderr[-500:]}")
    return out


@pytest.fixture(scope="module")
def uploaded(client: TestClient, checkpoints: Path) -> dict[str, dict]:
    result = {}
    for name in FILES:
        with open(checkpoints / name, "rb") as f:
            r = client.post("/api/models/weights", files={"file": (name, f, "application/octet-stream")},
                            data={"name": f"테스트 {name}"})
        assert r.status_code == 200, r.text
        result[name] = r.json()
    return result


def test_inspection(uploaded: dict[str, dict]) -> None:
    seg, obb, det = uploaded["seg.pt"], uploaded["obb.pt"], uploaded["detect.pt"]
    for m, task in ((seg, "segment"), (obb, "obb"), (det, "detect")):
        assert m["state"] == "ready" and m["error"] is None, m
        assert m["task"] == task and m["kind"] == "weights" and m["key"].startswith("w.")
        assert m["class_names"][:4] == ["red", "green", "blue", "wood"] and len(m["class_names"]) == 80
        assert m["file_bytes"] > 1_000_000 and m["name"].startswith("테스트 ")
    params = {p["name"]: p for p in seg["params"]}
    assert params["imgsz"]["default"] == 640, "train imgsz becomes the default"
    assert params["retina_masks"]["default"] is True and params["roi"]["group"] == "검출 영역"
    assert params["proto_red"]["type"] == "color"
    obb_params = {p["name"]: p for p in obb["params"]}
    assert obb_params["imgsz"]["default"] == 1280 and "retina_masks" not in obb_params
    for name in ("cls.pt", "garbage.pt", "plain.pt"):
        m = uploaded[name]
        assert m["state"] == "error" and m["params"] == [], m
        assert any("가" <= ch <= "힣" for ch in m["error"]), m["error"]
    assert "지원하지 않는 작업" in uploaded["cls.pt"]["error"]


def test_upload_rules(client: TestClient, uploaded: dict[str, dict], checkpoints: Path) -> None:
    r = client.post("/api/models/weights", files={"file": ("model.onnx", b"onnx", "application/octet-stream")})
    assert r.status_code == 415
    with open(checkpoints / "obb.pt", "rb") as f:
        again = client.post("/api/models/weights", files={"file": ("copy.pt", f, "application/octet-stream")}).json()
    assert again["key"] == uploaded["obb.pt"]["key"], "the same file is registered once"
    keys = [m["key"] for m in client.get("/api/models").json()]
    assert all(m["key"] in keys for m in uploaded.values())


def test_preview_is_refused_for_weights(client: TestClient, uploaded: dict[str, dict], image: dict) -> None:
    r = client.post("/api/preview", json={"media_id": image["id"], "model_key": uploaded["seg.pt"]["key"]})
    assert r.status_code == 400


def test_error_weights_cannot_run(client: TestClient, uploaded: dict[str, dict], image: dict) -> None:
    r = client.post("/api/runs", json={"media_id": image["id"], "model_key": uploaded["garbage.pt"]["key"]})
    assert r.status_code == 400


@pytest.mark.parametrize("name", ["seg.pt", "obb.pt", "detect.pt"])
def test_run_each_task(client: TestClient, uploaded: dict[str, dict], media_dir: Path, name: str) -> None:
    scene = upload(client, media_dir / "scene-1280.png")["media"]
    run = client.post("/api/runs", json={"media_id": scene["id"], "model_key": uploaded[name]["key"],
                                         "params": {"imgsz": "640", "color_source": "auto"}}).json()
    assert run["params"]["imgsz"] == 640 and run["params"]["device"] == "auto"
    run = wait_run(client, run["id"], timeout=240)
    assert run["status"] == "done", run["error"]
    assert run["device"] in ("mps", "cpu")
    frames = client.get(f"/api/runs/{run['id']}/frames").json()["frames"]
    assert len(frames) == 1
    cubes = frames[0]["cubes"]
    # The test checkpoints fire class 0 ("red") everywhere with random geometry (build_checkpoints.py).
    if name == "obb.pt":
        assert cubes, "cube_from_quad always returns a cube"
    for cube in cubes:
        assert set(cube) == CUBE_KEYS and cube["label"] == "red"
        assert cube["color"] in ("red", "green", "blue", "wood", "unknown")
    print(f"\n{name}: {len(cubes)} random cubes, first frame {run['timing']['first_ms']:.0f} ms on {run['device']}")
    worker = client.get("/api/system").json()
    assert worker["mps_available"] in (True, False)
    assert worker["worker"]["loaded_model"].startswith(f"테스트 {name}")


def test_video_run_with_weights_on_cpu(client: TestClient, uploaded: dict[str, dict], video: dict) -> None:
    run = client.post("/api/runs", json={"media_id": video["id"], "model_key": uploaded["obb.pt"]["key"], "stride": 25,
                                         "params": {"imgsz": "640", "device": "cpu", "color_source": "pixels"}}).json()
    run = wait_run(client, run["id"], timeout=240)
    assert run["status"] == "done", run["error"]
    assert run["device"] == "cpu" and run["frames_done"] == 3
    t = run["timing"]
    assert t["frames"] == 3 and t["first_ms"] > 0
    frames = client.get(f"/api/runs/{run['id']}/frames").json()["frames"]
    assert [f["frame"] for f in frames] == [0, 25, 50]


def test_patch_and_delete_weights(client: TestClient, uploaded: dict[str, dict], image: dict, data_dir: Path) -> None:
    key = uploaded["detect.pt"]["key"]
    m = client.patch(f"/api/models/{key}", json={"name": "검출기", "note": "임의 가중치"}).json()
    assert (m["name"], m["note"]) == ("검출기", "임의 가중치")
    assert client.delete(f"/api/models/{key}").json() == {"ok": True}
    assert key not in [x["key"] for x in client.get("/api/models").json()]
    assert not (data_dir / "weights" / f"{key[2:]}.pt").exists()
    assert client.post("/api/runs", json={"media_id": image["id"], "model_key": key}).status_code == 404
    # A run queued before its weights were deleted fails with a clear message.
    conn = connect(data_dir / "lab.sqlite3")
    rid = conn.execute(
        "INSERT INTO runs (media_id, model_key, model_name, params, frame_start, frame_end, stride, frames_total, "
        "status, created_at) VALUES (?, ?, '검출기', '{}', 0, 1, 1, 1, 'queued', ?)", (image["id"], key, now_iso())
    ).lastrowid
    conn.close()
    run = wait_run(client, rid)
    assert run["status"] == "failed" and run["error"] == "가중치가 삭제되어 실행할 수 없습니다"
    assert run["model_name"] == "검출기"

"""Baked exports: PNG for image runs, H.264 MP4 for video runs."""

from __future__ import annotations

import cv2
import numpy as np
from fastapi.testclient import TestClient

from lab_testkit import decode_index, ffprobe_json, wait_frames, wait_run


def test_export_image_png(client: TestClient, image: dict) -> None:
    run = wait_run(client, client.post("/api/runs", json={"media_id": image["id"], "model_key": "test.index"}).json()["id"])
    assert run["export_url"] is None
    info = client.post(f"/api/runs/{run['id']}/export").json()
    assert info["kind"] == "image" and info["bytes"] > 0
    assert client.get(f"/api/runs/{run['id']}").json()["export_url"] == info["url"]
    r = client.get(info["url"])
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert "attachment" in r.headers["content-disposition"]
    png = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    assert png.shape == (240, 320, 3)
    original = cv2.imdecode(np.frombuffer(client.get(image["file_url"]).content, np.uint8), cv2.IMREAD_COLOR)
    assert np.abs(png.astype(int) - original.astype(int)).sum() > 0, "the cube overlay was drawn"


def test_export_video_mp4(client: TestClient, video: dict, tmp_path) -> None:
    body = {"media_id": video["id"], "model_key": "test.index", "frame_start": 10, "frame_end": 40, "stride": 4}
    run = wait_run(client, client.post("/api/runs", json=body).json()["id"])
    last = max(f["frame"] for f in client.get(f"/api/runs/{run['id']}/frames").json()["frames"])
    assert last == 38
    info = client.post(f"/api/runs/{run['id']}/export").json()
    assert info["kind"] == "video"
    out = tmp_path / "export.mp4"
    out.write_bytes(client.get(info["url"]).content)
    s = ffprobe_json(out, "-show_streams", "-count_packets")["streams"][0]
    assert s["codec_name"] == "h264" and s["pix_fmt"] == "yuv420p"
    assert s["avg_frame_rate"] == "25/1"
    assert int(s["nb_read_packets"]) == last - 10 + 1, "every proxy frame from frame_start to the last processed one"
    assert (int(s["width"]), int(s["height"])) == (320, 240)
    cap = cv2.VideoCapture(str(out))
    ok, first = cap.read()
    cap.release()
    assert ok and decode_index(first) == 10


def test_export_refused_while_running(client: TestClient, video: dict) -> None:
    slow = client.post("/api/runs", json={"media_id": video["id"], "model_key": "test.index",
                                          "params": {"delay_ms": 40}}).json()
    wait_frames(client, slow["id"], 1)
    r = client.post(f"/api/runs/{slow['id']}/export")
    assert r.status_code == 409 and "끝난 뒤" in r.json()["detail"]
    client.post(f"/api/runs/{slow['id']}/cancel")
    wait_run(client, slow["id"])
    assert client.get(f"/api/runs/{slow['id']}/export").status_code == 404
    assert client.post(f"/api/runs/{slow['id']}/export").status_code == 200, "a cancelled run with frames can export"

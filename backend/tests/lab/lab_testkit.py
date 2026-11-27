"""Test media whose frames encode their own index, and a deterministic test-only pipeline.

The index is drawn as 10 black/white 16 px blocks along the top-left of each frame, so any
decoded frame can say which frame it is. ``IndexPipeline`` reads it back and reports it as the
cube label: a run's stored frame numbers can then be checked against the pictures the worker
actually decoded.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi.testclient import TestClient

from cubevision.pipelines.base import Pipeline
from cubevision.roi import filter_cubes
from cubevision.types import Cube, FrameResult, ParamSpec

VIDEO_FRAMES = 75  # the session's test video: 3 s at 25 fps
BITS = 10
BLOCK = 16
ORIGIN = 8


def encode_index(frame: np.ndarray, index: int) -> None:
    for b in range(BITS):
        x = ORIGIN + b * BLOCK
        on = (index >> b) & 1
        cv2.rectangle(frame, (x, ORIGIN), (x + BLOCK - 1, ORIGIN + BLOCK - 1), (255, 255, 255) if on else (0, 0, 0), -1)


def decode_index(frame: np.ndarray) -> int:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    value = 0
    for b in range(BITS):
        x = ORIGIN + b * BLOCK + BLOCK // 2
        y = ORIGIN + BLOCK // 2
        if gray[y - 3 : y + 4, x - 3 : x + 4].mean() > 127:
            value |= 1 << b
    return value


def index_frame(index: int, size: tuple[int, int] = (320, 240)) -> np.ndarray:
    w, h = size
    frame = np.full((h, w, 3), 128, np.uint8)
    x = 40 + (index * 3) % max(1, w - 100)
    cv2.rectangle(frame, (x, h // 2), (x + 30, h // 2 + 30), (40, 40, 200), -1)
    encode_index(frame, index)
    return frame


def write_index_video(path: Path, n_frames: int, size: tuple[int, int] = (320, 240), fps: float = 25.0) -> Path:
    """mp4v via cv2.VideoWriter, like cubevision.synth writes its sample (the lab must transcode it)."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    assert writer.isOpened()
    for i in range(n_frames):
        writer.write(index_frame(i, size))
    writer.release()
    return path


def ffprobe_json(path: Path, *args: str) -> dict:
    import json

    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", *args, str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    return json.loads(out)


class IndexPipeline(Pipeline):
    """Returns one red cube in the frame centre labelled with the frame's encoded index."""

    key = "test.index"
    name = "테스트: 프레임 번호"
    summary = "테스트 전용"
    description = "프레임에 그려진 번호를 읽어 큐브 라벨로 돌려줍니다."
    params = [
        ParamSpec("delay_ms", "지연", "int", 0, 0, 20000, 1, help="느린 모델 흉내"),
        ParamSpec("proto_red", "빨강 기준색", "color", "#c8323c"),
        ParamSpec("roi", "검출 영역", "roi", [], group="검출 영역"),
    ]
    debug_labels = {"bits": "번호 칸", "gray": "회색조"}

    def run(self, frame_bgr: np.ndarray, params: dict[str, Any], *, debug: bool = False) -> FrameResult:
        t0 = time.perf_counter()
        if params["delay_ms"]:
            time.sleep(params["delay_ms"] / 1000.0)
        index = decode_index(frame_bgr)
        h, w = frame_bgr.shape[:2]
        cx, cy = w / 2, h / 2
        quad = [(cx - 10, cy - 10), (cx + 10, cy - 10), (cx + 10, cy + 10), (cx - 10, cy + 10)]
        cube = Cube(color="red", score=1.0, center=(cx, cy), quad=quad, angle=0.0, area=400.0, label=str(index))
        layers: dict[str, np.ndarray] = {}
        if debug:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            layers = {"bits": (gray > 127).astype(np.uint8) * 255, "gray": cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)}
        ms = (time.perf_counter() - t0) * 1000
        return FrameResult(cubes=filter_cubes([cube], params["roi"]), ms=ms, stages_ms={"decode": ms}, debug=layers,
                           notes=[f"frame {index}"])


INDEX_PIPELINE = IndexPipeline()


# HTTP helpers ------------------------------------------------------------------------------


def upload(client: TestClient, path: Path, name: str | None = None) -> dict:
    with open(path, "rb") as f:
        r = client.post("/api/media", files={"file": (name or path.name, f, "application/octet-stream")})
    assert r.status_code == 200, r.text
    return r.json()


def wait_run(client: TestClient, run_id: int, timeout: float = 60.0, until=("done", "failed", "cancelled")) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        run = client.get(f"/api/runs/{run_id}").json()
        if run["status"] in until:
            return run
        if time.monotonic() > deadline:
            raise AssertionError(f"run {run_id} still {run['status']} after {timeout} s")
        time.sleep(0.05)


def wait_frames(client: TestClient, run_id: int, at_least: int, timeout: float = 30.0) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        run = client.get(f"/api/runs/{run_id}").json()
        if run["frames_done"] >= at_least or run["status"] not in ("queued", "running"):
            return run
        if time.monotonic() > deadline:
            raise AssertionError(f"run {run_id} has {run['frames_done']} frames after {timeout} s")
        time.sleep(0.05)

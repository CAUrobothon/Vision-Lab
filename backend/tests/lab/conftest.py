"""Shared fixtures: one app (with its real worker process) per test session, on a tmp data dir.

The data dir name contains Korean on purpose: the real project path does too.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from lab.main import create_app
from lab_testkit import INDEX_PIPELINE, VIDEO_FRAMES, index_frame, upload, write_index_video


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("랩 데이터")


@pytest.fixture(scope="session")
def media_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Source files for uploads (outside the data dir)."""
    d = tmp_path_factory.mktemp("media")
    img = index_frame(0, (320, 240))
    cv2.imwrite(str(d / "cubes.png"), img)
    cv2.imwrite(str(d / "cubes.tif"), img)
    big = np.full((720, 1280, 3), 230, np.uint8)
    for i, colour in enumerate([(40, 40, 200), (60, 150, 60), (180, 80, 40)]):
        cv2.rectangle(big, (300 + i * 200, 300), (361 + i * 200, 361), colour, -1)
    cv2.imwrite(str(d / "scene-1280.png"), big)
    # Odd size on purpose: the proxy must round down to even dimensions.
    write_index_video(d / "index.mp4", VIDEO_FRAMES, size=(321, 241), fps=25)
    # Same clip as a .mov with an audio track and a 90° display rotation.
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-display_rotation", "90", "-i", str(d / "index.mp4"),
         "-f", "lavfi", "-i", "sine=frequency=440:duration=3", "-map", "0:v", "-map", "1:a",
         "-c:v", "copy", "-c:a", "aac", "-shortest", str(d / "rotated.mov")],
        check=True,
    )
    # 29.97 fps H.264 with B-frames: the proxy must become 30 fps CFR without B-frames.
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
         "-i", "testsrc2=size=160x120:rate=30000/1001", "-t", "1", "-c:v", "libx264", "-bf", "2",
         "-pix_fmt", "yuv420p", str(d / "ntsc.mp4")],
        check=True,
    )
    (d / "notes.txt").write_text("이건 이미지가 아닙니다")
    (d / "broken.jpg").write_bytes(b"\xff\xd8\xff\xe0 not really a jpeg")
    return d


@pytest.fixture(scope="session")
def client(data_dir: Path) -> Iterator[TestClient]:
    app = create_app(data_dir, extra_pipelines=[INDEX_PIPELINE])
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture(scope="session")
def image(client: TestClient, media_dir: Path) -> dict:
    return upload(client, media_dir / "cubes.png")["media"]


@pytest.fixture(scope="session")
def video(client: TestClient, media_dir: Path) -> dict:
    return upload(client, media_dir / "index.mp4")["media"]

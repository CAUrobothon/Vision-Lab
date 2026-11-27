"""Frame access is exact: frame n decoded on the server shows the picture of frame n."""

from __future__ import annotations

import random
from pathlib import Path

from lab import frames
from lab.config import DataPaths
from lab.db import connect
from lab.media import frame_source, get_row
from lab_testkit import VIDEO_FRAMES, decode_index


def _source(data_dir: Path, media_id: int) -> frames.FrameSource:
    conn = connect(DataPaths(data_dir).db)
    try:
        return frame_source(DataPaths(data_dir), get_row(conn, media_id))
    finally:
        conn.close()


def test_random_seeks_are_exact(video: dict, data_dir: Path) -> None:
    src = _source(data_dir, video["id"])
    order = list(range(VIDEO_FRAMES))
    random.Random(7).shuffle(order)
    order += [0, 1, 2, 3, 74, 73, 15, 14, 16, 30]  # forward steps, backward steps, keyframe edges
    for n in order:
        assert decode_index(frames.get_frame(src, n)) == n


def test_get_frame_returns_a_private_copy(video: dict, data_dir: Path) -> None:
    src = _source(data_dir, video["id"])
    a = frames.get_frame(src, 5)
    a[:] = 0
    assert decode_index(frames.get_frame(src, 5)) == 5


def test_iter_frames_with_start_and_stride(video: dict, data_dir: Path) -> None:
    src = _source(data_dir, video["id"])
    got = [(i, decode_index(f)) for i, f in frames.iter_frames(src, 7, 60, 4)]
    assert [i for i, _ in got] == list(range(7, 60, 4))
    assert all(i == d for i, d in got)
    # end past the last frame stops at the last frame
    assert [i for i, _ in frames.iter_frames(src, 70, 500, 2)] == [70, 72, 74]


def test_image_source(image: dict, data_dir: Path) -> None:
    src = _source(data_dir, image["id"])
    assert decode_index(frames.get_frame(src, 0)) == 0
    assert [i for i, _ in frames.iter_frames(src, 0, 1)] == [0]

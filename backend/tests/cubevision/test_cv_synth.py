"""Synthetic scenes: determinism, ground truth and the lab's sample assets."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pytest

from cubevision.postprocess import order_quad
from cubevision.synth import make_scene, sample_assets


def test_same_seed_same_scene() -> None:
    a_img, a_gt = make_scene(7, mission="B")
    b_img, b_gt = make_scene(7, mission="B")
    assert np.array_equal(a_img, b_img)
    assert a_gt == b_gt
    c_img, _ = make_scene(8, mission="B")
    assert not np.array_equal(a_img, c_img)


@pytest.mark.parametrize("mission", ["A", "B"])
def test_ground_truth_fields(mission: str) -> None:
    img, gt = make_scene(5, mission=mission, arm=False)  # type: ignore[arg-type]
    assert img.shape == (720, 1280, 3) and img.dtype == np.uint8
    assert gt
    for g in gt:
        q = [tuple(p) for p in g["quad"]]
        assert np.allclose(order_quad(q), q)
        assert 0.0 <= g["angle"] < 90.0
        assert g["center"] == pytest.approx(tuple(np.mean(q, axis=0)), abs=1e-2)
        assert 0.0 < g["visible"] <= 1.0
    colours = Counter(g["color"] for g in gt)
    if mission == "B":
        assert colours == {"red": 3, "green": 3, "blue": 3, "wood": 3}
    else:
        assert set(colours) <= {"red", "green", "blue"} and sum(colours.values()) <= 9


def test_size_scales_the_camera() -> None:
    img, gt = make_scene(5, size=(640, 360), tower=0, arm=False)
    assert img.shape == (360, 640, 3)
    assert all(abs(g["side_px"] - 30.6) < 0.1 for g in gt)


def test_explicit_cubes_and_tower_visibility() -> None:
    cubes = [dict(color="red", x=-20.0, y=150.0, angle=0.0, level=k) for k in (1, 2, 3)]
    _, gt = make_scene(0, cubes=cubes, arm=False)
    assert [g["level"] for g in gt] == [3]  # lower levels are hidden under the top cube


def test_sample_assets(tmp_path: Path) -> None:
    paths = sample_assets(tmp_path / "out")
    assert [p.suffix for p in paths] == [".jpg", ".jpg", ".mp4"]
    for p in paths[:2]:
        img = cv2.imread(str(p))
        assert img is not None and img.shape == (720, 1280, 3)
    cap = cv2.VideoCapture(str(paths[2]))
    try:
        assert cap.isOpened()
        assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(30.0)
        n = 0
        while cap.grab():
            n += 1
    finally:
        cap.release()
    assert 140 <= n <= 160  # about 5 s at 30 fps

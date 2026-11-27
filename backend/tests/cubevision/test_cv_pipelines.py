"""Behaviour of single pipelines on layouts the shared benchmark rarely draws (all synthetic)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cubevision.bench import match
from cubevision.pipelines.base import Pipeline
from cubevision.pipelines.hsv_ranges import _hue_band
from cubevision.postprocess import angle_diff
from cubevision.registry import get_pipeline, list_pipelines
from cubevision.synth import CLEAN, make_scene

L3 = [(0, 0), (1, 0), (0, 1)]
ROW3 = [(0, 0), (1, 0), (2, 0)]
BLOCK = [(0, 0), (1, 0), (0, 1), (1, 1)]


def flush(x: float, y: float, angle: float, offsets: list[tuple[int, int]], color: str) -> list[dict]:
    """Cubes touching face to face (25 mm apart), all turned by ``angle``, in arena mm."""
    a = math.radians(angle)
    u, v = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
    return [dict(color=color, x=x + 25 * (i * u[0] + j * v[0]), y=y + 25 * (i * u[1] + j * v[1]), angle=angle) for i, j in offsets]


@pytest.mark.parametrize("h, hues", [
    ([10, 30], set(range(10, 31))),
    ([-12, 10], set(range(168, 180)) | set(range(0, 11))),
    ([-40, 0], set(range(140, 180)) | {0}),
    ([-20, -5], set(range(160, 176))),  # used to let hue 0 through as well
])
def test_hue_band_for_every_sign_combination(h: list[int], hues: set[int]) -> None:
    hsv = np.zeros((1, 180, 3), np.uint8)
    hsv[0, :, 0] = np.arange(180)
    hsv[0, :, 1:] = 200
    m = _hue_band(hsv, h, [90, 255], [60, 255])[0]
    assert {i for i in range(180) if m[i]} == hues


@pytest.mark.parametrize("offsets", [BLOCK, ROW3], ids=["2x2", "row3"])
@pytest.mark.parametrize("angle", [0.0, 30.0, 45.0, 60.0])
def test_edge_quads_keeps_turned_same_colour_groups(offsets: list[tuple[int, int]], angle: float) -> None:
    """The size gate measured the upright bounding box, so a 2x2 block turned 30-60 degrees
    (2.8 faces wide upright) and every row of three were dropped whole."""
    img, gt = make_scene(4, mission="A", cubes=flush(-120.0, 130.0, angle, offsets, "blue"), **{**CLEAN, "paper": "all"})
    res = get_pipeline("cv.edge_quads")(img)
    assert len(gt) == len(offsets)
    assert len(res.cubes) == len(gt)
    assert len(match(res.cubes, gt, 3.0)) == len(gt)


@pytest.mark.parametrize("p", list_pipelines(), ids=lambda p: p.key)
@pytest.mark.parametrize("x, y, angle", [(-120.0, 130.0, 30.0), (150.0, 200.0, 0.0), (-200.0, 60.0, 0.0)])
def test_l_of_three_with_paper_on_the_top_only(p: Pipeline, x: float, y: float, angle: float) -> None:
    """With plain wooden sides the blob holds no side band; the count assumed one and found two
    cubes, one about 30 px off (62 of 96 such cases)."""
    img, gt = make_scene(4, mission="A", cubes=flush(x, y, angle, L3, "red"), **{**CLEAN, "paper": "top"})
    res = p(img)
    assert len(res.cubes) == len(gt) == 3
    pairs = match(res.cubes, gt, 2.0)
    assert len(pairs) == 3
    for i, j in pairs:
        assert abs(angle_diff(res.cubes[i].angle, gt[j]["angle"])) < 1.0

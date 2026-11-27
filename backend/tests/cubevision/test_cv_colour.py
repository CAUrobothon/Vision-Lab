"""Colour classification: prototypes, board-relative adaptation, board / shadow / dark rejection."""

from __future__ import annotations

import numpy as np
import pytest

from cubevision.colour import (
    DEFAULT_PROTOTYPES, board_colour, classify_pixels, classify_region, colour_model, hex_to_bgr,
)
from cubevision.synth import CLEAN, make_scene


@pytest.mark.parametrize("name", list(DEFAULT_PROTOTYPES))
def test_prototype_colour_classifies_as_itself(name: str) -> None:
    px = np.tile(np.array(hex_to_bgr(DEFAULT_PROTOTYPES[name]), np.uint8), (50, 1))
    colour, score = classify_pixels(px)
    assert colour == name and score > 0.5


def test_board_shadow_and_dark_grey_are_not_cube_colours() -> None:
    model = colour_model(None)
    for bgr in ((224, 228, 228), (150, 152, 152), (40, 42, 46), (60, 66, 72)):  # board, shadow, arm, warm table
        assert model.classify(_lab(bgr))[0] == "none", bgr
    assert model.classify(_lab((170, 80, 44)))[0] == "blue"  # printed blue is dark but strongly chromatic


def _lab(bgr: tuple[int, int, int]) -> np.ndarray:
    from cubevision.colour import bgr_to_lab

    return bgr_to_lab(np.array([bgr], np.float32))[0]


def test_unknown_when_far_from_every_prototype() -> None:
    purple = np.tile(np.array((200, 40, 200), np.uint8), (10, 1))
    assert classify_pixels(purple, max_dist=15)[0] == "unknown"


@pytest.mark.parametrize("cast", [(1.0, 1.0, 1.0), (0.80, 0.92, 1.10), (1.12, 1.0, 0.82), (0.7, 0.7, 0.7)])
def test_classify_region_survives_white_balance_and_exposure(cast: tuple[float, float, float]) -> None:
    cubes = [dict(color=c, x=-200.0 + 60 * i, y=150.0, angle=10.0) for i, c in enumerate(("red", "green", "blue", "wood"))]
    img, gt = make_scene(1, cubes=cubes, **CLEAN)
    cast_img = np.clip(img.astype(np.float32) * np.array(cast, np.float32), 0, 255).astype(np.uint8)
    for g in gt:
        m = np.zeros(img.shape[:2], np.uint8)
        x, y = (int(round(v)) for v in g["center"])
        m[y - 15 : y + 15, x - 15 : x + 15] = 255
        assert classify_region(cast_img, m)[0] == g["color"], (cast, g["color"])


def test_board_colour_is_measured_and_drives_the_gains() -> None:
    img, _ = make_scene(2, mission="B", **CLEAN)
    board = board_colour(img)
    assert board is not None and np.all(board > 200)
    model = colour_model(img)
    assert np.all(np.abs(model.gains - 1.0) < 0.15)
    warm = np.clip(img.astype(np.float32) * np.array([0.85, 1.0, 1.1], np.float32), 0, 255).astype(np.uint8)
    gains = colour_model(warm).gains  # frame -> reference: undo less blue and more red
    assert gains[0] > model.gains[0] and gains[2] < model.gains[2]


def test_label_lut() -> None:
    model = colour_model(None)
    lut = model.label_lut()
    assert lut.shape == (256, 256) and lut.dtype == np.uint8
    a0, b0 = (int(round(v + 128)) for v in model.board_lab[1:])
    assert lut[a0, b0] == 0  # the board itself is no class
    r = model.lab[model.names.index("red")]
    assert lut[int(round(r[1] + 128)), int(round(r[2] + 128))] == model.names.index("red") + 1

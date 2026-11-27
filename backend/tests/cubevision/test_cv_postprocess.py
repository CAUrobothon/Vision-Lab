"""Geometry and post-processing: quads, angles, splitting, top-face isolation, contract entry points."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from cubevision.postprocess import (
    FACE_PX, Geometry, angle_diff, blobs_from_mask, coarse_angle, cube_from_box, cube_from_mask,
    cube_from_polygon, cube_from_quad, fit_blob, order_quad, quad_angle,
)
from cubevision.synth import CLEAN, make_scene


def square(cx: float, cy: float, side: float, angle_deg: float) -> np.ndarray:
    a = math.radians(angle_deg)
    u, v = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
    h = side / 2
    return np.array([(cx, cy) + s * h * u + t * h * v for s, t in ((-1, -1), (1, -1), (1, 1), (-1, 1))])


def raster(polys: list[np.ndarray], shape: tuple[int, int] = (720, 1280)) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    for p in polys:
        cv2.fillPoly(m, [np.round(p * 16).astype(np.int32).reshape(-1, 1, 2)], 255, cv2.LINE_8, shift=4)
    return m


def colour_mask(img: np.ndarray) -> np.ndarray:
    """Saturated, bright pixels: the cube faces of a clean render (board and table excluded)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return (((hsv[..., 1] > 70) & (hsv[..., 2] > 50)) * 255).astype(np.uint8)


# --- quads and angles ----------------------------------------------------------------------
def test_order_quad_is_clockwise_from_smallest_x_plus_y() -> None:
    q = [(10, 10), (20, 10), (20, 20), (10, 20)]
    for perm in ([2, 0, 3, 1], [3, 2, 1, 0], [1, 3, 0, 2]):
        assert order_quad([q[i] for i in perm]) == [(10.0, 10.0), (20.0, 10.0), (20.0, 20.0), (10.0, 20.0)]
    with pytest.raises(ValueError):
        order_quad(q[:3])


@pytest.mark.parametrize("angle", [0.0, 0.3, 12.0, 44.9, 45.0, 67.5, 89.7])
def test_quad_angle_and_wraparound(angle: float) -> None:
    a = quad_angle(square(300, 200, 61, angle))
    assert 0.0 <= a < 90.0
    assert abs(angle_diff(a, angle)) < 1e-6
    # 89.7 and 0.3 are 0.6 degrees apart, not 89.4
    assert abs(angle_diff(quad_angle(square(0, 0, 61, 89.7)), quad_angle(square(0, 0, 61, 0.3)))) == pytest.approx(0.6, abs=1e-6)


def test_quad_angle_of_an_almost_upright_square_is_not_90() -> None:
    """One edge tilted by 2e-14 px makes the mean angle a tiny negative number, and
    -1e-15 % 90.0 rounds to 90.0 itself (203 of 20000 random upright quads did this)."""
    q = [(858.7192434823833, 91.58071794666134), (919.7192434823833, 91.58071794666132),
         (919.7192434823833, 152.58071794666134), (858.7192434823833, 152.58071794666134)]
    a = quad_angle(q)
    assert 0.0 <= a < 90.0
    assert abs(angle_diff(a, 0.0)) < 1e-9


@pytest.mark.parametrize("angle", [0.0, 0.8, 30.0, 89.2])
def test_coarse_angle_of_rasterised_square(angle: float) -> None:
    # Pixel staircases cost about 1 degree near the axes even on a clean raster.
    m = raster([square(400, 300, 61, angle)])
    ys, xs = np.nonzero(m)
    assert abs(angle_diff(coarse_angle(np.column_stack([xs, ys])), angle)) < 1.5


# --- gates and splitting -------------------------------------------------------------------
def fit_mask(m: np.ndarray, geo: Geometry | None = None):
    blobs = blobs_from_mask(m)
    assert len(blobs) == 1
    return fit_blob(blobs[0], geo or Geometry(), m.shape)


def test_noise_sliver_and_large_region_are_dropped() -> None:
    assert fit_mask(raster([square(300, 300, 15, 0)]))[1] == "noise"
    strip = np.array([(300, 300), (361, 300), (361, 328), (300, 328)], float)  # a 28 px side band
    assert fit_mask(raster([strip]))[1] == "sliver"
    assert fit_mask(raster([square(500, 360, 200, 10)]))[1] == "large"


@pytest.mark.parametrize("angle", [0.0, 25.0, 70.0])
def test_flush_pair_is_tiled_into_two_cubes(angle: float) -> None:
    a = math.radians(angle)
    c0 = np.array([640.0, 360.0])  # at the principal point: no side band to confuse things
    c1 = c0 + FACE_PX * np.array([math.cos(a), math.sin(a)])
    fits, status = fit_mask(raster([square(*c0, FACE_PX, angle), square(*c1, FACE_PX, angle)]))
    assert status == "block" and len(fits) == 2
    for truth in (c0, c1):
        assert min(math.dist(f.center, truth) for f in fits) < 2.0


def test_two_by_two_block_and_offset_pair() -> None:
    u, v = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    c = np.array([640.0, 360.0])
    centres = [c, c + FACE_PX * u, c + FACE_PX * v, c + FACE_PX * (u + v)]
    fits, status = fit_mask(raster([square(*p, FACE_PX, 0) for p in centres]))
    assert status == "block" and len(fits) == 4
    for truth in centres:
        assert min(math.dist(f.center, truth) for f in fits) < 1.5

    shifted = [c, c + FACE_PX * u + 0.5 * FACE_PX * v]  # flush, shifted half a face
    fits, status = fit_mask(raster([square(*p, FACE_PX, 0) for p in shifted]))
    assert status == "split" and len(fits) == 2
    for truth in shifted:
        assert min(math.dist(f.center, truth) for f in fits) < 4.0


@pytest.mark.parametrize("angle", [20.0, 67.7])
def test_shifted_pair_at_an_angle_keeps_its_angle(angle: float) -> None:
    """The rectangle around two flush cubes shifted by half a face is tilted; the outline is not."""
    a = math.radians(angle)
    u, v = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
    c0 = np.array([400.0, 250.0])
    c1 = c0 + FACE_PX * u + 0.5 * FACE_PX * v
    fits, status = fit_mask(raster([square(*c0, FACE_PX, angle), square(*c1, FACE_PX, angle)]))
    assert status == "split" and len(fits) == 2
    for f in fits:
        assert abs(angle_diff(quad_angle(f.quad), angle)) < 1.5
    for truth in (c0, c1):
        assert min(math.dist(f.center, truth) for f in fits) < 2.5


def test_corner_touching_pair_is_split() -> None:
    c0 = np.array([600.0, 330.0])
    c1 = c0 + FACE_PX * np.array([1.0, 1.0])  # corners meet
    fits, status = fit_mask(raster([square(*c0, FACE_PX, 0), square(*c1, FACE_PX, 0)]))
    assert status == "split" and len(fits) == 2
    for truth in (c0, c1):
        assert min(math.dist(f.center, truth) for f in fits) < 3.0


@pytest.mark.parametrize("angle", [0.0, 30.0])
def test_l_of_three_without_a_side_band_is_three_cubes(angle: float) -> None:
    """Paper on the top face only: the blob has no side band even far from the principal point.
    The count from the area assumed the predicted band (about 11 px per axis here) and gave 2."""
    a = math.radians(angle)
    u, v = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
    c0 = np.array([300.0, 520.0])  # about 380 px from the principal point
    centres = [c0, c0 + FACE_PX * u, c0 + FACE_PX * v]
    fits, status = fit_mask(raster([square(*c, FACE_PX, angle) for c in centres]))
    assert status == "split" and len(fits) == 3
    for truth in centres:
        assert min(math.dist(f.center, truth) for f in fits) < 2.0


# --- top-face isolation on rendered cubes -------------------------------------------------
@pytest.mark.parametrize("x, y, angle", [(-230.0, 60.0, 20.0), (200.0, 40.0, 65.0), (-60.0, 230.0, 5.0)])
def test_top_face_is_isolated_from_the_side_band(x: float, y: float, angle: float) -> None:
    img, gt = make_scene(0, cubes=[dict(color="blue", x=x, y=y, angle=angle)], paper="all", **CLEAN)
    m = colour_mask(img)
    blob = max(blobs_from_mask(m), key=lambda b: b.area)
    ys, xs = np.nonzero(blob.mask)
    centroid = (xs.mean() + blob.x0, ys.mean() + blob.y0)
    truth = gt[0]["center"]
    assert math.dist(centroid, truth) > 4.0  # the band really pulls the blob towards the centre
    fits, status = fit_blob(blob, Geometry(), img.shape[:2], img)
    assert status == "ok" and len(fits) == 1
    assert "side_band" in fits[0].flags
    assert math.dist(fits[0].center, truth) < 2.0
    assert abs(angle_diff(quad_angle(fits[0].quad), gt[0]["angle"])) < 1.5


def test_tower_top_gets_its_larger_face() -> None:
    cubes = [dict(color="green", x=-13.0, y=150.0, angle=30.0, level=k) for k in (1, 2, 3, 4)]
    img, gt = make_scene(0, cubes=cubes, paper="all", **CLEAN)
    top = [g for g in gt if g["level"] == 4][0]
    assert top["side_px"] > 69  # 61.2 * 576 / (601 - 100)
    blob = max(blobs_from_mask(colour_mask(img)), key=lambda b: b.area)
    fits, _ = fit_blob(blob, Geometry(), img.shape[:2], img)
    assert math.dist(fits[0].center, top["center"]) < 2.5


# --- contract entry points (used by the lab's Ultralytics adapter) --------------------------
@pytest.fixture(scope="module")
def rim_cube() -> tuple[np.ndarray, dict, np.ndarray]:
    img, gt = make_scene(0, cubes=[dict(color="red", x=-225.0, y=70.0, angle=35.0)], paper="all", **CLEAN)
    m = colour_mask(img)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return img, gt[0], max(cnts, key=cv2.contourArea).reshape(-1, 2)


def test_cube_from_quad(rim_cube) -> None:
    img, g, _ = rim_cube
    c = cube_from_quad(img, g["quad"], score=0.9)
    assert c.color == "red" and c.score == pytest.approx(0.9)
    assert c.quad == order_quad(g["quad"])
    assert abs(angle_diff(c.angle, g["angle"])) < 1e-3


def test_cube_from_polygon_cuts_the_band_only_with_face_px(rim_cube) -> None:
    img, g, outline = rim_cube
    with_face = cube_from_polygon(img, outline, score=0.8, face_px=FACE_PX)
    without = cube_from_polygon(img, outline, score=0.8)
    assert with_face is not None and without is not None
    assert with_face.color == "red" and with_face.polygon is not None
    assert math.dist(with_face.center, g["center"]) < 2.0
    assert math.dist(without.center, g["center"]) > 4.0  # baseline: whole blob, band included
    m = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(m, [outline.reshape(-1, 1, 2)], 255)
    via_mask = cube_from_mask(img, m, score=0.8, face_px=FACE_PX)
    assert via_mask is not None and math.dist(via_mask.center, g["center"]) < 2.0


def test_cube_from_box_recovers_the_angle(rim_cube) -> None:
    img, g, outline = rim_cube
    x1, y1 = outline.min(axis=0)
    x2, y2 = outline.max(axis=0)
    c = cube_from_box(img, (x1, y1, x2, y2), score=0.7, face_px=FACE_PX)
    assert c is not None and "from_box" in c.flags and c.angle is not None
    assert abs(angle_diff(c.angle, g["angle"])) < 3.0
    assert math.dist(c.center, g["center"]) < 4.0
    assert cube_from_box(img, (5, 5, 6, 6), score=0.5) is None  # degenerate box


def test_given_colour_is_never_dropped_or_overwritten() -> None:
    """A trained model's class is final: a 'wood' box on bare white board stays a wood cube."""
    img, _ = make_scene(0, cubes=[], **CLEAN)
    quad = square(500, 400, 61, 10)
    q = cube_from_quad(img, quad, score=0.6, color="wood")
    assert q.color == "wood" and "class_mismatch" in q.flags
    poly = cube_from_polygon(img, quad, score=0.6, color="wood", face_px=FACE_PX)
    assert poly is not None and poly.color == "wood"
    box = cube_from_box(img, (470, 370, 530, 430), score=0.6, color="wood", face_px=FACE_PX)
    assert box is not None and box.color == "wood"
    # Without a class the pixels decide, and board pixels are reported, not hidden.
    none = cube_from_quad(img, quad, score=0.6)
    assert none.color == "unknown" and "no_colour" in none.flags
    assert cube_from_polygon(img, quad, score=0.6, face_px=FACE_PX) is not None


def test_partly_hidden_cube_is_anchored_on_its_visible_edges() -> None:
    """A dark 'arm' covers a third of the top face: the face square hangs from the edges that
    have board outside them, so the centre stays right although a third of the cube is gone."""
    img, gt = make_scene(0, cubes=[dict(color="red", x=-150.0, y=150.0, angle=0.0)], paper="top", **CLEAN)
    g = gt[0]
    q = np.asarray(g["quad"])
    x0, x1 = q[:, 0].min(), q[:, 0].max()
    img = img.copy()
    img[:, int(x0 + 0.66 * (x1 - x0)) : int(x1) + 25] = (50, 50, 55)  # the arm, right third and beyond
    blob = max(blobs_from_mask(colour_mask(img)), key=lambda b: b.area)
    fits, status = fit_blob(blob, Geometry(), img.shape[:2], img)
    assert status == "ok" and "partial" in fits[0].flags and fits[0].quad is not None
    assert math.dist(fits[0].center, g["center"]) < 3.0

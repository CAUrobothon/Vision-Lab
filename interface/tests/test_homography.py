"""The homography TEMPLATE: known mappings, fitting, the plane-height correction, no shipped numbers."""

from __future__ import annotations

import inspect
import math

import pytest

from vision_interface import HomographyPixelToWorld, PixelToWorld, fit_homography


def _close(a, b, tol=1e-6) -> bool:
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def test_known_scale_and_offset() -> None:
    # 0.5 mm per pixel, pixel (100, 100) at robot (0, 0), image v pointing to robot -y.
    H = [[0.5, 0.0, -50.0], [0.0, -0.5, 50.0], [0.0, 0.0, 1.0]]
    p2w = HomographyPixelToWorld(H, calibrated_height_mm=25.0, board_top_z_mm=0.0, name="test")
    assert isinstance(p2w, PixelToWorld)
    assert _close(p2w.pixel_to_world(100, 100, 25.0), (0.0, 0.0, 25.0))
    assert _close(p2w.pixel_to_world(300, 0, 25.0), (100.0, 50.0, 25.0))
    assert _close(p2w.world_to_pixel(100.0, 50.0, 25.0), (300.0, 0.0))


def test_board_top_offset_sets_z() -> None:
    p2w = HomographyPixelToWorld([[1, 0, 0], [0, 1, 0], [0, 0, 1]], calibrated_height_mm=25.0,
                                 board_top_z_mm=-17.5, name="frame z=0 on the table")
    assert p2w.pixel_to_world(3, 4, 25.0) == (3.0, 4.0, 7.5)


def test_fit_four_points_exactly() -> None:
    px = [(0, 0), (100, 0), (100, 100), (0, 100)]
    mm = [(0, 0), (50, 0), (50, 50), (0, 50)]
    H = fit_homography(px, mm)
    assert _close([v for r in H for v in r], [0.5, 0, 0, 0, 0.5, 0, 0, 0, 1], 1e-9)
    p2w = HomographyPixelToWorld(H, calibrated_height_mm=25.0, board_top_z_mm=0.0, name="t")
    assert _close(p2w.pixel_to_world(50, 50, 25.0)[:2], (25.0, 25.0), 1e-9)


def test_fit_recovers_a_projective_matrix_from_many_points() -> None:
    true = ((0.42, -0.03, -270.0), (0.02, -0.40, 290.0), (1.5e-5, -2.0e-5, 1.0))
    p_true = HomographyPixelToWorld(true, calibrated_height_mm=25.0, board_top_z_mm=0.0, name="true")
    px = [(u, v) for u in (40, 400, 800, 1240) for v in (30, 360, 690)]
    mm = [p_true.pixel_to_world(u, v, 25.0)[:2] for u, v in px]
    p_fit = HomographyPixelToWorld(fit_homography(px, mm), calibrated_height_mm=25.0, board_top_z_mm=0.0, name="fit")
    for u, v in [(10, 10), (640, 360), (1270, 710), (333, 555)]:
        assert _close(p_fit.pixel_to_world(u, v, 25.0), p_true.pixel_to_world(u, v, 25.0), 1e-6)


def test_other_heights_follow_the_camera_ray() -> None:
    # Calibrated on the board (height 0); lens 600 mm straight above the origin.
    H = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    p2w = HomographyPixelToWorld(H, calibrated_height_mm=0.0, board_top_z_mm=0.0, name="t",
                                 camera_xyz_mm=(0.0, 0.0, 600.0))
    # The ray through board point (100, 0) passes height 25 at 100 * 575 / 600.
    x, y, z = p2w.pixel_to_world(100, 0, 25.0)
    assert math.isclose(x, 100 * 575 / 600) and y == 0.0 and z == 25.0
    assert _close(p2w.world_to_pixel(x, y, 25.0), (100.0, 0.0))
    # Without the lens position only the calibrated height can be mapped.
    bare = HomographyPixelToWorld(H, calibrated_height_mm=0.0, board_top_z_mm=0.0, name="t")
    with pytest.raises(ValueError, match="camera_xyz_mm"):
        bare.pixel_to_world(100, 0, 25.0)


def test_template_ships_no_numbers() -> None:
    params = inspect.signature(HomographyPixelToWorld.__init__).parameters
    for name in ("matrix", "calibrated_height_mm", "board_top_z_mm", "name"):
        assert params[name].default is inspect.Parameter.empty, name
    assert params["camera_xyz_mm"].default is None
    with pytest.raises(TypeError):
        HomographyPixelToWorld()  # type: ignore[call-arg]


@pytest.mark.parametrize("m", [[[1, 0, 0], [0, 1, 0]], [[1, 2, 3], [2, 4, 6], [0, 0, 1]]])
def test_bad_matrices_are_rejected(m) -> None:
    with pytest.raises(ValueError):
        HomographyPixelToWorld(m, calibrated_height_mm=25.0, board_top_z_mm=0.0, name="t")


def test_collinear_points_are_rejected() -> None:
    with pytest.raises(ValueError):
        fit_homography([(0, 0), (1, 1), (2, 2), (3, 3)], [(0, 0), (1, 0), (0, 1), (1, 1)])

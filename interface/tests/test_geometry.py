"""90-degree yaw symmetry, localisation through a PixelToWorld, list helpers, rule helpers."""

from __future__ import annotations

import math

import pytest

from vision_interface import (
    Color, Cube, Scene, SyntheticPixelToWorld, by_color, closest_grasp_yaw, fold90, localize, localize_cube,
    nearest_first, random_scene,
)
from vision_interface import rules


@pytest.mark.parametrize("deg, want", [(0, 0), (90, 0), (450, 0), (-10, 80), (135, 45), (89.999, 89.999), (-1e-17, 0)])
def test_fold90(deg: float, want: float) -> None:
    got = fold90(deg)
    assert 0.0 <= got < 90.0
    assert math.isclose(got, want, abs_tol=1e-9)


@pytest.mark.parametrize("yaw, ref, want", [
    (10, 0, 10), (80, 0, -10), (44, 0, 44), (46, 0, -44), (45, 0, -45),  # tie goes to the lower angle
    (30, 170, 120 + 90), (30, 100, 120), (0, -135, -180), (89.9, 90, 89.9),
])
def test_closest_grasp_yaw(yaw: float, ref: float, want: float) -> None:
    got = closest_grasp_yaw(yaw, ref)
    assert math.isclose(got, want, abs_tol=1e-9)
    assert -45.0 <= got - ref < 45.0
    assert math.isclose(fold90(got), fold90(yaw), abs_tol=1e-9)  # same square pose


def _same_pose(a: float, b: float) -> bool:
    """Equal as square orientations (mod 90), tolerant of the 0/90 wrap."""
    return math.isclose(fold90(a - b + 45.0) - 45.0, 0.0, abs_tol=1e-6)


class _Affine:
    """A test PixelToWorld: rotate the image by `rot` degrees, optionally mirror, then scale."""

    def __init__(self, rot_deg: float, mirror: bool) -> None:
        self.c, self.s = math.cos(math.radians(rot_deg)), math.sin(math.radians(rot_deg))
        self.mirror = mirror
        self.name = "SYNTHETIC affine test"

    def pixel_to_world(self, u: float, v: float, height_mm: float):
        if self.mirror:
            u = -u
        return (0.4 * (self.c * u - self.s * v) + 30.0, 0.4 * (self.s * u + self.c * v) - 20.0, height_mm)


def _square_cube(yaw_img: float) -> Cube:
    c, s, h = math.cos(math.radians(yaw_img)), math.sin(math.radians(yaw_img)), 30.0
    pts = [(500 + c * dx - s * dy, 300 + s * dx + c * dy) for dx, dy in ((-h, -h), (h, -h), (h, h), (-h, h))]
    from vision_interface.fake import order_quad
    return Cube(color="blue", center_px=(500.0, 300.0), corners_px=order_quad(pts),
                image_angle_deg=fold90(yaw_img), area_px2=3600.0, score=1.0)


@pytest.mark.parametrize("rot, mirror", [(0, False), (90, False), (-33, False), (0, True), (57, True)])
@pytest.mark.parametrize("yaw_img", [0.0, 12.0, 44.0, 71.5])
def test_yaw_is_measured_in_the_robot_frame(rot: float, mirror: bool, yaw_img: float) -> None:
    p2w = _Affine(rot, mirror)
    cube = _square_cube(yaw_img)
    want = fold90((-yaw_img if mirror else yaw_img) + rot)
    assert _same_pose(localize_cube(cube, p2w).pose.yaw_deg, want)
    no_corners = Cube(color="blue", center_px=cube.center_px, corners_px=None,
                      image_angle_deg=cube.image_angle_deg, area_px2=1.0, score=1.0)
    assert _same_pose(localize_cube(no_corners, p2w).pose.yaw_deg, want)


def test_localize_fills_every_pose_and_names_the_calibration() -> None:
    scene = random_scene("B", seed=2)
    assert scene.localized_by is None and all(c.pose is None for c in scene.cubes)
    p2w = SyntheticPixelToWorld()
    loc = localize(scene, p2w)
    assert loc.localized_by == p2w.name and loc.is_synthetic
    for raw, c in zip(scene.cubes, loc.cubes):
        assert c.center_px == raw.center_px and c.pose is not None
        assert c.pose.z_mm == rules.CUBE_TOP_HEIGHT_MM
        assert -45.0 <= c.pose.grasp_yaw_deg - c.pose.bearing_deg < 45.0
        # SyntheticPixelToWorld flips v (image down = robot -y), so the yaw is the mirrored image angle.
        assert _same_pose(c.pose.yaw_deg, -raw.image_angle_deg)


def test_localize_needs_a_name() -> None:
    class NoName:
        name = " "

        def pixel_to_world(self, u, v, h):
            return (u, v, h)

    with pytest.raises(ValueError, match="name"):
        localize(random_scene("A", seed=1), NoName())


def test_localize_cube_at_other_heights() -> None:
    cube = random_scene("A", seed=1).cubes[0]
    on_tower = localize_cube(cube, SyntheticPixelToWorld(), rules.tower_top_height_mm(3))
    assert on_tower.pose.z_mm == 75.0
    in_basket = localize_cube(cube, SyntheticPixelToWorld(), rules.BASKET_CUBE_TOP_HEIGHT_MM)
    assert in_basket.pose.z_mm == 29.0


def test_by_color_and_nearest_first() -> None:
    scene = localize(random_scene("B", seed=4), SyntheticPixelToWorld())
    reds = by_color(scene.cubes, Color.RED)
    assert len(reds) == 3 and all(c.color is Color.RED for c in reds)
    assert len(by_color(scene.cubes, "green", Color.BLUE)) == 6
    order = nearest_first(scene.cubes)
    d = [c.pose.r_mm for c in order]
    assert d == sorted(d)
    far = [math.hypot(c.pose.x_mm + 1000.0, c.pose.y_mm) for c in nearest_first(scene.cubes, x_mm=-1000.0)]
    assert far == sorted(far) and len(far) == 12
    with pytest.raises(ValueError, match="localize"):
        nearest_first(random_scene("B", seed=4).cubes)


def test_tower_target_colors_follow_the_rule_example() -> None:
    # Rulebook §5 p.7: R-B-G on levels 1-3 -> R-B-G-R-B-G-R-B-G.
    got = rules.tower_target_colors(["red", "blue", "green"])
    assert [c.value[0].upper() for c in got] == list("RBGRBGRBG")
    with pytest.raises(ValueError):
        rules.tower_target_colors(["red", "red", "green"])
    assert rules.tower_top_height_mm(1) == 25.0 and rules.tower_top_height_mm(9) == 225.0


@pytest.mark.parametrize("x, y, mission, want", [
    (150, 10, "A", "A.zone1.right"), (-150, 10, "A", "A.zone1.left"), (0, 150, "A", "A.zone2"),
    (0, 50, "A", None), (0, 260, "A", None), (-100, 100, "B", "B.placement"),
    (170 * math.cos(math.radians(10)), 170 * math.sin(math.radians(10)), "B", "B.basket.blue"),
    (170 * math.cos(math.radians(35)), 170 * math.sin(math.radians(35)), "B", "B.basket.green"),
    (170 * math.cos(math.radians(70)), 170 * math.sin(math.radians(70)), "B", "B.basket.red"),
    (100, -5, "B", None),
])
def test_zone_of(x: float, y: float, mission: str, want: str | None) -> None:
    assert rules.zone_of(x, y, mission) == want


def test_scene_without_cubes_is_fine() -> None:
    empty = Scene(frame_id=1, timestamp_s=0.0, image_size_px=(640, 480), cubes=())
    assert localize(empty, SyntheticPixelToWorld()).cubes == ()

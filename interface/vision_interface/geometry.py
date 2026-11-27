"""Pure geometry helpers: localisation (pixels -> robot mm through a PixelToWorld), the 90-degree
yaw symmetry, and small list filters. No motion planning.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Iterable

from vision_interface.protocols import PixelToWorld
from vision_interface.rules import CUBE_TOP_HEIGHT_MM
from vision_interface.scene import Color, Cube, CubePose, Scene


def fold90(deg: float) -> float:
    """Fold any angle into [0, 90): a square's orientation repeats every 90 degrees."""
    r = float(deg) % 90.0
    return 0.0 if r >= 90.0 else r  # -1e-17 % 90.0 == 90.0 in floating point


def closest_grasp_yaw(yaw_deg: float, reference_deg: float) -> float:
    """The member of {yaw_deg + k*90} closest to ``reference_deg`` (e.g. the gripper's current
    yaw, or the bearing to the cube), so the result minus reference_deg is in [-45, 45).
    An exact tie (45 degrees away) goes to the lower angle.

    Why 90: the cube's top face is a square (same after a 90-degree turn) and a parallel-jaw
    gripper is the same after 180 degrees, so every yaw + k*90 is an equally good grasp. It does
    not matter whether you measure the gripper by its jaw-closing axis or by its finger plane:
    the two differ by 90 degrees and land in the same set.
    """
    d = fold90(float(yaw_deg) - float(reference_deg) + 45.0) - 45.0
    return float(reference_deg) + d


def _yaw_from_points(pts: list[tuple[float, float]]) -> float:
    """Mean direction (mod 90) of the closed polygon's edges: 4-theta circular mean, so edges
    at 0.5 and 89.5 degrees average to 0, not 45."""
    c = s = 0.0
    for i in range(len(pts)):
        (x0, y0), (x1, y1) = pts[i], pts[(i + 1) % len(pts)]
        a = math.atan2(y1 - y0, x1 - x0)
        c += math.cos(4.0 * a)
        s += math.sin(4.0 * a)
    if math.hypot(c, s) < 1e-9:
        raise ValueError("corners do not form a square-like quad (edge directions cancel out)")
    return fold90(math.degrees(math.atan2(s, c)) / 4.0)


def localize_cube(cube: Cube, pixel_to_world: PixelToWorld, height_mm: float = CUBE_TOP_HEIGHT_MM) -> Cube:
    """Return ``cube`` with ``pose`` filled by mapping its top face, assumed to lie ``height_mm``
    above the board top (25 = standing on the board; rules.tower_top_height_mm(n) for a tower
    top; rules.BASKET_CUBE_TOP_HEIGHT_MM for a cube on a basket floor).

    Yaw is measured in the ROBOT frame from the mapped corners, so it is right however the
    camera image is rotated or mirrored. Without corners it falls back to mapping a short step
    along the image angle; with neither, yaw is None.
    """
    u, v = cube.center_px
    x, y, z = (float(t) for t in pixel_to_world.pixel_to_world(u, v, height_mm))
    yaw: float | None = None
    if cube.corners_px is not None:
        pts = [pixel_to_world.pixel_to_world(pu, pv, height_mm)[:2] for pu, pv in cube.corners_px]
        yaw = _yaw_from_points([(float(a), float(b)) for a, b in pts])
    elif cube.image_angle_deg is not None:
        t = math.radians(cube.image_angle_deg)
        step = 10.0  # px; any short step inside the top face works
        x1, y1, _ = pixel_to_world.pixel_to_world(u + step * math.cos(t), v + step * math.sin(t), height_mm)
        yaw = fold90(math.degrees(math.atan2(float(y1) - y, float(x1) - x)))
    grasp = closest_grasp_yaw(yaw, math.degrees(math.atan2(y, x))) if yaw is not None else None
    return replace(cube, pose=CubePose(x_mm=x, y_mm=y, z_mm=z, yaw_deg=yaw, grasp_yaw_deg=grasp))


def localize(scene: Scene, pixel_to_world: PixelToWorld, height_mm: float = CUBE_TOP_HEIGHT_MM) -> Scene:
    """Fill every cube's pose with the calibration teammate's PixelToWorld, assuming every cube
    stands on the board (top face ``height_mm`` = 25 mm above the board top). Re-map a cube that
    is known to sit higher (tower top, basket) with localize_cube(). Returns a new Scene whose
    ``localized_by`` is ``pixel_to_world.name``."""
    name = str(getattr(pixel_to_world, "name", "") or "").strip()
    if not name:
        raise ValueError("the PixelToWorld needs a non-empty .name (it is recorded in Scene.localized_by)")
    cubes = tuple(localize_cube(c, pixel_to_world, height_mm) for c in scene.cubes)
    return replace(scene, cubes=cubes, localized_by=name)


def by_color(cubes: Iterable[Cube], *colors: Color | str) -> list[Cube]:
    """Cubes of the given colour(s), in their original order."""
    wanted = {Color(c) for c in colors}
    return [c for c in cubes if c.color in wanted]


def nearest_first(cubes: Iterable[Cube], x_mm: float = 0.0, y_mm: float = 0.0) -> list[Cube]:
    """Localised cubes sorted by x-y distance from (x_mm, y_mm) in the robot frame (default: the
    frame origin). Raises if a cube has no pose -- run localize() first."""
    cubes = list(cubes)
    if any(c.pose is None for c in cubes):
        raise ValueError("nearest_first needs localised cubes: call localize(scene, pixel_to_world) first")
    return sorted(cubes, key=lambda c: math.hypot(c.pose.x_mm - x_mm, c.pose.y_mm - y_mm))  # type: ignore[union-attr]

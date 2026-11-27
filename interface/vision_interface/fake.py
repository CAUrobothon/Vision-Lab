"""FakeVisionSource: develop the kinematics side without a camera.

Everything here is SYNTHETIC. The fake returns pixel-only scenes, exactly like the real source.
For millimetre tests, localise them with ``SyntheticPixelToWorld`` -- a made-up mapping whose
name starts with "SYNTHETIC", so ``scene.is_synthetic`` is True and nobody mistakes its numbers
for a calibration. On the day, swap it for the calibration teammate's PixelToWorld.

Command line (prints JSON; see README):
    python -m vision_interface.fake --mission B --seed 1
    python -m vision_interface.fake --mission A --seed 1 --count 5 --out scenes.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from vision_interface.geometry import fold90, localize
from vision_interface.rules import CUBE_SIZE_MM, MISSION_CUBES, START_SECTORS
from vision_interface.scene import Cube, Mission, Scene

SYNTHETIC_NAME = "SYNTHETIC test mapping (vision_interface.fake.SyntheticPixelToWorld) - not a calibration"


class SyntheticPixelToWorld:
    """SYNTHETIC, made-up pixel <-> mm mapping for tests. Not the real camera, not a calibration.

    Orthographic and axis-aligned: x = (u - origin_u) * mm_per_px, y = (origin_v - v) * mm_per_px,
    z = height_mm. Its frame imitates the rulebook frame (origin = robot base axis, +x right,
    +y towards the pillar = image up, z = 0 at the board top), so synthetic layouts land in the
    rulebook zones. The defaults are round numbers chosen only so the R250 arena fits in a
    1280 x 720 image; they say nothing about the competition camera.
    """

    def __init__(self, mm_per_px: float = 0.5, origin_px: tuple[float, float] = (640.0, 700.0)) -> None:
        self.mm_per_px = float(mm_per_px)
        self.origin_px = (float(origin_px[0]), float(origin_px[1]))
        self.name = SYNTHETIC_NAME

    def pixel_to_world(self, u_px: float, v_px: float, height_mm: float) -> tuple[float, float, float]:
        ou, ov = self.origin_px
        return ((u_px - ou) * self.mm_per_px, (ov - v_px) * self.mm_per_px, float(height_mm))

    def world_to_pixel(self, x_mm: float, y_mm: float) -> tuple[float, float]:
        ou, ov = self.origin_px
        return (ou + x_mm / self.mm_per_px, ov - y_mm / self.mm_per_px)


def order_quad(points: Sequence[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
    """4 corners clockwise ON SCREEN (v grows down), first = smallest u + v (cubevision's order)."""
    cu = sum(p[0] for p in points) / len(points)
    cv = sum(p[1] for p in points) / len(points)
    # With v pointing down, increasing atan2(v, u) runs clockwise on screen.
    ring = sorted(points, key=lambda p: math.atan2(p[1] - cv, p[0] - cu))
    start = min(range(len(ring)), key=lambda i: ring[i][0] + ring[i][1])
    return tuple(ring[start:] + ring[:start])


def _shoelace(pts: Sequence[tuple[float, float]]) -> float:
    return abs(sum(pts[i][0] * pts[(i + 1) % 4][1] - pts[(i + 1) % 4][0] * pts[i][1] for i in range(4))) / 2.0


def _footprint(x: float, y: float, yaw_deg: float) -> list[tuple[float, float]]:
    """The 4 corners (mm) of a 25 mm square centred at (x, y), edges at yaw_deg."""
    h = CUBE_SIZE_MM / 2.0
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    return [(x + c * dx - s * dy, y + s * dx + c * dy) for dx, dy in ((-h, -h), (h, -h), (h, h), (-h, h))]


def _random_layout(mission: Mission, rng: random.Random, positions_mm: Sequence[tuple[float, float]] | None
                   ) -> list[tuple[float, float, float]]:
    """(x, y, yaw) per cube in the rule order. With positions_mm (the 20 marked spots, once they
    are revealed on the day, §5 p.7) the draw follows the rule: pick len(cubes) spots without
    replacement, in drawn order. Without them: uniform spots in the start zones (§3 p.5, §4 p.6),
    whole footprint inside the zone, no two cubes closer than their circumscribed circles.
    The rulebook does not fix cube orientation, so yaw is uniform in [0, 90)."""
    n = len(MISSION_CUBES[mission])
    if positions_mm is not None:
        if len(positions_mm) < n:
            raise ValueError(f"mission {mission.value} needs {n} positions, got {len(positions_mm)}")
        return [(float(x), float(y), rng.uniform(0.0, 90.0)) for x, y in rng.sample(list(positions_mm), n)]
    sectors = START_SECTORS[mission]
    min_gap = CUBE_SIZE_MM * math.sqrt(2.0) + 1.0  # circumscribed circles never touch
    out: list[tuple[float, float, float]] = []
    for _ in range(20000):
        if len(out) == n:
            return out
        s = rng.choice(sectors)
        r = math.sqrt(rng.uniform(s.r_min_mm ** 2, s.r_max_mm ** 2))  # uniform over the area
        a = math.radians(rng.uniform(s.a_min_deg, s.a_max_deg))
        x, y, yaw = r * math.cos(a), r * math.sin(a), rng.uniform(0.0, 90.0)
        if not all(s.contains(px, py) for px, py in _footprint(x, y, yaw)):
            continue
        if any(math.hypot(x - ox, y - oy) < min_gap for ox, oy, _ in out):
            continue
        out.append((x, y, yaw))
    raise RuntimeError("could not place the cubes without overlap (should not happen)")


def random_scene(
    mission: Mission | str,
    *,
    seed: int | None = None,
    positions_mm: Sequence[tuple[float, float]] | None = None,
    mapping: SyntheticPixelToWorld | None = None,
    image_size_px: tuple[int, int] = (1280, 720),
    frame_id: int = 0,
    timestamp_s: float | None = None,
) -> Scene:
    """One random, rule-respecting starting layout as a PIXEL-ONLY Scene (no poses).

    Rule-respecting: cube count and colours per mission in the rule order (§5 p.7), each cube
    fully inside its start zone, no overlaps. The layout is drawn in the synthetic frame and
    projected through ``mapping`` (default SyntheticPixelToWorld()), so the pixel positions are
    synthetic too -- they are not where the competition camera would see the cubes.
    """
    m = Mission(mission)
    rng = random.Random(seed)
    mp = mapping or SyntheticPixelToWorld()
    w, h = image_size_px
    cubes = []
    for color, (x, y, yaw) in zip(MISSION_CUBES[m], _random_layout(m, rng, positions_mm)):
        corners = order_quad([mp.world_to_pixel(px, py) for px, py in _footprint(x, y, yaw)])
        if not all(0.0 <= u < w and 0.0 <= v < h for u, v in corners):
            raise ValueError("a cube falls outside the image: mapping and image_size_px do not match")
        (u0, v0), (u1, v1) = corners[0], corners[1]
        cubes.append(Cube(
            color=color,
            center_px=mp.world_to_pixel(x, y),
            corners_px=corners,  # type: ignore[arg-type]
            image_angle_deg=fold90(math.degrees(math.atan2(v1 - v0, u1 - u0))),
            area_px2=_shoelace(corners),
            score=1.0,
        ))
    return Scene(
        frame_id=frame_id,
        timestamp_s=time.time() if timestamp_s is None else timestamp_s,
        image_size_px=(w, h),
        cubes=tuple(cubes),
        mission=m,
        source="fake:random",
        notes=("synthetic layout: pixel positions come from SyntheticPixelToWorld, not a real camera",),
    )


class FakeVisionSource:
    """A VisionSource without a camera.

    - ``FakeVisionSource(scenes)`` / ``from_json_file(path)``: replays recorded scenes as they
      are (frame_id and timestamp unchanged), cycling when ``loop`` is True, otherwise repeating
      the last one.
    - ``FakeVisionSource.random(mission, seed=...)``: one random layout that stays put; every
      get_scene() returns it with a new frame_id and the current time. ``remove(cube)``
      simulates a pick: the cube is gone from the next scene.
    """

    def __init__(self, scenes: Sequence[Scene], *, loop: bool = True, restamp: bool = False) -> None:
        if not scenes:
            raise ValueError("FakeVisionSource needs at least one scene")
        self._scenes = list(scenes)
        self._loop = loop
        self._restamp = restamp
        self._i = 0
        self._frame = 0

    @classmethod
    def from_json_file(cls, path: str | Path, *, loop: bool = True) -> FakeVisionSource:
        """A file holding one scene object or a JSON array of scene objects."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        items = data if isinstance(data, list) else [data]
        return cls([Scene.from_dict(d) for d in items], loop=loop)

    @classmethod
    def random(cls, mission: Mission | str, *, seed: int | None = None,
               positions_mm: Sequence[tuple[float, float]] | None = None) -> FakeVisionSource:
        return cls([random_scene(mission, seed=seed, positions_mm=positions_mm)], restamp=True)

    def get_scene(self) -> Scene:
        scene = self._scenes[self._i]
        if self._loop:
            self._i = (self._i + 1) % len(self._scenes)
        else:
            self._i = min(self._i + 1, len(self._scenes) - 1)
        if self._restamp:
            self._frame += 1
            scene = replace(scene, frame_id=self._frame, timestamp_s=time.time())
        return scene

    def remove(self, cube: Cube) -> bool:
        """Simulate picking ``cube`` (matched by its pixel centre; a localised copy works too).
        Only for a source holding one scene. Returns False if no such cube is there."""
        if len(self._scenes) != 1:
            raise RuntimeError("remove() works on a single-scene source (e.g. FakeVisionSource.random)")
        scene = self._scenes[0]
        keep = tuple(c for c in scene.cubes if c.center_px != cube.center_px)
        if len(keep) == len(scene.cubes):
            return False
        self._scenes[0] = replace(scene, cubes=keep)
        return True


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m vision_interface.fake",
                                 description="Print random, rule-respecting SYNTHETIC scenes as JSON.")
    ap.add_argument("--mission", choices=["A", "B"], required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--count", type=int, default=1, help="number of scenes (seed, seed+1, ...)")
    ap.add_argument("--localize-synthetic", action="store_true",
                    help="also fill poses with SyntheticPixelToWorld (labelled SYNTHETIC)")
    ap.add_argument("--out", type=Path, default=None, help="write to this file instead of stdout")
    args = ap.parse_args(argv)
    scenes = []
    for k in range(args.count):
        seed = None if args.seed is None else args.seed + k
        s = random_scene(args.mission, seed=seed, frame_id=k)
        if args.localize_synthetic:
            s = localize(s, SyntheticPixelToWorld())
        scenes.append(s.to_dict())
    text = json.dumps(scenes[0] if args.count == 1 else scenes, ensure_ascii=False, indent=2)
    if args.out is None:
        sys.stdout.write(text + "\n")
    else:
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

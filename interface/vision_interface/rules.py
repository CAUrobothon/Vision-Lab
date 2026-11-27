"""Facts taken from the competition rulebook, each with its source.

Source documents (in ~/Desktop/로보톤/):
- "Pick and Place 대회 규정집.pdf" (8 pages) -- the rulebook. It wins over the OT slides.
  p.1 §1 + 도면 1/3 · p.2 도면 2/3, 3/3 · p.3 §2 · p.4 timeline, §3 start · p.5 §3 · p.6 §4 ·
  p.7 §5, §6 · p.8 §6 end, §7.
- "26_로보톤_OT.pptx.pdf" (orientation slides), cited as "OT p.N".

Nothing in here is a vision measurement or a calibration. Heights are given above the BOARD TOP
(rulebook z = 0), because that is the one height reference the rulebook defines; the robot
frame's own z zero is the calibration teammate's choice (see protocols.PixelToWorld).

The angular sectors at the bottom are in the RULEBOOK frame (§1.1, p.1). They describe the
robot frame only if the calibration teammate's PixelToWorld outputs that same frame -- an open
question in README.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from vision_interface.scene import Color, Mission

# --- cube ------------------------------------------------------------------------------------
CUBE_SIZE_MM = 25.0  # wooden cube, 25 mm edge, 10 g (§3 경기 구성 p.5; 도면 3/3 C p.2)
# Height above the board top of the TOP FACE of a cube standing on the board: the plane every
# Cube.center_px / corners_px refers to (z = 0 is the board top: §1.1 p.1, 도면 3/3 A p.2).
CUBE_TOP_HEIGHT_MM = CUBE_SIZE_MM

# --- board and baskets -------------------------------------------------------------------------
BOARD_THICKNESS_MM = 17.5  # board top is 17.5 mm above the table (§1.6 p.1; 도면 3/3 A p.2)
BASKET_FLOOR_HEIGHT_MM = 4.0  # basket inner floor is 4 mm above the board top (도면 3/3 B p.2)
BASKET_RIM_HEIGHT_MM = 32.0  # basket wall top: 35 mm tall, 32 mm above the board top (§4 p.6; 도면 3/3 B)
BASKET_WALL_MM = 7.0  # wall thickness (§4 p.6; 도면 2/3 note 3, 3/3 B)
# Top face of a cube lying flat on a basket floor: 4 + 25 = 29 mm above the board top.
BASKET_CUBE_TOP_HEIGHT_MM = BASKET_FLOOR_HEIGHT_MM + CUBE_SIZE_MM
# Points per cube in a basket: +points if the colour matches, -points otherwise, wood included
# (§4 배점 p.6: R 10, G 15, B 20; score = 225 + sum(points * (matching - non-matching))).
BASKET_POINTS = {Color.RED: 10, Color.GREEN: 15, Color.BLUE: 20}

# --- camera (CAD value from the rulebook; NOT used by any code path in this package) ----------
# Lens centre in the rulebook frame, camera looking straight down (§1.3 p.1; 도면 3/3 A p.2).
# During matches the organisers' camera is used (§1.5 p.1); model and resolution are not stated.
RULEBOOK_CAMERA_LENS_MM = (-13.1, 131.4, 601.0)

# --- cubes per mission, in the rulebook's placement order (§5 p.7) -----------------------------
_R, _G, _B, _W = Color.RED, Color.GREEN, Color.BLUE, Color.WOOD
MISSION_CUBES: dict[Mission, tuple[Color, ...]] = {
    Mission.A: (_R, _R, _R, _G, _G, _G, _B, _B, _B),  # 9, no wood (§3 p.5, §5 p.7, 도면 3/3 C)
    Mission.B: (_R, _R, _R, _G, _G, _G, _B, _B, _B, _W, _W, _W),  # 12 (§4 p.6, §5 p.7)
}
PLACEMENT_SLOTS = 20  # organisers mark positions 1-20, revealed on the day; cubes go to a random draw (§5 p.7)

# --- mission A tower ---------------------------------------------------------------------------
TOWER_MAX_LEVELS = 9  # level scores are listed up to 9 levels (§3 배점 p.5); there are 9 cubes


def tower_target_colors(first_three: Sequence[Color | str], levels: int = TOWER_MAX_LEVELS) -> tuple[Color, ...]:
    """Target colour per level, bottom first. The organisers draw an order of R, G, B for levels
    1-3 and repeat it for 4-6 and 7-9 (§3 용어 p.4; §5 p.7: R-B-G -> R-B-G-R-B-G-R-B-G).
    The order is announced per attempt; vision cannot see it, so someone has to type it in."""
    order = tuple(Color(c) for c in first_three)
    if sorted(c.value for c in order) != sorted(c.value for c in (_R, _G, _B)):
        raise ValueError(f"first_three must be R, G and B once each, got {[c.value for c in order]}")
    if not 1 <= levels <= TOWER_MAX_LEVELS:
        raise ValueError(f"levels must be 1..{TOWER_MAX_LEVELS}")
    return tuple(order[i % 3] for i in range(levels))


def tower_top_height_mm(levels: int) -> float:
    """Height above the board top of the top face of an n-cube tower: 25 mm per level
    (OT p.10 '층마다 놓는 높이를 블록 한 변(25mm)씩 올리기'). Pass it to localize_cube() to map a
    tower top seen in the image; vision cannot measure the level count reliably by itself."""
    if not 1 <= levels <= TOWER_MAX_LEVELS:
        raise ValueError(f"levels must be 1..{TOWER_MAX_LEVELS}")
    return CUBE_SIZE_MM * levels


# --- zones (RULEBOOK frame: origin = robot base rotation axis, +x right, +y towards the pillar,
#     angles from +x towards +y about the origin; §1.1 p.1, 도면 1/3 notes 1-4, 도면 2/3 notes 2, 5) ---
ZONE_R_MIN_MM = 90.0  # every zone spans R90-R250 measured to the inner edge of the grooves (§1.2 p.1; 도면 1/3 note 3)
ZONE_R_MAX_MM = 250.0


@dataclass(frozen=True)
class Sector:
    """An annular sector about the rulebook origin. Bounds are inclusive; the 0.5 mm grooves
    between zones belong to no zone (§3 경계선 p.5), which is below vision accuracy."""

    name: str
    a_min_deg: float
    a_max_deg: float
    r_min_mm: float = ZONE_R_MIN_MM
    r_max_mm: float = ZONE_R_MAX_MM

    def contains(self, x_mm: float, y_mm: float) -> bool:
        r = math.hypot(x_mm, y_mm)
        a = math.degrees(math.atan2(y_mm, x_mm))
        return self.r_min_mm <= r <= self.r_max_mm and self.a_min_deg <= a <= self.a_max_deg


# Mission A: cubes start in zone 1 (0-60 deg and 120-180 deg), the tower must stand in zone 2
# (60-120 deg); a tower whose bottom cube touches a groove does not count (§3 경기 구성 p.5; 도면 1/3).
MISSION_A_ZONE1 = (Sector("A.zone1.right", 0.0, 60.0), Sector("A.zone1.left", 120.0, 180.0))
MISSION_A_ZONE2 = Sector("A.zone2", 60.0, 120.0)
# Mission B: cubes start in 90-180 deg; baskets B 0-20, G 20-50, R 50-90 deg, outer R90-R250
# including their 7 mm walls (§4 p.6; 도면 2/3 notes 2, 3, 5). The R basket's corner at the pillar
# is cut away (도면 2/3 note 4); the pillar's bottom fillet R10 is no-placement (§1.4 p.1).
MISSION_B_PLACEMENT = Sector("B.placement", 90.0, 180.0)
MISSION_B_BASKETS: dict[Color, Sector] = {
    Color.BLUE: Sector("B.basket.blue", 0.0, 20.0),
    Color.GREEN: Sector("B.basket.green", 20.0, 50.0),
    Color.RED: Sector("B.basket.red", 50.0, 90.0),
}

START_SECTORS: dict[Mission, tuple[Sector, ...]] = {
    Mission.A: MISSION_A_ZONE1,
    Mission.B: (MISSION_B_PLACEMENT,),
}


def zone_of(x_mm: float, y_mm: float, mission: Mission | str) -> str | None:
    """Name of the rulebook zone containing a RULEBOOK-frame point, or None (outside / origin side).
    Mission A: "A.zone1.right", "A.zone1.left", "A.zone2"; mission B: "B.placement",
    "B.basket.blue", "B.basket.green", "B.basket.red". Only meaningful if the PixelToWorld in use
    outputs the rulebook frame."""
    m = Mission(mission)
    sectors = (*MISSION_A_ZONE1, MISSION_A_ZONE2) if m is Mission.A else (MISSION_B_PLACEMENT, *MISSION_B_BASKETS.values())
    for s in sectors:
        if s.contains(x_mm, y_mm):
            return s.name
    return None

"""The data that crosses the vision -> kinematics boundary.

Two layers, kept apart on purpose:

1. Image space (always filled, produced by vision): colour, top-face centre and corners in
   pixels, image angle, confidence, flags. Vision does not know where the robot base is, so it
   never produces millimetres.
2. Robot space (``Cube.pose``, optional): filled ONLY by ``localize(scene, pixel_to_world)`` with
   a PixelToWorld supplied by the calibration teammate. ``Scene.localized_by`` names that mapping;
   it is None exactly when no cube has a pose.

Pixel convention (same as cubevision/types.py): u = x grows to the right, v = y grows
downwards, origin at the top-left of the image. Angles in image space are measured from +u
towards +v, i.e. CLOCKWISE on screen.

Everything is a frozen dataclass of plain floats, strings and tuples, so a Scene can be
compared with ``==``, used across threads, and written to / read from JSON exactly
(``to_json`` / ``from_json``; floats survive the round trip bit for bit).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

SCHEMA = "robothon.vision.scene"
SCHEMA_VERSION = 1  # bump on any change that an old reader would misread

PointPx = tuple[float, float]


class Color(str, Enum):
    """Cube colour class. The values are the strings cubevision uses (cubevision/types.py)."""

    RED = "red"
    GREEN = "green"
    BLUE = "blue"
    WOOD = "wood"  # bare wooden cube (rulebook drawing 3/3 C: wood cubes, colour cubes carry printed paper)
    UNKNOWN = "unknown"  # vision found a cube but could not tell its colour


class Mission(str, Enum):
    A = "A"  # tower (rulebook §3, p.4-5)
    B = "B"  # baskets (rulebook §4, p.6)


def _point(p: Sequence[Any], what: str) -> PointPx:
    if len(p) != 2:
        raise ValueError(f"{what}: expected 2 numbers (u, v), got {p!r}")
    u, v = float(p[0]), float(p[1])
    if not (math.isfinite(u) and math.isfinite(v)):
        raise ValueError(f"{what}: not finite: {p!r}")
    return (u, v)


def _angle90(a: Any, what: str) -> float | None:
    if a is None:
        return None
    a = float(a)
    if not (0.0 <= a < 90.0):
        raise ValueError(f"{what} must be in [0, 90) degrees, got {a!r} (fold it with geometry.fold90)")
    return a


@dataclass(frozen=True)
class CubePose:
    """Where a cube is in the ROBOT frame. Produced only by ``localize`` from a PixelToWorld.

    The frame (origin, axes, z zero) is whatever the calibration teammate's PixelToWorld
    outputs -- see README "열린 질문". All lengths in millimetres, angles in degrees.
    """

    x_mm: float  # top-face CENTRE of the cube
    y_mm: float
    z_mm: float  # z of the top face (the plane the pixel was mapped on), in the same frame
    # Direction of one pair of top-face edges, from +x towards +y of that frame, folded into
    # [0, 90). A square looks the same every 90 degrees, so yaw and yaw + k*90 are the SAME pose.
    yaw_deg: float | None
    # The member of {yaw_deg + k*90} closest to bearing_deg (the direction from the frame origin
    # to the cube), so grasp_yaw_deg - bearing_deg is in [-45, 45). If the origin is the robot
    # base rotation axis, that difference is the smallest wrist roll away from a radial
    # top-down approach. None when yaw_deg is None.
    grasp_yaw_deg: float | None

    def __post_init__(self) -> None:
        for name in ("x_mm", "y_mm", "z_mm"):
            v = float(getattr(self, name))
            if not math.isfinite(v):
                raise ValueError(f"CubePose.{name} is not finite: {v!r}")
            object.__setattr__(self, name, v)
        object.__setattr__(self, "yaw_deg", _angle90(self.yaw_deg, "CubePose.yaw_deg"))
        if self.grasp_yaw_deg is not None:
            object.__setattr__(self, "grasp_yaw_deg", float(self.grasp_yaw_deg))

    @property
    def r_mm(self) -> float:
        """Distance from the frame origin in the x-y plane."""
        return math.hypot(self.x_mm, self.y_mm)

    @property
    def bearing_deg(self) -> float:
        """atan2(y, x) in degrees, (-180, 180]: direction from the frame origin to the cube."""
        return math.degrees(math.atan2(self.y_mm, self.x_mm))

    def to_dict(self) -> dict[str, Any]:
        return {"x_mm": self.x_mm, "y_mm": self.y_mm, "z_mm": self.z_mm,
                "yaw_deg": self.yaw_deg, "grasp_yaw_deg": self.grasp_yaw_deg}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CubePose:
        return cls(x_mm=d["x_mm"], y_mm=d["y_mm"], z_mm=d["z_mm"],
                   yaw_deg=d.get("yaw_deg"), grasp_yaw_deg=d.get("grasp_yaw_deg"))


@dataclass(frozen=True)
class Cube:
    """One detected cube, described by its TOP FACE in image pixels (+ optional robot pose)."""

    color: Color
    # Top-face centre in pixels. Physically this is the centre of the top face, i.e. a point
    # CUBE_TOP_HEIGHT_MM = 25 mm above the board top for a cube standing on the board -- not the
    # cube's footprint on the board (see README "어느 평면의 픽셀인가").
    center_px: PointPx
    # The 4 top-face corners in pixels, clockwise on screen, first = smallest u + v.
    # None when the model could not fit the top face (centre and score still valid).
    corners_px: tuple[PointPx, PointPx, PointPx, PointPx] | None
    # Direction of one pair of top-face edges from +u towards +v (clockwise on screen), folded
    # into [0, 90). 0 and 90 are the same pose. None if unknown. NOT a robot angle: the image
    # may be rotated or mirrored relative to the robot; use pose.yaw_deg for motion.
    image_angle_deg: float | None
    area_px2: float  # top-face area in px^2
    score: float  # detection confidence in [0, 1]
    color_score: float = 1.0  # confidence of the colour class in [0, 1]
    # Free-form warnings from vision. Known values (cubevision/postprocess.py):
    # "touches_border" (blob touches the image border), "partial" (partly hidden, e.g. by the
    # arm), "split" (cut out of a blob of several cubes), "side_band" (perspective side face was
    # removed), "large_face" (larger than one cube top: a tower top, or face size is off),
    # "shade_edge", "class_mismatch" (model class and pixel colour disagree), "from_box"/"box_only".
    flags: tuple[str, ...] = ()
    pose: CubePose | None = None  # None until localize() ran with a calibration

    def __post_init__(self) -> None:
        object.__setattr__(self, "color", Color(self.color))
        object.__setattr__(self, "center_px", _point(self.center_px, "Cube.center_px"))
        if self.corners_px is not None:
            if len(self.corners_px) != 4:
                raise ValueError(f"Cube.corners_px needs 4 points, got {len(self.corners_px)}")
            object.__setattr__(self, "corners_px",
                               tuple(_point(p, "Cube.corners_px") for p in self.corners_px))
        object.__setattr__(self, "image_angle_deg", _angle90(self.image_angle_deg, "Cube.image_angle_deg"))
        object.__setattr__(self, "area_px2", float(self.area_px2))
        for name in ("score", "color_score"):
            v = float(getattr(self, name))
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"Cube.{name} must be in [0, 1], got {v!r}")
            object.__setattr__(self, name, v)
        if isinstance(self.flags, str):
            raise TypeError("Cube.flags must be a sequence of strings, not one string")
        object.__setattr__(self, "flags", tuple(str(f) for f in self.flags))
        if self.pose is not None and not isinstance(self.pose, CubePose):
            raise TypeError("Cube.pose must be a CubePose or None")

    def to_dict(self) -> dict[str, Any]:
        return {
            "color": self.color.value,
            "center_px": list(self.center_px),
            "corners_px": [list(p) for p in self.corners_px] if self.corners_px is not None else None,
            "image_angle_deg": self.image_angle_deg,
            "area_px2": self.area_px2,
            "score": self.score,
            "color_score": self.color_score,
            "flags": list(self.flags),
            "pose": self.pose.to_dict() if self.pose is not None else None,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Cube:
        corners = d.get("corners_px")
        pose = d.get("pose")
        return cls(
            color=d["color"],
            center_px=tuple(d["center_px"]),
            corners_px=tuple(tuple(p) for p in corners) if corners is not None else None,
            image_angle_deg=d.get("image_angle_deg"),
            area_px2=d["area_px2"],
            score=d["score"],
            color_score=d.get("color_score", 1.0),
            flags=tuple(d.get("flags", ())),
            pose=CubePose.from_dict(pose) if pose is not None else None,
        )


@dataclass(frozen=True)
class Scene:
    """Everything vision reports about one camera frame."""

    frame_id: int  # increases by 1 per frame from one source; not comparable across sources
    timestamp_s: float  # Unix time (time.time()) when the frame was captured
    image_size_px: tuple[int, int]  # (width, height) of the frame the pixels refer to
    cubes: tuple[Cube, ...]
    mission: Mission | None = None
    source: str = ""  # who made the pixels, e.g. "cubevision:cv.hsv_ranges" or "fake:random"
    # PixelToWorld.name of the calibration that filled the poses; None = pixels only.
    # A name starting with "SYNTHETIC" means test values, never real robot coordinates.
    localized_by: str | None = None
    notes: tuple[str, ...] = ()  # human-readable warnings from vision

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_id", int(self.frame_id))
        object.__setattr__(self, "timestamp_s", float(self.timestamp_s))
        w, h = (int(x) for x in self.image_size_px)
        if w <= 0 or h <= 0:
            raise ValueError(f"Scene.image_size_px must be positive, got {self.image_size_px!r}")
        object.__setattr__(self, "image_size_px", (w, h))
        cubes = tuple(self.cubes)
        for c in cubes:
            if not isinstance(c, Cube):
                raise TypeError(f"Scene.cubes must hold Cube objects, got {type(c).__name__}")
        object.__setattr__(self, "cubes", cubes)
        if self.mission is not None:
            object.__setattr__(self, "mission", Mission(self.mission))
        object.__setattr__(self, "notes", tuple(str(n) for n in self.notes))
        # A scene is either pixels only or fully localised by ONE named calibration.
        with_pose = sum(c.pose is not None for c in cubes)
        if self.localized_by is None and with_pose:
            raise ValueError("cubes have poses but Scene.localized_by is None (use localize())")
        if self.localized_by is not None:
            if not str(self.localized_by).strip():
                raise ValueError("Scene.localized_by must name the PixelToWorld (non-empty)")
            if with_pose != len(cubes):
                raise ValueError("Scene.localized_by is set but some cubes have no pose")

    @property
    def is_localized(self) -> bool:
        return self.localized_by is not None

    @property
    def is_synthetic(self) -> bool:
        """True when the poses are synthetic test values (never use them on the real robot)."""
        return self.localized_by is not None and self.localized_by.upper().startswith("SYNTHETIC")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "frame_id": self.frame_id,
            "timestamp_s": self.timestamp_s,
            "image_size_px": list(self.image_size_px),
            "mission": self.mission.value if self.mission is not None else None,
            "source": self.source,
            "localized_by": self.localized_by,
            "notes": list(self.notes),
            "cubes": [c.to_dict() for c in self.cubes],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Scene:
        if d.get("schema") != SCHEMA:
            raise ValueError(f"not a vision scene: schema={d.get('schema')!r}, expected {SCHEMA!r}")
        if d.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"scene schema_version {d.get('schema_version')!r} is not supported "
                             f"(this reader knows {SCHEMA_VERSION})")
        return cls(
            frame_id=d["frame_id"],
            timestamp_s=d["timestamp_s"],
            image_size_px=tuple(d["image_size_px"]),
            cubes=tuple(Cube.from_dict(c) for c in d["cubes"]),
            mission=d.get("mission"),
            source=d.get("source", ""),
            localized_by=d.get("localized_by"),
            notes=tuple(d.get("notes", ())),
        )

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, allow_nan=False)

    @classmethod
    def from_json(cls, text: str) -> Scene:
        return cls.from_dict(json.loads(text))

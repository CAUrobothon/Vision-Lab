"""Shared result and parameter types for every vision model in the lab.

Every model -- a built-in OpenCV pipeline or an uploaded Ultralytics checkpoint --
returns the same thing: a list of ``Cube`` objects in image pixels.

Coordinate conventions (image space, never millimetres -- calibration is out of scope):
- x grows to the right, y grows downwards, origin at the top-left pixel corner.
- ``quad`` holds the 4 corners of the cube's TOP face, ordered clockwise as seen on
  screen, starting with the corner that has the smallest x + y.
- ``angle`` is the direction of one pair of top-face edges, in degrees, measured from
  the image +x axis towards +y (clockwise on screen), folded into [0, 90).
  A square's orientation repeats every 90 degrees, so 0 and 90 are the same pose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

Color = Literal["red", "green", "blue", "wood", "unknown"]
COLORS: tuple[Color, ...] = ("red", "green", "blue", "wood", "unknown")

Point = tuple[float, float]


def _pt(p: Any) -> list[float]:
    return [round(float(p[0]), 2), round(float(p[1]), 2)]


@dataclass
class Cube:
    """One detected cube (its top face) in image pixels."""

    color: Color
    score: float  # detection confidence in [0, 1]
    center: Point  # top-face centre
    quad: list[Point] | None  # 4 top-face corners (see module docstring), None if unknown
    angle: float | None  # degrees in [0, 90), None if unknown
    area: float  # top-face area in px^2 (quad area, else mask/box area)
    color_score: float = 1.0  # confidence of the colour class in [0, 1]
    polygon: list[Point] | None = None  # raw outline from the model (mask contour or polygon)
    box: tuple[float, float, float, float] | None = None  # raw axis-aligned box x1, y1, x2, y2
    label: str | None = None  # raw class name from a model, if any
    flags: list[str] = field(default_factory=list)  # e.g. "split", "clipped", "side_band"

    def to_dict(self) -> dict[str, Any]:
        return {
            "color": self.color,
            "score": round(float(self.score), 4),
            "color_score": round(float(self.color_score), 4),
            "center": _pt(self.center),
            "quad": [_pt(p) for p in self.quad] if self.quad is not None else None,
            "angle": round(float(self.angle), 2) if self.angle is not None else None,
            "area": round(float(self.area), 1),
            "polygon": [_pt(p) for p in self.polygon] if self.polygon is not None else None,
            "box": [round(float(v), 2) for v in self.box] if self.box is not None else None,
            "label": self.label,
            "flags": list(self.flags),
        }


@dataclass
class FrameResult:
    """What a model returns for one frame."""

    cubes: list[Cube]
    ms: float  # processing time for this frame in milliseconds (decode excluded)
    stages_ms: dict[str, float] = field(default_factory=dict)  # optional per-stage timings
    # Optional debug layers, only filled when the caller asks for debug output:
    # name -> uint8 array, either HxW (mask, 0/255) or HxWx3 (BGR image).
    debug: dict[str, np.ndarray] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)  # human-readable warnings

    def to_dict(self) -> dict[str, Any]:
        return {
            "cubes": [c.to_dict() for c in self.cubes],
            "ms": round(float(self.ms), 3),
            "stages_ms": {k: round(float(v), 3) for k, v in self.stages_ms.items()},
            "notes": list(self.notes),
        }


ParamType = Literal["int", "float", "bool", "select", "multiselect", "range", "color", "roi"]


@dataclass(frozen=True)
class ParamSpec:
    """A tunable parameter. The web UI renders one control per spec.

    Value shapes per ``type``:
    - int / float: a number within [min, max], moved in ``step`` increments (slider)
    - bool: true / false (toggle)
    - select: one of ``options`` values (dropdown or segmented control)
    - multiselect: a list of ``options`` values, in option order (toggle chips); empty = none chosen
    - range: [low, high] within [min, max] (two-handle slider)
    - color: "#rrggbb" sRGB hex (colour swatch; the viewer can pick it from the image)
    - roi: list of {"mode": "include" | "exclude", "points": [[x, y], ...]} polygons in
      image pixels (edited by drawing on the viewer). Empty list = whole image.
    """

    name: str
    label: str  # Korean display label
    type: ParamType
    default: Any
    min: float | None = None
    max: float | None = None
    step: float | None = None
    options: tuple[tuple[str, str], ...] | None = None  # (value, label) pairs for select
    group: str = "기본"  # section heading in the UI
    help: str = ""  # one-line Korean explanation shown under the control
    unit: str = ""  # e.g. "px", "°", "ms"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "type": self.type,
            "default": self.default,
            "min": self.min,
            "max": self.max,
            "step": self.step,
            "options": [{"value": v, "label": lab} for v, lab in self.options] if self.options else None,
            "group": self.group,
            "help": self.help,
            "unit": self.unit,
        }


class ParamError(ValueError):
    """Raised when a parameter value cannot be used."""


def _decimals(x: float | None) -> int:
    """Decimal places of a step or bound as written (0.05 -> 2, 0.5 -> 1, 3 -> 0)."""
    if x is None:
        return 0
    text = repr(float(x))
    return 0 if text.endswith(".0") else len(text.split(".")[1]) if "." in text else 0


def _fit(v: float, s: ParamSpec) -> float:
    """Clamp to [min, max] and snap to the step grid that starts at min, so the value used is
    the value a slider can show. Rounded to the step's decimals (17 x 0.05 is 0.8500000000000001)."""
    lo = float(s.min) if s.min is not None else None
    hi = float(s.max) if s.max is not None else None
    if lo is not None:
        v = max(lo, v)
    if hi is not None:
        v = min(hi, v)
    if s.step:
        base = lo if lo is not None else 0.0
        v = round(base + round((v - base) / float(s.step)) * float(s.step), max(_decimals(s.step), _decimals(lo)))
        if hi is not None and v > hi:
            v = hi
    return v


def _option(v: Any, s: ParamSpec) -> str:
    """A select value as its option string ("5" for 5 or 5.0, "true" for True). Not an option = error."""
    allowed = [o[0] for o in (s.options or ())]
    if isinstance(v, bool):
        text = str(v).lower()
    elif isinstance(v, float) and v.is_integer():
        text = str(int(v))
    else:
        text = str(v)
    if text not in allowed:
        raise ParamError(f"{s.name}: {v!r}은(는) 고를 수 있는 값이 아닙니다 (가능: {', '.join(allowed)})")
    return text


def resolve_params(specs: list[ParamSpec], given: dict[str, Any] | None) -> dict[str, Any]:
    """Fill defaults, coerce types, clamp numbers into [min, max] and snap them to the step grid.
    Unknown keys are dropped; a value that cannot be used at all raises ParamError."""
    given = given or {}
    out: dict[str, Any] = {}
    for s in specs:
        v = given.get(s.name, s.default)
        try:
            if s.type == "int":
                v = int(round(_fit(float(v), s)))
            elif s.type == "float":
                v = _fit(float(v), s)
            elif s.type == "bool":
                v = v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes", "on")
            elif s.type == "select":
                v = _option(v, s)
            elif s.type == "multiselect":
                chosen = {str(x) for x in (v if isinstance(v, (list, tuple)) else [])}
                v = [o[0] for o in (s.options or ()) if o[0] in chosen]
            elif s.type == "range":
                lo, hi = float(v[0]), float(v[1])
                if lo > hi:
                    lo, hi = hi, lo
                # Both ends inside [min, max]: [600, 700] on 0-500 is [500, 500], never [600, 500].
                lo, hi = _fit(lo, s), _fit(hi, s)
                v = [lo, hi]
                if isinstance(s.default[0], int) and isinstance(s.default[1], int):
                    v = [int(round(lo)), int(round(hi))]
            elif s.type == "color":
                v = str(v).strip().lower()
                if not (len(v) == 7 and v[0] == "#" and all(c in "0123456789abcdef" for c in v[1:])):
                    raise ParamError(f"{s.name}: '#rrggbb' 형식이 아닙니다")
            elif s.type == "roi":
                polys = []
                for poly in v or []:
                    mode = poly.get("mode", "include")
                    pts = [[float(p[0]), float(p[1])] for p in poly.get("points", [])]
                    if mode in ("include", "exclude") and len(pts) >= 3:
                        polys.append({"mode": mode, "points": pts})
                v = polys
        except ParamError:
            raise
        except Exception as exc:  # noqa: BLE001 - report any malformed value the same way
            raise ParamError(f"{s.name}: 값을 해석할 수 없습니다 ({v!r})") from exc
        out[s.name] = v
    return out

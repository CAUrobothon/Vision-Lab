"""Draw cubes onto a BGR frame (thumbnails, exported PNG/MP4, debugging).

The web UI draws its own overlays from JSON; this is only for baked images.
Accepts Cube objects or the dicts produced by Cube.to_dict().
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from cubevision.types import Cube

# BGR colours for overlays (bright enough to read on a white board and on dark frames).
OVERLAY_BGR: dict[str, tuple[int, int, int]] = {
    "red": (40, 40, 235),
    "green": (60, 190, 40),
    "blue": (230, 110, 20),
    "wood": (60, 150, 215),
    "unknown": (200, 0, 200),
}
SHORT = {"red": "R", "green": "G", "blue": "B", "wood": "W", "unknown": "?"}


def _as_dict(c: Cube | dict[str, Any]) -> dict[str, Any]:
    return c.to_dict() if isinstance(c, Cube) else c


def draw_cubes(
    frame_bgr: np.ndarray,
    cubes: list[Cube] | list[dict[str, Any]],
    *,
    thickness: int | None = None,
    labels: bool = True,
    show_polygon: bool = False,
) -> np.ndarray:
    """Return a copy of ``frame_bgr`` with quads, centres, angle ticks and labels drawn."""
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    t = thickness or max(1, round(min(h, w) / 360))
    font_scale = max(0.4, min(h, w) / 1400)
    for raw in cubes:
        c = _as_dict(raw)
        col = OVERLAY_BGR.get(c.get("color", "unknown"), OVERLAY_BGR["unknown"])
        if show_polygon and c.get("polygon"):
            poly = np.round(np.asarray(c["polygon"], np.float64)).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(out, [poly], True, (160, 160, 160), 1, cv2.LINE_AA)
        if c.get("quad"):
            q = np.round(np.asarray(c["quad"], np.float64)).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(out, [q], True, col, t, cv2.LINE_AA)
        elif c.get("box"):
            x1, y1, x2, y2 = (int(round(v)) for v in c["box"])
            cv2.rectangle(out, (x1, y1), (x2, y2), col, t, cv2.LINE_AA)
        cx, cy = c["center"]
        cpt = (int(round(cx)), int(round(cy)))
        cv2.circle(out, cpt, t + 2, col, -1, cv2.LINE_AA)
        ang = c.get("angle")
        if ang is not None:
            r = 0.35 * math.sqrt(max(c.get("area", 0.0), 1.0))
            a = math.radians(ang)
            tip = (int(round(cx + r * math.cos(a))), int(round(cy + r * math.sin(a))))
            cv2.line(out, cpt, tip, col, t, cv2.LINE_AA)
        if labels:
            txt = SHORT.get(c.get("color", "unknown"), "?")
            if ang is not None:
                txt += f" {ang:.0f}"
            org = (cpt[0] + 6, cpt[1] - 6)
            cv2.putText(out, txt, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), t + 2, cv2.LINE_AA)
            cv2.putText(out, txt, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), max(1, t - 1), cv2.LINE_AA)
    return out

"""Detection region (ROI): include / exclude polygons in image pixels.

Value shape (ParamSpec type "roi"): [{"mode": "include" | "exclude", "points": [[x, y], ...]}, ...]
Semantics: if any include polygon exists, the region is the union of the include polygons;
otherwise it is the whole image. Exclude polygons are then removed from it.
An empty list therefore means "the whole image".
"""

from __future__ import annotations

from typing import Any, Iterable

import cv2
import numpy as np

from cubevision.types import Cube

Roi = list[dict[str, Any]]


def _polys(roi: Roi | None, mode: str) -> list[np.ndarray]:
    out = []
    for poly in roi or []:
        if poly.get("mode") == mode and len(poly.get("points", [])) >= 3:
            out.append(np.round(np.asarray(poly["points"], dtype=np.float64)).astype(np.int32))
    return out


def roi_mask(shape_hw: tuple[int, int], roi: Roi | None) -> np.ndarray | None:
    """uint8 mask (255 = inside the region) or None when the region is the whole image."""
    if not roi:
        return None
    h, w = int(shape_hw[0]), int(shape_hw[1])
    inc, exc = _polys(roi, "include"), _polys(roi, "exclude")
    if not inc and not exc:
        return None
    mask = np.zeros((h, w), np.uint8) if inc else np.full((h, w), 255, np.uint8)
    # One fillPoly call per polygon: a single call with several polygons fills even-odd, so
    # where two of them overlap they cancel out (a hole), unlike the union point_in_roi uses.
    for p in inc:
        cv2.fillPoly(mask, [p], 255)
    for p in exc:
        cv2.fillPoly(mask, [p], 0)
    return mask


def point_in_roi(pt: Iterable[float], roi: Roi | None) -> bool:
    if not roi:
        return True
    x, y = (float(v) for v in pt)
    inc, exc = _polys(roi, "include"), _polys(roi, "exclude")
    inside = True
    if inc:
        inside = any(cv2.pointPolygonTest(p.reshape(-1, 1, 2), (x, y), False) >= 0 for p in inc)
    if inside and exc:
        inside = not any(cv2.pointPolygonTest(p.reshape(-1, 1, 2), (x, y), False) >= 0 for p in exc)
    return inside


def filter_cubes(cubes: list[Cube], roi: Roi | None) -> list[Cube]:
    """Keep cubes whose centre lies inside the region."""
    if not roi:
        return cubes
    return [c for c in cubes if point_in_roi(c.center, roi)]

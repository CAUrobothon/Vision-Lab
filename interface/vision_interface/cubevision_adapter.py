"""Vision side: turn a cubevision result into a Scene, and run a camera + pipeline as a VisionSource.

This module never imports cubevision, numpy or OpenCV. It reads cubevision's result by duck
typing, so it accepts either live objects (cubevision.types.FrameResult / Cube) or their
``to_dict()`` form (e.g. received as JSON from another process), and the package stays
importable on a kinematics laptop that has none of those installed.

Field mapping (cubevision/types.py -> scene.Cube):
    color        -> color          (same strings: red/green/blue/wood/unknown)
    center       -> center_px      (top-face centre)
    quad         -> corners_px     (clockwise on screen, first = smallest x+y; unchanged)
    angle        -> image_angle_deg (folded into [0, 90): to_dict() rounds 89.996 to 90.0)
    area         -> area_px2
    score, color_score, flags      -> same names
    FrameResult.notes              -> Scene.notes
Nothing is converted to millimetres here: poses stay None (see geometry.localize).
"""

from __future__ import annotations

import time
from typing import Any, Callable, Mapping

from vision_interface.geometry import fold90
from vision_interface.scene import Cube, Mission, Scene


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def cube_from_cubevision(c: Any) -> Cube:
    """One cubevision Cube (object or dict) -> Cube (pixels only)."""
    quad = _get(c, "quad")
    angle = _get(c, "angle")
    return Cube(
        color=str(_get(c, "color")),
        center_px=tuple(float(v) for v in _get(c, "center")),  # type: ignore[arg-type]
        corners_px=tuple((float(p[0]), float(p[1])) for p in quad) if quad is not None else None,  # type: ignore[arg-type]
        image_angle_deg=fold90(float(angle)) if angle is not None else None,
        area_px2=float(_get(c, "area")),
        score=float(_get(c, "score")),
        color_score=float(_get(c, "color_score", 1.0)),
        flags=tuple(_get(c, "flags", None) or ()),
    )


def scene_from_cubevision(
    result: Any,
    *,
    frame_id: int,
    timestamp_s: float,
    image_size_px: tuple[int, int],
    mission: Mission | str | None = None,
    source: str = "cubevision",
) -> Scene:
    """A cubevision FrameResult (object or dict), or a plain list of its cubes -> pixel-only Scene."""
    if isinstance(result, (list, tuple)):
        raw_cubes, notes = result, ()
    else:
        raw_cubes, notes = _get(result, "cubes"), tuple(_get(result, "notes", None) or ())
    return Scene(
        frame_id=frame_id,
        timestamp_s=timestamp_s,
        image_size_px=image_size_px,
        cubes=tuple(cube_from_cubevision(c) for c in raw_cubes),
        mission=Mission(mission) if mission is not None else None,
        source=source,
        notes=notes,
    )


class CubevisionSource:
    """The real VisionSource: grab a frame, run a cubevision pipeline, return a pixel-only Scene.

        import cv2
        from cubevision.registry import get_pipeline
        cap = cv2.VideoCapture(0)
        pipe = get_pipeline("cv.hsv_ranges")
        def grab():
            for _ in range(BUFFERED_FRAMES):   # TODO measure: frames the driver buffers
                cap.grab()
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("camera read failed")
            return frame
        source = CubevisionSource(grab, lambda f: pipe(f, my_params), source="cubevision:cv.hsv_ranges")

    ``grab_frame`` must return an HxWx3 BGR frame captured AFTER the call started (drain the
    driver's buffer; how many frames it holds depends on the camera and OS -- measure it).
    """

    def __init__(self, grab_frame: Callable[[], Any], detect: Callable[[Any], Any], *,
                 mission: Mission | str | None = None, source: str = "cubevision") -> None:
        self._grab = grab_frame
        self._detect = detect
        self._mission = Mission(mission) if mission is not None else None
        self._source = source
        self._frame_id = 0

    def get_scene(self) -> Scene:
        frame = self._grab()
        t = time.time()
        h, w = int(frame.shape[0]), int(frame.shape[1])
        result = self._detect(frame)
        self._frame_id += 1
        return scene_from_cubevision(result, frame_id=self._frame_id, timestamp_s=t, image_size_px=(w, h),
                                     mission=self._mission, source=self._source)

"""cubevision: find 25 mm cubes (red / green / blue / wood) seen from an overhead camera.

Pure OpenCV + NumPy. No web, no torch. Safe to import from the competition program.
"""

from cubevision.types import COLORS, Cube, FrameResult, ParamSpec, resolve_params

__all__ = ["COLORS", "Cube", "FrameResult", "ParamSpec", "resolve_params"]

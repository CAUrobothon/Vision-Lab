"""vision_interface: the contract between vision (pixels), calibration (pixel -> robot mm) and
kinematics (motion) for the 2026 robothon pick-and-place competition.

Pure standard library, Python >= 3.10. README.md (Korean) explains units, frames and open questions.

    from vision_interface import Color, FakeVisionSource, localize, by_color, nearest_first
"""

from vision_interface.cubevision_adapter import CubevisionSource, cube_from_cubevision, scene_from_cubevision
from vision_interface.fake import FakeVisionSource, SyntheticPixelToWorld, random_scene
from vision_interface.geometry import by_color, closest_grasp_yaw, fold90, localize, localize_cube, nearest_first
from vision_interface.homography import HomographyPixelToWorld, fit_homography
from vision_interface.protocols import PixelToWorld, VisionSource
from vision_interface.scene import SCHEMA, SCHEMA_VERSION, Color, Cube, CubePose, Mission, Scene

__version__ = "0.1.0"

__all__ = [
    "SCHEMA", "SCHEMA_VERSION", "Color", "Cube", "CubePose", "Mission", "Scene",
    "VisionSource", "PixelToWorld",
    "localize", "localize_cube", "fold90", "closest_grasp_yaw", "by_color", "nearest_first",
    "HomographyPixelToWorld", "fit_homography",
    "FakeVisionSource", "SyntheticPixelToWorld", "random_scene",
    "CubevisionSource", "scene_from_cubevision", "cube_from_cubevision",
]

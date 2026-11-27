"""The two contracts at the team boundaries.

    vision (this package)          calibration teammate              kinematics teammates
    VisionSource.get_scene()  -->  PixelToWorld.pixel_to_world()  -->  read cube.pose (mm)
    pixels only                    pixel -> robot-base mm             motion, IK, strategy

``localize(scene, pixel_to_world)`` (geometry.py) is the only place where the two meet.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from vision_interface.scene import Scene


@runtime_checkable
class VisionSource(Protocol):
    """What the kinematics code calls to see the arena. Implemented by the vision side
    (cubevision_adapter.CubevisionSource) and by fake.FakeVisionSource for development."""

    def get_scene(self) -> Scene:
        """Return the cubes seen in ONE camera frame, in image pixels (Scene.localized_by is None).

        Real sources capture a new frame after this call starts (never a frame cached from
        before the call), run detection on it and return; they block for that long. Call it
        while the arm does not hide the cubes you care about: vision only reports what the
        camera sees (a partly hidden cube carries the flag "partial").
        """
        ...


@runtime_checkable
class PixelToWorld(Protocol):
    """Pixel -> robot-base millimetres. OWNED BY THE CALIBRATION TEAMMATE.

    Vision never implements this for the competition: it does not know where the robot base is.
    ``homography.HomographyPixelToWorld`` is a template the calibration teammate may fill with
    their own matrix; ``fake.SyntheticPixelToWorld`` is a synthetic stand-in for tests.
    """

    #: Short description of THIS calibration (e.g. "homography table B 2026-11-29 18:20").
    #: Copied into Scene.localized_by, so every pose says where its millimetres came from.
    #: Start it with "SYNTHETIC" for test mappings.
    name: str

    def pixel_to_world(self, u_px: float, v_px: float, height_mm: float) -> tuple[float, float, float]:
        """Robot-frame (x_mm, y_mm, z_mm) of the point that pixel (u_px, v_px) sees, given that the
        point lies ``height_mm`` above the TOP SURFACE OF THE BOARD (rulebook §1.1, p.1: z = 0 is
        the board top on which cubes stand).

        Inputs:  u_px, v_px in the same frames vision uses (same resolution, same undistortion);
                 height_mm = 25.0 for the top-face centre/corners of a cube standing on the board
                 (rules.CUBE_TOP_HEIGHT_MM), 29.0 for a cube on a basket floor, 25*n for the top
                 of an n-cube tower.
        Output:  (x_mm, y_mm, z_mm) in the robot frame chosen by the calibration teammate,
                 including z (so the kinematics side never has to know where the board top is).
        Raise ValueError for a height the calibration cannot map.
        """
        ...

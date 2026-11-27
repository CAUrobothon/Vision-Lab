"""TEMPLATE PixelToWorld for the calibration teammate: a 3x3 homography you fill with YOUR
calibrated matrix. This module ships no numbers -- every value comes from your calibration.

How to use it (pure Python, no OpenCV needed; a matrix from cv2.findHomography works as well):

    # pixels: top-face centres vision reported (Cube.center_px) for cubes you put at known spots
    # world:  where those top-face centres are in YOUR robot frame (x_mm, y_mm)
    H = fit_homography(pixels, world)              # >= 4 points, not all on one line
    p2w = HomographyPixelToWorld(H, calibrated_height_mm=25.0, board_top_z_mm=<your z of the
                                 board top>, name="homography table B 2026-11-29 18:20")

Why "calibrated_height_mm": a homography is exact for ONE plane only. Vision's pixels are
top-face centres, 25 mm above the board top. A matrix fitted on board-level marks (height 0)
and applied to them puts every cube about 4.3 % too far from the point under the camera
(lens 601 mm above the board, §1.3 p.1: 601/576 = 1.043), i.e. 4 mm at 100 mm and ~13 mm at the
arena rim. Fit on the 25 mm plane (recommended), or give ``camera_xyz_mm`` so other heights are
corrected along the camera ray.
"""

from __future__ import annotations

import math
from typing import Sequence

Matrix3 = tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]


def _as_matrix(m: Sequence[Sequence[float]]) -> Matrix3:
    rows = [list(r) for r in m]
    if len(rows) != 3 or any(len(r) != 3 for r in rows):
        raise ValueError("homography must be a 3x3 matrix")
    out = tuple(tuple(float(v) for v in r) for r in rows)
    if not all(math.isfinite(v) for r in out for v in r):
        raise ValueError("homography has non-finite entries")
    return out  # type: ignore[return-value]


def _det3(m: Matrix3) -> float:
    (a, b, c), (d, e, f), (g, h, i) = m
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def _inv3(m: Matrix3) -> Matrix3:
    (a, b, c), (d, e, f), (g, h, i) = m
    det = _det3(m)
    if abs(det) < 1e-12:
        raise ValueError("homography is singular (not invertible)")
    return (
        ((e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det),
        ((f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det),
        ((d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det),
    )


def _mul3(a: Matrix3, b: Matrix3) -> Matrix3:
    return tuple(tuple(sum(a[r][k] * b[k][c] for k in range(3)) for c in range(3)) for r in range(3))  # type: ignore[return-value]


def _apply(m: Matrix3, x: float, y: float) -> tuple[float, float]:
    w = m[2][0] * x + m[2][1] * y + m[2][2]
    if abs(w) < 1e-12:
        raise ValueError(f"point ({x}, {y}) maps to infinity under this homography")
    return ((m[0][0] * x + m[0][1] * y + m[0][2]) / w, (m[1][0] * x + m[1][1] * y + m[1][2]) / w)


class HomographyPixelToWorld:
    """PixelToWorld TEMPLATE: fill with your calibrated matrix. Every argument is required except
    camera_xyz_mm; nothing has a default value.

    matrix:               3x3, maps pixel (u, v, 1) to robot-frame (x, y, 1) for points on the
                          calibrated plane.
    calibrated_height_mm: height above the board top of the plane the matrix was fitted on
                          (25.0 if you fitted on cube top-face centres; 0.0 for board marks).
    board_top_z_mm:       robot-frame z of the board top surface, so that
                          z_mm = board_top_z_mm + height_mm.
    camera_xyz_mm:        optional robot-frame position of the lens centre. Only needed to map
                          heights other than calibrated_height_mm (exact for an undistorted
                          pinhole camera at any tilt). Without it such heights raise ValueError.
    name:                 recorded in Scene.localized_by.
    """

    def __init__(
        self,
        matrix: Sequence[Sequence[float]],
        *,
        calibrated_height_mm: float,
        board_top_z_mm: float,
        name: str,
        camera_xyz_mm: Sequence[float] | None = None,
    ) -> None:
        self.matrix = _as_matrix(matrix)
        self._inverse = _inv3(self.matrix)
        self.calibrated_height_mm = float(calibrated_height_mm)
        self.board_top_z_mm = float(board_top_z_mm)
        if not str(name).strip():
            raise ValueError("name must describe this calibration")
        self.name = str(name)
        self.camera_xyz_mm: tuple[float, float, float] | None = None
        if camera_xyz_mm is not None:
            cx, cy, cz = (float(v) for v in camera_xyz_mm)
            if cz <= self.board_top_z_mm + self.calibrated_height_mm:
                raise ValueError("camera_xyz_mm must be above the calibrated plane")
            self.camera_xyz_mm = (cx, cy, cz)

    def _scale(self, height_mm: float) -> float:
        """Ratio by which a point at height_mm sits closer to the lens's foot point than the point
        on the calibrated plane that the same pixel hits (similar triangles along the ray)."""
        if abs(height_mm - self.calibrated_height_mm) < 1e-9:
            return 1.0
        if self.camera_xyz_mm is None:
            raise ValueError(
                f"this homography is calibrated at {self.calibrated_height_mm} mm above the board; "
                f"mapping {height_mm} mm needs camera_xyz_mm (lens centre in your robot frame)")
        cz = self.camera_xyz_mm[2]
        z0 = self.board_top_z_mm + self.calibrated_height_mm
        z = self.board_top_z_mm + height_mm
        if z >= cz:
            raise ValueError("height is at or above the camera")
        return (cz - z) / (cz - z0)

    def pixel_to_world(self, u_px: float, v_px: float, height_mm: float) -> tuple[float, float, float]:
        x0, y0 = _apply(self.matrix, float(u_px), float(v_px))
        s = self._scale(float(height_mm))
        z = self.board_top_z_mm + float(height_mm)
        if s == 1.0:
            return (x0, y0, z)
        cx, cy, _ = self.camera_xyz_mm  # type: ignore[misc]
        return (cx + (x0 - cx) * s, cy + (y0 - cy) * s, z)

    def world_to_pixel(self, x_mm: float, y_mm: float, height_mm: float) -> tuple[float, float]:
        """Inverse mapping (e.g. to draw a robot target on the image)."""
        s = self._scale(float(height_mm))
        if s != 1.0:
            cx, cy, _ = self.camera_xyz_mm  # type: ignore[misc]
            x_mm, y_mm = cx + (x_mm - cx) / s, cy + (y_mm - cy) / s
        return _apply(self._inverse, float(x_mm), float(y_mm))


def _normaliser(pts: list[tuple[float, float]]) -> Matrix3:
    """Hartley normalisation: move the centroid to 0 and scale the mean distance to sqrt(2)."""
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    d = sum(math.hypot(p[0] - mx, p[1] - my) for p in pts) / n
    if d < 1e-12:
        raise ValueError("all points coincide")
    s = math.sqrt(2.0) / d
    return ((s, 0.0, -s * mx), (0.0, s, -s * my), (0.0, 0.0, 1.0))


def _solve(a: list[list[float]], b: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting (small dense systems)."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            raise ValueError("points are degenerate (e.g. three or more on one line)")
        m[col], m[piv] = m[piv], m[col]
        for r in range(col + 1, n):
            f = m[r][col] / m[col][col]
            for k in range(col, n + 1):
                m[r][k] -= f * m[col][k]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        x[r] = (m[r][n] - sum(m[r][k] * x[k] for k in range(r + 1, n))) / m[r][r]
    return x


def fit_homography(pixels: Sequence[Sequence[float]], world_xy: Sequence[Sequence[float]]) -> Matrix3:
    """Least-squares homography from >= 4 pixel <-> robot-frame (x, y) pairs on ONE plane
    (normalised DLT with h33 = 1). Exact for 4 points in general position."""
    src = [(float(p[0]), float(p[1])) for p in pixels]
    dst = [(float(p[0]), float(p[1])) for p in world_xy]
    if len(src) != len(dst) or len(src) < 4:
        raise ValueError("need the same number (>= 4) of pixel and world points")
    tp, tw = _normaliser(src), _normaliser(dst)
    ns = [_apply(tp, *p) for p in src]
    nd = [_apply(tw, *p) for p in dst]
    rows: list[list[float]] = []
    rhs: list[float] = []
    for (x, y), (X, Y) in zip(ns, nd):
        rows.append([x, y, 1.0, 0.0, 0.0, 0.0, -x * X, -y * X])
        rhs.append(X)
        rows.append([0.0, 0.0, 0.0, x, y, 1.0, -x * Y, -y * Y])
        rhs.append(Y)
    # Normal equations (8x8); well conditioned thanks to the normalisation.
    ata = [[sum(r[i] * r[j] for r in rows) for j in range(8)] for i in range(8)]
    atb = [sum(r[i] * v for r, v in zip(rows, rhs)) for i in range(8)]
    h = _solve(ata, atb)
    hn: Matrix3 = ((h[0], h[1], h[2]), (h[3], h[4], h[5]), (h[6], h[7], 1.0))
    full = _mul3(_mul3(_inv3(tw), hn), tp)
    k = full[2][2]
    if abs(k) < 1e-12:
        raise ValueError("degenerate homography")
    return tuple(tuple(v / k for v in r) for r in full)  # type: ignore[return-value]

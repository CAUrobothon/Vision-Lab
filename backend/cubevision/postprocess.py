"""Turn any model output (quad, polygon, mask, box) into a Cube with a top-face quad.

CONTRACT (the lab's Ultralytics adapter and every pipeline rely on these signatures;
internals may be improved, e.g. better top-face isolation):

- order_quad(points) -> list[Point]            4 corners, clockwise on screen, first = smallest x+y
- quad_angle(quad) -> float                     degrees in [0, 90)
- cube_from_quad(frame_bgr, quad, *, score, color=None, label=None, prototypes=None,
                 polygon=None, box=None, flags=()) -> Cube
- cube_from_polygon(frame_bgr, polygon, *, score, color=None, label=None, prototypes=None,
                    top_face=True, face_px=None, box=None) -> Cube | None
- cube_from_mask(frame_bgr, mask, *, score, color=None, label=None, prototypes=None,
                 top_face=True, face_px=None) -> Cube | None
- cube_from_box(frame_bgr, box, *, score, color=None, label=None, prototypes=None,
                face_px=None) -> Cube | None
  Every entry point also takes ``geo: Geometry | None`` (all post-processing tunables; its
  face_px is replaced by the ``face_px`` argument) and ``colour_opts: dict | None`` (keyword
  arguments of colour.colour_model). None = the defaults.

``color=None`` means: classify from the pixels inside the top face (colour.classify_region).
A given ``color`` is used as it is (the pixels can only add the flag class_mismatch), and these
functions never drop a model's detection for its colour.
``face_px`` is the expected top-face edge length in pixels (about 61 px for the competition
camera); None means unknown, so no size-based correction is applied.

Shared post-processing (``cubes_from_blobs``, used by every built-in pipeline) follows
docs/research/classical-cv.json, stages 0-7 (docs/PIPELINES.md lists where and why it differs):

0. Any model output becomes a blob: a binary mask crop (``Blob``).
1. Gates: too small = noise; larger than ``max_area_ratio`` faces = a large region (basket
   floor, arm), never a cube; thinner than ``sliver_ratio`` faces = a side-face strip.
2. Coarse angle from ``cv2.boxPoints(cv2.minAreaRect(...))`` (never the rect's angle field, whose
   range differs between OpenCV releases; never PCA, which is isotropic for a square). For a
   blob of several cubes whose rectangle disagrees with its outline, the outline's direction.
3. Extents in the cube frame from 1-px cross-section counts (thin spurs never reach half a face,
   so they do not move the edges). The side band that perspective shows on the camera-axis side
   is predicted as kappa * distance from the principal point; from that, how many cubes the
   blob holds. A flush block is tiled from its outer corner, a single row (shifted or not) is cut
   into face-wide slabs, anything else is split (distance-transform watershed, else k-means).
4. Top face: for a straight-down camera the pixel farthest from the principal point is always
   an outer top-face corner, so the top face is the face_px square anchored on the outer edges.
   Without a band (cube near the centre, or paper on the top only) the measured edges are used;
   a visible top/side brightness step or a clearly larger blob (a tower top) sets the size.
   A partly hidden cube is anchored on the edges that have board outside them.
5. A candidate lying mostly inside another cube's silhouette is that cube's side face: dropped.
6. Colour from the median of the top face shrunk by 15 %, corrected to the reference light
   (colour.py). A built-in pipeline's board / shadow coloured candidates are dropped.
7. Output a Cube: quad, centre, angle in [0, 90), flags.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Iterable

import cv2
import numpy as np

from cubevision.colour import ColourModel, colour_model, lab8_to_real, median_bgr
from cubevision.types import Color, Cube, Point

FACE_PX = 61.0  # top-face edge of a cube on the board: 1410.6 px * 25 mm / 576 mm
KAPPA = 25.0 / 601.0  # base = top face pulled towards the principal point by this fraction


# --- quad helpers (contract) ---------------------------------------------------------------
def order_quad(points: Iterable[Iterable[float]]) -> list[Point]:
    pts = np.asarray(list(points), np.float64).reshape(-1, 2)
    if len(pts) != 4:
        raise ValueError("a quad needs exactly 4 points")
    c = pts.mean(axis=0)
    # Angle from centre; in image coords (y down) increasing atan2 is clockwise on screen.
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    pts = pts[np.argsort(ang)]
    start = int(np.argmin(pts.sum(axis=1)))
    pts = np.roll(pts, -start, axis=0)
    return [(float(x), float(y)) for x, y in pts]


def quad_angle(quad: Iterable[Iterable[float]]) -> float:
    q = np.asarray(list(quad), np.float64).reshape(4, 2)
    # Each edge direction folded into [0, 90), then a circular mean on the 90-degree period
    # (average cos 4a and sin 4a), so 0.5 and 89.5 average to 0, not 45.
    angs = []
    for i in range(4):
        dx, dy = q[(i + 1) % 4] - q[i]
        angs.append(np.degrees(np.arctan2(dy, dx)) % 90.0)
    rad = np.radians(np.asarray(angs) * 4.0)
    mean = np.degrees(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())) / 4.0
    return _fold90(mean)


def _fold90(deg: float) -> float:
    """``deg`` folded into [0, 90). Plain ``% 90.0`` returns 90.0 itself for a tiny negative
    input (-1e-15 % 90.0 rounds to 90.0), which is the same orientation as 0."""
    a = float(deg) % 90.0
    return 0.0 if a >= 90.0 else a


def angle_diff(a: float, b: float) -> float:
    """Signed difference of two square orientations in degrees, in [-45, 45)."""
    return _fold90(a - b + 45.0) - 45.0


def coarse_angle(points: np.ndarray) -> float:
    """Orientation in [0, 90) of the minimum-area rectangle around ``points`` (N, 2)."""
    box = cv2.boxPoints(cv2.minAreaRect(np.asarray(points, np.float32).reshape(-1, 1, 2)))
    e = box[1] - box[0]
    if float(np.hypot(*e)) < 1e-6:
        e = box[2] - box[1]
    return _fold90(math.degrees(math.atan2(float(e[1]), float(e[0]))))


# --- blobs -----------------------------------------------------------------------------------
@dataclass
class Blob:
    """A candidate region from a model: a binary mask crop and its offset in the frame."""

    mask: np.ndarray  # uint8 crop, non-zero = inside
    x0: int
    y0: int
    color: str | None = None  # class proposed by the model; None = decide from the pixels
    score: float = 1.0
    label: str | None = None

    @property
    def area(self) -> int:
        return int(np.count_nonzero(self.mask))

    def outline(self, eps: float = 1.0) -> list[Point] | None:
        """Simplified outer contour in frame pixels (for the raw-output overlay)."""
        cnts, _ = cv2.findContours((self.mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        c = cv2.approxPolyDP(max(cnts, key=cv2.contourArea), eps, True).reshape(-1, 2)
        return [(float(x + self.x0), float(y + self.y0)) for x, y in c]


def blobs_from_mask(
    mask: np.ndarray,
    color: str | None = None,
    *,
    min_area: float = 0.0,
    score: float = 1.0,
    fill_holes: bool = True,
    fill_max: float = 1.5,
    connectivity: int = 8,
) -> list[Blob]:
    """One Blob per ``connectivity``-connected component of ``mask``.

    Small holes (glare, a printed digit) are filled. A blob whose holes would make it more than
    ``fill_max`` times larger is a ring (e.g. mixed-colour edge pixels around a cube of another
    colour) and is kept as it is, so it cannot pose as a solid face."""
    m = (mask > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=int(connectivity))
    out = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area < min_area:
            continue
        crop = (labels[y : y + h, x : x + w] == i).astype(np.uint8) * 255
        if fill_holes:
            filled = crop.copy()
            cnts, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(filled, cnts, -1, 255, cv2.FILLED)
            if np.count_nonzero(filled) <= fill_max * area:
                crop = filled
        out.append(Blob(crop, x, y, color, score))
    return out


def blob_from_polygon(polygon: np.ndarray, shape_hw: tuple[int, int], color: str | None = None, score: float = 1.0) -> Blob | None:
    """Rasterise a polygon (frame pixels) into a Blob clipped to the frame."""
    pts = np.asarray(polygon, np.float64).reshape(-1, 2)
    h, w = shape_hw
    x0 = int(max(0, math.floor(pts[:, 0].min()) - 1))
    y0 = int(max(0, math.floor(pts[:, 1].min()) - 1))
    x1 = int(min(w, math.ceil(pts[:, 0].max()) + 2))
    y1 = int(min(h, math.ceil(pts[:, 1].max()) + 2))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    crop = np.zeros((y1 - y0, x1 - x0), np.uint8)
    p = np.round((pts - (x0, y0)) * 16).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(crop, [p], 255, cv2.LINE_8, shift=4)
    if not crop.any():
        return None
    return Blob(crop, x0, y0, color, score)


# --- geometry of one blob (stages 1-4) -------------------------------------------------------
@dataclass
class Geometry:
    """Scene constants and every tunable of the shared post-processing.

    The lab shows each field as a parameter (pipelines.common.geometry_params); the defaults are
    the values the pipelines were built and measured with."""

    face_px: float = FACE_PX
    pp: tuple[float, float] | None = None  # principal point; None = image centre
    kappa: float = KAPPA  # cube height / camera height above the board (side band per px from pp)
    min_area_ratio: float = 0.35  # below this many face areas: noise
    max_area_ratio: float = 8.0  # above: large region (basket floor, arm), never a cube
    sliver_ratio: float = 0.6  # short extent below this many faces: a side-face strip
    partial_ratio: float = 0.85  # an extent below this: the cube is partly hidden
    snap_tol: float = 0.06  # an extent above (1 + snap_tol) faces holds a side band
    split: bool = True  # split merged blobs of touching cubes
    shrink: float = 0.15  # colour is sampled on the top face shrunk by this fraction per side
    border_px: int = 2  # a blob this close to the frame edge is flagged touches_border
    shade_snap: bool = True  # move the inner top-face edge onto the top/side brightness step
    shade_t: float = 6.0  # minimum brightness step for that, in L* units
    drop_side_faces: bool = True  # drop candidates lying inside another cube's silhouette
    # Extents: an edge is where the 1-px cross-section count reaches this fraction of a face.
    edge_level: float = 0.5
    # Counting: up to this many faces (side band removed) along an axis is still one cube.
    single_max: float = 1.6
    block_fill: float = 0.85  # a multi-cube blob filling its rectangle this much is a flush block
    tile_coverage: float = 0.6  # a tile of a block covered less than this is an empty corner
    split_peak: float = 0.7  # watershed seeds: distance transform above this x half a face
    split_iters: int = 15  # k-means iterations of the fallback split
    min_coverage: float = 0.5  # a top face covered less than this by its blob is a ring: dropped
    angle_disagree: float = 8.0  # degrees between rectangle and outline angle that mark a composite blob
    outline_eps: float = 2.0  # px, contour simplification for the outline angle
    shade_search: tuple[float, float] = (0.85, 1.6)  # faces in from the outer edge searched for the step
    shade_min_count: float = 0.25  # profile slabs thinner than this x a face are ignored
    shade_window: int = 3  # px averaged on each side of a candidate step
    partial_contrast: float = 20.0  # grey-level difference that tells the true edge of a hidden cube
    partial_probe: tuple[float, float] = (3.0, 7.0)  # px beyond the blob end probed for board
    partial_probe_n: int = 3  # probe distances in that range
    partial_across_n: int = 9  # probe points across the blob
    partial_trim: float = 0.2  # fraction of the blob width left out at each end of the probe line
    side_face_frac: float = 0.5  # a candidate this much inside another cube's silhouette is its side face
    partial_penalty: float = 0.5  # score factor of a partly hidden cube
    border_penalty: float = 0.8  # score factor of a cube touching the frame border
    fill_holes: bool = True  # fill holes (glare, printed digits) of candidate blobs
    hole_fill_max: float = 1.5  # ... unless filling would grow the blob more than this (a ring)
    # Axis-aligned boxes of a trained detector (cube_from_box): the cube's mask is rebuilt inside.
    box_pad: float = 0.25  # crop padding around the box, in faces (or box sides without face_px)
    box_seed: float = 0.3  # the centre window that gives the seed colour leaves this much of each side
    box_w_l: float = 0.5  # weight of L* in the colour distance to the seed
    box_t: tuple[float, float] = (10.0, 30.0)  # Otsu threshold (Lab distance) clamped to this range
    box_size: tuple[float, float] = (0.3, 1.2)  # rebuilt mask area must be this many box areas
    box_blur: int = 3  # Gaussian blur size before rebuilding the mask (odd)


@dataclass
class Fit:
    """Top-face estimate for one cube."""

    center: tuple[float, float]
    quad: np.ndarray | None  # (4, 2) frame pixels
    angle: float | None
    flags: list[str] = field(default_factory=list)
    coverage: float = 1.0  # fraction of the quad covered by the blob mask
    outline: list[Point] | None = None


def _edges(proj: np.ndarray, face: float, level: float = 0.5) -> tuple[float, float]:
    """Interval [lo, hi] where the 1-px cross-section count reaches ``level`` x a face (or that
    fraction of the largest count for a thin blob), with linear interpolation between slabs."""
    base = math.floor(float(proj.min()))
    counts = np.bincount(np.floor(proj - base + 0.5).astype(np.int64)).astype(np.float64)
    t = level * min(face, float(counts.max()))
    above = np.nonzero(counts >= t)[0]
    i, j = int(above[0]), int(above[-1])
    lo = i - 0.5 if i == 0 else (i - 1) + (t - counts[i - 1]) / max(counts[i] - counts[i - 1], 1e-9)
    hi = j + 0.5 if j == len(counts) - 1 else j + (counts[j] - t) / max(counts[j] - counts[j + 1], 1e-9)
    return lo + base, hi + base


def _count(extent: float, band: float, face: float, single_max: float = 1.6) -> int:
    """Number of cubes side by side along one axis. Up to ``single_max`` faces (after removing
    the side band; default 1.6) is still one cube: a tower top is up to 1.53 faces wide (9 levels)."""
    e = extent - band
    return 1 if e < single_max * face else max(1, int(round(e / face)))


def _axis_frame(angle: float) -> tuple[np.ndarray, np.ndarray]:
    a = math.radians(angle)
    return np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])


def _kmeans_xy(pts: np.ndarray, k: int, iters: int = 15) -> np.ndarray:
    """Deterministic k-means on pixel positions (farthest-point start). Returns labels."""
    c = [pts[np.argmax(((pts - pts.mean(0)) ** 2).sum(1))]]
    for _ in range(1, k):
        d = np.min([((pts - ci) ** 2).sum(1) for ci in c], axis=0)
        c.append(pts[int(np.argmax(d))])
    cent = np.array(c, np.float64)
    lab = np.zeros(len(pts), np.int64)
    for _ in range(iters):
        lab = np.argmin(((pts[:, None, :] - cent[None]) ** 2).sum(2), axis=1)
        new = np.array([pts[lab == j].mean(0) if np.any(lab == j) else cent[j] for j in range(k)])
        if np.allclose(new, cent, atol=0.05):
            break
        cent = new
    return lab


def _split_mask(
    mask: np.ndarray, n: int, face: float, image: np.ndarray | None, peak: float = 0.7, iters: int = 15,
) -> list[np.ndarray]:
    """Split a crop mask into ``n`` parts: watershed on the distance transform when the cubes
    meet at a neck, else k-means on pixel positions (always gives n compact parts)."""
    m = (mask > 0).astype(np.uint8)
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    peaks = (dist > peak * face / 2).astype(np.uint8)
    k, markers = cv2.connectedComponents(peaks)
    if k - 1 == n:
        markers = markers + 1  # background 1, peaks 2..n+1
        markers[(m > 0) & (peaks == 0)] = 0  # unknown, to be flooded
        img = image if image is not None and image.shape[:2] == m.shape else cv2.cvtColor(m * 255, cv2.COLOR_GRAY2BGR)
        cv2.watershed(np.ascontiguousarray(img), markers)
        parts = [((markers == j + 2) & (m > 0)).astype(np.uint8) * 255 for j in range(n)]
        if all(p.any() for p in parts):
            return parts
    ys, xs = np.nonzero(m)
    lab = _kmeans_xy(np.column_stack([xs, ys]).astype(np.float64), n, int(iters))
    parts = []
    for j in range(n):
        p = np.zeros_like(m)
        p[ys[lab == j], xs[lab == j]] = 255
        parts.append(p)
    return parts


def _quad_from_intervals(u: np.ndarray, v: np.ndarray, iu: tuple[float, float], iv: tuple[float, float]) -> np.ndarray:
    return np.array([iu[0] * u + iv[0] * v, iu[1] * u + iv[0] * v, iu[1] * u + iv[1] * v, iu[0] * u + iv[1] * v])


def _coverage(quad: np.ndarray, mask: np.ndarray, x0: int, y0: int) -> float:
    """Fraction of the quad's pixels that are inside the blob mask."""
    q = np.round((quad - (x0, y0)) * 16).astype(np.int32).reshape(-1, 1, 2)
    h, w = mask.shape
    canvas = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(canvas, q, 255, cv2.LINE_8, shift=4)
    inside = int(np.count_nonzero(canvas))
    area = abs(float(cv2.contourArea(quad.astype(np.float32))))
    if area <= 0:
        return 0.0
    return float(np.count_nonzero(canvas & (mask > 0))) / max(area, inside, 1)


@dataclass
class _Shape:
    """Cross-section analysis of one mask in its own cube frame."""

    u: np.ndarray
    v: np.ndarray
    lo_u: float
    hi_u: float
    lo_v: float
    hi_v: float
    out_u_hi: bool  # True when the outer side (away from the principal point) is the high end
    out_v_hi: bool
    band_u: float
    band_v: float

    @property
    def ext_u(self) -> float:
        return self.hi_u - self.lo_u

    @property
    def ext_v(self) -> float:
        return self.hi_v - self.lo_v


def _shape(
    xs: np.ndarray, ys: np.ndarray, angle: float, face: float, pp: tuple[float, float],
    kappa: float = KAPPA, level: float = 0.5,
) -> _Shape:
    u, v = _axis_frame(angle)
    pu, pv = xs * u[0] + ys * u[1], xs * v[0] + ys * v[1]
    lo_u, hi_u = _edges(pu, face, level)
    lo_v, hi_v = _edges(pv, face, level)
    ppu, ppv = pp[0] * u[0] + pp[1] * u[1], pp[0] * v[0] + pp[1] * v[1]
    out_u_hi = ppu < (lo_u + hi_u) / 2
    out_v_hi = ppv < (lo_v + hi_v) / 2
    # A band appears only once the top-face centre is more than half a face from the principal
    # point; its width is kappa * (distance of the outer edge - one face).
    band_u = kappa * max(0.0, abs((hi_u if out_u_hi else lo_u) - ppu) - face)
    band_v = kappa * max(0.0, abs((hi_v if out_v_hi else lo_v) - ppv) - face)
    return _Shape(u, v, lo_u, hi_u, lo_v, hi_v, out_u_hi, out_v_hi, band_u, band_v)


def _face_interval(lo: float, hi: float, out_hi: bool, face: float, snap_tol: float) -> tuple[tuple[float, float], bool]:
    """Top-face interval along one axis and whether a side band was cut off."""
    if hi - lo > (1.0 + snap_tol) * face:
        return ((hi - face, hi) if out_hi else (lo, lo + face)), True
    return (lo, hi), False


def _shade_edge(
    proj: np.ndarray, gray: np.ndarray, lo: float, hi: float, out_hi: bool, face: float, shade_t: float,
    search: tuple[float, float] = (0.85, 1.6), min_count: float = 0.25, window: int = 3,
) -> float | None:
    """Inner edge of the top face along one axis, from the brightness step between the top face
    (lit from above) and the side band. Searched ``search`` faces in from the outer edge (default
    0.85-1.6: a tower top is up to 1.53 faces), comparing the mean of ``window`` px on each side.
    None when no step reaches ``shade_t`` (L* units)."""
    base = math.floor(float(proj.min()))
    idx = np.floor(proj - base + 0.5).astype(np.int64)
    cnt = np.bincount(idx).astype(np.float64)
    prof = np.bincount(idx, weights=gray) / np.maximum(cnt, 1.0)
    prof[cnt < min_count * face] = np.nan
    outer = (hi if out_hi else lo) - base
    best, best_k = shade_t * 255.0 / 100.0, None
    w = max(1, int(window))
    for d in range(int(search[0] * face), int(min(hi - lo - 2, search[1] * face)) + 1):
        k = outer - d if out_hi else outer + d  # candidate edge position (between slabs)
        kk = int(math.floor(k))
        top = prof[kk + 1 : kk + 1 + w] if out_hi else prof[kk + 1 - w : kk + 1]
        side = prof[kk + 1 - w : kk + 1] if out_hi else prof[kk + 1 : kk + 1 + w]
        if len(top) < w or len(side) < w or np.isnan(top).any() or np.isnan(side).any():
            continue
        step = float(top.mean() - side.mean())
        if step > best:
            best, best_k = step, kk + 0.5
    return None if best_k is None else best_k + base


def outline_angle(mask: np.ndarray, eps: float = 2.0) -> float | None:
    """Orientation in [0, 90) of the blob outline: length-weighted circular mean of 4 x the
    edge directions of its simplified contour. Exact for any union of flush cubes (all their
    edges are parallel or perpendicular), but biased by a few degrees on a single cube with a
    side band (its slanted vertical edges), so it is only used where the rectangle fails."""
    cnts, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    poly = cv2.approxPolyDP(max(cnts, key=cv2.contourArea), float(eps), True).reshape(-1, 2).astype(np.float64)
    if len(poly) < 3:
        return None
    d = np.roll(poly, -1, axis=0) - poly
    length, phi = np.hypot(d[:, 0], d[:, 1]), np.arctan2(d[:, 1], d[:, 0])
    return _fold90(math.degrees(math.atan2((length * np.sin(4 * phi)).sum(), (length * np.cos(4 * phi)).sum())) / 4.0)


ANGLE_DISAGREE = 8.0  # default of Geometry.angle_disagree


def fit_blob(
    blob: Blob,
    geo: Geometry,
    frame_shape: tuple[int, int],
    image: np.ndarray | None = None,
    *,
    allow_split: bool = True,
    angle_hint: float | None = None,
) -> tuple[list[Fit], str]:
    """Stages 1-4 for one blob -> (fits, status). status is "ok", "split", "block" or the reason
    the blob was dropped: "noise", "large", "sliver", "hollow".

    ``angle_hint`` is the orientation of the blob this part was cut from; it is used when the
    part's own rectangle disagrees with it (a cut edge tilts the rectangle)."""
    face = float(geo.face_px)
    area = blob.area
    if area < geo.min_area_ratio * face * face:
        return [], "noise"
    if area > geo.max_area_ratio * face * face:
        return [], "large"
    h, w = frame_shape[:2]
    pp = geo.pp if geo.pp is not None else (w / 2.0, h / 2.0)
    ys, xs = np.nonzero(blob.mask)
    xs = xs.astype(np.float64) + blob.x0
    ys = ys.astype(np.float64) + blob.y0
    # The rectangle depends only on the convex hull, so the outer contour is enough (and fast).
    cnts, _ = cv2.findContours((blob.mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    angle = coarse_angle(np.vstack([c.reshape(-1, 2) for c in cnts]) + (blob.x0, blob.y0))
    if angle_hint is not None and abs(angle_diff(angle, angle_hint)) > geo.angle_disagree:
        angle = angle_hint
    s = _shape(xs, ys, angle, face, pp, geo.kappa, geo.edge_level)
    n_u, n_v = _count(s.ext_u, s.band_u, face, geo.single_max), _count(s.ext_v, s.band_v, face, geo.single_max)
    composite = False
    if n_u * n_v > 1 and angle_hint is None:
        # Several cubes: the rectangle around e.g. two flush cubes shifted by half a face is not
        # aligned with them, but every outline edge is. (Single cubes keep the rectangle angle.)
        a4 = outline_angle(blob.mask, geo.outline_eps)
        if a4 is not None and abs(angle_diff(angle, a4)) > geo.angle_disagree:
            angle, composite = a4, True
            s = _shape(xs, ys, angle, face, pp, geo.kappa, geo.edge_level)
            n_u, n_v = _count(s.ext_u, s.band_u, face, geo.single_max), _count(s.ext_v, s.band_v, face, geo.single_max)
    if min(s.ext_u, s.ext_v) < geo.sliver_ratio * face:
        return [], "sliver"

    flags: list[str] = []
    mh, mw = blob.mask.shape
    if blob.x0 <= geo.border_px or blob.y0 <= geo.border_px or blob.x0 + mw >= w - geo.border_px or blob.y0 + mh >= h - geo.border_px:
        flags.append("touches_border")

    if n_u * n_v > 1 and allow_split and geo.split:
        fill = area / max(s.ext_u * s.ext_v, 1.0)
        if fill >= geo.block_fill:
            return _tile_block(blob, s, n_u, n_v, face, flags, geo.tile_coverage), "block"
        if min(n_u, n_v) == 1:
            fits = _split_row(blob, s, max(n_u, n_v), n_u > 1, xs, ys, face, geo, frame_shape, image, angle)
            if fits is not None:
                return fits, "split"
        # The extents already prove more than one cube (1.6+ faces without the band; even a
        # 9-level tower top is 1.53). The area says how many: one face each plus the side band,
        # but only as much band as the extents show (none when only the top face is paper,
        # since the plain side then falls outside the blob and the count would come out low).
        band_u = min(s.band_u, max(0.0, s.ext_u - n_u * face))
        band_v = min(s.band_v, max(0.0, s.ext_v - n_v * face))
        a_pred = face * face + face * (band_u + band_v)
        n = min(max(int(round(area / a_pred)), 2), n_u * n_v)
        fits = []
        crop_img = image[blob.y0 : blob.y0 + mh, blob.x0 : blob.x0 + mw] if image is not None else None
        for part in _split_mask(blob.mask, n, face, crop_img, geo.split_peak, geo.split_iters):
            sub = Blob(part, blob.x0, blob.y0, blob.color, blob.score)
            part_fits, _ = fit_blob(sub, geo, frame_shape, image, allow_split=False, angle_hint=angle if composite else None)
            for f in part_fits:
                f.flags.append("split")
            fits += part_fits
        return fits, "split"

    if min(s.ext_u, s.ext_v) < geo.partial_ratio * face:
        return [_fit_partial(blob, s, face, geo, image, flags + ["partial"])], "ok"
    side = face
    snapped: dict[str, float] = {}
    if geo.shade_snap and image is not None:
        iy, ix = ys.astype(np.int64), xs.astype(np.int64)
        px = image[iy, ix].astype(np.float32)
        gray = (0.114 * px[:, 0] + 0.587 * px[:, 1] + 0.299 * px[:, 2]).astype(np.float64)
        for name, lo, hi, out_hi, axis in (("u", s.lo_u, s.hi_u, s.out_u_hi, s.u), ("v", s.lo_v, s.hi_v, s.out_v_hi, s.v)):
            if hi - lo > (1.0 + geo.snap_tol) * face:
                edge = _shade_edge(
                    xs * axis[0] + ys * axis[1], gray, lo, hi, out_hi, face, geo.shade_t,
                    geo.shade_search, geo.shade_min_count, geo.shade_window,
                )
                size = None if edge is None else ((hi - edge) if out_hi else (edge - lo))
                # Only confirm or enlarge the face: a smaller "face" is a shadow edge (the arm's)
                # across the top face, not the top/side boundary.
                if size is not None and size >= (1.0 - geo.snap_tol) * face:
                    snapped[name] = size
        if snapped:
            side = float(np.mean(list(snapped.values())))  # a tower top is larger than face_px
            flags.append("shade_edge")
    est_u, est_v = s.ext_u - s.band_u, s.ext_v - s.band_v
    if not snapped and min(est_u, est_v) > (1.0 + geo.snap_tol) * face:
        # Larger than face_px on both axes even without the predicted band: a tower top, or
        # face_px is off. The predicted band never overshoots by more than ~1.6 px.
        side = min(est_u, est_v)
        flags.append("large_face")
    iu, cut_u = _face_interval(s.lo_u, s.hi_u, s.out_u_hi, side, geo.snap_tol)
    iv, cut_v = _face_interval(s.lo_v, s.hi_v, s.out_v_hi, side, geo.snap_tol)
    if cut_u or cut_v:
        flags.append("side_band")
    quad = _quad_from_intervals(s.u, s.v, iu, iv)
    coverage = _coverage(quad, blob.mask, blob.x0, blob.y0)
    if coverage < geo.min_coverage:
        return [], "hollow"  # a ring (e.g. mixed edge pixels around another cube), not a face
    c = quad.mean(axis=0)
    return [Fit((float(c[0]), float(c[1])), quad, angle, flags, coverage, blob.outline())], "ok"


def _outside_brightness(
    image: np.ndarray, along: np.ndarray, across: np.ndarray, end: float, step: float, lo2: float, hi2: float,
    geo: Geometry | None = None,
) -> float:
    """Mean grey level just outside one end of a blob (``partial_probe`` px beyond ``end`` along
    ``along``; default 3, 5 and 7 px), over the other axis without ``partial_trim`` of it at each
    end (default the middle 60 %). Points outside the frame count as black."""
    geo = geo or Geometry()
    h, w = image.shape[:2]
    trim = geo.partial_trim
    t = np.linspace(lo2 + trim * (hi2 - lo2), hi2 - trim * (hi2 - lo2), max(1, int(geo.partial_across_n)))
    vals = []
    for d in np.linspace(float(geo.partial_probe[0]), float(geo.partial_probe[1]), max(1, int(geo.partial_probe_n))):
        p = (end + step * d) * along[None, :] + t[:, None] * across[None, :]
        x, y = np.round(p[:, 0]).astype(np.int64), np.round(p[:, 1]).astype(np.int64)
        inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        px = np.zeros((len(t), 3), np.float64)
        px[inside] = image[y[inside], x[inside]]
        vals.append(0.114 * px[:, 0] + 0.587 * px[:, 1] + 0.299 * px[:, 2])
    return float(np.mean(vals))


def _fit_partial(blob: Blob, s: _Shape, face: float, geo: Geometry, image: np.ndarray | None, flags: list[str]) -> Fit:
    """A cube partly hidden (the arm, the frame border): along each cut axis keep the end that
    has bright board just outside it (a true edge) and hang the face square from it. When both
    ends look alike there is no telling which edge is real: blob centroid, no angle."""
    ys, xs = np.nonzero(blob.mask)
    centroid = Fit((float(xs.mean() + blob.x0), float(ys.mean() + blob.y0)), None, None, flags, 1.0, blob.outline())
    if image is None:
        return centroid
    a4 = outline_angle(blob.mask, geo.outline_eps)
    if a4 is None or abs(angle_diff(a4, math.degrees(math.atan2(s.u[1], s.u[0])) % 90.0)) > geo.angle_disagree:
        return centroid  # the cut edge tilts the rectangle: no trustworthy angle
    intervals = []
    axes = ((s.u, s.v, s.lo_u, s.hi_u, s.out_u_hi, s.lo_v, s.hi_v), (s.v, s.u, s.lo_v, s.hi_v, s.out_v_hi, s.lo_u, s.hi_u))
    for along, across, lo, hi, out_hi, lo2, hi2 in axes:
        if hi - lo >= geo.partial_ratio * face:
            intervals.append(_face_interval(lo, hi, out_hi, face, geo.snap_tol)[0])
            continue
        b_lo = _outside_brightness(image, along, across, lo, -1.0, lo2, hi2, geo)
        b_hi = _outside_brightness(image, along, across, hi, 1.0, lo2, hi2, geo)
        if abs(b_hi - b_lo) < geo.partial_contrast:
            return centroid
        intervals.append((hi - face, hi) if b_hi > b_lo else (lo, lo + face))
    quad = _quad_from_intervals(s.u, s.v, intervals[0], intervals[1])
    c = quad.mean(axis=0)
    return Fit((float(c[0]), float(c[1])), quad, coarse_angle(quad), flags, _coverage(quad, blob.mask, blob.x0, blob.y0), blob.outline())


def _split_row(
    blob: Blob, s: _Shape, n: int, along_u: bool, xs: np.ndarray, ys: np.ndarray, face: float,
    geo: Geometry, frame_shape: tuple[int, int], image: np.ndarray | None, angle: float,
) -> list[Fit] | None:
    """One row of n flush cubes, possibly shifted sideways against each other: cut it into
    face-wide slabs from the outer end (where flush cubes meet) and fit each slab on its own,
    with the row's angle. None if a slab does not hold a cube (then the caller splits otherwise)."""
    axis = s.u if along_u else s.v
    lo, hi, out_hi = (s.lo_u, s.hi_u, s.out_u_hi) if along_u else (s.lo_v, s.hi_v, s.out_v_hi)
    proj = xs * axis[0] + ys * axis[1]
    idx = np.floor(((hi - proj) if out_hi else (proj - lo)) / face).clip(0, n - 1).astype(np.int64)
    my, mx = (ys - blob.y0).astype(np.int64), (xs - blob.x0).astype(np.int64)
    fits: list[Fit] = []
    for k in range(n):
        part = np.zeros_like(blob.mask)
        part[my[idx == k], mx[idx == k]] = 255
        part_fits, _ = fit_blob(Blob(part, blob.x0, blob.y0, blob.color, blob.score), geo, frame_shape, image, allow_split=False, angle_hint=angle)
        if not part_fits:
            return None
        for f in part_fits:
            f.flags.append("split")
        fits += part_fits
    return fits


def _tile_block(blob: Blob, s: _Shape, n_u: int, n_v: int, face: float, flags: list[str], min_cov: float = 0.6) -> list[Fit]:
    """Tile a flush block of n_u x n_v cubes with face squares, starting at its outer corner
    (top faces of flush cubes lie in one plane, so they are flush in the image too)."""
    fits = []
    outline = blob.outline()
    for i in range(n_u):
        iu = (s.hi_u - (i + 1) * face, s.hi_u - i * face) if s.out_u_hi else (s.lo_u + i * face, s.lo_u + (i + 1) * face)
        for j in range(n_v):
            iv = (s.hi_v - (j + 1) * face, s.hi_v - j * face) if s.out_v_hi else (s.lo_v + j * face, s.lo_v + (j + 1) * face)
            quad = _quad_from_intervals(s.u, s.v, iu, iv)
            cov = _coverage(quad, blob.mask, blob.x0, blob.y0)
            if cov < min_cov:
                continue  # an empty corner of an L-shaped group
            c = quad.mean(axis=0)
            angle = coarse_angle(quad)
            fits.append(Fit((float(c[0]), float(c[1])), quad, angle, flags + ["split"], cov, outline))
    return fits


# --- colour and output (stages 6-7) ----------------------------------------------------------
def _sample_mask(fit: Fit, blob: Blob | None, shrink: float, shape_hw: tuple[int, int]) -> tuple[np.ndarray, int, int] | None:
    """Mask (crop, offset) of the pixels whose colour decides the cube."""
    h, w = shape_hw
    if fit.quad is not None:
        c = fit.quad.mean(axis=0)
        q = c + (fit.quad - c) * (1.0 - 2.0 * shrink)
        x0, y0 = int(max(0, math.floor(q[:, 0].min()))), int(max(0, math.floor(q[:, 1].min())))
        x1, y1 = int(min(w, math.ceil(q[:, 0].max()) + 1)), int(min(h, math.ceil(q[:, 1].max()) + 1))
        if x1 - x0 < 1 or y1 - y0 < 1:
            return None
        m = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillConvexPoly(m, np.round((q - (x0, y0)) * 16).astype(np.int32).reshape(-1, 1, 2), 255, cv2.LINE_8, shift=4)
        return (m, x0, y0) if m.any() else None
    if blob is not None:
        k = max(1, int(round(math.sqrt(blob.area) * shrink)))
        m = cv2.erode(blob.mask, np.ones((3, 3), np.uint8), iterations=k)
        return (m if m.any() else blob.mask), blob.x0, blob.y0
    return None


def _face_lab(frame_bgr: np.ndarray, colour: ColourModel, m: np.ndarray, x0: int, y0: int) -> np.ndarray | None:
    """Lab (reference light) of the median raw colour under the sample mask crop."""
    h, w = m.shape
    bgr = median_bgr(frame_bgr[y0 : y0 + h, x0 : x0 + w], m)
    return None if bgr is None else colour.lab_of_bgr(bgr)


@dataclass
class Report:
    """What the shared post-processing did with the candidates of one frame."""

    dropped: dict[str, int] = field(default_factory=dict)  # reason -> count
    splits: int = 0  # blobs that held several cubes
    records: list[tuple[list[Point] | None, str]] = field(default_factory=list)  # (outline, status) for debug

    def add(self, reason: str, outline: list[Point] | None = None) -> None:
        self.dropped[reason] = self.dropped.get(reason, 0) + 1
        self.records.append((outline, reason))

    def notes(self) -> list[str]:
        """Short Korean warnings for the UI."""
        msgs = {
            "large": "큰 색 영역 {n}개를 바구니 바닥·팔 등으로 보고 제외",
            "sliver": "가는 띠 {n}개를 큐브 옆면·그림자로 보고 제외",
            "no_colour": "판·그림자 색인 후보 {n}개 제외",
            "side_face": "다른 큐브의 옆면으로 보이는 후보 {n}개 제외",
        }
        out = [msgs[k].format(n=v) for k, v in self.dropped.items() if k in msgs and v]
        if self.splits:
            out.append(f"붙어 있는 큐브 묶음 {self.splits}개를 나눔")
        return out


def cubes_from_blobs(
    frame_bgr: np.ndarray,
    blobs: list[Blob],
    *,
    colour: ColourModel,
    geo: Geometry,
    report: Report | None = None,
) -> tuple[list[Cube], Report]:
    """Shared post-processing for a list of candidate blobs (stages 1-7). ``frame_bgr`` is the
    raw frame: colours are corrected with ``colour.gains`` (see colour.py)."""
    report = report or Report()
    found: list[tuple[Fit, Blob]] = []
    for blob in blobs:
        fits, status = fit_blob(blob, geo, frame_bgr.shape[:2], frame_bgr)
        if not fits:
            report.add(status, blob.outline() if status != "noise" else None)
            continue
        if status in ("split", "block"):
            report.splits += 1
        report.records.append((blob.outline(), status))
        found += [(f, blob) for f in fits]
    if geo.drop_side_faces and len(found) > 1:
        h, w = frame_bgr.shape[:2]
        keep = _not_side_faces(
            found, geo.pp if geo.pp is not None else (w / 2.0, h / 2.0), (h, w), geo.kappa, geo.side_face_frac,
        )
        for (fit, _), k in zip(found, keep):
            if not k:
                report.add("side_face", fit.outline)
        found = [fb for fb, k in zip(found, keep) if k]
    cubes: list[Cube] = []
    for fit, blob in found:
        cube = _cube_from_fit(
            frame_bgr, fit, blob, colour, geo.shrink,
            partial_penalty=geo.partial_penalty, border_penalty=geo.border_penalty,
        )
        if cube is None:
            report.add("no_colour", fit.outline)
        else:
            cubes.append(cube)
    return cubes, report


def silhouette(quad: np.ndarray, pp: tuple[float, float], kappa: float = KAPPA) -> np.ndarray:
    """Image outline of a whole cube from its top face: the convex hull of the top face and
    the base (top face pulled towards the principal point by kappa)."""
    q = np.asarray(quad, np.float64).reshape(4, 2)
    base = np.asarray(pp) + (q - pp) * (1.0 - kappa)
    return cv2.convexHull(np.vstack([q, base]).astype(np.float32)).reshape(-1, 2)


def _not_side_faces(
    found: list[tuple[Fit, Blob]], pp: tuple[float, float], shape_hw: tuple[int, int],
    kappa: float = KAPPA, side_frac: float = 0.5,
) -> list[bool]:
    """A candidate whose pixels lie mostly inside the silhouette of another, complete cube is
    that cube's side face (e.g. the bare wood band of a cube with paper on the top only).

    Candidates are accepted largest first, and only an accepted cube can explain away a
    later one, so two candidates can never remove each other."""
    def own_mask(fit: Fit, blob: Blob) -> np.ndarray:
        own = blob.mask > 0
        if fit.quad is not None:
            qm = np.zeros(blob.mask.shape, np.uint8)
            cv2.fillConvexPoly(qm, np.round((fit.quad - (blob.x0, blob.y0)) * 16).astype(np.int32).reshape(-1, 1, 2), 1, cv2.LINE_8, shift=4)
            own &= qm > 0
        return own

    owns = [own_mask(f, b) for f, b in found]
    order = sorted(range(len(found)), key=lambda i: -int(np.count_nonzero(owns[i])))
    keep = [False] * len(found)
    accepted: list[int] = []
    for i in order:
        fit, blob = found[i]
        others = [j for j in accepted if found[j][1] is not blob]
        frac = 0.0
        if others:
            inside = np.zeros(blob.mask.shape, np.uint8)
            for j in others:
                poly = silhouette(found[j][0].quad, pp, kappa)  # type: ignore[arg-type]
                cv2.fillConvexPoly(inside, np.round((poly - (blob.x0, blob.y0)) * 16).astype(np.int32).reshape(-1, 1, 2), 1, cv2.LINE_8, shift=4)
            frac = np.count_nonzero(owns[i] & (inside > 0)) / max(int(np.count_nonzero(owns[i])), 1)
        keep[i] = frac < side_frac
        if keep[i] and fit.quad is not None and "partial" not in fit.flags:
            accepted.append(i)
    return keep


def _cube_from_fit(
    frame_bgr: np.ndarray,
    fit: Fit,
    blob: Blob | None,
    colour: ColourModel,
    shrink: float,
    *,
    color: Color | None = None,
    label: str | None = None,
    polygon: list[Point] | None = None,
    box: tuple[float, float, float, float] | None = None,
    partial_penalty: float = 0.5,
    border_penalty: float = 0.8,
) -> Cube | None:
    """Stage 6-7: decide the colour and build the Cube.

    ``color`` (given by the caller, e.g. a trained model's class) is used as it is: the pixels
    can only add the flag class_mismatch. A built-in pipeline's own class (``blob.color``) is
    kept the same way, but its candidate is dropped (None) when the face has board / shadow
    colour. With no class at all the pixels decide, and board colour also gives None."""
    flags = list(fit.flags)
    sm = _sample_mask(fit, blob, shrink, frame_bgr.shape[:2])
    lab = _face_lab(frame_bgr, colour, *sm) if sm is not None else None
    proposed = color if color is not None else (blob.color if blob is not None else None)
    if lab is None:
        name, cscore = (proposed or "unknown"), 0.0
    else:
        name, cscore, _ = colour.classify(lab)
        if name == "none" and color is None:
            return None  # board, shadow or a grey part of the arm
        if proposed is not None:
            if name != proposed:
                flags.append("class_mismatch")
            cscore = colour.probability(lab, proposed) if (name != "none" and proposed in colour.names) else 0.0
            name = proposed
    score = (blob.score if blob is not None else 1.0) * min(1.0, fit.coverage)
    if "partial" in flags:
        score *= partial_penalty
    if "touches_border" in flags:
        score *= border_penalty
    if fit.quad is not None:
        q = order_quad(fit.quad)
        area = abs(float(cv2.contourArea(np.asarray(q, np.float32))))
        angle = quad_angle(q)
    else:
        q, angle = None, None
        area = float(blob.area) if blob is not None else 0.0
    return Cube(
        color=name if name in ("red", "green", "blue", "wood") else "unknown",  # type: ignore[arg-type]
        score=float(np.clip(score, 0.0, 1.0)),
        center=fit.center,
        quad=q,
        angle=angle,
        area=area,
        color_score=float(np.clip(cscore, 0.0, 1.0)),
        polygon=polygon if polygon is not None else fit.outline,
        box=box,
        label=label if label is not None else (blob.label if blob is not None else None),
        flags=flags,
    )


# --- contract entry points for models outside cubevision (the lab's Ultralytics adapter) -----
def _poly_area(pts: np.ndarray) -> float:
    return float(abs(cv2.contourArea(pts.astype(np.float32).reshape(-1, 1, 2))))


def cube_from_quad(
    frame_bgr: np.ndarray,
    quad: Iterable[Iterable[float]],
    *,
    score: float,
    color: Color | None = None,
    label: str | None = None,
    prototypes: dict[str, str] | None = None,
    polygon: list[Point] | None = None,
    box: tuple[float, float, float, float] | None = None,
    flags: Iterable[str] = (),
    geo: Geometry | None = None,
    colour_opts: dict[str, Any] | None = None,
) -> Cube:
    """A quad that already is the top face (e.g. an oriented box). ``color`` given = used as is;
    None = classified from the pixels ("unknown" + flag no_colour when they look like board)."""
    geo = geo or Geometry()
    q = np.asarray(order_quad(quad), np.float64)
    c = q.mean(axis=0)
    fit = Fit((float(c[0]), float(c[1])), q, quad_angle(q), list(flags))
    cube = _cube_from_fit(
        frame_bgr, fit, None, colour_model(frame_bgr, prototypes, **(colour_opts or {})), geo.shrink,
        color=color, label=label, polygon=polygon, box=box,
    )
    if cube is None:
        cube = _uncoloured(fit, None, label, polygon, box)
    cube.score = float(score)
    return cube


def _uncoloured(fit: Fit, blob: Blob | None, label: str | None, polygon: list[Point] | None, box: tuple[float, float, float, float] | None) -> Cube:
    """A detection from an outside model whose pixels look like board: keep it, as unknown.
    The contract functions never drop a model's detection for its colour."""
    q = order_quad(fit.quad) if fit.quad is not None else None
    area = _poly_area(np.asarray(q)) if q is not None else float(blob.area if blob is not None else 0.0)
    angle = quad_angle(q) if q is not None else None
    return Cube("unknown", 1.0, fit.center, q, angle, area, 0.0, polygon if polygon is not None else fit.outline, box, label, list(fit.flags) + ["no_colour"])


def _cube_from_single_blob(
    frame_bgr: np.ndarray,
    blob: Blob,
    *,
    score: float,
    color: Color | None,
    label: str | None,
    prototypes: dict[str, str] | None,
    face_px: float,
    polygon: list[Point] | None,
    box: tuple[float, float, float, float] | None,
    extra_flags: Iterable[str] = (),
    geo: Geometry | None = None,
    colour_opts: dict[str, Any] | None = None,
) -> Cube:
    """Top-face fit of one outside model's mask. Never drops the detection: when the fit fails
    (a ring, a scrap) the minimum-area rectangle of the mask is used instead. So the gates that
    drop candidates (size, sliver) are off and nothing is split, whatever ``geo`` says."""
    geo = replace(
        geo or Geometry(), face_px=float(face_px), min_area_ratio=0.0, max_area_ratio=1e9, sliver_ratio=0.0, split=False,
    )
    fits, status = fit_blob(blob, geo, frame_bgr.shape[:2], frame_bgr, allow_split=False)
    if fits:
        fit = fits[0]
    else:
        ys, xs = np.nonzero(blob.mask)
        pts = np.column_stack([xs + blob.x0, ys + blob.y0]).astype(np.float32).reshape(-1, 1, 2)
        q = cv2.boxPoints(cv2.minAreaRect(pts)).astype(np.float64)
        c = q.mean(axis=0)
        fit = Fit((float(c[0]), float(c[1])), q, coarse_angle(q), [f"fit_{status}"], 1.0, blob.outline())
    fit.flags += list(extra_flags)
    cube = _cube_from_fit(
        frame_bgr, fit, blob, colour_model(frame_bgr, prototypes, **(colour_opts or {})), geo.shrink,
        color=color, label=label, polygon=polygon, box=box,
    )
    if cube is None:
        cube = _uncoloured(fit, blob, label, polygon, box)
    cube.score = float(score) * (geo.partial_penalty if "partial" in fit.flags else 1.0)
    return cube


def cube_from_polygon(
    frame_bgr: np.ndarray,
    polygon: Iterable[Iterable[float]],
    *,
    score: float,
    color: Color | None = None,
    label: str | None = None,
    prototypes: dict[str, str] | None = None,
    top_face: bool = True,
    face_px: float | None = None,
    box: tuple[float, float, float, float] | None = None,
    geo: Geometry | None = None,
    colour_opts: dict[str, Any] | None = None,
) -> Cube | None:
    """A segmentation outline. With ``face_px`` and ``top_face`` the side band is cut off
    (anchored top face); otherwise the minimum-area rectangle of the outline is the quad."""
    pts = np.asarray(list(polygon), np.float32).reshape(-1, 2)
    if len(pts) < 3 or _poly_area(pts) < 4:
        return None
    raw = [(float(x), float(y)) for x, y in pts]
    if top_face and face_px:
        blob = blob_from_polygon(pts, frame_bgr.shape[:2], None, score)
        if blob is not None:
            return _cube_from_single_blob(
                frame_bgr, blob, score=score, color=color, label=label, prototypes=prototypes,
                face_px=face_px, polygon=raw, box=box, geo=geo, colour_opts=colour_opts,
            )
    quad = cv2.boxPoints(cv2.minAreaRect(pts.reshape(-1, 1, 2)))
    return cube_from_quad(
        frame_bgr, quad, score=score, color=color, label=label, prototypes=prototypes, polygon=raw, box=box,
        geo=geo, colour_opts=colour_opts,
    )


def cube_from_mask(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    *,
    score: float,
    color: Color | None = None,
    label: str | None = None,
    prototypes: dict[str, str] | None = None,
    top_face: bool = True,
    face_px: float | None = None,
    geo: Geometry | None = None,
    colour_opts: dict[str, Any] | None = None,
) -> Cube | None:
    """A full-frame mask of ONE cube (its largest component is used)."""
    m = (mask > 0).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    return cube_from_polygon(
        frame_bgr, cnt.reshape(-1, 2), score=score, color=color, label=label,
        prototypes=prototypes, top_face=top_face, face_px=face_px, geo=geo, colour_opts=colour_opts,
    )


def cube_from_box(
    frame_bgr: np.ndarray,
    box: Iterable[float],
    *,
    score: float,
    color: Color | None = None,
    label: str | None = None,
    prototypes: dict[str, str] | None = None,
    face_px: float | None = None,
    geo: Geometry | None = None,
    colour_opts: dict[str, Any] | None = None,
) -> Cube | None:
    """An axis-aligned box carries no angle (its width L(|cos t| + |sin t|) is the same for t and
    90 - t). So rebuild the cube's mask inside the box: pixels close in Lab to the box centre's
    colour (Otsu threshold, clamped to ``geo.box_t``, default 10-30), keep the component at the
    centre, then fit it like a mask. Falls back to the box itself (angle None, flag box_only)
    when that fails."""
    x1, y1, x2, y2 = (float(v) for v in box)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    bx = (x1, y1, x2, y2)
    blob = _mask_in_box(frame_bgr, bx, face_px, geo or Geometry())
    if blob is not None:
        if face_px:
            return _cube_from_single_blob(
                frame_bgr, blob, score=score, color=color, label=label, prototypes=prototypes,
                face_px=face_px, polygon=None, box=bx, extra_flags=("from_box",), geo=geo, colour_opts=colour_opts,
            )
        ys, xs = np.nonzero(blob.mask)
        pts = np.column_stack([xs + blob.x0, ys + blob.y0]).astype(np.float32)
        quad = cv2.boxPoints(cv2.minAreaRect(pts.reshape(-1, 1, 2)))
        return cube_from_quad(
            frame_bgr, quad, score=score, color=color, label=label, prototypes=prototypes, box=bx, flags=("from_box",),
            geo=geo, colour_opts=colour_opts,
        )
    quad = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    cube = cube_from_quad(
        frame_bgr, quad, score=score, color=color, label=label, prototypes=prototypes, box=bx,
        geo=geo, colour_opts=colour_opts,
    )
    cube.angle = None  # an axis-aligned box carries no orientation
    cube.flags.append("box_only")
    return cube


def _mask_in_box(frame_bgr: np.ndarray, box: tuple[float, float, float, float], face_px: float | None, geo: Geometry) -> Blob | None:
    x1, y1, x2, y2 = box
    h, w = frame_bgr.shape[:2]
    pad = geo.box_pad * (face_px if face_px else min(x2 - x1, y2 - y1))
    cx0, cy0 = int(max(0, math.floor(x1 - pad))), int(max(0, math.floor(y1 - pad)))
    cx1, cy1 = int(min(w, math.ceil(x2 + pad))), int(min(h, math.ceil(y2 + pad)))
    k = int(geo.box_blur) | 1
    crop = cv2.GaussianBlur(frame_bgr[cy0:cy1, cx0:cx1], (k, k), 0) if k > 1 else frame_bgr[cy0:cy1, cx0:cx1]
    if crop.size == 0:
        return None
    lab = lab8_to_real(cv2.cvtColor(crop, cv2.COLOR_BGR2Lab))
    bw, bh = x2 - x1, y2 - y1
    sx0, sx1 = int(x1 + geo.box_seed * bw) - cx0, int(math.ceil(x2 - geo.box_seed * bw)) - cx0
    sy0, sy1 = int(y1 + geo.box_seed * bh) - cy0, int(math.ceil(y2 - geo.box_seed * bh)) - cy0
    seed_px = lab[max(0, sy0) : max(sy0 + 1, sy1), max(0, sx0) : max(sx0 + 1, sx1)].reshape(-1, 3)
    if len(seed_px) == 0:
        return None
    seed = np.median(seed_px, axis=0)
    d = lab - seed
    d[..., 0] *= geo.box_w_l
    de = np.sqrt((d**2).sum(axis=2))
    de8 = np.clip(de * 2.0, 0, 255).astype(np.uint8)  # 0.5 Lab units per level
    t, _ = cv2.threshold(de8, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    t = float(np.clip(t / 2.0, float(geo.box_t[0]), float(geo.box_t[1])))
    m = (de < t).astype(np.uint8)
    n, labels = cv2.connectedComponents(m, connectivity=8)
    cx, cy = int(round((x1 + x2) / 2)) - cx0, int(round((y1 + y2) / 2)) - cy0
    if n < 2 or not (0 <= cy < labels.shape[0] and 0 <= cx < labels.shape[1]) or labels[cy, cx] == 0:
        return None
    part = (labels == labels[cy, cx]).astype(np.uint8) * 255
    n_part = np.count_nonzero(part)
    if n_part < geo.box_size[0] * bw * bh or n_part > geo.box_size[1] * bw * bh:
        return None  # nothing solid at the centre, or it leaked into the board around the box
    cnts, _ = cv2.findContours(part, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(part, cnts, -1, 255, cv2.FILLED)
    return Blob(part, cx0, cy0)

"""Colour classification of cube pixels (shared by every model).

CONTRACT (other modules and the lab rely on these signatures; internals may improve):
- DEFAULT_PROTOTYPES: {"red" | "green" | "blue" | "wood": "#rrggbb"}
- classify_pixels(bgr_pixels, prototypes=None, *, max_dist=...) -> (Color, score)
- classify_region(frame_bgr, mask, prototypes=None, *, max_dist=...) -> (Color, score)

Method (board-referenced nearest prototype):
1. The white board is always in view, so it is a white reference. Each frame we measure the
   board colour (median of the most neutral bright pixels) and compare it with the board colour
   the prototypes were picked under (``DEFAULT_BOARD``, or a "board" entry in the prototypes
   dict). Per-channel gains reference / measured undo the change of light, white balance and
   exposure since then (von Kries). A per-channel scale of sRGB values is exactly a scale of
   linear light for a power-law gamma, and a per-channel median commutes with it, so a cube's
   colour is "median BGR x gains", and per-pixel pipelines scale the whole frame once
   (``ColourModel.balance``). After that every colour is compared in the reference light: a grey
   object (pillar, arm, table) is grey there too, whatever the cast.
2. A region's colour is its median, converted to CIELAB. Its distance to each prototype weighs
   a* and b* fully and lightness by ``W_LIGHTNESS``.
3. Board, shadow and dark grey objects are rejected before the prototypes are asked: anything
   whose chroma relative to the reference board is below ``chroma_min`` is "no colour" (a shadow
   darkens the board but hardly changes its chroma; a test that is only "darker than the board"
   would turn every shadow into a wood cube), and so is anything dark (under half the board's
   lightness) with only weak chroma (the arm, a table, the robot base).
Classes beyond ``max_dist`` from every prototype are "unknown".

Units: Lab in real units (L 0-100, a and b about -128..127), as everywhere in the lab.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from cubevision.types import Color

# sRGB hex. Rough guesses for printed paper under indoor light; re-pick from a real image in the lab.
DEFAULT_PROTOTYPES: dict[str, str] = {
    "red": "#c8323c",
    "green": "#3c9650",
    "blue": "#2850b4",
    "wood": "#c8aa82",
}
# Board colour under the light in which the prototypes were picked (white PLA under indoor light).
DEFAULT_BOARD = "#e4e4e0"
DEFAULT_MAX_DIST = 45.0  # Lab distance beyond which the colour is "unknown"
DEFAULT_CHROMA_MIN = 8.0  # board-relative chroma below this is board / shadow, never a cube
W_LIGHTNESS = 0.3  # weight of L* next to a* and b* in the prototype distance
DARK_RATIO = 0.5  # lightness below this fraction of the board's counts as dark
SIGMA = 10.0  # Lab distance scale of the class probabilities
WEAK_FACTOR = 2.0  # "weak" chroma = below this many times chroma_min (dark AND weak = grey object)
GAIN_RANGE = (0.4, 2.5)  # light-correction gains are clipped to this range
BOARD_SAMPLES = 20000  # board_colour works on a subsample of about this many pixels
BOARD_V_PCT = 50.0  # board candidates: V at or above this percentile
BOARD_S_PCT = 40.0  # ... and S at most this percentile of those bright pixels
BOARD_S_FLOOR = 40.0  # ... but never a stricter S limit than this
CLASSES: tuple[str, ...] = ("red", "green", "blue", "wood")


def hex_to_bgr(hex_str: str) -> tuple[int, int, int]:
    h = hex_str.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return b, g, r


def bgr_to_hex(bgr: np.ndarray | tuple[float, float, float]) -> str:
    b, g, r = (int(round(float(np.clip(v, 0, 255)))) for v in bgr)
    return f"#{r:02x}{g:02x}{b:02x}"


def bgr_to_lab(bgr: np.ndarray) -> np.ndarray:
    """(N,3) BGR 0-255 (uint8 or float) -> (N,3) float32 Lab in real units."""
    px = np.asarray(bgr, np.float32).reshape(-1, 1, 3) / 255.0
    return cv2.cvtColor(np.clip(px, 0, 1), cv2.COLOR_BGR2Lab).reshape(-1, 3)


def lab8_to_real(lab8: np.ndarray) -> np.ndarray:
    """OpenCV 8-bit Lab (L*255/100, a+128, b+128) -> float32 real units."""
    lab = np.asarray(lab8, np.float32).copy()
    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1:] -= 128.0
    return lab


def _class_protos(prototypes: dict[str, str] | None) -> dict[str, str]:
    protos = dict(DEFAULT_PROTOTYPES)
    protos.update({k: v for k, v in (prototypes or {}).items() if k in CLASSES})
    return protos


def board_colour(
    frame_bgr: np.ndarray,
    region: np.ndarray | None = None,
    *,
    samples: float = BOARD_SAMPLES,
    v_pct: float = BOARD_V_PCT,
    s_pct: float = BOARD_S_PCT,
    s_floor: float = BOARD_S_FLOOR,
) -> np.ndarray | None:
    """Median BGR of the white board: the most neutral of the bright pixels.

    Bright = V at or above the ``v_pct`` percentile (default: the median). Neutral = S at most the
    ``s_pct`` percentile of those bright pixels, and at least up to ``s_floor``: a fixed "S < 40"
    would find no board at all under a strong colour cast, exactly when the measurement matters most.
    Works on a subsample (about ``samples`` pixels), so it costs well under a millisecond.
    ``region`` (HxW mask) limits the search, e.g. to the user's ROI. None if too few pixels.
    """
    h, w = frame_bgr.shape[:2]
    step = max(1, int(np.sqrt(h * w / max(float(samples), 1.0))))
    small = frame_bgr[::step, ::step]
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    s, v = hsv[..., 1], hsv[..., 2]
    sel = np.ones(s.shape, bool) if region is None else region[::step, ::step] > 0
    if np.count_nonzero(sel) < 50:
        return None
    sel &= v >= np.percentile(v[sel], float(v_pct))
    sel &= s <= max(float(s_floor), float(np.percentile(s[sel], float(s_pct))))
    if np.count_nonzero(sel) < 25:
        return None
    return np.median(small[sel].reshape(-1, 3), axis=0).astype(np.float64)


@dataclass
class ColourModel:
    """Prototypes plus this frame's light correction. Build once per frame with ``colour_model``.

    All Lab values it compares are in the reference light: convert raw pixels with
    ``lab_of_bgr`` (one colour) or ``balance`` (a whole frame) first."""

    names: tuple[str, ...]  # class names, same order as ``lab``
    lab: np.ndarray  # (K, 3) prototype Lab (reference light)
    board_lab: np.ndarray  # (3,) reference board Lab
    gains: np.ndarray  # (3,) BGR multipliers: this frame -> reference light
    max_dist: float = DEFAULT_MAX_DIST
    chroma_min: float = DEFAULT_CHROMA_MIN
    board_now: np.ndarray | None = None  # (3,) board BGR measured in this frame, None if not found
    clipped: float = 0.0  # fraction of the frame (or region) with a channel at 250 or more
    w_lightness: float = W_LIGHTNESS  # weight of L* next to a* and b* in the prototype distance
    dark_ratio: float = DARK_RATIO  # lightness below this fraction of the board's counts as dark
    sigma: float = SIGMA  # Lab distance scale of the class probabilities
    weak_factor: float = WEAK_FACTOR  # weak chroma = below weak_factor x chroma_min

    def balance(self, frame_bgr: np.ndarray) -> np.ndarray:
        """The frame as it would look in the reference light (uint8, clipped)."""
        if np.allclose(self.gains, 1.0):
            return frame_bgr
        return cv2.transform(frame_bgr, np.diag(self.gains).astype(np.float32))

    def lab_of_bgr(self, bgr: np.ndarray) -> np.ndarray:
        """Lab (reference light) of one raw BGR colour, e.g. the median of a face."""
        return bgr_to_lab(np.clip(np.asarray(bgr, np.float64).reshape(1, 3) * self.gains, 0, 255))[0]

    def chroma(self, lab: np.ndarray) -> np.ndarray:
        """Chroma relative to the board (distance from the board in the a*b* plane)."""
        lab = np.asarray(lab, np.float32)
        return np.hypot(lab[..., 1] - self.board_lab[1], lab[..., 2] - self.board_lab[2])

    def is_neutral(self, lab: np.ndarray) -> bool:
        """Board, shadow, or a dark grey object (arm, table, robot base): not a cube colour.

        Weak chroma alone (below ``chroma_min``) is board or shadow. Dark AND only weakly
        chromatic (below weak_factor x chroma_min, lightness under dark_ratio x the board's) is a dark grey
        object: dark-warm greys are otherwise nearest to wood. Printed blue is dark too but
        strongly chromatic, so it is not affected."""
        lab = np.asarray(lab, np.float32).reshape(3)
        c = float(self.chroma(lab))
        if c < self.chroma_min:
            return True
        return c < self.weak_factor * self.chroma_min and float(lab[0]) < self.dark_ratio * float(self.board_lab[0])

    def distances(self, lab: np.ndarray, w_l: float | None = None) -> np.ndarray:
        d = np.asarray(lab, np.float32).reshape(-1, 1, 3) - self.lab[None, :, :]
        d[..., 0] *= self.w_lightness if w_l is None else w_l
        return np.sqrt((d**2).sum(axis=2))

    def classify(self, lab: np.ndarray) -> tuple[str, float, float]:
        """Classify one Lab colour -> (name, score, distance).

        name is a class, "unknown" (far from every prototype) or "none" (board / shadow / grey:
        chroma below ``chroma_min``). score = class probability x closeness, in [0, 1].
        """
        lab = np.asarray(lab, np.float32).reshape(3)
        if self.is_neutral(lab):
            return "none", 0.0, float("inf")
        d = self.distances(lab)[0]
        best = int(np.argmin(d))
        if d[best] > self.max_dist:
            return "unknown", 0.0, float(d[best])
        p = np.exp(-(d**2) / (2 * self.sigma**2))
        p_best = float(p[best] / p.sum()) if p.sum() > 1e-12 else 1.0
        closeness = max(0.0, 1.0 - (float(d[best]) / self.max_dist) ** 2)
        return self.names[best], p_best * closeness, float(d[best])

    def probability(self, lab: np.ndarray, name: str) -> float:
        """Probability of class ``name`` for one Lab colour (for a class proposed by a model)."""
        if name not in self.names:
            return 0.0
        d = self.distances(np.asarray(lab, np.float32).reshape(3))[0]
        p = np.exp(-(d**2) / (2 * self.sigma**2))
        if p.sum() <= 1e-12:  # far from everything: fall back to relative closeness
            p = 1.0 / np.maximum(d, 1e-6)
        return float(p[self.names.index(name)] / p.sum())

    def label_lut(self, max_dist: float | None = None) -> np.ndarray:
        """256x256 table over 8-bit (a, b): index+1 of the nearest prototype in the a*b* plane,
        0 where the nearest is farther than ``max_dist`` or the chroma is below ``chroma_min``.
        Index it as ``lut[lab8[..., 1], lab8[..., 2]]``."""
        md = self.max_dist if max_dist is None else max_dist
        a = np.arange(256, dtype=np.float32) - 128.0
        aa, bb = np.meshgrid(a, a, indexing="ij")
        d = np.stack([np.hypot(aa - p[1], bb - p[2]) for p in self.lab], axis=0)
        lut = (d.argmin(axis=0) + 1).astype(np.uint8)
        lut[d.min(axis=0) > md] = 0
        lut[np.hypot(aa - self.board_lab[1], bb - self.board_lab[2]) < self.chroma_min] = 0
        return lut


def colour_model(
    frame_bgr: np.ndarray | None,
    prototypes: dict[str, str] | None = None,
    *,
    region: np.ndarray | None = None,
    max_dist: float = DEFAULT_MAX_DIST,
    chroma_min: float = DEFAULT_CHROMA_MIN,
    adapt: bool = True,
    w_lightness: float = W_LIGHTNESS,
    dark_ratio: float = DARK_RATIO,
    sigma: float = SIGMA,
    weak_factor: float = WEAK_FACTOR,
    gain_range: tuple[float, float] = GAIN_RANGE,
    board_samples: float = BOARD_SAMPLES,
    board_v_pct: float = BOARD_V_PCT,
    board_s_pct: float = BOARD_S_PCT,
    board_s_floor: float = BOARD_S_FLOOR,
) -> ColourModel:
    """Build the colour model for one frame (``frame_bgr`` None or ``adapt`` False = no light
    correction: gains 1). The keyword arguments are the tunables the lab exposes
    (pipelines.common.colour_params)."""
    protos = _class_protos(prototypes)
    names = tuple(protos.keys())
    ref = np.array(hex_to_bgr((prototypes or {}).get("board", DEFAULT_BOARD)), np.float64)
    now = (
        board_colour(frame_bgr, region, samples=board_samples, v_pct=board_v_pct, s_pct=board_s_pct, s_floor=board_s_floor)
        if frame_bgr is not None else None
    )
    g_lo, g_hi = float(gain_range[0]), float(gain_range[1])
    gains = np.ones(3) if (now is None or not adapt) else np.clip(ref / np.maximum(now, 1.0), g_lo, g_hi)
    bgr = np.array([hex_to_bgr(protos[n]) for n in names], np.float64)
    return ColourModel(
        names=names,
        lab=bgr_to_lab(bgr),
        board_lab=bgr_to_lab(ref)[0],
        gains=gains,
        max_dist=float(max_dist),
        chroma_min=float(chroma_min),
        board_now=now,
        clipped=clipped_fraction(frame_bgr, region) if frame_bgr is not None else 0.0,
        w_lightness=float(w_lightness),
        dark_ratio=float(dark_ratio),
        sigma=float(sigma),
        weak_factor=float(weak_factor),
    )


def clipped_fraction(frame_bgr: np.ndarray, region: np.ndarray | None = None) -> float:
    """Fraction of pixels (in ``region``) with any channel at 250 or more, on a subsample.
    Above a few percent the board is overexposed: its colour, and so the light correction
    and the wood-versus-board decision, cannot be trusted."""
    h, w = frame_bgr.shape[:2]
    step = max(1, int(np.sqrt(h * w / 20000)))
    hot = frame_bgr[::step, ::step].max(axis=2) >= 250
    if region is not None:
        sel = region[::step, ::step] > 0
        return float(np.count_nonzero(hot & sel)) / max(int(np.count_nonzero(sel)), 1)
    return float(hot.mean())


def median_bgr(frame_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    """Per-channel median BGR of the pixels where ``mask`` (same height and width) is non-zero."""
    sel = mask > 0
    if not np.any(sel):
        return None
    return np.median(frame_bgr[sel].reshape(-1, 3), axis=0)


def _public(name: str, score: float) -> tuple[Color, float]:
    if name in CLASSES:
        return name, score  # type: ignore[return-value]
    return "unknown", 0.0


def classify_pixels(
    bgr_pixels: np.ndarray,
    prototypes: dict[str, str] | None = None,
    *,
    max_dist: float = DEFAULT_MAX_DIST,
) -> tuple[Color, float]:
    """Classify a set of BGR pixels by their median colour. Without the frame there is no board
    to measure, so the prototypes are used as given (no light adaptation)."""
    px = np.asarray(bgr_pixels, np.uint8).reshape(-1, 3)
    if len(px) == 0:
        return "unknown", 0.0
    model = colour_model(None, prototypes, max_dist=max_dist)
    name, score, _ = model.classify(model.lab_of_bgr(np.median(px, axis=0)))
    return _public(name, score)


def classify_region(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    prototypes: dict[str, str] | None = None,
    *,
    max_dist: float = DEFAULT_MAX_DIST,
) -> tuple[Color, float]:
    """Classify the pixels where ``mask`` is non-zero (mask is HxW, same size as the frame),
    relative to the board measured in the same frame."""
    bgr = median_bgr(frame_bgr, mask)
    if bgr is None:
        return "unknown", 0.0
    model = colour_model(frame_bgr, prototypes, max_dist=max_dist)
    name, score, _ = model.classify(model.lab_of_bgr(bgr))
    return _public(name, score)

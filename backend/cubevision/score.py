"""Scoring detections against a ground-truth set (docs/SCORING.md §2-§7; that file is the spec).

Pure functions on in-memory values: no database, no file I/O, no lab imports.

    gt = parse_gt(json.load(f))                       # cubevision.gt
    scores = [score_image(im, preds_of(im), ms_of(im), crit) for im in gt.images]
    metrics = aggregate(scores)                        # §3
    rows = slices(scores)                              # §6
    cmp = paired_bootstrap(scores_a, scores_b)         # §5, B - A
    side = bootstrap_side(scores_a)                    # A's half, made once for many B:
    cmp_c = paired_bootstrap(scores_a, scores_c, side_a=side, metrics_b=aggregate(scores_c))
    aug = augmentation_effect(scores)                  # §6, variant - original

Predictions may be ``Cube`` objects, ``Cube.to_dict()`` dicts or anything with ``color``,
``center``, ``quad`` and ``angle``. A prediction without a finite centre cannot be matched and
counts as a false positive. A colour outside red/green/blue/wood/unknown is read as "unknown".
Missing values are None, never 0: an undefined ratio has value None, an empty sample has
median / P95 / mean None.
"""

from __future__ import annotations

import csv
import io
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from cubevision.gt import GT_COLORS, ORIGINAL, GtCube, GtImage, Point, finite_number, finite_point, finite_quad
from cubevision.types import COLORS


# --- criteria ------------------------------------------------------------------------------
@dataclass(frozen=True)
class Criteria:
    """Tolerances (provisional, not the robot's real tolerance), the matching gate and the
    bootstrap settings. Scores carry the criteria they were made with."""

    center_px: float = 5.0
    angle_deg: float = 5.0
    vertex_px: float = 5.0
    gate_px: float = 29.0
    bootstrap_reps: int = 2000
    bootstrap_level: float = 0.95
    bootstrap_seed: int = 0

    def __post_init__(self) -> None:
        for name in ("center_px", "angle_deg", "vertex_px", "gate_px"):
            if finite_number(getattr(self, name)) is None or getattr(self, name) < 0:
                raise ValueError(f"기준값 {name}은(는) 0 이상의 유한한 수여야 합니다 ({getattr(self, name)!r})")
        if isinstance(self.bootstrap_reps, bool) or not isinstance(self.bootstrap_reps, int) or self.bootstrap_reps < 1:
            raise ValueError(f"bootstrap 반복 횟수는 1 이상의 정수여야 합니다 ({self.bootstrap_reps!r})")
        if finite_number(self.bootstrap_level) is None or not 0 < self.bootstrap_level < 1:
            raise ValueError(f"bootstrap 구간 수준은 0과 1 사이여야 합니다 ({self.bootstrap_level!r})")
        if isinstance(self.bootstrap_seed, bool) or not isinstance(self.bootstrap_seed, int) or self.bootstrap_seed < 0:
            raise ValueError(f"bootstrap 시드는 0 이상의 정수여야 합니다 ({self.bootstrap_seed!r})")

    @property
    def thresholds(self) -> tuple[float, float, float, float]:
        """(τc, τa, τv, g): what must be equal for two scorings to be comparable."""
        return (float(self.center_px), float(self.angle_deg), float(self.vertex_px), float(self.gate_px))


# --- predictions and per-cube errors -------------------------------------------------------
@dataclass(frozen=True)
class Pred:
    """A model output reduced to what scoring reads. Invalid fields are None."""

    color: str  # one of COLORS
    center: Point | None
    quad: tuple[Point, Point, Point, Point] | None
    angle: float | None


def as_pred(obj: Any) -> Pred:
    """Read a Cube, a Cube.to_dict() dict or a Pred."""
    if isinstance(obj, Pred):
        return obj
    get = obj.get if isinstance(obj, Mapping) else (lambda k: getattr(obj, k, None))
    color = get("color")
    return Pred(color if color in COLORS else "unknown", finite_point(get("center")),
                finite_quad(get("quad")), finite_number(get("angle")))


def center_error(pred_center: Any, gt_center: Point) -> float | None:
    """‖pred − gt‖ in px; None when the prediction is not two finite numbers."""
    p = finite_point(pred_center)
    return None if p is None else math.hypot(p[0] - gt_center[0], p[1] - gt_center[1])


def angle_error(pred_angle: Any, gt_angle: float) -> float | None:
    """min(d, 90 − d) with d = |a − b| mod 90 (a square repeats every 90°); None if not finite."""
    a = finite_number(pred_angle)
    if a is None:
        return None
    d = abs(a - gt_angle) % 90.0
    return min(d, 90.0 - d)


def vertex_rms(pred_quad: Any, gt_quad: Sequence[Point]) -> float | None:
    """RMS corner distance, minimised over the 4 cyclic shifts (the label starts at the topmost
    corner, models at the smallest x + y; both are clockwise). None unless 4 finite points."""
    q = finite_quad(pred_quad)
    return None if q is None else _rms(q, gt_quad)


def _rms(q: Sequence[Point], gt_quad: Sequence[Point]) -> float:
    best = min(
        sum((q[(i + k) % 4][0] - gt_quad[i][0]) ** 2 + (q[(i + k) % 4][1] - gt_quad[i][1]) ** 2 for i in range(4))
        for k in range(4)
    )
    return math.sqrt(best / 4.0)


# --- matching (§2) -------------------------------------------------------------------------
@dataclass(frozen=True)
class Pair:
    gt: int  # index into the GT cubes
    pred: int  # index into the predictions
    distance: float  # ‖pred center − GT detection_match_point‖, ≤ gate


@dataclass(frozen=True)
class MatchResult:
    pairs: tuple[Pair, ...]  # sorted by GT index
    missed: tuple[int, ...]  # GT indices without a pair (FN)
    false_pos: tuple[int, ...]  # prediction indices without a pair (FP)


def match(gt_cubes: Sequence[GtCube], preds: Sequence[Any], gate: float) -> MatchResult:
    """One-to-one pairs within the gate (d ≤ gate): as many pairs as possible, then the smallest
    total distance. Colour is not used."""
    return match_points([c.match_point for c in gt_cubes], [as_pred(p).center for p in preds], gate)


def match_points(gt_points: Sequence[Point], pred_points: Sequence[Point | None], gate: float) -> MatchResult:
    """``match`` on bare points; a None prediction point can never be paired."""
    allowed: dict[tuple[int, int], float] = {}
    for i, g in enumerate(gt_points):
        for j, p in enumerate(pred_points):
            if p is not None:
                d = math.hypot(p[0] - g[0], p[1] - g[1])
                if d <= gate:
                    allowed[i, j] = d
    pairs: list[Pair] = []
    if allowed:
        # Only GT / predictions with at least one allowed partner take part. A forbidden cell costs
        # more than any sum of allowed distances (k · gate + 1), so the cheapest assignment first
        # minimises the number of forbidden cells used, i.e. maximises the number of real pairs.
        rows = sorted({i for i, _ in allowed})
        cols = sorted({j for _, j in allowed})
        big = min(len(rows), len(cols)) * float(gate) + 1.0
        cost = [[allowed.get((i, j), big) for j in cols] for i in rows]
        for r, c in _assignment(cost):
            key = (rows[r], cols[c])
            if key in allowed:
                pairs.append(Pair(key[0], key[1], allowed[key]))
    pairs.sort(key=lambda p: p.gt)
    used_gt = {p.gt for p in pairs}
    used_pred = {p.pred for p in pairs}
    return MatchResult(tuple(pairs), tuple(i for i in range(len(gt_points)) if i not in used_gt),
                       tuple(j for j in range(len(pred_points)) if j not in used_pred))


def _assignment(cost: list[list[float]]) -> list[tuple[int, int]]:
    """Minimum-cost assignment of min(n, m) (row, col) pairs: the Hungarian method with
    potentials, O(n² m). Small inputs only (a handful of cubes per image)."""
    n, m = len(cost), len(cost[0])
    if n > m:
        return [(r, c) for c, r in _assignment([list(col) for col in zip(*cost)])]
    inf = math.inf
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    owner = [0] * (m + 1)  # owner[j] = row (1-based) assigned to column j, 0 = free
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        owner[0] = i
        j0 = 0
        minv = [inf] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = owner[j0], inf, 0
            row = cost[i0 - 1]
            for j in range(1, m + 1):
                if not used[j]:
                    cur = row[j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[owner[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if owner[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            owner[j0] = owner[j1]
            j0 = j1
    return [(owner[j] - 1, j - 1) for j in range(1, m + 1) if owner[j]]


# --- one image (§2, §3) --------------------------------------------------------------------
Outcome = Literal["pass", "fail", "missed"]


@dataclass(frozen=True)
class CubeScore:
    """One GT cube's result. The §3 pass rule lives here and only here: the checks, the cube's
    pass, its outcome in words and (in score_image) the image's pass are all read from these."""

    gt: GtCube
    matched: bool
    pred_index: int | None
    distance: float | None  # match distance (to the GT match point)
    pred_color: str | None
    pred_center: Point | None
    color_ok: bool  # matched and the predicted colour equals the GT colour ("unknown" is wrong)
    center_err: float | None  # the three errors: valid-geometry GT with a pair and a valid output only
    angle_err: float | None
    vertex_err: float | None
    # Tolerance checks of a valid-geometry GT cube (error ≤ τ; a missing error fails), all None for
    # invalid geometry, where only match and colour apply.
    center_ok: bool | None
    angle_ok: bool | None
    vertex_ok: bool | None
    geom_ok: bool | None  # all three
    passed: bool | None  # color_ok and geom_ok (허용 오차 충족); None for invalid geometry
    outcome: Outcome  # "missed" without a pair, else "pass" when the colour is right and geom_ok is not False


@dataclass(frozen=True)
class FalsePositive:
    pred_index: int
    color: str
    center: Point | None


@dataclass(frozen=True)
class ImageScore:
    image: GtImage
    criteria: Criteria
    ran: bool  # False = the run failed: zero detections, no time sample
    ms: float | None  # processing time, a time sample only when ran
    n_preds: int
    cubes: tuple[CubeScore, ...]  # in GT order
    false_pos: tuple[FalsePositive, ...]
    passed: bool  # every GT cube's outcome is "pass" and zero FP


@dataclass(frozen=True)
class OutcomeCounts:
    """One scored image in numbers (the viewer card and the per-image list show these)."""

    n_gt: int
    n_missed: int  # GT cubes without a pair (FN)
    n_fp: int  # detections without a pair
    n_color_wrong: int  # pairs with the wrong colour
    n_geom_fail: int  # valid-geometry pairs over a threshold or without centre / angle / quad


def outcome_counts(s: ImageScore) -> OutcomeCounts:
    return OutcomeCounts(
        n_gt=len(s.cubes),
        n_missed=sum(not c.matched for c in s.cubes),
        n_fp=len(s.false_pos),
        n_color_wrong=sum(c.matched and not c.color_ok for c in s.cubes),
        n_geom_fail=sum(c.matched and c.geom_ok is False for c in s.cubes),
    )


def score_image(gt_image: GtImage, preds: Sequence[Any] | None, ms: float | None,
                criteria: Criteria = Criteria()) -> ImageScore:
    """Score one image. ``preds`` None means the run failed (§4)."""
    ran = preds is not None
    ps = [as_pred(p) for p in preds] if preds is not None else []
    m = match_points([c.match_point for c in gt_image.cubes], [p.center for p in ps], criteria.gate_px)
    by_gt = {p.gt: p for p in m.pairs}
    cubes = []
    for i, g in enumerate(gt_image.cubes):
        pair = by_gt.get(i)
        pred = ps[pair.pred] if pair is not None else None
        color_ok = pred is not None and pred.color == g.color
        ce = ae = ve = None
        center_ok = angle_ok = vertex_ok = geom_ok = passed = None
        if g.valid_geometry:
            if pred is not None:
                ce = center_error(pred.center, g.center)  # type: ignore[arg-type]
                ae = angle_error(pred.angle, g.angle)  # type: ignore[arg-type]
                ve = _rms(pred.quad, g.quad) if pred.quad is not None else None  # type: ignore[arg-type]
            center_ok = ce is not None and ce <= criteria.center_px
            angle_ok = ae is not None and ae <= criteria.angle_deg
            vertex_ok = ve is not None and ve <= criteria.vertex_px
            geom_ok = center_ok and angle_ok and vertex_ok
            passed = color_ok and geom_ok
        outcome: Outcome = "missed" if pred is None else "pass" if color_ok and geom_ok is not False else "fail"
        cubes.append(CubeScore(
            gt=g, matched=pred is not None, pred_index=pair.pred if pair is not None else None,
            distance=pair.distance if pair is not None else None, pred_color=pred.color if pred else None,
            pred_center=pred.center if pred else None, color_ok=color_ok, center_err=ce, angle_err=ae,
            vertex_err=ve, center_ok=center_ok, angle_ok=angle_ok, vertex_ok=vertex_ok, geom_ok=geom_ok,
            passed=passed, outcome=outcome,
        ))
    fps = tuple(FalsePositive(j, ps[j].color, ps[j].center) for j in m.false_pos)
    image_ok = not fps and all(c.outcome == "pass" for c in cubes)
    t = finite_number(ms) if ran else None
    return ImageScore(gt_image, criteria, ran, t if t is not None and t >= 0 else None, len(ps), tuple(cubes),
                      fps, image_ok)


# --- metrics (§3) --------------------------------------------------------------------------
@dataclass(frozen=True)
class Ratio:
    numerator: int
    denominator: int
    value: float | None  # None when the denominator is 0

    @staticmethod
    def of(num: int, den: int) -> Ratio:
        return Ratio(int(num), int(den), num / den if den else None)


@dataclass(frozen=True)
class Stats:
    n: int
    median: float | None
    p95: float | None
    mean: float | None

    @staticmethod
    def of(values: Iterable[float]) -> Stats:
        arr = np.asarray(list(values), dtype=float)
        if arr.size == 0:
            return Stats(0, None, None, None)
        med, p95 = np.quantile(arr, [0.5, 0.95])  # numpy default = linear interpolation
        return Stats(int(arr.size), float(med), float(p95), float(arr.mean()))


def quantile(values: Iterable[float], q: float) -> float | None:
    """Linear-interpolation quantile (numpy default); None for no values."""
    arr = np.asarray(list(values), dtype=float)
    return float(np.quantile(arr, q)) if arr.size else None


@dataclass(frozen=True)
class MetricDef:
    key: str  # attribute of Metrics
    label: str  # Korean display label (summary.csv)
    kind: Literal["ratio", "stat"]
    unit: str


METRICS: tuple[MetricDef, ...] = (
    MetricDef("tolerance_rate", "허용 오차 충족률", "ratio", ""),
    MetricDef("image_pass_rate", "이미지 전체 기준 충족률", "ratio", ""),
    MetricDef("recall", "Recall", "ratio", ""),
    MetricDef("precision", "Precision", "ratio", ""),
    MetricDef("color_accuracy", "색상 정확도", "ratio", ""),
    MetricDef("vertex_err", "꼭짓점 RMS 오차", "stat", "px"),
    MetricDef("center_err", "중심 거리 오차", "stat", "px"),
    MetricDef("angle_err", "회전각 오차", "stat", "°"),
    MetricDef("time_ms", "이미지당 처리 시간", "stat", "ms"),
)
METRIC_BY_KEY = {m.key: m for m in METRICS}

# Scalars compared by the bootstrap: each ratio, and the median and P95 of each sample.
BOOTSTRAP_KEYS: tuple[str, ...] = tuple(
    k for m in METRICS for k in ((m.key,) if m.kind == "ratio" else (f"{m.key}_median", f"{m.key}_p95"))
)


@dataclass(frozen=True)
class Metrics:
    """§3 metrics over a set of images (or, for a cube slice, a set of GT cubes: then the
    image-level fields n_images / n_failed / n_preds / precision / image_pass_rate / time_ms are None)."""

    criteria: Criteria | None  # None only for an empty input
    n_images: int | None
    n_failed: int | None  # images whose run failed
    n_preds: int | None
    n_cubes: int
    n_valid: int
    tolerance_rate: Ratio  # passed valid-geometry GT / valid-geometry GT
    image_pass_rate: Ratio | None  # passed images / images
    recall: Ratio  # pairs / GT
    precision: Ratio | None  # pairs / detections
    color_accuracy: Ratio  # right colour / pairs
    confusion: dict[str, dict[str, int]]  # GT colour -> predicted value -> pairs
    vertex_err: Stats
    center_err: Stats
    angle_err: Stats
    time_ms: Stats | None

    def value(self, key: str) -> float | None:
        """A BOOTSTRAP_KEYS scalar ("recall", "center_err_median", "time_ms_p95", ...)."""
        if key in METRIC_BY_KEY and METRIC_BY_KEY[key].kind == "ratio":
            r = getattr(self, key)
            return None if r is None else r.value
        base, _, part = key.rpartition("_")
        if part not in ("median", "p95") or base not in METRIC_BY_KEY or METRIC_BY_KEY[base].kind != "stat":
            raise KeyError(key)
        s = getattr(self, base)
        return None if s is None else getattr(s, part)


def _common_criteria(scores: Sequence[ImageScore]) -> Criteria | None:
    if not scores:
        return None
    first = scores[0].criteria
    if any(s.criteria.thresholds != first.thresholds for s in scores):
        raise ValueError("서로 다른 기준값(τc, τa, τv, g)으로 채점한 결과는 함께 집계할 수 없습니다")
    return first


# Per-image sufficient statistics. Every §3 ratio is a sum of per-image counts over another, and
# every error statistic is taken over the per-image samples put together, so aggregate() and each
# bootstrap resample are computed from the same counts and a metric is defined once, here.
_CUBE_COUNTS = ("gt", "matched", "color_ok", "valid", "valid_passed")  # also defined on a cube slice
_COUNTS = ("images", "images_passed", "preds", *_CUBE_COUNTS)
_RATIOS = {
    "tolerance_rate": ("valid_passed", "valid"),
    "image_pass_rate": ("images_passed", "images"),
    "recall": ("matched", "gt"),
    "precision": ("matched", "preds"),
    "color_accuracy": ("color_ok", "matched"),
}
_CUBE_SAMPLES = ("vertex_err", "center_err", "angle_err")  # CubeScore attributes; also on a cube slice
_SAMPLES = (*_CUBE_SAMPLES, "time_ms")


def _cube_counts(cubes: Sequence[CubeScore]) -> list[int]:
    matched = color_ok = valid = valid_passed = 0
    for c in cubes:
        matched += c.matched
        color_ok += c.color_ok  # only a pair can have the right colour
        valid += c.gt.valid_geometry
        valid_passed += bool(c.passed)  # None (invalid geometry) never passes
    return [len(cubes), matched, color_ok, valid, valid_passed]


def _image_counts(s: ImageScore) -> list[int]:
    return [1, int(s.passed), s.n_preds, *_cube_counts(s.cubes)]


def _cube_samples(cubes: Sequence[CubeScore]) -> dict[str, list[float]]:
    valid = [c for c in cubes if c.gt.valid_geometry]
    return {k: [v for c in valid if (v := getattr(c, k)) is not None] for k in _CUBE_SAMPLES}


def _image_samples(s: ImageScore) -> dict[str, list[float]]:
    return {**_cube_samples(s.cubes), "time_ms": [s.ms] if s.ms is not None else []}


def _confusion(cubes: Iterable[CubeScore]) -> dict[str, dict[str, int]]:
    out = {g: {p: 0 for p in COLORS} for g in GT_COLORS}
    for c in cubes:
        if c.matched:
            out[c.gt.color][c.pred_color] += 1  # type: ignore[index]
    return out


def _metrics(criteria: Criteria | None, counts: Mapping[str, int], samples: Mapping[str, list[float]],
             cubes: Iterable[CubeScore], n_failed: int | None) -> Metrics:
    """Metrics from summed counts and pooled samples. Without the image-level counts (a cube
    slice) the image-level fields are None: _RATIOS / _SAMPLES entries they need are missing."""
    ratios = {k: Ratio.of(counts[num], counts[den]) for k, (num, den) in _RATIOS.items() if num in counts and den in counts}
    stats = {k: Stats.of(v) for k, v in samples.items()}
    return Metrics(
        criteria=criteria, n_images=counts.get("images"), n_failed=n_failed, n_preds=counts.get("preds"),
        n_cubes=counts["gt"], n_valid=counts["valid"],
        tolerance_rate=ratios["tolerance_rate"], image_pass_rate=ratios.get("image_pass_rate"),
        recall=ratios["recall"], precision=ratios.get("precision"), color_accuracy=ratios["color_accuracy"],
        confusion=_confusion(cubes),
        vertex_err=stats["vertex_err"], center_err=stats["center_err"], angle_err=stats["angle_err"],
        time_ms=stats.get("time_ms"),
    )


def _cube_metrics(cubes: Sequence[CubeScore], criteria: Criteria | None) -> Metrics:
    return _metrics(criteria, dict(zip(_CUBE_COUNTS, _cube_counts(cubes))), _cube_samples(cubes), cubes, None)


def aggregate(image_scores: Sequence[ImageScore]) -> Metrics:
    """All §3 metrics over the given images (each listed image counts once per listing)."""
    scores = list(image_scores)
    crit = _common_criteria(scores)
    counts = dict.fromkeys(_COUNTS, 0)
    samples: dict[str, list[float]] = {k: [] for k in _SAMPLES}
    for s in scores:
        for k, n in zip(_COUNTS, _image_counts(s)):
            counts[k] += n
        for k, v in _image_samples(s).items():
            samples[k].extend(v)
    return _metrics(crit, counts, samples, (c for s in scores for c in s.cubes), sum(not s.ran for s in scores))


# --- slices (§6) ---------------------------------------------------------------------------
SliceValue = str | bool | None

# (dimension, Korean label, unit, preferred value order)
SLICE_DIMENSIONS: tuple[tuple[str, str, Literal["image", "cube"], tuple[Any, ...]], ...] = (
    ("variant", "증강", "image", (ORIGINAL,)),
    ("region", "위치", "cube", ("center", "middle", "outer")),
    ("image_side", "좌우", "cube", ("left", "right")),
    ("color", "색", "cube", GT_COLORS),
    ("near_contact", "근접", "cube", (True, False)),
    ("weak_boundary", "경계 약함", "cube", (True, False)),
)


@dataclass(frozen=True)
class SliceRow:
    dimension: str  # a SLICE_DIMENSIONS key
    value: SliceValue
    n: int  # images for "variant", GT cubes otherwise
    metrics: Metrics  # cube slices: image-level fields are None (undefined there)


def _ordered(values: Iterable[Any], preferred: tuple[Any, ...]) -> list[Any]:
    seen = list(dict.fromkeys(values))
    return [v for v in preferred if v in seen] + [v for v in seen if v not in preferred]


def slices(image_scores: Sequence[ImageScore]) -> tuple[SliceRow, ...]:
    """The variant slice (image level, all metrics) then the cube-level slices."""
    scores = list(image_scores)
    crit = _common_criteria(scores)
    rows: list[SliceRow] = []
    cubes = [c for s in scores for c in s.cubes]
    for dim, _label, unit, preferred in SLICE_DIMENSIONS:
        if unit == "image":
            for v in _ordered((s.image.variant for s in scores), preferred):
                part = [s for s in scores if s.image.variant == v]
                rows.append(SliceRow(dim, v, len(part), aggregate(part)))
        else:
            for v in _ordered((getattr(c.gt, dim) for c in cubes), preferred):
                part = [c for c in cubes if getattr(c.gt, dim) == v]
                rows.append(SliceRow(dim, v, len(part), _cube_metrics(part, crit)))
    return tuple(rows)


# --- paired bootstrap (§5) and augmentation effect (§6) ------------------------------------
Verdict = Literal["higher", "lower", "unclear"]


@dataclass(frozen=True)
class Diff:
    key: str  # a BOOTSTRAP_KEYS scalar
    a: float | None  # on the full image set
    b: float | None
    delta: float | None  # b − a
    ci_low: float | None  # percentile interval of the resampled b − a
    ci_high: float | None
    verdict: Verdict | None  # "unclear" when the interval contains 0; None when never defined
    reps_used: int  # resamples where the metric was defined for both A and B


@dataclass(frozen=True)
class Comparison:
    n_images_a: int
    n_images_b: int
    n_groups: int  # resampling units (layout_group)
    reps: int
    level: float
    seed: int
    diffs: tuple[Diff, ...]  # in BOOTSTRAP_KEYS order

    def get(self, key: str) -> Diff:
        return next(d for d in self.diffs if d.key == key)


@dataclass(frozen=True, eq=False)
class BootstrapSide:
    """Side A of a paired bootstrap: its metrics on the full image set, the draw (how often each
    resample takes each layout_group) and its value of every BOOTSTRAP_KEYS scalar on each
    resample. It depends only on A's images and the draw (its groups, reps, seed), so one side
    serves every B compared with it: the reference for every batch, the originals for every
    augmentation variant. Results are the same as computing it afresh for each B."""

    images: tuple[str, ...]  # sha256 of A's images, in order
    groups: tuple[str, ...]  # the resampled units: A's layout_groups, sorted
    reps: int
    seed: int
    metrics: Metrics  # aggregate(A)
    draw: np.ndarray  # (reps, groups)
    values: dict[str, np.ndarray]  # BOOTSTRAP_KEYS scalar -> (reps,), NaN where undefined


def bootstrap_side(scores: Sequence[ImageScore], criteria: Criteria | None = None,
                   metrics: Metrics | None = None) -> BootstrapSide:
    """A's side for paired_bootstrap(scores, ...), each image resampled with its own layout_group.
    ``criteria`` supplies reps / seed (default: the scores' own); ``metrics`` is aggregate(scores)
    when the caller has it already."""
    a = list(scores)
    if not a:
        raise ValueError("비교할 채점 결과가 비어 있습니다")
    crit = _bootstrap_criteria(a, a, criteria)
    names = tuple(sorted({s.image.layout_group for s in a}))  # sorted: the draw does not depend on image order
    index = {g: i for i, g in enumerate(names)}
    draw = _draw_counts(len(names), crit.bootstrap_reps, crit.bootstrap_seed)
    values = _resampled(_group_stats(a, [index[s.image.layout_group] for s in a], len(names)), draw)
    return BootstrapSide(tuple(s.image.sha256 for s in a), names, crit.bootstrap_reps, crit.bootstrap_seed,
                         metrics if metrics is not None else aggregate(a), draw, values)


def paired_bootstrap(scores_a: Sequence[ImageScore], scores_b: Sequence[ImageScore],
                     criteria: Criteria | None = None, *, side_a: BootstrapSide | None = None,
                     metrics_b: Metrics | None = None) -> Comparison:
    """B − A per metric, resampling layout_groups with replacement (never single images) and
    scoring both batches on the same draw. Both must cover the same images (by sha256) with the
    same thresholds. ``criteria`` supplies reps / level / seed (default: the scores' own).
    ``side_a`` (bootstrap_side(scores_a, criteria)) and ``metrics_b`` (aggregate(scores_b)) are
    reused instead of recomputed when given: one reference compared with many batches."""
    a, b = list(scores_a), list(scores_b)
    if not a or not b:
        raise ValueError("비교할 채점 결과가 비어 있습니다")
    crit = _bootstrap_criteria(a, b, criteria)
    sha_a = [s.image.sha256 for s in a]
    sha_b = {s.image.sha256: s for s in b}
    if len(set(sha_a)) != len(sha_a) or len(sha_b) != len(b):
        raise ValueError("한 배치 안에 같은 이미지가 두 번 들어 있습니다")
    if set(sha_a) != set(sha_b):
        raise ValueError(f"두 배치의 평가 이미지 집합이 다릅니다 (A만 {len(set(sha_a) - set(sha_b))}장, "
                         f"B만 {len(set(sha_b) - set(sha_a))}장)")
    groups = {s.image.sha256: s.image.layout_group for s in a}
    if any(groups[s.image.sha256] != s.image.layout_group for s in b):
        raise ValueError("같은 이미지의 layout_group이 두 배치에서 다릅니다 (다른 정답셋으로 채점한 결과)")
    if side_a is None:
        side_a = bootstrap_side(a, crit)
    elif (side_a.images != tuple(sha_a) or (side_a.reps, side_a.seed) != (crit.bootstrap_reps, crit.bootstrap_seed)
          or side_a.metrics.criteria is None or side_a.metrics.criteria.thresholds != crit.thresholds):
        raise ValueError("bootstrap의 A쪽(side_a)이 이 배치와 설정으로 만든 것이 아닙니다")
    return _compare(side_a, b, [groups[s.image.sha256] for s in b], crit, metrics_b)


@dataclass(frozen=True)
class AugmentationEffect:
    variant: str
    n_sources: int  # originals paired with this variant (by source_sha256)
    n_unpaired: int  # variant images whose original is not in the scores
    comparison: Comparison  # A = the paired originals, B = the variant images


def augmentation_effect(image_scores: Sequence[ImageScore],
                        criteria: Criteria | None = None) -> tuple[AugmentationEffect, ...]:
    """For each variant ≠ original: Δ = metric(variant images) − metric(their originals), with
    the same layout_group bootstrap. Variants without any paired original are left out."""
    scores = list(image_scores)
    if not scores:
        return ()
    crit = _bootstrap_criteria(scores, scores, criteria)
    originals = {s.image.sha256: s for s in scores if s.image.variant == ORIGINAL}
    by_variant: dict[str, list[ImageScore]] = {}
    for s in scores:
        if s.image.variant != ORIGINAL:
            by_variant.setdefault(s.image.variant, []).append(s)
    sides: dict[tuple[str, ...], BootstrapSide] = {}  # variants of the same originals share their side
    out = []
    for variant, vs in by_variant.items():
        paired = [s for s in vs if s.image.source_sha256 in originals]
        if not paired:
            continue
        sources = tuple(dict.fromkeys(s.image.source_sha256 for s in paired))
        if sources not in sides:
            sides[sources] = bootstrap_side([originals[src] for src in sources], crit)
        group_of = {src: originals[src].image.layout_group for src in sources}  # the original's group for both sides
        cmp = _compare(sides[sources], paired, [group_of[s.image.source_sha256] for s in paired], crit, None)
        out.append(AugmentationEffect(variant, len(sources), len(vs) - len(paired), cmp))
    return tuple(out)


def _bootstrap_criteria(a: Sequence[ImageScore], b: Sequence[ImageScore], criteria: Criteria | None) -> Criteria:
    ca, cb = _common_criteria(a), _common_criteria(b)
    assert ca is not None and cb is not None
    if ca.thresholds != cb.thresholds:
        raise ValueError("두 배치의 기준값(τc, τa, τv, g)이 달라 비교할 수 없습니다")
    if criteria is not None and criteria.thresholds != ca.thresholds:
        raise ValueError("bootstrap에 준 기준값이 채점에 쓴 기준값과 다릅니다")
    return criteria or ca


_QS = (0.5, 0.95)
_CHUNK = 256  # resamples per vectorised block (bounds memory: CHUNK × values × 8 bytes)


@dataclass(frozen=True)
class _GroupStats:
    counts: np.ndarray  # (groups, len(_COUNTS)) int64
    # key -> (sorted values, (groups, values) count of each group's values among the first j + 1)
    samples: dict[str, tuple[np.ndarray, np.ndarray]]


def _group_stats(scores: Sequence[ImageScore], group_idx: Sequence[int], n_groups: int) -> _GroupStats:
    counts = np.zeros((n_groups, len(_COUNTS)), np.int64)
    vals: dict[str, list[float]] = {k: [] for k in _SAMPLES}
    labs: dict[str, list[int]] = {k: [] for k in _SAMPLES}
    for s, g in zip(scores, group_idx):
        counts[g] += _image_counts(s)
        for k, v in _image_samples(s).items():
            vals[k].extend(v)
            labs[k].extend([g] * len(v))
    samples = {}
    for k in _SAMPLES:
        v = np.asarray(vals[k], dtype=float)
        order = np.argsort(v, kind="stable")
        member = np.zeros((n_groups, v.size))
        member[np.asarray(labs[k], dtype=np.int64)[order], np.arange(v.size)] = 1.0
        samples[k] = (v[order], np.cumsum(member, axis=1))
    return _GroupStats(counts, samples)


def _draw_counts(n_groups: int, reps: int, seed: int) -> np.ndarray:
    """(reps, groups): how many times each group was drawn in each resample of n_groups draws."""
    draws = np.random.default_rng(seed).integers(0, n_groups, size=(reps, n_groups))
    flat = (draws + np.arange(reps)[:, None] * n_groups).ravel()
    return np.bincount(flat, minlength=reps * n_groups).reshape(reps, n_groups)


def _resampled_quantiles(values: np.ndarray, member: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """np.quantile(..., _QS) of every resample without building it. A resample repeats each value
    as often as its group was drawn, so on the sorted values its k-th element is the first value
    whose cumulative weight exceeds k; the cumulative weights of all resamples are one matrix
    product (counts @ member, exact integers in float64). -> (len(_QS), reps), NaN if empty."""
    reps, n = counts.shape[0], values.size
    out = np.full((len(_QS), reps), np.nan)
    if n == 0:
        return out
    weights = counts.astype(float)
    for start in range(0, reps, _CHUNK):
        cum = weights[start:start + _CHUNK] @ member  # (r, n), rows nondecreasing
        r = cum.shape[0]
        total = cum[:, -1].copy()  # a copy: cum is shifted in place below
        offset = np.arange(r) * (total.max() + 1.0)
        cum += offset[:, None]  # rows stacked into one sorted array: a single searchsorted for all rows
        flat = cum.ravel()
        base = np.arange(r) * n
        for qi, q in enumerate(_QS):
            h = (total - 1) * q  # numpy "linear": virtual index (N − 1) · q
            lo = np.floor(h)
            gamma = h - lo
            hi = np.minimum(lo + 1, total - 1)
            jlo = np.clip(np.searchsorted(flat, lo + offset, side="right") - base, 0, n - 1)
            jhi = np.clip(np.searchsorted(flat, hi + offset, side="right") - base, 0, n - 1)
            x0, x1 = values[jlo], values[jhi]
            diff = x1 - x0
            val = np.where(gamma >= 0.5, x1 - diff * (1 - gamma), x0 + diff * gamma)  # numpy's lerp
            out[qi, start:start + r] = np.where(total > 0, val, np.nan)
    return out


def _resampled(stats: _GroupStats, counts: np.ndarray) -> dict[str, np.ndarray]:
    sums = counts @ stats.counts  # (reps, len(_COUNTS)) exact integer sums
    col = {name: i for i, name in enumerate(_COUNTS)}
    out: dict[str, np.ndarray] = {}
    for key, (num, den) in _RATIOS.items():
        d = sums[:, col[den]].astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[key] = np.where(d > 0, sums[:, col[num]] / d, np.nan)
    for key in _SAMPLES:
        med, p95 = _resampled_quantiles(*stats.samples[key], counts)
        out[f"{key}_median"], out[f"{key}_p95"] = med, p95
    return out


def _compare(side: BootstrapSide, b: Sequence[ImageScore], groups_b: Sequence[str], crit: Criteria,
             metrics_b: Metrics | None) -> Comparison:
    """B − A on A's draw. Every group of B is one of A's (the callers check that)."""
    index = {g: i for i, g in enumerate(side.groups)}
    rb = _resampled(_group_stats(b, [index[g] for g in groups_b], len(side.groups)), side.draw)
    ma, mb = side.metrics, metrics_b if metrics_b is not None else aggregate(b)
    alpha = (1.0 - crit.bootstrap_level) / 2.0
    diffs = []
    for key in BOOTSTRAP_KEYS:
        va, vb = ma.value(key), mb.value(key)
        d = rb[key] - side.values[key]
        d = d[np.isfinite(d)]
        lo = hi = verdict = None
        if d.size:
            lo, hi = (float(x) for x in np.quantile(d, [alpha, 1.0 - alpha]))
            verdict = "higher" if lo > 0 else "lower" if hi < 0 else "unclear"
        diffs.append(Diff(key, va, vb, None if va is None or vb is None else vb - va, lo, hi, verdict, int(d.size)))
    return Comparison(len(side.images), len(b), len(side.groups), crit.bootstrap_reps, crit.bootstrap_level,
                      crit.bootstrap_seed, tuple(diffs))


# --- CSV rows (§7) -------------------------------------------------------------------------
_CRITERIA_FIELDS = ("tau_center_px", "tau_angle_deg", "tau_vertex_px", "gate_px")

CUBE_CSV_FIELDS: tuple[str, ...] = (
    "batch", "status", "image", "sha256", "variant", "source_sha256", "layout_group", "row",
    "gt_id", "gt_color", "valid_geometry", "pred_color", "pred_x", "pred_y", "match_distance_px",
    "center_err_px", "angle_err_deg", "vertex_rms_px", "passed",
    "region", "image_side", "near_contact", "weak_boundary", "run_failed",
) + _CRITERIA_FIELDS

SUMMARY_CSV_FIELDS: tuple[str, ...] = (
    "batch", "status", "metric", "label", "unit", "value", "numerator", "denominator", "n", "median", "p95", "mean",
) + _CRITERIA_FIELDS


def _cell(x: Any, digits: int = 4) -> Any:
    """CSV cell: booleans as 1 / 0, floats rounded, None stays None (an empty cell)."""
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, float):
        return round(x, digits)
    return x


def _criteria_cells(c: Criteria | None) -> dict[str, Any]:
    vals = c.thresholds if c is not None else (None,) * 4
    return dict(zip(_CRITERIA_FIELDS, vals))


def _status(complete: bool) -> str:
    """A batch's status column: 완료, or 중간 for an interim batch (§4)."""
    return "완료" if complete else "중간"


def cube_csv_rows(batch: str | int, score: ImageScore, complete: bool) -> list[dict[str, Any]]:
    """cubes.csv rows for one image: one per GT cube ("gt"), then one per false positive ("fp")."""
    im = score.image
    head = {"batch": batch, "status": _status(complete), "image": im.filename, "sha256": im.sha256,
            "variant": im.variant, "source_sha256": im.source_sha256, "layout_group": im.layout_group}
    tail = {"run_failed": _cell(not score.ran), **_criteria_cells(score.criteria)}
    rows = []
    for c in score.cubes:
        g = c.gt
        rows.append({
            **head, "row": "gt", "gt_id": g.id, "gt_color": g.color, "valid_geometry": _cell(g.valid_geometry),
            "pred_color": c.pred_color, "pred_x": _cell(c.pred_center[0]) if c.pred_center else None,
            "pred_y": _cell(c.pred_center[1]) if c.pred_center else None, "match_distance_px": _cell(c.distance),
            "center_err_px": _cell(c.center_err), "angle_err_deg": _cell(c.angle_err),
            "vertex_rms_px": _cell(c.vertex_err), "passed": _cell(c.passed),
            "region": g.region, "image_side": g.image_side, "near_contact": _cell(g.near_contact),
            "weak_boundary": _cell(g.weak_boundary), **tail,
        })
    for fp in score.false_pos:
        rows.append({
            **head, "row": "fp", "gt_id": None, "gt_color": None, "valid_geometry": None,
            "pred_color": fp.color, "pred_x": _cell(fp.center[0]) if fp.center else None,
            "pred_y": _cell(fp.center[1]) if fp.center else None, "match_distance_px": None,
            "center_err_px": None, "angle_err_deg": None, "vertex_rms_px": None, "passed": None,
            "region": None, "image_side": None, "near_contact": None, "weak_boundary": None, **tail,
        })
    return rows


def summary_csv_rows(batch: str | int, metrics: Metrics, complete: bool) -> list[dict[str, Any]]:
    """summary.csv rows: one per metric of METRICS. ``complete`` False marks an interim batch (중간)."""
    head = {"batch": batch, "status": _status(complete)}
    rows = []
    for m in METRICS:
        v = getattr(metrics, m.key)
        row = {**head, "metric": m.key, "label": m.label, "unit": m.unit, "value": None, "numerator": None,
               "denominator": None, "n": None, "median": None, "p95": None, "mean": None,
               **_criteria_cells(metrics.criteria)}
        if isinstance(v, Ratio):
            row.update(value=_cell(v.value, 6), numerator=v.numerator, denominator=v.denominator, n=v.denominator)
        elif isinstance(v, Stats):
            row.update(n=v.n, median=_cell(v.median), p95=_cell(v.p95), mean=_cell(v.mean))
        rows.append(row)
    return rows


def csv_bytes(rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    """UTF-8 with BOM (Excel reads Korean); None is an empty cell."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
    return ("﻿" + buf.getvalue()).encode("utf-8")

"""Scoring against the ground truth (docs/SCORING.md): hand-computed cases, matching against brute
force, the group bootstrap, and the two real ground-truth files in tests/data/gt."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cubevision.gt import SCHEMA_AUGMENTED, SCHEMA_V11, GtCube, GtImage, GtSet, parse_gt
from cubevision.score import (
    BOOTSTRAP_KEYS, CUBE_CSV_FIELDS, METRICS, SUMMARY_CSV_FIELDS, Criteria, ImageScore, OutcomeCounts, Stats,
    _draw_counts, _group_stats, _resampled, aggregate, angle_error, augmentation_effect, bootstrap_side,
    center_error, csv_bytes, cube_csv_rows, match, match_points, outcome_counts, paired_bootstrap, quantile,
    score_image, slices, summary_csv_rows, vertex_rms,
)
from cubevision.types import Cube

GT_DIR = Path(__file__).resolve().parents[1] / "data" / "gt"
GT41 = GT_DIR / "robothon_ground_truth_all41.json"
GT410 = GT_DIR / "robothon_labels_410.json"


# --- builders ------------------------------------------------------------------------------
def sq(cx: float, cy: float, side: float = 60.0, angle: float = 0.0) -> tuple[tuple[float, float], ...]:
    """Square corners, clockwise on screen (y down), starting top-left for angle 0."""
    a = math.radians(angle)
    u, v = (math.cos(a), math.sin(a)), (-math.sin(a), math.cos(a))
    h = side / 2
    return tuple((cx + s * h * u[0] + t * h * v[0], cy + s * h * u[1] + t * h * v[1])
                 for s, t in ((-1, -1), (1, -1), (1, 1), (-1, 1)))


def gcube(cid: str, x: float, y: float, color: str = "red", valid: bool = True, angle: float = 10.0,
          region: str | None = "middle", side: str | None = "left", near: bool = False,
          weak: bool = False) -> GtCube:
    return GtCube(cid, color, valid, (x, y) if valid else None, sq(x, y, angle=angle) if valid else None,
                  angle if valid else None, (x, y), region, side, near, weak, ("weak_boundary",) if weak else ())


def gimage(cubes: list[GtCube], sha: str = "s0", group: str = "g0", variant: str = "original",
           source: str | None = None) -> GtImage:
    return GtImage(0, sha, f"{sha}.png", f"{sha}.png", variant, source or sha, group, 1280, 720, tuple(cubes))


def perfect(im: GtImage) -> list[dict[str, Any]]:
    """Predictions equal to the label; an invalid-geometry cube at its match point with no quad / angle."""
    out = []
    for c in im.cubes:
        if c.valid_geometry:
            out.append({"color": c.color, "center": list(c.center), "quad": [list(p) for p in c.quad],
                        "angle": c.angle})
        else:
            out.append({"color": c.color, "center": list(c.match_point), "quad": None, "angle": None})
    return out


def noisy(im: GtImage, rng: random.Random, noise: float, miss: float = 0.1) -> list[dict[str, Any]] | None:
    """Predictions with Gaussian errors, misses, a wrong colour now and then, missing quads,
    an occasional false positive, and now and then a failed run (None)."""
    if rng.random() < 0.05:
        return None
    out = []
    for c in im.cubes:
        if rng.random() < miss:
            continue
        x, y = c.center or c.match_point
        dx, dy = rng.gauss(0, noise), rng.gauss(0, noise)
        quad = ([[p[0] + dx + rng.gauss(0, noise), p[1] + dy + rng.gauss(0, noise)] for p in c.quad]
                if c.quad and rng.random() > 0.1 else None)
        out.append({"color": c.color if rng.random() > 0.1 else "unknown", "center": [x + dx, y + dy],
                    "quad": quad, "angle": (c.angle + rng.gauss(0, noise)) % 90 if c.angle is not None else None})
    if rng.random() < 0.3:
        out.append({"color": "blue", "center": [rng.uniform(0, 1280), rng.uniform(0, 720)], "quad": None,
                    "angle": None})
    return out


def row_images(n: int, group: str, prefix: str, variant: str = "original",
               sources: list[str] | None = None) -> list[GtImage]:
    """n images of 4 cubes each (one invalid-geometry), all in one layout group."""
    out = []
    for k in range(n):
        cubes = [gcube("r", 100, 100, "red"), gcube("g", 300, 100, "green", angle=80.0),
                 gcube("b", 500, 300, "blue", region="outer", side="right", near=True),
                 gcube("w", 700, 300, "wood", valid=False, region="outer", side="right", weak=True)]
        sha = f"{prefix}{k}"
        out.append(gimage(cubes, sha, group, variant, sources[k] if sources else None))
    return out


def _brute(gt: list[tuple[float, float]], pr: list[tuple[float, float]], gate: float) -> tuple[int, float]:
    """Best (most pairs, then least total distance) over every one-to-one partial matching."""
    best = (0, 0.0)

    def go(i: int, used: frozenset[int], count: int, total: float) -> None:
        nonlocal best
        if i == len(gt):
            if count > best[0] or (count == best[0] and total < best[1] - 1e-12):
                best = (count, total)
            return
        go(i + 1, used, count, total)
        for j, p in enumerate(pr):
            d = math.dist(gt[i], p)
            if j not in used and d <= gate:
                go(i + 1, used | {j}, count + 1, total + d)

    go(0, frozenset(), 0, 0.0)
    return best


# --- per-cube errors -----------------------------------------------------------------------
def test_vertex_rms_same_square_any_start_corner_is_zero() -> None:
    """The label starts at the topmost corner, a model at the smallest x + y: a cyclic shift of the
    same clockwise square is the same square."""
    g = sq(400, 300, angle=20.0)
    for k in range(4):
        assert vertex_rms(g[k:] + g[:k], g) == pytest.approx(0.0, abs=1e-12)


def test_vertex_rms_known_values() -> None:
    """Every corner moved by (3, 4) -> RMS 5 whatever the start corner; one corner moved by 4 px ->
    sqrt(16 / 4) = 2."""
    g = sq(400, 300)
    moved = [(x + 3, y + 4) for x, y in g]
    for k in range(4):
        assert vertex_rms(moved[k:] + moved[:k], g) == pytest.approx(5.0)
    one = list(g)
    one[2] = (one[2][0] + 4, one[2][1])
    assert vertex_rms(one, g) == pytest.approx(2.0)


@pytest.mark.parametrize("quad", [None, [], [(0, 0)] * 3, [(0, 0)] * 5, [(0, 0), (1, 0), (1, 1), (0, math.nan)],
                                  [(0, 0), (1, 0), (1, 1), (0, math.inf)], "abcd", [(0, 0), (1, 0), (1, 1), None]])
def test_vertex_rms_without_four_finite_points_is_none(quad: Any) -> None:
    assert vertex_rms(quad, sq(0, 0)) is None


@pytest.mark.parametrize("a, b, want", [(89, 1, 2), (1, 89, 2), (0, 45, 45), (10, 100, 0), (-1, 1, 2),
                                        (44.5, 0, 44.5), (135, 0, 45), (0, 0, 0)])
def test_angle_error_wraps_at_90(a: float, b: float, want: float) -> None:
    """A square repeats every 90 degrees: 89 vs 1 is 2, 10 vs 100 is 0, the largest error is 45."""
    assert angle_error(a, b) == pytest.approx(want, abs=1e-9)


@pytest.mark.parametrize("bad", [None, math.nan, math.inf, "10", True])
def test_angle_and_center_error_need_finite_numbers(bad: Any) -> None:
    assert angle_error(bad, 10.0) is None
    assert center_error([bad, 0.0], (0.0, 0.0)) is None
    assert center_error(bad, (0.0, 0.0)) is None


def test_center_error_is_euclidean_and_reads_numpy_values() -> None:
    assert center_error([3, 4], (0.0, 0.0)) == pytest.approx(5.0)
    assert center_error(np.array([3, 4], np.float32), (0.0, 0.0)) == pytest.approx(5.0)


# --- matching ------------------------------------------------------------------------------
def test_gate_is_inclusive_at_29_px() -> None:
    """d ≤ g pairs, d > g never does: 29.0 px matches, 29.01 px does not (gate 29)."""
    g = [gcube("a", 100, 100)]
    assert len(match(g, [{"center": [129.0, 100]}], 29).pairs) == 1
    m = match(g, [{"center": [129.01, 100]}], 29)
    assert (m.pairs, m.missed, m.false_pos) == ((), (0,), (0,))


def test_matching_maximises_pair_count_before_distance() -> None:
    """GT A (0, 0), GT B (28, 0); pred p (5, 0), pred q (-20, 0). Nearest-first takes p-A (5) and
    leaves q unmatched (48 px from B); so does a pure minimum sum. Count first: q-A (20) + p-B (23)."""
    m = match_points([(0, 0), (28, 0)], [(5, 0), (-20, 0)], 29)
    assert [(p.gt, p.pred) for p in m.pairs] == [(0, 1), (1, 0)]
    assert sum(p.distance for p in m.pairs) == pytest.approx(43.0)
    assert m.missed == () and m.false_pos == ()


def test_matching_ignores_colour() -> None:
    """A blue prediction next to a red label pairs with it (and counts as a colour error)."""
    im = gimage([gcube("r", 100, 100, "red"), gcube("b", 140, 100, "blue")])
    s = score_image(im, [{"color": "blue", "center": [105, 100]}, {"color": "red", "center": [135, 100]}], 1.0)
    assert [c.matched for c in s.cubes] == [True, True]
    assert [c.color_ok for c in s.cubes] == [False, False]
    assert [c.pred_index for c in s.cubes] == [0, 1]


def test_matching_equals_brute_force_on_random_small_cases() -> None:
    """400 random cases with up to 4 GT and 6 predictions packed into a 70 px box so many pairs
    compete: same pair count and same total distance as exhaustive search."""
    rng = random.Random(7)
    for _ in range(400):
        gt = [(rng.uniform(0, 70), rng.uniform(0, 70)) for _ in range(rng.randint(0, 4))]
        pr = [(rng.uniform(0, 70), rng.uniform(0, 70)) for _ in range(rng.randint(0, 6))]
        m = match_points(gt, pr, 29.0)
        count, total = _brute(gt, pr, 29.0)
        assert len(m.pairs) == count
        assert sum(p.distance for p in m.pairs) == pytest.approx(total, abs=1e-9)
        assert len({p.pred for p in m.pairs}) == len(m.pairs) and all(p.distance <= 29.0 for p in m.pairs)
        assert sorted([p.gt for p in m.pairs] + list(m.missed)) == list(range(len(gt)))
        assert sorted([p.pred for p in m.pairs] + list(m.false_pos)) == list(range(len(pr)))


def test_prediction_without_a_centre_is_a_false_positive() -> None:
    im = gimage([gcube("r", 100, 100)])
    s = score_image(im, [{"color": "red", "center": None}, {"color": "red", "center": [math.nan, 1]}], 1.0)
    assert not s.cubes[0].matched and len(s.false_pos) == 2 and s.n_preds == 2


# --- one image -----------------------------------------------------------------------------
def test_tolerances_are_inclusive() -> None:
    """Centre and every corner off by (3, 4) -> centre 5 px, RMS 5 px; angle off by 5° -> all exactly 5
    (integer corners, so no rounding): pass. Off by (3, 4.01) fails."""
    im = gimage([gcube("r", 100, 100, angle=0.0)])
    ok = [{"color": "red", "center": [103, 104], "quad": [[x + 3, y + 4] for x, y in sq(100, 100)], "angle": 5.0}]
    s = score_image(im, ok, 1.0)
    c = s.cubes[0]
    assert (c.center_err, c.vertex_err, c.angle_err) == pytest.approx((5.0, 5.0, 5.0))
    assert c.passed and s.passed
    ok[0]["center"] = [103, 104.01]
    assert score_image(im, ok, 1.0).cubes[0].passed is False


def test_missing_quad_counts_centre_and_angle_but_fails_tolerance() -> None:
    """Each geometry error is collected from its own valid outputs: no quad -> out of the vertex
    sample only; the cube fails 허용 오차 충족률 because a required output is missing."""
    im = gimage([gcube("r", 100, 100), gcube("g", 300, 100, "green")])
    preds = perfect(im)
    preds[0]["quad"] = None
    s = score_image(im, preds, 7.0)
    assert s.cubes[0].vertex_err is None and s.cubes[0].center_err == 0 and s.cubes[0].passed is False
    assert s.cubes[1].passed is True and s.passed is False
    m = aggregate([s])
    assert (m.vertex_err.n, m.center_err.n, m.angle_err.n) == (1, 2, 2)
    assert (m.tolerance_rate.numerator, m.tolerance_rate.denominator) == (1, 2)
    assert m.recall.value == 1.0


def test_failed_run_misses_every_cube_and_has_no_time_sample() -> None:
    im1 = gimage([gcube("r", 100, 100), gcube("w", 300, 100, "wood", valid=False)], "s1")
    im2 = gimage([gcube("r", 100, 100)], "s2")
    failed = score_image(im1, None, 99.0)
    assert not failed.ran and failed.ms is None and failed.n_preds == 0 and not failed.passed
    assert all(not c.matched for c in failed.cubes)
    m = aggregate([failed])
    assert (m.recall.numerator, m.recall.denominator) == (0, 2)
    assert m.precision.value is None and m.color_accuracy.value is None  # 0 / 0 is undefined, not 0
    assert m.time_ms == Stats(0, None, None, None) and m.center_err.median is None
    m2 = aggregate([failed, score_image(im2, perfect(im2), 10.0)])
    assert (m2.n_failed, m2.time_ms.n, m2.time_ms.median) == (1, 1, 10.0)
    assert (m2.recall.numerator, m2.recall.denominator) == (1, 3)


def test_image_pass_needs_zero_false_positives() -> None:
    im = gimage([gcube("r", 100, 100), gcube("g", 300, 100, "green")])
    s = score_image(im, perfect(im) + [{"color": "red", "center": [900, 500]}], 1.0)
    assert all(c.passed for c in s.cubes) and len(s.false_pos) == 1 and not s.passed
    m = aggregate([s])
    assert (m.precision.numerator, m.precision.denominator) == (2, 3) and m.image_pass_rate.value == 0.0


def test_invalid_geometry_cube_needs_only_match_and_colour() -> None:
    """A truncated cube has no geometry: matched with the right colour, no quad or angle, it still lets
    the image pass and stays out of the 허용 오차 충족률 denominator."""
    im = gimage([gcube("r", 100, 100), gcube("w", 1270, 300, "wood", valid=False)])
    s = score_image(im, perfect(im), 1.0)
    assert s.cubes[1].passed is None and s.cubes[1].center_err is None and s.passed
    m = aggregate([s])
    assert m.tolerance_rate.denominator == 1 and m.n_valid == 1 and m.n_cubes == 2
    wrong = perfect(im)
    wrong[1]["color"] = "red"
    assert not score_image(im, wrong, 1.0).passed


def test_unknown_colour_is_wrong_and_shows_in_the_confusion_table() -> None:
    im = gimage([gcube("r", 100, 100, "red"), gcube("b", 300, 100, "blue")])
    preds = perfect(im)
    preds[0]["color"] = "unknown"
    preds[1]["color"] = "purple"  # not a model colour: read as unknown
    s = score_image(im, preds, 1.0)
    assert [c.color_ok for c in s.cubes] == [False, False] and not any(c.passed for c in s.cubes)
    m = aggregate([s])
    assert m.confusion["red"]["unknown"] == 1 and m.confusion["blue"]["unknown"] == 1
    assert set(m.confusion) == {"red", "green", "blue", "wood"}
    assert all(set(row) == {"red", "green", "blue", "wood", "unknown"} for row in m.confusion.values())
    assert m.color_accuracy.value == 0.0


def test_pass_rule_checks_outcomes_and_counts() -> None:
    """One source of the §3 pass rule: each check, geom_ok, passed, the outcome in words, the
    image's pass and its counts. A missing output fails its check; invalid geometry has no checks
    (None) and passes on match + colour; an unmatched cube is "missed", never a geometry failure."""
    im = gimage([gcube("ok", 100, 100), gcube("far", 300, 100), gcube("noquad", 500, 100),
                 gcube("blue", 700, 100, "blue"), gcube("lost", 900, 100), gcube("cut", 1100, 100, "wood", valid=False),
                 gcube("cutwrong", 1100, 400, "wood", valid=False)])
    preds = perfect(im)
    preds[1]["center"] = [306.0, 100.0]  # 6 px off: centre over τc, quad and angle right
    preds[2]["quad"] = None
    preds[3]["color"] = "green"
    preds[6]["color"] = "red"
    del preds[4]  # "lost" is not detected
    s = score_image(im, preds + [{"color": "red", "center": [40, 600]}], 1.0)
    got = [(c.center_ok, c.angle_ok, c.vertex_ok, c.geom_ok, c.passed, c.outcome) for c in s.cubes]
    assert got == [
        (True, True, True, True, True, "pass"),
        (False, True, True, False, False, "fail"),  # 6 px off: over τc only
        (True, True, False, False, False, "fail"),  # no quad: the vertex check fails
        (True, True, True, True, False, "fail"),  # geometry right, colour wrong
        (False, False, False, False, False, "missed"),  # no pair: no errors, every check fails
        (None, None, None, None, None, "pass"),  # invalid geometry: match + colour only
        (None, None, None, None, None, "fail"),
    ]
    assert not s.passed
    assert outcome_counts(s) == OutcomeCounts(n_gt=7, n_missed=1, n_fp=1, n_color_wrong=2, n_geom_fail=2)
    ok = score_image(im, perfect(im), 1.0)
    assert ok.passed and all(c.outcome == "pass" for c in ok.cubes)
    assert outcome_counts(ok) == OutcomeCounts(7, 0, 0, 0, 0)
    failed = score_image(im, None, None)
    assert outcome_counts(failed) == OutcomeCounts(7, 7, 0, 0, 0) and {c.outcome for c in failed.cubes} == {"missed"}


def test_cube_objects_are_read_like_dicts() -> None:
    im = gimage([gcube("r", 100, 100, angle=10.0)])
    q = sq(100, 100, angle=10.0)
    c = Cube("red", 0.9, (np.float32(100), np.float32(100)), [q[1], q[2], q[3], q[0]], 10.0, 3600.0)
    s = score_image(im, [c], 3.0)
    assert s.cubes[0].passed and s.ms == 3.0


# --- metrics -------------------------------------------------------------------------------
def test_quantiles_use_numpy_linear_interpolation() -> None:
    assert Stats.of([1, 2, 3, 4]) == Stats(4, 2.5, pytest.approx(3.85), 2.5)
    rng = np.random.default_rng(1)
    for n in (1, 2, 5, 37, 400):
        v = rng.exponential(3.0, n)
        s = Stats.of(v.tolist())
        assert s.median == np.quantile(v, 0.5) and s.p95 == np.quantile(v, 0.95) and quantile(v, 0.3) == np.quantile(v, 0.3)
    assert quantile([], 0.5) is None and Stats.of([]) == Stats(0, None, None, None)


def test_mixed_criteria_cannot_be_aggregated() -> None:
    im = gimage([gcube("r", 100, 100)])
    a = score_image(im, perfect(im), 1.0, Criteria())
    b = score_image(im, perfect(im), 1.0, Criteria(center_px=3))
    with pytest.raises(ValueError):
        aggregate([a, b])
    with pytest.raises(ValueError):
        Criteria(gate_px=-1)
    with pytest.raises(ValueError):
        Criteria(bootstrap_level=1.0)


def test_slices_by_variant_and_cube_attributes() -> None:
    images = row_images(2, "g0", "o") + row_images(2, "g0", "d", "dark", ["o0", "o1"])
    scores = [score_image(im, perfect(im), 5.0) for im in images]
    rows = slices(scores)
    variant = [(r.value, r.n) for r in rows if r.dimension == "variant"]
    assert variant == [("original", 2), ("dark", 2)]
    by = {(r.dimension, r.value): r for r in rows}
    assert by["region", "outer"].n == 8 and by["region", "middle"].n == 8  # the invalid cube's region counts
    assert by["near_contact", True].n == 4 and by["weak_boundary", True].n == 4
    assert [r.value for r in rows if r.dimension == "color"] == ["red", "green", "blue", "wood"]
    cube_row = by["image_side", "right"]
    assert cube_row.metrics.precision is None and cube_row.metrics.image_pass_rate is None
    assert cube_row.metrics.time_ms is None and cube_row.metrics.recall.value == 1.0
    assert cube_row.n == 8 and cube_row.metrics.tolerance_rate.denominator == 4  # wood has no geometry
    assert by["variant", "dark"].metrics.precision.value == 1.0


# --- bootstrap -----------------------------------------------------------------------------
def _random_batches(seed: int, groups: int = 5) -> tuple[list[ImageScore], list[ImageScore]]:
    rng = random.Random(seed)
    images = [im for g in range(groups) for im in row_images(rng.randint(1, 4), f"grp{g}", f"g{g}-")]
    a = [score_image(im, noisy(im, rng, 2.0), rng.uniform(5, 15)) for im in images]
    b = [score_image(im, noisy(im, rng, 3.0, 0.2), rng.uniform(5, 15)) for im in images]
    return a, b


def test_bootstrap_of_identical_batches_is_zero_and_unclear() -> None:
    a, _ = _random_batches(1)
    a2, _ = _random_batches(1)  # the same predictions scored again: equal values, other objects
    for x, y in ((a, a), (a, a2)):
        cmp = paired_bootstrap(x, y, Criteria(bootstrap_reps=300))
        for d in cmp.diffs:
            if d.verdict is not None:
                assert (d.delta, d.ci_low, d.ci_high, d.verdict) == (0.0, 0.0, 0.0, "unclear"), d


def test_bootstrap_is_deterministic_for_a_seed() -> None:
    a, b = _random_batches(2)
    c1 = paired_bootstrap(a, b, Criteria(bootstrap_reps=500, bootstrap_seed=4))
    c2 = paired_bootstrap(list(reversed(a)), list(reversed(b)), Criteria(bootstrap_reps=500, bootstrap_seed=4))
    c3 = paired_bootstrap(a, b, Criteria(bootstrap_reps=500, bootstrap_seed=5))
    assert c1 == c2  # image order does not matter either
    assert c1 != c3
    assert [d.key for d in c1.diffs] == list(BOOTSTRAP_KEYS) and c1.n_groups == 5


def test_bootstrap_resample_equals_scoring_the_resampled_images() -> None:
    """Regression check of the shortcut: the resampled values equal aggregate() on the drawn groups'
    images listed with their multiplicity. Both sum the same per-image counts (_image_counts), so
    the ratios agree by construction; the medians and P95s check the weighted-quantile search."""
    a, _ = _random_batches(3)
    names = sorted({s.image.layout_group for s in a})
    idx = [names.index(s.image.layout_group) for s in a]
    counts = _draw_counts(len(names), 40, 11)
    fast = _resampled(_group_stats(a, idx, len(names)), counts)
    for r in range(40):
        drawn = [s for g, name in enumerate(names) for _ in range(counts[r, g]) for s in a
                 if s.image.layout_group == name]
        m = aggregate(drawn)
        for key in BOOTSTRAP_KEYS:
            want = m.value(key)
            got = fast[key][r]
            assert (want is None and math.isnan(got)) or got == pytest.approx(want, rel=1e-12, abs=1e-12), (key, r)


def _grouped(n_groups: int, copies: int, own_group: bool) -> tuple[list[ImageScore], list[ImageScore]]:
    """A finds every cube; in group g, B misses g % 4 of the 4 cubes in every copy of the image."""
    a, b = [], []
    for g in range(n_groups):
        for k in range(copies):
            (im,) = row_images(1, f"g{g}-{k}" if own_group else f"g{g}", f"g{g}-{k}-")
            a.append(score_image(im, perfect(im), 10.0))
            b.append(score_image(im, perfect(im)[: 4 - g % 4], 10.0))
    return a, b


def test_bootstrap_resamples_layout_groups_not_images() -> None:
    """20 near-identical copies per group carry no more evidence than one: the interval is exactly the
    one-copy interval. Treating each copy as its own group would shrink it by about sqrt(20)."""
    crit = Criteria(bootstrap_reps=1000)
    one = paired_bootstrap(*_grouped(6, 1, False), crit).get("recall")
    many = paired_bootstrap(*_grouped(6, 20, False), crit).get("recall")
    naive = paired_bootstrap(*_grouped(6, 20, True), crit).get("recall")
    assert (many.ci_low, many.ci_high) == pytest.approx((one.ci_low, one.ci_high))
    assert many.verdict == "lower" and many.delta == pytest.approx(17 / 24 - 1)  # B finds 4+3+2+1+4+3 of 24
    assert (naive.ci_high - naive.ci_low) < 0.5 * (many.ci_high - many.ci_low)


def test_reused_bootstrap_side_gives_the_same_comparison() -> None:
    """A side made once (the reference, compared with many batches) and B's metrics passed in give
    exactly what a fresh paired_bootstrap gives; a side made for other images or another draw is refused."""
    a, b = _random_batches(6)
    c = [score_image(s.image, noisy(s.image, random.Random(i), 2.5), 9.0) for i, s in enumerate(a)]
    crit = Criteria(bootstrap_reps=500, bootstrap_seed=3, bootstrap_level=0.9)
    side = bootstrap_side(a, crit, aggregate(a))
    for x in (b, c):
        assert paired_bootstrap(a, x, crit, side_a=side, metrics_b=aggregate(x)) == paired_bootstrap(a, x, crit)
    with pytest.raises(ValueError, match="side_a"):
        paired_bootstrap(a, b, Criteria(bootstrap_reps=500, bootstrap_seed=4), side_a=side)
    with pytest.raises(ValueError, match="side_a"):
        paired_bootstrap(list(reversed(a)), b, crit, side_a=side)


def test_augmentation_effect_shares_the_originals_side_without_changing_results() -> None:
    """The variants of one set of originals share its side: each variant's effect is exactly what
    scoring that variant alone with its originals gives (where nothing can be shared)."""
    rng = random.Random(8)
    orig = [im for g in range(4) for im in row_images(2, f"grp{g}", f"o{g}-")]
    scores = [score_image(im, noisy(im, rng, 2.0, 0.1), 5.0) for im in orig]
    for v in ("dark", "warm", "blur"):
        ims = [im for g in range(4) for im in row_images(2, f"grp{g}", f"{v}{g}-", v, [f"o{g}-0", f"o{g}-1"])]
        scores += [score_image(im, noisy(im, rng, 3.0, 0.3), 5.0) for im in ims]
    crit = Criteria(bootstrap_reps=300, bootstrap_seed=2)
    effects = augmentation_effect(scores, crit)
    assert [e.variant for e in effects] == ["dark", "warm", "blur"]
    for e in effects:
        (alone,) = augmentation_effect(scores[:8] + [s for s in scores if s.image.variant == e.variant], crit)
        assert alone == e


def test_bootstrap_requires_the_same_images_and_thresholds() -> None:
    a, b = _random_batches(4)
    with pytest.raises(ValueError):
        paired_bootstrap(a, b[1:])
    other = [score_image(s.image, perfect(s.image), 1.0, Criteria(gate_px=20)) for s in a]
    with pytest.raises(ValueError):
        paired_bootstrap(a, other)
    with pytest.raises(ValueError):
        paired_bootstrap(a, b, Criteria(center_px=4))


def test_augmentation_effect_pairs_variants_with_their_originals() -> None:
    rng = random.Random(5)
    orig = [im for g in range(4) for im in row_images(2, f"grp{g}", f"o{g}-")]
    dark = [im for g in range(4) for im in row_images(2, f"grp{g}", f"d{g}-", "dark", [f"o{g}-0", f"o{g}-1"])]
    lonely = row_images(1, "grpX", "x", "warm", ["missing"])
    scores = ([score_image(im, perfect(im), 5.0) for im in orig]
              + [score_image(im, noisy(im, rng, 4.0, 0.4), 5.0) for im in dark]
              + [score_image(im, perfect(im), 5.0) for im in lonely])
    (eff,) = augmentation_effect(scores, Criteria(bootstrap_reps=400))
    assert (eff.variant, eff.n_sources, eff.n_unpaired, eff.comparison.n_groups) == ("dark", 8, 0, 4)
    want = aggregate(scores[8:16]).recall.value - aggregate(scores[:8]).recall.value
    assert eff.comparison.get("recall").delta == pytest.approx(want)


# --- CSV rows ------------------------------------------------------------------------------
def test_csv_rows_have_the_declared_columns() -> None:
    im = gimage([gcube("r", 100, 100), gcube("w", 1270, 300, "wood", valid=False)], "s9")
    s = score_image(im, perfect(im) + [{"color": "green", "center": [600, 400]}], 2.0)
    rows = cube_csv_rows("batch-1", s, complete=True)
    assert [r["row"] for r in rows] == ["gt", "gt", "fp"]
    assert all(tuple(r) == CUBE_CSV_FIELDS and r["status"] == "완료" for r in rows)
    assert CUBE_CSV_FIELDS[:3] == ("batch", "status", "image")
    assert {r["status"] for r in cube_csv_rows("batch-1", s, complete=False)} == {"중간"}
    assert rows[0]["passed"] == 1 and rows[1]["passed"] is None and rows[2]["pred_x"] == 600
    assert (rows[0]["tau_center_px"], rows[0]["gate_px"]) == (5.0, 29.0)
    summary = summary_csv_rows("batch-1", aggregate([s]), complete=False)
    assert [r["metric"] for r in summary] == [m.key for m in METRICS]
    assert all(tuple(r) == SUMMARY_CSV_FIELDS and r["status"] == "중간" for r in summary)
    prec = next(r for r in summary if r["metric"] == "precision")
    assert (prec["numerator"], prec["denominator"]) == (2, 3)
    data = csv_bytes(rows, CUBE_CSV_FIELDS)
    assert data.startswith("﻿".encode()) and data.decode("utf-8-sig").splitlines()[0] == ",".join(CUBE_CSV_FIELDS)


# --- parsing -------------------------------------------------------------------------------
def _doc(schema: str = SCHEMA_V11, **item: Any) -> dict[str, Any]:
    cube = {"id": "red_1", "color": "red", "valid_geometry": True, "center": [10, 10],
            "quad": [[0, 0], [20, 0], [20, 20], [0, 20]], "angle": 0.0, "detection_match_point": [10, 10],
            "region": "outer", "image_side": "left", "flags": []}
    base = {"media": "a/b.png", "sha256": "abc", "layout_group": "layout_00", "width": 1280, "height": 720,
            "near_top_face_pairs": [], "frames": [{"frame": 0, "cubes": [cube]}]}
    base.update(item)
    return {"metadata": {"schema_version": schema}, "items": [base]}


def test_parse_minimal_document_and_fallbacks() -> None:
    gt = parse_gt(_doc())
    (im,) = gt.images
    assert (im.filename, im.variant, im.source_sha256, im.media) == ("b.png", "original", "abc", "a/b.png")
    aug = parse_gt(_doc(SCHEMA_AUGMENTED, augmentation={"name": "warm"}, source_sha256="src", filename="f.png"))
    assert (aug.images[0].variant, aug.images[0].source_sha256, aug.images[0].filename) == ("warm", "src", "f.png")


@pytest.mark.parametrize("change, needle", [
    ({"schema": "robothon_top_face_gt_v9"}, "지원하지 않는"),
    ({"sha256": None}, "sha256"),
    ({"layout_group": ""}, "layout_group"),
    ({"frames": []}, "frames"),
    ({"near_top_face_pairs": [["red_1", "blue_9"]]}, "blue_9"),
    ({"cube": {"color": "purple"}}, "purple"),
    ({"cube": {"quad": [[0, 0], [1, 0], [1, 1]]}}, "quad"),
    ({"cube": {"angle": None}}, "angle"),
    ({"cube": {"detection_match_point": None}}, "detection_match_point"),
    ({"cube": {"valid_geometry": "yes"}}, "valid_geometry"),
    ({"width": -3}, "width"),
])
def test_parse_errors_name_the_problem(change: dict[str, Any], needle: str) -> None:
    change = dict(change)
    doc = _doc(change.pop("schema", SCHEMA_V11))
    item = doc["items"][0]
    item["frames"][0]["cubes"][0].update(change.pop("cube", {}))
    item.update(change)
    with pytest.raises(ValueError, match=needle):
        parse_gt(doc)


def test_parse_rejects_duplicate_images_and_cube_ids() -> None:
    doc = _doc()
    doc["items"].append(json.loads(json.dumps(doc["items"][0])))
    with pytest.raises(ValueError, match="sha256"):
        parse_gt(doc)
    doc = _doc()
    cubes = doc["items"][0]["frames"][0]["cubes"]
    cubes.append(dict(cubes[0]))
    with pytest.raises(ValueError, match="id"):
        parse_gt(doc)


def test_invalid_geometry_cube_drops_geometry_and_uses_approximate_region() -> None:
    doc = _doc()
    doc["items"][0]["frames"][0]["cubes"][0].update(valid_geometry=False, center=None, quad=None, angle=None,
                                                    region=None, approximate_region="middle")
    c = parse_gt(doc).images[0].cubes[0]
    assert (c.center, c.quad, c.angle, c.region, c.match_point) == (None, None, None, "middle", (10.0, 10.0))


# --- the real ground-truth files -----------------------------------------------------------
@pytest.fixture(scope="module")
def raw41() -> dict[str, Any]:
    return json.loads(GT41.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def raw410() -> dict[str, Any]:
    return json.loads(GT410.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def gt41(raw41: dict[str, Any]) -> GtSet:
    return parse_gt(raw41)


@pytest.fixture(scope="module")
def gt410(raw410: dict[str, Any]) -> GtSet:
    return parse_gt(raw410)


def test_real_41_counts(gt41: GtSet) -> None:
    assert gt41.schema_version == SCHEMA_V11
    assert (gt41.n_images, gt41.n_cubes, gt41.n_valid_geometry) == (41, 328, 318)
    assert gt41.variants == ("original",) and all(im.source_sha256 == im.sha256 for im in gt41.images)
    assert len(gt41.layout_groups) == 35
    assert sum(c.weak_boundary for im in gt41.images for c in im.cubes) == 53


def test_real_410_counts_and_provenance(gt410: GtSet, gt41: GtSet) -> None:
    assert gt410.schema_version == SCHEMA_AUGMENTED
    assert (gt410.n_images, gt410.n_cubes, gt410.n_valid_geometry) == (410, 3280, 3180)
    assert len(gt410.layout_groups) == 35
    assert len(gt410.variants) == 10 and gt410.variants[0] == "original"
    assert all(sum(im.variant == v for im in gt410.images) == 41 for v in gt410.variants)
    originals = {im.sha256 for im in gt410.images if im.variant == "original"}
    assert originals == {im.sha256 for im in gt41.images}
    assert all(im.source_sha256 in originals for im in gt410.images)
    assert sum(c.weak_boundary for im in gt410.images for c in im.cubes) == 530


def test_real_files_take_invalid_regions_from_approximate_region(gt410: GtSet, raw410: dict[str, Any]) -> None:
    invalid = 0
    for im, item in zip(gt410.images, raw410["items"]):
        raw = {c["id"]: c for c in item["frames"][0]["cubes"]}
        near = {i for p in item["near_top_face_pairs"] for i in p}
        for c in im.cubes:
            assert c.region is not None and c.near_contact == (c.id in near)
            if not c.valid_geometry:
                invalid += 1
                assert "region" not in raw[c.id] and c.region == raw[c.id]["approximate_region"]
                assert c.center is None and c.quad is None and c.angle is None
    assert invalid == 100


@pytest.mark.parametrize("which", ["41", "410"])
def test_real_ground_truth_scored_against_itself_is_perfect(which: str, gt41: GtSet, gt410: GtSet) -> None:
    gt = gt41 if which == "41" else gt410
    scores = [score_image(im, perfect(im), 10.0) for im in gt.images]
    m = aggregate(scores)
    assert (m.recall.value, m.precision.value, m.color_accuracy.value) == (1.0, 1.0, 1.0)
    assert (m.tolerance_rate.numerator, m.tolerance_rate.denominator) == (gt.n_valid_geometry,) * 2
    assert m.image_pass_rate.value == 1.0 and m.n_failed == 0
    assert (m.vertex_err.n, m.vertex_err.p95, m.center_err.median) == (gt.n_valid_geometry, 0.0, 0.0)
    if which == "410":
        rows = slices(scores)
        assert sum(r.n for r in rows if r.dimension == "region") == 3280
        assert [r.n for r in rows if r.dimension == "variant"] == [41] * 10
        effects = augmentation_effect(scores, Criteria(bootstrap_reps=200))
        assert len(effects) == 9 and all(e.n_sources == 41 and e.comparison.n_groups == 35 for e in effects)
        assert all(d.delta == 0 and d.verdict == "unclear" for e in effects for d in e.comparison.diffs)

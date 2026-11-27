"""Seeded synthetic benchmark of the built-in pipelines.

    cd backend && uv run python -m cubevision.bench            # all pipelines, 24 scenes
    uv run python -m cubevision.bench --scenes 48 --pipelines cv.hsv_ranges

SYNTHETIC DATA ONLY (합성 데이터 기준): it shows that the plumbing and the geometry work and how
the pipelines compare under imitated light and noise. Real-photo accuracy must be checked in the
lab with real captures.

Matching: a detection matches the nearest unmatched ground-truth cube within half a face
(30 px). Cubes whose top face is less than 60 % visible (under the arm) are optional: finding
them is not required and does not count as a false positive.
"""

from __future__ import annotations

import argparse
import math
import statistics
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from cubevision.pipelines.base import Pipeline
from cubevision.postprocess import FACE_PX, angle_diff
from cubevision.registry import get_pipeline, list_pipelines
from cubevision.synth import make_scene

BENCH_SEED0 = 1000
MATCH_PX = 0.5 * FACE_PX
OPTIONAL_BELOW = 0.6


@dataclass
class Scene:
    seed: int
    mission: str
    image: np.ndarray
    gt: list[dict[str, Any]]


def hard_light(seed: int) -> dict[str, Any]:
    """Harsher light than the default scenes: a strong colour cast (up to 18 % per channel), a
    steep light gradient, darker shadows, more sensor noise, and an exposure between 60 % and
    100 % of the brightest that keeps the board unclipped (an auto-exposure that protects the
    highlights). Clipped frames are a capture problem: the pipelines warn about them instead."""
    rng = np.random.default_rng(seed + 50_000)
    wb = rng.uniform(0.82, 1.18, 3)
    light = float(rng.uniform(0.15, 0.25))
    no_clip = 245.0 / (228.0 * float(wb.max()) * (1.0 + light))
    return {
        "wb": tuple(wb),
        "exposure": float(rng.uniform(0.6, 1.0)) * no_clip,
        "light": light,
        "noise": float(rng.uniform(3.5, 6.0)),
        "shadow_strength": float(rng.uniform(0.3, 0.5)),
    }


def bench_scenes(n: int = 24, seed0: int = BENCH_SEED0, *, hard: bool = False) -> list[Scene]:
    """The fixed benchmark: alternating missions A and B, every variation drawn from the seed.
    ``hard`` adds ``hard_light`` to every scene (same layouts)."""
    out = []
    for i in range(n):
        seed, mission = seed0 + i, ("A" if i % 2 == 0 else "B")
        extra = hard_light(seed) if hard else {}
        img, gt = make_scene(seed, mission=mission, **extra)  # type: ignore[arg-type]
        out.append(Scene(seed, mission, img, gt))
    return out


def match(cubes: list[Any], gt: list[dict[str, Any]], max_px: float = MATCH_PX) -> list[tuple[int, int]]:
    """Greedy nearest-centre matching -> list of (detection index, ground-truth index)."""
    pairs = []
    for i, c in enumerate(cubes):
        for j, g in enumerate(gt):
            d = math.hypot(c.center[0] - g["center"][0], c.center[1] - g["center"][1])
            if d <= max_px:
                pairs.append((d, i, j))
    pairs.sort()
    used_i: set[int] = set()
    used_j: set[int] = set()
    out = []
    for _, i, j in pairs:
        if i not in used_i and j not in used_j:
            used_i.add(i)
            used_j.add(j)
            out.append((i, j))
    return out


def evaluate(pipeline: Pipeline, scenes: list[Scene], params: dict[str, Any] | None = None) -> dict[str, float]:
    """Run ``pipeline`` on every scene and aggregate the metrics."""
    required = found = colour_ok = fp = 0
    centre_err: list[float] = []
    angle_err: list[float] = []
    times: list[float] = []
    pipeline(scenes[0].image, params)  # warm-up (first call pays for allocations and caches)
    for sc in scenes:
        t = time.perf_counter()
        res = pipeline(sc.image, params)
        times.append((time.perf_counter() - t) * 1000.0)
        pairs = match(res.cubes, sc.gt)
        matched_det = {i for i, _ in pairs}
        fp += sum(1 for i in range(len(res.cubes)) if i not in matched_det)
        req = [j for j, g in enumerate(sc.gt) if g["visible"] >= OPTIONAL_BELOW]
        required += len(req)
        for i, j in pairs:
            if j not in req:
                continue
            c, g = res.cubes[i], sc.gt[j]
            found += 1
            colour_ok += int(c.color == g["color"])
            centre_err.append(math.hypot(c.center[0] - g["center"][0], c.center[1] - g["center"][1]))
            if c.angle is not None:
                angle_err.append(abs(angle_diff(c.angle, g["angle"])))
    return {
        "scenes": len(scenes),
        "required": required,
        "recall": found / max(required, 1),
        "fp_per_image": fp / max(len(scenes), 1),
        "colour_acc": colour_ok / max(found, 1),
        "centre_mean": statistics.fmean(centre_err) if centre_err else float("nan"),
        "centre_max": max(centre_err) if centre_err else float("nan"),
        "angle_mean": statistics.fmean(angle_err) if angle_err else float("nan"),
        "angle_max": max(angle_err) if angle_err else float("nan"),
        "p50_ms": statistics.median(times),
        "max_ms": max(times),
    }


def table(rows: list[tuple[Pipeline, dict[str, float]]]) -> str:
    head = (
        "| 파이프라인 | 재현율 | 오검출/장 | 색 정확도 | 중심 오차 px (평균 / 최대) | 각도 오차 ° (평균 / 최대) | p50 ms |\n"
        "|---|---|---|---|---|---|---|"
    )
    lines = [head]
    for p, m in rows:
        lines.append(
            f"| `{p.key}` {p.name} | {m['recall'] * 100:.1f} % | {m['fp_per_image']:.2f} | {m['colour_acc'] * 100:.1f} % "
            f"| {m['centre_mean']:.2f} / {m['centre_max']:.2f} | {m['angle_mean']:.2f} / {m['angle_max']:.2f} | {m['p50_ms']:.1f} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Seeded synthetic benchmark (합성 데이터 기준)")
    ap.add_argument("--scenes", type=int, default=24)
    ap.add_argument("--seed0", type=int, default=BENCH_SEED0)
    ap.add_argument("--pipelines", type=str, default="", help="comma-separated keys (default: all)")
    ap.add_argument("--hard", action="store_true", help="only the hard-light variant of the scenes")
    ap.add_argument("--both", action="store_true", help="default scenes, then the hard-light variant")
    args = ap.parse_args(argv)
    keys = [k for k in args.pipelines.split(",") if k]
    pipes = [get_pipeline(k) for k in keys] if keys else list_pipelines()
    variants = [False, True] if args.both else [args.hard]
    for hard in variants:
        t = time.perf_counter()
        scenes = bench_scenes(args.scenes, args.seed0, hard=hard)
        n_gt = sum(len(s.gt) for s in scenes)
        title = "hard light (strong cast, exposure, noise)" if hard else "default scenes"
        print(f"\n## {title}: {len(scenes)} synthetic scenes ({n_gt} cubes in view), rendered in {time.perf_counter() - t:.1f} s\n")
        rows = [(p, evaluate(p, scenes)) for p in pipes]
        print(table(rows))
    print("\n합성 데이터 기준 — 실제 사진 성능은 랩에서 확인 필요")


if __name__ == "__main__":
    main()

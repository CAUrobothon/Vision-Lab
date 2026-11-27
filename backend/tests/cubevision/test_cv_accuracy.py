"""Accuracy of every pipeline on the fixed seeded synthetic benchmark (first 16 scenes).

Thresholds, and why: measured on 2026-11-25 (M4) every pipeline found 99-100 % of the required
cubes with no false positive, 100 % correct colours, a mean centre error of 0.4-0.9 px and a
mean angle error under 0.3 degrees. The limits below leave room for small, deliberate changes
but catch a real regression: losing one cube in twenty, a false cube every fifth image, a
wrong colour in thirty, or a centre bias of 1.5 px (0.6 mm, about the anchoring error the design
study predicts with 2 px of boundary noise). SYNTHETIC ONLY: real photos must be checked in the lab.
"""

from __future__ import annotations

import pytest

from cubevision.bench import evaluate
from cubevision.pipelines.base import Pipeline
from cubevision.registry import list_pipelines


@pytest.mark.parametrize("p", list_pipelines(), ids=lambda p: p.key)
def test_accuracy_on_synthetic_benchmark(p: Pipeline, bench16) -> None:
    m = evaluate(p, bench16)
    assert m["required"] > 100
    assert m["recall"] >= 0.95, m
    assert m["fp_per_image"] <= 0.2, m
    assert m["colour_acc"] >= 0.97, m
    assert m["centre_mean"] <= 1.5, m
    assert m["angle_mean"] <= 1.0, m
    assert m["p50_ms"] < 100.0, m  # loose: the 30 ms target is checked with cubevision.bench

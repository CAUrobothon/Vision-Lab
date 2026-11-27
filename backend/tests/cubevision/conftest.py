"""Shared fixtures for the cubevision tests (all offline, all synthetic)."""

from __future__ import annotations

import numpy as np
import pytest

from cubevision.bench import Scene, bench_scenes
from cubevision.synth import make_scene


@pytest.fixture(scope="session")
def bench16() -> list[Scene]:
    """The first 16 scenes of the fixed benchmark (8 mission A, 8 mission B)."""
    return bench_scenes(16)


@pytest.fixture(scope="session")
def scene_a() -> tuple[np.ndarray, list[dict]]:
    return make_scene(3, mission="A", arm=False)


@pytest.fixture(scope="session")
def scene_b() -> tuple[np.ndarray, list[dict]]:
    return make_scene(4, mission="B", arm=False)

"""Every parameter a pipeline shows is read by the pipeline, and no single value inside a
parameter's own range crashes it. (The lab shows every tunable; a shown knob that does nothing,
or one that breaks the run at the end of its slider, would be worse than a hidden constant.)"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from cubevision.pipelines.base import Pipeline
from cubevision.registry import get_pipeline, list_pipelines
from cubevision.synth import make_scene
from cubevision.types import ParamSpec

KEYS = [p["key"] if isinstance(p, dict) else p.key for p in list_pipelines()]
# Choices that switch a code path, so the params of every path get read.
MODES: list[dict[str, Any]] = [
    {},
    {"pp_mode": "manual", "denoise": "gauss", "init": "prototypes"},
    {"denoise": "none"},
]


class Tracking(dict):
    """A params dict that remembers which keys were read."""

    def __init__(self, *a: Any) -> None:
        super().__init__(*a)
        self.read: set[str] = set()

    def __getitem__(self, k: str) -> Any:
        self.read.add(k)
        return super().__getitem__(k)

    def get(self, k: str, default: Any = None) -> Any:
        self.read.add(k)
        return super().get(k, default)


@pytest.fixture(scope="module")
def scenes() -> list[np.ndarray]:
    # Touching groups, a tower, the arm and the frame border: the paths that read the
    # split, top-face and partly-hidden parameters.
    plain = [make_scene(s, mission=m, size=(640, 360))[0] for s, m in ((11, "A"), (12, "B"), (13, "A"))]
    return plain + [make_scene(s, mission="A", size=(640, 360), touching=True, tower=0)[0] for s in (14, 15)]


def _run(pipe: Pipeline, img: np.ndarray, given: dict[str, Any]) -> Tracking:
    p = Tracking(pipe.resolve({"face_px": 30.5, **given}))
    pipe.run(img, p, debug=True)
    return p


@pytest.mark.parametrize("key", KEYS)
def test_every_param_is_read(key: str, scenes: list[np.ndarray]) -> None:
    pipe = get_pipeline(key)
    read: set[str] = set()
    for mode in MODES:
        for img in scenes:
            read |= _run(pipe, img, mode).read
    unused = [s.name for s in pipe.params if s.name not in read]
    assert not unused, f"{key} shows params it never reads: {unused}"


def _extremes(s: ParamSpec) -> list[Any]:
    if s.type in ("int", "float") and s.min is not None and s.max is not None:
        return [s.min, s.max]
    if s.type == "range" and s.min is not None and s.max is not None:
        return [[s.min, s.max], [s.min, s.min], [s.max, s.max]]
    if s.type == "bool":
        return [not s.default]
    if s.type == "select":
        return [o[0] for o in (s.options or ()) if o[0] != s.default]
    return []


@pytest.mark.parametrize("key", KEYS)
def test_no_value_in_range_crashes(key: str, scenes: list[np.ndarray]) -> None:
    pipe = get_pipeline(key)
    img = scenes[0]
    tried = 0
    for s in pipe.params:
        extra = {"pp_mode": "manual"} if s.name in ("pp_x", "pp_y") else {}
        if s.name in ("bilateral_d", "sigma_color", "sigma_space"):
            extra = {"denoise": "bilateral"}
        elif s.name in ("gauss_ksize", "gauss_sigma"):
            extra = {"denoise": "gauss"}
        elif s.name == "dark_start_L":
            extra = {"init": "prototypes"}
        for v in _extremes(s):
            try:
                r = pipe(img, {"face_px": 30.5, **extra, s.name: v})
            except Exception as exc:  # noqa: BLE001 - report which value broke it
                pytest.fail(f"{key}: {s.name} = {v!r} raised {type(exc).__name__}: {exc}")
            assert all(0.0 <= c.score <= 1.0 for c in r.cubes)
            tried += 1
    assert tried > 50

"""Contract tests for every registered pipeline: what the lab and the competition code rely on."""

from __future__ import annotations

import json
import re

import numpy as np
import pytest

from cubevision.bench import match
from cubevision.colour import DEFAULT_PROTOTYPES
from cubevision.pipelines.base import Pipeline
from cubevision.registry import get_pipeline, list_pipelines
from cubevision.roi import point_in_roi, roi_mask
from cubevision.synth import CLEAN, make_scene
from cubevision.types import COLORS, FrameResult

PIPELINES = list_pipelines()
HANGUL = re.compile(r"[가-힣]")


def test_registry_has_four_or_more_unique_pipelines() -> None:
    keys = [p.key for p in PIPELINES]
    assert len(keys) >= 4
    assert len(set(keys)) == len(keys)
    assert get_pipeline(keys[0]) is PIPELINES[0]
    with pytest.raises(KeyError):
        get_pipeline("cv.does_not_exist")


@pytest.mark.parametrize("p", PIPELINES, ids=lambda p: p.key)
class TestPipelineContract:
    def test_info_shape(self, p: Pipeline) -> None:
        info = p.info()
        assert set(info) == {"key", "kind", "name", "summary", "description", "params", "debug_labels"}
        assert info["kind"] == "pipeline"
        assert re.fullmatch(r"cv\.[a-z0-9_]+", info["key"])
        for field in ("name", "summary", "description"):
            assert HANGUL.search(info[field]), field
        assert 3 <= info["description"].count("\n") + 1 <= 5
        json.dumps(info, ensure_ascii=False)  # the lab serves it as JSON

    def test_params(self, p: Pipeline) -> None:
        specs = {s.name: s for s in p.params}
        assert len(specs) == len(p.params), "duplicate parameter names"
        for s in p.params:
            assert HANGUL.search(s.label), s.name
            assert s.help, s.name
        assert specs["roi"].type == "roi" and specs["roi"].group == "검출 영역"
        assert specs["face_px"].type == "float" and specs["face_px"].default == pytest.approx(61.0)
        for k, hex_ in DEFAULT_PROTOTYPES.items():
            assert specs[f"proto_{k}"].type == "color"
            assert specs[f"proto_{k}"].default == hex_

    def test_resolve_defaults_and_clamps(self, p: Pipeline) -> None:
        resolved = p.resolve(None)
        assert set(resolved) == {s.name for s in p.params}
        assert resolved["face_px"] == pytest.approx(61.0)
        assert p.resolve({"face_px": 10_000})["face_px"] == 400
        assert p.resolve({"face_px": 1})["face_px"] == 10
        # A 4032x3024 phone photo of the same scene needs about 190 px: not clamped any more.
        assert p.resolve({"face_px": 190})["face_px"] == 190
        assert "unknown_key" not in p.resolve({"unknown_key": 1})

    def test_face_px_below_40_on_a_small_frame(self, p: Pipeline) -> None:
        """At 640x360 a top face is about 31 px; the old 40 px floor made every cube partial."""
        img, gt = make_scene(5, mission="A", size=(640, 360), **{**CLEAN, "tower": 0, "touching": False})
        face = float(np.median([g["side_px"] for g in gt]))
        res = p(img, {"face_px": face})
        # Snapped to the slider's 0.1 px grid, never clamped to a floor.
        assert p.resolve({"face_px": face})["face_px"] == pytest.approx(face, abs=0.05)
        assert len(match(res.cubes, gt, 3.0)) == len(gt) == len(res.cubes)
        assert not any("partial" in c.flags for c in res.cubes)

    def test_debug_labels(self, p: Pipeline) -> None:
        assert 2 <= len(p.debug_labels) <= 4
        assert all(HANGUL.search(v) for v in p.debug_labels.values())

    @pytest.mark.parametrize("fill", [0, 128, 230])
    def test_blank_frames_give_no_cubes(self, p: Pipeline, fill: int) -> None:
        frame = np.full((720, 1280, 3), fill, np.uint8)
        res = p(frame)
        assert isinstance(res, FrameResult)
        assert res.cubes == []
        assert res.ms > 0

    def test_tiny_and_noise_frames_do_not_crash(self, p: Pipeline) -> None:
        rng = np.random.default_rng(0)
        p(rng.integers(0, 256, (48, 64, 3), dtype=np.uint8))
        p(rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8))

    @pytest.mark.parametrize("shape", [(1, 1), (2, 2), (1, 640), (640, 1)])
    def test_frames_one_or_two_pixels_thin_give_no_cubes(self, p: Pipeline, shape: tuple[int, int]) -> None:
        # cv.lab_autocluster used to crash here: shrinking to 1/4 left an empty image.
        assert p(np.full((*shape, 3), 200, np.uint8), debug=True).cubes == []

    def test_rejects_non_bgr_input(self, p: Pipeline) -> None:
        with pytest.raises(ValueError):
            p(np.zeros((10, 10), np.uint8))

    def test_scene_result(self, p: Pipeline, scene_a: tuple) -> None:
        img, gt = scene_a
        res = p(img)
        assert len(match(res.cubes, gt)) >= 0.8 * len(gt)
        for c in res.cubes:
            assert c.color in COLORS
            assert 0.0 <= c.score <= 1.0 and 0.0 <= c.color_score <= 1.0
            if c.angle is not None:
                assert 0.0 <= c.angle < 90.0
            if c.quad is not None:
                assert len(c.quad) == 4
        assert res.stages_ms and all(v >= 0 for v in res.stages_ms.values())
        json.dumps(res.to_dict(), ensure_ascii=False)

    def test_deterministic(self, p: Pipeline, scene_b: tuple) -> None:
        img, _ = scene_b
        a, b = p(img).to_dict(), p(img).to_dict()
        assert a["cubes"] == b["cubes"]

    def test_debug_layers(self, p: Pipeline, scene_b: tuple) -> None:
        img, _ = scene_b
        res = p(img, debug=True)
        assert set(res.debug) == set(p.debug_labels)
        for name, layer in res.debug.items():
            assert layer.dtype == np.uint8, name
            assert layer.shape[:2] == img.shape[:2], name
            assert layer.ndim == 2 or (layer.ndim == 3 and layer.shape[2] == 3), name
            if layer.ndim == 2:  # a mask: the UI tints it, so it must be 0/255 only
                assert set(np.unique(layer).tolist()) <= {0, 255}, name
            else:  # a picture: shown as it is, so it must carry the frame (not black)
                assert layer.mean() > 20, name
        assert p(img).debug == {}  # layers only when asked for

    def test_roi_include_and_exclude(self, p: Pipeline, scene_a: tuple) -> None:
        img, gt = scene_a
        left = [{"mode": "include", "points": [[0, 0], [640, 0], [640, 720], [0, 720]]}]
        res = p(img, {"roi": left})
        assert res.cubes and all(c.center[0] <= 640 for c in res.cubes)

        target = gt[0]
        x, y = target["center"]
        box = [[x - 40, y - 40], [x + 40, y - 40], [x + 40, y + 40], [x - 40, y + 40]]
        res = p(img, {"roi": [{"mode": "exclude", "points": box}]})
        assert not match(res.cubes, [target])
        assert match(res.cubes, gt[1:])

    def test_roi_overlapping_includes_keep_the_overlap(self, p: Pipeline, scene_a: tuple) -> None:
        img, gt = scene_a
        x, y = gt[0]["center"]
        a = [[x - 150, y - 150], [x + 60, y - 150], [x + 60, y + 60], [x - 150, y + 60]]
        b = [[x - 60, y - 60], [x + 150, y - 60], [x + 150, y + 150], [x - 60, y + 150]]
        res = p(img, {"roi": [{"mode": "include", "points": a}, {"mode": "include", "points": b}]})
        assert match(res.cubes, [gt[0]])  # the cube sits where the two overlap


def test_roi_mask_is_the_union_point_in_roi_uses() -> None:
    """Polygons of one mode add up, as point_in_roi and the docstring say. Filling them in one
    fillPoly call used even-odd filling: where two overlapped (or one sat inside another) they
    cancelled out, so the mask had a hole that the point test did not."""
    def rect(x0: int, y0: int, x1: int, y1: int) -> list[list[int]]:
        return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]

    overlapping = [rect(100, 100, 400, 400), rect(300, 300, 600, 600)]
    nested = [rect(100, 100, 600, 600), rect(200, 200, 300, 300)]
    for mode, polys, probe, inside in (
        ("include", overlapping, (350, 350), True),
        ("include", nested, (250, 250), True),
        ("exclude", overlapping, (350, 350), False),
        ("exclude", nested, (250, 250), False),
    ):
        roi = [{"mode": mode, "points": pts} for pts in polys]
        mask = roi_mask((720, 1280), roi)
        assert mask is not None
        assert bool(mask[probe[1], probe[0]]) == inside, (mode, polys)
        for yy in range(60, 700, 25):  # never on a polygon edge (those sit on multiples of 100)
            for xx in range(60, 700, 25):
                assert (mask[yy, xx] > 0) == point_in_roi((xx, yy), roi), (mode, xx, yy)


@pytest.mark.parametrize("p", PIPELINES, ids=lambda p: p.key)
def test_grey_objects_under_a_colour_cast_are_not_cubes(p: Pipeline) -> None:
    """A dark grey patch (arm, table) next to a cube, under a strong cast and with noise, must not
    become a wood cube; the real cube must still be found."""
    img, gt = make_scene(0, cubes=[dict(color="red", x=-150.0, y=150.0, angle=20.0)], **{**CLEAN, "wb": (1.15, 0.95, 0.85), "noise": 4.0})
    x, y = (int(v) for v in gt[0]["center"])
    img = img.copy()
    img[y + 60 : y + 140, x - 40 : x + 60] = (np.array([70, 72, 80]) * np.array([1.15, 0.95, 0.85])).astype(np.uint8)
    res = p(img)
    assert [c.color for c in res.cubes] == ["red"]

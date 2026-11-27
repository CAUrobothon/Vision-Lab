"""cubevision result -> Scene. The first tests use cubevision-shaped data (dicts and objects) and
need nothing installed; the last one runs a real cubevision pipeline when cubevision is importable
(run with PYTHONPATH=../backend and the backend venv, see README)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vision_interface import Color, CubevisionSource, Scene, VisionSource, scene_from_cubevision

# Exactly what cubevision.types.FrameResult.to_dict() produces (values rounded as it rounds them).
FRAME_DICT = {
    "cubes": [
        {"color": "red", "score": 0.9731, "color_score": 0.88, "center": [412.5, 300.25],
         "quad": [[380.1, 270.0], [444.0, 268.9], [445.2, 331.0], [381.0, 332.4]], "angle": 89.996,
         "area": 3901.5, "polygon": [[380.0, 270.0]], "box": [380.0, 268.0, 446.0, 333.0], "label": None,
         "flags": ["side_band"]},
        {"color": "wood", "score": 0.61, "color_score": 0.55, "center": [900.0, 500.0], "quad": None,
         "angle": None, "area": 2100.0, "polygon": None, "box": None, "label": None, "flags": ["partial"]},
        {"color": "blue", "score": 0.95, "color_score": 1.0, "center": [100.0, 100.0],
         "quad": [[70, 70], [130, 70], [130, 130], [70, 130]], "angle": 90.0, "area": 3600.0,
         "polygon": None, "box": None, "label": "blue", "flags": []},
    ],
    "ms": 12.3, "stages_ms": {}, "notes": ["low light"],
}


def test_dict_result_maps_every_field() -> None:
    s = scene_from_cubevision(FRAME_DICT, frame_id=3, timestamp_s=10.0, image_size_px=(1280, 720),
                              mission="B", source="cubevision:cv.hsv_ranges")
    assert s.localized_by is None and all(c.pose is None for c in s.cubes)
    red, wood, blue = s.cubes
    assert red.color is Color.RED and red.center_px == (412.5, 300.25)
    assert red.corners_px[0] == (380.1, 270.0) and len(red.corners_px) == 4
    assert red.image_angle_deg == pytest.approx(89.996) and red.area_px2 == 3901.5
    assert red.score == 0.9731 and red.color_score == 0.88 and red.flags == ("side_band",)
    assert wood.color is Color.WOOD and wood.corners_px is None and wood.image_angle_deg is None
    assert wood.flags == ("partial",)
    assert blue.image_angle_deg == 0.0  # to_dict() may round 89.996 up to 90.0; 90 == 0 for a square
    assert s.notes == ("low light",) and s.source == "cubevision:cv.hsv_ranges"
    assert Scene.from_json(s.to_json()) == s


def test_object_result_and_plain_list() -> None:
    cubes = [SimpleNamespace(**{k: v for k, v in c.items()}) for c in FRAME_DICT["cubes"]]
    obj = SimpleNamespace(cubes=cubes, notes=[], ms=1.0)
    a = scene_from_cubevision(obj, frame_id=1, timestamp_s=0.0, image_size_px=(1280, 720))
    b = scene_from_cubevision(FRAME_DICT["cubes"], frame_id=1, timestamp_s=0.0, image_size_px=(1280, 720))
    assert a.cubes == b.cubes and len(a.cubes) == 3


def test_cubevision_source_runs_grab_then_detect() -> None:
    calls = []

    def grab():
        calls.append("grab")
        return SimpleNamespace(shape=(720, 1280, 3))

    def detect(frame):
        calls.append("detect")
        return FRAME_DICT

    src = CubevisionSource(grab, detect, mission="A", source="test")
    assert isinstance(src, VisionSource)
    s1, s2 = src.get_scene(), src.get_scene()
    assert calls == ["grab", "detect", "grab", "detect"]
    assert (s1.frame_id, s2.frame_id) == (1, 2) and s1.image_size_px == (1280, 720)
    assert s1.localized_by is None and len(s1.cubes) == 3


def test_real_cubevision_pipeline_on_a_synthetic_frame() -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    pytest.importorskip("cubevision")
    from cubevision.registry import get_pipeline
    from cubevision.synth import make_scene

    frame, truth = make_scene(4, mission="B", arm=False)
    result = get_pipeline("cv.hsv_ranges")(frame)
    h, w = frame.shape[:2]
    live = scene_from_cubevision(result, frame_id=1, timestamp_s=0.0, image_size_px=(w, h), mission="B")
    via_json = scene_from_cubevision(result.to_dict(), frame_id=1, timestamp_s=0.0, image_size_px=(w, h), mission="B")
    assert len(live.cubes) == len(result.cubes) >= 10
    assert [c.color for c in live.cubes] == [Color(c.color) for c in result.cubes]
    for a, b in zip(live.cubes, via_json.cubes):  # live objects and their rounded dicts agree
        assert a.color is b.color and abs(a.center_px[0] - b.center_px[0]) < 0.01
    assert live.localized_by is None and Scene.from_json(live.to_json()) == live

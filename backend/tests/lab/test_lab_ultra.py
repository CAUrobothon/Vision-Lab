"""The Ultralytics adapter's own logic, without torch: class-name colours, colour source,
task-specific conversion and the ROI filter, driven by fake Ultralytics results."""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pytest

from cubevision.types import resolve_params

from lab import ultra
from lab.ultra import UltraModel, colour_from_class


@pytest.mark.parametrize("name, colour", [
    ("red", "red"), ("Red_cube", "red"), ("RED", "red"), ("r", "red"), ("R-1", "red"), ("빨강", "red"),
    ("green", "green"), ("g", "green"), ("초록 큐브", "green"),
    ("blue", "blue"), ("B", "blue"), ("파랑", "blue"),
    ("wood", "wood"), ("w", "wood"), ("나무", "wood"),
    ("box", None), ("background", None), ("white", None), ("robot", None), ("cube", None), ("", None), (None, None),
])
def test_colour_from_class(name: str | None, colour: str | None) -> None:
    assert colour_from_class(name) == colour


# Every Ultralytics 8.4.215 predict argument that changes the result or the speed of one image.
PREDICT_ARGS = ["conf", "iou", "imgsz", "imgsz_h", "rect", "max_det", "classes", "agnostic_nms", "augment", "nms",
                "tile", "tile_px", "batch", "device", "quantize", "compile", "channels_last"]


def test_param_specs_follow_task_and_train_size() -> None:
    names = [p.name for p in ultra.param_specs("segment", 960, ["red", "green"])]
    assert names[: len(PREDICT_ARGS) + 1] == PREDICT_ARGS + ["retina_masks"]
    assert names[len(PREDICT_ARGS) + 1 : len(PREDICT_ARGS) + 7] == [
        "color_source", "proto_red", "proto_green", "proto_blue", "proto_wood", "proto_board"]
    assert {"adapt_light", "chroma_min", "max_dist", "w_lightness", "gain_range", "shrink"} <= set(names)
    assert {"top_face", "face_px", "cube_mm", "snap_tol", "shade_search", "partial_contrast"} <= set(names)
    # Detections of a trained model are never dropped or split, so those gates are not offered.
    assert not {"min_area_ratio", "max_area_ratio", "split", "drop_side_faces", "box_pad"} & set(names)
    assert names[-1] == "roi"
    # An OBB already gives the top face, so it has no top-face correction.
    obb = [p.name for p in ultra.param_specs("obb", None)]
    assert "face_px" not in obb and "top_face" not in obb and "proto_board" in obb and "shrink" in obb
    assert {"box_pad", "box_t", "box_size", "box_blur"} <= {p.name for p in ultra.param_specs("detect", None)}
    assert next(p for p in ultra.param_specs("segment", 960) if p.name == "imgsz").default == 960
    assert next(p for p in ultra.param_specs("detect", 320) if p.name == "imgsz").default == 320
    assert next(p for p in ultra.param_specs("detect", None) if p.name == "imgsz").default == 1280
    assert "retina_masks" not in [p.name for p in ultra.param_specs("obb", None)]
    classes = next(p for p in ultra.param_specs("obb", None, ["red", "wood"]) if p.name == "classes")
    assert classes.type == "multiselect" and classes.options == (("0", "red"), ("1", "wood"))


def test_predict_kwargs_map_every_argument() -> None:
    specs = ultra.param_specs("segment", 640, ["red", "green", "blue"])
    p = resolve_params(specs, {})
    assert ultra.predict_kwargs(p, "segment", "mps") == {
        "device": "mps", "imgsz": 640, "conf": 0.25, "iou": 0.5, "max_det": 300, "rect": True,
        "agnostic_nms": False, "augment": False, "verbose": False, "classes": None, "retina_masks": True}
    p = resolve_params(specs, {"imgsz_h": 384, "classes": ["2", "0", "9"], "nms": "false", "tile": "custom",
                               "tile_px": 512, "batch": 4, "quantize": "16", "compile": "reduce-overhead",
                               "channels_last": "false", "agnostic_nms": True, "augment": True, "rect": False})
    kw = ultra.predict_kwargs(p, "segment", "cpu")
    assert kw["imgsz"] == [384, 640] and kw["classes"] == [0, 2], "unknown class ids are dropped"
    assert kw["nms"] is False and kw["tile"] == 512 and kw["batch"] == 4 and kw["quantize"] == 16
    assert kw["compile"] == "reduce-overhead" and kw["channels_last"] is False
    assert kw["agnostic_nms"] is True and kw["augment"] is True and kw["rect"] is False
    assert ultra.predict_kwargs(resolve_params(specs, {"tile": "imgsz"}), "segment", "cpu")["tile"] is True


def test_setup_args_start_a_new_predictor() -> None:
    m = _model("obb", _Result())
    m.model.predictor = "first"
    m.predict(FRAME, PARAMS)
    assert m.model.predictor == "first", "the first call keeps the predictor"
    m.predict(FRAME, {**PARAMS, "conf": 0.5})
    assert m.model.predictor == "first", "conf is read on every call"
    m.predict(FRAME, {**PARAMS, "quantize": "16"})
    assert m.model.predictor is None, "quantize is read only when the predictor is built"


def test_face_px_reaches_large_images() -> None:
    """A 4032x3024 phone photo of the board needs a top face of about 190 px; resolving must not clamp it."""
    specs = ultra.param_specs("segment", None)
    face = next(p for p in specs if p.name == "face_px")
    assert (face.min, face.max) == (10, 400)
    assert resolve_params(specs, {"face_px": 190})["face_px"] == 190.0
    assert resolve_params(specs, {"face_px": 12})["face_px"] == 12.0


def test_adapter_module_does_not_import_torch() -> None:
    assert "torch" not in sys.modules, "the API process (and this test process) must never import torch"


class _T:
    """Stands in for a torch tensor: .cpu().numpy()."""

    def __init__(self, a: Any):
        self.a = np.asarray(a)

    def cpu(self) -> "_T":
        return self

    def numpy(self) -> np.ndarray:
        return self.a


class _Boxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy, self.conf, self.cls = _T(xyxy), _T(conf), _T(cls)

    def __len__(self) -> int:
        return len(self.conf.a)


class _Obb:
    def __init__(self, corners, conf, cls):
        self.xyxyxyxy, self.conf, self.cls = _T(corners), _T(conf), _T(cls)

    def __len__(self) -> int:
        return len(self.conf.a)


class _Masks:
    def __init__(self, polys):
        self.xy = [np.asarray(p, np.float32) for p in polys]


class _Result:
    def __init__(self, boxes=None, masks=None, obb=None):
        self.boxes, self.masks, self.obb = boxes, masks, obb


class _FakeYOLO:
    def __init__(self, result: _Result):
        self.result = result
        self.calls: list[dict] = []

    def predict(self, frame, **kwargs):
        self.calls.append(kwargs)
        return [self.result]


def _model(task: str, result: _Result) -> UltraModel:
    m = UltraModel.__new__(UltraModel)  # skip loading a checkpoint
    m.device, m.task, m.model = "cpu", task, _FakeYOLO(result)
    m.names = {0: "red", 1: "thing", 2: "b"}
    m._setup = None
    return m


PARAMS = {"conf": 0.3, "iou": 0.5, "imgsz": "960", "max_det": 50, "device": "cpu", "retina_masks": False,
          "color_source": "auto", "proto_red": "#c8323c", "proto_green": "#3c9650", "proto_blue": "#2850b4",
          "proto_wood": "#c8aa82", "roi": []}
FRAME = np.full((200, 300, 3), 200, np.uint8)
SQUARES = [[[20, 20], [60, 20], [60, 60], [20, 60]], [[120, 20], [160, 20], [160, 60], [120, 60]],
           [[220, 120], [260, 120], [260, 160], [220, 160]]]


def _record(monkeypatch: pytest.MonkeyPatch, fn_name: str) -> list[dict]:
    """Capture the colour/label/prototypes the adapter hands to cubevision.postprocess."""
    seen: list[dict] = []
    real = getattr(ultra, fn_name)

    def spy(frame, geom, **kw):
        seen.append(kw)
        return real(frame, geom, **kw)

    monkeypatch.setattr(ultra, fn_name, spy)
    return seen


@pytest.mark.parametrize("source, expected", [
    ("auto", ["red", None, "blue"]),  # class colour when the name is a colour, pixels otherwise
    ("class", ["red", "unknown", "blue"]),
    ("pixels", [None, None, None]),
])
def test_colour_source_for_obb(monkeypatch: pytest.MonkeyPatch, source: str, expected: list) -> None:
    seen = _record(monkeypatch, "cube_from_quad")
    m = _model("obb", _Result(obb=_Obb(SQUARES, [0.9, 0.8, 0.7], [0, 1, 2])))
    cubes = m.predict(FRAME, {**PARAMS, "color_source": source})
    assert [kw["color"] for kw in seen] == expected
    assert [kw["label"] for kw in seen] == ["red", "thing", "b"]
    assert all(kw["prototypes"]["red"] == "#c8323c" for kw in seen)
    assert len(cubes) == 3 and [c.score for c in cubes] == pytest.approx([0.9, 0.8, 0.7])
    call = m.model.calls[0]
    assert call == {"device": "cpu", "imgsz": 960, "conf": 0.3, "iou": 0.5, "max_det": 50, "verbose": False,
                    "rect": True, "agnostic_nms": False, "augment": False, "classes": None}


def test_segment_and_detect_conversion(monkeypatch: pytest.MonkeyPatch) -> None:
    boxes = _Boxes([[20, 20, 60, 60], [120, 20, 160, 60], [220, 120, 260, 160]], [0.9, 0.8, 0.7], [0, 0, 2])
    seen = _record(monkeypatch, "cube_from_polygon")
    seg = _model("segment", _Result(boxes=boxes, masks=_Masks(SQUARES)))
    seg.predict(FRAME, {**PARAMS, "retina_masks": True})
    assert seg.model.calls[0]["retina_masks"] is True
    assert [kw["box"] for kw in seen] == [(20.0, 20.0, 60.0, 60.0), (120.0, 20.0, 160.0, 60.0),
                                         (220.0, 120.0, 260.0, 160.0)]
    seen = _record(monkeypatch, "cube_from_box")
    det = _model("detect", _Result(boxes=boxes))
    det.predict(FRAME, PARAMS)
    assert "retina_masks" not in det.model.calls[0]
    assert [kw["color"] for kw in seen] == ["red", "red", "blue"]


def test_roi_filters_by_centre() -> None:
    m = _model("obb", _Result(obb=_Obb(SQUARES, [0.9, 0.8, 0.7], [0, 0, 0])))
    roi = [{"mode": "include", "points": [[0, 0], [100, 0], [100, 100], [0, 100]]}]
    cubes = m.predict(FRAME, {**PARAMS, "roi": roi})
    assert [round(c.center[0]) for c in cubes] == [40]


def test_empty_results() -> None:
    for task, result in (("obb", _Result(obb=_Obb(np.zeros((0, 4, 2)), [], []))),
                         ("segment", _Result(boxes=None, masks=None)), ("detect", _Result(boxes=None))):
        assert _model(task, result).predict(FRAME, PARAMS) == []

"""Ultralytics adapter: uploaded .pt checkpoints (detect / segment / obb) -> cubes.

Importing this module is cheap and never imports torch: the API process uses only
``param_specs`` and ``inspect_in_subprocess``. torch and ultralytics are imported by
``configure()``, which only the worker process and the ``inspect`` subprocess call.

    python -m lab.ultra inspect <file.pt> --data <data dir>    # prints one JSON object
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from cubevision.pipelines.common import (
    G_SIZE, box_params, colour_opts_from, colour_params, geometry_from, geometry_params, prototypes_from,
)
from cubevision.postprocess import cube_from_box, cube_from_polygon, cube_from_quad
from cubevision.roi import filter_cubes
from cubevision.types import Color, Cube, ParamSpec

TASKS = ("detect", "segment", "obb")
TASK_LABELS = {"detect": "검출(박스)", "segment": "세그멘테이션", "obb": "회전 박스(OBB)"}
DEFAULT_IMGSZ = 1280  # when the checkpoint does not say its training size
INSPECT_TIMEOUT_S = 180
G_DET = "검출"
G_RUN = "실행"
# Predict arguments read once, when Ultralytics sets the model up: changing one needs a new predictor.
SETUP_ARGS = ("device", "quantize", "nms", "compile", "channels_last")

# Class name -> cube colour: case-insensitive exact or prefix match ("red_cube", "Blue", "w").
_CLASS_COLOURS: list[tuple[str, Color]] = [
    ("red", "red"), ("빨강", "red"), ("green", "green"), ("초록", "green"),
    ("blue", "blue"), ("파랑", "blue"), ("wood", "wood"), ("나무", "wood"),
    ("r", "red"), ("g", "green"), ("b", "blue"), ("w", "wood"),
]


def colour_from_class(name: str | None) -> Color | None:
    n = (name or "").strip().lower()
    if not n:
        return None
    for prefix, colour in _CLASS_COLOURS:
        if len(prefix) == 1:
            # A single letter only counts on its own or before a separator ("r", "r_cube"),
            # otherwise "box" or "background" would be read as blue.
            if n == prefix or (n.startswith(prefix) and not n[1].isalnum()):
                return colour
        elif n.startswith(prefix):
            return colour
    return None


def param_specs(task: str | None, train_imgsz: int | None, class_names: list[str] | None = None) -> list[ParamSpec]:
    """Every Ultralytics predict argument that changes the result or the speed of one image
    (ultralytics/cfg/default.yaml, 8.4.215), then the lab's own post-processing of the detections.

    Left out: arguments that only save or show files (save*, show*, visualize, embed, line_width),
    stream and video-reader options (source, stream_buffer, vid_stride: the lab reads the frames),
    and ``dnn`` (ONNX only)."""
    default_imgsz = int(train_imgsz) if train_imgsz else DEFAULT_IMGSZ
    names = list(class_names or [])
    specs = [
        ParamSpec("conf", "신뢰도 하한 (conf)", "float", 0.25, 0.0, 1.0, 0.01, group=G_DET,
                  help="이 점수보다 낮은 검출은 버립니다"),
        ParamSpec("iou", "NMS IoU (iou)", "float", 0.5, 0.0, 1.0, 0.01, group=G_DET,
                  help="겹친 검출을 하나로 합치는 기준. 낮을수록 더 많이 합칩니다"),
        ParamSpec("imgsz", "입력 크기 (imgsz)", "int", default_imgsz, 32, 4096, 32, group=G_DET, unit="px",
                  help="모델에 넣는 이미지 크기 (32의 배수로 맞춰짐). 학습 때 크기가 기본값입니다"),
        ParamSpec("imgsz_h", "입력 높이 (imgsz [h, w])", "int", 0, 0, 4096, 32, group=G_DET, unit="px",
                  help="0이면 위 크기 하나만 씁니다. 0보다 크면 [이 높이, 위 크기] 직사각 입력"),
        ParamSpec("rect", "최소 여백 레터박스 (rect)", "bool", True, group=G_DET,
                  help="켜면 비율을 유지하며 32의 배수까지만 채웁니다. 끄면 정사각으로 채웁니다"),
        ParamSpec("max_det", "최대 검출 수 (max_det)", "int", 300, 1, 3000, 1, group=G_DET,
                  help="한 프레임에서 남길 최대 검출 수"),
        ParamSpec("classes", "클래스만 (classes)", "multiselect", [],
                  options=tuple((str(i), n) for i, n in enumerate(names)), group=G_DET,
                  help="고른 클래스만 남깁니다. 아무것도 안 고르면 전체"),
        ParamSpec("agnostic_nms", "클래스 무시 NMS (agnostic_nms)", "bool", False, group=G_DET,
                  help="다른 클래스끼리 겹쳐도 하나로 합칩니다"),
        ParamSpec("augment", "테스트 시 증강 (augment)", "bool", False, group=G_DET,
                  help="뒤집기·크기 변형을 함께 추론해 합칩니다 (느려짐)"),
        ParamSpec("nms", "NMS 방식 (nms)", "select", "auto",
                  options=(("auto", "모델 기본"), ("false", "NMS 없는 헤드 (있으면)")), group=G_DET,
                  help="YOLO26처럼 NMS 없는 헤드가 있는 모델은 그것을 씁니다"),
        ParamSpec("tile", "타일 추론 (tile)", "select", "off",
                  options=(("off", "끔"), ("imgsz", "입력 크기로"), ("custom", "직접 지정")), group=G_DET,
                  help="큰 이미지를 겹치는 타일로 나눠 추론한 뒤 합칩니다"),
        ParamSpec("tile_px", "타일 크기", "int", 640, 32, 4096, 32, group=G_DET, unit="px",
                  help="'직접 지정'일 때 원본 픽셀 기준 타일 한 변"),
        ParamSpec("batch", "타일 묶음 (batch)", "int", 1, 1, 64, 1, group=G_RUN,
                  help="타일 추론에서 한 번에 넣는 타일 수"),
        ParamSpec("device", "장치 (device)", "select", "auto",
                  options=(("auto", "자동 (MPS 우선)"), ("mps", "MPS (GPU)"), ("cpu", "CPU")), group=G_RUN,
                  help="자동은 Apple GPU(MPS)를 쓸 수 있으면 MPS, 아니면 CPU"),
        ParamSpec("quantize", "정밀도 (quantize)", "select", "32", options=(("32", "FP32"), ("16", "FP16")), group=G_RUN,
                  help="FP16은 GPU에서 빠를 수 있지만 장치에 따라 지원하지 않습니다"),
        ParamSpec("compile", "torch.compile (compile)", "select", "false",
                  options=(("false", "끔"), ("true", "기본"), ("reduce-overhead", "reduce-overhead"),
                           ("max-autotune-no-cudagraphs", "max-autotune-no-cudagraphs")), group=G_RUN,
                  help="첫 프레임이 오래 걸리는 대신 이후가 빨라질 수 있습니다"),
        ParamSpec("channels_last", "channels_last", "select", "auto",
                  options=(("auto", "자동"), ("true", "켬"), ("false", "끔")), group=G_RUN,
                  help="메모리 배치. 자동은 Ultralytics가 정합니다"),
    ]
    if task == "segment":
        specs.append(ParamSpec("retina_masks", "원본 해상도 마스크 (retina_masks)", "bool", True, group=G_RUN,
                               help="마스크를 원본 크기로 계산해 모서리·각도가 정확해집니다 (조금 느림)"))
    specs.append(ParamSpec("color_source", "색 결정 방식", "select", "auto",
                           options=(("auto", "자동"), ("class", "클래스 이름"), ("pixels", "영역 픽셀")), group="색 기준",
                           help="자동: 클래스 이름이 색 이름이면 그것을, 아니면 윗면 픽셀로 분류"))
    specs += colour_params()
    if task in ("segment", "detect"):
        specs.append(ParamSpec("top_face", "윗면만 남기기", "bool", True, group=G_SIZE,
                               help="마스크·박스에 옆면이 섞여 있으면 카메라에서 먼 쪽 모서리 기준으로 윗면 크기만큼만 남깁니다"))
        specs += geometry_params(outside=True)
    if task == "detect":
        specs += box_params()
    specs.append(ParamSpec("roi", "검출 영역", "roi", [], group="검출 영역",
                           help="중심이 이 영역 밖인 큐브는 버립니다. 비우면 전체 화면"))
    return specs


def predict_kwargs(p: dict[str, Any], task: str, device: str) -> dict[str, Any]:
    """Ultralytics predict arguments from resolved params (only those that differ from what
    Ultralytics would do anyway are left implicit: quantize FP32, channels_last auto, no tiles)."""
    imgsz: Any = int(p["imgsz"])
    if int(p.get("imgsz_h", 0)) > 0:
        imgsz = [int(p["imgsz_h"]), int(p["imgsz"])]
    kw: dict[str, Any] = dict(
        device=device, imgsz=imgsz, conf=float(p["conf"]), iou=float(p["iou"]), max_det=int(p["max_det"]),
        rect=bool(p.get("rect", True)), agnostic_nms=bool(p.get("agnostic_nms", False)),
        augment=bool(p.get("augment", False)), verbose=False,
    )
    classes = p.get("classes") or []
    kw["classes"] = [int(c) for c in classes] if classes else None
    if p.get("nms", "auto") == "false":
        kw["nms"] = False
    tile = p.get("tile", "off")
    if tile != "off":
        kw["tile"] = True if tile == "imgsz" else int(p.get("tile_px", 640))
        kw["batch"] = int(p.get("batch", 1))
    if str(p.get("quantize", "32")) == "16":
        kw["quantize"] = 16
    comp = p.get("compile", "false")
    if comp != "false":
        kw["compile"] = True if comp == "true" else comp
    cl = p.get("channels_last", "auto")
    if cl != "auto":
        kw["channels_last"] = cl == "true"
    if task == "segment":
        kw["retina_masks"] = bool(p.get("retina_masks", True))
    return kw


def configure(data_dir: Path) -> None:
    """Point Ultralytics at data/ultralytics, switch off downloads/telemetry, then import it."""
    cfg = data_dir / "ultralytics"
    cfg.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(cfg)
    os.environ["YOLO_AUTOINSTALL"] = "False"
    os.environ["YOLO_OFFLINE"] = "1"
    os.environ["YOLO_VERBOSE"] = "False"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    from ultralytics import settings

    settings.update({
        "sync": False,
        "weights_dir": str(data_dir / "weights"),
        "runs_dir": str(cfg / "runs"),
        "datasets_dir": str(cfg / "datasets"),
    })


def mps_available() -> bool:
    import torch

    return bool(torch.backends.mps.is_available())


def resolve_device(choice: str) -> str:
    if choice == "cpu":
        return "cpu"
    if choice == "mps" or mps_available():
        return "mps"
    return "cpu"


def _train_imgsz(ckpt: Any) -> int | None:
    args = (ckpt or {}).get("train_args") or {}
    imgsz = args.get("imgsz") if isinstance(args, dict) else None
    if isinstance(imgsz, (list, tuple)) and imgsz:
        imgsz = max(imgsz)
    return int(imgsz) if isinstance(imgsz, (int, float)) else None


def inspect_checkpoint(path: Path) -> dict[str, Any]:
    """Load a checkpoint and describe it. Never raises: failures come back as ok=False."""
    try:
        from ultralytics import YOLO

        model = YOLO(str(path.resolve()))
    except Exception as exc:  # noqa: BLE001 - any failure means "not a usable Ultralytics file"
        return {"ok": False, "error": f"Ultralytics 체크포인트로 읽을 수 없습니다 ({type(exc).__name__}: {exc})"[:500]}
    task = getattr(model, "task", None)
    names = getattr(model, "names", None) or {}
    class_names = [str(names[k]) for k in sorted(names)] if isinstance(names, dict) else [str(n) for n in names]
    if task not in TASKS:
        return {"ok": False, "task": task, "class_names": class_names,
                "error": f"지원하지 않는 작업 종류입니다: {task} (detect · segment · obb 만 시험할 수 있습니다)"}
    return {"ok": True, "task": task, "class_names": class_names, "imgsz": _train_imgsz(getattr(model, "ckpt", None))}


def inspect_in_subprocess(path: Path, data_dir: Path) -> dict[str, Any]:
    """Run ``inspect_checkpoint`` in a child interpreter so the API process never imports torch."""
    cmd = [sys.executable, "-m", "lab.ultra", "inspect", str(path), "--data", str(data_dir)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=INSPECT_TIMEOUT_S,
                              cwd=data_dir / "tmp")
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"검사 시간이 {INSPECT_TIMEOUT_S}초를 넘었습니다"}
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
    if proc.returncode != 0 or not lines:
        tail = (proc.stderr.strip().splitlines() or ["알 수 없는 오류"])[-1]
        return {"ok": False, "error": f"가중치 검사에 실패했습니다: {tail}"[:500]}
    return json.loads(lines[-1])


class UltraModel:
    """One loaded checkpoint on one device. The worker keeps at most one of these."""

    def __init__(self, path: Path, device: str):
        from ultralytics import YOLO

        self.device = device
        self.model = YOLO(str(path.resolve()))
        self.task = self.model.task
        if self.task not in TASKS:
            raise ValueError(f"지원하지 않는 작업 종류입니다: {self.task}")
        names = self.model.names or {}
        self.names: dict[int, str] = {int(k): str(v) for k, v in names.items()} if isinstance(names, dict) else dict(enumerate(names))
        self._setup: tuple[str, ...] | None = None

    def predict(self, frame_bgr: np.ndarray, p: dict[str, Any]) -> list[Cube]:
        kwargs = predict_kwargs(p, self.task, self.device)
        setup = tuple(repr(kwargs.get(k)) for k in SETUP_ARGS)
        if setup != self._setup:
            # Ultralytics reads these only when it builds its predictor: start a new one.
            if self._setup is not None:
                self.model.predictor = None
            self._setup = setup
        r = self.model.predict(frame_bgr, **kwargs)[0]
        prototypes = prototypes_from(p)
        geo = geometry_from(p)
        opts = colour_opts_from(p)
        face_px = float(p["face_px"]) if p.get("top_face", True) and "face_px" in p else None
        source = p.get("color_source", "auto")

        def colour_for(cls_id: int) -> tuple[Color | None, str]:
            label = self.names.get(cls_id, str(cls_id))
            if source == "pixels":
                return None, label
            named = colour_from_class(label)
            if named is None and source == "class":
                return "unknown", label
            return named, label

        cubes: list[Cube] = []
        if self.task == "obb" and r.obb is not None and len(r.obb):
            corners = r.obb.xyxyxyxy.cpu().numpy()
            confs = r.obb.conf.cpu().numpy()
            classes = r.obb.cls.cpu().numpy().astype(int)
            for quad, conf, cls_id in zip(corners, confs, classes):
                colour, label = colour_for(int(cls_id))
                cubes.append(cube_from_quad(frame_bgr, quad, score=float(conf), color=colour, label=label,
                                            prototypes=prototypes, geo=geo, colour_opts=opts))
        elif self.task == "segment" and r.masks is not None and r.boxes is not None and len(r.boxes):
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)
            for poly, box, conf, cls_id in zip(r.masks.xy, boxes, confs, classes):
                colour, label = colour_for(int(cls_id))
                cube = cube_from_polygon(frame_bgr, poly, score=float(conf), color=colour, label=label,
                                         prototypes=prototypes, box=tuple(float(v) for v in box),
                                         top_face=face_px is not None, face_px=face_px, geo=geo, colour_opts=opts)
                if cube is not None:
                    cubes.append(cube)
        elif self.task == "detect" and r.boxes is not None and len(r.boxes):
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)
            for box, conf, cls_id in zip(boxes, confs, classes):
                colour, label = colour_for(int(cls_id))
                cube = cube_from_box(frame_bgr, box, score=float(conf), color=colour, label=label,
                                     prototypes=prototypes, face_px=face_px, geo=geo, colour_opts=opts)
                if cube is not None:
                    cubes.append(cube)
        return filter_cubes(cubes, p.get("roi"))

    def close(self) -> None:
        del self.model
        gc.collect()
        try:
            import torch

            if self.device == "mps" and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:  # noqa: BLE001 - freeing memory is best effort
            pass


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m lab.ultra")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ins = sub.add_parser("inspect", help="print task, class names and train imgsz of a checkpoint as JSON")
    ins.add_argument("path", type=Path)
    ins.add_argument("--data", type=Path, required=True)
    args = ap.parse_args(argv)
    configure(args.data.resolve())
    print(json.dumps(inspect_checkpoint(args.path), ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

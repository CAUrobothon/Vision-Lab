"""/api/preview: run a built-in pipeline on one frame right now (nothing is stored).

A plain ``def`` endpoint, so it runs in the threadpool; OpenCV releases the GIL while it
works, which keeps the API responsive during a slider drag.
"""

from __future__ import annotations

import base64
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from fastapi import APIRouter

from lab import frames, models
from lab import media as media_mod
from lab.deps import Conn, Paths
from lab.errors import LabError
from lab.schemas import DebugLayer, PreviewRequest, PreviewResult

router = APIRouter(prefix="/api", tags=["preview"])

# cv2.imencode releases the GIL, so the layers of one preview are encoded side by side
# (1280x720 colour layers take 10-30 ms each at the fastest PNG compression).
_encoder = ThreadPoolExecutor(max_workers=4, thread_name_prefix="png")


def png_data_url(image: np.ndarray) -> str:
    """Masks (HxW) become 1-channel PNGs, images (HxWx3 BGR) colour PNGs; fast compression."""
    img = np.asarray(image)
    if img.dtype == bool:
        img = img.astype(np.uint8) * 255
    elif img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not ok:
        raise LabError(500, "디버그 이미지를 인코딩하지 못했습니다")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


@router.post("/preview")
def preview(req: PreviewRequest, conn: Conn, paths: Paths) -> PreviewResult:
    if models.weights_id(req.model_key) is not None:
        raise LabError(400, "가중치 모델은 미리보기 대신 실행을 사용하세요")
    pipeline = models.find_pipeline(req.model_key)
    row = media_mod.get_row(conn, req.media_id)
    params = models.resolve(conn, req.model_key, req.params).params
    try:
        frame = frames.get_frame(media_mod.frame_source(paths, row), req.frame)
    except IndexError:
        raise LabError(400, f"프레임 번호가 범위를 벗어났습니다 (0 ~ {(row['frame_count'] or 1) - 1})") from None
    except frames.FrameError as exc:
        raise LabError(500, str(exc)) from exc

    t0 = time.perf_counter()
    try:
        result = pipeline(frame, params, debug=req.debug)
    except Exception as exc:  # noqa: BLE001 - show pipeline bugs to the user instead of a bare 500
        raise LabError(500, f"파이프라인 실행 중 오류가 났습니다: {type(exc).__name__}: {exc}") from exc
    ms = (time.perf_counter() - t0) * 1000.0

    layers = []
    if req.debug and result.debug:
        urls = _encoder.map(png_data_url, result.debug.values())
        layers = [DebugLayer(name=name, label=pipeline.debug_labels.get(name, name), url=url)
                  for name, url in zip(result.debug, urls)]
    return PreviewResult(
        media_id=req.media_id,
        frame=req.frame,
        model_key=req.model_key,
        params=params,
        cubes=[c.to_dict() for c in result.cubes],
        ms=round(ms, 3),
        stages_ms={k: round(float(v), 3) for k, v in result.stages_ms.items()},
        notes=list(result.notes),
        debug_layers=layers,
    )

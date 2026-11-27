"""cv.hsv_ranges — one HSV gate per colour class; the class is the gate that fired.

The fastest and most familiar baseline. Every colour has the same three gates, cv2.inRange's
lower and upper bound on H, S and V. OpenCV's hue scale is 0-179 and red straddles its end, so a
hue range may start below 0 (−10 means 170). Optional white balance on the board first; by
default the wood gate (the loosest) gives way to R, G and B where they overlap; holes filled.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from cubevision.colour import board_colour
from cubevision.pipelines.common import (
    G_PRE, G_SIZE, BlobPipeline, Context, apply_blur, apply_morph, blob_params, blobs_kwargs, blur_params,
    board_opts_from, colour_params, geometry_params, morph_params, roi_param, union_mask,
)
from cubevision.postprocess import Blob, blobs_from_mask
from cubevision.types import ParamSpec

G_HSV = "HSV 범위"
COLOURS = (("red", "빨강"), ("green", "초록"), ("blue", "파랑"), ("wood", "나무"))
# (H, S, V) gate defaults: [low, high] each, OpenCV units (H 0-179, S and V 0-255).
DEFAULT_GATES = {
    "red": ([-12, 10], [90, 255], [60, 255]),
    "green": ([38, 88], [60, 255], [50, 255]),
    "blue": ([98, 135], [70, 255], [40, 255]),
    "wood": ([6, 30], [35, 140], [70, 255]),
}


def _gate(name: str, ko: str) -> list[ParamSpec]:
    h, s, v = DEFAULT_GATES[name]
    return [
        ParamSpec(f"{name}_h", f"{ko} H 범위", "range", list(h), -90, 179, 1, group=G_HSV,
                  help="OpenCV H (0–179). 음수는 179 쪽으로 감김: −10 = 170"),
        ParamSpec(f"{name}_s", f"{ko} S 범위", "range", list(s), 0, 255, 1, group=G_HSV, help="채도 하한–상한"),
        ParamSpec(f"{name}_v", f"{ko} V 범위", "range", list(v), 0, 255, 1, group=G_HSV, help="밝기 하한–상한"),
    ]


def _hue_band(hsv: np.ndarray, h: list[int], s: list[int], v: list[int]) -> np.ndarray:
    """Pixels whose hue is in ``h`` (and S in ``s``, V in ``v``). A negative hue counts back from
    180 (−10 = 170), so [−12, 10] wraps through 0 and [−20, −5] is 160–175 only."""
    lo, hi = int(h[0]), int(h[1])
    s0, s1, v0, v1 = int(s[0]), int(s[1]), int(v[0]), int(v[1])
    if hi < 0:  # both ends below 0: the same range on the 179 side, no wrap
        lo, hi = lo + 180, hi + 180
    if lo < 0:  # wraps around 179 -> 0
        a = cv2.inRange(hsv, (0, s0, v0), (hi, s1, v1))
        b = cv2.inRange(hsv, (180 + lo, s0, v0), (179, s1, v1))
        return cv2.bitwise_or(a, b)
    return cv2.inRange(hsv, (lo, s0, v0), (hi, s1, v1))


class HsvRanges(BlobPipeline):
    key = "cv.hsv_ranges"
    name = "HSV 색 범위"
    summary = "색마다 H·S·V 하한–상한으로 마스크를 만들고 그 범위가 곧 색인 가장 빠른 기본 방식"
    description = (
        "발상: 빨강·초록·파랑·나무마다 HSV 범위(게이트)를 두고, 통과한 픽셀 덩어리를 큐브 후보로 봅니다. "
        "색은 통과한 게이트가 정합니다.\n"
        "강점: 가장 빠르고 이해하기 쉬우며, 슬라이더로 바로 맞출 수 있습니다. 조명이 고정되면 세 원색에 강합니다.\n"
        "약점: 노출·화이트밸런스가 바뀌면 S·V가 움직여 다시 맞춰야 합니다. 나무 게이트는 따뜻한 조명의 판·그림자와 겹치기 쉽습니다.\n"
        "바구니 바닥은 같은 종이라 게이트를 통과하므로 큰 영역 제외와 검출 영역(제외 다각형)이 필요합니다."
    )
    params = [
        *[spec for name, ko in COLOURS for spec in _gate(name, ko)],
        ParamSpec("wood_yields", "나무는 원색에 양보", "bool", True, group=G_HSV,
                  help="빨강·초록·파랑 범위에도 든 픽셀은 나무 마스크에서 뺍니다"),
        ParamSpec("wb_normalise", "판 기준 화이트밸런스", "bool", True, group=G_PRE,
                  help="판이 무채색이 되도록 채널을 맞춘 뒤 HSV를 계산 (따뜻한 조명에서 판이 나무로 잡히는 것 방지)"),
        *blur_params(G_PRE),
        *morph_params(G_PRE),
        *blob_params(G_SIZE),
        *geometry_params(),
        *colour_params(),
        roi_param(),
    ]
    debug_labels = {
        "mask": "색 게이트를 통과한 픽셀 (네 색 합)",
        "hsv": "HSV 보기 (H를 색상으로, S를 밝기로)",
        "candidates": "후보 덩어리와 윗면",
    }

    def migrate(self, given: dict[str, Any] | None) -> dict[str, Any]:
        """Older records stored only the lower S / V bound (``red_s_min``); they become
        [low, 255], which is what the gate did then."""
        given = dict(given or {})
        for name, _ in COLOURS:
            for ch in ("s", "v"):
                old = given.pop(f"{name}_{ch}_min", None)
                if old is not None and f"{name}_{ch}" not in given:
                    given[f"{name}_{ch}"] = [old, 255]
        return given

    def candidates(self, ctx: Context) -> list[Blob]:
        p: dict[str, Any] = ctx.params
        bgr = apply_blur(ctx.frame, p)
        if p["wb_normalise"]:
            board = board_colour(bgr, ctx.roi, **board_opts_from(p))
            if board is not None:
                gains = board.mean() / np.maximum(board, 1.0)
                bgr = cv2.transform(bgr, np.diag(gains).astype(np.float32))
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        ctx.watch.lap("전처리")

        masks = {name: _hue_band(hsv, p[f"{name}_h"], p[f"{name}_s"], p[f"{name}_v"]) for name, _ in COLOURS}
        if p["wood_yields"]:
            taken = cv2.bitwise_or(cv2.bitwise_or(masks["red"], masks["green"]), masks["blue"])
            masks["wood"] = cv2.bitwise_and(masks["wood"], cv2.bitwise_not(taken))
        for name in masks:
            masks[name] = apply_morph(ctx.apply_roi(masks[name]), p)
        ctx.watch.lap("색 게이트")

        kw = blobs_kwargs(ctx)
        blobs = [b for name, m in masks.items() for b in blobs_from_mask(m, name, **kw)]
        ctx.watch.lap("덩어리")
        if ctx.debug:
            ctx.layers["mask"] = union_mask(masks, ctx.frame.shape[:2])
            vis = np.dstack([hsv[..., 0], np.full_like(hsv[..., 0], 255), hsv[..., 1]])
            ctx.layers["hsv"] = cv2.cvtColor(vis, cv2.COLOR_HSV2BGR)
        return blobs

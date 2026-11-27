"""cv.board_diff_lab — model the board instead of the cubes.

Every pixel whose chroma differs from the white board (measured in this frame) is foreground,
and each foreground pixel takes the class of the nearest colour prototype in the a*b* plane.
Shadows lower lightness but hardly change chroma, so they stay background; wood is found
because it is yellowish relative to the board; touching cubes of different colours separate
because the label changes.

Both decisions depend only on a pixel's 8-bit (a*, b*), so they are precomputed in one
65536-entry table per frame and applied with a single lookup.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from cubevision.colour import ColourModel
from cubevision.pipelines.common import (
    G_COLOUR, G_PRE, G_SIZE, BlobPipeline, Context, ab_grid, apply_blur, apply_morph, blob_params, blobs_kwargs,
    blur_params, class_overlay, colour_params, flag_weak_chroma, geometry_params, lookup, morph_params, roi_param,
)
from cubevision.postprocess import Blob, blobs_from_mask
from cubevision.types import ParamSpec

STRONG = 8  # table value offset: class k strong = k, weak = k + STRONG


def build_table(colour: ColourModel, t_lo: float, t_hi: float, t_dist: float) -> np.ndarray:
    """256x256 table over 8-bit (a, b): 0 background, k (1..4) strong class k, k+8 weak class k
    (plus WEAK_FLAG, see common.lookup).

    Strong = chroma distance from the board above t_hi; weak = above t_lo. A pixel farther than
    t_dist from every prototype in a*b* is background."""
    aa, bb = ab_grid()
    a0, b0 = float(colour.board_lab[1]), float(colour.board_lab[2])
    dc = np.hypot(aa - a0, bb - b0)
    d = np.stack([np.hypot(aa - p[1], bb - p[2]) for p in colour.lab], axis=0)
    cls = (d.argmin(axis=0) + 1).astype(np.uint8)
    cls[d.min(axis=0) > t_dist] = 0
    table = np.where(dc > t_hi, cls, np.where(dc > t_lo, cls + STRONG, 0)).astype(np.uint8)
    table[cls == 0] = 0
    return flag_weak_chroma(table, colour)


class BoardDiffLab(BlobPipeline):
    key = "cv.board_diff_lab"
    name = "판 차이 + Lab 최근접색"
    summary = "흰 판과 색(채도)이 다른 픽셀을 전경으로 보고, 픽셀마다 가장 가까운 기준색을 붙이는 방식"
    description = (
        "발상: 큐브 대신 판을 모델링합니다. 매 프레임 판의 색을 재고, 판과 a*b* 거리가 먼 픽셀을 전경으로 본 뒤 "
        "가장 가까운 기준색(빨강·초록·파랑·나무)을 붙입니다. 강한 픽셀이 씨앗이 되는 두 단계 문턱을 씁니다.\n"
        "강점: 그림자는 밝기만 낮추고 채도는 거의 바꾸지 않아 배경으로 남습니다. 나무도 같은 방식으로 찾고, "
        "색이 다른 큐브가 붙어 있어도 라벨이 바뀌어 저절로 나뉩니다. 판 색을 매 프레임 재므로 조명 변화에 강합니다.\n"
        "약점: 판과 채도 차이가 작은 밝은 나무는 빠지거나 가장자리가 거칠어집니다. 바구니 바닥은 같은 종이라 전경이 되므로 "
        "큰 영역 제외와 검출 영역이 필요합니다. 기준색이 실제와 많이 다르면 색이 틀립니다."
    )
    params = [
        ParamSpec("t_chroma_hi", "전경 씨앗 채도", "float", 18.0, 0, 80, 0.5, group=G_COLOUR,
                  help="판과 a*b* 거리가 이보다 크면 확실한 전경(씨앗)"),
        ParamSpec("t_chroma_lo", "전경 연결 채도", "float", 10.0, 0, 80, 0.5, group=G_COLOUR,
                  help="씨앗과 이어져 있으면 이 거리까지 전경. 나무와 판을 가르는 값"),
        ParamSpec("t_dist", "기준색 최대 거리 (픽셀)", "float", 40.0, 0, 150, 1, group=G_COLOUR,
                  help="픽셀이 가장 가까운 기준색과 a*b*로 이보다 멀면 배경"),
        ParamSpec("seed_connectivity", "씨앗 연결", "select", "8", options=(("8", "8방향"), ("4", "4방향")),
                  group=G_COLOUR, help="약한 전경이 씨앗과 이어졌는지 볼 때 대각선 연결을 칠지"),
        *colour_params(),
        *blur_params(G_PRE, help="색 판정 전 잡음 제거 (홀수, 1 = 끔)"),
        *morph_params(G_SIZE),
        *blob_params(G_SIZE),
        *geometry_params(),
        roi_param(),
    ]
    debug_labels = {
        "chroma": "판과의 채도 차이 (파랑 작음 → 빨강 큼)",
        "foreground": "전경 픽셀 (두 단계 문턱 통과)",
        "labels": "픽셀별 색 라벨",
        "candidates": "후보 덩어리와 윗면",
    }

    def candidates(self, ctx: Context) -> list[Blob]:
        p: dict[str, Any] = ctx.params
        frame = ctx.balanced()
        bgr = apply_blur(frame, p)
        lab8 = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
        ctx.watch.lap("전처리")

        t_lo = min(p["t_chroma_lo"], p["t_chroma_hi"])
        table = build_table(ctx.colour, t_lo, p["t_chroma_hi"], p["t_dist"])
        code = lookup(table, lab8, ctx.colour)
        if ctx.roi is not None:
            code[ctx.roi == 0] = 0
        # Hysteresis: keep weak components that contain at least one strong pixel.
        n, comp = cv2.connectedComponents((code > 0).astype(np.uint8), connectivity=int(p["seed_connectivity"]))
        seeded = np.zeros(n, np.uint8)
        seeded[comp[(code > 0) & (code < STRONG)]] = 1
        seeded[0] = 0
        cls = np.where(seeded[comp] > 0, code % STRONG, 0).astype(np.uint8)
        ctx.watch.lap("판 차이·라벨")

        kw = blobs_kwargs(ctx)
        masks: dict[str, np.ndarray] = {}
        blobs: list[Blob] = []
        for i, name in enumerate(ctx.colour.names, start=1):
            m = apply_morph((cls == i).astype(np.uint8) * 255, p)
            masks[name] = m
            blobs += blobs_from_mask(m, name, **kw)
        ctx.watch.lap("덩어리")
        if ctx.debug:
            a0, b0 = ctx.colour.board_lab[1] + 128.0, ctx.colour.board_lab[2] + 128.0
            dc = np.hypot(lab8[..., 1].astype(np.float32) - a0, lab8[..., 2].astype(np.float32) - b0)
            ctx.layers["chroma"] = cv2.applyColorMap(np.clip(dc * 6.0, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            ctx.layers["foreground"] = (cls > 0).astype(np.uint8) * 255
            ctx.layers["labels"] = class_overlay(frame, masks)
        return blobs

"""cv.edge_quads — find closed regions bounded by edges that have the size and shape of a face.

Colour plays no part in detection: Canny on L*, a* and b* (joined, so neighbours of equal
lightness still separate), small gaps sealed, and every region the edges enclose is a
candidate when its area is a few faces at most, it nearly fills its minimum-area rectangle and
its colour is uniform. The colour is read afterwards from the pixels; regions with board colour
(an empty placement mark, a cell between grooves) are dropped there. Follows the candidate
stage of ArUco marker detection.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from cubevision.pipelines.common import (
    G_PRE, G_SIZE, BlobPipeline, Context, colour_params, dimmed, geometry_params, odd, roi_param,
)
from cubevision.postprocess import Blob
from cubevision.types import ParamSpec

G_EDGE = "윤곽"
# Default longest side of a region's minimum-area rectangle, in faces: a row of three flush cubes
# (measured 2.5-3.0 faces on synthetic scenes) with room for the side band. Measured on the
# rotated rectangle so that a group turned 30-60 degrees is judged like an axis-aligned one.
MAX_LENGTH = 3.6


class EdgeQuads(BlobPipeline):
    key = "cv.edge_quads"
    name = "윤곽으로 닫힌 사각형"
    summary = "색을 보지 않고 윤곽선이 둘러싼 윗면 크기의 사각형 영역을 찾은 뒤 색을 읽는 방식"
    description = (
        "발상: L*·a*·b* 세 채널의 Canny 윤곽선이 둘러싼 영역 가운데 넓이가 윗면 몇 배 이내이고, 최소 사각형을 거의 채우고, "
        "안쪽 색이 고른 것을 큐브 후보로 봅니다. 색은 찾은 뒤에 픽셀에서 읽습니다 (ArUco 후보 단계와 같은 생각).\n"
        "강점: 색 문턱에 기대지 않아 화이트밸런스 변화에 가장 덜 민감합니다. 흰 판 위 나무도 색이 아니라 윤곽으로 찾고, "
        "윗면과 옆면 사이 밝기 경계가 보이면 윗면만 바로 얻습니다. 바구니 바닥은 크기에서 걸러집니다.\n"
        "약점: 움직임 흐림·약한 대비로 윤곽이 끊기면 영역이 닫히지 않아 놓칩니다. 붙은 큐브 사이 이음매가 흐리면 한 덩어리가 됩니다. "
        "윗면에 반사·글자가 있으면 영역이 쪼개집니다."
    )
    params = [
        ParamSpec("denoise", "잡음 제거", "select", "bilateral",
                  options=(("bilateral", "양방향 (윤곽 보존)"), ("gauss", "가우시안 (빠름)"), ("none", "없음")),
                  group=G_PRE, help="윤곽 찾기 전 평활화"),
        ParamSpec("bilateral_d", "양방향 지름", "int", 5, 1, 31, 1, group=G_PRE, unit="px",
                  help="양방향 필터가 섞는 이웃 지름 (양방향일 때만)"),
        ParamSpec("sigma_color", "양방향 색 시그마", "float", 35.0, 1, 200, 1, group=G_PRE,
                  help="이만큼 다른 색끼리는 섞지 않습니다 (양방향일 때만)"),
        ParamSpec("sigma_space", "양방향 공간 시그마", "float", 5.0, 0.5, 50, 0.5, group=G_PRE,
                  help="얼마나 먼 픽셀까지 섞을지 (양방향일 때만)"),
        ParamSpec("gauss_ksize", "가우시안 크기", "int", 5, 1, 31, 2, group=G_PRE, unit="px",
                  help="가우시안일 때 흐림 크기 (홀수)"),
        ParamSpec("gauss_sigma", "가우시안 시그마", "float", 0.0, 0, 20, 0.1, group=G_PRE,
                  help="가우시안일 때. 0 = 크기에서 자동 계산"),
        ParamSpec("canny", "Canny 문턱 (낮음·높음)", "range", [30, 90], 0, 500, 1, group=G_EDGE,
                  help="낮추면 희미한 나무 윤곽까지 잡지만 잡선이 늘어납니다"),
        ParamSpec("canny_aperture", "Canny Sobel 크기", "select", "3", options=(("3", "3"), ("5", "5"), ("7", "7")),
                  group=G_EDGE, help="기울기를 계산하는 Sobel 커널 크기"),
        ParamSpec("canny_l2", "Canny L2 기울기", "bool", True, group=G_EDGE,
                  help="기울기 크기를 L2(정확)로 계산. 끄면 L1(|dx|+|dy|)"),
        ParamSpec("canny_L", "L* 채널 윤곽", "bool", True, group=G_EDGE, help="밝기 채널에서 윤곽 찾기"),
        ParamSpec("canny_a", "a* 채널 윤곽", "bool", True, group=G_EDGE, help="빨강–초록 채널에서 윤곽 찾기"),
        ParamSpec("canny_b", "b* 채널 윤곽", "bool", True, group=G_EDGE, help="노랑–파랑 채널에서 윤곽 찾기"),
        ParamSpec("ab_scale", "a*·b* 문턱 나눔값", "float", 2.55, 0.1, 10, 0.05, group=G_EDGE,
                  help="a*·b* 채널에는 Canny 문턱을 이 값으로 나눠 씁니다 (8비트 L*는 2.55배로 저장되므로 2.55면 같은 색 차이)"),
        ParamSpec("edge_seal", "윤곽 틈 메우기", "int", 1, 0, 10, 1, group=G_EDGE, unit="회",
                  help="윤곽을 이만큼 두껍게 해 작은 틈을 닫습니다 (영역은 그만큼 다시 키움)"),
        ParamSpec("seal_ksize", "틈 메우기 커널", "int", 3, 3, 15, 2, group=G_EDGE, unit="px",
                  help="윤곽을 두껍게 하는 팽창 커널 크기"),
        ParamSpec("region_connectivity", "영역 연결", "select", "4", options=(("4", "4방향"), ("8", "8방향")),
                  group=G_EDGE, help="윤곽 사이 빈 공간을 영역으로 묶을 때 대각선 연결을 칠지 (8은 윤곽의 대각선 틈으로 새기 쉬움)"),
        ParamSpec("min_rectangularity", "최소 사각형 채움", "float", 0.8, 0, 1.0, 0.01, group=G_SIZE,
                  help="면 하나 크기 영역: 넓이 / 최소 회전 사각형 넓이 (1 = 완전한 사각형). 더 큰 영역은 볼록 껍질 대비 + 보정값"),
        ParamSpec("single_face_area", "면 하나 영역 한도", "float", 1.5, 0.5, 5, 0.05, group=G_SIZE, unit="면",
                  help="넓이가 윗면의 이 배 미만이면 면 하나로 보고 회전 사각형 채움을, 이상이면 볼록 껍질 채움을 봅니다"),
        ParamSpec("group_fill_bonus", "묶음 채움 보정", "float", 0.1, 0, 0.5, 0.01, group=G_SIZE,
                  help="여러 큐브 묶음(L·Z자)의 볼록 껍질 채움에 더하는 값"),
        ParamSpec("max_length", "최대 길이", "float", MAX_LENGTH, 1, 10, 0.1, group=G_SIZE, unit="면",
                  help="회전 사각형의 긴 변이 윗면의 이 배보다 길면 큐브 묶음이 아님 (세 개 일렬 + 옆면 여유)"),
        ParamSpec("t_uniform", "안쪽 색 균일도", "float", 12.0, 0, 60, 0.5, group=G_SIZE,
                  help="영역 안쪽 a*·b* 표준편차가 이보다 크면 큐브 면이 아님 (팔·글자 등)"),
        ParamSpec("uniform_inset", "균일도 검사 안쪽 여백", "float", 0.15, 0, 0.5, 0.01, group=G_SIZE, unit="면",
                  help="영역 가장자리를 윗면 한 변의 이 배만큼 깎고 안쪽에서 색 균일도를 잽니다"),
        *geometry_params(min_area=0.45, max_area=4.5),
        *colour_params(),
        roi_param(),
    ]
    debug_labels = {
        "edges": "윤곽선 (L*·a*·b* Canny 합)",
        "regions": "닫힌 영역 (초록 통과 · 빨강 모양 탈락 · 주황 색 불균일)",
        "candidates": "후보 영역과 윗면",
    }

    def candidates(self, ctx: Context) -> list[Blob]:
        p: dict[str, Any] = ctx.params
        frame = ctx.balanced()  # also normalises exposure, so the Canny thresholds keep their meaning
        if p["denoise"] == "bilateral":
            den = cv2.bilateralFilter(frame, int(p["bilateral_d"]), float(p["sigma_color"]), float(p["sigma_space"]))
        elif p["denoise"] == "gauss":
            k = odd(p["gauss_ksize"])
            den = cv2.GaussianBlur(frame, (k, k), float(p["gauss_sigma"])) if k > 1 else frame
        else:
            den = frame
        lab8 = cv2.cvtColor(den, cv2.COLOR_BGR2Lab)
        ctx.watch.lap("전처리")

        lo, hi = float(p["canny"][0]), float(p["canny"][1])
        ap, l2, ab = int(p["canny_aperture"]), bool(p["canny_l2"]), float(p["ab_scale"])
        edges = np.zeros(lab8.shape[:2], np.uint8)
        if p["canny_L"]:
            edges |= cv2.Canny(lab8[..., 0], lo, hi, apertureSize=ap, L2gradient=l2)
        # 8-bit Lab stores L* x 2.55 but a* and b* x 1: scale the thresholds so one unit means
        # the same colour difference on every channel (wood differs from the board mostly in b*).
        if p["canny_a"]:
            edges |= cv2.Canny(lab8[..., 1], lo / ab, hi / ab, apertureSize=ap, L2gradient=l2)
        if p["canny_b"]:
            edges |= cv2.Canny(lab8[..., 2], lo / ab, hi / ab, apertureSize=ap, L2gradient=l2)
        seal = int(p["edge_seal"])
        if seal > 0:
            ks = odd(p["seal_ksize"])
            edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (ks, ks)), iterations=seal)
        ctx.watch.lap("윤곽")

        face = ctx.face_px
        free = cv2.bitwise_not(edges)
        if ctx.roi is not None:
            free = cv2.bitwise_and(free, ctx.roi)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(free, connectivity=int(p["region_connectivity"]))
        area = stats[:, cv2.CC_STAT_AREA]
        w, h = stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]
        ok = (area >= 0.5 * p["min_area_ratio"] * face**2) & (area <= p["max_area_ratio"] * face**2)
        # Cheap pre-filter only: a region whose rotated rectangle passes the length gate below
        # always fits in an upright box sqrt(2) times as long.
        max_len = float(p["max_length"])
        ok &= np.maximum(w, h) <= 1.415 * max_len * face
        ok[0] = False
        grow = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        blobs: list[Blob] = []
        vis = dimmed(frame) if ctx.debug else None
        for i in np.nonzero(ok)[0]:
            x, y, bw, bh = (int(v) for v in stats[i, :4])
            pad = seal + 2
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(labels.shape[1], x + bw + pad), min(labels.shape[0], y + bh + pad)
            region = (labels[y0:y1, x0:x1] == i).astype(np.uint8) * 255
            cnts, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnt = max(cnts, key=cv2.contourArea)
            rw, rh = cv2.minAreaRect(cnt)[1]
            if max(rw, rh) > max_len * face:
                continue  # longer than a row of three cubes at any rotation: not a cube group
            if area[i] < float(p["single_face_area"]) * face**2:  # one face: must be (nearly) a rectangle
                fill = float(area[i]) / max(rw * rh, 1.0)
            else:  # several flush cubes (an L or Z shape): only ask for a compact region
                fill = float(area[i]) / max(cv2.contourArea(cv2.convexHull(cnt)), 1.0) + float(p["group_fill_bonus"])
            status = "ok"
            if fill < p["min_rectangularity"]:
                status = "shape"
            else:
                inner = cv2.erode(region, grow, iterations=max(1, int(float(p["uniform_inset"]) * face)))
                px = lab8[y0:y1, x0:x1][inner > 0]
                if len(px) and float(np.sqrt(px[:, 1].astype(np.float32).var() + px[:, 2].astype(np.float32).var())) > p["t_uniform"]:
                    status = "uniform"
            if status == "ok":
                # The edge line (Canny pixel + sealing) ate into the region: grow it back.
                region = cv2.dilate(region, grow, iterations=seal + 1)
                blobs.append(Blob(region, x0, y0, None))
            if vis is not None:
                col = {"ok": (60, 200, 60), "shape": (40, 40, 230), "uniform": (0, 140, 255)}[status]
                vis[y0:y1, x0:x1][region > 0] = col
        ctx.watch.lap("영역 검사")
        if ctx.debug:
            ctx.layers["edges"] = edges
            ctx.layers["regions"] = vis  # type: ignore[assignment]
        return blobs

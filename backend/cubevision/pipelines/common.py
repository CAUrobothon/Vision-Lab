"""Pieces every built-in pipeline shares: common parameters, the run template, debug drawing.

A built-in pipeline only has to turn a frame into candidate blobs (``BlobPipeline.candidates``);
the template measures the board colour, applies the ROI, runs the shared post-processing
(postprocess.cubes_from_blobs), filters by ROI, times the stages and draws the debug layers.

Parameters: every tunable of the shared steps is a parameter (colour_params, geometry_params,
morph_params, blob_params). Nothing that changes which cubes are found, where, or with what
colour or score is a hidden constant; the defaults are the values the pipelines were measured
with. The Ultralytics adapter (lab/ultra.py) reuses the same builders.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from cubevision.colour import (
    BOARD_S_FLOOR, BOARD_S_PCT, BOARD_SAMPLES, BOARD_V_PCT, DARK_RATIO, DEFAULT_BOARD, DEFAULT_CHROMA_MIN,
    DEFAULT_MAX_DIST, DEFAULT_PROTOTYPES, GAIN_RANGE, SIGMA, W_LIGHTNESS, WEAK_FACTOR, ColourModel, bgr_to_lab,
    colour_model,
)
from cubevision.draw import OVERLAY_BGR
from cubevision.pipelines.base import Pipeline
from cubevision.postprocess import FACE_PX, Blob, Geometry, Report, cubes_from_blobs
from cubevision.roi import filter_cubes, roi_mask
from cubevision.types import Cube, FrameResult, ParamSpec

G_COLOUR = "색 기준"
G_COLOUR_X = "색 판정 세부"
G_BOARD = "판 색 측정"
G_SIZE = "크기·모양"
G_CAMERA = "카메라"
G_TOP = "윗면 찾기"
G_SPLIT = "붙은 큐브·각도"
G_PARTIAL = "가려진 큐브"
G_SCORE = "점수"
G_BOX = "박스 → 윗면"
G_PRE = "전처리"
G_ROI = "검출 영역"

COLOUR_NAMES_KO = {"red": "빨강", "green": "초록", "blue": "파랑", "wood": "나무"}
_G = Geometry()  # defaults of the shared post-processing


# --- parameters --------------------------------------------------------------------------------
def roi_param() -> ParamSpec:
    return ParamSpec(
        "roi", "검출 영역", "roi", [], group=G_ROI,
        help="포함·제외 다각형. 바구니 바닥과 기둥은 제외로 그려 두세요. 비우면 화면 전체",
    )


def colour_params() -> list[ParamSpec]:
    """Colour prototypes, the light correction and every constant of the colour decision."""
    out = [
        ParamSpec(
            f"proto_{k}", f"{COLOUR_NAMES_KO[k]} 기준색", "color", DEFAULT_PROTOTYPES[k], group=G_COLOUR,
            help=f"실제 {COLOUR_NAMES_KO[k]} 큐브 윗면에서 색을 찍어 넣으세요",
        )
        for k in ("red", "green", "blue", "wood")
    ]
    out += [
        ParamSpec("proto_board", "판 색 (기준색을 고른 사진)", "color", DEFAULT_BOARD, group=G_COLOUR,
                  help="기준색을 찍은 같은 사진에서 빈 판을 찍으세요. 조명 보정의 기준이 됩니다"),
        ParamSpec("adapt_light", "조명 보정 (판 기준)", "bool", True, group=G_COLOUR,
                  help="매 프레임 판 색을 재서 영상을 기준색을 고른 조명으로 되돌린 뒤 색을 비교합니다 (화이트밸런스·노출 보정)"),
        ParamSpec("chroma_min", "판과 구별할 최소 채도", "float", DEFAULT_CHROMA_MIN, 0, 50, 0.5, group=G_COLOUR,
                  help="판 대비 채도(Lab a*b* 거리)가 이보다 낮은 후보는 판·그림자로 보고 버립니다"),
        ParamSpec("max_dist", "기준색 최대 거리", "float", DEFAULT_MAX_DIST, 0, 150, 1, group=G_COLOUR,
                  help="가장 가까운 기준색까지 Lab 거리가 이보다 멀면 unknown"),
        ParamSpec("w_lightness", "밝기(L*) 가중치", "float", W_LIGHTNESS, 0, 2, 0.05, group=G_COLOUR_X,
                  help="기준색 거리에서 a*·b* 옆에 L* 차이를 얼마나 볼지. 0이면 밝기 무시"),
        ParamSpec("sigma", "색 확률 폭", "float", SIGMA, 0.5, 50, 0.5, group=G_COLOUR_X,
                  help="색 점수(확률)를 만들 때의 Lab 거리 척도. 작을수록 가장 가까운 색에 점수가 몰립니다"),
        ParamSpec("dark_ratio", "어두움 기준", "float", DARK_RATIO, 0, 1, 0.05, group=G_COLOUR_X,
                  help="밝기가 판의 이 비율보다 낮고 채도도 약하면 팔·바닥 같은 어두운 회색으로 보고 버립니다"),
        ParamSpec("weak_factor", "약한 채도 배수", "float", WEAK_FACTOR, 1, 5, 0.1, group=G_COLOUR_X,
                  help="'약한 채도' = 최소 채도의 이 배 미만. 어두우면서 약한 채도면 회색 물체"),
        ParamSpec("shrink", "색 읽기 안쪽 여백", "float", _G.shrink, 0, 0.45, 0.01, group=G_COLOUR_X,
                  help="윗면의 가장자리를 한쪽마다 이 비율만큼 빼고 안쪽 픽셀의 중앙값으로 색을 정합니다"),
        ParamSpec("gain_range", "조명 보정 배율 한계", "range", list(GAIN_RANGE), 0.05, 10, 0.05, group=G_BOARD,
                  help="채널별 보정 배율(기준 판 색 / 지금 판 색)을 이 범위로 자릅니다"),
        ParamSpec("board_v_pct", "판 후보 밝기 백분위", "float", BOARD_V_PCT, 0, 100, 1, group=G_BOARD, unit="%",
                  help="V가 이 백분위 이상인 밝은 픽셀만 판 후보로 봅니다 (50 = 중앙값)"),
        ParamSpec("board_s_pct", "판 후보 채도 백분위", "float", BOARD_S_PCT, 0, 100, 1, group=G_BOARD, unit="%",
                  help="밝은 후보 중 S가 이 백분위 이하인 무채색 픽셀을 판으로 봅니다"),
        ParamSpec("board_s_floor", "판 후보 채도 하한", "float", BOARD_S_FLOOR, 0, 255, 1, group=G_BOARD,
                  help="위 S 문턱이 이보다 낮아지지 않게 합니다 (강한 색 조명에서도 판을 찾도록)"),
        ParamSpec("board_samples", "판 측정 표본 수", "int", BOARD_SAMPLES, 1000, 1000000, 1000, group=G_BOARD, unit="px",
                  help="판 색을 잴 때 이만큼의 픽셀로 줄여서 봅니다"),
    ]
    return out


proto_params = colour_params  # older name


def colour_opts_from(params: dict[str, Any]) -> dict[str, Any]:
    """Keyword arguments of colour.colour_model from resolved params (defaults where missing)."""
    g = params.get("gain_range", GAIN_RANGE)
    return {
        "max_dist": float(params.get("max_dist", DEFAULT_MAX_DIST)),
        "chroma_min": float(params.get("chroma_min", DEFAULT_CHROMA_MIN)),
        "adapt": bool(params.get("adapt_light", True)),
        "w_lightness": float(params.get("w_lightness", W_LIGHTNESS)),
        "dark_ratio": float(params.get("dark_ratio", DARK_RATIO)),
        "sigma": float(params.get("sigma", SIGMA)),
        "weak_factor": float(params.get("weak_factor", WEAK_FACTOR)),
        "gain_range": (float(g[0]), float(g[1])),
        "board_samples": float(params.get("board_samples", BOARD_SAMPLES)),
        "board_v_pct": float(params.get("board_v_pct", BOARD_V_PCT)),
        "board_s_pct": float(params.get("board_s_pct", BOARD_S_PCT)),
        "board_s_floor": float(params.get("board_s_floor", BOARD_S_FLOOR)),
    }


def board_opts_from(params: dict[str, Any]) -> dict[str, Any]:
    """Keyword arguments of colour.board_colour from resolved params."""
    o = colour_opts_from(params)
    return {"samples": o["board_samples"], "v_pct": o["board_v_pct"], "s_pct": o["board_s_pct"], "s_floor": o["board_s_floor"]}


def geometry_params(*, min_area: float = 0.35, max_area: float = 8.0, outside: bool = False) -> list[ParamSpec]:
    """Every tunable of the shared post-processing (postprocess.Geometry).

    ``outside`` = for a trained model's detections, which are never dropped or split: the gates
    that drop or split candidates do nothing there, so they are left out."""
    S, C, T, P, K, R = G_SIZE, G_CAMERA, G_TOP, G_SPLIT, G_PARTIAL, G_SCORE
    out: list[ParamSpec] = [
        # Wide on purpose: it scales with the image width (synthetic: 30.6 px at 640x360,
        # 183.7 px at 3840x2160), and a clamped value turns every cube "partial".
        ParamSpec("face_px", "윗면 한 변", "float", FACE_PX, 10, 400, 0.1, group=S, unit="px",
                  help="판 위 큐브 윗면 한 변 길이. 601 mm 높이 1280x720 카메라면 61"),
    ]
    if not outside:
        out += [
            ParamSpec("min_area_ratio", "최소 크기", "float", min_area, 0, 2, 0.05, group=S, unit="면",
                      help="윗면 넓이의 몇 배보다 작으면 잡음으로 버림"),
            ParamSpec("max_area_ratio", "큰 영역 제외", "float", max_area, 1, 50, 0.5, group=S, unit="면",
                      help="윗면 넓이의 몇 배보다 크면 바구니 바닥·팔로 보고 버림"),
            ParamSpec("sliver_ratio", "가는 띠 기준", "float", _G.sliver_ratio, 0, 1, 0.05, group=S, unit="면",
                      help="짧은 쪽 폭이 윗면 한 변의 이 배보다 작으면 옆면 띠·그림자로 보고 버림"),
        ]
    out += [
        ParamSpec("partial_ratio", "가려짐 기준", "float", _G.partial_ratio, 0, 1, 0.05, group=S, unit="면",
                  help="짧은 쪽 폭이 윗면 한 변의 이 배보다 작으면 일부가 가려진 큐브로 처리"),
        ParamSpec("min_coverage", "윗면 채움 최소", "float", _G.min_coverage, 0, 1, 0.05, group=S,
                  help="맞춘 윗면 사각형을 덩어리가 이 비율 이상 채워야 함. 미만이면 고리 모양으로 보고 버림"
                       + (" (학습 모델은 버리지 않고 최소 사각형으로 대신함)" if outside else "")),
    ]
    if not outside:
        out += [
            ParamSpec("fill_holes", "덩어리 구멍 메우기", "bool", _G.fill_holes, group=S,
                      help="덩어리 안의 구멍(반사·글자)을 메웁니다"),
            ParamSpec("hole_fill_max", "구멍 메우기 한도", "float", _G.hole_fill_max, 1, 5, 0.05, group=S, unit="배",
                      help="메운 넓이가 원래의 이 배를 넘으면 고리로 보고 메우지 않습니다"),
        ]
    out += [
        ParamSpec("pp_mode", "주점(광축 중심)", "select", "center",
                  options=(("center", "화면 중심"), ("manual", "직접 입력")), group=C,
                  help="옆면 띠가 어느 쪽에 생기는지 정하는 카메라 광축 위치"),
        ParamSpec("pp_x", "주점 x", "float", 640.0, 0, 8000, 0.5, group=C, unit="px", help="'직접 입력'일 때만 씁니다"),
        ParamSpec("pp_y", "주점 y", "float", 360.0, 0, 8000, 0.5, group=C, unit="px", help="'직접 입력'일 때만 씁니다"),
        ParamSpec("cube_mm", "큐브 높이", "float", 25.0, 1, 200, 0.5, group=C, unit="mm",
                  help="옆면 띠 폭 = (큐브 높이 / 카메라 높이) x 주점에서의 거리"),
        ParamSpec("camera_mm", "카메라 높이 (판 위)", "float", 601.0, 50, 5000, 1, group=C, unit="mm",
                  help="판 윗면에서 카메라까지 높이"),
        ParamSpec("border_px", "화면 테두리 거리", "int", _G.border_px, 0, 100, 1, group=C, unit="px",
                  help="덩어리가 화면 가장자리에서 이만큼 안쪽까지 닿으면 touches_border 표시"),
        ParamSpec("edge_level", "가장자리 기준", "float", _G.edge_level, 0.05, 1, 0.05, group=T,
                  help="1픽셀 단면의 픽셀 수가 윗면 한 변의 이 비율에 닿는 곳을 덩어리의 가장자리로 봅니다"),
        ParamSpec("snap_tol", "옆면 띠 허용", "float", _G.snap_tol, 0, 0.5, 0.01, group=T,
                  help="한 축 길이가 윗면의 (1 + 이 값) 배보다 길면 옆면 띠가 붙어 있다고 보고 잘라냅니다"),
        ParamSpec("shade_snap", "밝기 경계로 윗면 맞추기", "bool", _G.shade_snap, group=T,
                  help="윗면과 옆면 사이 밝기 차이가 보이면 그 경계를 윗면 안쪽 모서리로 씁니다"),
        ParamSpec("shade_t", "밝기 경계 최소 차이", "float", _G.shade_t, 0, 50, 0.5, group=T, unit="L*",
                  help="윗면과 옆면 밝기 차이가 이보다 작으면 경계로 보지 않습니다"),
        ParamSpec("shade_search", "밝기 경계 찾는 구간", "range", list(_G.shade_search), 0, 3, 0.05, group=T, unit="면",
                  help="바깥 모서리에서 윗면 한 변의 몇 배 안쪽 사이에서 경계를 찾습니다"),
        ParamSpec("shade_min_count", "밝기 단면 최소 폭", "float", _G.shade_min_count, 0, 1, 0.05, group=T, unit="면",
                  help="단면 픽셀 수가 윗면 한 변의 이 비율보다 적은 줄은 밝기 비교에서 뺍니다"),
        ParamSpec("shade_window", "밝기 비교 폭", "int", _G.shade_window, 1, 15, 1, group=T, unit="px",
                  help="경계 후보 양쪽 몇 픽셀의 평균 밝기를 비교할지"),
    ]
    if not outside:
        out.append(ParamSpec("split", "붙은 큐브 나누기", "bool", _G.split, group=P,
                             help="한 덩어리로 붙은 큐브를 윗면 크기로 나눕니다"))
    out.append(ParamSpec("single_max", "한 개로 보는 최대 길이", "float", _G.single_max, 1, 4, 0.05, group=P, unit="면",
                         help="옆면 띠를 뺀 길이가 윗면의 이 배보다 짧으면 큐브 한 개 (탑 꼭대기는 최대 1.53배)"))
    if not outside:
        out += [
            ParamSpec("block_fill", "빈틈없는 묶음 기준", "float", _G.block_fill, 0, 1, 0.01, group=P,
                      help="여러 개 덩어리가 외접 사각형을 이 비율 이상 채우면 바둑판처럼 윗면을 깔아 나눕니다"),
            ParamSpec("tile_coverage", "묶음 칸 최소 채움", "float", _G.tile_coverage, 0, 1, 0.05, group=P,
                      help="바둑판 칸을 덩어리가 이 비율보다 덜 채우면 빈 모서리(L자 묶음)로 보고 뺍니다"),
            ParamSpec("split_peak", "나누기 씨앗 깊이", "float", _G.split_peak, 0, 2, 0.05, group=P,
                      help="거리 변환 값이 윗면 반 변의 이 배를 넘는 곳을 큐브마다의 씨앗으로 워터셰드 분할"),
            ParamSpec("split_iters", "k-평균 나누기 반복", "int", _G.split_iters, 1, 100, 1, group=P,
                      help="워터셰드가 실패하면 쓰는 위치 k-평균의 최대 반복 횟수"),
        ]
    out += [
        ParamSpec("angle_disagree", "각도 불일치 기준", "float", _G.angle_disagree, 0, 45, 0.5, group=P, unit="°",
                  help="최소 사각형 각도와 윤곽선 각도가 이만큼 다르면 여러 큐브 묶음으로 보고 윤곽선 각도를 씁니다"),
        ParamSpec("outline_eps", "윤곽선 단순화", "float", _G.outline_eps, 0.5, 10, 0.5, group=P, unit="px",
                  help="윤곽선 각도를 잴 때 윤곽을 이 허용 오차로 다각형 근사합니다"),
    ]
    if not outside:
        out += [
            ParamSpec("drop_side_faces", "옆면 후보 버리기", "bool", _G.drop_side_faces, group=P,
                      help="다른 큐브의 실루엣 안에 거의 들어가는 후보는 그 큐브의 옆면으로 보고 버립니다"),
            ParamSpec("side_face_frac", "옆면 판정 비율", "float", _G.side_face_frac, 0, 1, 0.05, group=P,
                      help="후보 픽셀이 다른 큐브 실루엣 안에 이 비율 이상이면 옆면"),
        ]
    out += [
        ParamSpec("partial_contrast", "진짜 모서리 밝기 차", "float", _G.partial_contrast, 0, 255, 1, group=K,
                  help="가려진 큐브의 두 끝 바깥 밝기 차이가 이보다 작으면 어느 쪽이 진짜 모서리인지 몰라 중심만 냅니다"),
        ParamSpec("partial_probe", "바깥 밝기 재는 거리", "range", [3, 7], 0, 40, 1, group=K, unit="px",
                  help="덩어리 끝에서 이 거리 범위의 판 밝기를 잽니다"),
        ParamSpec("partial_probe_n", "바깥 거리 표본 수", "int", _G.partial_probe_n, 1, 15, 1, group=K,
                  help="위 거리 범위를 몇 점으로 나눠 잴지"),
        ParamSpec("partial_across_n", "가로 표본 수", "int", _G.partial_across_n, 1, 50, 1, group=K,
                  help="덩어리 폭 방향으로 몇 점에서 잴지"),
        ParamSpec("partial_trim", "가로 양끝 제외", "float", _G.partial_trim, 0, 0.49, 0.01, group=K,
                  help="덩어리 폭의 양끝에서 이 비율씩은 재지 않습니다"),
        ParamSpec("partial_penalty", "가려진 큐브 점수 배율", "float", _G.partial_penalty, 0, 1, 0.05, group=R,
                  help="일부가 가려진 큐브의 점수에 곱합니다"),
    ]
    if not outside:
        out.append(ParamSpec("border_penalty", "테두리 큐브 점수 배율", "float", _G.border_penalty, 0, 1, 0.05, group=R,
                             help="화면 테두리에 닿은 큐브의 점수에 곱합니다"))
    return out


def box_params() -> list[ParamSpec]:
    """How a detector's axis-aligned box becomes a cube mask (postprocess.cube_from_box)."""
    return [
        ParamSpec("box_pad", "박스 주변 여유", "float", _G.box_pad, 0, 2, 0.05, group=G_BOX, unit="면",
                  help="박스 둘레로 윗면 한 변(없으면 박스 짧은 변)의 이 배만큼 더 잘라 봅니다"),
        ParamSpec("box_seed", "씨앗 색 영역 여백", "float", _G.box_seed, 0, 0.49, 0.01, group=G_BOX,
                  help="박스 가장자리를 한쪽마다 이 비율씩 빼고 남은 가운데의 중앙값을 큐브 색으로 봅니다"),
        ParamSpec("box_w_l", "밝기(L*) 가중치", "float", _G.box_w_l, 0, 2, 0.05, group=G_BOX,
                  help="씨앗 색과의 거리에서 L* 차이를 얼마나 볼지"),
        ParamSpec("box_t", "색 거리 문턱 범위", "range", [10, 30], 0, 100, 1, group=G_BOX,
                  help="Otsu로 정한 색 거리 문턱을 이 범위로 자릅니다 (Lab)"),
        ParamSpec("box_size", "마스크 크기 허용", "range", list(_G.box_size), 0, 5, 0.05, group=G_BOX, unit="박스",
                  help="다시 만든 마스크 넓이가 박스 넓이의 이 범위 밖이면 박스 자체를 씁니다"),
        ParamSpec("box_blur", "박스 흐림 크기", "int", _G.box_blur, 1, 15, 2, group=G_BOX, unit="px",
                  help="마스크를 만들기 전 가우시안 흐림 (홀수, 1 = 끔)"),
    ]


def morph_params(group: str = G_PRE, *, open_k: int = 3, close_k: int = 5) -> list[ParamSpec]:
    """Morphological opening and closing of a mask (cv2.morphologyEx)."""
    return [
        ParamSpec("open_ksize", "열기 크기", "int", open_k, 1, 31, 2, group=group, unit="px", help="작은 점 제거 (1 = 끔)"),
        ParamSpec("open_iters", "열기 반복", "int", 1, 1, 10, 1, group=group, help="열기를 몇 번 할지"),
        ParamSpec("close_ksize", "닫기 크기", "int", close_k, 1, 31, 2, group=group, unit="px", help="작은 틈 메우기 (1 = 끔)"),
        ParamSpec("close_iters", "닫기 반복", "int", 1, 1, 10, 1, group=group, help="닫기를 몇 번 할지"),
        ParamSpec("morph_shape", "모폴로지 모양", "select", "rect",
                  options=(("rect", "사각형"), ("ellipse", "타원"), ("cross", "십자")), group=group,
                  help="열기·닫기 커널 모양"),
    ]


_MORPH_SHAPES = {"rect": cv2.MORPH_RECT, "ellipse": cv2.MORPH_ELLIPSE, "cross": cv2.MORPH_CROSS}


def apply_morph(mask: np.ndarray, p: dict[str, Any]) -> np.ndarray:
    """Opening then closing with the morph_params values."""
    shape = _MORPH_SHAPES.get(p.get("morph_shape", "rect"), cv2.MORPH_RECT)
    ko, kc = odd(p.get("open_ksize", 3)), odd(p.get("close_ksize", 5))
    if ko > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(shape, (ko, ko)),
                                iterations=int(p.get("open_iters", 1)))
    if kc > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(shape, (kc, kc)),
                                iterations=int(p.get("close_iters", 1)))
    return mask


def blur_params(group: str = G_PRE, *, ksize: int = 5, help: str = "센서 잡음 제거 (홀수, 1 = 끔)") -> list[ParamSpec]:
    """Gaussian blur (cv2.GaussianBlur)."""
    return [
        ParamSpec("blur_ksize", "흐림 크기", "int", ksize, 1, 31, 2, group=group, unit="px", help=help),
        ParamSpec("blur_sigma", "흐림 시그마", "float", 0.0, 0, 20, 0.1, group=group,
                  help="0 = 크기에서 자동 계산"),
    ]


def apply_blur(img: np.ndarray, p: dict[str, Any]) -> np.ndarray:
    k = odd(p.get("blur_ksize", 5))
    return cv2.GaussianBlur(img, (k, k), float(p.get("blur_sigma", 0.0))) if k > 1 else img


def blob_params(group: str = G_SIZE) -> list[ParamSpec]:
    """How mask pixels join into candidate blobs (cv2.connectedComponentsWithStats)."""
    return [
        ParamSpec("connectivity", "덩어리 연결", "select", "8",
                  options=(("8", "8방향"), ("4", "4방향")), group=group,
                  help="대각선으로만 닿은 픽셀도 같은 덩어리로 볼지 (8) 아닐지 (4)"),
    ]


def blobs_kwargs(ctx: "Context") -> dict[str, Any]:
    """Keyword arguments of postprocess.blobs_from_mask for this frame."""
    p = ctx.params
    return {
        "min_area": 0.5 * p["min_area_ratio"] * ctx.face_px**2,
        "fill_holes": ctx.geo.fill_holes,
        "fill_max": ctx.geo.hole_fill_max,
        "connectivity": int(p.get("connectivity", "8")),
    }


def prototypes_from(params: dict[str, Any]) -> dict[str, str]:
    protos = {k: params.get(f"proto_{k}", DEFAULT_PROTOTYPES[k]) for k in ("red", "green", "blue", "wood")}
    protos["board"] = params.get("proto_board", DEFAULT_BOARD)
    return protos


def _pair(v: Any, default: tuple[float, float]) -> tuple[float, float]:
    return (float(v[0]), float(v[1])) if v is not None else default


def geometry_from(params: dict[str, Any]) -> Geometry:
    """Geometry from resolved params; a param the model does not have keeps its default."""
    g = params.get
    pp = (float(g("pp_x", 640.0)), float(g("pp_y", 360.0))) if g("pp_mode", "center") == "manual" else None
    return Geometry(
        face_px=float(g("face_px", FACE_PX)),
        pp=pp,
        kappa=float(g("cube_mm", 25.0)) / max(float(g("camera_mm", 601.0)), 1e-9),
        min_area_ratio=float(g("min_area_ratio", _G.min_area_ratio)),
        max_area_ratio=float(g("max_area_ratio", _G.max_area_ratio)),
        sliver_ratio=float(g("sliver_ratio", _G.sliver_ratio)),
        partial_ratio=float(g("partial_ratio", _G.partial_ratio)),
        snap_tol=float(g("snap_tol", _G.snap_tol)),
        split=bool(g("split", _G.split)),
        shrink=float(g("shrink", _G.shrink)),
        border_px=int(g("border_px", _G.border_px)),
        shade_snap=bool(g("shade_snap", _G.shade_snap)),
        shade_t=float(g("shade_t", _G.shade_t)),
        drop_side_faces=bool(g("drop_side_faces", _G.drop_side_faces)),
        edge_level=float(g("edge_level", _G.edge_level)),
        single_max=float(g("single_max", _G.single_max)),
        block_fill=float(g("block_fill", _G.block_fill)),
        tile_coverage=float(g("tile_coverage", _G.tile_coverage)),
        split_peak=float(g("split_peak", _G.split_peak)),
        split_iters=int(g("split_iters", _G.split_iters)),
        min_coverage=float(g("min_coverage", _G.min_coverage)),
        angle_disagree=float(g("angle_disagree", _G.angle_disagree)),
        outline_eps=float(g("outline_eps", _G.outline_eps)),
        shade_search=_pair(g("shade_search"), _G.shade_search),
        shade_min_count=float(g("shade_min_count", _G.shade_min_count)),
        shade_window=int(g("shade_window", _G.shade_window)),
        partial_contrast=float(g("partial_contrast", _G.partial_contrast)),
        partial_probe=_pair(g("partial_probe"), _G.partial_probe),
        partial_probe_n=int(g("partial_probe_n", _G.partial_probe_n)),
        partial_across_n=int(g("partial_across_n", _G.partial_across_n)),
        partial_trim=float(g("partial_trim", _G.partial_trim)),
        side_face_frac=float(g("side_face_frac", _G.side_face_frac)),
        partial_penalty=float(g("partial_penalty", _G.partial_penalty)),
        border_penalty=float(g("border_penalty", _G.border_penalty)),
        fill_holes=bool(g("fill_holes", _G.fill_holes)),
        hole_fill_max=float(g("hole_fill_max", _G.hole_fill_max)),
        box_pad=float(g("box_pad", _G.box_pad)),
        box_seed=float(g("box_seed", _G.box_seed)),
        box_w_l=float(g("box_w_l", _G.box_w_l)),
        box_t=_pair(g("box_t"), _G.box_t),
        box_size=_pair(g("box_size"), _G.box_size),
        box_blur=int(g("box_blur", _G.box_blur)),
    )


def odd(k: int) -> int:
    return int(k) | 1


# --- per-pixel label tables over 8-bit (a*, b*) ------------------------------------------------
WEAK_FLAG = 16  # table bit: chroma below weak_factor x chroma_min (grey if the pixel is also dark)


def ab_grid() -> tuple[np.ndarray, np.ndarray]:
    """a* and b* (real units) of every 8-bit (a, b) table entry, indexed [a8, b8]."""
    a = np.arange(256, dtype=np.float32) - 128.0
    return np.meshgrid(a, a, indexing="ij")


def flag_weak_chroma(table: np.ndarray, colour: ColourModel) -> np.ndarray:
    """Set WEAK_FLAG on the 256x256 table entries whose chroma (vs the board) is weak."""
    aa, bb = ab_grid()
    weak = np.hypot(aa - colour.board_lab[1], bb - colour.board_lab[2]) < colour.weak_factor * colour.chroma_min
    return np.where(weak & (table > 0), table | WEAK_FLAG, table).astype(np.uint8)


def lookup(table: np.ndarray, lab8: np.ndarray, colour: ColourModel) -> np.ndarray:
    """Per-pixel table codes; pixels that are dark AND weakly chromatic get 0 (the per-pixel
    form of ColourModel.is_neutral: shadows, the arm, the table, noise in dark areas)."""
    code = table.reshape(-1)[(lab8[..., 1].astype(np.uint16) << 8) | lab8[..., 2]]
    dark = lab8[..., 0] < colour.dark_ratio * float(colour.board_lab[0]) * 2.55
    code[((code & WEAK_FLAG) > 0) & dark] = 0
    return code & ~np.uint8(WEAK_FLAG)


# --- run template ------------------------------------------------------------------------------
class Stopwatch:
    def __init__(self) -> None:
        self.t0 = self.t = time.perf_counter()
        self.stages: dict[str, float] = {}

    def lap(self, name: str) -> None:
        now = time.perf_counter()
        self.stages[name] = self.stages.get(name, 0.0) + (now - self.t) * 1000.0
        self.t = now

    def total(self) -> float:
        return (time.perf_counter() - self.t0) * 1000.0


@dataclass
class Context:
    """What a pipeline's ``candidates`` step gets: the frame, params and per-frame shared state."""

    frame: np.ndarray
    params: dict[str, Any]
    roi: np.ndarray | None  # 255 inside the detection region, None = whole frame
    colour: ColourModel
    face_px: float
    watch: Stopwatch
    geo: Geometry
    debug: bool
    layers: dict[str, np.ndarray] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    _balanced: np.ndarray | None = None

    def balanced(self) -> np.ndarray:
        """The frame in the reference light (board-referenced white balance and exposure),
        computed once per frame. Per-pixel colour decisions use this, never the raw frame."""
        if self._balanced is None:
            self._balanced = self.colour.balance(self.frame)
        return self._balanced

    def apply_roi(self, mask: np.ndarray) -> np.ndarray:
        return mask if self.roi is None else cv2.bitwise_and(mask, self.roi)


class BlobPipeline(Pipeline):
    """Template for pipelines that produce candidate blobs for the shared post-processing."""

    uses_prototypes = True

    def candidates(self, ctx: Context) -> list[Blob]:
        raise NotImplementedError

    def run(self, frame_bgr: np.ndarray, params: dict[str, Any], *, debug: bool = False) -> FrameResult:
        watch = Stopwatch()
        roi = roi_mask(frame_bgr.shape[:2], params.get("roi"))
        colour = colour_model(frame_bgr, prototypes_from(params), region=roi, **colour_opts_from(params))
        geo = geometry_from(params)
        ctx = Context(frame_bgr, params, roi, colour, geo.face_px, watch, geo, debug)
        watch.lap("판 색 측정")
        blobs = self.candidates(ctx)
        cubes, report = cubes_from_blobs(frame_bgr, blobs, colour=colour, geo=geo)
        cubes = filter_cubes(cubes, params.get("roi"))
        watch.lap("후처리")
        if debug:
            ctx.layers["candidates"] = draw_candidates(frame_bgr, report, cubes)
        cubes.sort(key=lambda c: (c.center[1], c.center[0]))
        notes = ctx.notes + report.notes() + light_notes(colour)
        return FrameResult(cubes=cubes, ms=watch.total(), stages_ms=watch.stages, debug=ctx.layers, notes=notes)


def light_notes(colour: ColourModel) -> list[str]:
    """Health checks of the light, from the board measured in this frame."""
    if colour.board_now is None:
        return ["흰 판을 찾지 못해 조명 보정을 하지 않았습니다"]
    out = []
    L0, a0, b0 = (float(v) for v in bgr_to_lab(colour.board_now.reshape(1, 3))[0])
    if colour.clipped > 0.02:
        out.append(f"화소 {colour.clipped * 100:.0f} %가 포화(노출 과다): 판 색을 믿을 수 없어 나무·판 구별과 조명 보정이 틀릴 수 있습니다")
    if L0 < 45:
        out.append(f"판이 어둡습니다 (L* {L0:.0f}): 노출 부족")
    if float(np.hypot(a0, b0)) > 12:
        out.append(f"판에 색이 끼어 있습니다 (채도 {np.hypot(a0, b0):.0f}): 판 기준으로 보정함, 화이트밸런스 고정 권장")
    return out


# --- debug drawing -----------------------------------------------------------------------------
STATUS_BGR = {
    "ok": (60, 200, 60),
    "split": (230, 200, 0),
    "block": (230, 200, 0),
    "large": (40, 40, 230),
    "sliver": (0, 140, 255),
    "no_colour": (160, 160, 160),
    "side_face": (200, 0, 200),
}


# Debug layer convention (FrameResult.debug): an HxW layer is a mask (0/255) that the UI tints
# over the frame; an HxWx3 layer is a picture shown as it is, so pictures carry the dimmed
# frame themselves.
def dimmed(frame: np.ndarray) -> np.ndarray:
    return cv2.convertScaleAbs(frame, alpha=0.45)


def union_mask(masks: dict[str, np.ndarray], shape_hw: tuple[int, int]) -> np.ndarray:
    """HxW 0/255 mask of all class masks together."""
    out = np.zeros(shape_hw, np.uint8)
    for m in masks.values():
        out |= m
    return out


def class_overlay(frame: np.ndarray, masks: dict[str, np.ndarray]) -> np.ndarray:
    """Picture: the dimmed frame with each class mask painted in its overlay colour."""
    out = dimmed(frame)
    for name, m in masks.items():
        out[m > 0] = OVERLAY_BGR.get(name, OVERLAY_BGR["unknown"])
    return out


def draw_candidates(frame: np.ndarray, report: Report, cubes: list[Cube]) -> np.ndarray:
    """Dimmed frame + every blob outline coloured by what post-processing did with it
    (green kept, cyan split, red large region, orange thin strip, grey board colour,
    magenta side face of another cube) +
    the fitted top faces in white."""
    out = dimmed(frame)
    for outline, status in report.records:
        if outline is None or len(outline) < 3:
            continue
        p = np.round(np.asarray(outline)).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [p], True, STATUS_BGR.get(status, (200, 0, 200)), 2, cv2.LINE_AA)
    for c in cubes:
        if c.quad is not None:
            q = np.round(np.asarray(c.quad)).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(out, [q], True, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(out, (int(round(c.center[0])), int(round(c.center[1]))), 3, (255, 255, 255), -1, cv2.LINE_AA)
    return out

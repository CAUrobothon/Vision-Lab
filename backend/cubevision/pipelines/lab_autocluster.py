"""cv.lab_autocluster — let the image state its own colours.

Remove the board (the pixels near the measured board colour in a*b*), cluster what remains
with k-means at quarter size, and name every cluster by the nearest prototype (or "other":
arm, pillar, table). Each pixel of the full frame then takes the class of its nearest cluster
centre. There are no hue or saturation thresholds, the centres follow today's light, and they
are reported in the notes as ready-made prototype suggestions for the other pipelines.

Stateless per frame (no re-use of centres between video frames) and repeatable: OpenCV's
k-means draws from OpenCV's global random generator, which is reset to a fixed seed first.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from cubevision.colour import bgr_to_hex, lab8_to_real
from cubevision.pipelines.common import (
    COLOUR_NAMES_KO, G_COLOUR, G_PRE, G_SIZE, BlobPipeline, Context, ab_grid, apply_blur, apply_morph, blob_params,
    blobs_kwargs, blur_params, class_overlay, colour_params, dimmed, flag_weak_chroma, geometry_params, lookup,
    morph_params, roi_param,
)
from cubevision.postprocess import Blob, blobs_from_mask
from cubevision.types import ParamSpec

G_CLUSTER = "군집"
KMEANS_SEED = 0  # default of the "seed" param
_INTERP = {"area": cv2.INTER_AREA, "linear": cv2.INTER_LINEAR, "nearest": cv2.INTER_NEAREST, "cubic": cv2.INTER_CUBIC}


class LabAutocluster(BlobPipeline):
    key = "cv.lab_autocluster"
    name = "Lab 자동 군집"
    summary = "판을 뺀 픽셀을 k-평균으로 묶고 각 군집에 가장 가까운 기준색 이름을 붙이는, 문턱 없는 방식"
    description = (
        "발상: 판 색 근처 픽셀을 지운 뒤 남은 픽셀을 a*·b*·(약한) L*로 k-평균 군집화하고, 군집 중심마다 가장 가까운 기준색 이름을 붙입니다. "
        "가까운 기준색이 없으면 '기타'(팔·기둥·바닥)입니다. 전체 해상도에서는 픽셀마다 가장 가까운 군집 중심의 이름을 씁니다.\n"
        "강점: 색 문턱이 전혀 없고 그 장면의 실제 색으로 중심이 맞춰집니다. 같은 색의 밝은 면·그늘진 면이 이름으로 합쳐집니다. "
        "군집 중심이 메모에 나오므로 다른 파이프라인의 기준색을 고를 때 쓸 수 있습니다.\n"
        "약점: 군집 수가 장면과 맞지 않으면 한 색이 쪼개지거나 섞입니다. 판에 가까운 옅은 나무는 판과 함께 지워질 수 있습니다. "
        "미션 B에서는 바구니 바닥 픽셀이 많아 중심이 바닥 쪽으로 끌립니다. 모양은 보지 않으므로 크기 검사는 후처리가 합니다."
    )
    params = [
        ParamSpec("k_clusters", "군집 수", "int", 6, 2, 20, 1, group=G_CLUSTER,
                  help="판 말고 찾을 색 무리 수 (빨강·초록·파랑·나무·어두운 것·여분)"),
        ParamSpec("init", "시작 중심", "select", "kmeans++",
                  options=(("kmeans++", "k-means++ (무작위, 고정 시드)"), ("prototypes", "기준색에서 시작")),
                  group=G_CLUSTER, help="기준색에서 시작하면 중심이 기준색에서 오늘 조명 쪽으로 미끄러집니다"),
        ParamSpec("attempts", "시도 횟수", "int", 3, 1, 20, 1, group=G_CLUSTER,
                  help="가장 촘촘한 결과를 고릅니다 (k-means++일 때)"),
        ParamSpec("kmeans_max_iter", "최대 반복", "int", 20, 1, 300, 1, group=G_CLUSTER, help="k-평균 종료 조건: 반복 횟수"),
        ParamSpec("kmeans_eps", "수렴 기준", "float", 0.5, 0, 5, 0.05, group=G_CLUSTER,
                  help="k-평균 종료 조건: 중심이 이보다 덜 움직이면 멈춤"),
        ParamSpec("seed", "난수 시드", "int", KMEANS_SEED, 0, 99999, 1, group=G_CLUSTER,
                  help="같은 시드면 같은 결과가 나옵니다"),
        ParamSpec("dark_start_L", "어두운 시작 중심 L*", "float", 25.0, 0, 100, 1, group=G_CLUSTER,
                  help="'기준색에서 시작'일 때 기준색 4개 옆에 두는 어두운 중심의 밝기"),
        ParamSpec("w_L", "밝기 가중치", "float", 0.3, 0.0, 2.0, 0.05, group=G_CLUSTER,
                  help="a*·b* 옆에 L*를 얼마나 볼지. 0이면 밝기 무시"),
        ParamSpec("board_radius", "판 제거 반경", "float", 8.0, 0, 50, 0.5, group=G_CLUSTER,
                  help="판 색에서 a*b* 거리가 이 안이면 판으로 보고 군집에서 뺍니다"),
        ParamSpec("min_samples", "최소 표본 수", "int", 300, 50, 100000, 50, group=G_CLUSTER,
                  help="판이 아닌 픽셀이 이보다 적으면 찾을 것이 없다고 봅니다"),
        ParamSpec("name_max_dist", "이름 붙일 최대 거리", "float", 30.0, 0, 150, 1, group=G_COLOUR,
                  help="군집 중심이 기준색과 이보다 멀면 '기타'"),
        *colour_params(),
        ParamSpec("cluster_scale", "군집 계산 배율", "float", 0.25, 0.05, 1.0, 0.025, group=G_PRE,
                  help="작을수록 빠르지만 거칠어집니다"),
        ParamSpec("cluster_interp", "축소 보간", "select", "area",
                  options=(("area", "면적 평균"), ("linear", "선형"), ("nearest", "최근접"), ("cubic", "3차")), group=G_PRE,
                  help="군집 계산용으로 줄일 때의 보간 방식"),
        *blur_params(G_PRE, help="군집 전 잡음 제거 (홀수, 1 = 끔)"),
        *morph_params(G_SIZE),
        *blob_params(G_SIZE),
        *geometry_params(),
        roi_param(),
    ]
    debug_labels = {
        "clusters": "군집 (픽셀을 군집 중심 색으로 칠함)",
        "labels": "픽셀별 색 이름",
        "candidates": "후보 덩어리와 윗면",
    }

    def candidates(self, ctx: Context) -> list[Blob]:
        p: dict[str, Any] = ctx.params
        frame, colour = ctx.balanced(), ctx.colour
        blurred = apply_blur(frame, p)
        lab8 = cv2.cvtColor(blurred, cv2.COLOR_BGR2Lab)
        scale = float(p["cluster_scale"])
        if min(lab8.shape[:2]) * scale < 1:  # shrinking would leave no pixels (OpenCV asserts)
            ctx.notes.append("화면이 너무 작아 군집을 만들지 않았습니다")
            return []
        interp = _INTERP.get(p["cluster_interp"], cv2.INTER_AREA)
        small8 = cv2.resize(lab8, None, fx=scale, fy=scale, interpolation=interp) if scale < 1 else lab8
        small = lab8_to_real(small8)
        a0, b0 = float(colour.board_lab[1]), float(colour.board_lab[2])
        keep = np.hypot(small[..., 1] - a0, small[..., 2] - b0) >= p["board_radius"]
        if ctx.roi is not None:
            keep &= cv2.resize(ctx.roi, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        w_l = float(p["w_L"])
        samples = np.column_stack([small[keep][:, 1], small[keep][:, 2], w_l * small[keep][:, 0]]).astype(np.float32)
        ctx.watch.lap("전처리")
        if len(samples) < max(int(p["min_samples"]), int(p["k_clusters"])):
            ctx.notes.append("판이 아닌 픽셀이 너무 적어 군집을 만들지 않았습니다")
            return []

        k = int(p["k_clusters"])
        protos = np.column_stack([colour.lab[:, 1], colour.lab[:, 2], w_l * colour.lab[:, 0]]).astype(np.float32)
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, int(p["kmeans_max_iter"]), float(p["kmeans_eps"]))
        cv2.setRNGSeed(int(p["seed"]))
        if p["init"] == "prototypes":
            # Start from the prototypes plus "dark" and spare centres; labels = nearest start.
            extra = [np.array([a0, b0, w_l * float(p["dark_start_L"])], np.float32)]
            starts = np.vstack([protos, *extra])
            while len(starts) < k:
                starts = np.vstack([starts, samples[len(starts) * 7919 % len(samples)]])
            starts = starts[:k]
            init = np.argmin(((samples[:, None, :] - starts[None]) ** 2).sum(2), axis=1).astype(np.int32).reshape(-1, 1)
            _, labels, centres = cv2.kmeans(samples, k, init, crit, 1, cv2.KMEANS_USE_INITIAL_LABELS)
        else:
            _, labels, centres = cv2.kmeans(samples, k, None, crit, int(p["attempts"]), cv2.KMEANS_PP_CENTERS)
        ctx.watch.lap("k-평균")

        # Name each cluster after the nearest prototype (a*, b*, weighted L*), or "other" (0).
        # A neutral centre (board, shadow, or dark grey: table, arm, pillar) is "other" even
        # when a prototype is near: wood is the closest class to every warm grey.
        d = np.sqrt(((centres[:, None, :] - protos[None]) ** 2).sum(2))
        neutral = np.array([colour.is_neutral(_centre_lab(c, w_l).reshape(3)) for c in centres])
        named = (d.min(1) <= p["name_max_dist"]) & ~neutral
        names = np.where(named, d.argmin(1) + 1, 0).astype(np.uint8)
        # Full-resolution label map: nearest centre in a*b* via a 256x256 table (board = 0).
        aa, bb = ab_grid()
        dist = np.stack([np.hypot(aa - c[0], bb - c[1]) for c in centres], axis=0)
        table = names[dist.argmin(0)]
        table[np.hypot(aa - a0, bb - b0) < p["board_radius"]] = 0
        cls = lookup(flag_weak_chroma(table, colour), lab8, colour)
        if ctx.roi is not None:
            cls[ctx.roi == 0] = 0
        ctx.watch.lap("라벨")

        kw = blobs_kwargs(ctx)
        masks: dict[str, np.ndarray] = {}
        blobs: list[Blob] = []
        for i, name in enumerate(colour.names, start=1):
            m = apply_morph((cls == i).astype(np.uint8) * 255, p)
            masks[name] = m
            blobs += blobs_from_mask(m, name, **kw)
        ctx.watch.lap("덩어리")

        ctx.notes.append(_centre_note(centres, names, labels, colour.names, w_l))
        if ctx.debug:
            ctx.layers["clusters"] = _cluster_image(frame, keep, labels, centres, w_l)
            ctx.layers["labels"] = class_overlay(frame, masks)
        return blobs


def _centre_lab(c: np.ndarray, w_l: float) -> np.ndarray:
    L = float(c[2]) / w_l if w_l > 0 else 60.0
    return np.array([[[L, c[0], c[1]]]], np.float32)


def _centre_note(centres: np.ndarray, names: np.ndarray, labels: np.ndarray, class_names: tuple[str, ...], w_l: float) -> str:
    """'군집 중심 → 빨강 #c43a40 (12 %), ...' — the largest named cluster per class."""
    sizes = np.bincount(labels.ravel(), minlength=len(centres)) / max(len(labels), 1)
    parts = []
    for i in np.argsort(-sizes):
        name = class_names[names[i] - 1] if names[i] else None
        rgb = cv2.cvtColor(_centre_lab(centres[i], w_l), cv2.COLOR_Lab2BGR).reshape(3) * 255.0
        label = COLOUR_NAMES_KO.get(name, "기타") if name else "기타"
        parts.append(f"{label} {bgr_to_hex(rgb)} ({sizes[i] * 100:.0f} %)")
    return "군집 중심: " + ", ".join(parts)


def _cluster_image(frame: np.ndarray, keep: np.ndarray, labels: np.ndarray, centres: np.ndarray, w_l: float) -> np.ndarray:
    """Picture: every clustered pixel (at cluster scale) painted in its cluster centre's colour."""
    small = np.zeros((*keep.shape, 3), np.uint8)
    cols = np.array([cv2.cvtColor(_centre_lab(c, w_l), cv2.COLOR_Lab2BGR).reshape(3) * 255.0 for c in centres]).clip(0, 255).astype(np.uint8)
    small[keep] = cols[labels.ravel()]
    h, w = frame.shape[:2]
    big = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    hit = cv2.resize(keep.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    out = dimmed(frame)
    out[hit] = big[hit]
    return out

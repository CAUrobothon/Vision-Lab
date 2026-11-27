"""Test media for the Playwright specs (run with the backend's environment, offline).

usage: uv run --project ../backend python tests/e2e/make_media.py <out_dir> [salt]

Writes
- scene_a.jpg / scene_b.jpg: seeded synthetic boards from cubevision.synth.
- scene_small.jpg: scene_b scaled to 640x480 (Compare with media of a different size and shape).
- frames.mp4: 60 frames, 30 fps, 1280x720. Each frame encodes its own index as 10 black/white
  blocks along the top edge, and one red 61 px cube moves 8 px right per frame (centre x = 200 + 8n).
  The browser decodes the blocks to prove which frame is on screen and checks the overlay against it.
- folders/업로드/: a directory tree for the folder upload (see write_folder_tree).
- gt/: images with a ground-truth set for scoring.spec.ts (see write_gt_set).

The salt changes one corner pixel of every file, so each test run uploads files the lab has not
seen yet (it deduplicates by sha256) even when a running server is reused. regress/ holds the same
files with another salt for regressions.spec.ts, so its runs never show up in the other specs' media.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from cubevision.gt import SCHEMA_AUGMENTED
from cubevision.pipelines.hsv_ranges import HsvRanges
from cubevision.synth import make_scene

BITS = 10
BLOCK = 40  # px; block i spans x in [20 + 48 i, 60 + 48 i), y in [10, 50)
FRAMES = 60
FACE = 61


def code_frame(n: int, salt: int) -> np.ndarray:
    img = np.full((720, 1280, 3), 225, np.uint8)  # light grey board
    img[719, 1279] = salt % 256
    img[0:60, :] = 128  # mid-grey band behind the code blocks
    for i in range(BITS):
        x = 20 + 48 * i
        img[10:10 + BLOCK, x:x + BLOCK] = 255 if (n >> i) & 1 else 0
    cx = 200 + 8 * n
    cy = 400
    h = FACE // 2
    img[cy - h:cy + h + 1, cx - h:cx + h + 1] = (60, 50, 200)  # BGR of printed red
    return img


def main(out: Path, salt: int) -> None:
    write_set(out, salt)
    # A different salt in both bytes: a change of 1 in the corner pixel can vanish in JPEG quantisation.
    write_set(out / "regress", salt ^ 0x8080)
    # Own files for the dataset spec, so its batch runs never show up in another spec's run strip.
    ds = out / "datasets"
    write_set(ds, salt ^ 0x4040)
    (ds / "scene_a.jpg").rename(ds / "ds_a.jpg")
    (ds / "scene_b.jpg").rename(ds / "ds_b.jpg")
    write_folder_tree(out / "folders", salt ^ 0x2020)
    write_gt_set(out / "gt", salt ^ 0x1010)


def write_folder_tree(out: Path, salt: int) -> None:
    """folders/업로드/ for the folder spec: a directory that holds only directories (원본/a.png,
    증강1/b.png, 증강1/sub/c.png), plus OS junk files the app must skip. Small PNGs, one seed each."""
    root = out / "업로드"
    for rel, seed in (("원본/a.png", 201), ("증강1/b.png", 202), ("증강1/sub/c.png", 203)):
        img, _ = make_scene(seed, mission="A", size=(640, 360))
        img[0:2, 0:2] = (salt % 256, (salt >> 8) % 256, 0)
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(root / rel), img)
    (root / ".DS_Store").write_bytes(b"\x00\x00\x00\x01Bud1")
    (root / "Thumbs.db").write_bytes(b"\xd0\xcf\x11\xe0")
    (root / "원본" / "._a.png").write_bytes(b"\x00\x05\x16\x07")


def write_set(out: Path, salt: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, seed, mission in (("scene_a.jpg", 101, "A"), ("scene_b.jpg", 102, "B")):
        img, _ = make_scene(seed, mission=mission)
        img[0:2, 0:2] = (salt % 256, (salt >> 8) % 256, 0)
        cv2.imwrite(str(out / name), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if name == "scene_b.jpg":
            small = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
            small[0:2, 0:2] = (salt % 256, (salt >> 8) % 256, 0)
            cv2.imwrite(str(out / "scene_small.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 92])
    writer = cv2.VideoWriter(str(out / "frames.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (1280, 720))
    for n in range(FRAMES):
        writer.write(code_frame(n, salt))
    writer.release()


# --- ground truth for scoring.spec.ts ------------------------------------------------------------
GT_SOURCES = (("s1", 302, "B"), ("s2", 303, "A"), ("s3", 304, "B"))  # clean HSV output, no 'unknown'
GT_VARIANTS = ("original", "dark_ev_m0p5", "cool")


def _variant(img: np.ndarray, name: str) -> np.ndarray:
    if name == "dark_ev_m0p5":
        return np.clip(img.astype(np.float32) * 0.75, 0, 255).astype(np.uint8)
    if name == "cool":
        return cv2.GaussianBlur(img, (5, 5), 1.2)
    return img


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jpg(path: Path, img: np.ndarray, salt: int) -> np.ndarray:
    """Writes the salted JPEG and returns what OpenCV (and so the lab's worker) decodes from it."""
    img = img.copy()
    img[0:2, 0:2] = (salt % 256, (salt >> 8) % 256, 0)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def _shift(cube: dict[str, Any], dx: float) -> None:
    cube["center"] = [cube["center"][0] + dx, cube["center"][1]]
    cube["detection_match_point"] = list(cube["center"])
    cube["quad"] = [[x + dx, y] for x, y in cube["quad"]]


def _gt_cubes(preds: list[dict[str, Any]], w: int, h: int) -> tuple[list[dict[str, Any]], list[list[str]]]:
    """The pipeline's own output as labels (a perfect score), with the GT fields the lab reads."""
    cubes = []
    for i, p in enumerate(preds):
        cx, cy = p["center"]
        d = math.hypot(cx - w / 2, cy - h / 2)
        cubes.append({
            "id": f"{p['color']}_{i}", "color": p["color"], "valid_geometry": True, "center": list(p["center"]),
            "quad": [list(q) for q in p["quad"]], "angle": p["angle"], "detection_match_point": list(p["center"]),
            "region": "center" if d < 220 else "middle" if d < 420 else "outer",
            "image_side": "left" if cx < w / 2 else "right", "flags": [],
        })
    near = [[a["id"], b["id"]] for k, a in enumerate(cubes) for b in cubes[k + 1:]
            if math.dist(a["center"], b["center"]) < 75]
    return cubes, near


def _empty_spot(preds: list[dict[str, Any]], w: int, h: int) -> list[float]:
    """A point at least 120 px from every detection (a labelled cube nothing found)."""
    for y in range(120, h - 80, 20):
        for x in range(120, w - 80, 20):
            if all(math.dist((x, y), p["center"]) > 120 for p in preds):
                return [float(x), float(y)]
    raise RuntimeError("no empty spot for the missed cube")


def write_gt_set(out: Path, salt: int) -> None:
    """3 sources x (original, dark_ev_m0p5, cool) = 9 images, one layout group per source, labelled
    with cv.hsv_ranges' default output (so a batch of it scores 1.0) except for six known changes:

    - s1 original: cube 0 moved 3 px (passes at τc 5 px, fails at 2 px); cube 1 flagged weak_boundary
    - s2 original: cube 0 moved 8 px (fails: centre and vertex RMS 8 px)
    - s3 original: cube 0 has invalid geometry (match and colour only)
    - s1 dark_ev_m0p5: cube 0 labelled with another colour (colour wrong)
    - s2 dark_ev_m0p5: one more labelled cube where nothing is (missed)
    - s3 cool: cube 0 unlabelled (its detection is a false positive)

    Writes the images, labels.json (augmented schema), labels_link.json (the same plus gt_extra.jpg
    whose label carries another file's sha256: same name, so only a candidate until confirmed),
    expected.json (the hand-derived metrics) and slow.mp4 (a long video that keeps the worker busy).
    """
    out.mkdir(parents=True, exist_ok=True)
    pipe = HsvRanges()
    items: list[dict[str, Any]] = []
    n_preds = 0
    for src, seed, mission in GT_SOURCES:
        base, _ = make_scene(seed, mission=mission)
        source_sha = None
        for variant in GT_VARIANTS:
            name = f"{src}_{variant}.jpg"
            decoded = _write_jpg(out / name, _variant(base, variant), salt)
            preds = [c.to_dict() for c in pipe(decoded, {}).cubes]
            assert preds and all(p["color"] != "unknown" and p["quad"] and p["angle"] is not None for p in preds), name
            n_preds += len(preds)
            h, w = decoded.shape[:2]
            sha = _sha(out / name)
            source_sha = source_sha or sha
            cubes, near = _gt_cubes(preds, w, h)
            key = (src, variant)
            if key == ("s1", "original"):
                _shift(cubes[0], 3.0)
                cubes[1]["flags"] = ["weak_boundary"]
            elif key == ("s2", "original"):
                _shift(cubes[0], 8.0)
            elif key == ("s3", "original"):
                c = cubes[0]
                c.update(valid_geometry=False, center=None, quad=None, angle=None, approximate_region=c.pop("region"))
            elif key == ("s1", "dark_ev_m0p5"):
                cubes[0]["color"] = "blue" if cubes[0]["color"] != "blue" else "red"
            elif key == ("s2", "dark_ev_m0p5"):
                x, y = _empty_spot(preds, w, h)
                cubes.append({"id": "red_missed", "color": "red", "valid_geometry": True, "center": [x, y],
                              "quad": [[x - 30, y - 30], [x + 30, y - 30], [x + 30, y + 30], [x - 30, y + 30]],
                              "angle": 0.0, "detection_match_point": [x, y], "region": "middle",
                              "image_side": "left" if x < w / 2 else "right", "flags": []})
            elif key == ("s3", "cool"):
                dropped = cubes.pop(0)["id"]
                near = [p for p in near if dropped not in p]
            items.append({
                "media": f"images/{variant}/{name}", "filename": name, "sha256": sha, "source_sha256": source_sha,
                "augmentation": {"name": variant}, "layout_group": f"group_{src}", "width": w, "height": h,
                "near_top_face_pairs": near, "frames": [{"frame": 0, "cubes": cubes}],
            })
    meta = {"schema_version": SCHEMA_AUGMENTED, "label_kind": "e2e: cv.hsv_ranges default output with known changes"}
    (out / "labels.json").write_text(json.dumps({"metadata": meta, "items": items}, ensure_ascii=False))

    # gt_extra.jpg: the uploaded file and the labelled one differ in a corner pixel (two sha256).
    extra, _ = make_scene(305, mission="A")
    decoded = _write_jpg(out / "gt_extra.jpg", extra, salt)
    labelled = out / "labelled" / "gt_extra.jpg"
    labelled.parent.mkdir(exist_ok=True)
    _write_jpg(labelled, extra, salt ^ 0x00FF)
    cubes, near = _gt_cubes([c.to_dict() for c in pipe(decoded, {}).cubes], decoded.shape[1], decoded.shape[0])
    extra_item = {"media": "images/original/gt_extra.jpg", "filename": "gt_extra.jpg", "sha256": _sha(labelled),
                  "source_sha256": _sha(labelled), "augmentation": {"name": "original"}, "layout_group": "group_s4",
                  "width": decoded.shape[1], "height": decoded.shape[0], "near_top_face_pairs": near,
                  "frames": [{"frame": 0, "cubes": cubes}]}
    (out / "labels_link.json").write_text(
        json.dumps({"metadata": meta, "items": [*items, extra_item]}, ensure_ascii=False))

    # Hand-derived from the changes above (not from cubevision.score).
    n_gt = n_preds - 1 + 1  # s3 cool lost one label, s2 dark gained one
    n_valid = n_gt - 1  # s3 original's invalid cube
    matched = n_gt - 1  # the added label is missed
    expected = {
        "n_images": len(items), "n_preds": n_preds, "n_cubes": n_gt, "n_valid": n_valid,
        "tolerance": [n_valid - 3, n_valid],  # 8 px, wrong colour, missed
        "tolerance_tc2": [n_valid - 4, n_valid],  # + the 3 px cube
        "image_pass": [len(items) - 4, len(items)],  # s2 original, s1 dark, s2 dark, s3 cool
        "image_pass_tc2": [len(items) - 5, len(items)],  # + s1 original
        "recall": [matched, n_gt], "precision": [matched, n_preds], "color_accuracy": [matched - 1, matched],
        "failing": ["s2_dark_ev_m0p5.jpg", "s2_original.jpg", "s1_dark_ev_m0p5.jpg", "s3_cool.jpg"],
    }
    (out / "expected.json").write_text(json.dumps(expected, indent=1))

    # A long video for the worker to chew on while a batch waits behind it (about 450 frames).
    writer = cv2.VideoWriter(str(out / "slow.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (1280, 720))
    for n in range(450):
        writer.write(code_frame(n % FRAMES, salt))
    writer.release()


if __name__ == "__main__":
    main(Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 0)

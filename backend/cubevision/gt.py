"""Ground-truth sets for scoring (docs/SCORING.md §1).

``parse_gt(doc)`` reads an already-loaded JSON document of one of the two supported schemas
and returns frozen dataclasses. No file I/O here: the caller loads the JSON.

Both schemas share the cube fields; only the augmented one carries ``augmentation``,
``source_sha256`` and ``filename``. Missing values fall back as the spec says
(variant "original", source = own sha256, filename = last part of ``media``).
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Any

import numpy as np

SCHEMA_V11 = "robothon_top_face_gt_v1.1"
SCHEMA_AUGMENTED = "robothon_top_face_gt_augmented_v1.0"
SCHEMAS = (SCHEMA_V11, SCHEMA_AUGMENTED)

GT_COLORS: tuple[str, ...] = ("red", "green", "blue", "wood")
ORIGINAL = "original"

Point = tuple[float, float]


@dataclass(frozen=True)
class GtCube:
    """One labelled cube. Geometry (center / quad / angle) is None unless ``valid_geometry``."""

    id: str
    color: str  # one of GT_COLORS
    valid_geometry: bool
    center: Point | None
    quad: tuple[Point, Point, Point, Point] | None  # clockwise from the topmost vertex (label order)
    angle: float | None  # degrees, compared modulo 90
    match_point: Point  # detection_match_point: used for matching only, never for geometry errors
    region: str | None  # region, else approximate_region (center / middle / outer)
    image_side: str | None  # left / right
    near_contact: bool  # id appears in the image's near_top_face_pairs
    weak_boundary: bool  # 'weak_boundary' in flags
    flags: tuple[str, ...]


@dataclass(frozen=True)
class GtImage:
    index: int  # position in the document's items
    sha256: str
    filename: str
    media: str
    variant: str  # augmentation.name, else "original"
    source_sha256: str  # the original image's sha256 (own sha256 for an original)
    layout_group: str  # bootstrap resampling unit
    width: int | None
    height: int | None
    cubes: tuple[GtCube, ...]


@dataclass(frozen=True)
class GtSet:
    schema_version: str
    label_kind: str | None
    images: tuple[GtImage, ...]

    @property
    def n_images(self) -> int:
        return len(self.images)

    @property
    def n_cubes(self) -> int:
        return sum(len(im.cubes) for im in self.images)

    @property
    def n_valid_geometry(self) -> int:
        return sum(c.valid_geometry for im in self.images for c in im.cubes)

    @property
    def layout_groups(self) -> tuple[str, ...]:
        return tuple(sorted({im.layout_group for im in self.images}))

    @property
    def variants(self) -> tuple[str, ...]:
        """Variant names in first-appearance order."""
        return tuple(dict.fromkeys(im.variant for im in self.images))


def parse_gt(doc: dict[str, Any]) -> GtSet:
    """Parse a ground-truth document. Raises ValueError (Korean message) for an unknown schema
    or a malformed item, naming the item and the field."""
    if not isinstance(doc, dict):
        raise ValueError("정답셋은 JSON 객체여야 합니다")
    meta = doc.get("metadata")
    schema = meta.get("schema_version") if isinstance(meta, dict) else None
    if schema not in SCHEMAS:
        raise ValueError(f"지원하지 않는 정답셋 형식입니다: {schema!r} (지원: {', '.join(SCHEMAS)})")
    items = doc.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("정답셋에 items 목록이 없거나 비어 있습니다")
    images = tuple(_image(i, item) for i, item in enumerate(items))
    seen: dict[str, int] = {}
    for im in images:
        if im.sha256 in seen:
            raise ValueError(f"정답 이미지 {seen[im.sha256]}번과 {im.index}번의 sha256이 같습니다: {im.sha256}")
        seen[im.sha256] = im.index
    label_kind = meta.get("label_kind")
    return GtSet(schema, label_kind if isinstance(label_kind, str) else None, images)


# --- helpers (also used by cubevision.score for model outputs) ---------------------------------
def finite_number(x: Any) -> float | None:
    """A real, finite number as float (numpy scalars included), else None. Booleans are not numbers."""
    if type(x) is not float and type(x) is not int:  # fast path first: the ABC check below is slow
        if isinstance(x, (bool, np.bool_)) or not isinstance(x, numbers.Real):
            return None
    v = float(x)
    return v if math.isfinite(v) else None


def finite_point(x: Any) -> Point | None:
    """Two finite numbers (list, tuple or array) as (x, y), else None."""
    if isinstance(x, (str, bytes)):
        return None
    try:
        if len(x) != 2:
            return None
    except TypeError:
        return None
    a, b = finite_number(x[0]), finite_number(x[1])
    return None if a is None or b is None else (a, b)


def finite_quad(x: Any) -> tuple[Point, Point, Point, Point] | None:
    """Exactly four finite points, else None."""
    if x is None or isinstance(x, (str, bytes)):
        return None
    try:
        if len(x) != 4:
            return None
    except TypeError:
        return None
    pts = [finite_point(p) for p in x]
    if any(p is None for p in pts):
        return None
    return (pts[0], pts[1], pts[2], pts[3])  # type: ignore[return-value]


def _text(x: Any) -> str | None:
    return x if isinstance(x, str) and x else None


def _image(index: int, item: Any) -> GtImage:
    where = f"정답 이미지 {index}번"
    if not isinstance(item, dict):
        raise ValueError(f"{where}: 객체가 아닙니다")
    media = _text(item.get("media"))
    if media:
        where = f"정답 이미지 {index}번({media})"
    sha = _text(item.get("sha256"))
    if sha is None:
        raise ValueError(f"{where}: sha256이 없습니다")
    group = _text(item.get("layout_group"))
    if group is None:
        raise ValueError(f"{where}: layout_group이 없습니다")
    filename = _text(item.get("filename")) or (media.replace("\\", "/").rsplit("/", 1)[-1] if media else None)
    if filename is None:
        raise ValueError(f"{where}: filename과 media가 모두 없습니다")
    aug = item.get("augmentation")
    variant = ORIGINAL
    if aug is not None:
        if not isinstance(aug, dict) or _text(aug.get("name")) is None:
            raise ValueError(f"{where}: augmentation.name이 없습니다")
        variant = aug["name"]
    source = item.get("source_sha256")
    if source is not None and _text(source) is None:
        raise ValueError(f"{where}: source_sha256이 문자열이 아닙니다")
    size = []
    for key in ("width", "height"):
        v = item.get(key)
        if v is not None and (isinstance(v, bool) or not isinstance(v, int) or v <= 0):
            raise ValueError(f"{where}: {key}가 양의 정수가 아닙니다 ({v!r})")
        size.append(v)

    frames = item.get("frames")
    if not isinstance(frames, list) or not frames or not isinstance(frames[0], dict):
        raise ValueError(f"{where}: frames[0]이 없습니다")
    raw_cubes = frames[0].get("cubes")
    if not isinstance(raw_cubes, list):
        raise ValueError(f"{where}: frames[0].cubes 목록이 없습니다")

    pairs = item.get("near_top_face_pairs") or []
    if not isinstance(pairs, list) or not all(isinstance(p, (list, tuple)) and len(p) == 2 for p in pairs):
        raise ValueError(f"{where}: near_top_face_pairs는 큐브 id 쌍의 목록이어야 합니다")
    near = {str(i) for p in pairs for i in p}

    cubes = tuple(_cube(f"{where} 큐브 {k}번", c, near) for k, c in enumerate(raw_cubes))
    ids = [c.id for c in cubes]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{where}: 큐브 id가 겹칩니다 ({', '.join(sorted({i for i in ids if ids.count(i) > 1}))})")
    unknown = near - set(ids)
    if unknown:
        raise ValueError(f"{where}: near_top_face_pairs에 없는 큐브 id가 있습니다 ({', '.join(sorted(unknown))})")
    return GtImage(index, sha, filename, media or filename, variant, source or sha, group, size[0], size[1], cubes)


def _cube(where: str, c: Any, near: set[str]) -> GtCube:
    if not isinstance(c, dict):
        raise ValueError(f"{where}: 객체가 아닙니다")
    cid = c.get("id")
    if not isinstance(cid, (str, int)) or isinstance(cid, bool) or cid == "":
        raise ValueError(f"{where}: id가 없습니다")
    cid = str(cid)
    where = f"{where}({cid})"
    color = c.get("color")
    if color not in GT_COLORS:
        raise ValueError(f"{where}: 색 {color!r}은(는) 정답 색이 아닙니다 (가능: {', '.join(GT_COLORS)})")
    valid = c.get("valid_geometry")
    if not isinstance(valid, bool):
        raise ValueError(f"{where}: valid_geometry가 true/false가 아닙니다")
    match_point = finite_point(c.get("detection_match_point"))
    if match_point is None:
        raise ValueError(f"{where}: detection_match_point가 유한한 [x, y]가 아닙니다")

    center = quad = angle = None
    if valid:
        center = finite_point(c.get("center"))
        angle = finite_number(c.get("angle"))
        quad = finite_quad(c.get("quad"))
        if center is None or angle is None or quad is None:
            missing = [k for k, v in (("center", center), ("quad", quad), ("angle", angle)) if v is None]
            raise ValueError(f"{where}: 기하 유효 큐브인데 {', '.join(missing)}이(가) 비었거나 형식이 틀렸습니다")

    flags = c.get("flags") or []
    if not isinstance(flags, list) or not all(isinstance(f, str) for f in flags):
        raise ValueError(f"{where}: flags는 문자열 목록이어야 합니다")
    region = _text(c.get("region")) or _text(c.get("approximate_region"))
    return GtCube(
        id=cid,
        color=color,
        valid_geometry=valid,
        center=center,
        quad=quad,
        angle=angle,
        match_point=match_point,
        region=region,
        image_side=_text(c.get("image_side")),
        near_contact=cid in near,
        weak_boundary="weak_boundary" in flags,
        flags=tuple(flags),
    )

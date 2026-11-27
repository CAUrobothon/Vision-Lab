"""Media items: ingest an uploaded file, turn rows into API objects, delete with all files."""

from __future__ import annotations

import hashlib
import shutil
import sqlite3
import uuid
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from lab import frames, video
from lab.config import DataPaths
from lab.db import dumps, loads, now_iso, transaction
from lab.errors import LabError
from lab.schemas import Media

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
BROWSER_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}  # shown as uploaded
ALPHA_IMAGE_EXTS = {".png", ".webp", ".tif", ".tiff"}  # may have transparent pixels
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}
THUMB_WIDTH = 320


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# Columns every media query adds to m.*: its run count and the datasets it belongs to.
EXTRA_COLUMNS = """
    (SELECT COUNT(*) FROM runs r WHERE r.media_id = m.id) AS run_count,
    (SELECT json_group_array(dm.dataset_id) FROM dataset_media dm WHERE dm.media_id = m.id) AS dataset_ids
"""


def get_row(conn: sqlite3.Connection, media_id: int) -> sqlite3.Row:
    row = conn.execute(f"SELECT m.*, {EXTRA_COLUMNS} FROM media m WHERE m.id = ?", (media_id,)).fetchone()
    if row is None:
        raise LabError(404, "미디어를 찾을 수 없습니다")
    return row


def url_version(row: sqlite3.Row) -> str:
    """Query value that makes a media URL unique to its file: ids restart at 1 in a new data
    dir, so /api/media/1/thumb alone could hit another folder's file in the browser cache."""
    return row["sha256"][:12]


def to_api(row: sqlite3.Row) -> Media:
    mid = row["id"]
    v = url_version(row)
    return Media(
        id=mid,
        kind=row["kind"],
        name=row["name"],
        sha256=row["sha256"],
        bytes=row["bytes"],
        width=row["width"],
        height=row["height"],
        fps=row["fps"],
        frame_count=row["frame_count"],
        duration=row["duration"],
        tags=loads(row["tags"], []),
        note=row["note"],
        starred=bool(row["starred"]),
        created_at=row["created_at"],
        run_count=row["run_count"] if "run_count" in row.keys() else 0,
        file_url=f"/api/media/{mid}/file?v={v}",
        original_url=f"/api/media/{mid}/original?v={v}",
        thumb_url=f"/api/media/{mid}/thumb?v={v}",
        dataset_ids=sorted(loads(row["dataset_ids"], [])) if "dataset_ids" in row.keys() else [],
        folder=row["ds_folder"] if "ds_folder" in row.keys() else None,
    )


def original_path(paths: DataPaths, row: sqlite3.Row) -> Path:
    return paths.media / f"{row['sha256']}{row['ext']}"


def display_path(paths: DataPaths, row: sqlite3.Row) -> Path:
    """The file the browser shows and the server decodes: the proxy if there is one."""
    return paths.root / row["proxy"] if row["proxy"] else original_path(paths, row)


def frame_source(paths: DataPaths, row: sqlite3.Row) -> frames.FrameSource:
    count = 1 if row["kind"] == "image" else int(row["frame_count"] or 0)
    return frames.FrameSource(kind=row["kind"], path=display_path(paths, row), frame_count=count)


def write_thumb(frame_bgr: np.ndarray, dst: Path, width: int = THUMB_WIDTH) -> None:
    h, w = frame_bgr.shape[:2]
    scale = width / w
    small = cv2.resize(frame_bgr, (width, max(1, round(h * scale))), interpolation=cv2.INTER_AREA) if w > width else frame_bgr
    tmp = dst.with_name(f".{dst.name}.{uuid.uuid4().hex}.jpg")
    if not cv2.imwrite(str(tmp), small, [cv2.IMWRITE_JPEG_QUALITY, 85]):
        raise RuntimeError(f"썸네일을 쓰지 못했습니다: {dst.name}")
    tmp.replace(dst)


def clean_tags(tags: list[str] | None) -> list[str]:
    out: list[str] = []
    for t in tags or []:
        t = str(t).strip()[:40]
        if t and t not in out:
            out.append(t)
    return out


def _colour_managed(path: Path) -> bool:
    """True when a browser would colour-manage the file (ICC profile, or PNG gamma without sRGB)."""
    try:
        with Image.open(path) as im:
            info = im.info
    except Exception:  # noqa: BLE001 - unreadable by Pillow: nothing to strip
        return False
    return bool(info.get("icc_profile")) or ("gamma" in info and "srgb" not in info)


def _jpeg_without_icc(src: Path, dst: Path) -> bool:
    """Copy a JPEG minus its ICC profile segments (APP2 "ICC_PROFILE"); image data and EXIF untouched."""
    data = src.read_bytes()
    if data[:2] != b"\xff\xd8":
        return False
    out = bytearray(data[:2])
    i = 2
    while i + 4 <= len(data) and data[i] == 0xFF:
        marker = data[i + 1]
        if marker == 0xDA:  # start of scan: the rest is compressed image data
            break
        end = i + 2 + int.from_bytes(data[i + 2 : i + 4], "big")
        if not (marker == 0xE2 and data[i + 4 : i + 16] == b"ICC_PROFILE\0"):
            out += data[i:end]
        i = end
    else:
        return False  # malformed header: the caller falls back to a PNG
    out += data[i:]
    dst.write_bytes(out)
    return True


def _png_proxy(paths: DataPaths, img: np.ndarray, work: list[Path]) -> Path:
    png = paths.tmp / f"proxy-{uuid.uuid4().hex}.png"
    work.append(png)
    if not cv2.imwrite(str(png), img, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
        raise RuntimeError("PNG 프록시를 쓰지 못했습니다")
    return png


def _flatten_alpha(path: Path, img: np.ndarray) -> np.ndarray | None:
    """``img`` (the IMREAD_COLOR decode of ``path``) composited onto white if the file has
    transparent pixels, else None.

    IMREAD_COLOR drops alpha and keeps whatever colour lies under transparent pixels, while a
    browser shows them see-through: the flattened copy is what both get to see.
    """
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None or raw.ndim != 3 or raw.shape[2] != 4 or raw.dtype not in (np.uint8, np.uint16):
        return None
    full = np.iinfo(raw.dtype).max
    if raw[..., 3].min() == full:
        return None  # an alpha channel, but every pixel is opaque
    # The alpha must line up with img (it would not if OpenCV rotated img by an EXIF tag).
    colour = raw[..., :3] if raw.dtype == np.uint8 else (raw[..., :3] >> 8).astype(np.uint8)
    if not np.array_equal(colour, img):
        return None
    a = raw[..., 3:].astype(np.float32) / full
    return (img * a + 255.0 * (1.0 - a) + 0.5).astype(np.uint8)


def _prepare_image(paths: DataPaths, src: Path, ext: str, work: list[Path]) -> tuple[np.ndarray, Path | None]:
    """Decode an image as OpenCV (and so the competition program) does, plus a browser proxy if needed.

    The browser must show exactly the decoded pixels, or the viewer's pixel values and colour
    picks would not match what the pipelines see. A proxy is made when browsers cannot show
    the format (TIFF, HEIC, ...), would colour-manage it (e.g. the Display P3 profile of
    iPhone photos, which OpenCV ignores: measured 26 units of red apart in Chromium), or the
    file has transparent pixels (flattened onto white for both the browser and the pipelines).
    """
    img = cv2.imread(str(src), cv2.IMREAD_COLOR) if ext in IMAGE_EXTS else None
    if img is None:
        converted = paths.tmp / f"convert-{uuid.uuid4().hex}.png"
        work.append(converted)
        try:
            video.image_to_png(src, converted)
        except video.FFmpegError as exc:
            raise LabError(415, f"이미지를 읽지 못했습니다. 파일이 손상됐거나 지원하지 않는 형식입니다 ({_brief(exc)})") from exc
        img = cv2.imread(str(converted), cv2.IMREAD_COLOR)
        if img is None:
            raise LabError(415, "이미지를 읽지 못했습니다")
        flat = _flatten_alpha(converted, img)
        img = img if flat is None else flat
        return img, _png_proxy(paths, img, work)  # re-written by OpenCV: no ICC chunk from ffmpeg
    flat = _flatten_alpha(src, img) if ext in ALPHA_IMAGE_EXTS else None
    if flat is not None:
        return flat, _png_proxy(paths, flat, work)  # frames are read from this proxy too
    if ext in BROWSER_IMAGE_EXTS and not _colour_managed(src):
        return img, None
    if ext in (".jpg", ".jpeg"):
        jpg = paths.tmp / f"proxy-{uuid.uuid4().hex}.jpg"
        work.append(jpg)
        if _jpeg_without_icc(src, jpg) and np.array_equal(cv2.imread(str(jpg), cv2.IMREAD_COLOR), img):
            return img, jpg  # lossless: same compressed data, no profile
    return img, _png_proxy(paths, img, work)


def _brief(exc: Exception, limit: int = 120) -> str:
    """The last meaningful line of a tool's error output, short enough for a toast (ffmpeg
    prints its whole banner and every decoder warning)."""
    lines = [ln.strip() for ln in str(exc).splitlines() if ln.strip()]
    last = lines[-1] if lines else type(exc).__name__
    return last if len(last) <= limit else last[: limit - 1] + "…"


def _prepare_video(paths: DataPaths, src: Path, info: dict | None, work: list[Path]) -> tuple[Path, video.ProxyInfo]:
    try:
        info = info or video.probe(src)
    except video.FFmpegError as exc:
        raise LabError(415, f"영상을 읽지 못했습니다. 파일이 손상됐거나 지원하지 않는 형식입니다 ({_brief(exc)})") from exc
    if not video.video_streams(info):
        raise LabError(415, "영상 트랙이 없는 파일입니다")
    mp4 = paths.tmp / f"proxy-{uuid.uuid4().hex}.mp4"
    work.append(mp4)
    try:
        video.make_proxy(src, mp4, video.proxy_fps(info))
        pinfo = video.proxy_info(mp4)
    except video.FFmpegError as exc:
        raise LabError(415, f"영상을 변환하지 못했습니다 ({_brief(exc)})") from exc
    if pinfo.frame_count < 1:
        raise LabError(415, "영상에 프레임이 없습니다")
    return mp4, pinfo


def ingest(
    conn: sqlite3.Connection,
    paths: DataPaths,
    src: Path,
    *,
    name: str,
    sha256: str,
    size: int,
    tags: list[str] | None = None,
) -> tuple[sqlite3.Row, bool]:
    """Store ``src`` (a file in data/tmp, consumed) as a media item. Returns (row, duplicate)."""
    work: list[Path] = [src]  # temporary files removed at the end, whatever happens
    stored: list[Path] = []  # final files removed if the insert fails
    try:
        existing = conn.execute("SELECT id FROM media WHERE sha256 = ?", (sha256,)).fetchone()
        if existing:
            return get_row(conn, existing["id"]), True

        ext = Path(name).suffix.lower()
        if not ext.isascii() or len(ext) > 12:
            ext = ""
        info = None
        if ext in IMAGE_EXTS:
            kind = "image"
        elif ext in VIDEO_EXTS:
            kind = "video"
        else:
            try:
                info = video.probe(src)
            except video.FFmpegError as exc:
                raise LabError(415, "지원하지 않는 파일 형식입니다 (이미지나 영상이 아닙니다)") from exc
            if not video.video_streams(info):
                raise LabError(415, "지원하지 않는 파일 형식입니다 (이미지나 영상이 아닙니다)")
            kind = "video" if video.looks_like_video(info) else "image"

        if kind == "image":
            first, proxy_tmp = _prepare_image(paths, src, ext, work)
            height, width = first.shape[:2]
            fps = frame_count = duration = None
            proxy_name = f"proxy/{sha256}{proxy_tmp.suffix}" if proxy_tmp else None
        else:
            proxy_tmp, pinfo = _prepare_video(paths, src, info, work)
            width, height, fps, frame_count = pinfo.width, pinfo.height, pinfo.fps, pinfo.frame_count
            duration = round(frame_count / fps, 3) if fps else None
            proxy_name = f"proxy/{sha256}.mp4"
            first = next(frames.iter_frames(frames.FrameSource("video", proxy_tmp, frame_count), 0, 1), (0, None))[1]
            if first is None:
                raise LabError(415, "영상의 첫 프레임을 읽지 못했습니다")

        thumb_tmp = paths.tmp / f"thumb-{uuid.uuid4().hex}.jpg"
        work.append(thumb_tmp)
        write_thumb(first, thumb_tmp)

        original = paths.media / f"{sha256}{ext}"
        shutil.move(src, original)
        stored.append(original)
        if proxy_tmp is not None and proxy_name is not None:
            proxy_final = paths.root / proxy_name
            shutil.move(proxy_tmp, proxy_final)
            stored.append(proxy_final)

        try:
            with transaction(conn):
                cur = conn.execute(
                    """INSERT INTO media (sha256, kind, name, ext, proxy, bytes, width, height, fps,
                                          frame_count, duration, tags, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (sha256, kind, name, ext, proxy_name, size, width, height, fps,
                     frame_count, duration, dumps(clean_tags(tags)), now_iso()),
                )
                media_id = int(cur.lastrowid)
        except sqlite3.IntegrityError:
            # The same file finished uploading in a parallel request a moment earlier.
            stored.clear()
            existing = conn.execute("SELECT id FROM media WHERE sha256 = ?", (sha256,)).fetchone()
            return get_row(conn, existing["id"]), True
        thumb_tmp.replace(paths.media_thumb(media_id))
        stored.clear()  # everything is in place
        return get_row(conn, media_id), False
    except BaseException:
        for p in stored:
            p.unlink(missing_ok=True)
        raise
    finally:
        for p in work:
            p.unlink(missing_ok=True)


def delete(conn: sqlite3.Connection, paths: DataPaths, media_id: int) -> None:
    """Delete a media item, its runs (rows cascade) and every file that belongs to them."""
    row = get_row(conn, media_id)
    run_ids = [r["id"] for r in conn.execute("SELECT id FROM runs WHERE media_id = ?", (media_id,))]
    frames.forget(display_path(paths, row))
    with transaction(conn):
        conn.execute("DELETE FROM media WHERE id = ?", (media_id,))
    files = [original_path(paths, row), paths.media_thumb(media_id)]
    if row["proxy"]:
        files.append(paths.root / row["proxy"])
    for rid in run_ids:
        files += [paths.run_thumb(rid), *paths.run_exports(rid)]
    for p in files:
        p.unlink(missing_ok=True)

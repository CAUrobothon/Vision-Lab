"""Read frame n of a media item as a BGR array.

Videos are always read from the proxy (no B-frames, keyframe every 15 frames), so seeking
with CAP_PROP_POS_FRAMES lands exactly on frame n; tests/lab/test_frames.py proves it with a
video whose frames encode their own index.

Random access (preview, /frame/n) keeps a few open captures and recently decoded frames in
this process. Runs and exports use ``iter_frames``, which decodes sequentially and never seeks
after the first frame.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MAX_CAPTURES = 4
MAX_FRAMES = 16


class FrameError(RuntimeError):
    pass


@dataclass(frozen=True)
class FrameSource:
    kind: str  # "image" | "video"
    path: Path  # image to decode (original or PNG proxy) or the video proxy
    frame_count: int  # 1 for images


@dataclass
class _Capture:
    cap: cv2.VideoCapture
    next_index: int  # frame that a plain read() would return; avoids a seek when stepping forward


_lock = threading.Lock()
_captures: OrderedDict[str, _Capture] = OrderedDict()
_frames: OrderedDict[tuple[str, int], np.ndarray] = OrderedDict()


def _read_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FrameError(f"이미지를 읽지 못했습니다: {path.name}")
    return img


def _capture(path: str) -> _Capture:
    entry = _captures.get(path)
    if entry is not None:
        _captures.move_to_end(path)
        return entry
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FrameError("영상을 열지 못했습니다")
    entry = _captures[path] = _Capture(cap, 0)
    while len(_captures) > MAX_CAPTURES:
        _, old = _captures.popitem(last=False)
        old.cap.release()
    return entry


def get_frame(src: FrameSource, n: int) -> np.ndarray:
    """Frame n as an HxWx3 uint8 BGR array (a copy the caller may modify)."""
    if not 0 <= n < src.frame_count:
        raise IndexError(n)
    key = (str(src.path), n)
    with _lock:
        cached = _frames.get(key)
        if cached is not None:
            _frames.move_to_end(key)
            return cached.copy()
        if src.kind == "image":
            frame = _read_image(src.path)
        else:
            entry = _capture(key[0])
            if entry.next_index != n:
                entry.cap.set(cv2.CAP_PROP_POS_FRAMES, n)
            ok, frame = entry.cap.read()
            entry.next_index = n + 1 if ok else -1
            if not ok or frame is None:
                raise FrameError(f"프레임 {n}을(를) 읽지 못했습니다")
        _frames[key] = frame
        while len(_frames) > MAX_FRAMES:
            _frames.popitem(last=False)
        return frame.copy()


def iter_frames(src: FrameSource, start: int, end: int, stride: int = 1) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (index, frame) for start, start+stride, ... < end, decoding sequentially.

    Skipped frames are grab()bed (demuxed and decoded without the colour conversion), so a
    run never pays for per-frame seeks.
    """
    if src.kind == "image":
        if start <= 0 < end:
            yield 0, _read_image(src.path)
        return
    cap = cv2.VideoCapture(str(src.path))
    if not cap.isOpened():
        raise FrameError("영상을 열지 못했습니다")
    try:
        if start > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for i in range(start, min(end, src.frame_count)):
            if (i - start) % stride == 0:
                ok, frame = cap.read()
                if not ok or frame is None:
                    return
                yield i, frame
            elif not cap.grab():
                return
    finally:
        cap.release()


def forget(path: Path) -> None:
    """Drop cached handles and frames of a file that is about to be deleted."""
    key = str(path)
    with _lock:
        entry = _captures.pop(key, None)
        if entry is not None:
            entry.cap.release()
        for k in [k for k in _frames if k[0] == key]:
            del _frames[k]

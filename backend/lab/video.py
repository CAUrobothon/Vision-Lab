"""ffmpeg / ffprobe calls: probing uploads, building video proxies, converting odd images.

The proxy is what both the browser and the server read, so frame n is the same picture on
both sides: H.264 yuv420p at a constant frame rate, rotation applied, even dimensions,
no B-frames and a keyframe every 15 frames (cheap, exact seeking), moov atom up front.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

PROXY_TIMEOUT_S = 3600
PROBE_TIMEOUT_S = 120


class FFmpegError(RuntimeError):
    """ffmpeg/ffprobe failed; ``str(exc)`` is the tail of its error output."""


def _tail(text: str, lines: int = 6) -> str:
    rows = [r for r in text.strip().splitlines() if r.strip()]
    return "\n".join(rows[-lines:]) or "알 수 없는 오류"


def run_ffmpeg(args: list[str], *, timeout: float = PROXY_TIMEOUT_S) -> None:
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y", *args]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise FFmpegError(f"ffmpeg 시간 초과 ({timeout:.0f}초)") from exc
    except FileNotFoundError as exc:
        raise FFmpegError("ffmpeg 를 찾을 수 없습니다 (PATH 확인)") from exc
    if proc.returncode != 0:
        raise FFmpegError(_tail(proc.stderr))


def probe(path: Path, *, count_packets: bool = False) -> dict[str, Any]:
    cmd = ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams"]
    if count_packets:
        cmd += ["-count_packets", "-select_streams", "v:0"]
    try:
        proc = subprocess.run([*cmd, str(path)], capture_output=True, text=True, timeout=PROBE_TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        raise FFmpegError("ffprobe 시간 초과") from exc
    except FileNotFoundError as exc:
        raise FFmpegError("ffprobe 를 찾을 수 없습니다 (PATH 확인)") from exc
    if proc.returncode != 0:
        raise FFmpegError(_tail(proc.stderr))
    return json.loads(proc.stdout or "{}")


def video_streams(info: dict[str, Any]) -> list[dict[str, Any]]:
    return [s for s in info.get("streams", []) if s.get("codec_type") == "video"]


def parse_rate(text: str | None) -> float:
    try:
        rate = Fraction(text or "0")
    except (ValueError, ZeroDivisionError):
        return 0.0
    return float(rate)


def looks_like_video(info: dict[str, Any]) -> bool:
    """For files without a known extension: more than one frame (or a real duration) = video."""
    streams = video_streams(info)
    if not streams:
        return False
    counts = [int(s["nb_frames"]) for s in streams if str(s.get("nb_frames", "")).isdigit()]
    if counts:
        return max(counts) > 1
    try:
        return float(info.get("format", {}).get("duration") or 0) >= 0.5
    except ValueError:
        return False


def proxy_fps(info: dict[str, Any]) -> int:
    """Rounded average frame rate of the first video stream, clamped to 1-60."""
    s = video_streams(info)[0]
    fps = parse_rate(s.get("avg_frame_rate")) or parse_rate(s.get("r_frame_rate")) or 30.0
    return int(min(60, max(1, round(fps))))


# Convert with the BT.709 matrix and say so in the stream: an untagged stream is decoded with
# whatever matrix the decoder guesses (Chromium and OpenCV assume BT.601, other players
# BT.709 for HD), and then the viewer's pixel values would not match the server's.
COLOUR = "out_color_matrix=bt709:out_range=tv"
COLOUR_TAGS = "setparams=range=tv:colorspace=bt709:color_primaries=bt709:color_trc=bt709"


def make_proxy(src: Path, dst: Path, fps: int) -> None:
    # Rotation metadata is applied by ffmpeg's default autorotate before these filters run.
    vf = f"fps={fps},scale=trunc(iw*sar/2)*2:trunc(ih/2)*2:{COLOUR},setsar=1,format=yuv420p,{COLOUR_TAGS}"
    run_ffmpeg([
        "-i", str(src), "-map", "0:v:0", "-an", "-sn", "-dn", "-vf", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-g", "15", "-keyint_min", "15", "-sc_threshold", "0", "-bf", "0",
        "-fps_mode", "cfr", "-movflags", "+faststart", str(dst),
    ])


@dataclass(frozen=True)
class ProxyInfo:
    width: int
    height: int
    fps: float
    frame_count: int


def proxy_info(path: Path) -> ProxyInfo:
    info = probe(path, count_packets=True)
    s = video_streams(info)[0]
    count = s.get("nb_read_packets") or s.get("nb_frames") or 0
    return ProxyInfo(
        width=int(s["width"]),
        height=int(s["height"]),
        fps=parse_rate(s.get("avg_frame_rate")) or parse_rate(s.get("r_frame_rate")),
        frame_count=int(count),
    )


def image_to_png(src: Path, dst: Path) -> None:
    """Decode formats OpenCV cannot read (HEIC, AVIF, ...) with ffmpeg; tile grids are stitched."""
    run_ffmpeg(["-i", str(src), "-frames:v", "1", "-update", "1", str(dst)], timeout=PROBE_TIMEOUT_S)

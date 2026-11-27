"""Upload, dedup, proxies, file endpoints, media list/patch."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from lab_testkit import VIDEO_FRAMES, decode_index, ffprobe_json, upload


def _decode(content: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)


def test_upload_image_and_dedup(client: TestClient, image: dict, media_dir: Path, data_dir: Path) -> None:
    assert image["kind"] == "image"
    assert (image["width"], image["height"]) == (320, 240)
    assert image["fps"] is None and image["frame_count"] is None
    assert image["file_url"] == f"/api/media/{image['id']}/file?v={image['sha256'][:12]}"
    again = upload(client, media_dir / "cubes.png", name="다른 이름.png")
    assert again["duplicate"] is True
    assert again["media"]["id"] == image["id"]
    assert (data_dir / "media" / f"{image['sha256']}.png").exists()
    assert not any((data_dir / "tmp").iterdir()), "upload temp files must be cleaned up"


def test_image_name_keeps_korean(client: TestClient, media_dir: Path, tmp_path: Path) -> None:
    img = cv2.imread(str(media_dir / "cubes.png"))
    img[0, 0] = (1, 2, 3)  # a different file (sha) than the session image
    src = tmp_path / "x.png"
    cv2.imwrite(str(src), img)
    m = upload(client, src, name="큐브 사진.png")["media"]
    assert m["name"] == "큐브 사진.png"
    r = client.get(m["original_url"])
    assert r.status_code == 200 and "attachment" in r.headers["content-disposition"]
    assert r.content == src.read_bytes()


def test_tiff_gets_a_png_for_the_browser(client: TestClient, media_dir: Path) -> None:
    m = upload(client, media_dir / "cubes.tif")["media"]
    r = client.get(m["file_url"])
    assert r.headers["content-type"] == "image/png"
    assert decode_index(_decode(r.content)) == 0
    assert client.get(m["original_url"]).content == (media_dir / "cubes.tif").read_bytes()


@pytest.mark.skipif(shutil.which("sips") is None, reason="needs macOS sips to make a HEIC file")
def test_heic_is_converted_with_ffmpeg(client: TestClient, media_dir: Path, tmp_path: Path) -> None:
    heic = tmp_path / "photo.heic"
    subprocess.run(["sips", "-s", "format", "heic", str(media_dir / "scene-1280.png"), "--out", str(heic)],
                   check=True, capture_output=True)
    m = upload(client, heic)["media"]
    assert m["kind"] == "image" and (m["width"], m["height"]) == (1280, 720)
    r = client.get(m["file_url"])
    assert r.headers["content-type"] == "image/png"


def test_rejects_files_that_are_not_media(client: TestClient, media_dir: Path, data_dir: Path) -> None:
    for name in ("notes.txt", "broken.jpg"):
        with open(media_dir / name, "rb") as f:
            r = client.post("/api/media", files={"file": (name, f, "application/octet-stream")})
        assert r.status_code == 415, r.text
        detail = r.json()["detail"]
        assert isinstance(detail, str) and detail
        # One short Korean sentence, not ffmpeg's whole stderr (found by exploratory QA).
        assert len(detail) < 250 and "\n" not in detail, detail
    assert not any((data_dir / "tmp").iterdir())


def test_upload_needs_a_file_field(client: TestClient) -> None:
    r = client.post("/api/media", files={"other": ("a.png", b"123", "image/png")})
    assert r.status_code == 400
    r = client.post("/api/media", content=b"raw", headers={"content-type": "application/octet-stream"})
    assert r.status_code == 400
    r = client.post("/api/media", files={"file": ("empty.png", b"", "image/png")})
    assert r.status_code == 400


def test_video_proxy_properties(video: dict, data_dir: Path) -> None:
    assert video["kind"] == "video"
    assert (video["width"], video["height"]) == (320, 240)  # 321x241 rounded down to even
    assert video["fps"] == 25
    assert video["frame_count"] == VIDEO_FRAMES
    assert video["duration"] == pytest.approx(VIDEO_FRAMES / 25)
    proxy = data_dir / "proxy" / f"{video['sha256']}.mp4"
    info = ffprobe_json(proxy, "-show_streams", "-show_format", "-count_packets")
    v = [s for s in info["streams"] if s["codec_type"] == "video"]
    assert len(v) == 1 and len(info["streams"]) == 1, "no audio or other streams"
    s = v[0]
    assert s["codec_name"] == "h264" and s["pix_fmt"] == "yuv420p"
    assert s["avg_frame_rate"] == s["r_frame_rate"] == "25/1", "constant frame rate"
    assert int(s["nb_read_packets"]) == VIDEO_FRAMES
    assert int(s.get("has_b_frames", 0)) == 0
    packets = ffprobe_json(proxy, "-select_streams", "v:0", "-show_entries", "packet=flags")["packets"]
    keyframes = [i for i, p in enumerate(packets) if "K" in p["flags"]]
    assert keyframes == list(range(0, VIDEO_FRAMES, 15))
    # +faststart: the moov atom comes before mdat.
    head = proxy.read_bytes()[:4096]
    assert head.find(b"moov") != -1 and (head.find(b"mdat") == -1 or head.find(b"moov") < head.find(b"mdat"))


def test_rotation_applied_and_audio_dropped(client: TestClient, media_dir: Path, data_dir: Path) -> None:
    m = upload(client, media_dir / "rotated.mov")["media"]
    assert (m["width"], m["height"]) == (240, 320)
    proxy = data_dir / "proxy" / f"{m['sha256']}.mp4"
    streams = ffprobe_json(proxy, "-show_streams")["streams"]
    assert [s["codec_type"] for s in streams] == ["video"]
    assert not any(sd.get("rotation") for sd in streams[0].get("side_data_list", []))


def test_ntsc_rate_becomes_30_fps_cfr(client: TestClient, media_dir: Path, data_dir: Path) -> None:
    m = upload(client, media_dir / "ntsc.mp4")["media"]
    assert m["fps"] == 30
    s = ffprobe_json(data_dir / "proxy" / f"{m['sha256']}.mp4", "-show_streams")["streams"][0]
    assert s["avg_frame_rate"] == "30/1" and int(s.get("has_b_frames", 0)) == 0


def test_file_endpoints(client: TestClient, video: dict) -> None:
    r = client.get(video["file_url"], headers={"Range": "bytes=0-99"})
    assert r.status_code == 206 and len(r.content) == 100
    assert r.headers["content-type"] == "video/mp4"
    thumb = _decode(client.get(video["thumb_url"]).content)
    assert thumb.shape[1] == 320
    frame = _decode(client.get(f"/api/media/{video['id']}/frame/42").content)
    assert frame.shape[:2] == (240, 320) and decode_index(frame) == 42
    r = client.get(f"/api/media/{video['id']}/frame/{VIDEO_FRAMES}")
    assert r.status_code == 400 and "범위" in r.json()["detail"]
    assert client.get("/api/media/99999").status_code == 404


def test_list_filters_and_patch(client: TestClient, image: dict, video: dict) -> None:
    kinds = {m["kind"] for m in client.get("/api/media", params={"kind": "video"}).json()}
    assert kinds == {"video"}
    r = client.patch(f"/api/media/{image['id']}", json={"name": " 판 사진 ", "tags": ["판", "판", " 조명 "],
                                                         "note": "메모", "starred": True})
    assert r.status_code == 200
    m = r.json()
    assert (m["name"], m["tags"], m["note"], m["starred"]) == ("판 사진", ["판", "조명"], "메모", True)
    assert [x["id"] for x in client.get("/api/media", params={"starred": True}).json()] == [image["id"]]
    assert [x["id"] for x in client.get("/api/media", params={"tag": "조명"}).json()] == [image["id"]]
    assert [x["id"] for x in client.get("/api/media", params={"q": "판 사"}).json()] == [image["id"]]
    ids = [x["id"] for x in client.get("/api/media").json()]
    assert ids == sorted(ids, reverse=True), "newest first"
    assert client.patch(f"/api/media/{image['id']}", json={"name": "  "}).status_code == 400


def _icc_bytes() -> bytes:
    from PIL import ImageCms

    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def test_images_with_a_colour_profile_get_a_profile_free_proxy(client: TestClient, media_dir: Path,
                                                               data_dir: Path, tmp_path: Path) -> None:
    """Browsers colour-manage embedded profiles, OpenCV does not: the browser must get the raw pixels."""
    from PIL import Image

    src = cv2.cvtColor(cv2.imread(str(media_dir / "cubes.png")), cv2.COLOR_BGR2RGB)
    src[2, 2] = (7, 7, 7)  # new sha
    Image.fromarray(src).save(tmp_path / "phone.jpg", quality=95, icc_profile=_icc_bytes())
    Image.fromarray(src).save(tmp_path / "phone.png", icc_profile=_icc_bytes())
    Image.fromarray(src).save(tmp_path / "plain.jpg", quality=95)

    jpg = upload(client, tmp_path / "phone.jpg")["media"]
    proxy = data_dir / "proxy" / f"{jpg['sha256']}.jpg"
    assert proxy.exists() and Image.open(proxy).info.get("icc_profile") is None
    original = cv2.imread(str(tmp_path / "phone.jpg"))
    assert np.array_equal(cv2.imread(str(proxy)), original), "lossless: same JPEG data without the profile"
    assert client.get(jpg["file_url"]).content == proxy.read_bytes()

    png = upload(client, tmp_path / "phone.png")["media"]
    served = client.get(png["file_url"])
    assert served.headers["content-type"] == "image/png"
    assert Image.open(data_dir / "proxy" / f"{png['sha256']}.png").info.get("icc_profile") is None
    assert np.array_equal(_decode(served.content), cv2.imread(str(tmp_path / "phone.png")))

    plain = upload(client, tmp_path / "plain.jpg")["media"]
    assert not list((data_dir / "proxy").glob(f"{plain['sha256']}.*")), "no proxy without a profile"


def test_video_proxy_is_tagged_bt709(video: dict, data_dir: Path) -> None:
    s = ffprobe_json(data_dir / "proxy" / f"{video['sha256']}.mp4", "-show_streams")["streams"][0]
    assert (s["color_space"], s["color_primaries"], s["color_transfer"], s["color_range"]) == (
        "bt709", "bt709", "bt709", "tv")


def test_media_urls_carry_the_file_version(client: TestClient, image: dict) -> None:
    """Ids restart at 1 in a new data dir: a cached /api/media/1/thumb would show another folder's
    picture. URLs carry the sha256 prefix, and only such a URL is marked cacheable."""
    v = image["sha256"][:12]
    base = f"/api/media/{image['id']}"
    assert (image["file_url"], image["original_url"], image["thumb_url"]) == (
        f"{base}/file?v={v}", f"{base}/original?v={v}", f"{base}/thumb?v={v}")
    assert client.get(image["thumb_url"]).headers["cache-control"] == "private, max-age=86400"
    for url in (f"{base}/thumb", f"{base}/thumb?v=000000000000", f"{base}/frame/0"):
        r = client.get(url)
        assert r.status_code == 200 and "cache-control" not in r.headers, url
    assert "cache-control" in client.get(f"{base}/frame/0?v={v}").headers


def test_transparent_png_is_flattened_onto_white(client: TestClient, data_dir: Path, tmp_path: Path) -> None:
    """Browsers show transparent pixels see-through; IMREAD_COLOR would give pipelines the colour
    hidden under them. Both must see the same picture: the image flattened onto white."""
    from lab import frames
    from lab import media as media_mod
    from lab.config import DataPaths
    from lab.db import connect

    bgra = np.zeros((40, 60, 4), np.uint8)
    bgra[..., :3] = (0, 0, 255)  # red everywhere ...
    bgra[:, :20, 3] = 0  # ... hidden under fully transparent pixels on the left,
    bgra[:, 20:40, 3] = 128  # half transparent in the middle,
    bgra[:, 40:] = (255, 0, 0, 255)  # opaque blue on the right
    cv2.imwrite(str(tmp_path / "alpha.png"), bgra)
    expected = np.zeros((40, 60, 3), np.uint8)
    expected[:, :20] = (255, 255, 255)
    expected[:, 20:40] = (127, 127, 255)  # half-transparent red over white
    expected[:, 40:] = (255, 0, 0)

    m = upload(client, tmp_path / "alpha.png")["media"]
    served = client.get(m["file_url"])
    assert served.headers["content-type"] == "image/png"
    assert served.content != (tmp_path / "alpha.png").read_bytes(), "the browser gets the flattened copy"
    assert np.array_equal(_decode(served.content), expected)
    conn = connect(data_dir / "lab.sqlite3")
    try:
        src = media_mod.frame_source(DataPaths(data_dir), media_mod.get_row(conn, m["id"]))
    finally:
        conn.close()
    assert np.array_equal(frames.get_frame(src, 0), expected), "previews see what the browser shows"
    assert np.array_equal(next(frames.iter_frames(src, 0, 1))[1], expected), "runs and exports too"
    assert client.get(m["original_url"]).content == (tmp_path / "alpha.png").read_bytes()

    opaque = bgra.copy()
    opaque[..., 3] = 255
    opaque[0, 0, 0] = 1  # new sha
    cv2.imwrite(str(tmp_path / "opaque.png"), opaque)
    o = upload(client, tmp_path / "opaque.png")["media"]
    assert client.get(o["file_url"]).content == (tmp_path / "opaque.png").read_bytes(), "opaque alpha: no proxy"

"""Datasets (groups of media) and batches (one run per media of a dataset), with the zip downloads:
the originals of a dataset, and a batch's drawn results + results.json + results.csv."""

from __future__ import annotations

import csv
import io
import json
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from lab_testkit import upload


def _wait_batch(client: TestClient, batch_id: int, timeout: float = 90.0) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        b = client.get(f"/api/batches/{batch_id}").json()
        if b["finished"]:
            return b
        assert time.monotonic() < deadline, f"batch {batch_id} unfinished: {b['counts']}"
        time.sleep(0.1)


def test_dataset_grouping(client: TestClient, image: dict, video: dict) -> None:
    r = client.post("/api/datasets", json={"name": "그룹 시험", "media_ids": [image["id"]]})
    assert r.status_code == 200, r.text
    ds = r.json()
    assert ds["media_count"] == 1 and ds["image_count"] == 1 and ds["video_count"] == 0
    assert client.post("/api/datasets", json={"name": "그룹 시험"}).status_code == 409, "names are unique"

    ds = client.post(f"/api/datasets/{ds['id']}/media", json={"media_ids": [video["id"], image["id"]]}).json()
    assert ds["media_count"] == 2 and ds["video_count"] == 1, "adding twice is harmless"
    listed = client.get("/api/media", params={"dataset_id": ds["id"]}).json()
    assert {m["id"] for m in listed} == {image["id"], video["id"]}
    assert ds["id"] in next(m for m in listed if m["id"] == image["id"])["dataset_ids"]
    assert ds["id"] in client.get(f"/api/media/{video['id']}").json()["dataset_ids"]

    assert client.post(f"/api/datasets/{ds['id']}/media", json={"media_ids": [999999]}).status_code == 404
    ds = client.patch(f"/api/datasets/{ds['id']}", json={"name": "그룹 시험 2", "note": "메모"}).json()
    assert ds["name"] == "그룹 시험 2" and ds["note"] == "메모"
    ds = client.post(f"/api/datasets/{ds['id']}/media/remove", json={"media_ids": [video["id"]]}).json()
    assert ds["media_count"] == 1
    assert client.get(f"/api/media/{video['id']}").status_code == 200, "removing from a dataset keeps the media"

    assert client.delete(f"/api/datasets/{ds['id']}").json()["ok"]
    assert client.get(f"/api/datasets/{ds['id']}").status_code == 404
    assert client.get(f"/api/media/{image['id']}").json()["dataset_ids"] == [], "only the grouping is gone"


def test_upload_into_a_dataset(client: TestClient, media_dir: Path, image: dict) -> None:
    ds = client.post("/api/datasets", json={"name": "올리기 시험"}).json()
    with open(media_dir / "cubes.png", "rb") as f:  # the same file as `image`: a duplicate
        r = client.post(f"/api/media?dataset_id={ds['id']}", files={"file": ("cubes.png", f, "image/png")})
    assert r.status_code == 200, r.text
    res = r.json()
    assert res["duplicate"] and res["media"]["id"] == image["id"] and ds["id"] in res["media"]["dataset_ids"]
    with open(media_dir / "cubes.png", "rb") as f:
        assert client.post("/api/media?dataset_id=999999", files={"file": ("cubes.png", f, "image/png")}).status_code == 404


def test_batch_results_and_downloads(client: TestClient, media_dir: Path, image: dict, video: dict) -> None:
    ds = client.post("/api/datasets", json={"name": "일괄/시험: 결과", "media_ids": [image["id"], video["id"]]}).json()
    empty = client.post("/api/datasets", json={"name": "빈 데이터셋"}).json()
    r = client.post(f"/api/datasets/{empty['id']}/batches", json={"model_key": "test.index"})
    assert r.status_code == 400

    r = client.post(f"/api/datasets/{ds['id']}/batches", json={"model_key": "test.index", "params": {"delay_ms": 0}, "stride": 10})
    assert r.status_code == 200, r.text
    batch = r.json()
    assert batch["counts"]["total"] == 2 and batch["dataset_name"] == "일괄/시험: 결과" and batch["stride"] == 10
    assert all(it["run_id"] for it in batch["items"])
    run = client.get(f"/api/runs/{batch['items'][0]['run_id']}").json()
    assert run["batch_id"] == batch["id"]

    batch = _wait_batch(client, batch["id"])
    assert batch["counts"]["done"] == 2, batch["counts"]
    assert [b["id"] for b in client.get(f"/api/datasets/{ds['id']}/batches").json()] == [batch["id"]]
    assert client.get(f"/api/datasets/{ds['id']}").json()["batch_count"] == 1

    # One item: the drawn result named after the media.
    img_item = next(it for it in batch["items"] if it["media_kind"] == "image")
    r = client.get(img_item["download_url"])
    assert r.status_code == 200 and "cubes_result.png" in r.headers["content-disposition"]
    drawn = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    assert drawn.shape[:2] == (image["height"], image["width"])

    # Everything: drawn image + drawn video + results.json + results.csv.
    r = client.get(batch["zip_url"])
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "application/zip"
    z = zipfile.ZipFile(io.BytesIO(r.content))
    names = set(z.namelist())
    assert {"cubes_result.png", "index_result.mp4", "results.json", "results.csv"} <= names, names
    meta = json.loads(z.read("results.json"))
    assert meta["batch"]["model_key"] == "test.index" and meta["batch"]["params"] == {"delay_ms": 0, "proto_red": "#c8323c", "roi": []}
    vid = next(it for it in meta["items"] if it["kind"] == "video")
    assert vid["file"] == "index_result.mp4" and [f["frame"] for f in vid["frames"]][:3] == [0, 10, 20]
    rows = list(csv.DictReader(io.StringIO(z.read("results.csv").decode("utf-8-sig"))))
    assert rows and {r["file"] for r in rows} == {"cubes_result.png", "index_result.mp4"}
    assert rows[0]["color"] == "red"

    r = client.get(f"/api/datasets/{ds['id']}/originals.zip")
    assert r.status_code == 200
    assert sorted(zipfile.ZipFile(io.BytesIO(r.content)).namelist()) == ["cubes.png", "index.mp4"]

    # Deleting the batch deletes its runs.
    run_ids = [it["run_id"] for it in batch["items"]]
    assert client.delete(f"/api/batches/{batch['id']}").json()["ok"]
    assert all(client.get(f"/api/runs/{rid}").status_code == 404 for rid in run_ids)
    assert client.get(f"/api/batches/{batch['id']}").status_code == 404


def test_cancel_batch(client: TestClient, image: dict) -> None:
    ds = client.post("/api/datasets", json={"name": "취소 시험", "media_ids": [image["id"]]}).json()
    batch = client.post(f"/api/datasets/{ds['id']}/batches",
                        json={"model_key": "test.index", "params": {"delay_ms": 3000}}).json()
    batch = client.post(f"/api/batches/{batch['id']}/cancel").json()
    batch = _wait_batch(client, batch["id"], timeout=30)
    assert batch["counts"]["cancelled"] == 1 and batch["counts"]["done"] == 0
    assert client.get(batch["zip_url"]).status_code == 409, "nothing finished to download"

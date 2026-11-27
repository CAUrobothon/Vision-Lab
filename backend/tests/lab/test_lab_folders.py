"""Folders inside a dataset: path rules, the tree operations (create, rename/move with the subtree,
delete taking media out of the dataset only), uploads into a folder and the report of files that
were in the dataset already, moving media, batches over one folder, and both zips keeping the
folder paths (exact entry names) with the folder in results.json and results.csv."""

from __future__ import annotations

import csv
import io
import json
import sqlite3
import time
import unicodedata
import zipfile
from pathlib import Path

import cv2
import pytest
from fastapi.testclient import TestClient

from lab.datasets import FOLDER_DEPTH_MAX, FOLDER_NAME_MAX, clean_folder, in_folder, is_within
from lab.errors import LabError
from lab_testkit import index_frame, upload, wait_run


def _wait_batch(client: TestClient, batch_id: int, timeout: float = 90.0) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        b = client.get(f"/api/batches/{batch_id}").json()
        if b["finished"]:
            return b
        assert time.monotonic() < deadline, f"batch {batch_id} unfinished: {b['counts']}"
        time.sleep(0.1)


def _image(tmp: Path, name: str, index: int) -> Path:
    """A PNG no other test uploads (its index blocks make it unique), saved under ``name``."""
    tmp.mkdir(parents=True, exist_ok=True)
    path = tmp / f"{index}-{name}"
    cv2.imwrite(str(path), index_frame(index))
    return path


def _upload_to(client: TestClient, path: Path, dataset_id: int | None, folder: str | None, name: str | None = None):
    with open(path, "rb") as f:
        return client.post(
            "/api/media" + (f"?dataset_id={dataset_id}" if dataset_id is not None else ""),
            data={} if folder is None else {"folder": folder},
            files={"file": (name or path.name, f, "image/png")},
        )


def _folders(client: TestClient, dataset_id: int) -> dict[str, tuple[int, int]]:
    ds = client.get(f"/api/datasets/{dataset_id}").json()
    return {f["path"]: (f["count"], f["total"]) for f in ds["folders"]}


def test_is_within_matches_the_sql_condition() -> None:
    """is_within (Python) and in_folder (SQL) say the same for every pair, prefixes and % _ included."""
    paths = ["", "a", "a/b", "a/b/c", "ab", "ab/c", "a_b", "a%", "a%/x", "증강", "증강/어둡게", "증강2"]
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t (folder TEXT)")
    conn.executemany("INSERT INTO t VALUES (?)", [(p,) for p in paths])
    for folder in paths:
        cond, args = in_folder("folder", folder)
        sql = {r[0] for r in conn.execute(f"SELECT folder FROM t WHERE {cond}", args)}
        assert sql == {p for p in paths if is_within(p, folder)}, folder
    assert is_within("a/b", "a") and not is_within("ab", "a") and is_within("x", "") and not is_within("", "a")


# --- path rules ----------------------------------------------------------------------------------
@pytest.mark.parametrize(("raw", "stored"), [
    ("", ""),
    ("   ", ""),
    ("원본", "원본"),
    (" 증강 / 어둡게 ", "증강/어둡게"),
    (unicodedata.normalize("NFD", "증강1/밝게"), "증강1/밝게"),  # macOS sends decomposed Hangul
    ("50% 밝게/a_b", "50% 밝게/a_b"),
    ("가" * FOLDER_NAME_MAX, "가" * FOLDER_NAME_MAX),
    ("/".join(["d"] * FOLDER_DEPTH_MAX), "/".join(["d"] * FOLDER_DEPTH_MAX)),
])
def test_folder_paths_that_are_kept(raw: str, stored: str) -> None:
    assert clean_folder(raw) == stored


@pytest.mark.parametrize("raw", [
    "..", "a/..", "../a", ".", "a/./b",  # no relative steps
    "/abs", "/", "C:/x", "c:x",  # no absolute paths
    "a//b", "a/", "a/ /b",  # no empty names
    "a\\b",  # no backslashes
    "a\x00b", "a\x1fb", "a\x7fb",  # no control characters
    "가" * (FOLDER_NAME_MAX + 1),
    "/".join(["d"] * (FOLDER_DEPTH_MAX + 1)),
])
def test_folder_paths_that_are_refused(raw: str) -> None:
    with pytest.raises(LabError) as err:
        clean_folder(raw)
    assert err.value.status == 400


def test_api_refuses_bad_folder_paths(client: TestClient) -> None:
    ds = client.post("/api/datasets", json={"name": "폴더 경로 시험"}).json()
    for bad in ["../밖", "/abs", "a//b", "a\\b", "x/" * 9 + "x", "가" * 81]:
        r = client.post(f"/api/datasets/{ds['id']}/folders", json={"path": bad})
        assert r.status_code == 400, (bad, r.text)
        assert r.json()["detail"]
    assert client.post(f"/api/datasets/{ds['id']}/folders", json={"path": " "}).status_code == 400
    assert client.post("/api/datasets/999999/folders", json={"path": "a"}).status_code == 404
    assert client.get("/api/media", params={"folder": "a"}).status_code == 400, "a folder needs its dataset"
    assert _folders(client, ds["id"]) == {}


# --- tree operations -----------------------------------------------------------------------------
def test_folder_tree_create_move_rename_delete(client: TestClient, image: dict, video: dict, data_dir: Path) -> None:
    ds = client.post("/api/datasets", json={"name": "폴더 나무 시험", "media_ids": [image["id"], video["id"]]}).json()
    did = ds["id"]
    folders = f"/api/datasets/{did}/folders"
    assert ds["folders"] == [] and ds["media_count"] == 2

    # Create: parents are made too, empty folders exist, names are unique among siblings (any letter case).
    r = client.post(folders, json={"path": "증강/어둡게"})
    assert r.status_code == 200, r.text
    assert [f["path"] for f in r.json()["folders"]] == ["증강", "증강/어둡게"]
    assert client.post(folders, json={"path": "증강"}).status_code == 409
    assert client.post(folders, json={"path": "Aug"}).status_code == 200
    r = client.post(folders, json={"path": "aug"})
    assert r.status_code == 409 and '"Aug"' in r.json()["detail"]
    r = client.post(folders, json={"path": "aug/정면"})  # a parent spelled in another case is the same folder
    assert [f["path"] for f in r.json()["folders"]] == ["Aug", "Aug/정면", "증강", "증강/어둡게"]

    # Move media between folders (the target is made if missing); only media of this dataset move.
    r = client.post(f"/api/datasets/{did}/media/move", json={"media_ids": [image["id"]], "folder": "증강/어둡게"})
    assert r.status_code == 200, r.text
    client.post(f"/api/datasets/{did}/media/move", json={"media_ids": [video["id"]], "folder": "원본"})
    other = upload(client, _image(data_dir.parent / "folder-src", "other.png", 301))["media"]
    r = client.post(f"/api/datasets/{did}/media/move", json={"media_ids": [other["id"]], "folder": "원본"})
    assert r.status_code == 404
    assert _folders(client, did) == {
        "Aug": (0, 0), "Aug/정면": (0, 0), "원본": (1, 1), "증강": (0, 1), "증강/어둡게": (1, 1),
    }

    # The media list of a folder includes its subfolders; each media says where it is.
    def listed(folder: str) -> dict[int, str]:
        return {m["id"]: m["folder"] for m in client.get("/api/media", params={"dataset_id": did, "folder": folder}).json()}

    assert listed("") == {image["id"]: "증강/어둡게", video["id"]: "원본"}
    assert listed("증강") == {image["id"]: "증강/어둡게"}
    assert listed("증강/어둡게") == {image["id"]: "증강/어둡게"}
    assert listed("원본") == {video["id"]: "원본"}
    assert listed("증") == {}, "a folder is matched by whole names, not by prefix"
    assert client.get(f"/api/media/{image['id']}").json()["folder"] is None, "no dataset given: no folder"

    # Rename moves the subtree and its media; a rename can also move a folder somewhere else.
    r = client.post(f"{folders}/rename", json={"path": "증강", "new_path": "변형"})
    assert r.status_code == 200, r.text
    assert listed("변형") == {image["id"]: "변형/어둡게"}
    r = client.post(f"{folders}/rename", json={"path": "변형", "new_path": "원본/변형"})
    assert r.status_code == 200, r.text
    assert _folders(client, did) == {
        "Aug": (0, 0), "Aug/정면": (0, 0), "원본": (1, 2), "원본/변형": (0, 1), "원본/변형/어둡게": (1, 1),
    }
    assert listed("원본") == {video["id"]: "원본", image["id"]: "원본/변형/어둡게"}
    assert client.post(f"{folders}/rename", json={"path": "원본", "new_path": "원본/안"}).status_code == 400
    assert client.post(f"{folders}/rename", json={"path": "원본/변형", "new_path": "aug"}).status_code == 409
    assert client.post(f"{folders}/rename", json={"path": "없음", "new_path": "x"}).status_code == 404
    assert client.post(f"{folders}/rename", json={"path": "", "new_path": "x"}).status_code == 400
    r = client.post(f"{folders}/rename", json={"path": "Aug", "new_path": "AUG"})  # a case-only rename
    assert r.status_code == 200 and "AUG/정면" in [f["path"] for f in r.json()["folders"]]
    deep = "깊이/2/3/4/5/6/7"
    client.post(folders, json={"path": deep})
    r = client.post(f"{folders}/rename", json={"path": "깊이", "new_path": "AUG/정면/3단"})  # would be 9 deep
    assert r.status_code == 400 and "8단계" in r.json()["detail"]
    r = client.post(f"{folders}/rename", json={"path": "깊이", "new_path": "AUG/2단"})  # exactly 8 deep
    assert r.status_code == 200 and "AUG/2단/2/3/4/5/6/7" in [f["path"] for f in r.json()["folders"]]

    # A run on the image, to show that deleting a folder keeps runs.
    run = client.post("/api/runs", json={"media_id": image["id"], "model_key": "test.index"}).json()
    assert wait_run(client, run["id"])["status"] == "done"

    # Delete takes the folder, its subfolders and their media out of THIS dataset only.
    r = client.post(f"{folders}/delete", json={"path": "원본"})
    assert r.status_code == 200, r.text
    ds = r.json()
    assert ds["media_count"] == 0
    assert [f["path"] for f in ds["folders"]] == [
        "AUG", "AUG/2단", "AUG/2단/2", "AUG/2단/2/3", "AUG/2단/2/3/4", "AUG/2단/2/3/4/5", "AUG/2단/2/3/4/5/6",
        "AUG/2단/2/3/4/5/6/7", "AUG/정면",
    ]
    for m in (image, video):
        again = client.get(f"/api/media/{m['id']}")
        assert again.status_code == 200 and did not in again.json()["dataset_ids"]
        assert client.get(m["original_url"]).status_code == 200, "the file stays"
    assert client.get(f"/api/runs/{run['id']}").json()["status"] == "done", "the run stays"
    assert client.post(f"{folders}/delete", json={"path": "원본"}).status_code == 404
    assert client.post(f"{folders}/delete", json={"path": ""}).status_code == 400

    # Deleting the dataset deletes its folders with it.
    assert client.delete(f"/api/datasets/{did}").json()["ok"]
    assert client.post(folders, json={"path": "x"}).status_code == 404


# --- uploads -------------------------------------------------------------------------------------
def test_upload_into_a_folder_and_report_files_already_in_the_dataset(client: TestClient, data_dir: Path) -> None:
    src = data_dir.parent / "folder-src"
    ds = client.post("/api/datasets", json={"name": "폴더 올리기 시험"}).json()
    did = ds["id"]

    a = _image(src, "a.png", 311)
    r = _upload_to(client, a, did, unicodedata.normalize("NFD", "원본/정면"))
    assert r.status_code == 200, r.text
    first = r.json()
    assert not first["duplicate"] and not first["already_in_dataset"] and first["folder"] == "원본/정면"
    assert _folders(client, did) == {"원본": (0, 1), "원본/정면": (1, 1)}

    # The same file again, aimed at another folder: it stays where it is and says so.
    r = _upload_to(client, a, did, "증강1")
    assert r.status_code == 200, r.text
    again = r.json()
    assert again["duplicate"] and again["already_in_dataset"] and again["folder"] == "원본/정면"
    assert again["media"]["id"] == first["media"]["id"]
    assert _folders(client, did) == {"원본": (0, 1), "원본/정면": (1, 1)}, "증강1 was not made"

    # A file the lab has but this dataset has not: it goes into the folder.
    b = _image(src, "b.png", 312)
    stored = upload(client, b)["media"]
    r = _upload_to(client, b, did, "증강1/sub")
    res = r.json()
    assert res["duplicate"] and not res["already_in_dataset"] and res["folder"] == "증강1/sub"
    assert res["media"]["id"] == stored["id"]
    # No folder field: the top level.
    r = _upload_to(client, _image(src, "c.png", 313), did, None)
    assert r.json()["folder"] == "" and not r.json()["already_in_dataset"]
    assert _folders(client, did) == {"원본": (0, 1), "원본/정면": (1, 1), "증강1": (0, 1), "증강1/sub": (1, 1)}
    assert client.get(f"/api/datasets/{did}").json()["media_count"] == 3

    # A bad folder is refused before anything is stored; a folder needs a dataset.
    before = len(client.get("/api/media").json())
    d = _image(src, "d.png", 314)
    r = _upload_to(client, d, did, "../밖")
    assert r.status_code == 400
    r = _upload_to(client, d, None, "원본")
    assert r.status_code == 400
    assert len(client.get("/api/media").json()) == before
    # Without a dataset the response has no folder.
    r = _upload_to(client, d, None, None)
    assert r.status_code == 200 and r.json()["folder"] is None and not r.json()["already_in_dataset"]


# --- batches and zips ----------------------------------------------------------------------------
def test_folder_batch_and_zips_keep_the_folders(client: TestClient, image: dict, video: dict, data_dir: Path) -> None:
    src = data_dir.parent / "folder-src"
    ds = client.post("/api/datasets", json={"name": "폴더 일괄 시험", "media_ids": [image["id"]]}).json()
    did = ds["id"]
    client.post(f"/api/datasets/{did}/folders", json={"path": "비어 있음"})
    client.post(f"/api/datasets/{did}/media", json={"media_ids": [video["id"]]})
    client.post(f"/api/datasets/{did}/media/move", json={"media_ids": [video["id"]], "folder": "증강1"})
    # Two more files called cubes.png (other pictures): one per folder needs no new name, two in one folder do.
    for i, index in enumerate((321, 322)):
        r = _upload_to(client, _image(src, f"same-{i}.png", index), did, "증강1/sub", name="cubes.png")
        assert r.status_code == 200, r.text

    # The originals: root files at the zip root, the others in their folders, names unique per folder.
    r = client.get(f"/api/datasets/{did}/originals.zip")
    assert r.status_code == 200, r.text
    assert sorted(zipfile.ZipFile(io.BytesIO(r.content)).namelist()) == [
        "cubes.png", "증강1/index.mp4", "증강1/sub/cubes-2.png", "증강1/sub/cubes.png",
    ]

    # A batch on one folder runs it with its subfolders.
    assert client.post(f"/api/datasets/{did}/batches", json={"model_key": "test.index", "folder": "없음"}).status_code == 404
    r = client.post(f"/api/datasets/{did}/batches", json={"model_key": "test.index", "folder": "비어 있음"})
    assert r.status_code == 400 and "비어 있습니다" in r.json()["detail"]
    r = client.post(f"/api/datasets/{did}/batches", json={"model_key": "test.index", "params": {"delay_ms": 0}, "stride": 25, "folder": "증강1"})
    assert r.status_code == 200, r.text
    batch = r.json()
    assert batch["folder"] == "증강1" and batch["counts"]["total"] == 3
    assert {it["folder"] for it in batch["items"]} == {"증강1", "증강1/sub"}
    run = client.get(f"/api/runs/{batch['items'][0]['run_id']}").json()
    assert "폴더 일괄 시험/증강1" in run["note"]
    batch = _wait_batch(client, batch["id"])
    assert batch["counts"]["done"] == 3, batch["counts"]
    whole = client.post(f"/api/datasets/{did}/batches", json={"model_key": "test.index", "stride": 25}).json()
    assert whole["folder"] is None and whole["counts"]["total"] == 4
    assert [(b["id"], b["folder"]) for b in client.get(f"/api/datasets/{did}/batches").json()] == [
        (whole["id"], None), (batch["id"], "증강1"),
    ]

    # Its results keep the paths relative to the dataset, so zips of different folders merge.
    r = client.get(batch["zip_url"])
    assert r.status_code == 200, r.text
    assert "%EC%A6%9D%EA%B0%951" in r.headers["content-disposition"], "the folder is in the file name"
    z = zipfile.ZipFile(io.BytesIO(r.content))
    assert sorted(z.namelist()) == [
        "results.csv", "results.json", "증강1/index_result.mp4", "증강1/sub/cubes_result-2.png", "증강1/sub/cubes_result.png",
    ]
    meta = json.loads(z.read("results.json"))
    assert meta["batch"]["folder"] == "증강1"
    assert sorted((it["folder"], it["file"]) for it in meta["items"]) == [
        ("증강1", "증강1/index_result.mp4"), ("증강1/sub", "증강1/sub/cubes_result-2.png"), ("증강1/sub", "증강1/sub/cubes_result.png"),
    ]
    rows = list(csv.DictReader(io.StringIO(z.read("results.csv").decode("utf-8-sig"))))
    assert rows and list(rows[0].keys())[:3] == ["file", "folder", "media"]
    assert {(row["folder"], row["file"]) for row in rows} == {
        ("증강1", "증강1/index_result.mp4"), ("증강1/sub", "증강1/sub/cubes_result-2.png"), ("증강1/sub", "증강1/sub/cubes_result.png"),
    }

    # Renaming the folder renames the batch's scope too.
    client.post(f"/api/datasets/{did}/folders/rename", json={"path": "증강1", "new_path": "증강A"})
    assert client.get(f"/api/batches/{batch['id']}").json()["folder"] == "증강A"
    _wait_batch(client, whole["id"])

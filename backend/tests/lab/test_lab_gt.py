"""Ground-truth sets and scoring (docs/SCORING.md): importing the two real files, links by sha256
(also for media added later), candidates that link only when confirmed, all-or-nothing
confirmation, sha256 winning over a confirmed link, organizing into variant folders, POST
/api/score with hand-computed metrics (perfect, perturbed, failed, interim, folder batches,
paired bootstrap, nulls, the summary detail), the per-image details, the CSVs, the criteria
limits, the caches and dataset deletion.

The test images are 320x240 index frames: test.index puts one red cube at (160, 120) with a
20 px axis-aligned quad and angle 0, so GT labels placed relative to that point give known errors.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import time
from pathlib import Path
from typing import Any

import cv2
import pytest
from fastapi.testclient import TestClient

from typing import get_args

from cubevision.gt import GT_COLORS, SCHEMA_AUGMENTED, SCHEMA_V11
from cubevision.score import CUBE_CSV_FIELDS, METRICS, SUMMARY_CSV_FIELDS, Criteria, Outcome
from cubevision.types import COLORS
from lab import scoring, schemas
from lab.db import connect, now_iso
from lab_testkit import index_frame

GT_DIR = Path(__file__).resolve().parents[1] / "data" / "gt"
GT41 = GT_DIR / "robothon_ground_truth_all41.json"
GT410 = GT_DIR / "robothon_labels_410.json"
CX, CY = 160.0, 120.0  # where test.index puts its cube on a 320x240 frame
HIDE_CENTRE = [{"mode": "exclude", "points": [[100, 60], [220, 60], [220, 180], [100, 180]]}]


# --- builders ------------------------------------------------------------------------------------
def _png(src: Path, name: str, index: int) -> Path:
    """A 320x240 PNG no other test makes (its index blocks make it unique)."""
    d = src / str(index)
    d.mkdir(parents=True, exist_ok=True)
    path = d / name
    cv2.imwrite(str(path), index_frame(index))
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cube(cid: str, color: str = "red", dx: float = 0.0, dy: float = 0.0, angle: float = 0.0, valid: bool = True,
          region: str = "center", side: str = "left", flags: tuple[str, ...] = ()) -> dict[str, Any]:
    x, y = CX + dx, CY + dy
    c: dict[str, Any] = {"id": cid, "color": color, "valid_geometry": valid, "detection_match_point": [x, y],
                         "image_side": side, "flags": list(flags)}
    if valid:
        c.update(center=[x, y], quad=[[x - 10, y - 10], [x + 10, y - 10], [x + 10, y + 10], [x - 10, y + 10]],
                 angle=angle, region=region)
    else:
        c.update(center=None, quad=None, angle=None, approximate_region=region)
    return c


def _item(sha: str, filename: str, group: str, cubes: list[dict[str, Any]], variant: str | None = None,
          source: str | None = None, near: tuple[tuple[str, str], ...] = ()) -> dict[str, Any]:
    item: dict[str, Any] = {"media": f"images/{filename}", "filename": filename, "sha256": sha, "layout_group": group,
                            "width": 320, "height": 240, "near_top_face_pairs": [list(p) for p in near],
                            "frames": [{"frame": 0, "cubes": cubes}]}
    if variant is not None:
        item["augmentation"] = {"name": variant}
        item["source_sha256"] = source or sha
    return item


def _doc(items: list[dict[str, Any]], augmented: bool = False) -> bytes:
    schema = SCHEMA_AUGMENTED if augmented else SCHEMA_V11
    return json.dumps({"metadata": {"schema_version": schema, "label_kind": "test labels"}, "items": items}).encode()


def _post_gt(client: TestClient, dataset_id: int, content: bytes, filename: str = "gt.json", name: str | None = None):
    return client.post(f"/api/datasets/{dataset_id}/gt", data={} if name is None else {"name": name},
                       files={"file": (filename, content, "application/json")})


def _put(client: TestClient, path: Path, dataset_id: int, folder: str | None = None) -> dict:
    with open(path, "rb") as f:
        r = client.post(f"/api/media?dataset_id={dataset_id}", data={} if folder is None else {"folder": folder},
                        files={"file": (path.name, f, "image/png")})
    assert r.status_code == 200, r.text
    return r.json()["media"]


def _dataset(client: TestClient, name: str) -> int:
    r = client.post("/api/datasets", json={"name": name})
    assert r.status_code == 200, r.text
    return r.json()["id"]


def _wait_batch(client: TestClient, batch_id: int, timeout: float = 90.0) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        b = client.get(f"/api/batches/{batch_id}").json()
        if b["finished"]:
            return b
        assert time.monotonic() < deadline, f"batch {batch_id} unfinished: {b['counts']}"
        time.sleep(0.1)


def _batch(client: TestClient, dataset_id: int, **body) -> dict:
    r = client.post(f"/api/datasets/{dataset_id}/batches", json={"model_key": "test.index", "stride": 25, **body})
    assert r.status_code == 200, r.text
    b = _wait_batch(client, r.json()["id"])
    assert b["counts"]["done"] == b["counts"]["total"], b["counts"]
    return b


def _score(client: TestClient, **body) -> dict:
    r = client.post("/api/score", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def _set_status(data_dir: Path, run_ids: dict[int, str]) -> None:
    conn = connect(data_dir / "lab.sqlite3")
    try:
        for rid, status in run_ids.items():
            conn.execute("UPDATE runs SET status = ? WHERE id = ?", (status, rid))
    finally:
        conn.close()


def _by_batch(result: dict) -> dict[int, dict]:
    return {b["batch_id"]: b for b in result["batches"]}


def _ratio(r: dict) -> tuple[int, int, float | None]:
    return (r["numerator"], r["denominator"], r["value"])


def _diff(cmp: dict, key: str) -> dict:
    return next(d for d in cmp["diffs"] if d["key"] == key)


# --- import --------------------------------------------------------------------------------------
def test_import_the_real_files(client: TestClient, image: dict, data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    did = _dataset(client, "정답셋 올리기 시험")
    client.post(f"/api/datasets/{did}/media", json={"media_ids": [image["id"]]})
    sets = {}
    for path, counts, schema in ((GT41, (41, 328, 318), SCHEMA_V11), (GT410, (410, 3280, 3180), SCHEMA_AUGMENTED)):
        r = _post_gt(client, did, path.read_bytes(), path.name)
        assert r.status_code == 200, r.text
        s = r.json()
        assert (s["n_images"], s["n_cubes"], s["n_valid"]) == counts
        assert s["schema_version"] == schema and s["name"] == path.name and s["n_layout_groups"] == 35
        assert "approximate image-plane geometry" in s["label_kind"]
        assert s["links"] == {"gt_images": counts[0], "linked_sha256": 0, "linked_confirmed": 0, "unlinked": counts[0],
                              "dataset_images_without_gt": 1, "dataset_videos": 0, "ignored": []}
        assert (data_dir / "gt" / f"gt-{s['id']}.json").read_bytes() == path.read_bytes(), "kept byte for byte"
        sets[path] = s
    assert [(v["name"], v["count"]) for v in sets[GT410]["variants"]][:2] == [("original", 41), ("dark_ev_m0p5", 41)]
    assert len(sets[GT410]["variants"]) == 10 and sets[GT41]["variants"] == [{"name": "original", "count": 41}]
    assert [s["id"] for s in client.get(f"/api/datasets/{did}/gt").json()] == [sets[GT410]["id"], sets[GT41]["id"]]

    # The same file again: refused, naming the set that has it, before it is parsed (the upload's
    # sha256 says so); another dataset may have its own copy.
    def never(doc: Any) -> Any:
        raise AssertionError("a known file is not parsed again")

    with monkeypatch.context() as m:
        m.setattr("lab.gtsets.parse_gt", never)
        r = _post_gt(client, did, GT41.read_bytes(), "다른 이름.json")
    assert r.status_code == 409 and GT41.name in r.json()["detail"] and f"#{sets[GT41]['id']}" in r.json()["detail"]
    other = _dataset(client, "정답셋 올리기 시험 2")
    assert _post_gt(client, other, GT41.read_bytes(), name="41장판").json()["name"] == "41장판"

    # Files the parser refuses: 400 with its message; nothing is stored.
    before = sorted(p.name for p in (data_dir / "gt").iterdir())
    r = _post_gt(client, did, b"{not json", "broken.json")
    assert r.status_code == 400 and "JSON" in r.json()["detail"] and "1행" in r.json()["detail"]
    r = _post_gt(client, did, json.dumps({"metadata": {"schema_version": "robothon_top_face_gt_v9"}, "items": []}).encode())
    assert r.status_code == 400 and "지원하지 않는 정답셋 형식" in r.json()["detail"]
    bad = json.loads(_doc([_item("abc", "x.png", "g", [_cube("red_1", color="purple")])]))
    r = _post_gt(client, did, json.dumps(bad).encode())
    assert r.status_code == 400 and "purple" in r.json()["detail"] and "x.png" in r.json()["detail"]
    assert _post_gt(client, 999999, GT41.read_bytes()).status_code == 404
    monkeypatch.setattr("lab.api.gt.MAX_GT_BYTES", 1000)
    assert _post_gt(client, did, GT41.read_bytes(), "큰 파일.json").status_code == 413
    assert sorted(p.name for p in (data_dir / "gt").iterdir()) == before
    assert not any((data_dir / "tmp").iterdir()), "upload temp files are removed"

    # Rename and delete.
    r = client.patch(f"/api/gt/{sets[GT41]['id']}", json={"name": " 41장 정답 "})
    assert r.status_code == 200 and r.json()["name"] == "41장 정답"
    assert client.patch(f"/api/gt/{sets[GT41]['id']}", json={"name": "  "}).status_code == 400
    assert client.delete(f"/api/gt/{sets[GT41]['id']}").json()["ok"]
    assert client.get(f"/api/gt/{sets[GT41]['id']}").status_code == 404
    assert not (data_dir / "gt" / f"gt-{sets[GT41]['id']}.json").exists()
    assert client.delete(f"/api/gt/{sets[GT41]['id']}").status_code == 404


# --- links ---------------------------------------------------------------------------------------
def test_links_by_sha256_candidates_confirmation_and_organize(client: TestClient, video: dict, image: dict,
                                                             data_dir: Path) -> None:
    src = data_dir.parent / "gt-links"
    a, b = _png(src, "a.png", 601), _png(src, "a__cool.png", 602)
    c = _png(src, "a__dark.png", 603)  # uploaded after the import
    d = _png(src, "cand.png", 604)  # same name as a GT image whose file is something else
    e = _png(src, "e.png", 605)  # uploaded after its look-alike was confirmed
    f = _png(src, "e.png", 606)  # the look-alike: GT file name e.png, another picture
    did = _dataset(client, "정답 연결 시험")
    ma, mb, md, mf = (_put(client, p, did) for p in (a, b, d, f))
    client.post(f"/api/datasets/{did}/media", json={"media_ids": [video["id"]]})
    sha_a = _sha(a)
    items = [
        _item(sha_a, "a.png", "g1", [_cube("red_1")], "original"),
        _item(_sha(b), "a__cool.png", "g1", [_cube("red_1")], "cool", sha_a),
        _item(_sha(c), "a__dark.png", "g1", [_cube("red_1")], "dark_ev_m0p5", sha_a),
        _item("f" * 64, "cand.png", "g2", [_cube("red_1")], "original"),
        _item(_sha(e), "e.png", "g2", [_cube("red_1")], "cool", "f" * 64),
    ]
    A, B, C, X, E = range(5)
    r = _post_gt(client, did, _doc(items, augmented=True))
    assert r.status_code == 200, r.text
    gid = r.json()["id"]
    assert r.json()["links"] == {"gt_images": 5, "linked_sha256": 2, "linked_confirmed": 0, "unlinked": 3,
                                 "dataset_images_without_gt": 2, "dataset_videos": 1, "ignored": []}

    # Candidates: dataset images with the GT file name that are not linked. They are not links.
    det = client.get(f"/api/gt/{gid}").json()
    assert [(x["gt_index"], x["method"], x["media"]["media_id"]) for x in det["linked"]] == [(A, "sha256", ma["id"]), (B, "sha256", mb["id"])]
    assert {x["gt_index"]: [m["media_id"] for m in x["candidates"]] for x in det["unlinked"]} == {C: [], X: [md["id"]], E: [mf["id"]]}
    assert sorted(m["media_id"] for m in det["images_without_gt"]) == sorted([md["id"], mf["id"]])
    assert client.get(f"/api/media/{md['id']}/gt").json() == []
    assert client.get(f"/api/media/{ma['id']}/gt").json() == [{
        "gt_set_id": gid, "gt_set_name": "gt.json", "dataset_id": did, "dataset_name": "정답 연결 시험", "gt_index": A,
        "filename": "a.png", "variant": "original", "method": "sha256"}]
    assert client.get("/api/media/999999/gt").status_code == 404

    # A media added after the import links by itself.
    mc = _put(client, c, did)
    assert client.get(f"/api/media/{mc['id']}/gt").json()[0]["gt_index"] == C
    assert client.get(f"/api/gt/{gid}").json()["links"]["linked_sha256"] == 3

    # Confirmation is all or nothing.
    links = f"/api/gt/{gid}/links"
    r = client.post(links, json={"links": [{"gt_index": X, "media_id": md["id"]}, {"gt_index": A, "media_id": mf["id"]}]})
    assert r.status_code == 409 and "0번" in r.json()["detail"]
    r = client.post(links, json={"links": [{"gt_index": X, "media_id": md["id"]}, {"gt_index": E, "media_id": ma["id"]}]})
    assert r.status_code == 409 and "이미 정답 이미지" in r.json()["detail"]
    r = client.post(links, json={"links": [{"gt_index": X, "media_id": video["id"]}]})
    assert r.status_code == 400 and "영상" in r.json()["detail"]
    r = client.post(links, json={"links": [{"gt_index": X, "media_id": image["id"]}]})
    assert r.status_code == 400 and "이 데이터셋에 없습니다" in r.json()["detail"]
    r = client.post(links, json={"links": [{"gt_index": X, "media_id": md["id"]}, {"gt_index": E, "media_id": md["id"]}]})
    assert r.status_code == 400 and "두 번" in r.json()["detail"]
    assert client.post(links, json={"links": [{"gt_index": 99, "media_id": md["id"]}]}).status_code == 400
    assert client.post(links, json={"links": []}).status_code == 400
    assert client.get(f"/api/gt/{gid}").json()["links"]["linked_confirmed"] == 0, "nothing was linked"

    r = client.post(links, json={"links": [{"gt_index": X, "media_id": md["id"]}, {"gt_index": E, "media_id": mf["id"]}]})
    assert r.status_code == 200, r.text
    det = r.json()
    assert det["links"]["linked_confirmed"] == 2 and det["unlinked"] == [] and det["images_without_gt"] == []
    assert [(x["gt_index"], x["method"]) for x in det["linked"]] == [(A, "sha256"), (B, "sha256"), (C, "sha256"),
                                                                    (X, "confirmed"), (E, "confirmed")]
    assert client.get(f"/api/media/{md['id']}/gt").json()[0]["method"] == "confirmed"

    # The labelled file itself arrives: its sha256 link wins, the confirmed one is ignored and reported.
    me = _put(client, e, did)
    det = client.get(f"/api/gt/{gid}").json()
    assert next(x for x in det["linked"] if x["gt_index"] == E)["media"]["media_id"] == me["id"]
    assert det["links"]["linked_sha256"] == 4 and det["links"]["linked_confirmed"] == 1
    (ignored,) = det["links"]["ignored"]
    assert (ignored["gt_index"], ignored["media_id"]) == (E, mf["id"]) and "sha256" in ignored["reason"]
    assert [m["media_id"] for m in det["images_without_gt"]] == [mf["id"]]
    assert client.get(f"/api/media/{mf['id']}/gt").json() == []

    # Removing links: a confirmed one (also an ignored one), never a sha256 one.
    r = client.delete(f"{links}/{X}")
    assert r.status_code == 200
    assert next(x for x in r.json()["unlinked"] if x["gt_index"] == X)["candidates"][0]["media_id"] == md["id"]
    assert client.delete(f"{links}/{X}").status_code == 404
    r = client.delete(f"{links}/{A}")
    assert r.status_code == 400 and "sha256" in r.json()["detail"]
    assert client.delete(f"{links}/{E}").json()["links"]["ignored"] == []

    # Organize: every linked media into <root>/<variant>; the others stay where they are.
    assert client.post(f"/api/gt/{gid}/organize", json={"root": "../밖"}).status_code == 400
    r = client.post(f"/api/gt/{gid}/organize", json={"root": "정답"})
    assert r.status_code == 200, r.text
    assert {f["path"] for f in r.json()["folders"]} >= {"정답", "정답/original", "정답/cool", "정답/dark_ev_m0p5"}
    where = {m["id"]: m["folder"] for m in client.get("/api/media", params={"dataset_id": did}).json()}
    assert where == {ma["id"]: "정답/original", mb["id"]: "정답/cool", mc["id"]: "정답/dark_ev_m0p5",
                     me["id"]: "정답/cool", md["id"]: "", mf["id"]: "", video["id"]: ""}
    empty = _dataset(client, "정답 연결 없음")
    gid2 = _post_gt(client, empty, _doc(items[:1])).json()["id"]
    assert client.post(f"/api/gt/{gid2}/organize", json={}).status_code == 400

    # A media that leaves the dataset keeps no link there.
    client.post(f"/api/datasets/{did}/media/remove", json={"media_ids": [mb["id"]]})
    assert client.get(f"/api/media/{mb['id']}/gt").json() == []
    assert client.get(f"/api/gt/{gid}").json()["links"]["unlinked"] == 2


# --- scoring -------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def scene(client: TestClient, video: dict, data_dir: Path) -> dict[str, Any]:
    """A dataset of 5 images + 1 video, two GT sets (perfect, perturbed), batch A (test.index)
    and batch B (the cube hidden by the ROI: nothing detected), both done."""
    src = data_dir.parent / "gt-score"
    pngs = [_png(src, f"p{i}.png", 700 + i) for i in range(5)]
    did = _dataset(client, "채점 시험")
    media = [_put(client, p, did) for p in pngs]
    client.post(f"/api/datasets/{did}/media", json={"media_ids": [video["id"]]})
    shas = [_sha(p) for p in pngs]
    groups = ["g1", "g1", "g2", "g2", "g3"]
    perfect = [_item(s, p.name, g, [_cube("red_1")]) for s, p, g in zip(shas, pngs, groups)]
    perturbed = [
        _item(shas[0], "p0.png", "g1", [_cube("red_1")]),  # exact
        _item(shas[1], "p1.png", "g1", [_cube("red_1", dx=3, dy=4, angle=4.0)]),  # 5 px, 5 px RMS, 4°: passes
        _item(shas[2], "p2.png", "g2", [_cube("red_1", dx=6, dy=8)]),  # 10 px: fails
        _item(shas[3], "p3.png", "g2", [_cube("red_1", color="blue"),  # wrong colour
                                        _cube("green_1", "green", dx=-120, dy=-80, region="outer")],  # missed
              near=(("red_1", "green_1"),)),
        _item(shas[4], "p4.png", "g3", [_cube("red_1", valid=False, region="outer", flags=("weak_boundary",))]),
    ]
    sets = {}
    for name, items in (("완벽", perfect), ("어긋남", perturbed)):
        r = _post_gt(client, did, _doc(items), f"{name}.json")
        assert r.status_code == 200 and r.json()["links"]["linked_sha256"] == 5, r.text
        sets[name] = r.json()["id"]
    a = _batch(client, did)
    b = _batch(client, did, params={"roi": HIDE_CENTRE})
    runs = {}
    for batch in (a, b):
        runs[batch["id"]] = {it["media_id"]: it["run_id"] for it in batch["items"]}
    return {"dataset": did, "media": [m["id"] for m in media], "video": video["id"], "perfect": sets["완벽"],
            "perturbed": sets["어긋남"], "A": a["id"], "B": b["id"], "runs": runs}


def test_perfect_and_empty_batches(client: TestClient, scene: dict) -> None:
    res = _score(client, gt_set_id=scene["perfect"], reference_batch_id=scene["A"])
    assert res["criteria"] == {"center_px": 5.0, "angle_deg": 5.0, "vertex_px": 5.0, "gate_px": 29.0,
                               "bootstrap_reps": 2000, "bootstrap_level": 0.95, "bootstrap_seed": 0}
    assert [b["batch_id"] for b in res["batches"]] == [scene["B"], scene["A"]], "all batches of the dataset, newest first"
    a, b = _by_batch(res)[scene["A"]], _by_batch(res)[scene["B"]]
    assert res["gt_set"]["n_images"] == 5
    for x in (a, b):
        assert x["status"] == "complete" and x["full_coverage"]
        assert x["pending"] == {"queued": 0, "running": 0, "cancelled": 0, "missing": 0, "total": 0}
        assert (x["n_eval_images"], x["metrics"]["n_images"], x["metrics"]["n_failed"], x["videos_skipped"]) == (5, 5, 0, 1)
        assert x["folder"] is None and x["model_key"] == "test.index"
        assert {"n_scored", "n_failed_runs", "coverage"}.isdisjoint(x)
    m = a["metrics"]
    for key in ("tolerance_rate", "image_pass_rate", "recall", "precision", "color_accuracy"):
        assert _ratio(m[key]) == (5, 5, 1.0), key
    assert m["center_err"] == {"n": 5, "median": 0.0, "p95": 0.0, "mean": 0.0}
    assert m["vertex_err"]["n"] == m["angle_err"]["n"] == 5 and m["time_ms"]["n"] == 5
    # Nothing detected: undefined ratios and empty samples are null, never 0.
    m = b["metrics"]
    assert _ratio(m["recall"]) == (0, 5, 0.0) and _ratio(m["tolerance_rate"]) == (0, 5, 0.0)
    assert _ratio(m["precision"]) == (0, 0, None) and _ratio(m["color_accuracy"]) == (0, 0, None)
    assert m["center_err"] == {"n": 0, "median": None, "p95": None, "mean": None}
    assert m["n_preds"] == 0 and m["time_ms"]["n"] == 5

    # Paired bootstrap against the requested reference A: B is lower wherever both are defined.
    assert res["reference_batch_id"] == scene["A"] and res["reference_note"] is None
    assert a["comparison"] == {"role": "reference", "reason_code": None, "reason": None, "result": None}
    cmp = b["comparison"]["result"]
    assert b["comparison"]["role"] == "compared" and cmp["n_groups"] == 3 and cmp["reps"] == 2000
    rec = _diff(cmp, "recall")
    assert (rec["a"], rec["b"], rec["delta"], rec["ci_low"], rec["ci_high"], rec["verdict"]) == (1.0, 0.0, -1.0, -1.0, -1.0, "lower")
    assert _diff(cmp, "tolerance_rate")["verdict"] == "lower"
    col = _diff(cmp, "color_accuracy")
    assert (col["b"], col["delta"], col["verdict"], col["reps_used"]) == (None, None, None, 0)
    assert _diff(cmp, "center_err_median")["verdict"] is None
    assert res["final_ranking"] == [scene["A"], scene["B"]]

    # Without a reference: the first complete full-coverage batch in the list (the newest).
    res = _score(client, gt_set_id=scene["perfect"])
    assert res["reference_batch_id"] == scene["B"] and res["final_ranking"] == [scene["A"], scene["B"]]
    assert _diff(_by_batch(res)[scene["A"]]["comparison"]["result"], "image_pass_rate")["verdict"] == "higher"
    assert "metric_defs" not in res
    assert [d["key"] for d in res["slice_dimensions"]][:2] == ["variant", "region"]

    # The rail's chips: status, pending, coverage and metrics only, the same numbers; no bootstrap.
    summary = _score(client, gt_set_id=scene["perfect"], detail="summary")
    assert summary["final_ranking"] == res["final_ranking"] and summary["reference_batch_id"] == scene["B"]
    for full_b, sum_b in zip(res["batches"], summary["batches"]):
        assert (sum_b["slices"], sum_b["augmentation_effect"], sum_b["comparison"]) == (None, None, None)
        assert {k: v for k, v in sum_b.items() if k not in ("slices", "augmentation_effect", "comparison")} == {
            k: v for k, v in full_b.items() if k not in ("slices", "augmentation_effect", "comparison")}
    assert client.post("/api/score", json={"gt_set_id": scene["perfect"], "detail": "some"}).status_code == 400

    # Bad requests.
    assert client.post("/api/score", json={"gt_set_id": 999999}).status_code == 404
    r = client.post("/api/score", json={"gt_set_id": scene["perfect"], "criteria": {"center_px": -1}})
    assert r.status_code == 400 and "center_px" in r.json()["detail"]
    r = client.post("/api/score", json={"gt_set_id": scene["perfect"], "criteria": {"bootstrap_reps": 10**7}})
    assert r.status_code == 400 and "bootstrap_reps" in r.json()["detail"]
    r = client.post("/api/score", json={"gt_set_id": scene["perfect"], "criteria": {"bootstrap_level": 1}})
    assert r.status_code == 400 and "bootstrap_level" in r.json()["detail"]
    # A null criterion takes its default.
    res = _score(client, gt_set_id=scene["perfect"], criteria={"center_px": None, "gate_px": 20})
    assert (res["criteria"]["center_px"], res["criteria"]["gate_px"]) == (5.0, 20.0)
    assert client.post("/api/score", json={"gt_set_id": scene["perfect"], "batch_ids": [999999]}).status_code == 404
    other = _dataset(client, "채점 시험 다른 데이터셋")
    client.post(f"/api/datasets/{other}/media", json={"media_ids": scene["media"][:1]})
    foreign = _batch(client, other)["id"]
    r = client.post("/api/score", json={"gt_set_id": scene["perfect"], "batch_ids": [foreign]})
    assert r.status_code == 400 and "데이터셋" in r.json()["detail"]


def test_perturbed_batch_hand_computed(client: TestClient, scene: dict) -> None:
    res = _score(client, gt_set_id=scene["perturbed"], batch_ids=[scene["A"], scene["B"]], reference_batch_id=scene["A"])
    m = _by_batch(res)[scene["A"]]["metrics"]
    # 6 GT cubes, 5 with valid geometry; P0 and P1 pass (P1 at exactly 5 px / 5 px RMS / 4°),
    # P2 is 10 px off, P3's red cube has the wrong colour and its green cube is missed, P4 is
    # invalid geometry (match + colour only).
    assert (m["n_images"], m["n_cubes"], m["n_valid"], m["n_preds"], m["n_failed"]) == (5, 6, 5, 5, 0)
    assert _ratio(m["tolerance_rate"]) == (2, 5, 0.4)
    assert _ratio(m["image_pass_rate"]) == (3, 5, 0.6)
    assert _ratio(m["recall"]) == (5, 6, pytest.approx(5 / 6))
    assert _ratio(m["precision"]) == (5, 5, 1.0)
    assert _ratio(m["color_accuracy"]) == (4, 5, 0.8)
    assert m["confusion"]["blue"]["red"] == 1 and m["confusion"]["red"]["red"] == 4
    # Errors over the 4 matched valid-geometry cubes: centre/vertex [0, 5, 10, 0], angle [0, 4, 0, 0].
    assert m["center_err"] == {"n": 4, "median": 2.5, "p95": pytest.approx(9.25), "mean": 3.75}
    assert m["vertex_err"] == {"n": 4, "median": 2.5, "p95": pytest.approx(9.25), "mean": 3.75}
    assert (m["angle_err"]["n"], m["angle_err"]["median"], m["angle_err"]["p95"]) == (4, 0.0, pytest.approx(3.4))
    rows = {(s["dimension"], s["value"]): s for s in _by_batch(res)[scene["A"]]["slices"]}
    assert rows[("variant", "original")]["n"] == 5
    assert (rows[("region", "center")]["n"], _ratio(rows[("region", "center")]["metrics"]["recall"])) == (4, (4, 4, 1.0))
    assert (rows[("region", "outer")]["n"], _ratio(rows[("region", "outer")]["metrics"]["recall"])) == (2, (1, 2, 0.5))
    assert _ratio(rows[("near_contact", True)]["metrics"]["recall"]) == (1, 2, 0.5)
    assert rows[("weak_boundary", True)]["n"] == 1
    assert rows[("color", "red")]["metrics"]["precision"] is None, "not defined for a cube slice"
    assert _by_batch(res)[scene["A"]]["augmentation_effect"] == [], "no variants in this set"

    # The bootstrap is deterministic for a seed (computed afresh, not from the cache).
    scoring.clear_caches()
    first = _score(client, gt_set_id=scene["perturbed"], reference_batch_id=scene["A"], criteria={"bootstrap_seed": 5})
    scoring.clear_caches()
    again = _score(client, gt_set_id=scene["perturbed"], reference_batch_id=scene["A"], criteria={"bootstrap_seed": 5})
    cmp = _by_batch(first)[scene["B"]]["comparison"]["result"]
    assert cmp == _by_batch(again)[scene["B"]]["comparison"]["result"] and cmp["seed"] == 5
    assert _diff(cmp, "recall")["verdict"] == "lower"

    # Thresholds change the outcome: with 12 px / 12 px RMS, P2 passes too.
    res = _score(client, gt_set_id=scene["perturbed"], batch_ids=[scene["A"]],
                 criteria={"center_px": 12, "vertex_px": 12})
    assert _ratio(_by_batch(res)[scene["A"]]["metrics"]["tolerance_rate"]) == (3, 5, 0.6)
    assert res["criteria"]["center_px"] == 12.0


def test_failed_run_misses_everything_and_has_no_time(client: TestClient, scene: dict, data_dir: Path) -> None:
    p0_run = scene["runs"][scene["A"]][scene["media"][0]]
    _set_status(data_dir, {p0_run: "failed"})
    try:
        res = _score(client, gt_set_id=scene["perturbed"], batch_ids=[scene["A"]])
        b = _by_batch(res)[scene["A"]]
        m = b["metrics"]
        assert b["status"] == "complete" and m["n_failed"] == 1 and m["n_images"] == 5
        assert _ratio(m["recall"]) == (4, 6, pytest.approx(4 / 6)) and _ratio(m["tolerance_rate"]) == (1, 5, 0.2)
        assert _ratio(m["image_pass_rate"]) == (2, 5, 0.4) and m["n_preds"] == 4
        assert m["time_ms"]["n"] == 4, "a failed run gives no time sample"
        detail = client.get("/api/score/image", params={"run_id": p0_run, "gt_set_id": scene["perturbed"]}).json()
        assert detail["scored"] and detail["ran"] is False and detail["n_missed"] == 1
        assert detail["cubes"][0]["matched"] is False and detail["cubes"][0]["outcome"] == "missed"
        assert detail["passed"] is False
    finally:
        _set_status(data_dir, {p0_run: "done"})


def test_interim_batch_is_scored_but_not_ranked_or_compared(client: TestClient, scene: dict, data_dir: Path) -> None:
    runs = scene["runs"][scene["A"]]
    p1, p2 = runs[scene["media"][1]], runs[scene["media"][2]]
    _set_status(data_dir, {p1: "running", p2: "cancelled"})
    try:
        res = _score(client, gt_set_id=scene["perfect"], reference_batch_id=scene["A"])
        a = _by_batch(res)[scene["A"]]
        assert a["status"] == "interim"
        assert a["pending"] == {"queued": 0, "running": 1, "cancelled": 1, "missing": 0, "total": 2}
        assert (a["n_eval_images"], a["metrics"]["n_images"]) == (5, 3) and a["full_coverage"]
        assert _ratio(a["metrics"]["recall"]) == (3, 3, 1.0), "scored on its finished images"
        assert a["comparison"]["role"] == "not_comparable" and a["comparison"]["reason_code"] == "interim"
        assert res["reference_batch_id"] == scene["B"] and "중간 결과" in res["reference_note"]
        assert res["final_ranking"] == [scene["B"]]
        detail = client.get("/api/score/image", params={"run_id": p1}).json()
        assert detail["scored"] is False and detail["cubes"] == [] and detail["passed"] is None
        assert len(detail["gt"]) == 1 and (detail["n_gt"], detail["n_missed"], detail["n_geom_fail"]) == (1, None, None)

        # The CSVs include it, marked 중간.
        r = client.get("/api/score/summary.csv", params={"gt_set_id": scene["perfect"], "batch_ids": str(scene["A"])})
        assert {row["status"] for row in csv.DictReader(io.StringIO(r.content.decode("utf-8-sig")))} == {"중간"}
        r = client.get("/api/score/cubes.csv", params={"gt_set_id": scene["perfect"], "batch_ids": str(scene["A"])})
        rows = list(csv.DictReader(io.StringIO(r.content.decode("utf-8-sig"))))
        assert len(rows) == 3 and {row["status"] for row in rows} == {"중간"}
    finally:
        _set_status(data_dir, {p1: "done", p2: "done"})


def test_folder_batch_has_partial_coverage(client: TestClient, scene: dict) -> None:
    did = scene["dataset"]
    client.post(f"/api/datasets/{did}/media/move", json={"media_ids": scene["media"][3:], "folder": "일부/안"})
    try:
        c = _batch(client, did, folder="일부")
        res = _score(client, gt_set_id=scene["perturbed"])
        assert [b["batch_id"] for b in res["batches"]] == [c["id"], scene["B"], scene["A"]]
        x = _by_batch(res)[c["id"]]
        assert x["folder"] == "일부" and x["status"] == "complete" and x["videos_skipped"] == 0
        assert (x["n_eval_images"], res["gt_set"]["n_images"]) == (2, 5) and not x["full_coverage"]
        assert _ratio(x["metrics"]["recall"]) == (2, 3, pytest.approx(2 / 3))
        assert x["comparison"]["reason_code"] == "different_images" and "이 배치만 0장, 기준만 3장" in x["comparison"]["reason"]
        assert res["reference_batch_id"] == scene["B"], "a partial batch is never the default reference"
        assert c["id"] not in res["final_ranking"]
        # Asked for as the reference, it is one: the full batches then have a different image set.
        res = _score(client, gt_set_id=scene["perturbed"], reference_batch_id=c["id"])
        assert res["reference_batch_id"] == c["id"]
        assert _by_batch(res)[scene["A"]]["comparison"]["reason_code"] == "different_images"
        assert [_by_batch(res)[bid]["comparison"]["role"] for bid in res["final_ranking"]] == ["not_comparable"] * 2
    finally:
        client.post(f"/api/datasets/{did}/media/move", json={"media_ids": scene["media"][3:], "folder": ""})


def test_image_and_frame_details(client: TestClient, scene: dict, image: dict, data_dir: Path) -> None:
    p2_run = scene["runs"][scene["A"]][scene["media"][2]]
    r = client.get("/api/score/image", params={"run_id": p2_run, "gt_set_id": scene["perturbed"]})
    assert r.status_code == 200, r.text
    d = r.json()
    assert (d["gt_set_name"], d["gt_index"], d["method"], d["media_id"], d["run_id"]) == (
        "어긋남.json", 2, "sha256", scene["media"][2], p2_run)
    assert d["filename"] == "p2.png" and d["variant"] == "original"
    assert {"gt_set_id", "run_status", "layout_group", "width", "height", "ms", "pairs"}.isdisjoint(d)
    (g,) = d["gt"]
    assert g == {"id": "red_1", "color": "red", "valid_geometry": True, "quad": g["quad"], "center": [166.0, 128.0],
                 "match_point": [166.0, 128.0]}
    assert g["quad"][0] == [156.0, 118.0]
    (c,) = d["cubes"]
    assert d["predictions"][0]["center"] == [CX, CY] and (c["pred_index"], c["distance"]) == (0, 10.0)
    assert (c["center_err"], c["vertex_err"], c["angle_err"]) == (10.0, 10.0, 0.0)
    assert (c["center_ok"], c["angle_ok"], c["vertex_ok"], c["geom_ok"], c["passed"], c["outcome"]) == (
        False, True, False, False, False, "fail")
    assert d["false_positives"] == [] and d["passed"] is False and d["scored"] and d["ran"]
    assert (d["n_gt"], d["n_missed"], d["n_fp"], d["n_color_wrong"], d["n_geom_fail"]) == (1, 0, 0, 0, 1)
    d = client.get("/api/score/image", params={"run_id": p2_run, "gt_set_id": scene["perturbed"],
                                               "center_px": 12, "vertex_px": 12}).json()
    assert d["passed"] is True and d["criteria"]["center_px"] == 12.0 and d["cubes"][0]["outcome"] == "pass"
    # No set given: the newest set that links the media.
    assert client.get("/api/score/image", params={"run_id": p2_run}).json()["gt_set_name"] == "어긋남.json"
    video_run = scene["runs"][scene["A"]][scene["video"]]
    r = client.get("/api/score/image", params={"run_id": video_run})
    assert r.status_code == 404 and "정답이 없습니다" in r.json()["detail"]
    assert client.get("/api/score/image", params={"run_id": 999999}).status_code == 404

    # The workbench's live score of cubes it has.
    cubes = [
        {"color": "red", "center": [CX, CY], "quad": [[150, 110], [170, 110], [170, 130], [150, 130]], "angle": 0.0},
        {"color": "green", "center": [40, 40], "quad": [[30, 30], [50, 30], [50, 50], [30, 50]], "angle": 0.0},
        {"color": "blue", "center": [300, 20], "quad": None, "angle": None},
    ]
    r = client.post("/api/score/frame", json={"media_id": scene["media"][3], "cubes": cubes})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["gt_set_name"] == "어긋남.json" and d["run_id"] is None
    assert [(c["matched"], c["color_ok"], c["passed"], c["outcome"]) for c in d["cubes"]] == [
        (True, False, False, "fail"), (True, True, True, "pass")]
    assert d["false_positives"] == [2] and d["passed"] is False
    assert (d["n_gt"], d["n_missed"], d["n_fp"], d["n_color_wrong"], d["n_geom_fail"]) == (2, 0, 1, 1, 0)
    d = client.post("/api/score/frame", json={"media_id": scene["media"][3], "cubes": [], "gt_set_id": scene["perfect"],
                                              "criteria": {"gate_px": 0}}).json()
    assert d["gt_set_name"] == "완벽.json" and d["cubes"][0]["matched"] is False and d["criteria"]["gate_px"] == 0.0
    r = client.post("/api/score/frame", json={"media_id": image["id"], "cubes": []})
    assert r.status_code == 404 and "정답이 없습니다" in r.json()["detail"]
    r = client.post("/api/score/frame", json={"media_id": image["id"], "cubes": [], "gt_set_id": scene["perfect"]})
    assert r.status_code == 404 and "연결되어 있지 않습니다" in r.json()["detail"]

    # Every set that links the media, newest first.
    sets = client.get(f"/api/media/{scene['media'][0]}/gt").json()
    assert [(s["gt_set_id"], s["gt_index"], s["method"]) for s in sets] == [
        (scene["perturbed"], 0, "sha256"), (scene["perfect"], 0, "sha256")]


def test_image_rows_of_a_batch(client: TestClient, scene: dict, data_dir: Path) -> None:
    """GET /api/score/images: the 채점 page's per-image list, from the same image scores."""
    runs = scene["runs"][scene["A"]]
    q = {"gt_set_id": scene["perturbed"], "batch_id": scene["A"]}
    r = client.get("/api/score/images", params=q)
    assert r.status_code == 200, r.text
    rows = r.json()
    assert [x["gt_index"] for x in rows] == [0, 1, 2, 3, 4], "GT order; the video is never an evaluation image"
    first = rows[0]
    assert first == {"gt_index": 0, "media_id": scene["media"][0], "media_name": "p0.png", "folder": "", "filename": "p0.png",
                     "run_id": runs[scene["media"][0]], "variant": "original", "source_sha256": first["source_sha256"],
                     "layout_group": "g1", "status": "done", "passed": True, "n_gt": 1, "n_missed": 0, "n_fp": 0,
                     "n_color_wrong": 0, "n_geom_fail": 0, "ms": first["ms"]}
    assert isinstance(first["ms"], float) and len(first["source_sha256"]) == 64
    counts = [(x["passed"], x["n_gt"], x["n_missed"], x["n_fp"], x["n_color_wrong"], x["n_geom_fail"]) for x in rows]
    assert counts == [
        (True, 1, 0, 0, 0, 0),  # exact
        (True, 1, 0, 0, 0, 0),  # 5 px / 5 px RMS / 4°: at the thresholds
        (False, 1, 0, 0, 0, 1),  # 10 px off
        (False, 2, 1, 0, 1, 0),  # wrong colour + a missed cube
        (True, 1, 0, 0, 0, 0),  # invalid geometry: match and colour only
    ]
    # The same numbers as POST /api/score for the same criteria, and as each image's detail.
    m = _by_batch(_score(client, gt_set_id=scene["perturbed"], batch_ids=[scene["A"]]))[scene["A"]]["metrics"]
    assert sum(x["passed"] for x in rows) == m["image_pass_rate"]["numerator"]
    assert sum(x["n_gt"] - x["n_missed"] for x in rows) == m["recall"]["numerator"]
    count_keys = ("n_gt", "n_missed", "n_fp", "n_color_wrong", "n_geom_fail", "passed")
    for x in rows:
        d = client.get("/api/score/image", params={"run_id": x["run_id"], "gt_set_id": scene["perturbed"]}).json()
        assert {k: d[k] for k in count_keys} == {k: x[k] for k in count_keys}
    # Thresholds apply: at 12 px / 12 px RMS the 10 px cube passes.
    rows = client.get("/api/score/images", params={**q, "center_px": 12, "vertex_px": 12}).json()
    assert (rows[2]["passed"], rows[2]["n_geom_fail"]) == (True, 0)
    # A gate of 4 px: the 5 px cube finds no partner (missed, and its detection is a false positive).
    rows = client.get("/api/score/images", params={**q, "gate_px": 4}).json()
    assert (rows[1]["passed"], rows[1]["n_missed"], rows[1]["n_fp"], rows[1]["n_geom_fail"]) == (False, 1, 1, 0)

    # Unfinished images keep their run state and null counts.
    _set_status(data_dir, {runs[scene["media"][1]]: "running"})
    try:
        row = client.get("/api/score/images", params=q).json()[1]
        assert (row["status"], row["run_id"], row["passed"], row["n_missed"], row["ms"], row["n_gt"]) == (
            "running", runs[scene["media"][1]], None, None, None, 1)
    finally:
        _set_status(data_dir, {runs[scene["media"][1]]: "done"})

    # Bad requests: a batch of another dataset, an unknown batch or set.
    other = _dataset(client, "채점 이미지 목록 다른 데이터셋")
    client.post(f"/api/datasets/{other}/media", json={"media_ids": scene["media"][:1]})
    foreign = _batch(client, other)["id"]
    r = client.get("/api/score/images", params={**q, "batch_id": foreign})
    assert r.status_code == 400 and "데이터셋" in r.json()["detail"]
    assert client.get("/api/score/images", params={**q, "batch_id": 999999}).status_code == 404
    assert client.get("/api/score/images", params={**q, "gt_set_id": 999999}).status_code == 404
    assert client.get("/api/score/images", params={"gt_set_id": scene["perturbed"]}).status_code == 400


def test_csv_exports(client: TestClient, scene: dict) -> None:
    query = {"gt_set_id": scene["perturbed"], "batch_ids": f"{scene['A']},{scene['B']}", "gate_px": 4}
    r = client.get("/api/score/cubes.csv", params=query)
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("text/csv")
    assert "attachment; filename*=utf-8''" in r.headers["content-disposition"]
    assert r.content.startswith("﻿".encode())
    text = r.content.decode("utf-8-sig")
    assert text.splitlines()[0] == ",".join(CUBE_CSV_FIELDS) and CUBE_CSV_FIELDS[:3] == ("batch", "status", "image")
    rows = list(csv.DictReader(io.StringIO(text)))
    a = [row for row in rows if row["batch"] == f"#{scene['A']} 테스트: 프레임 번호"]
    b = [row for row in rows if row["batch"].startswith(f"#{scene['B']} ")]
    # Gate 4 px: P1 (5 px) and P2 (10 px) find no partner, so A has 6 GT rows + 2 FP rows.
    assert [row["row"] for row in a].count("gt") == 6 and [row["row"] for row in a].count("fp") == 2
    assert [row["row"] for row in b] == ["gt"] * 6, "B detects nothing: no FP rows"
    p0 = next(row for row in a if row["image"] == "p0.png")
    assert (p0["passed"], p0["status"], p0["gate_px"], p0["tau_center_px"], p0["match_distance_px"]) == ("1", "완료", "4.0", "5.0", "0.0")
    p4 = next(row for row in a if row["image"] == "p4.png")
    assert (p4["passed"], p4["valid_geometry"], p4["weak_boundary"], p4["center_err_px"]) == ("", "0", "1", "")

    r = client.get("/api/score/summary.csv", params={"gt_set_id": scene["perfect"], "batch_ids": f"{scene['A']},{scene['B']}"})
    assert r.status_code == 200 and r.content.startswith("﻿".encode())
    text = r.content.decode("utf-8-sig")
    assert text.splitlines()[0] == ",".join(SUMMARY_CSV_FIELDS)
    rows = list(csv.DictReader(io.StringIO(text)))
    assert len(rows) == 2 * len(METRICS) and {row["status"] for row in rows} == {"완료"}
    prec_b = next(row for row in rows if row["batch"].startswith(f"#{scene['B']} ") and row["metric"] == "precision")
    assert (prec_b["value"], prec_b["numerator"], prec_b["denominator"]) == ("", "0", "0"), "null stays an empty cell"
    assert all(row["gate_px"] == "29.0" for row in rows)
    assert client.get("/api/score/summary.csv", params={"gt_set_id": scene["perfect"], "batch_ids": "x"}).status_code == 400


def test_polling_reuses_the_image_scores(client: TestClient, scene: dict, data_dir: Path,
                                         monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    real = scoring.score_image

    def counting(*args: Any, **kw: Any):
        calls.append(1)
        return real(*args, **kw)

    monkeypatch.setattr(scoring, "score_image", counting)
    scoring.clear_caches()
    body = {"gt_set_id": scene["perfect"], "batch_ids": [scene["A"], scene["B"]]}
    first = _score(client, **body)
    assert len(calls) == 10, "5 images x 2 batches"
    assert _score(client, **body)["batches"] == first["batches"] and len(calls) == 10, "a poll rescored nothing"
    conn = connect(data_dir / "lab.sqlite3")
    conn.execute("UPDATE runs SET finished_at = ? WHERE id = ?", (now_iso(), scene["runs"][scene["A"]][scene["media"][0]]))
    conn.close()
    _score(client, **body)
    assert len(calls) == 11, "only the run that changed"
    gens = scoring._generations.get((str(client.app.state.paths.root), scene["perfect"]))
    assert len(gens) == 1 and len(gens.get(Criteria().thresholds).scored) == 2, "a poll replaces its batch's entry"
    _score(client, **body, criteria={"bootstrap_seed": 3})
    assert len(calls) == 11, "bootstrap settings do not rescore images"
    _score(client, **body, criteria={"center_px": 4})
    assert len(calls) == 21, "new thresholds do"
    # Each set keeps the image scores of its last 3 threshold tuples.
    for tau in (3, 2):
        _score(client, **body, criteria={"center_px": tau}, detail="summary")
    assert len(calls) == 41 and len(gens) == scoring.GENERATIONS == 3
    _score(client, **body, detail="summary")
    assert len(calls) == 51, "the oldest generation (the defaults) was dropped"


def test_media_added_later_makes_batches_interim(client: TestClient, scene: dict, data_dir: Path) -> None:
    """Runs last in this module: it adds a sixth image and a third set to the scene."""
    late = _png(data_dir.parent / "gt-score", "p5.png", 705)
    items = [_item(_sha(data_dir.parent / "gt-score" / str(700 + i) / f"p{i}.png"), f"p{i}.png", "g1", [_cube("red_1")])
             for i in range(5)] + [_item(_sha(late), "p5.png", "g4", [_cube("red_1")])]
    gid = _post_gt(client, scene["dataset"], _doc(items), "여섯 장.json").json()["id"]
    res = _score(client, gt_set_id=gid)
    a = _by_batch(res)[scene["A"]]
    assert a["status"] == "complete" and (a["n_eval_images"], res["gt_set"]["n_images"]) == (5, 6) and not a["full_coverage"]
    assert res["final_ranking"] == [] and res["reference_batch_id"] is None
    assert a["comparison"]["reason_code"] == "no_reference"
    _put(client, late, scene["dataset"])  # links by sha256 on its own; batches have no run for it
    res = _score(client, gt_set_id=gid)
    a = _by_batch(res)[scene["A"]]
    assert a["status"] == "interim" and a["pending"]["missing"] == a["pending"]["total"] == 1 and a["full_coverage"]
    assert (a["n_eval_images"], a["metrics"]["n_images"]) == (6, 5)
    row = client.get("/api/score/images", params={"gt_set_id": gid, "batch_id": scene["A"]}).json()[5]
    assert (row["filename"], row["status"], row["run_id"], row["passed"], row["n_fp"]) == ("p5.png", "missing", None, None, None)


def test_criteria_defaults_and_limits(client: TestClient) -> None:
    """GET /api/score/criteria: the request schema's defaults and bounds, which are Criteria's."""
    r = client.get("/api/score/criteria")
    assert r.status_code == 200, r.text
    info = r.json()
    c = Criteria()
    assert info["defaults"] == {k: getattr(c, k) for k in info["defaults"]} and len(info["defaults"]) == 7
    lim = info["limits"]
    assert list(lim) == list(info["defaults"])
    assert lim["center_px"] == {"min": 0, "max": None, "min_exclusive": False, "max_exclusive": False, "step": 0.5,
                                "integer": False}
    assert (lim["bootstrap_reps"]["min"], lim["bootstrap_reps"]["max"], lim["bootstrap_reps"]["integer"]) == (1, 20000, True)
    assert (lim["bootstrap_level"]["min"], lim["bootstrap_level"]["max"]) == (0, 1)
    assert lim["bootstrap_level"]["min_exclusive"] and lim["bootstrap_level"]["max_exclusive"]
    assert lim["bootstrap_seed"]["integer"] and lim["bootstrap_seed"]["min"] == 0
    schema = client.get("/openapi.json").json()["components"]["schemas"]["ScoreCriteriaIn"]["properties"]
    assert schema["center_px"]["default"] == 5.0 and schema["bootstrap_reps"]["maximum"] == 20000
    # The colour and outcome types of the API are the scoring's own.
    assert get_args(schemas.GtColor) == GT_COLORS and get_args(schemas.Color) == COLORS
    assert get_args(schemas.Outcome) == get_args(Outcome) == ("pass", "fail", "missed")


# --- deletion ------------------------------------------------------------------------------------
def test_dataset_delete_removes_its_gt_sets_and_files(client: TestClient, data_dir: Path) -> None:
    did = _dataset(client, "정답셋 지우기 시험")
    ids = [_post_gt(client, did, _doc([_item(f"{i}" * 64, "z.png", "g", [_cube("red_1")])])).json()["id"] for i in (1, 2)]
    files = [data_dir / "gt" / f"gt-{gid}.json" for gid in ids]
    assert all(f.exists() for f in files)
    assert client.delete(f"/api/datasets/{did}").json()["ok"]
    assert not any(f.exists() for f in files)
    assert all(client.get(f"/api/gt/{gid}").status_code == 404 for gid in ids)
    conn = connect(data_dir / "lab.sqlite3")
    try:
        assert conn.execute(f"SELECT COUNT(*) FROM gt_images WHERE gt_set_id IN ({ids[0]}, {ids[1]})").fetchone()[0] == 0
    finally:
        conn.close()
    assert client.get(f"/api/datasets/{did}/gt").status_code == 404

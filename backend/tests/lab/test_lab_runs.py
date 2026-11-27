"""Runs end to end through the real worker process: queue, process, cancel, list, delete."""

from __future__ import annotations

import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import cv2
import pytest
from fastapi.testclient import TestClient

from lab.config import DataPaths
from lab.db import connect, now_iso
from lab.main import DataDirBusy, create_app, lock_data_dir
from lab_testkit import INDEX_PIPELINE, VIDEO_FRAMES, upload, wait_frames, wait_run


def create(client: TestClient, **body) -> dict:
    r = client.post("/api/runs", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def test_image_run(client: TestClient, image: dict, data_dir: Path) -> None:
    run = create(client, media_id=image["id"], model_key="test.index", params={"delay_ms": 5},
                 frame_start=9, frame_end=99, stride=3, note="첫 실행")
    assert run["status"] == "queued"
    assert (run["frame_start"], run["frame_end"], run["stride"], run["frames_total"]) == (0, 1, 1, 1)
    assert run["model_name"] == INDEX_PIPELINE.name and run["media_kind"] == "image"
    assert run["params"] == {"delay_ms": 5, "proto_red": "#c8323c", "roi": []}
    run = wait_run(client, run["id"])
    assert run["status"] == "done", run["error"]
    assert run["device"] == "opencv" and run["frames_done"] == 1
    assert run["timing"]["frames"] == 1 and run["timing"]["first_ms"] >= 5
    assert run["timing"]["p50_ms"] == run["timing"]["first_ms"], "a single frame is its own statistics"
    assert run["summary"] == {"frames_with_cubes": 1, "cubes_mean": 1.0, "cubes_min": 1, "cubes_max": 1,
                              "color_mean": {"red": 1.0, "green": 0.0, "blue": 0.0, "wood": 0.0, "unknown": 0.0}}
    assert run["started_at"] and run["finished_at"] and run["note"] == "첫 실행"
    thumb = data_dir / "thumbs" / f"run-{run['id']}.jpg"
    assert run["thumb_url"] == f"/api/runs/{run['id']}/thumb?v={thumb.stat().st_mtime_ns}", "versioned like the export"
    assert client.get(run["thumb_url"]).headers["content-type"] == "image/jpeg"
    frames = client.get(f"/api/runs/{run['id']}/frames").json()
    assert frames["fps"] is None and [f["frame"] for f in frames["frames"]] == [0]
    assert frames["frames"][0]["cubes"][0]["label"] == "0"
    assert client.get(f"/api/media/{image['id']}").json()["run_count"] >= 1


def test_video_run_with_stride_stores_the_right_frames(client: TestClient, video: dict) -> None:
    run = create(client, media_id=video["id"], model_key="test.index", frame_start=5, frame_end=50, stride=3)
    assert run["frames_total"] == math.ceil((50 - 5) / 3)
    run = wait_run(client, run["id"])
    assert run["status"] == "done", run["error"]
    assert run["frames_done"] == run["frames_total"] == 15
    frames = client.get(f"/api/runs/{run['id']}/frames").json()
    assert frames["fps"] == 25 and frames["stride"] == 3
    numbers = [f["frame"] for f in frames["frames"]]
    assert numbers == list(range(5, 50, 3))
    # The worker decoded exactly the picture of each stored frame number.
    assert [int(f["cubes"][0]["label"]) for f in frames["frames"]] == numbers
    t = run["timing"]
    assert t["frames"] == 15 and t["p50_ms"] <= t["p95_ms"] <= t["max_ms"]


def test_whole_video_by_default(client: TestClient, video: dict) -> None:
    run = wait_run(client, create(client, media_id=video["id"], model_key="test.index", stride=10)["id"])
    assert (run["frame_end"], run["frames_total"], run["frames_done"]) == (VIDEO_FRAMES, 8, 8)


def test_run_validation(client: TestClient, image: dict, video: dict) -> None:
    def post(**body):
        return client.post("/api/runs", json={"media_id": video["id"], "model_key": "test.index", **body})

    assert post(frame_start=VIDEO_FRAMES).status_code == 400
    assert post(frame_start=10, frame_end=10).status_code == 400
    assert post(stride=0).status_code == 400
    assert post(media_id=99999).status_code == 404
    assert post(model_key="cv.none").status_code == 404
    assert post(model_key="w.12345").status_code == 404
    assert post(parent_run_id=99999).status_code == 400
    r = post(params={"proto_red": "#12"})
    assert r.status_code == 400 and isinstance(r.json()["detail"], str)


def test_cancel_queued_and_running(client: TestClient, video: dict) -> None:
    slow = create(client, media_id=video["id"], model_key="test.index", params={"delay_ms": 40})
    queued = create(client, media_id=video["id"], model_key="test.index", stride=5)
    r = client.post(f"/api/runs/{queued['id']}/cancel").json()
    assert r["status"] == "cancelled" and r["finished_at"], "a queued run is cancelled at once"
    wait_frames(client, slow["id"], 5)
    client.post(f"/api/runs/{slow['id']}/cancel")
    run = wait_run(client, slow["id"], timeout=5)
    assert run["status"] == "cancelled"
    assert 5 <= run["frames_done"] < VIDEO_FRAMES
    stored = client.get(f"/api/runs/{slow['id']}/frames").json()["frames"]
    assert len(stored) == run["frames_done"], "frames processed before the cancel are kept"
    assert run["timing"]["frames"] == run["frames_done"]
    assert client.get(f"/api/runs/{queued['id']}").json()["frames_done"] == 0


def test_list_filters_patch_and_delete(client: TestClient, image: dict, video: dict, data_dir: Path) -> None:
    a = wait_run(client, create(client, media_id=image["id"], model_key="test.index")["id"])
    b = wait_run(client, create(client, media_id=video["id"], model_key="test.index", stride=25,
                                parent_run_id=a["id"])["id"])
    assert b["parent_run_id"] == a["id"]
    r = client.patch(f"/api/runs/{b['id']}", json={"note": "비교용", "tags": ["기준", "기준"], "starred": True}).json()
    assert (r["note"], r["tags"], r["starred"]) == ("비교용", ["기준"], True)

    def ids(**params) -> list[int]:
        return [x["id"] for x in client.get("/api/runs", params=params).json()["items"]]

    assert ids(media_id=image["id"])[0] == a["id"]
    assert b["id"] not in ids(media_id=image["id"])
    assert ids(starred=True) == [b["id"]]
    assert ids(tag="기준") == [b["id"]]
    assert ids(q="비교") == [b["id"]] and ids(q=f"#{a['id']}") == [a["id"]]
    assert set(ids(status="done,cancelled")) >= {a["id"], b["id"]}
    assert client.get("/api/runs", params={"status": "nope"}).status_code == 400
    assert ids(model_key="cv.none") == []
    page = client.get("/api/runs", params={"limit": 2, "offset": 1}).json()
    everything = ids(limit=1000)
    assert page["total"] == len(everything) and [x["id"] for x in page["items"]] == everything[1:3]
    assert everything == sorted(everything, reverse=True)

    assert client.post(f"/api/runs/{b['id']}/export").status_code == 200
    export = data_dir / "exports" / f"run-{b['id']}.mp4"
    thumb = data_dir / "thumbs" / f"run-{b['id']}.jpg"
    assert export.exists() and thumb.exists()
    assert client.delete(f"/api/runs/{b['id']}").json() == {"ok": True}
    assert client.get(f"/api/runs/{b['id']}").status_code == 404
    assert client.get(f"/api/runs/{b['id']}/frames").status_code == 404
    assert not export.exists() and not thumb.exists()
    conn = connect(data_dir / "lab.sqlite3")
    assert conn.execute("SELECT COUNT(*) FROM run_frames WHERE run_id = ?", (b["id"],)).fetchone()[0] == 0
    conn.close()


def test_delete_media_cascades(client: TestClient, media_dir: Path, data_dir: Path, tmp_path: Path) -> None:
    import cv2

    img = cv2.imread(str(media_dir / "cubes.png"))
    img[1, 1] = (9, 9, 9)
    cv2.imwrite(str(tmp_path / "doomed.png"), img)
    m = upload(client, tmp_path / "doomed.png")["media"]
    run = wait_run(client, create(client, media_id=m["id"], model_key="test.index")["id"])
    client.post(f"/api/runs/{run['id']}/export")
    files = [data_dir / "media" / f"{m['sha256']}.png", data_dir / "thumbs" / f"media-{m['id']}.jpg",
             data_dir / "thumbs" / f"run-{run['id']}.jpg", data_dir / "exports" / f"run-{run['id']}.png"]
    assert all(f.exists() for f in files)
    assert client.delete(f"/api/media/{m['id']}").status_code == 200
    assert not any(f.exists() for f in files)
    assert client.get(f"/api/runs/{run['id']}").status_code == 404
    assert client.delete(f"/api/media/{m['id']}").status_code == 404


def test_deleting_a_running_run_leaves_the_worker_healthy(client: TestClient, video: dict) -> None:
    slow = create(client, media_id=video["id"], model_key="test.index", params={"delay_ms": 30})
    wait_frames(client, slow["id"], 3)
    pid = client.get("/api/system").json()["worker"]["pid"]
    assert client.delete(f"/api/runs/{slow['id']}").status_code == 200
    nxt = wait_run(client, create(client, media_id=video["id"], model_key="test.index", stride=25)["id"])
    assert nxt["status"] == "done"
    assert client.get("/api/system").json()["worker"]["pid"] == pid


def test_worker_restart_fails_the_running_run(client: TestClient, video: dict) -> None:
    slow = create(client, media_id=video["id"], model_key="test.index", params={"delay_ms": 50})
    wait_frames(client, slow["id"], 2)
    before = client.get("/api/system").json()["worker"]
    assert before["busy_run_id"] == slow["id"]
    after = client.post("/api/worker/restart").json()
    assert after["alive"] and after["pid"] != before["pid"]
    run = client.get(f"/api/runs/{slow['id']}").json()
    assert run["status"] == "failed" and run["error"] == "작업자를 다시 시작해서 중단됨"
    assert_stats_match_stored_frames(client, run)
    # the new worker takes new runs
    assert wait_run(client, create(client, media_id=video["id"], model_key="test.index", stride=25)["id"])[
        "status"] == "done"


def assert_stats_match_stored_frames(client: TestClient, run: dict) -> None:
    """A run cut short keeps its stored frames and gets timing and summary from them."""
    stored = client.get(f"/api/runs/{run['id']}/frames").json()["frames"]
    assert len(stored) >= 2
    assert run["frames_done"] == run["timing"]["frames"] == len(stored)
    assert run["timing"]["first_ms"] == stored[0]["ms"]
    assert run["summary"]["frames_with_cubes"] == len(stored)


def test_startup_marks_interrupted_runs_failed(tmp_path: Path) -> None:
    app = create_app(tmp_path / "재시작", extra_pipelines=[INDEX_PIPELINE])
    db = app.state.paths.db
    conn = connect(db)
    conn.execute("INSERT INTO media (sha256, kind, name, ext, bytes, width, height, created_at) "
                 "VALUES ('x', 'image', 'a.png', '.png', 1, 1, 1, ?)", (now_iso(),))
    for cancel in (0, 1):
        conn.execute("INSERT INTO runs (media_id, model_key, model_name, params, frame_start, frame_end, stride, "
                     "frames_total, status, cancel_requested, created_at) "
                     "VALUES (1, 'test.index', 't', '{}', 0, 3, 1, 3, 'running', ?, ?)", (cancel, now_iso()))
    conn.executemany("INSERT INTO run_frames (run_id, frame, ms, cubes) VALUES (1, ?, ?, ?)", [
        (0, 12.0, '[{"color": "red"}]'), (1, 4.0, '[{"color": "red"}, {"color": "blue"}]'), (2, 6.0, "[]")])
    conn.close()
    with TestClient(app) as c:
        run = c.get("/api/runs/1").json()
        cancelled = c.get("/api/runs/2").json()
    assert run["status"] == "failed" and run["error"] == "서버가 다시 시작되어 중단됨"
    assert run["frames_done"] == 3
    assert run["timing"] == {"frames": 3, "first_ms": 12.0, "mean_ms": 5.0, "p50_ms": 5.0, "p95_ms": 5.9,
                             "max_ms": 6.0}
    assert run["summary"] == {"frames_with_cubes": 2, "cubes_mean": 1.0, "cubes_min": 0, "cubes_max": 2,
                              "color_mean": {"red": 0.667, "green": 0.0, "blue": 0.333, "wood": 0.0, "unknown": 0.0}}
    assert cancelled["status"] == "cancelled" and cancelled["error"] is None and cancelled["timing"] is None


def test_ignored_cancel_forces_a_worker_restart(client: TestClient, image: dict) -> None:
    """A frame stuck longer than the grace period (e.g. a hung MPS call) is ended by restarting the worker."""
    stuck = create(client, media_id=image["id"], model_key="test.index", params={"delay_ms": 15000})
    wait_run(client, stuck["id"], until=("running",))
    pid = client.get("/api/system").json()["worker"]["pid"]
    client.post(f"/api/runs/{stuck['id']}/cancel")
    run = wait_run(client, stuck["id"], timeout=12)
    assert run["status"] == "cancelled"
    assert client.get("/api/system").json()["worker"]["pid"] != pid


def test_crashed_worker_is_replaced(client: TestClient, video: dict) -> None:
    import os
    import signal

    slow = create(client, media_id=video["id"], model_key="test.index", params={"delay_ms": 50})
    wait_frames(client, slow["id"], 2)
    pid = client.get("/api/system").json()["worker"]["pid"]
    os.kill(pid, signal.SIGKILL)
    run = wait_run(client, slow["id"], timeout=10)
    assert run["status"] == "failed" and run["error"] == "작업자가 비정상 종료되어 중단됨"
    assert_stats_match_stored_frames(client, run)
    nxt = wait_run(client, create(client, media_id=video["id"], model_key="test.index", stride=25)["id"], timeout=40)
    assert nxt["status"] == "done"
    assert client.get("/api/system").json()["worker"]["pid"] != pid


def test_deleting_during_the_first_frame_leaves_no_thumbnail(client: TestClient, image: dict, media_dir: Path,
                                                             data_dir: Path, tmp_path: Path) -> None:
    """The worker writes the thumbnail after the first frame, so a delete during that frame found
    nothing to remove: the worker must remove what it wrote once it sees the run is gone."""
    first = create(client, media_id=image["id"], model_key="test.index", params={"delay_ms": 1500})
    wait_run(client, first["id"], until=("running",))
    time.sleep(0.3)
    assert client.delete(f"/api/runs/{first['id']}").status_code == 200

    img = cv2.imread(str(media_dir / "cubes.png"))
    img[3, 3] = (5, 5, 5)  # new sha
    cv2.imwrite(str(tmp_path / "doomed.png"), img)
    m = upload(client, tmp_path / "doomed.png")["media"]
    second = create(client, media_id=m["id"], model_key="test.index", params={"delay_ms": 1500})
    wait_run(client, second["id"], until=("running",))
    time.sleep(0.3)
    assert client.delete(f"/api/media/{m['id']}").status_code == 200

    # The worker takes runs in order: once this one is done, it has finished with both.
    assert wait_run(client, create(client, media_id=image["id"], model_key="test.index")["id"])["status"] == "done"
    for rid in (first["id"], second["id"]):
        assert not (data_dir / "thumbs" / f"run-{rid}.jpg").exists(), rid


def test_worker_never_overwrites_a_run_ended_elsewhere(client: TestClient, video: dict, data_dir: Path) -> None:
    """A server started after this worker's API died ends its run; the old worker must stop and
    not write 'done' over that later."""
    slow = create(client, media_id=video["id"], model_key="test.index", params={"delay_ms": 30})
    wait_frames(client, slow["id"], 3)
    conn = connect(data_dir / "lab.sqlite3")
    try:
        conn.execute("UPDATE runs SET status = 'failed', error = '다른 서버가 끝냄' WHERE id = ?", (slow["id"],))
    finally:
        conn.close()
    nxt = wait_run(client, create(client, media_id=video["id"], model_key="test.index", stride=25)["id"])
    assert nxt["status"] == "done"
    run = client.get(f"/api/runs/{slow['id']}").json()
    assert (run["status"], run["error"]) == ("failed", "다른 서버가 끝냄")
    assert run["frames_done"] < VIDEO_FRAMES


def test_a_second_server_on_the_same_data_dir_is_refused(client: TestClient, video: dict, data_dir: Path) -> None:
    """It would mark the first server's running run failed and start a second worker."""
    slow = create(client, media_id=video["id"], model_key="test.index", params={"delay_ms": 20})
    wait_frames(client, slow["id"], 2)
    with pytest.raises(DataDirBusy, match=f"pid {os.getpid()}"):
        with TestClient(create_app(data_dir, extra_pipelines=[INDEX_PIPELINE])):
            pass
    with pytest.raises(DataDirBusy):  # the refused attempt did not release the first server's lock
        lock_data_dir(DataPaths(data_dir))
    assert wait_run(client, slow["id"])["status"] == "done"


SERVER_THEN_HANG = """
import sys, time
from pathlib import Path
sys.path.insert(0, {testkit!r})
from fastapi.testclient import TestClient
from lab.main import create_app
from lab_testkit import INDEX_PIPELINE, upload, wait_frames

if __name__ == "__main__":
    with TestClient(create_app(Path(sys.argv[1]), extra_pipelines=[INDEX_PIPELINE])) as c:
        m = upload(c, Path(sys.argv[2]))["media"]
        run = c.post("/api/runs", json={{"media_id": m["id"], "model_key": "test.index",
                                         "params": {{"delay_ms": 100}}}}).json()
        wait_frames(c, run["id"], 10)
        print(c.get("/api/system").json()["worker"]["pid"], run["id"], flush=True)
        time.sleep(120)
"""


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_worker_stops_when_its_server_is_killed(media_dir: Path, tmp_path: Path) -> None:
    """kill -9 of the API leaves its worker running. It must stop at the next frame instead of
    finishing the run next to the new server's worker (two torch processes on a 16 GB Mac)."""
    script = tmp_path / "server.py"
    script.write_text(SERVER_THEN_HANG.format(testkit=str(Path(__file__).parent)))
    data = tmp_path / "서버 데이터"
    proc = subprocess.Popen([sys.executable, str(script), str(data), str(media_dir / "index.mp4")],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    worker_pid = None
    try:
        line = proc.stdout.readline()
        assert line, "the test server did not start"
        worker_pid, run_id = map(int, line.split())
        proc.send_signal(signal.SIGKILL)
        proc.wait(10)
        deadline = time.monotonic() + 3  # the rest of the run would take about 6 s
        while _alive(worker_pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not _alive(worker_pid), "the orphaned worker kept running"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        if worker_pid and _alive(worker_pid):
            os.kill(worker_pid, signal.SIGKILL)

    conn = connect(data / "lab.sqlite3")
    try:
        stored = conn.execute("SELECT COUNT(*) FROM run_frames WHERE run_id = ?", (run_id,)).fetchone()[0]
        assert conn.execute("SELECT status FROM runs WHERE id = ?", (run_id,)).fetchone()[0] == "running"
    finally:
        conn.close()
    assert 10 <= stored < VIDEO_FRAMES
    # The kill released the data dir lock: a new server starts and ends the run from its stored frames.
    with TestClient(create_app(data, extra_pipelines=[INDEX_PIPELINE])) as c:
        run = c.get(f"/api/runs/{run_id}").json()
    assert run["status"] == "failed" and run["error"] == "서버가 다시 시작되어 중단됨"
    assert run["frames_done"] == run["timing"]["frames"] == stored

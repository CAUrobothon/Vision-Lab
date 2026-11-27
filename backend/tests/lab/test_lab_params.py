"""POST /api/params/check: hand-written params (e.g. JSON from a chat assistant) resolved the way a run
resolves them, with a report of what was applied, adjusted and ignored."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_check_reports_applied_adjusted_and_ignored(client: TestClient) -> None:
    r = client.post("/api/params/check", json={"model_key": "cv.hsv_ranges", "params": {
        "red_h": [-20, 12],  # as given
        "blur_ksize": 99,  # above its range (max 31): clamped
        "open_iters": "2",  # a string number: coerced
        "red_s": [200, 100],  # reversed: swapped
        "wood_yields": False,
        "not_a_param": 1,
    }})
    assert r.status_code == 200, r.text
    res = r.json()
    assert res["model_key"] == "cv.hsv_ranges"
    assert res["applied"] == ["red_h", "red_s", "wood_yields", "blur_ksize", "open_iters"]  # spec order
    assert res["ignored"] == ["not_a_param"]
    adjusted = {a["name"]: (a["given"], a["used"]) for a in res["adjusted"]}
    assert adjusted == {"blur_ksize": (99, 31), "red_s": ([200, 100], [100, 200])}, "'2' -> 2 is not a change"
    p = res["params"]
    assert p["red_h"] == [-20, 12] and p["blur_ksize"] == 31 and p["open_iters"] == 2 and p["wood_yields"] is False
    assert p["green_h"] == [38, 88], "params not given come back with their defaults"


def test_old_names_count_as_applied(client: TestClient) -> None:
    res = client.post("/api/params/check", json={"model_key": "cv.hsv_ranges", "params": {"red_s_min": 120}}).json()
    assert res["applied"] == ["red_s"] and res["ignored"] == [] and res["params"]["red_s"] == [120, 255]


def test_check_errors(client: TestClient) -> None:
    r = client.post("/api/params/check", json={"model_key": "cv.board_diff_lab", "params": {"proto_red": "red"}})
    assert r.status_code == 400 and "proto_red" in r.json()["detail"]
    r = client.post("/api/params/check", json={"model_key": "cv.nope", "params": {}})
    assert r.status_code == 404


def test_values_are_what_a_slider_can_show(client: TestClient) -> None:
    """Found by exploratory QA: numbers for select params fell back to the default, out-of-range
    ranges came out reversed, and off-grid values were shown but used differently."""
    res = client.post("/api/params/check", json={"model_key": "cv.edge_quads", "params": {
        "canny_aperture": 5, "region_connectivity": 8.0,  # numbers, as an assistant writes them
        "canny": [600, 700],  # both above the 0-500 range
        "gauss_ksize": 8,  # odd only (step 2 from 1)
        "min_rectangularity": 0.123,  # step 0.01
    }}).json()
    p = res["params"]
    assert p["canny_aperture"] == "5" and p["region_connectivity"] == "8"
    assert p["canny"] == [500, 500], "never low > high"
    assert p["gauss_ksize"] == 9 and p["min_rectangularity"] == 0.12
    adjusted = {a["name"]: a["used"] for a in res["adjusted"]}
    assert adjusted == {"canny": [500, 500], "gauss_ksize": 9, "min_rectangularity": 0.12}, "5 -> '5' is no change"
    assert client.post("/api/params/check", json={"model_key": "cv.edge_quads", "params": {"canny": [-50, -10]}}).json()["params"]["canny"] == [0, 0]


def test_a_value_that_is_not_an_option_is_an_error(client: TestClient) -> None:
    r = client.post("/api/params/check", json={"model_key": "cv.edge_quads", "params": {"denoise": "invalid"}})
    assert r.status_code == 400
    assert "denoise" in r.json()["detail"] and "bilateral" in r.json()["detail"], "names the allowed values"
    # Presets go through the same rules.
    r = client.post("/api/presets", json={"model_key": "cv.edge_quads", "name": "범위 시험", "params": {"canny": [600, 700]}})
    assert r.status_code == 200 and r.json()["params"]["canny"] == [500, 500]

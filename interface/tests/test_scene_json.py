"""Scene <-> JSON round trip and the invariants of the data model."""

from __future__ import annotations

import json

import pytest

from vision_interface import (
    SCHEMA, SCHEMA_VERSION, Color, Cube, CubePose, FakeVisionSource, Mission, Scene,
    SyntheticPixelToWorld, localize, random_scene,
)


def _cube(**kw) -> Cube:
    base = dict(color="red", center_px=(100.25, 200.5), corners_px=((70, 170), (130, 171), (131, 230), (69, 229)),
                image_angle_deg=1.0, area_px2=3600.0, score=0.9, color_score=0.8, flags=["side_band"])
    base.update(kw)
    return Cube(**base)


def test_pixel_only_scene_round_trips_exactly() -> None:
    s = Scene(frame_id=7, timestamp_s=1764400000.123456, image_size_px=(1280, 720),
              cubes=(_cube(), _cube(color=Color.WOOD, corners_px=None, image_angle_deg=None, flags=())),
              mission=Mission.B, source="test", notes=("hello",))
    text = s.to_json()
    back = Scene.from_json(text)
    assert back == s
    d = json.loads(text)
    assert d["schema"] == SCHEMA and d["schema_version"] == SCHEMA_VERSION
    assert d["localized_by"] is None and d["cubes"][0]["pose"] is None
    assert d["cubes"][0]["color"] == "red" and d["mission"] == "B"


def test_localized_scene_round_trips_exactly() -> None:
    s = localize(random_scene("A", seed=5, timestamp_s=1.5), SyntheticPixelToWorld())
    back = Scene.from_json(s.to_json(indent=2))
    assert back == s
    assert all(isinstance(c.pose, CubePose) for c in back.cubes)
    assert back.is_synthetic


def test_types_are_coerced() -> None:
    c = _cube(color="green", center_px=[1, 2], flags=["a", "b"])
    assert c.color is Color.GREEN and c.center_px == (1.0, 2.0) and c.flags == ("a", "b")
    assert isinstance(c.corners_px, tuple) and len(c.corners_px) == 4


@pytest.mark.parametrize("bad", [
    dict(image_angle_deg=90.0), dict(image_angle_deg=-0.1), dict(score=1.2), dict(color="purple"),
    dict(corners_px=((0, 0), (1, 0), (1, 1))), dict(center_px=(float("nan"), 1.0)),
])
def test_invalid_cube_is_rejected(bad: dict) -> None:
    with pytest.raises((ValueError, TypeError)):
        _cube(**bad)


def test_pose_requires_a_named_calibration() -> None:
    pose = CubePose(x_mm=1, y_mm=2, z_mm=25, yaw_deg=10, grasp_yaw_deg=10)
    with pytest.raises(ValueError, match="localized_by"):
        Scene(frame_id=0, timestamp_s=0, image_size_px=(10, 10), cubes=(_cube(pose=pose),))
    with pytest.raises(ValueError, match="some cubes have no pose"):
        Scene(frame_id=0, timestamp_s=0, image_size_px=(10, 10), cubes=(_cube(pose=pose), _cube()),
              localized_by="calib")


def test_wrong_schema_is_rejected() -> None:
    d = random_scene("A", seed=1).to_dict()
    with pytest.raises(ValueError, match="schema_version"):
        Scene.from_dict({**d, "schema_version": 2})
    with pytest.raises(ValueError, match="not a vision scene"):
        Scene.from_dict({**d, "schema": "something.else"})


def test_json_file_replay(tmp_path) -> None:
    scenes = [random_scene("B", seed=k, frame_id=k, timestamp_s=float(k)) for k in range(3)]
    path = tmp_path / "scenes.json"
    path.write_text(json.dumps([s.to_dict() for s in scenes]), encoding="utf-8")
    src = FakeVisionSource.from_json_file(path)
    assert [src.get_scene() for _ in range(4)] == [scenes[0], scenes[1], scenes[2], scenes[0]]
    single = tmp_path / "one.json"
    single.write_text(scenes[1].to_json(), encoding="utf-8")
    once = FakeVisionSource.from_json_file(single, loop=False)
    assert once.get_scene() == scenes[1] and once.get_scene() == scenes[1]

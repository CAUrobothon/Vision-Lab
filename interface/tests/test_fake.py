"""FakeVisionSource: rule-respecting scenes, pixel-only by default, synthetic mm clearly labelled."""

from __future__ import annotations

import json
import math
from collections import Counter

import pytest

from vision_interface import (
    Color, FakeVisionSource, Mission, Scene, SyntheticPixelToWorld, VisionSource, localize, random_scene,
)
from vision_interface import rules
from vision_interface.fake import main as fake_main

SEEDS = range(200)


def test_fake_is_a_vision_source() -> None:
    assert isinstance(FakeVisionSource.random("A", seed=1), VisionSource)


@pytest.mark.parametrize("mission", ["A", "B"])
def test_default_path_has_no_millimetres(mission: str) -> None:
    scene = FakeVisionSource.random(mission, seed=3).get_scene()
    assert scene.localized_by is None and not scene.is_localized
    assert all(c.pose is None for c in scene.cubes)
    assert scene.mission is Mission(mission)


@pytest.mark.parametrize("mission, counts", [
    ("A", {Color.RED: 3, Color.GREEN: 3, Color.BLUE: 3}),  # §3 p.5, §5 p.7: no wood in mission A
    ("B", {Color.RED: 3, Color.GREEN: 3, Color.BLUE: 3, Color.WOOD: 3}),  # §4 p.6, §5 p.7
])
def test_random_scenes_obey_the_rules(mission: str, counts: dict) -> None:
    p2w = SyntheticPixelToWorld()
    for seed in SEEDS:
        scene = random_scene(mission, seed=seed)
        assert Counter(c.color for c in scene.cubes) == counts
        assert [c.color for c in scene.cubes] == list(rules.MISSION_CUBES[Mission(mission)])  # rule order
        w, h = scene.image_size_px
        for c in scene.cubes:
            assert all(0 <= u < w and 0 <= v < h for u, v in c.corners_px)
            assert 0.0 <= c.image_angle_deg < 90.0
        loc = localize(scene, p2w)
        for c in loc.cubes:  # every cube starts inside a start zone of its mission (synthetic frame)
            zone = rules.zone_of(c.pose.x_mm, c.pose.y_mm, mission)
            assert zone in {s.name for s in rules.START_SECTORS[Mission(mission)]}, (seed, c.pose)
        pts = [(c.pose.x_mm, c.pose.y_mm) for c in loc.cubes]
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                assert math.dist(pts[i], pts[j]) > rules.CUBE_SIZE_MM * math.sqrt(2), seed  # no overlap
        side = [math.dist(a, b) for a, b in zip(c.corners_px, c.corners_px[1:] + c.corners_px[:1])]
        assert all(math.isclose(s, rules.CUBE_SIZE_MM / p2w.mm_per_px) for s in side)


def test_seed_is_deterministic() -> None:
    a = random_scene("B", seed=11, timestamp_s=0.0)
    b = random_scene("B", seed=11, timestamp_s=0.0)
    assert a == b and a != random_scene("B", seed=12, timestamp_s=0.0)


def test_marked_positions_follow_the_draw() -> None:
    slots = [(-100.0 - 5 * k, 120.0 + 3 * k) for k in range(rules.PLACEMENT_SLOTS)]
    scene = localize(random_scene("B", seed=1, positions_mm=slots), SyntheticPixelToWorld())
    used = [(round(c.pose.x_mm, 6), round(c.pose.y_mm, 6)) for c in scene.cubes]
    assert len(set(used)) == 12 and set(used) <= {(round(x, 6), round(y, 6)) for x, y in slots}
    with pytest.raises(ValueError):
        random_scene("B", seed=1, positions_mm=slots[:5])


def test_random_source_restamps_and_remove_simulates_a_pick() -> None:
    src = FakeVisionSource.random("A", seed=7)
    s1, s2 = src.get_scene(), src.get_scene()
    assert s2.frame_id == s1.frame_id + 1 and s1.cubes == s2.cubes
    picked = localize(s2, SyntheticPixelToWorld()).cubes[0]  # a localised copy is matched by pixels
    assert src.remove(picked) is True
    assert len(src.get_scene().cubes) == 8
    assert src.remove(picked) is False


def test_synthetic_mapping_is_labelled() -> None:
    p2w = SyntheticPixelToWorld()
    assert p2w.name.startswith("SYNTHETIC")
    assert localize(random_scene("A", seed=1), p2w).is_synthetic


def test_cli_prints_scenes(capsys, tmp_path) -> None:
    assert fake_main(["--mission", "A", "--seed", "3"]) == 0
    one = Scene.from_json(capsys.readouterr().out)
    assert len(one.cubes) == 9 and one.localized_by is None
    out = tmp_path / "s.json"
    assert fake_main(["--mission", "B", "--seed", "3", "--count", "2", "--localize-synthetic", "--out", str(out)]) == 0
    many = [Scene.from_dict(d) for d in json.loads(out.read_text(encoding="utf-8"))]
    assert len(many) == 2 and all(s.is_synthetic and len(s.cubes) == 12 for s in many)
    assert FakeVisionSource.from_json_file(out).get_scene() == many[0]


def test_readme_example_runs(capsys) -> None:
    import runpy
    from pathlib import Path

    runpy.run_path(str(Path(__file__).resolve().parents[1] / "examples" / "pick_red_fake.py"), run_name="__main__")
    out = capsys.readouterr().out
    assert out.count("집기:") == 3  # three red cubes in mission B (§4 p.6)

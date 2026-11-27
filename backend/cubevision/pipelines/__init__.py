"""Built-in pipelines, in display order. Add new pipeline instances to ALL."""

from __future__ import annotations

from cubevision.pipelines.base import Pipeline
from cubevision.pipelines.board_diff_lab import BoardDiffLab
from cubevision.pipelines.edge_quads import EdgeQuads
from cubevision.pipelines.hsv_ranges import HsvRanges
from cubevision.pipelines.lab_autocluster import LabAutocluster

ALL: list[Pipeline] = [HsvRanges(), BoardDiffLab(), EdgeQuads(), LabAutocluster()]

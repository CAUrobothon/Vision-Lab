"""The interface every built-in OpenCV pipeline implements.

A pipeline is a pure function of (BGR frame, params) -> FrameResult. It must not keep
state between frames, must not read or write files, and must not import torch.
The same code is meant to run inside the competition program later, so keep it free
of any web-server dependency.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from cubevision.types import FrameResult, ParamSpec, resolve_params


class Pipeline(ABC):
    key: str  # stable id, e.g. "cv.hsv_threshold" (stored in run records -- never rename)
    name: str  # Korean display name
    summary: str  # one-line Korean description shown in the model picker
    description: str  # a few lines: idea, strengths, known weak spots (Korean)
    params: list[ParamSpec]
    # Korean labels for the debug layers this pipeline can return (FrameResult.debug keys).
    debug_labels: dict[str, str] = {}

    def migrate(self, given: dict[str, Any] | None) -> dict[str, Any]:
        """Rewrite parameter names of older records into today's names (default: none)."""
        return dict(given or {})

    def resolve(self, given: dict[str, Any] | None) -> dict[str, Any]:
        return resolve_params(self.params, self.migrate(given))

    def __call__(self, frame_bgr: np.ndarray, params: dict[str, Any] | None = None, *, debug: bool = False) -> FrameResult:
        if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("frame_bgr must be an HxWx3 uint8 BGR image")
        return self.run(frame_bgr, self.resolve(params), debug=debug)

    @abstractmethod
    def run(self, frame_bgr: np.ndarray, params: dict[str, Any], *, debug: bool = False) -> FrameResult:
        """``params`` is already resolved (defaults filled, numbers clamped)."""

    def info(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "kind": "pipeline",
            "name": self.name,
            "summary": self.summary,
            "description": self.description,
            "params": [p.to_dict() for p in self.params],
            "debug_labels": dict(self.debug_labels),
        }

"""Where the lab keeps its files, and the limits it enforces."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"

# Uploads, runs and exports are refused below this much free disk (macOS swap shares the volume).
MIN_FREE_BYTES = 2 * 1024**3
MAX_MEDIA_BYTES = 4 * 1024**3
MAX_WEIGHTS_BYTES = 1024**3
MAX_GT_BYTES = 50 * 1024**2  # a ground-truth JSON (the 410-image set is about 5 MB)
MAX_BOOTSTRAP_REPS = 20000  # resamples per scoring request: beyond this one request holds a thread for minutes


def resolve_data_dir(given: Path | str | None = None) -> Path:
    """--data / create_app(data_dir) first, then $LAB_DATA_DIR, then <project>/data."""
    raw = given or os.environ.get("LAB_DATA_DIR") or PROJECT_ROOT / "data"
    return Path(raw).expanduser().resolve()


@dataclass(frozen=True)
class DataPaths:
    """Every directory and file the lab writes, all below one data directory."""

    root: Path

    @property
    def db(self) -> Path:
        return self.root / "lab.sqlite3"

    @property
    def lock(self) -> Path:
        """Locked by the one server that uses this data dir (main.lock_data_dir)."""
        return self.root / "lab.lock"

    @property
    def media(self) -> Path:
        return self.root / "media"

    @property
    def proxy(self) -> Path:
        return self.root / "proxy"

    @property
    def thumbs(self) -> Path:
        return self.root / "thumbs"

    @property
    def weights(self) -> Path:
        return self.root / "weights"

    @property
    def exports(self) -> Path:
        return self.root / "exports"

    @property
    def tmp(self) -> Path:
        return self.root / "tmp"

    @property
    def ultralytics(self) -> Path:
        return self.root / "ultralytics"

    @property
    def gt(self) -> Path:
        """Uploaded ground-truth JSON files, kept byte for byte (docs/SCORING.md)."""
        return self.root / "gt"

    def ensure(self) -> None:
        for d in (self.media, self.proxy, self.thumbs, self.weights, self.exports, self.tmp, self.ultralytics, self.gt):
            d.mkdir(parents=True, exist_ok=True)

    # File naming (DESIGN.md §4). Ids are never reused (AUTOINCREMENT), so names are stable.
    def media_thumb(self, media_id: int) -> Path:
        return self.thumbs / f"media-{media_id}.jpg"

    def run_thumb(self, run_id: int) -> Path:
        return self.thumbs / f"run-{run_id}.jpg"

    def run_exports(self, run_id: int) -> list[Path]:
        return [self.exports / f"run-{run_id}.png", self.exports / f"run-{run_id}.mp4"]

    def gt_file(self, gt_set_id: int) -> Path:
        return self.gt / f"gt-{gt_set_id}.json"

"""SQLite storage (stdlib sqlite3). Shared by the API process and the worker process.

One connection per request or thread; WAL lets the worker write frames while the API reads.
Connections run in autocommit mode: single statements commit on their own, and
``transaction()`` groups several statements (BEGIN IMMEDIATE takes the write lock up front,
so two writers wait for each other via busy_timeout instead of failing on upgrade).
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

# Each entry upgrades the schema by one version (PRAGMA user_version). Append, never edit.
MIGRATIONS: list[str] = [
    """
    CREATE TABLE media (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        sha256      TEXT NOT NULL UNIQUE,
        kind        TEXT NOT NULL CHECK (kind IN ('image', 'video')),
        name        TEXT NOT NULL,
        ext         TEXT NOT NULL,             -- extension of the stored original, e.g. '.jpg'
        proxy       TEXT,                      -- browser/decoder copy relative to the data dir, or NULL
        bytes       INTEGER NOT NULL,
        width       INTEGER NOT NULL,
        height      INTEGER NOT NULL,
        fps         REAL,
        frame_count INTEGER,
        duration    REAL,
        tags        TEXT NOT NULL DEFAULT '[]',
        note        TEXT NOT NULL DEFAULT '',
        starred     INTEGER NOT NULL DEFAULT 0,
        created_at  TEXT NOT NULL
    );
    CREATE TABLE weights (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        name        TEXT NOT NULL,
        file        TEXT NOT NULL,             -- relative to the data dir
        bytes       INTEGER NOT NULL,
        sha256      TEXT NOT NULL,
        task        TEXT,
        class_names TEXT NOT NULL DEFAULT '[]',
        imgsz       INTEGER,
        state       TEXT NOT NULL CHECK (state IN ('ready', 'error')),
        error       TEXT,
        note        TEXT NOT NULL DEFAULT '',
        created_at  TEXT NOT NULL
    );
    CREATE TABLE presets (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        model_key   TEXT NOT NULL,
        name        TEXT NOT NULL,
        params      TEXT NOT NULL,
        created_at  TEXT NOT NULL,
        UNIQUE (model_key, name)
    );
    CREATE TABLE runs (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        media_id         INTEGER NOT NULL REFERENCES media(id) ON DELETE CASCADE,
        model_key        TEXT NOT NULL,
        model_name       TEXT NOT NULL,
        params           TEXT NOT NULL,
        frame_start      INTEGER NOT NULL,
        frame_end        INTEGER,
        stride           INTEGER NOT NULL,
        frames_total     INTEGER,
        frames_done      INTEGER NOT NULL DEFAULT 0,
        status           TEXT NOT NULL CHECK (status IN ('queued', 'running', 'done', 'failed', 'cancelled')),
        cancel_requested INTEGER NOT NULL DEFAULT 0,
        device           TEXT,
        timing           TEXT,
        summary          TEXT,
        error            TEXT,
        parent_run_id    INTEGER REFERENCES runs(id) ON DELETE SET NULL,
        tags             TEXT NOT NULL DEFAULT '[]',
        note             TEXT NOT NULL DEFAULT '',
        starred          INTEGER NOT NULL DEFAULT 0,
        created_at       TEXT NOT NULL,
        started_at       TEXT,
        finished_at      TEXT
    );
    CREATE INDEX runs_media ON runs (media_id);
    CREATE INDEX runs_status ON runs (status);
    CREATE TABLE run_frames (
        run_id  INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
        frame   INTEGER NOT NULL,
        ms      REAL NOT NULL,
        cubes   TEXT NOT NULL,
        PRIMARY KEY (run_id, frame)
    ) WITHOUT ROWID;
    -- One row written by the worker process so the API can report what it is doing.
    CREATE TABLE worker (
        id            INTEGER PRIMARY KEY CHECK (id = 1),
        pid           INTEGER,
        busy_run_id   INTEGER,
        loaded_model  TEXT,
        mps_available INTEGER,                 -- NULL until the worker has imported torch
        updated_at    TEXT NOT NULL
    );
    """,
    # 2: datasets (named groups of media; one media can be in several) and batches (one run per
    # media of a dataset, started together with the same model and params).
    """
    CREATE TABLE datasets (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        name        TEXT NOT NULL UNIQUE,
        note        TEXT NOT NULL DEFAULT '',
        created_at  TEXT NOT NULL
    );
    CREATE TABLE dataset_media (
        dataset_id  INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
        media_id    INTEGER NOT NULL REFERENCES media(id) ON DELETE CASCADE,
        added_at    TEXT NOT NULL,
        PRIMARY KEY (dataset_id, media_id)
    ) WITHOUT ROWID;
    CREATE INDEX dataset_media_media ON dataset_media (media_id);
    CREATE TABLE batches (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        dataset_id   INTEGER REFERENCES datasets(id) ON DELETE SET NULL,
        dataset_name TEXT NOT NULL,             -- kept when the dataset is deleted or renamed
        model_key    TEXT NOT NULL,
        model_name   TEXT NOT NULL,
        params       TEXT NOT NULL,
        stride       INTEGER NOT NULL,
        created_at   TEXT NOT NULL
    );
    ALTER TABLE runs ADD COLUMN batch_id INTEGER REFERENCES batches(id) ON DELETE SET NULL;
    CREATE INDEX runs_batch ON runs (batch_id);
    """,
    # 3: folders inside a dataset. A folder is its path of names joined by '/' ('증강/어둡게');
    # every folder's parents have rows too, so empty folders exist. dataset_media.folder '' = the
    # dataset's top level. A batch runs over the whole dataset (folder NULL) or one folder with
    # everything below it.
    """
    CREATE TABLE dataset_folders (
        dataset_id  INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
        path        TEXT NOT NULL,
        created_at  TEXT NOT NULL,
        PRIMARY KEY (dataset_id, path)
    ) WITHOUT ROWID;
    ALTER TABLE dataset_media ADD COLUMN folder TEXT NOT NULL DEFAULT '';
    CREATE INDEX dataset_media_folder ON dataset_media (dataset_id, folder);
    ALTER TABLE batches ADD COLUMN folder TEXT;
    """,
    # 4: ground-truth sets of a dataset (docs/SCORING.md). The uploaded JSON is kept byte for byte
    # in data/gt/gt-<id>.json; gt_images indexes its images so links are plain lookups. Links by
    # sha256 are never stored: they follow the dataset's current media. gt_links holds only the
    # links the user confirmed (same file name, different sha256).
    """
    CREATE TABLE gt_sets (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        dataset_id      INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
        name            TEXT NOT NULL,
        schema_version  TEXT NOT NULL,
        label_kind      TEXT,
        file_sha256     TEXT NOT NULL,
        file_bytes      INTEGER NOT NULL,
        n_images        INTEGER NOT NULL,
        n_cubes         INTEGER NOT NULL,
        n_valid         INTEGER NOT NULL,      -- cubes with valid_geometry
        created_at      TEXT NOT NULL,
        UNIQUE (dataset_id, file_sha256)
    );
    CREATE TABLE gt_images (
        gt_set_id     INTEGER NOT NULL REFERENCES gt_sets(id) ON DELETE CASCADE,
        gt_index      INTEGER NOT NULL,        -- position in the file's items
        sha256        TEXT NOT NULL,
        filename      TEXT NOT NULL,           -- NFC
        variant       TEXT NOT NULL,           -- augmentation.name, 'original' when absent
        layout_group  TEXT NOT NULL,
        PRIMARY KEY (gt_set_id, gt_index)
    ) WITHOUT ROWID;
    CREATE INDEX gt_images_sha256 ON gt_images (sha256);
    CREATE TABLE gt_links (
        gt_set_id   INTEGER NOT NULL,
        gt_index    INTEGER NOT NULL,
        media_id    INTEGER NOT NULL REFERENCES media(id) ON DELETE CASCADE,
        created_at  TEXT NOT NULL,
        PRIMARY KEY (gt_set_id, gt_index),
        UNIQUE (gt_set_id, media_id),
        FOREIGN KEY (gt_set_id, gt_index) REFERENCES gt_images(gt_set_id, gt_index) ON DELETE CASCADE
    ) WITHOUT ROWID;
    CREATE INDEX gt_links_media ON gt_links (media_id);
    """,
]


def now_iso() -> str:
    """Local time with its UTC offset, e.g. 2026-11-25T14:03:12.123+09:00."""
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def connect(db_path: Path) -> sqlite3.Connection:
    # check_same_thread=False: FastAPI may run a dependency and its endpoint on different
    # threadpool threads; the connection is still used by one request at a time.
    conn = sqlite3.connect(db_path, timeout=5.0, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def migrate(db_path: Path) -> None:
    conn = connect(db_path)
    try:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        for i, sql in enumerate(MIGRATIONS[version:], start=version + 1):
            conn.executescript(f"BEGIN IMMEDIATE;\n{sql}\nPRAGMA user_version = {i};\nCOMMIT;")
    finally:
        conn.close()


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def loads(text: str | None, default: Any = None) -> Any:
    return default if text is None else json.loads(text)

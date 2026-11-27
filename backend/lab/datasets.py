"""Datasets (named groups of media) and batches (one run per media of a dataset, started together
with the same model and params), plus the zip downloads of both.

A media can be in several datasets; deleting a dataset only removes the grouping. A batch keeps
the dataset's name, so its runs stay readable after the dataset is renamed or deleted.

A dataset holds nested folders, each a path of names joined by '/' ('증강/어둡게'). Every folder's
parents are stored too, so empty folders exist; a media sits in one folder of each dataset it is
in ('' = the dataset's top level). Deleting a folder takes its media out of that dataset only.
Zips keep the folders as directories, and a batch runs over the whole dataset or one folder.
"""

from __future__ import annotations

import csv
import io
import json
import re
import sqlite3
import uuid
import zipfile
from pathlib import Path
from typing import Any

from lab import export as export_mod
from lab import media as media_mod
from lab import models
from lab import runs as runs_mod
from lab.config import DataPaths
from lab.db import dumps, loads, now_iso, transaction
from lab.errors import LabError
from lab.schemas import (
    Batch, BatchCreate, BatchDetail, BatchItem, Dataset, DatasetFolder, RunCreate, RunSummary,
)
from lab.storage import guard_disk
from lab.text import nfc

STATUSES = ("queued", "running", "done", "failed", "cancelled")

_SELECT = """
    SELECT d.*,
        (SELECT COUNT(*) FROM dataset_media dm WHERE dm.dataset_id = d.id) AS media_count,
        (SELECT COUNT(*) FROM dataset_media dm JOIN media m ON m.id = dm.media_id
          WHERE dm.dataset_id = d.id AND m.kind = 'image') AS image_count,
        (SELECT COUNT(*) FROM dataset_media dm JOIN media m ON m.id = dm.media_id
          WHERE dm.dataset_id = d.id AND m.kind = 'video') AS video_count,
        (SELECT COUNT(*) FROM batches b WHERE b.dataset_id = d.id) AS batch_count
    FROM datasets d
"""


# --- datasets ----------------------------------------------------------------------------------
def get_row(conn: sqlite3.Connection, dataset_id: int) -> sqlite3.Row:
    row = conn.execute(_SELECT + " WHERE d.id = ?", (dataset_id,)).fetchone()
    if row is None:
        raise LabError(404, "데이터셋을 찾을 수 없습니다")
    return row


def to_api(row: sqlite3.Row, folders: list[DatasetFolder] | None = None) -> Dataset:
    return Dataset(
        id=row["id"], name=row["name"], note=row["note"], created_at=row["created_at"],
        media_count=row["media_count"], image_count=row["image_count"], video_count=row["video_count"],
        batch_count=row["batch_count"], originals_url=f"/api/datasets/{row['id']}/originals.zip",
        folders=folders or [],
    )


def get(conn: sqlite3.Connection, dataset_id: int) -> Dataset:
    row = get_row(conn, dataset_id)
    return to_api(row, _folders_api(conn, dataset_id).get(dataset_id))


def list_datasets(conn: sqlite3.Connection) -> list[Dataset]:
    folders = _folders_api(conn)
    return [to_api(r, folders.get(r["id"])) for r in conn.execute(_SELECT + " ORDER BY d.name COLLATE NOCASE, d.id")]


def _name_taken(conn: sqlite3.Connection, name: str, but: int | None = None) -> bool:
    row = conn.execute("SELECT id FROM datasets WHERE name = ?", (name,)).fetchone()
    return row is not None and row["id"] != but


def create(conn: sqlite3.Connection, name: str, note: str = "", media_ids: list[int] | None = None) -> Dataset:
    name = name.strip()
    if not name:
        raise LabError(400, "데이터셋 이름을 입력하세요")
    if _name_taken(conn, name):
        raise LabError(409, f'"{name}" 데이터셋이 이미 있습니다')
    _check_media(conn, media_ids or [])
    with transaction(conn):
        cur = conn.execute("INSERT INTO datasets (name, note, created_at) VALUES (?, ?, ?)", (name, note, now_iso()))
        did = int(cur.lastrowid)
        _insert_media(conn, did, media_ids or [])
    return get(conn, did)


def patch(conn: sqlite3.Connection, dataset_id: int, name: str | None, note: str | None) -> Dataset:
    get_row(conn, dataset_id)
    if name is not None:
        name = name.strip()
        if not name:
            raise LabError(400, "데이터셋 이름을 입력하세요")
        if _name_taken(conn, name, but=dataset_id):
            raise LabError(409, f'"{name}" 데이터셋이 이미 있습니다')
        conn.execute("UPDATE datasets SET name = ? WHERE id = ?", (name, dataset_id))
    if note is not None:
        conn.execute("UPDATE datasets SET note = ? WHERE id = ?", (note, dataset_id))
    return get(conn, dataset_id)


def delete(conn: sqlite3.Connection, dataset_id: int) -> int:
    """Removes the grouping and its batch records (the zip downloads); the media and every run
    stay, listed on the runs page. Returns the number of batches removed."""
    get_row(conn, dataset_id)
    with transaction(conn):
        n = conn.execute("DELETE FROM batches WHERE dataset_id = ?", (dataset_id,)).rowcount
        conn.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))
    return n


def _check_media(conn: sqlite3.Connection, media_ids: list[int]) -> None:
    if not media_ids:
        return
    marks = ",".join("?" * len(media_ids))
    found = {r["id"] for r in conn.execute(f"SELECT id FROM media WHERE id IN ({marks})", media_ids)}
    missing = sorted(set(media_ids) - found)
    if missing:
        raise LabError(404, f"미디어를 찾을 수 없습니다: {', '.join(map(str, missing))}")


def _insert_media(conn: sqlite3.Connection, dataset_id: int, media_ids: list[int]) -> None:
    now = now_iso()
    conn.executemany(
        "INSERT OR IGNORE INTO dataset_media (dataset_id, media_id, added_at) VALUES (?, ?, ?)",
        [(dataset_id, mid, now) for mid in dict.fromkeys(media_ids)],
    )


def add_media(conn: sqlite3.Connection, dataset_id: int, media_ids: list[int]) -> Dataset:
    get_row(conn, dataset_id)
    _check_media(conn, media_ids)
    with transaction(conn):
        _insert_media(conn, dataset_id, media_ids)
    return get(conn, dataset_id)


def remove_media(conn: sqlite3.Connection, dataset_id: int, media_ids: list[int]) -> Dataset:
    get_row(conn, dataset_id)
    marks = ",".join("?" * len(media_ids))
    conn.execute(f"DELETE FROM dataset_media WHERE dataset_id = ? AND media_id IN ({marks})", [dataset_id, *media_ids])
    return get(conn, dataset_id)


def media_rows(conn: sqlite3.Connection, dataset_id: int, folder: str = "") -> list[sqlite3.Row]:
    """The media of a dataset, or of one folder with everything below it; ``ds_folder`` = where each is."""
    cond, args = in_folder("dm.folder", folder)
    return conn.execute(
        f"""SELECT m.*, dm.folder AS ds_folder FROM media m JOIN dataset_media dm ON dm.media_id = m.id
            WHERE dm.dataset_id = ? AND {cond} ORDER BY dm.folder, m.name COLLATE NOCASE, m.id""",
        (dataset_id, *args),
    ).fetchall()


# --- folders -----------------------------------------------------------------------------------
FOLDER_NAME_MAX = 80  # characters per folder name
FOLDER_DEPTH_MAX = 8  # names per path
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


def clean_folder(raw: str | None) -> str:
    """A folder path as stored: NFC (macOS may send decomposed Hangul), each name stripped, joined
    by '/'; '' = the dataset's top level. Anything but a plain relative path is refused (400)."""
    path = nfc(raw or "").strip()
    if not path:
        return ""
    if "\\" in path:
        raise LabError(400, "폴더 경로에는 \\ 대신 / 를 쓰세요")
    if _CONTROL.search(path):
        raise LabError(400, "폴더 이름에 제어 문자를 쓸 수 없습니다")
    if path.startswith("/") or re.match(r"[A-Za-z]:", path):
        raise LabError(400, "폴더 경로는 데이터셋 안의 상대 경로여야 합니다 (/ 나 드라이브 문자로 시작할 수 없음)")
    names = [n.strip() for n in path.split("/")]
    if "" in names:
        raise LabError(400, "폴더 경로에 빈 이름이 있습니다 (// 나 끝의 /)")
    if "." in names or ".." in names:
        raise LabError(400, "폴더 이름으로 . 이나 .. 은 쓸 수 없습니다")
    if len(names) > FOLDER_DEPTH_MAX:
        raise LabError(400, f"폴더는 {FOLDER_DEPTH_MAX}단계까지만 만들 수 있습니다")
    for n in names:
        if len(n) > FOLDER_NAME_MAX:
            raise LabError(400, f"폴더 이름은 {FOLDER_NAME_MAX}자까지입니다: {n[:20]}…")
    return "/".join(names)


def in_folder(column: str, folder: str) -> tuple[str, list[Any]]:
    """SQL condition: ``column`` is ``folder`` or a folder below it ('' matches everything).
    Prefixes are compared with substr, not LIKE, so % and _ in names stay plain characters."""
    if not folder:
        return "1", []
    return f"({column} = ? OR substr({column}, 1, ?) = ?)", [folder, len(folder) + 1, folder + "/"]


def is_within(path: str, folder: str) -> bool:
    """``in_folder`` in Python: ``path`` is ``folder`` or a folder below it ('' holds everything)."""
    return not folder or path == folder or path.startswith(folder + "/")


def _with_parents(path: str) -> list[str]:
    """'a/b/c' -> ['a', 'a/b', 'a/b/c']."""
    names = path.split("/")
    return ["/".join(names[: i + 1]) for i in range(len(names))]


def folder_paths(conn: sqlite3.Connection, dataset_id: int) -> list[str]:
    return [r["path"] for r in conn.execute("SELECT path FROM dataset_folders WHERE dataset_id = ? ORDER BY path", (dataset_id,))]


def _canonical(paths: list[str], path: str) -> str:
    """``path`` with each level spelled like an existing folder that differs only in letter case:
    unpacked on macOS or Windows, 'Aug/' and 'aug/' would be one directory."""
    known = {p.casefold(): p for p in paths}
    out = ""
    for name in path.split("/") if path else []:
        candidate = f"{out}/{name}" if out else name
        out = known.get(candidate.casefold(), candidate)
    return out


def _ensure_folder(conn: sqlite3.Connection, dataset_id: int, path: str) -> str:
    """Makes ``path`` and any missing parent (inside the caller's transaction); returns it as stored."""
    if not path:
        return ""
    path = _canonical(folder_paths(conn, dataset_id), path)
    now = now_iso()
    conn.executemany(
        "INSERT OR IGNORE INTO dataset_folders (dataset_id, path, created_at) VALUES (?, ?, ?)",
        [(dataset_id, p, now) for p in _with_parents(path)],
    )
    return path


def _existing_folder(conn: sqlite3.Connection, dataset_id: int, path: str) -> str:
    """``path`` as stored; 404 when the dataset has no such folder."""
    paths = folder_paths(conn, dataset_id)
    found = _canonical(paths, path)
    if found not in paths:
        raise LabError(404, f'"{path}" 폴더를 찾을 수 없습니다')
    return found


def _folders_api(conn: sqlite3.Connection, dataset_id: int | None = None) -> dict[int, list[DatasetFolder]]:
    """Every folder with its media counts, per dataset (one dataset, or all)."""
    where, args = ("WHERE dataset_id = ?", [dataset_id]) if dataset_id is not None else ("", [])
    counts: dict[int, dict[str, int]] = {}
    for r in conn.execute(f"SELECT dataset_id, folder, COUNT(*) AS n FROM dataset_media {where} GROUP BY dataset_id, folder", args):
        counts.setdefault(r["dataset_id"], {})[r["folder"]] = r["n"]
    out: dict[int, list[DatasetFolder]] = {}
    for r in conn.execute(f"SELECT dataset_id, path FROM dataset_folders {where} ORDER BY dataset_id, path", args):
        here, path = counts.get(r["dataset_id"], {}), r["path"]
        total = sum(n for f, n in here.items() if is_within(f, path))
        out.setdefault(r["dataset_id"], []).append(DatasetFolder(path=path, count=here.get(path, 0), total=total))
    return out


def create_folder(conn: sqlite3.Connection, dataset_id: int, raw_path: str) -> Dataset:
    """A new empty folder (missing parents are made too); 409 if it exists."""
    get_row(conn, dataset_id)
    path = clean_folder(raw_path)
    if not path:
        raise LabError(400, "폴더 이름을 입력하세요")
    with transaction(conn):
        paths = folder_paths(conn, dataset_id)
        if (same := _canonical(paths, path)) in paths:
            raise LabError(409, f'"{same}" 폴더가 이미 있습니다')
        _ensure_folder(conn, dataset_id, path)
    return get(conn, dataset_id)


def rename_folder(conn: sqlite3.Connection, dataset_id: int, raw_path: str, raw_new: str) -> Dataset:
    """Renames or moves a folder; its subfolders, its media and the batches run on it go along."""
    get_row(conn, dataset_id)
    old, new = clean_folder(raw_path), clean_folder(raw_new)
    if not old:
        raise LabError(400, "데이터셋 맨 위는 이름을 바꿀 수 없습니다")
    if not new:
        raise LabError(400, "새 폴더 이름을 입력하세요")
    with transaction(conn):
        old = _existing_folder(conn, dataset_id, old)
        paths = folder_paths(conn, dataset_id)
        parent, _, name = new.rpartition("/")
        parent = _canonical(paths, parent)  # the last name keeps its spelling: 'aug' -> 'Aug' is a rename
        new = f"{parent}/{name}" if parent else name
        if new == old:
            return get(conn, dataset_id)
        if is_within(new, old):
            raise LabError(400, "폴더를 자기 안으로 옮길 수 없습니다")
        clash = next((p for p in paths if p.casefold() == new.casefold() and p != old), None)
        if clash is not None:
            raise LabError(409, f'"{clash}" 폴더가 이미 있습니다')
        depth = max(p.count("/") for p in paths if is_within(p, old)) - old.count("/") + new.count("/") + 1
        if depth > FOLDER_DEPTH_MAX:
            raise LabError(400, f"옮기면 {depth}단계가 됩니다. 폴더는 {FOLDER_DEPTH_MAX}단계까지만 만들 수 있습니다")
        if parent:
            _ensure_folder(conn, dataset_id, parent)
        for table, column in (("dataset_folders", "path"), ("dataset_media", "folder"), ("batches", "folder")):
            cond, args = in_folder(column, old)
            conn.execute(
                f"UPDATE {table} SET {column} = ? || substr({column}, ?) WHERE dataset_id = ? AND {cond}",
                [new, len(old) + 1, dataset_id, *args],
            )
    return get(conn, dataset_id)


def delete_folder(conn: sqlite3.Connection, dataset_id: int, raw_path: str) -> Dataset:
    """Deletes a folder and its subfolders. Their media leave this dataset only: the media, their
    files, their runs and the batches stay, as with remove_media."""
    get_row(conn, dataset_id)
    path = clean_folder(raw_path)
    if not path:
        raise LabError(400, "데이터셋 맨 위는 지울 수 없습니다 (데이터셋을 지우세요)")
    with transaction(conn):
        path = _existing_folder(conn, dataset_id, path)
        cond, args = in_folder("folder", path)
        conn.execute(f"DELETE FROM dataset_media WHERE dataset_id = ? AND {cond}", [dataset_id, *args])
        cond, args = in_folder("path", path)
        conn.execute(f"DELETE FROM dataset_folders WHERE dataset_id = ? AND {cond}", [dataset_id, *args])
    return get(conn, dataset_id)


def move_media(conn: sqlite3.Connection, dataset_id: int, media_ids: list[int], raw_folder: str) -> Dataset:
    """Moves media of the dataset into ``folder`` ('' = the top level; made if missing)."""
    return move_media_groups(conn, dataset_id, {clean_folder(raw_folder): media_ids})


def move_media_groups(conn: sqlite3.Connection, dataset_id: int, groups: dict[str, list[int]]) -> Dataset:
    """Moves media of the dataset into several folders at once, all or nothing. Keys are clean
    folder paths (made if missing), values the media to put there."""
    get_row(conn, dataset_id)
    ids = list(dict.fromkeys(i for group in groups.values() for i in group))
    with transaction(conn):
        found = {r["media_id"] for r in conn.execute(
            "SELECT media_id FROM dataset_media WHERE dataset_id = ?", (dataset_id,))}
        missing = [i for i in ids if i not in found]
        if missing:
            raise LabError(404, f"이 데이터셋에 없는 미디어입니다: {', '.join(map(str, missing))}")
        for folder, group in groups.items():
            folder = _ensure_folder(conn, dataset_id, folder)
            conn.executemany("UPDATE dataset_media SET folder = ? WHERE dataset_id = ? AND media_id = ?",
                             [(folder, dataset_id, i) for i in dict.fromkeys(group)])
    return get(conn, dataset_id)


def put_uploaded(conn: sqlite3.Connection, dataset_id: int, media_id: int, folder: str) -> tuple[str, bool]:
    """Puts an uploaded media into ``folder`` (a clean path, made if missing). A media that is in
    the dataset already stays where it is. Returns (its folder, whether it was there already)."""
    get_row(conn, dataset_id)
    with transaction(conn):
        row = conn.execute(
            "SELECT folder FROM dataset_media WHERE dataset_id = ? AND media_id = ?", (dataset_id, media_id)).fetchone()
        if row is not None:
            return row["folder"], True
        folder = _ensure_folder(conn, dataset_id, folder)
        conn.execute(
            "INSERT INTO dataset_media (dataset_id, media_id, added_at, folder) VALUES (?, ?, ?, ?)",
            (dataset_id, media_id, now_iso(), folder),
        )
    return folder, False


# --- batches -----------------------------------------------------------------------------------
def batch_row(conn: sqlite3.Connection, batch_id: int) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM batches WHERE id = ?", (batch_id,)).fetchone()
    if row is None:
        raise LabError(404, f"일괄 실행 기록을 찾을 수 없습니다: #{batch_id}")
    return row


def batch_rows(conn: sqlite3.Connection, dataset_id: int) -> list[sqlite3.Row]:
    """The dataset's batches, newest first."""
    return conn.execute("SELECT * FROM batches WHERE dataset_id = ? ORDER BY id DESC", (dataset_id,)).fetchall()


def batch_fields(row: sqlite3.Row) -> dict[str, Any]:
    """What a batch row says about the batch, as API fields (shared by Batch and ScoreBatch)."""
    return {
        "dataset_id": row["dataset_id"], "dataset_name": row["dataset_name"], "folder": row["folder"],
        "model_key": row["model_key"], "model_name": row["model_name"], "params": loads(row["params"], {}),
        "stride": row["stride"], "created_at": row["created_at"],
    }


def _counts(conn: sqlite3.Connection, batch_id: int) -> dict[str, int]:
    counts = {s: 0 for s in STATUSES}
    for r in conn.execute("SELECT status, COUNT(*) AS n FROM runs WHERE batch_id = ? GROUP BY status", (batch_id,)):
        counts[r["status"]] = r["n"]
    counts["total"] = sum(counts[s] for s in STATUSES)
    return counts


def _batch_api(conn: sqlite3.Connection, row: sqlite3.Row) -> Batch:
    counts = _counts(conn, row["id"])
    return Batch(
        id=row["id"], **batch_fields(row), counts=counts, finished=counts["queued"] + counts["running"] == 0,
        zip_url=f"/api/batches/{row['id']}/results.zip",
    )


def list_batches(conn: sqlite3.Connection, dataset_id: int) -> list[Batch]:
    get_row(conn, dataset_id)
    return [_batch_api(conn, r) for r in batch_rows(conn, dataset_id)]


def get_batch(conn: sqlite3.Connection, paths: DataPaths, batch_id: int) -> BatchDetail:
    row = batch_row(conn, batch_id)
    items = []
    for r in conn.execute(
        """SELECT r.*, m.name AS media_name, m.kind AS media_kind, dm.folder AS ds_folder
           FROM runs r JOIN media m ON m.id = r.media_id
           LEFT JOIN dataset_media dm ON dm.media_id = r.media_id AND dm.dataset_id = ?
           WHERE r.batch_id = ? ORDER BY COALESCE(dm.folder, ''), m.name COLLATE NOCASE, r.id""",
        (row["dataset_id"], batch_id),
    ):
        summary = loads(r["summary"])
        thumb = paths.run_thumb(r["id"])
        items.append(BatchItem(
            run_id=r["id"], media_id=r["media_id"], media_name=r["media_name"], media_kind=r["media_kind"],
            folder=r["ds_folder"], status=r["status"], frames_done=r["frames_done"], frames_total=r["frames_total"],
            summary=RunSummary(**summary) if summary else None, error=r["error"],
            thumb_url=f"/api/runs/{r['id']}/thumb?v={thumb.stat().st_mtime_ns}" if thumb.exists() else None,
            download_url=f"/api/runs/{r['id']}/download" if r["status"] == "done" else None,
        ))
    return BatchDetail(**_batch_api(conn, row).model_dump(), items=items)


def create_batch(conn: sqlite3.Connection, paths: DataPaths, dataset_id: int, req: BatchCreate) -> BatchDetail:
    """One run per media of the dataset, or of ``req.folder`` and the folders below it."""
    ds = get_row(conn, dataset_id)
    folder = clean_folder(req.folder)
    if folder:
        folder = _existing_folder(conn, dataset_id, folder)
    media = media_rows(conn, dataset_id, folder)
    if not media:
        raise LabError(400, f'"{folder}" 폴더가 비어 있습니다. 먼저 이미지나 영상을 넣으세요' if folder
                       else "데이터셋이 비어 있습니다. 먼저 이미지나 영상을 넣으세요")
    guard_disk(paths)
    model = models.resolve(conn, req.model_key, req.params)  # 400 before anything is written
    with transaction(conn):
        cur = conn.execute(
            """INSERT INTO batches (dataset_id, dataset_name, folder, model_key, model_name, params, stride, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (dataset_id, ds["name"], folder or None, model.key, model.name, dumps(model.params), req.stride, now_iso()),
        )
    bid = int(cur.lastrowid)
    scope = f"{ds['name']}/{folder}" if folder else ds["name"]
    for m in media:
        runs_mod.create(
            conn, paths,
            RunCreate(media_id=m["id"], model_key=model.key, params=model.params, stride=req.stride,
                      note=f"데이터셋 {scope} 일괄 실행 #{bid}"),
            batch_id=bid,
        )
    return get_batch(conn, paths, bid)


def cancel_batch(conn: sqlite3.Connection, paths: DataPaths, batch_id: int) -> BatchDetail:
    batch_row(conn, batch_id)
    conn.execute(
        "UPDATE runs SET status = 'cancelled', cancel_requested = 1, finished_at = ? WHERE batch_id = ? AND status = 'queued'",
        (now_iso(), batch_id),
    )
    conn.execute("UPDATE runs SET cancel_requested = 1 WHERE batch_id = ? AND status = 'running'", (batch_id,))
    return get_batch(conn, paths, batch_id)


def delete_batch(conn: sqlite3.Connection, paths: DataPaths, batch_id: int) -> None:
    """The batch and all its runs (a running one stops at its next frame)."""
    batch_row(conn, batch_id)
    run_ids = [r["id"] for r in conn.execute("SELECT id FROM runs WHERE batch_id = ?", (batch_id,))]
    with transaction(conn):
        conn.execute("DELETE FROM runs WHERE batch_id = ?", (batch_id,))
        conn.execute("DELETE FROM batches WHERE id = ?", (batch_id,))
    for rid in run_ids:
        for p in [paths.run_thumb(rid), *paths.run_exports(rid)]:
            p.unlink(missing_ok=True)
    paths.exports.joinpath(f"batch-{batch_id}.zip").unlink(missing_ok=True)


# --- zip downloads -----------------------------------------------------------------------------
def safe_name(name: str) -> str:
    """A file name that is safe inside a zip on every OS (keeps Korean)."""
    name = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", name).strip(" .")
    return name or "file"


def _zip_path(folder: str | None, name: str, used: dict[str, set[str]]) -> str:
    """``name`` inside the zip directory of a dataset folder ('' or None = the zip's top level),
    each folder name made safe; names are made unique per directory."""
    folder_arc = "/".join(safe_name(n) for n in folder.split("/")) if folder else ""
    name = _unique(name, used.setdefault(folder_arc.lower(), set()))
    return f"{folder_arc}/{name}" if folder_arc else name


def _unique(name: str, used: set[str]) -> str:
    stem, dot, ext = name.rpartition(".")
    if not dot:
        stem, ext = name, ""
    out, n = name, 2
    while out.lower() in used:
        out = f"{stem}-{n}{dot}{ext}"
        n += 1
    used.add(out.lower())
    return out


def _write_zip(paths: DataPaths, dst: Path, entries: list[tuple[str, Path | bytes]]) -> Path:
    """Stored (not deflated: images and videos are compressed already), written via a temp file."""
    tmp = paths.tmp / f"zip-{uuid.uuid4().hex}.zip"
    try:
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
            for arc, src in entries:
                if isinstance(src, bytes):
                    z.writestr(zipfile.ZipInfo(arc, date_time=_zip_time()), src, compress_type=zipfile.ZIP_DEFLATED)
                else:
                    z.write(src, arc)
        tmp.replace(dst)
    finally:
        tmp.unlink(missing_ok=True)
    return dst


def _zip_time() -> tuple[int, int, int, int, int, int]:
    from datetime import datetime

    t = datetime.now()
    return (t.year, t.month, t.day, t.hour, t.minute, t.second)


def originals_zip(conn: sqlite3.Connection, paths: DataPaths, dataset_id: int) -> tuple[Path, str]:
    ds = get_row(conn, dataset_id)
    media = media_rows(conn, dataset_id)
    if not media:
        raise LabError(400, "데이터셋이 비어 있습니다")
    guard_disk(paths)
    used: dict[str, set[str]] = {}
    entries: list[tuple[str, Path | bytes]] = []
    for m in media:
        src = media_mod.original_path(paths, m)
        name = safe_name(m["name"])
        if not Path(name).suffix and m["ext"]:
            name += m["ext"]
        entries.append((_zip_path(m["ds_folder"], name, used), src))
    dst = _write_zip(paths, paths.exports / f"dataset-{dataset_id}-originals.zip", entries)
    return dst, f"{safe_name(ds['name'])}-원본.zip"


CSV_FIELDS = ["file", "folder", "media", "frame", "cube", "color", "center_x", "center_y", "angle", "score", "color_score",
              "area", "flags"]


def results_zip(conn: sqlite3.Connection, paths: DataPaths, batch_id: int) -> tuple[Path, str]:
    """Every finished run of the batch drawn (PNG for an image, MP4 for a video), plus
    results.json (batch, params, every frame's cubes) and results.csv (one row per cube).
    Files sit in their folder's directory, relative to the dataset (also for a folder batch),
    so the zips of different folders merge."""
    detail = get_batch(conn, paths, batch_id)
    done = [it for it in detail.items if it.status == "done"]
    if not done:
        raise LabError(409, "끝난 실행이 아직 없습니다")
    guard_disk(paths)
    used: dict[str, set[str]] = {}
    entries: list[tuple[str, Path | bytes]] = []
    out_items: list[dict[str, Any]] = []
    rows = io.StringIO()
    writer = csv.DictWriter(rows, fieldnames=CSV_FIELDS)
    writer.writeheader()
    for it in detail.items:
        item: dict[str, Any] = {"media_id": it.media_id, "media": it.media_name, "kind": it.media_kind,
                                "folder": it.folder, "run_id": it.run_id, "status": it.status, "file": None}
        if it.status == "done":
            path = runs_mod.export_file(paths, it.run_id)
            if path is None:
                export_mod.export_run(conn, paths, it.run_id)
                path = runs_mod.export_file(paths, it.run_id)
            assert path is not None
            stem = safe_name(it.media_name.rsplit(".", 1)[0]) if "." in it.media_name else safe_name(it.media_name)
            arc = _zip_path(it.folder, f"{stem}_result{path.suffix}", used)
            entries.append((arc, path))
            item["file"] = arc
            frames = runs_mod.frames_of(conn, it.run_id).frames
            item["frames"] = [{"frame": f.frame, "ms": f.ms, "cubes": f.cubes} for f in frames]
            for f in frames:
                for i, c in enumerate(f.cubes):
                    writer.writerow({
                        "file": arc, "folder": it.folder or "", "media": it.media_name, "frame": f.frame, "cube": i, "color": c.get("color"),
                        "center_x": c["center"][0], "center_y": c["center"][1], "angle": c.get("angle"),
                        "score": c.get("score"), "color_score": c.get("color_score"), "area": c.get("area"),
                        "flags": " ".join(c.get("flags") or []),
                    })
        elif it.error:
            item["error"] = it.error
        out_items.append(item)
    meta = {
        "batch": {"id": detail.id, "dataset": detail.dataset_name, "folder": detail.folder, "model_key": detail.model_key,
                  "model_name": detail.model_name, "params": detail.params, "stride": detail.stride,
                  "created_at": detail.created_at, "counts": detail.counts},
        "coordinates": "image pixels; center = top-face centre; angle in degrees [0, 90)",
        "items": out_items,
    }
    entries.append(("results.json", json.dumps(meta, ensure_ascii=False, indent=1).encode("utf-8")))
    entries.append(("results.csv", ("﻿" + rows.getvalue()).encode("utf-8")))  # BOM: Excel reads Korean
    dst = _write_zip(paths, paths.exports / f"batch-{batch_id}.zip", entries)
    scope = f"-{safe_name(detail.folder)}" if detail.folder else ""
    return dst, f"{safe_name(detail.dataset_name)}{scope}-{safe_name(detail.model_name)}-결과-{batch_id}.zip"

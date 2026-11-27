"""Ground-truth sets of a dataset: import, the links between GT images and media, and the
in-memory cache of parsed sets (docs/SCORING.md §1 is the spec).

A set is one uploaded JSON file, kept byte for byte at data/gt/gt-<id>.json and parsed once per
process. gt_images indexes its images (sha256, file name, variant) so links are plain lookups.

Links between a GT image and a dataset image:
- by sha256: the media is exactly the labelled file. Derived from the dataset's current media
  every time and never stored, so media added later link on their own.
- confirmed: the user linked them (normally a media with the GT file name but another sha256);
  stored in gt_links. It counts only while its media is an image of the dataset and no sha256 link
  takes its GT image or its media; otherwise it is ignored and reported (sha256 always wins).
A media is linked to at most one GT image of a set, and a GT image to at most one media.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from dataclasses import dataclass, field

from cubevision.gt import GtSet, parse_gt

from lab import datasets as datasets_mod
from lab import media as media_mod
from lab.cache import Lru
from lab.config import DataPaths
from lab.db import now_iso, transaction
from lab.errors import LabError
from lab.schemas import (
    Dataset, GtIgnoredLink, GtLinkIn, GtLinkedImage, GtLinkReport, GtMediaRef, GtSetDetail, GtSetInfo,
    GtUnlinkedImage, GtVariant, LinkMethod, MediaGtLink,
)
from lab.text import nfc
from lab.upload import Upload

NAME_MAX = 200


# --- parsed sets, cached -------------------------------------------------------------------------
# Parsed sets by (data dir, set id); the 410-image set is a few MB of objects.
_sets: Lru[tuple[str, int], GtSet] = Lru(8)


def load(paths: DataPaths, gt_set_id: int) -> GtSet:
    """The parsed set; read from its file only the first time (per process)."""
    def parse() -> GtSet:
        path = paths.gt_file(gt_set_id)
        try:
            return parse_gt(json.loads(path.read_bytes()))
        except FileNotFoundError:
            raise LabError(500, f"정답셋 #{gt_set_id}의 파일이 없습니다 ({path.name})") from None

    return _sets.get_or_make((str(paths.root), gt_set_id), parse)


def forget(paths: DataPaths, gt_set_ids: list[int]) -> None:
    """Removes the files and cache entries of deleted sets (their rows are gone already)."""
    for gid in gt_set_ids:
        paths.gt_file(gid).unlink(missing_ok=True)
        _sets.pop((str(paths.root), gid))


# --- rows and links ------------------------------------------------------------------------------
_SELECT = "SELECT gs.*, d.name AS dataset_name FROM gt_sets gs JOIN datasets d ON d.id = gs.dataset_id"


def set_row(conn: sqlite3.Connection, gt_set_id: int) -> sqlite3.Row:
    row = conn.execute(_SELECT + " WHERE gs.id = ?", (gt_set_id,)).fetchone()
    if row is None:
        raise LabError(404, "정답셋을 찾을 수 없습니다")
    return row


def set_ids_of_dataset(conn: sqlite3.Connection, dataset_id: int) -> list[int]:
    return [r["id"] for r in conn.execute("SELECT id FROM gt_sets WHERE dataset_id = ?", (dataset_id,))]


@dataclass(frozen=True)
class GtImageRow:
    index: int
    sha256: str
    filename: str
    variant: str
    layout_group: str


@dataclass(frozen=True)
class DatasetMedia:
    id: int
    kind: str
    sha256: str
    name: str
    folder: str  # '' = the dataset's top level


@dataclass
class Links:
    """The links of one set right now, against its dataset's current media."""

    gt: list[GtImageRow]  # by GT index
    images: dict[int, DatasetMedia]  # the dataset's images by media id
    videos: list[DatasetMedia]  # the dataset's videos (never scored)
    by_index: dict[int, tuple[int, LinkMethod]] = field(default_factory=dict)  # GT index -> (media id, method)
    by_media: dict[int, int] = field(default_factory=dict)  # media id -> GT index
    ignored: list[GtIgnoredLink] = field(default_factory=list)


def links_of(conn: sqlite3.Connection, row: sqlite3.Row) -> Links:
    gt = [GtImageRow(r["gt_index"], r["sha256"], r["filename"], r["variant"], r["layout_group"]) for r in conn.execute(
        "SELECT gt_index, sha256, filename, variant, layout_group FROM gt_images WHERE gt_set_id = ? ORDER BY gt_index",
        (row["id"],))]
    images: dict[int, DatasetMedia] = {}
    videos: list[DatasetMedia] = []
    for r in datasets_mod.media_rows(conn, row["dataset_id"]):
        m = DatasetMedia(r["id"], r["kind"], r["sha256"].lower(), r["name"], r["ds_folder"])
        if m.kind == "image":
            images[m.id] = m
        else:
            videos.append(m)
    links = Links(gt, images, videos)
    by_sha = {m.sha256: m.id for m in images.values()}
    for g in gt:
        mid = by_sha.get(g.sha256)
        if mid is not None:
            links.by_index[g.index] = (mid, "sha256")
            links.by_media[mid] = g.index
    for r in conn.execute("SELECT gt_index, media_id FROM gt_links WHERE gt_set_id = ? ORDER BY gt_index", (row["id"],)):
        i, mid = r["gt_index"], r["media_id"]
        if mid not in images:
            reason = "영상은 채점하지 않습니다" if any(v.id == mid for v in videos) else "미디어가 이 데이터셋에 없습니다"
        elif i in links.by_index:
            reason = "같은 파일(sha256)인 미디어가 이 정답 이미지에 연결되어 우선합니다"
        elif mid in links.by_media:
            reason = f"이 미디어는 sha256으로 정답 이미지 {links.by_media[mid]}번에 연결되어 있습니다"
        else:
            links.by_index[i] = (mid, "confirmed")
            links.by_media[mid] = i
            continue
        links.ignored.append(GtIgnoredLink(gt_index=i, media_id=mid, reason=reason))
    return links


def _report(links: Links) -> GtLinkReport:
    methods = [m for _, m in links.by_index.values()]
    return GtLinkReport(
        gt_images=len(links.gt), linked_sha256=methods.count("sha256"), linked_confirmed=methods.count("confirmed"),
        unlinked=len(links.gt) - len(links.by_index), dataset_images_without_gt=len(links.images) - len(links.by_media),
        dataset_videos=len(links.videos), ignored=links.ignored,
    )


def info_of(row: sqlite3.Row, links: Links) -> GtSetInfo:
    counts: dict[str, int] = {}
    for g in links.gt:
        counts[g.variant] = counts.get(g.variant, 0) + 1
    return GtSetInfo(
        id=row["id"], dataset_id=row["dataset_id"], name=row["name"], schema_version=row["schema_version"],
        label_kind=row["label_kind"], file_sha256=row["file_sha256"], file_bytes=row["file_bytes"],
        n_images=row["n_images"], n_cubes=row["n_cubes"], n_valid=row["n_valid"],
        n_layout_groups=len({g.layout_group for g in links.gt}),
        variants=[GtVariant(name=k, count=v) for k, v in counts.items()], created_at=row["created_at"],
        links=_report(links),
    )


def info(conn: sqlite3.Connection, gt_set_id: int) -> GtSetInfo:
    row = set_row(conn, gt_set_id)
    return info_of(row, links_of(conn, row))


def list_sets(conn: sqlite3.Connection, dataset_id: int) -> list[GtSetInfo]:
    datasets_mod.get_row(conn, dataset_id)
    rows = conn.execute(_SELECT + " WHERE gs.dataset_id = ? ORDER BY gs.id DESC", (dataset_id,)).fetchall()
    return [info_of(r, links_of(conn, r)) for r in rows]


def _ref(m: DatasetMedia) -> GtMediaRef:
    return GtMediaRef(media_id=m.id, name=m.name, folder=m.folder)


def _by_place(m: DatasetMedia) -> tuple[str, str, int]:
    return (m.folder, m.name.casefold(), m.id)


def detail(conn: sqlite3.Connection, gt_set_id: int) -> GtSetDetail:
    row = set_row(conn, gt_set_id)
    links = links_of(conn, row)
    free = sorted((m for m in links.images.values() if m.id not in links.by_media), key=_by_place)
    by_name: dict[str, list[DatasetMedia]] = {}
    for m in free:
        by_name.setdefault(nfc(m.name), []).append(m)
    linked, unlinked = [], []
    for g in links.gt:
        if g.index in links.by_index:
            mid, method = links.by_index[g.index]
            linked.append(GtLinkedImage(gt_index=g.index, filename=g.filename, variant=g.variant, method=method,
                                        media=_ref(links.images[mid])))
        else:
            unlinked.append(GtUnlinkedImage(gt_index=g.index, filename=g.filename, variant=g.variant,
                                            candidates=[_ref(m) for m in by_name.get(g.filename, [])]))
    return GtSetDetail(**info_of(row, links).model_dump(), linked=linked, unlinked=unlinked,
                       images_without_gt=[_ref(m) for m in free])


# --- import, edit, delete ------------------------------------------------------------------------
def _json_error(exc: Exception) -> str:
    if isinstance(exc, json.JSONDecodeError):
        return f"{exc.lineno}행 {exc.colno}열: {exc.msg}"
    return "UTF-8 텍스트가 아닙니다"


def import_set(conn: sqlite3.Connection, paths: DataPaths, dataset_id: int, up: Upload) -> GtSetInfo:
    """Parse the uploaded JSON (a file in data/tmp, consumed), store it as a new set of the
    dataset and keep the file byte for byte. 409 when this dataset has the same file already
    (known from the upload's sha256, before parsing), 400 for a file parse_gt refuses (its message)."""
    try:
        datasets_mod.get_row(conn, dataset_id)
        same = conn.execute("SELECT id, name FROM gt_sets WHERE dataset_id = ? AND file_sha256 = ?",
                            (dataset_id, up.sha256)).fetchone()
        if same is not None:
            raise LabError(409, f'같은 정답셋 파일이 이 데이터셋에 이미 있습니다: "{same["name"]}" (#{same["id"]})')
        name = nfc((up.fields.get("name") or "").strip() or up.filename)[:NAME_MAX]
        try:
            doc = json.loads(up.path.read_bytes())
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LabError(400, f"정답셋 파일을 JSON으로 읽지 못했습니다 ({_json_error(exc)})") from None
        try:
            gt = parse_gt(doc)
        except ValueError as exc:
            raise LabError(400, str(exc)) from None
        gid = None
        try:
            with transaction(conn):
                cur = conn.execute(
                    """INSERT INTO gt_sets (dataset_id, name, schema_version, label_kind, file_sha256, file_bytes,
                                            n_images, n_cubes, n_valid, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (dataset_id, name, gt.schema_version, gt.label_kind, up.sha256, up.bytes,
                     gt.n_images, gt.n_cubes, gt.n_valid_geometry, now_iso()),
                )
                gid = int(cur.lastrowid)
                conn.executemany(
                    """INSERT INTO gt_images (gt_set_id, gt_index, sha256, filename, variant, layout_group)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    [(gid, im.index, im.sha256.lower(), nfc(im.filename), im.variant, im.layout_group) for im in gt.images],
                )
                shutil.move(up.path, paths.gt_file(gid))  # inside: a failed move rolls the rows back
        except sqlite3.IntegrityError:
            # The same file finished uploading in a parallel request a moment earlier.
            raise LabError(409, "같은 정답셋 파일이 이 데이터셋에 이미 있습니다") from None
        except BaseException:
            if gid is not None:
                paths.gt_file(gid).unlink(missing_ok=True)
            raise
        _sets.put((str(paths.root), gid), gt)
        return info(conn, gid)
    finally:
        up.discard()


def rename(conn: sqlite3.Connection, gt_set_id: int, name: str) -> GtSetInfo:
    set_row(conn, gt_set_id)
    name = nfc(name.strip())
    if not name:
        raise LabError(400, "정답셋 이름을 입력하세요")
    conn.execute("UPDATE gt_sets SET name = ? WHERE id = ?", (name[:NAME_MAX], gt_set_id))
    return info(conn, gt_set_id)


def delete(conn: sqlite3.Connection, paths: DataPaths, gt_set_id: int) -> None:
    """The set, its confirmed links and its file. Media, runs and batches stay."""
    set_row(conn, gt_set_id)
    with transaction(conn):
        conn.execute("DELETE FROM gt_sets WHERE id = ?", (gt_set_id,))
    forget(paths, [gt_set_id])


# --- confirmed links -----------------------------------------------------------------------------
def confirm_links(conn: sqlite3.Connection, gt_set_id: int, wanted: list[GtLinkIn]) -> GtSetDetail:
    """Links GT images to dataset images, all or nothing. Each GT image must be unlinked and each
    media an unlinked image of the set's dataset; a confirmed link that is being ignored (see the
    module docstring) is replaced."""
    row = set_row(conn, gt_set_id)
    with transaction(conn):
        links = links_of(conn, row)
        bad: list[str] = []  # the request itself is wrong (400)
        taken: list[str] = []  # something is linked already (409)
        indices = [w.gt_index for w in wanted]
        media_ids = [w.media_id for w in wanted]
        if len(set(indices)) != len(indices):
            bad.append("같은 정답 이미지가 요청에 두 번 있습니다")
        if len(set(media_ids)) != len(media_ids):
            bad.append("같은 미디어가 요청에 두 번 있습니다")
        for w in wanted:
            if not 0 <= w.gt_index < len(links.gt):
                bad.append(f"정답 이미지 {w.gt_index}번은 이 정답셋에 없습니다")
            elif w.gt_index in links.by_index:
                taken.append(f"정답 이미지 {w.gt_index}번({links.gt[w.gt_index].filename})은 이미 연결되어 있습니다")
            if w.media_id not in links.images:
                kind = "영상이라 연결할 수 없습니다" if any(v.id == w.media_id for v in links.videos) else "이 데이터셋에 없습니다"
                bad.append(f"미디어 #{w.media_id}은(는) {kind}")
            elif w.media_id in links.by_media:
                taken.append(f"미디어 #{w.media_id}({links.images[w.media_id].name})은(는) 이미 정답 이미지 "
                             f"{links.by_media[w.media_id]}번에 연결되어 있습니다")
        problems = bad + taken
        if problems:
            more = f" 외 {len(problems) - 3}건" if len(problems) > 3 else ""
            raise LabError(400 if bad else 409, "연결하지 않았습니다: " + "; ".join(problems[:3]) + more)
        now = now_iso()
        conn.executemany(
            "INSERT OR REPLACE INTO gt_links (gt_set_id, gt_index, media_id, created_at) VALUES (?, ?, ?, ?)",
            [(gt_set_id, w.gt_index, w.media_id, now) for w in wanted],
        )
    return detail(conn, gt_set_id)


def remove_link(conn: sqlite3.Connection, gt_set_id: int, gt_index: int) -> GtSetDetail:
    """Removes a confirmed link. A sha256 link cannot be removed: the media is the labelled file."""
    row = set_row(conn, gt_set_id)
    if conn.execute("DELETE FROM gt_links WHERE gt_set_id = ? AND gt_index = ?", (gt_set_id, gt_index)).rowcount == 0:
        link = links_of(conn, row).by_index.get(gt_index)
        if link is not None and link[1] == "sha256":
            raise LabError(400, f"정답 이미지 {gt_index}번은 같은 파일(sha256)로 연결되어 있어 끊을 수 없습니다")
        raise LabError(404, f"정답 이미지 {gt_index}번에는 확인된 연결이 없습니다")
    return detail(conn, gt_set_id)


# --- organize ------------------------------------------------------------------------------------
def organize(conn: sqlite3.Connection, gt_set_id: int, raw_root: str) -> Dataset:
    """Moves every linked media of the set into the folder "<root>/<variant>" of its dataset."""
    row = set_row(conn, gt_set_id)
    root = datasets_mod.clean_folder(raw_root)
    links = links_of(conn, row)
    if not links.by_index:
        raise LabError(400, "연결된 이미지가 없어 옮길 것이 없습니다")
    groups: dict[str, list[int]] = {}
    for g in links.gt:
        if g.index in links.by_index:
            folder = datasets_mod.clean_folder(f"{root}/{g.variant}" if root else g.variant)
            groups.setdefault(folder, []).append(links.by_index[g.index][0])
    return datasets_mod.move_media_groups(conn, row["dataset_id"], groups)


# --- one media -----------------------------------------------------------------------------------
def media_links(conn: sqlite3.Connection, media_id: int) -> list[MediaGtLink]:
    """The sets (of any dataset) that link this media, newest first."""
    media_mod.get_row(conn, media_id)
    rows = conn.execute(
        _SELECT + """ JOIN dataset_media dm ON dm.dataset_id = gs.dataset_id AND dm.media_id = ?
                      ORDER BY gs.id DESC""", (media_id,)).fetchall()
    out = []
    for r in rows:
        links = links_of(conn, r)
        index = links.by_media.get(media_id)
        if index is not None:
            g = links.gt[index]
            out.append(MediaGtLink(gt_set_id=r["id"], gt_set_name=r["name"], dataset_id=r["dataset_id"],
                                   dataset_name=r["dataset_name"], gt_index=index, filename=g.filename,
                                   variant=g.variant, method=links.by_index[index][1]))
    return out

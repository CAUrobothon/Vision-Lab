"""Scoring batches (and single images) against a ground-truth set. docs/SCORING.md is the spec;
the maths is cubevision.score. This module picks each batch's evaluation images, reads their
runs, caches the work and turns results into API objects.

Evaluation images of a batch = the set's linked GT images whose media lies in the batch's scope
(the whole dataset, or its folder with every folder below it). Per image the batch's run decides:
done -> scored on frame 0; failed -> scored with zero detections (every GT cube missed, no time
sample); queued / running / cancelled, or no run in the batch -> pending. A batch with a pending
image is interim (§4): scored on its finished images only, never ranked or compared.

Caches (per process, bounded, keyed by the data dir so another data dir never hits them; all
lab.cache.Lru, so two identical requests at once compute once):
- the parsed predictions of each finished run (frame 0's cubes and ms) per (run, finished_at):
  new thresholds only redo the matching;
- per GT set, the last 3 threshold tuples (τc τa τv g) each keep: one ImageScore per (GT image,
  run), stamped with the run's status / frames_done / finished_at, so a page polling a running
  batch only scores what finished since the last poll; and per batch its scored images with
  their metrics, and its slices, stamped with a fingerprint of those images and their runs, so
  each poll replaces the batch's previous entry. Bootstrap settings never recompute these;
- per (criteria, batch) and fingerprint: the augmentation effect; per (thresholds, reps, seed,
  reference) the reference's bootstrap side, shared by every batch compared with it; per
  (criteria, reference, batch) and both fingerprints: the paired bootstrap.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import asdict, dataclass, replace
from typing import Any

from cubevision.gt import GtImage, GtSet
from cubevision.score import (
    CUBE_CSV_FIELDS, SLICE_DIMENSIONS, SUMMARY_CSV_FIELDS, BootstrapSide, Criteria, ImageScore, Metrics, Pred,
    aggregate, as_pred, augmentation_effect, bootstrap_side, csv_bytes, cube_csv_rows, outcome_counts,
    paired_bootstrap, score_image, slices, summary_csv_rows,
)

from lab import datasets as datasets_mod
from lab import gtsets
from lab import media as media_mod
from lab import runs as runs_mod
from lab.cache import Lru
from lab.config import DataPaths
from lab.db import loads, now_iso
from lab.errors import LabError
from lab.schemas import (
    CubeDict, MediaGtLink, ScoreAugmentationEffect, ScoreBatch, ScoreBatchComparison, ScoreComparison,
    ScoreCriteria, ScoreCriteriaIn, ScoreCriteriaInfo, ScoreCriterionLimit, ScoreImageDetail, ScoreImageRow,
    ScoreMetrics, ScorePending, ScoreRequest, ScoreResult, ScoreSlice, ScoreSliceDimension,
)

PENDING = ("queued", "running", "cancelled")  # run states of an image that is not finished


def criteria_of(raw: ScoreCriteriaIn | None) -> Criteria:
    """The request's criteria (ScoreCriteriaIn holds the defaults and bounds); 400 with
    Criteria's own message for a value the schema lets through but Criteria refuses."""
    values = raw.model_dump(include=set(ScoreCriteriaIn.model_fields)) if raw is not None else {}
    try:
        return Criteria(**values)
    except ValueError as exc:
        raise LabError(400, str(exc)) from None


def criteria_info() -> ScoreCriteriaInfo:
    """GET /api/score/criteria: defaults and limits exactly as ScoreCriteriaIn's schema states them."""
    limits = {}
    for name, p in ScoreCriteriaIn.model_json_schema()["properties"].items():
        limits[name] = ScoreCriterionLimit(
            min=p.get("minimum", p.get("exclusiveMinimum")),
            max=p.get("maximum", p.get("exclusiveMaximum")), min_exclusive="exclusiveMinimum" in p,
            max_exclusive="exclusiveMaximum" in p, step=p["step"], integer=p["type"] == "integer",
        )
    return ScoreCriteriaInfo(defaults=ScoreCriteria.model_validate(Criteria()), limits=limits)


# --- caches --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class _Scored:
    scores: tuple[ImageScore, ...]  # the batch's finished evaluation images, in GT order
    metrics: Metrics
    api: ScoreMetrics


class _Generation:
    """What one GT set scored with one threshold tuple keeps."""

    def __init__(self) -> None:
        self.images: Lru[tuple[int, int], ImageScore] = Lru(60_000)  # (GT index, run id); stamp: the run's state
        self.scored: Lru[int, _Scored] = Lru(256)  # batch id; stamp: the batch's fingerprint
        self.sliced: Lru[int, list[ScoreSlice]] = Lru(256)  # the same


GENERATIONS = 3  # threshold tuples kept per GT set
_preds: Lru[tuple[str, int, str | None], tuple[tuple[Pred, ...] | None, float | None]] = Lru(20_000)
_generations: Lru[tuple[str, int], Lru[tuple[float, ...], _Generation]] = Lru(8)  # (data dir, set id)
_augmented: Lru[tuple[Any, ...], list[ScoreAugmentationEffect]] = Lru(64)
_sides: Lru[tuple[Any, ...], BootstrapSide] = Lru(8)
_compared: Lru[tuple[Any, ...], ScoreComparison] = Lru(256)


def clear_caches() -> None:
    for c in (_preds, _generations, _augmented, _sides, _compared):
        c.clear()


def _generation(root: str, gid: int, thresholds: tuple[float, ...]) -> _Generation:
    return _generations.get_or_make((root, gid), lambda: Lru(GENERATIONS)).get_or_make(thresholds, _Generation)


SLICE_DEFS = [ScoreSliceDimension(key=k, label=label, unit=unit) for k, label, unit, _ in SLICE_DIMENSIONS]


# --- one batch -----------------------------------------------------------------------------------
@dataclass
class _Batch:
    row: sqlite3.Row
    entries: list[tuple[int, int, sqlite3.Row | None]]  # (GT index, media id, its run or None) in GT order
    finished: list[tuple[int, sqlite3.Row]]  # (GT index, run) of the entries with a done or failed run
    pending: ScorePending
    fingerprint: str  # of the finished entries and their runs' states
    videos: int

    @property
    def id(self) -> int:
        return self.row["id"]

    @property
    def eval_indices(self) -> frozenset[int]:
        return frozenset(index for index, _, _ in self.entries)

    @property
    def complete(self) -> bool:
        return self.pending.total == 0


def _state(run: sqlite3.Row) -> tuple[str, int, str | None]:
    return (run["status"], run["frames_done"], run["finished_at"])


def _frames0(conn: sqlite3.Connection, run_ids: list[int]) -> dict[int, tuple[float, list[CubeDict]]]:
    out = {}
    for start in range(0, len(run_ids), 500):
        part = run_ids[start:start + 500]
        for r in conn.execute(
                f"SELECT run_id, ms, cubes FROM run_frames WHERE frame = 0 AND run_id IN ({','.join('?' * len(part))})",
                part):
            out[r["run_id"]] = (r["ms"], loads(r["cubes"], []))
    return out


def _preds_of(conn: sqlite3.Connection, root: str,
              runs: list[sqlite3.Row]) -> dict[int, tuple[tuple[Pred, ...] | None, float | None]]:
    """Frame 0 of done runs as (predictions, ms); (None, None) for a done run without frame 0."""
    out, need = {}, []
    for run in runs:
        hit = _preds.get((root, run["id"], run["finished_at"]))
        if hit is None:
            need.append(run)
        else:
            out[run["id"]] = hit
    frames = _frames0(conn, [run["id"] for run in need])
    for run in need:
        frame = frames.get(run["id"])
        value = (tuple(as_pred(c) for c in frame[1]), frame[0]) if frame else (None, None)
        _preds.put((root, run["id"], run["finished_at"]), value)
        out[run["id"]] = value
    return out


def _evaluate(conn: sqlite3.Connection, links: gtsets.Links, row: sqlite3.Row) -> _Batch:
    """The batch's evaluation images and the state of their runs (nothing is scored here)."""
    scope = row["folder"] or ""
    in_scope = sorted((links.by_media[m.id], m.id) for m in links.images.values()
                      if m.id in links.by_media and datasets_mod.is_within(m.folder, scope))
    videos = sum(datasets_mod.is_within(v.folder, scope) for v in links.videos)
    runs: dict[int, sqlite3.Row] = {}
    for r in conn.execute("SELECT id, media_id, status, frames_done, finished_at FROM runs WHERE batch_id = ? ORDER BY id",
                          (row["id"],)):
        runs[r["media_id"]] = r  # the newest run of a media (a batch makes one per media)
    pending = dict.fromkeys(("queued", "running", "cancelled", "missing"), 0)
    entries, finished = [], []
    for index, mid in in_scope:
        run = runs.get(mid)
        entries.append((index, mid, run))
        if run is None:
            pending["missing"] += 1
        elif run["status"] in PENDING:
            pending[run["status"]] += 1
        else:
            finished.append((index, run))
    fp = hashlib.blake2b(repr([(index, run["id"], *_state(run)) for index, run in finished]).encode(),
                         digest_size=16).hexdigest()
    return _Batch(row, entries, finished, ScorePending(**pending), fp, videos)


def _batch_rows(conn: sqlite3.Connection, set_row: sqlite3.Row, batch_ids: list[int]) -> list[sqlite3.Row]:
    """The requested batches in the given order; none = every batch of the set's dataset, newest first."""
    if not batch_ids:
        return datasets_mod.batch_rows(conn, set_row["dataset_id"])
    rows = []
    for bid in dict.fromkeys(batch_ids):
        r = datasets_mod.batch_row(conn, bid)
        if r["dataset_id"] != set_row["dataset_id"]:
            raise LabError(400, f"일괄 실행 #{bid}은(는) 정답셋 \"{set_row['name']}\"의 데이터셋"
                                f"(\"{set_row['dataset_name']}\")에서 실행한 것이 아닙니다")
        rows.append(r)
    return rows


@dataclass
class _Selection:
    conn: sqlite3.Connection
    root: str
    gid: int
    set_row: sqlite3.Row
    gt: GtSet
    links: gtsets.Links
    crit: Criteria
    gen: _Generation
    batches: list[_Batch]


def _select(conn: sqlite3.Connection, paths: DataPaths, gt_set_id: int, batch_ids: list[int],
            raw: ScoreCriteriaIn | None) -> _Selection:
    crit = criteria_of(raw)
    set_row = gtsets.set_row(conn, gt_set_id)
    gt = gtsets.load(paths, gt_set_id)
    links = gtsets.links_of(conn, set_row)
    root = str(paths.root)
    batches = [_evaluate(conn, links, r) for r in _batch_rows(conn, set_row, batch_ids)]
    return _Selection(conn, root, gt_set_id, set_row, gt, links, crit, _generation(root, gt_set_id, crit.thresholds),
                      batches)


def _scored(sel: _Selection, b: _Batch) -> _Scored:
    """The batch's finished evaluation images scored, and their metrics."""
    def make() -> _Scored:
        found: dict[int, ImageScore] = {}
        todo = []
        for index, run in b.finished:
            s = sel.gen.images.get((index, run["id"]), _state(run))
            if s is None:
                todo.append((index, run))
            else:
                found[index] = s
        preds = _preds_of(sel.conn, sel.root, [run for _, run in todo if run["status"] == "done"])
        for index, run in todo:
            # failed -> None (zero detections, no time); a done run without frame 0 reads the same.
            p, ms = preds.get(run["id"], (None, None))
            s = score_image(sel.gt.images[index], p, ms, sel.crit)
            sel.gen.images.put((index, run["id"]), s, _state(run))
            found[index] = s
        scores = tuple(found[index] for index, _ in b.finished)
        metrics = aggregate(scores)
        return _Scored(scores, metrics, ScoreMetrics.model_validate(metrics))

    return sel.gen.scored.get_or_make(b.id, make, stamp=b.fingerprint)


def _slices(sel: _Selection, b: _Batch) -> list[ScoreSlice]:
    return sel.gen.sliced.get_or_make(
        b.id, lambda: [ScoreSlice.model_validate(r) for r in slices(_scored(sel, b).scores)], stamp=b.fingerprint)


def _augmentation(sel: _Selection, b: _Batch) -> list[ScoreAugmentationEffect]:
    return _augmented.get_or_make(
        (sel.root, sel.gid, sel.crit, b.id),
        lambda: [ScoreAugmentationEffect.model_validate(e) for e in augmentation_effect(_scored(sel, b).scores, sel.crit)],
        stamp=b.fingerprint)


def _side(sel: _Selection, ref: _Batch) -> BootstrapSide:
    c = sel.crit

    def make() -> BootstrapSide:
        s = _scored(sel, ref)
        return bootstrap_side(s.scores, c, s.metrics)

    return _sides.get_or_make((sel.root, sel.gid, c.thresholds, c.bootstrap_reps, c.bootstrap_seed, ref.id), make,
                              stamp=ref.fingerprint)


def _comparison(sel: _Selection, ref: _Batch, b: _Batch) -> ScoreComparison:
    def make() -> ScoreComparison:
        s = _scored(sel, b)
        return ScoreComparison.model_validate(
            paired_bootstrap(_scored(sel, ref).scores, s.scores, sel.crit, side_a=_side(sel, ref), metrics_b=s.metrics))

    return _compared.get_or_make((sel.root, sel.gid, sel.crit, ref.id, b.id), make,
                                 stamp=(ref.fingerprint, b.fingerprint))


# --- POST /api/score -----------------------------------------------------------------------------
def _not_comparable(code: str, reason: str) -> ScoreBatchComparison:
    return ScoreBatchComparison(role="not_comparable", reason_code=code, reason=reason)  # type: ignore[arg-type]


def _reference(batches: list[_Batch], asked_id: int | None, n_gt: int) -> tuple[_Batch | None, str | None]:
    """The reference (§5): the requested batch if it can be one, else the first complete batch
    that covers the whole set; and why the requested one was not used."""
    ref: _Batch | None = None
    note = None
    if asked_id is not None:
        asked = next((b for b in batches if b.id == asked_id), None)
        if asked is None:
            note = f"기준으로 고른 일괄 실행 #{asked_id}이(가) 채점 대상에 없습니다"
        elif not asked.complete:
            note = f"기준으로 고른 일괄 실행 #{asked_id}은(는) 중간 결과라 기준이 될 수 없습니다"
        elif not asked.finished:
            note = f"기준으로 고른 일괄 실행 #{asked_id}에는 평가할 이미지가 없습니다"
        else:
            ref = asked
    if ref is None:
        ref = next((b for b in batches if b.complete and b.finished and len(b.entries) == n_gt), None)
        if note is not None:
            note += f". 대신 #{ref.id}을(를) 기준으로 했습니다" if ref else ". 기준으로 쓸 완료 배치가 없습니다"
    return ref, note


def _batch_comparison(sel: _Selection, ref: _Batch | None, b: _Batch) -> ScoreBatchComparison:
    if b is ref:
        return ScoreBatchComparison(role="reference")
    if not b.complete:
        return _not_comparable("interim", f"중간 결과라 비교하지 않습니다 (끝나지 않은 이미지 {b.pending.total}장)")
    if not b.finished:
        return _not_comparable("no_images", "평가할 이미지가 없습니다 (정답이 연결된 이미지가 범위에 없음)")
    if ref is None:
        return _not_comparable("no_reference", "비교 기준이 될 완료 배치가 없습니다 (정답셋 전체를 채점한 완료 배치가 필요합니다)")
    if b.eval_indices != ref.eval_indices:
        only_b, only_a = len(b.eval_indices - ref.eval_indices), len(ref.eval_indices - b.eval_indices)
        return _not_comparable("different_images", f"평가 이미지 집합이 기준 #{ref.id}과(와) 다릅니다 "
                                                   f"(이 배치만 {only_b}장, 기준만 {only_a}장)")
    return ScoreBatchComparison(role="compared", result=_comparison(sel, ref, b))


def score(conn: sqlite3.Connection, paths: DataPaths, req: ScoreRequest) -> ScoreResult:
    sel = _select(conn, paths, req.gt_set_id, req.batch_ids, req.criteria)
    full = req.detail == "full"
    n_gt = len(sel.links.gt)
    ref, note = _reference(sel.batches, req.reference_batch_id, n_gt)
    out: list[ScoreBatch] = []
    for b in sel.batches:
        n_eval = len(b.entries)
        out.append(ScoreBatch(
            batch_id=b.id, **datasets_mod.batch_fields(b.row), status="complete" if b.complete else "interim",
            pending=b.pending, n_eval_images=n_eval, videos_skipped=b.videos, full_coverage=n_eval == n_gt and n_gt > 0,
            metrics=_scored(sel, b).api,
            slices=_slices(sel, b) if full else None,
            augmentation_effect=_augmentation(sel, b) if full else None,
            comparison=_batch_comparison(sel, ref, b) if full else None,
        ))

    # Final ranking: complete, full-coverage batches by 허용 오차 충족률 then 이미지 전체 기준 충족률.
    def rank_key(s: ScoreBatch) -> tuple[float, float]:
        tol, img = s.metrics.tolerance_rate.value, s.metrics.image_pass_rate.value  # type: ignore[union-attr]
        return (-(tol if tol is not None else -1.0), -(img if img is not None else -1.0))

    ranked = sorted((s for s in out if s.status == "complete" and s.full_coverage), key=rank_key)
    return ScoreResult(
        gt_set=gtsets.info_of(sel.set_row, sel.links), criteria=ScoreCriteria.model_validate(sel.crit),
        reference_batch_id=ref.id if ref else None, reference_note=note, batches=out,
        final_ranking=[s.batch_id for s in ranked], slice_dimensions=SLICE_DEFS, scored_at=now_iso(),
    )


# --- one image -----------------------------------------------------------------------------------
def _counts(im: GtImage, s: ImageScore | None) -> dict[str, int | None]:
    """The ScoreImageCounts fields (outcome_counts); only n_gt while the image is not scored."""
    if s is None:
        return {"n_gt": len(im.cubes), "n_missed": None, "n_fp": None, "n_color_wrong": None, "n_geom_fail": None}
    return asdict(outcome_counts(s))


def _link_for_media(conn: sqlite3.Connection, media_id: int, gt_set_id: int | None) -> MediaGtLink:
    """The media's link in the given set, or in the newest set linking it."""
    media = media_mod.get_row(conn, media_id)
    found = gtsets.media_links(conn, media_id)
    if gt_set_id is not None:
        row = gtsets.set_row(conn, gt_set_id)
        found = [link for link in found if link.gt_set_id == gt_set_id]
        if not found:
            raise LabError(404, f'"{media["name"]}"은(는) 정답셋 "{row["name"]}"에 연결되어 있지 않습니다')
    elif not found:
        raise LabError(404, f'"{media["name"]}"에는 정답이 없습니다 (연결된 정답셋이 없음)')
    return found[0]


def _detail(link: MediaGtLink, im: GtImage, media_id: int, run_id: int | None, crit: Criteria,
            preds: list[CubeDict], s: ImageScore | None) -> ScoreImageDetail:
    return ScoreImageDetail(
        gt_set_name=link.gt_set_name, gt_index=link.gt_index, method=link.method, media_id=media_id, run_id=run_id,
        filename=im.filename, variant=im.variant, criteria=ScoreCriteria.model_validate(crit), scored=s is not None,
        ran=None if s is None else s.ran, passed=None if s is None else s.passed, **_counts(im, s),
        gt=list(im.cubes), predictions=preds, cubes=[] if s is None else list(s.cubes),
        false_positives=[] if s is None else [fp.pred_index for fp in s.false_pos],
    )


def image_detail(conn: sqlite3.Connection, paths: DataPaths, run_id: int, gt_set_id: int | None,
                 raw: ScoreCriteriaIn | None) -> ScoreImageDetail:
    """A run's frame 0 scored against its media's GT image (the viewer overlay)."""
    crit = criteria_of(raw)
    run = runs_mod.get_row(conn, run_id)
    link = _link_for_media(conn, run["media_id"], gt_set_id)
    im = gtsets.load(paths, link.gt_set_id).images[link.gt_index]
    preds: list[CubeDict] = []
    s = None
    if run["status"] == "done":
        frame = _frames0(conn, [run_id]).get(run_id)
        preds = frame[1] if frame else []
        s = score_image(im, preds if frame else None, frame[0] if frame else None, crit)
    elif run["status"] == "failed":
        s = score_image(im, None, None, crit)
    return _detail(link, im, run["media_id"], run_id, crit, preds, s)


def frame_detail(conn: sqlite3.Connection, paths: DataPaths, media_id: int, cubes: list[CubeDict],
                 gt_set_id: int | None, raw: ScoreCriteriaIn | None) -> ScoreImageDetail:
    """Cubes the caller has (a workbench preview) scored against the media's GT image."""
    crit = criteria_of(raw)
    link = _link_for_media(conn, media_id, gt_set_id)
    im = gtsets.load(paths, link.gt_set_id).images[link.gt_index]
    return _detail(link, im, media_id, None, crit, cubes, score_image(im, cubes, None, crit))


# --- GET /api/score/images -----------------------------------------------------------------------
def image_rows(conn: sqlite3.Connection, paths: DataPaths, gt_set_id: int, batch_id: int,
               raw: ScoreCriteriaIn | None) -> list[ScoreImageRow]:
    """Every evaluation image of one batch, in GT order, from the same cached image scores as
    POST /api/score. Images that are not finished carry their run state and null counts."""
    sel = _select(conn, paths, gt_set_id, [batch_id], raw)
    b = sel.batches[0]
    scored = {s.image.index: s for s in _scored(sel, b).scores}
    out = []
    for index, mid, run in b.entries:
        im = sel.gt.images[index]
        media = sel.links.images[mid]
        s = scored.get(index)
        out.append(ScoreImageRow(
            gt_index=index, media_id=mid, media_name=media.name, folder=media.folder, filename=im.filename,
            run_id=run["id"] if run is not None else None, variant=im.variant, source_sha256=im.source_sha256,
            layout_group=im.layout_group, status=run["status"] if run is not None else "missing",
            passed=None if s is None else s.passed, ms=None if s is None else s.ms, **_counts(im, s),
        ))
    return out


# --- CSV (§7) ------------------------------------------------------------------------------------
def _label(row: sqlite3.Row) -> str:
    return f"#{row['id']} {row['model_name']}" + (f" ({row['folder']})" if row["folder"] else "")


def _csv_name(sel: _Selection, what: str) -> str:
    stem = sel.set_row["name"].rsplit(".", 1)[0] if sel.set_row["name"].lower().endswith(".json") else sel.set_row["name"]
    return f"{datasets_mod.safe_name(sel.set_row['dataset_name'])}-{datasets_mod.safe_name(stem)}-채점-{what}.csv"


def cubes_csv(conn: sqlite3.Connection, paths: DataPaths, gt_set_id: int, batch_ids: list[int],
              raw: ScoreCriteriaIn | None) -> tuple[bytes, str]:
    """One row per GT cube and one per false positive of every scored image; interim batches
    included with status 중간."""
    sel = _select(conn, paths, gt_set_id, batch_ids, raw)
    rows = []
    for b in sel.batches:
        label = _label(b.row)
        for s in _scored(sel, b).scores:
            rows += cube_csv_rows(label, s, b.complete)
    return csv_bytes(rows, CUBE_CSV_FIELDS), _csv_name(sel, "큐브별")


def summary_csv(conn: sqlite3.Connection, paths: DataPaths, gt_set_id: int, batch_ids: list[int],
                raw: ScoreCriteriaIn | None) -> tuple[bytes, str]:
    """One row per batch and metric, from the cached metrics."""
    sel = _select(conn, paths, gt_set_id, batch_ids, raw)
    rows = []
    for b in sel.batches:
        metrics = _scored(sel, b).metrics
        if metrics.criteria is None:  # nothing scored yet: still say which thresholds were asked for
            metrics = replace(metrics, criteria=sel.crit)
        rows += summary_csv_rows(_label(b.row), metrics, b.complete)
    return csv_bytes(rows, SUMMARY_CSV_FIELDS), _csv_name(sel, "요약")

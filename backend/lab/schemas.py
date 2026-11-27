"""API contract. Mirrors frontend/src/api/types.ts -- keep the two files in sync.

Cube dictionaries come straight from cubevision.types.Cube.to_dict(); they are typed
here as plain dicts to avoid a second, drifting definition.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from cubevision.score import Criteria, Outcome
from cubevision.types import Color
from lab.config import MAX_BOOTSTRAP_REPS

CubeDict = dict[str, Any]
Params = dict[str, Any]

MediaKind = Literal["image", "video"]
RunStatus = Literal["queued", "running", "done", "failed", "cancelled"]


class Media(BaseModel):
    id: int
    kind: MediaKind
    name: str
    sha256: str
    bytes: int
    width: int
    height: int
    fps: float | None
    frame_count: int | None
    duration: float | None
    tags: list[str]
    note: str
    starred: bool
    created_at: str
    run_count: int
    file_url: str
    original_url: str
    thumb_url: str
    dataset_ids: list[int] = Field(default_factory=list)
    folder: str | None = None  # its folder in the dataset of ?dataset_id ('' = top level); null otherwise


class MediaUploadResult(BaseModel):
    media: Media
    duplicate: bool
    folder: str | None = None  # with a dataset: the media's folder there ('' = top level)
    already_in_dataset: bool = False  # it was in the dataset before: it stays in its folder


class MediaPatch(BaseModel):
    name: str | None = None
    tags: list[str] | None = None
    note: str | None = None
    starred: bool | None = None


class ModelInfo(BaseModel):
    key: str
    kind: Literal["pipeline", "weights"]
    name: str
    summary: str
    description: str
    params: list[dict[str, Any]]  # ParamSpec.to_dict()
    state: Literal["ready", "error"]
    error: str | None = None
    task: Literal["detect", "segment", "obb"] | None = None
    class_names: list[str] | None = None
    file_bytes: int | None = None
    note: str = ""
    created_at: str | None = None
    debug_labels: dict[str, str] = Field(default_factory=dict)  # pipelines: debug layer key -> Korean label


class ModelPatch(BaseModel):
    name: str | None = None
    note: str | None = None


class Preset(BaseModel):
    id: int
    model_key: str
    name: str
    params: Params
    created_at: str


class ParamCheckRequest(BaseModel):
    model_key: str
    params: Params = Field(default_factory=dict)


class ParamAdjust(BaseModel):
    name: str
    given: Any
    used: Any


class ParamCheck(BaseModel):
    """Params resolved exactly as a run would resolve them, and what happened to the input."""

    model_key: str
    params: Params  # every param of the model: the input where it was given, else the default
    applied: list[str]  # params the input set (today's names)
    ignored: list[str]  # input keys the model does not have
    adjusted: list[ParamAdjust]  # applied params whose value had to change (clamped, coerced, filtered)


class PresetCreate(BaseModel):
    model_key: str
    name: str = Field(min_length=1, max_length=80)
    params: Params


class DebugLayer(BaseModel):
    name: str
    label: str
    url: str


class PreviewRequest(BaseModel):
    media_id: int
    frame: int = 0
    model_key: str
    params: Params = Field(default_factory=dict)
    debug: bool = False


class PreviewResult(BaseModel):
    media_id: int
    frame: int
    model_key: str
    params: Params
    cubes: list[CubeDict]
    ms: float
    stages_ms: dict[str, float]
    notes: list[str]
    debug_layers: list[DebugLayer]


class Timing(BaseModel):
    frames: int
    first_ms: float
    mean_ms: float
    p50_ms: float
    p95_ms: float
    max_ms: float


class RunSummary(BaseModel):
    frames_with_cubes: int
    cubes_mean: float
    cubes_min: int
    cubes_max: int
    color_mean: dict[str, float]


class Run(BaseModel):
    id: int
    media_id: int
    media_name: str
    media_kind: MediaKind
    model_key: str
    model_name: str
    params: Params
    frame_start: int
    frame_end: int | None
    stride: int
    frames_total: int | None
    frames_done: int
    status: RunStatus
    device: str | None
    timing: Timing | None
    summary: RunSummary | None
    error: str | None
    parent_run_id: int | None
    batch_id: int | None = None  # set when the run was started for a whole dataset
    tags: list[str]
    note: str
    starred: bool
    created_at: str
    started_at: str | None
    finished_at: str | None
    thumb_url: str | None
    export_url: str | None


class RunCreate(BaseModel):
    media_id: int
    model_key: str
    params: Params = Field(default_factory=dict)
    frame_start: int = 0
    frame_end: int | None = None
    stride: int = Field(default=1, ge=1, le=1000)
    parent_run_id: int | None = None
    note: str = ""


class RunPatch(BaseModel):
    note: str | None = None
    tags: list[str] | None = None
    starred: bool | None = None


class RunList(BaseModel):
    items: list[Run]
    total: int


class FrameRecord(BaseModel):
    frame: int
    ms: float
    cubes: list[CubeDict]


class RunFrames(BaseModel):
    run_id: int
    fps: float | None
    stride: int
    frames: list[FrameRecord]


class ExportInfo(BaseModel):
    url: str
    bytes: int
    kind: MediaKind


class StorageUsage(BaseModel):
    media_bytes: int
    proxy_bytes: int
    thumbs_bytes: int
    weights_bytes: int
    exports_bytes: int
    db_bytes: int


class StorageInfo(BaseModel):
    data_dir: str
    free_bytes: int
    total_bytes: int
    usage: StorageUsage


class WorkerInfo(BaseModel):
    alive: bool
    pid: int | None
    busy_run_id: int | None
    loaded_model: str | None


class SystemInfo(BaseModel):
    version: str
    torch_available: bool
    mps_available: bool | None
    worker: WorkerInfo
    storage: StorageInfo


# --- Added by the backend (additive; mirrored in types.ts) ----------------------------------


class Ok(BaseModel):
    """Body of DELETE responses."""

    ok: bool = True


class Health(BaseModel):
    """GET /api/health. ``build`` changes whenever the frontend is rebuilt, so an open page can
    tell that it runs old code and offer a reload. ``features`` names the route groups this server
    has ("folders", "gt", "score"), so a page can tell an older server from a missing record."""

    ok: bool = True
    build: str | None = None
    features: list[str] = Field(default_factory=list)


class DatasetFolder(BaseModel):
    path: str  # names joined by '/', e.g. '증강/어둡게'
    count: int  # media directly in this folder
    total: int  # media in this folder and every folder below it


class Dataset(BaseModel):
    id: int
    name: str
    note: str
    created_at: str
    media_count: int
    image_count: int
    video_count: int
    batch_count: int
    originals_url: str  # zip of the original files
    folders: list[DatasetFolder] = Field(default_factory=list)  # sorted by path; parents before children


class DatasetCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    note: str = ""
    media_ids: list[int] = Field(default_factory=list)


class DatasetPatch(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=80)
    note: str | None = None


class DatasetMediaChange(BaseModel):
    media_ids: list[int] = Field(min_length=1)


class DatasetMediaMove(BaseModel):
    media_ids: list[int] = Field(min_length=1)
    folder: str = Field(default="", max_length=1000)  # '' = the dataset's top level; made if missing


class FolderCreate(BaseModel):
    path: str = Field(max_length=1000)  # missing parents are made too


class FolderRename(BaseModel):
    """Renames or moves a folder; its subfolders and media go with it."""

    path: str = Field(max_length=1000)
    new_path: str = Field(max_length=1000)


class FolderDelete(BaseModel):
    """Deletes a folder and its subfolders; their media leave this dataset only."""

    path: str = Field(max_length=1000)


class BatchCreate(BaseModel):
    """Run one model with one set of params on every media of a dataset (or of one folder)."""

    model_key: str
    params: Params = Field(default_factory=dict)
    stride: int = Field(default=1, ge=1, le=1000)  # videos: every N-th frame
    folder: str | None = Field(default=None, max_length=1000)  # one folder and its subfolders; null = all


class Batch(BaseModel):
    id: int
    dataset_id: int | None
    dataset_name: str
    folder: str | None = None  # the folder it ran on (with its subfolders); null = the whole dataset
    model_key: str
    model_name: str
    params: Params
    stride: int
    created_at: str
    counts: dict[str, int]  # total and per run status
    finished: bool  # nothing queued or running
    zip_url: str  # results of the finished runs: drawn images / videos + results.json + results.csv


class BatchItem(BaseModel):
    run_id: int
    media_id: int
    media_name: str
    media_kind: MediaKind
    folder: str | None = None  # the media's folder in the dataset now; null when no longer in it
    status: RunStatus
    frames_done: int
    frames_total: int | None
    summary: RunSummary | None
    error: str | None
    thumb_url: str | None
    download_url: str | None  # the drawn result, once the run is done


class BatchDetail(Batch):
    items: list[BatchItem]


class CleanupRequest(BaseModel):
    delete_exports: bool = False
    runs_older_than_days: int | None = Field(default=None, ge=0)  # null = keep all runs
    unstarred_only: bool = True  # starred runs are deleted only when this is false


class CleanupResult(BaseModel):
    freed_bytes: int


# --- Ground truth and scoring (docs/SCORING.md is the spec) ---------------------------------

LinkMethod = Literal["sha256", "confirmed"]
Verdict = Literal["higher", "lower", "unclear"]
GtColor = Literal["red", "green", "blue", "wood"]  # cubevision.gt.GT_COLORS (a label never says unknown)


class GtVariant(BaseModel):
    name: str  # augmentation.name ("original" when the file has none)
    count: int  # GT images of this variant


class GtIgnoredLink(BaseModel):
    """A confirmed link that does not count (and why): a sha256 link wins, or the media left the dataset."""

    gt_index: int
    media_id: int
    reason: str


class GtLinkReport(BaseModel):
    gt_images: int
    linked_sha256: int
    linked_confirmed: int
    unlinked: int  # GT images without a media
    dataset_images_without_gt: int
    dataset_videos: int  # never scored
    ignored: list[GtIgnoredLink] = Field(default_factory=list)


class GtSetInfo(BaseModel):
    id: int
    dataset_id: int
    name: str
    schema_version: str
    label_kind: str | None
    file_sha256: str
    file_bytes: int
    n_images: int
    n_cubes: int
    n_valid: int  # cubes with valid geometry
    n_layout_groups: int  # bootstrap resampling units
    variants: list[GtVariant]  # in file order
    created_at: str
    links: GtLinkReport


class GtMediaRef(BaseModel):
    media_id: int
    name: str
    folder: str  # its folder in the set's dataset ('' = top level)


class GtLinkedImage(BaseModel):
    gt_index: int
    filename: str
    variant: str
    method: LinkMethod
    media: GtMediaRef


class GtUnlinkedImage(BaseModel):
    gt_index: int
    filename: str
    variant: str
    candidates: list[GtMediaRef]  # dataset images with the same file name (NFC) that are not linked yet


class GtSetDetail(GtSetInfo):
    linked: list[GtLinkedImage]
    unlinked: list[GtUnlinkedImage]
    images_without_gt: list[GtMediaRef]


class GtSetPatch(BaseModel):
    name: str = Field(min_length=1, max_length=200)


class GtLinkIn(BaseModel):
    gt_index: int
    media_id: int


class GtLinksConfirm(BaseModel):
    links: list[GtLinkIn] = Field(min_length=1)


class GtOrganize(BaseModel):
    """Moves every linked media into "<root>/<variant>"."""

    root: str = Field(default="", max_length=1000)


class MediaGtLink(BaseModel):
    """A ground-truth set that has a label for this media."""

    gt_set_id: int
    gt_set_name: str
    dataset_id: int
    dataset_name: str
    gt_index: int
    filename: str
    variant: str
    method: LinkMethod


class ScoreCriteriaIn(BaseModel):
    """Every field has its docs/SCORING.md default and the bounds Criteria accepts (plus
    MAX_BOOTSTRAP_REPS): this schema is where the defaults and limits come from, also for
    GET /api/score/criteria. ``step`` is a hint for an input's arrows, not a constraint. A field
    left out or null takes the default."""

    center_px: float = Field(Criteria.center_px, ge=0, allow_inf_nan=False, json_schema_extra={"step": 0.5})  # τc
    angle_deg: float = Field(Criteria.angle_deg, ge=0, allow_inf_nan=False, json_schema_extra={"step": 0.5})  # τa
    vertex_px: float = Field(Criteria.vertex_px, ge=0, allow_inf_nan=False, json_schema_extra={"step": 0.5})  # τv
    gate_px: float = Field(Criteria.gate_px, ge=0, allow_inf_nan=False, json_schema_extra={"step": 1})  # g
    bootstrap_reps: int = Field(Criteria.bootstrap_reps, ge=1, le=MAX_BOOTSTRAP_REPS, json_schema_extra={"step": 100})
    bootstrap_level: float = Field(Criteria.bootstrap_level, gt=0, lt=1, json_schema_extra={"step": 0.01})
    bootstrap_seed: int = Field(Criteria.bootstrap_seed, ge=0, json_schema_extra={"step": 1})

    @model_validator(mode="before")
    @classmethod
    def _null_is_default(cls, data: Any) -> Any:
        return {k: v for k, v in data.items() if v is not None} if isinstance(data, dict) else data


class _FromScore(BaseModel):
    """Built with model_validate(obj) from the cubevision.score / cubevision.gt dataclass that
    has the same field names."""

    model_config = ConfigDict(from_attributes=True)


class ScoreCriteria(_FromScore):
    center_px: float
    angle_deg: float
    vertex_px: float
    gate_px: float
    bootstrap_reps: int
    bootstrap_level: float
    bootstrap_seed: int


class ScoreCriterionLimit(BaseModel):
    min: int | float | None  # null = no bound
    max: int | float | None
    min_exclusive: bool  # the bound itself is not allowed (bootstrap_level: 0 < level < 1)
    max_exclusive: bool
    step: int | float  # a hint for an input's arrows, not a constraint
    integer: bool


class ScoreCriteriaInfo(BaseModel):
    """GET /api/score/criteria: the defaults and limits of ScoreCriteriaIn, read from its schema."""

    defaults: ScoreCriteria
    limits: dict[str, ScoreCriterionLimit]  # by criteria field, in field order


class ScoreRatio(_FromScore):
    numerator: int
    denominator: int
    value: float | None  # null when the denominator is 0


class ScoreStats(_FromScore):
    n: int
    median: float | None  # null for an empty sample
    p95: float | None
    mean: float | None


class ScoreMetrics(_FromScore):
    """§3. In a cube slice the image-level fields (n_images, n_failed, n_preds, image_pass_rate,
    precision, time_ms) are null: they are not defined there."""

    n_images: int | None  # images scored (done or failed runs)
    n_failed: int | None  # of which the run failed (zero detections)
    n_preds: int | None
    n_cubes: int
    n_valid: int
    tolerance_rate: ScoreRatio
    image_pass_rate: ScoreRatio | None
    recall: ScoreRatio
    precision: ScoreRatio | None
    color_accuracy: ScoreRatio
    confusion: dict[GtColor, dict[Color, int]]  # GT colour -> predicted colour -> pairs
    vertex_err: ScoreStats
    center_err: ScoreStats
    angle_err: ScoreStats
    time_ms: ScoreStats | None


class ScoreSlice(_FromScore):
    dimension: str  # variant | region | image_side | color | near_contact | weak_boundary
    value: str | bool | None
    n: int  # images (variant) or GT cubes (the others)
    metrics: ScoreMetrics


class ScoreDiff(_FromScore):
    key: str  # a ratio key or "<stat>_median" / "<stat>_p95"
    a: float | None
    b: float | None
    delta: float | None  # b − a
    ci_low: float | None
    ci_high: float | None
    verdict: Verdict | None  # null when the metric was never defined on both sides
    reps_used: int


class ScoreComparison(_FromScore):
    n_images_a: int
    n_images_b: int
    n_groups: int  # layout groups resampled
    reps: int
    level: float
    seed: int
    diffs: list[ScoreDiff]


class ScoreAugmentationEffect(_FromScore):
    variant: str
    n_sources: int
    n_unpaired: int
    comparison: ScoreComparison  # A = the paired originals, B = this variant


class ScorePending(BaseModel):
    """Evaluation images that are not finished, by reason. All 0 for a complete batch."""

    queued: int = 0
    running: int = 0
    cancelled: int = 0
    missing: int = 0  # no run for this media in the batch (e.g. added after the batch ran)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total(self) -> int:
        return self.queued + self.running + self.cancelled + self.missing


class ScoreBatchComparison(BaseModel):
    role: Literal["reference", "compared", "not_comparable"]
    reason_code: Literal["interim", "different_images", "no_images", "no_reference"] | None = None
    reason: str | None = None  # Korean, when not compared
    result: ScoreComparison | None = None  # this batch (B) − the reference (A)


class ScoreBatch(BaseModel):
    batch_id: int
    dataset_id: int | None
    dataset_name: str
    folder: str | None  # scope: null = the whole dataset, else this folder with its subfolders
    model_key: str
    model_name: str
    params: Params
    stride: int
    created_at: str
    status: Literal["complete", "interim"]
    pending: ScorePending
    n_eval_images: int  # linked GT images in the scope (finished or not); coverage = this / gt_set.n_images
    videos_skipped: int  # videos in the scope: no ground truth
    full_coverage: bool
    metrics: ScoreMetrics  # over the scored images (metrics.n_images of them; interim: the finished ones only)
    slices: list[ScoreSlice] | None  # null for detail "summary" (also the two below)
    augmentation_effect: list[ScoreAugmentationEffect] | None
    comparison: ScoreBatchComparison | None


class ScoreSliceDimension(BaseModel):
    key: str
    label: str
    unit: Literal["image", "cube"]


class ScoreRequest(BaseModel):
    gt_set_id: int
    batch_ids: list[int] = Field(default_factory=list)  # empty = every batch of the set's dataset
    criteria: ScoreCriteriaIn = Field(default_factory=ScoreCriteriaIn)
    reference_batch_id: int | None = None
    # "summary": status, pending, coverage and metrics per batch only (no slices, augmentation
    # effect or comparison; the ranking is still ordered) -- the rail's score chips.
    detail: Literal["full", "summary"] = "full"


class ScoreResult(BaseModel):
    gt_set: GtSetInfo
    criteria: ScoreCriteria
    reference_batch_id: int | None  # the reference actually used
    reference_note: str | None  # why the requested reference was not used
    batches: list[ScoreBatch]
    # Batch ids, best first: complete, full-coverage batches by 허용 오차 충족률 then 이미지 전체 기준 충족률.
    final_ranking: list[int]
    slice_dimensions: list[ScoreSliceDimension]
    scored_at: str


class ScoreGtCube(_FromScore):
    id: str
    color: GtColor
    valid_geometry: bool
    quad: list[list[float]] | None
    center: list[float] | None
    match_point: list[float]


class ScoreCubeResult(_FromScore):
    """One GT cube's outcome, in GT order (cubevision.score.CubeScore holds the §3 pass rule)."""

    matched: bool
    pred_index: int | None  # the paired prediction
    distance: float | None  # match distance to the GT match point
    pred_color: Color | None
    color_ok: bool
    center_err: float | None
    angle_err: float | None
    vertex_err: float | None
    # Tolerance checks (error ≤ τ; a missing error fails); null for invalid geometry.
    center_ok: bool | None
    angle_ok: bool | None
    vertex_ok: bool | None
    geom_ok: bool | None  # all three
    passed: bool | None  # color_ok and geom_ok; null for invalid geometry (only match and colour apply)
    outcome: Outcome  # missed (no pair), pass, or fail; invalid geometry: pass = right colour


class ScoreImageCounts(BaseModel):
    """One image's counts (cubevision.score.outcome_counts); null while the image is not scored."""

    n_gt: int
    n_missed: int | None  # GT cubes without a pair
    n_fp: int | None  # detections without a pair
    n_color_wrong: int | None  # pairs with the wrong colour
    n_geom_fail: int | None  # valid-geometry pairs over a threshold or without centre / angle / quad


class ScoreImageDetail(ScoreImageCounts):
    gt_set_name: str
    gt_index: int
    method: LinkMethod
    media_id: int
    run_id: int | None
    filename: str
    variant: str
    criteria: ScoreCriteria
    scored: bool  # false while the run is queued, running or was cancelled
    ran: bool | None  # false for a failed run (zero detections)
    passed: bool | None
    gt: list[ScoreGtCube]
    predictions: list[CubeDict]
    cubes: list[ScoreCubeResult]  # one per GT cube; empty while not scored
    false_positives: list[int]  # prediction indices


class ScoreImageRow(ScoreImageCounts):
    """GET /api/score/images: one evaluation image of a batch (the 채점 page's per-image and
    per-source views). The counts are null while the image is not scored."""

    gt_index: int
    media_id: int
    media_name: str
    folder: str  # the media's folder in the set's dataset ('' = top level)
    filename: str  # the GT image's file name
    run_id: int | None  # null: the batch has no run for this media
    variant: str
    source_sha256: str
    layout_group: str
    status: Literal["queued", "running", "done", "failed", "cancelled", "missing"]  # missing = no run in the batch
    passed: bool | None  # image pass (§3); null while not scored
    ms: float | None


class ScoreFrameRequest(BaseModel):
    """Live score of cubes the workbench has (a preview), against the media's ground truth."""

    media_id: int
    cubes: list[CubeDict]
    gt_set_id: int | None = None  # null = the newest set that links the media
    criteria: ScoreCriteriaIn = Field(default_factory=ScoreCriteriaIn)

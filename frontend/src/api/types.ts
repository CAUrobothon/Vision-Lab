// API contract shared with backend/lab/schemas.py. Keep the two files in sync.
// All coordinates are image pixels (x right, y down). See backend/cubevision/types.py.

export type CubeColor = 'red' | 'green' | 'blue' | 'wood' | 'unknown'
export const CUBE_COLORS: CubeColor[] = ['red', 'green', 'blue', 'wood', 'unknown']
/** A ground-truth label's colour (a label never says unknown). */
export type GtColor = Exclude<CubeColor, 'unknown'>

export type Pt = [number, number]

export interface Cube {
  color: CubeColor
  score: number // detection confidence 0..1
  color_score: number // colour confidence 0..1
  center: Pt // top-face centre
  quad: Pt[] | null // 4 top-face corners, clockwise on screen, first = smallest x+y
  angle: number | null // degrees in [0, 90), from +x towards +y (clockwise on screen)
  area: number // px^2
  polygon: Pt[] | null // raw outline from the model
  box: [number, number, number, number] | null // raw x1, y1, x2, y2
  label: string | null // raw class name from a model
  flags: string[]
}

export type MediaKind = 'image' | 'video'

export interface Media {
  id: number
  kind: MediaKind
  name: string
  sha256: string
  bytes: number // original upload size
  width: number
  height: number
  fps: number | null // video only (proxy fps)
  frame_count: number | null // video only (proxy frames)
  duration: number | null // seconds, video only
  tags: string[]
  note: string
  starred: boolean
  created_at: string // ISO 8601
  run_count: number
  file_url: string // image file, or the browser-safe H.264 proxy for videos
  original_url: string // the untouched upload (download)
  thumb_url: string // 320 px JPEG
  dataset_ids: number[] // datasets this media is in
  folder?: string | null // its folder in the dataset of ?dataset_id ('' = top level); null otherwise
}

/** A folder inside a dataset. */
export interface DatasetFolder {
  path: string // names joined by '/', e.g. '증강/어둡게'
  count: number // media directly in this folder
  total: number // media in this folder and every folder below it
}

/** A named group of media (one media can be in several). */
export interface Dataset {
  id: number
  name: string
  note: string
  created_at: string
  media_count: number
  image_count: number
  video_count: number
  batch_count: number
  originals_url: string // zip of the original files
  folders: DatasetFolder[] // sorted by path; parents before children
}

/** One model with one set of params run on every media of a dataset. */
export interface Batch {
  id: number
  dataset_id: number | null
  dataset_name: string
  folder: string | null // the folder it ran on (with its subfolders); null = the whole dataset
  model_key: string
  model_name: string
  params: Params
  stride: number
  created_at: string
  counts: Record<RunStatus | 'total', number>
  finished: boolean // nothing queued or running
  zip_url: string // drawn results of the finished runs + results.json + results.csv
}

export interface BatchItem {
  run_id: number
  media_id: number
  media_name: string
  media_kind: MediaKind
  folder: string | null // the media's folder in the dataset now; null when no longer in it
  status: RunStatus
  frames_done: number
  frames_total: number | null
  summary: RunSummary | null
  error: string | null
  thumb_url: string | null
  download_url: string | null // the drawn result, once done
}

export interface BatchDetail extends Batch {
  items: BatchItem[]
}

export interface MediaUploadResult {
  media: Media
  duplicate: boolean // true if the same file (sha256) was already stored
  folder: string | null // with a dataset: the media's folder there ('' = top level)
  already_in_dataset: boolean // it was in the dataset before: it stays in its folder
}

export type ParamType = 'int' | 'float' | 'bool' | 'select' | 'multiselect' | 'range' | 'color' | 'roi'

export interface RoiPolygon {
  mode: 'include' | 'exclude'
  points: Pt[]
}

export type ParamValue = number | boolean | string | [number, number] | string[] | RoiPolygon[]
export type Params = Record<string, ParamValue>

export interface ParamOption {
  value: string
  label: string
}

export interface ParamSpec {
  name: string
  label: string
  type: ParamType
  default: ParamValue
  min: number | null
  max: number | null
  step: number | null
  options: ParamOption[] | null
  group: string
  help: string
  unit: string
}

export type ModelKind = 'pipeline' | 'weights'
export type ModelTask = 'detect' | 'segment' | 'obb'

export interface ModelInfo {
  key: string // "cv.*" for pipelines, "w.<id>" for uploaded weights
  kind: ModelKind
  name: string
  summary: string
  description: string
  params: ParamSpec[]
  state: 'ready' | 'error'
  error: string | null
  task: ModelTask | null // weights only
  class_names: string[] | null // weights only
  file_bytes: number | null // weights only
  note: string
  created_at: string | null
  debug_labels: Record<string, string> // pipelines: debug layer key -> Korean label (empty for weights)
}

/** POST /api/params/check: params resolved exactly as a run resolves them, and what happened to the input. */
export interface ParamAdjust {
  name: string
  given: unknown
  used: ParamValue
}

export interface ParamCheck {
  model_key: string
  params: Params // every param of the model: the input where given, else the default
  applied: string[] // params the input set
  ignored: string[] // input keys the model does not have
  adjusted: ParamAdjust[] // applied params whose value changed to fit (clamped, coerced, filtered)
}

export interface Preset {
  id: number
  model_key: string
  name: string
  params: Params
  created_at: string
}

export interface DebugLayer {
  name: string
  label: string // Korean label
  url: string // data:image/png;base64,... same size as the frame
}

export interface PreviewRequest {
  media_id: number
  frame: number // 0 for images
  model_key: string // pipelines only
  params: Params
  debug: boolean
}

export interface PreviewResult {
  media_id: number
  frame: number
  model_key: string
  params: Params // resolved (defaults filled, numbers clamped)
  cubes: Cube[]
  ms: number // pipeline time on the server
  stages_ms: Record<string, number>
  notes: string[]
  debug_layers: DebugLayer[]
}

export type RunStatus = 'queued' | 'running' | 'done' | 'failed' | 'cancelled'

export interface Timing {
  frames: number // frames measured
  first_ms: number // first frame (includes model warm-up)
  mean_ms: number // mean over frames after the first
  p50_ms: number
  p95_ms: number
  max_ms: number
}

export interface RunSummary {
  frames_with_cubes: number
  cubes_mean: number // mean cubes per processed frame
  cubes_min: number
  cubes_max: number
  color_mean: Record<CubeColor, number> // mean count per frame for each colour
}

export interface Run {
  id: number
  media_id: number
  media_name: string
  media_kind: MediaKind
  model_key: string
  model_name: string // copied at creation, survives model deletion
  params: Params // resolved
  frame_start: number // inclusive
  frame_end: number | null // exclusive, null = until the end
  stride: number // process every Nth frame
  frames_total: number | null
  frames_done: number
  status: RunStatus
  device: string | null // "cpu" | "mps" | "opencv"
  timing: Timing | null
  summary: RunSummary | null
  error: string | null
  parent_run_id: number | null // set by "re-run with changes"
  batch_id: number | null // set when the run was started for a whole dataset
  tags: string[]
  note: string
  starred: boolean
  created_at: string
  started_at: string | null
  finished_at: string | null
  thumb_url: string | null // annotated 320 px JPEG once a frame exists
  export_url: string | null // baked PNG/MP4 once exported
}

export interface RunCreate {
  media_id: number
  model_key: string
  params: Params
  frame_start?: number
  frame_end?: number | null
  stride?: number
  parent_run_id?: number | null
  note?: string
}

export interface RunList {
  items: Run[]
  total: number
}

export interface FrameRecord {
  frame: number
  ms: number
  cubes: Cube[]
}

export interface RunFrames {
  run_id: number
  fps: number | null
  stride: number
  frames: FrameRecord[] // sorted by frame
}

export interface ExportInfo {
  url: string
  bytes: number
  kind: 'image' | 'video'
}

export interface StorageInfo {
  data_dir: string
  free_bytes: number
  total_bytes: number
  usage: {
    media_bytes: number
    proxy_bytes: number
    thumbs_bytes: number
    weights_bytes: number
    exports_bytes: number
    db_bytes: number
  }
}

export interface SystemInfo {
  version: string
  torch_available: boolean
  mps_available: boolean | null
  worker: {
    alive: boolean
    pid: number | null
    busy_run_id: number | null
    loaded_model: string | null
  }
  storage: StorageInfo
}

export interface ApiError {
  detail: string // Korean, human-readable
}

// --- Added by the backend (additive; mirrored in schemas.py) ---

// Body of DELETE responses.
export interface Ok {
  ok: boolean
}

// GET /api/health. `build` changes whenever the frontend is rebuilt; `features` names the route
// groups this server has ("folders", "gt", "score"), so an older server is told apart.
export interface Health {
  ok: boolean
  build: string | null
  features: string[]
}

// POST /api/storage/cleanup
export interface CleanupRequest {
  delete_exports: boolean
  runs_older_than_days: number | null // null = keep all runs; 0 = every finished run
  unstarred_only: boolean // starred runs are deleted only when this is false
}

export interface CleanupResult {
  freed_bytes: number
}

// --- Ground truth and scoring (docs/SCORING.md is the spec; mirrors schemas.py) ---

export type LinkMethod = 'sha256' | 'confirmed'
export type Verdict = 'higher' | 'lower' | 'unclear'

export interface GtVariant {
  name: string // augmentation.name ("original" when the file has none)
  count: number // GT images of this variant
}

/** A confirmed link that does not count (and why): a sha256 link wins, or the media left the dataset. */
export interface GtIgnoredLink {
  gt_index: number
  media_id: number
  reason: string
}

export interface GtLinkReport {
  gt_images: number
  linked_sha256: number
  linked_confirmed: number
  unlinked: number // GT images without a media
  dataset_images_without_gt: number
  dataset_videos: number // never scored
  ignored: GtIgnoredLink[]
}

export interface GtSetInfo {
  id: number
  dataset_id: number
  name: string
  schema_version: string
  label_kind: string | null
  file_sha256: string
  file_bytes: number
  n_images: number
  n_cubes: number
  n_valid: number // cubes with valid geometry
  n_layout_groups: number // bootstrap resampling units
  variants: GtVariant[] // in file order
  created_at: string
  links: GtLinkReport
}

export interface GtMediaRef {
  media_id: number
  name: string
  folder: string // its folder in the set's dataset ('' = top level)
}

export interface GtLinkedImage {
  gt_index: number
  filename: string
  variant: string
  method: LinkMethod
  media: GtMediaRef
}

export interface GtUnlinkedImage {
  gt_index: number
  filename: string
  variant: string
  candidates: GtMediaRef[] // dataset images with the same file name (NFC) that are not linked yet
}

export interface GtSetDetail extends GtSetInfo {
  linked: GtLinkedImage[]
  unlinked: GtUnlinkedImage[]
  images_without_gt: GtMediaRef[]
}

export interface GtLinkIn {
  gt_index: number
  media_id: number
}

/** A ground-truth set that has a label for this media. */
export interface MediaGtLink {
  gt_set_id: number
  gt_set_name: string
  dataset_id: number
  dataset_name: string
  gt_index: number
  filename: string
  variant: string
  method: LinkMethod
}

/** Any field left out or null takes the server's default (GET /api/score/criteria); a value out
 *  of its limits is refused (400). */
export interface ScoreCriteriaIn {
  center_px?: number | null // τc
  angle_deg?: number | null // τa
  vertex_px?: number | null // τv
  gate_px?: number | null // g
  bootstrap_reps?: number | null
  bootstrap_level?: number | null
  bootstrap_seed?: number | null
}

export interface ScoreCriteria {
  center_px: number
  angle_deg: number
  vertex_px: number
  gate_px: number
  bootstrap_reps: number
  bootstrap_level: number
  bootstrap_seed: number
}

/** A criteria field's limits: null = no bound; an exclusive bound itself is not allowed. `step`
 *  is a hint for an input's arrows, not a constraint. */
export interface ScoreCriterionLimit {
  min: number | null
  max: number | null
  min_exclusive: boolean
  max_exclusive: boolean
  step: number
  integer: boolean
}

/** GET /api/score/criteria */
export interface ScoreCriteriaInfo {
  defaults: ScoreCriteria
  limits: Record<keyof ScoreCriteria, ScoreCriterionLimit>
}

export interface ScoreRatio {
  numerator: number
  denominator: number
  value: number | null // null when the denominator is 0
}

export interface ScoreStats {
  n: number
  median: number | null // null for an empty sample
  p95: number | null
  mean: number | null
}

/** §3. In a cube slice the image-level fields are null: they are not defined there. */
export interface ScoreMetrics {
  n_images: number | null
  n_failed: number | null
  n_preds: number | null
  n_cubes: number
  n_valid: number
  tolerance_rate: ScoreRatio
  image_pass_rate: ScoreRatio | null
  recall: ScoreRatio
  precision: ScoreRatio | null
  color_accuracy: ScoreRatio
  confusion: Record<GtColor, Record<CubeColor, number>> // GT colour -> predicted colour -> pairs
  vertex_err: ScoreStats
  center_err: ScoreStats
  angle_err: ScoreStats
  time_ms: ScoreStats | null
}

export interface ScoreSlice {
  dimension: string // variant | region | image_side | color | near_contact | weak_boundary
  value: string | boolean | null
  n: number // images (variant) or GT cubes (the others)
  metrics: ScoreMetrics
}

export interface ScoreDiff {
  key: string // a ratio key or "<stat>_median" / "<stat>_p95"
  a: number | null
  b: number | null
  delta: number | null // b − a
  ci_low: number | null
  ci_high: number | null
  verdict: Verdict | null // null when the metric was never defined on both sides
  reps_used: number
}

export interface ScoreComparison {
  n_images_a: number
  n_images_b: number
  n_groups: number // layout groups resampled
  reps: number
  level: number
  seed: number
  diffs: ScoreDiff[]
}

export interface ScoreAugmentationEffect {
  variant: string
  n_sources: number
  n_unpaired: number
  comparison: ScoreComparison // A = the paired originals, B = this variant
}

/** Evaluation images that are not finished, by reason. All 0 for a complete batch. */
export interface ScorePending {
  queued: number
  running: number
  cancelled: number
  missing: number // no run for this media in the batch (e.g. added after the batch ran)
  total: number // all four
}

export interface ScoreBatchComparison {
  role: 'reference' | 'compared' | 'not_comparable'
  reason_code: 'interim' | 'different_images' | 'no_images' | 'no_reference' | null
  reason: string | null // Korean, when not compared
  result: ScoreComparison | null // this batch (B) − the reference (A)
}

export interface ScoreBatch {
  batch_id: number
  dataset_id: number | null
  dataset_name: string
  folder: string | null // scope: null = the whole dataset, else this folder with its subfolders
  model_key: string
  model_name: string
  params: Params
  stride: number
  created_at: string
  status: 'complete' | 'interim'
  pending: ScorePending
  n_eval_images: number // linked GT images in the scope (finished or not); coverage = this / gt_set.n_images
  videos_skipped: number // videos in the scope: no ground truth
  full_coverage: boolean
  metrics: ScoreMetrics // over the scored images (metrics.n_images of them; interim: the finished ones only)
  slices: ScoreSlice[] | null // null for detail "summary" (also the two below)
  augmentation_effect: ScoreAugmentationEffect[] | null
  comparison: ScoreBatchComparison | null
}

export interface ScoreSliceDimension {
  key: string
  label: string
  unit: 'image' | 'cube'
}

export interface ScoreRequest {
  gt_set_id: number
  batch_ids: number[] // empty = every batch of the set's dataset
  criteria: ScoreCriteriaIn
  reference_batch_id: number | null
  // "summary": status, pending, coverage and metrics per batch only (slices, augmentation effect
  // and comparison null, no bootstrap; the ranking is still ordered). Default "full".
  detail?: 'full' | 'summary'
}

export interface ScoreResult {
  gt_set: GtSetInfo
  criteria: ScoreCriteria
  reference_batch_id: number | null // the reference actually used
  reference_note: string | null // why the requested reference was not used
  batches: ScoreBatch[]
  // Batch ids, best first (rank = index + 1): complete, full-coverage batches by 허용 오차 충족률,
  // then 이미지 전체 기준 충족률.
  final_ranking: number[]
  slice_dimensions: ScoreSliceDimension[]
  scored_at: string
}

export interface ScoreGtCube {
  id: string
  color: GtColor
  valid_geometry: boolean
  quad: Pt[] | null
  center: Pt | null
  match_point: Pt
}

/** One GT cube's outcome, in GT order (the server applies the §3 pass rule). */
export interface ScoreCubeResult {
  matched: boolean
  pred_index: number | null // the paired prediction
  distance: number | null // match distance to the GT match point
  pred_color: CubeColor | null
  color_ok: boolean
  center_err: number | null
  angle_err: number | null
  vertex_err: number | null
  // Tolerance checks (error ≤ τ; a missing error fails); null for invalid geometry.
  center_ok: boolean | null
  angle_ok: boolean | null
  vertex_ok: boolean | null
  geom_ok: boolean | null // all three
  passed: boolean | null // color_ok and geom_ok; null for invalid geometry (only match and colour apply)
  outcome: 'pass' | 'fail' | 'missed' // missed = no pair; invalid geometry: pass = right colour
}

/** One image's counts (the same server function for every view); null while it is not scored. */
export interface ScoreImageCounts {
  n_gt: number
  n_missed: number | null // GT cubes without a pair
  n_fp: number | null // detections without a pair
  n_color_wrong: number | null // pairs with the wrong colour
  n_geom_fail: number | null // valid-geometry pairs over a threshold or without centre / angle / quad
}

export interface ScoreImageDetail extends ScoreImageCounts {
  gt_set_name: string
  gt_index: number
  method: LinkMethod
  media_id: number
  run_id: number | null
  filename: string
  variant: string
  criteria: ScoreCriteria
  scored: boolean // false while the run is queued, running or was cancelled
  ran: boolean | null // false for a failed run (zero detections)
  passed: boolean | null
  gt: ScoreGtCube[]
  predictions: Cube[]
  cubes: ScoreCubeResult[] // one per GT cube; empty while not scored
  false_positives: number[] // prediction indices
}

/** GET /api/score/images: one evaluation image of a batch. Counts are null while not scored. */
export interface ScoreImageRow extends ScoreImageCounts {
  gt_index: number
  media_id: number
  media_name: string
  folder: string // the media's folder in the set's dataset ('' = top level)
  filename: string // the GT image's file name
  run_id: number | null // null: the batch has no run for this media
  variant: string
  source_sha256: string
  layout_group: string
  status: RunStatus | 'missing' // missing = no run in the batch
  passed: boolean | null // image pass (§3); null while not scored
  ms: number | null
}

/** Live score of cubes the workbench has (a preview), against the media's ground truth. */
export interface ScoreFrameRequest {
  media_id: number
  cubes: Cube[]
  gt_set_id: number | null // null = the newest set that links the media
  criteria: ScoreCriteriaIn
}

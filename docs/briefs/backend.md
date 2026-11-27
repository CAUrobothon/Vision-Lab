# Build brief — `backend/lab` (FastAPI app, storage, worker, Ultralytics adapter)

Read `docs/briefs/common.md` and `docs/DESIGN.md` first. You own `backend/lab/**` (except
`schemas.py`, which you implement against but may only extend additively, mirrored in
`frontend/src/api/types.ts`) and `backend/tests/lab/**`. You do NOT edit `backend/cubevision/**`
(another agent is improving it in parallel); call it only through the contracts.

## Entry points
- `lab.main:create_app(data_dir: Path | None = None) -> FastAPI` (tests pass a tmp dir).
- `lab.main:cli()` = the `lab` console script: `uv run lab [--host 127.0.0.1] [--port 8010] [--data PATH]`
  runs uvicorn (no reload) and serves the API under `/api` plus, if `frontend/dist` exists, the
  built SPA at `/` with an index.html fallback for non-API paths (client-side routes such as
  `/runs`, `/compare`, `/models` must load).
- Default data dir: `<project>/data` (override with `LAB_DATA_DIR` or `--data`).

## Storage (`DESIGN.md` §4)
- stdlib `sqlite3`, one connection per request/thread, `PRAGMA journal_mode=WAL`,
  `foreign_keys=ON`, `busy_timeout=5000`, migrations via `PRAGMA user_version`.
- JSON columns stored as TEXT. Timestamps ISO 8601 with timezone (local time is fine).
- Files: `data/media/<sha256>.<ext>`, `data/proxy/<sha256>.mp4`, `data/thumbs/`, `data/weights/`,
  `data/exports/`, `data/tmp/`, `data/ultralytics/` (YOLO_CONFIG_DIR).
- Refuse uploads, runs and exports with HTTP 507 and a Korean message when free disk < 2 GB.

## Media ingest (`POST /api/media`, multipart field `file`)
- Stream the upload to `data/tmp` in chunks while computing sha256 (do not read it all into RAM).
  Duplicate sha256 → delete the temp file, return the existing media with `duplicate: true`.
- Images: jpg/jpeg/png/webp/bmp/tif(f) kept as uploaded (decode with `cv2.imread`). Anything else
  that is an image (e.g. HEIC) → try `ffmpeg -i in -frames:v 1 out.png`; on failure return 415 with
  a Korean message. Store width/height of the decoded image.
- Videos (mp4/mov/m4v/webm/mkv/avi or anything ffprobe reports as video): always build the proxy
  with ffmpeg: H.264 (libx264, `-preset veryfast -crf 18`), `yuv420p`, constant frame rate equal to
  the rounded average frame rate (clamp 1–60), even dimensions, rotation metadata applied (ffmpeg's
  default autorotate), no audio, `-g 15 -bf 0`, `-movflags +faststart`. Read fps, frame count
  (`ffprobe -count_packets` on the proxy) and size from the proxy. If ffmpeg fails, delete
  everything and return 415 with the ffmpeg error tail.
- Thumbnail: 320 px wide JPEG of frame 0.
- `GET /api/media/{id}/file` serves the image or the proxy with HTTP Range (FileResponse);
  `/original` serves the upload as an attachment; `/frame/{n}` returns frame n as JPEG (quality 92).

## Frame access (`lab/frames.py`)
- `get_frame(media, n) -> np.ndarray` (BGR). Images ignore n (must be 0). Videos: a small LRU of
  open `cv2.VideoCapture` handles on the proxy plus an LRU of decoded frames (≈16). Seek with
  `CAP_PROP_POS_FRAMES`; the proxy has no B-frames and a keyframe every 15 frames, so seeking must
  be exact — prove it with a test video whose frames encode their own index.
- Sequential iteration helper for runs/exports: `iter_frames(media, start, end, stride)` using
  `read()` for kept frames and `grab()` for skipped ones (no per-frame seeking).

## Preview (`POST /api/preview`)
- Pipelines only (`cv.*`); weights models → 400 "가중치 모델은 미리보기 대신 실행을 사용하세요".
- Runs in the API process (OpenCV releases the GIL; use a plain `def` endpoint so it runs in the
  threadpool). Resolve params, call the pipeline with `debug=req.debug`, encode each debug layer as
  a PNG data URL (masks as 1-channel PNG), label = `pipeline.debug_labels.get(name, name)`.
- Target: round trip under 60 ms for a 1280x720 image with the default pipeline. Measure and report.

## Runs and the worker process
- `POST /api/runs` validates media, model and params (resolve them; store resolved params),
  computes `frames_total = ceil((end - start) / stride)` (images: start 0, end 1, stride 1),
  inserts status `queued`, returns the Run.
- One worker process (`multiprocessing.get_context("spawn")`), started in the FastAPI lifespan and
  terminated on shutdown. It never shares objects with the API; it talks only through SQLite.
  Loop: claim the oldest queued run atomically (`UPDATE ... WHERE id=? AND status='queued'`),
  process it, repeat; poll every 100 ms when idle. Write a heartbeat row (pid, busy_run_id,
  loaded_model, updated_at) so `GET /api/system` can report it.
- Processing: iterate frames, run the model, measure wall-clock ms per frame (inference +
  post-processing, decode excluded), filter cubes by the run's ROI param if the model did not
  already, store `run_frames` in batches (every 10 frames or 0.5 s) and update `frames_done`.
  Check `cancel_requested` at least every 0.2 s → status `cancelled`, keep stored frames.
  On exception → `failed` with a Korean summary + the exception text (no traceback spam).
  On finish compute `timing` (first frame separately; mean/p50/p95/max over the rest, or over the
  single frame) and `summary` (see schemas), and write a 320 px annotated thumbnail of the first
  processed frame with `cubevision.draw.draw_cubes`.
- On API startup mark runs left `running` as `failed` ("서버가 다시 시작되어 중단됨").
- `POST /api/runs/{id}/cancel` sets the flag (queued runs become `cancelled` immediately).
- `POST /api/worker/restart` terminates and respawns the worker (marks its running run failed).
- `GET /api/runs` supports the filters in DESIGN.md §5, newest first, `limit` (default 100) and
  `offset`, plus `total`. `GET /api/runs/{id}/frames` returns all stored frames sorted.
- Deleting a run removes its frames, thumbnail and export. Deleting media removes its runs and files.

## Ultralytics adapter (`lab/ultra.py`, used only inside the worker and in a subprocess)
Facts verified on 2026-11-25 (ultralytics 8.4.215, torch 2.15, M4):
- Before importing ultralytics set: `YOLO_CONFIG_DIR=<data>/ultralytics`, `YOLO_AUTOINSTALL=False`,
  `YOLO_OFFLINE=1`, `YOLO_VERBOSE=False`, `PYTORCH_ENABLE_MPS_FALLBACK=1`; after import
  `ultralytics.settings.update({"sync": False, "weights_dir": <data>/weights, "runs_dir":
  <data>/ultralytics/runs, "datasets_dir": <data>/ultralytics/datasets})`.
- On a Mac an unset device means CPU: pass `device="mps"` (or "cpu") explicitly on every predict.
- `Results.speed` is not synchronised on MPS; time with perf_counter around predict + conversion.
- Load with an absolute path: `YOLO("/abs/path.pt")`. `model.task` in {"detect","segment","obb",...};
  `model.names` is {id: name}. Train imgsz: `(model.ckpt or {}).get("train_args", {}).get("imgsz")`.
- Results: `r.obb.xyxyxyxy` (N,4,2) corners, `r.obb.conf`, `r.obb.cls`; `r.masks.xy` list of (P,2)
  polygons (use `retina_masks=True` for full-resolution masks); `r.boxes.xyxy/conf/cls`.
- Pass frames as HxWx3 uint8 BGR numpy arrays. Use `verbose=False`. Never call `r.plot()`.
- Free memory when switching models: `del model; gc.collect(); torch.mps.empty_cache()`.
  Keep at most one loaded model in the worker.
- Pretrained COCO/DOTA checkpoints have no cube class: zero detections is a valid result.
Upload flow (`POST /api/models/weights`, multipart `file` + optional `name`): store as
`data/weights/<id>.pt`, then inspect it in a SUBPROCESS (`uv`'s interpreter: `sys.executable -m
lab.ultra inspect <path>` printing JSON) so the API process never imports torch. Accept tasks
detect / segment / obb; anything else or a non-Ultralytics file → keep the row with
`state="error"` and a Korean `error` message (do not 500). Weights model key: `w.<id>`.
Params for weights models (ParamSpec list built in `ultra.py` without importing torch):
`conf` float 0.01–1 (0.25), `iou` float 0.05–0.95 (0.5), `imgsz` select 640/960/1280 (default:
train imgsz if it is one of them, else 1280), `max_det` int 1–300 (100), `device` select
auto/mps/cpu (auto = mps when available), `retina_masks` bool (true, segment only),
`color_source` select auto/class/pixels (auto = class name when it names a colour, else pixels),
`proto_red/green/blue/wood` color params (defaults from `cubevision.colour.DEFAULT_PROTOTYPES`),
`roi` (group "검출 영역"). Class name → colour: case-insensitive exact or prefix match of
red/r/빨강 → red, green/g/초록 → green, blue/b/파랑 → blue, wood/w/나무 → wood.
Conversion: obb → `postprocess.cube_from_quad`, segment → `cube_from_polygon`, detect →
`cube_from_box`; pass `color=` when the class decides the colour, else `prototypes=`.
Tests must not download anything: build a random-weight checkpoint offline with
`YOLO("yolo26n-seg.yaml")` (and an obb/detect yaml) and save it to a tmp `.pt`, then exercise
upload → inspect → run (zero or random detections are fine; the plumbing is what is tested).
If building from yaml turns out to need a download, mark that test skipped and say why.

## Export (`POST /api/runs/{id}/export`, `GET` downloads it)
- Image run → PNG of `draw_cubes(frame, cubes)`. Video run → every proxy frame from frame_start to
  the last processed frame, overlay = nearest processed frame ≤ current frame, piped as raw BGR to
  `ffmpeg -f rawvideo -pix_fmt bgr24 -s WxH -r FPS -i - -c:v libx264 -preset veryfast -crf 20
  -pix_fmt yuv420p -movflags +faststart`. Synchronous is acceptable; stream frames, never hold the
  whole video in RAM.

## Samples (`POST /api/samples`)
- Call `cubevision.synth.sample_assets(out_dir: Path) -> list[Path]` (another agent writes it: a
  couple of synthetic still images and one short synthetic video) and ingest each file like an
  upload, tags `["sample", "합성"]`. Return `list[Media]`. If the function does not exist yet when
  you test, write the endpoint anyway and mark its test xfail with a clear reason.

## Storage endpoints
- `GET /api/storage` → StorageInfo (walk the data dirs; cache sizes for ~5 s).
- `POST /api/storage/cleanup` body `{delete_exports: bool, runs_older_than_days: int | null,
  unstarred_only: bool}` → `{freed_bytes: int}`. Never delete starred runs unless explicitly asked.

## System
- `GET /api/system`: version, `torch_available` (importlib.util.find_spec, no import),
  `mps_available` (from the worker heartbeat; the worker checks once at start), worker info, storage.

## Errors
Every error response is `{"detail": "<Korean sentence>"}` with a fitting status (400 bad input,
404 missing, 409 conflict, 413/415 bad upload, 507 no disk). Unknown exceptions → 500 with a short
Korean message; log the traceback to stderr.

## Tests (`backend/tests/lab/`, run with `cd backend && uv run pytest -q`)
Use a tmp data dir per test session. Generate tiny media in the tests (cv2 images; a 2–3 s video
via ffmpeg). Cover: upload image/video + dedup, proxy properties (ffprobe: h264, yuv420p, CFR,
frame count), exact frame access, preview (with a registered test pipeline if the real ones are
not there yet), runs end-to-end through the real worker process (image + video with stride,
cancel), frames endpoint, filters, patch/delete cascades, export image + video (ffprobe the
output), storage + cleanup, weights upload with an offline random checkpoint, SPA fallback.
Report the test count and timings.

## Ports while you work
Other agents work in parallel: use TestClient for tests; if you need a real server, run it on
127.0.0.1:8011 with a tmp `--data` dir and stop it afterwards. Never touch 8010, 8012 or 5190.
The cubevision agent is rewriting pipelines in parallel: if `list_pipelines()` is still empty when
you test, register a tiny test-only pipeline from your test code (do not add it to cubevision).

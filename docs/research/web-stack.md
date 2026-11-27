# Vision lab: architecture memo

Verdict: keep custom FastAPI + React, trimmed to three routes, no ORM, no UI kit, no auth.

## Build vs reuse
- **Gradio 6.32**: fastest first demo (`live=True`), but each slider tick round-trips a rendered image through its queue (latency unverified); history, tags, compare and synced video overlays would all be custom components. Meets 1-3, not 4-5.
- **Streamlit 1.67**: "reruns your entire Python script from top to bottom" on each widget change; same gaps, weaker viewer.
- **NiceGUI 4.1**: strongest contender; `ui.interactive_image` is an image with SVG overlay and image-coordinate mouse events. Video sync and history screens still need custom JS in a stack the student does not know.
- **FiftyOne 1.22**: compares predictions over a dataset; not upload-and-run, no slider loop; brings MongoDB (`fiftyone-db`, 41 MB wheel) and pins `opencv-python-headless<5`.
- **Label Studio / CVAT**: annotation tools (74 MB wheel; Docker); relevant only if labelling starts.
- **Roboflow Inference 1.8.6**: inference server with no upload or history UI; `inference server start` pulls a Docker image; 85 requirements whose pins (`opencv<4.13`, `av==14.2.0`) miss the local cache. **Supervision** (0.6 MB) is fine as an optional drawing library.

Challenge to the default: Gradio wins the first hour. But the defining requirements (large zoomable viewer, record management, compare, 10 ms slider loop) are where those frameworks must be fought, and React boilerplate is cheap for an agent. If the day slips, cut scope (compare, export), not the framework.

## Backend architecture
- **Fast lane**: OpenCV pipelines run in-process in sync `def` endpoints (threadpool). `POST /api/preview {media_id, frame, pipeline, params}` returns detections JSON only; the client draws. Previews are ephemeral: nothing enters history until "Save as run", otherwise slider drags flood the records.
- **Heavy lane**: one dedicated worker process (`multiprocessing` spawn, the macOS default; fork is unsafe there) owning torch and MPS; the API process never imports torch. One process per model family would duplicate torch's footprint in 16 GB.
- **Queue and cancel**: `runs.status` in SQLite, single FIFO consumer; stale `running` rows become `failed` on startup. Cancel is a cooperative `mp.Event` checked between frames; if not honoured within about 3 s, `terminate()` and respawn. Partial frames are kept.
- **Memory**: one heavy model loaded at a time; evict with `del`, `gc.collect()`, `torch.mps.empty_cache()`. "Restart worker" is the guaranteed release. macOS swap sits on the same nearly full volume, so memory pressure also eats disk.
- **Progress**: one SSE stream `/api/events` for all jobs (native `fastapi.sse.EventSourceResponse`, FastAPI 0.135+) that invalidates react-query keys; polling as fallback.
- **Weights**: catalogue with state (ready, missing, downloading, gated, error) and declared size. Download is an explicit job with a free-space pre-check, never implicit inside a run. Gated repositories (no Hugging Face token) show `gated`, not a stack trace.
- **Upload**: raw-body `PUT /api/media?name=` read with `request.stream()` into `data/tmp` while hashing; multipart would spool a temporary copy first.

## Storage
SQLite via stdlib `sqlite3` (WAL, foreign keys, `PRAGMA user_version` migrations):

```
media(id, sha256 UNIQUE, kind, name, bytes, w, h, fps, n_frames, proxy_path, tags, note, starred, created_at)
models(key PK, family, weights_ref, weights_bytes, param_schema, state)
presets(id, model_key, name, params, UNIQUE(model_key, name))
runs(id, media_id, model_key, params, parent_run_id, device, status, stride, max_frames, frames_done, infer_ms_p50, summary, error, tags, note, starred, created_at)
run_frames(run_id, frame_idx, infer_ms, cubes, raw, PRIMARY KEY(run_id, frame_idx)) -- cascade delete
```

Files under `data/`: `lab.sqlite3`, `media/<sha>.<ext>`, `proxy/<sha>.mp4`, `thumbs/`, `runs/<id>/`, `tmp/`, `weights/` (point `HF_HOME`, `TORCH_HOME` and the Ultralytics weights directory here).
- **Dedup**: sha256 computed while streaming; on a hit, delete the temp file and return the existing row.
- **Per run**: detections in `run_frames` (masks as polygons, never rasters), a 320 px annotated thumbnail, timings. No full annotated image or video: both are redrawn client-side.
- **Cleanup**: always-visible disk bar (`shutil.disk_usage` plus per-category bytes); refuse uploads and downloads below a 1.5 GB floor; bulk delete for old unstarred runs, exports, unused weights, originals that have a proxy.
- **Measured on this Mac**: the uv cache already holds torch 2.15.0 (570 MB), torchvision 0.29.2, ultralytics 8.4.215, opencv-python 5.1.0.95, fastapi 0.142.2. uv's default link mode on macOS is `clone`, so pinning exactly these versions makes the venv nearly free.

## Video pipeline
- **Ingest**: `ffprobe`, then transcode to a proxy unless already H.264 8-bit yuv420p, constant frame rate, unrotated: `ffmpeg -i in -an -vf "fps=30,format=yuv420p" -c:v libx264 -preset veryfast -crf 20 -g 15 -bf 0 -movflags +faststart`. HEVC plays in Safari and in Chrome 107+ on macOS, but variable frame rate, rotation metadata and 10-bit HDR make frame indexing unreliable, so normalise anyway. Inference and the browser read the same proxy, so frame N is the same picture in both; `idx = round(mediaTime * fps)` (heuristic, must be tested).
- **Decode**: sequential `cv2.VideoCapture.read()/grab()` on the proxy, no `CAP_PROP_POS_FRAMES` seeking inside jobs. Stride and max-frames are run parameters (suggested heavy-model defaults: stride 5, max 150). PyAV is unnecessary.
- **Encode**: `mp4v` is MPEG-4 Part 2; MDN lists Chrome and Safari as "No". Pipe raw BGR frames to ffmpeg (libx264 or h264_videotoolbox, yuv420p, faststart).
- **Overlays, both with roles**: primary is a client-side SVG overlay on the proxy `<video>` from per-frame JSON, synced with `requestVideoFrameCallback` (Chrome 83, Safari 15.4, Firefox 132; the callback can be one v-sync late). It costs no disk per run, toggles layers instantly, shows results while the job runs, and puts two runs on one video. Secondary is a baked "Export MP4" on demand.
- Serve media with `FileResponse` (HTTP Range since Starlette 0.39).

## Frontend
- **Routes**: `/` Workbench, `/runs` History, `/storage` Storage and models.
- **Workbench**: media rail (drop anywhere, paste), centre viewer, model picker with schema-generated sliders and cube table, strip of this media's runs. An upload is auto-selected and auto-run with the last model and preset.
- **Viewer** (one component for image and video): CSS-transform zoom about the cursor, pan, fit and 100 %, SVG overlay in media pixels, layer toggles (quads, centres, angle, labels, raw boxes, masks), pixel inspector (x, y, RGB, HSV) for threshold tuning, frame-step keys.
- **Compare** (same media, two runs): overlay mode (A solid, B dashed, diff table matched by nearest centre) and split mode with linked zoom and playhead.
- **History**: filter by media, model, status, tag, star; notes; bulk delete; "re-run with changes" prefills the Workbench and stores `parent_run_id`; save parameters as a preset.
- **Libraries**: the student's stack (React 19, Vite 7, TypeScript, react-query, react-router 7, lucide-react, Pretendard) with plain CSS; nothing else. npm latest is Vite 8.3, react-router 8.5, TypeScript 7.0.
- **Ports**: 5190 (web) and 8010 (API) were not listening when checked.

## Risks
1. **Disk exhaustion** (weights, swap, exports): pinned cached versions, explicit sized downloads, floor guard, disk bar from the first hour.
2. **Detections drawn on the wrong frame**: Playwright test in Chromium and WebKit with a synthetic clip whose content encodes the frame index.
3. **MPS failures** (unsupported operators, leaks, hangs): worker isolation, per-model device choice, warm-up excluded from timings, hard kill.
4. **Slider loop slow**: JSON-only preview, latest-wins with `AbortController`; measure the round trip first (target under 50 ms).
5. **Time** (four days to the competition): vertical slice first (image, HSV pipeline, overlay, saved run), then video, heavy models, compare. Adapter interface `param_schema / load / infer(frame, params)` makes a new model one file.

## Rejected
Heavy models in API threads (cannot be killed); Celery, Redis, WebSocket, ORM (one user, one-way progress, five tables); baked video or annotated image per run (megabytes each on a 5.7 GB disk).

Unverified: cv2 releasing the GIL, ffmpeg auto-rotation default, Gradio latency, whether `cv2.VideoWriter` can write H.264 from the pip wheel, gating of specific models.

Sources (read 2026-11-25): https://developer.mozilla.org/en-US/docs/Web/API/HTMLVideoElement/requestVideoFrameCallback ; https://caniuse.com/mdn-api_htmlvideoelement_requestvideoframecallback ; https://developer.mozilla.org/en-US/docs/Web/Media/Guides/Formats/Video_codecs ; https://caniuse.com/hevc ; https://fastapi.tiangolo.com/tutorial/server-sent-events/ ; https://starlette.dev/requests/ ; https://starlette.dev/responses/ ; https://starlette.dev/release-notes/ ; https://docs.astral.sh/uv/reference/settings/ ; https://docs.pytorch.org/docs/2.15/mps.html ; https://docs.python.org/3.12/library/multiprocessing.html ; https://docs.streamlit.io/get-started/fundamentals/main-concepts ; https://nicegui.io/documentation/interactive_image ; https://docs.roboflow.com/deployment/self-hosted/self-hosted ; versions and wheel sizes from https://pypi.org/pypi/<package>/json and https://registry.npmjs.org/<package>/latest
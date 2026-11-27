# Build brief — `frontend/` (React app)

Read `docs/briefs/common.md` and `docs/DESIGN.md` (§2 terms, §6 screens, §7 video) first.
You own `frontend/**` except `frontend/src/api/types.ts` (the contract: implement against it; only
additive changes mirrored in `backend/lab/schemas.py`). The backend is being built in parallel; build
against the contract and verify against the real backend when it is available (`cd backend &&
uv run lab --port 8010`), otherwise against a small fixture server or MSW-free fetch mocks inside
your own tests. Do not edit backend code.

## Skills to load before you start
- `impeccable` — use it to set the design direction and to critique your own screens (the user has
  asked for it on earlier projects). Then `anti-ui-slop` as a finish gate.
- `vercel-react-best-practices` — follow its rules for data fetching, re-renders and bundle size.
- `webapp-testing` — for the Playwright checks at the end.

## What the user asked for (their words, translated)
Test many vision models on images and videos; upload an image or video inside the site and test it
immediately; see the result image or video LARGE right away; manage the records of tests; keep the
data in the site so testing can continue. Everything else serves those five.

## Product shape
Routes (react-router 7, data in @tanstack/react-query, selection in the URL so reloads and
back/forward work):
1. `/` **작업대 (Workbench)** — `?media=<id>&run=<id>`.
   - Left rail (~260 px): upload button + media list (thumbnail, name, kind, resolution, duration or
     "이미지", run count, star), filter chips (전체 / 이미지 / 영상 / 별표), search by name.
     Drag-and-drop ANYWHERE on the page shows a full-page drop overlay; ⌘V / Ctrl+V pastes an image
     from the clipboard; multiple files upload in sequence with per-file progress (XMLHttpRequest
     upload progress). After upload the new media is selected and, if the current model is a
     pipeline, previewed at once. Empty library → a large drop zone plus a "합성 샘플 넣기" button
     (`POST /api/samples`).
   - Centre: the **viewer** (largest area on screen; see below) with, under it, a strip of this
     media's runs (thumbnail, model, status / progress bar, speed p50, star; click to show that
     run's result in the viewer; check two to open Compare).
   - Right panel (~360 px): model picker grouped "내장 파이프라인" / "내 가중치" with each model's
     summary; presets (select, save as, delete); parameter controls generated from `ParamSpec`
     grouped by `group` (collapsible), each with label, value, unit and the `help` line; "기본값"
     reset per group. Actions: for pipelines, live preview is automatic on every change (latest
     request wins: abort the previous fetch with AbortController) and **"기록으로 저장"** creates a
     run with the same params; for weights models **"실행"**. For videos: frame range + stride inputs
     ("전체 / 현재 구간", "N 프레임마다") and **"영상 전체 실행"**. Result section: counts per colour
     (chips), a table (색, 중심 x·y, 각도, 점수) where hovering/clicking a row highlights that cube in
     the viewer, and the time (ms) with per-stage times when present. Notes from the pipeline appear
     as small warnings.
   - Make the mode obvious: a badge on the viewer reads "미리보기 · 저장 안 됨" or "기록 #12 · 모델명".
     "파라미터 바꿔 다시 실행" on a run loads its model + params into the panel and remembers
     `parent_run_id` for the next saved run.
2. `/runs` **기록** — filter bar (미디어, 모델, 상태, 별표, 태그, 검색), list with 96 px annotated
   thumbnail, #id, media, model, changed params (only those different from the model defaults),
   cubes per frame, p50 ms, status, created time, star, note (inline edit), tags (inline add/remove);
   bulk select → delete / compare; open → Workbench with `?media&run`.
3. `/compare?a=<run>&b=<run>` **비교** — two viewers side by side with linked zoom, pan and video
   playhead; toggle "겹쳐 보기" (one viewer, A solid, B dashed). A diff panel: parameter differences,
   cubes per frame (A vs B), colour counts, timing. If the runs are on different media, say so and
   still show both.
4. `/models` **모델 · 저장공간** — built-in pipeline cards (name, summary, description, parameters),
   `.pt` upload (drag or button) with the inspection result (task, class names, size) or the
   Korean error, rename / note / delete; storage bar (free / total, per category) and cleanup
   actions with confirmation; worker status with "작업자 재시작".

## The viewer (one component for image and video, reused by Compare)
- The media element (img or video) and an SVG overlay share one transformed container
  (`transform: translate() scale()`), SVG `viewBox="0 0 W H"` in media pixels, strokes with
  `vector-effect: non-scaling-stroke`, labels kept at constant screen size.
- Zoom about the cursor with the wheel / pinch, drag to pan, buttons 맞춤 (fit) and 100 %, keyboard
  0 = fit, 1 = 100 %, F = fullscreen (Fullscreen API on the viewer element), and a zoom readout.
- Overlay layers (toggle menu, remembered in localStorage): 윗면 사각형, 중심·각도, 라벨(색 + 각도),
  원 출력(polygon / box), 검출 영역(ROI), 디버그 레이어 (select one of the preview's debug layers,
  opacity slider; masks are drawn tinted, not as raw black-and-white). Colour per cube class.
- Pixel inspector toggle: shows x, y, RGB, HSV (OpenCV scale) and Lab (real units, same formula as
  `cubevision/colour.py` — implement sRGB→Lab in TS and check a few values against Python) under the
  cursor, read from an offscreen canvas of the current image / paused video frame.
- Colour picking for `color` params: "이미지에서 고르기" → crosshair → click → median of a 5x5
  patch → hex.
- ROI editing for `roi` params: draw a polygon by clicking points (Enter or double-click closes,
  Esc cancels, Backspace removes the last point), choose 포함 / 제외, list with delete. Stored in
  the params, so presets and runs keep it.
- Video: a transport bar (play/pause, frame step ← → with Shift = 10 frames, frame number "f 123 /
  1800", time, scrubber with marks where the run has frames). Frame index of what is on screen =
  `Math.round(metadata.mediaTime * fps)` from `requestVideoFrameCallback` (fallback: currentTime).
  For a run, draw the stored frame with the largest index ≤ the current frame; show a small
  "처리 프레임 f N" note when it is not the exact frame. When paused and a pipeline is selected
  with no run shown, preview the current frame (`frame` in the preview request).
- Cube table ↔ overlay highlight in both directions.

## Visual direction
A focused tool, not a marketing page. The media is the hero: neutral dark viewer surface so colours
read true, quiet chrome, one accent colour, Pretendard, tabular numbers, dense but calm panels. Cube
class colours must be distinguishable on white boards and dark frames and must not depend on colour
alone (labels R/G/B/W). Let `impeccable` refine this; do not ship generic dashboard styling
(gradients, glass cards, giant rounded tiles, emoji). Target 1440x900 and 1280x800 laptops; the
layout must not break at 1180 px wide (collapse the side panels into toggles if needed). Keyboard
focus visible everywhere; buttons have accessible names.

## Data and state
- One small typed API client (`src/api/client.ts`) over `fetch`, errors surfaced with the backend's
  Korean `detail`. Uploads via XMLHttpRequest for progress.
- react-query keys per resource; poll runs every 1 s only while any visible run is queued/running;
  invalidate on mutations. Preview results are not cached across params.
- Remember the last model and its params per model in localStorage (try/catch around storage).
- Show a clear banner when the backend is unreachable ("서버에 연결할 수 없습니다 — ./run.sh 로
  서버를 켜세요").

## Done means
- `npm run typecheck` and `npm run build` pass with no warnings you introduced.
- Against the real backend: upload an image → preview with cubes drawn → change a slider and see
  the overlay update → save → run appears in the strip and on `/runs` → compare two runs; upload a
  video → run every 2nd frame → play it with the overlay following the video. Verify with Playwright
  (`frontend/tests/e2e/*.spec.ts`, config in `frontend/playwright.config.ts`, Chromium; it may
  start the backend itself with `--data` pointing at a temporary folder) and keep the specs.
- Screenshots of each screen saved to `docs/screenshots/` (1440x900).

## Ports while you work
Other agents work in parallel. For your own checks run the backend on 127.0.0.1:8012 with a tmp data
dir (`cd backend && uv run lab --port 8012 --data /tmp/...` once `lab` exists) and Vite on 5190 with
`LAB_API_URL=http://127.0.0.1:8012`. Stop both when you finish. Never touch 8010 or 8011. If the
backend is not ready yet, build the UI first and come back to the end-to-end checks later; if it is
still missing pieces at the end, report exactly which flows you could not verify.

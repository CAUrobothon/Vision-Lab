# Build brief — `backend/cubevision` (OpenCV pipelines, shared post-processing, synthetic scenes)

Read `docs/briefs/common.md` and `docs/DESIGN.md` first. You own `backend/cubevision/**`,
`backend/tests/cubevision/**` and `docs/PIPELINES.md`. Do not edit `backend/lab/**` or the frontend.
`cubevision` must stay free of web and torch imports: it will be imported by the competition program.

## Input you must read
`docs/research/classical-cv.json` — a design study done today for exactly this scene: geometry
notes, 3–5 candidate pipelines with steps / parameters / failure modes, a shared post-processing
design (top-face isolation, angle, splitting touching cubes), colour classification, and a capture
checklist. Treat it as a strong starting point, not gospel: re-derive anything that looks wrong and
say so in your report.

## Scene facts (from the rulebook and drawings)
- Camera 1280x720, about 55° diagonal field of view → fx ≈ fy ≈ 1410 px, looking straight down from
  601 mm above the board; principal point ≈ image centre. A 25 mm cube top face is ≈ 61 px wide
  (scale ≈ 2.45 px/mm at the top-face plane, 576 mm away).
- Perspective: the top face of a cube at horizontal distance r from the point under the camera is
  pushed outward, and the side faces that face the image centre show as a band ≈ 0.10·r px wide on the
  centre side of the top face (up to ≈ 30 px at the arena edge). Whether the coloured paper covers
  only the top or all faces is unknown — handle both. A blob = top face + side band; the top face is
  the part farthest from the principal point.
- Board: white PLA with 0.5 mm engraved zone grooves, a grey pillar at the top centre, probably
  printed position marks 1–20. Mission A: 9 cubes (3 R, 3 G, 3 B) + a growing tower (the top cube of
  a k-level tower looks larger by 576/(576 − 25k)). Mission B: 12 cubes (3 each R, G, B, wood) on
  the left half; the right half has three white baskets (32 mm walls) whose FLOORS are large R/G/B
  paper areas that must never be reported as cubes (by size/shape checks, and by the ROI the user
  draws). The robot arm may occlude or cast shadows. Printed colours ≠ pure RGB; lighting unknown.

## Deliverables
1. **Pipelines** (3 to 5, genuinely different ideas, keys `cv.<snake_name>`, never renamed later),
   each a `Pipeline` subclass registered in `cubevision/pipelines/__init__.py` (`ALL`, best first).
   Each has Korean `name`, `summary` (one line), `description` (idea · strengths · weak spots, 3–5
   lines), `params` (Korean labels, `help` lines, sensible groups such as "색 기준", "크기·모양",
   "전처리", "검출 영역"), and 2–4 informative debug layers when `debug=True` with Korean
   `debug_labels`. Every pipeline has a `roi` param (group "검출 영역") and applies it, and the
   colour-prototype params `proto_red`, `proto_green`, `proto_blue`, `proto_wood` (type `color`,
   defaults from `colour.DEFAULT_PROTOTYPES`) wherever colour is decided by prototypes. Expose a
   `face_px` param (expected top-face edge, default 61) where size matters. Fill `stages_ms` with
   the main stages. Put short warnings in `notes` (e.g. "큰 색 영역 2개를 바구니 바닥으로 보고 제외").
   Performance target: ≤ 30 ms p50 per 1280x720 frame on this M4 for the default params — measure.
2. **Shared post-processing** — improve the internals of `postprocess.py` (top-face isolation using
   `face_px` and the principal point, robust angle, splitting merged blobs of touching cubes by the
   known face area) and of `colour.py` (better decision rule, wood vs white board vs shadow) WITHOUT
   changing their public signatures; the lab's Ultralytics adapter calls them too. Keep `roi.py`
   and `draw.py` signatures.
3. **Synthetic scenes** `cubevision/synth.py`:
   - `make_scene(seed: int, *, mission: "A" | "B" = "A", size=(1280, 720), **variation) ->
     tuple[np.ndarray, list[dict]]` returns a BGR image and the ground truth per cube
     (`color`, top-face `quad`, `center`, `angle`). Render with the real camera geometry (pinhole,
     f ≈ 1410 px, 601 mm, cube 25 mm, side faces visible by perspective; paper on top only or on all
     faces as a variation), white board with grooves and marks, mission-B baskets with coloured
     floors, lighting gradient, soft shadows, sensor noise, slight blur, JPEG artefacts, optional
     touching cubes, a tower, an arm-like occluder. Deterministic per seed.
   - `sample_assets(out_dir: Path) -> list[Path]` writes 2 still images (mission A, mission B) as
     JPEG and one ≈ 5 s, 30 fps video (cubes being moved one at a time, written with
     `cv2.VideoWriter` mp4v — the lab transcodes it). Used by the lab's "합성 샘플 넣기" button.
   Synthetic data proves the plumbing and catches geometry bugs; it does NOT prove real-world
   accuracy. Say that plainly wherever you report numbers.
4. **Tests** (`backend/tests/cubevision/`, `cd backend && uv run pytest -q tests/cubevision`):
   contract tests (every registered pipeline: info() shape, params resolve, runs on a blank frame
   and on a scene, debug layers have labels and the frame's size, ROI respected), geometry unit
   tests (order_quad, quad_angle wrap-around near 0/90, splitting two touching cubes, top-face
   isolation on a scene with side bands), synth determinism, and an accuracy test per pipeline on a
   fixed seeded benchmark with thresholds you can justify.
5. **`docs/PIPELINES.md`** (Korean): per pipeline the idea, parameters that matter most and how to
   tune them on competition day (colour pick from a real cube, ROI over the baskets), and a measured
   table on the seeded synthetic benchmark: recall, false positives per image, colour accuracy,
   centre error px (mean / max), angle error ° (mean / max), p50 ms on this Mac — with the command
   that reproduces it (`uv run python -m cubevision.bench`), and a clearly marked "합성 데이터 기준,
   실제 사진 성능은 랩에서 확인 필요" caveat. Also fill DESIGN.md §9 with the list of pipelines
   (key, name, one line) — that one section of DESIGN.md is yours to edit.

## Priorities (time is short: competition in 4 days, the user reviews in the morning)
1. Shared post-processing stages 0–7 of `classical-cv.json` → `shared_postprocess` (gates, coarse
   angle via `cv2.boxPoints(cv2.minAreaRect(...))` — never read the rect's angle field —, splitting,
   anchored top face, colour, output). Stage 8 (video tracking) is OUT: pipelines are stateless.
   `angle_method` profile/grad4 and `refine=edges` are optional extras, only if time allows.
2. Pipelines in this order: `cv.hsv_ranges`, `cv.board_diff_lab`, `cv.edge_quads`,
   `cv.lab_autocluster` (k-means; stateless per frame, fixed RNG seed), then `cv.square_fit` only if
   the others are done, tested and benchmarked. Four solid pipelines beat five shaky ones.
3. Board-relative colour (measure the board's median Lab each frame) inside `colour.py`, with the
   prototypes still given as `#rrggbb` params (the lab's colour picker sets them from a real cube).
4. Optional extra output: add `"centre_base"` (the same cube projected to the board plane,
   `pp + (centre_top − pp)·(1 − 25/601)`) as a flag-free extra ONLY by putting it into `flags` is
   wrong — instead mention it in docs/PIPELINES.md as a formula for the calibration owner. Do not
   change the Cube contract.
Ports: you do not need servers. Never start anything on 8010–8012 or 5190.

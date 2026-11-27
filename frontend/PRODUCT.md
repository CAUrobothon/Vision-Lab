# Product

<!-- impeccable:product-schema 1 -->

> Inferred on 2026-11-25 from docs/DESIGN.md and docs/briefs/{common,frontend}.md while the user
> was unavailable (no question tool in the build session). Every line below restates those briefs;
> nothing was confirmed in an interview. Kept in frontend/ because the build agent owns only this folder.

## Platform

web

## Users
The vision lead of a 4-person student team competing in a robot-arm pick-and-place contest
(SO-101, 2026-11-29). Korean speaker. Works alone on a MacBook (Apple M4, 16 GB) in the four days
before the competition, tuning how a camera 601 mm above a white PLA board finds 25 mm cubes
(red / green / blue printed paper, plain wood).

## Product Purpose
A local, single-user lab to try many vision models on images and videos: upload a file inside the
site and test it at once, see the result image or video large right away, manage test records, and
keep the data so testing continues across sessions. Success = the user picks and tunes the cube
detector they will take to the competition.

## Positioning
Built for one scene: 1280x720 top-down frames where a cube top face is about 61x61 px. Built-in
models are OpenCV pipelines tuned with sliders and re-run on every change; uploaded Ultralytics
`.pt` checkpoints (detect / segment / obb) run in a worker. Results are cube lists in image pixels
(colour, top-face quad, centre, angle in [0, 90), score) drawn over the untouched media.

## Operating Context
Used at a desk and at the practice arena, next to the robot, often with real photos and short
videos from the competition camera. Two other teammates own calibration (pixel -> mm) and the
competition program, which later imports the same `cubevision` package.

## Capabilities and Constraints
- Local only, no login, no external hosting. All UI text is Korean.
- Visual comparison and processing time on every screen; since 2026-11-27 also scoring against a ground-truth JSON
  uploaded to a dataset (docs/SCORING.md: AI-assisted approximate labels, provisional tolerances, differences judged only
  by a paired bootstrap interval).
- Pixel coordinates only (x right, y down); HSV in OpenCV scale; Lab in real units.
- Frontend: React 19, Vite 7, TypeScript, react-router 7, react-query 5, lucide-react, Pretendard;
  plain CSS, no UI kits.
- Target laptops 1440x900 and 1280x800; must not break at 1180 px.

## Evidence on Hand
Synthetic sample scenes from `POST /api/samples` (clearly synthetic). No real competition photos are
in the repository; never fabricate results or accuracy claims.

## Product Principles
- The media is the hero: the result is seen large, with true colours, immediately.
- Every number on screen is the same number the code uses (units shown).
- Fast loop: a slider change re-draws the overlay within a frame or two; nothing is saved until asked.
- Records are cheap to make and easy to compare; nothing is lost on reload (state in the URL).

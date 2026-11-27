# Build brief — common rules (read first)

Project root: `/Users/limgiheon/Desktop/로보톤/vision-lab` (the folder name is Korean; quote paths in shell).
Read `docs/DESIGN.md` (Korean) for the product design. This file and the per-area briefs are the
engineering contract. If a brief and DESIGN.md disagree, the brief wins; report the conflict.

## Who and what
- User: the vision lead of a 4-person student team in a robot-arm pick-and-place competition on
  2026-11-29 (today is 2026-11-25). They speak Korean; every user-visible string is Korean.
- The lab is a LOCAL single-user web app on their MacBook (Apple M4, 16 GB RAM, macOS 15.7,
  MPS GPU). It is used to try vision models on images and videos of 25 mm cubes (red / green /
  blue printed paper, plus plain wood) on a white PLA board, seen from a camera 601 mm straight
  above (1280x720, a cube top face is about 61x61 px). See DESIGN.md §1 for the user's decisions:
  built-in models are OpenCV pipelines only; uploaded Ultralytics `.pt` checkpoints can be tested;
  no ground-truth scoring (visual comparison + speed only); pixel coordinates only.

## Hard rules
- Stay inside the project root. Do not touch `/Users/limgiheon/Desktop/캡스톤` or anything else.
- Do not `git commit`, push or create branches. The user commits.
- Do not install new Python packages unless unavoidable. The environment is `backend/.venv`
  (uv, Python 3.12) pinned to cached versions in `backend/pyproject.toml`. Never add
  `opencv-python-headless` (it overwrites `opencv-python`). If you truly need a package, add it to
  pyproject with an exact pin and run `uv sync`, and say so in your report.
- Frontend packages: React 19, Vite 7, TypeScript 5.9, react-router 7, @tanstack/react-query 5,
  lucide-react, pretendard (already installed with npm in `frontend/`). No UI kits, no CSS
  frameworks, no state libraries. Ask yourself before adding anything; prefer none.
- Never download model weights or datasets. Tests must run offline.
- Ports: backend 127.0.0.1:8010, Vite dev 127.0.0.1:5190. Ports 5000, 5180, 7000, 8000, 8001 belong
  to other programs — never kill processes you did not start. Stop servers you start before you finish.
- The machine has only 16 GB RAM and is already swapping: never run two torch processes at once,
  keep test media small, and kill stray processes you spawned.
- Commands: Python via `cd backend && uv run ...`; frontend via `cd frontend && npm run ...`.

## Contracts you must not break (owned by the orchestrator)
- `backend/cubevision/types.py` — Cube, FrameResult, ParamSpec, resolve_params, conventions.
- `backend/cubevision/pipelines/base.py` — Pipeline interface (incl. `debug_labels`).
- `backend/cubevision/registry.py` — list_pipelines(), get_pipeline(key).
- Public signatures in `backend/cubevision/{colour,postprocess,roi,draw}.py` (see their docstrings).
  Internals may be improved; signatures and return shapes may not change.
- `backend/lab/schemas.py` and `frontend/src/api/types.ts` — the HTTP contract (kept in sync).
If you think a contract is wrong, implement around it and explain the problem in your report
instead of silently changing it. Additive, backward-compatible fields are allowed if you update
BOTH schemas.py and types.ts.

## Units used everywhere (UI shows the same numbers the code uses)
- Pixel coordinates: x right, y down, origin top-left. Angles: degrees in [0, 90) from +x towards +y.
- HSV: OpenCV 8-bit scale (H 0–179, S 0–255, V 0–255).
- Lab: real units (L 0–100, a and b about −128..127), converted as in `cubevision/colour.py`.
- Colours in params: `"#rrggbb"` sRGB hex.
- Times: milliseconds, wall clock (`time.perf_counter`), first frame reported separately (warm-up).

## Quality bar
- Code is read by a student who will reuse `cubevision` in the competition program: clear names,
  short docstrings that say why, no dead code, no speculative abstractions.
- Every claim in your final report must come from something you ran. Report what you could not
  verify. Keep the report short: what you built (files), how you verified it (commands + results),
  known gaps.

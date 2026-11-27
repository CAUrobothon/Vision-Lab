#!/usr/bin/env bash
# Vision Lab: build the web app if needed, then start the server.
#   ./run.sh                 http://127.0.0.1:8010, data in ./data
#   ./run.sh --data ~/lab    any extra arguments go to `uv run lab`
#   PORT=8020 ./run.sh       another port
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST="127.0.0.1"
PORT="${PORT:-8010}"
DIST="$ROOT/frontend/dist/index.html"

for tool in uv ffmpeg ffprobe; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "필요한 프로그램이 없습니다: $tool" >&2
    exit 1
  fi
done

# Rebuild the frontend when dist is missing or any source file is newer than the last build.
if [ ! -f "$DIST" ] || [ -n "$(find "$ROOT/frontend/src" "$ROOT/frontend/index.html" "$ROOT/frontend/package.json" \
    "$ROOT/frontend/vite.config.ts" -newer "$DIST" -print -quit 2>/dev/null)" ]; then
  if ! command -v npm >/dev/null 2>&1; then
    echo "npm 이 없어 화면을 빌드할 수 없습니다 (API 만 실행합니다)." >&2
  else
    echo "화면(frontend)을 빌드합니다..."
    if [ ! -d "$ROOT/frontend/node_modules" ]; then
      (cd "$ROOT/frontend" && npm install)
    fi
    (cd "$ROOT/frontend" && npm run build)
  fi
fi

if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "포트 $PORT 를 이미 다른 프로그램이 쓰고 있습니다. 이미 켜져 있다면 http://$HOST:$PORT 를 여세요." >&2
  echo "다른 포트로 실행하려면: PORT=8020 ./run.sh" >&2
  exit 1
fi

echo "Vision Lab → http://$HOST:$PORT   (끄려면 Ctrl+C)"
cd "$ROOT/backend"
exec uv run lab --host "$HOST" --port "$PORT" "$@"

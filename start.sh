#!/usr/bin/env sh
# One command for macOS/Linux: create .venv on first use, install requirements, then start the app.
#   sh start.sh               (demo DRES, port 8000)
#   sh start.sh --real-dres   (DRES from .env)
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python
if [ ! -x "$PY" ]; then
  echo "Creating .venv (Python 3.13)..."
  if command -v python3.13 >/dev/null 2>&1; then python3.13 -m venv .venv; else python3 -m venv .venv; fi
fi
STAMP=$( (sha256sum requirements.txt 2>/dev/null || shasum -a 256 requirements.txt) | cut -d' ' -f1)
if [ ! -f .venv/.requirements-installed ] || [ "$(cat .venv/.requirements-installed)" != "$STAMP" ]; then
  echo "Installing requirements (first run takes a while)..."
  "$PY" -m pip install --upgrade pip
  "$PY" -m pip install -r requirements.txt
  echo "$STAMP" > .venv/.requirements-installed
fi
if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env from .env.example. Add your Gemini keys there to enable QA and Gemini checks."
fi
PYTHONUTF8=1 exec "$PY" tools/start_local.py "$@"

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"

cd "$ROOT"

if [[ ! -x ".venv/bin/python" ]]; then
  "$PYTHON" -m venv .venv
fi

.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -r requirements-cuda.txt

echo "Downloading/loading the Whisper model once. This can take a while..."
.venv/bin/python dictate_hold.py --hotkey f13 --model large-v3-turbo --device cuda --compute-type float16 --language auto --download-only

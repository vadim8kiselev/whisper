#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

exec .venv/bin/python dictate_hold.py --hotkey f13 --model large-v3-turbo --device cpu --compute-type int8 --language auto

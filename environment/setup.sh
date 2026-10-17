#!/usr/bin/env bash
# ============================================================
#  LINGORA environment setup (offline, no network needed)
#  Only dependency: jieba (installed from local wheel)
# ============================================================
set -e
cd "$(dirname "$0")"

echo "[1/3] Creating isolated venv .venv ..."
python3 -m venv .venv || {
    echo "[ERROR] python3 not found. Install Python 3.10+ first."
    exit 1
}

source .venv/bin/activate

echo "[2/3] Installing jieba offline (local wheel)..."
python -m pip install --no-index --find-links=. jieba==0.42.1

echo "[3/3] Done."
echo "Run LINGORA from the project root:"
echo "    environment/.venv/bin/python main.py"

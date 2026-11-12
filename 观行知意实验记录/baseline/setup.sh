#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
"${PYTHON:-python3.10}" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python check_environment.py

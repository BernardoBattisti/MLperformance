#!/usr/bin/env sh
# Runs MLperformance from source on Linux/macOS. Creates the virtual environment on first run.
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install -r requirements.txt
fi
exec .venv/bin/python MLperformance.py

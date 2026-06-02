#!/usr/bin/env bash
# Double-click launcher for macOS.
#
# What it does, in plain terms:
#   1. Finds the project folder (the folder this file lives in, one level up).
#   2. Makes sure a Python environment with the toolbox installed exists
#      (creates one the first time, which can take a few minutes).
#   3. Opens the graphical EEG/DBS synchronization window.
#
# A clinician should be able to just double-click this file in Finder.
# (The first run may ask macOS for permission to run a downloaded script.)

set -e

# Resolve the directory of this script, then the project root (its parent).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

VENV_DIR="${PROJECT_DIR}/.venv"
PYTHON="${VENV_DIR}/bin/python"

# Pick a system Python to bootstrap the environment if needed.
if command -v python3 >/dev/null 2>&1; then
  SYS_PYTHON="python3"
elif command -v python >/dev/null 2>&1; then
  SYS_PYTHON="python"
else
  osascript -e 'display dialog "Python 3 was not found on this Mac.\n\nPlease install Python 3.12+ from https://www.python.org/downloads/ and try again." buttons {"OK"} with icon stop' >/dev/null 2>&1 || true
  echo "Python 3 not found. Install it from https://www.python.org/downloads/"
  exit 1
fi

# Create the environment and install the toolbox (with GUI extras) on first run.
if [ ! -x "${PYTHON}" ]; then
  echo "First-time setup: creating a Python environment (this may take a few minutes)…"
  "${SYS_PYTHON}" -m venv "${VENV_DIR}"
  "${PYTHON}" -m pip install --upgrade pip
  "${PYTHON}" -m pip install -e ".[gui]"
fi

# Make sure the GUI front-end is importable; install GUI extras if missing.
if ! "${PYTHON}" -c "import PyQt6" >/dev/null 2>&1; then
  echo "Installing the graphical-interface components…"
  "${PYTHON}" -m pip install -e ".[gui]"
fi

# Launch the graphical window.
exec "${PYTHON}" -m dbs_eeg_sync

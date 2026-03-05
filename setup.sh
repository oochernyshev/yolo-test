#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Error: Python not found: $PYTHON_BIN"
  echo "Tip: run with a specific interpreter, e.g. PYTHON_BIN=python3.11 ./setup.sh"
  exit 1
fi

cd "$PROJECT_DIR"

if [[ ! -d ".venv" ]]; then
  echo "Creating virtual environment in .venv"
  "$PYTHON_BIN" -m venv .venv
else
  echo "Using existing virtual environment: .venv"
fi

source .venv/bin/activate

echo "Upgrading pip tooling"
python -m pip install --upgrade pip setuptools wheel

echo "Installing dependencies"
python -m pip install -r requirements.txt

echo
echo "Setup complete."
echo "Activate env: source .venv/bin/activate"
echo "Run app:      python -m streamlit run app.py"
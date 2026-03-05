#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$PROJECT_DIR"

if [[ ! -f ".venv/bin/activate" ]]; then
  echo "Error: local virtual environment not found at .venv"
  echo "Run ./setup.sh first"
  exit 1
fi

source .venv/bin/activate

exec python -m streamlit run app.py "$@"
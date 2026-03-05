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

# Run Streamlit as a managed background job and forward a gentle TERM
# on Ctrl+C to avoid noisy shutdown traces.
set +e
(trap '' INT; exec python run_streamlit.py "$@") &
STREAMLIT_PID=$!

shutdown() {
  if kill -0 "$STREAMLIT_PID" 2>/dev/null; then
    echo "Stopping Streamlit..."
    kill -TERM "$STREAMLIT_PID" 2>/dev/null || true
    wait "$STREAMLIT_PID" 2>/dev/null || true
  fi
  exit 0
}

trap shutdown INT TERM

wait "$STREAMLIT_PID"
EXIT_CODE=$?
trap - INT TERM
exit "$EXIT_CODE"
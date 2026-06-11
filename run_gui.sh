#!/usr/bin/env bash
# Launch the interactive photo-to-mesh GUI. Open http://localhost:7860 in a browser.
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# shellcheck disable=SC1091
source .venv/bin/activate
exec python app.py

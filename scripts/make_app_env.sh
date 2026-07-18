#!/usr/bin/env bash
# Build the consolidated `tem-app` env for the desktop app (segment + review +
# export in one napari window). Reproduces the validated stack:
# Python 3.11 / numpy 1.26 / PyQt6 / napari + npsam (docs/batch_design.md).
#
#     bash scripts/make_app_env.sh
#     conda run -n tem-app python gui/napari_app.py
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"

echo "[1/2] creating the tem-app conda env (this downloads torch/napari — a few minutes)…"
conda env create -f "$HERE/environment.yml"

echo "[2/2] pre-fetching FastSAM weights (else the first tier-2 run prompts to download)…"
printf 'y\n' | conda run -n tem-app python -c \
  "from npsam.utils import download_weights; download_weights('fast')" \
  && echo "  weights ready." \
  || echo "  (prefetch skipped; the first tier-2 run will offer to download them)"

echo
echo "done. launch the app with:"
echo "    conda run -n tem-app python $HERE/gui/napari_app.py"

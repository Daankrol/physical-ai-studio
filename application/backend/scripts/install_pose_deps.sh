#!/bin/bash
set -euo pipefail
# -----------------------------------------------------------------------------
# install_pose_deps.sh - Install the MediaPipe pose-estimation backend.
#
# Not a pyproject.toml dependency: mediapipe's PyPI metadata pulls in
# opencv-contrib-python, which collides with this project's opencv-python /
# opencv-python-headless (all three ship files at the same `cv2/` import
# path — whichever installs last wins, silently breaking the others). The
# task-vision API this project actually uses does not need cv2 at all, so we
# install mediapipe with --no-deps and its real runtime dependencies by hand.
#
# Run this once after `uv sync`, from application/backend/.
#
# A plain `uv sync` (without --inexact) will not remove these packages since
# they are not project dependencies, but re-run this script if you ever
# reinstall opencv-contrib-python by mistake.
# -----------------------------------------------------------------------------

cd "$(dirname "${BASH_SOURCE[0]}")/.."

uv pip install --python .venv/bin/python --no-deps mediapipe
uv pip install --python .venv/bin/python matplotlib absl-py sounddevice flatbuffers

.venv/bin/python -c "
import cv2
import mediapipe
import torch
print('cv2:', cv2.__version__)
print('mediapipe:', mediapipe.__version__)
print('torch:', torch.__version__, 'cuda available:', torch.cuda.is_available())
"

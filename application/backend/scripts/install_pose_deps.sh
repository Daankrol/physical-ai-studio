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
#
# mediapipe's compiled task-vision library also needs a real OpenGL ES
# library at runtime, which this pip package does not provide. On Debian/
# Ubuntu it's the `libgles2` package (`libGLESv2.so.2`); without it,
# constructing a PoseLandmarker fails with:
#   OSError: libGLESv2.so.2: cannot open shared object file: No such file or directory
# On other platforms, install whatever OS package provides an OpenGL ES 2
# ICD/dispatch library. macOS ships this as part of the system frameworks, so
# it should not need a separate install there; if it does fail, the error
# will look similar (a dynamic-library load failure), just for a .dylib.
# -----------------------------------------------------------------------------

cd "$(dirname "${BASH_SOURCE[0]}")/.."

uv pip install --python .venv/bin/python --no-deps mediapipe
uv pip install --python .venv/bin/python matplotlib absl-py sounddevice flatbuffers

if command -v apt-get >/dev/null 2>&1 && ! ldconfig -p 2>/dev/null | grep -q libGLESv2.so.2; then
	echo "Installing libgles2 (provides libGLESv2.so.2, required by mediapipe's native library)..."
	sudo apt-get install -y libgles2
fi

echo "Verifying mediapipe can actually construct a PoseLandmarker (not just import)..."
.venv/bin/python -c "
import cv2
import torch
print('cv2:', cv2.__version__)
print('torch:', torch.__version__, 'cuda available:', torch.cuda.is_available())

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
import tempfile, urllib.request

with tempfile.NamedTemporaryFile(suffix='.task') as model_file:
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/mediapipe-models/pose_landmarker/'
        'pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
        model_file.name,
    )
    landmarker = PoseLandmarker.create_from_options(
        PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_file.name),
            running_mode=RunningMode.VIDEO,
        )
    )
    landmarker.close()
print('mediapipe pose landmarker loads correctly.')
"

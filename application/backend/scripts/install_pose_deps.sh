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
# Linux: mediapipe's compiled task-vision library also needs a real OpenGL ES
# library at runtime, which the pip package does not provide. On Debian/
# Ubuntu that's the `libgles2` package (`libGLESv2.so.2`); without it,
# constructing a PoseLandmarker fails with:
#   OSError: libGLESv2.so.2: cannot open shared object file: No such file or directory
# On other Linux distros, install whatever OS package provides an OpenGL ES 2
# ICD/dispatch library.
#
# macOS: the latest mediapipe (1.0.x) has a known upstream regression where
# every Tasks Vision graph with a detector/landmarker calculator (which
# includes PoseLandmarker) SIGABRTs during graph construction, unconditionally,
# even with delegate=CPU:
#   F0000 graph_service.h:139] Check failed: service_ Service is unavailable.
#       @ -[DrishtiMetalHelper initWithCalculatorContext:]
#       @ mediapipe::api2::TensorsToDetectionsCalculator::Open()
# https://github.com/google-ai-edge/mediapipe/issues/6356 — the calculator
# unconditionally builds a Metal GPU helper that a CPU-only graph never
# registers a service for. Not something this project's code can work around;
# the confirmed-working release is mediapipe==0.10.21, which only ships
# wheels up to Python 3.12 (no cp313 build). If this venv is Python 3.13 on
# macOS, pose estimation cannot be installed until either mediapipe fixes the
# regression or you recreate the venv on Python 3.12 (`uv venv --python 3.12`).
#
# mediapipe==0.10.21's package __init__ unconditionally imports its older
# "solutions" API, which needs a working `cv2` — unlike 1.0.x's task-vision
# path, which needs no cv2 at all. This project's own `opencv-python`
# dependency (already installed by `uv sync`) satisfies that import; nothing
# extra to install here. Only skip `--no-deps` mediapipe pulling in a SECOND,
# conflicting `opencv-contrib-python`, which is exactly what --no-deps avoids.
# -----------------------------------------------------------------------------

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ "$(uname -s)" == "Darwin" ]]; then
	mediapipe_spec="mediapipe==0.10.21"
	python_minor="$(.venv/bin/python -c 'import sys; print(sys.version_info[1])')"
	if [[ "${python_minor}" -ge 13 ]]; then
		echo "mediapipe==0.10.21 (the macOS-working release, see the comment above) has no" >&2
		echo "wheel for Python 3.1${python_minor}. Recreate this venv on Python 3.12 first:" >&2
		echo "  uv venv --python 3.12 && uv sync --extra cpu --extra tests" >&2
		exit 1
	fi
else
	mediapipe_spec="mediapipe"
fi

uv pip install --python .venv/bin/python --no-deps "${mediapipe_spec}"
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

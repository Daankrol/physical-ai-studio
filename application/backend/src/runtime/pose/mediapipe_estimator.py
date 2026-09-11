"""MediaPipe BlazePose GHUM 3D estimator.

Monocular 3D: depth is inferred from a single RGB frame by a learned body
model, not measured. No depth camera required. Absolute scale is
approximate, but the retargeter only reads vector angles between landmarks,
which are scale-invariant, so this is fine for teleoperation.

Not a pyproject.toml dependency — see
``application/backend/scripts/install_pose_deps.sh`` for why and how to
install it.
"""

from __future__ import annotations

import urllib.request
from typing import TYPE_CHECKING

from loguru import logger

from runtime.pose.estimator import PoseEstimate

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np

_MODEL_FILENAME = "pose_landmarker_lite.task"
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
# Milliseconds must strictly increase between VIDEO-mode calls; the runtime
# loop's own frame timestamps are monotonic-clock floats, not milliseconds,
# so this counts synthetic timestamps instead of converting them.
_TIMESTAMP_STEP_MS = 33


def default_model_path(cache_dir: Path) -> Path:
    """Return the cached model path, downloading it on first use."""
    path = cache_dir / _MODEL_FILENAME
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading MediaPipe pose model to {}", path)
        tmp_path = path.with_suffix(".tmp")
        urllib.request.urlretrieve(_MODEL_URL, tmp_path)  # noqa: S310 - fixed https URL, not user input
        tmp_path.rename(path)
    return path


class MediaPipePoseEstimator:
    """Wraps MediaPipe's ``PoseLandmarker`` in ``VIDEO`` running mode."""

    def __init__(self, *, model_path: Path) -> None:
        # Imported lazily: mediapipe is not a project dependency (see module
        # docstring), so importing it must not happen unless a session
        # actually configures a pose teleoperator.
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.VIDEO,
            num_poses=1,
        )
        self._landmarker = PoseLandmarker.create_from_options(options)
        self._timestamp_ms = 0

    def estimate(self, frame_rgb: np.ndarray) -> PoseEstimate | None:
        import mediapipe as mp

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        self._timestamp_ms += _TIMESTAMP_STEP_MS
        result = self._landmarker.detect_for_video(image, self._timestamp_ms)
        if not result.pose_world_landmarks or not result.pose_landmarks:
            return None
        return PoseEstimate(
            world_landmarks=[
                (landmark.x, landmark.y, landmark.z, landmark.visibility) for landmark in result.pose_world_landmarks[0]
            ],
            image_landmarks=[
                (landmark.x, landmark.y, landmark.z, landmark.visibility) for landmark in result.pose_landmarks[0]
            ],
        )

    def close(self) -> None:
        self._landmarker.close()

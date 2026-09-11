"""Backend human-pose estimation: camera frame -> 3D landmarks -> joint angles.

Runs inside the runtime session process, reading a camera the session already
subscribes to (no second camera subscriber). ``PoseEstimator`` is a narrow
protocol so the concrete model (MediaPipe today) stays swappable.
"""

from .estimator import Landmark, PoseEstimator
from .mediapipe_estimator import MediaPipePoseEstimator
from .worker import PoseWorker

__all__ = ["Landmark", "MediaPipePoseEstimator", "PoseEstimator", "PoseWorker"]

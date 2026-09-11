"""The narrow interface a pose estimation backend implements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import numpy as np

# (x, y, z, visibility) in the estimator's own world-space convention. The
# retargeter (``runtime.pose_retarget``) only reads ratios and vector angles
# between these, never absolute position, so a scale-inconsistent monocular
# 3D lift is good enough here even though it is not metrically accurate.
Landmark = tuple[float, float, float, float]


@dataclass(frozen=True)
class PoseEstimate:
    """One detection: two landmark sets serving two different consumers.

    ``world_landmarks`` are metric-ish 3D (hip-centered), for the joint-angle
    retargeter. ``image_landmarks`` are normalized ``[0, 1]`` image-space
    coordinates, for drawing a skeleton overlay on the camera feed. These are
    not interchangeable — a 3D world position is not a screen position.
    """

    world_landmarks: list[Landmark]
    image_landmarks: list[Landmark]


class PoseEstimator(Protocol):
    """One human, one frame in, landmarks out (or ``None``)."""

    def estimate(self, frame_rgb: np.ndarray) -> PoseEstimate | None:
        """Return landmarks for the most prominent detected person."""

    def close(self) -> None:
        """Release the estimator's resources. Idempotent."""

"""Runs a ``PoseEstimator`` on the newest camera frame off the control thread.

Mirrors the decoupling ``physicalai.runtime.AsyncExecution`` already does for
policy inference: the 30 Hz control loop must never block on a model that
can take longer than one tick. A single-slot mailbox in, a single-slot
result out; frames the worker cannot keep up with are simply skipped
rather than queued.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    import numpy as np

    from runtime.pose.estimator import Landmark, PoseEstimator


class PoseWorker:
    """Background thread that keeps re-running estimation on the latest frame."""

    def __init__(self, estimator: PoseEstimator) -> None:
        self._estimator = estimator
        self._lock = threading.Lock()
        self._pending: tuple[np.ndarray, int] | None = None
        self._latest_world_landmarks: list[Landmark] | None = None
        self._latest_image_landmarks: list[Landmark] | None = None
        self._latest_landmarks_at: float | None = None
        self._last_processed_sequence: int | None = None
        self._error_logged = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="pose-worker", daemon=True)
        self._thread.start()

    def submit_frame(self, frame_rgb: np.ndarray, sequence: int) -> None:
        """Replace the pending frame. Cheap; called from the control thread every tick."""
        with self._lock:
            if sequence == self._last_processed_sequence:
                return
            self._pending = (frame_rgb, sequence)

    def latest_landmarks(self) -> list[Landmark] | None:
        """Return the most recent metric-ish 3D world landmarks, for retargeting."""
        with self._lock:
            return self._latest_world_landmarks

    def latest_overlay_landmarks(self) -> list[Landmark] | None:
        """Return the most recent normalized image-space landmarks, for a UI overlay."""
        with self._lock:
            return self._latest_image_landmarks

    def seconds_since_update(self) -> float | None:
        """Return how long ago landmarks last changed, or ``None`` before the first detection."""
        with self._lock:
            if self._latest_landmarks_at is None:
                return None
            return time.monotonic() - self._latest_landmarks_at

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._estimator.close()

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                pending = self._pending
                self._pending = None
            if pending is None:
                self._stop.wait(0.005)
                continue
            frame_rgb, sequence = pending
            try:
                estimate = self._estimator.estimate(frame_rgb)
            except Exception as exc:
                if not self._error_logged:
                    logger.warning("Pose estimation failed; holding the last landmarks: {}", exc)
                    self._error_logged = True
                estimate = None
            else:
                self._error_logged = False
            with self._lock:
                self._last_processed_sequence = sequence
                if estimate is not None:
                    self._latest_world_landmarks = estimate.world_landmarks
                    self._latest_image_landmarks = estimate.image_landmarks
                    self._latest_landmarks_at = time.monotonic()

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Action-chunking inference runner (decorator).

Wraps any ``InferenceRunner`` to add temporal action buffering.  The inner
runner produces an output dict whose action value has shape
``(batch, horizon, action_dim)``.  This wrapper queues the individual
timesteps, dispensing one per call.  Only invokes the inner runner again
when the queue is exhausted.

This is the GoF Decorator pattern: ``ActionChunking`` *is* an
``InferenceRunner`` and *has* an ``InferenceRunner``.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import numpy as np

from physicalai.inference.constants import ACTION
from physicalai.inference.runners.base import InferenceRunner

if TYPE_CHECKING:
    from physicalai.inference.adapters.base import RuntimeAdapter


class ActionChunking(InferenceRunner):
    """Wrap a runner with temporal action buffering.

    On the first call (or when the queue is empty), delegates to the
    inner runner which returns an output dict containing an action with
    shape ``(batch, chunk_size, action_dim)``.  All chunk steps are
    enqueued and one is returned.  Subsequent calls pop from the queue
    without running inference.

    When ``rtc_max_delay > 0``, enables Training-Time Real-Time Chunking
    (TT-RTC): inference is triggered early — when the queue has
    ``rtc_max_delay`` or fewer actions remaining — and the remaining
    actions are passed as ``action_prefix`` to condition the next chunk.
    Only the newly generated postfix actions are enqueued, avoiding
    duplicate execution of prefix actions.

    Args:
        runner: The inner runner to delegate inference to.
        chunk_size: Number of actions per chunk.  Must match the inner
            runner's output temporal dimension.
        action_key: Key in the runner output dict that holds the action
            tensor.  Defaults to ``"action"``.
        rtc_max_delay: Maximum prefix length for TT-RTC conditioning.
            0 disables TT-RTC (default).
        action_dim: Action dimension (required when ``rtc_max_delay > 0``
            to construct the padded prefix tensor).

    Examples:
        Standard action chunking (no RTC):

        >>> runner = ActionChunking(SinglePass(), chunk_size=10)
        >>> outputs = runner.run(adapter, inputs)

        With TT-RTC conditioning:

        >>> runner = ActionChunking(SinglePass(), chunk_size=50, rtc_max_delay=10, action_dim=7)
    """

    def __init__(
        self,
        runner: InferenceRunner,
        chunk_size: int = 1,
        action_key: str = ACTION,
        rtc_max_delay: int = 0,
        action_dim: int = 0,
    ) -> None:
        self.runner = runner
        self.chunk_size = chunk_size
        self.action_key = action_key
        self.rtc_max_delay = rtc_max_delay
        self.action_dim = action_dim
        self._action_queue: deque[np.ndarray] = deque()

    def run(
        self,
        adapter: RuntimeAdapter,
        inputs: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Return the next action, running inference when the queue runs low.

        Without TT-RTC (``rtc_max_delay=0``): inference runs when the
        queue is empty.

        With TT-RTC: inference runs when the queue has
        ``rtc_max_delay`` or fewer items remaining.  The remaining
        actions are injected as ``action_prefix`` and ``delay`` into
        the model inputs so the denoiser conditions on them.

        .. note::

            Currently synchronous — the call blocks while inference
            runs.  A future optimization could trigger inference
            asynchronously (e.g. in a background thread) when the
            queue crosses the delay threshold, so the robot keeps
            executing queued actions while the next chunk is computed.

        Args:
            adapter: The loaded runtime adapter.
            inputs: Pre-processed model inputs.

        Returns:
            Output dict with a single action of shape
            ``(batch_size, action_dim)``.
        """
        if self.rtc_max_delay > 0:
            return self._run_with_rtc(adapter, inputs)

        if len(self._action_queue) > 0:
            return {self.action_key: self._action_queue.popleft()}

        outputs = self.runner.run(adapter, inputs)
        actions = outputs[self.action_key]

        batch_actions = np.transpose(actions, (1, 0, 2))
        self._action_queue.extend(batch_actions)

        return {self.action_key: self._action_queue.popleft()}

    def _run_with_rtc(
        self,
        adapter: RuntimeAdapter,
        inputs: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        queue_len = len(self._action_queue)

        if queue_len > self.rtc_max_delay:
            return {self.action_key: self._action_queue.popleft()}

        delay = queue_len

        if delay > 0:
            remaining = list(self._action_queue)
            # remaining is list of (batch, action_dim) arrays
            prefix_actions = np.stack(remaining, axis=0)  # (delay, batch, action_dim)
            prefix_actions = np.transpose(prefix_actions, (1, 0, 2))  # (batch, delay, action_dim)
            batch_size = prefix_actions.shape[0]

            # Pad to full chunk_size along temporal axis
            pad_width = self.chunk_size - delay
            action_prefix = np.pad(
                prefix_actions,
                ((0, 0), (0, pad_width), (0, 0)),
                mode="constant",
                constant_values=0.0,
            )  # (batch, chunk_size, action_dim)

            inputs = {**inputs, "action_prefix": action_prefix, "delay": np.array(delay)}
        outputs = self.runner.run(adapter, inputs)
        actions = outputs[self.action_key]  # (batch, chunk_size, action_dim)

        # Discard prefix echo, keep only postfix
        postfix_actions = actions[:, delay:]  # (batch, chunk_size - delay, action_dim)
        batch_actions = np.transpose(postfix_actions, (1, 0, 2))

        self._action_queue.clear()
        self._action_queue.extend(batch_actions)

        return {self.action_key: self._action_queue.popleft()}

    def reset(self) -> None:
        """Clear the action queue and reset the inner runner."""
        self._action_queue.clear()
        self.runner.reset()

    def __repr__(self) -> str:
        """Return string representation of the runner."""
        return f"{self.__class__.__name__}(runner={self.runner!r}, chunk_size={self.chunk_size})"

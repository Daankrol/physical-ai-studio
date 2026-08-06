# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base torch nn.Module for Models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn

from physicalai.data.observation import EXTRA


class Model(nn.Module, ABC):
    """Base class for Models.

    Model is an entity that is fully compatible with torch.nn.Module,
    and is used to define the architecture of the neural network inside Policy.

    Subclasses must implement:

    - ``forward(batch)``: standard PyTorch forward pass.  In training mode it
      should return ``(loss, loss_dict)``; in eval mode it should return
      predicted actions.
    - ``compute_loss(batch)``: compute the **training** loss (with gradients).
      Called by ``forward()`` when ``self.training`` is ``True``.
    - ``compute_val_loss(batch)``: compute the **validation** loss (no
      gradients).  Override this to use a more meaningful metric than the
      training loss (e.g. action prediction MSE for diffusion / flow-matching
      models).  The default falls back to ``compute_loss``.
    """

    @abstractmethod
    def compute_loss(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
        """Compute the training loss for this model.

        Args:
            batch: Preprocessed batch dict.

        Returns:
            Tuple of (loss tensor with grad, dict with at least a ``"loss"`` key).
            Values in the dict should generally be left as (detached) tensors
            rather than converted to Python scalars via ``.item()``: when the
            model's ``forward`` is wrapped with ``torch.compile``, calling
            ``.item()`` inside this method forces a host sync and breaks the
            compiled graph. The enclosing ``Policy`` (a ``LightningModule``)
            logs these values via ``self.log(...)``, which accepts tensors
            directly.
        """

    @torch.no_grad()
    def compute_val_loss(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
        """Compute the validation loss for this model.

        Override in subclasses to use a different metric from the training
        loss (e.g. action prediction MSE after full denoising).  The default
        delegates to :meth:`compute_loss`.

        Args:
            batch: Preprocessed batch dict.

        Returns:
            Tuple of (loss tensor, dict with at least a ``"loss"`` key).
        """
        return self.compute_loss(batch)

    @staticmethod
    def in_episode_bound(batch: dict[str, Any], exempt_idx: torch.Tensor | None = None) -> torch.Tensor | None:
        """Build the mask of action steps that carry real supervision.

        LeRobot clamps action-chunk queries at episode boundaries, repeating the
        terminal action to fill the chunk and flagging the clamped steps as
        ``action_is_pad``.  Those steps must not supervise the policy, otherwise
        the tail of every episode trains towards a frozen pose.

        Args:
            batch: Preprocessed batch dict, optionally containing
                ``extra.action_is_pad`` as a ``(batch, chunk)`` bool tensor.
            exempt_idx: Optional indices of samples that should stay fully
                weighted despite the padding, for objectives that do not regress
                onto the dataset action (e.g. self-distillation).

        Returns:
            A ``(batch, chunk)`` bool mask, ``True`` where the step should
            contribute to the loss, or ``None`` when the batch carries no
            padding information (e.g. non-chunked datasets).
        """
        actions_is_pad = batch.get(EXTRA + ".action_is_pad")
        if actions_is_pad is None:
            return None
        bound = ~actions_is_pad
        if exempt_idx is not None:
            bound = bound.clone()
            bound[exempt_idx] = True
        return bound

    @staticmethod
    def reduce_losses(losses: torch.Tensor, in_episode_bound: torch.Tensor | None) -> torch.Tensor:
        """Reduce per-element losses to a scalar, ignoring padded action steps.

        Args:
            losses: Per-element losses shaped ``(batch, chunk, action_dim)``.
                Padded steps are zeroed here if they were not already.
            in_episode_bound: Optional ``(batch, chunk)`` bool mask from
                :meth:`in_episode_bound`.

        Returns:
            Scalar loss averaged over the valid elements only.
        """
        if in_episode_bound is None:
            return losses.mean()
        # Not an in-place `*=`: `losses` is often a slice of a larger tensor and
        # mutating a caller's argument here would be a surprising side effect.
        masked = losses * in_episode_bound.unsqueeze(-1)
        # Divide by the number of *valid* elements, not the full tensor. A plain
        # .mean() would count the zeroed-out padded steps in the denominator and
        # shrink the loss (and gradient) in proportion to the padding fraction.
        num_valid = (in_episode_bound.sum() * losses.shape[-1]).clamp_min(1)
        return masked.sum() / num_valid

    @property
    @abstractmethod
    def reward_delta_indices(self) -> list | None:
        """Return reward indices.

        Currently returns `None` as rewards are not implemented.

        Returns:
            None or a list of reward indices.
        """

    @property
    @abstractmethod
    def action_delta_indices(self) -> list | None:
        """Get indices of actions relative to the current timestep.

        Returns:
            None or a list of relative action indices.
        """

    @property
    @abstractmethod
    def observation_delta_indices(self) -> list | None:
        """Get indices of observations relative to the current timestep.

        Returns:
            None or a list of relative observation indices.
        """

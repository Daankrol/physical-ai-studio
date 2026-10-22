# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Scatter and gather dataset joints in LingBot's 55-dimensional checkpoint space."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from .config import LingBotVLA2Config


def camera_name(key: str) -> str:
    """Remove layout prefixes without discarding dots in camera names.

    Returns:
        The dataset camera name.
    """
    return key.removeprefix("observation.").removeprefix("images.").removeprefix("image.")


class JointLayout(nn.Module):
    """Map flat dataset vectors to canonical groups and back, without mixing arms and grippers."""

    def __init__(self, config: LingBotVLA2Config, *, action: bool = False) -> None:
        """Build index tensors for either state or action vectors."""
        super().__init__()
        mapping = config.action_map if action else config.state_map
        width = config.max_action_dim if action else config.max_state_dim
        source: list[int] = []
        target: list[int] = []
        offset = 0
        for group, size in config.joint_layout:
            destination = offset
            for start, end in mapping.get(group, []):
                source.extend(range(start, end))
                target.extend(range(destination, destination + end - start))
                destination += end - start
            offset += size
        self.width = width
        self.source: Tensor
        self.target: Tensor
        self.register_buffer("source", torch.tensor(source, dtype=torch.long), persistent=False)
        self.register_buffer("target", torch.tensor(target, dtype=torch.long), persistent=False)

    def forward(self, values: Tensor) -> tuple[Tensor, Tensor]:
        """Return canonical values and occupied dimensions.

        Raises:
            ValueError: If the input width is not the mapped dataset width.
        """
        if values.shape[-1] != self.source.numel():
            msg = f"Expected {self.source.numel()} dataset dimensions, got {values.shape[-1]}"
            raise ValueError(msg)
        canonical = values.new_zeros((*values.shape[:-1], self.width))
        canonical[..., self.target] = values[..., self.source]
        mask = torch.zeros(self.width, dtype=torch.bool, device=values.device)
        mask[self.target] = True
        return canonical, mask

    def gather(self, canonical: Tensor) -> Tensor:
        """Restore dataset order from a canonical prediction.

        Returns:
            The flat dataset vector.

        Raises:
            ValueError: If the input width is not the canonical width.
        """
        if canonical.shape[-1] != self.width:
            msg = f"Expected {self.width} canonical dimensions, got {canonical.shape[-1]}"
            raise ValueError(msg)
        dataset = canonical.new_empty((*canonical.shape[:-1], self.source.numel()))
        dataset[..., self.source] = canonical[..., self.target]
        return dataset

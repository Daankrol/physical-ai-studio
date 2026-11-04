# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Restore dataset action order and units from the canonical checkpoint space."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor, nn

from physicalai.data import FeatureType
from physicalai.data.observation import ACTION
from physicalai.policies.utils.normalization import FeatureNormalizeTransform, NormalizationType

from .layout import JointLayout
from .normalization import UpstreamNormalizeTransform

if TYPE_CHECKING:
    from .config import LingBotVLA2Config
    from .preprocessor import LingBotVLA2Preprocessor


class LingBotVLA2Postprocessor(nn.Module):
    """Gather canonical joints, undo feature normalization, and trim for execution."""

    def __init__(self, config: LingBotVLA2Config) -> None:
        """Build the inverse action transform from the dataset feature statistics."""
        super().__init__()
        self.n_action_steps = config.n_action_steps
        self.layout = JointLayout(config, action=True)
        features = {
            feature.name: feature
            for feature in config.output_features
            if feature.ftype == FeatureType.ACTION and feature.name is not None
        }
        normalizer = UpstreamNormalizeTransform if config.training_recipe == "upstream" else FeatureNormalizeTransform
        self.relative_pairs = [
            (
                [index for start, end in config.action_map[group] for index in range(start, end)],
                [index for start, end in config.state_map[group] for index in range(start, end)],
            )
            for group in config.relative_action_groups
        ]
        self.denormalizer = normalizer(
            features,
            {FeatureType.ACTION: NormalizationType(config.normalization_mode)},
            inverse=True,
        )

    def forward(self, canonical: Tensor, state: Tensor | None = None) -> Tensor:
        """Return the action chunk in dataset order and units.

        Returns:
            Tensor with n_action_steps and dataset action width.

        Raises:
            ValueError: If relative actions have no current raw state.
        """
        dataset_actions = self.layout.gather(canonical)
        actions = self.denormalizer({ACTION: dataset_actions})[ACTION][:, : self.n_action_steps]
        if self.relative_pairs:
            if state is None:
                msg = "Relative actions require the current raw state for denormalization"
                raise ValueError(msg)
            if state.ndim == 3:  # noqa: PLR2004 -- batch, time, joints
                state = state[:, 0]
            actions = actions.clone()
            for action_indices, state_indices in self.relative_pairs:
                actions[..., action_indices] += state[:, None, state_indices]
        return actions


def make_lingbotvla2_processors(config: LingBotVLA2Config) -> tuple[LingBotVLA2Preprocessor, LingBotVLA2Postprocessor]:
    """Build a matching pair of observation and action processors.

    Returns:
        Preprocessor and postprocessor.
    """
    from .preprocessor import LingBotVLA2Preprocessor  # noqa: PLC0415

    return LingBotVLA2Preprocessor(config), LingBotVLA2Postprocessor(config)

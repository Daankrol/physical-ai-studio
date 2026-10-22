# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Training-only statistics in the representation consumed by LingBot."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from physicalai.data import FeatureType, NormalizationParameters

if TYPE_CHECKING:
    from physicalai.data.lerobot.dataset import _LeRobotDatasetAdapter

    from .config import LingBotVLA2Config


def _statistics(values: Tensor) -> NormalizationParameters:
    values = values.double()
    if not torch.isfinite(values).all():
        msg = "Training state/action values must be finite"
        raise ValueError(msg)
    return NormalizationParameters(
        mean=values.mean(0).float().tolist(),
        std=values.std(0, correction=0).float().tolist(),
        min=values.amin(0).float().tolist(),
        max=values.amax(0).float().tolist(),
        q01=values.quantile(0.01, dim=0).float().tolist(),
        q99=values.quantile(0.99, dim=0).float().tolist(),
    )


def training_statistics(
    state: Tensor,
    action: Tensor,
    episodes: Tensor,
    config: LingBotVLA2Config,
) -> tuple[NormalizationParameters, NormalizationParameters]:
    """Match upstream's per-horizon relative stats and pooled absolute stats.

    Returns:
        State statistics and action statistics in the training representation.

    Raises:
        ValueError: If columns are empty or misaligned.
    """
    count = len(state)
    if count == 0 or len(action) != count or len(episodes) != count:
        msg = "Statistics require nonempty aligned state/action/episode columns"
        raise ValueError(msg)
    ends = torch.full((count,), count - 1, dtype=torch.long)
    boundary = episodes[:-1] != episodes[1:]
    ends[:-1] = torch.where(boundary, torch.arange(count - 1), ends[:-1])
    ends = ends.flip(0).cummin(0).values.flip(0)
    indices = torch.minimum(torch.arange(count)[:, None] + torch.arange(config.chunk_size), ends[:, None])
    # shortcut: chunks materialize on CPU; stream statistics when datasets no longer fit RAM.
    chunks = action[indices].clone()
    relative = []
    for group in config.relative_action_groups:
        action_indices = [i for start, end in config.action_map[group] for i in range(start, end)]
        state_indices = [i for start, end in config.state_map[group] for i in range(start, end)]
        chunks[..., action_indices] -= state[:, None, state_indices]
        relative.extend(action_indices)
    per_horizon = _statistics(chunks)
    pooled = _statistics(chunks.flatten(0, 1))
    updates = {}
    for field in ("mean", "std", "min", "max", "q01", "q99"):
        horizon_value = torch.tensor(getattr(per_horizon, field))
        absolute_value = torch.tensor(getattr(pooled, field)).expand_as(horizon_value).clone()
        absolute_value[:, relative] = horizon_value[:, relative]
        updates[field] = absolute_value.tolist()
    return _statistics(state), replace(per_horizon, **updates)


def adopt_training_statistics(dataset: _LeRobotDatasetAdapter, config: LingBotVLA2Config) -> LingBotVLA2Config:
    """Read only tabular columns of the selected training episodes, never camera frames.

    Returns:
        Config with training-only statistics and dataset FPS.
    """
    source = dataset._lerobot_dataset.hf_dataset  # noqa: SLF001 -- adapter owns the episode-filtered table
    state_columns = dataset._state_columns or ["observation.state"]  # noqa: SLF001
    action_columns = dataset._action_columns or ["action"]  # noqa: SLF001
    columns = [*state_columns, *action_columns, "episode_index"]
    table = source.select_columns(columns).with_format("torch")[:]
    state = torch.cat([torch.as_tensor(table[key]).reshape(len(source), -1) for key in state_columns], dim=-1)
    action = torch.cat([torch.as_tensor(table[key]).reshape(len(source), -1) for key in action_columns], dim=-1)
    state_stats, action_stats = training_statistics(state, action, torch.as_tensor(table["episode_index"]), config)
    return replace(
        config,
        input_features=[
            replace(feature, normalization_data=state_stats) if feature.ftype == FeatureType.STATE else feature
            for feature in config.input_features
        ],
        output_features=[replace(feature, normalization_data=action_stats) for feature in config.output_features],
        dataset_fps=dataset.fps,
    )

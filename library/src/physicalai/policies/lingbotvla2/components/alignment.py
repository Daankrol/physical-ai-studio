# Copyright (C) 2026 Robbyant Inc.
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Released task-token resamplers, adapted from upstream's OpenFlamingo-derived heads."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn


class PerceiverAttention(nn.Module):
    """Cross-attend learned queries to image and task features."""

    def __init__(self, dimension: int, heads: int, head_dim: int) -> None:
        """Allocate checkpoint-compatible attention projections."""
        super().__init__()
        self.heads = heads
        self.scale = head_dim**-0.25
        self.norm1 = nn.LayerNorm(dimension)
        self.norm2 = nn.LayerNorm(dimension)
        self.to_q = nn.Linear(dimension, heads * head_dim, bias=False)
        self.to_kv = nn.Linear(dimension, 2 * heads * head_dim, bias=False)
        self.to_out = nn.Linear(heads * head_dim, dimension, bias=False)

    def forward(self, features: Tensor, queries: Tensor) -> Tensor:
        """Return query updates attending to both context and queries."""
        features, queries = self.norm1(features), self.norm2(queries)
        query = self.to_q(queries)
        key, value = self.to_kv(torch.cat((features, queries), dim=1)).chunk(2, dim=-1)
        query, key, value = [
            tensor.reshape(tensor.shape[0], tensor.shape[1], self.heads, -1).transpose(1, 2)
            for tensor in (query, key, value)
        ]
        weights = (query * self.scale) @ (key * self.scale).transpose(-2, -1)
        weights = weights.float().softmax(-1).to(query.dtype)
        output = (weights @ value).transpose(1, 2).reshape(queries.shape[0], queries.shape[1], -1)
        return self.to_out(output)


class TaskTokenResampler(nn.Module):
    """Preserve upstream module names so alignment weights load strictly."""

    def __init__(self, hidden_size: int, output_size: int, heads: int = 4, head_dim: int = 32) -> None:
        """Allocate the released one-layer resampler."""
        super().__init__()
        self.proj_in1 = nn.Linear(hidden_size, hidden_size)
        self.proj_in2 = nn.Linear(hidden_size, hidden_size)
        self.proj_out = nn.Linear(hidden_size, output_size)
        self.norm_out = nn.LayerNorm(output_size)
        self.layers = nn.ModuleList([
            nn.ModuleList([
                PerceiverAttention(hidden_size, heads, head_dim),
                nn.Sequential(
                    nn.LayerNorm(hidden_size),
                    nn.Linear(hidden_size, hidden_size, bias=False),
                    nn.GELU(),
                    nn.Linear(hidden_size, hidden_size, bias=False),
                ),
            ]),
        ])

    def forward(self, features: Tensor, queries: Tensor) -> Tensor:
        """Return resampled teacher-feature predictions."""
        queries, features = self.proj_in1(queries), self.proj_in2(features)
        for layer in self.layers:
            attention, feedforward = cast("nn.ModuleList", layer)
            queries = queries + attention(features, queries)  # noqa: PLR6104 -- preserve saved activations
            queries = queries + feedforward(queries)  # noqa: PLR6104
        return self.norm_out(self.proj_out(queries))


class TaskTokenHead(nn.Module):
    """The released single-layer depth/video head."""

    def __init__(self, hidden_size: int, output_size: int, heads: int = 4, head_dim: int = 32) -> None:
        """Keep the checkpoint's projector namespace."""
        super().__init__()
        self.projector = TaskTokenResampler(hidden_size, output_size, heads, head_dim)

    def forward(self, features: Tensor, queries: Tensor) -> Tensor:
        """Return teacher-space features for the learned queries."""
        return self.projector(features, queries)

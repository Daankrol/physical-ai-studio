# Copyright (C) 2026 Robbyant Inc.
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Differentiable routed experts using PyTorch grouped matrix multiplication."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.distributed.tensor import DTensor
from torch.nn import functional


def fused_experts(hidden: Tensor, selected: Tensor, routing: Tensor, gate: Tensor, up: Tensor, down: Tensor) -> Tensor:  # noqa: PLR0914
    """Match upstream's BF16 kernel boundary while preserving FP32 master weights.

    Returns:
        Routed expert activations in the input dtype.

    Raises:
        ValueError: If called without CUDA.
    """
    if not hidden.is_cuda:
        msg = "The fused LingBot MoE backend requires CUDA"
        raise ValueError(msg)
    shape, dtype = hidden.shape, hidden.dtype
    tensors = [hidden, routing, gate, up, down]
    hidden, routing, gate, up, down = [
        (tensor.to_local() if isinstance(tensor, DTensor) else tensor).to(torch.bfloat16).contiguous()
        for tensor in tensors
    ]
    hidden = hidden.flatten(0, 1)
    routes = selected.flatten()
    order = routes.argsort(stable=True)
    counts = torch.zeros(gate.shape[0], device=routes.device, dtype=torch.int32)
    offsets = counts.scatter_add(0, routes, torch.ones_like(routes, dtype=torch.int32)).cumsum(0, dtype=torch.int32)
    tokens = hidden.index_select(0, order // selected.shape[-1])
    gated = functional.silu(functional.grouped_mm(tokens, gate.transpose(1, 2), offs=offsets))
    activated = gated * functional.grouped_mm(tokens, up.transpose(1, 2), offs=offsets)
    activated *= routing.flatten().index_select(0, order)[:, None]
    result = functional.grouped_mm(activated, down.transpose(1, 2), offs=offsets)
    result = result.index_select(0, order.argsort()).reshape(*shape[:2], selected.shape[-1], shape[-1])
    return result.float().sum(-2).to(torch.bfloat16).to(dtype)

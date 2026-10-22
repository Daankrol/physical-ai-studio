# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""FSDP2 units for the joint decoder, whose layer methods bypass stock forward hooks."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn
from torch.distributed.fsdp import fully_shard, register_fsdp_forward_method

from .components.alignment import TaskTokenHead
from .components.expert import AdaRMSNorm, FusedExperts, TokenMoE

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

    from .policy import LingBotVLA2


def shard_policy(policy: LingBotVLA2, mesh: DeviceMesh) -> None:
    """Shard invoked modules rather than decoder containers that are never called.

    Raises:
        RuntimeError: If the policy is not configured.
    """

    def wrap(module: nn.Module) -> None:
        # The joint loop invokes projections directly, not Qwen decoder-layer forward.
        unit = isinstance(module, (TaskTokenHead, TokenMoE, AdaRMSNorm, FusedExperts)) or module.__class__.__name__ in {
            "Qwen3VLVisionBlock",
            "Qwen3VLTextMLP",
        }
        if unit:
            if isinstance(module, TokenMoE):
                fully_shard(module.experts, mesh=mesh, reshard_after_forward=True)
            fully_shard(module, mesh=mesh, reshard_after_forward=True)
            return
        if isinstance(module, nn.Linear):
            fully_shard(module, mesh=mesh, reshard_after_forward=True)
            return
        for child in module.children():
            wrap(child)

    if policy.model is None:
        msg = "Configure LingBot before applying FSDP"
        raise RuntimeError(msg)
    wrap(policy.model)
    fully_shard(policy, mesh=mesh, reshard_after_forward=True)
    register_fsdp_forward_method(policy, "predict_action_chunk")
    register_fsdp_forward_method(policy, "compute_val_loss")

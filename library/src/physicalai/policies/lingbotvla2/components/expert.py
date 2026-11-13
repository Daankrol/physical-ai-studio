# Copyright (C) 2026 Robbyant Inc.
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint-compatible adaptive norms and token-level MoE for the action expert."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional


class AdaRMSNorm(nn.Module):
    """RMSNorm followed by zero-initialized time-conditioned scale and shift."""

    def __init__(self, hidden_size: int, cond_dim: int, eps: float = 1e-6) -> None:
        """Initialize the norm and time-conditioning projections."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.gamma = nn.Linear(cond_dim, hidden_size)
        self.beta = nn.Linear(cond_dim, hidden_size)
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, hidden: Tensor, cond: Tensor) -> Tensor:
        """Apply time-conditioned RMS normalization.

        Returns:
            Normalized tokens.
        """
        dtype = hidden.dtype
        normalized = hidden.float()
        variance = normalized.square().mean(dim=-1, keepdim=True)
        normalized = torch.rsqrt(variance + self.variance_epsilon) * normalized * self.weight
        gamma = self.gamma(cond).unsqueeze(1).float()
        beta = self.beta(cond).unsqueeze(1).float()
        return ((1 + gamma) * normalized + beta).to(dtype)


class FusedExperts(nn.Module):
    """Fused parameter layout from the published sharded safetensors checkpoint."""

    def __init__(self, count: int, hidden_size: int, intermediate_size: int) -> None:
        """Allocate routed experts in the upstream fused tensor layout."""
        super().__init__()
        self.backend = "dense"
        self.gate_proj = nn.Parameter(torch.empty(count, intermediate_size, hidden_size))
        self.up_proj = nn.Parameter(torch.empty(count, intermediate_size, hidden_size))
        self.down_proj = nn.Parameter(torch.empty(count, hidden_size, intermediate_size))
        for parameter in self.parameters():
            nn.init.normal_(parameter, std=0.02)

    def forward(self, hidden: Tensor, selected: Tensor | None = None, routing: Tensor | None = None) -> Tensor:
        """Compute each expert output for each token.

        Returns:
            Token activations for every routed expert.

        Raises:
            ValueError: If fused routing inputs are absent.
        """
        if self.backend == "triton":
            from .fused_moe import fused_experts  # noqa: PLC0415

            if selected is None or routing is None:
                msg = "The fused expert backend requires selected routes and weights"
                raise ValueError(msg)
            return fused_experts(hidden, selected, routing, self.gate_proj, self.up_proj, self.down_proj)
        # Dense compute is retained as the strict FP32 reference.
        gated = functional.silu(torch.einsum("bsh,eih->bsei", hidden, self.gate_proj))
        up = torch.einsum("bsh,eih->bsei", hidden, self.up_proj)
        return torch.einsum("bsei,ehi->bseh", gated * up, self.down_proj)


class SharedExpert(nn.Module):
    """Always-active SwiGLU MLP."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        """Allocate the checkpoint's shared SwiGLU projections."""
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden: Tensor) -> Tensor:
        """Apply the shared expert to all tokens.

        Returns:
            Shared expert activations.
        """
        return self.down_proj(functional.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


class TokenMoE(nn.Module):
    """Sigmoid top-k router, fused routed weights, and ungated shared expert."""

    def __init__(
        self,
        hidden_size: int,
        *,
        num_experts: int = 32,
        top_k: int = 4,
        intermediate_size: int = 512,
        shared_intermediate_size: int = 704,
        scaling_factor: float = 4.0,
    ) -> None:
        """Initialize the sigmoid router and fused expert weights.

        Raises:
            ValueError: If top-k routing exceeds the number of experts.
        """
        super().__init__()
        if not 1 <= top_k <= num_experts:
            msg = "top_k must be in [1, num_experts]"
            raise ValueError(msg)
        self.top_k = top_k
        self.routed_scaling_factor = scaling_factor
        self.e_score_correction_bias: Tensor
        self.register_buffer("e_score_correction_bias", torch.zeros(num_experts))
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = FusedExperts(num_experts, hidden_size, intermediate_size)
        self.shared_expert = SharedExpert(hidden_size, shared_intermediate_size)

    def forward(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
        """Return expert activations and fp32 router logits."""
        with torch.amp.autocast(hidden.device.type, enabled=False):
            logits = functional.linear(hidden.float(), self.gate.weight.float())
        scores = logits.sigmoid()
        selected = (scores + self.e_score_correction_bias).topk(self.top_k, dim=-1).indices
        weights = scores.gather(-1, selected)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20) * self.routed_scaling_factor
        if self.experts.backend == "triton":
            output = self.experts(hidden, selected, weights)
        else:
            routes = torch.zeros_like(scores).scatter_(-1, selected, weights).to(hidden.dtype)
            output = (self.experts(hidden) * routes.unsqueeze(-1)).sum(dim=-2)
        return output + self.shared_expert(hidden), logits

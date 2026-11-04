# Copyright (C) 2026 Robbyant Inc.
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""One Qwen2 action expert layer, with checkpoint-compatible parameter names."""

from __future__ import annotations

from torch import Tensor, nn

from .expert import AdaRMSNorm, TokenMoE


class _AttentionProjections(nn.Module):
    def __init__(self, hidden: int, queries: int, keys: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden, queries, bias=True)
        self.k_proj = nn.Linear(hidden, keys, bias=True)
        self.v_proj = nn.Linear(hidden, keys, bias=True)
        self.o_proj = nn.Linear(queries, hidden, bias=False)


class ExpertLayer(nn.Module):
    """Compute expert attention projections and residual MoE separately for joint attention."""

    def __init__(
        self,
        hidden_size: int,
        *,
        num_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        num_experts: int = 32,
        top_k: int = 4,
        intermediate_size: int = 512,
        shared_intermediate_size: int = 704,
        scaling_factor: float = 4.0,
    ) -> None:
        """Build the same projection and norm hierarchy as upstream's expert decoder."""
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        q_width = num_heads * head_dim
        kv_width = num_key_value_heads * head_dim
        self.self_attn = _AttentionProjections(hidden_size, q_width, kv_width)
        self.input_layernorm = AdaRMSNorm(hidden_size, hidden_size)
        self.post_attention_layernorm = AdaRMSNorm(hidden_size, hidden_size)
        self.mlp = TokenMoE(
            hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            intermediate_size=intermediate_size,
            shared_intermediate_size=shared_intermediate_size,
            scaling_factor=scaling_factor,
        )

    def qkv(self, hidden: Tensor, cond: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return the query, key and value tensors in (B, S, heads, head_dim) order."""
        normalized = self.input_layernorm(hidden, cond)
        shape = (*hidden.shape[:-1], -1, self.head_dim)
        return (
            self.self_attn.q_proj(normalized).view(shape),
            self.self_attn.k_proj(normalized).view(shape),
            self.self_attn.v_proj(normalized).view(shape),
        )

    def finish(self, hidden: Tensor, attention: Tensor, cond: Tensor) -> tuple[Tensor, Tensor]:
        """Apply the attention residual and MoE residual.

        Returns:
            Updated hidden states and fp32 router logits.
        """
        residual = hidden + self.self_attn.o_proj(attention.reshape(*hidden.shape[:-1], -1).to(hidden.dtype))
        mlp_output, router_logits = self.mlp(self.post_attention_layernorm(residual, cond))
        return residual + mlp_output, router_logits

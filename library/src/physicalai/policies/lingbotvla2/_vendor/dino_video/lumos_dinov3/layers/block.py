# Adapted for Physical AI Studio; see the vendor README for source, licenses, and changes.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from typing import Callable

import torch
from torch import Tensor, nn

from .attention import SelfAttention, rope_apply
from .ffn_layers import Mlp
from .layer_scale import LayerScale


class SelfAttentionBlock(nn.Module):
    """Frozen video-teacher block with packed, frame-causal PyTorch attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        init_values=None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        mask_k_bias: bool = False,
        attention_mode: str = "flex_block_causal",
        device=None,
    ) -> None:
        super().__init__()
        if attention_mode not in {"flex_block_causal", "sdpa_block_causal"}:
            raise ValueError(f"Unsupported video-teacher attention: {attention_mode}")
        self.norm1 = norm_layer(dim)
        self.attn = SelfAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            mask_k_bias=mask_k_bias,
            device=device,
        )
        self.ls1 = LayerScale(dim, init_values=init_values, device=device) if init_values else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=int(dim * ffn_ratio),
            act_layer=act_layer,
            bias=ffn_bias,
            device=device,
        )
        self.ls2 = LayerScale(dim, init_values=init_values, device=device) if init_values else nn.Identity()
        self.attention_mode = attention_mode

    def forward(self, x: Tensor) -> Tensor:
        packed_seq_info = getattr(self, "_packed_seq_info", None)
        if packed_seq_info is None:
            raise RuntimeError("Video-teacher blocks require packed sequence metadata")
        rope_sin, rope_cos, attention_extra = packed_seq_info
        attn = self.attn
        qkv = attn.qkv(self.norm1(x))
        C = attn.qkv.in_features
        qkv = qkv.reshape(qkv.shape[0], 3, attn.num_heads, C // attn.num_heads)
        q, k, v = qkv.unbind(1)
        q_dtype, k_dtype = q.dtype, k.dtype
        q = rope_apply(q.to(rope_sin.dtype), rope_sin.unsqueeze(-2), rope_cos.unsqueeze(-2)).to(q_dtype)
        k = rope_apply(k.to(rope_sin.dtype), rope_sin.unsqueeze(-2), rope_cos.unsqueeze(-2)).to(k_dtype)
        shapes, crops_per_shape, n_prefix, block_masks = attention_extra
        if self.attention_mode == "flex_block_causal":
            from .flex_attention_packed import flex_attention_block_causal

            attn_out = flex_attention_block_causal(
                q.contiguous(), k.contiguous(), v.contiguous(),
                shapes, crops_per_shape, n_prefix, block_masks,
                has_regional_compile=getattr(self, "_has_regional_compile", True),
            )
        else:
            from .sdpa_attention_packed import sdpa_attention_block_causal

            attn_out = sdpa_attention_block_causal(
                q.contiguous(), k.contiguous(), v.contiguous(),
                shapes, crops_per_shape, n_prefix, block_masks,
            )
        x = x + self.ls1(attn.proj(attn_out.reshape(-1, C)))
        return x + self.ls2(self.mlp(self.norm2(x)))

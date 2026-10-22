# Adapted for Physical AI Studio; see the vendor README for source, licenses, and changes.
"""Checkpoint-compatible video backbone construction and frame tokenization."""

import logging
from typing import Any, Literal, Tuple

import torch
from torch import Tensor, nn

from ..layers import PatchEmbed, RopePositionEmbedding, RopePositionEmbedding3D, SelfAttentionBlock
from ..utils import named_apply
from .vision_transformer import dtype_dict, ffn_layer_dict, init_weights_vit, norm_layer_dict

logger = logging.getLogger("dinov3")


class UnifiedVideoViT(nn.Module):
    """Frozen video teacher with per-frame CLS, storage, and patch tokens."""

    def __init__(
        self,
        *,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        pos_embed_rope_base: float = 100.0,
        pos_embed_rope_min_period: float | None = None,
        pos_embed_rope_max_period: float | None = None,
        pos_embed_rope_normalize_coords: Literal["min", "max", "separate"] = "separate",
        pos_embed_rope_shift_coords: float | None = None,
        pos_embed_rope_jitter_coords: float | None = None,
        pos_embed_rope_rescale_coords: float | None = None,
        pos_embed_rope_temporal_base: float = 10000.0,
        pos_embed_rope_fps: float | None = None,
        pos_embed_rope_base_fps: float = 24.0,
        pos_embed_rope_dtype: str = "bf16",
        pos_embed_rope_3d: bool = False,
        pos_embed_rope_prefix_temporal: bool = False,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = True,
        layerscale_init: float | None = None,
        norm_layer: str = "layernorm",
        ffn_layer: str = "mlp",
        ffn_bias: bool = True,
        proj_bias: bool = True,
        n_storage_tokens: int = 0,
        mask_k_bias: bool = False,
        untie_cls_and_patch_norms: bool = False,
        untie_global_and_local_cls_norm: bool = False,
        attention_mode: str = "flex_block_causal",
        device: Any | None = None,
        **ignored_kwargs,
    ):
        super().__init__()
        if ignored_kwargs:
            logger.warning(f"Ignored kwargs: {ignored_kwargs}")
        norm_layer_cls = norm_layer_dict[norm_layer]
        ffn_layer_cls = ffn_layer_dict[ffn_layer]
        self.num_features = self.embed_dim = embed_dim
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.attention_mode = attention_mode
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            flatten_embedding=False,
        )
        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim, device=device))
        self.n_storage_tokens = n_storage_tokens
        if n_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(torch.empty(1, n_storage_tokens, embed_dim, device=device))
        self.mask_token = nn.Parameter(torch.empty(1, embed_dim, device=device))
        rope_cls = RopePositionEmbedding3D if pos_embed_rope_3d else RopePositionEmbedding
        rope_kwargs = dict(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=pos_embed_rope_base,
            min_period=pos_embed_rope_min_period,
            max_period=pos_embed_rope_max_period,
            normalize_coords=pos_embed_rope_normalize_coords,
            shift_coords=pos_embed_rope_shift_coords,
            jitter_coords=pos_embed_rope_jitter_coords,
            rescale_coords=pos_embed_rope_rescale_coords,
            dtype=dtype_dict[pos_embed_rope_dtype],
            device=device,
        )
        if pos_embed_rope_3d:
            rope_kwargs["temporal_base"] = pos_embed_rope_temporal_base
            rope_kwargs["fps"] = pos_embed_rope_fps
            rope_kwargs["base_fps"] = pos_embed_rope_base_fps
        self.rope_embed = rope_cls(**rope_kwargs)
        self.rope_3d = pos_embed_rope_3d
        self.rope_prefix_temporal = pos_embed_rope_prefix_temporal
        self.blocks = nn.ModuleList([
            SelfAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                ffn_ratio=ffn_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                norm_layer=norm_layer_cls,
                act_layer=nn.GELU,
                ffn_layer=ffn_layer_cls,
                init_values=layerscale_init,
                mask_k_bias=mask_k_bias,
                attention_mode=attention_mode,
                device=device,
            )
            for _ in range(depth)
        ])
        self.norm = norm_layer_cls(embed_dim)
        self.untie_cls_and_patch_norms = untie_cls_and_patch_norms
        self.cls_norm = norm_layer_cls(embed_dim) if untie_cls_and_patch_norms else None
        self.untie_global_and_local_cls_norm = untie_global_and_local_cls_norm
        self.local_cls_norm = norm_layer_cls(embed_dim) if untie_global_and_local_cls_norm else None
        self.head = nn.Identity()

    def init_weights(self):
        self.rope_embed._init_weights()
        nn.init.normal_(self.cls_token, std=0.02)
        if self.n_storage_tokens > 0:
            nn.init.normal_(self.storage_tokens, std=0.02)
        nn.init.zeros_(self.mask_token)
        named_apply(init_weights_vit, self)

    def prepare_tokens_with_masks(
        self, x: Tensor, masks: Tensor | None = None
    ) -> Tuple[Tensor, Tuple[int, int, int]]:
        """Tokenize [B,C,T,H,W] video with per-frame CLS and storage prefixes."""
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        x = self.patch_embed(x)
        _, Hp, Wp, D = x.shape
        x = x.reshape(B, T, Hp * Wp, D)
        if masks is not None:
            masks_reshaped = masks.reshape(B, T, Hp * Wp)
            x = torch.where(masks_reshaped.unsqueeze(-1), self.mask_token.to(x.dtype), x)
            cls_token = self.cls_token
        else:
            cls_token = self.cls_token + 0 * self.mask_token
        cls_tokens = cls_token.unsqueeze(0).expand(B, T, 1, -1)
        if self.n_storage_tokens > 0:
            storage = self.storage_tokens.unsqueeze(0).expand(B, T, -1, -1)
        else:
            storage = x.new_empty(B, T, 0, D)
        frame_tokens = torch.cat([cls_tokens, storage, x], dim=2)
        return frame_tokens.reshape(B, T * (1 + self.n_storage_tokens + Hp * Wp), D), (T, Hp, Wp)

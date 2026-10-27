# Copyright (C) 2026 Robbyant Inc.
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Layer-wise joint Qwen3-VL / action-expert attention."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch
from torch import Tensor
from torch.nn import functional
from torch.utils.checkpoint import checkpoint
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb

if TYPE_CHECKING:
    from transformers import Qwen3VLForConditionalGeneration
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextDecoderLayer, Qwen3VLTextModel

    from physicalai.policies.lingbotvla2.model import _ExpertDecoder

    from .layer import ExpertLayer


class JointBackbone:
    """Drive stock Qwen3-VL layers and the action expert through one attention call per layer."""

    def __init__(
        self,
        vlm: Qwen3VLForConditionalGeneration,
        expert: _ExpertDecoder,
        *,
        attention_fp32: bool,
        attention_backend: str = "sdpa",
    ) -> None:
        """Reference registered modules without adding duplicate state-dict namespaces."""
        self.vlm = vlm
        self.expert = expert
        self.attention_fp32 = attention_fp32
        self.gradient_checkpointing = False
        self.attention_backend = attention_backend
        self._flex_attention: Any = None

    def __call__(
        self,
        prefix: Tensor | None,
        suffix: Tensor | None,
        mask: Tensor,
        position_ids: Tensor,
        *,
        cond: Tensor | None = None,
        cache: dict[int, tuple[Tensor, Tensor]] | None = None,
        fill_cache: bool = False,
        visual_pos_masks: Tensor | None = None,
        deepstack: list[Tensor] | None = None,
    ) -> tuple[Tensor | None, Tensor | None, list[Tensor]]:
        """Return both hidden streams and the MoE logits from every layer.

        Returns:
            Prefix, suffix, and per-layer router logits.
        """
        text = cast("Qwen3VLTextModel", cast("Any", self.vlm.model).language_model)
        routers: list[Tensor] = []
        for index, (vlm_module, expert_module) in enumerate(zip(text.layers, self.expert.layers, strict=True)):
            vlm_layer = cast("Qwen3VLTextDecoderLayer", vlm_module)
            expert_layer = cast("ExpertLayer", expert_module)
            visual = deepstack[index] if deepstack is not None and index < len(deepstack) else None

            def run(
                pre: Tensor | None,
                suf: Tensor | None,
                *,
                layer: Qwen3VLTextDecoderLayer = vlm_layer,
                expert: ExpertLayer = expert_layer,
                visual_features: Tensor | None = visual,
                layer_index: int = index,
            ) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
                return self._layer(
                    layer,
                    expert,
                    pre,
                    suf,
                    mask,
                    position_ids,
                    cond,
                    cache,
                    layer_index,
                    fill_cache,
                    visual_pos_masks,
                    visual_features,
                )

            if self.gradient_checkpointing and self.vlm.training and prefix is not None and suffix is not None:
                output = cast(
                    "tuple[Tensor | None, Tensor | None, Tensor | None]",
                    checkpoint(run, prefix, suffix, use_reentrant=False),
                )
            else:
                output = run(prefix, suffix)
            prefix, suffix, logits = output
            if logits is not None:
                routers.append(logits)
        if prefix is not None:
            prefix = text.norm(prefix)
        if suffix is not None:
            suffix = self.expert.norm(suffix)
        return prefix, suffix, routers

    def _layer(  # noqa: PLR0912, PLR0914, PLR0915
        self,
        vlm_layer: Qwen3VLTextDecoderLayer,
        expert_layer: ExpertLayer,
        prefix: Tensor | None,
        suffix: Tensor | None,
        mask: Tensor,
        position_ids: Tensor,
        cond: Tensor | None,
        cache: dict[int, tuple[Tensor, Tensor]] | None,
        index: int,
        fill_cache: bool,  # noqa: FBT001
        visual_pos_masks: Tensor | None,
        visual: Tensor | None,
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
        queries, keys, values = [], [], []
        if prefix is not None:
            normalized = vlm_layer.input_layernorm(prefix)
            attn = vlm_layer.self_attn
            shape = (*prefix.shape[:-1], -1, attn.head_dim)
            queries.append(attn.q_norm(attn.q_proj(normalized).view(shape)))
            keys.append(attn.k_norm(attn.k_proj(normalized).view(shape)))
            values.append(attn.v_proj(normalized).view(shape))
        if suffix is not None:
            if cond is None:
                msg = "An action suffix needs its time embedding"
                raise ValueError(msg)
            q, k, v = expert_layer.qkv(suffix, cond)
            queries.append(q)
            keys.append(k)
            values.append(v)
        q = torch.cat(queries, dim=1)
        k = torch.cat(keys, dim=1)
        v = torch.cat(values, dim=1)
        if self.attention_fp32:
            q, k, v = q.float(), k.float(), v.float()
        cos, sin = self.vlm.model.language_model.rotary_emb(q, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2)
        if cache is not None:
            if fill_cache:
                cache[index] = (k, v)
            else:
                past_k, past_v = cache[index]
                k, v = torch.cat((past_k, k), dim=1), torch.cat((past_v, v), dim=1)
        if self.attention_backend == "flex":
            if not q.is_cuda:
                msg_0 = "LingBot Flex attention requires CUDA"
                raise ValueError(msg_0)
            if self._flex_attention is None:
                from torch.nn.attention.flex_attention import flex_attention  # noqa: PLC0415

                self._flex_attention = torch.compile(flex_attention, dynamic=False)

            def score_mod(score: Tensor, batch: Tensor, _head: Tensor, query: Tensor, key: Tensor) -> Tensor:
                return torch.where(mask[batch, query, key], score, float("-inf"))

            attended = cast(
                "Tensor",
                self._flex_attention(
                    q.transpose(1, 2),
                    k.transpose(1, 2),
                    v.transpose(1, 2),
                    score_mod=score_mod,
                    enable_gqa=q.shape[2] != k.shape[2],
                ),
            ).transpose(1, 2)
        else:
            attended = functional.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                attn_mask=mask[:, None],
                enable_gqa=q.shape[2] != k.shape[2],
            ).transpose(1, 2)
        offset = 0
        if prefix is not None:
            offset = prefix.shape[1]
            projected = vlm_layer.self_attn.o_proj(
                attended[:, :offset].reshape(*prefix.shape[:-1], -1).to(prefix.dtype),
            )
            prefix = prefix + projected  # noqa: PLR6104 -- avoid in-place writes on checkpointed activations
            prefix = prefix + vlm_layer.mlp(vlm_layer.post_attention_layernorm(prefix))  # noqa: PLR6104
            if visual is not None:
                prefix = self.vlm.model.language_model._deepstack_process(  # noqa: SLF001
                    prefix,
                    visual_pos_masks,
                    visual,
                )
        logits = None
        if suffix is not None:
            if cond is None:
                msg = "An action suffix needs its time embedding"
                raise ValueError(msg)
            suffix, logits = expert_layer.finish(suffix, attended[:, offset:], cond)
        return prefix, suffix, logits

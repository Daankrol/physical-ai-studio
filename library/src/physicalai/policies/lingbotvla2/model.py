# Copyright (C) 2026 Robbyant Inc.
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-VL and MoE action expert with continuous flow matching."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, cast, override

import torch
from torch import Tensor, nn
from torch.nn import functional
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

from physicalai.policies.base import Model
from physicalai.policies.mixins.peft import PeftModelMixin
from physicalai.policies.utils.weights import load_safetensors_weights

from .components.alignment import TaskTokenHead
from .components.joint import JointBackbone
from .components.layer import ExpertLayer

if TYPE_CHECKING:
    from pathlib import Path

    from .config import LingBotVLA2Config


def sample_upstream_time(batch_size: int, device: torch.device) -> Tensor:
    """Reproduce upstream sample_beta, which is not an exact Beta distribution.

    Returns:
        Scaled flow-matching times on the requested device.
    """
    first = torch.rand(batch_size, device=device).pow(1 / 1.5)
    second = torch.rand(batch_size, device=device)
    return first / (first + second) * 0.999 + 0.001


def _attention_mask(valid: Tensor, blocks: Tensor) -> Tensor:
    levels = blocks.long().cumsum(-1)
    return (levels[:, :, None] >= levels[:, None, :]) & valid[:, :, None] & valid[:, None, :]


def _masked_mean(error: Tensor, joint_mask: Tensor, action_is_pad: Tensor | None = None) -> Tensor:
    valid = joint_mask[:, None, :].expand_as(error)
    if action_is_pad is not None:
        valid = valid & ~action_is_pad[:, :, None]  # noqa: PLR6104 -- expanded mask cannot be written in place
    return (error * valid).sum() / valid.sum().clamp_min(1)


class _ExpertDecoder(nn.Module):
    def __init__(self, layers: nn.ModuleList, hidden_size: int) -> None:
        super().__init__()
        self.layers = layers
        self.norm = Qwen2RMSNorm(hidden_size)


class _ExpertStack(nn.Module):
    def __init__(self, model: _ExpertDecoder) -> None:
        super().__init__()
        self.model = model


class _JointModules(nn.Module):
    def __init__(self, vlm: Qwen3VLForConditionalGeneration, expert: _ExpertStack) -> None:
        super().__init__()
        self.qwenvl = vlm
        self.qwen_expert = expert


class LingBotVLA2Model(PeftModelMixin, Model):
    """Released LingBot-VLA 2.0 architecture; no data features or tokenizer dependencies."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        chunk_size: int = 50,
        num_denoise_steps: int = 10,
        max_state_dim: int = 55,
        max_action_dim: int = 55,
        num_layers: int = 36,
        hidden_size: int = 2560,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        vocab_size: int = 151936,
        intermediate_size: int = 9728,
        rope_theta: float = 5_000_000.0,
        mrope_section: tuple[int, int, int] = (24, 20, 20),
        vision_hidden_size: int = 1024,
        vision_intermediate_size: int = 4096,
        vision_num_heads: int = 16,
        vision_num_layers: int = 24,
        vision_deepstack_indexes: tuple[int, ...] = (5, 11, 17),
        vision_patch_size: int = 16,
        vision_spatial_merge_size: int = 2,
        vision_temporal_patch_size: int = 2,
        expert_hidden_size: int = 768,
        token_num_experts: int = 32,
        token_top_k: int = 4,
        token_moe_intermediate_size: int = 512,
        token_shared_intermediate_size: int = 704,
        routed_scaling_factor: float = 4.0,
        num_task_tokens: int = 8,
        num_query_backbone_tokens: int = 256,
        loss_type: str = "L1_fm",
        sequence_wise_loss_coeff: float = 1e-3,
        router_z_loss_coeff: float = 1e-4,
        attention_fp32: bool = True,
        training_recipe: str = "action_only",
        moe_backend: str = "dense",
        attention_backend: str = "sdpa",
        distillation_heads: bool = False,
        action_fp32: bool = False,
        alignment_feature_dim: int = 1024,
        alignment_num_heads: int = 4,
        alignment_head_dim: int = 32,
        depth_loss_weight: float = 0.004,
        future_depth_loss_weight: float = 0.004,
        video_loss_weight: float = 0.004,
        video_cosine_weight: float = 0.2,
    ) -> None:
        """Create the released VLM, joint decoder, and learned query tokens."""
        super().__init__()
        self.training_recipe = training_recipe
        self.distillation_heads = distillation_heads
        self.action_fp32 = action_fp32
        self.depth_loss_weight = depth_loss_weight
        self.future_depth_loss_weight = future_depth_loss_weight
        self.video_loss_weight = video_loss_weight
        self.video_cosine_weight = video_cosine_weight
        self.chunk_size = chunk_size
        self.num_denoise_steps = num_denoise_steps
        self.max_action_dim = max_action_dim
        self.num_task_tokens = num_task_tokens
        self.loss_type = loss_type
        self.sequence_wise_loss_coeff = sequence_wise_loss_coeff
        self.router_z_loss_coeff = router_z_loss_coeff
        self.token_top_k = token_top_k
        self.num_experts = token_num_experts
        self.vision_spatial_merge_size = vision_spatial_merge_size
        config = Qwen3VLConfig(
            text_config={
                "hidden_size": hidden_size,
                "num_hidden_layers": num_layers,
                "num_attention_heads": num_attention_heads,
                "num_key_value_heads": num_key_value_heads,
                "head_dim": head_dim,
                "vocab_size": vocab_size,
                "intermediate_size": intermediate_size,
                "eos_token_id": 151645,  # nosec B105 -- Qwen token ID, not a secret
                "rope_parameters": {
                    "rope_type": "default",
                    "rope_theta": rope_theta,
                    "mrope_section": list(mrope_section),
                    "mrope_interleaved": True,
                },
            },
            vision_config={
                "depth": vision_num_layers,
                "hidden_size": vision_hidden_size,
                "intermediate_size": vision_intermediate_size,
                "num_heads": vision_num_heads,
                "out_hidden_size": hidden_size,
                "deepstack_visual_indexes": list(vision_deepstack_indexes),
                "patch_size": vision_patch_size,
                "spatial_merge_size": vision_spatial_merge_size,
                "temporal_patch_size": vision_temporal_patch_size,
            },
        )
        vlm = Qwen3VLForConditionalGeneration(config)
        del vlm.lm_head
        layers = nn.ModuleList([
            ExpertLayer(
                expert_hidden_size,
                num_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                num_experts=token_num_experts,
                top_k=token_top_k,
                intermediate_size=token_moe_intermediate_size,
                shared_intermediate_size=token_shared_intermediate_size,
                scaling_factor=routed_scaling_factor,
            )
            for _ in range(num_layers)
        ])
        for layer in layers:
            cast("ExpertLayer", layer).mlp.experts.backend = moe_backend
        expert = _ExpertDecoder(layers, expert_hidden_size)
        self.qwenvl_with_expert = _JointModules(vlm, _ExpertStack(expert))
        self._joint = JointBackbone(vlm, expert, attention_fp32=attention_fp32, attention_backend=attention_backend)
        self.state_proj = nn.Linear(max_state_dim, expert_hidden_size)
        self.action_in_proj = nn.Linear(max_action_dim, expert_hidden_size)
        self.action_out_proj = nn.Linear(expert_hidden_size, max_action_dim)
        self.action_time_mlp_in = nn.Linear(2 * expert_hidden_size, expert_hidden_size)
        self.action_time_mlp_out = nn.Linear(expert_hidden_size, expert_hidden_size)
        self.depth_align_embs: nn.Parameter
        self.future_depth_align_embs: nn.Parameter
        self.current_video_align_embs: nn.Parameter
        self.future_video_align_embs: nn.Parameter
        for name in (
            "depth_align_embs",
            "future_depth_align_embs",
            "current_video_align_embs",
            "future_video_align_embs",
        ):
            self.register_parameter(name, nn.Parameter(torch.randn(num_query_backbone_tokens, hidden_size)))
        self.current_shared_task_proj = nn.Linear(2 * hidden_size, hidden_size)
        self.future_shared_task_proj = nn.Linear(2 * hidden_size, hidden_size)
        if distillation_heads:
            for name in (
                "depth_align_head",
                "future_depth_align_head",
                "current_video_align_head",
                "future_video_align_head",
            ):
                self.add_module(
                    name,
                    TaskTokenHead(hidden_size, alignment_feature_dim, alignment_num_heads, alignment_head_dim),
                )

    @classmethod
    def get_default_peft_targets(cls) -> str:
        """Adapt VLM linears and expert attention while keeping MoE experts fully trainable.

        Returns:
            Regex for PEFT target modules.
        """
        return (
            r"qwenvl_with_expert\.qwenvl\.model\.(?:visual|language_model)\..*\.(?:q_proj|k_proj|v_proj|o_proj|qkv|gate_proj|up_proj|down_proj|proj|fc1|fc2)$"
            r"|qwenvl_with_expert\.qwen_expert\.model\.layers\.\d+\.self_attn\.(?:q_proj|k_proj|v_proj|o_proj)$"
        )

    def enable_gradient_checkpointing(self) -> None:
        """Recompute joint attention and vision blocks during backward passes."""
        self._joint.gradient_checkpointing = True
        cast("Any", self.qwenvl_with_expert.qwenvl.model).visual.gradient_checkpointing_enable()

    def load_weights(self, index_path: str | Path) -> None:
        """Strictly load the released checkpoint without its training-only teacher heads."""

        def key_map(key: str) -> str | None:
            if not key.startswith("model."):
                return key
            if "_align_head." in key and not self.distillation_heads:
                return None
            return key.removeprefix("model.")

        load_safetensors_weights(self, index_path, key_map=key_map)
        # Upstream loads these tensors into bf16 modules before converting to fp32 for inference.
        with torch.no_grad():
            for name, parameter in self.named_parameters():
                if (
                    name.startswith("qwenvl_with_expert.qwenvl.model.language_model.")
                    or (name.startswith("qwenvl_with_expert.qwen_expert.model.layers.") and ".self_attn." in name)
                    or name == "qwenvl_with_expert.qwen_expert.model.norm.weight"
                ):
                    parameter.copy_(parameter.to(torch.bfloat16).to(parameter.dtype))

    @classmethod
    def from_config(cls, config: LingBotVLA2Config) -> LingBotVLA2Model:
        """Construct the model without copying dataset features or fetching hub assets.

        Returns:
            Model matching the architecture in the configuration.
        """
        fields = (
            "chunk_size",
            "num_denoise_steps",
            "max_state_dim",
            "max_action_dim",
            "num_layers",
            "hidden_size",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "vocab_size",
            "intermediate_size",
            "rope_theta",
            "mrope_section",
            "vision_hidden_size",
            "vision_intermediate_size",
            "vision_num_heads",
            "vision_num_layers",
            "vision_deepstack_indexes",
            "vision_patch_size",
            "vision_spatial_merge_size",
            "vision_temporal_patch_size",
            "expert_hidden_size",
            "token_num_experts",
            "token_top_k",
            "token_moe_intermediate_size",
            "token_shared_intermediate_size",
            "routed_scaling_factor",
            "num_task_tokens",
            "num_query_backbone_tokens",
            "loss_type",
            "sequence_wise_loss_coeff",
            "router_z_loss_coeff",
            "attention_fp32",
            "training_recipe",
            "moe_backend",
            "attention_backend",
            "distillation_heads",
            "action_fp32",
            "alignment_feature_dim",
            "alignment_num_heads",
            "alignment_head_dim",
            "depth_loss_weight",
            "future_depth_loss_weight",
            "video_loss_weight",
            "video_cosine_weight",
        )
        return cls(**{field: getattr(config, field) for field in fields})

    def _queries(self, current: Tensor, video: Tensor, projection: nn.Linear) -> Tensor:
        shape = (self.num_task_tokens, -1, current.shape[-1])
        pooled = torch.cat((current.view(shape).mean(1), video.view(shape).mean(1)), dim=-1)
        return projection(pooled)

    def embed_prefix(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, list[Tensor]]:  # noqa: PLR0914
        """Encode camera slots, text, and the two shared query segments.

        Returns:
            Prefix embeddings, validity, attention blocks, MRoPE ids, visual mask and deepstack features.
        """
        images, grid = batch["pixel_values"], batch["image_grid_thw"]
        masks = batch["img_masks"].bool()
        bsize, cameras = masks.shape
        vlm = self.qwenvl_with_expert.qwenvl
        vlm_model = cast("Any", vlm.model)
        grid = grid.reshape(-1, 3)
        vision = vlm_model.visual(images.reshape(-1, *images.shape[-2:]), grid_thw=grid)
        patch_count = int(grid[0].prod().item()) // self.vision_spatial_merge_size**2
        image_embeddings = vision.pooler_output.reshape(bsize, cameras, patch_count, -1)
        language = vlm_model.language_model.embed_tokens(batch["lang_tokens"])
        start = vlm_model.language_model.embed_tokens.weight[vlm.config.vision_start_token_id]
        end = vlm_model.language_model.embed_tokens.weight[vlm.config.vision_end_token_id]
        image_embeddings = torch.cat(
            (start.expand(bsize, cameras, 1, -1), image_embeddings, end.expand(bsize, cameras, 1, -1)),
            dim=2,
        )
        image_embeddings = image_embeddings.flatten(1, 2)
        current = self._queries(self.depth_align_embs, self.current_video_align_embs, self.current_shared_task_proj)
        future = self._queries(self.future_depth_align_embs, self.future_video_align_embs, self.future_shared_task_proj)
        queries = torch.cat((current, future), dim=0).to(language).expand(bsize, -1, -1)
        prefix = torch.cat((image_embeddings, language, queries), dim=1)
        image_valid = masks.repeat_interleave(patch_count + 2, dim=1)
        pad_mask = torch.cat(
            (
                image_valid,
                batch["lang_masks"].bool(),
                torch.ones(bsize, 2 * self.num_task_tokens, device=masks.device, dtype=torch.bool),
            ),
            dim=1,
        )
        image_ids = torch.full(
            (bsize, cameras, patch_count + 2),
            vlm.config.image_token_id,
            device=masks.device,
            dtype=torch.long,
        )
        image_ids[:, :, 0] = vlm.config.vision_start_token_id
        image_ids[:, :, -1] = vlm.config.vision_end_token_id
        fake_ids = torch.cat(
            (
                image_ids.flatten(1, 2),
                batch["lang_tokens"],
                torch.full(
                    (bsize, 2 * self.num_task_tokens),
                    cast("Any", vlm.config.text_config).eos_token_id,
                    device=masks.device,
                    dtype=torch.long,
                ),
            ),
            dim=1,
        )
        visual_mask = torch.zeros_like(pad_mask)
        image_positions = visual_mask[:, : cameras * (patch_count + 2)].view(bsize, cameras, patch_count + 2)
        image_positions[:, :, 1:-1] = masks[:, :, None]
        mm_types = torch.zeros_like(fake_ids)
        mm_types[:, : cameras * (patch_count + 2)] = image_positions.reshape(bsize, -1).long()
        positions, _ = vlm_model.get_rope_index(
            fake_ids,
            mm_types,
            image_grid_thw=grid[masks.reshape(-1)],
            attention_mask=pad_mask.long(),
        )
        positions = positions.masked_fill(~pad_mask[None], 1)  # Match transformers 4.57's padding MRoPE ids.
        deepstack = [
            features.reshape(bsize, cameras, patch_count, -1)[masks[:, :, None].expand(-1, -1, patch_count)]
            for features in vision.deepstack_features
        ]
        return prefix, pad_mask, torch.ones_like(pad_mask), positions, visual_mask, deepstack

    def embed_suffix(self, state: Tensor, actions: Tensor, time: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Embed state, noisy actions and time, with a bidirectional action block.

        Returns:
            Time embedding, suffix embeddings, and attention block markers.
        """
        dimension = self.action_in_proj.out_features
        fractions = torch.linspace(0, 1, dimension // 2, device=time.device)
        periods = 4e-3 * (4.0 / 4e-3) ** fractions
        angles = 2 * math.pi * time[:, None] / periods[None, :]
        time_emb = torch.cat((angles.sin(), angles.cos()), dim=-1).to(state.dtype)
        action_emb = self.action_in_proj(actions)
        action_time = torch.cat((action_emb, time_emb[:, None].expand_as(action_emb)), dim=-1)
        action_time = self.action_time_mlp_out(functional.silu(self.action_time_mlp_in(action_time)))
        suffix = torch.cat((self.state_proj(state)[:, None], action_time), dim=1)
        blocks = torch.zeros(suffix.shape[:2], dtype=torch.bool, device=state.device)
        blocks[:, :2] = True
        return time_emb, suffix, blocks

    @staticmethod
    def _suffix_positions(prefix_positions: Tensor, prefix_valid: Tensor, suffix_len: int) -> Tensor:
        prefix_max = prefix_positions.masked_fill(~prefix_valid[None], 0).amax((0, 2))
        positions = prefix_max[:, None] + torch.arange(1, suffix_len + 1, device=prefix_positions.device)
        return positions[None].expand(3, -1, -1)

    def _velocity(  # noqa: PLR0914
        self,
        batch: dict[str, Tensor],
        actions: Tensor,
        time: Tensor,
        prefix: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, list[Tensor]] | None = None,
        cache: dict[int, tuple[Tensor, Tensor]] | None = None,
    ) -> tuple[Tensor, list[Tensor], Tensor | None]:
        state = batch["state"]
        if self.action_fp32 and torch.is_autocast_enabled(state.device.type):
            msg = "FP32 action-expert training/inference requires autocast disabled (Lightning precision='32-true')"
            raise ValueError(msg)
        prefix_hidden = None
        cond, suffix, blocks = self.embed_suffix(state, actions, time)
        if prefix is None:
            valid = batch["prefix_valid"]
            positions = batch["prefix_positions"]
            mask = torch.cat(
                (valid[:, None, :].expand(-1, suffix.shape[1], -1), _attention_mask(torch.ones_like(blocks), blocks)),
                dim=-1,
            )
            suffix_positions = self._suffix_positions(positions, valid, suffix.shape[1])
            _, result, routers = self._joint(None, suffix, mask, suffix_positions, cond=cond, cache=cache)
        else:
            embeddings, valid, prefix_blocks, positions, visual_mask, deepstack = prefix
            all_valid = torch.cat((valid, torch.ones_like(blocks)), dim=1)
            mask = _attention_mask(all_valid, torch.cat((prefix_blocks, blocks), dim=1))
            full_positions = torch.cat((positions, self._suffix_positions(positions, valid, suffix.shape[1])), dim=-1)
            prefix_hidden, result, routers = self._joint(
                embeddings,
                suffix,
                mask,
                full_positions,
                cond=cond,
                visual_pos_masks=visual_mask,
                deepstack=deepstack,
            )
        if result is None:
            msg = "Joint backbone produced no action suffix"
            raise RuntimeError(msg)
        return self.action_out_proj(result[:, 1:]), routers, prefix_hidden

    def _alignment_losses(self, hidden: Tensor, batch: dict[str, Tensor]) -> dict[str, Tensor]:  # noqa: PLR0914
        grid = batch["image_grid_thw"][0, 0]
        image_count = int(grid.prod().item()) // self.vision_spatial_merge_size**2
        image_features = hidden[:, 1 : image_count + 1]
        current_task = hidden[:, -2 * self.num_task_tokens : -self.num_task_tokens]
        future_task = hidden[:, -self.num_task_tokens :]
        losses = {}
        branches = (
            ("depth", "depth_align_head", self.depth_align_embs, current_task, image_features),
            (
                "future_depth",
                "future_depth_align_head",
                self.future_depth_align_embs,
                future_task,
                image_features.detach(),
            ),
            ("current_video", "current_video_align_head", self.current_video_align_embs, current_task, image_features),
            (
                "future_video",
                "future_video_align_head",
                self.future_video_align_embs,
                future_task,
                image_features.detach(),
            ),
        )
        for name, head_name, queries, task, images in branches:
            target_key = f"{name}_target"
            if target_key not in batch:
                msg = f"Distillation requires {target_key}"
                raise ValueError(msg)
            head = getattr(self, head_name)
            context = torch.cat((images, task), dim=1)
            prediction = head(context, queries.to(context).expand(hidden.shape[0], -1, -1)).float()
            target = batch[target_key].detach().to(prediction)
            if prediction.shape != target.shape:
                msg = f"Incompatible {target_key} shape: {target.shape}, expected {prediction.shape}"
                raise ValueError(msg)
            if "video" in name:
                loss = functional.mse_loss(prediction, target)
                pred_norm = functional.normalize(prediction, dim=-1, eps=1e-6)
                target_norm = functional.normalize(target, dim=-1, eps=1e-6)
                cosine = 1 - functional.cosine_similarity(pred_norm, target_norm, dim=-1, eps=1e-6).mean()
                losses[f"{name}_mse_loss"] = loss
                losses[f"{name}_cosine_loss"] = cosine
                losses[f"{name}_loss"] = loss + self.video_cosine_weight * cosine
            else:
                losses[f"{name}_loss"] = functional.smooth_l1_loss(prediction, target)
        return losses

    def compute_loss(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, Tensor | float]]:  # noqa: PLR0914
        """Compute masked L1 flow loss and per-sequence MoE routing penalties.

        Returns:
            Differentiable loss and detached training metrics.

        Raises:
            RuntimeError: If the distillation path has no prefix hidden states.
        """
        actions = batch["action"]
        noise = batch["noise"] if "noise" in batch else torch.randn_like(actions)
        if "time" in batch:
            time = batch["time"]
        elif self.training_recipe == "upstream":
            time = sample_upstream_time(actions.shape[0], actions.device)
        else:
            time = torch.distributions.Beta(1.5, 1.0).sample((actions.shape[0],)).to(actions.device)
            time = time * 0.999 + 0.001
        interpolated = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
        prediction, routers, hidden = self._velocity(batch, interpolated, time, prefix=self.embed_prefix(batch))
        error = prediction - (noise - actions)
        flow = _masked_mean(
            error.abs() if self.loss_type == "L1_fm" else error.square(),
            batch["joint_mask"].bool(),
            batch.get("action_is_pad") if self.training_recipe == "action_only" else None,
        )
        seq_losses, z_losses = [], []
        for layer_logits in routers:
            logits = layer_logits.float()
            scores = logits.sigmoid()
            probability = scores / scores.sum(dim=-1, keepdim=True)
            selection = logits.topk(self.token_top_k, dim=-1).indices
            assignment = functional.one_hot(selection, self.num_experts).sum(dim=-2).float().detach()
            frequency = assignment.mean(dim=1) * (self.num_experts / self.token_top_k)
            seq_losses.append((frequency * probability.mean(dim=1)).sum(dim=-1).mean())
            z_losses.append(logits.logsumexp(dim=-1).square().mean())
        seq = self.sequence_wise_loss_coeff * torch.stack(seq_losses).mean() if seq_losses else flow.new_zeros(())
        z = self.router_z_loss_coeff * torch.stack(z_losses).mean() if z_losses else flow.new_zeros(())
        loss = flow + seq + z
        alignment = {}
        if self.distillation_heads:
            if hidden is None:
                msg = "Distillation requires prefix hidden states"
                raise RuntimeError(msg)
            alignment = self._alignment_losses(hidden, batch)
            for name, weight in (
                ("depth", self.depth_loss_weight),
                ("future_depth", self.future_depth_loss_weight),
                ("current_video", self.video_loss_weight),
                ("future_video", self.video_loss_weight),
            ):
                weighted = alignment[f"{name}_loss"] * weight
                alignment[f"{name}_loss_weighted"] = weighted
                loss += weighted
        metrics = {"loss": loss, "flow_loss": flow, "moe_seq_loss": seq, "moe_z_loss": z, **alignment}
        return loss, {key: value.detach() for key, value in metrics.items()}

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Denoise a full canonical action chunk with the cached image/language prefix.

        Returns:
            Actions of shape (batch, chunk_size, max_action_dim).
        """
        prefix = self.embed_prefix(batch)
        embeddings, valid, blocks, positions, visual_mask, deepstack = prefix
        cache: dict[int, tuple[Tensor, Tensor]] = {}
        self._joint(
            embeddings,
            None,
            _attention_mask(valid, blocks),
            positions,
            cache=cache,
            fill_cache=True,
            visual_pos_masks=visual_mask,
            deepstack=deepstack,
        )
        current = (
            torch.randn(
                batch["state"].shape[0],
                self.chunk_size,
                self.max_action_dim,
                device=batch["state"].device,
                dtype=batch["state"].dtype,
            )
            if noise is None
            else noise
        )
        inference_batch = {**batch, "prefix_valid": valid, "prefix_positions": positions}
        dt = current.new_tensor(-1.0 / self.num_denoise_steps)
        time = current.new_tensor(1.0)
        for _ in range(self.num_denoise_steps):
            velocity, _, _ = self._velocity(inference_batch, current, time.expand(current.shape[0]), cache=cache)
            current = current + dt * velocity  # noqa: PLR6104 -- do not mutate supplied noise
            time += dt
        return current

    @torch.no_grad()
    @override
    def compute_val_loss(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, Tensor | float]]:
        """Measure denoised action MSE and the flow objective.

        Returns:
            Action MSE and detached validation metrics.
        """
        prediction = self.predict_action_chunk(batch, noise=torch.zeros_like(batch["action"]))
        action_mse = _masked_mean(
            (prediction - batch["action"]).square(),
            batch["joint_mask"].bool(),
            batch.get("action_is_pad"),
        )
        _, metrics = self.compute_loss(batch)
        return action_mse, {**metrics, "loss": action_mse.detach(), "action_mse": action_mse.detach()}

    @override
    def forward(self, batch: dict[str, Any]) -> Tensor | tuple[Tensor, dict[str, Tensor | float]]:
        """Train with flow matching or sample a full chunk in evaluation mode.

        Returns:
            Loss tuple in training, canonical actions in evaluation.
        """
        return self.compute_loss(batch) if self.training else self.predict_action_chunk(batch)

    @property
    def action_delta_indices(self) -> list[int]:
        """Future action offsets."""
        return list(range(self.chunk_size))

    @property
    def image_delta_indices(self) -> list[int] | None:
        """Request future RGB only for the distillation training path."""
        return [0, self.chunk_size - 1] if self.distillation_heads else None

    @property
    def observation_delta_indices(self) -> None:
        """Use only the current observation."""
        return None

    @property
    def reward_delta_indices(self) -> None:
        """Do not train on rewards."""
        return None

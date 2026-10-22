# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Observation to fixed-size Qwen3-VL images, chat tokens, and canonical joints."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch
from torch import Tensor, nn
from torch.nn import functional
from transformers import AutoTokenizer

from physicalai.data import FeatureType
from physicalai.data.observation import ACTION, STATE, TASK
from physicalai.policies.utils.normalization import FeatureNormalizeTransform, NormalizationType

from .layout import JointLayout, camera_name
from .normalization import UpstreamNormalizeTransform

if TYPE_CHECKING:
    from .config import LingBotVLA2Config


class LingBotVLA2Preprocessor(nn.Module):
    """Prepare the released three-slot image and 55-dimensional action layout."""

    def __init__(self, config: LingBotVLA2Config) -> None:
        """Build normalization buffers and fixed joint mappings from dataset features."""
        super().__init__()
        self.camera_map = config.camera_map
        self.camera_slots = config.camera_slots
        self.image_size = config.image_size
        self.patch_size = config.vision_patch_size
        self.merge_size = config.vision_spatial_merge_size
        self.temporal_patch_size = config.vision_temporal_patch_size
        self.tokenizer_name_or_path = config.tokenizer_name_or_path
        self.tokenizer_revision = config.tokenizer_revision
        self.tokenizer_max_length = config.tokenizer_max_length
        self._tokenizer: Any = None
        features = {
            feature.name: feature
            for feature in config.input_features + config.output_features
            if feature.ftype in {FeatureType.STATE, FeatureType.ACTION} and feature.name is not None
        }
        mode = NormalizationType(config.normalization_mode)
        self.upstream = config.training_recipe == "upstream"
        self.distillation_heads = config.distillation_heads
        normalizer = UpstreamNormalizeTransform if self.upstream else FeatureNormalizeTransform
        self.normalizer = normalizer(features, {FeatureType.STATE: mode, FeatureType.ACTION: mode})
        self.relative_pairs = [
            (
                [index for start, end in config.action_map[group] for index in range(start, end)],
                [index for start, end in config.state_map[group] for index in range(start, end)],
            )
            for group in config.relative_action_groups
        ]
        self.chunk_size = config.chunk_size
        self.dataset_fps = config.dataset_fps
        self.teacher_depth_key = config.teacher_depth_key
        self.teacher_depth_aligned = config.teacher_depth_aligned
        self.teacher_targets_cached = config.teacher_targets_cached
        self.state_layout = JointLayout(config)
        self.action_layout = JointLayout(config, action=True)

    def _images(self, batch: dict[str, Any], batch_size: int, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:  # noqa: PLR0914
        if self.image_size % (self.patch_size * self.merge_size):
            msg = "image_size must be divisible by vision_patch_size * vision_spatial_merge_size"
            raise ValueError(msg)
        image_by_slot = {slot: name for name, slot in self.camera_map.items()}
        images = []
        masks = []
        for slot in self.camera_slots:
            name = image_by_slot.get(slot)
            image = batch.get(name, batch.get(f"images.{camera_name(name)}")) if name is not None else None
            if image is None:
                image = torch.full((batch_size, 3, self.image_size, self.image_size), -1.0, device=device)
                masks.append(False)
            else:
                if image.ndim == 5:  # noqa: PLR2004 -- (batch, time, channel, height, width)
                    image = image[:, 0 if self.upstream or self.distillation_heads else -1]
                image = image.float().div(255) if image.dtype == torch.uint8 else image.float()
                image = functional.interpolate(
                    image,
                    size=(self.image_size, self.image_size),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )
                if self.upstream:
                    image = (image * 255).round().clamp(0, 255) / 255
                image = 2 * image - 1
                masks.append(True)
            images.append(image)
        stacked = torch.stack(images, dim=1)
        bsize, num_images, channels, height, width = stacked.shape
        grid_h, grid_w = height // self.patch_size, width // self.patch_size
        patches = stacked.reshape(
            bsize * num_images,
            channels,
            grid_h // self.merge_size,
            self.merge_size,
            self.patch_size,
            grid_w // self.merge_size,
            self.merge_size,
            self.patch_size,
        ).permute(0, 2, 5, 3, 6, 1, 4, 7)
        patches = patches.unsqueeze(6).expand(-1, -1, -1, -1, -1, -1, self.temporal_patch_size, -1, -1)
        patches = patches.reshape(
            bsize,
            num_images,
            grid_h * grid_w,
            channels * self.temporal_patch_size * self.patch_size**2,
        )
        grid = torch.tensor((1, grid_h, grid_w), dtype=torch.long, device=device).expand(bsize, num_images, 3)
        image_masks = torch.tensor(masks, dtype=torch.bool, device=device).expand(bsize, -1)
        return patches, image_masks, grid

    def _language(self, task: list[str] | str, device: torch.device) -> tuple[Tensor, Tensor]:
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name_or_path,
                revision=self.tokenizer_revision,
                local_files_only=True,
            )
        tokenizer = cast("Any", self._tokenizer)
        prompts = [task] if isinstance(task, str) else task
        conversations = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=False,
            )
            for prompt in prompts
        ]
        encoded = tokenizer(
            conversations,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer_max_length,
            return_tensors="pt",
        )
        return encoded["input_ids"].to(device), encoded["attention_mask"].to(device).bool()

    def forward(self, batch: dict[str, Any]) -> dict[str, Tensor]:  # noqa: C901, PLR0912, PLR0914, PLR0915
        """Normalize and pack a batch without modifying its original tensors.

        Returns:
            Named tensors consumed by the model (and action targets when present).

        Raises:
            ValueError: If distillation frames or effective FPS are missing.
        """
        state = batch[STATE]
        if state.ndim == 3:  # noqa: PLR2004 -- (batch, time, joints)
            state = state[:, 0 if self.upstream or self.distillation_heads else -1]
        actions = batch.get(ACTION)
        if actions is not None and self.relative_pairs:
            actions = actions.clone()
            for action_indices, state_indices in self.relative_pairs:
                actions[..., action_indices] -= state[:, None, state_indices]
        normalized = self.normalizer({STATE: state, ACTION: actions})
        state, state_mask = self.state_layout(normalized[STATE])
        pixels, img_masks, grid = self._images(batch, state.shape[0], state.device)
        tokens, lang_masks = self._language(batch.get(TASK) or [""] * state.shape[0], state.device)
        output = {
            "state": state,
            "state_joint_mask": state_mask.expand(state.shape[0], -1),
            "pixel_values": pixels,
            "img_masks": img_masks,
            "image_grid_thw": grid,
            "lang_tokens": tokens,
            "lang_masks": lang_masks,
        }
        if normalized[ACTION] is not None:
            output["action"], joint_mask = self.action_layout(normalized[ACTION])
            output["joint_mask"] = joint_mask.expand(state.shape[0], -1)
            extra = batch.get("extra")
            pad = extra.get("action_is_pad") if isinstance(extra, dict) else batch.get("extra.action_is_pad")
            if pad is not None:
                output["action_is_pad"] = pad.bool()
            if self.distillation_heads and self.teacher_targets_cached:
                for target in ("depth", "future_depth", "current_video", "future_video"):
                    key = f"extra.teacher_features.{target}"
                    if key not in batch:
                        msg = f"Missing cached teacher feature: {target}"
                        raise ValueError(msg)
                    output[f"{target}_target"] = batch[key]
            elif self.distillation_heads:
                name = next((name for name, slot in self.camera_map.items() if slot == self.camera_slots[0]), None)
                image = batch.get(name, batch.get(f"images.{camera_name(name)}")) if name is not None else None
                if image is None or image.ndim != 5 or image.shape[1] != 2:  # noqa: PLR2004 -- B, current/future, C, H, W
                    msg = "Distillation requires current/future RGB for the first camera slot"
                    raise ValueError(msg)
                if self.teacher_depth_key is not None:
                    depth = batch.get(f"depths.{self.teacher_depth_key}")
                    if not self.teacher_depth_aligned:
                        msg = "Teacher depth must be calibrated/aligned to the first RGB camera"
                        raise ValueError(msg)
                    if (
                        depth is None
                        or depth.ndim != 5  # noqa: PLR2004 -- B,T,C,H,W
                        or depth.shape[1:3] != (2, 1)
                        or depth.shape[-2:] != image.shape[-2:]
                    ):
                        msg = "Teacher depth needs aligned current/future (B,2,1,H,W) maps in metres"
                        raise ValueError(msg)
                    resized = functional.interpolate(
                        depth.float().flatten(0, 1),
                        size=(self.image_size, self.image_size),
                        mode="nearest",
                    )
                    resized = resized.reshape(depth.shape[0], 2, 1, self.image_size, self.image_size)
                    output["teacher_current_depth"], output["teacher_future_depth"] = resized[:, 0], resized[:, 1]
                image = image.float() if image.dtype == torch.uint8 else image.float() * 255
                rgb = (
                    functional
                    .interpolate(
                        image.flatten(0, 1),
                        size=(self.image_size, self.image_size),
                        mode="bilinear",
                        align_corners=False,
                        antialias=True,
                    )
                    .round()
                    .clamp(0, 255)
                    .reshape(*image.shape[:3], self.image_size, self.image_size)
                )
                output["teacher_current_rgb"] = rgb[:, :1]
                output["teacher_future_rgb"] = rgb[:, 1:]
                fps = batch.get("extra.future_video_effective_fps")
                if fps is None and self.dataset_fps is not None:
                    fps = state.new_full((state.shape[0],), self.dataset_fps / max(1, self.chunk_size - 1))
                if fps is None:
                    msg = "Distillation requires dataset effective video FPS metadata"
                    raise ValueError(msg)
                output["teacher_fps"] = fps
        return output

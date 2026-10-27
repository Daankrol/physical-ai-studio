# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""LingBot-VLA 2.0 architecture and dataset-to-checkpoint layout."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Literal

from physicalai.config import Config

from physicalai.data import Feature, FeatureType
from physicalai.policies.mixins.peft import PeftConfigMixin

DEFAULT_JOINT_LAYOUT = (
    ("arm.position", 14),
    ("end.position", 14),
    ("effector.position", 2),
    ("waist.position", 4),
    ("head.position", 2),
    ("base.position", 3),
    ("hand.position", 12),
)


@dataclass(frozen=True, kw_only=True)
class LingBotVLA2Config(PeftConfigMixin, Config):
    """Checkpoint geometry and the explicit mapping of dataset joints to canonical slots."""

    input_features: list[Feature] = field(default_factory=list)
    output_features: list[Feature] = field(default_factory=list)
    joint_layout: tuple[tuple[str, int], ...] = DEFAULT_JOINT_LAYOUT
    state_map: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    action_map: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    camera_slots: tuple[str, ...] = ("camera_top", "camera_wrist_left", "camera_wrist_right")
    camera_map: dict[str, str] = field(default_factory=dict)

    chunk_size: int = 50
    n_action_steps: int = 50
    num_denoise_steps: int = 10
    max_state_dim: int = 55
    max_action_dim: int = 55
    tokenizer_name_or_path: str = "Qwen/Qwen3-VL-4B-Instruct"
    tokenizer_revision: str | None = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
    tokenizer_max_length: int = 72
    image_size: int = 256

    num_layers: int = 36
    hidden_size: int = 2560
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    vocab_size: int = 151936
    intermediate_size: int = 9728
    rope_theta: float = 5_000_000.0
    mrope_section: tuple[int, int, int] = (24, 20, 20)
    vision_hidden_size: int = 1024
    vision_intermediate_size: int = 4096
    vision_num_heads: int = 16
    vision_num_layers: int = 24
    vision_deepstack_indexes: tuple[int, ...] = (5, 11, 17)
    vision_patch_size: int = 16
    vision_spatial_merge_size: int = 2
    vision_temporal_patch_size: int = 2
    expert_hidden_size: int = 768
    expert_num_attention_heads: int = 32
    expert_num_key_value_heads: int = 8
    expert_head_dim: int = 128
    token_moe_layers: tuple[int, ...] = tuple(range(36))
    token_num_experts: int = 32
    token_top_k: int = 4
    token_moe_intermediate_size: int = 512
    token_shared_intermediate_size: int = 704
    routed_scaling_factor: float = 4.0
    num_task_tokens: int = 8
    num_query_backbone_tokens: int = 256
    query_layout: Literal["shared_current_future"] = "shared_current_future"
    distillation_heads: bool = False
    training_recipe: Literal["action_only", "upstream"] = "action_only"
    moe_backend: Literal["dense", "triton"] = "dense"
    attention_backend: Literal["sdpa", "flex"] = "sdpa"
    relative_action_groups: tuple[str, ...] = ()
    dataset_fps: float | None = None
    alignment_feature_dim: int = 1024
    alignment_num_heads: int = 4
    alignment_head_dim: int = 32
    depth_loss_weight: float = 0.004
    future_depth_loss_weight: float = 0.004
    video_loss_weight: float = 0.004
    video_cosine_weight: float = 0.2
    loss_type: Literal["fm", "L1_fm"] = "L1_fm"
    sequence_wise_loss_coeff: float = 1e-3
    router_z_loss_coeff: float = 1e-4
    action_fp32: bool = False
    attention_fp32: bool = True
    normalization_mode: Literal["QUANTILES", "MEAN_STD"] = "QUANTILES"
    freeze_vision_encoder: bool = False
    train_expert_only: bool = False

    _PEFT_EXCLUSIVE_FLAGS: ClassVar[dict[str, object]] = {
        "train_expert_only": False,
        "freeze_vision_encoder": False,
    }

    def __post_init__(self) -> None:
        """Validate architecture and reject incomplete dataset mappings.

        Raises:
            ValueError: If the architecture or dataset mapping is inconsistent.
        """
        super().__post_init__()
        self._validate_geometry()
        if not self.state_map or not self.action_map:
            msg = "Supply state_map and action_map; training infers them from recorded joint names"
            raise ValueError(msg)

        self._validate_map(self.state_map, self.input_features, FeatureType.STATE, "state_map")
        self._validate_map(self.action_map, self.output_features, FeatureType.ACTION, "action_map")
        for group in self.relative_action_groups:
            if group not in self.action_map or group not in self.state_map:
                msg = f"Relative action group {group} needs both state and action mappings"
                raise ValueError(msg)
            if sum(end - start for start, end in self.action_map[group]) != sum(
                end - start for start, end in self.state_map[group]
            ):
                msg = f"Relative action group {group} needs equal state and action widths"
                raise ValueError(msg)
        images = {feature.name for feature in self.input_features if feature.ftype == FeatureType.VISUAL}
        if images and (set(self.camera_map) != images or len(images) > len(self.camera_slots)):
            msg = "camera_map must select each dataset image exactly once (or select cameras before training)"
            raise ValueError(msg)

    def _validate_geometry(self) -> None:  # noqa: C901, PLR0912, PLR0915
        if self.chunk_size < 1 or not 1 <= self.n_action_steps <= self.chunk_size or self.num_denoise_steps < 1:
            msg = "chunk_size, n_action_steps and num_denoise_steps must be positive; n_action_steps <= chunk_size"
            raise ValueError(msg)
        if self.moe_backend not in {"dense", "triton"} or self.attention_backend not in {"sdpa", "flex"}:
            msg_0 = "Unknown LingBot attention or MoE backend"
            raise ValueError(msg_0)
        if self.dataset_fps is not None and self.dataset_fps <= 0:
            msg = "dataset_fps must be positive"
            raise ValueError(msg)
        if min(self.alignment_feature_dim, self.alignment_num_heads, self.alignment_head_dim) < 1:
            msg = "Alignment head dimensions must be positive"
            raise ValueError(msg)
        if self.training_recipe not in {"action_only", "upstream"}:
            msg = f"Unsupported training_recipe: {self.training_recipe}"
            raise ValueError(msg)
        if set(self.relative_action_groups) - {"arm.position"}:
            msg = "Only arm.position supports relative actions; grippers and poses stay absolute"
            raise ValueError(msg)
        if self.loss_type not in {"fm", "L1_fm"}:
            msg = f"Unsupported flow loss_type: {self.loss_type}"
            raise ValueError(msg)
        if self.normalization_mode not in {"QUANTILES", "MEAN_STD"}:
            msg = f"Unsupported normalization_mode: {self.normalization_mode}"
            raise ValueError(msg)
        if not self.camera_slots:
            msg = "At least one camera slot is required"
            raise ValueError(msg)
        if self.query_layout != "shared_current_future":
            msg = f"Unsupported query_layout: {self.query_layout}"
            raise ValueError(msg)
        if (
            self.num_layers < 1
            or len(set(self.token_moe_layers)) != len(self.token_moe_layers)
            or any(layer < 0 or layer >= self.num_layers for layer in self.token_moe_layers)
        ):
            msg = "token_moe_layers must contain distinct layer indices in [0, num_layers)"
            raise ValueError(msg)
        if sum(self.mrope_section) != self.head_dim // 2:
            msg = "mrope_section must partition half the attention head width"
            raise ValueError(msg)
        if tuple(sorted(self.token_moe_layers)) != tuple(range(self.num_layers)):
            msg = "Only a MoE expert in every layer is supported"
            raise ValueError(msg)
        if (self.num_attention_heads, self.num_key_value_heads, self.head_dim) != (
            self.expert_num_attention_heads,
            self.expert_num_key_value_heads,
            self.expert_head_dim,
        ):
            msg = "VLM and action expert attention geometry must match"
            raise ValueError(msg)
        if self.token_top_k < 1 or self.token_top_k > self.token_num_experts:
            msg = "token_top_k must be within [1, token_num_experts]"
            raise ValueError(msg)
        if any(index < 0 or index >= self.vision_num_layers for index in self.vision_deepstack_indexes):
            msg = "vision_deepstack_indexes must refer to vision layers"
            raise ValueError(msg)
        if self.num_query_backbone_tokens % self.num_task_tokens:
            msg = "num_query_backbone_tokens must be divisible by num_task_tokens"
            raise ValueError(msg)
        if len(set(self.camera_slots)) != len(self.camera_slots) or set(self.camera_map.values()) - set(
            self.camera_slots,
        ):
            msg = "camera_map must target distinct known camera_slots"
            raise ValueError(msg)
        if len(set(self.camera_map.values())) != len(self.camera_map):
            msg = "Two dataset cameras cannot occupy the same camera slot"
            raise ValueError(msg)
        if any(width < 1 for _, width in self.joint_layout) or len({name for name, _ in self.joint_layout}) != len(
            self.joint_layout,
        ):
            msg = "joint_layout needs unique, positive-width groups"
            raise ValueError(msg)
        if sum(width for _, width in self.joint_layout) > min(self.max_state_dim, self.max_action_dim):
            msg = "joint_layout exceeds the canonical state or action width"
            raise ValueError(msg)

    def _validate_map(
        self,
        mapping: dict[str, list[tuple[int, int]]],
        features: list[Feature],
        ftype: FeatureType,
        name: str,
    ) -> None:
        """Check canonical and dataset bounds.

        Raises:
            ValueError: If a slice is invalid, overlaps, or leaves dimensions unmapped.
        """
        widths = dict(self.joint_layout)
        if set(mapping) - set(widths):
            msg = f"{name} contains unknown groups: {set(mapping) - set(widths)}"
            raise ValueError(msg)
        feature = [item for item in features if item.ftype == ftype]
        if len(feature) > 1:
            msg = f"{name} needs exactly one {ftype} feature"
            raise ValueError(msg)
        width = feature[0].shape[-1] if feature and feature[0].shape else None
        used: set[int] = set()
        for group, slices in mapping.items():
            filled = 0
            for start, end in slices:
                if start < 0 or end <= start or (width is not None and end > width):
                    msg = f"Invalid dataset slice ({start}, {end}) in {name}"
                    raise ValueError(msg)
                indices = set(range(start, end))
                if used & indices:
                    msg = f"Overlapping dataset slices in {name}"
                    raise ValueError(msg)
                used |= indices
                filled += end - start
            if filled > widths[group]:
                msg = f"{name} exceeds canonical width of {group}"
                raise ValueError(msg)
        if width is not None and used != set(range(width)):
            msg = f"{name} must map all {width} dataset dimensions exactly once"
            raise ValueError(msg)

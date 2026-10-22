# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Resolve LingBot weights and supported architecture settings without constructing a model."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import snapshot_download

_REVISIONS = {
    "robbyant/lingbot-vla-v2-6b": "11c703bf6a5c1f45b3b69168482da11fdbba53d7",
    "robbyant/lingbot-vla-v2-6b-robotwin": "0451855729ec904f970600e0aec8b84661423afe",
}
_TRAIN_FIELDS = {
    "chunk_size",
    "max_state_dim",
    "max_action_dim",
    "expert_hidden_size",
    "token_num_experts",
    "token_top_k",
    "token_moe_intermediate_size",
    "token_shared_intermediate_size",
    "routed_scaling_factor",
    "token_moe_layers",
    "loss_type",
    "router_z_loss_coeff",
    "sequence_wise_loss_coeff",
    "action_fp32",
    "tokenizer_max_length",
}


def resolve_pretrained(name_or_path: str | Path, *, revision: str | None = None) -> tuple[dict[str, Any], Path]:  # noqa: PLR0912
    """Return supported architecture overrides and the safetensors index path.

    Returns:
        Architecture overrides and the weights index, without loading any tensors.

    Raises:
        ValueError: If a checkpoint requests an unsupported architecture or uses a floating revision.
        TypeError: If the recipe's model or training fields have an invalid type.
        FileNotFoundError: If the checkpoint has no unambiguous safetensors index.
    """
    local = Path(name_or_path)
    if not local.exists():
        pinned_revision = revision or _REVISIONS.get(str(name_or_path))
        if pinned_revision is None or re.fullmatch(r"[0-9a-f]{40}", pinned_revision) is None:
            msg = "Hub checkpoints need a pinned 40-character commit revision"
            raise ValueError(msg)
        snapshot_download(  # nosec B615 -- hardcoded repo, pinned revision, and explicit asset allowlist
            repo_id="Qwen/Qwen3-VL-4B-Instruct",
            revision="ebb281ec70b05090aa6165b016eac8ec08e71b17",
            allow_patterns=[
                "config.json",
                "tokenizer*",
                "added_tokens.json",
                "special_tokens_map.json",
                "vocab.json",
                "merges.txt",
            ],
        )
        local = Path(
            snapshot_download(  # nosec B615 -- pinned revision and explicit asset allowlist
                repo_id=str(name_or_path),
                revision=pinned_revision,
                allow_patterns=[
                    "*.json",
                    "*.txt",
                    "*.model",
                    "*.safetensors",
                    "*.safetensors.index.json",
                    "*.yaml",
                    "checkpoints/*/hf_ckpt/*.safetensors",
                    "checkpoints/*/hf_ckpt/*.json",
                ],
                ignore_patterns=["depth/*", "dino_video/*"],
            ),
        )
    if local.is_file():
        local = local.parent
    matches = [
        path
        for path in [
            local / "model.safetensors.index.json",
            *local.glob("checkpoints/*/hf_ckpt/model.safetensors.index.json"),
        ]
        if path.is_file()
    ]
    if len(matches) != 1:
        msg = f"Expected one LingBot safetensors index in {local}, found {len(matches)}"
        raise FileNotFoundError(msg)
    recipe = local / "lingbotvla_cli.yaml"
    if not recipe.exists():
        return {}, matches[0]
    content = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    if not isinstance(content, Mapping):
        msg = f"Invalid LingBot recipe in {recipe}"
        raise TypeError(msg)
    model, train = content.get("model", {}), content.get("train", {})
    if not isinstance(model, Mapping) or not isinstance(train, Mapping):
        msg = f"Invalid model/train fields in {recipe}"
        raise TypeError(msg)
    align = train.get("align_params", {})
    if not isinstance(align, Mapping):
        msg = f"Invalid align_params in {recipe}"
        raise TypeError(msg)
    video = align.get("video", {})
    depth = align.get("depth", {})
    if not isinstance(video, Mapping) or not isinstance(depth, Mapping):
        msg = f"Invalid depth/video alignment in {recipe}"
        raise TypeError(msg)
    unsupported = (
        model.get("use_lm_head", False),
        model.get("adanorm_time", True) is False,
        model.get("final_norm_adanorm", False),
        train.get("use_moe") is False,
        align.get("mode", "query") != "query",
        depth.get("use_future_depth", True) is False,
        depth.get("block_future_depth_to_action", False),
        video.get("block_suffix_to_future_video", False),
        video.get("share_future_depth_query", True) is False,
        video.get("use_shared_future_task_proj", True) is False,
        video.get("use_current_shared_task_proj", True) is False,
    )
    if any(unsupported):
        msg = "Unsupported LingBot checkpoint architecture or query layout"
        raise ValueError(msg)
    overrides = {key: train[key] for key in _TRAIN_FIELDS & train.keys()}
    if "my_tokenizer_max_length" in train:
        overrides["tokenizer_max_length"] = train["my_tokenizer_max_length"]
    if "num_task_tokens" in align:
        overrides["num_task_tokens"] = align["num_task_tokens"]
    if "num_backbone_tokens" in depth:
        overrides["num_query_backbone_tokens"] = depth["num_backbone_tokens"]
    if "token_moe_layers" in overrides:
        overrides["token_moe_layers"] = tuple(overrides["token_moe_layers"])
    return overrides, matches[0]

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Strict mapping of sharded released-format weights into a small module."""

import json
from dataclasses import replace

import pytest
import torch
from safetensors.torch import save_file

from physicalai.policies.lingbotvla2.model import LingBotVLA2Model
from physicalai.policies.utils.weights import load_safetensors_weights

from .test_policy import tiny_config


def test_pretrained_load_matches_upstreams_bf16_to_fp32_rounding(tmp_path) -> None:
    model = LingBotVLA2Model.from_config(replace(tiny_config(), vocab_size=256))
    weights = {f"model.{key}": torch.full_like(value, 0.3333) for key, value in model.state_dict().items()}
    save_file(weights, tmp_path / "model.safetensors")
    model.load_weights(tmp_path)
    rounded = torch.tensor(0.3333).bfloat16().float()
    assert model.qwenvl_with_expert.qwenvl.model.language_model.embed_tokens.weight[0, 0] == rounded
    assert model.qwenvl_with_expert.qwen_expert.model.layers[0].self_attn.q_proj.weight[0, 0] == rounded
    assert model.action_in_proj.weight[0, 0] == torch.tensor(0.3333)


def test_hub_snapshot_symlinks_into_blobs_are_allowed(tmp_path) -> None:
    hub = tmp_path / "hub"
    snapshot = hub / "models--robot" / "snapshots" / "commit"
    blobs = hub / "blobs" / "3d"
    snapshot.mkdir(parents=True)
    blobs.mkdir(parents=True)
    model = torch.nn.Linear(2, 1)
    save_file({"weight": torch.ones(1, 2), "bias": torch.zeros(1)}, blobs / "digest")
    (snapshot / "model-00001.safetensors").symlink_to(blobs / "digest")
    index = snapshot / "model.safetensors.index.json"
    index.write_text(
        json.dumps({
            "weight_map": {
                "weight": "model-00001.safetensors",
                "bias": "model-00001.safetensors",
            }
        })
    )
    load_safetensors_weights(model, index)
    torch.testing.assert_close(model.weight, torch.ones(1, 2))


def test_mapped_shards_skip_teacher_and_reject_missing_keys(tmp_path) -> None:
    module = torch.nn.Linear(3, 2)
    first = {"model.weight": torch.arange(6.0).reshape(2, 3), "model.depth_align_head.weight": torch.ones(1)}
    second = {"model.bias": torch.tensor([1.0, 2.0])}
    save_file(first, tmp_path / "first.safetensors")
    save_file(second, tmp_path / "second.safetensors")
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(
        json.dumps({
            "weight_map": {
                **dict.fromkeys(first, "first.safetensors"),
                **dict.fromkeys(second, "second.safetensors"),
            }
        })
    )

    def key_map(key: str) -> str | None:
        return None if "_align_head." in key else key.removeprefix("model.")

    load_safetensors_weights(module, index, key_map=key_map)
    torch.testing.assert_close(module.weight, first["model.weight"])
    torch.testing.assert_close(module.bias, second["model.bias"])
    index.write_text(json.dumps({"weight_map": dict.fromkeys(first, "first.safetensors")}))
    with pytest.raises(RuntimeError, match="Missing"):
        load_safetensors_weights(module, index, key_map=key_map)
    index.write_text(json.dumps({"weight_map": {"model.weight": "../outside.safetensors"}}))
    with pytest.raises(ValueError, match="shard path"):
        load_safetensors_weights(module, index, key_map=key_map)
    index.write_text(json.dumps({"weight_map": {"model.weight": "..\\outside.safetensors"}}))
    with pytest.raises(ValueError, match="shard path"):
        load_safetensors_weights(module, index, key_map=key_map)
    (tmp_path / "escape.safetensors").symlink_to(tmp_path.parent / "other.safetensors")
    index.write_text(json.dumps({"weight_map": {"model.weight": "escape.safetensors"}}))
    with pytest.raises(ValueError, match="shard path"):
        load_safetensors_weights(module, index, key_map=key_map)

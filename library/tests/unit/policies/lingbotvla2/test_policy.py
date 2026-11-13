# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Policy construction and checkpoint restore must never contact the pretrained hub."""

from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch

from physicalai.data import Feature, FeatureType, NormalizationParameters, Observation
from physicalai.policies.lingbotvla2.config import LingBotVLA2Config
from physicalai.policies.lingbotvla2.policy import LingBotVLA2

from .test_processors import StubTokenizer

SO101 = {"arm.position": [(0, 5)], "effector.position": [(5, 6)]}


def tiny_config() -> LingBotVLA2Config:
    stats = NormalizationParameters(q01=[0.0] * 6, q99=[10.0] * 6)
    state = Feature(name="state", ftype=FeatureType.STATE, shape=(6,), normalization_data=stats)
    action = Feature(name="action", ftype=FeatureType.ACTION, shape=(6,), normalization_data=stats)
    image = Feature(name="images.front", ftype=FeatureType.VISUAL, shape=(3, 32, 32))
    return LingBotVLA2Config(
        state_map=SO101, action_map=SO101,
        input_features=[state, image],
        output_features=[action],
        camera_map={"images.front": "camera_top"},
        camera_slots=("camera_top",),
        image_size=32,
        chunk_size=2,
        n_action_steps=1,
        num_denoise_steps=2,
        num_layers=2,
        token_moe_layers=(0, 1),
        hidden_size=64,
        intermediate_size=128,
        head_dim=16,
        mrope_section=(4, 2, 2),
        num_attention_heads=4,
        num_key_value_heads=2,
        expert_num_attention_heads=4,
        expert_num_key_value_heads=2,
        expert_head_dim=16,
        expert_hidden_size=32,
        token_num_experts=4,
        token_top_k=2,
        token_moe_intermediate_size=16,
        token_shared_intermediate_size=16,
        vision_hidden_size=32,
        vision_intermediate_size=64,
        vision_num_heads=4,
        vision_num_layers=2,
        vision_deepstack_indexes=(0, 1),
        num_task_tokens=2,
        num_query_backbone_tokens=4,
    )


def test_configure_once_and_checkpoint_restore_without_hub() -> None:
    config = tiny_config()
    with patch("physicalai.policies.lingbotvla2.policy.resolve_pretrained", side_effect=AssertionError("hub touched")):
        policy = LingBotVLA2.from_config(config)
        model = policy.model
        policy.configure_model()
        assert policy.model is model
        checkpoint: dict[str, Any] = {}
        policy.on_save_checkpoint(checkpoint)
        restored = LingBotVLA2(pretrained_name_or_path=None)
        restored.on_load_checkpoint(checkpoint)
        assert restored.config == config
        restored.load_state_dict(policy.state_dict())
    policy._preprocessor._tokenizer = StubTokenizer()
    observation = Observation(
        state=torch.ones(1, 6), action=torch.ones(1, 2, 6), images={"front": torch.rand(1, 3, 32, 32)}, task=["move"]
    )
    policy.eval()
    output = policy.predict_action_chunk(observation)
    assert output.shape == (1, 1, 6)
    assert torch.isfinite(output).all()


def test_renaming_camera_rebuilds_processors_without_reloading_model() -> None:
    policy = LingBotVLA2.from_config(tiny_config())
    model = policy.model
    policy.rename_features({"images.front": "images.top"})
    assert policy.model is model
    assert policy.config is not None
    assert policy.config.camera_map == {"images.top": "camera_top"}
    assert "images.top" in [feature.name for feature in policy.config.input_features]
    with pytest.raises(ValueError, match="unknown|known"):
        policy.rename_features({"images.unknown": "images.top"})


def test_camera_mapping_accepts_studio_feature_keys() -> None:
    images = [Feature(name="top", ftype=FeatureType.VISUAL, shape=(3, 32, 32))]
    assert LingBotVLA2._camera_map(images, {"images.top": 0}) == {"top": "camera_top"}
    assert LingBotVLA2._camera_map(images, {"top": 0}) == {"top": "camera_top"}


def test_lora_keeps_expert_trainable_and_restores_adapters() -> None:
    config = replace(tiny_config(), lora_enabled=True)
    policy = LingBotVLA2.from_config(config)
    assert any("lora_A" in name for name, _ in policy.model.named_parameters())
    assert policy.model.qwenvl_with_expert.qwen_expert.model.layers[0].mlp.experts.gate_proj.requires_grad
    checkpoint: dict[str, Any] = {}
    policy.on_save_checkpoint(checkpoint)
    restored = LingBotVLA2(pretrained_name_or_path=None)
    restored.on_load_checkpoint(checkpoint)
    restored.load_state_dict(policy.state_dict())


def test_unresolved_features_defer_construction() -> None:
    policy = LingBotVLA2(pretrained_name_or_path=None)
    policy.configure_model()
    assert policy.model is None
    assert policy._preprocessor is None
    with pytest.raises(RuntimeError, match="not initialized"):
        _ = policy.config


def test_statistics_refresh_processors_but_architecture_changes_fail(dummy_dataset) -> None:
    config = tiny_config()
    policy = LingBotVLA2.from_config(config)
    model, preprocessor = policy.model, policy._preprocessor
    policy.configure_model()
    assert policy._preprocessor is preprocessor
    stats = NormalizationParameters(q01=[1.0] * 6, q99=[2.0] * 6)
    state = replace(config.input_features[0], normalization_data=stats)
    checkpoint = {"model_config": replace(config, input_features=[state, config.input_features[1]]).to_dict()}
    policy.on_load_checkpoint(checkpoint)
    assert policy.model is model
    assert policy._preprocessor is not preprocessor
    assert policy.config.input_features[0].normalization_data == stats
    checkpoint = {"model_config": replace(config, chunk_size=3, n_action_steps=1).to_dict()}
    with pytest.raises(ValueError, match="does not match"):
        policy.on_load_checkpoint(checkpoint)
    policy._trainer = SimpleNamespace(datamodule=SimpleNamespace(train_dataset=dummy_dataset()))
    with pytest.raises(ValueError, match="do not match"):
        policy.setup("fit")
    assert policy.model is model


def test_bf16_weights_train_with_kahan_and_scale_only_lora_and_experts() -> None:
    from physicalai.train.optimizers import AdamWBF16Kahan

    policy = LingBotVLA2.from_config(replace(tiny_config(), dtype="bfloat16", lora_enabled=True))
    policy._preprocessor._tokenizer = StubTokenizer()
    assert policy.model.state_proj.weight.dtype == torch.bfloat16
    assert policy.model.qwenvl_with_expert.qwenvl.model.language_model.rotary_emb.inv_freq.dtype == torch.float32
    policy._trainer = SimpleNamespace(estimated_stepping_batches=10)
    optimizer = policy.configure_optimizers()["optimizer"]
    assert isinstance(optimizer, AdamWBF16Kahan)
    rates = {id(p): group["lr"] for group in optimizer.param_groups for p in group["params"]}
    named = dict(policy.model.named_parameters())
    lr = policy.optimizer_lr
    assert rates[id(named["state_proj.weight"])] == pytest.approx(lr)
    expert = next(p for n, p in named.items() if ".mlp.experts." in n and p.requires_grad)
    assert rates[id(expert)] == pytest.approx(lr * 2**0.5)  # sqrt(4 experts / top-2)
    lora = next(p for n, p in named.items() if "lora_A" in n and ".visual." not in n)
    assert rates[id(lora)] == pytest.approx(lr * policy.config.lora_lr_scale)

    observation = Observation(
        state=torch.ones(1, 6), action=torch.ones(1, 2, 6), images={"front": torch.rand(1, 3, 32, 32)}, task=["move"]
    )
    policy.train()
    with torch.autocast("cpu", dtype=torch.bfloat16):  # as under a bf16-mixed trainer
        policy(observation)
    loss, _ = policy(observation)
    loss.backward()
    optimizer.step()
    policy.eval()
    output = policy.predict_action_chunk(observation)
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()
    with pytest.raises(ValueError, match="bfloat16"):
        replace(tiny_config(), dtype="bfloat16", training_recipe="upstream")

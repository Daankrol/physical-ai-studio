# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Upstream recipe transforms and backward-compatible action-only defaults."""

from dataclasses import replace

import pytest
import torch

from physicalai.data import Feature, FeatureType, NormalizationParameters
from physicalai.policies.lingbotvla2.config import LingBotVLA2Config
from physicalai.policies.lingbotvla2.model import sample_upstream_time
from physicalai.policies.lingbotvla2.postprocessor import make_lingbotvla2_processors
from tests.unit.policies.lingbotvla2.test_processors import StubTokenizer


def config(**changes) -> LingBotVLA2Config:
    stats = NormalizationParameters(mean=[0.] * 6, std=[1.] * 6, q01=[0.] * 6, q99=[0.] * 6)
    base = LingBotVLA2Config(
        robot_layout="so101", training_recipe="upstream", relative_action_groups=("arm.position",),
        normalization_mode="MEAN_STD", image_size=32, chunk_size=2, n_action_steps=2, dataset_fps=30,
        input_features=[
            Feature(name="state", ftype=FeatureType.STATE, shape=(6,), normalization_data=stats),
            Feature(name="images.front", ftype=FeatureType.VISUAL, shape=(3, 32, 32)),
        ],
        output_features=[Feature(name="action", ftype=FeatureType.ACTION, shape=(6,), normalization_data=stats)],
        camera_map={"images.front": "camera_top"},
    )
    return replace(base, **changes)


def test_powered_uniform_sampler_and_rng() -> None:
    torch.manual_seed(123)
    first, second = torch.rand(8).pow(1 / 1.5), torch.rand(8)
    expected = first / (first + second) * .999 + .001
    torch.manual_seed(123)
    torch.testing.assert_close(sample_upstream_time(8, torch.device("cpu")), expected, atol=0, rtol=0)


def test_relative_arm_roundtrip_and_no_future_leakage() -> None:
    pre, post = make_lingbotvla2_processors(config())
    pre._tokenizer = StubTokenizer()
    state = torch.arange(6.).reshape(1, 6)
    actions = torch.full((1, 2, 6), 8.)
    image = torch.stack((torch.zeros(1, 3, 32, 32), torch.ones(1, 3, 32, 32)), dim=1)
    batch = {"state": state, "action": actions, "images.front": image, "task": ["move"]}
    processed = pre(batch)
    torch.testing.assert_close(post(processed["action"], state), actions)
    torch.testing.assert_close(processed["action"][0, :, 28], actions[0, :, 5] / (1 + 1e-6))
    changed = pre({**batch, "images.front": image.flip(1) * 0})
    torch.testing.assert_close(processed["pixel_values"], changed["pixel_values"])
    torch.testing.assert_close(batch["action"], actions)
    with pytest.raises(ValueError, match="current raw state"):
        post(processed["action"])


def test_nested_rgb_feature_names_are_not_truncated() -> None:
    cfg = config()
    cfg = replace(cfg, input_features=[
        replace(feature, name="front.rgb") if feature.ftype == FeatureType.VISUAL else feature
        for feature in cfg.input_features
    ], camera_map={"front.rgb": "camera_top"})
    pre, _ = make_lingbotvla2_processors(cfg)
    pre._tokenizer = StubTokenizer()
    batch = pre({"state": torch.zeros(1,6), "images.front.rgb": torch.ones(1,3,32,32), "task": ["move"]})
    assert batch["img_masks"][0,0]
    assert batch["pixel_values"][0,0].eq(1).all()


def test_per_horizon_action_statistics_roundtrip() -> None:
    cfg = config()
    stats = NormalizationParameters(mean=[[1.] * 6, [2.] * 6], std=[[1.] * 6, [3.] * 6])
    cfg = replace(cfg, output_features=[replace(cfg.output_features[0], normalization_data=stats)])
    pre, post = make_lingbotvla2_processors(cfg)
    pre._tokenizer = StubTokenizer()
    state = torch.full((1,6), 3.)
    actions = torch.full((1,2,6), 7.)
    output = pre({"state": state, "action": actions, "task": ["move"]})
    torch.testing.assert_close(post(output["action"], state), actions)


def test_upstream_quantiles_include_additive_epsilon() -> None:
    pre, post = make_lingbotvla2_processors(config(normalization_mode="QUANTILES", relative_action_groups=()))
    pre._tokenizer = StubTokenizer()
    batch = {"state": torch.zeros(1, 6), "action": torch.full((1, 2, 6), .1), "task": ["move"]}
    output = pre(batch)
    assert torch.isfinite(output["action"]).all()
    torch.testing.assert_close(output["action"][0, 0, :5], torch.full((5,), 199999.))
    torch.testing.assert_close(post(output["action"]), batch["action"])


@pytest.mark.parametrize("short_name", [False, True])
def test_teacher_frames_are_separate_and_missing_future_fails(short_name) -> None:
    cfg = config(distillation_heads=True)
    if short_name:
        cfg = replace(cfg, camera_map={"front": "camera_top"}, input_features=[
            replace(feature, name="front") if feature.ftype == FeatureType.VISUAL else feature
            for feature in cfg.input_features
        ])
    pre, _ = make_lingbotvla2_processors(cfg)
    pre._tokenizer = StubTokenizer()
    batch = {"state": torch.zeros(1, 6), "action": torch.zeros(1, 2, 6), "task": ["move"]}
    with pytest.raises(ValueError, match="current/future RGB"):
        pre({**batch, "images.front": torch.zeros(1, 3, 32, 32)})
    images = torch.stack((torch.zeros(1, 3, 32, 32), torch.ones(1, 3, 32, 32)), dim=1)
    output = pre({**batch, "images.front": images})
    assert output["teacher_current_rgb"].eq(0).all()
    assert output["teacher_future_rgb"].eq(255).all()
    assert output["teacher_fps"].item() == 30
    assert output["pixel_values"][0, 0].eq(-1).all()

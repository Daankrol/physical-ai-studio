# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Opt-in training step with online depth/video targets from the pinned teachers."""

import os
from dataclasses import replace

import pytest
import torch

from physicalai.data import Observation
from physicalai.policies.lingbotvla2.policy import LingBotVLA2
from tests.unit.policies.lingbotvla2.test_processors import StubTokenizer
from tests.unit.policies.lingbotvla2.test_training import training_config


@pytest.mark.integration
@pytest.mark.requires_download
@pytest.mark.skipif(
    os.environ.get("RUN_LINGBOT_TEACHERS") != "1" or not torch.cuda.is_available(),
    reason="opt-in CUDA teacher weights",
)
def test_online_teachers_train_student_from_rgb() -> None:
    torch.manual_seed(123)
    config = replace(
        training_config(), image_size=256, dataset_fps=30,
        num_query_backbone_tokens=256, alignment_feature_dim=1024,
        moe_backend="triton", attention_backend="flex", action_fp32=True,
    )
    policy = LingBotVLA2.from_config(config, gradient_checkpointing=True).cuda()
    policy._preprocessor._tokenizer = StubTokenizer()
    batch = Observation(
        state=torch.rand(1, 6, device="cuda"), action=torch.rand(1, 2, 6, device="cuda"),
        images={"front": torch.rand(1, 2, 3, 256, 256, device="cuda")}, task=["move"],
    )
    loss, metrics = policy(batch)
    loss.backward()
    assert loss.isfinite()
    assert policy.model.depth_align_head.projector.proj_out.weight.grad is not None
    assert all(f"{name}_loss_weighted" in metrics for name in ("depth", "future_depth", "current_video", "future_video"))
    assert policy.teachers is not None
    for teacher in (policy.teachers.moge, policy.teachers.depth, policy.teachers.video):
        assert not teacher.training
        assert all(not parameter.requires_grad and parameter.grad is None for parameter in teacher.parameters())
    assert not any("teachers" in name for name in policy.state_dict())

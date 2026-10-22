# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Opt-in comparison against upstream's released pretrain weights and dense MoE path.

Both references were captured with Robbyant lingbot-vla-v2 at be969b8, torch 2.8,
transformers 4.57.3, fp32 eager attention and a dense fp32 implementation of its
fused MoE. The shipped Triton MoE uses approximate matmuls and does not meet
this tolerance on H100; this test checks the model math, not that kernel.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from physicalai.data import Feature, FeatureType, NormalizationParameters, Observation
from physicalai.policies.lingbotvla2.config import LingBotVLA2Config
from physicalai.policies.lingbotvla2.model import LingBotVLA2Model
from physicalai.policies.lingbotvla2.preprocessor import LingBotVLA2Preprocessor
from physicalai.policies.lingbotvla2.pretrained import resolve_pretrained


@pytest.mark.integration
@pytest.mark.requires_download
@pytest.mark.skipif(
    os.environ.get("RUN_LINGBOT_PARITY") != "1" or not torch.cuda.is_available(),
    reason="opt-in H100 parity",
)
@pytest.mark.parametrize("variant", ["pretrain", "robotwin"])
def test_released_checkpoints_match_upstream_dense_reference(variant: str) -> None:
    """Compare both released normalized chunks at fixed observations and noise."""
    torch.set_float32_matmul_precision("highest")
    stats = NormalizationParameters(q01=[0.0] * 6, q99=[1.0] * 6)
    config = LingBotVLA2Config(
        robot_layout="so101",
        input_features=[
            Feature(name="state", ftype=FeatureType.STATE, shape=(6,), normalization_data=stats),
            Feature(name="images.front", ftype=FeatureType.VISUAL, shape=(3, 256, 256)),
        ],
        output_features=[Feature(name="action", ftype=FeatureType.ACTION, shape=(6,), normalization_data=stats)],
        camera_map={"images.front": "camera_top"},
    )
    checkpoint_id = "robbyant/lingbot-vla-v2-6b" + ("-robotwin" if variant == "robotwin" else "")
    _, weights = resolve_pretrained(checkpoint_id)
    model = LingBotVLA2Model.from_config(config)
    model.load_weights(weights)
    model.to("cuda").eval()
    observation = Observation.from_dict(
        {
            "state": torch.zeros(1, 6),
            "images.front": torch.zeros(1, 3, 256, 256),
            "task": ["move"],
        },
    )
    batch = LingBotVLA2Preprocessor(config)(observation.to_dict())
    batch = {key: tensor.to("cuda") for key, tensor in batch.items()}
    noise = torch.randn(1, 50, 55, generator=torch.Generator().manual_seed(9876)).to("cuda")
    with torch.no_grad():
        prediction = model.predict_action_chunk(batch, noise=noise)
    reference_path = Path(__file__).parent / "fixtures" / f"lingbotvla2_{variant}_eager_dense.safetensors"
    reference = load_file(reference_path)["actions"].to(prediction)
    torch.testing.assert_close(prediction, reference, atol=2e-4, rtol=1e-4)

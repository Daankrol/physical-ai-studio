# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Check uint8/float images and normalized action scattering without hub access."""

from typing import Any

import torch
from transformers import Qwen2VLImageProcessor

from physicalai.data import Feature, FeatureType, NormalizationParameters
from physicalai.policies.lingbotvla2.config import LingBotVLA2Config
from physicalai.policies.lingbotvla2.postprocessor import make_lingbotvla2_processors

SO101 = {"arm.position": [(0, 5)], "effector.position": [(5, 6)]}


class StubTokenizer:
    def save_pretrained(self, path):
        from pathlib import Path

        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "tokenizer.json").write_text("{}")

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        return messages[0]["content"]

    def __call__(self, prompts, **kwargs):
        return {
            "input_ids": torch.ones(len(prompts), 3, dtype=torch.long),
            "attention_mask": torch.ones(len(prompts), 3, dtype=torch.long),
        }


def test_joint_and_image_processors_roundtrip() -> None:
    stats = NormalizationParameters(q01=[0.0] * 6, q99=[10.0] * 6)
    state = Feature(name="state", ftype=FeatureType.STATE, shape=(6,), normalization_data=stats)
    action = Feature(name="action", ftype=FeatureType.ACTION, shape=(6,), normalization_data=stats)
    image = Feature(name="images.front", ftype=FeatureType.VISUAL, shape=(3, 32, 32))
    config = LingBotVLA2Config(
        state_map=SO101, action_map=SO101,
        input_features=[state, image],
        output_features=[action],
        camera_map={"images.front": "top"},
        camera_slots=("top", "wrist"),
        image_size=32,
        chunk_size=2,
        n_action_steps=1,
    )
    pre, post = make_lingbotvla2_processors(config)
    pre._tokenizer = StubTokenizer()
    float_image = torch.rand(1, 3, 32, 32)
    batch: dict[str, Any] = {
        "state": torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]),
        "action": torch.ones(1, 2, 6) * 7,
        "images.front": float_image,
        "task": ["move"],
        "extra.action_is_pad": torch.tensor([[False, True]]),
    }
    output = pre(batch)
    assert output["pixel_values"].shape == (1, 2, 4, 1536)
    assert output["img_masks"].tolist() == [[True, False]]
    assert output["action_is_pad"].tolist() == [[False, True]]
    assert output["state_joint_mask"].sum() == 6
    torch.testing.assert_close(post(output["action"]), batch["action"][:, :1])
    assert output["pixel_values"][0, 1].eq(-1).all()
    image_processor = Qwen2VLImageProcessor()
    expected, _, _ = image_processor.patchify(float_image * 2 - 1, patch_size=16, merge_size=2, temporal_patch_size=2)
    torch.testing.assert_close(output["pixel_values"][0, 0], expected[0])
    batch["images.front"] = (float_image * 255).round().to(torch.uint8)
    torch.testing.assert_close(
        pre(batch)["pixel_values"][0, 0],
        pre({**batch, "images.front": batch["images.front"].float() / 255})["pixel_values"][0, 0],
    )

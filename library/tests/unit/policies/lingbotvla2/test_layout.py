# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint joint layout must preserve dataset order and leave absent groups masked."""

from dataclasses import replace

import pytest
import torch

from physicalai.data import Feature, FeatureType
from physicalai.policies.lingbotvla2.config import LingBotVLA2Config
from physicalai.policies.lingbotvla2.layout import JointLayout


def features(width: int) -> tuple[list[Feature], list[Feature]]:
    return (
        [Feature(ftype=FeatureType.STATE, shape=(width,), name="observation.state")],
        [Feature(ftype=FeatureType.ACTION, shape=(width,), name="action")],
    )


@pytest.mark.parametrize("layout,width", [("so101", 6), ("bimanual_6dof", 14)])
def test_scatter_gather_and_mask(layout: str, width: int) -> None:
    inputs, outputs = features(width)
    config = LingBotVLA2Config(robot_layout=layout, input_features=inputs, output_features=outputs)
    value = torch.arange(width, dtype=torch.float32).reshape(1, 1, width)
    for mapper in (JointLayout(config), JointLayout(config, action=True)):
        canonical, mask = mapper(value)
        assert canonical.shape == (1, 1, 55)
        torch.testing.assert_close(mapper.gather(canonical), value)
        assert canonical[..., ~mask].count_nonzero() == 0
        assert mask.count_nonzero() == width
        if layout == "so101":
            assert canonical[0, 0, 28] == value[0, 0, 5]
        else:
            torch.testing.assert_close(canonical[0, 0, 0:12], value[0, 0, [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]])
            torch.testing.assert_close(canonical[0, 0, 28:30], value[0, 0, [6, 13]])


def test_roundtrip_config_preserves_maps() -> None:
    inputs, outputs = features(14)
    config = LingBotVLA2Config(robot_layout="bimanual_6dof", input_features=inputs, output_features=outputs)
    restored = LingBotVLA2Config.from_dict(config.to_dict())
    torch.testing.assert_close(
        JointLayout(restored).gather(JointLayout(restored)(torch.arange(14))[0]), torch.arange(14)
    )
    assert restored.action_map == config.action_map


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({}, "Supply state_map"),
        ({"n_action_steps": 51, "robot_layout": "so101"}, "n_action_steps"),
        ({"robot_layout": "so101", "num_layers": 2}, "token_moe_layers"),
        ({"robot_layout": "so101", "expert_head_dim": 64}, "attention geometry"),
        ({"robot_layout": "so101", "camera_map": {"front": "not_a_slot"}}, "camera_map"),
        ({"robot_layout": "so101", "training_recipe": "unknown"}, "training_recipe"),
        ({"robot_layout": "so101", "relative_action_groups": ("end.position",)}, "Only arm.position"),
        ({"robot_layout": "so101", "loss_type": "other"}, "flow loss_type"),
        ({"robot_layout": "so101", "normalization_mode": "none"}, "normalization_mode"),
        ({"robot_layout": "so101", "camera_slots": ()}, "camera slot"),
        ({"state_map": {"arm.position": [(0, 5), (4, 6)]}, "action_map": {"arm.position": [(0, 6)]}}, "Overlapping"),
        ({"state_map": {"arm.position": [(0, 5)]}, "action_map": {"arm.position": [(0, 6)]}}, "must map all"),
        ({"state_map": {"arm.position": [(0, 15)]}, "action_map": {"arm.position": [(0, 6)]}}, "Invalid dataset slice"),
    ],
)
def test_invalid_layout_is_rejected(changes: dict, match: str) -> None:
    inputs, outputs = features(6)
    with pytest.raises(ValueError, match=match):
        LingBotVLA2Config(input_features=inputs, output_features=outputs, **changes)


def test_maps_must_match_features_after_replacement() -> None:
    inputs, outputs = features(6)
    config = LingBotVLA2Config(robot_layout="so101", input_features=inputs, output_features=outputs)
    with pytest.raises(ValueError, match="must map all"):
        replace(config, input_features=features(7)[0])


def test_camera_selection_is_explicit() -> None:
    inputs, outputs = features(6)
    inputs.append(Feature(ftype=FeatureType.VISUAL, shape=(3, 256, 256), name="front"))
    with pytest.raises(ValueError, match="camera_map must select"):
        LingBotVLA2Config(robot_layout="so101", input_features=inputs, output_features=outputs)
    config = LingBotVLA2Config(
        robot_layout="so101", input_features=inputs, output_features=outputs, camera_map={"front": "camera_top"}
    )
    assert config.camera_map["front"] == "camera_top"

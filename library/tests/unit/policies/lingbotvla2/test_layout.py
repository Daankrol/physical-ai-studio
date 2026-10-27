# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Joint groups come from recorded joint names; the checkpoint layout preserves dataset order."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from physicalai.data import Feature, FeatureType
from physicalai.data.lerobot.dataset import _LeRobotDatasetAdapter
from physicalai.policies.lingbotvla2.config import LingBotVLA2Config
from physicalai.policies.lingbotvla2.data import recorded_joint_maps
from physicalai.policies.lingbotvla2.layout import JointLayout, joint_map
from physicalai.policies.lingbotvla2.policy import LingBotVLA2

SO101 = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
# WidowX AI; ReBot B601 and StarArm 102 also report six arm joints followed by "gripper".
SIX_DOF = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll", "gripper"]


def pos(names: list[str], prefix: str = "") -> list[str]:
    return [f"{prefix}{name}.pos" for name in names]


def features(width: int) -> tuple[list[Feature], list[Feature]]:
    return (
        [Feature(ftype=FeatureType.STATE, shape=(width,), name="observation.state")],
        [Feature(ftype=FeatureType.ACTION, shape=(width,), name="action")],
    )


def config_for(names: list[str], **changes) -> LingBotVLA2Config:
    inputs, outputs = features(len(names))
    groups = joint_map(names)
    return LingBotVLA2Config(state_map=groups, action_map=groups, input_features=inputs, output_features=outputs, **changes)


@pytest.mark.parametrize(
    ("names", "expected"),
    [
        (pos(SO101), {"arm.position": [(0, 5)], "effector.position": [(5, 6)]}),
        (pos(SIX_DOF), {"arm.position": [(0, 6)], "effector.position": [(6, 7)]}),
        (
            pos(SO101, "left_") + pos(SO101, "right_"),
            {"arm.position": [(0, 5), (6, 11)], "effector.position": [(5, 6), (11, 12)]},
        ),
        (
            pos(SIX_DOF, "left_") + pos(SIX_DOF, "right_"),
            {"arm.position": [(0, 6), (7, 13)], "effector.position": [(6, 7), (13, 14)]},
        ),
        (
            pos(SO101, "right_") + pos(SO101, "left_"),
            {"arm.position": [(6, 11), (0, 5)], "effector.position": [(11, 12), (5, 6)]},
        ),
        ([f"main_{name}" for name in SO101], {"arm.position": [(0, 5)], "effector.position": [(5, 6)]}),
        (pos(SO101[:-1]), {"arm.position": [(0, 5)]}),
    ],
    ids=["so101", "six_dof", "bimanual_so101", "bimanual_six_dof", "right_recorded_first", "legacy_prefix", "no_gripper"],
)
def test_joint_groups_follow_recorded_names(names: list[str], expected: dict) -> None:
    assert joint_map(names) == expected


@pytest.mark.parametrize(
    "names",
    [
        pos(SO101) + ["gripper.pos"],
        pos(SO101) + pos(SO101, "left_"),
        [f"{name}.vel" for name in SO101],
        [*pos(SO101), "tool_gripper.pos"],
    ],
    ids=["duplicate", "mixed_prefixes", "velocities", "two_grippers"],
)
def test_unrecognized_names_require_explicit_maps(names: list[str]) -> None:
    with pytest.raises(ValueError, match="Cannot infer LingBot joint groups"):
        joint_map(names)


@pytest.mark.parametrize("names", [pos(SO101), pos(SIX_DOF, "left_") + pos(SIX_DOF, "right_")])
def test_scatter_gather_and_mask(names: list[str]) -> None:
    width = len(names)
    config = config_for(names)
    value = torch.arange(width, dtype=torch.float32).reshape(1, 1, width)
    for mapper in (JointLayout(config), JointLayout(config, action=True)):
        canonical, mask = mapper(value)
        assert canonical.shape == (1, 1, 55)
        torch.testing.assert_close(mapper.gather(canonical), value)
        assert canonical[..., ~mask].count_nonzero() == 0
        assert mask.count_nonzero() == width
        if width == 6:
            assert canonical[0, 0, 28] == value[0, 0, 5]
        else:
            torch.testing.assert_close(canonical[0, 0, 0:12], value[0, 0, [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]])
            torch.testing.assert_close(canonical[0, 0, 28:30], value[0, 0, [6, 13]])


def test_recorded_names_build_maps_and_missing_names_fail() -> None:
    names = pos(SO101, "left_") + pos(SO101, "right_")
    recorded = {"observation.state": {"names": names}, "action": {"names": names}}
    state_map, action_map = recorded_joint_maps(_LeRobotDatasetAdapter.from_lerobot(SimpleNamespace(features=recorded)))
    assert state_map == action_map == joint_map(names)
    unnamed = {"observation.state": {"names": None}, "action": {"names": names}}
    with pytest.raises(ValueError, match="no recorded joint names"):
        recorded_joint_maps(_LeRobotDatasetAdapter.from_lerobot(SimpleNamespace(features=unnamed)))


def test_training_setup_infers_maps_from_dataset(dummy_dataset, monkeypatch) -> None:
    class NamedDataset(dummy_dataset):
        @property
        def raw_features(self) -> dict:
            features = dict(super().raw_features)
            features["observation.state"] = {**features["observation.state"], "names": pos(["a", "b", "c", "gripper"])}
            features["action"] = {**features["action"], "names": pos(["a", "gripper"])}
            return features

    policy = LingBotVLA2(pretrained_name_or_path=None)
    monkeypatch.setattr(policy, "configure_model", lambda: None)
    policy._trainer = SimpleNamespace(datamodule=SimpleNamespace(train_dataset=NamedDataset()))
    policy.setup("fit")
    assert policy.state_map == {"arm.position": [(0, 3)], "effector.position": [(3, 4)]}
    assert policy.action_map == {"arm.position": [(0, 1)], "effector.position": [(1, 2)]}


def test_roundtrip_config_preserves_maps() -> None:
    names = pos(SIX_DOF, "left_") + pos(SIX_DOF, "right_")
    config = config_for(names)
    restored = LingBotVLA2Config.from_dict(config.to_dict())
    torch.testing.assert_close(
        JointLayout(restored).gather(JointLayout(restored)(torch.arange(14))[0]), torch.arange(14)
    )
    assert restored.action_map == config.action_map


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"n_action_steps": 51}, "n_action_steps"),
        ({"num_layers": 2}, "token_moe_layers"),
        ({"expert_head_dim": 64}, "attention geometry"),
        ({"camera_map": {"front": "not_a_slot"}}, "camera_map"),
        ({"training_recipe": "unknown"}, "training_recipe"),
        ({"relative_action_groups": ("end.position",)}, "Only arm.position"),
        ({"loss_type": "other"}, "flow loss_type"),
        ({"normalization_mode": "none"}, "normalization_mode"),
        ({"camera_slots": ()}, "camera slot"),
    ],
)
def test_invalid_config_is_rejected(changes: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        config_for(pos(SO101), **changes)


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({}, "Supply state_map and action_map"),
        ({"state_map": {"arm.position": [(0, 5), (4, 6)]}, "action_map": {"arm.position": [(0, 6)]}}, "Overlapping"),
        ({"state_map": {"arm.position": [(0, 5)]}, "action_map": {"arm.position": [(0, 6)]}}, "must map all"),
        ({"state_map": {"arm.position": [(0, 15)]}, "action_map": {"arm.position": [(0, 6)]}}, "Invalid dataset slice"),
    ],
)
def test_invalid_maps_are_rejected(changes: dict, match: str) -> None:
    inputs, outputs = features(6)
    with pytest.raises(ValueError, match=match):
        LingBotVLA2Config(input_features=inputs, output_features=outputs, **changes)


def test_maps_must_match_features_after_replacement() -> None:
    config = config_for(pos(SO101))
    with pytest.raises(ValueError, match="must map all"):
        replace(config, input_features=features(7)[0])


def test_camera_selection_is_explicit() -> None:
    inputs, outputs = features(6)
    inputs.append(Feature(ftype=FeatureType.VISUAL, shape=(3, 256, 256), name="front"))
    groups = joint_map(pos(SO101))
    with pytest.raises(ValueError, match="camera_map must select"):
        LingBotVLA2Config(state_map=groups, action_map=groups, input_features=inputs, output_features=outputs)
    config = LingBotVLA2Config(
        state_map=groups, action_map=groups, input_features=inputs, output_features=outputs,
        camera_map={"front": "camera_top"},
    )
    assert config.camera_map["front"] == "camera_top"
    with pytest.raises(ValueError, match="image_key_reorder_map"):
        LingBotVLA2._camera_map(inputs, None)

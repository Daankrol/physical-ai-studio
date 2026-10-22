# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Full-objective student losses, statistics, checkpointing, and optimizer updates."""

from copy import deepcopy
from dataclasses import replace
from typing import Any

import pytest
import torch

from physicalai.policies.lingbotvla2.data import training_statistics, adopt_training_statistics
from physicalai.policies.lingbotvla2.model import LingBotVLA2Model
from physicalai.policies.lingbotvla2.optimizer import MuonAdamW, newton_schulz
from physicalai.policies.lingbotvla2.teachers import TeacherTargets
from physicalai.policies.lingbotvla2.policy import LingBotVLA2
from physicalai.data import Observation
from .test_processors import StubTokenizer
from .test_policy import tiny_config


def training_config():
    return replace(tiny_config(), training_recipe="upstream", distillation_heads=True,
                   loss_type="fm", alignment_feature_dim=16, alignment_num_heads=2, alignment_head_dim=8)


def training_batch():
    return {
        "pixel_values": torch.randn(1, 1, 4, 1536),
        "image_grid_thw": torch.tensor([[[1, 2, 2]]]), "img_masks": torch.ones(1, 1, dtype=torch.bool),
        "lang_tokens": torch.tensor([[3, 4, 5]]), "lang_masks": torch.ones(1, 3, dtype=torch.bool),
        "state": torch.randn(1, 55), "action": torch.randn(1, 2, 55),
        "joint_mask": torch.ones(1, 55, dtype=torch.bool), "noise": torch.randn(1, 2, 55),
        "time": torch.tensor([.4]), "action_is_pad": torch.tensor([[False, True]]),
        **{f"{name}_target": torch.randn(1, 4, 16) for name in ("depth", "future_depth", "current_video", "future_video")},
    }


def test_full_objective_and_checkpointed_gradients() -> None:
    torch.manual_seed(123)
    model = LingBotVLA2Model.from_config(training_config())
    reference = deepcopy(model)
    batch = training_batch()
    reference_loss, metrics = reference.compute_loss(batch)
    reference_loss.backward()
    model.enable_gradient_checkpointing()
    loss, _ = model.compute_loss(batch)
    loss.backward()
    torch.testing.assert_close(loss, reference_loss)
    for name, parameter in model.named_parameters():
        expected = dict(reference.named_parameters())[name]
        if expected.grad is not None:
            torch.testing.assert_close(parameter.grad, expected.grad)
    for name in ("depth_align_head", "future_depth_align_head", "current_video_align_head", "future_video_align_head"):
        assert getattr(model, name).projector.proj_out.weight.grad is not None
    assert model.depth_align_embs.grad is not None
    assert all(f"{name}_loss_weighted" in metrics for name in ("depth", "future_depth", "current_video", "future_video"))
    unpadded = model.compute_loss({**batch, "action_is_pad": torch.zeros_like(batch["action_is_pad"])})[0]
    torch.testing.assert_close(loss, unpadded)


def test_teachers_load_lazily_for_training_only(monkeypatch) -> None:
    calls = []
    def load(*args, **kwargs):
        calls.append(kwargs)
        return TeacherTargets(FakeMoge(), FakeDepth(), FakeVideo())
    monkeypatch.setattr("physicalai.policies.lingbotvla2.teacher_models.load_teachers", load)
    policy = LingBotVLA2.from_config(replace(training_config(), dataset_fps=30))
    policy._preprocessor._tokenizer = StubTokenizer()
    assert not calls
    batch = Observation(state=torch.ones(1, 6), action=torch.ones(1, 2, 6),
                        images={"front": torch.rand(1, 2, 3, 32, 32)}, task=["move"])
    assert policy(batch)[0].isfinite()
    assert len(calls) == 1
    policy.eval()
    batch.action = None
    assert policy.predict_action_chunk(batch).isfinite().all()
    assert len(calls) == 1


def test_fp32_action_expert_rejects_outer_autocast() -> None:
    model = LingBotVLA2Model.from_config(replace(training_config(), action_fp32=True))
    batch = training_batch()
    with torch.autocast("cpu", dtype=torch.bfloat16), pytest.raises(ValueError, match="autocast disabled"):
        model._velocity(batch, batch["action"], batch["time"])


def test_training_stats_respect_episode_boundaries_and_action_representation() -> None:
    config = replace(tiny_config(), relative_action_groups=("arm.position",))
    state = torch.tensor([[1.] * 6, [2.] * 6, [100.] * 6])
    action = state + 3
    state_stats, action_stats = training_statistics(state, action, torch.tensor([0, 0, 1]), config)
    torch.testing.assert_close(torch.tensor(state_stats.mean), state.mean(0))
    mean = torch.tensor(action_stats.mean)
    torch.testing.assert_close(mean[0, :5], torch.full((5,), 3.))
    torch.testing.assert_close(mean[1, :5], torch.full((5,), 3. + 1/3))
    assert mean[1, 0] < 4  # No jump from episode 0 into episode 1.
    assert mean[0, 5] == mean[1, 5]  # Absolute gripper statistics pool all horizons.


class FakeMoge(torch.nn.Module):
    def infer(self, image, **kwargs):
        return {"depth": image.mean(1)}


class FakeDepth(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(16))

    def infer_feat(self, image, depth, **kwargs):
        pixels = torch.nn.functional.adaptive_avg_pool2d(image.mean(1, keepdim=True), (2, 2))
        return pixels * self.scale[None, :, None, None], None


class FakeVideo(torch.nn.Module):
    def get_future_feature(self, video, *, current_index, **kwargs):
        def features(image):
            return torch.nn.functional.adaptive_avg_pool2d(image.mean(1, keepdim=True), (2, 2)).flatten(2).transpose(1, 2).expand(-1, -1, 16)
        assert current_index == 1
        torch.testing.assert_close(video[:, :, 0], video[:, :, 1])
        return features(video[:, :, -1]), features(video[:, :, current_index])


def test_frozen_teacher_extraction_and_student_only_checkpoint() -> None:
    teachers = TeacherTargets(FakeMoge(), FakeDepth(), FakeVideo(), {"source": "test"})
    policy = LingBotVLA2.from_config(replace(training_config(), dataset_fps=30), teachers=teachers)
    policy._preprocessor._tokenizer = StubTokenizer()
    observation = Observation(
        state=torch.ones(1, 6), action=torch.ones(1, 2, 6),
        images={"front": torch.rand(1, 2, 3, 32, 32)}, task=["move"],
    )
    loss, metrics = policy(observation)
    loss.backward()
    assert "current_video_loss_weighted" in metrics
    for model in (teachers.moge, teachers.depth, teachers.video):
        assert not model.training
        assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    assert not any("teachers" in name for name in policy.state_dict())
    checkpoint: dict[str, Any] = {}
    policy.on_save_checkpoint(checkpoint)
    assert checkpoint["teacher_references"] == {"source": "test"}


def test_resume_can_enable_cache_but_cannot_change_objective() -> None:
    base = LingBotVLA2.from_config(training_config())
    checkpoint: dict[str, Any] = {}
    base.on_save_checkpoint(checkpoint)
    cached = LingBotVLA2.from_config(replace(training_config(), teacher_targets_cached=True))
    cached.on_load_checkpoint(checkpoint)
    assert cached.config.teacher_targets_cached and cached.model.image_delta_indices is None
    changed = LingBotVLA2.from_config(replace(training_config(), loss_type="L1_fm"))
    with pytest.raises(ValueError, match="does not match"):
        changed.on_load_checkpoint(checkpoint)


def test_measured_depth_skips_rgb_depth_prediction() -> None:
    teachers = TeacherTargets(None, FakeDepth(), FakeVideo())
    policy = LingBotVLA2.from_config(replace(training_config(), dataset_fps=30,
        teacher_depth_key="observation.images.depth", teacher_depth_aligned=True), teachers=teachers)
    policy._preprocessor._tokenizer = StubTokenizer()
    batch = Observation(state=torch.ones(1,6), action=torch.ones(1,2,6),
        images={"front": torch.rand(1,2,3,32,32)},
        depths={"observation.images.depth": torch.full((1,2,1,32,32), 1.5)}, task=["move"])
    loss, metrics = policy(batch)
    assert loss.isfinite() and "depth_loss" in metrics
    policy.config = replace(policy.config, teacher_depth_aligned=False)
    policy._preprocessor.teacher_depth_aligned = False
    with pytest.raises(ValueError, match="calibrated/aligned"):
        policy(batch)


def test_cached_targets_train_without_future_frames_or_teacher_models(monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("Teacher models should not load")
    monkeypatch.setattr("physicalai.policies.lingbotvla2.teacher_models.load_teachers", forbidden)
    policy = LingBotVLA2.from_config(replace(training_config(), teacher_targets_cached=True))
    policy._preprocessor._tokenizer = StubTokenizer()
    batch = Observation(state=torch.ones(1,6), action=torch.ones(1,2,6),
        images={"front": torch.rand(1,3,32,32)}, task=["move"],
        extra={f"teacher_features.{name}": torch.rand(1,4,16) for name in ("depth","future_depth","current_video","future_video")})
    assert policy(batch)[0].isfinite()
    assert policy.teachers is None
    assert policy.model.image_delta_indices is None


def test_statistics_read_only_the_selected_training_table() -> None:
    from datasets import Dataset as HFDataset
    from types import SimpleNamespace
    from physicalai.data.lerobot.dataset import _LeRobotDatasetAdapter

    table = HFDataset.from_dict({
        "observation.state": [[1.] * 6, [2.] * 6, [1000.] * 6],
        "action": [[4.] * 6, [5.] * 6, [1003.] * 6], "episode_index": [0, 0, 1],
    }).select([0, 1])
    underlying: Any = SimpleNamespace(hf_dataset=table, fps=30)
    adapter = _LeRobotDatasetAdapter.from_lerobot(underlying)
    config = adopt_training_statistics(adapter, replace(tiny_config(), relative_action_groups=("arm.position",)))
    assert config.dataset_fps == 30
    state = next(feature for feature in config.input_features if feature.ftype.value == "STATE")
    assert state.normalization_data.mean == [1.5] * 6
    assert config.output_features[0].normalization_data.mean[0][:5] == [3.] * 5


def test_muon_step_matches_upstream_formula_and_resumes() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(8, 4), torch.nn.LayerNorm(4))
    optimizer = MuonAdamW(model, lr=1e-3)
    parameter = model[0].weight
    gradient = torch.arange(parameter.numel()).reshape_as(parameter).float() / 10
    parameter.grad = gradient.clone()
    before = parameter.detach().clone()
    expected = before - .001 * .2 * (8 ** .5) * newton_schulz(gradient * (.05 * .95 + .05))
    # Upstream: buffer=.05*g; nesterov update=lerp(g,buffer,.95)=.0975*g.
    optimizer.step()
    torch.testing.assert_close(parameter, expected)
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    optimizer.step()
    restored = deepcopy(model)
    resumed = MuonAdamW(restored, lr=1e-3)
    resumed.load_state_dict(deepcopy(optimizer.state_dict()))
    for module in (model, restored):
        for p in module.parameters():
            p.grad = torch.ones_like(p)
    optimizer.step()
    resumed.step()
    for actual, reference in zip(restored.parameters(), model.parameters(), strict=True):
        torch.testing.assert_close(actual, reference)

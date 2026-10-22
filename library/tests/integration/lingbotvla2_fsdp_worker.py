# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Run from library with torchrun --standalone --nproc-per-node=4 -m tests.integration.lingbotvla2_fsdp_worker."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor

from physicalai.policies.lingbotvla2.config import LingBotVLA2Config
from physicalai.policies.lingbotvla2.pretrained import resolve_pretrained
from physicalai.policies.lingbotvla2.distributed import shard_policy
from physicalai.policies.lingbotvla2.optimizer import MuonAdamW
from physicalai.policies.lingbotvla2.policy import LingBotVLA2
from tests.unit.policies.lingbotvla2.test_training import training_batch, training_config


class PreparedPolicy(LingBotVLA2):
    """Use fixed teacher targets to isolate student sharding from teacher packaging."""

    def forward(self, batch):
        return self.model(batch)


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    mesh = init_device_mesh("cuda", (dist.get_world_size(),))
    torch.manual_seed(123)
    full = os.environ.get("LINGBOT_FULL_WEIGHTS") == "1"
    config = replace(training_config(), moe_backend=os.environ.get("LINGBOT_MOE", "dense"))
    batch = training_batch()
    if full:
        config = LingBotVLA2Config(
            robot_layout="so101", input_features=config.input_features, output_features=config.output_features,
            camera_map=config.camera_map, training_recipe="upstream", distillation_heads=True,
            loss_type="fm", action_fp32=True,
            moe_backend=os.environ.get("LINGBOT_MOE", "dense"),
            attention_backend=os.environ.get("LINGBOT_ATTENTION", "sdpa"),
        )
        batch = {
            "pixel_values": torch.randn(1, 3, 256, 1536),
            "image_grid_thw": torch.tensor([[[1, 16, 16]] * 3]), "img_masks": torch.ones(1, 3, dtype=torch.bool),
            "lang_tokens": torch.full((1, 72), 3), "lang_masks": torch.ones(1, 72, dtype=torch.bool),
            "state": torch.randn(1, 55), "action": torch.randn(1, 50, 55),
            "joint_mask": torch.ones(1, 55, dtype=torch.bool), "noise": torch.randn(1, 50, 55),
            "time": torch.tensor([.4]),
            **{f"{name}_target": torch.randn(1, 256, 1024) for name in ("depth", "future_depth", "current_video", "future_video")},
        }
    dataset_root = os.environ.get("LINGBOT_DATASET")
    observation = None
    if full and dataset_root:
        from physicalai.data.datamodules import _collate_observations
        from physicalai.data.lerobot.dataset import _LeRobotDatasetAdapter
        from physicalai.data.lerobot.feature_cache import TeacherFeatureCache
        from physicalai.policies.lingbotvla2.data import adopt_training_statistics
        info = json.loads((Path(dataset_root) / "meta/info.json").read_text())
        adapter = _LeRobotDatasetAdapter(repo_id="local/fsdp", root=dataset_root,
            delta_timestamps={"action": [step / info["fps"] for step in range(50)]})
        adapter.extra_features = TeacherFeatureCache(Path(dataset_root), chunk_size=50, feature_dim=1024, tokens=256)
        config = replace(config, input_features=list(adapter.observation_features.values()),
            output_features=list(adapter.action_features.values()), camera_map={"front": "camera_top"},
            normalization_mode="MEAN_STD", relative_action_groups=("arm.position",), teacher_targets_cached=True)
        config = adopt_training_statistics(adapter, config)
        observation = _collate_observations([adapter[0]])
    policy = PreparedPolicy.from_config(config, gradient_checkpointing=True)
    if observation is not None:
        batch = policy._preprocessor(observation.to_dict())
    if full:
        _, weights = resolve_pretrained("robbyant/lingbot-vla-v2-6b")
        policy.model.load_weights(weights)
    policy.cuda()
    batch = {key: value.cuda() for key, value in batch.items()}
    reference = None if full else deepcopy(policy)
    if reference is not None:
        reference_optimizer = MuonAdamW(reference.model, expert_scale=2 ** .5)
        expected_loss = reference(batch)[0]
        expected_loss.backward()
        reference_optimizer.step()
    shard_policy(policy, mesh)
    optimizer = MuonAdamW(policy.model, expert_scale=(config.token_num_experts/config.token_top_k) ** .5)
    actual_loss = policy(batch)[0]
    actual_loss.backward()
    optimizer.step()
    if reference is not None:
        torch.testing.assert_close(actual_loss, expected_loss)
    maximum_error = 0.
    for actual, expected in zip(policy.model.parameters() if reference is not None else [], reference.model.parameters() if reference is not None else [], strict=True):
        full_parameter = actual.full_tensor() if isinstance(actual, DTensor) else actual
        error = (full_parameter - expected).abs().max().item()
        maximum_error = max(maximum_error, error)
        torch.testing.assert_close(full_parameter, expected, atol=2e-5, rtol=2e-4)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = policy(batch)[0]
        loss.backward()
        optimizer.step()
    if full:
        results = [None] * dist.get_world_size()
        dist.all_gather_object(results, {"rank": dist.get_rank(), "loss": loss.item(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "synthetic_teacher_targets": not bool(dataset_root)})
        if dist.get_rank() == 0:
            Path(os.environ.get("LINGBOT_RESULT", "/tmp/lingbotvla2-full-fsdp-result.json")).write_text(json.dumps(results, indent=2) + "\n")
        dist.destroy_process_group()
        return
    # Optimizer state can be restored on the same sharded parameter geometry.
    restored = MuonAdamW(policy.model, expert_scale=2 ** .5)
    restored.load_state_dict(deepcopy(optimizer.state_dict()))
    optimizer.zero_grad(set_to_none=True)
    policy(batch)[0].backward()
    before = {name: parameter.detach().clone() for name, parameter in policy.model.named_parameters()}
    optimizer.step()
    expected = {name: parameter.detach().clone() for name, parameter in policy.model.named_parameters()}
    with torch.no_grad():
        for name, parameter in policy.model.named_parameters():
            parameter.copy_(before[name])
    restored.step()
    for name, parameter in policy.model.named_parameters():
        actual = parameter.to_local() if isinstance(parameter, DTensor) else parameter
        target = expected[name].to_local() if isinstance(expected[name], DTensor) else expected[name]
        torch.testing.assert_close(actual, target, atol=0, rtol=0)
    results = [None] * dist.get_world_size()
    dist.all_gather_object(results, {"rank": dist.get_rank(), "loss": loss.item(),
        "max_update_error": maximum_error, "peak_allocated_bytes": torch.cuda.max_memory_allocated()})
    if dist.get_rank() == 0:
        output = Path(os.environ.get("LINGBOT_RESULT", "/tmp/lingbotvla2-fsdp-result.json"))
        output.write_text(json.dumps(results, indent=2) + "\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Opt-in four-GPU Lightning/FSDP2 checkpoint and resume smoke test."""

import os
from dataclasses import replace
from pathlib import Path

import torch
from lightning.pytorch.strategies import ModelParallelStrategy
from torch.utils.data import DataLoader

from physicalai.policies.lingbotvla2.policy import LingBotVLA2
from physicalai.train import Trainer
from tests.unit.policies.lingbotvla2.test_training import training_batch, training_config


class PreparedPolicy(LingBotVLA2):
    def setup(self, stage):
        pass

    def forward(self, batch):
        return self.model(batch)


def first(batch):
    return batch[0]


def trainer(steps):
    return Trainer(
        accelerator="gpu", devices=4,
        strategy=ModelParallelStrategy(data_parallel_size=4, tensor_parallel_size=1, save_distributed_checkpoint=False),
        precision="32-true", max_steps=steps, logger=False, enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False, gradient_clip_val=1.,
        num_sanity_val_steps=0,
    )


def main():
    torch.manual_seed(123)
    loader = DataLoader([training_batch() for _ in range(16)], collate_fn=first)
    config = replace(training_config(), moe_backend=os.environ.get("LINGBOT_MOE", "dense"))
    policy = PreparedPolicy.from_config(config, gradient_checkpointing=True)
    initial = trainer(2)
    initial.fit(policy, train_dataloaders=loader)
    checkpoint = Path("/tmp/lingbotvla2-lightning-fsdp-full.ckpt")
    initial.save_checkpoint(checkpoint)
    resumed = trainer(3)
    policy = PreparedPolicy.from_config(config, gradient_checkpointing=True)
    resumed.fit(policy, train_dataloaders=loader, ckpt_path=checkpoint)
    assert resumed.global_step == 3


if __name__ == "__main__":
    main()

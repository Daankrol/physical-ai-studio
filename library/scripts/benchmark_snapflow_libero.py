#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Verify SnapFlow phase 2 (distillation) for Pi05 on LIBERO.

Phase 1 already exists as the published ``lerobot/pi05_libero_finetuned`` checkpoint, so this
script only runs phase 2: it benchmarks that checkpoint as the multi-step teacher, distills it
with the VLM frozen, and benchmarks the distilled model at 1-NFE. Tweak the constants below;
there are no CLI arguments. Run from ``library/`` with ``python scripts/benchmark_snapflow_libero.py``.
"""

# This is a CLI entry point: progress banners are the intended user interface.
# ruff: noqa: T201

from __future__ import annotations

import time
from pathlib import Path

import lightning
import torch
from lightning.pytorch.callbacks import ModelCheckpoint

from physicalai.benchmark.gyms import LiberoBenchmark
from physicalai.data import LeRobotDataModule
from physicalai.policies import Pi05
from physicalai.train import Trainer

PRETRAINED = "lerobot/pi05_libero_finetuned"  # phase-1 teacher
DATASET = "HuggingFaceVLA/libero"
OUTPUT_DIR = Path("./results/snapflow_libero")
SNAPFLOW_CKPT: Path | None = None  # set to skip training and only benchmark

TASK_SUITE = "libero_10"
TASK_IDS: list[int] | None = [1, 3, 5, 6]  # None runs every task in the suite
NUM_EPISODES = 10
BASELINE_STEPS = 10  # denoising steps for the teacher

TRAIN_STEPS = 20_000
CKPT_EVERY_STEPS = 5_000
BATCH_SIZE = 8
WARMUP_STEPS = 1_000
DTYPE = "float32"  # same for both runs, or the latency comparison is meaningless
SEED = 42

# SnapFlow paper defaults (arXiv:2604.05656).
ALPHA = 0.5
LAMBDA = 0.1
SNAPFLOW_STEPS = 1


def evaluate(policy: Pi05, name: str) -> tuple[float, float]:
    """Time one denoising pass, run the LIBERO rollout, and write results.

    Returns:
        Tuple of (success rate in %, mean inference latency in ms).
    """
    bench = LiberoBenchmark(task_suite=TASK_SUITE, task_ids=TASK_IDS, num_episodes=NUM_EPISODES, seed=SEED)
    try:
        # predict_action_chunk bypasses the action queue, so this times the denoising
        # loop rather than the env stepping that dominates rollout FPS.
        obs, _ = bench.gyms[0].reset()
        for _ in range(3):
            policy.predict_action_chunk(obs)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            policy.predict_action_chunk(obs)
        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) / 20 * 1e3

        results = bench.evaluate(policy)
    finally:
        # Release the EGL contexts while CUDA is still alive.
        for gym in bench.gyms:
            gym.close()

    print(results.summary(), flush=True)
    out_dir = OUTPUT_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_json(out_dir / "results.json")
    results.to_csv(out_dir / "results.csv")
    return results.overall_success_rate, latency_ms


def train_phase2() -> Path:
    """Distill the phase-1 teacher into a 1-NFE SnapFlow model.

    Returns:
        Path to the final checkpoint.
    """
    save_dir = OUTPUT_DIR / "phase2_snapflow"
    save_dir.mkdir(parents=True, exist_ok=True)

    policy = Pi05(
        pretrained_name_or_path=PRETRAINED,
        dtype=DTYPE,
        snapflow_enabled=True,
        snapflow_alpha=ALPHA,
        snapflow_lambda=LAMBDA,
        snapflow_num_inference_steps=SNAPFLOW_STEPS,
        train_expert_only=True,  # freeze the VLM: only the action expert + target-time embedding train
        gradient_checkpointing=True,
        scheduler_decay_steps=None,  # cosine horizon follows the real step budget
        scheduler_warmup_steps=WARMUP_STEPS,
    )
    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    total = sum(p.numel() for p in policy.parameters())
    print(f"Phase-2 trainable: {trainable / 1e6:.0f}M / {total / 1e6:.0f}M (expect ~10%)", flush=True)

    # Step-based cadence: a LIBERO epoch is longer than the whole phase-2 budget,
    # so an epoch-based checkpoint callback would never fire.
    ckpt_cb = ModelCheckpoint(
        dirpath=str(save_dir),
        filename="snapflow-step{step:07d}",
        every_n_train_steps=CKPT_EVERY_STEPS,
        save_top_k=-1,
        save_last="link",
        auto_insert_metric_name=False,
    )
    datamodule = LeRobotDataModule(
        repo_id=DATASET,
        train_batch_size=BATCH_SIZE,
        data_format="physicalai",
        revision="main",
    )
    Trainer(
        max_steps=TRAIN_STEPS,
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        default_root_dir=str(save_dir),
        callbacks=[ckpt_cb],
    ).fit(model=policy, datamodule=datamodule)

    ckpt = Path(ckpt_cb.last_model_path or ckpt_cb.best_model_path)
    print(f"Phase-2 checkpoint: {ckpt}", flush=True)
    return ckpt


def main() -> None:
    """Benchmark the teacher, distill it, and benchmark the distilled model."""
    lightning.seed_everything(SEED)

    print(f"\n{'=' * 60}\nBASELINE - {BASELINE_STEPS}-step flow matching\n{'=' * 60}", flush=True)
    # Set the step count through the constructor. The old version poked `_num_steps` on the
    # model, which Pi05Model never reads (it reads `_num_inference_steps`), so it was a no-op.
    teacher = Pi05(
        pretrained_name_or_path=PRETRAINED,
        dtype=DTYPE,
        snapflow_enabled=False,
        num_inference_steps=BASELINE_STEPS,
    )
    teacher.to("cuda").eval()
    b_rate, b_ms = evaluate(teacher, "baseline")
    teacher.to("cpu")
    del teacher
    torch.cuda.empty_cache()

    ckpt = SNAPFLOW_CKPT or train_phase2()

    print(f"\n{'=' * 60}\nSNAPFLOW - {SNAPFLOW_STEPS}-step inference\n{'=' * 60}", flush=True)
    # load_from_checkpoint rebuilds the policy from its own hparams (including dataset
    # stats) and loads weights strictly, so a key mismatch fails loudly instead of
    # silently benchmarking the undistilled teacher.
    student = Pi05.load_from_checkpoint(
        ckpt,
        map_location="cpu",
        dtype=DTYPE,
        snapflow_enabled=True,
        snapflow_alpha=ALPHA,
        snapflow_lambda=LAMBDA,
        snapflow_num_inference_steps=SNAPFLOW_STEPS,
        train_expert_only=True,
        compile_model=False,  # excluded from saved hparams, must be re-passed
    )
    student.to("cuda").eval()
    s_rate, s_ms = evaluate(student, "snapflow")

    print(
        f"\n{'=' * 60}\n"
        f"                       Baseline    SnapFlow\n"
        f"  Success Rate:        {b_rate:>7.1f}%    {s_rate:>7.1f}%\n"
        f"  Inference latency:   {b_ms:>7.1f}ms   {s_ms:>7.1f}ms\n"
        f"  Speedup:             {b_ms / s_ms:>7.2f}x\n"
        f"  Success Delta:       {s_rate - b_rate:>+7.1f}%\n"
        f"{'=' * 60}",
        flush=True,
    )


if __name__ == "__main__":
    main()

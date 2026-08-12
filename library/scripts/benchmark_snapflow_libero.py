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
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from physicalai.benchmark.gyms import LiberoBenchmark
from physicalai.data import LeRobotDataModule
from physicalai.policies import Pi05
from physicalai.train import Trainer

PRETRAINED = "lerobot/pi05_libero_finetuned"  # phase-1 teacher
DATASET = "HuggingFaceVLA/libero"
OUTPUT_DIR = Path("/mnt/data/experiments/snapflow_libero_30k")
SNAPFLOW_CKPT: Path | None = None  # set to skip training and only benchmark

TASK_SUITE = "libero_10"
TASK_IDS: list[int] | None = None  # None runs every task in the suite
NUM_EPISODES = 10
BASELINE_STEPS = 10  # denoising steps for the teacher

TRAIN_STEPS = 30_000
CKPT_EVERY_STEPS = TRAIN_STEPS // 10  # save a checkpoint every 1/10 of the budget
BATCH_SIZE = 16
WARMUP_STEPS = 1000
DTYPE = "bfloat16"
SEED = 42

# SnapFlow paper defaults (arXiv:2604.05656).
ALPHA = 0.5
LAMBDA = 0.1
SNAPFLOW_STEPS = 1

# Held-out episodes for eval-loss validation during phase-2 distillation.
VAL_SPLIT = 0.1
VAL_SPLIT_SEED = 42

# Weights & Biases. Set WANDB_ENABLED = False to fall back to Lightning's
# default (TensorBoard) logger.
WANDB_ENABLED = True
WANDB_PROJECT = "physicalai_snapflow"
WANDB_ENTITY: str | None = None
WANDB_NAME = "snapflow-libero"
WANDB_OFFLINE = False
WANDB_LOG_MODEL = False


def _make_logger(save_dir: Path) -> WandbLogger | None:
    """Build the phase-2 training W&B logger, or ``None`` to keep Lightning's default.

    Args:
        save_dir: Directory W&B buffers run data into.

    Returns:
        A configured ``WandbLogger``, or ``None`` when ``WANDB_ENABLED`` is ``False``.

    Raises:
        SystemExit: If ``WANDB_ENABLED`` is ``True`` but wandb is not installed.
    """
    if not WANDB_ENABLED:
        return None

    try:
        import wandb  # noqa: F401, PLC0415
    except ImportError:  # pragma: no cover - depends on the install extras
        msg = "WANDB_ENABLED requires the wandb package. Install it with: uv pip install wandb"
        raise SystemExit(msg) from None

    logger_ = WandbLogger(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=f"{WANDB_NAME}-phase2",
        group=WANDB_NAME,
        job_type="phase2-snapflow",
        save_dir=str(save_dir),
        offline=WANDB_OFFLINE,
        log_model=WANDB_LOG_MODEL,
    )
    logger_.experiment.config.update(
        {
            "pretrained": PRETRAINED,
            "dataset": DATASET,
            "task_suite": TASK_SUITE,
            "task_ids": TASK_IDS,
            "num_episodes": NUM_EPISODES,
            "baseline_steps": BASELINE_STEPS,
            "train_steps": TRAIN_STEPS,
            "batch_size": BATCH_SIZE,
            "warmup_steps": WARMUP_STEPS,
            "val_split": VAL_SPLIT,
            "val_split_seed": VAL_SPLIT_SEED,
            "snapflow_alpha": ALPHA,
            "snapflow_lambda": LAMBDA,
            "snapflow_steps": SNAPFLOW_STEPS,
            "seed": SEED,
        },
        allow_val_change=True,
    )
    return logger_


def evaluate(policy: Pi05, name: str, logger_: WandbLogger | None = None) -> tuple[float, float]:
    """Time one denoising pass, run the LIBERO rollout, and write results.

    Args:
        policy: Policy to benchmark.
        name: Run name, used for the output directory and metric prefix.
        logger_: Optional W&B logger to report the success rate and latency to.

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

    if logger_ is not None:
        logger_.log_metrics(
            {
                f"benchmark/{name}/success_rate": results.overall_success_rate,
                f"benchmark/{name}/latency_ms": latency_ms,
            },
        )

    return results.overall_success_rate, latency_ms


def train_phase2() -> Path:
    """Distill the phase-1 teacher into a 1-NFE SnapFlow model.

    Returns:
        Path to the best (lowest ``val/loss``) checkpoint seen during distillation.
    """
    save_dir = OUTPUT_DIR / "phase2_snapflow"
    save_dir.mkdir(parents=True, exist_ok=True)

    policy = Pi05(
        pretrained_name_or_path=PRETRAINED,
        dtype=DTYPE,
        snapflow_enabled=True,
        compile_model=True,
        snapflow_alpha=ALPHA,
        snapflow_lambda=LAMBDA,
        snapflow_num_inference_steps=SNAPFLOW_STEPS,
        train_expert_only=True,  # freeze the VLM: only the action expert + target-time embedding train
        gradient_checkpointing=False,  # enable if you run out of GPU memory
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
        save_last=True,
        auto_insert_metric_name=False,
    )
    # Separate best-val-loss tracker: distillation loss can wobble near the end of
    # the budget, so benchmarking should use the lowest-val/loss checkpoint rather
    # than whatever happens to be last.
    best_ckpt_cb = ModelCheckpoint(
        dirpath=str(save_dir),
        filename="snapflow-best",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        auto_insert_metric_name=False,
    )
    datamodule = LeRobotDataModule(
        repo_id=DATASET,
        train_batch_size=BATCH_SIZE,
        data_format="physicalai",
        revision="main",
        val_split=VAL_SPLIT,
        val_split_seed=VAL_SPLIT_SEED,
        val_batch_size=BATCH_SIZE,
    )
    logger_ = _make_logger(save_dir)
    callbacks: list = [ckpt_cb, best_ckpt_cb]
    if logger_ is not None:
        callbacks.append(LearningRateMonitor(logging_interval="step"))
    try:
        Trainer(
            max_steps=TRAIN_STEPS,
            accelerator="gpu",
            devices=1,
            precision="bf16-mixed",
            default_root_dir=str(save_dir),
            callbacks=callbacks,
            logger=logger_,
            # Step-based training: validate on the same cadence as checkpointing.
            val_check_interval=CKPT_EVERY_STEPS,
            check_val_every_n_epoch=None,
        ).fit(model=policy, datamodule=datamodule)

        if best_ckpt_cb.best_model_path:
            ckpt = Path(best_ckpt_cb.best_model_path)
            print(
                f"Phase-2 best checkpoint (val/loss={best_ckpt_cb.best_model_score:.4f}): {ckpt}",
                flush=True,
            )
        else:
            # VAL_SPLIT == 0 (no val/loss logged) -> nothing to rank, fall back to last.
            ckpt = Path(ckpt_cb.last_model_path or ckpt_cb.best_model_path)
            print(f"Phase-2 checkpoint (no val/loss available; using last): {ckpt}", flush=True)
        return ckpt
    finally:
        if logger_ is not None:
            logger_.experiment.finish()


def main() -> None:
    """Benchmark the teacher, distill it, and benchmark the distilled model.

    Raises:
        SystemExit: If ``WANDB_ENABLED`` is ``True`` but wandb is not installed.
    """
    lightning.seed_everything(SEED)

    bench_logger = None
    if WANDB_ENABLED:
        try:
            import wandb  # noqa: F401, PLC0415
        except ImportError:  # pragma: no cover - depends on the install extras
            msg = "WANDB_ENABLED requires the wandb package. Install it with: uv pip install wandb"
            raise SystemExit(msg) from None
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        bench_logger = WandbLogger(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=f"{WANDB_NAME}-benchmark",
            group=WANDB_NAME,
            job_type="benchmark",
            save_dir=str(OUTPUT_DIR),
            offline=WANDB_OFFLINE,
            log_model=False,
        )

    print(f"\n{'=' * 60}\nBASELINE - {BASELINE_STEPS}-step flow matching\n{'=' * 60}", flush=True)
    teacher = Pi05(
        pretrained_name_or_path=PRETRAINED,
        dtype=DTYPE,
        snapflow_enabled=False,
        num_inference_steps=BASELINE_STEPS,
    )
    teacher.to("cuda").eval()
    b_rate, b_ms = evaluate(teacher, "baseline", bench_logger)
    teacher.to("cpu")
    del teacher
    torch.cuda.empty_cache()

    ckpt = SNAPFLOW_CKPT or train_phase2()

    print(f"\n{'=' * 60}\nSNAPFLOW - {SNAPFLOW_STEPS}-step inference\n{'=' * 60}", flush=True)
    student = Pi05.load_from_checkpoint(
        ckpt,
        map_location="cpu",
        dtype=DTYPE,
        snapflow_enabled=True,
        snapflow_alpha=ALPHA,
        snapflow_lambda=LAMBDA,
        snapflow_num_inference_steps=SNAPFLOW_STEPS,
    )
    student.to("cuda").eval()
    s_rate, s_ms = evaluate(student, "snapflow", bench_logger)

    speedup = b_ms / s_ms
    if bench_logger is not None:
        bench_logger.log_metrics(
            {
                "benchmark/speedup": speedup,
                "benchmark/success_delta": s_rate - b_rate,
            },
        )
        bench_logger.experiment.finish()

    print(
        f"\n{'=' * 60}\n"
        f"                       Baseline    SnapFlow\n"
        f"  Success Rate:        {b_rate:>7.1f}%    {s_rate:>7.1f}%\n"
        f"  Inference latency:   {b_ms:>7.1f}ms   {s_ms:>7.1f}ms\n"
        f"  Speedup:             {speedup:>7.2f}x\n"
        f"  Success Delta:       {s_rate - b_rate:>+7.1f}%\n"
        f"{'=' * 60}",
        flush=True,
    )


if __name__ == "__main__":
    main()

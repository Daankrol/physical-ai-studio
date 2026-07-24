#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Two-phase Pi05 + SnapFlow training on a real SO101 dataset.

Workflow
--------
1. **Phase 1 - finetune.** Warm-start Pi05 from the ``lerobot/pi05_base`` VLA and
   finetune it on the target LeRobot dataset with standard flow matching.
   Intermediate checkpoints are saved every ``--ckpt-every`` steps so you can
   compare how much finetuning is actually needed.
2. **Phase 2 - SnapFlow distillation.** Take the *final* phase-1 checkpoint,
   enable SnapFlow self-distillation (freezes the VLM, trains only the action
   expert + target-time embedding), and distill toward 1-NFE inference.
   Again, a checkpoint is saved every ``--ckpt-every`` step so you can compare
   how many "snapflow-steps" are required.

Recommended step budgets (defaults)
-----------------------------------
- Phase 1: ~30k steps. openpi finetunes Pi0.5 for ~30k steps and Pi05's own
  scheduler decays over 30k steps.
- Phase 2: ~30k steps. This is the SnapFlow paper recipe (arXiv:2604.05656,
  §3.6 + Appendix J) and matches ``docs/how-to/training/snapflow-two-phase.md``.

Both are overridable via ``--phase1-steps`` / ``--phase2-steps``.

Usage
-----
From ``library/``::

    # Full two-phase run with defaults
    python scripts/train_snapflow_so101.py

    # Custom step budgets / checkpoint cadence
    python scripts/train_snapflow_so101.py \
        --phase1-steps 30000 --phase2-steps 30000 --ckpt-every 5000

    # Skip phase 1 and distill from an existing finetuned checkpoint
    python scripts/train_snapflow_so101.py \
        --skip-phase1 --phase1-ckpt experiments/so101_snapflow/phase1/.../last.ckpt

    # Quick smoke-test
    python scripts/train_snapflow_so101.py \
        --phase1-steps 20 --phase2-steps 20 --ckpt-every 10 --batch-size 2
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("train_snapflow_so101")

# Defaults
_PRETRAINED = "lerobot/pi05_base"
_DATASET = "Daankrol/pick-and-place-so101"

# SnapFlow paper hyperparameters (arXiv:2604.05656)
_SNAPFLOW_ALPHA = 0.5
_SNAPFLOW_LAMBDA = 0.1
_SNAPFLOW_STEPS = 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=_DATASET, help=f"LeRobot dataset repo ID. Default: {_DATASET}")
    p.add_argument(
        "--pretrained",
        default=_PRETRAINED,
        help=f"Pi05 base checkpoint to warm-start phase 1 from. Default: {_PRETRAINED}",
    )
    p.add_argument(
        "--phase1-steps",
        type=int,
        default=30000,
        help="Phase-1 flow-matching finetuning steps. Default: 30000.",
    )
    p.add_argument(
        "--phase2-steps",
        type=int,
        default=30000,
        help="Phase-2 SnapFlow distillation steps. Default: 30000.",
    )
    p.add_argument(
        "--ckpt-every",
        type=int,
        default=5000,
        help="Save a checkpoint every N training steps in each phase. Default: 5000.",
    )
    p.add_argument("--batch-size", type=int, default=8, help="Training batch size. Default: 8.")
    p.add_argument(
        "--snapflow-alpha",
        type=float,
        default=_SNAPFLOW_ALPHA,
        help=f"SnapFlow FM-loss weight. Default: {_SNAPFLOW_ALPHA} (paper).",
    )
    p.add_argument(
        "--snapflow-lambda",
        type=float,
        default=_SNAPFLOW_LAMBDA,
        help=f"SnapFlow shortcut-loss scale. Default: {_SNAPFLOW_LAMBDA} (paper).",
    )
    p.add_argument(
        "--snapflow-steps",
        type=int,
        default=_SNAPFLOW_STEPS,
        help=f"SnapFlow inference denoising steps. Default: {_SNAPFLOW_STEPS} (1-NFE).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./experiments/so101_snapflow"),
        help="Root directory for both phases' checkpoints. Default: ./experiments/so101_snapflow",
    )
    p.add_argument(
        "--skip-phase1",
        action="store_true",
        help="Skip phase-1 finetuning. Requires --phase1-ckpt.",
    )
    p.add_argument(
        "--phase1-ckpt",
        type=Path,
        default=None,
        help="Existing phase-1 checkpoint to distill from (used with --skip-phase1).",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42.")

    args = p.parse_args()
    if args.skip_phase1 and args.phase1_ckpt is None:
        p.error("--skip-phase1 requires --phase1-ckpt to point at a finetuned checkpoint.")
    return args


def _make_checkpoint_callback(save_dir: Path, ckpt_every: int, phase: str):  # noqa: ANN202
    """Build a ModelCheckpoint that saves every N steps and keeps them all."""
    from lightning.pytorch.callbacks import ModelCheckpoint  # noqa: PLC0415

    return ModelCheckpoint(
        dirpath=str(save_dir),
        filename=f"{phase}-step{{step}}",
        every_n_train_steps=ckpt_every,
        save_top_k=-1,  # keep every staged checkpoint for comparison
        save_last=True,  # always write last.ckpt
        auto_insert_metric_name=False,
    )


def _make_datamodule(args: argparse.Namespace):  # noqa: ANN202
    from physicalai.data import LeRobotDataModule  # noqa: PLC0415

    # revision="main" bypasses lerobot's get_safe_version() tag lookup, which
    # fails when the dataset has no HF git tags due to a lerobot 0.6.0 /
    # huggingface_hub ≥1.23 incompatibility (HfHubHTTPError missing `response`).
    return LeRobotDataModule(
        repo_id=args.dataset,
        train_batch_size=args.batch_size,
        data_format="physicalai",
        revision="main",
    )


def _find_last_ckpt(save_dir: Path) -> Path:
    ckpts = sorted(save_dir.glob("**/last.ckpt"))
    if not ckpts:
        ckpts = sorted(save_dir.glob("**/*.ckpt"))
    if not ckpts:
        msg = f"No checkpoint found under {save_dir} after training."
        raise FileNotFoundError(msg)
    return ckpts[-1]


def train_phase1(args: argparse.Namespace) -> Path:
    """Finetune Pi05 from the base VLA with standard flow matching."""
    from physicalai.policies import Pi05  # noqa: PLC0415
    from physicalai.train import Trainer  # noqa: PLC0415

    save_dir = args.output_dir / "phase1"
    save_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"\n{'=' * 70}\n"
        f"PHASE 1 - Pi05 finetuning ({args.phase1_steps} steps)\n"
        f"  base       : {args.pretrained}\n"
        f"  dataset    : {args.dataset}\n"
        f"  ckpt every : {args.ckpt_every} steps -> {save_dir}\n"
        f"{'=' * 70}",
        flush=True,
    )

    policy = Pi05(
        pretrained_name_or_path=args.pretrained,
        dtype="bfloat16",
        gradient_checkpointing=True,
        # compile_model=True,
    )
    datamodule = _make_datamodule(args)

    trainer = Trainer(
        max_steps=args.phase1_steps,
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        default_root_dir=str(save_dir),
        callbacks=[_make_checkpoint_callback(save_dir, args.ckpt_every, "phase1")],
        log_every_n_steps=100,
    )
    trainer.fit(model=policy, datamodule=datamodule)

    ckpt = _find_last_ckpt(save_dir)
    print(f"  Phase-1 final checkpoint: {ckpt}", flush=True)
    return ckpt


def train_phase2(args: argparse.Namespace, phase1_ckpt: Path) -> Path:
    """Distill the finetuned model into a 1-NFE SnapFlow model."""
    import torch  # noqa: PLC0415

    from physicalai.policies import Pi05  # noqa: PLC0415
    from physicalai.train import Trainer  # noqa: PLC0415

    save_dir = args.output_dir / "phase2_snapflow"
    save_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"\n{'=' * 70}\n"
        f"PHASE 2 - SnapFlow distillation ({args.phase2_steps} steps)\n"
        f"  from ckpt  : {phase1_ckpt}\n"
        f"  alpha={args.snapflow_alpha}  lambda={args.snapflow_lambda}  "
        f"num_inference_steps={args.snapflow_steps}\n"
        f"  ckpt every : {args.ckpt_every} steps -> {save_dir}\n"
        f"{'=' * 70}",
        flush=True,
    )

    # Rebuild the policy from the base VLA, then enable SnapFlow (freezes the VLM,
    # adds the zero-init target-time embedding) and load the phase-1 weights on
    # top. enable_snapflow() sets plain-attribute flags that are not part of the
    # state dict, so they survive load_state_dict unchanged.
    policy = Pi05(pretrained_name_or_path=args.pretrained, dtype="bfloat16", 
        gradient_checkpointing=True, 
        # compile_model=True
        )
    policy.enable_snapflow(
        alpha=args.snapflow_alpha,
        lambda_=args.snapflow_lambda,
        num_inference_steps=args.snapflow_steps,
    )
    state = torch.load(phase1_ckpt, map_location="cpu", weights_only=False)
    missing, unexpected = policy.load_state_dict(state.get("state_dict", state), strict=False)
    if missing:
        logger.info("load_state_dict missing keys (expected: zero-init target-time embedding): %d", len(missing))
    if unexpected:
        logger.info("load_state_dict unexpected keys: %d", len(unexpected))

    datamodule = _make_datamodule(args)

    trainer = Trainer(
        max_steps=args.phase2_steps,
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        default_root_dir=str(save_dir),
        callbacks=[_make_checkpoint_callback(save_dir, args.ckpt_every, "snapflow")],
        log_every_n_steps=100,
    )
    trainer.fit(model=policy, datamodule=datamodule)

    ckpt = _find_last_ckpt(save_dir)
    print(f"  Phase-2 final SnapFlow checkpoint: {ckpt}", flush=True)
    return ckpt


def main() -> None:
    import lightning  # noqa: PLC0415

    args = parse_args()
    lightning.seed_everything(args.seed)

    if args.skip_phase1:
        phase1_ckpt = args.phase1_ckpt
        print(f"Skipping phase 1; distilling from {phase1_ckpt}", flush=True)
    else:
        phase1_ckpt = train_phase1(args)

    phase2_ckpt = train_phase2(args, phase1_ckpt)

    print(
        f"\n{'=' * 70}\n"
        f"DONE\n"
        f"  Phase-1 checkpoints : {args.output_dir / 'phase1'}\n"
        f"  Phase-2 checkpoints : {args.output_dir / 'phase2_snapflow'}\n"
        f"  Final SnapFlow ckpt : {phase2_ckpt}\n"
        f"{'=' * 70}",
        flush=True,
    )


if __name__ == "__main__":
    main()

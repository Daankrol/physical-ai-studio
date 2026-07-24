#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Two-phase SnapFlow benchmark for Pi05 on LIBERO.

Workflow
--------
1. Benchmark the pretrained Pi05 libero model (standard multi-step FM).
2. Train SnapFlow phase 2 on libero data (action expert + target-time embedding only).
3. Benchmark the trained SnapFlow model (1-NFE).

Usage
-----
From library/:

    # Full two-phase run (default)
    python scripts/benchmark_snapflow_libero.py --policy pi05

    # Use an existing SnapFlow checkpoint (skip training)
    python scripts/benchmark_snapflow_libero.py --snapflow-ckpt ./path/to/last.ckpt

    # Quick smoke-test
    python scripts/benchmark_snapflow_libero.py --task-ids 0 1 2 --num-episodes 5

    # Skip baseline benchmark
    python scripts/benchmark_snapflow_libero.py --skip-baseline
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Default pretrained checkpoints per policy type
_DEFAULTS: dict[str, str] = {
    "smolvla": "HuggingFaceVLA/smolvla_libero",
    # "pi05": "lerobot/pi05_libero_finetuned_v044", # this checkpoint was trained on un-rotated images.
    "pi05": "lerobot/pi05_libero_finetuned",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--policy",
        default="pi05",
        choices=["smolvla", "pi05"],
        help="Policy architecture to benchmark. Default: pi05",
    )
    p.add_argument(
        "--pretrained",
        default=None,
        help=(
            "HuggingFace repo ID or local path to pretrained weights. "
            f"Defaults: smolvla={_DEFAULTS['smolvla']!r}, pi05={_DEFAULTS['pi05']!r}"
        ),
    )
    p.add_argument(
        "--task-suite",
        default="libero_10",
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
        help="LIBERO task suite. Default: libero_10",
    )
    p.add_argument(
        "--task-ids",
        nargs="+",
        type=int,
        default=None,
        metavar="ID",
        help="Subset of task IDs to evaluate. Default: all tasks in the suite.",
    )
    p.add_argument(
        "--num-episodes",
        type=int,
        default=20,
        help="Episodes per task. Default: 20 (paper setting).",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Max steps per episode. Default: suite-specific LIBERO default.",
    )
    p.add_argument(
        "--snapflow-alpha",
        type=float,
        default=0.5,
        help="SnapFlow FM-loss weight. Default: 0.5 (paper).",
    )
    p.add_argument(
        "--snapflow-lambda",
        type=float,
        default=0.1,
        help="SnapFlow shortcut-loss scale. Default: 0.1 (paper).",
    )
    p.add_argument(
        "--snapflow-steps",
        type=int,
        default=1,
        help="SnapFlow inference denoising steps. Default: 1 (1-NFE).",
    )
    p.add_argument(
        "--baseline-steps",
        type=int,
        default=10,
        help="Number of denoising steps for the baseline (no-SnapFlow) run. Default: 10.",
    )
    p.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip the standard flow-matching baseline run.",
    )
    p.add_argument(
        "--skip-snapflow",
        action="store_true",
        help="Skip the SnapFlow training and benchmark run.",
    )
    p.add_argument(
        "--train-dataset",
        default="HuggingFaceVLA/libero",
        help="LeRobot dataset repo ID for SnapFlow phase-2 training. Default: HuggingFaceVLA/libero",
    )
    p.add_argument(
        "--train-steps",
        type=int,
        default=30000,
        help="SnapFlow phase-2 training steps. Default: 30000 (paper setting).",
    )
    p.add_argument(
        "--snapflow-ckpt",
        type=Path,
        default=None,
        help="Path to an existing SnapFlow checkpoint. Skips training when provided.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./results/snapflow_libero"),
        help="Directory for JSON/CSV results. Default: ./results/snapflow_libero",
    )
    p.add_argument(
        "--video-dir",
        type=Path,
        default=None,
        help="Directory for episode videos. Default: no videos.",
    )
    p.add_argument(
        "--record-mode",
        choices=["all", "successes", "failures", "none"],
        default="none",
        help="Video recording mode. Default: none.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Default: 42.",
    )

    args = p.parse_args()

    # Apply default pretrained path based on selected policy
    if args.pretrained is None:
        args.pretrained = _DEFAULTS[args.policy]

    return args


def load_policy(args: argparse.Namespace):  # noqa: ANN201
    """Load a SmolVLA or Pi05 policy from a pretrained HF repo or local path."""
    pretrained = args.pretrained
    t0 = time.monotonic()

    if args.policy == "smolvla":
        from physicalai.policies import SmolVLA  # noqa: PLC0415

        print(f"Loading SmolVLA from '{pretrained}' …", flush=True)
        policy = SmolVLA(pretrained_name_or_path=pretrained)
    else:
        from physicalai.policies import Pi05  # noqa: PLC0415

        print(f"Loading Pi05 from '{pretrained}' …", flush=True)
        policy = Pi05(pretrained_name_or_path=pretrained, dtype="float32")

    policy.to("cuda")
    policy.eval()
    print(f"  Loaded in {time.monotonic() - t0:.1f}s", flush=True)
    return policy


def _set_baseline_flags(policy, baseline_steps: int) -> None:
    """Disable SnapFlow and set multi-step inference count on the inner model."""
    from physicalai.policies import Pi05, SmolVLA  # noqa: PLC0415

    if isinstance(policy, SmolVLA):
        inner = policy.model._model  # noqa: SLF001
        inner._snapflow_enabled = False  # noqa: SLF001
        inner._num_steps = baseline_steps  # noqa: SLF001
    elif isinstance(policy, Pi05):
        inner = policy.model  # noqa: SLF001
        inner._snapflow_enabled = False  # noqa: SLF001
        inner._num_steps = baseline_steps  # noqa: SLF001
    else:
        msg = f"Unsupported policy type for baseline flag override: {type(policy).__name__}"
        raise TypeError(msg)


def make_benchmark(args: argparse.Namespace, video_subdir: str | None = None):  # noqa: ANN201
    """Build a LiberoBenchmark from CLI args."""
    from physicalai.benchmark.gyms import LiberoBenchmark  # noqa: PLC0415

    video_dir = (args.video_dir / video_subdir) if (args.video_dir and video_subdir) else None
    return LiberoBenchmark(
        task_suite=args.task_suite,
        task_ids=args.task_ids,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        video_dir=video_dir,
        record_mode=args.record_mode,
    )


def measure_latency(args: argparse.Namespace, policy, n_warmup: int = 3, n_iters: int = 20) -> dict:  # noqa: ANN001
    """Time a single full action-chunk inference (denoising) pass.

    Resets a real LIBERO gym to obtain an observation in the exact format the
    policy consumes, then repeatedly calls ``predict_action_chunk`` (bypassing
    the action-queue replay) so the measurement reflects the model's denoising
    cost, not the env stepping that dominates rollout FPS.

    Returns:
        Dict with mean/std/median latency in milliseconds and inferences/sec.
    """
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    benchmark = make_benchmark(args)
    gym = benchmark.gyms[0]
    obs, _ = gym.reset()

    def _one_pass() -> None:
        policy.predict_action_chunk(obs)
        torch.cuda.synchronize()

    for _ in range(n_warmup):
        _one_pass()

    times_ms: list[float] = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        _one_pass()
        times_ms.append((time.perf_counter() - t0) * 1e3)

    gym.close()

    arr = np.asarray(times_ms)
    return {
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std()),
        "median_ms": float(np.median(arr)),
        "infer_per_s": float(1e3 / arr.mean()),
    }


def train_snapflow(args: argparse.Namespace) -> Path:
    """Run SnapFlow phase-2 training via the Python API and return the checkpoint path."""
    from physicalai.data import LeRobotDataModule  # noqa: PLC0415
    from physicalai.policies import Pi05, SmolVLA  # noqa: PLC0415
    from physicalai.train import Trainer  # noqa: PLC0415

    train_dir = args.output_dir / "snapflow_training"
    train_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"\n{'=' * 60}\nTRAINING  —  SnapFlow phase-2 ({args.train_steps} steps)\n{'=' * 60}",
        flush=True,
    )

    if args.policy == "pi05":
        policy = Pi05(
            pretrained_name_or_path=args.pretrained,
            dtype="float32",
            snapflow_enabled=True,
            snapflow_alpha=args.snapflow_alpha,
            snapflow_lambda=args.snapflow_lambda,
            snapflow_num_inference_steps=args.snapflow_steps,
            train_expert_only=True,
        )
    else:
        policy = SmolVLA(
            pretrained_name_or_path=args.pretrained,
            snapflow_enabled=True,
            snapflow_alpha=args.snapflow_alpha,
            snapflow_lambda=args.snapflow_lambda,
            snapflow_num_inference_steps=args.snapflow_steps,
            train_expert_only=True,
        )

    datamodule = LeRobotDataModule(repo_id=args.train_dataset, train_batch_size=8, data_format="physicalai")
    trainer = Trainer(
        max_steps=args.train_steps,
        accelerator="gpu",
        devices=1,
        default_root_dir=str(train_dir),
    )
    trainer.fit(model=policy, datamodule=datamodule)

    ckpts = sorted(train_dir.glob("**/last.ckpt"))
    if not ckpts:
        ckpts = sorted(train_dir.glob("**/*.ckpt"))
    if not ckpts:
        msg = f"No checkpoint found under {train_dir} after training."
        raise FileNotFoundError(msg)
    ckpt = ckpts[-1]
    print(f"  Checkpoint: {ckpt}", flush=True)
    return ckpt


def _run_benchmark(args: argparse.Namespace, policy, video_subdir: str) -> object:  # noqa: ANN001
    """Run benchmark and explicitly close gyms to avoid EGL destructor errors."""
    benchmark = make_benchmark(args, video_subdir=video_subdir)
    t0 = time.monotonic()
    results = benchmark.evaluate(policy)
    elapsed = time.monotonic() - t0
    # Explicitly close all gyms before returning so the EGL context is released
    # while CUDA is still alive, preventing OpenGL destructor errors at exit.
    for gym in benchmark.gyms:
        gym.close()
    return results, elapsed


def run_baseline(args: argparse.Namespace, policy) -> object:  # noqa: ANN001
    """Evaluate the policy with standard flow-matching (no SnapFlow)."""
    print(
        f"\n{'=' * 60}\nBASELINE  —  standard flow-matching ({args.baseline_steps}-step Euler)\n{'=' * 60}",
        flush=True,
    )
    _set_baseline_flags(policy, args.baseline_steps)

    latency = measure_latency(args, policy)
    print(
        f"  Inference latency: {latency['mean_ms']:.1f} ± {latency['std_ms']:.1f} ms "
        f"({latency['infer_per_s']:.1f} inf/s)",
        flush=True,
    )

    results, elapsed = _run_benchmark(args, policy, video_subdir="baseline")

    print(results.summary(), flush=True)
    print(f"  Elapsed: {elapsed:.1f}s", flush=True)

    out_dir = args.output_dir / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_json(out_dir / "results.json")
    results.to_csv(out_dir / "results.csv")
    print(f"  Results written to {out_dir}", flush=True)
    return results, latency


def run_snapflow(args: argparse.Namespace, ckpt_path: Path) -> object:  # noqa: ANN001
    """Load a trained SnapFlow checkpoint and evaluate it."""
    print(
        f"\n{'=' * 60}\n"
        f"SNAPFLOW  —  {args.snapflow_steps}-step inference  "
        f"(alpha={args.snapflow_alpha}, lambda={args.snapflow_lambda})\n"
        f"checkpoint: {ckpt_path}\n"
        f"{'=' * 60}",
        flush=True,
    )
    policy = load_policy(args)
    policy.enable_snapflow(
        alpha=args.snapflow_alpha,
        lambda_=args.snapflow_lambda,
        num_inference_steps=args.snapflow_steps,
    )
    # Load trained weights on top of the pretrained model.
    # enable_snapflow() sets _snapflow_enabled and _snapflow_num_inference_steps
    # as plain Python attributes — they are not part of the state dict and
    # survive load_state_dict unchanged.
    import torch  # noqa: PLC0415

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    policy.load_state_dict(state.get("state_dict", state), strict=False)
    policy.to("cuda")
    policy.eval()

    latency = measure_latency(args, policy)
    print(
        f"  Inference latency: {latency['mean_ms']:.1f} ± {latency['std_ms']:.1f} ms "
        f"({latency['infer_per_s']:.1f} inf/s)",
        flush=True,
    )

    results, elapsed = _run_benchmark(args, policy, video_subdir="snapflow")

    print(results.summary(), flush=True)
    print(f"  Elapsed: {elapsed:.1f}s", flush=True)

    out_dir = args.output_dir / "snapflow"
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_json(out_dir / "results.json")
    results.to_csv(out_dir / "results.csv")
    print(f"  Results written to {out_dir}", flush=True)
    return results, latency


def print_comparison(baseline, snapflow) -> None:  # noqa: ANN001
    """Print a side-by-side success-rate and inference-latency comparison."""
    if baseline is None or snapflow is None:
        return

    b_results, b_lat = baseline
    s_results, s_lat = snapflow
    # success_rate is already on a 0-100 scale.
    b_rate = b_results.overall_success_rate
    s_rate = s_results.overall_success_rate
    b_ms = b_lat["mean_ms"]
    s_ms = s_lat["mean_ms"]
    speedup = b_ms / s_ms if s_ms > 0 else float("nan")

    print(
        f"\n{'=' * 60}\n"
        f"COMPARISON\n"
        f"{'=' * 60}\n"
        f"                       Baseline    SnapFlow\n"
        f"  Success Rate:        {b_rate:>7.1f}%    {s_rate:>7.1f}%\n"
        f"  Inference latency:   {b_ms:>7.1f}ms   {s_ms:>7.1f}ms\n"
        f"  Speedup:             {speedup:>7.2f}x\n"
        f"  Success Delta:       {s_rate - b_rate:>+7.1f}%\n"
        f"{'=' * 60}",
        flush=True,
    )


def main() -> None:
    args = parse_args()

    if args.skip_baseline and args.skip_snapflow:
        print("Nothing to do (--skip-baseline and --skip-snapflow both set).", file=sys.stderr)
        sys.exit(1)

    baseline_results = None
    snapflow_results = None

    if not args.skip_baseline:
        policy = load_policy(args)
        baseline_results = run_baseline(args, policy)
        del policy  # free GPU memory before training

    if not args.skip_snapflow:
        ckpt_path = args.snapflow_ckpt
        if ckpt_path is None:
            ckpt_path = train_snapflow(args)
        snapflow_results = run_snapflow(args, ckpt_path)

    print_comparison(baseline_results, snapflow_results)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Benchmark SmolVLA on LIBERO: standard flow-matching vs. SnapFlow 1-NFE.

Loads HuggingFaceVLA/smolvla_libero (cached after first run), runs the
requested number of episodes per task, and prints a side-by-side comparison.

Usage
-----
From library/:

    uv run python scripts/benchmark_snapflow_libero.py

Common overrides:

    # Fewer tasks / episodes for a quick smoke-test
    uv run python scripts/benchmark_snapflow_libero.py \\
        --task-ids 0 1 2 --num-episodes 5

    # Skip the baseline (no-SnapFlow) run
    uv run python scripts/benchmark_snapflow_libero.py --skip-baseline

    # Full libero_10 suite, 20 episodes (paper numbers)
    uv run python scripts/benchmark_snapflow_libero.py \\
        --task-suite libero_10 --num-episodes 20

    # Save failure videos
    uv run python scripts/benchmark_snapflow_libero.py \\
        --video-dir ./results/snapflow_libero --record-mode failures
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--pretrained",
        default="HuggingFaceVLA/smolvla_libero",
        help="HuggingFace repo ID or local path to SmolVLA pretrained weights. Default: HuggingFaceVLA/smolvla_libero",
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
        help="Skip the SnapFlow run.",
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
    return p.parse_args()


def load_policy(pretrained: str):  # noqa: ANN201
    """Load SmolVLA from a pretrained HF repo or local path."""
    from physicalai.policies import SmolVLA  # noqa: PLC0415

    print(f"Loading SmolVLA from '{pretrained}' …", flush=True)
    t0 = time.monotonic()
    policy = SmolVLA(pretrained_name_or_path=pretrained)
    policy.eval()
    elapsed = time.monotonic() - t0
    print(f"  Loaded in {elapsed:.1f}s", flush=True)
    return policy


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


def run_baseline(args: argparse.Namespace, policy) -> object:  # noqa: ANN001
    """Evaluate the policy with standard flow-matching (no SnapFlow)."""
    from physicalai.policies import SmolVLA  # noqa: PLC0415

    print(
        f"\n{'=' * 60}\nBASELINE  —  standard flow-matching ({args.baseline_steps}-step Euler)\n{'=' * 60}",
        flush=True,
    )
    # Ensure SnapFlow is off and set desired inference step count
    assert isinstance(policy, SmolVLA)
    policy.model._model._snapflow_enabled = False  # noqa: SLF001
    policy.model._model._num_steps = args.baseline_steps  # noqa: SLF001

    benchmark = make_benchmark(args, video_subdir="baseline")
    t0 = time.monotonic()
    results = benchmark.evaluate(policy)
    elapsed = time.monotonic() - t0

    print(results.summary(), flush=True)
    print(f"  Elapsed: {elapsed:.1f}s", flush=True)

    out_dir = args.output_dir / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_json(out_dir / "results.json")
    results.to_csv(out_dir / "results.csv")
    print(f"  Results written to {out_dir}", flush=True)
    return results


def run_snapflow(args: argparse.Namespace, policy) -> object:  # noqa: ANN001
    """Evaluate the policy with SnapFlow 1-NFE enabled."""
    print(
        f"\n{'=' * 60}\n"
        f"SNAPFLOW  —  {args.snapflow_steps}-step inference  "
        f"(alpha={args.snapflow_alpha}, lambda={args.snapflow_lambda})\n"
        f"{'=' * 60}",
        flush=True,
    )
    policy.enable_snapflow(
        alpha=args.snapflow_alpha,
        lambda_=args.snapflow_lambda,
        num_inference_steps=args.snapflow_steps,
    )
    vlm_frozen = not any(p.requires_grad for p in policy.model._model.vlm_with_expert.vlm.parameters())
    print(f"  VLM frozen: {vlm_frozen}", flush=True)
    print(f"  snapflow_enabled: {policy.config.snapflow_enabled}", flush=True)

    benchmark = make_benchmark(args, video_subdir="snapflow")
    t0 = time.monotonic()
    results = benchmark.evaluate(policy)
    elapsed = time.monotonic() - t0

    print(results.summary(), flush=True)
    print(f"  Elapsed: {elapsed:.1f}s", flush=True)

    out_dir = args.output_dir / "snapflow"
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_json(out_dir / "results.json")
    results.to_csv(out_dir / "results.csv")
    print(f"  Results written to {out_dir}", flush=True)
    return results


def print_comparison(baseline, snapflow) -> None:  # noqa: ANN001
    """Print a side-by-side success-rate comparison."""
    if baseline is None or snapflow is None:
        return

    b_rate = baseline.overall_success_rate
    s_rate = snapflow.overall_success_rate
    delta = s_rate - b_rate

    print(
        f"\n{'=' * 60}\n"
        f"COMPARISON\n"
        f"{'=' * 60}\n"
        f"  Baseline  (multi-step FM):  {b_rate:.1%}\n"
        f"  SnapFlow  (1-NFE):          {s_rate:.1%}\n"
        f"  Delta:                      {delta:+.1%}\n"
        f"{'=' * 60}",
        flush=True,
    )


def main() -> None:
    args = parse_args()

    if args.skip_baseline and args.skip_snapflow:
        print("Nothing to do (--skip-baseline and --skip-snapflow both set).", file=sys.stderr)
        sys.exit(1)

    policy = load_policy(args.pretrained)

    baseline_results = None
    snapflow_results = None

    if not args.skip_baseline:
        baseline_results = run_baseline(args, policy)

    if not args.skip_snapflow:
        snapflow_results = run_snapflow(args, policy)

    print_comparison(baseline_results, snapflow_results)


if __name__ == "__main__":
    main()

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: INP001
"""Benchmark Training-Time Real-Time Chunking (TT-RTC) for Pi05.

Compares training throughput and inference latency with TT-RTC enabled vs disabled.
Runs on a local dataset using the Lightning Trainer for training benchmarks and
direct model calls for inference latency.

Training benchmark:
  Measures steady-state steps/sec and peak GPU memory for Pi05 with and without
  enable_training_time_rtc. Training with TT-RTC is expected to be slightly slower
  due to per-token time embedding and prefix masking overhead.

Inference benchmark:
  Measures per-chunk latency for standard flow matching denoising vs TT-RTC
  prefix-conditioned denoising. TT-RTC inference eliminates the need for
  pseudoinverse inpainting (VJPs per denoising step), so should be similar or
  slightly faster per-chunk, with the real win being that chunks are shorter
  (postfix-only) enabling more frequent re-planning.

Usage:
    python scripts/benchmark_rtc.py --dataset-path /path/to/dataset
    python scripts/benchmark_rtc.py --inference-only  # skip training, use synthetic data
    python scripts/benchmark_rtc.py --training-only   # skip inference benchmark

Examples:
    # Full benchmark (training + inference) with local dataset
    python scripts/benchmark_rtc.py --dataset-path ~/.cache/physicalai/datasets/pick_and_place

    # Quick inference-only benchmark (no dataset needed)
    python scripts/benchmark_rtc.py --inference-only --inference-steps 50

    # Training-only with custom steps
    python scripts/benchmark_rtc.py --training-only --max-steps 100 --batch-size 4
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from lightning.pytorch.callbacks import Callback

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

DATASET_PATH = Path.home() / ".cache" / "physicalai" / "datasets" / "pick_and_place"


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class TrainingResult:
    """Stores training benchmark results for a single RTC configuration."""

    rtc_enabled: bool
    precision: str
    total_time_s: float
    warmup_steps: int
    warmup_time_s: float
    steady_steps: int
    steady_step_times: list[float] = field(default_factory=list)
    peak_memory_mb: float = 0.0
    error: str | None = None

    @property
    def label(self) -> str:
        return f"rtc={'ON' if self.rtc_enabled else 'OFF'}, precision={self.precision}"

    @property
    def steady_mean_ms(self) -> float:
        if not self.steady_step_times:
            return 0.0
        return statistics.mean(self.steady_step_times) * 1000

    @property
    def steady_median_ms(self) -> float:
        if not self.steady_step_times:
            return 0.0
        return statistics.median(self.steady_step_times) * 1000

    def to_dict(self) -> dict[str, Any]:
        return {
            "rtc_enabled": self.rtc_enabled,
            "precision": self.precision,
            "label": self.label,
            "total_time_s": self.total_time_s,
            "warmup_steps": self.warmup_steps,
            "warmup_time_s": self.warmup_time_s,
            "steady_steps": self.steady_steps,
            "steady_mean_ms": self.steady_mean_ms,
            "steady_median_ms": self.steady_median_ms,
            "peak_memory_mb": self.peak_memory_mb,
            "error": self.error,
        }


@dataclass
class InferenceResult:
    """Stores inference benchmark results for a single RTC configuration."""

    rtc_enabled: bool
    num_inference_steps: int
    num_iterations: int
    warmup_iterations: int
    latencies_ms: list[float] = field(default_factory=list)
    peak_memory_mb: float = 0.0
    error: str | None = None

    @property
    def label(self) -> str:
        return f"rtc={'ON' if self.rtc_enabled else 'OFF'}, denoise_steps={self.num_inference_steps}"

    @property
    def mean_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        return statistics.mean(self.latencies_ms)

    @property
    def median_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        return statistics.median(self.latencies_ms)

    @property
    def p95_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        sorted_lat = sorted(self.latencies_ms)
        idx = int(len(sorted_lat) * 0.95)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "rtc_enabled": self.rtc_enabled,
            "label": self.label,
            "num_inference_steps": self.num_inference_steps,
            "num_iterations": self.num_iterations,
            "warmup_iterations": self.warmup_iterations,
            "mean_ms": self.mean_ms,
            "median_ms": self.median_ms,
            "p95_ms": self.p95_ms,
            "peak_memory_mb": self.peak_memory_mb,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Training benchmark
# ---------------------------------------------------------------------------


class TimingCallback(Callback):
    """Records per-step wall-clock times, separating warmup from steady-state."""

    def __init__(self, warmup_steps: int = 5) -> None:
        self.warmup_steps = warmup_steps
        self.step_start: float = 0.0
        self.step_times: list[float] = []

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:  # noqa: ANN001, ARG002
        self.step_start = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:  # noqa: ANN001, ARG002
        self.step_times.append(time.perf_counter() - self.step_start)

    @property
    def warmup_times(self) -> list[float]:
        return self.step_times[: self.warmup_steps]

    @property
    def steady_times(self) -> list[float]:
        return self.step_times[self.warmup_steps :]


def run_training_benchmark(
    *,
    rtc_enabled: bool,
    precision: str,
    max_steps: int,
    batch_size: int,
    dataset_path: Path,
    warmup_steps: int,
    variant: str,
) -> TrainingResult:
    """Run a single training benchmark with or without TT-RTC."""
    from physicalai.data.lerobot import LeRobotDataModule  # noqa: PLC0415
    from physicalai.policies import Pi05  # noqa: PLC0415
    from physicalai.train import Trainer  # noqa: PLC0415

    label = f"rtc={'ON' if rtc_enabled else 'OFF'}, precision={precision}"
    logger.info("\n%s", "=" * 70)
    logger.info("  TRAINING: %s", label)
    logger.info("  (%d warmup + %d measured steps)", warmup_steps, max_steps - warmup_steps)
    logger.info("%s", "=" * 70)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    try:
        datamodule = LeRobotDataModule(
            repo_id="local",
            root=str(dataset_path),
            train_batch_size=batch_size,
            data_format="physicalai",
        )

        policy = Pi05(
            paligemma_variant=variant,
            action_expert_variant="gemma_300m",
            compile_model=False,
            gradient_checkpointing=False,
            enable_training_time_rtc=rtc_enabled,
        )

        timer = TimingCallback(warmup_steps=warmup_steps)
        trainer = Trainer(
            max_steps=max_steps,
            precision=precision,
            accelerator="gpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=True,
            callbacks=[timer],
        )

        start = time.perf_counter()
        trainer.fit(policy, datamodule)
        total_time = time.perf_counter() - start

        peak_mem = 0.0
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)

        return TrainingResult(
            rtc_enabled=rtc_enabled,
            precision=precision,
            total_time_s=total_time,
            warmup_steps=len(timer.warmup_times),
            warmup_time_s=sum(timer.warmup_times),
            steady_steps=len(timer.steady_times),
            steady_step_times=timer.steady_times,
            peak_memory_mb=peak_mem,
        )
    except Exception as e:
        logger.exception("  TRAINING ERROR")
        return TrainingResult(
            rtc_enabled=rtc_enabled,
            precision=precision,
            total_time_s=0.0,
            warmup_steps=0,
            warmup_time_s=0.0,
            steady_steps=0,
            peak_memory_mb=0.0,
            error=str(e),
        )
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Inference benchmark
# ---------------------------------------------------------------------------


def _build_synthetic_model(*, rtc_enabled: bool, variant: str, dtype: str, device: torch.device) -> Any:
    """Instantiate a Pi05Model with synthetic dataset_stats."""
    from physicalai.policies.pi05.model import Pi05Model  # noqa: PLC0415

    dataset_stats = {
        "observation.state": {
            "name": "observation.state",
            "shape": (8,),
            "type": "STATE",
            "mean": [0.0] * 8,
            "std": [1.0] * 8,
            "q01": [-1.0] * 8,
            "q99": [1.0] * 8,
        },
        "action": {
            "name": "action",
            "shape": (32,),
            "type": "ACTION",
            "mean": [0.0] * 32,
            "std": [1.0] * 32,
            "q01": [-1.0] * 32,
            "q99": [1.0] * 32,
        },
        "observation.image.top": {
            "name": "observation.image.top",
            "shape": (3, 224, 224),
            "type": "VISUAL",
        },
    }

    model = Pi05Model(
        dataset_stats,
        paligemma_variant=variant,
        action_expert_variant="gemma_300m",
        dtype=dtype,
        compile_model=False,
        gradient_checkpointing=False,
        enable_training_time_rtc=rtc_enabled,
    )
    model.to(device)
    model.eval()
    return model


def run_inference_benchmark(
    *,
    rtc_enabled: bool,
    num_iterations: int,
    warmup_iterations: int,
    num_inference_steps: int,
    variant: str,
    dtype: str,
) -> InferenceResult:
    """Run inference latency benchmark with or without TT-RTC prefix conditioning."""
    label = f"rtc={'ON' if rtc_enabled else 'OFF'}"
    logger.info("\n%s", "=" * 70)
    logger.info("  INFERENCE: %s (%d denoise steps)", label, num_inference_steps)
    logger.info("  (%d warmup + %d measured iterations)", warmup_iterations, num_iterations)
    logger.info("%s", "=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    try:
        model = _build_synthetic_model(
            rtc_enabled=rtc_enabled,
            variant=variant,
            dtype=dtype,
            device=device,
        )

        batch = model.sample_input

        if rtc_enabled and "action_prefix" not in batch:
            chunk_size = model._chunk_size
            action_dim = int(model._dataset_stats["action"]["shape"][-1])
            batch["action_prefix"] = torch.zeros(
                1,
                chunk_size,
                action_dim,
                device=device,
            )
            batch["delay"] = torch.tensor(5, device=device)

        total_iters = warmup_iterations + num_iterations
        latencies: list[float] = []

        with torch.no_grad():
            for i in range(total_iters):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                start = time.perf_counter()
                model.predict_action_chunk(batch)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                elapsed_ms = (time.perf_counter() - start) * 1000

                if i >= warmup_iterations:
                    latencies.append(elapsed_ms)

                if i < warmup_iterations:
                    logger.info("    warmup %d/%d: %.1f ms", i + 1, warmup_iterations, elapsed_ms)
                elif i < warmup_iterations + 5:
                    logger.info("    iter %d: %.1f ms", i - warmup_iterations + 1, elapsed_ms)

        peak_mem = 0.0
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)

        return InferenceResult(
            rtc_enabled=rtc_enabled,
            num_inference_steps=num_inference_steps,
            num_iterations=num_iterations,
            warmup_iterations=warmup_iterations,
            latencies_ms=latencies,
            peak_memory_mb=peak_mem,
        )
    except Exception as e:
        logger.exception("  INFERENCE ERROR")
        return InferenceResult(
            rtc_enabled=rtc_enabled,
            num_inference_steps=num_inference_steps,
            num_iterations=0,
            warmup_iterations=0,
            error=str(e),
        )
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_training_results(results: list[TrainingResult]) -> None:
    """Print training benchmark comparison table."""
    logger.info("\n%s", "=" * 90)
    logger.info("  TRAINING BENCHMARK RESULTS (steady-state, excluding warmup)")
    logger.info("%s", "=" * 90)

    header = f"{'Configuration':<35} {'Steps':>6} {'Mean':>10} {'Median':>10} {'Peak Mem':>10} {'vs Baseline':>12}"
    logger.info(header)
    logger.info("%s", "-" * 90)

    baseline = next((r for r in results if not r.rtc_enabled and not r.error), None)

    for r in results:
        if r.error:
            logger.info("%-35s %6s   %s", r.label, "FAILED", r.error)
            continue

        speedup = ""
        if baseline and r.rtc_enabled and baseline.steady_mean_ms > 0:
            ratio = baseline.steady_mean_ms / r.steady_mean_ms
            speedup = f"{ratio:.3f}x"

        mem_str = f"{r.peak_memory_mb:.0f} MB" if r.peak_memory_mb > 0 else "N/A"
        logger.info(
            "%-35s %6d %7.1f ms %7.1f ms %10s %12s",
            r.label,
            r.steady_steps,
            r.steady_mean_ms,
            r.steady_median_ms,
            mem_str,
            speedup,
        )

    logger.info("%s", "=" * 90)


def print_inference_results(results: list[InferenceResult]) -> None:
    """Print inference benchmark comparison table."""
    logger.info("\n%s", "=" * 90)
    logger.info("  INFERENCE BENCHMARK RESULTS (per-chunk latency)")
    logger.info("%s", "=" * 90)

    header = f"{'Configuration':<35} {'Iters':>6} {'Mean':>10} {'Median':>10} {'P95':>10} {'Peak Mem':>10} {'vs Baseline':>12}"
    logger.info(header)
    logger.info("%s", "-" * 90)

    baseline = next((r for r in results if not r.rtc_enabled and not r.error), None)

    for r in results:
        if r.error:
            logger.info("%-35s %6s   %s", r.label, "FAILED", r.error)
            continue

        speedup = ""
        if baseline and r.rtc_enabled and baseline.mean_ms > 0:
            ratio = baseline.mean_ms / r.mean_ms
            speedup = f"{ratio:.3f}x"

        mem_str = f"{r.peak_memory_mb:.0f} MB" if r.peak_memory_mb > 0 else "N/A"
        logger.info(
            "%-35s %6d %7.1f ms %7.1f ms %7.1f ms %10s %12s",
            r.label,
            r.num_iterations,
            r.mean_ms,
            r.median_ms,
            r.p95_ms,
            mem_str,
            speedup,
        )

    logger.info("%s", "=" * 90)
    logger.info("")
    logger.info("  NOTE: Per-chunk latency should be similar for RTC ON vs OFF.")
    logger.info("  The real-time benefit of TT-RTC is at the system level:")
    logger.info("  - Standard: must wait for full chunk before acting (chunk_size steps)")
    logger.info("  - TT-RTC: acts on postfix immediately, re-plans after chunk_size - delay steps")
    logger.info("  - No pseudoinverse inpainting (VJPs per denoising step) needed at inference")
    logger.info("")


def save_results(
    training_results: list[TrainingResult],
    inference_results: list[InferenceResult],
    output_path: Path,
    args: argparse.Namespace,
) -> None:
    """Save all results to JSON."""
    data: dict[str, Any] = {
        "config": {
            "variant": args.variant,
            "precision": args.precision,
            "max_steps": args.max_steps,
            "batch_size": args.batch_size,
            "warmup_steps": args.warmup_steps,
            "inference_steps": args.inference_steps,
            "inference_warmup": args.inference_warmup,
            "num_inference_denoise_steps": args.num_inference_steps,
        },
    }
    if training_results:
        data["training"] = [r.to_dict() for r in training_results]
    if inference_results:
        data["inference"] = [r.to_dict() for r in inference_results]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.info("Results saved to %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the TT-RTC benchmark."""
    parser = argparse.ArgumentParser(
        description="Benchmark TT-RTC for Pi05: training throughput and inference latency",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset-path", type=str, default=str(DATASET_PATH), help="Path to local dataset")
    parser.add_argument("--max-steps", type=int, default=100, help="Total training steps (default: 100)")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size (default: 4)")
    parser.add_argument("--warmup-steps", type=int, default=10, help="Training warmup steps (default: 10)")
    parser.add_argument("--precision", type=str, default="bf16-mixed", help="Training precision (default: bf16-mixed)")
    parser.add_argument(
        "--variant",
        type=str,
        default="gemma_2b",
        help="PaliGemma variant: gemma_300m or gemma_2b (default: gemma_2b)",
    )
    parser.add_argument("--inference-steps", type=int, default=30, help="Inference iterations to measure (default: 30)")
    parser.add_argument("--inference-warmup", type=int, default=5, help="Inference warmup iterations (default: 5)")
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=10,
        help="Number of denoising steps per chunk (default: 10)",
    )
    parser.add_argument("--training-only", action="store_true", help="Skip inference benchmark")
    parser.add_argument("--inference-only", action="store_true", help="Skip training benchmark")
    parser.add_argument(
        "--output-path",
        type=str,
        default="benchmark_rtc_results.json",
        help="Path to save results JSON (default: benchmark_rtc_results.json)",
    )
    args = parser.parse_args()

    if args.training_only and args.inference_only:
        logger.error("Cannot use both --training-only and --inference-only")
        return

    training_results: list[TrainingResult] = []
    inference_results: list[InferenceResult] = []

    # --- Training benchmark ---
    if not args.inference_only:
        dataset_path = Path(args.dataset_path)
        if not dataset_path.exists():
            logger.error("Dataset not found at %s", dataset_path)
            logger.error("Use --dataset-path to specify the local dataset, or --inference-only to skip training.")
            return

        logger.info("\n%s", "#" * 70)
        logger.info("  PI05 TT-RTC TRAINING BENCHMARK")
        logger.info("  variant=%s, precision=%s, batch_size=%d", args.variant, args.precision, args.batch_size)
        logger.info("%s", "#" * 70)

        for rtc_enabled in [False, True]:
            result = run_training_benchmark(
                rtc_enabled=rtc_enabled,
                precision=args.precision,
                max_steps=args.max_steps,
                batch_size=args.batch_size,
                dataset_path=dataset_path,
                warmup_steps=args.warmup_steps,
                variant=args.variant,
            )
            training_results.append(result)

        print_training_results(training_results)

    # --- Inference benchmark ---
    if not args.training_only:
        dtype = "float32"
        if "bf16" in args.precision:
            dtype = "bfloat16"

        logger.info("\n%s", "#" * 70)
        logger.info("  PI05 TT-RTC INFERENCE BENCHMARK")
        logger.info("  variant=%s, dtype=%s, denoise_steps=%d", args.variant, dtype, args.num_inference_steps)
        logger.info("%s", "#" * 70)

        for rtc_enabled in [False, True]:
            result = run_inference_benchmark(
                rtc_enabled=rtc_enabled,
                num_iterations=args.inference_steps,
                warmup_iterations=args.inference_warmup,
                num_inference_steps=args.num_inference_steps,
                variant=args.variant,
                dtype=dtype,
            )
            inference_results.append(result)

        print_inference_results(inference_results)

    # --- Save ---
    save_results(training_results, inference_results, Path(args.output_path), args)


if __name__ == "__main__":
    main()

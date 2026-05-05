"""Train Pi05 with and without SnapFlow, export to PyTorch + OpenVINO, and benchmark all variants.

Usage:
    cd physical-ai-studio
    python scripts/snapflow_benchmark.py [--device gpu|cpu] [--warmup 10] [--iterations 50]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from physicalai.data.lerobot import LeRobotDataModule
from physicalai.inference import InferenceModel
from physicalai.policies import Pi05
from physicalai.train import Trainer

EXPORT_DIR = Path("experiments/snapflow_benchmark")

BACKENDS = ["torch", "openvino"]
VARIANTS = [
    {"tag": "baseline", "snapflow_enabled": False},
    {"tag": "snapflow", "snapflow_enabled": True},
]


def export_path(tag: str, backend: str) -> Path:
    return EXPORT_DIR / f"pi05_{tag}_{backend}"


def train_and_export(snapflow_enabled: bool, tag: str, device: str) -> None:
    """Train Pi05 for 1 step and export to all backends."""
    accelerator = "gpu" if device == "gpu" else "cpu"
    precision = "bf16-mixed" if device == "gpu" else 32

    datamodule = LeRobotDataModule(
        repo_id="lerobot/aloha_sim_transfer_cube_human",
        train_batch_size=2,
        data_format="physicalai",
    )

    model = Pi05(
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        dtype="bfloat16" if device == "gpu" else "float32",
        chunk_size=50,
        n_action_steps=50,
        max_state_dim=32,
        max_action_dim=32,
        num_inference_steps=10,
        snapflow_enabled=snapflow_enabled,
        snapflow_alpha=0.5,
        snapflow_lambda=1.0,
        snapflow_num_inference_steps=1,
    )

    trainer = Trainer(
        max_steps=1,
        accelerator=accelerator,
        devices=1,
        precision=precision,
        default_root_dir=str(EXPORT_DIR / f"training_{tag}"),
        enable_checkpointing=True,
    )
    print(f"\n{'=' * 60}")
    print(f"Training {tag} model (1 step)...")
    print(f"{'=' * 60}")
    trainer.fit(model=model, datamodule=datamodule)

    model.eval()
    for backend in BACKENDS:
        path = export_path(tag, backend)
        print(f"Exporting {tag} → {backend} at {path}...")
        model.export(str(path), backend=backend)
    print(f"All exports complete for {tag}")


def benchmark_latency(
    model_path: Path,
    label: str,
    warmup: int = 10,
    iterations: int = 50,
) -> dict[str, float]:
    """Benchmark inference latency for an exported model."""
    print(f"\n  {label}")
    print(f"  {'-' * 40}")

    model = InferenceModel.load(model_path)
    model.reset()

    obs: dict[str, np.ndarray] = {
        "observation.state": np.random.randn(1, 32).astype(np.float32),
        "observation.images.top": np.random.randn(1, 3, 224, 224).astype(np.float32),
        "task": np.array(["pick up the cube"]),
    }

    for _ in range(warmup):
        model.reset()
        try:
            model.select_action(obs)
        except Exception:
            pass

    latencies = []
    for _ in range(iterations):
        model.reset()
        t0 = time.perf_counter()
        try:
            model.select_action(obs)
        except Exception:
            pass
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)

    arr = np.array(latencies)
    results = {
        "mean_ms": float(np.mean(arr)),
        "median_ms": float(np.median(arr)),
        "std_ms": float(np.std(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
    }

    for k in ["mean_ms", "median_ms", "p95_ms", "min_ms", "max_ms"]:
        print(f"    {k.replace('_ms', '').capitalize():<8} {results[k]:>10.2f} ms")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="SnapFlow benchmark for Pi05")
    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--skip-training", action="store_true", help="Skip training, only benchmark existing exports")
    args = parser.parse_args()

    if not args.skip_training:
        for variant in VARIANTS:
            train_and_export(
                snapflow_enabled=variant["snapflow_enabled"],
                tag=variant["tag"],
                device=args.device,
            )
    else:
        missing = [
            export_path(v["tag"], b) for v in VARIANTS for b in BACKENDS if not export_path(v["tag"], b).exists()
        ]
        if missing:
            print(f"Error: missing exports: {[str(p) for p in missing]}")
            print("Run without --skip-training first.")
            return

    all_results: dict[str, dict[str, float]] = {}

    print(f"\n{'=' * 60}")
    print("BENCHMARKING ALL VARIANTS")
    print(f"{'=' * 60}")
    print(f"  Warmup: {args.warmup} | Iterations: {args.iterations}")

    for variant in VARIANTS:
        for backend in BACKENDS:
            path = export_path(variant["tag"], backend)
            label = f"{variant['tag']} / {backend}"
            all_results[label] = benchmark_latency(path, label, warmup=args.warmup, iterations=args.iterations)

    header_labels = list(all_results.keys())
    col_w = max(len(l) for l in header_labels) + 2

    print(f"\n{'=' * 60}")
    print("COMPARISON SUMMARY")
    print(f"{'=' * 60}")

    header = f"{'Metric':<10}"
    for label in header_labels:
        header += f" {label:>{col_w}}"
    print(header)
    print("-" * len(header))

    for metric in ["mean_ms", "median_ms", "p95_ms"]:
        row = f"{metric.replace('_ms', '').capitalize():<10}"
        for label in header_labels:
            val = all_results[label][metric]
            row += f" {val:>{col_w - 2}.2f}ms"
        print(row)

    ref_key = header_labels[0]
    ref_mean = all_results[ref_key]["mean_ms"]
    speedup_row = f"{'Speedup':<10}"
    for label in header_labels:
        val = all_results[label]["mean_ms"]
        speedup = ref_mean / val if val > 0 else float("inf")
        speedup_row += f" {speedup:>{col_w - 2}.2f}x"
    print(speedup_row)


if __name__ == "__main__":
    main()

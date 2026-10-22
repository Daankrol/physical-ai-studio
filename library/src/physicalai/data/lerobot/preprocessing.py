# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Offline depth maps and teacher targets, staged for safe dataset publication."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
from io import BytesIO
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from datasets import Features
from datasets import Image as ImageFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image
from safetensors.torch import save_file
from torch.nn import functional

from .feature_cache import CACHE_DIRECTORY, TARGET_NAMES, dataset_fingerprint

if TYPE_CHECKING:
    from collections.abc import Callable


def _validate_image_paths(root: Path, features: dict) -> None:
    keys = [key for key, feature in features.items() if feature.get("dtype") == "image"]
    if not keys:
        return
    for path in (root / "data").glob("*/*.parquet"):
        for batch in pq.ParquetFile(path).iter_batches(columns=keys, batch_size=8):
            for key in keys:
                images = batch.column(key)
                if not pa.types.is_struct(images.type):
                    msg = "Image features must use LeRobot image storage"
                    raise ValueError(msg)
                for image_path, missing_bytes in zip(
                    images.field("path").to_pylist(),
                    images.field("bytes").is_null().to_pylist(),
                    strict=True,
                ):
                    if not missing_bytes:
                        continue
                    if (
                        not isinstance(image_path, str)
                        or (urlsplit(image_path).scheme and not Path(image_path).is_absolute())
                        or not (root / image_path).resolve().is_relative_to(root)
                    ):
                        msg = "Image references must remain inside the source dataset"
                        raise ValueError(msg)


def _rgb(images: torch.Tensor, device: str) -> torch.Tensor:
    images = images.to(device).float() if images.dtype == torch.uint8 else images.to(device).float() * 255
    return (
        functional
        .interpolate(images, (256, 256), mode="bilinear", align_corners=False, antialias=True)
        .round()
        .clamp(0, 255)
    )


def _write_depth(  # noqa: PLR0914 -- streaming rewrite owns the depth statistics
    source: LeRobotDataset,
    output: Path,
    rgb_key: str,
    depth_key: str,
    model: Any,  # noqa: ANN401 -- upstream inference API
    device: str,
    batch_size: int,
    progress: Callable[[int], None],
) -> None:
    info = json.loads((output / "meta/info.json").read_text())
    if depth_key in info["features"]:
        msg = "The output depth feature already exists; choose a new feature name"
        raise ValueError(msg)
    first = source[0][rgb_key]
    height, width = first.shape[-2:]
    info["features"][depth_key] = {
        "dtype": "image",
        "shape": [height, width, 1],
        "names": ["height", "width", "channels"],
        "info": {
            "is_depth_map": True,
            "depth_unit": "m",
            "aligned_to": rgb_key,
            "source": "moge-2-vitb-normal",
            "preview_path": f".physicalai/depth/{depth_key}/episode_{{episode_index:06d}}.mp4",
        },
    }
    count, total, squared, low, high = 0, 0.0, 0.0, float("inf"), 0.0
    done = 0
    for path in sorted((source.root / "data").glob("*/*.parquet")):
        relative = path.relative_to(source.root)
        destination = output / relative
        temporary = destination.with_suffix(".tmp")
        parquet = pq.ParquetFile(path)
        features = Features.from_arrow_schema(parquet.schema_arrow)
        features[depth_key] = ImageFeature()
        with pq.ParquetWriter(temporary, features.arrow_schema) as writer:
            for batch in parquet.iter_batches(batch_size=batch_size):
                rows = pa.Table.from_batches([batch])
                images = torch.stack([source[int(index)][rgb_key] for index in rows.column("index").to_pylist()])
                images = images.float() / 255 if images.dtype == torch.uint8 else images.float()
                with (
                    torch.inference_mode(),
                    torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")),
                ):
                    predicted = model.infer(
                        images.to(device),
                        resolution_level=3,
                        num_tokens=256,
                        apply_mask=False,
                    )["depth"]
                predicted = torch.nan_to_num(predicted.float().cpu(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0)
                if tuple(predicted.shape) != (rows.num_rows, height, width):
                    msg = "Predicted depth must match the batch and RGB pixel grid"
                    raise ValueError(msg)
                depths = []
                for depth in predicted:
                    array = depth.numpy()
                    buffer = BytesIO()
                    Image.fromarray(array).save(buffer, format="TIFF")
                    depths.append({"bytes": buffer.getvalue(), "path": None})
                    count += array.size
                    total += depth.double().sum().item()
                    squared += depth.double().square().sum().item()
                    low, high = min(low, depth.min().item()), max(high, depth.max().item())
                    done += 1
                    progress(done)
                rows = rows.append_column(depth_key, pa.array(depths, type=features.arrow_schema.field(depth_key).type))
                writer.write_table(rows.cast(features.arrow_schema))
        temporary.replace(destination)
    (output / "meta/info.json").write_text(json.dumps(info, indent=2) + "\n")
    stats_path = output / "meta/stats.json"
    stats = json.loads(stats_path.read_text())
    mean = total / count
    stats[depth_key] = {
        "mean": [[[mean]]],
        "std": [[[max(0.0, squared / count - mean * mean) ** 0.5]]],
        "min": [[[low]]],
        "max": [[[high]]],
        "count": [done],
    }
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")


def _write_depth_previews(output: Path, depth_key: str, fps: int) -> None:
    """Encode display-only previews; lossless metric maps stay in parquet."""
    info_path = output / "meta/info.json"
    info = json.loads(info_path.read_text())
    feature = info["features"][depth_key]
    decoder = (
        LeRobotDataset(repo_id=output.name, root=output, depth_output_unit="m") if feature["dtype"] == "video" else None
    )
    directory = output / ".physicalai/depth" / depth_key
    directory.mkdir(parents=True, exist_ok=True)
    rows = (
        row
        for path in sorted((output / "data").glob("*/*.parquet"))
        for batch in pq.ParquetFile(path).iter_batches(columns=["index", "episode_index", depth_key], batch_size=8)
        for row in batch.to_pylist()
    )
    for episode, frames in groupby(rows, key=itemgetter("episode_index")):
        with av.open(str(directory / f"episode_{episode:06d}.mp4"), mode="w") as container:
            stream = container.add_stream("libx264", rate=fps)
            height, width = feature["shape"][:2]
            stream.width, stream.height = width + width % 2, height + height % 2
            stream.pix_fmt = "yuv420p"
            for row in frames:
                if decoder is not None:
                    depth = decoder[row["index"]][depth_key].squeeze(0).numpy()
                else:
                    stored = row[depth_key]
                    image_source = BytesIO(stored["bytes"]) if stored["bytes"] is not None else output / stored["path"]
                    with Image.open(image_source) as image:
                        depth = np.array(image).astype(np.float32)
                    if (feature.get("info") or {}).get("depth_unit") == "mm":
                        depth /= 1000
                # A fixed metric scale avoids flicker from per-frame normalization.
                gray = np.where(depth > 0, 255 / (1 + depth), 0).astype(np.uint8)
                gray = np.pad(gray, ((0, gray.shape[0] % 2), (0, gray.shape[1] % 2)))
                for packet in stream.encode(av.VideoFrame.from_ndarray(gray, format="gray")):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
    feature.setdefault("info", {})["preview_path"] = f".physicalai/depth/{depth_key}/episode_{{episode_index:06d}}.mp4"
    info_path.write_text(json.dumps(info, indent=2) + "\n")


def _write_targets(
    source: LeRobotDataset,
    output: Path,
    rgb_key: str,
    depth_key: str | None,
    chunk_size: int,
    teachers: Any,  # noqa: ANN401 -- injectable target extractor
    device: str,
    progress: Callable[[int], None],
) -> None:
    directory = output / CACHE_DIRECTORY
    directory.mkdir(parents=True)
    shard_size = 16
    pending: dict[str, list[torch.Tensor]] = {key: [] for key in TARGET_NAMES}
    for index in range(len(source)):
        sample = source[index]
        rgb = _rgb(sample[rgb_key], device)[None]
        batch = {
            "teacher_current_rgb": rgb[:, :1],
            "teacher_future_rgb": rgb[:, 1:],
            "teacher_fps": torch.tensor([source.fps / max(1, chunk_size - 1)], device=device),
        }
        if depth_key is not None:
            depth = sample[depth_key]
            if depth.shape[0:2] != (2, 1) or depth.shape[-2:] != sample[rgb_key].shape[-2:]:
                msg = "Measured depth must share the RGB pixel grid"
                raise ValueError(msg)
            depth = functional.interpolate(depth.to(device).float(), (256, 256), mode="nearest")
            batch["teacher_current_depth"], batch["teacher_future_depth"] = depth[:1], depth[1:]
        result = teachers(batch)
        for name in TARGET_NAMES:
            values = result[f"{name}_target"].detach().cpu().to(torch.bfloat16)
            if values.shape != (1, 256, 1024) or not values.isfinite().all():
                msg = "Teacher produced invalid released-checkpoint target features"
                raise ValueError(msg)
            pending[name].append(values)
        if (index + 1) % shard_size == 0 or index == len(source) - 1:
            save_file(
                {name: torch.cat(values).contiguous() for name, values in pending.items()},
                directory / f"{index // shard_size:06d}.safetensors",
            )
            pending = {key: [] for key in TARGET_NAMES}
        progress(index + 1)
    manifest = {
        "version": 1,
        "targets": list(TARGET_NAMES),
        "frames": len(source),
        "shard_size": shard_size,
        "chunk_size": chunk_size,
        "tokens": 256,
        "feature_dim": 1024,
        "rgb_key": rgb_key,
        "depth_key": depth_key,
        "teachers": teachers.references,
        "source_sha256": dataset_fingerprint(output),
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def preprocess_dataset(  # noqa: C901, PLR0912, PLR0914, PLR0915 -- one atomic publication lifecycle
    source_root: str | Path,
    output_root: str | Path,
    *,
    operation: str,
    rgb_key: str | None = None,
    depth_key: str | None = None,
    depth_aligned: bool = False,
    chunk_size: int = 50,
    batch_size: int = 1,
    device: str = "cuda:0",
    teacher_weights_dir: str | Path | None = None,
    progress: Callable[[int], None] | None = None,
    models: Any = None,  # noqa: ANN401 -- injectable model/target extractor
) -> Path:
    """Produce a new dataset atomically; never overwrite recorded source data.

    Returns:
        The new dataset root after successful publication.

    Raises:
        ValueError: If source metadata, alignment, destination, or disk space is invalid.
    """
    if type(batch_size) is not int or not 1 <= batch_size <= 64:  # noqa: PLR2004 -- bounded GPU batch
        msg = "Depth batch size must be between 1 and 64"
        raise ValueError(msg)
    if operation == "teacher_targets" and batch_size != 1:
        msg = "Teacher-target caching supports batch size 1"
        raise ValueError(msg)
    source_root, output_root = Path(source_root).resolve(), Path(output_root).resolve()
    if output_root.exists() or output_root.is_relative_to(source_root) or source_root.is_relative_to(output_root):
        msg = "Preprocessing output must be a new directory outside the source dataset"
        raise ValueError(msg)
    for feature in (rgb_key, depth_key):
        if feature is not None and ("/" in feature or "\\" in feature or ".." in feature):
            msg = "Dataset feature names cannot contain path components"
            raise ValueError(msg)
    if operation not in {"depth", "teacher_targets"} or chunk_size < 1:
        msg = "Unknown preprocessing operation or invalid action horizon"
        raise ValueError(msg)
    if any(path.is_symlink() for path in source_root.rglob("*")):
        msg = "Materialize dataset symlinks before preprocessing"
        raise ValueError(msg)
    source_info = json.loads((source_root / "meta/info.json").read_text())
    for key in ("data_path", "video_path"):
        template = source_info.get(key)
        if template is not None and (
            not isinstance(template, str) or Path(template).is_absolute() or ".." in Path(template).parts
        ):
            msg = "Dataset path templates must remain inside the dataset"
            raise ValueError(msg)
    features = source_info["features"]
    depth_keys = {
        key
        for key, feature in features.items()
        if (feature.get("info") or {}).get("is_depth_map")
        or (feature.get("info") or {}).get("video.is_depth_map")
        or (feature.get("video_info") or {}).get("video.is_depth_map")
    }
    cameras = [
        key for key, feature in features.items() if feature.get("dtype") in {"image", "video"} and key not in depth_keys
    ]
    missing = [
        key
        for key in cameras
        if not any((features[depth].get("info") or {}).get("aligned_to") == key for depth in depth_keys)
    ]
    if operation == "depth":
        if rgb_key is not None or depth_key is not None:
            msg = "Depth computation automatically processes all RGB cameras"
            raise ValueError(msg)
        for key in missing:
            if not key.startswith("observation.") or "/" in key or "\\" in key or ".." in key:
                msg = "Invalid RGB camera feature name"
                raise ValueError(msg)
            if f"{key}_depth" in features:
                msg = "Automatic depth feature name conflicts with existing data"
                raise ValueError(msg)
        rgb_key = next(iter(cameras), None)
    rgb_feature = features.get(rgb_key, {})
    rgb_info = rgb_feature.get("info") or {}
    rgb_legacy = rgb_feature.get("video_info") or {}
    if (
        rgb_key is None
        or rgb_feature.get("dtype") not in {"image", "video"}
        or any((
            rgb_info.get("is_depth_map"),
            rgb_info.get("video.is_depth_map"),
            rgb_legacy.get("video.is_depth_map"),
        ))
    ):
        msg = "Select an RGB source feature"
        raise ValueError(msg)
    if operation == "teacher_targets" and depth_key is not None:
        depth_feature = source_info["features"].get(depth_key, {})
        metadata = depth_feature.get("info") or {}
        legacy = depth_feature.get("video_info") or {}
        marked = metadata.get("is_depth_map") or metadata.get("video.is_depth_map") or legacy.get("video.is_depth_map")
        if not marked or metadata.get("depth_unit") not in {"m", "mm"}:
            msg = "Measured depth needs LeRobot depth metadata and explicit units"
            raise ValueError(msg)
        if metadata.get("aligned_to") != rgb_key and not depth_aligned:
            msg = "Confirm calibrated depth alignment to the selected RGB camera"
            raise ValueError(msg)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    original = dataset_fingerprint(source_root)
    _validate_image_paths(source_root, source_info["features"])
    frames = source_info.get("total_frames")
    shape = source_info["features"][rgb_key].get("shape")
    if type(frames) is not int or frames < 1:
        msg = "Cannot preprocess an empty dataset"
        raise ValueError(msg)
    if (
        not isinstance(shape, (list, tuple))
        or len(shape) != 3  # noqa: PLR2004 -- image axes
        or any(type(size) is not int or size < 1 for size in shape)
    ):
        msg = "RGB metadata must contain positive image dimensions"
        raise ValueError(msg)
    bytes_required = sum(path.stat().st_size for path in source_root.rglob("*") if path.is_file())
    bytes_required += frames * (
        sum(math.prod(features[key]["shape"]) * 4 for key in missing) if operation == "depth" else 4 * 256 * 1024 * 2
    )
    if shutil.disk_usage(output_root.parent).free < bytes_required * 1.1:
        msg = "Insufficient disk space for a non-destructive preprocessing copy"
        raise ValueError(msg)
    stage = Path(tempfile.mkdtemp(prefix=".preprocess-", dir=output_root.parent))
    try:

        def ignore_cache(directory: str, _names: list[str]) -> list[str]:
            return ["lingbotvla2"] if Path(directory) == source_root / ".physicalai" else []

        shutil.copytree(source_root, stage, dirs_exist_ok=True, ignore=ignore_cache)
        source = LeRobotDataset(repo_id=source_root.name, root=stage, return_uint8=True, depth_output_unit="m")
        report = progress or (lambda _: None)
        if operation == "depth":
            if missing:
                from physicalai.policies.lingbotvla2.teacher_models import load_moge  # noqa: PLC0415

                model = models if models is not None else load_moge(teacher_weights_dir)
                model.eval().requires_grad_(False).to(device)  # noqa: FBT003
                for camera_index, camera in enumerate(missing):

                    def camera_progress(done: int, offset: int = camera_index * frames) -> None:
                        report((offset + done) // len(missing))

                    _write_depth(source, stage, camera, f"{camera}_depth", model, device, batch_size, camera_progress)
            preview_keys = {f"{camera}_depth" for camera in missing} | {
                key
                for key in depth_keys
                if (features[key].get("info") or {}).get("aligned_to") in cameras
                and (features[key].get("info") or {}).get("depth_unit") in {"m", "mm"}
            }
            for key in sorted(preview_keys):
                if "/" in key or "\\" in key or ".." in key:
                    msg = "Invalid depth camera feature name"
                    raise ValueError(msg)
                _write_depth_previews(stage, key, source.fps)
        else:
            offsets = [0, (chunk_size - 1) / source.fps]
            source = LeRobotDataset(
                repo_id=source_root.name,
                root=stage,
                return_uint8=True,
                depth_output_unit="m",
                delta_timestamps={key: offsets for key in (rgb_key, depth_key) if key is not None},
            )
            from physicalai.policies.lingbotvla2.teacher_models import load_teachers  # noqa: PLC0415

            teachers = (
                models
                if models is not None
                else load_teachers(teacher_weights_dir, measured_depth=depth_key is not None, device=device)
            )
            _write_targets(source, stage, rgb_key, depth_key, chunk_size, teachers, device, report)
        if dataset_fingerprint(source_root) != original:
            msg = "Source dataset changed during preprocessing; retry after recording stops"
            raise ValueError(msg)
        stage.rename(output_root)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return output_root


def main() -> None:
    """Run the preprocessing subprocess used by Studio's job worker."""
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--operation", choices=["depth", "teacher_targets"], required=True)
    parser.add_argument("--rgb-key")
    parser.add_argument("--depth-key")
    parser.add_argument("--depth-aligned", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--parent-pid", type=int)
    args = parser.parse_args()
    total = json.loads((Path(args.source) / "meta/info.json").read_text())["total_frames"]

    def progress(done: int) -> None:
        if args.parent_pid is not None and os.getppid() != args.parent_pid:
            msg = "Preprocessing owner exited"
            raise RuntimeError(msg)
        print(json.dumps({"progress": min(99, int(100 * done / total))}), flush=True)  # noqa: T201 -- CLI worker protocol

    preprocess_dataset(
        args.source,
        args.output,
        operation=args.operation,
        rgb_key=args.rgb_key,
        depth_key=args.depth_key,
        depth_aligned=args.depth_aligned,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        device=args.device,
        progress=progress,
    )


if __name__ == "__main__":
    main()

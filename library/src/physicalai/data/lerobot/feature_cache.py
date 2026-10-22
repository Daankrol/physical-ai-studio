# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Versioned, sharded auxiliary features attached to an immutable LeRobot dataset."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

from safetensors import safe_open

if TYPE_CHECKING:
    from pathlib import Path

    from torch import Tensor

CACHE_DIRECTORY = ".physicalai/lingbotvla2"
TARGET_NAMES = ("depth", "future_depth", "current_video", "future_video")


def dataset_fingerprint(root: Path) -> str:
    """Hash source metadata and data, excluding derived caches and normalization stats.

    Returns:
        SHA-256 digest independent of the dataset directory name.

    Raises:
        ValueError: If source files escape the dataset root.
    """
    root = root.resolve()
    digest = hashlib.sha256()
    paths = [root / "meta/info.json", root / "meta/tasks.parquet"]
    for name in ("data", "videos", "images", "meta/episodes"):
        paths.extend(path for path in (root / name).rglob("*") if path.is_file())
    for path in sorted(paths):
        if not path.resolve().is_relative_to(root):
            msg = "Dataset source files must not escape the dataset root"
            raise ValueError(msg)
        if not path.is_file():
            continue
        digest.update(str(path.relative_to(root)).encode())
        with path.open("rb") as source:
            while block := source.read(8 * 1024 * 1024):
                digest.update(block)
    return digest.hexdigest()


class TeacherFeatureCache:
    """Read one frame's detached targets without loading a whole shard into RAM."""

    def __init__(self, dataset_root: Path, *, chunk_size: int, feature_dim: int, tokens: int) -> None:
        """Validate provenance and target geometry before reading cache tensors.

        Raises:
            ValueError: If metadata, provenance, or geometry is invalid.
        """
        self.root = (dataset_root / CACHE_DIRECTORY).resolve()
        if not self.root.is_relative_to(dataset_root.resolve()):
            msg = "Teacher cache must remain inside its dataset"
            raise ValueError(msg)
        manifest_path = self.root / "manifest.json"
        if manifest_path.stat().st_size > 1024 * 1024:
            msg = "Teacher cache manifest is too large"
            raise ValueError(msg)
        manifest = json.loads(manifest_path.read_text())
        if (
            not isinstance(manifest, dict)
            or manifest.get("version") != 1
            or manifest.get("targets") != list(TARGET_NAMES)
        ):
            msg = "Invalid teacher cache manifest"
            raise ValueError(msg)
        for key in ("frames", "shard_size", "chunk_size", "feature_dim", "tokens"):
            if type(manifest.get(key)) is not int or manifest[key] < 1:
                msg = "Invalid teacher cache dimensions"
                raise ValueError(msg)
        if (manifest["chunk_size"], manifest["feature_dim"], manifest["tokens"]) != (chunk_size, feature_dim, tokens):
            msg = "Teacher cache does not match the training horizon or head geometry"
            raise ValueError(msg)
        if manifest.get("source_sha256") != dataset_fingerprint(dataset_root):
            msg = "Teacher cache is stale; recompute targets for this dataset"
            raise ValueError(msg)
        self.rgb_key, self.depth_key = manifest.get("rgb_key"), manifest.get("depth_key")
        if not isinstance(self.rgb_key, str) or (self.depth_key is not None and not isinstance(self.depth_key, str)):
            msg = "Invalid teacher cache camera mapping"
            raise ValueError(msg)
        self.frames, self.shard_size = manifest["frames"], manifest["shard_size"]
        self.tokens, self.feature_dim = tokens, feature_dim
        self.references = manifest.get("teachers", {})
        if not isinstance(self.references, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in self.references.items()
        ):
            msg = "Invalid cached teacher references"
            raise ValueError(msg)

    def __call__(self, frame_index: int) -> dict[str, Tensor]:
        """Return only the requested frame from its memory-mapped safetensors shard.

        Raises:
            ValueError: If the frame or cached tensor is invalid.
        """
        if not 0 <= frame_index < self.frames:
            msg = "Frame index lies outside the teacher cache"
            raise ValueError(msg)
        path = (self.root / f"{frame_index // self.shard_size:06d}.safetensors").resolve()
        if not path.is_relative_to(self.root):
            msg = "Teacher cache shard escapes its directory"
            raise ValueError(msg)
        output = {}
        with safe_open(path, framework="pt", device="cpu") as source:
            for name in TARGET_NAMES:
                values = source.get_slice(name)
                expected_rows = min(self.shard_size, self.frames - (frame_index // self.shard_size) * self.shard_size)
                if values.get_shape() != [expected_rows, self.tokens, self.feature_dim] or values.get_dtype() != "BF16":
                    msg = "Invalid cached teacher tensor shape"
                    raise ValueError(msg)
                target = values[frame_index % self.shard_size]
                if not target.isfinite().all():
                    msg = "Cached teacher targets must be finite"
                    raise ValueError(msg)
                output[f"teacher_features.{name}"] = target
        return output

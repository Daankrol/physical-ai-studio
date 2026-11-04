# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Strict, shard-by-shard loading of safetensors model weights."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from safetensors.torch import load_file

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import nn


_SAFE_WEIGHTS_NAME = "model.safetensors"
_SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"


def load_safetensors_weights(
    model: nn.Module,
    weights_path: str | Path,
    *,
    key_map: Callable[[str], str | None] | None = None,
) -> None:
    """Load one file or sharded weights, validating all keys before loading any shard.

    Args:
        model: Module to populate.
        weights_path: Safetensors file, index, or containing directory.
        key_map: Translate upstream names, or return None for deliberately skipped keys.

    Raises:
        FileNotFoundError: If no supported checkpoint is found.
        TypeError: If the index weight map is malformed.
        RuntimeError: If mapped keys disagree with the model or a shard is incomplete.
        ValueError: If a shard path escapes the checkpoint directory.
    """

    def translate(key: str) -> str | None:
        return key_map(key) if key_map is not None else key

    path = Path(weights_path)
    if path.is_dir():
        path /= _SAFE_WEIGHTS_INDEX_NAME if (path / _SAFE_WEIGHTS_INDEX_NAME).is_file() else _SAFE_WEIGHTS_NAME
    if not path.is_file():
        msg = f"No safetensors checkpoint found at {weights_path!s}."
        raise FileNotFoundError(msg)
    if path.name != _SAFE_WEIGHTS_INDEX_NAME:
        state = load_file(path, device="cpu")
        mapped = {name: value for key, value in state.items() if (name := translate(key)) is not None}
        if len(mapped) != sum(translate(key) is not None for key in state):
            msg = "Checkpoint key_map mapped distinct keys to one model parameter."
            raise RuntimeError(msg)
        model.load_state_dict(mapped, strict=True)
        return

    with path.open(encoding="utf-8") as stream:
        index = json.load(stream)
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not all(
        isinstance(key, str) and isinstance(shard, str) for key, shard in weight_map.items()
    ):
        msg = f"Invalid safetensors weight map in {path}."
        raise TypeError(msg)
    # Hub snapshots symlink shards into content-addressed blobs in the Hub cache.
    snapshot = next(
        (parent for parent in path.parents if parent.name == "snapshots" and parent.parent.name.startswith("models--")),
        None,
    )
    allowed_roots = [path.parent.resolve()]
    if snapshot is not None:
        allowed_roots.append((snapshot.parent.parent / "blobs").resolve())
    if any(
        not shard
        or "/" in shard
        or "\\" in shard
        or ".." in shard
        or Path(shard).suffix != ".safetensors"
        or not any((path.parent / shard).resolve().is_relative_to(root) for root in allowed_roots)
        for shard in weight_map.values()
    ):
        msg = f"Invalid shard path in {path}."
        raise ValueError(msg)
    mapped_keys = {name for key in weight_map if (name := translate(key)) is not None}
    expected = set(model.state_dict())
    missing, unexpected = sorted(expected - mapped_keys), sorted(mapped_keys - expected)
    if missing or unexpected or len(mapped_keys) != sum(translate(key) is not None for key in weight_map):
        msg = f"Checkpoint keys mismatch. Missing: {missing[:6]} Unexpected: {unexpected[:6]}"
        raise RuntimeError(msg)

    loaded: set[str] = set()
    for shard in sorted(set(weight_map.values())):
        state = load_file(path.parent / shard, device="cpu")
        shard_keys = {key for key, value in weight_map.items() if value == shard}
        if set(state) != shard_keys:
            msg = f"Safetensors shard keys mismatch for {shard!r}."
            raise RuntimeError(msg)
        mapped = {name: value for key, value in state.items() if (name := translate(key)) is not None}
        model.load_state_dict(mapped, strict=False)
        loaded.update(mapped)
    if loaded != expected:
        msg = "Sharded checkpoint did not load every model parameter."
        raise RuntimeError(msg)

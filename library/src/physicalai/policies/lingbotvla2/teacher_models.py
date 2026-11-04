# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pinned, lazy teacher loading without upstream's Transformers patches."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

import torch
from huggingface_hub import hf_hub_download
from torch import nn

LINGBOT_REVISION = "11c703bf6a5c1f45b3b69168482da11fdbba53d7"
MOGE_REVISION = "ca5f0e07ff01d3e5a364c1d954ed12ee1814b368"

if TYPE_CHECKING:
    from ._vendor.moge.model.v2 import MoGeModel
    from .teachers import TeacherTargets


def teacher_asset(name: str, root: str | Path | None = None) -> Path:
    """Resolve one known asset locally or from its pinned model revision.

    Returns:
        The local checkpoint or config path.

    Raises:
        ValueError: If the asset name is unknown.
        FileNotFoundError: If a supplied asset directory is incomplete or unsafe.
    """
    assets = {
        "moge": ("Ruicheng/moge-2-vitb-normal", MOGE_REVISION, "model.pt", "moge/model.pt"),
        "depth": ("robbyant/lingbot-vla-v2-6b", LINGBOT_REVISION, "depth/model.pt", "depth/model.pt"),
        "video": (
            "robbyant/lingbot-vla-v2-6b",
            LINGBOT_REVISION,
            "dino_video/teacher_step_10000.pth",
            "dino_video/teacher_step_10000.pth",
        ),
        "video_config": (
            "robbyant/lingbot-vla-v2-6b",
            LINGBOT_REVISION,
            "dino_video/config.yaml",
            "dino_video/config.yaml",
        ),
    }
    if name not in assets:
        msg = "Unknown teacher asset"
        raise ValueError(msg)
    repo, revision, filename, local_name = assets[name]
    if root is not None:
        base = Path(root).resolve()
        path = (base / local_name).resolve()
        if not path.is_relative_to(base) or not path.is_file():
            msg = f"Missing or invalid {name} teacher asset"
            raise FileNotFoundError(msg)
        return path
    return Path(hf_hub_download(repo_id=repo, revision=revision, filename=filename))  # nosec B615 -- fixed asset names and pinned revisions


ModelT = TypeVar("ModelT", bound=nn.Module)


def _geometry_model(model_type: type[ModelT], path: Path) -> ModelT:  # noqa: UP047 -- CI Bandit parses with Python 3.10
    # reason: weights_only=True restricts unpickling to tensors + safe primitives; no arbitrary code runs.
    # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(checkpoint, Mapping)
        or not isinstance(checkpoint.get("model_config"), Mapping)
        or not isinstance(checkpoint.get("model"), Mapping)
    ):
        msg = "Invalid geometry teacher checkpoint"
        raise TypeError(msg)
    model = model_type(**dict(checkpoint["model_config"]))
    weights = dict(checkpoint["model"])
    if model_type.__name__ == "MDMModel":
        # Released MoRGBD includes a normal decoder omitted by its feature-extraction class.
        weights = {name: value for name, value in weights.items() if not name.startswith("normal_head.")}
    model.load_state_dict(weights, strict=True)
    return model.eval().requires_grad_(False)  # noqa: FBT003


def load_moge(root: str | Path | None = None) -> MoGeModel:
    """Load the frozen RGB-to-metric-depth teacher.

    Returns:
        Frozen RGB-to-depth model.
    """
    from ._vendor.moge.model.v2 import MoGeModel  # noqa: PLC0415

    model = _geometry_model(MoGeModel, teacher_asset("moge", root))
    model.enable_pytorch_native_sdpa()
    return model


def load_teachers(
    root: str | Path | None = None,
    *,
    device: str | torch.device = "cuda",
) -> TeacherTargets:
    """Load the frozen RGB-to-depth, RGB-D, and video teachers.

    Returns:
        Training-only target extractor with pinned model references.
    """
    from ._vendor.dino_video.teacher import DinoVideoTeacher  # noqa: PLC0415
    from ._vendor.mdm.model.v2 import MDMModel  # noqa: PLC0415
    from .teachers import TeacherTargets  # noqa: PLC0415

    depth = _geometry_model(MDMModel, teacher_asset("depth", root))
    depth.enable_pytorch_native_sdpa()
    video = DinoVideoTeacher(
        str(teacher_asset("video", root)),
        str(teacher_asset("video_config", root)),
        attention_mode="flex_block_causal",
        input_size=256,
        n_blocks=1,
        device=device,
        cls_pool="last",
    ).build()
    return TeacherTargets(
        load_moge(root),
        depth,
        video,
        {
            "lingbot_revision": LINGBOT_REVISION,
            "moge_revision": MOGE_REVISION,
            "depth_source": "moge",
        },
    )

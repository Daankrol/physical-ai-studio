# Copyright (C) 2026 Robbyant Inc.
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""RGB target extraction for frozen upstream MoGe, MoRGBD, and DINO-Video models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.nn import functional


@dataclass
class TeacherTargets:
    """Keep frozen teachers outside the student's state dict and FSDP parameter tree.

    Models must expose upstream ``infer``, ``infer_feat``, and ``get_future_feature`` APIs.
    Construct them in the training process; Studio does not import upstream's patched package.
    """

    moge: nn.Module
    depth: nn.Module
    video: nn.Module
    references: dict[str, str] = field(default_factory=dict)
    _device: torch.device | None = field(default=None, init=False)

    @torch.no_grad()
    def __call__(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Generate current/future depth and video targets from RGB.

        Returns:
            Detached BF16 alignment targets.
        """
        current, future = batch["teacher_current_rgb"][:, 0] / 255, batch["teacher_future_rgb"][:, 0] / 255
        if self._device != current.device:
            for model in (self.moge, self.depth, self.video):
                model.requires_grad_(False).to(current.device)  # noqa: FBT003
            self._device = current.device
        for model in (self.moge, self.depth, self.video):
            model.eval()
        output = {}
        with torch.autocast(current.device.type, dtype=torch.bfloat16, enabled=current.is_cuda):
            for name, image in (("depth", current), ("future_depth", future)):
                geometry: Any = cast("Any", self.moge).infer(
                    image,
                    resolution_level=3,
                    num_tokens=256,
                    apply_mask=False,
                )
                depth = geometry["depth"]
                depth = torch.nan_to_num(depth.squeeze().detach().clone(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0)
                features, _ = cast("Any", self.depth).infer_feat(
                    image,
                    depth,
                    depth_down_scale=1,
                    resolution_level=3,
                    num_tokens=256,
                    enable_depth_mask=False,
                )
                output[f"{name}_target"] = features.permute(0, 2, 3, 1).flatten(1, 2).detach().to(torch.bfloat16)
            current = functional.interpolate(current, size=(256, 256), mode="bilinear", align_corners=False)
            future = functional.interpolate(future, size=(256, 256), mode="bilinear", align_corners=False)
            mean = current.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
            std = current.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
            current, future = (current - mean) / std, (future - mean) / std
            video = torch.stack((current.clone(), current, future), dim=2).contiguous()
            future_features, current_features = cast("Any", self.video).get_future_feature(
                video,
                return_cls=False,
                return_current=True,
                current_index=1,
                fps=batch["teacher_fps"],
            )
            output["future_video_target"] = future_features.detach().to(torch.bfloat16)
            output["current_video_target"] = current_features.detach().to(torch.bfloat16)
        return output

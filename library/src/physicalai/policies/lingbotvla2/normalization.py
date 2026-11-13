# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Upstream normalization without changing other policies' epsilon semantics."""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import nn

from physicalai.data import Feature, FeatureType
from physicalai.policies.utils.normalization import FeatureNormalizeTransform, NormalizationType


class UpstreamNormalizeTransform(FeatureNormalizeTransform):
    """Use LingBot's additive epsilon in both directions."""

    @staticmethod
    def _create_stats_buffers(
        features: dict[str, Feature],
        norm_map: dict[FeatureType, NormalizationType],
    ) -> dict[str, dict[str, nn.ParameterDict]]:
        adjusted = dict(features)
        for name, feature in features.items():
            if feature.ftype != FeatureType.ACTION or feature.normalization_data is None:
                continue
            data = feature.normalization_data
            values = data.mean if norm_map.get(FeatureType.ACTION) == NormalizationType.MEAN_STD else data.q01
            if values is not None and torch.as_tensor(values).ndim == 2:  # noqa: PLR2004 -- time, action dimensions
                shape = tuple(torch.as_tensor(values).shape)
                if feature.shape is None or shape[-1] != feature.shape[-1]:
                    msg = "Per-horizon action statistics must match the action width"
                    raise ValueError(msg)
                if data.mask is not None and torch.as_tensor(data.mask).ndim == 1:
                    data = replace(data, mask=torch.as_tensor(data.mask).expand(shape).tolist())
                adjusted[name] = replace(feature, shape=shape, normalization_data=data)
        return FeatureNormalizeTransform._create_stats_buffers(adjusted, norm_map)  # noqa: SLF001

    @staticmethod
    def _apply_normalization(
        batch: dict,
        key: str,
        norm_mode: NormalizationType,
        buffer: nn.ParameterDict,
        *,
        inverse: bool,
    ) -> None:
        value = batch[key]
        if value is None:
            return None
        if norm_mode == NormalizationType.MEAN_STD:
            center, width = buffer["mean"], buffer["std"] + 1e-6
            transformed = value * width + center if inverse else (value - center) / width
        elif norm_mode == NormalizationType.QUANTILES:
            center, width = buffer["q01"], buffer["q99"] - buffer["q01"] + 1e-6
            transformed = (value + 1) * width / 2 + center if inverse else 2 * (value - center) / width - 1
        else:
            return FeatureNormalizeTransform._apply_normalization(  # noqa: SLF001 -- override delegates other modes
                batch,
                key,
                norm_mode,
                buffer,
                inverse=inverse,
            )
        if not torch.compiler.is_compiling() and (not torch.isfinite(center).all() or not torch.isfinite(width).all()):
            msg = "LingBot normalization requires finite dataset statistics"
            raise ValueError(msg)
        mask = buffer.get("mask")
        batch[key] = transformed if mask is None else torch.where(mask.bool(), transformed, value)
        return None

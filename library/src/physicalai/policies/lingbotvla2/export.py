# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Torch checkpoint export metadata for LingBot-VLA 2.0."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from physicalai.inference.data import InferenceFeature, InferenceFeatureDtype, InferenceFeatureType
from physicalai.inference.manifest import ComponentSpec
from transformers import AutoTokenizer

from physicalai.data import FeatureType
from physicalai.data.observation import ACTION, TASK
from physicalai.export import ExportablePolicyMixin, ExportBackend
from physicalai.export.backends import ExportParameters, TorchExportParameters

if TYPE_CHECKING:
    from os import PathLike

    from .config import LingBotVLA2Config


class LingBotVLA2ExportMixin(ExportablePolicyMixin):
    """Export Torch checkpoints with the raw observation and trimmed action schema."""

    config: LingBotVLA2Config | None

    @override
    def to_torch(self, checkpoint_path: str | PathLike[str]) -> None:
        """Ship the pinned tokenizer alongside the checkpoint for offline Runtime loading.

        Raises:
            RuntimeError: If the policy is not initialized.
            ValueError: If the pinned tokenizer cannot be loaded.
        """
        if self.config is None:
            msg = "Initialize LingBot before Torch export"
            raise RuntimeError(msg)
        model_path = self._prepare_export_path(checkpoint_path, ".pt")
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer_name_or_path,
            revision=self.config.tokenizer_revision,
            local_files_only=True,
        )
        if tokenizer is None:
            msg = "The LingBot tokenizer was not found in the local cache"
            raise ValueError(msg)
        tokenizer.save_pretrained(model_path.parent / "tokenizer")
        super().to_torch(checkpoint_path)

    @property
    @override
    def inputs_schema(self) -> list[InferenceFeature] | None:
        """Describe the selected camera, state, and task inputs.

        Returns:
            Raw feature schema, or None before model initialization.

        Raises:
            ValueError: If a selected input feature has no shape.
        """
        if self.config is None:
            return None
        schema = []
        for feature in self.config.input_features:
            if feature.ftype not in {FeatureType.STATE, FeatureType.VISUAL}:
                continue
            if feature.shape is None:
                msg = f"Export requires the shape of {feature.name}"
                raise ValueError(msg)
            schema.append(
                InferenceFeature(
                    name=str(feature.name),
                    shape=feature.shape,
                    ftype=InferenceFeatureType.STATE
                    if feature.ftype == FeatureType.STATE
                    else InferenceFeatureType.VISUAL,
                    dtype=InferenceFeatureDtype.FLOAT32,
                ),
            )
        schema.append(
            InferenceFeature(
                name=TASK,
                shape=(),
                ftype=InferenceFeatureType.LANGUAGE,
                dtype=InferenceFeatureDtype.STRING,
            ),
        )
        return schema

    @property
    @override
    def outputs_schema(self) -> list[InferenceFeature] | None:
        """Describe actions as emitted by the policy's postprocessor.

        Returns:
            Dataset action schema, or None before initialization.

        Raises:
            ValueError: If the action feature is absent or lacks a shape.
        """
        if self.config is None:
            return None
        actions = [feature for feature in self.config.output_features if feature.ftype == FeatureType.ACTION]
        if len(actions) != 1 or actions[0].shape is None:
            msg = "Export requires one action feature with a concrete shape"
            raise ValueError(msg)
        return [
            InferenceFeature(
                name=ACTION,
                shape=(self.config.n_action_steps, *actions[0].shape),
                ftype=InferenceFeatureType.ACTION,
                dtype=InferenceFeatureDtype.FLOAT32,
            ),
        ]

    @property
    @override
    def extra_export_args(self) -> dict[str, ExportParameters]:
        """Keep Runtime's input tensor conversion; the policy handles all other processing.

        Returns:
            Torch-only export parameters.
        """
        return {ExportBackend.TORCH: TorchExportParameters(preprocessors_specs=[ComponentSpec(type="to_float_tensor")])}

    @staticmethod
    @override
    def get_supported_export_backends() -> list[str | ExportBackend]:
        """Advertise only the checkpoint backend.

        Returns:
            The Torch backend.
        """
        return [ExportBackend.TORCH]

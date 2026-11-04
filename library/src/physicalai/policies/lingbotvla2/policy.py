# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Lightning policy for LingBot-VLA 2.0."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, override

import torch
from torch import Tensor

from physicalai.data import Feature, FeatureType, Observation
from physicalai.data.dataset import Dataset
from physicalai.policies.base import Policy
from physicalai.policies.mixins.peft import PeftPolicyMixin

from .config import LingBotVLA2Config
from .export import LingBotVLA2ExportMixin
from .model import LingBotVLA2Model
from .postprocessor import LingBotVLA2Postprocessor, make_lingbotvla2_processors
from .pretrained import resolve_pretrained

if TYPE_CHECKING:
    from collections.abc import Callable

    from lightning.pytorch.utilities.types import OptimizerLRScheduler

    from .preprocessor import LingBotVLA2Preprocessor
    from .teachers import TeacherTargets


class LingBotVLA2(PeftPolicyMixin, LingBotVLA2ExportMixin, Policy):
    """Fine-tune or sample from the released LingBot checkpoint; joint groups come from recorded joint names."""

    def __init__(  # noqa: PLR0913
        self,
        input_features: list[Feature] | None = None,
        output_features: list[Feature] | None = None,
        pretrained_name_or_path: str | Path | None = "robbyant/lingbot-vla-v2-6b",
        *,
        state_map: dict[str, list[tuple[int, int]]] | None = None,
        action_map: dict[str, list[tuple[int, int]]] | None = None,
        image_key_reorder_map: dict[str, int] | None = None,
        num_cameras: int = 3,
        chunk_size: int = 50,
        n_action_steps: int = 50,
        num_denoise_steps: int = 10,
        normalization_mode: str = "QUANTILES",
        training_recipe: str = "action_only",
        moe_backend: str = "dense",
        attention_backend: str = "sdpa",
        relative_action_groups: tuple[str, ...] = (),
        distillation_heads: bool = False,
        teacher_weights_dir: str | Path | None = None,
        loss_type: str | None = None,
        gradient_checkpointing: bool = False,
        compile_model: bool = False,
        lora_enabled: bool = False,
        lora_rank: int = 32,
        lora_alpha: int | None = None,
        lora_dropout: float = 0.05,
        lora_use_dora: bool = False,
        lora_target_modules: str | None = None,
        train_expert_only: bool = False,
        freeze_vision_encoder: bool = False,
        optimizer_lr: float = 1e-4,
        optimizer_vit_lr: float = 1e-6,
        optimizer_betas: tuple[float, float] = (0.9, 0.95),
        optimizer_weight_decay: float = 0.0,
        scheduler_warmup_steps: int = 0,
        scheduler_decay_lr: float = 5e-5,
        model_config: dict[str, Any] | None = None,
        tokenizer_dir_override: str | Path | None = None,
        teachers: TeacherTargets | None = None,
    ) -> None:
        """Defer hub resolution until the dataset features and layout are known."""
        super().__init__(n_action_steps=n_action_steps)
        self.input_features = input_features
        self.output_features = output_features
        self.pretrained_name_or_path = pretrained_name_or_path
        self.state_map = state_map
        self.action_map = action_map
        self.image_key_reorder_map = image_key_reorder_map
        self.num_cameras = num_cameras
        self.chunk_size = chunk_size
        self.n_action_steps = n_action_steps
        self.num_denoise_steps = num_denoise_steps
        self.normalization_mode = normalization_mode
        self.training_recipe = training_recipe
        self.moe_backend = moe_backend
        self.attention_backend = attention_backend
        self.relative_action_groups = relative_action_groups
        self.distillation_heads = distillation_heads
        self.teacher_weights_dir = teacher_weights_dir
        self.loss_type = loss_type
        self.gradient_checkpointing = gradient_checkpointing
        self.compile_model = compile_model
        self.lora_enabled = lora_enabled
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_use_dora = lora_use_dora
        self.lora_target_modules = lora_target_modules
        self.train_expert_only = train_expert_only
        self.freeze_vision_encoder = freeze_vision_encoder
        self.optimizer_lr = optimizer_lr
        self.optimizer_vit_lr = optimizer_vit_lr
        self.optimizer_betas = optimizer_betas
        self.optimizer_weight_decay = optimizer_weight_decay
        self.scheduler_warmup_steps = scheduler_warmup_steps
        self.scheduler_decay_lr = scheduler_decay_lr
        self.model: LingBotVLA2Model | None = None  # pyrefly: ignore[bad-override-mutable-attribute]
        self.teachers = teachers
        self.teacher_references: dict[str, str] = {}
        self.tokenizer_dir_override = tokenizer_dir_override
        self.config: LingBotVLA2Config | None = None
        self._config: LingBotVLA2Config | None = LingBotVLA2Config.from_dict(model_config) if model_config else None
        self._preprocessor: LingBotVLA2Preprocessor | None = None  # pyrefly: ignore[bad-override-mutable-attribute]
        self._postprocessor: LingBotVLA2Postprocessor | None = None
        self.save_hyperparameters(
            ignore=["input_features", "output_features", "model_config", "tokenizer_dir_override", "teachers"],
        )
        if input_features is not None and output_features is not None:
            self.configure_model()

    @staticmethod
    def _camera_map(features: list[Feature], reorder: dict[str, int] | None) -> dict[str, str]:
        slots = ("camera_top", "camera_wrist_left", "camera_wrist_right")
        result = {}
        for feature in features:
            if feature.ftype != FeatureType.VISUAL or feature.name is None:
                continue
            name = feature.name
            short = name.rsplit(".", 1)[-1]
            position = (
                reorder.get(name, reorder.get(short, reorder.get(f"images.{short}")))
                if reorder is not None
                else slots.index(short)
                if short in slots
                else None
            )
            if position is None or position not in range(len(slots)):
                msg = f"No LingBot camera slot selected for {name!r}; set image_key_reorder_map"
                raise ValueError(msg)
            result[name] = slots[position]
        return result

    @classmethod
    def from_config(cls, config: LingBotVLA2Config, **options: Any) -> LingBotVLA2:  # noqa: ANN401
        """Build a model from a resolved config without contacting the pretrained hub.

        Returns:
            Policy with an initialized model and processors.
        """
        policy = cls(pretrained_name_or_path=None, n_action_steps=config.n_action_steps, **options)
        policy._config = config
        policy.configure_model()
        return policy

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str | Path | IO[bytes],
        map_location: torch.device | str | int | Callable | dict | None = None,
        hparams_file: str | Path | None = None,
        strict: bool | None = None,  # noqa: FBT001
        weights_only: bool | None = None,  # noqa: FBT001
        **kwargs: Any,  # noqa: ANN401
    ) -> LingBotVLA2:
        """Restore from the saved model config rather than downloading pretrained shards.

        Returns:
            Restored policy.
        """
        kwargs["pretrained_name_or_path"] = None
        if isinstance(checkpoint_path, (str, Path)) and (Path(checkpoint_path).parent / "tokenizer").is_dir():
            kwargs["tokenizer_dir_override"] = Path(checkpoint_path).parent / "tokenizer"
        return super().load_from_checkpoint(
            checkpoint_path,
            map_location=map_location,
            hparams_file=hparams_file,
            strict=strict,
            weights_only=weights_only,
            **kwargs,
        )

    def configure_model(self) -> None:  # noqa: PLR0912
        """Build once from a saved config or selected features, then load optional pretrained weights.

        Raises:
            ValueError: If input/output features or canonical joint mappings are absent.
        """
        if self.model is not None:
            mesh = getattr(self, "_device_mesh", None)
            if mesh is not None and not getattr(self, "_fsdp_configured", False):
                from .distributed import shard_policy  # noqa: PLC0415

                shard_policy(self, mesh["data_parallel"])
                self._fsdp_configured = True
            return
        weights = None
        if self._config is not None:
            config = self._config
        else:
            if self.input_features is None or self.output_features is None:
                msg = "Input and output features are required to build LingBot-VLA 2.0"
                raise ValueError(msg)
            pretrained_fields, weights = (
                resolve_pretrained(self.pretrained_name_or_path) if self.pretrained_name_or_path else ({}, None)
            )
            config_fields = dict(pretrained_fields)
            config_fields.update(
                input_features=self.input_features,
                output_features=self.output_features,
                state_map=self.state_map or {},
                action_map=self.action_map or {},
                camera_map=self._camera_map(self.input_features, self.image_key_reorder_map),
                chunk_size=self.chunk_size,
                n_action_steps=self.n_action_steps,
                num_denoise_steps=self.num_denoise_steps,
                normalization_mode=self.normalization_mode,
                training_recipe=self.training_recipe,
                moe_backend=self.moe_backend,
                attention_backend=self.attention_backend,
                relative_action_groups=self.relative_action_groups,
                distillation_heads=self.distillation_heads,
                loss_type=self.loss_type
                or ("fm" if self.training_recipe == "upstream" else pretrained_fields.get("loss_type", "L1_fm")),
                action_fp32=self.training_recipe == "upstream",
                lora_enabled=self.lora_enabled,
                lora_rank=self.lora_rank,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                lora_use_dora=self.lora_use_dora,
                lora_target_modules=self.lora_target_modules,
                freeze_vision_encoder=self.freeze_vision_encoder,
                train_expert_only=self.train_expert_only,
            )
            config = LingBotVLA2Config(**config_fields)
        self.config = config
        self.input_features, self.output_features = config.input_features, config.output_features
        model = LingBotVLA2Model.from_config(config)
        self.model = model
        self._preprocessor, self._postprocessor = make_lingbotvla2_processors(config)
        if self.tokenizer_dir_override is not None:
            self._preprocessor.tokenizer_name_or_path = str(self.tokenizer_dir_override)
            self._preprocessor.tokenizer_revision = None
        if weights is not None:
            model.load_weights(weights)
        if config.train_expert_only or config.freeze_vision_encoder:
            tower = model.qwenvl_with_expert.qwenvl
            if config.train_expert_only:
                tower.requires_grad_(False)  # noqa: FBT003
            else:
                tower.model.visual.requires_grad_(False)  # noqa: FBT003
        if config.use_lora:
            self._inject_lora()
            for parameter in model.qwenvl_with_expert.qwen_expert.parameters():
                parameter.requires_grad_(True)  # noqa: FBT003
            for name, parameter in model.named_parameters():
                if (
                    name.startswith(("state_proj.", "action_in_proj.", "action_out_proj.", "action_time_mlp_"))
                    or "align_embs" in name
                    or "shared_task_proj" in name
                    or "_align_head." in name
                ):
                    parameter.requires_grad_(True)  # noqa: FBT003
        if self.gradient_checkpointing:
            model.enable_gradient_checkpointing()
        if self.compile_model:
            model.predict_action_chunk = torch.compile(model.predict_action_chunk)  # type: ignore[method-assign]

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Persist resolved architecture and dataset maps for offline restoration."""
        super().on_save_checkpoint(checkpoint)
        references = self.teachers.references if self.teachers is not None else self.teacher_references
        if references:
            checkpoint["teacher_references"] = dict(references)
        if self.config is not None:
            checkpoint["model_config"] = self.config.to_dict()
            checkpoint.setdefault("hyper_parameters", {})["model_config"] = self.config.to_dict()

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Construct the exact saved model before Lightning loads its parameters.

        Raises:
            TypeError: If the checkpoint has no serialized LingBot config.
            ValueError: If a preconstructed model disagrees with the checkpoint.
        """
        super().on_load_checkpoint(checkpoint)
        references = checkpoint.get("teacher_references", {})
        if not isinstance(references, Mapping) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in references.items()
        ):
            msg = "Invalid checkpoint teacher references"
            raise TypeError(msg)
        if not self.teacher_references:
            self.teacher_references = dict(references)
        data = checkpoint.get("model_config")
        if not isinstance(data, Mapping):
            msg = "LingBot checkpoint is missing model_config"
            raise TypeError(msg)
        config = LingBotVLA2Config.from_dict(data)
        if self.model is not None:
            current = self.config.to_dict() if self.config is not None else {}
            if current != config.to_dict():
                msg = "Initialized LingBot config does not match the checkpoint"
                raise ValueError(msg)
            return
        self._config = config
        self.configure_model()

    def set_features(self, input_features: list[Feature], output_features: list[Feature]) -> None:
        """Update names and statistics without rebuilding the VLM or reloading weights.

        Raises:
            RuntimeError: If the policy has no model yet.
        """
        if self.config is None or self.model is None:
            msg = "Initialize the policy before replacing features"
            raise RuntimeError(msg)
        config = replace(self.config, input_features=input_features, output_features=output_features)
        self._set_config(config)

    def rename_features(self, mapping: Mapping[str, str]) -> None:
        """Rename selected input features and their camera-slot mappings.

        Raises:
            ValueError: If a source name is unknown or the new names collide.
            RuntimeError: If the policy has not been initialized.
        """
        if self.config is None or self.model is None:
            msg = "Initialize the policy before renaming features"
            raise RuntimeError(msg)
        known = {feature.name for feature in self.config.input_features}
        if set(mapping) - known or any(not isinstance(name, str) or not name for name in mapping.values()):
            msg = "Feature renames must use known sources and non-empty names"
            raise ValueError(msg)
        renamed = [
            replace(feature, name=mapping.get(feature.name, feature.name)) if feature.name is not None else feature
            for feature in self.config.input_features
        ]
        if len({feature.name for feature in renamed}) != len(renamed):
            msg = "Feature renames create duplicate names"
            raise ValueError(msg)
        cameras = {mapping.get(name, name): slot for name, slot in self.config.camera_map.items()}
        self._set_config(replace(self.config, input_features=renamed, camera_map=cameras))

    def _set_config(self, config: LingBotVLA2Config) -> None:
        if self.model is None:
            msg = "Initialize the policy before updating features"
            raise RuntimeError(msg)
        pre, post = make_lingbotvla2_processors(config)
        if self.tokenizer_dir_override is not None:
            pre.tokenizer_name_or_path = str(self.tokenizer_dir_override)
            pre.tokenizer_revision = None
        parameter = next(self.model.parameters())
        self._preprocessor = pre.to(parameter.device)
        self._postprocessor = post.to(parameter.device)
        self.config = config
        self.input_features, self.output_features = config.input_features, config.output_features
        self.reset()

    def setup(self, stage: str) -> None:
        """Adopt dataset features once, or validate and replace only processors.

        Raises:
            TypeError: If the datamodule supplies an unsupported dataset type.
            ValueError: If state or action features do not match the policy.
        """
        if stage != "fit":
            return
        dataset = self.trainer.datamodule.train_dataset  # type: ignore[attr-defined]  # pyrefly: ignore[missing-attribute]
        if not isinstance(dataset, Dataset):
            msg = "Expected a PhysicalAI training Dataset"
            raise TypeError(msg)
        inputs, outputs = list(dataset.observation_features.values()), list(dataset.action_features.values())
        if sum(feature.ftype == FeatureType.STATE for feature in inputs) != 1 or len(outputs) != 1:
            msg = "LingBot training requires one state vector and one action vector"
            raise ValueError(msg)
        if self.model is None:
            self.input_features, self.output_features = inputs, outputs
            if not self.state_map or not self.action_map:
                from .data import recorded_joint_maps  # noqa: PLC0415

                state_map, action_map = recorded_joint_maps(dataset)
                self.state_map, self.action_map = self.state_map or state_map, self.action_map or action_map
            self.configure_model()
        elif self.config is not None and (
            inputs != self.config.input_features or outputs != self.config.output_features
        ):
            self.set_features(inputs, outputs)
        if self.config is not None and self.config.training_recipe == "upstream":
            from physicalai.data.lerobot.dataset import _LeRobotDatasetAdapter  # noqa: PLC0415, PLC2701

            from .data import adopt_training_statistics  # noqa: PLC0415

            if not isinstance(dataset, _LeRobotDatasetAdapter):
                msg_0 = "The upstream recipe requires a LeRobot training dataset for train-only statistics"
                raise TypeError(msg_0)
            self._set_config(adopt_training_statistics(dataset, self.config))

    @override
    def forward(self, batch: Observation) -> Tensor | tuple[Tensor, dict[str, Tensor | float]]:
        """Compute training loss or generate a denormalized action chunk.

        Returns:
            Loss and metrics during training; actions otherwise.

        Raises:
            RuntimeError: If the policy is not yet configured.
        """
        if not self.training:
            return self.predict_action_chunk(batch)
        if self.model is None or self._preprocessor is None:
            msg = "LingBot policy is not configured"
            raise RuntimeError(msg)
        return self.model(self._training_batch(batch))

    @torch.no_grad()
    @override
    def predict_action_chunk(self, batch: Observation) -> Tensor:
        """Predict the next dataset action chunk.

        Returns:
            Actions in dataset units and order, trimmed to n_action_steps.

        Raises:
            RuntimeError: If the policy is not yet configured.
        """
        if self.model is None or self._preprocessor is None or self._postprocessor is None:
            msg = "LingBot policy is not configured"
            raise RuntimeError(msg)
        batch = batch.to(self.device)
        processed = self._preprocessor({**batch.to_dict(), "action": None})
        return self._postprocessor(self.model.predict_action_chunk(processed), batch.state)

    def training_step(self, batch: Observation, batch_idx: int) -> Tensor:
        """Log the flow and MoE losses for Lightning.

        Returns:
            Differentiable training loss.
        """
        del batch_idx
        loss, metrics = self(batch)
        for name, value in metrics.items():
            self.log(f"train/{name}", value, prog_bar=name == "loss")
        return loss

    @override
    def compute_val_loss(self, batch: Observation) -> tuple[Tensor, dict[str, Tensor | float]]:
        """Compute denoised action MSE for validation.

        Returns:
            Validation loss and metrics.

        Raises:
            RuntimeError: If the policy is not yet configured.
        """
        if self.model is None or self._preprocessor is None:
            msg = "LingBot policy is not configured"
            raise RuntimeError(msg)
        return self.model.compute_val_loss(self._training_batch(batch))

    def _training_batch(self, batch: Observation) -> dict[str, Tensor]:
        if self._preprocessor is None or self.config is None:
            msg = "Configure LingBot before preparing a training batch"
            raise RuntimeError(msg)
        processed = self._preprocessor(batch.to_dict())
        if self.config.distillation_heads:
            if self.teachers is None:
                from .teacher_models import load_teachers  # noqa: PLC0415

                self.teachers = load_teachers(
                    self.teacher_weights_dir,
                    device=processed["state"].device,
                )
            processed.update(self.teachers(processed))
        return processed

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Use AdamW with a low-rate vision group and a warmup-cosine schedule.

        Returns:
            Lightning optimizer and step scheduler.

        Raises:
            RuntimeError: If the policy is not yet configured.
        """
        if self.model is None or self.config is None:
            msg = "LingBot policy is not configured"
            raise RuntimeError(msg)
        scale = self.config.lora_lr_scale if self.config.use_lora else 1.0
        vision: list[torch.nn.Parameter] = []
        other: list[torch.nn.Parameter] = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                (vision if ".qwenvl.model.visual." in name else other).append(param)
        if self.config.training_recipe == "upstream":
            from .optimizer import MuonAdamW  # noqa: PLC0415

            optimizer = MuonAdamW(
                self.model,
                lr=self.optimizer_lr * scale,
                weight_decay=self.optimizer_weight_decay,
                betas=self.optimizer_betas,
                expert_scale=math.sqrt(self.config.token_num_experts / self.config.token_top_k),
            )
        else:
            optimizer = torch.optim.AdamW(
                [
                    {"params": other, "lr": self.optimizer_lr * scale},
                    {"params": vision, "lr": self.optimizer_vit_lr * scale},
                ],
                betas=self.optimizer_betas,
                weight_decay=self.optimizer_weight_decay,
            )
        steps = max(1, int(self.trainer.estimated_stepping_batches))
        warmup = self.scheduler_warmup_steps
        minimum = self.scheduler_decay_lr / self.optimizer_lr

        def multiplier(step: int) -> float:
            if step < warmup:
                return (step + 1) / max(1, warmup)
            progress = min(1.0, (step - warmup) / max(1, steps - warmup))
            return minimum + (1 - minimum) * (1 + math.cos(math.pi * progress)) / 2

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

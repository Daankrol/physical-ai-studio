# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Lightning policy for LingBot-VLA 2.0."""

from __future__ import annotations

import math
from collections.abc import Mapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import replace
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, override

import torch
from torch import Tensor

from physicalai.data import Feature, FeatureType, Observation
from physicalai.data.dataset import Dataset
from physicalai.policies.base import Policy
from physicalai.policies.mixins.peft import PeftPolicyMixin
from physicalai.train.optimizers import build_adamw_auto_precision

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


def _without_normalization(features: list[Feature]) -> list[Feature]:
    return [replace(feature, normalization_data=None) for feature in features]


def _same_architecture(left: LingBotVLA2Config, right: LingBotVLA2Config) -> bool:
    """Compare configs while ignoring feature statistics, which only processors consume.

    Returns:
        Whether both configs describe the same model and ordered feature contract.
    """
    return replace(
        left,
        input_features=_without_normalization(left.input_features),
        output_features=_without_normalization(left.output_features),
    ) == replace(
        right,
        input_features=_without_normalization(right.input_features),
        output_features=_without_normalization(right.output_features),
    )


class LingBotVLA2(PeftPolicyMixin, LingBotVLA2ExportMixin, Policy):
    """Fine-tune or sample from the released LingBot checkpoint; joint groups come from recorded joint names."""

    def __init__(  # noqa: PLR0913, PLR0915
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
        dtype: str = "float32",
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
        tokenizer_dir_override: str | Path | None = None,
        teachers: TeacherTargets | None = None,
    ) -> None:
        """Build eagerly from features, or defer until the dataset supplies them."""
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
        self.model_dtype = dtype  # LightningModule owns `dtype`
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
        self._config: LingBotVLA2Config | None = None
        self._processor_config: LingBotVLA2Config | None = None
        self._pretrained_fields: dict[str, Any] = {}
        self._base_weights: Path | None = None
        self._preprocessor: LingBotVLA2Preprocessor | None = None  # pyrefly: ignore[bad-override-mutable-attribute]
        self._postprocessor: LingBotVLA2Postprocessor | None = None
        self.save_hyperparameters(
            ignore=["input_features", "output_features", "tokenizer_dir_override", "teachers"],
        )
        if input_features is not None and output_features is not None:
            self._resolve_pretrained()
            self.configure_model()

    @property
    def config(self) -> LingBotVLA2Config:
        """Resolved policy-owned config.

        Raises:
            RuntimeError: If no config has been resolved yet.
        """
        if self._config is None:
            msg = "LingBot policy config is not initialized"
            raise RuntimeError(msg)
        return self._config

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

    def _resolve_pretrained(self) -> None:
        """Fetch checkpoint metadata and weights before Lightning enters ``configure_model()``."""
        if self.pretrained_name_or_path is not None and self._base_weights is None and self.model is None:
            self._pretrained_fields, self._base_weights = resolve_pretrained(self.pretrained_name_or_path)

    def _new_config(self) -> LingBotVLA2Config | None:
        """Resolve a config from constructor values and features, or None while features are unknown.

        Returns:
            Resolved config, or None when features are not yet available.

        Raises:
            RuntimeError: If pretrained weights were requested but not resolved first.
        """
        if self.input_features is None or self.output_features is None:
            return None
        if self.pretrained_name_or_path is not None and self._base_weights is None:
            msg = "Resolve the pretrained checkpoint before configuring LingBot"
            raise RuntimeError(msg)
        fields: dict[str, Any] = {
            **self._pretrained_fields,
            "input_features": self.input_features,
            "output_features": self.output_features,
            "state_map": self.state_map or {},
            "action_map": self.action_map or {},
            "camera_map": self._camera_map(self.input_features, self.image_key_reorder_map),
            "chunk_size": self.chunk_size,
            "n_action_steps": self.n_action_steps,
            "num_denoise_steps": self.num_denoise_steps,
            "normalization_mode": self.normalization_mode,
            "dtype": self.model_dtype,
            "training_recipe": self.training_recipe,
            "moe_backend": self.moe_backend,
            "attention_backend": self.attention_backend,
            "relative_action_groups": self.relative_action_groups,
            "distillation_heads": self.distillation_heads,
            "loss_type": self.loss_type
            or ("fm" if self.training_recipe == "upstream" else self._pretrained_fields.get("loss_type", "L1_fm")),
            "action_fp32": self.training_recipe == "upstream",
            "lora_enabled": self.lora_enabled,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "lora_use_dora": self.lora_use_dora,
            "lora_target_modules": self.lora_target_modules,
            "freeze_vision_encoder": self.freeze_vision_encoder,
            "train_expert_only": self.train_expert_only,
        }
        return LingBotVLA2Config(**fields)

    def configure_model(self) -> None:
        """Create the model once, then keep processors and FSDP sharding in sync with the config."""
        config = self._config if self._config is not None else self._new_config()
        if config is None:
            return
        self._config = config
        if self.model is None:
            self.model = LingBotVLA2Model.from_config(config)
            if self._base_weights is not None:
                self.model.load_weights(self._base_weights)
            if config.dtype == "bfloat16":
                # Parameters only: buffers such as RoPE frequencies stay fp32.
                for parameter in self.model.parameters():
                    parameter.data = parameter.data.to(torch.bfloat16)
            self._apply_model_modifications(self.model, config)
        if config != self._processor_config or self._preprocessor is None or self._postprocessor is None:
            pre, post = make_lingbotvla2_processors(config)
            if self.tokenizer_dir_override is not None:
                pre.tokenizer_name_or_path = str(self.tokenizer_dir_override)
                pre.tokenizer_revision = None
            device = next(self.model.parameters()).device
            self._preprocessor, self._postprocessor = pre.to(device), post.to(device)
            self._processor_config = config
        mesh = getattr(self, "_device_mesh", None)
        if mesh is not None and not getattr(self, "_fsdp_configured", False):
            from .distributed import shard_policy  # noqa: PLC0415

            shard_policy(self, mesh["data_parallel"])
            self._fsdp_configured = True

    def _apply_model_modifications(self, model: LingBotVLA2Model, config: LingBotVLA2Config) -> None:
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
        """Persist resolved architecture, features and joint maps for offline restoration."""
        super().on_save_checkpoint(checkpoint)
        references = self.teachers.references if self.teachers is not None else self.teacher_references
        if references:
            checkpoint["teacher_references"] = dict(references)
        checkpoint["model_config"] = self.config.to_dict()

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
        if self.model is not None and not _same_architecture(self.config, config):
            msg = "Initialized LingBot config does not match the checkpoint"
            raise ValueError(msg)
        self._config = config
        self.configure_model()

    def set_features(self, input_features: list[Feature], output_features: list[Feature]) -> None:
        """Update names and statistics without rebuilding the VLM or reloading weights.

        Raises:
            RuntimeError: If the policy has no model yet.
        """
        if self.model is None:
            msg = "Initialize the policy before replacing features"
            raise RuntimeError(msg)
        self._adapt(replace(self.config, input_features=input_features, output_features=output_features))

    def rename_features(self, mapping: Mapping[str, str]) -> None:
        """Rename selected input features and their camera-slot mappings.

        Raises:
            ValueError: If a source name is unknown or the new names collide.
            RuntimeError: If the policy has not been initialized.
        """
        if self.model is None:
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
        self._adapt(replace(self.config, input_features=renamed, camera_map=cameras))

    def _adapt(self, config: LingBotVLA2Config) -> None:
        """Adopt a validated feature change; ``configure_model()`` refreshes only the processors."""
        self._config = config
        self.configure_model()
        self.reset()

    def setup(self, stage: str) -> None:
        """Adopt or validate dataset features and statistics; ``configure_model()`` builds modules.

        Raises:
            TypeError: If the datamodule supplies an unsupported dataset type.
            ValueError: If dataset features do not match the policy.
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
        if self._config is None:
            self.input_features, self.output_features = inputs, outputs
            if not self.state_map or not self.action_map:
                from .data import recorded_joint_maps  # noqa: PLC0415

                state_map, action_map = recorded_joint_maps(dataset)
                self.state_map, self.action_map = self.state_map or state_map, self.action_map or action_map
            self._resolve_pretrained()
            config = self._new_config()
        else:
            if _without_normalization(inputs) != _without_normalization(self._config.input_features) or (
                _without_normalization(outputs) != _without_normalization(self._config.output_features)
            ):
                msg = "Dataset features do not match the initialized LingBot policy"
                raise ValueError(msg)
            config = replace(self._config, input_features=inputs, output_features=outputs)
        if config is not None and config.training_recipe == "upstream":
            from physicalai.data.lerobot.dataset import _LeRobotDatasetAdapter  # noqa: PLC0415, PLC2701

            from .data import adopt_training_statistics  # noqa: PLC0415

            if not isinstance(dataset, _LeRobotDatasetAdapter):
                msg = "The upstream recipe requires a LeRobot training dataset for train-only statistics"
                raise TypeError(msg)
            config = adopt_training_statistics(dataset, config)
        self._config = config

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
        with self._autocast():
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
        with self._autocast():
            actions = self.model.predict_action_chunk(processed)
        return self._postprocessor(actions.float(), batch.state)

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
        with self._autocast():
            return self.model.compute_val_loss(self._training_batch(batch))

    def _autocast(self) -> AbstractContextManager:
        """Compute in bf16 when the weights are bf16 and no outer autocast (e.g. a bf16-mixed trainer) is active.

        Returns:
            Autocast context, or a no-op that leaves any outer autocast in effect.
        """
        weight = self.model.state_proj.weight if self.model is not None else None
        if weight is None or weight.dtype != torch.bfloat16 or torch.is_autocast_enabled(weight.device.type):
            return nullcontext()
        return torch.autocast(weight.device.type, dtype=torch.bfloat16)

    def _training_batch(self, batch: Observation) -> dict[str, Tensor]:
        if self._preprocessor is None or self._config is None:
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
        """Use AdamW (Kahan-compensated for bf16 weights) with per-group rates and a warmup-cosine schedule.

        Returns:
            Lightning optimizer and step scheduler.

        Raises:
            RuntimeError: If the policy is not yet configured.
        """
        if self.model is None or self._config is None:
            msg = "LingBot policy is not configured"
            raise RuntimeError(msg)
        lora_scale = self.config.lora_lr_scale if self.config.use_lora else 1.0
        expert_scale = math.sqrt(self.config.token_num_experts / self.config.token_top_k)
        if self.config.training_recipe == "upstream":
            from .optimizer import MuonAdamW  # noqa: PLC0415

            optimizer = MuonAdamW(
                self.model,
                lr=self.optimizer_lr * lora_scale,
                weight_decay=self.optimizer_weight_decay,
                betas=self.optimizer_betas,
                expert_scale=expert_scale,
            )
        else:
            # Routed experts train at lr * sqrt(experts / top_k) as upstream; only LoRA adapters get the LoRA scale.
            groups: dict[float, list[torch.nn.Parameter]] = {}
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                if ".qwenvl.model.visual." in name:
                    lr = self.optimizer_vit_lr
                else:
                    lr = self.optimizer_lr * (expert_scale if ".mlp.experts." in name else 1.0)
                groups.setdefault(lr * (lora_scale if "lora_" in name else 1.0), []).append(param)
            optimizer = build_adamw_auto_precision(
                [{"params": params, "lr": lr} for lr, params in groups.items()],
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

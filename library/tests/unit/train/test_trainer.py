from pathlib import Path

import pytest
import torch
from lightning.pytorch.callbacks import BatchSizeFinder
from torch.utils.data import DataLoader

from physicalai.policies.base import Policy
from physicalai.train.trainer import Trainer


class TestTrainer:
    """Tests for physicalai.train.Trainer (Lightning Trainer subclass)."""

    def test_trainer_is_lightning_subclass(self):
        """Verify Trainer is a subclass of Lightning Trainer."""
        import lightning

        assert issubclass(Trainer, lightning.Trainer)

    def test_trainer_defaults(self):
        """Verify physicalai-specific defaults are set."""
        trainer = Trainer(accelerator="cpu", logger=False, enable_checkpointing=False)

        # physicalai default: num_sanity_val_steps=0 (instead of Lightning's 2)
        assert trainer.num_sanity_val_steps == 0

    def test_policy_dataset_interaction_callback_injected(self):
        """Verify PolicyDatasetInteraction callback is automatically added."""
        from physicalai.train.callbacks import PolicyDatasetInteraction

        trainer = Trainer(accelerator="cpu", logger=False, enable_checkpointing=False)

        # Check that PolicyDatasetInteraction callback was auto-injected
        callback_types = [type(cb) for cb in trainer.callbacks]
        assert PolicyDatasetInteraction in callback_types

    def test_learning_rate_monitor_callback_injected(self, tmp_path):
        """Verify LearningRateMonitor callback is automatically added with per-step logging."""
        from lightning.pytorch.callbacks import LearningRateMonitor

        trainer = Trainer(accelerator="cpu", default_root_dir=str(tmp_path), enable_checkpointing=False)

        monitors = [cb for cb in trainer.callbacks if isinstance(cb, LearningRateMonitor)]
        assert len(monitors) == 1
        assert monitors[0].logging_interval == "step"

    def test_learning_rate_monitor_skipped_without_logger(self):
        """Verify LearningRateMonitor is not injected when logging is disabled."""
        from lightning.pytorch.callbacks import LearningRateMonitor

        trainer = Trainer(accelerator="cpu", logger=False, enable_checkpointing=False)

        assert not any(isinstance(cb, LearningRateMonitor) for cb in trainer.callbacks)

    def test_user_learning_rate_monitor_not_duplicated(self, tmp_path):
        """Verify a user-provided LearningRateMonitor is used instead of injecting another."""
        from lightning.pytorch.callbacks import LearningRateMonitor

        user_monitor = LearningRateMonitor(logging_interval="epoch")
        trainer = Trainer(
            accelerator="cpu",
            default_root_dir=str(tmp_path),
            enable_checkpointing=False,
            callbacks=[user_monitor],
        )

        monitors = [cb for cb in trainer.callbacks if isinstance(cb, LearningRateMonitor)]
        assert monitors == [user_monitor]

    def test_user_callbacks_preserved(self):
        """Verify user callbacks are preserved alongside auto-injected callback."""
        from lightning.pytorch.callbacks import EarlyStopping
        from physicalai.train.callbacks import PolicyDatasetInteraction

        user_callback = EarlyStopping(monitor="val_loss")
        trainer = Trainer(
            accelerator="cpu",
            logger=False,
            enable_checkpointing=False,
            callbacks=[user_callback],
        )

        # Both user callback and auto-injected callback should be present
        callback_types = [type(cb) for cb in trainer.callbacks]
        assert EarlyStopping in callback_types
        assert PolicyDatasetInteraction in callback_types

    def test_callback_specs_from_config_are_instantiated(self):
        """Verify callback dict specs are instantiated before Lightning receives them."""
        from lightning.pytorch.callbacks import EarlyStopping

        trainer = Trainer(
            accelerator="cpu",
            logger=False,
            enable_checkpointing=False,
            callbacks=[
                {
                    "class_path": "lightning.pytorch.callbacks.EarlyStopping",
                    "init_args": {"monitor": "val/loss", "patience": 2},
                }
            ],
        )

        assert any(isinstance(cb, EarlyStopping) for cb in trainer.callbacks)

class _LowPrecisionPolicy(Policy):
    def __init__(self, build_in: str = "setup", dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.build_in = build_in
        self.weight_dtype = dtype
        self.full_precision = torch.nn.Parameter(torch.ones(1, dtype=torch.float64))
        self.register_buffer("floating_buffer", torch.ones(1, dtype=dtype))
        self.register_buffer("integer_buffer", torch.ones(1, dtype=torch.int64))
        if build_in == "init":
            self._build_model()

    def _build_model(self) -> None:
        self.model = torch.nn.Sequential(
            torch.nn.Linear(2, 2, bias=False, dtype=self.weight_dtype),
            torch.nn.Linear(2, 1, bias=False, dtype=self.weight_dtype),
        )
        with torch.no_grad():
            for parameter in self.model.parameters():
                parameter.fill_(1)
        self.model[0].requires_grad_(False)
        self.original_parameter_id = id(self.model[1].weight)

    def setup(self, stage: str) -> None:
        if self.build_in == "setup" and self.model is None:
            self._build_model()

    def configure_model(self) -> None:
        if self.build_in == "configure_model" and self.model is None:
            self._build_model()
        super().configure_model()

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model(batch.to(self.model[0].weight.dtype))

    def predict_action_chunk(self, batch: torch.Tensor) -> torch.Tensor:
        return self(batch)

    def predict_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self(batch)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self(batch).float().square().mean()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        self.weight_at_optimizer_setup = self.model[1].weight.detach().clone()
        return torch.optim.AdamW(self.parameters(), lr=1e-5, weight_decay=0)


def _precision_trainer(tmp_path: Path, precision: str = "bf16-mixed", **kwargs) -> Trainer:
    return Trainer(
        accelerator=kwargs.pop("accelerator", "cpu"),
        precision=precision,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
        max_steps=kwargs.pop("max_steps", 1),
        **kwargs,
    )


class TestMixedPrecisionWeights:
    def test_configure_model_without_trainer_keeps_weight_dtype(self) -> None:
        policy = _LowPrecisionPolicy(build_in="init")
        policy.configure_model()
        assert policy.model[1].weight.dtype == torch.bfloat16

    @pytest.mark.parametrize("build_in", ["init", "setup", "configure_model"])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_promotes_only_trainable_low_precision_weights(
        self, tmp_path: Path, build_in: str, dtype: torch.dtype,
    ) -> None:
        policy = _LowPrecisionPolicy(build_in, dtype)
        trainer = _precision_trainer(tmp_path)
        trainer.fit(policy, train_dataloaders=DataLoader(torch.ones(4, 2), batch_size=2))

        weight = policy.model[1].weight
        assert id(weight) == policy.original_parameter_id
        assert weight.dtype == torch.float32
        assert weight.grad.dtype == torch.float32
        assert policy.weight_at_optimizer_setup.dtype == torch.float32
        assert torch.all(weight < 1), "Small AdamW updates must not round away"
        assert policy.model[0].weight.dtype == dtype
        assert torch.equal(policy.model[0].weight, torch.ones_like(policy.model[0].weight))
        assert policy.full_precision.dtype == torch.float64
        assert policy.floating_buffer.dtype == dtype
        assert policy.integer_buffer.dtype == torch.int64
        state = trainer.optimizers[0].state[weight]
        assert state["exp_avg"].dtype == torch.float32
        assert state["exp_avg_sq"].dtype == torch.float32

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for fp16 AMP")
    @pytest.mark.parametrize("precision", ["bf16-mixed", "16-mixed"])
    def test_cuda_mixed_precision_updates_fp32_weights(self, tmp_path: Path, precision: str) -> None:
        policy = _LowPrecisionPolicy()
        trainer = _precision_trainer(tmp_path, precision=precision, accelerator="gpu", devices=1)
        # Avoid overflow at GradScaler's initial scale so the first fp16 step updates.
        trainer.fit(policy, train_dataloaders=DataLoader(torch.full((4, 2), 0.01), batch_size=2))
        assert policy.model[1].weight.dtype == torch.float32
        assert torch.all(policy.model[1].weight < 1)
        assert policy.model[0].weight.dtype == torch.bfloat16

    @pytest.mark.parametrize("precision", ["32-true", "bf16-true"])
    def test_non_mixed_fit_keeps_weight_dtype(self, tmp_path: Path, precision: str) -> None:
        policy = _LowPrecisionPolicy()
        trainer = _precision_trainer(tmp_path, precision=precision)
        trainer.fit(policy, train_dataloaders=DataLoader(torch.ones(4, 2), batch_size=2))
        assert policy.model[1].weight.dtype == torch.bfloat16

    def test_predict_does_not_promote_weights(self, tmp_path: Path) -> None:
        policy = _LowPrecisionPolicy()
        trainer = _precision_trainer(tmp_path)
        trainer.predict(policy, dataloaders=DataLoader(torch.ones(4, 2), batch_size=2))
        assert policy.model[1].weight.dtype == torch.bfloat16

    def test_resume_preserves_fp32_updates(self, tmp_path: Path) -> None:
        loader = DataLoader(torch.ones(8, 2), batch_size=2)
        policy = _LowPrecisionPolicy()
        trainer = _precision_trainer(tmp_path)
        trainer.fit(policy, train_dataloaders=loader)
        expected = policy.model[1].weight.detach().clone()
        checkpoint = tmp_path / "resume.ckpt"
        trainer.save_checkpoint(checkpoint)

        resumed = _LowPrecisionPolicy()
        resumed_trainer = _precision_trainer(tmp_path, max_steps=2)
        # The checkpoint was generated by this test, not received from an external source.
        resumed_trainer.fit(resumed, train_dataloaders=loader, ckpt_path=checkpoint, weights_only=True)
        torch.testing.assert_close(resumed.weight_at_optimizer_setup, expected, rtol=0, atol=0)
        assert torch.all(resumed.model[1].weight < expected)
        assert resumed_trainer.global_step == 2


class TestTensorBoardLogging:
    """Tests that TensorBoard logging works end to end."""

    def test_experiment_name_creates_tensorboard_logger(self, tmp_path):
        """Verify experiment_name auto-creates a TensorBoardLogger rooted at default_root_dir."""
        pytest.importorskip("tensorboard")
        from lightning.pytorch.loggers import TensorBoardLogger

        trainer = Trainer(
            accelerator="cpu",
            default_root_dir=str(tmp_path),
            experiment_name="pusht_act",
            enable_checkpointing=False,
        )

        assert isinstance(trainer.logger, TensorBoardLogger)
        assert trainer.logger.name == "pusht_act"
        assert trainer.logger.save_dir == str(tmp_path)


class TestAutoScaleBatchSize:
    """Tests for the auto_scale_batch_size feature."""

    def _has_batch_size_finder(self, trainer: Trainer) -> bool:
        return any(isinstance(cb, BatchSizeFinder) for cb in trainer.callbacks)

    def _get_batch_size_finder(self, trainer: Trainer) -> BatchSizeFinder:
        for cb in trainer.callbacks:
            if isinstance(cb, BatchSizeFinder):
                return cb
        raise AssertionError("BatchSizeFinder not found in callbacks")

    def test_disabled_by_default(self):
        trainer = Trainer(accelerator="cpu", logger=False, enable_checkpointing=False)
        assert not self._has_batch_size_finder(trainer)

    def test_enabled_adds_callback(self):
        trainer = Trainer(accelerator="cpu", logger=False, enable_checkpointing=False, auto_scale_batch_size=True)
        assert self._has_batch_size_finder(trainer)

    def test_uses_power_mode(self):
        trainer = Trainer(accelerator="cpu", logger=False, enable_checkpointing=False, auto_scale_batch_size=True)
        finder = self._get_batch_size_finder(trainer)
        assert finder._mode == "power"

    def test_disabled_no_callback(self):
        trainer = Trainer(accelerator="cpu", logger=False, enable_checkpointing=False, auto_scale_batch_size=False)
        assert not self._has_batch_size_finder(trainer)

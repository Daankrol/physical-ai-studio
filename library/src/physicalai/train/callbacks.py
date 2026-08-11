# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Callbacks for training."""

import logging
import time
from collections.abc import Callable, Mapping
from typing import Protocol, cast, runtime_checkable

import lightning as L  # noqa: N812
from lightning.pytorch.callbacks import Callback

from physicalai.train.utils import reformat_dataset_to_match_policy

logger = logging.getLogger(__name__)

ReportFn = Callable[[int, str | None, dict[str, object]], None]
"""Telemetry sink: ``(progress, message, extra_info)`` with progress in 0-100."""

StopFn = Callable[[], bool]
"""Cooperative cancellation probe: returns True when training should stop."""


@runtime_checkable
class SnapFlowCapable(Protocol):
    """A policy that can switch into SnapFlow self-distillation.

    Satisfied by any policy mixing in
    :class:`~physicalai.policies.common.SnapFlowPolicyMixin`, which covers
    :class:`~physicalai.policies.Pi05` and
    :class:`~physicalai.policies.SmolVLA`.

    Note:
        Used for static typing and documentation. Runtime detection goes through
        ``hasattr`` rather than ``isinstance``, because protocol ``isinstance``
        checks inspect the class and would reject duck-typed objects that carry
        the method on the instance.
    """

    def enable_snapflow(self, alpha: float, lambda_: float, num_inference_steps: int) -> None:
        """Activate the SnapFlow objective and freeze the VLM backbone."""
        ...


class ProgressReportingCallback(Callback):
    """Stream standardized training/validation telemetry to a sink.

    Forwards progress, step loss, and validation events through a ``report``
    callable so any consumer (a job store, an SSE stream, or logs) sees an
    identical telemetry schema, and honors cooperative cancellation via
    ``should_stop``. This keeps in-process and remote runners emitting the same
    data without duplicating the Lightning hook logic.

    ``report`` receives ``(progress, message, extra_info)`` where ``progress``
    is 0-100 and ``extra_info`` carries:

    - train batch: ``{"train/loss_step": float | None}``, plus
      ``{"global_step", "max_steps", "epoch"}`` on the logging cadence.
    - validation start: ``{"val_event": "start", "global_step", "max_steps"}``.
    - validation batch: ``{"val_event": "batch", "val_batch", "val/loss_step"}``
      (throttled to the logging cadence).
    - validation end: ``{"val_event": "end", "global_step", "val/loss",
      "val_elapsed_s"}``.

    Example:
        >>> cb = ProgressReportingCallback(report=sink, should_stop=lambda: False)
        >>> trainer = Trainer(callbacks=[cb])
    """

    def __init__(self, *, report: ReportFn, should_stop: StopFn) -> None:
        """Store the telemetry sink and cancellation probe.

        Args:
            report: Sink called with ``(progress, message, extra_info)``.
            should_stop: Returns True when training should stop cooperatively.
        """
        super().__init__()
        self._report = report
        self._should_stop = should_stop
        # Logging cadence in steps; resolved from the step budget on fit start.
        self._every_n_steps = 1
        self._val_start_t: float | None = None

    @staticmethod
    def _auto_every_n_steps(total_steps: int) -> int:
        """Pick a logging cadence in steps.

        Args:
            total_steps: Configured maximum number of steps.

        Returns:
            Cadence in steps. Targets ~1000 entries for budgets up to 100k steps,
            then caps at every 100 steps. Above 100k steps the cap dominates and
            the total entry count grows past 1000.
        """
        if total_steps <= 0:
            return 1
        return min(100, max(1, total_steps // 1000))

    @staticmethod
    def _to_scalar(value: object) -> float | None:
        """Coerce a metric to a float, handling tensors and plain scalars.

        Args:
            value: A 0-d tensor, a Python scalar, or None.

        Returns:
            The float value, or None when ``value`` is None.
        """
        if value is None:
            return None
        item = getattr(value, "item", None)
        if callable(item):
            return float(cast("float", item()))
        return float(value)  # type: ignore[arg-type]

    @staticmethod
    def _max_steps(trainer: L.Trainer) -> int | None:
        """Return the configured step budget, or None when unset/disabled.

        Lightning uses -1 (or any non-positive value) to signal an unbounded step
        budget. Surface that as None so consumers do not misread it as a real
        limit, keeping the value consistent with ``_progress`` returning 0.
        """
        return trainer.max_steps if trainer.max_steps > 0 else None

    @staticmethod
    def _extract_loss(outputs: object) -> float | None:
        """Return a scalar loss from a step output, or None when unavailable.

        Handles a ``{"loss": tensor}`` mapping (training) or a bare loss tensor
        (eval-loss validation).
        """
        candidate = outputs.get("loss") if isinstance(outputs, Mapping) else outputs
        detach = getattr(candidate, "detach", None)
        if detach is None:
            return None
        try:
            return detach().cpu().item()
        except (RuntimeError, ValueError):
            return None

    @staticmethod
    def _progress(trainer: L.Trainer) -> int:
        """Compute step-based completion.

        Args:
            trainer: The active Lightning trainer.

        Returns:
            Completion percentage clamped to 0-100. Returns 0 when ``max_steps``
            is unset (-1) or otherwise non-positive. Emits 100 only once
            ``global_step >= max_steps`` so partial steps never round up to
            completion.
        """
        max_steps = trainer.max_steps
        if max_steps <= 0:
            return 0
        if trainer.global_step >= max_steps:
            return 100
        return min(99, int(trainer.global_step / max_steps * 100))

    def _check_stop(self, trainer: L.Trainer) -> None:
        """Stop the trainer cooperatively when cancellation was requested."""
        if self._should_stop():
            trainer.should_stop = True

    def on_fit_start(self, trainer: L.Trainer, _pl_module: L.LightningModule) -> None:
        """Resolve the logging cadence and honor a cancel requested before training."""
        self._every_n_steps = self._auto_every_n_steps(trainer.max_steps)
        self._check_stop(trainer)

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        _pl_module: L.LightningModule,
        outputs: object,
        _batch: object,
        _batch_idx: int,
    ) -> None:
        """Report step progress and loss; honor cancellation."""
        global_step = trainer.global_step
        extra: dict[str, object] = {"train/loss_step": self._extract_loss(outputs)}
        # Attach the detailed cadence fields so consumers can throttle logs.
        if global_step <= 1 or global_step % self._every_n_steps == 0:
            extra["global_step"] = global_step
            extra["max_steps"] = self._max_steps(trainer)
            extra["epoch"] = trainer.current_epoch
        self._report(self._progress(trainer), None, extra)
        self._check_stop(trainer)

    def on_validation_start(self, trainer: L.Trainer, _pl_module: L.LightningModule) -> None:
        """Report the start of a validation pass; honor cancellation."""
        self._val_start_t = time.monotonic()
        self._report(
            self._progress(trainer),
            None,
            {"val_event": "start", "global_step": trainer.global_step, "max_steps": self._max_steps(trainer)},
        )
        self._check_stop(trainer)

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        _pl_module: L.LightningModule,
        outputs: object,
        _batch: object,
        batch_idx: int,
        dataloader_idx: int = 0,  # noqa: ARG002  # Lightning hook signature; unused here.
    ) -> None:
        """Report a throttled validation batch; honor cancellation."""
        current = batch_idx + 1
        if current == 1 or current % self._every_n_steps == 0:
            self._report(
                self._progress(trainer),
                None,
                {"val_event": "batch", "val_batch": current, "val/loss_step": self._extract_loss(outputs)},
            )
        self._check_stop(trainer)

    def on_validation_epoch_end(self, trainer: L.Trainer, _pl_module: L.LightningModule) -> None:
        """Report the validation summary with aggregated loss and elapsed time."""
        val_loss = trainer.callback_metrics.get("val/loss")
        val_loss_val = self._to_scalar(val_loss)
        elapsed = time.monotonic() - self._val_start_t if self._val_start_t is not None else 0.0
        self._report(
            self._progress(trainer),
            None,
            {
                "val_event": "end",
                "global_step": trainer.global_step,
                "val/loss": val_loss_val,
                "val_elapsed_s": elapsed,
            },
        )


class IterationTimer(Callback):
    """Log wall-clock time per training step in seconds.

    Logs ``train/iter_time_s`` on every training batch end.

    Example:
        >>> from physicalai.train.callbacks import IterationTimer
        >>> trainer = Trainer(callbacks=[IterationTimer()])
    """

    def on_train_batch_start(
        self,
        _trainer: L.Trainer,
        _pl_module: L.LightningModule,
        _batch: object,
        _batch_idx: int,
    ) -> None:
        """Record the batch start time."""
        self._start = time.perf_counter()

    def on_train_batch_end(
        self,
        _trainer: L.Trainer,
        pl_module: L.LightningModule,
        _outputs: object,
        _batch: object,
        _batch_idx: int,
    ) -> None:
        """Log elapsed time since batch start."""
        elapsed_s = time.perf_counter() - self._start
        pl_module.log("train/iter_time_s", elapsed_s, prog_bar=True)


class PolicyDatasetInteraction(Callback):
    """Callback to interact the policy and dataset before training starts."""

    @staticmethod
    def _interact_policy_dataset(trainer: L.Trainer, model: L.LightningModule) -> None:
        # Assumes trainer has a datamodule attached
        if hasattr(trainer, "datamodule") and trainer.datamodule is not None:
            reformat_dataset_to_match_policy(policy=model, datamodule=trainer.datamodule)

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Called at the start of `trainer.fit()`."""
        self._interact_policy_dataset(trainer, pl_module)


class SnapFlowPhaseCallback(Callback):
    """Enable SnapFlow self-distillation at a phase boundary within a single training run.

    At the configured boundary the callback transitions the policy from standard
    flow-matching (phase 1) into SnapFlow distillation (phase 2) without
    requiring a second CLI invocation:

    - Calls ``policy.enable_snapflow(alpha, lambda_, num_inference_steps)``
      which activates the mixed FM/consistency loss and freezes the VLM
      backbone via the policy's existing ``set_requires_grad`` primitive.
    - Reconfigures the optimizer so it only covers the now-trainable
      parameters (action expert + target-time embedding), giving a clean
      optimizer state for phase 2.

    The boundary is expressed either in optimizer steps (``start_step``) or in
    epochs (``start_epoch``); exactly one must be given. Use ``start_epoch``
    when the run is budgeted with ``Trainer(max_epochs=...)`` so the boundary
    does not have to be converted to steps by hand.

    The policy must expose an ``enable_snapflow`` method — both
    :class:`~physicalai.policies.SmolVLA` and
    :class:`~physicalai.policies.Pi05` satisfy this contract.

    Note:
        Reconfiguring the optimizer rebuilds the LR scheduler, so phase 2 starts
        with a fresh warmup. The cosine decay horizon is derived from
        ``Trainer.estimated_stepping_batches``, which reports the *total* run
        budget rather than the phase-2 remainder, so the phase-2 LR decays more
        slowly than a standalone phase-2 run would. Use two explicit
        ``physicalai fit`` runs if you need an exact phase-2 decay horizon.

    Args:
        start_step: Optimizer step at which to activate SnapFlow. Mutually
            exclusive with ``start_epoch``.
        start_epoch: Epoch at which to activate SnapFlow, applied at the start
            of that epoch. Mutually exclusive with ``start_step``.
        alpha: Weight for the flow-matching loss branch (``L_FM``).
            Paper default: ``0.5``.
        lambda_: Scaling factor for the shortcut consistency loss
            (``L_shortcut``).  Paper default: ``0.1``.
        num_inference_steps: Denoising steps at inference time.
            Set to ``1`` for the full single-step SnapFlow speedup.

    Example:
        >>> from physicalai.train.callbacks import SnapFlowPhaseCallback
        >>> cb = SnapFlowPhaseCallback(start_step=50_000)
        >>> trainer = Trainer(max_steps=80_000, callbacks=[cb])

        >>> # Epoch-budgeted run: 10 epochs of flow matching, then 5 of distillation.
        >>> cb = SnapFlowPhaseCallback(start_epoch=10)
        >>> trainer = Trainer(max_epochs=15, callbacks=[cb])
    """

    def __init__(
        self,
        start_step: int | None = None,
        alpha: float = 0.5,
        lambda_: float = 0.1,
        num_inference_steps: int = 1,
        start_epoch: int | None = None,
    ) -> None:
        """Store phase-transition hyperparameters.

        Args:
            start_step: Optimizer step at which SnapFlow distillation begins.
            alpha: FM-loss weight.  Paper default: ``0.5``.
            lambda_: Shortcut-loss scale.  Paper default: ``0.1``.
            num_inference_steps: Inference denoising steps.  Use ``1`` for
                1-NFE SnapFlow.
            start_epoch: Epoch at which SnapFlow distillation begins.

        Raises:
            ValueError: If neither or both of ``start_step`` and ``start_epoch``
                are given, or if the given boundary is negative.
        """
        super().__init__()
        if (start_step is None) == (start_epoch is None):
            msg = (
                "SnapFlowPhaseCallback requires exactly one of start_step or start_epoch, "
                f"got start_step={start_step}, start_epoch={start_epoch}."
            )
            raise ValueError(msg)
        boundary = start_step if start_step is not None else start_epoch
        if boundary is not None and boundary < 0:
            msg = f"SnapFlow phase boundary must be >= 0, got {boundary}."
            raise ValueError(msg)

        self.start_step = start_step
        self.start_epoch = start_epoch
        self.alpha = alpha
        self.lambda_ = lambda_
        self.num_inference_steps = num_inference_steps
        self._activated = False

    def state_dict(self) -> dict[str, object]:
        """Persist the activation flag so a resume does not re-trigger phase 2.

        Returns:
            Callback state to embed in the checkpoint.
        """
        return {"activated": self._activated}

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        """Restore the activation flag from a checkpoint.

        Args:
            state_dict: State previously produced by :meth:`state_dict`.
        """
        self._activated = bool(state_dict.get("activated"))

    def _activate(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Flip the policy into SnapFlow mode and rebuild the optimizer.

        Args:
            trainer: The active Lightning trainer.
            pl_module: The policy being trained.

        Raises:
            TypeError: If ``pl_module`` does not expose an ``enable_snapflow``
                method (i.e. is not a SmolVLA or Pi05 policy).
        """
        logger.info(
            "SnapFlowPhaseCallback: phase boundary reached at step %d / epoch %d, "
            "starting SnapFlow activation",
            trainer.global_step,
            trainer.current_epoch,
        )
        if not hasattr(pl_module, "enable_snapflow"):
            msg = (
                f"{type(pl_module).__name__} does not implement enable_snapflow(); "
                "SnapFlowPhaseCallback requires a SmolVLA or Pi05 policy."
            )
            raise TypeError(msg)
        policy = cast("SnapFlowCapable", pl_module)

        logger.info(
            "SnapFlowPhaseCallback: activating SnapFlow distillation at step %d / epoch %d "
            "(alpha=%.2f, lambda_=%.2f, num_inference_steps=%d)",
            trainer.global_step,
            trainer.current_epoch,
            self.alpha,
            self.lambda_,
            self.num_inference_steps,
        )
        policy.enable_snapflow(
            alpha=self.alpha,
            lambda_=self.lambda_,
            num_inference_steps=self.num_inference_steps,
        )
        # Reconfigure optimizers so frozen VLM params are excluded and phase 2
        # starts with a fresh optimizer state (no stale momentum from phase 1).
        trainer.strategy.setup_optimizers(trainer)
        self._activated = True

        trainable = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)
        total = sum(p.numel() for p in pl_module.parameters())
        logger.info(
            "SnapFlowPhaseCallback: phase-2 trainable params %.1fM / %.1fM (%.1f%%)",
            trainable / 1e6,
            total / 1e6,
            100 * trainable / max(1, total),
        )

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Activate SnapFlow at an epoch boundary when configured with ``start_epoch``.

        Args:
            trainer: The active Lightning trainer.
            pl_module: The policy being trained.
        """
        if self._activated or self.start_epoch is None or trainer.current_epoch < self.start_epoch:
            return
        self._activate(trainer, pl_module)

    def on_train_batch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        batch: object,  # noqa: ARG002
        batch_idx: int,  # noqa: ARG002
    ) -> None:
        """Activate SnapFlow at a step boundary when configured with ``start_step``.

        Args:
            trainer: The active Lightning trainer.
            pl_module: The policy being trained.
            batch: Current batch (unused).
            batch_idx: Current batch index within the epoch (unused).
        """
        if self._activated or self.start_step is None or trainer.global_step < self.start_step:
            return
        self._activate(trainer, pl_module)

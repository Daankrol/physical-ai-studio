# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared SnapFlow self-distillation surface for flow-matching policies.

SnapFlow ([arXiv:2604.05656](https://arxiv.org/abs/2604.05656)) compresses the
multi-step Euler denoising loop of a flow-matching VLA into a single forward
pass by self-distillation. It is trained in two phases: standard flow matching
first, then a short distillation phase with the VLM backbone frozen.

The *math* lives in each policy's model (the target-time embedding, the mixed
FM/consistency objective, and the 1-NFE sampling loop are all inside
``nn.Module`` forward paths that are ``torch.compile``-wrapped and traced during
export). What this module owns is the policy-level surface that was otherwise
duplicated verbatim between :class:`~physicalai.policies.Pi05` and
:class:`~physicalai.policies.SmolVLA`:

- :class:`SnapFlowConfigFields` — the four config flags and their validation.
- :class:`SnapFlowPolicyMixin` — the ``enable_snapflow()`` phase-2 entry point
  used by :class:`~physicalai.train.callbacks.SnapFlowPhaseCallback`.

Example:
    >>> from physicalai.policies import Pi05
    >>> policy = Pi05(pretrained_name_or_path="lerobot/pi05_base")
    >>> policy.enable_snapflow(alpha=0.5, lambda_=0.1, num_inference_steps=1)  # doctest: +SKIP
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import nn

# Paper defaults (arXiv:2604.05656 §3.6 + Appendix J). Note that the config
# default for lambda is 1.0 for backwards compatibility with checkpoints trained
# before the two-phase recipe landed; the phase-2 entry point uses 0.1.
SNAPFLOW_DEFAULT_ALPHA = 0.5
SNAPFLOW_DEFAULT_LAMBDA = 0.1
SNAPFLOW_DEFAULT_NUM_INFERENCE_STEPS = 1


@dataclass(frozen=True)
class SnapFlowConfigFields:
    """SnapFlow self-distillation flags shared by flow-matching policy configs.

    Mix into a policy config ahead of :class:`~physicalai.config.Config` and call
    :meth:`_validate_snapflow` from the config's ``__post_init__``.

    Attributes:
        snapflow_enabled: Enable SnapFlow self-distillation training mode for 1-NFE inference.
            When True, training mixes standard flow-matching with consistency objectives.
            See: arxiv.org/abs/2604.05656. Defaults to False.
        snapflow_alpha: Mixing ratio between FM and consistency objectives. ``alpha`` fraction of samples
            use standard flow-matching loss, ``1-alpha`` use the two-step Euler shortcut consistency loss.
            Must be in [0, 1]. Defaults to 0.5.
        snapflow_lambda: Weight for the consistency (shortcut) loss component. Balances gradient magnitudes
            between FM and consistency objectives. Defaults to 1.0.
        snapflow_num_inference_steps: Number of denoising steps at inference when SnapFlow is enabled.
            Set to 1 for single-step (1-NFE) generation. Defaults to 1.

    Example:
        >>> from dataclasses import dataclass
        >>> from physicalai.config import Config
        >>> from physicalai.policies.common import SnapFlowConfigFields
        >>> @dataclass(frozen=True)
        ... class MyConfig(SnapFlowConfigFields, Config):
        ...     hidden_dim: int = 256
        ...
        ...     def __post_init__(self) -> None:
        ...         self._validate_snapflow()
        >>> MyConfig().snapflow_enabled
        False
    """

    # SnapFlow self-distillation (arxiv.org/abs/2604.05656)
    snapflow_enabled: bool = False
    snapflow_alpha: float = SNAPFLOW_DEFAULT_ALPHA
    snapflow_lambda: float = 1.0
    snapflow_num_inference_steps: int = SNAPFLOW_DEFAULT_NUM_INFERENCE_STEPS

    def _validate_snapflow(self) -> None:
        """Validate the SnapFlow flags.

        Raises:
            ValueError: If ``snapflow_alpha`` falls outside ``[0, 1]`` or
                ``snapflow_num_inference_steps`` is below 1.
        """
        if not 0.0 <= self.snapflow_alpha <= 1.0:
            msg = f"snapflow_alpha must be in [0, 1], got {self.snapflow_alpha}"
            raise ValueError(msg)

        if self.snapflow_num_inference_steps < 1:
            msg = f"snapflow_num_inference_steps must be >= 1, got {self.snapflow_num_inference_steps}"
            raise ValueError(msg)


class SnapFlowPolicyMixin:
    """Give a flow-matching policy the SnapFlow phase-2 entry point.

    Implements :meth:`enable_snapflow`, which switches a policy that has been
    trained with standard flow matching into SnapFlow self-distillation and
    freezes the VLM backbone so only the action expert and the zero-initialised
    target-time embedding keep training (~10% of parameters).

    The mixin depends on two capabilities that are not SnapFlow-specific — they
    are ordinary parts of a VLA policy's API, and each policy family implements
    them differently because the VLM wrapper is named and frozen differently:

    - :attr:`inner_model` — the unwrapped flow-matching ``nn.Module``.
    - :meth:`freeze_vlm` — freeze the VLM backbone so only the action expert
      trains.

    Attributes:
        config: The policy's frozen config dataclass, which must mix in
            :class:`SnapFlowConfigFields` and carry a ``train_expert_only`` flag.
        _set_hparam_keys: Policy hook that re-syncs checkpoint hparams from
            ``config``.

    Example:
        >>> class MyPolicy(SnapFlowPolicyMixin, Policy):  # doctest: +SKIP
        ...     @property
        ...     def inner_model(self):
        ...         return self.model
        ...
        ...     def freeze_vlm(self):
        ...         object.__setattr__(self.config, "train_expert_only", True)
        ...         self.model.vlm.train_expert_only = True
        ...         self.model.vlm.set_requires_grad()
        ...         self.model.train()
    """

    # Declared for type checkers only; provided by the host policy.
    config: Any
    _set_hparam_keys: Callable[[], None]

    @property
    def inner_model(self) -> nn.Module:
        """The unwrapped flow-matching module.

        Implementations return the module that owns the velocity field and the
        target-time embedding, and should raise ``RuntimeError`` when it has not
        been built yet.

        Raises:
            NotImplementedError: If the host policy does not implement the hook.
        """
        msg = f"{type(self).__name__} must implement the inner_model property."
        raise NotImplementedError(msg)

    def freeze_vlm(self) -> None:
        """Freeze the VLM backbone so only the action expert keeps training.

        Implementations set ``config.train_expert_only``, flip ``requires_grad``
        on the backbone, and re-apply train/eval modes.

        Raises:
            NotImplementedError: If the host policy does not implement the hook.
        """
        msg = f"{type(self).__name__} must implement freeze_vlm()."
        raise NotImplementedError(msg)

    def enable_snapflow(
        self,
        alpha: float = SNAPFLOW_DEFAULT_ALPHA,
        lambda_: float = SNAPFLOW_DEFAULT_LAMBDA,
        num_inference_steps: int = SNAPFLOW_DEFAULT_NUM_INFERENCE_STEPS,
    ) -> None:
        """Enable SnapFlow self-distillation and freeze the VLM backbone.

        Activates the SnapFlow mixed FM/consistency objective and freezes the VLM
        so only the action expert and target-time embedding are trained. This is
        the phase-2 entry point used by
        :class:`~physicalai.train.callbacks.SnapFlowPhaseCallback` and can also
        be called manually before ``trainer.fit()``.

        Warm-starting from a well-trained flow-matching checkpoint is a
        precondition: the shortcut target is bootstrapped from the model's own
        marginal-velocity predictions, so distilling an undertrained model
        distills noise.

        Args:
            alpha: Weight for the flow-matching loss branch (``L_FM``).
                Paper default: ``0.5``.
            lambda_: Scaling factor for the shortcut consistency loss
                (``L_shortcut``). Paper default: ``0.1``.
            num_inference_steps: Number of denoising steps at inference time.
                Set to ``1`` for the full single-step SnapFlow speedup.

        Raises:
            ValueError: If ``alpha`` falls outside ``[0, 1]`` or
                ``num_inference_steps`` is below 1.

        Note:
            :attr:`inner_model` raises ``RuntimeError`` when accessed before the
            model has been initialized (i.e. before ``setup()`` runs).
        """
        if not 0.0 <= alpha <= 1.0:
            msg = f"alpha must be in [0, 1], got {alpha}"
            raise ValueError(msg)
        if num_inference_steps < 1:
            msg = f"num_inference_steps must be >= 1, got {num_inference_steps}"
            raise ValueError(msg)

        inner = self.inner_model
        inner._snapflow_enabled = True  # type: ignore[assignment]  # noqa: SLF001
        inner._snapflow_alpha = alpha  # type: ignore[assignment]  # noqa: SLF001
        inner._snapflow_lambda = lambda_  # type: ignore[assignment]  # noqa: SLF001
        inner._snapflow_num_inference_steps = num_inference_steps  # type: ignore[assignment]  # noqa: SLF001

        # Config is a frozen dataclass — bypass the immutability check so the
        # updated flags are included in checkpoint hparams. train_expert_only is
        # set by freeze_vlm(), which owns that half of the state.
        for key, value in (
            ("snapflow_enabled", True),
            ("snapflow_alpha", alpha),
            ("snapflow_lambda", lambda_),
            ("snapflow_num_inference_steps", num_inference_steps),
        ):
            object.__setattr__(self.config, key, value)  # noqa: PLC2801

        self.freeze_vlm()

        # Keep top-level checkpoint hparams in sync with the mutated config so
        # checkpoints saved after this phase transition reload as SnapFlow
        # policies. hparams["config"] is a to_dict() snapshot, not a live view,
        # so it must be refreshed explicitly.
        self._set_hparam_keys()

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared LoRA/DoRA policy-lifecycle mixin for Studio policies."""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, Protocol, cast

import torch

from .functions import build_lora_config, inject_lora, is_lora_injected, merge_lora_

if TYPE_CHECKING:
    from torch import nn

    from .config import PeftConfigMixin
    from .model import PeftModelMixin

    class _PeftCapableModel(PeftModelMixin, nn.Module):
        """Structural type for the ``model`` attribute expected by :class:`PeftPolicyMixin`."""

    class _PeftPolicyHost(Protocol):
        """Structural type for the ``self`` a :class:`PeftPolicyMixin` method is mixed into."""

        config: PeftConfigMixin
        model: _PeftCapableModel | None


logger = logging.getLogger(__name__)


class PeftPolicyMixin:
    """Mixin providing the LoRA injection/export lifecycle for a Studio ``Policy``.

    Expects the concrete ``Policy`` subclass to expose:

    - ``self.config``: a config mixing in :class:`physicalai.policies.peft.PeftConfigMixin`.
    - ``self.model``: an ``nn.Module`` mixing in
      :class:`physicalai.policies.peft.PeftModelMixin` (i.e. implementing
      ``get_default_peft_targets()``), once initialized.

    Typical usage in a policy's ``_initialize_model``::

        if self.config.use_lora:
            self._inject_lora()

    And in ``export()``, before handing the model to an export backend::

        model_to_export = self._merged_lora_model_for_export() or self.model
    """

    def _inject_lora(self) -> None:
        """Inject LoRA adapters into ``self.model``, freezing all base parameters.

        Intended to be called from ``_initialize_model`` (or equivalent) once the model
        has been constructed and any pretrained weights have been loaded. Also useful to
        re-inject adapters when a checkpoint is restored, since Lightning's
        ``load_from_checkpoint`` reruns model construction from hyperparameters before
        restoring the state dict.

        Raises:
            RuntimeError: If ``self.model`` has not been initialized yet.
        """
        self_ = cast("_PeftPolicyHost", self)
        if self_.model is None:
            msg = "Cannot inject LoRA before the model has been initialized."
            raise RuntimeError(msg)

        target_modules = self_.config.lora_target_modules or self_.model.get_default_peft_targets()
        adapter_dtype = None if self_.config.lora_adapter_dtype == "auto" else torch.float32
        lora_config = build_lora_config(
            rank=self_.config.lora_rank,
            alpha=self_.config.effective_lora_alpha,
            dropout=self_.config.lora_dropout,
            target_modules=target_modules,
            use_dora=self_.config.lora_use_dora,
        )
        inject_lora(self_.model, lora_config, adapter_dtype=adapter_dtype)

    def _merged_lora_model_for_export(self) -> nn.Module | None:
        """Return a disposable deep copy of ``self.model`` with LoRA adapters merged in.

        Intended for use inside ``export()`` so exported artifacts fold LoRA adaptation
        into the base layer weights and carry no ``peft`` dependency. Returns ``None`` if
        LoRA is not enabled or not currently injected, in which case callers should export
        ``self.model`` directly.

        Returns:
            A merged deep copy of ``self.model``, or ``None`` if there is nothing to merge.
        """
        self_ = cast("_PeftPolicyHost", self)
        if not (self_.config.use_lora and self_.model is not None and is_lora_injected(self_.model)):
            return None
        merged_model = copy.deepcopy(self_.model)
        merge_lora_(merged_model)
        return merged_model

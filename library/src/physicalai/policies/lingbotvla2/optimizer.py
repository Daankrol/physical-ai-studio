# Copyright (C) 2026 Robbyant Inc.
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Upstream Muon updates for matrix/expert tensors, with AdamW for other parameters."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn
from torch.distributed.tensor import DTensor, Replicate, Shard

if TYPE_CHECKING:
    from collections.abc import Callable


def newton_schulz(update: Tensor, steps: int = 5) -> Tensor:
    """Orthogonalize each trailing matrix with upstream BF16 Newton-Schulz arithmetic.

    Returns:
        Orthogonalized updates in the original dtype.
    """
    dtype = update.dtype
    value = update.to(torch.bfloat16)
    transpose = value.shape[-2] > value.shape[-1]
    if transpose:
        value = value.mT
    value /= value.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-7)
    for _ in range(steps):
        gram = value @ value.mT
        if value.ndim == 2:  # noqa: PLR2004 -- unbatched matrix
            polynomial = torch.addmm(gram, gram, gram, beta=-4.775, alpha=2.0315)
            value = torch.addmm(value, polynomial, value, beta=3.4445)
        else:
            polynomial = torch.baddbmm(gram, gram, gram, beta=-4.775, alpha=2.0315)
            value = torch.baddbmm(value, polynomial, value, beta=3.4445)
    return (value.mT if transpose else value).to(dtype)


class MuonAdamW(torch.optim.Optimizer):
    """A checkpointable optimizer preserving matrix geometry under FSDP2.

    Two-dimensional shards gather one matrix at a time; expert-axis shards update locally.
    This is the ordinary upstream Muon recipe, not its separate FlexShard DistMuon recipe.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 5e-5,
        *,
        weight_decay: float = 0.0,
        betas: tuple[float, float] = (0.9, 0.95),
        expert_scale: float = 1.0,
    ) -> None:
        """Split matrix/expert weights from embedding, bias, and norm parameters.

        Raises:
            ValueError: If learning rate, decay, or expert scaling is invalid.
        """
        if lr < 0 or weight_decay < 0 or expert_scale <= 0:
            msg = "Muon requires nonnegative learning rate/decay and positive expert scaling"
            raise ValueError(msg)
        groups: dict[tuple[bool, float], list[nn.Parameter]] = {}
        embeddings = {
            id(parameter)
            for module in model.modules()
            if isinstance(module, nn.Embedding)
            for parameter in module.parameters()
        }
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            use_muon = (
                parameter.ndim in {2, 3}
                and id(parameter) not in embeddings
                and not any(
                    marker in name.lower() for marker in ("embed_tokens", "embedding", "lm_head", "output_layer")
                )
            )
            rate = lr * expert_scale if ".mlp.experts." in name else lr
            groups.setdefault((use_muon, rate), []).append(parameter)
        super().__init__(
            [{"params": parameters, "muon": muon, "lr": rate} for (muon, rate), parameters in groups.items()],
            {
                "lr": lr,
                "weight_decay": weight_decay,
                "betas": betas,
                "eps": 1e-8,
                "momentum": 0.95,
                "nesterov": True,
                "ns_steps": 5,
            },
        )
        self._attach_adamw()

    def _attach_adamw(self) -> None:
        groups = [group for group in self.param_groups if not group["muon"]]
        self._adamw = torch.optim.AdamW(groups, foreach=True) if groups else None
        if self._adamw is not None:
            self._adamw.state = self.state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore shared optimizer state and reconnect the AdamW groups."""
        super().load_state_dict(state_dict)
        self._attach_adamw()

    @torch.no_grad()
    def step(self, closure: Callable[[], Tensor] | None = None) -> Tensor | None:  # type: ignore[override]  # pyrefly: ignore[bad-override]
        """Update matrix weights with Muon and remaining weights with AdamW.

        Returns:
            Closure loss, when supplied.

        Raises:
            RuntimeError: If a sharded matrix lacks gradients or gradients are sparse.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if not group["muon"]:
                continue
            for parameter in group["params"]:
                if parameter.grad is None:
                    if isinstance(parameter, DTensor):
                        msg = "Sharded Muon requires gradients for every configured matrix"
                        raise RuntimeError(msg)
                    continue
                if parameter.grad.is_sparse:
                    msg = "Muon does not support sparse gradients"
                    raise RuntimeError(msg)
                state = self.state[parameter]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(parameter)
                buffer = state["momentum_buffer"]
                local_parameter = parameter.to_local() if isinstance(parameter, DTensor) else parameter
                gradient = parameter.grad.to_local() if isinstance(parameter.grad, DTensor) else parameter.grad
                local_buffer = buffer.to_local() if isinstance(buffer, DTensor) else buffer
                local_buffer.lerp_(gradient, 1 - group["momentum"])
                update = gradient.lerp(local_buffer, group["momentum"]) if group["nesterov"] else local_buffer.clone()
                gather = isinstance(parameter, DTensor) and any(
                    isinstance(placement, Shard) and (parameter.ndim == 2 or placement.dim != 0)  # noqa: PLR2004
                    for placement in parameter.placements
                )
                if gather:
                    update = DTensor.from_local(
                        update,
                        parameter.device_mesh,
                        parameter.placements,
                        shape=parameter.shape,
                        stride=parameter.stride(),
                        run_check=False,
                    ).full_tensor()
                orthogonal = newton_schulz(update, group["ns_steps"])
                if gather:
                    orthogonal = (
                        DTensor
                        .from_local(
                            orthogonal,
                            parameter.device_mesh,
                            [Replicate()] * parameter.device_mesh.ndim,
                            run_check=False,
                        )
                        .redistribute(placements=parameter.placements)
                        .to_local()
                    )
                adjusted_lr = group["lr"] * 0.2 * math.sqrt(max(parameter.shape[-2:]))
                local_parameter.mul_(1 - group["lr"] * group["weight_decay"])
                local_parameter.add_(orthogonal.to(local_parameter.dtype), alpha=-adjusted_lr)
        if self._adamw is not None:
            self._adamw.step()
        return loss

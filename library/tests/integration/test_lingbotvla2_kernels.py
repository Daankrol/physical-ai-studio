# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Opt-in CUDA forward/backward checks for routed experts and joint attention."""

import os
from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from physicalai.policies.lingbotvla2.components.expert import TokenMoE
from physicalai.policies.lingbotvla2.model import LingBotVLA2Model
from tests.unit.policies.lingbotvla2.test_training import training_batch, training_config

pytestmark = [pytest.mark.integration, pytest.mark.skipif(
    os.environ.get("RUN_LINGBOT_KERNELS") != "1" or not torch.cuda.is_available(), reason="opt-in CUDA kernels",
)]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("num_experts,top_k,steps,hidden_size,intermediate_size", [
    (4, 2, 7, 64, 32),
    (32, 4, 1, 64, 32),  # Most experts receive no tokens.
    (32, 4, 51, 768, 512),
])
def test_fused_expert_forward_backward(dtype, num_experts, top_k, steps, hidden_size, intermediate_size) -> None:
    from physicalai.policies.lingbotvla2.components.fused_moe import fused_experts

    torch.manual_seed(123)
    dense = TokenMoE(hidden_size, num_experts=num_experts, top_k=top_k,
                     intermediate_size=intermediate_size, shared_intermediate_size=32).cuda()
    fused = deepcopy(dense)
    hidden = torch.randn(2, steps, hidden_size, device="cuda", dtype=dtype, requires_grad=True)
    other = hidden.detach().clone().requires_grad_()
    selected = torch.rand(2, steps, num_experts, device="cuda").topk(top_k, dim=-1).indices
    routing = torch.rand(2, steps, top_k, device="cuda", requires_grad=True)
    other_routing = routing.detach().clone().requires_grad_()
    tokens = hidden.to(torch.bfloat16).reshape(-1, hidden_size)
    gate, up, down = [parameter.to(torch.bfloat16) for parameter in dense.experts.parameters()]
    routes = torch.zeros(2 * steps, top_k, hidden_size, device="cuda", dtype=torch.bfloat16)
    for expert in range(num_experts):
        rows, slots = torch.where(selected.flatten(0, 1) == expert)
        activated = torch.nn.functional.silu(torch.nn.functional.linear(tokens[rows], gate[expert]))
        activated = activated * torch.nn.functional.linear(tokens[rows], up[expert])
        activated = activated * routing.flatten(0, 1)[rows, slots, None].to(torch.bfloat16)
        routes[rows, slots] = torch.nn.functional.linear(activated, down[expert])
    expected = routes.reshape(2, steps, top_k, hidden_size).float().sum(-2).to(torch.bfloat16).to(dtype)
    actual = fused_experts(other, selected, other_routing, *fused.experts.parameters())
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=.02)
    gradient = torch.randn_like(expected)
    expected.backward(gradient)
    actual.backward(gradient)
    # Separate experts accumulate BF16 input gradients in a different order.
    torch.testing.assert_close(other.grad, hidden.grad, atol=.004 if hidden_size == 768 else 1e-4, rtol=.03)
    assert (other.grad.float() - hidden.grad.float()).norm() < .01 * hidden.grad.float().norm()
    torch.testing.assert_close(other_routing.grad, routing.grad, atol=.002, rtol=.03)
    for actual_parameter, expected_parameter in zip(fused.experts.parameters(), dense.experts.parameters(), strict=True):
        torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad, atol=.002, rtol=.03)


def test_flex_video_teacher_matches_sdpa() -> None:
    from physicalai.policies.lingbotvla2._vendor.dino_video.lumos_dinov3.models.inference import CausalEvalAdapter
    from tests.unit.policies.lingbotvla2.test_teachers import video_backbone

    torch.manual_seed(123)
    dense = video_backbone()
    CausalEvalAdapter(dense).get_intermediate_layers(torch.zeros(2, 3, 3, 32, 32))
    dense = dense.cuda()
    flex = video_backbone("flex_block_causal").cuda()
    flex.load_state_dict(dense.state_dict(), strict=True)
    for block in flex.blocks:
        block._has_regional_compile = False
    video = torch.randn(2, 3, 3, 32, 32, device="cuda")
    fps = torch.tensor([15., 30.], device="cuda")
    expected = CausalEvalAdapter(dense).get_intermediate_layers(video, fps=fps)[0]
    actual = CausalEvalAdapter(flex).get_intermediate_layers(video, fps=fps)[0]
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-4)


def test_flex_joint_forward_backward() -> None:
    torch.manual_seed(123)
    dense = LingBotVLA2Model.from_config(training_config()).cuda()
    flex = LingBotVLA2Model.from_config(replace(training_config(), attention_backend="flex")).cuda()
    flex.load_state_dict(dense.state_dict())
    batch = {key: value.cuda() for key, value in training_batch().items()}
    expected = dense(batch)[0]
    actual = flex(batch)[0]
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-4)
    expected.backward()
    actual.backward()
    for actual_parameter, expected_parameter in zip(flex.parameters(), dense.parameters(), strict=True):
        if expected_parameter.grad is not None:
            torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad, atol=2e-5, rtol=1e-4)

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Small CPU checks for checkpoint-compatible action expert components."""

import torch
from torch.nn import functional as F

from physicalai.policies.lingbotvla2.components.expert import AdaRMSNorm, TokenMoE
from physicalai.policies.lingbotvla2.components.layer import ExpertLayer


def test_adarmsnorm_zero_condition_matches_rmsnorm() -> None:
    norm = AdaRMSNorm(8, 8)
    inputs = torch.randn(2, 3, 8)
    expected = F.rms_norm(inputs, (8,), norm.weight, eps=1e-6)
    torch.testing.assert_close(norm(inputs, torch.randn(2, 8)), expected)
    norm.gamma.bias.data.fill_(0.5)
    torch.testing.assert_close(norm(inputs, torch.zeros(2, 8)), 1.5 * expected)


def test_dense_moe_matches_per_token_reference_and_backpropagates() -> None:
    torch.manual_seed(7)
    moe = TokenMoE(8, num_experts=4, top_k=2, intermediate_size=6, shared_intermediate_size=6)
    hidden = torch.randn(2, 3, 8, requires_grad=True)
    output, logits = moe(hidden)
    assert logits.shape == (2, 3, 4)
    expected = torch.empty_like(output)
    for batch in range(2):
        for time in range(3):
            token = hidden[batch, time]
            scores = F.linear(token.float(), moe.gate.weight.float()).sigmoid()
            selected = (scores + moe.e_score_correction_bias).topk(2).indices
            weights = scores[selected] / scores[selected].sum() * moe.routed_scaling_factor
            expected[batch, time] = moe.shared_expert(token) + sum(
                weights[index]
                * F.linear(
                    F.silu(F.linear(token, moe.experts.gate_proj[expert]))
                    * F.linear(token, moe.experts.up_proj[expert]),
                    moe.experts.down_proj[expert],
                )
                for index, expert in enumerate(selected)
            )
    torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-5)
    output.sum().backward()
    assert moe.experts.gate_proj.grad is not None
    assert moe.gate.weight.grad is not None
    assert hidden.grad is not None
    assert set(moe.state_dict()) >= {
        "gate.weight",
        "experts.gate_proj",
        "experts.up_proj",
        "experts.down_proj",
        "e_score_correction_bias",
    }


def test_expert_layer_joint_attention_contract() -> None:
    layer = ExpertLayer(
        16,
        num_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        num_experts=4,
        top_k=2,
        intermediate_size=8,
        shared_intermediate_size=8,
    )
    hidden = torch.randn(2, 5, 16, requires_grad=True)
    cond = torch.randn(2, 16)
    q, k, v = layer.qkv(hidden, cond)
    assert q.shape == (2, 5, 4, 4)
    assert k.shape == v.shape == (2, 5, 2, 4)
    updated, logits = layer.finish(hidden, torch.randn(2, 5, 4, 4), cond)
    assert updated.shape == hidden.shape
    assert logits.shape == (2, 5, 4)
    updated.square().mean().backward()
    assert hidden.grad is not None
    assert set(layer.state_dict()) >= {
        "self_attn.q_proj.bias",
        "self_attn.o_proj.weight",
        "input_layernorm.gamma.weight",
        "post_attention_layernorm.beta.bias",
        "mlp.experts.gate_proj",
    }

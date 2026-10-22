# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Frozen video-teacher causality and supported attention paths."""

import pytest
import torch

from physicalai.policies.lingbotvla2._vendor.dino_video.lumos_dinov3.models.inference import CausalEvalAdapter
from physicalai.policies.lingbotvla2._vendor.dino_video.lumos_dinov3.models.navit_video_vision_transformer import NaviTVideoViT


def video_backbone(attention_mode: str = "sdpa_block_causal") -> NaviTVideoViT:
    backbone = NaviTVideoViT(
        img_size=32, patch_size=16, embed_dim=64, depth=2, num_heads=4,
        layerscale_init=1., n_storage_tokens=2, mask_k_bias=True,
        pos_embed_rope_3d=True, pos_embed_rope_prefix_temporal=True,
        pos_embed_rope_dtype="fp32", attention_mode=attention_mode,
    )
    backbone.init_weights()
    return backbone.eval()


def test_video_teacher_preserves_frame_causality_and_temporal_positions() -> None:
    torch.manual_seed(123)
    teacher = CausalEvalAdapter(video_backbone())
    video = torch.randn(1, 3, 3, 32, 32)
    expected = teacher.get_intermediate_layers(video, fps=30.)[0]
    changed = video.clone()
    changed[:, :, -1] += torch.randn_like(changed[:, :, -1])
    actual = teacher.get_intermediate_layers(changed, fps=30.)[0]
    assert actual.shape == (1, 12, 64)
    torch.testing.assert_close(actual[:, :8], expected[:, :8], atol=0, rtol=0)
    assert not torch.allclose(actual[:, 8:], expected[:, 8:])
    slower = teacher.get_intermediate_layers(video, fps=15.)[0]
    assert not torch.allclose(slower, expected)
    assert expected.isfinite().all() and not expected.requires_grad


def test_video_teacher_rejects_unsupported_attention() -> None:
    with pytest.raises(ValueError, match="Unsupported video-teacher attention"):
        video_backbone("flash_attn3")


@pytest.mark.parametrize("blocks", [0, 3, [], [0, 0], [-1]])
def test_video_teacher_keeps_block_validation(blocks: int | list[int]) -> None:
    teacher = CausalEvalAdapter(video_backbone())
    with pytest.raises(ValueError):
        teacher.get_intermediate_layers(torch.zeros(1, 3, 3, 32, 32), n=blocks)

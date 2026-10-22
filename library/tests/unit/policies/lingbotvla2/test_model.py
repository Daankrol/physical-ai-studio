# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Run the small joint VLM/action expert on CPU."""

import torch

from physicalai.policies.lingbotvla2.config import LingBotVLA2Config
from physicalai.policies.lingbotvla2.model import LingBotVLA2Model, _masked_mean


def test_flow_mask_ignores_unmapped_dimensions_and_episode_padding() -> None:
    errors = torch.tensor([[[1.0, 100.0], [100.0, 100.0]]])
    mask = torch.tensor([[True, False]])
    padded = torch.tensor([[False, True]])
    torch.testing.assert_close(_masked_mean(errors, mask, padded), torch.tensor(1.0))
    torch.testing.assert_close(_masked_mean(errors, torch.zeros_like(mask), padded), torch.tensor(0.0))


def test_tiny_model_train_and_cached_sample() -> None:
    config = LingBotVLA2Config(
        robot_layout="so101",
        chunk_size=2,
        n_action_steps=1,
        num_denoise_steps=2,
        num_layers=2,
        token_moe_layers=(0, 1),
        hidden_size=64,
        intermediate_size=128,
        head_dim=16,
        mrope_section=(4, 2, 2),
        num_attention_heads=4,
        num_key_value_heads=2,
        expert_num_attention_heads=4,
        expert_num_key_value_heads=2,
        expert_head_dim=16,
        expert_hidden_size=32,
        max_action_dim=55,
        max_state_dim=55,
        token_num_experts=4,
        token_top_k=2,
        token_moe_intermediate_size=16,
        token_shared_intermediate_size=16,
        vision_hidden_size=32,
        vision_intermediate_size=64,
        vision_num_heads=4,
        vision_num_layers=2,
        vision_deepstack_indexes=(0, 1),
        num_task_tokens=2,
        num_query_backbone_tokens=4,
        camera_slots=("top",),
    )
    model = LingBotVLA2Model.from_config(config)
    batch = {
        "pixel_values": torch.randn(1, 1, 4, 1536),
        "image_grid_thw": torch.tensor([[[1, 2, 2]]]),
        "img_masks": torch.tensor([[True]]),
        "lang_tokens": torch.tensor([[3, 4, 5]]),
        "lang_masks": torch.tensor([[True, True, True]]),
        "state": torch.randn(1, 55),
        "action": torch.randn(1, 2, 55),
        "joint_mask": torch.ones(1, 55, dtype=torch.bool),
        "action_is_pad": torch.tensor([[False, True]]),
        "noise": torch.randn(1, 2, 55),
        "time": torch.tensor([0.4]),
    }
    with torch.no_grad():
        prefix, valid, blocks, positions, _, _ = model.embed_prefix(batch)
        from physicalai.policies.lingbotvla2.model import _attention_mask

        stock = model.qwenvl_with_expert.qwenvl.model.language_model(
            inputs_embeds=prefix,
            attention_mask=valid.long(),
            position_ids=positions,
            use_cache=False,
        ).last_hidden_state
        joint, _, _ = model._joint(prefix, None, _attention_mask(valid, blocks), positions)
        torch.testing.assert_close(joint, stock, atol=1e-5, rtol=1e-5)
        missing_camera = {**batch, "img_masks": torch.tensor([[False]])}
        _, valid_without_image, _, missing_positions, _, _ = model.embed_prefix(missing_camera)
        assert not valid_without_image[0, :3].any()
        assert missing_positions[:, 0, :3].eq(1).all()
    loss, metrics = model.compute_loss(batch)
    assert loss.isfinite() and set(metrics) == {"loss", "flow_loss", "moe_seq_loss", "moe_z_loss"}
    loss.backward()
    assert model.action_out_proj.weight.grad is not None
    assert model.qwenvl_with_expert.qwenvl.model.language_model.layers[0].self_attn.q_proj.weight.grad is not None
    assert model.qwenvl_with_expert.qwen_expert.model.layers[0].mlp.experts.gate_proj.grad is not None
    model.zero_grad(set_to_none=True)
    model._joint.gradient_checkpointing = True
    model.compute_loss(batch)[0].backward()
    assert model.qwenvl_with_expert.qwenvl.model.language_model.layers[0].self_attn.q_proj.weight.grad is not None
    model.eval()
    with torch.no_grad():
        prefix = model.embed_prefix(batch)
        embeddings, valid, blocks, positions, visual_mask, deepstack = prefix
        from physicalai.policies.lingbotvla2.model import _attention_mask

        cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        model._joint(
            embeddings,
            None,
            _attention_mask(valid, blocks),
            positions,
            cache=cache,
            fill_cache=True,
            visual_pos_masks=visual_mask,
            deepstack=deepstack,
        )
        time = torch.tensor([0.5])
        velocity_full, _, _ = model._velocity(batch, batch["action"], time, prefix=prefix)
        velocity_cached, _, _ = model._velocity(
            {**batch, "prefix_valid": valid, "prefix_positions": positions}, batch["action"], time, cache=cache
        )
        torch.testing.assert_close(velocity_cached, velocity_full, atol=1e-5, rtol=1e-5)
    prediction = model.predict_action_chunk(batch, noise=torch.zeros(1, 2, 55))
    assert prediction.shape == (1, 2, 55)
    assert prediction.isfinite().all()

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Local pretrained recipe resolution must reject incompatible query layouts."""

from typing import Any

import pytest
import yaml

from physicalai.policies.lingbotvla2.pretrained import resolve_pretrained


def test_hub_repo_requires_pinned_revision() -> None:
    with pytest.raises(ValueError, match="pinned 40-character"):
        resolve_pretrained("some-org/some-checkpoint")
    with pytest.raises(ValueError, match="pinned 40-character"):
        resolve_pretrained("some-org/some-checkpoint", revision="main")


def test_local_recipe_mapping(tmp_path) -> None:
    (tmp_path / "model.safetensors.index.json").write_text('{"weight_map": {}}')
    recipe: dict[str, Any] = {
        "model": {"use_lm_head": False},
        "train": {
            "chunk_size": 50,
            "token_moe_layers": [0, 1],
            "token_top_k": 4,
            "my_tokenizer_max_length": 72,
            "align_params": {
                "mode": "query",
                "num_task_tokens": 8,
                "depth": {"num_backbone_tokens": 256},
                "video": {
                    "share_future_depth_query": True,
                    "use_current_shared_task_proj": True,
                    "use_shared_future_task_proj": True,
                },
            },
        },
    }
    (tmp_path / "lingbotvla_cli.yaml").write_text(yaml.safe_dump(recipe))
    overrides, path = resolve_pretrained(tmp_path)
    assert path.name == "model.safetensors.index.json"
    assert overrides == {
        "chunk_size": 50,
        "token_moe_layers": (0, 1),
        "token_top_k": 4,
        "tokenizer_max_length": 72,
        "num_task_tokens": 8,
        "num_query_backbone_tokens": 256,
    }
    recipe["train"]["align_params"]["video"]["share_future_depth_query"] = False
    (tmp_path / "lingbotvla_cli.yaml").write_text(yaml.safe_dump(recipe))
    with pytest.raises(ValueError, match="Unsupported"):
        resolve_pretrained(tmp_path)

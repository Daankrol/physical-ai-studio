# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Torch artifact roundtrip through the Runtime adapter with fixed noise."""

from unittest.mock import patch

import numpy as np
import torch

from physicalai.data import Observation
from physicalai.export import ExportBackend
from physicalai.inference import InferenceModel
from physicalai.inference.adapters.pytorch import TorchAdapter
from physicalai.policies.lingbotvla2.policy import LingBotVLA2

from .test_policy import tiny_config
from .test_processors import StubTokenizer


def test_torch_export_and_adapter_restore(tmp_path) -> None:
    policy = LingBotVLA2.from_config(tiny_config()).eval()
    policy._preprocessor._tokenizer = StubTokenizer()
    inputs = {
        "state": np.ones((1, 6), dtype=np.float32),
        "images.front": np.ones((1, 3, 32, 32), dtype=np.float32) * 0.4,
        "task": ["move"],
    }
    torch.manual_seed(42)
    expected = policy.predict_action_chunk(Observation.from_dict(inputs).to_torch()).detach().numpy()
    with patch("physicalai.policies.lingbotvla2.export.AutoTokenizer.from_pretrained", return_value=StubTokenizer()):
        policy.export(tmp_path, backend=ExportBackend.TORCH)
    assert (tmp_path / "tokenizer" / "tokenizer.json").exists()
    assert (tmp_path / "lingbotvla2.pt").exists()
    assert (tmp_path / "manifest.json").exists()
    with (
        patch(
            "physicalai.policies.lingbotvla2.preprocessor.AutoTokenizer.from_pretrained", return_value=StubTokenizer()
        ),
        patch("physicalai.policies.lingbotvla2.policy.resolve_pretrained", side_effect=AssertionError("hub touched")),
    ):
        adapter = TorchAdapter()
        adapter.load(tmp_path / "lingbotvla2.pt")
        assert adapter._policy._preprocessor.tokenizer_name_or_path == str(tmp_path / "tokenizer")
        torch.manual_seed(42)
        actual = adapter.predict(inputs)["action"]
        runtime = InferenceModel(tmp_path, device="cpu")
        torch.manual_seed(42)
        runtime_actions = runtime.predict_action_chunk(inputs)
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(runtime_actions, expected[0], atol=1e-5, rtol=1e-5)

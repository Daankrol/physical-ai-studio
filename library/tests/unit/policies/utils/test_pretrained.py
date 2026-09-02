# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit Tests - Pretrained Config Loading Helpers

Regression coverage for issue physical-ai-studio#1069: pretrained lerobot
``config.json`` files carry extra keys (e.g. ``input_features``,
``output_features``, ``repo_id``, ``license``) that
``physicalai.config.Config.from_dict(..., strict=False)`` no longer silently
ignores (upstream regression, see known_config_fields_only() docstring).
``known_config_fields_only`` is the Studio-side workaround; these tests pin
its contract across all first-party policies that load pretrained weights.
"""

from __future__ import annotations

import dataclasses

import pytest
from physicalai.policies.pi05.config import Pi05Config
from physicalai.policies.rldx1.config import Rldx1Config
from physicalai.policies.smolvla.config import SmolVLAConfig
from physicalai.policies.utils.pretrained import known_config_fields_only

# Junk keys observed on real lerobot pretrained config.json files (e.g.
# lerobot/pi05_base), none of which are fields on any first-party policy
# config dataclass.
_REAL_WORLD_JUNK_KEYS: dict[str, object] = {
    "type": "pi05",
    "device": "cuda",
    "use_amp": False,
    "push_to_hub": True,
    "repo_id": "lerobot/pi05_base",
    "private": None,
    "tags": None,
    "license": "apache-2.0",
    "input_features": {
        "observation.images.base_0_rgb": {"type": "VISUAL", "shape": [3, 224, 224]},
        "observation.state": {"type": "STATE", "shape": [8]},
    },
    "output_features": {"action": {"type": "ACTION", "shape": [7]}},
}


@pytest.mark.parametrize("config_cls", [Pi05Config, Rldx1Config, SmolVLAConfig])
class TestKnownConfigFieldsOnly:
    """Contract tests for known_config_fields_only, parametrized per policy config."""

    def test_drops_unknown_keys(self, config_cls: type) -> None:
        """Keys with no matching dataclass field are dropped."""
        data = {**_REAL_WORLD_JUNK_KEYS, "not_a_real_field_either": 123}
        filtered = known_config_fields_only(config_cls, data)
        known_fields = {field.name for field in dataclasses.fields(config_cls)}
        assert set(filtered) <= known_fields
        assert "input_features" not in filtered
        assert "output_features" not in filtered
        assert "not_a_real_field_either" not in filtered

    def test_preserves_known_keys(self, config_cls: type) -> None:
        """Keys that do match a dataclass field survive filtering, values intact."""
        known_fields = list(dataclasses.fields(config_cls))
        assert known_fields, f"{config_cls.__name__} unexpectedly has no fields"

        sentinel_field = known_fields[0].name
        data = {**_REAL_WORLD_JUNK_KEYS, sentinel_field: "sentinel-value"}
        filtered = known_config_fields_only(config_cls, data)

        assert filtered[sentinel_field] == "sentinel-value"

    def test_does_not_mutate_input(self, config_cls: type) -> None:
        """The input mapping is not mutated in place."""
        data = dict(_REAL_WORLD_JUNK_KEYS)
        original = dict(data)
        known_config_fields_only(config_cls, data)
        assert data == original

    def test_empty_input_returns_empty(self, config_cls: type) -> None:
        """Filtering an empty mapping returns an empty mapping."""
        assert known_config_fields_only(config_cls, {}) == {}

    def test_logs_dropped_keys(
        self,
        config_cls: type,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Dropped keys are logged at debug level for observability."""
        with caplog.at_level("DEBUG", logger="physicalai.policies.utils.pretrained"):
            known_config_fields_only(config_cls, _REAL_WORLD_JUNK_KEYS)
        assert any("input_features" in record.message for record in caplog.records)


def test_raises_for_non_dataclass() -> None:
    """A non-dataclass target raises TypeError instead of failing obscurely."""

    class NotADataclass:
        pass

    with pytest.raises(TypeError, match="must be a dataclass"):
        known_config_fields_only(NotADataclass, {"a": 1})

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for loading pretrained ``config.json`` files across policies."""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)

__all__ = ["known_config_fields_only"]


def known_config_fields_only[ConfigT](config_cls: type[ConfigT], data: Mapping[str, object]) -> dict[str, object]:
    """Drop keys from *data* that have no matching field on *config_cls*.

    This is a workaround for an upstream regression in runtime-owned
    ``physicalai.config.Config.from_dict``: passing ``strict=False`` no longer
    silently ignores unknown keys. The current implementation builds a
    jsonargparse parser from only the dataclass's known fields and always
    rejects any key without a matching constructor argument, regardless of
    ``strict``. Pretrained ``lerobot`` ``config.json`` files always carry extra
    bookkeeping keys (e.g. ``input_features``, ``output_features``, ``repo_id``,
    ``license``, ``push_to_hub``) that trip this up (see physical-ai-studio#1069).

    Filtering the keys ourselves before calling ``from_dict`` sidesteps the
    issue without depending on ``strict`` doing anything. Tracked upstream at
    openvinotoolkit/physicalai#251; this helper should be revisited (and
    likely removed) once the runtime's ``Config.from_dict`` honours
    ``strict=False`` again.

    Args:
        config_cls: The dataclass ``Config`` subclass being constructed.
        data: Raw mapping (typically parsed from a pretrained ``config.json``).

    Returns:
        A copy of *data* containing only keys that are fields on *config_cls*.

    Raises:
        TypeError: If *config_cls* is not a dataclass.
    """
    if not dataclasses.is_dataclass(config_cls):
        msg = f"{config_cls.__name__} must be a dataclass"
        raise TypeError(msg)

    known_fields = {field.name for field in dataclasses.fields(config_cls)}
    dropped = sorted(set(data) - known_fields)
    if dropped:
        logger.debug(
            "Ignoring %d pretrained config key(s) with no matching %s field: %s",
            len(dropped),
            config_cls.__name__,
            dropped,
        )

    return {key: value for key, value in data.items() if key in known_fields}

# Adapted for Physical AI Studio; see the vendor README for source, licenses, and changes.
from __future__ import annotations

from typing import Sequence


def validate_cls_pool(cls_pool: str) -> str:
    if cls_pool not in {"mean", "last"}:
        raise ValueError(f"cls_pool must be 'mean' or 'last', got {cls_pool!r}")
    return cls_pool


BLOCK_CAUSAL_ATTENTION_MODES = frozenset(
    {"flex_block_causal", "sdpa_block_causal"}
)


def validate_block_causal_backbone(backbone) -> None:
    """Reject backbones whose attention is not frame-level causal."""
    mode = getattr(backbone, "attention_mode", None)
    if mode not in BLOCK_CAUSAL_ATTENTION_MODES:
        raise ValueError(
            f"causal inference requires a block-causal attention_mode "
            f"(one of {sorted(BLOCK_CAUSAL_ATTENTION_MODES)}), but the backbone has "
            f"attention_mode={mode!r}, which runs bidirectional attention. "
            f"Rebuild the backbone with a block-causal attention_mode."
        )


def resolve_block_indices(n: int | Sequence[int], depth: int) -> list[int]:
    if isinstance(n, int):
        if not (0 < n <= depth):
            raise ValueError(f"n must be in [1, {depth}], got {n}")
        return list(range(depth - n, depth))
    indices = [int(i) for i in n]
    if not indices:
        raise ValueError("n sequence must not be empty")
    if len(set(indices)) != len(indices):
        raise ValueError(f"n contains duplicate block indices: {indices}")
    if any(i < 0 or i >= depth for i in indices):
        raise ValueError(f"n contains out-of-range block index for depth {depth}: {indices}")
    return sorted(indices)

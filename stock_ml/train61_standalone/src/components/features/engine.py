from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import pandas as pd

from src.components.features.base import FeatureBlock

if TYPE_CHECKING:
    pass


class ComposableFeatureEngine:
    """Composes multiple FeatureBlocks into a single feature set."""

    def __init__(self, blocks: list[FeatureBlock]) -> None:
        self.blocks = blocks
        self._validate_dependencies()

    def _validate_dependencies(self) -> None:
        available: set[str] = set()
        for block in self.blocks:
            for req in block.requires:
                if req not in available and req not in {
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "traded_value",
                    "symbol",
                    "timestamp",
                }:
                    pass  # soft check — cross-block columns (e.g. bb_width) are handled lazily
            available.update(block.get_feature_names())

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all blocks sequentially. Returns enriched DataFrame."""
        for block in self.blocks:
            df = block.compute(df)
        return df

    def compute_for_all_symbols(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply per-symbol blocks, then cross-sectional blocks."""
        from src.components.features.blocks.relative_strength import RelativeStrengthBlock

        per_symbol_blocks = [b for b in self.blocks if not isinstance(b, RelativeStrengthBlock)]
        cross_blocks = [b for b in self.blocks if isinstance(b, RelativeStrengthBlock)]

        parts = []
        for _symbol, group in df.groupby("symbol"):
            for block in per_symbol_blocks:
                group = block.compute(group)
            parts.append(group)
        result = pd.concat(parts, ignore_index=True)

        for block in cross_blocks:
            result = block.compute(result)

        return result

    def get_feature_names(self) -> list[str]:
        return [name for block in self.blocks for name in block.get_feature_names()]

    def signature(self) -> str:
        """Stable hash for cache key."""
        names = [block.name for block in self.blocks]
        return hashlib.sha256(json.dumps(names, sort_keys=True).encode()).hexdigest()[:16]

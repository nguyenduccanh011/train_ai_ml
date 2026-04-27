from __future__ import annotations

from pathlib import Path

import yaml

from src.components.features.blocks.accumulation import AccumulationBlock
from src.components.features.blocks.exhaustion import ExhaustionBlock
from src.components.features.blocks.heikin_ashi import HeikinAshiBlock
from src.components.features.blocks.leading_signals import LeadingSignalsBlock
from src.components.features.blocks.market_structure import MarketStructureBlock
from src.components.features.blocks.momentum import MomentumBlock
from src.components.features.blocks.moving_averages import MovingAverageBlock
from src.components.features.blocks.multi_timeframe import MultiTimeframeBlock
from src.components.features.blocks.ohlcv_basic import OhlcvBasicBlock
from src.components.features.blocks.relative_strength import RelativeStrengthBlock
from src.components.features.blocks.trend import TrendBlock
from src.components.features.blocks.volatility import VolatilityBlock
from src.components.features.blocks.volatility_regime import VolatilityRegimeBlock
from src.components.features.blocks.volume_advanced import VolumeAdvancedBlock
from src.components.features.engine import ComposableFeatureEngine

_BLOCK_REGISTRY: dict[str, type] = {
    "ohlcv_basic": OhlcvBasicBlock,
    "moving_averages": MovingAverageBlock,
    "momentum": MomentumBlock,
    "trend": TrendBlock,
    "volatility": VolatilityBlock,
    "volume_advanced": VolumeAdvancedBlock,
    "leading_signals": LeadingSignalsBlock,
    "market_structure": MarketStructureBlock,
    "exhaustion": ExhaustionBlock,
    "volatility_regime": VolatilityRegimeBlock,
    "multi_timeframe": MultiTimeframeBlock,
    "accumulation": AccumulationBlock,
    "relative_strength": RelativeStrengthBlock,
    "heikin_ashi": HeikinAshiBlock,
}


def get_block(name: str):
    if name not in _BLOCK_REGISTRY:
        raise KeyError(f"Unknown feature block: '{name}'. Available: {sorted(_BLOCK_REGISTRY)}")
    return _BLOCK_REGISTRY[name]()


def build_engine_from_yaml(path: Path) -> ComposableFeatureEngine:
    """Load a feature_set YAML and instantiate the corresponding engine."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    block_names: list[str] = cfg["blocks"]
    blocks = [get_block(name) for name in block_names]
    return ComposableFeatureEngine(blocks)


def build_engine_from_name(name: str, config_dir: Path | None = None) -> ComposableFeatureEngine:
    """Resolve a feature set name to a YAML config and build engine."""
    if config_dir is None:
        config_dir = Path(__file__).parents[4] / "config" / "feature_sets"
    yaml_path = config_dir / f"{name}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Feature set config not found: {yaml_path}")
    return build_engine_from_yaml(yaml_path)

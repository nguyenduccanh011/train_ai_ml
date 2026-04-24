from .engine import backtest_unified
from .indicators import compute_indicators, detect_trend_strength, get_regime_adapter
from .defaults import DEFAULT_PARAMS, FEATURE_NAMES, FEATURE_DEFAULTS, SYMBOL_PROFILES

__all__ = [
    "backtest_unified",
    "compute_indicators",
    "detect_trend_strength",
    "get_regime_adapter",
    "DEFAULT_PARAMS",
    "FEATURE_NAMES",
    "FEATURE_DEFAULTS",
    "SYMBOL_PROFILES",
]

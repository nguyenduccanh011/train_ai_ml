"""Target registry — manages label definitions for models.

Targets define how to compute labels {-1, 0, 1} from raw OHLCV data.
"""

from __future__ import annotations

import math
from typing import Protocol

import pandas as pd

from stock_ml.src.targets.continuation_entry import ContinuationEntryRegressionTarget
from stock_ml.src.targets.bottom_structure_entry import (
    BottomStructureEntryRegressionTarget,
)
from stock_ml.src.targets.action_oracle import ActionOracleTarget
from stock_ml.src.targets.amplitude_oracle import MaxProfitActionTarget
from stock_ml.src.targets.continuation_recov_entry import (
    ContinuationRecovEntryRegressionTarget,
)
from stock_ml.src.targets.downleg_depth import DownlegDepthRegressionTarget
from stock_ml.src.targets.early_wave import EarlyWaveExitTarget, EarlyWaveV2Target
from stock_ml.src.targets.forward import ForwardReturnTarget
from stock_ml.src.targets.forward_drawdown import ForwardDrawdownRegressionTarget
from stock_ml.src.targets.forward_regression import ForwardReturnRegressionTarget
from stock_ml.src.targets.forward_return_penalized import (
    ForwardReturnPenalizedRegressionTarget,
)
from stock_ml.src.targets.path_quality import (
    MFERegressionTarget,
    MultiHorizonReturnTarget,
    RewardRiskRegressionTarget,
)
from stock_ml.src.targets.reversal_entry import ReversalEntryRegressionTarget
from stock_ml.src.targets.risk_exit import RiskExitRegressionTarget
from stock_ml.src.targets.swing_value import SwingValueRegressionTarget
from stock_ml.src.targets.trend_scanning import TrendScanningExitTarget
from stock_ml.src.targets.triple_barrier import TripleBarrierTarget
from stock_ml.src.targets.velocity_exit import VelocityExitRegressionTarget
from stock_ml.src.targets.zigzag import ZigzagPivotTarget


class TargetProtocol(Protocol):
    """Target interface — applies labels to DataFrame."""

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add 'target' column with {-1, 0, 1, NaN} labels.

        Args:
            df: DataFrame with [symbol, date, close] columns

        Returns:
            DataFrame with 'target' column added
        """
        ...


class TrendRegimeTarget:
    """Trend regime target — labels based on SMA crossover.

    Buy (1): short SMA > long SMA and close above long SMA
    Sell (-1): short SMA < long SMA and close below long SMA
    Neutral (0): otherwise
    """

    def __init__(self, short_window: int = 5, long_window: int = 20):
        self.short_window = short_window
        self.long_window = long_window

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute trend regime labels.

        Args:
            df: DataFrame with [symbol, date, close] columns

        Returns:
            DataFrame with 'target' column added
        """
        out = df.copy()

        def _labels_per_symbol(g):
            g = g.copy()
            close = g["close"]
            sma_short = close.rolling(self.short_window, min_periods=1).mean()
            sma_long = close.rolling(self.long_window, min_periods=1).mean()

            target = pd.Series(0, index=g.index, dtype="int8")
            buy_mask = (sma_short > sma_long) & (close > sma_long)
            sell_mask = (sma_short < sma_long) & (close < sma_long)

            target[buy_mask] = 1
            target[sell_mask] = -1
            g["target"] = target
            return g

        out = out.groupby("symbol", group_keys=False).apply(_labels_per_symbol)
        return out


_REGISTRY: dict[str, type] = {
    "forward_return": ForwardReturnTarget,
    "forward_return_regression": ForwardReturnRegressionTarget,
    "forward_return_penalized_regression": ForwardReturnPenalizedRegressionTarget,
    "forward_drawdown_regression": ForwardDrawdownRegressionTarget,
    "continuation_entry_regression": ContinuationEntryRegressionTarget,
    "continuation_recov_entry_regression": ContinuationRecovEntryRegressionTarget,
    "reversal_entry_regression": ReversalEntryRegressionTarget,
    "bottom_structure_entry_regression": BottomStructureEntryRegressionTarget,
    "risk_exit_regression": RiskExitRegressionTarget,
    "swing_value_regression": SwingValueRegressionTarget,
    "velocity_exit_regression": VelocityExitRegressionTarget,
    "trend_scanning_exit": TrendScanningExitTarget,
    "trend_regime": TrendRegimeTarget,
    "zigzag_pivot": ZigzagPivotTarget,
    "action_oracle": ActionOracleTarget,
    "amplitude_oracle": MaxProfitActionTarget,
    "triple_barrier": TripleBarrierTarget,
    "mfe_regression": MFERegressionTarget,
    "reward_risk_regression": RewardRiskRegressionTarget,
    "multi_horizon_return": MultiHorizonReturnTarget,
    "downleg_depth_regression": DownlegDepthRegressionTarget,
    "early_wave_v2": EarlyWaveV2Target,
    "early_wave_exit": EarlyWaveExitTarget,
}


# Zigzag pivots are confirmed only by a future pct-reversal (forward distance to the next
# pivot is unbounded), so there is no clean `horizon`. The soft label decays as exp(-dist/tau),
# so the magnitude that can leak across the boundary at distance d is <= exp(-d/tau); covering
# down to 1% needs span >= tau*ln(100) ~= 4.6*tau (see target_forward_span / zigzag tau scaling).
# This constant is the FLOOR (the empirically-sufficient purge at the default tau=5, which gives
# required_gap 2*40+5=85; see project_perslot_horizon_leak). The actual span scales UP with tau.
ZIGZAG_FORWARD_SPAN = 40

# Targets whose forward label spans exactly `horizon` bars (tail-NaN at `horizon`).
_HORIZON_TARGETS = frozenset({
    "forward_return",
    "forward_return_regression",
    "forward_return_penalized_regression",
    "forward_drawdown_regression",
    "continuation_entry_regression",
    "continuation_recov_entry_regression",
    "reversal_entry_regression",
    "risk_exit_regression",
    "swing_value_regression",  # asymmetric forward swing value over `horizon`
    "mfe_regression",
    "reward_risk_regression",
    "triple_barrier",  # horizon = vertical barrier
})


def target_forward_span(config: dict | None) -> int:
    """Max #bars into the future a target's label depends on — for sizing the train/test gap.

    The walk-forward splitter blocks forward-label leakage only via ``gap_days``; the gap
    must cover the LONGEST forward span of any target actually trained (per-slot targets
    included). Fail-loud on an unknown forward target so a newly added one can't silently
    under-gap the split (the bug that let exit risk_exit h40 / zigzag pivots leak).
    """
    if not config:
        return 0
    cfg = dict(config)
    ttype = cfg.get("type")
    if not ttype:
        return 0  # slot uses the global target; the caller sizes from that separately

    def _i(key: str, default: int = 0) -> int:
        try:
            return int(cfg.get(key, default) or default)
        except (TypeError, ValueError):
            return default

    if ttype in ("early_wave_v2", "early_wave_exit"):
        # Forward label spans `forward_window` bars (close[i+1 : i+1+fw]).
        return _i("forward_window", 21)
    if ttype in _HORIZON_TARGETS:
        return _i("horizon", 10)
    if ttype == "velocity_exit_regression":
        return max(_i("horizon", 10), _i("upside_horizon", 10))
    if ttype == "trend_scanning_exit":
        ws = cfg.get("windows")
        return max(int(x) for x in ws) if ws else _i("max_window", 20)
    if ttype == "multi_horizon_return":
        hs = cfg.get("horizons") or [10]
        return max(int(x) for x in hs)
    if ttype == "zigzag_pivot":
        # Confirmation distance is unbounded, but the leaked label magnitude at distance d is
        # bounded by exp(-d/tau); size the span to where that drops below 1% (tau*ln(100)),
        # never below the empirical floor. Fixed 40 silently under-gapped wide-tau labels
        # (verify: tau=20 leaks 0.135 at 40 bars; verify_label_boundary_leak.py).
        tau = float(cfg.get("tau", 5.0) or 5.0)
        decay_span = int(math.ceil(tau * math.log(100.0)))
        return max(ZIGZAG_FORWARD_SPAN, decay_span)
    if ttype in ("action_oracle", "amplitude_oracle"):
        # Hard 4/5-class swing labels (action_oracle.py) or the perfect-foresight optimal
        # long-only policy (amplitude_oracle.py). Forward dependence is the next swing,
        # unbounded in principle; reuse the zigzag-grade floor and rely on the train/test
        # gap (>= 85) to block boundary contamination.
        return ZIGZAG_FORWARD_SPAN
    if ttype == "downleg_depth_regression":
        # Pivot-anchored but the forward dependence is explicitly capped at max_span.
        return _i("max_span", 40)
    if ttype == "bottom_structure_entry_regression":
        # forward label spans both the early-advance horizon and the longer parked window.
        return max(_i("horizon", 8), _i("park_window", 20))
    if ttype == "trend_regime":
        return 0  # backward-only SMA crossover — no forward label
    raise KeyError(
        f"target_forward_span: unhandled target type {ttype!r}; declare its forward span so "
        "the leakage audit can size the train/test gap (fail-loud, no silent under-gap)."
    )


def build_target(config: dict) -> TargetProtocol:
    """Build a target by type and config.

    Args:
        config: dict with 'type' key + type-specific kwargs
            forward_return: {type, horizon, gain_threshold, loss_threshold}
            trend_regime: {type, short_window, long_window}

    Returns:
        Target instance implementing TargetProtocol

    Raises:
        KeyError: if config['type'] not in registry
        TypeError: if required kwargs missing
    """
    config = dict(config)
    target_type = config.pop("type")

    if target_type not in _REGISTRY:
        raise KeyError(f"Unknown target type: {target_type}. Available: {sorted(_REGISTRY.keys())}")

    # Filter config to only include parameters that target accepts
    target_class = _REGISTRY[target_type]
    import inspect

    sig = inspect.signature(target_class.__init__)
    valid_params = set(sig.parameters.keys()) - {"self"}
    filtered_config = {k: v for k, v in config.items() if k in valid_params}
    return target_class(**filtered_config)

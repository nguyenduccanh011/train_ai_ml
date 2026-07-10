"""Regime gating and size scaling — applied on top of any sizing policy.

Both helpers operate on a per-date weight Series (indexed by symbol). They are the
concrete wiring point for the previously-dead `regime_model` / `size_model` slots.
"""

from __future__ import annotations

import pandas as pd


def _lookup(signal: pd.DataFrame | None, date, value_col: str) -> pd.Series | None:
    """Extract a {symbol: value} series for `date` from a [date, symbol, value] frame."""
    if signal is None or signal.empty or value_col not in signal.columns:
        return None
    rows = signal[signal["date"] == date]
    if rows.empty:
        return None
    return rows.set_index("symbol")[value_col]


def apply_regime_gate(weights: pd.Series, regime_signal: pd.DataFrame | None, date) -> tuple[pd.Series, pd.Series]:
    """Zero out names whose regime gate == 0 for this date.

    Returns (gated_weights, gated_mask) where gated_mask[symbol] is True if the name was
    forced flat by the regime. Names absent from the regime signal are left untouched.
    """
    gated_mask = pd.Series(False, index=weights.index)
    gate = _lookup(regime_signal, date, "gate")
    if gate is None:
        return weights, gated_mask
    aligned = gate.reindex(weights.index)
    blocked = aligned == 0  # NaN (unknown) → not blocked
    gated_mask = blocked.fillna(False)
    return weights.where(~gated_mask, 0.0), gated_mask


def apply_size_scale(weights: pd.Series, size_signal: pd.DataFrame | None, date) -> pd.Series:
    """Multiply weights by a per-name size score (default 1.0 when absent)."""
    size = _lookup(size_signal, date, "size_score")
    if size is None:
        return weights
    aligned = size.reindex(weights.index).fillna(1.0)
    return weights * aligned

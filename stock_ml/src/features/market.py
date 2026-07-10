"""Market index construction for regime (``$market_*``) features.

When no external benchmark series is configured, professional desks fall back to
an index built from the tradable universe itself. We use an **equal-weight return
index**: the per-date mean of single-name daily returns, compounded into a price
series. It is deterministic from the OHLCV passed in (so it folds cleanly into the
feature store's content fingerprint) and leakage-safe (each day uses only that
day's and prior returns).

This is an explicit, documented construction — not a silent zero-fill. The old
builder passed ``market_index=None`` and emitted constant-0 regime features; this
restores real market-regime signal.
"""

from __future__ import annotations

import pandas as pd

_INDEX_BASE = 100.0


def build_equal_weight_index(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight market index from a multi-symbol OHLCV frame.

    Args:
        ohlcv: frame with at least [symbol, date, close].

    Returns:
        DataFrame [date, close] — one row per date, ``close`` the index level.
    """
    required = {"symbol", "date", "close"}
    if not required.issubset(ohlcv.columns):
        raise ValueError(f"build_equal_weight_index needs columns {sorted(required)}")

    tmp = ohlcv.loc[:, ["symbol", "date", "close"]].sort_values(["symbol", "date"])
    tmp = tmp.assign(ret=tmp.groupby("symbol", sort=False)["close"].pct_change())
    mkt_ret = tmp.groupby("date")["ret"].mean().sort_index()
    level = (1.0 + mkt_ret.fillna(0.0)).cumprod() * _INDEX_BASE
    return pd.DataFrame({"date": level.index, "close": level.to_numpy()})

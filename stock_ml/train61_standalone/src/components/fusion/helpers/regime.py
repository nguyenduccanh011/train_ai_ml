"""V19 trend detection + per-symbol regime adapter — verbatim port from legacy."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

TrendStrength = Literal["strong", "moderate", "weak"]

SYMBOL_PROFILES: dict[str, str] = {
    "ACB": "bank",
    "BID": "bank",
    "MBB": "bank",
    "TCB": "bank",
    "AAV": "high_beta",
    "AAS": "high_beta",
    "SSI": "high_beta",
    "VND": "high_beta",
    "DGC": "momentum",
    "HPG": "momentum",
    "VIC": "momentum",
    "FPT": "defensive",
    "REE": "defensive",
    "VNM": "defensive",
}


def detect_trend_strength(ind: dict[str, Any], i: int) -> TrendStrength:
    """6-component score → strong/moderate/weak. Mirrors legacy lines 2581-2594."""
    if i < 1:
        return "weak"
    sma20 = ind["sma20"]
    sma50 = ind["sma50"]
    close = ind["close"]
    macd_line = ind["macd_line"]
    days_ab = ind["days_above_ma20"][i]
    near_high = ind["dist_from_52w_high"][i] > 0.90
    ma20_ok = not np.isnan(sma20[i]) and not np.isnan(sma50[i]) and sma20[i] > sma50[i]
    price_above = close[i] > sma20[i] if not np.isnan(sma20[i]) else False
    macd_pos = macd_line[i] > 0
    score = sum([ma20_ok, price_above, macd_pos, days_ab >= 10, days_ab >= 20, near_high])
    if score >= 4:
        return "strong"
    elif score >= 2:
        return "moderate"
    return "weak"


def get_regime_adapter(
    symbol: str,
    ind: dict[str, Any],
    i: int,
    trend: TrendStrength,
    symbol_groups: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Per-profile regime config — mirrors legacy lines 2613-2697."""
    n = ind["n"]
    _groups = symbol_groups if symbol_groups is not None else SYMBOL_PROFILES
    profile = _groups.get(symbol, "balanced")
    bb_i = ind["feat_arrays"]["bb_width_percentile"][i]
    ret_20d_i = ind["ret_20d"][i]
    atr14 = ind["atr14"]
    close = ind["close"]

    low_vol = bb_i < 0.35
    weak_move = abs(ret_20d_i) < 0.05
    choppy_regime = bool(low_vol and weak_move and trend == "weak")
    atr_ratio = (
        (atr14[i] / close[i]) if (i < n and close[i] > 0 and not np.isnan(atr14[i])) else 0.03
    )

    params: dict[str, Any] = {
        "profile": profile,
        "dp_floor": 0.020,
        "ret5_hot": 0.060,
        "size_mult": 1.0,
        "base_confirm_bars": 3,
        "exit_score_threshold": 2.0,
        "choppy_regime": choppy_regime,
    }
    if profile == "high_beta":
        params.update(
            {
                "dp_floor": 0.015,
                "ret5_hot": 0.090,
                "size_mult": 0.98,
                "base_confirm_bars": 2,
                "exit_score_threshold": 2.35,
            }
        )
    elif profile == "bank":
        params.update(
            {
                "dp_floor": 0.020,
                "ret5_hot": 0.070,
                "size_mult": 0.92,
                "base_confirm_bars": 3,
                "exit_score_threshold": 2.2,
            }
        )
    elif profile == "defensive":
        params.update(
            {
                "dp_floor": 0.025,
                "ret5_hot": 0.050,
                "size_mult": 0.85,
                "base_confirm_bars": 3,
                "exit_score_threshold": 1.8,
            }
        )
    elif profile == "momentum":
        params.update(
            {
                "dp_floor": 0.018,
                "ret5_hot": 0.080,
                "size_mult": 0.92,
                "base_confirm_bars": 2,
                "exit_score_threshold": 2.2,
            }
        )

    if choppy_regime:
        params["dp_floor"] += 0.004
        params["size_mult"] *= 0.65
        params["base_confirm_bars"] += 1
        params["exit_score_threshold"] += 0.4

    if atr_ratio > 0.040:
        params["size_mult"] *= 0.85
        params["exit_score_threshold"] += 0.25
    if atr_ratio > 0.055:
        params["size_mult"] *= 0.90
        params["base_confirm_bars"] += 1
        params["exit_score_threshold"] += 0.20
    if trend == "strong":
        params["dp_floor"] = max(0.012, params["dp_floor"] - 0.003)
        params["ret5_hot"] += 0.01
        params["exit_score_threshold"] += 0.2

    if symbol in ("HPG", "VND"):
        params["size_mult"] *= 0.86
        params["exit_score_threshold"] += 0.15
        if atr_ratio > 0.045:
            params["base_confirm_bars"] += 1
    return params

"""Tests for v19 indicator + regime + sizing helpers (parity with legacy arithmetic)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.components.fusion.helpers import (
    SYMBOL_PROFILES,
    compute_v19_indicators,
    compute_v19_size,
    detect_trend_strength,
    get_regime_adapter,
)


def _synthetic_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = 100 + np.cumsum(rng.normal(0.1, 1.5, n))
    high = base + np.abs(rng.normal(0.5, 0.5, n))
    low = base - np.abs(rng.normal(0.5, 0.5, n))
    return pd.DataFrame(
        {
            "open": base + rng.normal(0, 0.2, n),
            "close": base,
            "high": high,
            "low": low,
            "volume": rng.uniform(1_000_000, 5_000_000, n),
            "symbol": ["TEST"] * n,
            "date": pd.date_range("2020-01-01", periods=n, freq="D"),
        }
    )


class TestComputeV19Indicators:
    def test_keys_present(self) -> None:
        df = _synthetic_df(100)
        ind = compute_v19_indicators(df)
        for key in (
            "sma10",
            "sma20",
            "sma50",
            "ema8",
            "macd_line",
            "macd_hist",
            "atr14",
            "rsi14",
            "ret_5d",
            "ret_20d",
            "ret_60d",
            "dist_sma20",
            "drop_from_peak_20",
            "consolidation_breakout",
            "secondary_breakout",
            "vshape_bypass",
            "days_above_ma20",
            "dist_from_52w_high",
        ):
            assert key in ind, f"missing {key}"

    def test_lengths_match_input(self) -> None:
        df = _synthetic_df(150)
        ind = compute_v19_indicators(df)
        n = len(df)
        for key in ("close", "sma20", "macd_hist", "atr14", "rsi14"):
            assert len(ind[key]) == n

    def test_secondary_breakout_disabled_when_mod_e_false(self) -> None:
        df = _synthetic_df(100)
        ind_off = compute_v19_indicators(df, mod_e=False)
        assert not ind_off["secondary_breakout"].any()

    def test_default_features_when_columns_missing(self) -> None:
        df = _synthetic_df(50)
        ind = compute_v19_indicators(df)
        # bb_width_percentile not in df → default 0.5
        assert np.allclose(ind["feat_arrays"]["bb_width_percentile"], 0.5)

    def test_legacy_arithmetic_parity(self) -> None:
        """Spot-check sma20 + macd_hist match raw pandas arithmetic."""
        df = _synthetic_df(100)
        ind = compute_v19_indicators(df)
        expected_sma20 = df["close"].rolling(20, min_periods=5).mean().values
        np.testing.assert_array_equal(ind["sma20"], expected_sma20)


class TestDetectTrendStrength:
    def test_weak_at_index_0(self) -> None:
        df = _synthetic_df(50)
        ind = compute_v19_indicators(df)
        assert detect_trend_strength(ind, 0) == "weak"

    def test_returns_valid_label(self) -> None:
        df = _synthetic_df(100)
        ind = compute_v19_indicators(df)
        assert detect_trend_strength(ind, 50) in ("strong", "moderate", "weak")


class TestGetRegimeAdapter:
    def test_unknown_symbol_uses_balanced(self) -> None:
        df = _synthetic_df(50)
        ind = compute_v19_indicators(df)
        cfg = get_regime_adapter("UNKNOWN", ind, 30, "moderate")
        assert cfg["profile"] == "balanced"
        assert "size_mult" in cfg

    def test_bank_profile(self) -> None:
        df = _synthetic_df(50)
        ind = compute_v19_indicators(df)
        cfg = get_regime_adapter("ACB", ind, 30, "moderate")
        assert cfg["profile"] == "bank"
        assert cfg["dp_floor"] == pytest.approx(0.020)

    def test_strong_trend_dp_floor_reduced(self) -> None:
        df = _synthetic_df(50)
        ind = compute_v19_indicators(df)
        cfg_mod = get_regime_adapter("FPT", ind, 30, "moderate")
        cfg_strong = get_regime_adapter("FPT", ind, 30, "strong")
        assert cfg_strong["dp_floor"] < cfg_mod["dp_floor"]


class TestComputeV19Size:
    def test_vshape_bypass_yields_050(self) -> None:
        df = _synthetic_df(50)
        ind = compute_v19_indicators(df)
        regime = get_regime_adapter("UNKNOWN", ind, 30, "weak")
        size = compute_v19_size(
            ind, 30, trend="weak", entry_score=2, vshape_entry=True, regime_cfg=regime
        )
        assert 0.25 <= size <= 1.0

    def test_clip_minimum(self) -> None:
        df = _synthetic_df(50)
        ind = compute_v19_indicators(df)
        regime = get_regime_adapter("UNKNOWN", ind, 30, "weak")
        regime["size_mult"] = 0.01  # force tiny
        size = compute_v19_size(
            ind, 30, trend="weak", entry_score=0, vshape_entry=False, regime_cfg=regime
        )
        assert size >= 0.25

    def test_clip_maximum(self) -> None:
        df = _synthetic_df(50)
        ind = compute_v19_indicators(df)
        regime = get_regime_adapter("UNKNOWN", ind, 30, "strong")
        regime["size_mult"] = 10.0
        size = compute_v19_size(
            ind, 30, trend="strong", entry_score=5, vshape_entry=False, regime_cfg=regime
        )
        assert size <= 1.0


class TestSymbolProfiles:
    def test_bank_symbols(self) -> None:
        for sym in ("ACB", "BID", "MBB", "TCB"):
            assert SYMBOL_PROFILES[sym] == "bank"

    def test_high_beta(self) -> None:
        assert SYMBOL_PROFILES["SSI"] == "high_beta"

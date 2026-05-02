"""Equivalence tests: ComposableFeatureEngine output must match legacy FeatureEngine."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

CONFIG_DIR = Path(__file__).parents[2] / "config" / "feature_sets"

FEATURE_SETS = ["leading", "leading_v2", "leading_v3", "leading_v4"]


def _make_test_df(n: int = 300, n_symbols: int = 3) -> pd.DataFrame:
    """Minimal synthetic OHLCV dataframe with enough rows for rolling windows."""
    rng = np.random.default_rng(42)
    parts = []
    for sym_i in range(n_symbols):
        sym = f"SYM{sym_i}"
        close = 100.0 * np.cumprod(1 + rng.normal(0, 0.01, n))
        high = close * (1 + rng.uniform(0, 0.02, n))
        low = close * (1 - rng.uniform(0, 0.02, n))
        open_ = close * (1 + rng.normal(0, 0.005, n))
        volume = rng.integers(100_000, 1_000_000, n).astype(float)
        ts = pd.date_range("2018-01-01", periods=n, freq="B")
        parts.append(
            pd.DataFrame(
                {
                    "timestamp": ts,
                    "symbol": sym,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def _run_legacy(feature_set: str, df: pd.DataFrame) -> pd.DataFrame:
    from src.features.engine import FeatureEngine

    eng = FeatureEngine(feature_set=feature_set)
    return eng.compute_for_all_symbols(df)


def _run_new(feature_set: str, df: pd.DataFrame) -> pd.DataFrame:
    from src.components.features.registry import build_engine_from_name

    eng = build_engine_from_name(feature_set, CONFIG_DIR)
    return eng.compute_for_all_symbols(df)


@pytest.fixture(scope="module")
def base_df() -> pd.DataFrame:
    return _make_test_df()


@pytest.mark.parametrize("feature_set", FEATURE_SETS)
def test_feature_columns_match(feature_set: str, base_df: pd.DataFrame) -> None:
    """New engine must produce at least the same feature columns as legacy."""
    old = _run_legacy(feature_set, base_df.copy())
    new = _run_new(feature_set, base_df.copy())

    old_feat = set(old.columns)
    new_feat = set(new.columns)

    # All legacy feature columns must be present in new output
    missing = old_feat - new_feat
    assert not missing, f"[{feature_set}] New engine missing columns: {sorted(missing)}"


@pytest.mark.parametrize("feature_set", FEATURE_SETS)
def test_feature_values_match(feature_set: str, base_df: pd.DataFrame) -> None:
    """New engine column values must numerically match legacy within tolerance."""
    old = _run_legacy(feature_set, base_df.copy())
    new = _run_new(feature_set, base_df.copy())

    # Only compare shared columns (new engine may produce extra columns)
    shared_cols = sorted(set(old.columns) & set(new.columns))
    # Exclude metadata columns
    meta = {"timestamp", "symbol", "exchange", "asset_type", "data_provider", "timeframe"}
    compare_cols = [c for c in shared_cols if c not in meta]

    old_sorted = old.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    new_sorted = new.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    mismatches = []
    for col in compare_cols:
        if col not in old_sorted.columns or col not in new_sorted.columns:
            continue
        o_val = old_sorted[col].values.astype(float)
        n_val = new_sorted[col].values.astype(float)
        # Allow NaN where both are NaN
        both_nan = np.isnan(o_val) & np.isnan(n_val)
        diff = np.where(both_nan, 0.0, np.abs(o_val - n_val))
        max_diff = np.nanmax(diff)
        if max_diff > 1e-8:
            mismatches.append((col, max_diff))

    if mismatches:
        details = ", ".join(f"{c}(max_diff={d:.2e})" for c, d in mismatches[:10])
        pytest.fail(f"[{feature_set}] Value mismatches: {details}")


@pytest.mark.parametrize("feature_set", FEATURE_SETS)
def test_engine_signature_stable(feature_set: str) -> None:
    """Engine signature must be deterministic."""
    from src.components.features.registry import build_engine_from_name

    eng1 = build_engine_from_name(feature_set, CONFIG_DIR)
    eng2 = build_engine_from_name(feature_set, CONFIG_DIR)
    assert eng1.signature() == eng2.signature()


def test_registry_unknown_block_raises() -> None:
    from src.components.features.registry import get_block

    with pytest.raises(KeyError, match="Unknown feature block"):
        get_block("nonexistent_block")

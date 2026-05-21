"""Leakage invariant tests: future data perturbation must not affect past labels/features.

Strategy:
1. Generate target/feature on baseline df.
2. Perturb rows after split_idx in a copy of df.
3. Compute target/feature on the perturbed df.
4. Assert rows BEFORE (split_idx - max_lookahead) are byte-identical between baseline and perturbed.
   Rows in [split_idx - max_lookahead, split_idx) are allowed to differ because the target
   intentionally looks forward by `forward_window` bars.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.data.target import TargetGenerator
from src.features.engine import FeatureEngine


def _make_df(n: int = 400, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1 + rng.normal(0, 0.01, n))
    high = close * (1 + rng.uniform(0, 0.02, n))
    low = close * (1 - rng.uniform(0, 0.02, n))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    volume = rng.integers(100_000, 1_000_000, n).astype(float)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC"),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "symbol": "TEST",
        }
    )


# ── Target leakage tests ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "target_type,kwargs,forward_window",
    [
        ("trend_regime", {}, 1),  # shift(-1)
        ("early_wave", {"forward_window": 10}, 11),  # fw forward + shift(-1)
        ("early_wave_v2", {"forward_window": 10}, 11),
        ("forward_risk_reward", {"forward_window": 10}, 11),
    ],
)
def test_target_no_leakage_from_future(target_type: str, kwargs: dict, forward_window: int) -> None:
    """Perturbing rows >= split_idx must not change target[:split_idx - forward_window]."""
    df = _make_df()
    split_idx = 200

    gen = TargetGenerator(target_type=target_type, **kwargs)
    baseline = gen.generate(df.copy())

    df_perturbed = df.copy()
    df_perturbed.loc[split_idx:, ["close", "high", "low", "open"]] *= 1.5
    perturbed = gen.generate(df_perturbed)

    safe_idx = split_idx - forward_window
    base_target = baseline["target"].iloc[:safe_idx].values
    pert_target = perturbed["target"].iloc[:safe_idx].values

    base_nan = np.isnan(base_target.astype(float))
    pert_nan = np.isnan(pert_target.astype(float))
    np.testing.assert_array_equal(base_nan, pert_nan)

    valid = ~base_nan
    np.testing.assert_array_equal(base_target[valid], pert_target[valid])


def test_target_sell_no_leakage_from_future() -> None:
    """target_sell uses forward_window lookahead; past labels must be invariant."""
    df = _make_df()
    split_idx = 200
    forward_window = 15

    df_with_sell = TargetGenerator.generate_exit_labels(
        df.copy(), forward_window=forward_window, loss_threshold=0.05
    )
    df_perturbed = df.copy()
    df_perturbed.loc[split_idx:, ["close", "high", "low", "open"]] *= 1.5
    perturbed_with_sell = TargetGenerator.generate_exit_labels(
        df_perturbed, forward_window=forward_window, loss_threshold=0.05
    )

    safe_idx = split_idx - forward_window - 1
    base_sell = df_with_sell["target_sell"].iloc[:safe_idx].values
    pert_sell = perturbed_with_sell["target_sell"].iloc[:safe_idx].values
    np.testing.assert_array_equal(base_sell, pert_sell)


# ── Feature leakage tests ──────────────────────────────────────────────────────


def test_tail_rows_preserved_for_inference() -> None:
    """Inference path must keep the last bars, even when labels are NaN."""
    df = _make_df(n=128)
    fw = 12

    gen = TargetGenerator(target_type="early_wave_dual", forward_window=fw)
    labeled = gen.generate_for_all_symbols(df.copy(), drop_na=False)

    assert len(labeled) == len(df)
    tail = labeled.tail(fw + 1)
    assert tail["target"].isna().any()
    assert tail["target_sell"].isna().any()

    sell_only = TargetGenerator.generate_exit_labels(
        df.copy(), forward_window=fw, loss_threshold=0.05, drop_na=False
    )
    assert len(sell_only) == len(df)
    assert sell_only["target_sell"].tail(fw + 1).isna().any()


def test_features_no_leakage_from_future() -> None:
    """Features at row i must depend only on data[:i] (no forward window)."""
    df = _make_df()
    split_idx = 250

    engine = FeatureEngine(feature_set="leading_v2")
    baseline = engine.compute(df.copy())

    df_perturbed = df.copy()
    df_perturbed.loc[split_idx:, ["close", "high", "low", "open", "volume"]] *= 1.5
    perturbed = engine.compute(df_perturbed)

    feature_cols = [
        c
        for c in baseline.columns
        if c not in {"timestamp", "open", "high", "low", "close", "volume", "symbol", "target"}
    ]

    leaking_features: list[str] = []
    for col in feature_cols:
        b = baseline[col].iloc[:split_idx].values
        p = perturbed[col].iloc[:split_idx].values
        b_nan = pd.isna(b)
        p_nan = pd.isna(p)
        if not np.array_equal(b_nan, p_nan):
            leaking_features.append(f"{col} (NaN mask differs)")
            continue
        valid = ~b_nan
        if not np.allclose(b[valid].astype(float), p[valid].astype(float), rtol=1e-9, atol=1e-9):
            leaking_features.append(col)

    assert not leaking_features, f"Features leak future data into past rows: {leaking_features}"


# ── Walk-forward split leakage ─────────────────────────────────────────────────


def test_walk_forward_train_test_disjoint() -> None:
    """Train rows must end strictly before test rows in every fold."""
    from src.data.splitter import WalkForwardSplitter

    df = _make_df(n=2000)
    df["timestamp"] = pd.date_range("2015-01-01", periods=len(df), freq="D", tz="UTC")
    splitter = WalkForwardSplitter(
        method="walk_forward",
        train_years=4,
        test_years=1,
        gap_days=0,
        first_test_year=2019,
        last_test_year=2020,
    )

    folds = list(splitter.split(df))
    assert folds, "splitter produced no folds"

    for window, train_df, test_df in folds:
        assert train_df["timestamp"].max() < test_df["timestamp"].min(), (
            f"fold {window.label}: train_end {train_df['timestamp'].max()} "
            f">= test_start {test_df['timestamp'].min()}"
        )

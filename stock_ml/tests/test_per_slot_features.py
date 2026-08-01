"""Test per-slot feature sets and targets (Phase 0.4)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from stock_ml.src.data.splitter import YearSplitter  # noqa: E402
from stock_ml.src.features.resolver import FeatureResolver  # noqa: E402
from stock_ml.src.pipeline.experiment import ExperimentConfig, train_fold  # noqa: E402
from stock_ml.src.targets.forward_regression import ForwardReturnRegressionTarget  # noqa: E402


def _synthetic_ohlcv(symbols: list[str], start: str, end: str, seed: int = 0) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, end=end)
    frames = []
    for s, sym in enumerate(symbols):
        drift = 0.0003 + 0.0001 * s
        vol = 0.02
        rets = rng.normal(drift, vol, size=len(dates))
        close = 100.0 * np.exp(np.cumsum(rets))
        opn = close * (1.0 + rng.normal(0.0, 0.002, size=len(dates)))
        high = np.maximum(opn, close) * (1.0 + np.abs(rng.normal(0.0, 0.004, size=len(dates))))
        low = np.minimum(opn, close) * (1.0 - np.abs(rng.normal(0.0, 0.004, size=len(dates))))
        volume = rng.integers(1_000_00, 1_000_000, size=len(dates))
        frames.append(
            pd.DataFrame(
                {
                    "symbol": sym,
                    "date": dates,
                    "open": opn,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_resolver_unions_per_slot_sets():
    """The resolver materializes the union of entry+exit set members (no duplication)."""
    bars = _synthetic_ohlcv(["A", "B"], "2019-01-01", "2020-12-31")
    resolver = FeatureResolver.from_catalog()
    feat, cols, _hits = resolver.resolve(bars, ["leading_v2", "basic_v1"], data_root="t")
    assert set(cols["basic_v1"]).issubset(set(cols["leading_v2"]))
    feature_cols = [
        c
        for c in feat.columns
        if c not in {"symbol", "date", "open", "high", "low", "close", "volume"}
    ]
    assert len(feature_cols) == 37  # union == leading_v2, basic_v1 shares its features


def test_train_fold_per_slot_different_targets():
    """Test train_fold with per-slot feature columns and target columns."""
    # Generate synthetic data (2 symbols so the model has cross-sectional spread).
    # Start well before the splitter's train window (2019+) so leading_v2's longest
    # lookback (dist_52w = 252 bars) is warm inside it — burn-in buffer, production-
    # style. The YearSplitter then slices to warm [2019,2021)/[2021,2022) windows, so
    # no row needs dropping.
    bars = _synthetic_ohlcv(["A", "B"], "2017-06-01", "2021-12-31")

    # Resolve per-slot feature columns via the DSL feature store
    resolver = FeatureResolver.from_catalog()
    feat, cols, _hits = resolver.resolve(bars, ["leading_v2"], data_root="t")
    entry_feat_cols = cols["leading_v2"]
    exit_feat_cols = cols["leading_v2"]

    # Apply per-slot targets
    entry_target = ForwardReturnRegressionTarget(horizon=7)
    feat["target_entry"] = entry_target.apply(feat.copy())["target"]

    exit_target = ForwardReturnRegressionTarget(horizon=3)
    feat["target_exit"] = exit_target.apply(feat.copy())["target"]

    # Backward compat: global target = entry target
    feat["target"] = feat["target_entry"]

    # Split and train
    sp = YearSplitter(
        train_years=2, test_years=1, gap_days=25, first_test_year=2021, last_test_year=2021
    )
    for _, train_df, test_df in sp.split(feat):
        cfg = ExperimentConfig(
            name="test_per_slot",
            strategy="ml_only_regression",
            market="vn_stock",
            feature_set="leading_v3",
            target={"type": "forward_return_regression", "horizon": 5},
            entry_model={"type": "lightgbm", "params": {"n_estimators": 10, "num_leaves": 4}},
            exit_model={"type": "none", "enabled": False, "params": {}},
            split={},
            engine={},
            seed=42,
        )

        # Call train_fold with per-slot target columns
        entry_model, exit_model, signals = train_fold(
            train_df,
            test_df,
            entry_feat_cols,
            cfg,
            exit_feat_cols=exit_feat_cols,
            entry_target_col="target_entry",
            exit_target_col="target_exit",
        )

        assert not signals.empty, "signals should not be empty"
        assert set(signals.columns).issuperset({"symbol", "date", "signal", "score"})
        assert len(signals) > 0, "should have generated signals"
        break  # only test first fold

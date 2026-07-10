"""predict_slot_signals (serving predict-only) must match train_fold's dual-ML output.

Proves the serving predict path reproduces backtest signal generation bit-for-bit
(before the aggregate recombine), so a bundle's models produce the same per-bar
score / exit_score / raw signal that the walk-forward fold did.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "stock_ml"))

from src.features.catalog import set_members  # noqa: E402
from src.features.resolver import add_features  # noqa: E402
from src.pipeline.experiment import ExperimentConfig, predict_slot_signals, train_fold  # noqa: E402
from src.targets.forward_regression import ForwardReturnRegressionTarget  # noqa: E402

FEATURE_COLS = set_members("basic_v1")


def _synthetic_ohlcv(symbols, start, end, seed=3):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, end=end)
    frames = []
    for s, sym in enumerate(symbols):
        rets = rng.normal(0.0004 + 0.0001 * s, 0.02, size=len(dates))
        close = 100.0 * np.exp(np.cumsum(rets))
        opn = close * (1.0 + rng.normal(0.0, 0.002, size=len(dates)))
        high = np.maximum(opn, close) * (1.0 + np.abs(rng.normal(0.0, 0.004, size=len(dates))))
        low = np.minimum(opn, close) * (1.0 - np.abs(rng.normal(0.0, 0.004, size=len(dates))))
        volume = rng.integers(100_000, 1_000_000, size=len(dates))
        frames.append(pd.DataFrame({
            "symbol": sym, "date": dates, "open": opn, "high": high,
            "low": low, "close": close, "volume": volume,
        }))
    return pd.concat(frames, ignore_index=True)


def _dual_ml_cfg() -> ExperimentConfig:
    return ExperimentConfig(
        name="dual_ml_predict_test",
        strategy="regression_dual_ml_recombine_decoupled",
        market="test",
        feature_set="basic_v1",
        target={"type": "forward_return_regression", "horizon": 5},
        entry_model={"type": "lightgbm", "params": {}},
        exit_model={"type": "lightgbm", "enabled": True, "params": {}},
        split={},
        engine={"exit_force_gate": "downleg12"},
        seed=42,
        signal_threshold=0.0,
        entry_threshold=-1.2,
        exit_threshold=0.07,
        entry_target={"type": "forward_return_regression", "horizon": 5},
        exit_target={"type": "forward_return_regression", "horizon": 10},
        direction="long",
    )


def _build_dual_data(symbols, buffer_start, win_start, split, end):
    bars = _synthetic_ohlcv(symbols, buffer_start, end)
    feat = add_features(bars)
    feat["target_entry"] = ForwardReturnRegressionTarget(horizon=5).apply(feat.copy())["target"]
    feat["target_exit"] = ForwardReturnRegressionTarget(horizon=10).apply(feat.copy())["target"]
    # drop warmup head so require_no_nan in train_fold doesn't trip on feature warmup
    feat = feat.groupby("symbol", group_keys=False).apply(lambda g: g.iloc[30:])
    train = feat[(feat["date"] >= win_start) & (feat["date"] < split)]
    test = feat[(feat["date"] >= split) & (feat["date"] < end)]
    return train, test


def test_predict_slot_matches_train_fold():
    cfg = _dual_ml_cfg()
    train, test = _build_dual_data(
        ["X", "Y"], "2019-09-01", "2020-01-01", "2020-06-01", "2020-12-31"
    )

    entry_model, exit_model, sig_tf = train_fold(
        train, test, FEATURE_COLS, cfg,
        entry_target_col="target_entry", exit_target_col="target_exit",
    )
    assert exit_model is not None, "dual-ML branch should build an exit model"

    sig_ps = predict_slot_signals(entry_model, exit_model, test, FEATURE_COLS, cfg)

    # same rows, same order
    sig_tf = sig_tf.sort_values(["symbol", "date"]).reset_index(drop=True)
    sig_ps = sig_ps.sort_values(["symbol", "date"]).reset_index(drop=True)

    assert list(sig_ps["signal"]) == list(sig_tf["signal"])
    assert np.allclose(sig_ps["score"], sig_tf["score"])
    assert np.allclose(sig_ps["exit_score"], sig_tf["exit_score"])
    # OHLCV carried for the downstream downleg force-gate
    assert {"open", "close", "high", "low"}.issubset(sig_ps.columns)


def test_predict_slot_matches_train_fold_3head():
    """3-head (entry2/entry3 ensemble) serving predict must match train_fold's score2/score3.

    Regression guard for the bundle/serving 3-head fix: a 3-head champion (n2_3h_*) trained
    with out_models exposes entry2/entry3; predict_slot_signals must reproduce score2/score3
    bit-for-bit so the recombine's union buys fire as in the backtest (not silently dropped).
    """
    cfg = _dual_ml_cfg()
    # Two extra entry heads on DIFFERENT-horizon targets (stand-ins for reversal/continuation).
    cfg.engine = {
        "exit_force_gate": "downleg12",
        "entry_ensemble": {"target": {"type": "forward_return_regression", "horizon": 8}, "z_threshold": 0.9},
        "entry_ensemble2": {"target": {"type": "forward_return_regression", "horizon": 3}, "z_threshold": 0.7},
    }
    train, test = _build_dual_data(
        ["X", "Y"], "2019-09-01", "2020-01-01", "2020-06-01", "2020-12-31"
    )
    train = train.copy()
    train["target_entry2"] = ForwardReturnRegressionTarget(horizon=8).apply(train.copy())["target"]
    train["target_entry3"] = ForwardReturnRegressionTarget(horizon=3).apply(train.copy())["target"]

    ensemble: dict = {}
    entry_model, exit_model, sig_tf = train_fold(
        train, test, FEATURE_COLS, cfg,
        entry_target_col="target_entry", exit_target_col="target_exit",
        entry2_target_col="target_entry2", entry3_target_col="target_entry3",
        out_models=ensemble,
    )
    assert {"entry2", "entry3"}.issubset(ensemble), "train_fold should expose ensemble models"
    assert "score2" in sig_tf.columns and "score3" in sig_tf.columns

    sig_ps = predict_slot_signals(
        entry_model, exit_model, test, FEATURE_COLS, cfg,
        entry_ensemble=[
            ("score2", ensemble["entry2"], None),
            ("score3", ensemble["entry3"], None),
        ],
    )
    sig_tf = sig_tf.sort_values(["symbol", "date"]).reset_index(drop=True)
    sig_ps = sig_ps.sort_values(["symbol", "date"]).reset_index(drop=True)

    assert "score2" in sig_ps.columns and "score3" in sig_ps.columns
    assert np.allclose(sig_ps["score2"], sig_tf["score2"])
    assert np.allclose(sig_ps["score3"], sig_tf["score3"])
    assert list(sig_ps["signal"]) == list(sig_tf["signal"])

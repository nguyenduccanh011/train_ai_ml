"""Tests for live simulator — strict, no lookahead, no hidden errors."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "stock_ml"))

from src.backtest.engine import CostModel, EngineConfig  # noqa: E402
from src.live_sim.config import LiveSimConfig  # noqa: E402
from src.live_sim.executor import EntryExecutor, ExitEvaluator  # noqa: E402
from src.live_sim.signals import FrozenSignalSet, SignalGenerator  # noqa: E402
from src.live_sim.state import Position, SimState  # noqa: E402
from src.models.baseline import BaselineModel  # noqa: E402


def _synthetic_ohlcv(symbols: list[str], start: str, end: str, seed: int = 0) -> pd.DataFrame:
    """Generate synthetic OHLCV data."""
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
        volume = rng.integers(100_000, 1_000_000, size=len(dates))
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


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal feature building for test."""
    from src.features.basic import add_features

    return add_features(df)


def test_frozen_signal_set_integrity_hash():
    """FrozenSignalSet should have valid integrity_hash."""
    frozen = FrozenSignalSet(
        generated_at=pd.Timestamp("2025-01-02"),
        for_execution_date=pd.Timestamp("2025-01-03"),
        signals={"AAA": 1, "BBB": -1},
        n_buy=1,
        n_sell=1,
        n_neutral=0,
        filters_applied=[],
        integrity_hash="abc123",
    )

    assert len(frozen.integrity_hash) > 0
    assert frozen.signals == {"AAA": 1, "BBB": -1}


def test_signal_generator_detects_lookahead():
    """SignalGenerator must reject history_feat with date > yesterday."""
    cfg = LiveSimConfig(
        data_root="/tmp",
        symbols=["AAA"],
        out_dir="/tmp",
        sim_start="2025-01-02",
        sim_end="2025-01-31",
    )

    model = BaselineModel(seed=42)
    X_dummy = np.random.randn(10, 8).astype(np.float32)
    y_dummy = np.array([0, 1, -1] * 3 + [0]).astype(np.float64)
    model.fit(X_dummy, y_dummy)

    # History with lookahead: contains future data
    yesterday = pd.Timestamp("2025-01-02")
    today = pd.Timestamp("2025-01-03")
    bad_feat = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA"],
            "date": [pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-03")],  # oops, has today
            "close": [100.0, 101.0],
        }
    )

    sig_gen = SignalGenerator(model, cfg)
    with pytest.raises(ValueError, match="lookahead"):
        sig_gen.generate(yesterday, today, bad_feat)


def test_signal_generator_fails_on_nan_features():
    """SignalGenerator must reject NaN features."""
    from src.features.basic import FEATURE_COLS

    cfg = LiveSimConfig(
        data_root="/tmp",
        symbols=["AAA"],
        out_dir="/tmp",
        sim_start="2025-01-02",
        sim_end="2025-01-31",
    )

    model = BaselineModel(seed=42)
    X_dummy = np.random.randn(10, 8).astype(np.float32)
    y_dummy = np.array([0, 1, -1] * 3 + [0]).astype(np.float64)
    model.fit(X_dummy, y_dummy)

    yesterday = pd.Timestamp("2025-01-02")
    today = pd.Timestamp("2025-01-03")

    # Features with NaN values
    bad_feat_dict = {
        "symbol": ["AAA"],
        "date": [yesterday.date()],  # Use .date() to match what code expects
    }
    for col in FEATURE_COLS:
        bad_feat_dict[col] = [np.nan]  # All features are NaN

    bad_feat = pd.DataFrame(bad_feat_dict)

    sig_gen = SignalGenerator(model, cfg)
    with pytest.raises(ValueError, match="NaN features"):
        sig_gen.generate(yesterday, today, bad_feat)


def test_entry_executor_respects_frozen_signals():
    """EntryExecutor must only execute signals from frozen set."""
    cost_model = CostModel(commission=0.0015, tax=0.001, slippage=0.001)
    engine_cfg = EngineConfig(max_hold_bars=20, hard_stop_pct=-0.08)
    executor = EntryExecutor(cost_model, engine_cfg)

    frozen = FrozenSignalSet(
        generated_at=pd.Timestamp("2025-01-02"),
        for_execution_date=pd.Timestamp("2025-01-03"),
        signals={"AAA": 1, "BBB": 0, "CCC": -1},  # only AAA should entry
        n_buy=1,
        n_sell=1,
        n_neutral=1,
        filters_applied=[],
        integrity_hash="abc",
    )

    bars = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC"],
            "date": [pd.Timestamp("2025-01-03")] * 3,
            "open": [100.0, 100.0, 100.0],
            "high": [101.0, 101.0, 101.0],
            "low": [99.0, 99.0, 99.0],
            "close": [100.5, 100.5, 100.5],
        }
    )

    state = SimState()
    entries = executor.execute(frozen, bars, state)

    assert len(entries) == 1
    assert entries[0].symbol == "AAA"


def test_entry_executor_fails_date_mismatch():
    """EntryExecutor must validate frozen.for_execution_date == bars date."""
    cost_model = CostModel()
    engine_cfg = EngineConfig()
    executor = EntryExecutor(cost_model, engine_cfg)

    frozen = FrozenSignalSet(
        generated_at=pd.Timestamp("2025-01-02"),
        for_execution_date=pd.Timestamp("2025-01-03"),
        signals={"AAA": 1},
        n_buy=1,
        n_sell=0,
        n_neutral=0,
        filters_applied=[],
        integrity_hash="abc",
    )

    # Wrong date in bars
    bars = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "date": [pd.Timestamp("2025-01-04")],  # mismatch!
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
        }
    )

    state = SimState()
    with pytest.raises(ValueError, match="for_execution_date.*!=.*bars_today date"):
        executor.execute(frozen, bars, state)


def test_exit_evaluator_priority_hard_stop_first():
    """ExitEvaluator must prioritize hard_stop > signal > max_hold."""
    cost_model = CostModel()
    engine_cfg = EngineConfig(hard_stop_pct=-0.05, max_hold_bars=20, min_hold_bars=0)
    evaluator = ExitEvaluator(cost_model, engine_cfg)

    pos = Position(
        symbol="AAA",
        entry_date=pd.Timestamp("2025-01-02"),
        entry_price=100.0,
        entry_signal_date=pd.Timestamp("2025-01-02"),
        hold_bars=1,
    )

    state = SimState()
    state.open_positions["AAA"] = pos

    # Bars with low dipping below hard_stop
    bars = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "date": [pd.Timestamp("2025-01-03")],
            "open": [94.0],
            "high": [95.0],
            "low": [94.0],  # 94/100 - 1 = -0.06 < -0.05
            "close": [95.0],
        }
    )

    frozen = FrozenSignalSet(
        generated_at=pd.Timestamp("2025-01-02"),
        for_execution_date=pd.Timestamp("2025-01-03"),
        signals={"AAA": 1},  # even if buy signal, hard_stop takes priority
        n_buy=1,
        n_sell=0,
        n_neutral=0,
        filters_applied=[],
        integrity_hash="abc",
    )

    exits = evaluator.evaluate(state, frozen, bars)
    assert len(exits) == 1
    assert exits[0].exit_reason == "hard_stop"


def test_exit_evaluator_respects_min_hold_bars():
    """ExitEvaluator must respect min_hold_bars before exiting."""
    cost_model = CostModel()
    engine_cfg = EngineConfig(
        hard_stop_pct=-0.08,
        max_hold_bars=20,
        min_hold_bars=2,  # must hold at least 2 bars
    )
    evaluator = ExitEvaluator(cost_model, engine_cfg)

    pos = Position(
        symbol="AAA",
        entry_date=pd.Timestamp("2025-01-02"),
        entry_price=100.0,
        entry_signal_date=pd.Timestamp("2025-01-02"),
        hold_bars=1,  # only held 1 bar
    )

    state = SimState()
    state.open_positions["AAA"] = pos

    bars = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "date": [pd.Timestamp("2025-01-03")],
            "open": [94.0],
            "high": [95.0],
            "low": [94.0],
            "close": [95.0],
        }
    )

    frozen = FrozenSignalSet(
        generated_at=pd.Timestamp("2025-01-02"),
        for_execution_date=pd.Timestamp("2025-01-03"),
        signals={"AAA": -1},  # sell signal
        n_buy=0,
        n_sell=1,
        n_neutral=0,
        filters_applied=[],
        integrity_hash="abc",
    )

    exits = evaluator.evaluate(state, frozen, bars)
    assert len(exits) == 0  # should NOT exit yet (min_hold not met)


def test_no_pyramiding():
    """EntryExecutor must not pyramid (no multiple positions per symbol)."""
    cost_model = CostModel()
    engine_cfg = EngineConfig()
    executor = EntryExecutor(cost_model, engine_cfg)

    frozen = FrozenSignalSet(
        generated_at=pd.Timestamp("2025-01-02"),
        for_execution_date=pd.Timestamp("2025-01-03"),
        signals={"AAA": 1},
        n_buy=1,
        n_sell=0,
        n_neutral=0,
        filters_applied=[],
        integrity_hash="abc",
    )

    bars = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "date": [pd.Timestamp("2025-01-03")],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
        }
    )

    state = SimState()
    state.open_position(
        Position(
            symbol="AAA",
            entry_date=pd.Timestamp("2025-01-02"),
            entry_price=100.0,
            entry_signal_date=pd.Timestamp("2025-01-01"),
            hold_bars=1,
        )
    )

    entries = executor.execute(frozen, bars, state)
    assert len(entries) == 0  # should skip AAA, already open


def test_sim_state_position_tracking():
    """SimState must track open positions correctly."""
    state = SimState()

    pos1 = Position(
        symbol="AAA",
        entry_date=pd.Timestamp("2025-01-02"),
        entry_price=100.0,
        entry_signal_date=pd.Timestamp("2025-01-02"),
    )

    state.open_position(pos1)
    assert state.n_open() == 1
    assert state.has_position("AAA")

    state.advance_holds()
    assert state.open_positions["AAA"].hold_bars == 1

    closed = state.close_position("AAA")
    assert closed.symbol == "AAA"
    assert state.n_open() == 0


def test_frozen_signal_set_buys_and_sells():
    """FrozenSignalSet.buys() and sells() must return correct subsets."""
    frozen = FrozenSignalSet(
        generated_at=pd.Timestamp("2025-01-02"),
        for_execution_date=pd.Timestamp("2025-01-03"),
        signals={"AAA": 1, "BBB": -1, "CCC": 0},
        n_buy=1,
        n_sell=1,
        n_neutral=1,
        filters_applied=[],
        integrity_hash="abc",
    )

    assert frozen.buys() == frozenset(["AAA"])
    assert frozen.sells() == frozenset(["BBB"])
    assert frozen.neutrals() == frozenset(["CCC"])

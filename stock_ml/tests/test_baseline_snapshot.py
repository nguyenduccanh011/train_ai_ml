"""Baseline snapshot harness (Phase 0 of the alpha/portfolio/execution refactor).

Locks the CURRENT behavior of the alpha -> (legacy per-symbol) execution path so
that later phases can prove they did not change it unintentionally.

Invariant guarded here:
- Phase 1 (continuous score + hysteresis with entry_threshold == exit_threshold ==
  signal_threshold) MUST reproduce this snapshot bit-for-bit.
- Phase 3 (new portfolio/execution engine) is EXPECTED to diverge; at that point this
  fixture becomes the "legacy golden" kept only for parity comparison until Phase 6.

Regenerate intentionally with:  REGEN_BASELINE=1 pytest stock_ml/tests/test_baseline_snapshot.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from stock_ml.src.backtest.engine import (  # noqa: E402
    EngineConfig,
    run_backtest,
    trades_to_dataframe,
)
from stock_ml.src.backtest.stats import aggregate_stats  # noqa: E402
from stock_ml.src.features.catalog import set_members  # noqa: E402
from stock_ml.src.features.resolver import add_features  # noqa: E402
from stock_ml.src.pipeline.experiment import ExperimentConfig, train_fold  # noqa: E402
from stock_ml.src.targets.forward_regression import ForwardReturnRegressionTarget  # noqa: E402

FEATURE_COLS = set_members("basic_v1")

FIXTURE = Path(__file__).parent / "fixtures" / "baseline_snapshot.json"
SYMBOLS = ["A", "B", "C", "D"]


def _synthetic_ohlcv(symbols: list[str], start: str, end: str, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, end=end)
    frames = []
    for s, sym in enumerate(symbols):
        drift = 0.0003 + 0.0001 * s
        rets = rng.normal(drift, 0.02, size=len(dates))
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


def _baseline_config() -> ExperimentConfig:
    return ExperimentConfig(
        name="baseline_snapshot",
        strategy="entry_exit",
        market="test",
        feature_set="basic_v1",
        target={"type": "forward_return_regression", "horizon": 5},
        entry_model={"type": "lightgbm", "params": {}},
        exit_model={"type": "none", "enabled": False},
        split={},
        engine={},
        seed=42,
        signal_threshold=0.0,
    )


def compute_snapshot() -> dict:
    """Deterministic alpha -> legacy execution run, reduced to a stable comparable dict."""
    # Burn-in buffer: start before the window so basic_v1's rolling features (max
    # lookback 20 bars) are warm inside it — production computes features on the full
    # continuous history. The window is then selected by date; no row is dropped as
    # NaN, so the snapshot reflects the full intended train window [2020-01-01,
    # 2021-06-01).
    bars = _synthetic_ohlcv(SYMBOLS, "2019-10-01", "2021-12-31", seed=7)
    feat = add_features(bars)
    data = ForwardReturnRegressionTarget(horizon=5).apply(feat)

    train = data[(data["date"] >= "2020-01-01") & (data["date"] < "2021-06-01")]
    test = data[data["date"] >= "2021-06-01"]

    _, _, signals = train_fold(train, test, FEATURE_COLS, _baseline_config())

    trades = run_backtest(signals, bars, EngineConfig(max_hold_bars=20, hard_stop_pct=-0.08))
    trades_df = trades_to_dataframe(trades)
    agg = aggregate_stats(trades_df)

    sig_counts = {str(int(k)): int(v) for k, v in signals["signal"].value_counts().items()}
    trade_rows = [
        {
            "symbol": str(r.symbol),
            "entry_date": pd.Timestamp(r.entry_date).date().isoformat(),
            "exit_date": pd.Timestamp(r.exit_date).date().isoformat(),
            "pnl_pct": round(float(r.pnl_pct), 8),
            "holding_days": int(r.holding_days),
        }
        for r in trades_df.sort_values(["symbol", "entry_date"]).itertuples(index=False)
    ]
    return {
        "signal_counts": sig_counts,
        "score_sum": round(float(signals["score"].sum()), 6),
        "score_mean": round(float(signals["score"].mean()), 8),
        "n_trades": len(trade_rows),
        "trades": trade_rows,
        "aggregate_stats": {k: round(float(v), 8) for k, v in agg.items()},
    }


def test_baseline_snapshot():
    snap = compute_snapshot()

    if os.environ.get("REGEN_BASELINE") or not FIXTURE.exists():
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_text(json.dumps(snap, indent=2, sort_keys=True), encoding="utf-8")
        if not os.environ.get("REGEN_BASELINE"):
            # First-ever run created the fixture; nothing to compare against yet.
            return

    golden = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert snap == golden, (
        "Baseline behavior changed unexpectedly. If this change is intentional "
        "(Phase 3+), regenerate with REGEN_BASELINE=1 and review the diff."
    )

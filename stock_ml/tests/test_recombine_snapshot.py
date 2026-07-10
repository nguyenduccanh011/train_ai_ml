"""Snapshot harness locking the dual-ML recombine behavior.

This is the regression safety-net for the upcoming Hướng-A refactor (extracting
``recombine_signals`` from ``run_experiment`` so serving can reuse it). The
extraction must keep ``_recombine_dual_ml_signals`` and the surrounding
parameter-prep byte-for-byte; this test proves the recombine output (which drives
the champion-958-style signals: canonical z-sum, decoupled bands, and the
``downleg12`` force-sell gate) is unchanged.

Regenerate intentionally with:
    REGEN_RECOMBINE=1 pytest stock_ml/tests/test_recombine_snapshot.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "stock_ml"))

from src.pipeline.experiment import (  # noqa: E402
    ExperimentConfig,
    _recombine_dual_ml_signals,
    recombine_signals,
)

FIXTURE = Path(__file__).parent / "fixtures" / "recombine_snapshot.json"
SYMBOLS = ["AAA", "BBB", "CCC"]
N_BARS = 320  # > 252 z-window so the trailing causal z is fully warm


def _synthetic_signals() -> pd.DataFrame:
    """Deterministic dual-ML signal frame: symbol/date + entry/exit head preds + OHLCV."""
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2020-01-01", periods=N_BARS)
    frames = []
    for s, sym in enumerate(SYMBOLS):
        rets = rng.normal(0.0004 + 0.0001 * s, 0.02, size=N_BARS)
        close = 100.0 * np.exp(np.cumsum(rets))
        opn = close * (1.0 + rng.normal(0.0, 0.003, size=N_BARS))
        high = np.maximum(opn, close) * (1.0 + np.abs(rng.normal(0.0, 0.005, size=N_BARS)))
        low = np.minimum(opn, close) * (1.0 - np.abs(rng.normal(0.0, 0.005, size=N_BARS)))
        frames.append(
            pd.DataFrame(
                {
                    "symbol": sym,
                    "date": dates,
                    # raw per-fold signal (recombine overwrites it; only read for a log count)
                    "signal": np.zeros(N_BARS, dtype=np.int8),
                    # entry head (continuation alpha) + exit head (downside risk) preds
                    "score": rng.normal(0.0, 1.0, size=N_BARS),
                    "exit_score": np.abs(rng.normal(0.0, 1.0, size=N_BARS)),
                    "score2": rng.normal(0.0, 1.0, size=N_BARS),
                    "open": opn,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": rng.integers(100_000, 1_000_000, size=N_BARS),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _reduce(signals: pd.DataFrame) -> dict:
    """Stable comparable summary of a recombined signal frame."""
    sig = signals.sort_values(["symbol", "date"])["signal"].to_numpy().astype(int)
    counts = {str(int(k)): int(v) for k, v in pd.Series(sig).value_counts().items()}
    seq_hash = hashlib.sha256(sig.tobytes()).hexdigest()[:16]
    return {"counts": counts, "sum": int(sig.sum()), "seq_sha": seq_hash}


def compute_snapshot() -> dict:
    sig = _synthetic_signals()

    # Config A — canonical recombine: buy on z-sum, sell on raw exit head.
    canonical = _recombine_dual_ml_signals(
        sig.copy(),
        sum_threshold=1.0,
        exit_threshold=0.8,
        window=252,
        min_periods=60,
        direction="long",
        use_raw_exit=True,
    )

    # Config B — champion-958-style: decoupled entry z-band, rising-risk exit z-band,
    # raw-exit off, plus the downleg12 force-sell backstop on the right slope.
    champion = _recombine_dual_ml_signals(
        sig.copy(),
        sum_threshold=1.0,
        exit_threshold=0.0,
        window=252,
        min_periods=60,
        direction="long",
        use_raw_exit=False,
        entry_z_threshold=1.0,
        exit_z_threshold=1.0,
        exit_force_gate="downleg12",
    )

    return {"canonical": _reduce(canonical), "champion_downleg": _reduce(champion)}


def test_recombine_snapshot():
    snap = compute_snapshot()

    if os.environ.get("REGEN_RECOMBINE") or not FIXTURE.exists():
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_text(json.dumps(snap, indent=2, sort_keys=True), encoding="utf-8")
        if not os.environ.get("REGEN_RECOMBINE"):
            return  # first run created the fixture; nothing to diff yet

    golden = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert snap == golden, (
        "Dual-ML recombine behavior changed. If intentional, regenerate with "
        "REGEN_RECOMBINE=1 and review the diff."
    )


def test_recombine_is_deterministic():
    """Same input twice → identical signals (no hidden randomness in recombine)."""
    a = compute_snapshot()
    b = compute_snapshot()
    assert a == b


def _decoupled_downleg_cfg() -> ExperimentConfig:
    """Config whose recombine_signals() param-prep must reproduce snapshot config B."""
    return ExperimentConfig(
        name="recombine_wrapper_test",
        strategy="regression_dual_ml_recombine_decoupled",
        market="test",
        feature_set="basic_v1",
        target={"type": "zigzag_pivot"},
        entry_model={"type": "lightgbm", "params": {}},
        exit_model={"type": "lightgbm", "enabled": True, "params": {}},
        split={},
        engine={"exit_force_gate": "downleg12"},
        seed=42,
        signal_threshold=1.0,
        entry_threshold=1.0,
        exit_threshold=0.0,
        direction="long",
    )


def test_recombine_signals_wrapper_matches_direct():
    """recombine_signals(sig, cfg) must equal the direct _recombine call it wraps.

    Proves the Hướng-A extraction preserved the cfg -> _recombine_dual_ml_signals
    parameter mapping (decoupled entry z-band + rising-risk exit z-band + downleg12).
    """
    sig = _synthetic_signals()

    via_wrapper = recombine_signals(sig.copy(), _decoupled_downleg_cfg())
    direct = _recombine_dual_ml_signals(
        sig.copy(),
        sum_threshold=1.0,
        exit_threshold=0.0,
        window=252,
        min_periods=60,
        direction="long",
        use_raw_exit=False,
        entry_z_threshold=1.0,
        exit_z_threshold=1.0,
        exit_force_gate="downleg12",
    )
    assert _reduce(via_wrapper) == _reduce(direct)

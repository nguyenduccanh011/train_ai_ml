"""Hysteresis-band signal generation tests (Phase 1 of the refactor).

Verifies the continuous-score → discrete-signal mapping with asymmetric
entry/exit thresholds, and that the default (no explicit thresholds) reproduces
the legacy symmetric `signal_threshold` rule exactly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "stock_ml"))

from src.signals.core import generate_signals_from_predictions  # noqa: E402


def _frame(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {"symbol": ["X"] * n, "date": pd.date_range("2020-01-01", periods=n, freq="B")}
    )


def test_default_thresholds_match_symmetric_rule():
    preds = np.array([0.05, -0.05, 0.005, -0.005, 0.0], dtype=np.float32)
    df = _frame(len(preds))

    # New code with default thresholds
    out = generate_signals_from_predictions(preds, df, signal_threshold=0.01)
    # Legacy symmetric rule: >+0.01 → 1, <-0.01 → -1, else 0
    expected = np.where(preds > 0.01, 1, np.where(preds < -0.01, -1, 0))
    assert (out["signal"].to_numpy() == expected).all()
    # Score is the continuous prediction, untouched
    assert np.allclose(out["score"].to_numpy(), preds)


def test_hysteresis_dead_band():
    # Asymmetric band: enter long only above 0.02, exit only below -0.01.
    preds = np.array([0.03, 0.015, 0.0, -0.005, -0.02], dtype=np.float32)
    df = _frame(len(preds))
    out = generate_signals_from_predictions(preds, df, entry_threshold=0.02, exit_threshold=-0.01)
    # 0.03 > 0.02 → 1 ; 0.015 in band → 0 ; 0.0 in band → 0 ;
    # -0.005 in band → 0 ; -0.02 < -0.01 → -1
    assert out["signal"].tolist() == [1, 0, 0, 0, -1]


def test_short_direction_flips():
    preds = np.array([0.05, -0.05], dtype=np.float32)
    df = _frame(len(preds))
    long_out = generate_signals_from_predictions(preds, df, signal_threshold=0.0)
    short_out = generate_signals_from_predictions(
        preds, df, signal_threshold=0.0, direction="short"
    )
    assert (short_out["signal"].to_numpy() == -long_out["signal"].to_numpy()).all()

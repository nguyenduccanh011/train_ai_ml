"""End-to-end regression for run_experiment (real market.duckdb, tiny scope).

This locks the FULL pipeline output (load → resolve features → targets → folds →
recombine → backtest) so the Hướng-A extractions (recombine_signals,
build_feature_frame, predict-only) provably do not change run_experiment's
behavior. Snapshots only deterministic integer counts (trades + buy/sell signals).

Skips automatically if market.duckdb is absent. Regenerate intentionally with:
    REGEN_E2E=1 pytest stock_ml/tests/test_run_experiment_e2e.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "stock_ml"))

from src.pipeline.experiment import ExperimentConfig, run_experiment  # noqa: E402

DUCKDB = REPO_ROOT / "market_data" / "market.duckdb"
FIXTURE = Path(__file__).parent / "fixtures" / "run_experiment_e2e.json"
SYMBOLS = ["AAA", "AAS", "AAT"]


def _cfg() -> ExperimentConfig:
    return ExperimentConfig(
        name="e2e_regression",
        strategy="entry_exit",  # ml_only path (not a recombine strategy)
        market="vn_stock",
        feature_set="basic_v1",
        target={"type": "forward_return_regression", "horizon": 5},
        entry_model={"type": "lightgbm", "params": {}},
        exit_model={"type": "none", "enabled": False},
        split={
            "type": "walk_forward_year",
            "train_years": 2,
            "test_years": 1,
            "gap_days": 25,
            "first_test_year": 2022,
            "last_test_year": 2023,
        },
        engine={"max_hold_bars": 20, "hard_stop_pct": -0.08},
        seed=42,
        signal_threshold=0.0,
        strict_audit=False,  # 3-symbol toy universe may trip audit; not what we test here
    )


def compute_snapshot() -> dict:
    with tempfile.TemporaryDirectory() as td:
        summary = run_experiment(
            _cfg(), str(DUCKDB), SYMBOLS, out_dir=td, run_id=None, export_csv=False
        )
    return {
        "n_trades": int(summary["n_trades"]),
        "n_signals_buy": int(summary["n_signals_buy"]),
        "n_signals_sell": int(summary["n_signals_sell"]),
        "n_symbols": int(summary["n_symbols"]),
    }


@pytest.mark.skipif(not DUCKDB.exists(), reason="market.duckdb not available")
def test_run_experiment_e2e_snapshot():
    snap = compute_snapshot()

    if os.environ.get("REGEN_E2E") or not FIXTURE.exists():
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_text(json.dumps(snap, indent=2, sort_keys=True), encoding="utf-8")
        if not os.environ.get("REGEN_E2E"):
            return

    golden = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert snap == golden, (
        "run_experiment behavior changed. If intentional, regenerate with "
        "REGEN_E2E=1 and review the diff."
    )

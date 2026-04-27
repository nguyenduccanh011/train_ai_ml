"""Regression: rule baseline qua FusionStack + Backtester phải match golden trades_rule.csv exact.

Khác `test_champions.py` (hash file đã regen từ `run_pipeline.py`), test này
tự chạy `run_rule_baseline()` từ component pipeline mới (Phase 2.3a) và so DataFrame
trực tiếp với golden CSV.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
GOLDEN_DIR = REPO_ROOT / "stock_ml" / "tests" / "regression" / "golden"
GOLDEN_CSV = GOLDEN_DIR / "trades_rule.csv"
GOLDEN_META = GOLDEN_DIR / "trades_rule.meta.json"


@pytest.mark.regression
def test_rule_baseline_matches_golden() -> None:
    from src.components.runners import run_rule_baseline, trades_to_dataframe
    from src.env import resolve_data_dir

    if not GOLDEN_CSV.exists() or not GOLDEN_META.exists():
        pytest.fail(f"Golden artefacts missing: {GOLDEN_CSV} / {GOLDEN_META}")

    meta = json.loads(GOLDEN_META.read_text())
    symbols: list[str] = meta["symbols"]
    expected_n = int(meta["n_trades"])

    data_dir = resolve_data_dir("../portable_data/vn_stock_ai_dataset_cleaned")
    if not Path(data_dir).is_dir():
        pytest.skip(f"Data dir không tồn tại: {data_dir}")

    trades = run_rule_baseline(symbols=symbols, data_dir=data_dir, first_test_year=2020)
    df_new = trades_to_dataframe(trades).reset_index(drop=True)

    df_gold = pd.read_csv(GOLDEN_CSV).reset_index(drop=True)

    assert len(df_new) == expected_n, (
        f"Trade count mismatch: new={len(df_new)} expected={expected_n}"
    )

    df_new["entry_date"] = df_new["entry_date"].astype(str)
    df_new["exit_date"] = df_new["exit_date"].astype(str)
    df_gold["entry_date"] = df_gold["entry_date"].astype(str)
    df_gold["exit_date"] = df_gold["exit_date"].astype(str)

    pd.testing.assert_frame_equal(df_new, df_gold, check_exact=True)

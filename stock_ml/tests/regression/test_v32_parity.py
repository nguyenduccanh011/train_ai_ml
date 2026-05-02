"""Regression: componentized v32 runner should match golden trades_v32.csv."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
GOLDEN_DIR = REPO_ROOT / "stock_ml" / "tests" / "regression" / "golden"
GOLDEN_CSV = GOLDEN_DIR / "trades_v32.csv"
GOLDEN_META = GOLDEN_DIR / "trades_v32.meta.json"


@pytest.mark.regression
def test_v32_matches_golden() -> None:
    from src.components.runners import run_v32, trades_to_v32_dataframe
    from src.env import resolve_data_dir
    from src.pipeline.build_predictions import _build_predictions

    if not GOLDEN_CSV.exists() or not GOLDEN_META.exists():
        pytest.fail(f"Golden artefacts missing: {GOLDEN_CSV} / {GOLDEN_META}")

    meta = json.loads(GOLDEN_META.read_text())
    symbols: list[str] = meta["symbols"]
    data_dir = resolve_data_dir("../portable_data/vn_stock_ai_dataset_cleaned")
    if not Path(data_dir).is_dir():
        pytest.skip(f"Data dir không tồn tại: {data_dir}")

    prediction_cache = _build_predictions(
        symbols,
        meta["feature_set"],
        meta["target_config"],
        "cpu",
        model_type=meta["model_type"],
        exit_model_cfg=meta["exit_model_config"],
    )
    trades = run_v32(
        symbols=symbols,
        data_dir=str(data_dir),
        prediction_cache=prediction_cache,
    )
    df_raw = trades_to_v32_dataframe(trades).reset_index(drop=True)
    df_new = pd.read_csv(io.StringIO(df_raw.to_csv(index=False))).reset_index(drop=True)
    df_gold = pd.read_csv(GOLDEN_CSV).reset_index(drop=True)

    assert len(df_new) == int(meta["n_trades"]), (
        f"Trade count mismatch: new={len(df_new)} expected={meta['n_trades']}"
    )

    df_new["max_profit_pct"] = pd.to_numeric(df_new["max_profit_pct"], errors="coerce")
    df_gold["max_profit_pct"] = pd.to_numeric(df_gold["max_profit_pct"], errors="coerce")
    for col in ("entry_date", "exit_date", "entry_symbol", "symbol", "exit_reason"):
        df_new[col] = df_new[col].astype(str)
        df_gold[col] = df_gold[col].astype(str)

    pd.testing.assert_frame_equal(df_new, df_gold, check_exact=True)

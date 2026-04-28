"""Regression: v22 via Pipeline orchestrator should match golden trades_v22.csv."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
GOLDEN_DIR = REPO_ROOT / "stock_ml" / "tests" / "regression" / "golden"
GOLDEN_CSV = GOLDEN_DIR / "trades_v22.csv"
GOLDEN_META = GOLDEN_DIR / "trades_v22.meta.json"
CHAMPION_YAML = REPO_ROOT / "stock_ml" / "config" / "experiments" / "champions" / "v22.yaml"


@pytest.mark.regression
def test_v22_via_pipeline_matches_golden() -> None:
    from run_pipeline import _build_predictions
    from src.env import resolve_data_dir
    from src.pipeline import ExperimentConfig, Pipeline

    if not GOLDEN_CSV.exists() or not GOLDEN_META.exists():
        pytest.fail(f"Golden artefacts missing: {GOLDEN_CSV} / {GOLDEN_META}")

    meta = json.loads(GOLDEN_META.read_text())
    symbols: list[str] = meta["symbols"]
    data_dir = resolve_data_dir("../portable_data/vn_stock_ai_dataset_cleaned")
    if not Path(data_dir).is_dir():
        pytest.skip(f"Data dir không tồn tại: {data_dir}")

    # Build prediction cache using legacy helper (identical to direct runner test)
    prediction_cache = _build_predictions(
        symbols,
        meta["feature_set"],
        meta["target_config"],
        "cpu",
        model_type=meta["model_type"],
        exit_model_cfg=meta["exit_model_config"],
    )

    cfg = ExperimentConfig.from_yaml(CHAMPION_YAML)
    pipeline = Pipeline(cfg, symbols=symbols, device="cpu", prediction_cache=prediction_cache)
    result = pipeline.run()

    from src.components.runners import trades_to_v22_dataframe

    df_raw = trades_to_v22_dataframe(result.trades).reset_index(drop=True)
    df_new = pd.read_csv(io.StringIO(df_raw.to_csv(index=False))).reset_index(drop=True)
    df_gold = pd.read_csv(GOLDEN_CSV).reset_index(drop=True)

    assert result.n_trades == int(meta["n_trades"]), (
        f"Trade count mismatch: new={result.n_trades} expected={meta['n_trades']}"
    )

    df_new["max_profit_pct"] = pd.to_numeric(df_new["max_profit_pct"], errors="coerce")
    df_gold["max_profit_pct"] = pd.to_numeric(df_gold["max_profit_pct"], errors="coerce")
    for col in ("entry_date", "exit_date", "entry_symbol", "symbol", "exit_reason"):
        df_new[col] = df_new[col].astype(str)
        df_gold[col] = df_gold[col].astype(str)

    pd.testing.assert_frame_equal(df_new, df_gold, check_exact=True)

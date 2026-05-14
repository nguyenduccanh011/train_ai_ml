from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

import pandas as pd
from model_registry import DEFAULT_MODEL, get_model_cfg
from serve_train61_model import (
    _build_live_prediction_cache_on_demand,
    _build_prediction_cache_from_model,
    _load_cfg,
)
from src.pipeline import Pipeline


def _build_cache(
    model_id: str, model_cfg: dict[str, Any], cfg: Any, symbols: list[str]
) -> list[dict[str, Any]]:
    cache_items = []
    model_type = str(model_cfg.get("type", "pkl"))
    for symbol in symbols:
        if model_type == "on_demand":
            cache_items.extend(_build_live_prediction_cache_on_demand(model_id, cfg, symbol))
        else:
            cache_items.extend(_build_prediction_cache_from_model(model_id, symbol, model_cfg))
    return cache_items


def _run_variant(model_id: str, symbols: list[str], lag: int) -> dict[str, Any]:
    model_cfg = get_model_cfg(model_id)
    cfg = _load_cfg(model_id, model_cfg=model_cfg)
    cfg.params = dict(cfg.params or {})
    cfg.params["entry_filter_lag"] = lag

    cache_items = _build_cache(model_id, model_cfg, cfg, symbols)

    result = Pipeline(cfg, symbols=symbols, device="cpu", prediction_cache=cache_items).run()
    trades_df = (
        result.trades_df
        if isinstance(result.trades_df, pd.DataFrame)
        else pd.DataFrame(result.trades_df)
    )
    metrics = dict(result.metrics or {})
    metrics["entry_filter_lag"] = lag
    metrics["n_trades"] = int(len(trades_df))
    metrics["n_cache_items"] = int(len(cache_items))
    metrics["n_prediction_rows"] = int(
        sum(len(item.get("sym_test_df", [])) for item in cache_items)
    )
    metrics["symbols_with_trades"] = (
        int(trades_df["symbol"].nunique()) if "symbol" in trades_df else 0
    )
    return {"metrics": metrics, "trades_df": trades_df}


def _metric_row(name: str, a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    av = a.get(name)
    bv = b.get(name)
    diff = None
    if isinstance(av, (int, float)) and isinstance(bv, (int, float)):
        diff = bv - av
    return {"metric": name, "lag0_entry_bar": av, "lag1_predict_bar": bv, "diff": diff}


def main() -> None:
    model_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    if len(sys.argv) > 2:
        symbols = [s.upper() for s in sys.argv[2:]]
    else:
        symbols_payload = json.loads(
            (ROOT / "config" / "train61_symbols.json").read_text(encoding="utf-8")
        )
        symbols = [str(s).upper() for s in symbols_payload["symbols"]]

    base = _run_variant(model_id, symbols, 0)
    pred_bar = _run_variant(model_id, symbols, 1)

    metric_names = [
        "total_pnl",
        "total_pnl_simple",
        "wr",
        "avg_pnl",
        "pf",
        "max_loss",
        "avg_hold",
        "mdd_per_symbol",
        "yearly_consistency",
        "composite_score",
        "trades",
        "n_trades",
        "symbols_with_trades",
        "n_cache_items",
        "n_prediction_rows",
    ]
    rows = [_metric_row(name, base["metrics"], pred_bar["metrics"]) for name in metric_names]
    print(f"model_id={model_id}")
    print(f"symbols={','.join(symbols)}")
    print(pd.DataFrame(rows).to_string(index=False))

    out_dir = ROOT / "results" / "entry_filter_lag_compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    base["trades_df"].to_csv(out_dir / "lag0_entry_bar_trades.csv", index=False)
    pred_bar["trades_df"].to_csv(out_dir / "lag1_predict_bar_trades.csv", index=False)
    pd.DataFrame(rows).to_csv(out_dir / "summary.csv", index=False)
    print(f"saved={out_dir}")


if __name__ == "__main__":
    main()

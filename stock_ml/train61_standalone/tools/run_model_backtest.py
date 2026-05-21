from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from tools.env import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

import serve_train61_model as server  # noqa: E402
from model_registry import DEFAULT_MODEL, get_model_cfg  # noqa: E402
from src.pipeline import Pipeline  # noqa: E402
from src.pipeline.trainer import (
    build_prediction_cache as build_standard_prediction_cache,  # noqa: E402
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a backtest for one registered model.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--symbols", default="", help="Comma-separated symbols or JSON/TXT file")
    parser.add_argument("--out-dir", default=str(ROOT / "results" / "model_backtests"))
    parser.add_argument("--entry-filter-lag", type=int, default=None)
    parser.add_argument("--force-last-test-year", type=int, default=None)
    parser.add_argument("--compare-standard-cache", action="store_true")
    return parser.parse_args()


def load_symbols(raw: str, model_cfg: dict[str, Any]) -> list[str]:
    raw = str(raw or "").strip()
    if raw:
        path = Path(raw)
        if path.exists():
            if path.suffix.lower() == ".json":
                payload = json.loads(path.read_text(encoding="utf-8"))
                values = payload.get("symbols", payload) if isinstance(payload, dict) else payload
            else:
                values = path.read_text(encoding="utf-8").splitlines()
            return sorted({str(sym).strip().upper() for sym in values if str(sym).strip()})
        return sorted({part.strip().upper() for part in raw.split(",") if part.strip()})
    return server._load_symbols_for_model(model_cfg)


def build_prediction_cache(
    model_id: str, model_cfg: dict[str, Any], cfg: Any, symbols: list[str]
) -> list[dict[str, Any]]:
    model_type = str(model_cfg.get("type", "")).lower()
    if model_type == "pooled_global_rerun":
        return server._build_live_prediction_cache_pooled_global(
            model_id,
            cfg,
            symbols,
            server._dataset_dir_for_model(model_cfg),
        )

    cache_items: list[dict[str, Any]] = []
    for idx, symbol in enumerate(symbols, start=1):
        print(f"[{idx}/{len(symbols)}] build prediction cache {symbol}", flush=True)
        if model_type == "on_demand":
            cache_items.extend(server._build_live_prediction_cache_on_demand(model_id, cfg, symbol))
        elif model_type == "pkl":
            cache_items.extend(
                server._build_prediction_cache_from_model(model_id, symbol, model_cfg)
            )
        else:
            raise ValueError(
                f"Model type '{model_type}' is not supported by this script. "
                "Use pkl/on_demand/pooled_global_rerun models for this runner."
            )
    return cache_items


def compare_standard_cache(
    cfg: Any, symbols: list[str], live_items: list[dict[str, Any]]
) -> dict[str, Any]:
    standard_items = build_standard_prediction_cache(cfg, symbols, device="cpu")
    active_year = int(cfg.split.last_test_year)
    standard_items = [
        item
        for item in standard_items
        if str(
            item.get("sym_test_df", pd.DataFrame()).get("timestamp", pd.Series(dtype=str)).min()
        )[:4]
        == str(active_year)
    ]
    standard_by_symbol = {str(item.get("symbol", "")).upper(): item for item in standard_items}
    live_by_symbol = {str(item.get("symbol", "")).upper(): item for item in live_items}
    entry_diff_total = 0
    exit_diff_total = 0
    compared_symbols = 0
    for sym, standard in standard_by_symbol.items():
        live = live_by_symbol.get(sym)
        if live is None:
            continue
        n = min(len(standard.get("y_pred", [])), len(live.get("y_pred", [])))
        if n <= 0:
            continue
        entry_diff_total += int((standard["y_pred"][:n] != live["y_pred"][:n]).sum())
        std_exit = standard.get("y_pred_exit")
        live_exit = live.get("y_pred_exit")
        if std_exit is not None and live_exit is not None:
            exit_diff_total += int((std_exit[:n] != live_exit[:n]).sum())
        compared_symbols += 1
    return {
        "active_year": active_year,
        "compared_symbols": compared_symbols,
        "standard_items": len(standard_items),
        "live_items": len(live_items),
        "entry_diff_total": entry_diff_total,
        "exit_diff_total": exit_diff_total,
    }


def build_daily_signals(cache_items: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in cache_items:
        df = item.get("sym_test_df")
        y_pred = item.get("y_pred")
        y_pred_exit = item.get("y_pred_exit")
        if not isinstance(df, pd.DataFrame) or df.empty or y_pred is None:
            continue
        time_col = "timestamp" if "timestamp" in df.columns else "date"
        for i, row in df.reset_index(drop=True).iterrows():
            rows.append(
                {
                    "date": str(pd.Timestamp(row[time_col]).date()),
                    "symbol": str(row.get("symbol", item.get("symbol", ""))).upper(),
                    "close": float(row["close"])
                    if "close" in row and pd.notna(row["close"])
                    else None,
                    "entry_signal_for_next_bar": int(y_pred[i]) if i < len(y_pred) else None,
                    "exit_signal_for_next_bar": int(y_pred_exit[i])
                    if y_pred_exit is not None and i < len(y_pred_exit)
                    else None,
                    "model_mode": item.get("model_mode", ""),
                    "window_label": item.get("window_label", ""),
                }
            )
    return pd.DataFrame(rows)


def build_yearly_summary(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty or "exit_date" not in trades_df.columns:
        return pd.DataFrame()
    df = trades_df.copy()
    df["exit_date"] = pd.to_datetime(df["exit_date"], errors="coerce")
    df = df[df["exit_date"].notna()].copy()
    if df.empty:
        return pd.DataFrame()
    df["year"] = df["exit_date"].dt.year
    df["pnl_pct"] = pd.to_numeric(df.get("pnl_pct"), errors="coerce")
    rows = []
    for year, group in df.groupby("year"):
        pnl = group["pnl_pct"].dropna()
        gross_profit = float(pnl[pnl > 0].sum()) if not pnl.empty else 0.0
        gross_loss = float(-pnl[pnl < 0].sum()) if not pnl.empty else 0.0
        rows.append(
            {
                "year": int(year),
                "trades": int(len(group)),
                "wins": int((pnl > 0).sum()) if not pnl.empty else 0,
                "losses": int((pnl < 0).sum()) if not pnl.empty else 0,
                "win_rate": round(float((pnl > 0).mean() * 100), 2) if not pnl.empty else None,
                "total_pnl_pct": round(float(pnl.sum()), 4) if not pnl.empty else 0.0,
                "avg_pnl_pct": round(float(pnl.mean()), 4) if not pnl.empty else None,
                "pf": round(gross_profit / gross_loss, 4) if gross_loss else None,
            }
        )
    return pd.DataFrame(rows).sort_values("year")


def build_benchmark_yearly(model_cfg: dict[str, Any], years: list[int]) -> pd.DataFrame:
    if not years:
        return pd.DataFrame()
    loader = server._data_loader(server._dataset_dir_for_model(model_cfg))
    rows = []
    for symbol in ("VNINDEX", "VN30F1M", "HNXINDEX"):
        try:
            df = loader.load_symbol(symbol)
        except Exception:
            continue
        if df.empty:
            continue
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
        df["year"] = df["timestamp"].dt.year
        for year in years:
            group = df[df["year"] == year]
            if group.empty:
                continue
            first = float(group.iloc[0]["close"])
            last = float(group.iloc[-1]["close"])
            rows.append(
                {
                    "year": int(year),
                    "benchmark": symbol,
                    "start_close": first,
                    "end_close": last,
                    "return_pct": round((last / first - 1.0) * 100.0, 4) if first else None,
                }
            )
    return pd.DataFrame(rows)


def summarize(
    result: Any,
    *,
    model_id: str,
    symbols: list[str],
    runtime_sec: float,
    cache_items: list[dict[str, Any]],
) -> dict[str, Any]:
    metrics = dict(result.metrics or {})
    trades_df = result.trades_df if isinstance(result.trades_df, pd.DataFrame) else pd.DataFrame()
    pnl = (
        pd.to_numeric(trades_df["pnl_pct"], errors="coerce").dropna()
        if not trades_df.empty and "pnl_pct" in trades_df.columns
        else pd.Series(dtype=float)
    )
    gross_profit = float(pnl[pnl > 0].sum()) if not pnl.empty else 0.0
    gross_loss = float(-pnl[pnl < 0].sum()) if not pnl.empty else 0.0
    return {
        "model_id": model_id,
        "symbols": len(symbols),
        "symbols_with_trades": int(trades_df["symbol"].nunique())
        if not trades_df.empty and "symbol" in trades_df.columns
        else 0,
        "prediction_cache_items": len(cache_items),
        "prediction_rows": int(sum(len(item.get("sym_test_df", [])) for item in cache_items)),
        "runtime_sec": round(runtime_sec, 2),
        "trades": int(len(trades_df)),
        "wins": int((pnl > 0).sum()) if not pnl.empty else 0,
        "losses": int((pnl < 0).sum()) if not pnl.empty else 0,
        "win_rate": round(float((pnl > 0).mean() * 100), 2) if not pnl.empty else None,
        "total_pnl_pct": round(float(pnl.sum()), 4) if not pnl.empty else 0.0,
        "avg_pnl_pct": round(float(pnl.mean()), 4) if not pnl.empty else None,
        "median_pnl_pct": round(float(pnl.median()), 4) if not pnl.empty else None,
        "pf": round(gross_profit / gross_loss, 4) if gross_loss else None,
        "max_win_pct": round(float(pnl.max()), 4) if not pnl.empty else None,
        "max_loss_pct": round(float(pnl.min()), 4) if not pnl.empty else None,
        "avg_hold": metrics.get("avg_hold"),
        "mdd_per_symbol": metrics.get("mdd_per_symbol"),
        "yearly_consistency": metrics.get("yearly_consistency"),
        "composite_score": metrics.get("composite_score"),
        "raw_metrics": metrics,
    }


def main() -> int:
    args = parse_args()
    model_cfg = get_model_cfg(args.model_id)
    cfg = server._load_cfg(args.model_id, model_cfg=model_cfg)
    if args.force_last_test_year is not None:
        cfg.split.last_test_year = int(args.force_last_test_year)
    if args.entry_filter_lag is not None:
        cfg.params = dict(cfg.params or {})
        cfg.params["entry_filter_lag"] = args.entry_filter_lag
    symbols = load_symbols(args.symbols, model_cfg)

    started = time.time()
    cache_items = build_prediction_cache(args.model_id, model_cfg, cfg, symbols)
    parity = (
        compare_standard_cache(cfg, symbols, cache_items) if args.compare_standard_cache else None
    )
    result = Pipeline(cfg, symbols=symbols, device="cpu", prediction_cache=cache_items).run()
    runtime_sec = time.time() - started

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / f"{args.model_id}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    trades_path = out_dir / "trades.csv"
    closed_trades_path = out_dir / "closed_trades.csv"
    daily_signals_path = out_dir / "daily_signals.csv"
    yearly_summary_path = out_dir / "yearly_summary.csv"
    benchmark_yearly_path = out_dir / "benchmark_yearly.csv"
    summary_path = out_dir / "summary.json"

    trades_df = result.trades_df if isinstance(result.trades_df, pd.DataFrame) else pd.DataFrame()
    trades_df.to_csv(trades_path, index=False)
    closed_trades = trades_df.copy()
    if not closed_trades.empty and "exit_date" in closed_trades.columns:
        closed_trades = closed_trades[
            closed_trades["exit_date"].astype(str).str.strip() != ""
        ].copy()
    closed_trades.to_csv(closed_trades_path, index=False)

    daily_signals = build_daily_signals(cache_items)
    daily_signals.to_csv(daily_signals_path, index=False)
    yearly_summary = build_yearly_summary(closed_trades)
    yearly_summary.to_csv(yearly_summary_path, index=False)
    years = yearly_summary["year"].astype(int).tolist() if not yearly_summary.empty else []
    benchmark_yearly = build_benchmark_yearly(model_cfg, years)
    benchmark_yearly.to_csv(benchmark_yearly_path, index=False)

    summary = summarize(
        result,
        model_id=args.model_id,
        symbols=symbols,
        runtime_sec=runtime_sec,
        cache_items=cache_items,
    )
    summary["trades_path"] = str(trades_path)
    summary["closed_trades_path"] = str(closed_trades_path)
    summary["daily_signals_path"] = str(daily_signals_path)
    summary["yearly_summary_path"] = str(yearly_summary_path)
    summary["benchmark_yearly_path"] = str(benchmark_yearly_path)
    summary["summary_path"] = str(summary_path)
    if parity is not None:
        summary["standard_cache_comparison"] = parity
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

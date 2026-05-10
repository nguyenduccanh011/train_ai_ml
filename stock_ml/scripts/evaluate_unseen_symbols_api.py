from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cache.feature_cache import FeatureCacheManager  # noqa: E402
from src.components.exit_models.registry import get_exit_model  # noqa: E402
from src.components.models.registry import get_model  # noqa: E402
from src.config_loader import load_config, resolve_data_dir  # noqa: E402
from src.data.loader import DataLoader  # noqa: E402
from src.data.splitter import WalkForwardSplitter  # noqa: E402
from src.data.target import TargetGenerator  # noqa: E402
from src.env import get_results_dir  # noqa: E402
from src.evaluation.scoring import (  # noqa: E402
    calc_max_drawdown,
    calc_mdd_per_symbol,
    calc_metrics,
    calc_symbol_coverage,
    calc_yearly_consistency,
    composite_score,
)
from src.features.engine import FeatureEngine  # noqa: E402
from src.pipeline import ExperimentConfig, Pipeline  # noqa: E402
from src.signal_adapter import canonicalize_predictions  # noqa: E402

TOP1_CONFIG_PATH = (
    ROOT
    / "results"
    / "experiments"
    / "v22_exit_ablation_round42"
    / "v22_exit_ablation_round42_signals_features-leading-signals_entry_model_type-random_forest-signals_target-earlyv2_fw21_g033125_l0165625-exit_model-exit_fw21_l03725-fusion-peak_dist_only"
    / "config.resolved.yaml"
)


def summarize_trades(trades: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = calc_metrics(trades)
    return {
        "trades": int(metrics.get("trades", 0)),
        "wr": float(metrics.get("wr", 0.0)),
        "avg_pnl": float(metrics.get("avg_pnl", 0.0)),
        "total_pnl": float(metrics.get("total_pnl", 0.0)),
        "pf": float(metrics.get("pf", 0.0)),
        "max_loss": float(metrics.get("max_loss", 0.0)),
        "avg_hold": float(metrics.get("avg_hold", 0.0)),
        "max_drawdown": round(float(calc_max_drawdown(trades)), 2),
        "mdd_per_symbol": round(float(calc_mdd_per_symbol(trades)), 2),
        "yearly_consistency": round(float(calc_yearly_consistency(trades)), 4),
        "composite_score": float(composite_score(metrics, trades)),
        "symbol_coverage": calc_symbol_coverage(trades),
    }


def _get(url: str, *, timeout: int = 30) -> Any:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def fetch_all_symbols(
    base_url: str,
    *,
    asset_type: str = "stock",
    exchanges: set[str] | None = None,
) -> list[str]:
    symbols: list[str] = []
    limit = 1000
    offset = 0
    while True:
        url = f"{base_url}/symbols/?limit={limit}&offset={offset}"
        payload = _get(url)
        items = payload.get("items", [])
        if not items:
            break
        for item in items:
            if not item.get("is_active", True):
                continue
            if asset_type and str(item.get("asset_type", "")).lower() != asset_type.lower():
                continue
            if exchanges:
                ex = str(item.get("exchange", "")).upper()
                if ex not in exchanges:
                    continue
            sym = str(item.get("symbol", "")).upper()
            if sym:
                symbols.append(sym)
        offset += limit
        total = int(payload.get("total", 0) or 0)
        if total and offset >= total:
            break
    return sorted(set(symbols))


def fetch_symbol_count(base_url: str, symbol: str, timeframe: str) -> int:
    url = f"{base_url}/ohlcv/symbols/{symbol}/count?timeframe={timeframe}"
    payload = _get(url)
    return int(payload.get("count", 0) or 0)


def fetch_symbol_ohlcv(
    base_url: str,
    symbol: str,
    timeframe: str,
    *,
    end_date: str | None = None,
) -> pd.DataFrame:
    limit = 1000
    offset = 0
    rows: list[dict[str, Any]] = []
    while True:
        url = (
            f"{base_url}/ohlcv/?symbol={symbol}&timeframe={timeframe}&limit={limit}&offset={offset}"
        )
        if end_date:
            url += f"&end_date={end_date}"
        payload = _get(url)
        items = payload.get("items", [])
        if not items:
            break
        rows.extend(items)
        offset += limit
        total = int(payload.get("total", 0) or 0)
        if total and offset >= total:
            break
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    keep = ["timestamp", "open", "high", "low", "close", "volume", "traded_value"]
    for col in keep:
        if col not in df.columns:
            df[col] = np.nan
    df = df[keep].copy()
    df["symbol"] = symbol
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    for c in ("open", "high", "low", "close", "volume", "traded_value"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close"]).copy()
    df["volume"] = df["volume"].fillna(0.0)
    df["traded_value"] = df["traded_value"].fillna(df["close"] * df["volume"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def load_train_universe(cfg: ExperimentConfig) -> tuple[list[str], pd.DataFrame]:
    pipeline_cfg = load_config().get("pipeline", {})
    data_dir = pipeline_cfg.get("data_dir", "../portable_data/vn_stock_ai_dataset_cleaned")
    abs_data_dir = resolve_data_dir(data_dir)
    loader = DataLoader(abs_data_dir)
    symbols = sorted([s.upper() for s in loader.symbols])

    # Reuse feature cache when possible for speed.
    cache_root = Path(get_results_dir()) / "cache" / "features"
    cache_mgr = FeatureCacheManager(str(cache_root))
    feature_set = cfg.feature_set()
    target_cfg = cfg.target_dict()
    df_cached, _ = cache_mgr.load(
        data_dir=abs_data_dir,
        symbols=symbols,
        timeframe=loader.timeframe,
        feature_set=feature_set,
        target_config=target_cfg,
        code_paths=[],
    )
    if df_cached is not None:
        return symbols, df_cached

    raw_df = loader.load_all(symbols=symbols, show_progress=True)
    return symbols, raw_df


def build_prediction_cache_for_unseen(
    cfg: ExperimentConfig,
    train_raw_df: pd.DataFrame,
    unseen_raw_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    feature_set = cfg.feature_set()
    engine = FeatureEngine(feature_set=feature_set)

    train_feat = engine.compute_for_all_symbols(train_raw_df)
    unseen_feat = engine.compute_for_all_symbols(unseen_raw_df)
    feature_cols = engine.get_feature_columns(train_feat)

    split_cfg = cfg.split
    legacy_split = {
        "split": {
            "method": split_cfg.method,
            "train_years": split_cfg.train_years,
            "test_years": split_cfg.test_years,
            "gap_days": split_cfg.gap_days,
            "first_test_year": split_cfg.first_test_year,
            "last_test_year": split_cfg.last_test_year,
        },
        "target": cfg.target_dict(),
    }

    target_gen = TargetGenerator.from_config(legacy_split)
    train_df = target_gen.generate_for_all_symbols(train_feat.copy())
    exit_model_dict = cfg.exit_model_dict()
    if exit_model_dict:
        train_df = TargetGenerator.generate_exit_labels(
            train_df,
            forward_window=exit_model_dict.get("forward_window", 15),
            loss_threshold=exit_model_dict.get("loss_threshold", 0.05),
        )

    drop_cols = feature_cols + ["target"]
    has_exit = "target_sell" in train_df.columns
    if has_exit:
        drop_cols.append("target_sell")
    train_df = train_df.dropna(subset=drop_cols).copy()
    unseen_df = unseen_feat.dropna(subset=feature_cols).copy()

    splitter = WalkForwardSplitter(
        method=split_cfg.method,
        train_years=split_cfg.train_years,
        test_years=split_cfg.test_years,
        gap_days=split_cfg.gap_days,
        first_test_year=split_cfg.first_test_year,
        last_test_year=split_cfg.last_test_year,
    )

    pred_cache: list[dict[str, Any]] = []
    exit_model_cfg = cfg.signals.exit_model
    target_cfg_dict = legacy_split.get("target", {})

    for window in splitter.get_windows():
        train_mask = (train_df["timestamp"] >= window.train_start) & (
            train_df["timestamp"] <= window.train_end
        )
        test_mask = (unseen_df["timestamp"] >= window.test_start) & (
            unseen_df["timestamp"] <= window.test_end
        )
        tr = train_df[train_mask]
        te = unseen_df[test_mask]
        if tr.empty or te.empty:
            continue

        X_train = np.nan_to_num(tr[feature_cols].values)
        y_train = tr["target"].values.astype(int)
        model = get_model(cfg.entry_model_type(), device="cpu", **cfg.signals.entry_model.extras)
        model.fit(X_train, y_train)

        sell_model = None
        if has_exit:
            sell_model = get_exit_model(exit_model_cfg.type, device="cpu", **exit_model_cfg.extras)
            sell_model.fit(X_train, tr["target_sell"].values.astype(int))

        for sym, sym_df in te.groupby("symbol"):
            sym_df = sym_df.sort_values("timestamp").reset_index(drop=True)
            if len(sym_df) < 10:
                continue
            X_sym = np.nan_to_num(sym_df[feature_cols].values)
            y_pred_raw = model.predict(X_sym)
            y_pred = canonicalize_predictions(y_pred_raw, target_cfg_dict)

            y_proba = None
            classes = None
            try:
                if hasattr(model, "predict_proba"):
                    y_proba = model.predict_proba(X_sym)
                    final_est = model.steps[-1][1] if hasattr(model, "steps") else model
                    classes = list(final_est.classes_)
            except Exception:
                y_proba = None

            pred_cache.append(
                {
                    "symbol": str(sym).upper(),
                    "y_pred": y_pred,
                    "y_pred_exit": sell_model.predict(X_sym).astype(int)
                    if sell_model is not None
                    else None,
                    "y_proba": y_proba,
                    "classes": classes,
                    "returns": sym_df["return_1d"].values,
                    "sym_test_df": sym_df,
                    "feature_cols": feature_cols,
                }
            )

    return pred_cache


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate pooled model (trained on local universe) on unseen symbols from API."
    )
    parser.add_argument("--base-url", default="https://sieutinhieu.vn/api/v1", help="API base url")
    parser.add_argument("--timeframe", default="1D", help="OHLCV timeframe (default: 1D)")
    parser.add_argument(
        "--exchanges",
        default="HOSE,HNX,UPCOM",
        help="Comma-separated exchanges to include (empty = all exchanges)",
    )
    parser.add_argument(
        "--unseen-limit", type=int, default=30, help="Max unseen symbols to evaluate"
    )
    parser.add_argument(
        "--min-bars", type=int, default=800, help="Min OHLCV bars to keep unseen symbol"
    )
    parser.add_argument(
        "--end-date",
        default="2025-12-31T23:59:59Z",
        help="End date for unseen OHLCV download (ISO 8601). Use empty string for latest.",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "results" / "unseen_symbols_eval.json"),
        help="Output JSON path",
    )
    args = parser.parse_args()

    started = time.time()
    cfg = ExperimentConfig.from_yaml(TOP1_CONFIG_PATH)

    print("Loading local train universe...")
    train_symbols, train_raw = load_train_universe(cfg)
    train_symbol_set = set(train_symbols)
    print(f"  local symbols: {len(train_symbols)}")

    exchanges = {x.strip().upper() for x in str(args.exchanges).split(",") if x.strip()}
    print("Fetching API symbols...")
    api_symbols = fetch_all_symbols(args.base_url, asset_type="stock", exchanges=exchanges or None)
    unseen_candidates = [s for s in api_symbols if s not in train_symbol_set]
    print(f"  api stock symbols: {len(api_symbols)}")
    print(f"  unseen candidates: {len(unseen_candidates)}")
    if not unseen_candidates:
        print("No unseen symbols found.")
        return 0

    print("Filtering unseen by bar count...")
    picked: list[str] = []
    counts: dict[str, int] = {}
    for sym in unseen_candidates:
        try:
            n = fetch_symbol_count(args.base_url, sym, args.timeframe)
            counts[sym] = n
            if n >= args.min_bars:
                picked.append(sym)
            if len(picked) >= args.unseen_limit:
                break
        except Exception:
            continue
    print(f"  picked unseen symbols: {len(picked)} (min bars={args.min_bars})")
    if not picked:
        print("No unseen symbols satisfy min-bars condition.")
        return 0

    print("Downloading unseen OHLCV...")
    unseen_parts: list[pd.DataFrame] = []
    end_date = args.end_date.strip() if args.end_date else None
    for i, sym in enumerate(picked, 1):
        try:
            df = fetch_symbol_ohlcv(args.base_url, sym, args.timeframe, end_date=end_date)
            if len(df) < args.min_bars:
                continue
            unseen_parts.append(df)
            print(f"  [{i}/{len(picked)}] {sym}: {len(df)} bars")
        except Exception as exc:
            print(f"  [{i}/{len(picked)}] {sym}: error {exc}")
            continue

    if not unseen_parts:
        print("No unseen OHLCV data downloaded.")
        return 0

    unseen_raw = pd.concat(unseen_parts, ignore_index=True)
    unseen_raw = unseen_raw.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    unseen_symbols = sorted(unseen_raw["symbol"].astype(str).str.upper().unique().tolist())
    print(f"  unseen usable symbols: {len(unseen_symbols)}")

    print("Building prediction cache for unseen symbols (pooled train, unseen test)...")
    pred_cache = build_prediction_cache_for_unseen(cfg, train_raw, unseen_raw)
    print(f"  prediction cache items: {len(pred_cache)}")
    if not pred_cache:
        print("No predictions generated for unseen symbols.")
        return 0

    print("Running strategy backtest on unseen prediction cache...")
    result = Pipeline(cfg, symbols=unseen_symbols, device="cpu", prediction_cache=pred_cache).run()
    trades_df = result.trades_df if result.trades_df is not None else pd.DataFrame()
    trades: list[dict[str, Any]] = trades_df.to_dict("records") if not trades_df.empty else []
    metrics = summarize_trades(trades)

    by_symbol: dict[str, dict[str, Any]] = {}
    if trades:
        for sym, g in trades_df.groupby("symbol"):
            by_symbol[str(sym).upper()] = summarize_trades(g.to_dict("records"))

    output = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config_path": str(TOP1_CONFIG_PATH),
        "base_url": args.base_url,
        "timeframe": args.timeframe,
        "train_universe_symbol_count": len(train_symbols),
        "unseen_candidates_count": len(unseen_candidates),
        "unseen_requested_limit": args.unseen_limit,
        "unseen_min_bars": args.min_bars,
        "unseen_symbols_evaluated": unseen_symbols,
        "unseen_symbol_count": len(unseen_symbols),
        "end_date": end_date,
        "prediction_cache_items": len(pred_cache),
        "overall_metrics": metrics,
        "per_symbol_metrics": by_symbol,
        "runtime_sec": round(time.time() - started, 1),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")
    print("Overall:", metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

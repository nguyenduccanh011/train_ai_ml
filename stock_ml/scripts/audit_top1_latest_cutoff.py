from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.realtime_top1_common import _build_day_snapshot, load_top1_config
from src.data.loader import DataLoader
from src.env import resolve_data_dir
from src.features.engine import FeatureEngine
from src.market_profile import load_market_profile


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit top1 snapshot on latest live data.")
    p.add_argument("--cutoff", default="2026-05-18", help="Snapshot cutoff YYYY-MM-DD")
    p.add_argument("--base-url", default="https://sieutinhieu.vn/api/v1", help="API base url")
    p.add_argument("--max-workers", type=int, default=20)
    p.add_argument(
        "--output",
        default=str(ROOT / "results" / "leakage_check" / "top1_latest_cutoff_audit.json"),
    )
    return p.parse_args()


def _load_local_raw() -> pd.DataFrame:
    profile = load_market_profile("vn_stock")
    data_dir = resolve_data_dir(profile.data.data_dir)
    loader = DataLoader(
        str(data_dir),
        timeframe=profile.data.default_timeframe,
        timestamp_column=profile.data.timestamp_column,
        timezone=profile.data.timezone,
        required_columns=profile.data.required_columns,
        optional_columns=profile.data.optional_columns,
    )
    raw_df = loader.load_all(show_progress=False)
    raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"], utc=True, errors="coerce")
    raw_df = (
        raw_df.dropna(subset=["timestamp"])
        .sort_values(["symbol", "timestamp"])
        .reset_index(drop=True)
    )
    return raw_df


def _fetch_latest_rows(base_url: str, symbol: str) -> pd.DataFrame:
    url = f"{base_url}/ohlcv/latest?symbol={symbol}&timeframe=1D&limit=100"
    payload = requests.get(url, timeout=60).json()
    if not isinstance(payload, list) or not payload:
        return pd.DataFrame()
    df = pd.DataFrame(payload)
    df["symbol"] = symbol
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
    if "traded_value" in df.columns:
        df["traded_value"] = pd.to_numeric(df["traded_value"], errors="coerce")
    else:
        df["traded_value"] = df["close"] * df["volume"]
    return df[["symbol", "timestamp", "open", "high", "low", "close", "volume", "traded_value"]]


def _merge_live(raw_df: pd.DataFrame, base_url: str, max_workers: int) -> pd.DataFrame:
    symbols = sorted(raw_df["symbol"].astype(str).str.upper().unique().tolist())
    parts: list[pd.DataFrame] = []
    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_fetch_latest_rows, base_url, sym) for sym in symbols]
        for idx, fut in enumerate(cf.as_completed(futs), 1):
            df = fut.result()
            if not df.empty:
                parts.append(df)
            if idx % 50 == 0:
                print(f"[fetch] {idx}/{len(symbols)}")
    api_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=raw_df.columns)
    api_df = api_df.dropna(subset=["timestamp"])
    merged = pd.concat([raw_df, api_df], ignore_index=True)
    merged["symbol"] = merged["symbol"].astype(str).str.upper()
    merged = (
        merged.sort_values(["symbol", "timestamp"])
        .drop_duplicates(["symbol", "timestamp"], keep="last")
        .reset_index(drop=True)
    )
    return merged


def main() -> int:
    args = parse_args()
    cutoff = str(args.cutoff)[:10]
    raw_df = _load_local_raw()
    print(f"[local] rows={len(raw_df)} max={raw_df['timestamp'].max()}")
    merged = _merge_live(raw_df, args.base_url, args.max_workers)
    print(f"[merged] rows={len(merged)} max={merged['timestamp'].max()}")

    cfg = load_top1_config()
    feature_engine = FeatureEngine(feature_set=cfg.feature_set())
    snapshot = _build_day_snapshot(
        cfg=cfg,
        cutoff=cutoff,
        raw_df=merged,
        feature_engine=feature_engine,
        symbols=sorted(raw_df["symbol"].astype(str).str.upper().unique().tolist()),
        watchlist_top=10,
        min_history=260,
    )

    report = {
        "cutoff": cutoff,
        "stats": snapshot.stats,
        "new_entries": snapshot.new_entries,
        "open_positions": snapshot.open_positions,
        "all_predictions": snapshot.all_predictions,
        "next_session_predictions": snapshot.next_session_predictions,
        "watchlist_top_buy_proba": snapshot.watchlist_top_buy_proba,
        "errors": snapshot.errors,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(f"[saved] {out_path}")
    print(json.dumps(snapshot.stats, ensure_ascii=False))
    print(f"[new_entries] {snapshot.new_entries}")
    print(f"[top_next] {snapshot.next_session_predictions[:10]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

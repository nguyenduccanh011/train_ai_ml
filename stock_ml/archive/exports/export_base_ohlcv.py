"""
Export base OHLCV data for ALL symbols to visualization/data/{SYM}.json

This generates the base price data that the dashboard needs.
The unified_export.py only generates model overlay data (data_v25/, data_v24/, etc.)
but the dashboard also needs base OHLCV candlestick data in data/{SYM}.json.

Usage:
    python export_base_ohlcv.py                    # Export all symbols from trades
    python export_base_ohlcv.py --symbols ACB,FPT  # Export specific symbols
    python export_base_ohlcv.py --all-clean         # Export ALL clean symbols (486)
"""
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

# Setup path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

DATA_DIR = os.path.join(BASE_DIR, "..", "portable_data", "vn_stock_ai_dataset_cleaned")
VIZ_DIR = os.path.join(BASE_DIR, "visualization")
OUT_DIR = os.path.join(VIZ_DIR, "data")


def get_symbols_from_trades():
    """Get union of all symbols that appear in any model's trades."""
    results_dir = os.path.join(BASE_DIR, "results")
    all_symbols = set()
    
    trade_files = [
        "trades_v25.csv", "trades_v24.csv", "trades_v23.csv",
        "trades_v22.csv", "trades_v19_1.csv", "trades_rule.csv",
    ]
    
    for tf in trade_files:
        path = os.path.join(results_dir, tf)
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path, usecols=lambda c: c in ("symbol", "entry_symbol"))
            if "symbol" in df.columns:
                all_symbols.update(df["symbol"].dropna().astype(str).unique())
            elif "entry_symbol" in df.columns:
                all_symbols.update(df["entry_symbol"].dropna().astype(str).unique())
        except Exception as e:
            print(f"  ⚠ Error reading {tf}: {e}")
    
    return sorted(all_symbols)


def get_all_clean_symbols():
    """Get all symbols from clean_symbols.txt."""
    path = os.path.join(DATA_DIR, "clean_symbols.txt")
    if os.path.exists(path):
        return [s.strip() for s in open(path).read().strip().split("\n") if s.strip()]
    return []


def load_ohlcv(symbol):
    """Load OHLCV data for a symbol from the cleaned dataset."""
    csv_path = os.path.join(DATA_DIR, "all_symbols", f"symbol={symbol}", "timeframe=1D", "data.csv")
    if not os.path.exists(csv_path):
        return None
    
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    ohlcv = []
    for _, row in df.iterrows():
        d = str(row["timestamp"])[:10]
        ohlcv.append({
            "time": d,
            "open": round(float(row["open"]), 2) if pd.notna(row.get("open")) else 0,
            "high": round(float(row["high"]), 2) if pd.notna(row.get("high")) else 0,
            "low": round(float(row["low"]), 2) if pd.notna(row.get("low")) else 0,
            "close": round(float(row["close"]), 2) if pd.notna(row.get("close")) else 0,
            "volume": int(row["volume"]) if pd.notna(row.get("volume")) else 0,
        })
    
    return ohlcv


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Export base OHLCV data for dashboard")
    parser.add_argument("--symbols", type=str, default="",
                        help="Comma-separated symbols to export")
    parser.add_argument("--all-clean", action="store_true",
                        help="Export ALL clean symbols (not just those with trades)")
    args = parser.parse_args()
    
    # Determine symbols to export
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    elif args.all_clean:
        symbols = get_all_clean_symbols()
    else:
        symbols = get_symbols_from_trades()
    
    if not symbols:
        print("⚠ No symbols to export!")
        return
    
    os.makedirs(OUT_DIR, exist_ok=True)
    
    print("=" * 80)
    print("EXPORT BASE OHLCV DATA → visualization/data/")
    print("=" * 80)
    print(f"  Symbols to export: {len(symbols)}")
    print(f"  Source: {DATA_DIR}")
    print(f"  Output: {OUT_DIR}")
    print()
    
    index_entries = []
    exported = 0
    skipped = 0
    
    for i, sym in enumerate(symbols):
        ohlcv = load_ohlcv(sym)
        if ohlcv is None:
            skipped += 1
            continue
        
        payload = {
            "symbol": sym,
            "ohlcv": ohlcv,
        }
        
        out_path = os.path.join(OUT_DIR, f"{sym}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        
        index_entries.append({
            "symbol": sym,
            "file": f"data/{sym}.json",
            "data_points": len(ohlcv),
        })
        exported += 1
        
        if (i + 1) % 50 == 0 or (i + 1) == len(symbols):
            print(f"  Progress: {i + 1}/{len(symbols)} ({exported} exported, {skipped} skipped)")
    
    # Write index.json
    index_entries.sort(key=lambda x: x["symbol"])
    index = {
        "symbols": [{"symbol": e["symbol"], "file": e["file"]} for e in index_entries],
        "total_symbols": len(index_entries),
        "generated_at": datetime.now().isoformat(),
    }
    
    idx_path = os.path.join(OUT_DIR, "index.json")
    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Exported {exported} symbols to {OUT_DIR}")
    if skipped > 0:
        print(f"  ⚠ Skipped {skipped} symbols (no source data)")
    print(f"  📄 Index: {idx_path}")
    
    # Now update manifest.json to include base_symbols
    manifest_path = os.path.join(VIZ_DIR, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        
        manifest["base_symbols"] = [e["symbol"] for e in index_entries]
        manifest["base_data_dir"] = "data"
        
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        
        print(f"  ✓ Updated manifest.json with {len(index_entries)} base symbols")
    
    print("\n" + "=" * 80)
    print("DONE — Open dashboard.html to verify")
    print("=" * 80)


if __name__ == "__main__":
    main()

"""
Unified Export — exports trades from CSV files to JSON for web visualization.

Reads from results/trades_{version}.csv → outputs to visualization/data_{version}/{SYM}.json
Also generates a unified manifest (visualization/manifest.json) for the dynamic dashboard.

Usage:
    python -m src.export.unified_export                    # Export all active models
    python -m src.export.unified_export --versions v24,v23 # Export specific versions
    python -m src.export.unified_export --include-retired   # Include retired models
"""
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import src.safe_io  # noqa: F401 — fix UnicodeEncodeError on Windows console

from src.config_loader import (
    get_active_models,
    get_all_models,
    get_exit_abbreviations,
    get_model_config,
    get_visualization_config,
)
from src.env import get_results_dir


def make_markers(trades, version_key, color, marker_shape="arrowUp"):
    """Convert trade records into chart marker objects."""
    markers = []
    abbrevs = get_exit_abbreviations()
    win_color, loss_color = "#4caf50", "#f44336"
    label = version_key.upper().replace("_", ".")

    for t in trades:
        ed = t.get("entry_date", "")
        xd = t.get("exit_date", "")
        pnl = float(t.get("pnl_pct", 0))
        reason = t.get("exit_reason", "")
        tag = abbrevs.get(reason, reason[:2].upper() if reason else "")

        if ed:
            markers.append({
                "time": ed,
                "position": "belowBar",
                "color": color,
                "shape": marker_shape,
                "text": f"{label} Buy",
                "size": 1,
                "method": version_key,
            })
        if xd:
            markers.append({
                "time": xd,
                "position": "aboveBar",
                "color": win_color if pnl >= 0 else loss_color,
                "shape": "arrowDown",
                "text": f"{label} {pnl:+.1f}%{(' ' + tag) if tag else ''}",
                "size": 1,
                "method": version_key,
            })
    return markers


def compute_stats(trades, version_key):
    """Compute aggregate statistics for a list of trades."""
    if not trades:
        return {
            "total_trades": 0, "wins": 0, "losses": 0, "win_rate": 0,
            "total_pnl_pct": 0, "avg_pnl_pct": 0, "median_pnl_pct": 0,
            "std_pnl_pct": 0, "avg_win_pct": 0, "avg_loss_pct": 0,
            "payoff_ratio": 0, "max_win_pct": 0, "max_loss_pct": 0,
            "avg_hold": 0, "pf": 0, "version": version_key,
        }

    pnls = [float(t["pnl_pct"]) for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    holds = [float(t.get("holding_days", 0)) for t in trades]
    gross_loss = abs(sum(losses))
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    payoff = (avg_win / abs(avg_loss)) if avg_loss < 0 else 0.0

    return {
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(trades) * 100, 1),
        "total_pnl_pct": round(sum(pnls), 1),
        "avg_pnl_pct": round(float(np.mean(pnls)), 2),
        "median_pnl_pct": round(float(np.median(pnls)), 2),
        "std_pnl_pct": round(float(np.std(pnls)), 2),
        "avg_win_pct": round(avg_win, 2),
        "avg_loss_pct": round(avg_loss, 2),
        "payoff_ratio": round(payoff, 2),
        "max_win_pct": round(float(max(pnls)), 2),
        "max_loss_pct": round(float(min(pnls)), 2),
        "avg_hold": round(float(np.mean(holds)), 1) if holds else 0,
        "pf": round(sum(wins) / gross_loss, 2) if gross_loss > 0 else 99,
        "version": version_key,
    }


# Fields to include in exported trade records
TRADE_FIELDS = (
    "entry_date", "exit_date", "pnl_pct", "holding_days", "exit_reason",
    "entry_trend", "quick_reentry", "breakout_entry", "vshape_entry",
    "entry_profile", "entry_choppy_regime", "position_size",
    "secondary_breakout",
)


def select_fields(df):
    """Filter trade dicts to only include specified fields."""
    out = []
    for row in df.to_dict("records"):
        clean = {}
        for key in TRADE_FIELDS:
            if key not in row:
                continue
            val = row[key]
            if pd.isna(val):
                continue
            if isinstance(val, (np.generic,)):
                val = val.item()
            clean[key] = val
        out.append(clean)
    return out


def export_version(version_key, model_cfg, results_dir, viz_dir):
    """Export a single model version's trades to JSON files."""
    # Determine trades CSV path
    trades_csv = os.path.join(results_dir, f"trades_{version_key}.csv")
    if not os.path.exists(trades_csv):
        print(f"  ⚠ Skipping {version_key}: {trades_csv} not found")
        return None

    out_dir = os.path.join(viz_dir, f"data_{version_key}")
    os.makedirs(out_dir, exist_ok=True)

    color = model_cfg.get("color", "#888888")
    marker_shape = model_cfg.get("marker_shape", "arrowUp")

    # Load trades
    df = pd.read_csv(trades_csv)
    if "symbol" not in df.columns:
        if "entry_symbol" in df.columns:
            df["symbol"] = df["entry_symbol"].astype(str)
        else:
            print(f"  ⚠ Skipping {version_key}: no symbol column in CSV")
            return None

    for c in ("entry_date", "exit_date"):
        if c in df.columns:
            df[c] = df[c].astype(str).str[:10]

    grouped = {
        sym: g.copy().sort_values(["entry_date", "exit_date"]).reset_index(drop=True)
        for sym, g in df.groupby("symbol")
    }

    symbols_from_trades = sorted(df["symbol"].dropna().astype(str).unique().tolist())
    index_entries = []

    for symbol in symbols_from_trades:
        sym_df = grouped.get(symbol, pd.DataFrame())
        trades = select_fields(sym_df) if len(sym_df) > 0 else []
        stats = compute_stats(trades, version_key)
        markers = make_markers(trades, version_key, color, marker_shape)

        payload = {
            "symbol": symbol,
            f"{version_key}_markers": markers,
            f"{version_key}_trades": trades,
            f"{version_key}_stats": stats,
        }
        with open(os.path.join(out_dir, f"{symbol}.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

        index_entries.append({
            "symbol": symbol,
            f"{version_key}_trades": stats["total_trades"],
            f"{version_key}_pnl": stats["total_pnl_pct"],
            f"{version_key}_wr": stats["win_rate"],
        })

    index_entries.sort(key=lambda x: x.get(f"{version_key}_pnl", 0), reverse=True)
    with open(os.path.join(out_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump({
            "symbols": index_entries,
            "version": version_key,
            "generated_at": datetime.now().isoformat(),
        }, f, indent=2, ensure_ascii=False)

    total_pnl = sum(row.get(f"{version_key}_pnl", 0) for row in index_entries)
    with_trades = sum(1 for row in index_entries if row.get(f"{version_key}_trades", 0) > 0)
    print(f"  ✓ {version_key}: {len(index_entries)} symbols, "
          f"{with_trades} with trades, PnL={total_pnl:+.1f}%")

    return {
        "version_key": version_key,
        "name": model_cfg.get("name", version_key),
        "color": color,
        "marker_shape": marker_shape,
        "data_dir": f"data_{version_key}",
        "total_symbols": len(index_entries),
        "symbols_with_trades": with_trades,
        "total_pnl": total_pnl,
        "active": model_cfg.get("active", True),
        "order": model_cfg.get("order", 99),
    }


def generate_manifest(exported_versions, viz_dir, base_data_dir="data"):
    """Generate manifest.json for the dynamic dashboard."""
    # Check if base OHLCV data exists
    base_index_path = os.path.join(viz_dir, base_data_dir, "index.json")
    base_symbols = []
    if os.path.exists(base_index_path):
        with open(base_index_path, "r", encoding="utf-8") as f:
            base_index = json.load(f)
        base_symbols = [row.get("symbol") for row in base_index.get("symbols", []) if row.get("symbol")]

    manifest = {
        "generated_at": datetime.now().isoformat(),
        "base_data_dir": base_data_dir,
        "base_symbols": base_symbols,
        "models": sorted(exported_versions, key=lambda x: x["order"]),
        "exit_abbreviations": get_exit_abbreviations(),
    }

    manifest_path = os.path.join(viz_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Manifest saved to {manifest_path}")
    print(f"  Models: {', '.join(m['version_key'] for m in manifest['models'])}")
    return manifest


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Unified export for all model versions")
    parser.add_argument("--versions", type=str, default="",
                        help="Comma-separated version keys to export. Empty = all active.")
    parser.add_argument("--include-retired", action="store_true",
                        help="Include retired models in export")
    parser.add_argument("--base-data-dir", type=str, default="data",
                        help="Directory name for base OHLCV data (default: data)")
    args = parser.parse_args()

    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    results_dir = get_results_dir()
    viz_dir = os.path.join(base_dir, "visualization")

    # Determine which models to export
    if args.versions:
        version_keys = [v.strip() for v in args.versions.split(",") if v.strip()]
        models_to_export = {}
        for vk in version_keys:
            try:
                models_to_export[vk] = get_model_config(vk)
            except KeyError as e:
                print(f"  ⚠ {e}")
    else:
        models_to_export = get_all_models(include_retired=args.include_retired)

    print("=" * 80)
    print("UNIFIED EXPORT — Trades CSV → JSON for Dashboard")
    print("=" * 80)
    print(f"  Models to export: {', '.join(models_to_export.keys())}")
    print()

    exported = []
    for vk, mcfg in models_to_export.items():
        result = export_version(vk, mcfg, results_dir, viz_dir)
        if result:
            exported.append(result)

    if exported:
        generate_manifest(exported, viz_dir, args.base_data_dir)
    else:
        print("\n⚠ No models exported. Make sure trades CSV files exist in results/")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()

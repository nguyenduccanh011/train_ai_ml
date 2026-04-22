"""
Unified Pipeline Runner — train, backtest, export, and visualize in one command.

Usage:
    python run_pipeline.py --version v24                    # Run V24 only
    python run_pipeline.py --version v24 --export-only      # Skip training, just export
    python run_pipeline.py --version v24 --compare v23,rule # Run V24 + compare with V23 and Rule
    python run_pipeline.py --version v25 --compare v24,v23,rule  # Smart: only run v25, reuse v24/v23/rule CSVs
    python run_pipeline.py --all                            # Run all active models
    python run_pipeline.py --all --skip-existing            # Run only models missing CSV
    python run_pipeline.py --export-all                     # Export all models with existing CSVs
    python run_pipeline.py --version v25 --compare v24,v23 --force  # Force re-run all (no skip)

Workflow:
    1. Resolve symbols ONCE (shared across all models)
    2. Group models by feature_set → train ML once per group
    3. Run backtest per model (reusing cached predictions)
    4. Save trades CSV + metadata to results/
    5. Export JSON for visualization
    6. Generate manifest.json for dashboard

Smart Cache:
    - When using --compare, versions in --compare list are auto-skipped if CSV exists
    - The --version (primary) is ALWAYS run (unless --skip-existing is used)
    - Use --force to override and re-run everything
    - Use --skip-existing to skip ALL versions that already have CSV
    - Metadata (.meta.json) tracks conditions; mismatched caches trigger warnings
"""
import sys
import os
import json
import argparse
import time
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import src.safe_io  # noqa: F401 — fix UnicodeEncodeError on Windows console

from src.config_loader import (
    get_active_models, get_model_config, get_pipeline_symbols, load_config,
)
from src.env import resolve_data_dir, get_results_dir


RESULTS_DIR = get_results_dir()


def get_backtest_function(strategy_key):
    """Dynamically import the backtest function for a given strategy."""
    strategy_map = {
        "v27": ("run_v27", "backtest_v27_default"),
        "v26": ("run_v26", "backtest_v26_default"),
        "v25": ("run_v25", "backtest_v25"),
        "v24": ("run_v24", "backtest_v24"),
        "v23": ("run_v23_optimal", "backtest_v23"),
        "v22": ("run_v22_final", "backtest_v22_final"),
        "v19_3": ("run_v19_3_compare", "backtest_v19_3"),
        "v19_1": ("run_v19_1_compare", "backtest_v19_1"),
        "v18": ("run_v18_compare", "backtest_v18"),
        "v17": ("run_v17_compare", "backtest_v17"),
        "rule": ("compare_rule_vs_model", "backtest_rule"),
    }

    if strategy_key not in strategy_map:
        raise ValueError(f"Unknown strategy '{strategy_key}'. "
                         f"Available: {list(strategy_map.keys())}")

    module_name, func_name = strategy_map[strategy_key]
    module = __import__(module_name)
    return getattr(module, func_name)


# ─── Phase 2: Shared prediction cache ───────────────────────────────

def _build_predictions(symbols_list, feature_set, device):
    """Train LightGBM once and return per-(fold, symbol) predictions + test data.

    Returns:
        list of dict: each dict has keys
            {symbol, y_pred, returns, sym_test_df, feature_cols}
    """
    import numpy as np
    from src.data.loader import DataLoader
    from src.data.splitter import WalkForwardSplitter
    from src.data.target import TargetGenerator
    from src.features.engine import FeatureEngine
    from src.models.registry import build_model, detect_device
    from src.config_loader import get_training_device

    pipeline_cfg = load_config().get("pipeline", {})
    data_dir = pipeline_cfg.get("data_dir", "../portable_data/vn_stock_ai_dataset_cleaned")
    abs_data_dir = resolve_data_dir(data_dir)

    config = {
        "split": {
            "method": "walk_forward",
            "train_years": pipeline_cfg.get("train_years", 4),
            "test_years": pipeline_cfg.get("test_years", 1),
            "gap_days": 0,
            "first_test_year": pipeline_cfg.get("first_test_year", 2020),
            "last_test_year": pipeline_cfg.get("last_test_year", 2025),
        },
        "target": pipeline_cfg.get("target", {
            "type": "trend_regime", "trend_method": "dual_ma",
            "short_window": 5, "long_window": 20, "classes": 3,
        }),
    }

    if device is None:
        device = get_training_device()
    resolved_device = detect_device(device)
    print(f"    Training device: {resolved_device.upper()}"
          f"{' (auto-detected)' if device == 'auto' else ''}")

    loader = DataLoader(abs_data_dir)
    splitter = WalkForwardSplitter.from_config(config)
    target_gen = TargetGenerator.from_config(config)

    engine = FeatureEngine(feature_set=feature_set)

    raw_df = loader.load_all(symbols=symbols_list)
    df = engine.compute_for_all_symbols(raw_df)
    df = target_gen.generate_for_all_symbols(df)
    feature_cols = engine.get_feature_columns(df)
    df = df.dropna(subset=feature_cols + ["target"])

    results = []
    for window, train_df, test_df in splitter.split(df):
        model = build_model(pipeline_cfg.get("model_type", "lightgbm"), device=device)
        X_train = np.nan_to_num(train_df[feature_cols].values)
        y_train = train_df["target"].values.astype(int)
        model.fit(X_train, y_train)

        for sym in test_df["symbol"].unique():
            if sym not in symbols_list:
                continue
            sym_test = test_df[test_df["symbol"] == sym].reset_index(drop=True)
            if len(sym_test) < 10:
                continue
            X_sym = np.nan_to_num(sym_test[feature_cols].values)
            y_pred = model.predict(X_sym)
            rets = sym_test["return_1d"].values

            results.append({
                "symbol": sym,
                "y_pred": y_pred,
                "returns": rets,
                "sym_test_df": sym_test,
                "feature_cols": feature_cols,
            })

    return results


def _run_backtest_from_cache(prediction_cache, version_key, model_cfg):
    """Run backtest on cached predictions — no ML training needed."""
    import numpy as np

    strategy = model_cfg.get("strategy", version_key)
    backtest_fn = get_backtest_function(strategy)
    mods = model_cfg.get("mods", {})
    params = model_cfg.get("params", {})

    if params:
        base_fn = backtest_fn
        def backtest_fn(y_pred, returns, df_test, feature_cols, **kwargs):
            merged = {**kwargs, **params}
            return base_fn(y_pred, returns, df_test, feature_cols, **merged)

    all_trades = []
    for item in prediction_cache:
        r = backtest_fn(
            item["y_pred"], item["returns"],
            item["sym_test_df"], item["feature_cols"],
            mod_a=mods.get("a", True), mod_b=mods.get("b", True),
            mod_c=mods.get("c", False), mod_d=mods.get("d", False),
            mod_e=mods.get("e", True), mod_f=mods.get("f", True),
            mod_g=mods.get("g", True), mod_h=mods.get("h", True),
            mod_i=mods.get("i", True), mod_j=mods.get("j", True),
        )
        for t in r["trades"]:
            t["symbol"] = item["symbol"]
        all_trades.extend(r["trades"])

    return all_trades


# ─── Phase 4: Fair rule baseline ─────────────────────────────────────

def _run_rule_backtest_fair(symbols_list):
    """Run rule baseline through same walk-forward split windows as ML models."""
    import pandas as pd
    from src.data.loader import DataLoader
    from src.data.splitter import WalkForwardSplitter
    from compare_rule_vs_model import backtest_rule

    pipeline_cfg = load_config().get("pipeline", {})
    data_dir = pipeline_cfg.get("data_dir", "../portable_data/vn_stock_ai_dataset_cleaned")
    abs_data_dir = resolve_data_dir(data_dir)

    config = {
        "split": {
            "method": "walk_forward",
            "train_years": pipeline_cfg.get("train_years", 4),
            "test_years": pipeline_cfg.get("test_years", 1),
            "gap_days": 0,
            "first_test_year": pipeline_cfg.get("first_test_year", 2020),
            "last_test_year": pipeline_cfg.get("last_test_year", 2025),
        },
    }

    loader = DataLoader(abs_data_dir)
    splitter = WalkForwardSplitter.from_config(config)

    raw_df = loader.load_all(symbols=symbols_list)
    date_col = "timestamp" if "timestamp" in raw_df.columns else "date"
    raw_df[date_col] = pd.to_datetime(raw_df[date_col], utc=True)

    all_trades = []
    for window, _train_df, test_df in splitter.split(raw_df, time_col=date_col):
        for sym in test_df["symbol"].unique():
            if sym not in symbols_list:
                continue
            sym_test = test_df[test_df["symbol"] == sym].reset_index(drop=True)
            if len(sym_test) < 50:
                continue
            trades = backtest_rule(sym_test)
            for t in trades:
                t["symbol"] = sym
            all_trades.extend(trades)

    return all_trades


# ─── Phase 5: Metadata ──────────────────────────────────────────────

def _save_trades_with_meta(trades, version_key, symbols_list, feature_set, min_rows):
    """Save trades CSV and companion metadata JSON."""
    import pandas as pd

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, f"trades_{version_key}.csv")

    df = pd.DataFrame(trades)
    df.to_csv(csv_path, index=False)

    meta = {
        "version": version_key,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "generator": "run_pipeline.py",
        "symbols": sorted(symbols_list),
        "n_symbols": len(symbols_list),
        "min_rows": min_rows,
        "feature_set": feature_set or "leading",
        "n_trades": len(trades),
    }
    meta_path = os.path.join(RESULTS_DIR, f"trades_{version_key}.meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return csv_path


def _validate_cache_meta(version_key, symbols_list, min_rows):
    """Check if cached CSV was generated with matching conditions.

    Returns (ok: bool, reason: str).
    """
    meta_path = os.path.join(RESULTS_DIR, f"trades_{version_key}.meta.json")
    if not os.path.exists(meta_path):
        return False, "no metadata"

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    cached_syms = set(meta.get("symbols", []))
    current_syms = set(symbols_list)
    if cached_syms != current_syms:
        diff = len(cached_syms.symmetric_difference(current_syms))
        return False, f"symbol list differs ({diff} changes: cached={len(cached_syms)}, current={len(current_syms)})"

    if meta.get("min_rows", 0) != min_rows:
        return False, f"min_rows differs (cached={meta.get('min_rows')}, current={min_rows})"

    return True, "ok"


# ─── Export ──────────────────────────────────────────────────────────

def run_export(versions=None):
    """Run unified export for specified versions."""
    from src.export.unified_export import main as export_main
    old_argv = sys.argv
    if versions:
        sys.argv = ["unified_export", "--versions", ",".join(versions)]
    else:
        sys.argv = ["unified_export"]
    try:
        export_main()
    finally:
        sys.argv = old_argv


# ─── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Unified Pipeline Runner")
    parser.add_argument("--version", type=str, default="",
                        help="Version key to run (e.g., v24)")
    parser.add_argument("--compare", type=str, default="",
                        help="Comma-separated versions to compare against")
    parser.add_argument("--all", action="store_true",
                        help="Run all active models")
    parser.add_argument("--export-only", action="store_true",
                        help="Skip training, just export existing CSVs")
    parser.add_argument("--export-all", action="store_true",
                        help="Export all models with existing CSVs")
    parser.add_argument("--symbols", type=str, default="",
                        help="Comma-separated symbols (empty = auto from config)")
    parser.add_argument("--min-rows", type=int, default=2000,
                        help="Minimum rows per symbol")
    parser.add_argument("--no-export", action="store_true",
                        help="Skip export step")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip versions that already have trades CSV (smart cache)")
    parser.add_argument("--force", action="store_true",
                        help="Force re-run all versions even if CSV exists")
    parser.add_argument("--device", type=str, default=None,
                        choices=["auto", "gpu", "cuda", "cpu"],
                        help="Training device: auto (detect GPU), gpu, cuda, cpu. "
                             "Default reads from config/base.yaml")
    args = parser.parse_args()

    start = time.time()

    print("=" * 100)
    print("UNIFIED PIPELINE RUNNER")
    print("=" * 100)

    versions_to_run = []
    versions_to_export = []

    if args.export_all:
        print("  Mode: Export all existing CSVs")
        run_export()
        elapsed = time.time() - start
        print(f"\n  Total time: {elapsed:.1f}s")
        return

    if args.export_only and args.version:
        versions_to_export = [args.version]
        if args.compare:
            versions_to_export += [v.strip() for v in args.compare.split(",")]
        print(f"  Mode: Export only ({', '.join(versions_to_export)})")
        run_export(versions_to_export)
        elapsed = time.time() - start
        print(f"\n  Total time: {elapsed:.1f}s")
        return

    # Determine which versions to process
    primary_version = args.version
    compare_versions = []

    if args.all:
        active = get_active_models()
        versions_to_run = list(active.keys())
        compare_versions = versions_to_run
        primary_version = ""
        print(f"  Mode: Run ALL active models ({', '.join(versions_to_run)})")
    elif args.version:
        versions_to_run = [args.version]
        if args.compare:
            compare_versions = [v.strip() for v in args.compare.split(",")]
            versions_to_run += compare_versions
        print(f"  Mode: Run {', '.join(versions_to_run)}")
    else:
        parser.print_help()
        return

    # ── Phase 1: Resolve symbols ONCE for all models ──
    symbols_list = get_pipeline_symbols(
        symbols_arg=args.symbols,
        min_rows_override=args.min_rows,
    )
    if not symbols_list:
        print("  ERROR: No symbols resolved. Check data directory and min_rows.")
        return

    symbols_str = ",".join(symbols_list)
    print(f"\n  Symbols: {len(symbols_list)} (shared across all models)")
    print(f"  Min rows: {args.min_rows}")

    # ── Smart Cache: determine which versions actually need backtest ──
    all_versions = versions_to_run[:]
    versions_to_actually_run = []
    skipped_versions = []
    cache_warnings = []

    for vk in versions_to_run:
        csv_path = os.path.join(RESULTS_DIR, f"trades_{vk}.csv")
        csv_exists = os.path.exists(csv_path)

        if args.force:
            versions_to_actually_run.append(vk)
        elif args.skip_existing and csv_exists:
            meta_ok, reason = _validate_cache_meta(vk, symbols_list, args.min_rows)
            if not meta_ok:
                cache_warnings.append((vk, reason))
            skipped_versions.append(vk)
        elif vk in compare_versions and csv_exists and not args.force:
            meta_ok, reason = _validate_cache_meta(vk, symbols_list, args.min_rows)
            if not meta_ok:
                cache_warnings.append((vk, reason))
            skipped_versions.append(vk)
        else:
            versions_to_actually_run.append(vk)

    if skipped_versions:
        import pandas as pd
        print(f"\n  SMART CACHE: Reusing existing CSV for {len(skipped_versions)} version(s):")
        for vk in skipped_versions:
            csv_path = os.path.join(RESULTS_DIR, f"trades_{vk}.csv")
            df = pd.read_csv(csv_path)
            mod_time = time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(csv_path)))
            print(f"    {vk}: {len(df)} trades (cached {mod_time})")

    if cache_warnings:
        print(f"\n  WARNING: {len(cache_warnings)} cached CSV(s) have mismatched conditions:")
        for vk, reason in cache_warnings:
            print(f"    {vk}: {reason}")
        print(f"    Use --force to re-generate these for a fair comparison.")

    if versions_to_actually_run:
        print(f"\n  WILL RUN backtest for: {', '.join(versions_to_actually_run)}")
    else:
        print(f"\n  All versions have cached results. Nothing to run.")
        print(f"     Use --force to re-run all versions.")

    # ── Phase 2: Group by feature_set, train once per group ──
    if versions_to_actually_run:
        import numpy as np

        groups = defaultdict(list)
        for vk in versions_to_actually_run:
            try:
                model_cfg = get_model_config(vk)
            except KeyError as e:
                print(f"  ERROR: {e}")
                continue
            strategy = model_cfg.get("strategy", vk)
            if strategy == "rule":
                groups["__rule__"].append((vk, model_cfg))
            else:
                feat_set = model_cfg.get("feature_set", "leading")
                groups[feat_set].append((vk, model_cfg))

        # Train ML once per feature_set group
        prediction_caches = {}
        for feat_set, models_in_group in groups.items():
            if feat_set == "__rule__":
                continue
            version_names = [vk for vk, _ in models_in_group]
            print(f"\n{'─' * 100}")
            print(f"  ML TRAINING: feature_set='{feat_set}' "
                  f"(shared by: {', '.join(version_names)})")
            print(f"{'─' * 100}")
            t0 = time.time()
            prediction_caches[feat_set] = _build_predictions(
                symbols_list, feat_set, args.device
            )
            dt = time.time() - t0
            print(f"  Training done in {dt:.1f}s — "
                  f"{len(prediction_caches[feat_set])} (symbol x fold) prediction blocks cached")

        # Run backtests using cached predictions
        for feat_set, models_in_group in groups.items():
            if feat_set == "__rule__":
                continue
            cache = prediction_caches[feat_set]
            for vk, model_cfg in models_in_group:
                print(f"\n  Backtest {vk} ({model_cfg.get('name', '')})...")
                t0 = time.time()
                trades = _run_backtest_from_cache(cache, vk, model_cfg)
                dt = time.time() - t0

                feat = model_cfg.get("feature_set", "leading")
                csv_path = _save_trades_with_meta(
                    trades, vk, symbols_list, feat, args.min_rows
                )
                print(f"  Saved {len(trades)} trades to {csv_path} ({dt:.1f}s)")

                if trades:
                    pnls = np.array([t["pnl_pct"] for t in trades])
                    wins = pnls[pnls > 0]
                    print(f"    Trades={len(trades)}, WR={len(wins)/len(pnls)*100:.1f}%, "
                          f"TotalPnL={pnls.sum():+.1f}%")

        # Run rule baseline through walk-forward split (Phase 4)
        if "__rule__" in groups:
            for vk, model_cfg in groups["__rule__"]:
                print(f"\n  Backtest {vk} (Rule-based, walk-forward split)...")
                t0 = time.time()
                trades = _run_rule_backtest_fair(symbols_list)
                dt = time.time() - t0

                csv_path = _save_trades_with_meta(
                    trades, vk, symbols_list, "rule", args.min_rows
                )
                print(f"  Saved {len(trades)} trades to {csv_path} ({dt:.1f}s)")

                if trades:
                    pnls = np.array([t["pnl_pct"] for t in trades])
                    wins = pnls[pnls > 0]
                    print(f"    Trades={len(trades)}, WR={len(wins)/len(pnls)*100:.1f}%, "
                          f"TotalPnL={pnls.sum():+.1f}%")

    # Export (all versions — including cached ones)
    if not args.no_export:
        print("\n" + "=" * 100)
        print("EXPORT PHASE")
        print("=" * 100)
        run_export(all_versions)

    # Compare (all versions — using CSV files)
    if len(all_versions) > 1:
        print("\n" + "=" * 100)
        print("COMPARISON")
        print("=" * 100)
        from model_manager import cmd_compare
        cmd_compare(argparse.Namespace(versions=",".join(all_versions)))

    elapsed = time.time() - start
    print(f"\n{'=' * 100}")
    print(f"DONE — Total time: {elapsed:.1f}s")
    print(f"  Open visualization/dashboard.html to view results")
    print(f"{'=' * 100}")


if __name__ == "__main__":
    main()

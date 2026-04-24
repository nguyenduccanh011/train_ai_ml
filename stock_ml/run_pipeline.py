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
from src.env import resolve_data_dir, get_results_dir, get_experiment_dir
from src.signal_adapter import canonicalize_predictions, target_fingerprint


def get_backtest_function(strategy_key):
    """Dynamically import the backtest function for a given strategy."""
    strategy_map = {
        "v37a_exit": ("experiments.run_v37a", "backtest_v37a"),
        "v42_a":    ("experiments.run_v42",  "backtest_v42"),
        "v42_base": ("experiments.run_v42",  "backtest_v42"),
        "v39g":   ("experiments.run_v39g",  "backtest_v39g"),
        "v39f":   ("experiments.run_v39f",  "backtest_v39f"),
        "v39d":   ("experiments.run_v39d",  "backtest_v39d"),
        "v39e":   ("experiments.run_v39e",  "backtest_v39e"),
        "v39b":   ("experiments.run_v39b",  "backtest_v39b"),
        "v39a2":  ("experiments.run_v39a2", "backtest_v39a2"),
        "v39a":   ("experiments.run_v39a",  "backtest_v39a"),
        "v38b":   ("experiments.run_v38b", "backtest_v38b"),
        "v38c":   ("experiments.run_v38c", "backtest_v38c"),
        "v38d":   ("experiments.run_v38d", "backtest_v38d"),
        "v38b2":  ("experiments.run_v38b2", "backtest_v38b2"),
        "v38b3":  ("experiments.run_v38b3", "backtest_v38b3"),
        "v38e":   ("experiments.run_v38e",  "backtest_v38e"),
        "v38c2":  ("experiments.run_v38c2", "backtest_v38c2"),
        "v38d2":  ("experiments.run_v38d2", "backtest_v38d2"),
        "v38bc":  ("experiments.run_v38_combos", "backtest_v38bc"),
        "v38bd":  ("experiments.run_v38_combos", "backtest_v38bd"),
        "v38cd":  ("experiments.run_v38_combos", "backtest_v38cd"),
        "v38bcd": ("experiments.run_v38_combos", "backtest_v38bcd"),
        "v37a":   ("experiments.run_v37a", "backtest_v37a"),
        "v37b":   ("experiments.run_v37b", "backtest_v37b"),
        "v37c":   ("experiments.run_v37c", "backtest_v37c"),
        "v37d":   ("experiments.run_v37d", "backtest_v37d"),
        "v36a":   ("experiments.run_v34_final", "backtest_v36a"),
        "v36b":   ("experiments.run_v34_final", "backtest_v36b"),
        "v36c":   ("experiments.run_v34_final", "backtest_v36c"),
        "v35a":   ("experiments.run_v34_final", "backtest_v35a"),
        "v35b":   ("experiments.run_v34_final", "backtest_v35b"),
        "v35c":   ("experiments.run_v34_final", "backtest_v35c"),
        "v34":    ("experiments.run_v34_final", "backtest_v34"),
        "v33":    ("experiments.run_v33_final", "backtest_v33"),
        "v32":    ("experiments.run_v32_final", "backtest_v32"),
        "v31":    ("experiments.run_v31_final", "backtest_v31"),
        "v30":    ("experiments.run_v30", "backtest_v30"),
        "v29":    ("experiments.run_v29", "backtest_v29"),
        "v28":    ("experiments.run_v28", "backtest_v28"),
        "v27":    ("experiments.run_v27", "backtest_v27"),
        "v26":    ("experiments.run_v26", "backtest_v26"),
        "v25":    ("experiments.run_v25", "backtest_v25"),
        "v24":    ("experiments.run_v24", "backtest_v24"),
        "v23":    ("experiments.run_v23_optimal", "backtest_v23"),
        "v22":    ("experiments.run_v22_final", "backtest_v22"),
        "v21":    ("src.strategies.legacy", "backtest_v21"),
        "v20":    ("src.strategies.legacy", "backtest_v20"),
        "v19_4":  ("src.strategies.legacy", "backtest_v19_4"),
        "v19_3":  ("src.strategies.legacy", "backtest_v19_3"),
        "v19_2":  ("src.strategies.legacy", "backtest_v19_2"),
        "v19_1":  ("src.strategies.legacy", "backtest_v19_1"),
        "v19":    ("src.strategies.legacy", "backtest_v19"),
        "v18":    ("src.strategies.legacy", "backtest_v18"),
        "v17":    ("src.strategies.legacy", "backtest_v17"),
        "v16":    ("src.strategies.legacy", "backtest_v16"),
        "v15":    ("src.strategies.legacy", "backtest_v15"),
        "v14":    ("src.strategies.legacy", "backtest_v14"),
        "v13":    ("src.strategies.legacy", "backtest_v13"),
        "v12":    ("src.strategies.legacy", "backtest_v12"),
        "v11":    ("src.strategies.legacy", "backtest_v11"),
        "rule":   ("compare_rule_vs_model", "backtest_rule"),
    }

    if strategy_key not in strategy_map:
        raise ValueError(f"Unknown strategy '{strategy_key}'. "
                         f"Available: {list(strategy_map.keys())}")

    module_name, func_name = strategy_map[strategy_key]
    import importlib
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


# ─── Phase 2: Shared prediction cache ───────────────────────────────

def _build_predictions(symbols_list, feature_set, target_cfg, device, model_type=None,
                       exit_model_cfg=None):
    """Train ML model once and return per-(fold, symbol) predictions + test data.

    Args:
        model_type: override ML model (e.g. "xgboost"). None = read from config.
        exit_model_cfg: dict with keys {forward_window, loss_threshold} to train
            an independent exit model. None = no exit model.

    Returns:
        list of dict: each dict has keys
            {symbol, y_pred, y_pred_exit, returns, sym_test_df, feature_cols}
    """
    import numpy as np
    from pathlib import Path
    from src.data.loader import DataLoader
    from src.data.splitter import WalkForwardSplitter
    from src.data.target import TargetGenerator
    from src.features.engine import FeatureEngine
    from src.models.registry import build_model, detect_device
    from src.config_loader import get_training_device
    from src.cache.feature_cache import FeatureCacheManager
    import src.features.engine as feature_engine_module
    import src.data.target as target_module

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
        "target": target_cfg or pipeline_cfg.get("target", {
            "type": "trend_regime", "trend_method": "dual_ma",
            "short_window": 5, "long_window": 20, "classes": 3,
        }),
    }
    target_type = config["target"].get("type", "trend_regime")
    if str(target_type).lower() == "return_regression":
        raise ValueError(
            "target.type='return_regression' is not supported by current classifier pipeline. "
            "Use a regression model path before enabling this target."
        )

    if device is None:
        device = get_training_device()
    resolved_device = detect_device(device)
    print(f"    Training device: {resolved_device.upper()}"
          f"{' (auto-detected)' if device == 'auto' else ''}")

    effective_model_type = model_type or pipeline_cfg.get("model_type", "lightgbm")

    loader = DataLoader(abs_data_dir)
    splitter = WalkForwardSplitter.from_config(config)
    target_gen = TargetGenerator.from_config(config)

    engine = FeatureEngine(feature_set=feature_set)

    cache_root = Path(get_results_dir()) / "cache" / "features"
    cache_mgr = FeatureCacheManager(str(cache_root))
    code_paths = [feature_engine_module.__file__, target_module.__file__]
    df, cache_key = cache_mgr.load(
        data_dir=abs_data_dir,
        symbols=symbols_list,
        timeframe=loader.timeframe,
        feature_set=feature_set,
        target_config=config.get("target", {}),
        code_paths=code_paths,
    )
    if df is None:
        print(f"    Feature cache: MISS ({feature_set}) key={cache_key[:8]}")
        raw_df = loader.load_all(symbols=symbols_list)
        df = engine.compute_for_all_symbols(raw_df)
        saved_key, saved_fmt = cache_mgr.save(
            df=df,
            data_dir=abs_data_dir,
            symbols=symbols_list,
            timeframe=loader.timeframe,
            feature_set=feature_set,
            target_config=config.get("target", {}),
            code_paths=code_paths,
        )
        print(f"    Feature cache: STORED ({feature_set}) key={saved_key[:8]} format={saved_fmt}")
    else:
        print(f"    Feature cache: HIT ({feature_set}) key={cache_key[:8]}")

    df = target_gen.generate_for_all_symbols(df)

    # Generate exit labels independently if exit_model_cfg provided
    if exit_model_cfg:
        from src.data.target import TargetGenerator as _TG
        df = _TG.generate_exit_labels(
            df,
            forward_window=exit_model_cfg.get("forward_window", 15),
            loss_threshold=exit_model_cfg.get("loss_threshold", 0.05),
        )

    feature_cols = engine.get_feature_columns(df)
    drop_cols = feature_cols + ["target"]
    has_exit = "target_sell" in df.columns
    if has_exit:
        drop_cols.append("target_sell")
    df = df.dropna(subset=drop_cols)

    results = []
    for window, train_df, test_df in splitter.split(df):
        model = build_model(effective_model_type, device=device)
        X_train = np.nan_to_num(train_df[feature_cols].values)
        y_train = train_df["target"].values.astype(int)
        model.fit(X_train, y_train)

        sell_model = None
        if has_exit:
            sell_model = build_model(effective_model_type, device=device)
            sell_model.fit(X_train, train_df["target_sell"].values.astype(int))

        for sym in test_df["symbol"].unique():
            if sym not in symbols_list:
                continue
            sym_test = test_df[test_df["symbol"] == sym].reset_index(drop=True)
            if len(sym_test) < 10:
                continue
            X_sym = np.nan_to_num(sym_test[feature_cols].values)
            y_pred_raw = model.predict(X_sym)
            y_pred = canonicalize_predictions(y_pred_raw, config["target"])
            rets = sym_test["return_1d"].values

            # V37c: capture proba + class mapping for per-profile threshold tuning
            y_proba = None
            classes = None
            try:
                if hasattr(model, "predict_proba"):
                    y_proba = model.predict_proba(X_sym)
                    final_est = model.steps[-1][1] if hasattr(model, "steps") else model
                    classes = list(final_est.classes_)
            except Exception:
                y_proba = None

            results.append({
                "symbol": sym,
                "y_pred": y_pred,
                "y_pred_exit": (sell_model.predict(X_sym).astype(int) if sell_model is not None else None),
                "y_proba": y_proba,
                "classes": classes,
                "returns": rets,
                "sym_test_df": sym_test,
                "feature_cols": feature_cols,
            })

    return results


def _apply_proba_thresholds(item, proba_thresholds):
    """V37c: rebuild y_pred from y_proba using per-profile threshold.

    proba_thresholds: dict {profile_name: float}. Default profile threshold
    ("balanced") is used as fallback. Returns modified y_pred numpy array.
    """
    import numpy as np
    from src.backtest.defaults import SYMBOL_PROFILES

    proba = item.get("y_proba")
    classes = item.get("classes")
    if proba is None or classes is None:
        return item["y_pred"]

    sym = str(item.get("symbol", "?"))
    profile = SYMBOL_PROFILES.get(sym, "balanced")
    thresh = proba_thresholds.get(profile, proba_thresholds.get("balanced", 0.35))

    # Find buy / sell class indexes (labels from target config: 1=buy, -1=sell, 0=neutral)
    def _idx(label):
        try:
            return classes.index(label)
        except ValueError:
            return None

    buy_i = _idx(1)
    sell_i = _idx(-1)
    n = proba.shape[0]
    out = np.zeros(n, dtype=int)
    if buy_i is not None:
        out[proba[:, buy_i] >= thresh] = 1
    if sell_i is not None:
        # Sell threshold symmetric (same value). Only overwrite where buy didn't fire.
        sell_mask = (proba[:, sell_i] >= thresh) & (out == 0)
        out[sell_mask] = -1
    return out


def _run_backtest_from_cache(prediction_cache, version_key, model_cfg):
    """Run backtest on cached predictions — no ML training needed."""
    import inspect
    import numpy as np

    strategy = model_cfg.get("strategy", version_key)
    backtest_fn = get_backtest_function(strategy)
    mods = model_cfg.get("mods", {})
    params = model_cfg.get("params", {})
    proba_thresholds = model_cfg.get("proba_thresholds")  # V37c

    # Detect which mod_* kwargs this backtest function accepts
    sig_params = set(inspect.signature(backtest_fn).parameters.keys())
    accepts_mods = "mod_a" in sig_params

    if params:
        base_fn = backtest_fn
        def backtest_fn(y_pred, returns, df_test, feature_cols, **kwargs):
            merged = {**kwargs, **params}
            return base_fn(y_pred, returns, df_test, feature_cols, **merged)

    all_trades = []
    # Build mod kwargs filtered to only params the function actually accepts
    all_mod_kwargs = {
        "mod_a": mods.get("a", True), "mod_b": mods.get("b", True),
        "mod_c": mods.get("c", False), "mod_d": mods.get("d", False),
        "mod_e": mods.get("e", True),  "mod_f": mods.get("f", True),
        "mod_g": mods.get("g", True),  "mod_h": mods.get("h", True),
        "mod_i": mods.get("i", True),  "mod_j": mods.get("j", True),
    }
    mod_kwargs = {k: v for k, v in all_mod_kwargs.items() if k in sig_params}

    for item in prediction_cache:
        y_pred_eff = (
            _apply_proba_thresholds(item, proba_thresholds)
            if proba_thresholds else item["y_pred"]
        )
        extra_kwargs = {}
        y_pred_exit = item.get("y_pred_exit")
        if y_pred_exit is not None:
            extra_kwargs["y_pred_exit"] = y_pred_exit
        r = backtest_fn(
            y_pred_eff, item["returns"],
            item["sym_test_df"], item["feature_cols"],
            **mod_kwargs, **extra_kwargs,
        )
        for t in r["trades"]:
            t["symbol"] = item["symbol"]
        all_trades.extend(r["trades"])

    return all_trades


# ─── Phase 4: Fair rule baseline ─────────────────────────────────────

def _run_rule_backtest_fair(symbols_list):
    """Run rule baseline on full test period (2020+) per symbol.

    Rule strategy has no training phase, so walk-forward split is not needed
    and would incorrectly drop open positions at fold boundaries.
    """
    import pandas as pd
    from src.data.loader import DataLoader
    from compare_rule_vs_model import backtest_rule

    pipeline_cfg = load_config().get("pipeline", {})
    data_dir = pipeline_cfg.get("data_dir", "../portable_data/vn_stock_ai_dataset_cleaned")
    abs_data_dir = resolve_data_dir(data_dir)
    first_test_year = pipeline_cfg.get("first_test_year", 2020)

    loader = DataLoader(abs_data_dir)
    raw_df = loader.load_all(symbols=symbols_list)
    date_col = "timestamp" if "timestamp" in raw_df.columns else "date"
    raw_df[date_col] = pd.to_datetime(raw_df[date_col], utc=True)

    all_trades = []
    for sym in symbols_list:
        sym_data = raw_df[raw_df["symbol"] == sym].copy()
        sym_data = sym_data.sort_values(date_col).reset_index(drop=True)
        sym_test = sym_data[sym_data[date_col].dt.year >= first_test_year].reset_index(drop=True)
        if len(sym_test) < 50:
            continue
        trades = backtest_rule(sym_test)
        for t in trades:
            t["symbol"] = sym
        all_trades.extend(trades)

    return all_trades


# ─── Phase 5: Metadata ──────────────────────────────────────────────

def _save_trades_with_meta(
    trades, version_key, symbols_list, feature_set, min_rows,
    target_cfg=None, model_type="lightgbm", results_dir=None, exit_model_cfg=None,
):
    """Save trades CSV and companion metadata JSON."""
    import pandas as pd

    out_dir = results_dir or get_results_dir()
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"trades_{version_key}.csv")

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
        "model_type": model_type,
        "target_config": target_cfg or {},
        "target_fingerprint": target_fingerprint(target_cfg or {}),
        "n_trades": len(trades),
    }
    if exit_model_cfg:
        meta["exit_model_config"] = exit_model_cfg
    meta_path = os.path.join(out_dir, f"trades_{version_key}.meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return csv_path


def _validate_cache_meta(
    version_key, symbols_list, min_rows,
    expected_feature_set=None,
    expected_target_fingerprint=None,
    strategy_key=None,
    results_dir=None,
):
    """Check if cached CSV was generated with matching conditions.

    Returns (ok: bool, reason: str).
    """
    out_dir = results_dir or get_results_dir()
    meta_path = os.path.join(out_dir, f"trades_{version_key}.meta.json")
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

    if expected_feature_set is not None:
        cached_feature_set = meta.get("feature_set")
        if cached_feature_set != expected_feature_set:
            return False, (
                "feature_set differs "
                f"(cached={cached_feature_set}, current={expected_feature_set})"
            )

    # Rule baseline is independent from ML target definition.
    if strategy_key != "rule" and expected_target_fingerprint is not None:
        cached_fp = meta.get("target_fingerprint")
        if cached_fp != expected_target_fingerprint:
            return False, "target config differs (fingerprint mismatch)"

    return True, "ok"


# ─── Experiment helpers ──────────────────────────────────────────────

def _make_experiment_key(feature_set: str, ml_model: str) -> str:
    """Build subfolder name from feature_set and ml_model."""
    return f"{feature_set}__{ml_model}"


def _save_experiment_json(exp_dir, experiment_key, feature_set, ml_model,
                          target_cfg, target_fp, symbols_list, versions):
    """Write experiment.json snapshot to the experiment subfolder."""
    meta = {
        "experiment_key": experiment_key,
        "feature_set": feature_set,
        "ml_model": ml_model,
        "target_config": target_cfg or {},
        "target_fingerprint": target_fp,
        "symbols": sorted(symbols_list),
        "n_symbols": len(symbols_list),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "versions": sorted(versions),
    }
    path = os.path.join(exp_dir, "experiment.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def _run_matrix(versions_to_run, feature_sets, ml_models, symbols_list,
                target_cfg, target_fp, pipeline_cfg, args):
    """Run all combinations of feature_set × ml_model for the given versions.

    Trains ML once per (feature_set, ml_model) combination, then runs backtest
    for each version in that group. Results saved to results/{feat}__{ml}/.
    """
    import numpy as np
    from itertools import product

    for feat, ml in product(feature_sets, ml_models):
        experiment_key = _make_experiment_key(feat, ml)
        exp_dir = get_experiment_dir(experiment_key)
        os.makedirs(exp_dir, exist_ok=True)

        print(f"\n{'=' * 100}")
        print(f"  EXPERIMENT: {experiment_key}")
        print(f"  feature_set={feat}, ml_model={ml}")
        print(f"  versions: {', '.join(versions_to_run)}")
        print(f"  output: results/{experiment_key}/")
        print(f"{'=' * 100}")

        # Smart cache: determine which versions need backtest in this experiment
        versions_needed = []
        versions_cached = []
        for vk in versions_to_run:
            csv_path = os.path.join(exp_dir, f"trades_{vk}.csv")
            if args.force or not os.path.exists(csv_path):
                versions_needed.append(vk)
            else:
                meta_ok, reason = _validate_cache_meta(
                    vk, symbols_list, args.min_rows,
                    expected_feature_set=feat,
                    expected_target_fingerprint=target_fp,
                    results_dir=exp_dir,
                )
                if meta_ok:
                    versions_cached.append(vk)
                else:
                    print(f"    Cache invalid for {vk} ({reason}) — re-running")
                    versions_needed.append(vk)

        if versions_cached:
            print(f"\n  CACHE: reusing {', '.join(versions_cached)}")

        if not versions_needed:
            print(f"  All versions cached. Use --force to re-run.")
            continue

        # Train predictions once for this (feat, ml) combo
        model_cfg_map = {}
        groups = defaultdict(list)
        for vk in versions_needed:
            try:
                model_cfg = get_model_config(vk)
            except KeyError as e:
                print(f"  ERROR: {e}")
                continue
            model_cfg_map[vk] = model_cfg
            strategy = model_cfg.get("strategy", vk)
            if strategy == "rule":
                groups["__rule__"].append((vk, model_cfg))
            else:
                groups["__ml__"].append((vk, model_cfg))

        prediction_cache = None
        if "__ml__" in groups:
            version_names = [vk for vk, _ in groups["__ml__"]]
            print(f"\n  ML TRAINING: feature_set='{feat}', model='{ml}' "
                  f"(for: {', '.join(version_names)})")
            t0 = time.time()
            prediction_cache = _build_predictions(
                symbols_list, feat, target_cfg, args.device, model_type=ml
            )
            dt = time.time() - t0
            print(f"  Training done in {dt:.1f}s — "
                  f"{len(prediction_cache)} (symbol x fold) blocks cached")

        # Run backtests
        run_versions = []
        for vk, model_cfg in groups.get("__ml__", []):
            print(f"\n  Backtest {vk} ({model_cfg.get('name', '')})...")
            t0 = time.time()
            trades = _run_backtest_from_cache(prediction_cache, vk, model_cfg)
            dt = time.time() - t0

            csv_path = _save_trades_with_meta(
                trades, vk, symbols_list, feat, args.min_rows,
                target_cfg=target_cfg, model_type=ml, results_dir=exp_dir,
            )
            print(f"  Saved {len(trades)} trades to {csv_path} ({dt:.1f}s)")
            if trades:
                pnls = np.array([t["pnl_pct"] for t in trades])
                wins = pnls[pnls > 0]
                print(f"    WR={len(wins)/len(pnls)*100:.1f}%, TotalPnL={pnls.sum():+.1f}%")
            run_versions.append(vk)

        for vk, model_cfg in groups.get("__rule__", []):
            print(f"\n  Backtest {vk} (Rule-based)...")
            t0 = time.time()
            trades = _run_rule_backtest_fair(symbols_list)
            dt = time.time() - t0
            csv_path = _save_trades_with_meta(
                trades, vk, symbols_list, "rule", args.min_rows,
                target_cfg=target_cfg, model_type="rule", results_dir=exp_dir,
            )
            print(f"  Saved {len(trades)} trades to {csv_path} ({dt:.1f}s)")
            run_versions.append(vk)

        # Save experiment snapshot
        all_done = list(set(versions_to_run))  # includes cached
        _save_experiment_json(
            exp_dir, experiment_key, feat, ml,
            target_cfg, target_fp, symbols_list, all_done,
        )

        # Compare within this experiment
        all_exp_versions = versions_to_run[:]
        if len(all_exp_versions) > 1:
            print(f"\n{'─' * 100}")
            print(f"  COMPARISON — {experiment_key}")
            print(f"{'─' * 100}")
            from model_manager import cmd_compare
            cmd_compare(argparse.Namespace(
                versions=",".join(all_exp_versions),
                experiment=experiment_key,
            ))


# ─── Export ──────────────────────────────────────────────────────────

def run_export(versions=None, experiment=None):
    """Run unified export for specified versions."""
    from src.export.unified_export import main as export_main
    old_argv = sys.argv
    args_list = ["unified_export"]
    if versions:
        args_list += ["--versions", ",".join(versions)]
    if experiment:
        args_list += ["--experiment", experiment]
    sys.argv = args_list
    try:
        export_main()
    finally:
        sys.argv = old_argv


# ─── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Unified Pipeline Runner")
    parser.add_argument("--version", type=str, default="",
                        help="Version key to run (e.g., v24) — backward compat")
    parser.add_argument("--versions", type=str, default="",
                        help="Comma-separated version keys for matrix mode (e.g., v26,v27)")
    parser.add_argument("--compare", type=str, default="",
                        help="Comma-separated versions to compare against")
    parser.add_argument("--all", action="store_true",
                        help="Run all active models")
    parser.add_argument("--feature-sets", type=str, default="",
                        help="Comma-separated feature sets for matrix mode (e.g., leading,leading_v2)")
    parser.add_argument("--ml-models", type=str, default="",
                        help="Comma-separated ML models for matrix mode (e.g., lightgbm,xgboost)")
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

    # ── Matrix mode: --versions × --feature-sets × --ml-models ──
    feature_sets = [f.strip() for f in args.feature_sets.split(",") if f.strip()]
    ml_models = [m.strip() for m in args.ml_models.split(",") if m.strip()]
    matrix_versions = [v.strip() for v in args.versions.split(",") if v.strip()]

    if feature_sets or ml_models or matrix_versions:
        # Matrix mode — resolve versions
        if not matrix_versions:
            if args.version:
                matrix_versions = [args.version]
                if args.compare:
                    matrix_versions += [v.strip() for v in args.compare.split(",")]
            elif args.all:
                matrix_versions = list(get_active_models().keys())
            else:
                print("  ERROR: matrix mode requires --versions, --version, or --all")
                return
        if not feature_sets:
            print("  ERROR: --feature-sets is required in matrix mode")
            return
        if not ml_models:
            pipeline_cfg_tmp = load_config().get("pipeline", {})
            ml_models = [pipeline_cfg_tmp.get("model_type", "lightgbm")]

        print(f"  Mode: MATRIX")
        print(f"    versions:      {', '.join(matrix_versions)}")
        print(f"    feature_sets:  {', '.join(feature_sets)}")
        print(f"    ml_models:     {', '.join(ml_models)}")
        print(f"    combinations:  {len(feature_sets) * len(ml_models)}")

        cfg_all = load_config()
        pipeline_cfg = cfg_all.get("pipeline", {})
        target_cfg = pipeline_cfg.get("target", {
            "type": "trend_regime", "trend_method": "dual_ma",
            "short_window": 5, "long_window": 20, "classes": 3,
        })
        target_fp = target_fingerprint(target_cfg)

        symbols_list = get_pipeline_symbols(
            symbols_arg=args.symbols,
            min_rows_override=args.min_rows,
        )
        if not symbols_list:
            print("  ERROR: No symbols resolved.")
            return

        print(f"\n  Symbols: {len(symbols_list)} (shared across all experiments)")

        _run_matrix(
            matrix_versions, feature_sets, ml_models,
            symbols_list, target_cfg, target_fp, pipeline_cfg, args,
        )

        elapsed = time.time() - start
        print(f"\n{'=' * 100}")
        print(f"MATRIX DONE — {len(feature_sets) * len(ml_models)} experiment(s), "
              f"{len(matrix_versions)} version(s) each. Total time: {elapsed:.1f}s")
        print(f"{'=' * 100}")
        return

    # Determine which versions to process (backward compat — flat results/)
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
    cfg_all = load_config()
    pipeline_cfg = cfg_all.get("pipeline", {})
    target_cfg = pipeline_cfg.get("target", {
        "type": "trend_regime", "trend_method": "dual_ma",
        "short_window": 5, "long_window": 20, "classes": 3,
    })
    target_fp = target_fingerprint(target_cfg)

    symbols_list = get_pipeline_symbols(
        symbols_arg=args.symbols,
        min_rows_override=args.min_rows,
    )
    if not symbols_list:
        print("  ERROR: No symbols resolved. Check data directory and min_rows.")
        return

    print(f"\n  Symbols: {len(symbols_list)} (shared across all models)")
    print(f"  Min rows: {args.min_rows}")
    print(f"  Target: {target_cfg.get('type', 'trend_regime')} (cache fingerprint enabled)")

    # ── Smart Cache: determine which versions actually need backtest ──
    all_versions = versions_to_run[:]
    versions_to_actually_run = []
    skipped_versions = []
    invalidated_cache = []

    model_cfg_map = {}
    for vk in versions_to_run:
        try:
            model_cfg_map[vk] = get_model_config(vk)
        except KeyError:
            model_cfg_map[vk] = {}

    def _resolve_target_for(vk):
        cfg = model_cfg_map.get(vk, {})
        return cfg.get("target") or target_cfg

    for vk in versions_to_run:
        csv_path = os.path.join(get_results_dir(), f"trades_{vk}.csv")
        csv_exists = os.path.exists(csv_path)
        model_cfg = model_cfg_map.get(vk, {})
        strategy_key = model_cfg.get("strategy", vk)
        expected_feature_set = "rule" if strategy_key == "rule" else model_cfg.get("feature_set", "leading")
        version_target = _resolve_target_for(vk)
        version_target_fp = target_fingerprint(version_target)

        if args.force:
            versions_to_actually_run.append(vk)
            continue

        if not csv_exists:
            versions_to_actually_run.append(vk)
            continue

        should_consider_cache = args.skip_existing or (vk in compare_versions)
        if not should_consider_cache:
            versions_to_actually_run.append(vk)
            continue

        meta_ok, reason = _validate_cache_meta(
            vk, symbols_list, args.min_rows,
            expected_feature_set=expected_feature_set,
            expected_target_fingerprint=version_target_fp,
            strategy_key=strategy_key,
        )
        if meta_ok:
            skipped_versions.append(vk)
        else:
            invalidated_cache.append((vk, reason))
            versions_to_actually_run.append(vk)

    if skipped_versions:
        import pandas as pd
        print(f"\n  SMART CACHE: Reusing existing CSV for {len(skipped_versions)} version(s):")
        for vk in skipped_versions:
            csv_path = os.path.join(get_results_dir(), f"trades_{vk}.csv")
            df = pd.read_csv(csv_path)
            mod_time = time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(csv_path)))
            print(f"    {vk}: {len(df)} trades (cached {mod_time})")

    if invalidated_cache:
        print(f"\n  CACHE INVALIDATED: {len(invalidated_cache)} version(s) will be re-run:")
        for vk, reason in invalidated_cache:
            print(f"    {vk}: {reason}")
        print(f"    These mismatched CSVs are NOT reused to keep comparison fair.")

    if versions_to_actually_run:
        print(f"\n  WILL RUN backtest for: {', '.join(versions_to_actually_run)}")
    else:
        print(f"\n  All versions have cached results. Nothing to run.")
        print(f"     Use --force to re-run all versions.")

    # ── Phase 2: Group by (feature_set, target_fingerprint), train once per group ──
    if versions_to_actually_run:
        import numpy as np

        groups = defaultdict(list)
        group_targets = {}
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
                ver_tgt = model_cfg.get("target") or target_cfg
                ver_fp = target_fingerprint(ver_tgt)
                ver_ml = model_cfg.get("model_type")
                exit_cfg = model_cfg.get("exit_model") or {}
                exit_fp = json.dumps(exit_cfg, sort_keys=True) if exit_cfg.get("enabled") else ""
                gkey = (feat_set, ver_fp, ver_ml, exit_fp)
                groups[gkey].append((vk, model_cfg))
                group_targets[gkey] = ver_tgt

        # Train ML once per (feature_set, target, model_type, exit_model) group
        prediction_caches = {}
        for gkey, models_in_group in groups.items():
            if gkey == "__rule__":
                continue
            feat_set, _fp, ver_ml, exit_fp = gkey
            ver_tgt = group_targets[gkey]
            ver_exit_cfg = json.loads(exit_fp) if exit_fp else None
            version_names = [vk for vk, _ in models_in_group]
            exit_label = f" +exit_model(fw={ver_exit_cfg['forward_window']})" if ver_exit_cfg else ""
            print(f"\n{'─' * 100}")
            print(f"  ML TRAINING: feature_set='{feat_set}' target='{ver_tgt.get('type')}' "
                  f"model='{ver_ml or 'pipeline-default'}'{exit_label} "
                  f"(shared by: {', '.join(version_names)})")
            print(f"{'─' * 100}")
            t0 = time.time()
            prediction_caches[gkey] = _build_predictions(
                symbols_list, feat_set, ver_tgt, args.device,
                model_type=ver_ml, exit_model_cfg=ver_exit_cfg,
            )
            dt = time.time() - t0
            print(f"  Training done in {dt:.1f}s — "
                  f"{len(prediction_caches[gkey])} (symbol x fold) prediction blocks cached")

        # Run backtests using cached predictions
        for gkey, models_in_group in groups.items():
            if gkey == "__rule__":
                continue
            cache = prediction_caches[gkey]
            ver_tgt = group_targets[gkey]
            for vk, model_cfg in models_in_group:
                print(f"\n  Backtest {vk} ({model_cfg.get('name', '')})...")
                t0 = time.time()
                trades = _run_backtest_from_cache(cache, vk, model_cfg)
                dt = time.time() - t0

                feat = model_cfg.get("feature_set", "leading")
                ver_exit_cfg_save = json.loads(gkey[3]) if gkey[3] else None
                csv_path = _save_trades_with_meta(
                    trades, vk, symbols_list, feat, args.min_rows,
                    target_cfg=ver_tgt,
                    model_type=pipeline_cfg.get("model_type", "lightgbm"),
                    exit_model_cfg=ver_exit_cfg_save,
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
                    trades, vk, symbols_list, "rule", args.min_rows,
                    target_cfg=target_cfg,
                    model_type="rule",
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

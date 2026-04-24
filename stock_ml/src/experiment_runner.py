"""
Shared experiment runners for version comparison scripts.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from compare_rule_vs_model import backtest_rule
from src.cache.feature_cache import FeatureCacheManager
from src.config_loader import get_training_device, load_config
from src.data.loader import DataLoader
from src.data.splitter import WalkForwardSplitter
from src.data.target import TargetGenerator
from src.env import get_results_dir, resolve_data_dir
import src.data.target as target_module
from src.features.engine import FeatureEngine
import src.features.engine as feature_engine_module
from src.models.registry import build_model, detect_device
from src.signal_adapter import canonicalize_predictions


def _build_pipeline_config():
    cfg = load_config().get("pipeline", {})
    return {
        "split": {
            "method": "walk_forward",
            "train_years": cfg.get("train_years", 4),
            "test_years": cfg.get("test_years", 1),
            "gap_days": 0,
            "first_test_year": cfg.get("first_test_year", 2020),
            "last_test_year": cfg.get("last_test_year", 2025),
        },
        "target": cfg.get(
            "target",
            {
                "type": "trend_regime",
                "trend_method": "dual_ma",
                "short_window": 5,
                "long_window": 20,
                "classes": 3,
            },
        ),
    }


def run_test(
    symbols_str,
    mod_a,
    mod_b,
    mod_c=False,
    mod_d=False,
    mod_e=False,
    mod_f=False,
    mod_g=False,
    mod_h=False,
    mod_i=False,
    mod_j=False,
    backtest_fn=None,
    device=None,
    feature_set="leading_v2",
    target_override=None,
    train_exit_model=False,
):
    if backtest_fn is None:
        raise ValueError("run_test requires backtest_fn")

    pipeline_cfg = load_config().get("pipeline", {})
    data_dir = resolve_data_dir(
        pipeline_cfg.get("data_dir", "../portable_data/vn_stock_ai_dataset_cleaned")
    )
    config = _build_pipeline_config()
    if target_override is not None:
        config["target"] = target_override
    if str(config["target"].get("type", "trend_regime")).lower() == "return_regression":
        raise ValueError(
            "run_test currently supports classification targets only. "
            "Switch to classifier-friendly target.type."
        )

    if device is None:
        device = get_training_device()
    resolved_device = detect_device(device)
    print(
        f"    Training device: {resolved_device.upper()}"
        f"{' (auto-detected)' if device == 'auto' else ''}"
    )

    pick = [s.strip() for s in symbols_str.split(",") if s.strip()]
    loader = DataLoader(data_dir)
    available_symbols = set(loader.symbols)
    if pick:
        pick = [s for s in pick if s in available_symbols]
    else:
        pick = list(loader.symbols)
    if not pick:
        raise ValueError("No valid symbols to run_test after filtering by dataset.")
    pick_set = set(pick)

    splitter = WalkForwardSplitter.from_config(config)
    target_gen = TargetGenerator.from_config(config)
    engine = FeatureEngine(feature_set=feature_set)

    cache_root = Path(get_results_dir()) / "cache" / "features"
    cache_mgr = FeatureCacheManager(str(cache_root))
    code_paths = [feature_engine_module.__file__, target_module.__file__, __file__]
    df, cache_key = cache_mgr.load(
        data_dir=data_dir,
        symbols=pick,
        timeframe=loader.timeframe,
        feature_set=feature_set,
        target_config=config.get("target", {}),
        code_paths=code_paths,
    )
    if df is None:
        print(f"    Feature cache: MISS ({feature_set}) key={cache_key[:8]}")
        raw_df = loader.load_all(symbols=pick)
        df = engine.compute_for_all_symbols(raw_df)
        saved_key, saved_fmt = cache_mgr.save(
            df=df,
            data_dir=data_dir,
            symbols=pick,
            timeframe=loader.timeframe,
            feature_set=feature_set,
            target_config=config.get("target", {}),
            code_paths=code_paths,
        )
        print(
            f"    Feature cache: STORED ({feature_set}) "
            f"key={saved_key[:8]} format={saved_fmt}"
        )
    else:
        print(f"    Feature cache: HIT ({feature_set}) key={cache_key[:8]}")

    df = target_gen.generate_for_all_symbols(df)
    feature_cols = engine.get_feature_columns(df)
    drop_cols = feature_cols + ["target"]
    if "target_sell" in df.columns:
        drop_cols.append("target_sell")
    df = df.dropna(subset=drop_cols)

    all_trades = []
    for _, train_df, test_df in splitter.split(df):
        model = build_model("lightgbm", device=device)
        X_train = np.nan_to_num(train_df[feature_cols].values)
        y_train = train_df["target"].values.astype(int)
        model.fit(X_train, y_train)

        has_exit = train_exit_model and "target_sell" in train_df.columns
        model_exit = None
        if has_exit:
            model_exit = build_model("lightgbm", device=device)
            model_exit.fit(X_train, train_df["target_sell"].values.astype(int))

        for sym in test_df["symbol"].unique():
            if sym not in pick_set:
                continue
            sym_test = test_df[test_df["symbol"] == sym].reset_index(drop=True)
            if len(sym_test) < 10:
                continue
            X_sym = np.nan_to_num(sym_test[feature_cols].values)
            y_pred = model.predict(X_sym)
            y_pred = canonicalize_predictions(y_pred, config["target"])
            rets = sym_test["return_1d"].values

            y_pred_exit = None
            if model_exit is not None:
                y_pred_exit = model_exit.predict(X_sym).astype(int)

            result = backtest_fn(
                y_pred,
                rets,
                sym_test,
                feature_cols,
                mod_a=mod_a,
                mod_b=mod_b,
                mod_c=mod_c,
                mod_d=mod_d,
                mod_e=mod_e,
                mod_f=mod_f,
                mod_g=mod_g,
                mod_h=mod_h,
                mod_i=mod_i,
                mod_j=mod_j,
                y_pred_exit=y_pred_exit,
            )
            for trade in result["trades"]:
                trade["symbol"] = sym
            all_trades.extend(result["trades"])

    return all_trades


def run_rule_test(symbols_str):
    pipeline_cfg = load_config().get("pipeline", {})
    data_dir = resolve_data_dir(
        pipeline_cfg.get("data_dir", "../portable_data/vn_stock_ai_dataset_cleaned")
    )
    config = _build_pipeline_config()
    pick = [s.strip() for s in symbols_str.split(",") if s.strip()]
    loader = DataLoader(data_dir)
    splitter = WalkForwardSplitter.from_config(config)
    if pick:
        symbols = [s for s in pick if s in loader.symbols]
    else:
        symbols = list(loader.symbols)
    raw_df = loader.load_all(symbols=symbols)

    date_col = "timestamp" if "timestamp" in raw_df.columns else "date"
    raw_df[date_col] = pd.to_datetime(raw_df[date_col], utc=True)

    all_trades = []
    for _, _train_df, test_df in splitter.split(raw_df, time_col=date_col):
        for sym in test_df["symbol"].unique():
            if sym not in symbols:
                continue
            sym_test = test_df[test_df["symbol"] == sym].reset_index(drop=True)
            if len(sym_test) < 50:
                continue
            trades = backtest_rule(sym_test)
            for trade in trades:
                trade["symbol"] = sym
            all_trades.extend(trades)

    return all_trades

"""
Shared experiment runners for version comparison scripts.
"""
import numpy as np
import pandas as pd

from compare_rule_vs_model import backtest_rule
from src.config_loader import get_training_device, load_config
from src.data.loader import DataLoader
from src.data.splitter import WalkForwardSplitter
from src.data.target import TargetGenerator
from src.env import resolve_data_dir
from src.features.engine import FeatureEngine
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
):
    if backtest_fn is None:
        raise ValueError("run_test requires backtest_fn")

    pipeline_cfg = load_config().get("pipeline", {})
    data_dir = resolve_data_dir(
        pipeline_cfg.get("data_dir", "../portable_data/vn_stock_ai_dataset_cleaned")
    )
    config = _build_pipeline_config()
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
    splitter = WalkForwardSplitter.from_config(config)
    target_gen = TargetGenerator.from_config(config)

    raw_df = loader.load_all(symbols=pick)
    engine = FeatureEngine(feature_set=feature_set)
    df = engine.compute_for_all_symbols(raw_df)
    df = target_gen.generate_for_all_symbols(df)
    feature_cols = engine.get_feature_columns(df)
    df = df.dropna(subset=feature_cols + ["target"])

    all_trades = []
    for _, train_df, test_df in splitter.split(df):
        model = build_model("lightgbm", device=device)
        X_train = np.nan_to_num(train_df[feature_cols].values)
        y_train = train_df["target"].values.astype(int)
        model.fit(X_train, y_train)

        for sym in test_df["symbol"].unique():
            if sym not in pick:
                continue
            sym_test = test_df[test_df["symbol"] == sym].reset_index(drop=True)
            if len(sym_test) < 10:
                continue
            X_sym = np.nan_to_num(sym_test[feature_cols].values)
            y_pred = model.predict(X_sym)
            y_pred = canonicalize_predictions(y_pred, config["target"])
            rets = sym_test["return_1d"].values

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
    pick = [s.strip() for s in symbols_str.split(",") if s.strip()]
    loader = DataLoader(data_dir)
    symbols = [s for s in pick if s in loader.symbols]
    raw_df = loader.load_all(symbols=symbols)

    all_trades = []
    for sym in symbols:
        sym_data = raw_df[raw_df["symbol"] == sym].copy()
        date_col = "timestamp" if "timestamp" in sym_data.columns else "date"
        sym_data = sym_data.sort_values(date_col).reset_index(drop=True)
        sym_data[date_col] = pd.to_datetime(sym_data[date_col])
        sym_test = sym_data[sym_data[date_col] >= "2020-01-01"].reset_index(drop=True)
        if len(sym_test) < 50:
            continue
        trades = backtest_rule(sym_test)
        for trade in trades:
            trade["symbol"] = sym
        all_trades.extend(trades)

    return all_trades

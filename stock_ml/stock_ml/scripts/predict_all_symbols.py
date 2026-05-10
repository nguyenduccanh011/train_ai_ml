"""
Script để chạy prediction trên tất cả symbols và tìm tín hiệu mua/bán trong 10 ngày gần nhất
Sử dụng model top 1: v22_exit_ablation_round42 (config 748af6fb17f4)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datetime import datetime

import pandas as pd


def load_top_model_config():
    """Load config của model top 1"""
    from src.pipeline import ExperimentConfig

    config_path = (
        ROOT
        / "results"
        / "experiments"
        / "v22_exit_ablation_round42"
        / "v22_exit_ablation_round42_signals_features-leading-signals_entry_model_type-random_forest-signals_target-earlyv2_fw21_g033125_l0165625-exit_model-exit_fw21_l03725-fusion-peak_dist_only"
        / "config.resolved.yaml"
    )

    if not config_path.exists():
        raise FileNotFoundError(f"Config không tồn tại: {config_path}")

    cfg = ExperimentConfig.from_yaml(config_path)
    print(f"✓ Loaded config: {cfg.name}")
    print(f"  - Strategy: {cfg.strategy}")
    print(f"  - Feature set: {cfg.feature_set()}")
    print(f"  - Entry model: {cfg.signals.entry_model.type}")
    print(
        f"  - Exit model: {cfg.signals.exit_model.type if cfg.signals.exit_model.enabled else 'disabled'}"
    )
    return cfg


def get_all_symbols():
    """Lấy danh sách tất cả symbols từ data directory"""
    from src.config_loader import get_pipeline_config, resolve_data_dir

    pipeline_cfg = get_pipeline_config()
    data_dir = resolve_data_dir(
        pipeline_cfg.get("data_dir", "../portable_data/vn_stock_ai_dataset_cleaned")
    )

    # Lấy tất cả symbols từ thư mục data
    data_path = Path(data_dir)
    symbols = []

    # Tìm trong thư mục ohlcv_daily
    ohlcv_dir = data_path / "ohlcv_daily"
    if ohlcv_dir.exists():
        for file in ohlcv_dir.glob("symbol=*/timeframe=1D/data.csv"):
            symbol = file.parent.parent.name.replace("symbol=", "")
            symbols.append(symbol)

    symbols = sorted(set(symbols))
    print(f"✓ Found {len(symbols)} symbols")
    return symbols


def run_prediction_for_symbol(cfg, symbol: str):
    """Chạy prediction cho 1 symbol"""
    import src.data.target as target_module
    import src.features.engine as feature_engine_module
    from src.cache.feature_cache import FeatureCacheManager
    from src.components.exit_models.registry import get_exit_model
    from src.components.models.registry import get_model
    from src.config_loader import load_config, resolve_data_dir
    from src.data.loader import DataLoader
    from src.data.splitter import WalkForwardSplitter
    from src.data.target import TargetGenerator
    from src.env import get_results_dir
    from src.features.engine import FeatureEngine

    pipeline_cfg = load_config().get("pipeline", {})
    data_dir = pipeline_cfg.get("data_dir", "../portable_data/vn_stock_ai_dataset_cleaned")
    abs_data_dir = resolve_data_dir(data_dir)

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

    loader = DataLoader(abs_data_dir)
    feature_set = cfg.feature_set()
    engine = FeatureEngine(feature_set=feature_set)
    target_gen = TargetGenerator.from_config(legacy_split)

    # Load và compute features
    cache_root = Path(get_results_dir()) / "cache" / "features"
    cache_mgr = FeatureCacheManager(str(cache_root))
    code_paths = [feature_engine_module.__file__, target_module.__file__]

    df, cache_key = cache_mgr.load(
        data_dir=abs_data_dir,
        symbols=[symbol],
        timeframe=loader.timeframe,
        feature_set=feature_set,
        target_config=legacy_split.get("target", {}),
        code_paths=code_paths,
    )

    if df is None:
        raw_df = loader.load_all(symbols=[symbol])
        if raw_df.empty:
            return None
        df = engine.compute_for_all_symbols(raw_df)
        cache_mgr.save(
            df=df,
            data_dir=abs_data_dir,
            symbols=[symbol],
            timeframe=loader.timeframe,
            feature_set=feature_set,
            target_config=legacy_split.get("target", {}),
            code_paths=code_paths,
        )

    df = target_gen.generate_for_all_symbols(df)

    # Generate exit labels nếu có exit model
    exit_model_dict = cfg.exit_model_dict()
    if exit_model_dict:
        df = TargetGenerator.generate_exit_labels(
            df,
            forward_window=exit_model_dict["forward_window"],
            loss_threshold=exit_model_dict["loss_threshold"],
        )

    # Train model và predict
    splitter = WalkForwardSplitter.from_config(legacy_split)
    sym_df = df[df["symbol"] == symbol].copy()

    if sym_df.empty:
        return None

    all_predictions = []

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(sym_df)):
        train_df = sym_df.iloc[train_idx]
        test_df = sym_df.iloc[test_idx]

        if train_df.empty or test_df.empty:
            continue

        # Train entry model
        entry_model = get_model(
            cfg.signals.entry_model.type,
            device=cfg.signals.entry_model.device,
            extras=cfg.signals.entry_model.extras or {},
        )

        feature_cols = engine.get_feature_columns(train_df)
        X_train = train_df[feature_cols].values
        y_train = train_df["target"].values

        entry_model.fit(X_train, y_train)

        # Predict
        X_test = test_df[feature_cols].values
        y_pred = entry_model.predict(X_test)

        # Train exit model nếu có
        y_pred_exit = None
        if exit_model_dict:
            exit_model = get_exit_model(
                exit_model_dict["type"],
                device=cfg.signals.exit_model.device,
                extras=cfg.signals.exit_model.extras or {},
            )
            y_train_exit = train_df["target_exit"].values
            exit_model.fit(X_train, y_train_exit)
            y_pred_exit = exit_model.predict(X_test)

        # Lưu predictions
        pred_df = test_df.copy()
        pred_df["y_pred"] = y_pred
        if y_pred_exit is not None:
            pred_df["y_pred_exit"] = y_pred_exit

        all_predictions.append(pred_df)

    if not all_predictions:
        return None

    return pd.concat(all_predictions, ignore_index=True)


def find_recent_signals(predictions_df, days=10):
    """Tìm tín hiệu mua/bán trong N ngày gần nhất"""
    if predictions_df is None or predictions_df.empty:
        return None

    # Lấy N ngày gần nhất
    predictions_df = predictions_df.sort_values("timestamp", ascending=False)
    recent_df = predictions_df.head(days)

    # Tìm tín hiệu mua (y_pred == 2 hoặc 1 tùy target type)
    buy_signals = recent_df[recent_df["y_pred"] >= 1]

    if buy_signals.empty:
        return None

    return buy_signals


def main():
    print("=" * 80)
    print("PREDICTION SCRIPT - TOP 1 MODEL")
    print("=" * 80)
    print()

    # Load config
    print("[1/4] Loading model config...")
    cfg = load_top_model_config()
    print()

    # Get all symbols
    print("[2/4] Getting all symbols...")
    symbols = get_all_symbols()
    print()

    # Run predictions
    print("[3/4] Running predictions...")
    results = []

    for i, symbol in enumerate(symbols, 1):
        print(f"  [{i}/{len(symbols)}] Processing {symbol}...", end=" ")
        try:
            pred_df = run_prediction_for_symbol(cfg, symbol)
            signals = find_recent_signals(pred_df, days=10)

            if signals is not None and not signals.empty:
                latest_signal = signals.iloc[0]
                results.append(
                    {
                        "symbol": symbol,
                        "date": latest_signal["timestamp"],
                        "signal": int(latest_signal["y_pred"]),
                        "close": float(latest_signal.get("close", 0)),
                        "days_ago": (
                            datetime.now().date()
                            - pd.to_datetime(latest_signal["timestamp"]).date()
                        ).days,
                    }
                )
                print("✓ Found signal")
            else:
                print("- No signal")
        except Exception as e:
            print(f"✗ Error: {e}")

    print()

    # Display results
    print("[4/4] Results:")
    print("=" * 80)

    if not results:
        print("Không tìm thấy tín hiệu mua/bán nào trong 10 ngày gần nhất.")
    else:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values("days_ago")

        print(f"\nTìm thấy {len(results)} cổ phiếu có tín hiệu:")
        print()
        print(results_df.to_string(index=False))

        # Save to CSV
        output_file = ROOT / "results" / "recent_signals.csv"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_file, index=False)
        print()
        print(f"✓ Saved to: {output_file}")

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()

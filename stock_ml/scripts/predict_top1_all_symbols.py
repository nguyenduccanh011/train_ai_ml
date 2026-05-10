"""
Script to run predictions on all symbols using the top 1 model from leaderboard
Finds buy/sell signals in the last 10 days of data
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json

import pandas as pd


def get_top1_model():
    """Get top 1 model from leaderboard"""
    leaderboard_file = ROOT / "results" / "leaderboard" / "leaderboard.json"

    if not leaderboard_file.exists():
        raise FileNotFoundError(f"Leaderboard not found: {leaderboard_file}")

    with open(leaderboard_file, encoding="utf-8") as f:
        leaderboard = json.load(f)

    if not leaderboard:
        raise ValueError("Leaderboard is empty")

    top1 = leaderboard[0]
    print(f"[OK] Top 1 model: {top1['bundle']}")
    print(f"  - Run: {top1['run_name']}")
    print(f"  - Config hash: {top1['config_hash']}")
    print(f"  - Composite score: {top1['composite_score']:.1f}")
    print(f"  - Win rate: {top1['wr']:.2f}%")
    print(f"  - Total PnL: {top1['total_pnl']:.2f}")
    print(f"  - Trades: {top1['trades']}")

    return top1


def load_model_config(top1_info):
    """Load config of top 1 model"""
    from src.pipeline import ExperimentConfig

    bundle = top1_info["bundle"]
    run_name = top1_info["run_name"]

    config_path = ROOT / "results" / "experiments" / bundle / run_name / "config.resolved.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config does not exist: {config_path}")

    cfg = ExperimentConfig.from_yaml(config_path)
    print(f"[OK] Loaded config: {cfg.name}")
    print(f"  - Strategy: {cfg.strategy}")
    print(f"  - Feature set: {cfg.feature_set()}")
    print(f"  - Entry model: {cfg.signals.entry_model.type}")
    print(
        f"  - Exit model: {cfg.signals.exit_model.type if cfg.signals.exit_model.enabled else 'disabled'}"
    )

    return cfg


def get_all_symbols():
    """Get list of all symbols from data directory"""
    from src.config_loader import get_pipeline_config, resolve_data_dir

    pipeline_cfg = get_pipeline_config()
    data_dir = resolve_data_dir(
        pipeline_cfg.get("data_dir", "../portable_data/vn_stock_ai_dataset_cleaned")
    )

    # Get all symbols from data directory
    data_path = Path(data_dir)
    symbols = []

    # Search in ohlcv_daily directory
    ohlcv_dir = data_path / "ohlcv_daily"
    if ohlcv_dir.exists():
        for file in ohlcv_dir.glob("symbol=*/timeframe=1D/data.csv"):
            symbol = file.parent.parent.name.replace("symbol=", "")
            symbols.append(symbol)

    # If no symbols found, try to get from pipeline config
    if not symbols:
        explicit_list = pipeline_cfg.get("symbols", {}).get("explicit_list", "")
        if explicit_list:
            symbols = [s.strip() for s in explicit_list.split(",")]

    symbols = sorted(set(symbols))
    print(f"[OK] Found {len(symbols)} symbols")
    return symbols


def run_prediction_for_symbol(cfg, symbol: str):
    """Run prediction for 1 symbol and return predictions for last 10 days"""
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

    for fold_idx, (window, train_df, test_df) in enumerate(splitter.split(sym_df)):
        if train_df.empty or test_df.empty:
            continue

        # Train entry model
        entry_model = get_model(
            cfg.signals.entry_model.type,
            device=cfg.signals.entry_model.device,
        )

        feature_cols = engine.get_feature_columns(train_df)
        X_train = train_df[feature_cols].values
        y_train = train_df["target"].values

        entry_model.fit(X_train, y_train)

        # Predict
        X_test = test_df[feature_cols].values
        y_pred = entry_model.predict(X_test)

        # Train exit model if enabled
        y_pred_exit = None
        if exit_model_dict:
            exit_model = get_exit_model(
                exit_model_dict["type"],
                device=cfg.signals.exit_model.device,
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

    full_pred = pd.concat(all_predictions, ignore_index=True)

    # Chỉ lấy 10 ngày gần nhất
    full_pred = full_pred.sort_values("timestamp", ascending=False)
    recent_pred = full_pred.head(10)

    return recent_pred


def find_signals_in_predictions(predictions_df):
    """Find buy/sell signals from predictions"""
    if predictions_df is None or predictions_df.empty:
        return []

    signals = []

    for _, row in predictions_df.iterrows():
        # y_pred >= 1 means buy signal
        if row["y_pred"] >= 1:
            signal_info = {
                "symbol": row["symbol"],
                "date": row["timestamp"],
                "signal_type": "BUY",
                "y_pred": int(row["y_pred"]),
                "close": float(row.get("close", 0)),
            }

            # Add exit prediction if available
            if "y_pred_exit" in row:
                signal_info["y_pred_exit"] = int(row["y_pred_exit"])

            signals.append(signal_info)

    return signals


def main():
    print("=" * 80)
    print("PREDICT ALL SYMBOLS - TOP 1 MODEL FROM LEADERBOARD")
    print("=" * 80)
    print()

    # Get top 1 model
    print("[1/5] Getting top 1 model from leaderboard...")
    top1_info = get_top1_model()
    print()

    # Load config
    print("[2/5] Loading model config...")
    cfg = load_model_config(top1_info)
    print()

    # Get all symbols
    print("[3/5] Getting all symbols...")
    symbols = get_all_symbols()
    print()

    # Run predictions
    print("[4/5] Running predictions on all symbols...")
    print(f"  Processing {len(symbols)} symbols (this may take a while)...")
    print()

    all_signals = []
    processed = 0
    errors = 0

    for i, symbol in enumerate(symbols, 1):
        if i % 50 == 0 or i == 1:
            print(f"  Progress: {i}/{len(symbols)} ({i / len(symbols) * 100:.1f}%)")

        try:
            pred_df = run_prediction_for_symbol(cfg, symbol)
            signals = find_signals_in_predictions(pred_df)

            if signals:
                all_signals.extend(signals)

            processed += 1

        except Exception as e:
            errors += 1
            if errors <= 3:  # Only print first 3 errors
                print(f"  [!] Error processing {symbol}: {e}")

    print()
    print(f"  Processed: {processed}/{len(symbols)}")
    print(f"  Errors: {errors}")
    print()

    # Display and save results
    print("[5/5] Results:")
    print("=" * 80)

    if not all_signals:
        print("No buy/sell signals found in the last 10 days.")
    else:
        results_df = pd.DataFrame(all_signals)
        results_df["date"] = pd.to_datetime(results_df["date"])
        results_df = results_df.sort_values(["date", "symbol"], ascending=[False, True])

        print(
            f"\nFound {len(results_df)} signals from {results_df['symbol'].nunique()} unique symbols"
        )
        print()

        # Group by date
        print("Signals by date:")
        print("-" * 80)
        for date, group in results_df.groupby(results_df["date"].dt.date):
            print(f"\n{date} ({len(group)} signals):")
            for _, row in group.iterrows():
                exit_info = f", exit_pred={row['y_pred_exit']}" if "y_pred_exit" in row else ""
                print(
                    f"  {row['symbol']:8s} - {row['signal_type']:4s} (pred={row['y_pred']}{exit_info}, close={row['close']:.2f})"
                )

        # Save to CSV
        output_file = ROOT / "results" / "signals_top1_all_symbols.csv"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_file, index=False)
        print()
        print("=" * 80)
        print(f"[OK] Saved to: {output_file}")

        # Summary by symbol
        summary_file = ROOT / "results" / "signals_top1_summary.csv"
        summary = (
            results_df.groupby("symbol")
            .agg({"date": ["min", "max", "count"], "y_pred": "mean", "close": "last"})
            .reset_index()
        )
        summary.columns = [
            "symbol",
            "first_signal_date",
            "last_signal_date",
            "signal_count",
            "avg_pred",
            "last_close",
        ]
        summary = summary.sort_values("signal_count", ascending=False)
        summary.to_csv(summary_file, index=False)
        print(f"[OK] Summary saved to: {summary_file}")

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()

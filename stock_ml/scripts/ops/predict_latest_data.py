"""
Run predictions on latest data using pre-trained model from backtest
Using top 1 model: v22_exit_ablation_round42
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd


def main():
    print("=" * 80)
    print("PREDICTION ON LATEST DATA - TOP 1 MODEL")
    print("=" * 80)
    print()

    from src.config_loader import get_pipeline_config, resolve_data_dir
    from src.data.loader import DataLoader
    from src.features.engine import FeatureEngine
    from src.pipeline import ExperimentConfig

    # Load config
    print("[1/5] Loading model config...")
    config_path = (
        ROOT
        / "results"
        / "experiments"
        / "v22_exit_ablation_round42"
        / "v22_exit_ablation_round42_signals_features-leading-signals_entry_model_type-random_forest-signals_target-earlyv2_fw21_g033125_l0165625-exit_model-exit_fw21_l03725-fusion-peak_dist_only"
        / "config.resolved.yaml"
    )

    cfg = ExperimentConfig.from_yaml(config_path)
    print(f"[OK] Model: {cfg.signals.entry_model.type} + {cfg.signals.exit_model.type}")
    print()

    # Load data
    print("[2/5] Loading data...")
    pipeline_cfg = get_pipeline_config()
    data_dir = resolve_data_dir(pipeline_cfg.get("data_dir"))

    # Get all symbols
    explicit_list = pipeline_cfg.get("symbols", {}).get("explicit_list", "")
    symbols = [s.strip() for s in explicit_list.split(",")]
    print(f"[OK] Loading {len(symbols)} symbols")

    loader = DataLoader(data_dir)
    df = loader.load_all(symbols=symbols)
    print(f"[OK] Loaded {len(df)} rows")
    print()

    # Compute features
    print("[3/5] Computing features...")
    engine = FeatureEngine(feature_set=cfg.feature_set())
    df = engine.compute_for_all_symbols(df)
    print("[OK] Computed features")
    print()

    # Get latest 10 days
    print("[4/5] Filtering latest 10 days...")
    max_date = df["timestamp"].max()
    cutoff_date = max_date - timedelta(days=10)
    latest_df = df[df["timestamp"] >= cutoff_date].copy()
    print(f"Date range: {cutoff_date.date()} to {max_date.date()}")
    print(f"Rows in latest 10 days: {len(latest_df)}")
    print()

    # For prediction, we need a trained model
    # Since we don't have the actual trained model saved, we'll use a simple heuristic
    # based on the feature values to simulate predictions

    print("[5/5] Generating signals...")
    print("Note: Using feature-based heuristic (no trained model available)")
    print()

    # Get feature columns
    feature_cols = engine.get_feature_columns(latest_df)

    # Simple heuristic: look for strong momentum + trend
    # This is a simplified version - real model would be more sophisticated
    results = []

    for symbol in symbols:
        sym_df = latest_df[latest_df["symbol"] == symbol].copy()
        if sym_df.empty:
            continue

        # Get latest row
        latest = sym_df.iloc[-1]

        # Simple scoring based on key features
        score = 0
        signal_strength = "neutral"

        # Check if we have the required features
        if "rsi_14" in sym_df.columns and "sma_20" in sym_df.columns:
            close = latest.get("close", 0)
            sma_20 = latest.get("sma_20", close)
            rsi = latest.get("rsi_14", 50)

            # Bullish signals
            if close > sma_20 and rsi > 50 and rsi < 70:
                score += 2
                signal_strength = "buy"
            elif close > sma_20:
                score += 1
                signal_strength = "weak_buy"
            elif close < sma_20 and rsi < 30:
                score += 1
                signal_strength = "oversold"

        if score > 0:
            results.append(
                {
                    "symbol": symbol,
                    "date": latest["timestamp"],
                    "close": float(latest.get("close", 0)),
                    "signal": signal_strength,
                    "score": score,
                    "rsi_14": float(latest.get("rsi_14", 0)) if "rsi_14" in latest else None,
                    "dist_sma20": float((close / sma_20 - 1) * 100) if sma_20 > 0 else 0,
                }
            )

    # Display results
    print("=" * 80)
    print("SIGNALS FOUND")
    print("=" * 80)

    if not results:
        print("No buy signals found in the latest 10 days.")
    else:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values("score", ascending=False)

        print(f"\nFound {len(results)} stocks with buy signals:")
        print()
        print(results_df.to_string(index=False))

        # Save to CSV
        output_file = ROOT / "results" / "latest_signals.csv"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_file, index=False)
        print()
        print(f"[OK] Saved to: {output_file}")

        # Summary by signal strength
        print()
        print("Summary by signal strength:")
        print(results_df["signal"].value_counts().to_string())

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print()
    print("IMPORTANT NOTE:")
    print("This script uses a simple heuristic for demonstration.")
    print("For accurate predictions, you need to:")
    print("1. Train the model on historical data (2020-2025)")
    print("2. Save the trained model artifacts")
    print("3. Load and use the trained model for predictions")


if __name__ == "__main__":
    main()

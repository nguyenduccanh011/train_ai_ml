"""
Main experiment pipeline - orchestrates data loading, feature engineering,
model training, and evaluation across walk-forward windows.
"""
import os
import json
import time
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from .data.loader import DataLoader
from .data.splitter import WalkForwardSplitter
from .data.target import TargetGenerator
from .features.engine import FeatureEngine
from .models.registry import build_model, build_all_models, get_available_models
from .evaluation.metrics import (
    compute_metrics, compute_trading_metrics,
    format_results_table, print_leaderboard,
)
from .evaluation.backtest import backtest_predictions, format_backtest_report


class ExperimentPipeline:
    """
    Run systematic experiments across multiple:
    - Feature sets (minimal / technical / full)
    - Models (RF, XGB, LGB, LR, SVM, etc.)
    - Walk-forward windows
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_dir = config["data"]["data_dir"]
        self.output_dir = Path(config.get("output_dir", "results"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.loader = DataLoader(self.data_dir)
        self.splitter = WalkForwardSplitter.from_config(config)
        self.target_gen = TargetGenerator.from_config(config)

        self.all_results: List[Dict[str, Any]] = []

    def run(
        self,
        feature_sets: Optional[List[str]] = None,
        model_names: Optional[List[str]] = None,
        max_symbols: Optional[int] = None,
    ):
        """Run the full experiment grid."""
        feature_sets = feature_sets or self.config.get(
            "feature_sets", ["minimal", "technical"]
        )
        model_names = model_names or self.config.get(
            "models", get_available_models()
        )

        print("=" * 80)
        print("🚀 VN STOCK ML EXPERIMENT")
        print(f"   Feature sets: {feature_sets}")
        print(f"   Models: {model_names}")
        print(f"   Symbols: {max_symbols or 'all'}")
        print("=" * 80)

        # Load data once
        symbols = self.loader.symbols[:max_symbols] if max_symbols else None
        print("\n📦 Loading data...")
        raw_df = self.loader.load_all(symbols=symbols)
        print(f"   Loaded {len(raw_df)} rows, {raw_df['symbol'].nunique()} symbols")

        # Load context data
        context_data = {}
        try:
            context_data = self.loader.load_all_context()
            print(f"   Context: {list(context_data.keys())}")
        except Exception as e:
            print(f"   ⚠️ No context data: {e}")

        # Run experiments per feature set
        for feat_set in feature_sets:
            print(f"\n{'─' * 60}")
            print(f"📊 Feature set: {feat_set}")
            print(f"{'─' * 60}")

            engine = FeatureEngine(feature_set=feat_set)
            df = engine.compute_for_all_symbols(raw_df)

            # Add context if full feature set
            if feat_set == "full" and context_data:
                df = engine.add_market_context(df, context_data)

            # Generate targets
            df = self.target_gen.generate_for_all_symbols(df)

            # Get feature columns
            feature_cols = engine.get_feature_columns(df)
            print(f"   Features: {len(feature_cols)} columns")

            # Drop rows with NaN in features
            df = df.dropna(subset=feature_cols + ["target"])
            print(f"   Clean rows: {len(df)}")

            # Walk-forward evaluation
            for model_name in model_names:
                self._run_model_walkforward(
                    df, feature_cols, model_name, feat_set
                )

        # Final results
        results_df = format_results_table(self.all_results)
        self._save_results(results_df)
        print_leaderboard(results_df)

        return results_df

    def _run_model_walkforward(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        model_name: str,
        feature_set: str,
    ):
        """Train and evaluate a model across all walk-forward windows."""
        window_results = []

        for window, train_df, test_df in self.splitter.split(df):
            try:
                t0 = time.time()
                model = build_model(model_name)

                X_train = train_df[feature_cols].values
                y_train = train_df["target"].values.astype(int)
                X_test = test_df[feature_cols].values
                y_test = test_df["target"].values.astype(int)

                # XGBoost needs 0-indexed labels
                if model_name == "xgboost" and y_train.min() < 0:
                    label_offset = abs(y_train.min())
                    y_train = y_train + label_offset
                    y_test = y_test + label_offset
                    _xgb_offset = label_offset
                else:
                    _xgb_offset = 0

                # Handle inf values
                X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
                X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                elapsed = time.time() - t0

                # Metrics
                metrics = compute_metrics(y_test, y_pred)

                # Trading metrics (use return_1d if available)
                trading = {}
                if "return_1d" in test_df.columns:
                    trading = compute_trading_metrics(
                        y_test, y_pred, test_df["return_1d"].values
                    )

                result = {
                    "model": model_name,
                    "feature_set": feature_set,
                    "window": window.label,
                    "train_size": len(train_df),
                    "test_size": len(test_df),
                    "time_sec": round(elapsed, 2),
                    **metrics,
                    **trading,
                }
                window_results.append(result)
                self.all_results.append(result)

                print(
                    f"   ✅ {model_name:20s} | {window.label:30s} | "
                    f"F1={metrics['f1_macro']:.3f} | Acc={metrics['accuracy']:.3f} | "
                    f"{elapsed:.1f}s"
                )

            except Exception as e:
                print(f"   ❌ {model_name:20s} | {window.label:30s} | Error: {e}")
                self.all_results.append({
                    "model": model_name,
                    "feature_set": feature_set,
                    "window": window.label,
                    "error": str(e),
                })

    def _save_results(self, results_df: pd.DataFrame):
        """Save results to CSV and JSON."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = self.output_dir / f"results_{ts}.csv"
        json_path = self.output_dir / f"results_{ts}.json"

        results_df.to_csv(csv_path, index=False)
        results_df.to_json(json_path, orient="records", indent=2)

        print(f"\n💾 Results saved to:")
        print(f"   {csv_path}")
        print(f"   {json_path}")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)

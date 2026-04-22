"""
Quick run script for VN Stock ML experiments.
Usage:
    python run.py                          # Quick test (5 symbols, 2 models)
    python run.py --full                   # Full run (all symbols, all models)
    python run.py --symbols 20 --models lightgbm xgboost random_forest
"""
import argparse
import sys
import os

# Add parent to path so imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import ExperimentPipeline
from src.models.registry import get_available_models


def main():
    parser = argparse.ArgumentParser(description="VN Stock ML Experiment Runner")
    parser.add_argument("--full", action="store_true", help="Full run with all symbols")
    parser.add_argument("--symbols", type=int, default=5, help="Max symbols to use")
    parser.add_argument("--models", nargs="+", default=None, help="Model names")
    parser.add_argument("--features", nargs="+", default=None, help="Feature sets")
    parser.add_argument("--data-dir", type=str, default=None, help="Data directory")
    args = parser.parse_args()

    # Default config
    config = {
        "data": {
            "data_dir": args.data_dir or os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..", "portable_data", "vn_stock_ai_dataset_cleaned"
            ),
        },
        "split": {
            "method": "walk_forward",
            "train_years": 4,
            "test_years": 1,
            "gap_days": 0,
            "first_test_year": 2020,
            "last_test_year": 2025,
        },
        "target": {
            "type": "trend_regime",
            "trend_method": "dual_ma",
            "short_window": 10,
            "long_window": 40,
            "classes": 3,
        },
        "output_dir": os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "results"
        ),
    }

    # Determine scope
    if args.full:
        max_symbols = None
        feature_sets = args.features or ["minimal", "technical", "full"]
        model_names = args.models or get_available_models()
    else:
        max_symbols = args.symbols
        feature_sets = args.features or ["minimal", "technical"]
        model_names = args.models or ["random_forest", "logistic_regression"]
        # Add boost models if available
        available = get_available_models()
        for m in ["xgboost", "lightgbm"]:
            if m in available and m not in model_names:
                model_names.append(m)

    print(f"Available models: {get_available_models()}")

    pipeline = ExperimentPipeline(config)
    results = pipeline.run(
        feature_sets=feature_sets,
        model_names=model_names,
        max_symbols=max_symbols,
    )

    return results


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VIZ_DIR = ROOT / "web"
BASE_DATA_DIR = ROOT / "data" / "ohlcv"
STANDALONE_DATASET_DIR = ROOT / "data" / "vn_stock_ai_dataset_cleaned"
SIGNAL_CACHE_DIR = ROOT / "cache" / "signals"
FEATURE_CACHE_ROOT = ROOT / "cache" / "features"
MODEL_PATH = ROOT / "models" / "train61_single_model.pkl"
TOP1_FOLD_CHAIN_MODEL_PATH = ROOT / "models" / "train61_fold_chain.top1.no_context.pkl"
CONFIG_PATH = ROOT / "config" / "model_config.resolved.yaml"
TRAIN61_SYMBOLS_PATH = ROOT / "config" / "train61_symbols.json"
CONTEXT_CACHE_TAG = "with_context_v1"
TOP1_BACKTEST_DIR = (
    ROOT
    / "results"
    / "experiments"
    / "v22_exit_ablation_round42"
    / "v22_exit_ablation_round42_signals_features-leading-signals_entry_model_type-random_forest-signals_target-earlyv2_fw21_g033125_l0165625-exit_model-exit_fw21_l03725-fusion-peak_dist_only"
)
TOP1_BACKTEST_CONFIG = TOP1_BACKTEST_DIR / "config.resolved.yaml"

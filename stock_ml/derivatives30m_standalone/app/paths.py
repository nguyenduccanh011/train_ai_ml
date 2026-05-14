from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STOCK_ML_ROOT = ROOT.parent
REPO_ROOT = STOCK_ML_ROOT.parent

VIZ_DIR = ROOT / "web"
BASE_DATA_DIR = ROOT / "data" / "ohlcv"
STANDALONE_DATASET_DIR = ROOT / "data" / "derivatives_ai_dataset"
SIGNAL_CACHE_DIR = ROOT / "cache" / "signals"
FEATURE_CACHE_ROOT = ROOT / "cache" / "features"
MODEL_PATH = ROOT / "models" / "derivatives30m_top1.pkl"
CONFIG_PATH = ROOT / "config" / "model_config.resolved.yaml"
REALTIME_CONFIG_PATH = ROOT / "config" / "model_config.realtime_top1.yaml"
SYMBOLS_PATH = ROOT / "config" / "derivatives30m_symbols.json"

MARKET = "vn_derivatives_30m"
TIMEFRAME = "30m"
CONTEXT_CACHE_TAG = "no_context_v1"

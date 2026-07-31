"""stock_ml.core — the inference-only surface a production serving host imports.

A SEPARATE serving service (see memory ``project_production_bundle_deploy``)
depends on THIS package alone: load a bundle and turn OHLCV into signals, with no
research database, FastAPI, or training pipeline pulled in. The functions are
re-exported from stock_ml.src so backtest and serving share one implementation
(no copy-paste drift).

Verified import-clean (no sqlalchemy/fastapi/asyncpg) by test_core_facade.py.

Typical use in a serving repo::

    from stock_ml.core import load_bundle, generate_signals_from_bundle
    bundle = load_bundle("bundles/bundle_<name>_<cutoff>")
    signals = generate_signals_from_bundle(bundle, ohlcv_with_warmup)
"""

from __future__ import annotations

# Bumped when the inference contract (feature pipeline / signal chain) changes in a
# way that could alter a bundle's outputs. A serving host can assert this matches
# the bundle's expectations.
# 0.2.0: ensemble entry/exit heads (entry2/entry3/exit2) — 3-head champions (n2_3h_*)
#        now serve identically to the backtest; 2-head bundles are unchanged.
# 0.3.0: GENERIC N-head serving — predict_slot_signals takes entry_ensemble/exit_ensemble
#        lists; inference auto-detects entry2..entryN / exit2.. from the bundle. Serves
#        4-head (n2_4h_*) / 5-head (n2_5h_*) champions identically; ≤3-head unchanged.
__version__ = "0.3.0"

from stock_ml.src.backtest.engine import (
    CostModel,
    EngineConfig,
    engine_config_from_dict,
    run_backtest,
    trades_to_dataframe,
)
from stock_ml.src.pipeline.experiment import (
    ExperimentConfig,
    build_feature_frame,
    predict_slot_signals,
    recombine_signals,
)
from stock_ml.src.serving.bundle import LoadedBundle, load_bundle
from stock_ml.src.serving.inference import generate_signals_from_bundle

__all__ = [
    "__version__",
    "ExperimentConfig",
    "build_feature_frame",
    "predict_slot_signals",
    "recombine_signals",
    "load_bundle",
    "LoadedBundle",
    "generate_signals_from_bundle",
    # Backtest / trade path (R3/R9 facade — the serving trade layer builds its EngineConfig here so it
    # strips the same recombine keys as train, instead of importing stock_ml.src.backtest directly).
    "EngineConfig",
    "CostModel",
    "engine_config_from_dict",
    "run_backtest",
    "trades_to_dataframe",
]

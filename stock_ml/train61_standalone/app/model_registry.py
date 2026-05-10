from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from paths import (
    CONFIG_PATH,
    MODEL_PATH,
    STANDALONE_DATASET_DIR,
    TOP1_BACKTEST_CONFIG,
    TOP1_FOLD_CHAIN_MODEL_PATH,
    TRAIN61_SYMBOLS_PATH,
)

MODELS: dict[str, dict[str, Any]] = {
    "train61_pooled": {
        "type": "pkl",
        "path": MODEL_PATH,
        "config": CONFIG_PATH,
        "requires": ["model_path", "config", "dataset", "symbols"],
        "color": "#00E5FF",
        "label": "Train61 Pooled (61 symbols)",
    },
    "top1_on_demand": {
        "type": "on_demand",
        "config": TOP1_BACKTEST_CONFIG,
        "requires": ["config", "dataset", "symbols"],
        "color": "#FFD700",
        "label": "Top1 On-Demand",
    },
    "top1_fold_chain_no_context": {
        "type": "pkl",
        "path": TOP1_FOLD_CHAIN_MODEL_PATH,
        "config": CONFIG_PATH,
        "requires": ["model_path", "config", "dataset", "symbols"],
        "color": "#26A69A",
        "label": "Top1 Fold Chain (No Context)",
    },
    "v5_top1_pooled_global_rerun": {
        "type": "pooled_global_rerun",
        "config": TOP1_BACKTEST_CONFIG,
        "requires": ["config", "dataset", "symbols"],
        "color": "#43A047",
        "label": "V5 Top1 Pooled Global (Re-run)",
    },
}
DEFAULT_MODEL = str(
    os.getenv("TRAIN61_DEFAULT_MODEL", "train61_pooled") or "train61_pooled"
).strip()
if DEFAULT_MODEL not in MODELS:
    print(f"[WARN] Unknown TRAIN61_DEFAULT_MODEL='{DEFAULT_MODEL}', fallback to 'train61_pooled'")
    DEFAULT_MODEL = "train61_pooled"


def get_model_cfg(model_id: str) -> dict[str, Any]:
    cfg = MODELS.get(model_id)
    if cfg is None:
        raise ValueError(f"Unknown model: {model_id}")
    return cfg


def model_requirements(model_id: str) -> list[str]:
    return list(get_model_cfg(model_id).get("requires", []))


def model_availability(model_id: str) -> dict[str, Any]:
    cfg = get_model_cfg(model_id)
    missing: list[str] = []
    requirement_paths: dict[str, Path] = {
        "model_path": Path(cfg["path"]) if cfg.get("path") else Path(),
        "config": Path(cfg["config"]) if cfg.get("config") else Path(),
        "dataset": STANDALONE_DATASET_DIR,
        "symbols": TRAIN61_SYMBOLS_PATH,
    }
    for requirement in model_requirements(model_id):
        path = requirement_paths.get(requirement)
        if path is not None and not path.exists():
            missing.append(str(path))
    return {"available": not missing, "missing": missing}

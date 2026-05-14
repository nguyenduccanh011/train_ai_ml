from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from paths import MODEL_PATH, REALTIME_CONFIG_PATH, STANDALONE_DATASET_DIR, SYMBOLS_PATH

DEFAULT_MODEL = os.getenv("DERIVATIVES30M_DEFAULT_MODEL", "derivatives30m_top1")

MODELS: dict[str, dict[str, Any]] = {
    "derivatives30m_top1": {
        "type": "pkl",
        "label": "VN30F 30M Top1",
        "color": "#FFD54F",
        "path": MODEL_PATH,
        "config": REALTIME_CONFIG_PATH,
        "requires": ["model_path", "config", "dataset", "symbols"],
    }
}


def get_model_config(model_id: str | None = None) -> tuple[str, dict[str, Any]]:
    resolved = str(model_id or DEFAULT_MODEL).strip() or DEFAULT_MODEL
    if resolved not in MODELS:
        raise ValueError(f"Unknown model_id '{resolved}'. Available: {', '.join(sorted(MODELS))}")
    return resolved, MODELS[resolved]


def model_availability(model_id: str) -> dict[str, Any]:
    _, cfg = get_model_config(model_id)
    missing: list[str] = []
    checks = {
        "model_path": Path(cfg.get("path", "")),
        "config": Path(cfg.get("config", "")),
        "dataset": STANDALONE_DATASET_DIR,
        "symbols": SYMBOLS_PATH,
    }
    for key in cfg.get("requires", []):
        path = checks.get(str(key))
        if path is not None and not path.exists():
            missing.append(str(path))
    return {"available": not missing, "missing": missing}

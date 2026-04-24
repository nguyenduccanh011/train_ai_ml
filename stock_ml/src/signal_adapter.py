"""
Utilities to normalize model predictions into canonical trading labels.

Canonical labels used by backtests:
  1  -> buy / long bias
  0  -> neutral / no-entry
 -1  -> avoid / bearish
"""
from __future__ import annotations

import json
from typing import Any, Dict

import numpy as np


def _target_type(target_cfg: Dict[str, Any]) -> str:
    return str(target_cfg.get("type", "trend_regime")).strip().lower()


def canonicalize_predictions(y_pred, target_cfg: Dict[str, Any]):
    """Map raw model predictions to canonical {-1, 0, 1} labels."""
    cfg = target_cfg or {}
    target_type = _target_type(cfg)
    arr = np.asarray(y_pred)

    if target_type in ("trend_regime", "return_classification", "forward_risk_reward", "early_wave", "early_wave_v2", "early_wave_dual"):
        buy_label = cfg.get("buy_label", 1)
        neutral_label = cfg.get("neutral_label", 0)
        sell_label = cfg.get("sell_label", -1)
        n_classes = int(cfg.get("classes", 3))

        out = np.full(arr.shape, 0, dtype=int)
        out[arr == buy_label] = 1

        if n_classes >= 3:
            out[arr == sell_label] = -1
            out[arr == neutral_label] = 0
        return out

    if target_type == "return_regression":
        # Optional thresholds for future regression support.
        buy_threshold = float(cfg.get("buy_threshold", 0.0))
        sell_threshold = cfg.get("sell_threshold", None)

        arr = arr.astype(float)
        out = np.zeros(arr.shape, dtype=int)
        out[arr >= buy_threshold] = 1
        if sell_threshold is not None:
            out[arr <= float(sell_threshold)] = -1
        return out

    raise ValueError(f"Unknown target type for signal adapter: {target_type}")


def target_fingerprint(target_cfg: Dict[str, Any]) -> str:
    """Stable hash-like string for cache validation."""
    cfg = target_cfg or {}
    return json.dumps(cfg, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


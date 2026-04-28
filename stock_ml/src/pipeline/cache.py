"""PredictionCacheManager — persist walk-forward prediction cache to disk.

Cache layout:
    results/cache/predictions/<cache_key>.pkl

Cache key is a sha256 hash of: feature_set + target_config + model_type + sorted symbols.
Atomic write: write to .tmp then rename.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.pipeline.config import ExperimentConfig


def _build_prediction_cache_key(
    cfg: ExperimentConfig,
    symbols: list[str],
) -> str:
    payload = {
        "feature_set": cfg.feature_set(),
        "model_type": cfg.entry_model_type(),
        "target": cfg.target_dict(),
        "exit_model": cfg.exit_model_dict(),
        "symbols": sorted(symbols),
        "split": cfg.split.model_dump(),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


class PredictionCacheManager:
    """Save and load walk-forward prediction caches to/from disk.

    Avoids rebuilding ML predictions on repeated runs with the same config.
    """

    def __init__(self, cache_root: str | Path) -> None:
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def key(self, cfg: ExperimentConfig, symbols: list[str]) -> str:
        return _build_prediction_cache_key(cfg, symbols)

    def cache_path(self, key: str) -> Path:
        return self.cache_root / f"{key}.pkl"

    def load(
        self, cfg: ExperimentConfig, symbols: list[str]
    ) -> tuple[list[dict[str, Any]] | None, str]:
        """Load prediction cache from disk. Returns (cache, key). cache=None on miss."""
        k = self.key(cfg, symbols)
        path = self.cache_path(k)
        if not path.exists():
            return None, k
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            return data, k
        except Exception:
            return None, k

    def save(
        self,
        data: list[dict[str, Any]],
        cfg: ExperimentConfig,
        symbols: list[str],
    ) -> str:
        """Atomically write prediction cache to disk. Returns cache key."""
        k = self.key(cfg, symbols)
        path = self.cache_path(k)
        tmp_fd, tmp_path = tempfile.mkstemp(dir=self.cache_root, suffix=".tmp")
        try:
            with open(tmp_fd, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            Path(tmp_path).replace(path)
        except Exception:
            Path(tmp_path).unlink(missing_ok=True)
            raise
        return k

    def invalidate(self, cfg: ExperimentConfig, symbols: list[str]) -> bool:
        """Delete cached predictions. Returns True if something was deleted."""
        k = self.key(cfg, symbols)
        path = self.cache_path(k)
        if path.exists():
            path.unlink()
            return True
        return False

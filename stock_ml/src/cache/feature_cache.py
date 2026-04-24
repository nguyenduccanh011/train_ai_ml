"""
Feature cache manager for experiment pipelines.

Cache layout:
  results/cache/features/<feature_set>/<cache_key>.parquet
  results/cache/features/<feature_set>/<cache_key>.pkl
  results/cache/features/<feature_set>/<cache_key>.json
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import pandas as pd


class FeatureCacheManager:
    """Manage on-disk caches for expensive feature engineering output."""

    CACHE_SCHEMA_VERSION = 1

    def __init__(self, cache_root: str):
        self.cache_root = Path(cache_root)

    @staticmethod
    def _normalize_symbols(symbols: Sequence[str]) -> List[str]:
        return sorted({s.strip() for s in symbols if str(s).strip()})

    @staticmethod
    def _compute_source_fingerprint(
        data_dir: str,
        symbols: Sequence[str],
        timeframe: str,
    ) -> str:
        base_dir = Path(data_dir) / "all_symbols"
        rows = []
        for sym in symbols:
            csv_path = base_dir / f"symbol={sym}" / f"timeframe={timeframe}" / "data.csv"
            if not csv_path.exists():
                rows.append((sym, "missing", 0, 0))
                continue
            stat = csv_path.stat()
            rows.append((sym, "ok", stat.st_size, stat.st_mtime_ns))

        payload = json.dumps(rows, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _compute_code_fingerprint(code_paths: Optional[Iterable[str]]) -> str:
        if not code_paths:
            return "na"

        rows = []
        for p in code_paths:
            path = Path(p)
            if not path.exists():
                rows.append((str(path), "missing", 0))
                continue
            stat = path.stat()
            rows.append((str(path.resolve()), stat.st_size, stat.st_mtime_ns))
        payload = json.dumps(rows, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def build_signature(
        self,
        *,
        data_dir: str,
        symbols: Sequence[str],
        timeframe: str,
        feature_set: str,
        target_config: dict,
        extra_groups: Optional[Sequence[str]] = None,
        code_paths: Optional[Iterable[str]] = None,
    ) -> dict:
        norm_symbols = self._normalize_symbols(symbols)
        signature = {
            "schema_version": self.CACHE_SCHEMA_VERSION,
            "data_dir": str(Path(data_dir).resolve()),
            "timeframe": timeframe,
            "feature_set": feature_set,
            "extra_groups": sorted(set(extra_groups or [])),
            "target_config": target_config or {},
            "symbols": norm_symbols,
            "source_fingerprint": self._compute_source_fingerprint(
                data_dir=data_dir,
                symbols=norm_symbols,
                timeframe=timeframe,
            ),
            "code_fingerprint": self._compute_code_fingerprint(code_paths),
        }
        return signature

    @staticmethod
    def _cache_key(signature: dict) -> str:
        raw = json.dumps(signature, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]

    def _resolve_paths(self, feature_set: str, cache_key: str) -> tuple[Path, Path, Path]:
        cache_dir = self.cache_root / feature_set
        parquet_path = cache_dir / f"{cache_key}.parquet"
        pickle_path = cache_dir / f"{cache_key}.pkl"
        meta_path = cache_dir / f"{cache_key}.json"
        return parquet_path, pickle_path, meta_path

    def load(
        self,
        *,
        data_dir: str,
        symbols: Sequence[str],
        timeframe: str,
        feature_set: str,
        target_config: dict,
        extra_groups: Optional[Sequence[str]] = None,
        code_paths: Optional[Iterable[str]] = None,
    ) -> tuple[Optional[pd.DataFrame], str]:
        signature = self.build_signature(
            data_dir=data_dir,
            symbols=symbols,
            timeframe=timeframe,
            feature_set=feature_set,
            target_config=target_config,
            extra_groups=extra_groups,
            code_paths=code_paths,
        )
        cache_key = self._cache_key(signature)
        parquet_path, pickle_path, _meta_path = self._resolve_paths(feature_set, cache_key)

        if parquet_path.exists():
            return pd.read_parquet(parquet_path), cache_key
        if pickle_path.exists():
            return pd.read_pickle(pickle_path), cache_key
        return None, cache_key

    def save(
        self,
        *,
        df: pd.DataFrame,
        data_dir: str,
        symbols: Sequence[str],
        timeframe: str,
        feature_set: str,
        target_config: dict,
        extra_groups: Optional[Sequence[str]] = None,
        code_paths: Optional[Iterable[str]] = None,
    ) -> tuple[str, str]:
        signature = self.build_signature(
            data_dir=data_dir,
            symbols=symbols,
            timeframe=timeframe,
            feature_set=feature_set,
            target_config=target_config,
            extra_groups=extra_groups,
            code_paths=code_paths,
        )
        cache_key = self._cache_key(signature)
        parquet_path, pickle_path, meta_path = self._resolve_paths(feature_set, cache_key)
        parquet_path.parent.mkdir(parents=True, exist_ok=True)

        fmt = "parquet"
        try:
            df.to_parquet(parquet_path, index=False)
        except Exception:
            df.to_pickle(pickle_path)
            fmt = "pickle"

        metadata = {
            "cache_key": cache_key,
            "format": fmt,
            "rows": int(len(df)),
            "cols": int(len(df.columns)),
            "columns": list(df.columns),
            "signature": signature,
        }
        meta_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        return cache_key, fmt

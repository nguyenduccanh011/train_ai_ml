"""
Feature cache manager for experiment pipelines.

Cache layout:
  results/cache/features/<feature_set>/<cache_key>.parquet
  results/cache/features/<feature_set>/<cache_key>.pkl
  results/cache/features/<feature_set>/<cache_key>.json
"""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd


class FeatureCacheManager:
    """Manage on-disk caches for expensive feature engineering output."""

    CACHE_SCHEMA_VERSION = 2

    def __init__(self, cache_root: str):
        self.cache_root = Path(cache_root)

    @staticmethod
    def _normalize_symbols(symbols: Sequence[str]) -> list[str]:
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
            latest_ts = ""
            try:
                with csv_path.open("rb") as fh:
                    fh.seek(0, 2)
                    end = fh.tell()
                    size = min(end, 8192)
                    fh.seek(end - size)
                    tail = fh.read(size).decode("utf-8", errors="ignore")
                lines = [line for line in tail.splitlines() if line.strip()]
                if lines:
                    last_line = lines[-1]
                    parsed = next(csv.reader([last_line]), [])
                    if parsed:
                        latest_ts = parsed[0].strip()
            except Exception:
                latest_ts = ""
            rows.append((sym, "ok", stat.st_size, stat.st_mtime_ns, latest_ts))

        payload = json.dumps(rows, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _compute_code_fingerprint(code_paths: Iterable[str] | None) -> str:
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
        extra_groups: Sequence[str] | None = None,
        code_paths: Iterable[str] | None = None,
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
        extra_groups: Sequence[str] | None = None,
        code_paths: Iterable[str] | None = None,
    ) -> tuple[pd.DataFrame | None, str]:
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
            try:
                return pd.read_parquet(parquet_path), cache_key
            except Exception:
                parquet_path.unlink(missing_ok=True)
                return None, cache_key
        if pickle_path.exists():
            try:
                return pd.read_pickle(pickle_path), cache_key
            except Exception:
                pickle_path.unlink(missing_ok=True)
                return None, cache_key
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
        extra_groups: Sequence[str] | None = None,
        code_paths: Iterable[str] | None = None,
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
            with tempfile.NamedTemporaryFile(
                dir=parquet_path.parent,
                suffix=".parquet",
                delete=False,
            ) as tmp:
                tmp_parquet_path = Path(tmp.name)
            try:
                df.to_parquet(tmp_parquet_path, index=False)
                tmp_parquet_path.replace(parquet_path)
            except Exception:
                tmp_parquet_path.unlink(missing_ok=True)
                raise
        except Exception:
            with tempfile.NamedTemporaryFile(
                dir=pickle_path.parent,
                suffix=".pkl",
                delete=False,
            ) as tmp:
                tmp_pickle_path = Path(tmp.name)
            try:
                df.to_pickle(tmp_pickle_path)
                tmp_pickle_path.replace(pickle_path)
            except Exception:
                tmp_pickle_path.unlink(missing_ok=True)
                raise
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

"""Content-addressed feature store (the *values* layer of the feature system).

Each materialised feature is one physical parquet file keyed by its expression
hash and a ``data_version`` fingerprint::

    <root>/<expr_hash>/<data_version>.parquet      schema: [symbol, date, value]

Because the path is content-addressed, the same formula on the same
universe/timeframe/range is stored exactly once — every feature set that
references it points at the same file (0 duplication). A feature set's matrix is
a lazy DuckDB JOIN of its members on ``(symbol, date)``.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

DEFAULT_STORE_ROOT = "results/cache/features/store"
_VALUE_SCHEMA = ("symbol", "date", "value")


def content_fingerprint(frame: pd.DataFrame) -> str:
    """Deterministic content hash of the input frame (all columns).

    Any change to the underlying data — a corrected close, an inserted row, a
    different sector assignment, a different market index — flips this, so the
    content-addressed store never serves stale feature values when the inputs
    that produced them change. Order-independent (sorted by symbol/date).
    """
    if "symbol" not in frame.columns or "date" not in frame.columns:
        raise ValueError("content_fingerprint requires 'symbol' and 'date' columns")
    ordered = frame.sort_values(["symbol", "date"]).reset_index(drop=True)
    row_hashes = pd.util.hash_pandas_object(ordered, index=False).to_numpy()
    return hashlib.sha1(row_hashes.tobytes()).hexdigest()[:20]


class FeatureStore:
    """Save/load per-feature parquet values, content-addressed by expr_hash."""

    def __init__(self, root: str | Path = DEFAULT_STORE_ROOT):
        self.root = Path(root)

    # -- addressing -------------------------------------------------------

    def path(self, expr_hash: str, data_version: str) -> Path:
        return self.root / expr_hash / f"{data_version}.parquet"

    def exists(self, expr_hash: str, data_version: str) -> bool:
        return self.path(expr_hash, data_version).exists()

    @staticmethod
    def compute_data_version(
        *,
        symbols: Sequence[str],
        timeframe: str,
        start=None,
        end=None,
        source_fingerprint: str | None = None,
        extra: dict | None = None,
    ) -> str:
        """Fingerprint of (universe + timeframe + date range + data source).

        The universe (sorted symbols) is mandatory — cross-sectional features are
        only reproducible against the exact set of symbols they were computed on.
        """
        norm_symbols = sorted({str(s).strip() for s in symbols if str(s).strip()})
        if not norm_symbols:
            raise ValueError("data_version requires a non-empty universe (symbols)")
        payload = {
            "symbols": norm_symbols,
            "timeframe": timeframe,
            "start": str(start) if start is not None else None,
            "end": str(end) if end is not None else None,
            "source": source_fingerprint,
            "extra": extra or {},
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]

    # -- io ---------------------------------------------------------------

    @staticmethod
    def _validate(frame: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in _VALUE_SCHEMA if c not in frame.columns]
        if missing:
            raise ValueError(f"feature frame missing columns {missing}; need {list(_VALUE_SCHEMA)}")
        return frame.loc[:, list(_VALUE_SCHEMA)].copy()

    def save(
        self,
        *,
        expr_hash: str,
        data_version: str,
        frame: pd.DataFrame,
    ) -> Path:
        """Atomically write a feature's values; returns the file path."""
        out = self._validate(frame)
        path = self.path(expr_hash, data_version)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".parquet", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            out.to_parquet(tmp_path, index=False)
            tmp_path.replace(path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
        return path

    def load(self, expr_hash: str, data_version: str) -> pd.DataFrame | None:
        """Return [symbol, date, value] for a materialised feature, or None on miss."""
        path = self.path(expr_hash, data_version)
        if not path.exists():
            return None
        return pd.read_parquet(path)

    # -- set matrix (lazy JOIN) ------------------------------------------

    def join_matrix(self, members: list[dict]) -> pd.DataFrame:
        """Lazily JOIN member feature files on (symbol, date) into a wide matrix.

        Args:
            members: list of {"name", "expr_hash", "data_version"} dicts. Each
                contributes one value column named ``name``.

        Returns:
            DataFrame [symbol, date, <name1>, <name2>, ...].
        """
        if not members:
            raise ValueError("join_matrix requires at least one member")
        import duckdb

        paths = []
        for m in members:
            p = self.path(m["expr_hash"], m["data_version"])
            if not p.exists():
                raise FileNotFoundError(
                    f"feature '{m['name']}' not materialised at {p} — store it first"
                )
            paths.append(p.as_posix())

        def _sub(i: int, name: str) -> str:
            return f'(SELECT symbol, date, value AS "{name}" FROM read_parquet(?)) t{i}'

        sql = f"SELECT * FROM {_sub(0, members[0]['name'])}"
        for i in range(1, len(members)):
            sql += f" FULL JOIN {_sub(i, members[i]['name'])} USING (symbol, date)"
        sql += " ORDER BY symbol, date"

        con = duckdb.connect()
        try:
            return con.execute(sql, paths).df()
        finally:
            con.close()

"""DuckDB-based OHLCV loader — efficient SQL queries over parquetized datasets."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

_PRICE_COLS = ["open", "high", "low", "close"]


def _sanitize_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill non-positive OHLC prices.

    A handful of rows carry close/open/high/low == 0 (data glitches). Because most
    features divide by price or a rolling Min/Corr of price, ONE zero poisons the
    whole trailing window with ±inf (e.g. dist_52w_low = close/Min(close,252)-1 stays
    inf for 252 bars after a single zero). We mask non-positive OHLC to NaN and
    forward/back-fill per symbol so the on-disk data is untouched but features stay
    finite. Volume is left as-is (0 = a legitimate no-trade day).
    """
    cols = [c for c in _PRICE_COLS if c in df.columns]
    if df.empty or not cols:
        return df
    bad = df[cols] <= 0
    if not bad.to_numpy().any():
        return df
    df = df.copy()
    df[cols] = df[cols].mask(bad)
    if "symbol" in df.columns:
        df[cols] = df.groupby("symbol")[cols].transform(lambda g: g.ffill().bfill())
    else:
        df[cols] = df[cols].ffill().bfill()
    return df


class DuckDBLoader:
    """Load OHLCV data from DuckDB files.

    Replaces CSV-based DataLoader for scalability. Uses parametrized queries
    to prevent SQL injection and enable efficient date-range filtering.

    Schema:
        symbol    VARCHAR NOT NULL
        timeframe VARCHAR NOT NULL DEFAULT '1D'
        date      DATE NOT NULL
        open      DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE
        volume    DOUBLE, traded_value DOUBLE
    """

    def __init__(self, db_path: str | Path, timeframe: str = "1D") -> None:
        """Initialize loader.

        Args:
            db_path: path to .duckdb file
            timeframe: timeframe filter (default 1D)

        Raises:
            FileNotFoundError: if db_path doesn't exist
        """
        self.db_path = Path(db_path)
        self.timeframe = timeframe
        if not self.db_path.exists():
            raise FileNotFoundError(f"DuckDB file not found: {self.db_path}")

    def _conn(self) -> duckdb.DuckDBPyConnection:
        """Get read-only connection to DB."""
        return duckdb.connect(str(self.db_path), read_only=True)

    def list_symbols(self) -> list[str]:
        """List all unique symbols in the DB."""
        try:
            conn = self._conn()
            result = conn.execute(
                "SELECT DISTINCT symbol FROM ohlcv WHERE timeframe = ? ORDER BY symbol",
                [self.timeframe],
            ).fetchall()
            return [r[0] for r in result]
        finally:
            conn.close()

    def load_symbol(self, symbol: str, start_date: date | None = None, end_date: date | None = None) -> pd.DataFrame:
        """Load single symbol OHLCV data.

        Args:
            symbol: symbol to load
            start_date: optional start date filter (inclusive)
            end_date: optional end date filter (inclusive)

        Returns:
            DataFrame with columns [date, open, high, low, close, volume, symbol]

        Raises:
            ValueError: if no data found for symbol
        """
        try:
            conn = self._conn()
            query = "SELECT date, open, high, low, close, volume FROM ohlcv WHERE symbol = ? AND timeframe = ?"
            params = [symbol, self.timeframe]

            if start_date:
                query += " AND date >= ?"
                params.append(start_date)
            if end_date:
                query += " AND date <= ?"
                params.append(end_date)

            query += " ORDER BY date"

            result = conn.execute(query, params).fetchdf()

            if result.empty:
                raise ValueError(f"no data found for {symbol}")

            result["symbol"] = symbol
            result = _sanitize_prices(result)
            return result[["date", "open", "high", "low", "close", "volume", "symbol"]]
        finally:
            conn.close()

    def load_many(
        self, symbols: list[str], start_date: date | None = None, end_date: date | None = None
    ) -> pd.DataFrame:
        """Load multiple symbols into single DataFrame.

        Args:
            symbols: list of symbols
            start_date: optional start date filter
            end_date: optional end date filter

        Returns:
            Concatenated DataFrame with all symbols and columns [date, open, high, low, close, volume, symbol]
        """
        if not symbols:
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "symbol"])

        try:
            conn = self._conn()
            placeholders = ",".join(["?" for _ in symbols])
            query = f"SELECT date, open, high, low, close, volume, symbol FROM ohlcv WHERE symbol IN ({placeholders}) AND timeframe = ?"
            params = symbols + [self.timeframe]

            if start_date:
                query += " AND date >= ?"
                params.append(start_date)
            if end_date:
                query += " AND date <= ?"
                params.append(end_date)

            query += " ORDER BY symbol, date"

            result = conn.execute(query, params).fetchdf()
            if result.empty:
                return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "symbol"])
            return _sanitize_prices(result)
        finally:
            conn.close()

    def load_date_range(
        self, symbols: list[str], start_date: date, end_date: date
    ) -> pd.DataFrame:
        """Load multiple symbols for a specific date range.

        Convenience method that enforces date bounds.

        Args:
            symbols: list of symbols
            start_date: start date (inclusive)
            end_date: end date (inclusive)

        Returns:
            DataFrame filtered to [start_date, end_date]
        """
        return self.load_many(symbols, start_date=start_date, end_date=end_date)

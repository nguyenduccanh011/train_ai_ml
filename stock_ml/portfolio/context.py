"""PortfolioContext: abstracts the data sources so backtest and serving share ONE
portfolio implementation and differ only in where frames come from.

duckdb/sqlite are imported lazily INSIDE the concrete context (design doc §6:
duckdb stays out of the stock_ml_core wheel dependencies).
"""
from __future__ import annotations

import pandas as pd


class PortfolioContext:
    """Interface. Implementations must pin: market panel source (conviction is a
    cross-sectional rank — panel policy MUST match between backtest and serving),
    NAV-mark price source, and the date_hi cutoff."""

    def market_frame(self, start: str) -> pd.DataFrame:
        """Full-market OHLC slice: symbol,date,low,close,high — from `start`, no upper cut."""
        raise NotImplementedError

    def meta_frame(self, symbols: list[str]) -> pd.DataFrame:
        """OHLCV for trade symbols: symbol,date,open,high,low,close,volume."""
        raise NotImplementedError

    def price_frame(self, symbols: list[str], date_lo: str) -> pd.DataFrame:
        """NAV-mark closes: symbol,date,close in [date_lo, self.date_hi]."""
        raise NotImplementedError


class DuckDBContext(PortfolioContext):
    """Backtest/replay context: conviction+meta panels from a market.duckdb,
    NAV marked on a sqlite ohlcv store (same layout as serving's ohlcv.db)."""

    def __init__(self, market_db: str, ohlcv_db: str, date_hi: str):
        self.market_db = market_db
        self.ohlcv_db = ohlcv_db
        self.date_hi = date_hi

    def market_frame(self, start: str) -> pd.DataFrame:
        import duckdb
        cx = duckdb.connect(self.market_db, read_only=True)
        px = cx.execute("SELECT symbol,date,low,close,high FROM ohlcv WHERE timeframe='1D' AND date>=? "
                        "ORDER BY symbol,date", [start]).fetchdf()
        cx.close()
        return px

    def meta_frame(self, symbols: list[str]) -> pd.DataFrame:
        import duckdb
        d = duckdb.connect(self.market_db, read_only=True); ph = ",".join("?" * len(symbols))
        q = d.execute(f"select symbol,date,open,high,low,close,volume from ohlcv where timeframe='1D' "
                      f"and symbol in ({ph}) order by symbol,date", symbols).fetchdf()
        d.close()
        return q

    def price_frame(self, symbols: list[str], date_lo: str) -> pd.DataFrame:
        import sqlite3
        con = sqlite3.connect(self.ohlcv_db)
        ph = ",".join("?" * len(symbols))
        px = pd.read_sql_query(
            f"SELECT symbol,date,close FROM ohlcv WHERE symbol IN ({ph}) AND date>=? AND date<=?",
            con, params=list(symbols) + [date_lo, self.date_hi])
        con.close()
        return px

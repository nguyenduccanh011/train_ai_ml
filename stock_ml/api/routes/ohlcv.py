"""OHLCV (candlestick) endpoints — reads the DuckDB OLAP store (read-only).

Feeds the chart-comparison dashboard, which overlays per-model buy/sell markers
on top of the price candles served here.
"""

from __future__ import annotations

import calendar
import logging
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from stock_ml.api.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["ohlcv"])

# Intraday OHLCV (real 15m/1H bars) lives in the derivatives CSV dataset, NOT in market.duckdb
# (whose `date` column is DATE-typed -> any "15m" rows there are collapsed to one per day). Mounted
# into the API container at /portable_data. Bars are stored UTC; the VN futures session is shown in
# local time (UTC+7) so the chart axis reads 09:00-14:45 — register_leaderboard.py stores the trade
# entry/exit timestamps in the SAME local wall-clock, so markers land on the right candle.
_INTRADAY_CSV_BASE = Path("/portable_data/derivatives_ai_dataset")
_INTRADAY_TZ = "Asia/Ho_Chi_Minh"


def _is_intraday(timeframe: str) -> bool:
    """Minute ('15m') and hour ('1H') timeframes are intraday; '1D'/'1W'/'1M' are not.
    (Lowercase 'm' = minute, uppercase 'M' = month, so test the last char case-sensitively.)"""
    return bool(timeframe) and timeframe[-1] in ("m", "H", "h")


@lru_cache(maxsize=16)
def _load_intraday_csv(symbol: str, timeframe: str) -> tuple[dict, ...] | None:
    """Real intraday candles from the derivatives CSV (local-time epoch). None if no CSV for this
    symbol/timeframe -> caller falls back to the DuckDB (daily) store. Cached: the CSV is static."""
    path = _INTRADAY_CSV_BASE / f"symbol={symbol}" / f"timeframe={timeframe}" / "data.csv"
    if not path.exists():
        return None
    import pandas as pd

    df = pd.read_csv(path)
    # UTC bars -> VN local wall-clock (naive); timegm then yields the epoch lightweight-charts shows.
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(_INTRADAY_TZ).dt.tz_localize(None)
    rows: list[dict] = []
    for t, o, h, low, c, v in zip(ts, df["open"], df["high"], df["low"], df["close"], df["volume"]):
        rows.append(
            {
                "time": int(calendar.timegm(t.timetuple())),
                "open": float(o), "high": float(h), "low": float(low), "close": float(c),
                "volume": float(v) if pd.notna(v) else 0.0,
            }
        )
    return tuple(rows)


def _load_ohlcv(symbol: str, timeframe: str) -> list[dict]:
    """Intraday -> derivatives CSV (real bars, local-time epoch); else DuckDB daily candles.
    Synchronous — runs in a threadpool to avoid blocking the loop."""
    if _is_intraday(timeframe):
        csv_rows = _load_intraday_csv(symbol, timeframe)
        if csv_rows is not None:
            return list(csv_rows)

    from stock_ml.src.data.duckdb_loader import DuckDBLoader

    db_path = Path(settings.stock_data_dir)
    if not db_path.exists():
        raise FileNotFoundError(f"OHLCV store not found: {db_path}")

    loader = DuckDBLoader(db_path, timeframe=timeframe)
    try:
        df = loader.load_symbol(symbol)
    except ValueError:
        # No rows for this symbol/timeframe — return empty rather than 500.
        return []

    rows: list[dict] = []
    for rec in df.itertuples(index=False):
        rows.append(
            {
                "time": rec.date.strftime("%Y-%m-%d"),
                "open": float(rec.open),
                "high": float(rec.high),
                "low": float(rec.low),
                "close": float(rec.close),
                "volume": float(rec.volume) if rec.volume is not None else 0.0,
            }
        )
    return rows


@router.get("/ohlcv/{symbol}")
async def get_ohlcv(symbol: str, timeframe: str = "1D") -> dict:
    """Return OHLCV candles for a symbol in lightweight-charts format."""
    symbol = symbol.strip().upper()
    if not symbol:
        raise HTTPException(status_code=400, detail="symbol is required")
    try:
        rows = await run_in_threadpool(_load_ohlcv, symbol, timeframe)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "ohlcv": rows,
        "count": len(rows),
    }

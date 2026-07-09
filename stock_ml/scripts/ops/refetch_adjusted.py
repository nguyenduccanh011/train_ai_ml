"""Refetch full 1D OHLCV history for the production universe from the Sieu Tin Hieu API.

Writes RAW (as-served, still partially unadjusted) bars to market_data/market_raw_api.duckdb.
Resumable: re-running skips symbols already marked done in the _fetched manifest, so it can
be invoked repeatedly if the connection drops or the API rate-limits us.

Usage:
    python stock_ml/scripts/refetch_adjusted.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import duckdb
import pandas as pd
import requests

API = "https://sieutinhieu.vn/api/v1"
PROD_DB = "market_data/market.duckdb"
RAW_DB = "market_data/market_raw_api.duckdb"
TIMEFRAME = "1D"
PAGE = 1000          # API max limit
MAX_RETRY = 6
BACKOFF = 2.0        # seconds, exponential
SLEEP = 0.10         # polite delay between requests


def prod_symbols() -> list[str]:
    con = duckdb.connect(PROD_DB, read_only=True)
    try:
        rows = con.execute(
            "SELECT DISTINCT symbol FROM ohlcv WHERE timeframe = ? ORDER BY symbol", [TIMEFRAME]
        ).fetchall()
        return [r[0] for r in rows]
    finally:
        con.close()


def try_page(sess: requests.Session, symbol: str, offset: int, limit: int, retries: int = 3):
    """Fetch one page; return (json|None, ok). Does not raise."""
    last = None
    for attempt in range(retries):
        try:
            r = sess.get(f"{API}/ohlcv/",
                         params={"symbol": symbol, "timeframe": TIMEFRAME, "limit": limit, "offset": offset},
                         timeout=40)
            if r.status_code == 200:
                return r.json(), True
            last = f"HTTP {r.status_code}: {r.text[:80]}"
        except Exception as e:  # noqa: BLE001
            last = repr(e)
        time.sleep(BACKOFF * (attempt + 1))
    return last, False


def fetch_symbol(sess: requests.Session, symbol: str) -> pd.DataFrame:
    """Page through the full history (newest-first), advancing by rows actually returned.

    The API 500s on the final partial page when offset+limit >= total, and occasionally
    under load. On a persistent 500 we halve the limit and retry the same offset, so we
    recover everything except (at worst) the single oldest bar.
    """
    items: list[dict] = []
    offset, limit, total = 0, PAGE, None
    while True:
        d, ok = try_page(sess, symbol, offset, limit)
        if not ok:
            if limit > 64:
                limit //= 2
                continue
            print(f"   {symbol} giving up tail at offset={offset}: {d}", flush=True)
            break
        batch = d.get("items", [])
        total = d.get("total", total)
        items.extend(batch)
        got = len(batch)
        if got == 0:
            break
        offset += got
        if total is not None and offset >= total:
            break
        limit = PAGE
        time.sleep(SLEEP)
    if not items:
        return pd.DataFrame(columns=["symbol", "date", "open", "high", "low", "close", "volume"])
    df = pd.DataFrame(items)
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["symbol"] = symbol
    df = df[["symbol", "date", "open", "high", "low", "close", "volume"]]
    df = df.dropna(subset=["close"]).drop_duplicates(subset=["date"]).sort_values("date")
    return df


def ensure_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """CREATE TABLE IF NOT EXISTS ohlcv_raw (
            symbol VARCHAR, date DATE, open DOUBLE, high DOUBLE, low DOUBLE,
            close DOUBLE, volume DOUBLE
        )"""
    )
    con.execute(
        """CREATE TABLE IF NOT EXISTS _fetched (
            symbol VARCHAR PRIMARY KEY, nbars INTEGER, dmin DATE, dmax DATE, status VARCHAR
        )"""
    )


def main() -> int:
    Path(RAW_DB).parent.mkdir(parents=True, exist_ok=True)
    symbols = prod_symbols()
    con = duckdb.connect(RAW_DB)
    ensure_schema(con)
    done = {r[0] for r in con.execute("SELECT symbol FROM _fetched WHERE status='ok'").fetchall()}
    todo = [s for s in symbols if s not in done]
    print(f"universe={len(symbols)} done={len(done)} todo={len(todo)}", flush=True)

    sess = requests.Session()
    ok = fail = 0
    for i, sym in enumerate(todo, 1):
        try:
            df = fetch_symbol(sess, sym)
            con.execute("DELETE FROM ohlcv_raw WHERE symbol = ?", [sym])
            if len(df):
                con.register("df_tmp", df)
                con.execute("INSERT INTO ohlcv_raw SELECT * FROM df_tmp")
                con.unregister("df_tmp")
            con.execute(
                "INSERT OR REPLACE INTO _fetched VALUES (?,?,?,?,?)",
                [sym, len(df), (df["date"].min() if len(df) else None),
                 (df["date"].max() if len(df) else None), "ok"],
            )
            ok += 1
            print(f"[{i}/{len(todo)}] {sym} nbars={len(df)} "
                  f"{df['date'].min() if len(df) else '-'}..{df['date'].max() if len(df) else '-'}",
                  flush=True)
        except Exception as e:  # noqa: BLE001
            fail += 1
            con.execute("INSERT OR REPLACE INTO _fetched VALUES (?,?,?,?,?)", [sym, 0, None, None, "fail"])
            print(f"[{i}/{len(todo)}] {sym} FAIL {e!r}", flush=True)
        time.sleep(SLEEP)
    con.close()
    print(f"DONE ok={ok} fail={fail}", flush=True)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

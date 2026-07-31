"""DuckDB-based OHLCV loader — efficient SQL queries over parquetized datasets."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

_PRICE_COLS = ["open", "high", "low", "close"]


def _sanitize_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill non-positive OHLC prices and repair OHLC-coherence violations.

    (1) Non-positive prices: a handful of rows carry close/open/high/low == 0 (glitches).
    Because most features divide by price or a rolling Min/Corr of price, ONE zero poisons
    the whole trailing window with ±inf. We mask non-positive OHLC to NaN and forward-fill
    per symbol. Volume is left as-is (0 = a legitimate no-trade day).

    (2) OHLC-coherence: some rows are positive but inconsistent — open/close OUTSIDE
    [low, high], or high < low (vendor glitches, e.g. HCM 2020-06-05 open 4.80 vs low 6.66;
    ACV 2021-07 open > high; VTP 2022-05 close < low). These are *finite* so they survive
    the fail-loud NaN guard while silently corrupting candle-geometry features (clv,
    wick/body ratios, close_position_in_range). Inspection of the real violations shows the
    outlier is almost always ONE of open/high/close while the other three prices agree, so
    the minimal-damage repair is: trust the traded range [low, high] and CLAMP the offending
    open/close back into it (this pulls HCM's bogus 4.80 open up to the low, and ACV's 44.42
    open down to the high). Only in the rare high < low case do we swap them first.

    Forward-fill only (no back-fill): back-filling a leading bad bar would pull a FUTURE
    price into it (look-ahead). A leading NaN survives and is trimmed by feature warmup.
    """
    cols = [c for c in _PRICE_COLS if c in df.columns]
    if df.empty or not cols:
        return df
    df = df.copy()
    # (1) non-positive -> NaN -> ffill (per symbol; forward-only to avoid look-ahead)
    bad = df[cols] <= 0
    if bad.to_numpy().any():
        df[cols] = df[cols].mask(bad)
        if "symbol" in df.columns:
            df[cols] = df.groupby("symbol")[cols].transform(lambda g: g.ffill())
        else:
            df[cols] = df[cols].ffill()
    # (2) OHLC-coherence: trust [low, high] as the traded range; clamp open/close into it.
    have = all(c in df.columns for c in _PRICE_COLS)
    if have:
        # rare high<low: swap so the range is well-ordered before clamping
        swap = df["high"] < df["low"]
        if swap.to_numpy().any():
            hi_s, lo_s = df.loc[swap, "low"].copy(), df.loc[swap, "high"].copy()
            df.loc[swap, "high"], df.loc[swap, "low"] = hi_s, lo_s
        df["open"] = df["open"].clip(lower=df["low"], upper=df["high"])
        df["close"] = df["close"].clip(lower=df["low"], upper=df["high"])
    return df


def _trim_leading_phantom(df: pd.DataFrame) -> pd.DataFrame:
    """Drop each symbol's leading pre-listing PHANTOM block before real trading begins.

    Some symbols carry a long backfill head of o==h==l==c bars (no intraday range = no real
    trade) with placeholder volume, stepping the price up level-by-level, then jump to the
    true IPO price (e.g. VHM: 710 leading flat bars 2011-2014 with vol 0/5000, then a +268%
    step to the real 2018-05-17 listing at 64.36). These bars are FINITE so they survive the
    fail-loud NaN guard, yet they poison every rolling feature (std/ret/vol-ratio) and inject
    a fake +268% one-bar return the moment the symbol enters a train fold.

    A single "high > low" bar is NOT enough to mark the start of trading: the VHM backfill
    stepped its price with occasional 1-bar ranges while median volume stayed 0 for years.
    Real continuous trading is when intraday ranges become the NORM, not the exception. So we
    find the first date from which the next 20 sessions are MAJORITY real-range (>= half have
    high > low) and drop everything before it, cutting VHM's entire 2011-2018 placeholder head
    (incl. the +268% IPO seam) while costing a genuinely-listed name at most its warmup rows.
    Volume is deliberately not used (placeholder volumes are unreliable and legitimately-thin
    real names must not be trimmed).
    """
    need = ["open", "high", "low", "close"]
    if df.empty or not all(c in df.columns for c in need) or "symbol" not in df.columns:
        return df

    WIN, FRAC = 20, 0.5

    def _trim(g: pd.DataFrame) -> pd.DataFrame:
        traded = (g["high"] > g["low"]).to_numpy().astype(float)
        if traded.sum() == 0:
            return g.iloc[0:0]  # never actually traded in this window
        # forward-looking fraction of real-range bars over the next WIN sessions
        fwd = pd.Series(traded[::-1]).rolling(WIN, min_periods=1).mean().to_numpy()[::-1]
        ok = fwd >= FRAC
        g = g.iloc[int(ok.argmax()):] if ok.any() else g.iloc[int(traded.argmax()):]
        # A backfill head can pass the range-majority test yet still end in an IPO/re-listing
        # SEAM: a single >40% price jump from the placeholder level to the true opening price
        # (VHM: 17.47 flat -> 64.36 on 2018-05-17, +268%). If such a jump sits in the first 120
        # bars, the real listing starts AT the jump — drop everything up to and including it.
        if len(g) > 1:
            head = g.head(120)
            jump = (head["close"].pct_change().abs() > 0.40).to_numpy()
            if jump.any():
                last_jump = len(jump) - 1 - int(jump[::-1].argmax())
                g = g.iloc[last_jump:]
        return g

    return (
        df.sort_values(["symbol", "date"])
        .groupby("symbol", group_keys=False)
        .apply(_trim)
        .reset_index(drop=True)
    )


def ensure_symbols_cached(
    db_path: str | Path, symbols: list[str], *, timeframe: str = "1D"
) -> list[str]:
    """§13.3.1 fetch-on-miss: make sure every requested symbol has bars in the local cache.

    The corrected universe (§13.9) surfaces survivorship-correct names (ROS, delisted tickers) the
    local back-adjusted cache never had. Without this they'd be SILENTLY dropped at load time
    (``requested = [s for s in symbols if s in available]``), quietly shrinking the universe back to
    the survivors — the exact bias the fix removes. So: find the symbols missing from the cache, fetch
    their full back-adjusted history from the source, and upsert into the ``ohlcv`` table (the local
    file is a cache of the source, §13.3.1). Fail-loud inside the client if the source is unreachable.

    Returns the list of symbols actually fetched (empty when the cache already had everything).
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"ensure_symbols_cached: DuckDB file not found: {db_path}")
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        have = {r[0] for r in con.execute(
            "SELECT DISTINCT symbol FROM ohlcv WHERE timeframe = ?", [timeframe]).fetchall()}
    finally:
        con.close()
    missing = [s for s in dict.fromkeys(symbols) if s not in have]
    if not missing:
        return []

    from src.data.sieutinhieu import fetch_history

    # Per-symbol tolerant: a symbol the universe endpoint ranks (it has matched-flow) but whose /ohlcv/
    # 500s has NO price series on the source — it is UNTRADEABLE (no bars to backtest), so drop it with
    # a loud warning rather than aborting the whole universe. Only a fully-unreachable source (nothing
    # fetched) is fatal, matching the fail-loud contract (§1.1) without punishing fringe data gaps.
    frames, failed = [], []
    for sym in missing:
        try:
            frames.append(fetch_history(sym))
        except Exception as e:  # noqa: BLE001 — collect, decide after the loop
            failed.append((sym, str(e)[:80]))
    if failed:
        print(f"[cache] WARNING {len(failed)}/{len(missing)} symbol(s) have NO OHLCV on source "
              f"(untradeable → dropped): {[s for s, _ in failed]}")
    if not frames:
        # Nothing fetched. Distinguish a DOWN source (connection error → abort, we must not run on a
        # silently-shrunk universe) from "every missing symbol genuinely has no price series"
        # (per-symbol HTTP error → they are untradeable; proceed with zero fetched, they drop at load).
        conn_errs = [f for f in failed if "HTTP Error" not in f[1]]
        if conn_errs:
            raise RuntimeError(
                f"ensure_symbols_cached: source unreachable — {len(conn_errs)} connection failure(s) "
                f"(e.g. {conn_errs[:3]})")
        return []
    df = pd.concat(frames, ignore_index=True)
    df["timeframe"] = timeframe
    # Local convention: traded_value = volume*close (the endpoint doesn't ship it, and the universe
    # resolver no longer relies on it — it reads matched ADTV from the source).
    df["traded_value"] = df["volume"] * df["close"]
    cols = ["symbol", "timeframe", "date", "open", "high", "low", "close", "volume", "traded_value"]
    con = duckdb.connect(str(db_path))  # write connection
    try:
        con.register("_incoming", df[cols])
        con.execute(f"INSERT INTO ohlcv ({', '.join(cols)}) SELECT {', '.join(cols)} FROM _incoming")
        con.unregister("_incoming")
    finally:
        con.close()
    fetched = sorted(df["symbol"].unique())
    print(f"[cache] fetched-on-miss {len(fetched)} symbol(s) into {db_path.name}: {fetched[:8]}"
          + ("…" if len(fetched) > 8 else ""))
    return fetched


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
            result = _trim_leading_phantom(result)
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
            return _trim_leading_phantom(_sanitize_prices(result))
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

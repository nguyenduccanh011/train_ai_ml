"""Back-adjust unadjusted corporate-action gaps in the refetched 1D data.

Detection (exchange-aware, rock-solid): on every VN board the opening auction caps the
open within +/- band of the (already CA-adjusted) reference price, so an OVERNIGHT gap
beyond the band can ONLY mean the reference was reset by a corporate action that history
was not back-adjusted for. The adjustment factor is exactly open / prev_close.

  band: HOSE 7%, HNX 10%, UPCOM 15% (+1.5% buffer); unknown -> UPCOM.
  candidate ex-date: consecutive session (calendar gap <= 5d), prev_close > 0,
                     open/prev_close - 1 <= -(band + buffer).

A multi-day floor crash never triggers (each session only floors WITHIN the band, so the
open stays within band of the prior reference). Factors are applied CUMULATIVELY: each row
is multiplied by the product of factors of all ex-dates strictly after it (prices), volume
divided by the same (share-count grows), so the latest segment is left untouched and equals
the current live price -- matching how vendors anchor an adjusted series to the latest bar.

traded_value (not served by the API) is the real, adjustment-invariant turnover: taken from
the old prod db where (symbol, date) overlaps, else close_raw * volume_raw.

Usage:
    python stock_ml/scripts/back_adjust.py            # dry-run: writes audit CSV only
    python stock_ml/scripts/back_adjust.py --apply     # also writes market_adjusted.duckdb
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

RAW_DB = "market_data/market_raw_api.duckdb"
PROD_DB = "market_data/market.duckdb"
OUT_DB = "market_data/market_adjusted.duckdb"
EXCH_MAP = "market_data/symbol_exchange.json"
AUDIT = "market_data/back_adjust_audit.csv"

# Tier 1 (universal): no VN board (HOSE 7%, HNX 10%, UPCOM 15%) permits a single-session
# move beyond +/-15%, so an overnight gap past this is unambiguously a corporate-action
# reference reset -- regardless of exchange or era. Catches every big CA.
GAP_THR = -0.155
TOL = 0.16  # ex-date close must trade within this of its (reset) open

# Tier 2 (exchange-band, HOSE/HNX only): an overnight gap beyond the board band but under
# 15.5% is also impossible without a reference reset -- BUT only if the stock was actually
# on that board then (an UPCOM-era or illiquid stock routinely gaps >band, and exchange
# migrations / bogus pre-listing data mimic CAs). We gate each candidate on a DATA-DRIVEN
# local-band guard: in +/-40 bars around it the stock's typical overnight move must sit
# within the band (else it was on a wider band / illiquid -> skip), plus persistence (the
# drop must hold, not bounce back). This auto-excludes migration (ACB, GVR), bogus (VHM),
# and illiquid clusters (PGV, TDM) without a hardcoded list. UPCOM is skipped (its band is
# 15%, so sub-15.5% is negligible and indistinguishable from normal moves).
BANDS = {"HOSE": 0.07, "HNX": 0.10}
LOCAL_WIN = 40           # bars each side for the local-band estimate
LOCAL_MARGIN = 0.005     # a bar counts as "beyond band" only if it clears band by this much
MAX_BEYOND = 0           # a truly band-capped stock has ZERO non-CA overnight gaps past band;
                         # any => wider-band era (UPCOM/HNX migration), crash floors, or illiquid -> skip
MIN_WIN_BARS = 30        # need this many real surrounding sessions (excludes sparse pre-listing data)
PERSIST_MAX = 0.93       # median(next-5 close)/prev_close must be <= this (stayed down)
T2_MIN_DATE = "2015-01-01"  # tier2 only in the training era; pre-2015 has unreliable exchange/era data
PRICE_COLS = ["open", "high", "low", "close"]


def detect_events(df: pd.DataFrame, band: float | None = None, crash_dates: set | None = None) -> pd.DataFrame:
    """df: one symbol, sorted by date. Return ex-date corporate actions (tier1 + tier2).

    Common guards (both tiers): consecutive session (dgap<=5), both sessions traded
    (volume>0), and the ex-date close trades near its reset open (rejects scale errors).

    tier1 (universal): open-gap <= GAP_THR (-15.5%). Always a CA.
    tier2 (HOSE/HNX only, band passed in): open-gap in (-15.5%, -(band+buf)] AND the stock's
      local overnight volatility sits within the band (data-driven migration/illiquidity
      guard) AND the drop persists. Captures sub-15% dividends on genuinely-banded names.
    """
    d = df.reset_index(drop=True).copy()
    d["prev_close"] = d["close"].shift(1)
    d["prev_vol"] = d["volume"].shift(1)
    d["dgap"] = (pd.to_datetime(d["date"]) - pd.to_datetime(d["date"]).shift(1)).dt.days
    d["open_gap"] = d["open"] / d["prev_close"] - 1.0
    d["close_open"] = d["close"] / d["open"] - 1.0

    base = (
        (d["prev_close"] > 0) & (d["open"] > 0) & (d["dgap"] <= 5)
        & (d["volume"] > 0) & (d["prev_vol"] > 0)
    )
    t1 = base & (d["open_gap"] <= GAP_THR) & (d["close_open"].abs() <= TOL)
    ev = d[t1].copy()
    ev["tier"] = 1

    if band is not None:
        thr2 = -(band + 0.015)
        cand = (
            base & (d["open_gap"] <= thr2) & (d["open_gap"] > GAP_THR)
            & (d["close_open"].abs() <= band + 0.02)
            & (pd.to_datetime(d["date"]) >= pd.Timestamp(T2_MIN_DATE))
        )
        if crash_dates:
            cand &= ~d["date"].isin(crash_dates)  # market-crash gap-downs are not corporate actions
        og = d["open_gap"].to_numpy()
        dgaps = d["dgap"].to_numpy()
        keep = []
        for i in d.index[cand]:
            lo, hi = max(0, i - LOCAL_WIN), min(len(d), i + LOCAL_WIN + 1)
            beyond = valid = 0
            for j in range(lo, hi):
                if abs(j - i) <= 2:          # skip the event and its neighbours
                    continue
                g = og[j]
                if pd.isna(g) or dgaps[j] > 5:  # skip resumption gaps (not normal sessions)
                    continue
                valid += 1
                ag = abs(g)
                if ag >= abs(GAP_THR):       # skip other CA-sized gaps
                    continue
                if ag > band + LOCAL_MARGIN:
                    beyond += 1
            if valid < MIN_WIN_BARS or beyond > MAX_BEYOND:  # sparse, or not hard-capped at band
                continue
            nxt = d["close"].iloc[i + 1:i + 6]
            if len(nxt) and nxt.median() / d["prev_close"].iloc[i] <= PERSIST_MAX:
                keep.append(i)
        ev2 = d.loc[keep].copy()
        ev2["tier"] = 2
        ev = pd.concat([ev, ev2], ignore_index=True)

    ev["factor"] = ev["open"] / ev["prev_close"]
    return ev[["date", "prev_close", "open", "close", "open_gap", "factor", "tier"]]


def adjust_symbol(df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Apply cumulative back-adjustment. events.date are ex-dates with factor<1."""
    d = df.sort_values("date").reset_index(drop=True).copy()
    cum = np.ones(len(d), dtype=float)
    dates = pd.to_datetime(d["date"]).values
    for _, e in events.iterrows():
        exd = np.datetime64(pd.to_datetime(e["date"]))
        # rows STRICTLY before the ex-date get scaled by this factor
        cum[dates < exd] *= e["factor"]
    for c in PRICE_COLS:
        d[c] = d[c] * cum
    d["volume"] = d["volume"] / cum
    return d


def main() -> int:
    apply = "--apply" in sys.argv
    exch = json.load(open(EXCH_MAP)) if Path(EXCH_MAP).exists() else {}

    con = duckdb.connect(RAW_DB, read_only=True)
    raw = con.execute(
        "SELECT symbol, date, open, high, low, close, volume FROM ohlcv_raw ORDER BY symbol, date"
    ).fetchdf()
    con.close()
    print(f"raw rows={len(raw)} symbols={raw['symbol'].nunique()}", flush=True)

    import re
    # market-crash dates: a broad down day (median close-to-close <= -3%) gaps many stocks at
    # once -> a tier2 gap there is market-driven, not an idiosyncratic ex-date. Exclude them.
    mr = raw.copy()
    mr["ret"] = mr.groupby("symbol", sort=False)["close"].pct_change()
    med = mr.groupby("date")["ret"].median()
    crash_dates = set(med[med <= -0.03].index)
    print(f"market-crash dates excluded from tier2: {len(crash_dates)}", flush=True)

    audit_rows = []
    adjusted_parts = []
    for sym, g in raw.groupby("symbol", sort=False):
        if re.search(r"F\d+M$", sym) or sym.startswith("VN30F") or sym.startswith("VN100F"):
            adjusted_parts.append(g.sort_values("date"))  # derivatives have no corporate actions
            continue
        ev = detect_events(g, band=BANDS.get(exch.get(sym)), crash_dates=crash_dates)
        if len(ev):
            for _, e in ev.iterrows():
                audit_rows.append({
                    "symbol": sym, "exchange": exch.get(sym), "ex_date": e["date"],
                    "tier": int(e["tier"]),
                    "prev_close": round(e["prev_close"], 3), "ex_open": round(e["open"], 3),
                    "ex_close": round(e["close"], 3),
                    "open_gap": round(e["open_gap"], 4), "factor": round(e["factor"], 5),
                })
        adjusted_parts.append(adjust_symbol(g, ev) if len(ev) else g.sort_values("date"))

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(AUDIT, index=False)
    n_ev = len(audit)
    n_sym = audit["symbol"].nunique() if n_ev else 0
    print(f"corporate-action ex-dates detected: {n_ev} across {n_sym} symbols", flush=True)
    if n_ev:
        print("by exchange:", audit["exchange"].value_counts().to_dict(), flush=True)
        print(f"audit -> {AUDIT}", flush=True)

    if not apply:
        print("DRY-RUN (no db written). Re-run with --apply to build", OUT_DB, flush=True)
        return 0

    adj = pd.concat(adjusted_parts, ignore_index=True)
    # real, adjustment-invariant traded_value: prod where available else close_raw*vol_raw
    raw_tv = raw.assign(raw_tv=raw["close"] * raw["volume"])[["symbol", "date", "raw_tv"]]
    con = duckdb.connect(PROD_DB, read_only=True)
    prod_tv = con.execute(
        "SELECT symbol, date, traded_value FROM ohlcv WHERE timeframe='1D'"
    ).fetchdf()
    intraday = con.execute(
        "SELECT symbol, timeframe, date, open, high, low, close, volume, traded_value "
        "FROM ohlcv WHERE timeframe <> '1D'"
    ).fetchdf()
    con.close()

    adj = adj.merge(prod_tv, on=["symbol", "date"], how="left").merge(raw_tv, on=["symbol", "date"], how="left")
    adj["traded_value"] = adj["traded_value"].fillna(adj["raw_tv"])
    adj["timeframe"] = "1D"
    out = adj[["symbol", "timeframe", "date", "open", "high", "low", "close", "volume", "traded_value"]]

    Path(OUT_DB).unlink(missing_ok=True)
    con = duckdb.connect(OUT_DB)
    con.execute(
        """CREATE TABLE ohlcv (symbol VARCHAR, timeframe VARCHAR, date DATE,
            open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume DOUBLE, traded_value DOUBLE)"""
    )
    con.register("out_tmp", out)
    con.execute("INSERT INTO ohlcv SELECT * FROM out_tmp")
    con.unregister("out_tmp")
    if len(intraday):
        con.register("intra_tmp", intraday)
        con.execute("INSERT INTO ohlcv SELECT symbol, timeframe, date, open, high, low, close, "
                    "volume, traded_value FROM intra_tmp")
        con.unregister("intra_tmp")
    n = con.execute("SELECT count(*), count(distinct symbol) FROM ohlcv WHERE timeframe='1D'").fetchone()
    con.close()
    print(f"WROTE {OUT_DB}: 1D rows={n[0]} symbols={n[1]} (+intraday {len(intraday)})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

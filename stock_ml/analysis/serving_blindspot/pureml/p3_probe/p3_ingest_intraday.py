# -*- coding: utf-8 -*-
"""P3 Phase A — Ingest VN30F1M + VN30F2M 1m (2018 -> nay) tu Entrade/DNSE chart API
vao DB MOI market_data/market_intraday.duckdb (KHONG dung DB production de ghi).

Schema theo thiet ke P3_INTRADAY_PROBE.md muc 4a:
  ohlcv_intraday(symbol, timeframe, ts TIMESTAMP UTC, o/h/l/c/v, source,
                 PRIMARY KEY (symbol, timeframe, ts))
  _fetch_manifest(symbol, year, nbars, ts_min, ts_max, status, PK (symbol, year))

Verify sau fetch:
  (a) row count / bar-per-day theo nam (ky vong F1M ~243, 241 tu 2025)
  (b) close 1m cuoi ngay vs close 1D production market.duckdb (READ-ONLY) — % khop
  (c) map lo hong coverage vs lich giao dich 1D production (dac biet 2023-03..09)

Khuon fetch theo p3_01_entrade_probe.py / refetch_adjusted.py (sleep + retry/backoff).
"""
from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

import duckdb
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
API = "https://services.entrade.com.vn/chart-api/v2/ohlcs/derivative"
HDR = {"User-Agent": "Mozilla/5.0"}
SLEEP = 0.5
DB_NEW = r"f:\PROJECTS\train_ai_ml\market_data\market_intraday.duckdb"
DB_PROD = r"f:\PROJECTS\train_ai_ml\market_data\market.duckdb"  # READ-ONLY
SYMBOLS = ["VN30F1M", "VN30F2M"]
YEAR0 = 2018
VN_TZ = dt.timezone(dt.timedelta(hours=7))


def fetch_year(sess: requests.Session, sym: str, year: int) -> pd.DataFrame | None:
    """1 request = 1 nam (probe da xac nhan tra tron goi)."""
    f = int(dt.datetime(year, 1, 1, tzinfo=VN_TZ).timestamp())
    now = dt.datetime.now(VN_TZ)
    t_end = dt.datetime(year + 1, 1, 1, tzinfo=VN_TZ)
    t = int(min(t_end, now).timestamp())
    for attempt in range(4):
        try:
            r = sess.get(API, params={"symbol": sym, "resolution": "1",
                                      "from": f, "to": t}, timeout=90, headers=HDR)
            if r.status_code == 200:
                d = r.json()
                if not d or not d.get("t"):
                    return pd.DataFrame()
                df = pd.DataFrame({k: d[k] for k in ("t", "o", "h", "l", "c", "v") if k in d})
                df = df.drop_duplicates(subset="t")
                df["ts"] = pd.to_datetime(df["t"], unit="s", utc=True).dt.tz_localize(None)
                return df
            print(f"  HTTP {r.status_code} (attempt {attempt})", flush=True)
        except requests.RequestException as e:
            print(f"  err {e} (attempt {attempt})", flush=True)
        time.sleep(2.0 * (attempt + 1))
    return None


def ensure_schema(con: duckdb.DuckDBPyConnection):
    con.execute("""
        CREATE TABLE IF NOT EXISTS ohlcv_intraday (
            symbol    VARCHAR NOT NULL,
            timeframe VARCHAR NOT NULL,
            ts        TIMESTAMP NOT NULL,
            open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume DOUBLE,
            source    VARCHAR DEFAULT 'entrade',
            PRIMARY KEY (symbol, timeframe, ts)
        )""")
    con.execute("""
        CREATE TABLE IF NOT EXISTS _fetch_manifest (
            symbol VARCHAR, year INTEGER, nbars INTEGER,
            ts_min TIMESTAMP, ts_max TIMESTAMP, status VARCHAR,
            PRIMARY KEY (symbol, year))""")


def upsert(con, sym: str, df: pd.DataFrame) -> int:
    tmp = df.rename(columns={"o": "open", "h": "high", "l": "low",
                             "c": "close", "v": "volume"})
    tmp = tmp[["ts", "open", "high", "low", "close", "volume"]].copy()
    tmp.insert(0, "timeframe", "1m")
    tmp.insert(0, "symbol", sym)
    tmp["source"] = "entrade"
    con.register("_tmp", tmp)
    con.execute("INSERT OR REPLACE INTO ohlcv_intraday SELECT * FROM _tmp")
    con.unregister("_tmp")
    return len(tmp)


def main():
    import sys
    verify_only = "--verify-only" in sys.argv
    sess = requests.Session()
    con = duckdb.connect(DB_NEW)
    ensure_schema(con)
    year_now = dt.datetime.now(VN_TZ).year

    for sym in ([] if verify_only else SYMBOLS):
        for year in range(YEAR0, year_now + 1):
            df = fetch_year(sess, sym, year)
            if df is None:
                status, n, tmin, tmax = "FAIL", 0, None, None
            elif df.empty:
                status, n, tmin, tmax = "EMPTY", 0, None, None
            else:
                n = upsert(con, sym, df)
                status, tmin, tmax = "OK", df["ts"].min(), df["ts"].max()
            con.execute("INSERT OR REPLACE INTO _fetch_manifest VALUES (?,?,?,?,?,?)",
                        [sym, year, n, tmin, tmax, status])
            print(f"{sym} {year}: {status} n={n} [{tmin} .. {tmax}]", flush=True)
            time.sleep(SLEEP)

    # ---------------- VERIFY ----------------
    rep = {"run_at": dt.datetime.now().isoformat()}
    tot = con.execute("SELECT symbol, count(*), min(ts), max(ts) FROM ohlcv_intraday "
                      "GROUP BY symbol ORDER BY symbol").fetchall()
    rep["totals"] = [{"symbol": s, "rows": int(n), "ts_min": str(a), "ts_max": str(b)}
                     for s, n, a, b in tot]
    print("\n=== TOTALS ===")
    for r in rep["totals"]:
        print(r, flush=True)

    # (a) bar/ngay theo nam (ngay theo gio VN = ts + 7h)
    bpd = con.execute("""
        WITH d AS (SELECT symbol, CAST(ts + INTERVAL 7 HOUR AS DATE) AS day, count(*) n
                   FROM ohlcv_intraday GROUP BY 1, 2)
        SELECT symbol, year(day) yr, count(*) AS n_days,
               min(n) mn, median(n) md, max(n) mx
        FROM d GROUP BY 1, 2 ORDER BY 1, 2""").df()
    rep["bars_per_day"] = bpd.to_dict("records")
    print("\n=== BAR/NGAY THEO NAM ===")
    print(bpd.to_string(index=False), flush=True)

    # (b)+(c) doi chieu production 1D (READ-ONLY)
    prod = duckdb.connect(DB_PROD, read_only=True)
    d1 = prod.execute("SELECT symbol, date, close FROM ohlcv WHERE timeframe='1D' "
                      "AND symbol IN ('VN30F1M','VN30F2M') ORDER BY symbol, date").df()
    prod.close()
    d1["date"] = pd.to_datetime(d1["date"]).dt.date

    last1m = con.execute("""
        WITH b AS (SELECT symbol, CAST(ts + INTERVAL 7 HOUR AS DATE) AS day, ts, close,
                          row_number() OVER (PARTITION BY symbol,
                              CAST(ts + INTERVAL 7 HOUR AS DATE) ORDER BY ts DESC) rn
                   FROM ohlcv_intraday)
        SELECT symbol, day, close AS close_1m_last FROM b WHERE rn = 1""").df()
    last1m["day"] = pd.to_datetime(last1m["day"]).dt.date

    mrg = d1.merge(last1m, left_on=["symbol", "date"], right_on=["symbol", "day"],
                   how="left")
    rep["close_verify"], rep["coverage_gaps"] = {}, {}
    print("\n=== VERIFY CLOSE 1m-cuoi-ngay vs 1D PRODUCTION ===")
    for sym in SYMBOLS:
        m = mrg[mrg.symbol == sym].copy()
        have = m.close_1m_last.notna()
        diff = (m.close_1m_last / m.close - 1).abs()
        v = {"prod_days": int(len(m)),
             "intraday_days": int(have.sum()),
             "match_exact": round(float((diff[have] < 1e-9).mean()), 4),
             "match_10bp": round(float((diff[have] <= 0.001).mean()), 4),
             "match_50bp": round(float((diff[have] <= 0.005).mean()), 4),
             "med_absdiff_bp": round(float(diff[have].median() * 1e4), 2)}
        rep["close_verify"][sym] = v
        print(sym, v, flush=True)
        # (c) lo hong: ngay co 1D production nhung khong co intraday
        miss = m.loc[~have, "date"]
        bymon = pd.Series(pd.to_datetime(miss)).dt.to_period("M").value_counts().sort_index()
        rep["coverage_gaps"][sym] = {str(k): int(x) for k, x in bymon.items()}
        print(f"  ngay thieu intraday: {len(miss)}; theo thang (>=3 ngay): "
              f"{ {str(k): int(x) for k, x in bymon.items() if x >= 3} }", flush=True)

    (HERE / "p3_ingest_report.json").write_text(
        json.dumps(rep, indent=2, default=str), encoding="utf-8")
    con.close()
    print("\nsaved -> p3_ingest_report.json", flush=True)


if __name__ == "__main__":
    main()

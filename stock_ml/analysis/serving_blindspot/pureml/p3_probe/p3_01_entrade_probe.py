"""P3 probe buoc 2: chat luong intraday 1m tu Entrade/DNSE chart API.

- VN30F1M + VN30F2M 1m tai cac tuan moc 2018-08 / 2019 / 2020 / 2022 / 2024 / 2026.
- HPG 1m tuan 2026 (stock chi co tu ~2026-04 tren nguon nay).
- Do gioi han window/request (keo 1 thang, 1 quy, 1 nam mot lan duoc khong).
- Luu bars -> p3_entrade_bars.parquet, metrics -> p3_entrade_report.json.
KHONG dung duckdb production.
"""
from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

import pandas as pd
import requests

OUT = Path(__file__).resolve().parent
API = "https://services.entrade.com.vn/chart-api/v2/ohlcs"
HDR = {"User-Agent": "Mozilla/5.0"}
SLEEP = 0.25
VN_TZ = dt.timezone(dt.timedelta(hours=7))

PROBES = [
    ("derivative", "VN30F1M", "2018-08-13", "2018-08-18"),
    ("derivative", "VN30F1M", "2019-03-04", "2019-03-09"),
    ("derivative", "VN30F1M", "2020-03-02", "2020-03-07"),
    ("derivative", "VN30F1M", "2022-03-07", "2022-03-12"),
    ("derivative", "VN30F1M", "2024-03-04", "2024-03-09"),
    ("derivative", "VN30F1M", "2026-06-08", "2026-06-13"),
    ("derivative", "VN30F2M", "2020-03-02", "2020-03-07"),
    ("derivative", "VN30F2M", "2024-03-04", "2024-03-09"),
    ("stock", "HPG", "2026-06-08", "2026-06-13"),
]

# do gioi han window: 1 thang / 1 quy / 1 nam
WINDOW_TESTS = [
    ("derivative", "VN30F1M", "2022-03-01", "2022-04-01", "1 thang"),
    ("derivative", "VN30F1M", "2022-01-01", "2022-04-01", "1 quy"),
    ("derivative", "VN30F1M", "2022-01-01", "2023-01-01", "1 nam"),
]


def fetch(sess, kind, sym, d0, d1, res="1"):
    f = int(dt.datetime.fromisoformat(d0).timestamp())
    t = int(dt.datetime.fromisoformat(d1).timestamp())
    t0 = time.time()
    r = sess.get(f"{API}/{kind}", params={"symbol": sym, "resolution": res, "from": f, "to": t},
                 timeout=60, headers=HDR)
    elapsed = time.time() - t0
    if r.status_code != 200:
        return None, elapsed, f"HTTP {r.status_code}"
    d = r.json()
    if not d or not d.get("t"):
        return None, elapsed, "n=0"
    df = pd.DataFrame({k: d[k] for k in ("t", "o", "h", "l", "c", "v") if k in d})
    df["ts"] = pd.to_datetime(df["t"], unit="s", utc=True).dt.tz_convert(VN_TZ)
    return df, elapsed, None


def quality(df: pd.DataFrame) -> dict:
    d = df.copy()
    d["day"] = d["ts"].dt.date
    d["hm"] = d["ts"].dt.strftime("%H:%M")
    per_day = d.groupby("day").size()
    # gap trong phien: so phut thieu so voi luoi 1m chuan cua phien VN
    first_bar = d.groupby("day")["hm"].min().mode().iloc[0]
    last_bar = d.groupby("day")["hm"].max().mode().iloc[0]
    return {
        "n": int(len(d)),
        "days": int(per_day.size),
        "bars_per_day": {"min": int(per_day.min()), "med": float(per_day.median()),
                          "max": int(per_day.max())},
        "session_first_bar_VN": first_bar,
        "session_last_bar_VN": last_bar,
        "ts_min": str(d["ts"].min()),
        "ts_max": str(d["ts"].max()),
        "null_close": int(d["c"].isna().sum()),
        "zero_vol_pct": float((d["v"] == 0).mean().round(4)),
        "dup_ts": int(d["t"].duplicated().sum()),
    }


def main():
    sess = requests.Session()
    report = {"api": API, "run_at": dt.datetime.now().isoformat(), "probes": {}, "window_tests": {}}
    frames = []
    for kind, sym, d0, d1 in PROBES:
        df, el, err = fetch(sess, kind, sym, d0, d1)
        key = f"{sym}_1m_{d0}"
        if df is None:
            report["probes"][key] = {"error": err, "sec": round(el, 2)}
            print(key, "->", err, flush=True)
        else:
            q = quality(df)
            q["sec"] = round(el, 2)
            report["probes"][key] = q
            print(key, "->", json.dumps(q, default=str), flush=True)
            df["_symbol"], df["_kind"], df["_probe"] = sym, kind, d0
            frames.append(df)
        time.sleep(SLEEP)

    for kind, sym, d0, d1, label in WINDOW_TESTS:
        df, el, err = fetch(sess, kind, sym, d0, d1)
        r = {"label": label, "sec": round(el, 2)}
        if df is None:
            r["error"] = err
        else:
            r["n"] = int(len(df))
            r["ts_min"], r["ts_max"] = str(df["ts"].min()), str(df["ts"].max())
            r["full_range_returned"] = bool(
                df["ts"].min().date() <= dt.date.fromisoformat(d0) + dt.timedelta(days=3)
            )
        report["window_tests"][f"{label}"] = r
        print("window", label, "->", json.dumps(r, default=str), flush=True)
        time.sleep(SLEEP)

    (OUT / "p3_entrade_report.json").write_text(json.dumps(report, indent=2, default=str),
                                                encoding="utf-8")
    if frames:
        allb = pd.concat(frames, ignore_index=True)
        allb["ts"] = allb["ts"].dt.tz_localize(None)
        allb.to_parquet(OUT / "p3_entrade_bars.parquet", index=False)
        print(f"saved {len(allb)} bars -> p3_entrade_bars.parquet", flush=True)


if __name__ == "__main__":
    main()

"""P3 probe: kha thi refetch intraday tu sieutinhieu API.

CHI fetch tho -> luu parquet/json vao p3_probe/. KHONG dung duckdb production.

Cau hoi:
1. API co tra intraday (1m/5m) voi timestamp co GIO khong?
2. Lich su sau bao nhieu? (probe cac tuan 2019/2020/2022/2024/2026)
3. Chat luong: so bar/ngay, gio giao dich, gap.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

API = "https://sieutinhieu.vn/api/v1"
OUT = Path(__file__).resolve().parent
SLEEP = 0.15

# (symbol, timeframe) probes
PAIRS = [
    ("HPG", "1m"), ("HPG", "5m"),
    ("VN30F1M", "1m"), ("VN30F1M", "5m"),
]
# cac tuan probe (thu 2 -> thu 6)
WINDOWS = [
    ("2019-03-04", "2019-03-08"),
    ("2020-03-02", "2020-03-06"),
    ("2022-03-07", "2022-03-11"),
    ("2024-03-04", "2024-03-08"),
    ("2026-06-08", "2026-06-12"),
]


def get(sess: requests.Session, path: str, params: dict, retries: int = 4):
    last = None
    for a in range(retries):
        try:
            r = sess.get(f"{API}{path}", params=params, timeout=40,
                         headers={"Accept": "application/json"})
            if r.status_code == 200:
                return r.json(), None
            last = f"HTTP {r.status_code}: {r.text[:120]}"
        except Exception as e:  # noqa: BLE001
            last = repr(e)
        time.sleep(1.5 * (a + 1))
    return None, last


def fetch_window(sess, symbol, tf, d0, d1):
    """Fetch mot cua so ngay; page qua offset."""
    items, offset = [], 0
    while True:
        d, err = get(sess, "/ohlcv/", {
            "symbol": symbol, "timeframe": tf,
            "start_date": d0, "end_date": d1,
            "limit": 1000, "offset": offset,
        })
        if d is None:
            return items, err
        batch = d.get("items", d if isinstance(d, list) else [])
        if not isinstance(batch, list):
            for k in ("value", "data", "results"):
                if isinstance(d.get(k), list):
                    batch = d[k]
                    break
        items.extend(batch)
        total = d.get("total") if isinstance(d, dict) else None
        if not batch or (total is not None and offset + len(batch) >= total) or len(batch) < 1000:
            break
        offset += len(batch)
        time.sleep(SLEEP)
    return items, None


def total_and_edges(sess, symbol, tf):
    """Tong so bar + bar moi nhat + bar cu nhat (khong keo het)."""
    d, err = get(sess, "/ohlcv/", {"symbol": symbol, "timeframe": tf, "limit": 2, "offset": 0})
    if d is None:
        return {"error": err}
    total = d.get("total")
    newest = (d.get("items") or [None])[0]
    oldest = None
    if total and total > 2:
        d2, err2 = get(sess, "/ohlcv/", {"symbol": symbol, "timeframe": tf,
                                          "limit": 2, "offset": max(0, total - 2)})
        if d2 and d2.get("items"):
            oldest = d2["items"][-1]
        else:
            oldest = {"note": f"tail fetch fail: {err2}"}
    return {"total": total, "newest": newest, "oldest": oldest}


def summarize(items: list[dict]) -> dict:
    if not items:
        return {"n": 0}
    df = pd.DataFrame(items)
    ts = pd.to_datetime(df["timestamp"])
    df["_d"] = ts.dt.date
    df["_t"] = ts.dt.time
    per_day = df.groupby("_d").size()
    has_time = int((ts.dt.hour != 0).sum() + (ts.dt.minute != 0).sum() > 0)
    return {
        "n": len(df),
        "days": int(per_day.size),
        "bars_per_day_min": int(per_day.min()),
        "bars_per_day_med": float(per_day.median()),
        "bars_per_day_max": int(per_day.max()),
        "has_time_component": bool(has_time),
        "ts_min": str(ts.min()),
        "ts_max": str(ts.max()),
        "time_first_bar": str(df.sort_values("timestamp").groupby("_d")["_t"].first().mode().iloc[0]),
        "time_last_bar": str(df.sort_values("timestamp").groupby("_d")["_t"].last().mode().iloc[0]),
        "n_null_close": int(df["close"].isna().sum()) if "close" in df else -1,
    }


def main():
    sess = requests.Session()
    report: dict = {"api": API, "run_at": pd.Timestamp.now().isoformat(), "pairs": {}}
    all_rows = []
    for symbol, tf in PAIRS:
        key = f"{symbol}_{tf}"
        print(f"=== {key}", flush=True)
        edges = total_and_edges(sess, symbol, tf)
        print(f"  edges: {json.dumps(edges, default=str)[:300]}", flush=True)
        wins = {}
        for d0, d1 in WINDOWS:
            items, err = fetch_window(sess, symbol, tf, d0, d1)
            s = summarize(items)
            if err:
                s["error"] = err
            wins[f"{d0}..{d1}"] = s
            print(f"  {d0}..{d1}: {json.dumps(s, default=str)}", flush=True)
            for it in items:
                it["_symbol"], it["_tf"] = symbol, tf
            all_rows.extend(items)
            time.sleep(SLEEP)
        report["pairs"][key] = {"edges": edges, "windows": wins}

    (OUT / "p3_probe_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")
    if all_rows:
        pd.DataFrame(all_rows).to_parquet(OUT / "p3_probe_bars.parquet", index=False)
    print(f"saved -> {OUT / 'p3_probe_report.json'}", flush=True)


if __name__ == "__main__":
    main()

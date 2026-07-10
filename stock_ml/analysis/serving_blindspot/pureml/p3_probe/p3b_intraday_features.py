# -*- coding: utf-8 -*-
"""P3 Phase B — Feature intraday THAT tu 1m VN30F1M (F2M phu) trong market_intraday.duckdb.

TIMING / KHONG LOOKAHEAD: moi feature cua ngay t chi dung bar 1m ben trong phien t,
gio chot = ATC close 14:45 (bar cuoi phien). Dung cho quyet dinh CUOI phien t
(engine quyet dinh bar t, fill close t+1) — dong nhat timing voi dx04 daily screen.

Feature (15 ung vien, tinh tu 1m F1M; slope dung them F2M):
  rv_park      : Parkinson vol tu bar 5m trong phien
  rv_5m        : std log-return 5m trong phien
  mom_30/mom_60: momentum 30'/60' cuoi phien (close ATC / close(14:15|13:45) - 1)
  mom_pm       : return buoi chieu (ATC / close 13:05 - 1)
  vol_imb_pm   : (vol chieu - vol sang) / vol ngay
  gap_open     : open 09:00 vs ATC close hom truoc (cua chinh chuoi intraday)
  range_c_id   : (high-low)/close cua phien (tu 1m)
  range_comp   : range_c_id / mean-20d range_c_id (nen bien do)
  chop5        : so lan dao chieu dau return 5m trong phien
  maxdd_15m    : toc do roi toi da — min return 15' truot trong phien (<=0)
  vwap_dev     : close ATC / VWAP phien - 1
  atc_volshare : ty trong volume 15' cuoi (14:30->ATC) / vol ngay
  slope_id     : mean trong phien cua (F2M-F1M)/F1M (F2M ffill trong ngay)
  backwd_id    : streak so ngay lien tiep slope_id < 0 (ban intraday)

Ngay thieu bar (partial day < MIN_BARS) va lo 2023-03..09 -> NaN, de fold tu loai.
Output: p3b_intraday_features.parquet (index = date, cot = feature).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import duckdb
from pathlib import Path

HERE = Path(__file__).resolve().parent
DB = r"f:\PROJECTS\train_ai_ml\market_data\market_intraday.duckdb"
MIN_BARS_F1M = 200
MIN_BARS_F2M = 30


def load_1m(sym: str) -> pd.DataFrame:
    con = duckdb.connect(DB, read_only=True)
    df = con.execute(
        "SELECT ts, open, high, low, close, volume FROM ohlcv_intraday "
        "WHERE symbol=? AND timeframe='1m' ORDER BY ts", [sym]).df()
    con.close()
    df["ts_vn"] = df["ts"] + pd.Timedelta(hours=7)
    df["day"] = df["ts_vn"].dt.normalize()
    df["hm"] = df["ts_vn"].dt.hour * 60 + df["ts_vn"].dt.minute
    return df


def day_features(g: pd.DataFrame) -> dict | None:
    """g = 1m bars 1 ngay F1M, sorted theo ts."""
    if len(g) < MIN_BARS_F1M:
        return None
    c = g["close"].to_numpy(float)
    v = g["volume"].to_numpy(float)
    hm = g["hm"].to_numpy()
    atc = c[-1]
    end_hm = hm[-1]

    # --- resample 5m (theo block 5 phut cua clock VN) ---
    blk = hm // 5
    df5 = g.groupby(blk).agg(h=("high", "max"), l=("low", "min"), c=("close", "last"))
    lr5 = np.diff(np.log(df5["c"].to_numpy(float)))
    hl = np.log(df5["h"].to_numpy(float) / df5["l"].to_numpy(float))
    rv_park = float(np.sqrt(np.mean(hl ** 2) / (4 * np.log(2))))
    rv_5m = float(np.std(lr5)) if len(lr5) > 10 else np.nan

    # --- momentum cuoi phien: asof close tai (end - k phut) ---
    def close_at(minute):
        i = np.searchsorted(hm, minute, side="right") - 1
        return c[i] if i >= 0 else np.nan
    mom_30 = atc / close_at(end_hm - 30) - 1
    mom_60 = atc / close_at(end_hm - 60) - 1
    pm0 = close_at(13 * 60 + 5)
    mom_pm = atc / pm0 - 1 if np.isfinite(pm0) else np.nan

    # --- volume sang / chieu ---
    am = hm < 11 * 60 + 45
    pm = hm >= 13 * 60
    vtot = v.sum()
    vol_imb_pm = float((v[pm].sum() - v[am].sum()) / vtot) if vtot > 0 else np.nan

    hi, lo_ = g["high"].max(), g["low"].min()
    range_c_id = float((hi - lo_) / atc)

    # --- chop: dao chieu dau return 5m ---
    s = np.sign(lr5)
    s = s[s != 0]
    chop5 = float((s[1:] != s[:-1]).sum()) if len(s) > 1 else np.nan

    # --- max 15' drawdown (min return 15 phut truot) ---
    if len(c) > 16:
        r15 = c[15:] / c[:-15] - 1.0
        maxdd_15m = float(r15.min())
    else:
        maxdd_15m = np.nan

    vwap = float((c * v).sum() / vtot) if vtot > 0 else np.nan
    vwap_dev = atc / vwap - 1 if vwap and np.isfinite(vwap) else np.nan
    last15 = hm >= 14 * 60 + 30
    atc_volshare = float(v[last15].sum() / vtot) if vtot > 0 else np.nan

    return {"open_0900": float(g["open"].iloc[0]), "atc_close": float(atc),
            "rv_park": rv_park, "rv_5m": rv_5m, "mom_30": float(mom_30),
            "mom_60": float(mom_60), "mom_pm": float(mom_pm),
            "vol_imb_pm": vol_imb_pm, "range_c_id": range_c_id,
            "chop5": chop5, "maxdd_15m": maxdd_15m, "vwap_dev": float(vwap_dev),
            "atc_volshare": atc_volshare, "nbars": len(g)}


def main():
    f1 = load_1m("VN30F1M")
    f2 = load_1m("VN30F2M")
    print(f"F1M {len(f1)} bars, F2M {len(f2)} bars", flush=True)

    rows = {}
    for day, g in f1.groupby("day"):
        r = day_features(g.sort_values("ts"))
        if r is not None:
            rows[day] = r
    X = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    print(f"F1M feature days: {len(X)} (loai {f1['day'].nunique() - len(X)} partial day "
          f"< {MIN_BARS_F1M} bar)", flush=True)

    # gap mo cua vs ATC hom truoc (chuoi intraday; NaN qua lo hong >5 ngay lich)
    prev_close = X["atc_close"].shift(1)
    gap_days = X.index.to_series().diff().dt.days
    X["gap_open"] = np.where(gap_days <= 5, X["open_0900"] / prev_close - 1, np.nan)

    # nen bien do 20 ngay
    X["range_comp"] = X["range_c_id"] / X["range_c_id"].rolling(20, min_periods=15).mean()

    # slope intraday-mean F1M-F2M (ffill F2M trong ngay tren luoi phut F1M)
    f2d = {d: g.set_index("hm")["close"] for d, g in f2.groupby("day")
           if len(g) >= MIN_BARS_F2M}
    slope_id = {}
    for day, g in f1.groupby("day"):
        s2 = f2d.get(day)
        if s2 is None or day not in X.index:
            continue
        g = g.sort_values("hm")
        c2 = s2.reindex(g["hm"]).ffill().to_numpy(float)
        c1 = g["close"].to_numpy(float)
        m = np.isfinite(c2)
        if m.sum() < 60:
            continue
        slope_id[day] = float(((c2[m] - c1[m]) / c1[m]).mean())
    X["slope_id"] = pd.Series(slope_id)

    bw = X["slope_id"] < 0
    grp = (bw != bw.shift()).cumsum()
    X["backwd_id"] = bw.groupby(grp).cumsum().where(bw, 0).astype(float)
    X.loc[X["slope_id"].isna(), "backwd_id"] = np.nan

    feats = ["rv_park", "rv_5m", "mom_30", "mom_60", "mom_pm", "vol_imb_pm",
             "gap_open", "range_c_id", "range_comp", "chop5", "maxdd_15m",
             "vwap_dev", "atc_volshare", "slope_id", "backwd_id"]
    out = X[feats + ["atc_close", "nbars"]]
    out.to_parquet(HERE / "p3b_intraday_features.parquet")
    print("\ncoverage per feature (non-NaN):")
    print(out[feats].notna().sum().to_string())
    print("\nby year (rv_park non-NaN days):")
    print(out["rv_park"].notna().groupby(out.index.year).sum().to_string())
    print("\nsaved -> p3b_intraday_features.parquet", flush=True)


if __name__ == "__main__":
    main()

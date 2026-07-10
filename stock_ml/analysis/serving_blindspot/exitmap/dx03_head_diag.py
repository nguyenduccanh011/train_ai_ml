# -*- coding: utf-8 -*-
"""Buoc 0 — Vi sao velocity exit head im lang tu 2024?

Ba manh chung gian tiep (khong co model scores luu lai):
  A. INFO: cross-sectional IC cua cac feature price/vol loi (dung ho feature
     champion) vs CHINH label velocity_exit_regression (h20/u8/volnorm40),
     theo tung nam 2020..2026H1. Neu IC 2024+ sup do -> feature het thong tin
     o regime moi (b). Neu IC con -> nghieng ve score-drift/threshold (a).
  B. OPPORTUNITY: phan phoi label vexit theo nam (std cross-sectional, q90,
     ty le quan sat vuot q90 pooled 2020-23). Label co con "co hoi exit" khong?
  C. PRE-EMPTION: tu gbx08_enriched2.csv — % exit co force flag tai decision
     bar theo nam; neu 2024+ ~100% flag thi head bi force-gate chan cho.
"""
import duckdb
import numpy as np
import pandas as pd

DUCK = r"f:\PROJECTS\train_ai_ml\market_data\market.duckdb"
EM = r"f:\PROJECTS\train_ai_ml\stock_ml\analysis\serving_blindspot\exitmap"
MIN_SYM = 40
YEARS = [("2020", "2020-01-01", "2021-01-01"), ("2021", "2021-01-01", "2022-01-01"),
         ("2022", "2022-01-01", "2023-01-01"), ("2023", "2023-01-01", "2024-01-01"),
         ("2024", "2024-01-01", "2025-01-01"), ("2025", "2025-01-01", "2026-01-01"),
         ("2026H1", "2026-01-01", "2026-07-01")]


def _rsi(c, n):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def sym_features(g):
    c, h, l, v = g["close"], g["high"], g["low"], g["volume"].astype(float)
    f = pd.DataFrame(index=g.index)
    ret1 = c.pct_change()
    f["ret_5d"] = c.pct_change(5)
    f["ret_20d"] = c.pct_change(20)
    f["sma20_ratio"] = c / c.rolling(20).mean() - 1
    f["rsi_14"] = _rsi(c, 14)
    f["rv_10"] = ret1.rolling(10).std()
    upv = v.where(ret1 > 0, 0.0).rolling(20).sum()
    dnv = v.where(ret1 <= 0, 0.0).rolling(20).sum()
    f["updown_vol_20"] = upv / dnv.replace(0, np.nan)
    r20mx, r20mn = c.rolling(20).max(), c.rolling(20).min()
    f["range_pos_20"] = (c - r20mn) / (r20mx - r20mn).replace(0, np.nan)
    f["dist_10d_low"] = c / c.rolling(10).min() - 1
    # velocity exit label (exact port screen_ic.py / velocity_exit.py)
    dmin = pd.concat([c.shift(-k) for k in range(1, 21)], axis=1).min(axis=1)
    umax = pd.concat([c.shift(-k) for k in range(1, 9)], axis=1).max(axis=1)
    obs = c.shift(-20).notna()
    lab = (1.0 - dmin.where(obs) / c) - (umax.where(obs) / c - 1.0)
    vol40 = ret1.rolling(40, min_periods=2).std()
    f["t_vexit"] = lab / (vol40 + 1e-4)
    return f


FEATS = ["ret_5d", "ret_20d", "sma20_ratio", "rsi_14", "rv_10",
         "updown_vol_20", "range_pos_20", "dist_10d_low"]


def main():
    duck = duckdb.connect(DUCK, read_only=True)
    df = duck.execute(
        "SELECT symbol, date, open, high, low, close, volume FROM ohlcv "
        "WHERE timeframe='1D' AND symbol NOT IN ('VN30F1M','VN30F2M') "
        "ORDER BY symbol, date").df()
    duck.close()
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]
    print(f"panel: {df.symbol.nunique()} syms, {len(df)} bars")

    parts = []
    for s, g in df.groupby("symbol"):
        g = g.reset_index(drop=True)
        f = sym_features(g)
        f["symbol"], f["date"] = s, g["date"]
        parts.append(f)
    d = pd.concat(parts, ignore_index=True)
    d = d[d["date"] >= "2019-10-01"]

    # ---------- A. per-year IC of features vs vexit label ----------
    ic_rows = {}
    lab_rows = []
    for dt, gg in d.groupby("date", sort=True):
        y = gg["t_vexit"].to_numpy(float)
        my = ~np.isnan(y)
        if my.sum() < MIN_SYM:
            continue
        yr = pd.Series(y[my]).rank().to_numpy()
        for c in FEATS:
            x = gg[c].to_numpy(float)[my]
            m = ~np.isnan(x)
            if m.sum() < MIN_SYM:
                continue
            xr = pd.Series(x[m]).rank().to_numpy()
            yv = yr[m]
            sx, sy = xr.std(), yv.std()
            if sx <= 0 or sy <= 0:
                continue
            ic_rows.setdefault(c, []).append(
                (dt, float(np.mean((xr - xr.mean()) * (yv - yv.mean())) / (sx * sy))))
        lab_rows.append((dt, float(np.nanstd(y)), float(np.nanquantile(y[my], 0.9)),
                         float(np.nanmean(y[my]))))

    print("\n=== A. IC (cross-sectional Spearman) feature -> vexit label, theo nam ===")
    hdr = "feature".ljust(15) + "".join(n.rjust(9) for n, _, _ in YEARS)
    print(hdr)
    for c in FEATS:
        s = pd.Series({dt: v for dt, v in ic_rows[c]}).sort_index()
        row = c.ljust(15)
        for _, ts, te in YEARS:
            seg = s[(s.index >= ts) & (s.index < te)]
            row += (f"{seg.mean():+.4f}" if len(seg) > 30 else "   --  ").rjust(9)
        print(row)

    print("\n=== B. Phan phoi label vexit theo nam ===")
    ls = pd.DataFrame(lab_rows, columns=["date", "xstd", "q90", "mean"]).set_index("date")
    thr = ls[(ls.index >= "2020-01-01") & (ls.index < "2024-01-01")]["q90"].mean()
    print(f"pooled 2020-23 mean-of-daily-q90 = {thr:.3f}")
    print("year   xsec_std   q90_daily   mean    (label = downside - upside, vol-norm)")
    for n, ts, te in YEARS:
        seg = ls[(ls.index >= ts) & (ls.index < te)]
        if len(seg):
            print(f"{n:6s} {seg.xstd.mean():8.3f} {seg.q90.mean():10.3f} {seg['mean'].mean():8.3f}")

    # ---------- C. pre-emption tu enriched2 ----------
    tr = pd.read_csv(EM + r"\gbx08_enriched2.csv", parse_dates=["entry_date", "exit_date"])
    tr = tr[tr.label.notna() & (tr.label != "open")]
    tr["anyflag"] = tr[["f_dl12", "f_nb", "f_lb"]].any(axis=1)
    print("\n=== C. Force-flag coverage tai decision bar theo nam exit ===")
    print("year   n_exit  %anyflag  n_head  hold_med")
    for y, g in tr.groupby("year_exit"):
        print(f"{y}   {len(g):5d}   {g.anyflag.mean():6.1%}  "
              f"{(g.label == 'head_signal').sum():5d}   {g.holding_days.median():6.1f}")


if __name__ == "__main__":
    main()

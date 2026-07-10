# -*- coding: utf-8 -*-
"""Buoc 1+2 — Sang loc tin hieu phai sinh VN30F1M/F2M cho EXIT 2024+.

DATA TRUTH (audit dx01/dx02): intraday 1m/5m/15m/30m/1H trong market.duckdb
(va ca .bak) da bi COLLAPSE — PK (symbol,timeframe,date DATE) giu dung 1 bar
tuy tien/ngay (close bar '1m' lech xa close 1D). market_raw_api chi co daily
stocks. => Moi feature intraday that (RV trong phien, momentum 30-60' cuoi,
AM/PM imbalance) BAT KHA THI. Kenh kha dung: DAILY VN30F1M/F2M OHLCV+volume
(2017-08-10 .. 2026-06-16) — van la thong tin he chua tung thay (gia futures,
term structure, gap, volume phai sinh).

Khong co VN30 spot index trong bat ky nguon nao => basis LEVEL bat kha thi;
thay bang basis-CHANGE proxy: ret(F1M) - ret(EW universe proxy) (1/5/10/20d)
— chinh la thanh phan thay doi basis + noise thanh phan ro (VN30 vs EW-61).
Backwardation streak dung calendar-spread F2M<F1M thay cho F1M<spot.

Timing: bar daily t cua futures chot 14:45 (ATC) cung luc ATC co phieu;
engine quyet dinh tai bar t va fill close(t+1) => feature dung den close t,
KHONG lookahead.

IC = time-series Spearman theo fold nam (2022..2026H1) — day la tin hieu
MARKET-LEVEL (1 chuoi), khac harness cross-sectional screen_ic.py.
Orthogonal: residualize rank(x) tren rank(controls he da thay):
proxy_ret5/20, dist_ma20/50, breadth_ma50 full-univ, SNR20 universe,
drop5_z (market-drop gate), proxy_rv10.
Null: circular-shift 300 lan (giu autocorrelation ca 2 chuoi), band 2.5/97.5.

Targets dung cho dau exit:
  T1 open_dd10 : mean tren cac lenh DANG MO tai t cua forward-10-bar drawdown
                 cua chinh symbol lenh (1 - min(close t+1..t+10)/close_t).
  T1b open_gb5pct: ty le lenh mo bi rot >=5% trong 10 bar toi.
  T2 univ_dd10 : forward-10-bar drawdown cua EW proxy level.
  T3 rallied   : event-level tai decision bar cua 1378 exit — feature co tach
                 cohort sold-then-rallied (663) khong. Null: permute trong fold.
"""
import json
import numpy as np
import pandas as pd
import duckdb

DUCK = r"f:\PROJECTS\train_ai_ml\market_data\market.duckdb"
EM = r"f:\PROJECTS\train_ai_ml\stock_ml\analysis\serving_blindspot\exitmap"
FOLDS = [("2022", "2022-01-01", "2023-01-01"), ("2023", "2023-01-01", "2024-01-01"),
         ("2024", "2024-01-01", "2025-01-01"), ("2025", "2025-01-01", "2026-01-01"),
         ("2026H1", "2026-01-01", "2026-07-01")]
N_NULL = 300
RNG = np.random.default_rng(42)


# ------------------------------------------------------------------ data
def load():
    duck = duckdb.connect(DUCK, read_only=True)
    fut = duck.execute(
        "SELECT symbol, date, open, high, low, close, volume FROM ohlcv "
        "WHERE timeframe='1D' AND symbol IN ('VN30F1M','VN30F2M') ORDER BY date").df()
    st = duck.execute(
        "SELECT symbol, date, close FROM ohlcv WHERE timeframe='1D' "
        "AND symbol NOT IN ('VN30F1M','VN30F2M') ORDER BY symbol, date").df()
    duck.close()
    fut["date"] = pd.to_datetime(fut["date"])
    st["date"] = pd.to_datetime(st["date"])
    tr = pd.read_csv(EM + r"\gbx08_enriched2.csv",
                     parse_dates=["entry_date", "exit_date"])
    # universe 8 v2 (61 ma) — thu postgres nhu em02, fallback = symbols trong trades
    try:
        import psycopg2
        pg = psycopg2.connect(host="localhost", port=5433, dbname="stockml",
                              user="stockml", password="stockml_dev")
        cur = pg.cursor()
        cur.execute("SELECT symbols_json FROM universe_versions WHERE universe_id=8 AND version=2")
        uni = [d["symbol"] for d in json.loads(cur.fetchone()[0])]
        pg.close()
        print(f"universe tu postgres: {len(uni)}")
    except Exception as e:
        uni = sorted(tr["symbol"].unique())
        print(f"postgres fail ({e}); fallback universe = {len(uni)} symbols tu trades")
    return fut, st, tr, uni


def winz(s, q=0.005):
    lo, hi = s.quantile(q), s.quantile(1 - q)
    return s.clip(lo, hi)


# ------------------------------------------------------------------ features
def build_features(fut):
    f1 = fut[fut.symbol == "VN30F1M"].set_index("date").sort_index()
    f2 = fut[fut.symbol == "VN30F2M"].set_index("date").sort_index()
    ix = f1.index.intersection(f2.index)
    f1, f2 = f1.loc[ix], f2.loc[ix]
    c, o, h, l, v = f1["close"], f1["open"], f1["high"], f1["low"], f1["volume"]
    ret1 = c.pct_change()
    X = pd.DataFrame(index=ix)
    slope = (f2["close"] - c) / c
    X["slope"] = slope                              # term structure F2M-F1M
    X["slope_chg5"] = slope.diff(5)
    X["gap1"] = c.shift(1).rdiv(o) - 1              # gap mo cua F1M
    X["gap_abs5"] = X["gap1"].abs().rolling(5).mean()
    rng_ = (h - l).where(h > l)
    X["range_c"] = rng_ / c                         # bien do ngay
    X["range_r520"] = rng_.rolling(5).mean() / rng_.rolling(20).mean()
    X["rv10"] = ret1.rolling(10).std()              # realized vol daily
    clv = ((c - l) - (h - c)) / rng_
    X["clv5"] = clv.rolling(5).mean()               # vi tri close trong range
    X["co5"] = (c / o - 1).rolling(5).mean()        # than nen 5d
    X["volz20"] = (v - v.rolling(20).mean()) / v.rolling(20).std()
    bw = (slope < 0)
    grp = (bw != bw.shift()).cumsum()
    X["backwd_streak"] = bw.groupby(grp).cumsum().where(bw, 0).astype(float)
    X["f1_ret5"] = c.pct_change(5)
    X["f1_ret10"] = c.pct_change(10)
    return X, ret1


def build_market(st, uni):
    piv = st[st.symbol.isin(uni)].pivot_table(
        index="date", columns="symbol", values="close", aggfunc="last").sort_index()
    rets = piv.pct_change()
    mret = rets.replace([np.inf, -np.inf], np.nan).clip(-0.5, 0.5).mean(axis=1)
    lvl = (1.0 + mret.fillna(0.0)).cumprod()
    snr20 = rets.mean(axis=1).rolling(20).sum() / (rets.rolling(20).sum().std(axis=1) + 1e-9)
    pall = st.pivot_table(index="date", columns="symbol", values="close",
                          aggfunc="last").sort_index()
    ma50a = pall.rolling(50, min_periods=50).mean()
    ind = (pall > ma50a)
    breadth = ind.sum(axis=1) / ind.notna().sum(axis=1).clip(lower=1)
    drop5 = mret.rolling(5).sum()
    mu = drop5.rolling(252, min_periods=60).mean()
    sd = drop5.rolling(252, min_periods=60).std()
    C = pd.DataFrame({
        "proxy_ret5": lvl.pct_change(5), "proxy_ret20": lvl.pct_change(20),
        "dist_ma20": lvl / lvl.rolling(20).mean() - 1,
        "dist_ma50": lvl / lvl.rolling(50).mean() - 1,
        "breadth": breadth.reindex(lvl.index),
        "snr20": snr20, "drop5_z": (drop5 - mu) / sd.replace(0, np.nan),
        "proxy_rv10": mret.rolling(10).std()})
    return piv, mret, lvl, C


def build_targets(piv, lvl, tr):
    dates = piv.index
    fmin10 = piv.shift(-10).rolling(10, min_periods=10).min().shift(9)
    # fmin10[t] = min(close[t+1..t+10]); implement explicitly for clarity
    fmin10 = pd.concat([piv.shift(-k) for k in range(1, 11)], axis=1,
                       keys=range(1, 11)).groupby(level=1, axis=1).min()
    dd = 1.0 - fmin10 / piv          # per-symbol forward 10-bar drawdown (>=0 pain)
    # open-trade panel
    T1, T1b, ns = pd.Series(index=dates, dtype=float), pd.Series(index=dates, dtype=float), []
    trv = tr[["symbol", "entry_date", "exit_date"]].dropna()
    # build open mask per date via interval logic (vectorized per trade)
    open_dd = {dt: [] for dt in dates}
    pos = {s: i for i, s in enumerate(piv.columns)}
    ddv = dd.to_numpy()
    dix = {d: i for i, d in enumerate(dates)}
    for _, r in trv.iterrows():
        if r.symbol not in pos:
            continue
        i0 = dix.get(r.entry_date)
        if i0 is None:
            i0 = int(np.searchsorted(dates, r.entry_date))
        i1 = int(np.searchsorted(dates, r.exit_date))
        for i in range(i0, min(i1, len(dates))):
            val = ddv[i, pos[r.symbol]]
            if not np.isnan(val):
                open_dd[dates[i]].append(val)
    for dt in dates:
        a = open_dd[dt]
        if len(a) >= 3:
            a = np.array(a)
            T1[dt] = a.mean()
            T1b[dt] = (a >= 0.05).mean()
    lmin = pd.concat([lvl.shift(-k) for k in range(1, 11)], axis=1).min(axis=1)
    T2 = 1.0 - lmin / lvl
    return pd.DataFrame({"T1_open_dd10": T1, "T1b_open_gb5": T1b, "T2_univ_dd10": T2})


# ------------------------------------------------------------------ IC machinery
def rank_u(s):
    return s.rank() / s.notna().sum()


def resid(x, C):
    """rank-OLS residual of x on controls, fit tren toan mau co du lieu."""
    df = pd.concat([rank_u(x).rename("x")] + [rank_u(C[c]).rename(c) for c in C], axis=1).dropna()
    B = np.column_stack([np.ones(len(df))] + [df[c].to_numpy() for c in C.columns])
    beta, *_ = np.linalg.lstsq(B, df["x"].to_numpy(), rcond=None)
    r = pd.Series(df["x"].to_numpy() - B @ beta, index=df.index)
    return r.reindex(x.index)


def sp(a, b):
    m = a.notna() & b.notna()
    if m.sum() < 30:
        return np.nan, int(m.sum())
    return float(a[m].rank().corr(b[m].rank())), int(m.sum())


def fold_ics(x, y):
    out = {}
    for n, ts, te in FOLDS:
        seg = (x.index >= ts) & (x.index < te)
        out[n] = sp(x[seg], y[seg])[0]
    pooled = (x.index >= FOLDS[0][1]) & (x.index < FOLDS[-1][2])
    out["pooled"], out["n"] = sp(x[pooled], y[pooled])
    p24 = (x.index >= "2024-01-01") & (x.index < FOLDS[-1][2])
    out["pool24"], out["n24"] = sp(x[p24], y[p24])
    return out


def null_band(x, y, lo="2022-01-01"):
    seg = (x.index >= lo) & (x.index < FOLDS[-1][2])
    xs, ys = x[seg], y[seg]
    m = xs.notna() & ys.notna()
    xv, yv = xs[m].rank().to_numpy(float), ys[m].rank().to_numpy(float)
    n = len(xv)
    if n < 60:
        return None
    stats = []
    for _ in range(N_NULL):
        k = int(RNG.integers(30, n - 30))
        xp = np.roll(xv, k)
        stats.append(float(np.corrcoef(xp, yv)[0, 1]))
    return (round(float(np.percentile(stats, 2.5)), 4),
            round(float(np.percentile(stats, 97.5)), 4))


def main():
    fut, st, tr, uni = load()
    X, _ = build_features(fut)
    for c in X.columns:
        X[c] = winz(X[c])
    piv, mret, lvl, C = build_market(st, uni)
    cal = lvl.index
    X = X.reindex(cal)          # align futures calendar -> stock calendar
    C = C.reindex(cal)
    Y = build_targets(piv[sorted(set(tr.symbol) & set(piv.columns))], lvl, tr)
    print(f"features {X.shape}, targets {Y.shape}, controls {C.shape}")
    print("target coverage >=2022:",
          {c: int(Y.loc[Y.index >= '2022-01-01', c].notna().sum()) for c in Y})

    rows = []
    for feat in X.columns:
        x = X[feat]
        xr = resid(x, C)
        for tgt in Y.columns:
            y = Y[tgt]
            raw = fold_ics(x, y)
            ort = fold_ics(xr, y)
            nb_raw = null_band(x, y)
            nb_ort = null_band(xr, y)
            nb24 = null_band(xr, y, lo="2024-01-01")
            rows.append({"feature": feat, "target": tgt,
                         **{f"raw_{k}": v for k, v in raw.items()},
                         **{f"ort_{k}": v for k, v in ort.items()},
                         "raw_null": nb_raw, "ort_null": nb_ort, "ort_null24": nb24})
            print(f"{feat:15s} {tgt:14s} raw {raw['pooled']:+.3f} "
                  f"ort {ort['pooled']:+.3f} ort24 {ort['pool24']:+.3f} "
                  f"null {nb_ort} null24 {nb24}", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(EM + r"\dx04_deriv_ic.csv", index=False)

    # ---------------- T3: rallied discrimination tai decision bar ----------------
    ex = tr[tr.exit_date.notna() & tr.rallied.notna()].copy()
    fdates = X.dropna(how="all").index
    dec = []
    for d in ex.exit_date:
        i = int(np.searchsorted(fdates, d)) - 1
        dec.append(fdates[i] if i >= 0 else pd.NaT)
    ex["dec_date"] = dec
    ex["yr"] = ex.exit_date.dt.year
    ex = ex[ex.yr >= 2022]
    Xr = pd.DataFrame({f: resid(X[f], C) for f in X.columns})
    print("\n=== T3 rallied (exit >=2022, n per year:",
          ex.groupby("yr").size().to_dict(), ") ===")
    t3 = []
    for feat in X.columns:
        vals_r = Xr[feat].reindex(ex.dec_date).to_numpy()
        lab = ex.rallied.astype(float).to_numpy()
        yrs = ex.yr.to_numpy()
        percol, ok = {}, []
        for y in sorted(set(yrs)):
            m = (yrs == y) & ~np.isnan(vals_r)
            if m.sum() < 40:
                percol[str(y)] = None
                continue
            ic = float(pd.Series(vals_r[m]).rank().corr(pd.Series(lab[m]).rank()))
            percol[str(y)] = round(ic, 4)
            ok.append(ic)
        m = ~np.isnan(vals_r)
        pooled = float(pd.Series(vals_r[m]).rank().corr(pd.Series(lab[m]).rank()))
        # permutation null (shuffle rallied within year), 300 draws
        stats = []
        for _ in range(N_NULL):
            lp = lab.copy()
            for y in set(yrs):
                my = (yrs == y)
                lp[my] = RNG.permutation(lp[my])
            stats.append(float(pd.Series(vals_r[m]).rank().corr(pd.Series(lp[m]).rank())))
        nlo, nhi = np.percentile(stats, [2.5, 97.5])
        t3.append({"feature": feat, "pooled": round(pooled, 4), **percol,
                   "null": (round(float(nlo), 4), round(float(nhi), 4))})
        print(f"{feat:15s} pooled {pooled:+.4f} null ({nlo:+.4f},{nhi:+.4f}) {percol}")
    pd.DataFrame(t3).to_csv(EM + r"\dx04_t3_rallied.csv", index=False)
    print("\nsaved dx04_deriv_ic.csv / dx04_t3_rallied.csv")


if __name__ == "__main__":
    main()

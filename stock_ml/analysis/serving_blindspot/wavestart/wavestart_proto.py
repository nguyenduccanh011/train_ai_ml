# -*- coding: utf-8 -*-
"""Wave-start specialist head prototype.

Q: is there a learnable bar-level signal for high-quality wave starts (bottom
turns below MA20) that the production pullback-limit champion structurally misses?

Labels (forward-looking OK), features strictly causal (data <= t only).
Walk-forward LightGBM, folds test 2022 / 2023 / 2024 / 2025-2026H1,
train = all data ending 85 calendar days before test start.

Caveat: raw (non-back-adjusted) prices -> dividend gaps add label/feature noise.
missed_moves.csv was not found in repo; wave episodes (>=30% runup) are
reconstructed from OHLCV via trough->peak scan, captured flag derived from
champion trades file trades_n2_2643_wavestruct_la05_lamp02.csv.
"""
import json
import sqlite3

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr

RNG = 42
DB = r"C:\Users\DUC CANH PC\Desktop\stock-serving\data\ohlcv.db"
UNIV = r"f:\PROJECTS\train_ai_ml\_tmp_analysis\univ_2429.txt"
TRADES = (r"f:\PROJECTS\train_ai_ml\stock_ml\analysis\serving_blindspot\week0"
          r"\best_trades\trades_n2_2643_wavestruct_la05_lamp02.csv")
OUT = r"f:\PROJECTS\train_ai_ml\stock_ml\analysis\serving_blindspot\wavestart"

FOLDS = [  # (name, test_start, test_end_exclusive)
    ("2022", "2022-01-01", "2023-01-01"),
    ("2023", "2023-01-01", "2024-01-01"),
    ("2024", "2024-01-01", "2025-01-01"),
    ("2025H1+", "2025-01-01", "2026-07-01"),
]
GAP_DAYS = 85
FWD = 42          # forward horizon bars
RECLAIM = 15      # bars to reclaim MA20
GAIN = 0.10       # forward max close gain
BREAK = 0.03      # support break tolerance
TOPPCT = 0.05


# ---------------------------------------------------------------- data
def load_panel():
    univ = [l.strip() for l in open(UNIV) if l.strip()]
    con = sqlite3.connect(DB)
    q = ("select symbol, date, open, high, low, close, volume from ohlcv "
         f"where symbol in ({','.join('?' * len(univ))}) order by symbol, date")
    df = pd.read_sql(q, con, params=univ)
    con.close()
    df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)].reset_index(drop=True)
    return univ, df


# ---------------------------------------------------------------- features (STRICTLY CAUSAL)
def sym_features(g):
    """All rolling windows end at t inclusive -> uses only data <= t."""
    c, h, l, o, v = g["close"], g["high"], g["low"], g["open"], g["volume"]
    f = pd.DataFrame(index=g.index)
    for n in (1, 5, 10, 20, 60):
        f[f"ret{n}"] = c.pct_change(n)
    ma10, ma20, ma50 = c.rolling(10).mean(), c.rolling(20).mean(), c.rolling(50).mean()
    f["dist_ma10"] = c / ma10 - 1
    f["dist_ma20"] = c / ma20 - 1
    f["dist_ma50"] = c / ma50 - 1
    f["ma20_slope5"] = ma20 / ma20.shift(5) - 1
    for n in (20, 60, 120):
        f[f"dd{n}"] = c / c.rolling(n).max() - 1
    f["rebound60"] = c / c.rolling(60).min() - 1
    f["days_since_lo60"] = c.rolling(60).apply(lambda x: len(x) - 1 - np.argmin(x), raw=True)
    f["days_since_hi60"] = c.rolling(60).apply(lambda x: len(x) - 1 - np.argmax(x), raw=True)
    # RSI14 (Wilder via ewm -> causal)
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, min_periods=14).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, min_periods=14).mean()
    rsi = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    f["rsi14"] = rsi
    f["rsi14_slope3"] = rsi.diff(3)
    # volume
    vma20 = v.rolling(20).mean()
    f["vol_r5_20"] = v.rolling(5).mean() / vma20
    f["vol_surge"] = v / vma20.shift(1)
    updays = (d > 0)
    f["updown_vol10"] = (v.where(updays, 0).rolling(10).sum()
                         / v.where(~updays, 0).rolling(10).sum().replace(0, np.nan))
    # range / volatility
    hi20, lo20 = h.rolling(20).max(), l.rolling(20).min()
    f["range_pos20"] = (c - lo20) / (hi20 - lo20).replace(0, np.nan)
    f["day_range_pos"] = ((c - l) / (h - l).replace(0, np.nan))
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    f["atr14_pct"] = atr14 / c
    f["atr_r5_20"] = tr.rolling(5).mean() / tr.rolling(20).mean()
    f["retstd20"] = c.pct_change().rolling(20).std()
    f["gap_open"] = o / pc - 1
    # structure
    dnday = (d < 0).astype(int)
    f["consec_down"] = dnday.groupby((dnday == 0).cumsum()).cumsum()
    below = (c < ma20).astype(int)
    f["bars_below_ma20"] = below.groupby((below == 0).cumsum()).cumsum()
    f["thrust"] = (c > h.shift(1)).astype(int)
    f["thrust_str"] = c / h.shift(1) - 1
    return f


def build(df):
    parts = []
    for s, g in df.groupby("symbol"):
        g = g.reset_index(drop=True)
        f = sym_features(g)
        f["symbol"], f["date"], f["close"] = s, g["date"], g["close"]
        # forward outcomes (labels only)
        c = g["close"].to_numpy(float)
        lo = g["low"].to_numpy(float)
        ma20 = g["close"].rolling(20).mean().to_numpy(float)
        sup = g["low"].rolling(10).min().to_numpy(float)  # incl t
        n = len(c)
        fmax = np.full(n, np.nan); fmin = np.full(n, np.nan)
        rec = np.zeros(n, bool); pos = np.zeros(n, bool); lab_ok = np.zeros(n, bool)
        for t in range(n):
            if t + FWD >= n:
                continue
            w = c[t + 1:t + FWD + 1]
            fmax[t] = w.max() / c[t] - 1
            fmin[t] = w.min() / c[t] - 1
            rec[t] = bool((c[t + 1:t + RECLAIM + 1] > ma20[t + 1:t + RECLAIM + 1]).any())
            gi = np.nonzero(w >= c[t] * (1 + GAIN))[0]
            bi = np.nonzero(lo[t + 1:t + FWD + 1] < sup[t] * (1 - BREAK))[0]
            gain_first = len(gi) > 0 and (len(bi) == 0 or gi[0] < bi[0])
            pos[t] = gain_first and rec[t]
            lab_ok[t] = True
        f["fwd_max42"], f["fwd_min42"] = fmax, fmin
        f["pos_l1"] = pos          # reclaim & +10% before support-3% break
        f["pos_gain"] = np.array([  # gain-first only (no reclaim req) for L2/precision
            (not np.isnan(fmax[t])) and fmax[t] >= GAIN and pos[t] or False for t in range(n)])
        # recompute gain-first w/o reclaim cleanly
        pg = np.zeros(n, bool)
        for t in range(n):
            if t + FWD >= n:
                continue
            w = c[t + 1:t + FWD + 1]
            gi = np.nonzero(w >= c[t] * (1 + GAIN))[0]
            bi = np.nonzero(lo[t + 1:t + FWD + 1] < sup[t] * (1 - BREAK))[0]
            pg[t] = len(gi) > 0 and (len(bi) == 0 or gi[0] < bi[0])
        f["pos_gain"] = pg
        f["lab_ok"] = lab_ok
        # eligibility
        f["elig_l1"] = (f["dist_ma20"] < 0) & (f["dd60"] <= -0.12)
        prev_below = f["bars_below_ma20"].shift(1).fillna(0)
        f["elig_l2"] = ((prev_below >= 10) & (f["thrust"] == 1)
                        & (f["vol_surge"] >= 1.3))
        f["elig_l3"] = (f["dist_ma20"] < 0) & (f["dd60"] <= -0.08)
        f["y_l3"] = f["fwd_max42"] + 1.5 * f["fwd_min42"]  # penalized runup
        parts.append(f)
    d = pd.concat(parts, ignore_index=True)
    # market context (cross-sectional at date t from causal per-symbol cols)
    mkt = d.groupby("date").agg(mkt_ret5=("ret5", "mean"), mkt_ret20=("ret20", "mean"),
                                mkt_above_ma50=("dist_ma50", lambda x: (x > 0).mean()),
                                mkt_breadth1=("ret1", lambda x: (x > 0).mean()))
    d = d.merge(mkt, on="date", how="left")
    d["rel_ret20"] = d["ret20"] - d["mkt_ret20"]
    return d


FEATS = ["ret1", "ret5", "ret10", "ret20", "ret60", "dist_ma10", "dist_ma20",
         "dist_ma50", "ma20_slope5", "dd20", "dd60", "dd120", "rebound60",
         "days_since_lo60", "days_since_hi60", "rsi14", "rsi14_slope3",
         "vol_r5_20", "vol_surge", "updown_vol10", "range_pos20", "day_range_pos",
         "atr14_pct", "atr_r5_20", "retstd20", "gap_open", "consec_down",
         "bars_below_ma20", "thrust", "thrust_str", "mkt_ret5", "mkt_ret20",
         "mkt_above_ma50", "mkt_breadth1", "rel_ret20"]

LABELS = {  # name -> (elig col, target col, is_binary)
    "L1_bottom_turn": ("elig_l1", "pos_l1", True),
    "L2_thrust": ("elig_l2", "pos_gain", True),
    "L3_penalized_runup": ("elig_l3", "y_l3", False),
}


def make_model(binary):
    p = dict(n_estimators=400, learning_rate=0.05, num_leaves=15,
             min_child_samples=60, subsample=0.8, subsample_freq=1,
             colsample_bytree=0.8, random_state=RNG, n_jobs=4, verbose=-1)
    return lgb.LGBMClassifier(**p) if binary else lgb.LGBMRegressor(**p)


def run_folds(d, shuffle_null=False):
    res, oos = {}, []
    for lab, (ec, yc, binary) in LABELS.items():
        rows = d[d[ec] & d["lab_ok"]].dropna(subset=FEATS + [yc]).copy()
        res[lab] = {"n_rows": len(rows),
                    "base_rate_all": float(rows[yc].mean()) if binary
                    else float((rows["pos_gain"]).mean())}
        for fname, ts, te in FOLDS:
            tr_end = (pd.Timestamp(ts) - pd.Timedelta(days=GAP_DAYS)).strftime("%Y-%m-%d")
            tr = rows[rows["date"] < tr_end]
            tee = rows[(rows["date"] >= ts) & (rows["date"] < te)]
            if len(tee) < 30 or len(tr) < 300:
                continue
            ytr = tr[yc].to_numpy()
            if shuffle_null:
                ytr = np.random.default_rng(0).permutation(ytr)
            m = make_model(binary)
            m.fit(tr[FEATS], ytr.astype(int) if binary else ytr)
            sc = (m.predict_proba(tee[FEATS])[:, 1] if binary
                  else m.predict(tee[FEATS]))
            pos = tee[yc].to_numpy().astype(float) if binary else tee["pos_gain"].to_numpy().astype(float)
            ic = spearmanr(sc, tee[yc]).statistic
            ic_mag = spearmanr(sc, tee["fwd_max42"]).statistic
            ndays = tee["date"].nunique()
            base = float(pos.mean())
            r = {"n_test": len(tee), "n_train": len(tr), "base_rate": round(base, 4),
                 "ic": round(float(ic), 4), "ic_vs_fwdmax42": round(float(ic_mag), 4)}
            for pct, tag in [(0.05, "top5"), (0.10, "top10")]:
                k = max(1, int(len(sc) * pct))
                prec = float(pos[np.argsort(-sc)[:k]].mean())
                r[f"prec_{tag}"] = round(prec, 4)
                r[f"lift_{tag}"] = round(prec / base, 2) if base > 0 else None
                r[f"{tag}_sig_per_day"] = round(k / ndays, 2)
            res[lab][f"fold_{fname}"] = r
            if not shuffle_null:
                o = tee[["symbol", "date"]].copy()
                o["label"], o["fold"], o["score"] = lab, fname, sc
                o["y"], o["pos"] = tee[yc].to_numpy(), pos
                oos.append(o)
    return res, (pd.concat(oos, ignore_index=True) if oos else None)


# ---------------------------------------------------------------- episodes (>=30% waves)
def find_episodes(df):
    eps = []
    for s, g in df.groupby("symbol"):
        g = g.reset_index(drop=True)
        c = g["close"].to_numpy(float); dt = g["date"].to_numpy()
        i, n = 0, len(c)
        ti = 0  # trough idx
        state = "seek"; pk = 0
        for i in range(1, n):
            if state == "seek":
                if c[i] < c[ti]:
                    ti = i
                elif c[i] >= c[ti] * 1.30 and i - ti <= 90:
                    state, pk = "wave", i
            else:
                if c[i] > c[pk]:
                    pk = i
                elif c[i] <= c[pk] * 0.85:
                    eps.append((s, dt[ti], dt[pk], float(c[pk] / c[ti] - 1), ti, pk))
                    state, ti = "seek", i
        if state == "wave":
            eps.append((s, dt[ti], dt[pk], float(c[pk] / c[ti] - 1), ti, pk))
    return pd.DataFrame(eps, columns=["symbol", "start", "end", "gain", "ti", "pi"])


def episode_coverage(eps, oos, df, trades):
    # captured = champion trade entry inside [start-5bars ~ use dates, end]
    tset = trades.groupby("symbol")["entry_date"].apply(list).to_dict()
    cap = []
    for r in eps.itertuples():
        ent = tset.get(r.symbol, [])
        s0 = (pd.Timestamp(r.start) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
        cap.append(any(s0 <= e <= r.end for e in ent))
    eps = eps.copy(); eps["captured"] = cap
    out = {}
    dates_by_sym = {s: g["date"].reset_index(drop=True) for s, g in df.groupby("symbol")}
    for lab in LABELS:
        o = oos[oos["label"] == lab]
        thr = {f: np.quantile(o[o["fold"] == f]["score"], 0.90) for f in o["fold"].unique()}
        rows = []
        for r in eps.itertuples():
            f = next((fn for fn, ts, te in FOLDS if ts <= r.start < te), None)
            if f is None or f not in thr:
                continue
            ds = dates_by_sym[r.symbol]
            pos_arr = ds[ds == r.start].index
            if len(pos_arr) == 0:
                continue
            j = pos_arr[0]
            win = set(ds.iloc[max(0, j - 3):j + 4])
            sc = o[(o["symbol"] == r.symbol) & (o["date"].isin(win))]["score"]
            rows.append({"captured": r.captured, "gain": r.gain, "year_fold": f,
                         "n_scored_bars": len(sc),
                         "max_score": float(sc.max()) if len(sc) else np.nan,
                         "hit_top_decile": bool(len(sc) and sc.max() >= thr[f])})
        e = pd.DataFrame(rows)
        if len(e) == 0:
            continue
        g = {}
        for capv, tag in [(False, "missed"), (True, "captured")]:
            sub = e[e["captured"] == capv]
            g[tag] = {"n_episodes": len(sub),
                      "pct_with_scored_bar": round(float((sub["n_scored_bars"] > 0).mean()), 3) if len(sub) else None,
                      "pct_hit_top_decile": round(float(sub["hit_top_decile"].mean()), 3) if len(sub) else None,
                      "mean_max_score": round(float(sub["max_score"].mean()), 4) if len(sub) else None}
        out[lab] = g
    return eps, out


# ---------------------------------------------------------------- causality audit
def causality_audit(df, d_full, n_dates=4, tol=1e-9):
    """Rebuild features on data truncated at date T; rows at T must match full build."""
    rng = np.random.default_rng(RNG)
    dates = sorted(d_full.loc[d_full["date"] >= "2021-01-01", "date"].unique())
    bad = 0; checked = 0
    for T in rng.choice(dates, n_dates, replace=False):
        d_tr = build(df[df["date"] <= T])
        a = d_tr[d_tr["date"] == T].set_index("symbol")[FEATS].sort_index()
        b = d_full[d_full["date"] == T].set_index("symbol")[FEATS].sort_index()
        b = b.loc[a.index]
        diff = (a - b).abs().to_numpy()
        m = np.nanmax(diff) if diff.size else 0.0
        checked += a.size
        if m > tol:
            bad += 1
    return {"dates_checked": int(n_dates), "cells_checked": int(checked),
            "dates_with_mismatch": int(bad), "pass": bad == 0}


def main():
    univ, df = load_panel()
    print(f"panel: {df['symbol'].nunique()} symbols, {len(df)} bars,"
          f" {df['date'].min()}..{df['date'].max()}")
    d = build(df)
    d.to_parquet(rf"{OUT}\dataset.parquet")
    for lab, (ec, yc, binary) in LABELS.items():
        el = d[d[ec] & d["lab_ok"]]
        br = el[yc].mean() if binary else (el["pos_gain"]).mean()
        print(f"{lab}: eligible={len(el)} base_rate={br:.3f}")

    res, oos = run_folds(d)
    oos.to_parquet(rf"{OUT}\oos_scores.parquet")
    null_res, _ = run_folds(d, shuffle_null=True)
    nulls = {lab: {k: v["ic"] for k, v in null_res[lab].items() if k.startswith("fold_")}
             for lab in LABELS}

    eps = find_episodes(df)
    trades = pd.read_csv(TRADES)
    eps2, cov = episode_coverage(eps, oos, df, trades)
    eps2.to_csv(rf"{OUT}\episodes_reconstructed.csv", index=False)

    audit = causality_audit(df, d)

    out = {"folds": res, "null_shuffled_ic": nulls, "episode_coverage": cov,
           "n_episodes_total": len(eps2),
           "n_episodes_ge30": int((eps2["gain"] >= 0.30).sum()),
           "episodes_captured_share": round(float(eps2["captured"].mean()), 3),
           "causality_audit": audit,
           "caveats": ["raw prices (not back-adjusted)",
                       "missed_moves.csv not found; episodes reconstructed from OHLCV"
                       " (trough->+30% within 90 bars, end at 15% retrace),"
                       " captured=champion trade entry within [start-7d, end]"]}
    with open(rf"{OUT}\results.json", "w") as f:
        json.dump(out, f, indent=1, default=str)
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""OHLCV virgin-information screening for champion 2646.

For 25 candidate features covering the OHLCV channels the champion stack does
NOT use (overnight gap, multi-bar candle anatomy, HL-range structure, volume
microstructure, VN limit-move counts, price-path health), measure:

  1. RAW standalone OOS IC  — per-date cross-sectional Spearman vs target,
     averaged inside yearly folds 2022..2026H1 (screening only, no training,
     so "OOS" = the champion's walk-forward test years).
  2. ORTHOGONAL IC — per-date cross-sectional rank-OLS residual of the
     candidate on (a) 10 core close-based champion features, and (b) the
     strict base = those 10 + the 6 existing HL/O/V features the stack already
     has. Residual IC ~ 0 => old info in disguise.
  3. NULL band — 200 shuffle-within-date permutations of the candidate,
     pooled-mean-IC 2.5/97.5 percentiles (same missingness, same dates).

Targets:
  (a) fwd_ret_21   = close[t+21]/close[t] - 1            (entry proxy)
  (b) vexit        = champion exit label reproduced exactly:
      velocity_exit_regression(horizon=20, upside_horizon=8,
      vol_normalize=True, vol_window=40)  [stock_ml/src/targets/velocity_exit.py]
      NOTE: positive label = downside coming -> for an exit head a POSITIVE IC
      means "feature high => should exit".

Strictly causal features (rolling windows end at t). Raw non-back-adjusted
prices: overnight gaps are clipped to +/-7.5% (HOSE band) to damp
corporate-action artifacts; caveat recorded in the report.
"""
import json
import sqlite3

import numpy as np
import pandas as pd

DB = r"C:\Users\DUC CANH PC\Desktop\stock-serving\data\ohlcv.db"
UNIV = r"f:\PROJECTS\train_ai_ml\_tmp_analysis\univ_2429.txt"
OUT = r"f:\PROJECTS\train_ai_ml\stock_ml\analysis\serving_blindspot\ohlcv"

FOLDS = [("2022", "2022-01-01", "2023-01-01"),
         ("2023", "2023-01-01", "2024-01-01"),
         ("2024", "2024-01-01", "2025-01-01"),
         ("2025", "2025-01-01", "2026-01-01"),
         ("2026H1", "2026-01-01", "2026-07-01")]
MIN_SYM = 40          # min non-NaN symbols per date for an IC obs
N_SHUFFLE = 200
RNG = np.random.default_rng(42)

# ---------------------------------------------------------------- load
def load_panel():
    univ = [l.strip() for l in open(UNIV) if l.strip()]
    con = sqlite3.connect(DB)
    q = ("select symbol, date, open, high, low, close, volume from ohlcv "
         f"where symbol in ({','.join('?' * len(univ))}) order by symbol, date")
    df = pd.read_sql(q, con, params=univ)
    con.close()
    df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)].reset_index(drop=True)
    return df


# ---------------------------------------------------------------- features (causal, <= t)
def _rsi(c, n):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def sym_features(g):
    c, h, l, o, v = g["close"], g["high"], g["low"], g["open"], g["volume"].astype(float)
    f = pd.DataFrame(index=g.index)
    pc = c.shift(1)
    ret1 = c.pct_change()
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    rng_hl = (h - l).where(h > l)          # NaN on limit-lock bars (h == l)
    dv = (c * v).replace(0, np.nan)

    # ============ BASE (champion close-based core, 10) ============
    f["b_ret_5d"] = c.pct_change(5)
    f["b_ret_20d"] = c.pct_change(20)
    f["b_sma_20_ratio"] = c / c.rolling(20).mean() - 1
    f["b_sma_50_ratio"] = c / c.rolling(50).mean() - 1
    f["b_rsi_14"] = _rsi(c, 14)
    ema12, ema26 = c.ewm(span=12, min_periods=12).mean(), c.ewm(span=26, min_periods=26).mean()
    macd = ema12 - ema26
    f["b_macd_hist"] = macd - macd.ewm(span=9, min_periods=9).mean()
    f["b_realized_vol_10"] = ret1.rolling(10).std()
    f["b_dist_52w_high"] = c / c.rolling(252).max() - 1
    r20mx, r20mn = c.rolling(20).max(), c.rolling(20).min()
    f["b_range_pos_20"] = (c - r20mn) / (r20mx - r20mn).replace(0, np.nan)
    f["b_dist_10d_low"] = c / c.rolling(10).min() - 1
    # ============ STRICT-BASE extras (existing HL/O/V features, 6) ============
    atr14 = tr.rolling(14).mean()
    f["s_atr_14_ratio"] = atr14 / c
    f["s_high_low_pct_5d"] = (h.rolling(5).max() - l.rolling(5).min()) / c
    f["s_upper_wick_ratio"] = (h - pd.concat([o, c], axis=1).max(axis=1)) / rng_hl
    f["s_volume_ratio_20"] = v / v.rolling(20).mean().replace(0, np.nan)
    upv = v.where(ret1 > 0, 0.0).rolling(20).sum()
    dnv = v.where(ret1 <= 0, 0.0).rolling(20).sum()
    f["s_updown_vol_20"] = upv / dnv.replace(0, np.nan)
    tp = (h + l + c) / 3
    mf = tp * v
    pos_mf = mf.where(tp > tp.shift(1), 0.0).rolling(14).sum()
    neg_mf = mf.where(tp <= tp.shift(1), 0.0).rolling(14).sum()
    f["s_mfi_14"] = 100 - 100 / (1 + pos_mf / neg_mf.replace(0, np.nan))

    # ============ CANDIDATES (25) ============
    # --- GAP / overnight channel (fully virgin) ---
    gap = (o / pc - 1).clip(-0.075, 0.075)
    intra = c / o - 1
    f["c_gap_ret1"] = gap
    f["c_gap_abs_mean10"] = gap.abs().rolling(10).mean()
    f["c_gap_freq15_10"] = (gap.abs() > 0.015).rolling(10).mean()
    f["c_on_drift_20"] = np.log1p(gap).rolling(20).sum()
    f["c_gap_follow_10"] = (np.sign(gap) * intra).rolling(10).mean()
    # --- CANDLE anatomy, multi-bar (single-bar-only in champion) ---
    clv = ((c - l) - (h - c)) / rng_hl
    f["c_clv"] = clv
    f["c_clv_mean_10"] = clv.rolling(10).mean()
    f["c_uwick_mean_10"] = ((h - pd.concat([o, c], axis=1).max(axis=1)) / rng_hl).rolling(10).mean()
    f["c_lwick_mean_10"] = ((pd.concat([o, c], axis=1).min(axis=1) - l) / rng_hl).rolling(10).mean()
    # --- RANGE / HL structure ---
    f["c_nr_pos_7"] = tr / tr.rolling(7).max().replace(0, np.nan)
    f["c_tr_ratio_5_20"] = tr.rolling(5).mean() / tr.rolling(20).mean().replace(0, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        park = np.sqrt((np.log((h / l).where(h > l)) ** 2).rolling(20).mean() / (4 * np.log(2)))
    f["c_park_vs_close_20"] = park / ret1.rolling(20).std().replace(0, np.nan)
    f["c_range_pos_hl_20"] = (c - l.rolling(20).min()) / (
        h.rolling(20).max() - l.rolling(20).min()).replace(0, np.nan)
    f["c_path_eff_10"] = (c / c.shift(10) - 1).abs() / (tr / pc).rolling(10).sum().replace(0, np.nan)
    # --- VOLUME microstructure ---
    f["c_amihud_20"] = np.log((ret1.abs() / dv).rolling(20).mean().replace(0, np.nan))
    f["c_vol_cv_20"] = v.rolling(20).std() / v.rolling(20).mean().replace(0, np.nan)
    f["c_volu_ma5_20"] = v.rolling(5).mean() / v.rolling(20).mean().replace(0, np.nan)
    dlv = np.log(v.replace(0, np.nan)).diff()
    f["c_pv_corr_10"] = ret1.rolling(10).corr(dlv)
    sv = (v.where(ret1 > 0, 0.0) - v.where(ret1 < 0, 0.0)).rolling(10).sum()
    f["c_signed_vol_10"] = sv / v.rolling(10).sum().replace(0, np.nan)
    f["c_eff_result_10"] = (v * clv).rolling(10).mean() / v.rolling(10).mean().replace(0, np.nan)
    # --- VN limit-band behaviour ---
    f["c_limit_up_cnt_20"] = (ret1 >= 0.065).rolling(20).sum()
    f["c_limit_dn_cnt_20"] = (ret1 <= -0.065).rolling(20).sum()
    f["c_bigmove_freq_60"] = (ret1.abs() >= 0.05).rolling(60).mean()
    # --- price-path health ---
    trr = tr / tr.rolling(60).mean().replace(0, np.nan)
    f["c_range_spike_freq_20"] = (trr > 1.8).rolling(20).mean()
    f["c_zero_ret_freq_20"] = (ret1.abs() < 1e-4).rolling(20).mean()

    # ============ TARGETS (forward-looking, labels only) ============
    f["t_fwd21"] = c.shift(-21) / c - 1
    # champion exit label: velocity_exit_regression h20 u8 volnorm40 (exact port)
    dmin = pd.concat([c.shift(-k) for k in range(1, 21)], axis=1).min(axis=1)
    umax = pd.concat([c.shift(-k) for k in range(1, 9)], axis=1).max(axis=1)
    obs = c.shift(-20).notna()
    lab = (1.0 - dmin.where(obs) / c) - (umax.where(obs) / c - 1.0)
    vol40 = ret1.rolling(40, min_periods=2).std()
    f["t_vexit"] = lab / (vol40 + 1e-4)
    return f


BASE10 = ["b_ret_5d", "b_ret_20d", "b_sma_20_ratio", "b_sma_50_ratio", "b_rsi_14",
          "b_macd_hist", "b_realized_vol_10", "b_dist_52w_high", "b_range_pos_20",
          "b_dist_10d_low"]
STRICT = BASE10 + ["s_atr_14_ratio", "s_high_low_pct_5d", "s_upper_wick_ratio",
                   "s_volume_ratio_20", "s_updown_vol_20", "s_mfi_14"]
CANDS = ["c_gap_ret1", "c_gap_abs_mean10", "c_gap_freq15_10", "c_on_drift_20",
         "c_gap_follow_10", "c_clv", "c_clv_mean_10", "c_uwick_mean_10",
         "c_lwick_mean_10", "c_nr_pos_7", "c_tr_ratio_5_20", "c_park_vs_close_20",
         "c_range_pos_hl_20", "c_path_eff_10", "c_amihud_20", "c_vol_cv_20",
         "c_volu_ma5_20", "c_pv_corr_10", "c_signed_vol_10", "c_eff_result_10",
         "c_limit_up_cnt_20", "c_limit_dn_cnt_20", "c_bigmove_freq_60",
         "c_range_spike_freq_20", "c_zero_ret_freq_20"]
TARGETS = ["t_fwd21", "t_vexit"]


def build(df):
    parts = []
    for s, g in df.groupby("symbol"):
        g = g.reset_index(drop=True)
        f = sym_features(g)
        f["symbol"], f["date"] = s, g["date"]
        parts.append(f)
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------- IC machinery
def _rank_norm(a):
    """NaN-aware average-rank -> centered uniform, per 1-D array."""
    out = np.full(a.shape, np.nan)
    m = ~np.isnan(a)
    n = m.sum()
    if n < 3:
        return out
    r = pd.Series(a[m]).rank(method="average").to_numpy()
    out[m] = (r - (n + 1) / 2) / n
    return out


def precompute_groups(dat):
    """Per-date pre-ranked arrays: base cols + targets ranked ONCE (shared by all
    candidates); candidate columns kept raw and ranked on demand."""
    groups = []
    shared_cols = STRICT + TARGETS
    for dt, gg in dat.groupby("date", sort=True):
        ranked = {c: _rank_norm(gg[c].to_numpy(float)) for c in shared_cols}
        raw = {c: gg[c].to_numpy(float) for c in CANDS}
        groups.append((dt, ranked, raw))
    return groups


def per_date_ic(groups, xcol, ycol, resid_on=None):
    """Per-date Spearman IC of x (optionally residualized on base ranks) vs y."""
    ics, dates = [], []
    for dt, ranked, raw in groups:
        x = _rank_norm(raw[xcol])
        y = ranked[ycol]
        m = ~np.isnan(x) & ~np.isnan(y)
        if resid_on is not None:
            B = np.column_stack([ranked[b] for b in resid_on])
            m &= ~np.isnan(B).any(axis=1)
            if m.sum() < MIN_SYM:
                continue
            Bm = np.column_stack([np.ones(m.sum()), B[m]])
            beta, *_ = np.linalg.lstsq(Bm, x[m], rcond=None)
            xr = x[m] - Bm @ beta
        else:
            if m.sum() < MIN_SYM:
                continue
            xr = x[m]
        yv = y[m]
        sx, sy = xr.std(), yv.std()
        if sx <= 0 or sy <= 0:
            continue
        ics.append(float(np.mean((xr - xr.mean()) * (yv - yv.mean())) / (sx * sy)))
        dates.append(dt)
    return pd.Series(ics, index=pd.Index(dates, name="date"))


def fold_means(ic_series):
    out = {}
    for name, ts, te in FOLDS:
        s = ic_series[(ic_series.index >= ts) & (ic_series.index < te)]
        out[name] = round(float(s.mean()), 4) if len(s) >= 30 else None
    pooled = ic_series[ic_series.index >= FOLDS[0][1]]
    out["pooled"] = round(float(pooled.mean()), 4) if len(pooled) else None
    out["n_days"] = int(len(pooled))
    # conservative t: effective N = n/21 (overlapping fwd windows)
    if len(pooled) > 42 and pooled.std() > 0:
        out["t_cons"] = round(float(pooled.mean() / (pooled.std() / np.sqrt(len(pooled) / 21.0))), 2)
    else:
        out["t_cons"] = None
    return out


def null_band(groups, xcol, ycol, empirical=False):
    """Shuffle-within-date null for the pooled mean raw IC (2022+).

    Analytic: under within-date permutation the per-date IC has mean 0,
    var 1/(n_d - 1)  ->  pooled-mean sd = sqrt(mean_d 1/(n_d-1)) / sqrt(D).
    `empirical=True` additionally runs N_SHUFFLE real permutations to
    validate the analytic band (done for one candidate, reported alongside).
    """
    rows = []
    for dt, ranked, raw in groups:
        if dt < FOLDS[0][1]:
            continue
        x = _rank_norm(raw[xcol])
        y = ranked[ycol]
        m = ~np.isnan(x) & ~np.isnan(y)
        if m.sum() < MIN_SYM:
            continue
        rows.append((x[m], y[m]))
    if not rows:
        return None
    ns = np.array([len(x) for x, _ in rows], float)
    sd = float(np.sqrt(np.mean(1.0 / (ns - 1.0))) / np.sqrt(len(rows)))
    out = {"null_lo": round(-1.96 * sd, 4), "null_hi": round(1.96 * sd, 4),
           "null_sd": round(sd, 4), "n_days": len(rows)}
    if empirical:
        means = []
        for _ in range(N_SHUFFLE):
            tot = 0.0
            for x, y in rows:
                xp = RNG.permutation(x)
                tot += np.mean((xp - xp.mean()) * (y - y.mean())) / (xp.std() * y.std())
            means.append(tot / len(rows))
        means = np.array(means)
        out["emp_lo"] = round(float(np.percentile(means, 2.5)), 4)
        out["emp_hi"] = round(float(np.percentile(means, 97.5)), 4)
        out["emp_sd"] = round(float(means.std()), 4)
    return out


# ---------------------------------------------------------------- causality audit
def causality_audit(df, d_full, n_dates=3, tol=1e-9):
    dates = sorted(d_full.loc[d_full["date"] >= "2022-01-01", "date"].unique())
    rng = np.random.default_rng(7)
    bad = 0
    cols = CANDS + BASE10 + STRICT[len(BASE10):]
    for T in rng.choice(dates, n_dates, replace=False):
        d_tr = build(df[df["date"] <= T])
        a = d_tr[d_tr["date"] == T].set_index("symbol")[cols].sort_index()
        b = d_full[d_full["date"] == T].set_index("symbol")[cols].sort_index().loc[a.index]
        diff = (a - b).abs().to_numpy()
        if diff.size and np.nanmax(diff) > tol:
            bad += 1
    return {"dates_checked": n_dates, "dates_with_mismatch": bad, "pass": bad == 0}


def main():
    df = load_panel()
    print(f"panel: {df['symbol'].nunique()} syms, {len(df)} bars, "
          f"{df['date'].min()}..{df['date'].max()}", flush=True)
    d = build(df)
    # IC folds start 2022 -> only rank/score dates from 2021-12 on (features/labels
    # were computed on the FULL history above, so windows are already warm).
    groups = precompute_groups(d[d["date"] >= "2021-12-01"])
    print(f"scoring dates: {len(groups)}", flush=True)

    results = {}
    for cand in CANDS:
        r = {}
        for tgt in TARGETS:
            raw = per_date_ic(groups, cand, tgt)
            r[f"{tgt}_raw"] = fold_means(raw)
            r[f"{tgt}_resid_close10"] = fold_means(per_date_ic(groups, cand, tgt, resid_on=BASE10))
            r[f"{tgt}_resid_strict16"] = fold_means(per_date_ic(groups, cand, tgt, resid_on=STRICT))
            r[f"{tgt}_null"] = null_band(groups, cand, tgt,
                                         empirical=(cand == "c_clv"))
        results[cand] = r
        print(cand, "fwd21 raw", r["t_fwd21_raw"]["pooled"],
              "| resid16", r["t_fwd21_resid_strict16"]["pooled"],
              "| vexit resid16", r["t_vexit_resid_strict16"]["pooled"], flush=True)

    audit = causality_audit(df, d)
    print("causality:", audit, flush=True)

    out = {"panel": {"symbols": int(df["symbol"].nunique()), "bars": int(len(df)),
                     "span": [str(df["date"].min()), str(df["date"].max())]},
           "candidates": results, "causality_audit": audit,
           "notes": ["raw non-back-adjusted prices; gaps clipped +/-7.5%",
                     "IC = per-date cross-sectional Spearman, mean per yearly fold",
                     "resid = per-date rank-OLS residual on base features",
                     "t_vexit positive = downside coming (exit label)"]}
    with open(rf"{OUT}\screen_results.json", "w") as fh:
        json.dump(out, fh, indent=1)

    # flat CSV for the report
    rows = []
    for cand, r in results.items():
        row = {"candidate": cand}
        for tgt in TARGETS:
            for kind in ("raw", "resid_close10", "resid_strict16"):
                fm = r[f"{tgt}_{kind}"]
                row[f"{tgt}_{kind}_pooled"] = fm["pooled"]
                row[f"{tgt}_{kind}_t"] = fm["t_cons"]
                for fname, _, _ in FOLDS:
                    row[f"{tgt}_{kind}_{fname}"] = fm[fname]
            nb = r[f"{tgt}_null"]
            row[f"{tgt}_null_hi"] = nb["null_hi"] if nb else None
            row[f"{tgt}_null_lo"] = nb["null_lo"] if nb else None
        rows.append(row)
    pd.DataFrame(rows).to_csv(rf"{OUT}\screen_results.csv", index=False)
    print("saved screen_results.json / screen_results.csv")


if __name__ == "__main__":
    main()

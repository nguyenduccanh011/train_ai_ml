# -*- coding: utf-8 -*-
"""Bar-level anatomy of the first 10 bars of upswing episodes + fill-anchor candidates.

Per episode (start = local-low bar s, thrust = first bar j in (s, s+5] with
close[j] >= close[s]*1.02; fallback (s, s+10]):
  retest depths, thrust-close-minus-X touch rates, thrust-low revisit, MA20 reclaim,
  gain banked by reclaim.
Anchors (limit fills use entry = min(open_of_fill_bar, limit)):
  A1 limit thrust_close*0.98, active (th, s+10]
  A2 limit thrust_low,        active (th, s+10]
  A3 limit start_close,       active (th, s+10]
  A4 buy close of first bar closing > thrust high, (th, s+10]
  A5 limit reclaim_close*0.98, active (r, r+10]
Control set: trailing-20-bar-low bars with close<MA20, a >=2% bounce within 5 bars,
fwd 42-bar max close gain < 8%, outside episodes, non-overlapping (21-bar spacing).
"""
import json
import os
import sqlite3

import numpy as np
import pandas as pd

BASE = r"f:\PROJECTS\train_ai_ml\stock_ml\analysis\serving_blindspot"
OUT = os.path.join(BASE, "wavestart")
DB = r"C:\Users\DUC CANH PC\Desktop\stock-serving\data\ohlcv.db"

ep = pd.read_csv(os.path.join(OUT, "missed_moves.csv"))
universe = sorted(ep["symbol"].unique().tolist() +
                  [s for s in pd.read_csv(os.path.join(BASE, "signals.csv"),
                                          usecols=["symbol"])["symbol"].unique()])
universe = sorted(set(universe))

con = sqlite3.connect(DB)
q = "select symbol, date, open, high, low, close from ohlcv where symbol in (%s) order by symbol, date" % (
    ",".join("?" * len(universe)))
oh = pd.read_sql(q, con, params=universe)
con.close()

D = {}
for sym, g in oh.groupby("symbol"):
    g = g.reset_index(drop=True)
    D[sym] = dict(dates=g["date"].to_numpy(),
                  o=g["open"].to_numpy(float), h=g["high"].to_numpy(float),
                  lo=g["low"].to_numpy(float), c=g["close"].to_numpy(float),
                  ma20=g["close"].rolling(20).mean().to_numpy())

ANCHORS = ["A1", "A2", "A3", "A4", "A5"]


def find_thrust(c, s, n):
    for j in range(s + 1, min(s + 6, n)):
        if c[j] >= c[s] * 1.02:
            return j
    for j in range(s + 6, min(s + 11, n)):
        if c[j] >= c[s] * 1.02:
            return j
    return None


def limit_fill(o, lo, price, j0, j1, n):
    """First fill of a limit `price` scanning bars [j0, j1]. Returns (idx, entry) or None."""
    for j in range(j0, min(j1 + 1, n)):
        if lo[j] <= price:
            return j, min(o[j], price)
    return None


def anchor_stats(d, s, th, n, horizon_end):
    """Compute anchor fills for a start bar s with thrust th. horizon_end = s+10 cap."""
    o, h, lo, c, ma = d["o"], d["h"], d["lo"], d["c"], d["ma20"]
    res = {}
    H = min(horizon_end, n - 1)
    # A1 / A2 / A3
    for name, price in (("A1", c[th] * 0.98), ("A2", lo[th]), ("A3", c[s])):
        res[name] = limit_fill(o, lo, price, th + 1, H, n)
    # A4 confirmation: first close > thrust high
    res["A4"] = None
    for j in range(th + 1, H + 1):
        if c[j] > h[th]:
            res["A4"] = (j, c[j])
            break
    # A5: MA20 reclaim day r (first j>=s with close>MA20), limit reclaim_close*0.98 in (r, r+10]
    res["A5"] = None
    r = None
    for j in range(s, min(s + 22, n)):
        if not np.isnan(ma[j]) and c[j] > ma[j]:
            r = j
            break
    if r is not None:
        res["A5"] = limit_fill(o, lo, c[r] * 0.98, r + 1, min(r + 10, n - 1), n)
    return res, r


def fwd_metrics(d, fill_idx, entry, n):
    c, lo = d["c"], d["lo"]
    j21 = fill_idx + 21
    fwd21 = c[j21] / entry - 1.0 if j21 < n else np.nan
    end = min(fill_idx + 21, n - 1)
    mae = lo[fill_idx:end + 1].min() / entry - 1.0
    return fwd21, mae


# ---------------- per-episode anatomy ----------------
rows = []
anch_rows = []
for e in ep.itertuples():
    d = D[e.symbol]
    c, lo, h, ma = d["c"], d["lo"], d["h"], d["ma20"]
    n = len(c)
    s = int(e.start_idx)
    assert d["dates"][s] == e.start_date
    th = find_thrust(c, s, n)
    H = min(s + 10, n - 1)
    w_lo = lo[s + 1:H + 1]
    # (a) deepest retest vs start close & vs running-peak close
    retest_vs_start = w_lo.min() / c[s] - 1.0
    runpk = np.maximum.accumulate(c[s:H])  # peak close up to j-1
    pullback_run = float((1.0 - lo[s + 1:H + 1] / runpk).max())
    row = dict(symbol=e.symbol, start_date=e.start_date, gain_pct=e.gain_pct,
               captured=e.captured, thrust_bar=(th - s) if th else np.nan,
               retest_vs_start=retest_vs_start, max_pullback_run=pullback_run,
               below_start=w_lo.min() < c[s])
    if th is not None:
        tl = lo[th + 1:H + 1]
        tmin = tl.min() if len(tl) else np.inf
        for x in (1, 2, 3, 4.5):
            row["touch_%s" % x] = tmin <= c[th] * (1 - x / 100.0)
        row["thrust_low_5"] = lo[th + 1:min(th + 6, n)].min() <= lo[th] if th + 1 < n else False
        row["thrust_low_10"] = lo[th + 1:min(th + 11, n)].min() <= lo[th] if th + 1 < n else False
        row["retest_vs_thrust"] = (tmin / c[th] - 1.0) if np.isfinite(tmin) else np.nan
    # (d)(e) MA20 reclaim
    res, r = anchor_stats(d, s, th, n, s + 10) if th is not None else ({k: None for k in ANCHORS}, None)
    if r is not None:
        row["reclaim_bars"] = r - s
        banked = c[r] / c[s] - 1.0
        row["reclaim_banked"] = banked
        row["reclaim_banked_share"] = banked / (e.gain_pct / 100.0)
    rows.append(row)
    if th is None:
        continue
    for a in ANCHORS:
        f = res[a]
        rec = dict(symbol=e.symbol, start_date=e.start_date, gain_pct=e.gain_pct,
                   captured=e.captured, anchor=a, filled=f is not None)
        if f is not None:
            fi, entry = f
            fwd21, mae = fwd_metrics(d, fi, entry, n)
            s21 = s + 21
            rec.update(fill_bar=fi - s, entry=entry, fwd21=fwd21, mae=mae,
                       disc_vs_s21=(c[s21] / entry - 1.0) if s21 < n else np.nan)
        anch_rows.append(rec)

anat = pd.DataFrame(rows)
anch = pd.DataFrame(anch_rows)
anat.to_csv(os.path.join(OUT, "episode_anatomy.csv"), index=False)
anch.to_csv(os.path.join(OUT, "anchor_fills.csv"), index=False)

# ---------------- control set ----------------
ep_iv = {}
for e in ep.itertuples():
    ep_iv.setdefault(e.symbol, []).append((e.start_idx - 2, e.peak_idx))

ctrl_anch = []
n_ctrl = 0
for sym, d in D.items():
    c, lo, ma = d["c"], d["lo"], d["ma20"]
    dates = d["dates"]
    n = len(c)
    trail = pd.Series(lo).rolling(20).min().to_numpy()
    iv = ep_iv.get(sym, [])
    t = max(20, int(np.searchsorted(dates, "2020-01-01")))
    while t < n - 42:
        ok = (lo[t] == trail[t] and not np.isnan(ma[t]) and c[t] < ma[t]
              and not any(a <= t <= b for a, b in iv))
        if ok:
            th = find_thrust(c, t, n)
            if th is not None and th <= t + 5:
                fwd42 = c[t + 1:t + 43].max() / c[t] - 1.0
                if fwd42 < 0.08:
                    n_ctrl += 1
                    res, r = anchor_stats(d, t, th, n, t + 10)
                    for a in ANCHORS:
                        f = res[a]
                        rec = dict(symbol=sym, start_date=dates[t], anchor=a,
                                   filled=f is not None)
                        if f is not None:
                            fi, entry = f
                            fwd21, mae = fwd_metrics(d, fi, entry, n)
                            rec.update(fwd21=fwd21, mae=mae)
                        ctrl_anch.append(rec)
                    t += 21
                    continue
        t += 1

ctrl = pd.DataFrame(ctrl_anch)
ctrl.to_csv(os.path.join(OUT, "control_anchor_fills.csv"), index=False)

# ---------------- aggregation ----------------
anat["bucket"] = np.where(anat.gain_pct >= 30, ">=30%", "15-30%")
anch["bucket"] = np.where(anch.gain_pct >= 30, ">=30%", "15-30%")


def agg_anatomy(g):
    out = dict(n=len(g),
               thrust_bar_med=g.thrust_bar.median(),
               retest_vs_start_med=g.retest_vs_start.median(),
               pct_below_start=100 * g.below_start.mean(),
               max_pullback_run_med=g.max_pullback_run.median(),
               retest_vs_thrust_med=g.retest_vs_thrust.median())
    for x in (1, 2, 3, 4.5):
        out["touch_%s" % x] = 100 * g["touch_%s" % x].mean()
    out["thrust_low_5"] = 100 * g.thrust_low_5.mean()
    out["thrust_low_10"] = 100 * g.thrust_low_10.mean()
    out["reclaim_bars_med"] = g.reclaim_bars.median()
    out["reclaim_banked_med"] = 100 * g.reclaim_banked.median()
    out["reclaim_share_med"] = 100 * g.reclaim_banked_share.median()
    return pd.Series(out)


grp = anat.groupby(["captured", "bucket"]).apply(agg_anatomy)
grp_all = agg_anatomy(anat).to_frame("ALL").T
anat_summary = pd.concat([grp, grp_all])
anat_summary.round(2).to_csv(os.path.join(OUT, "anatomy_summary.csv"))


def agg_anchor(g):
    fills = g[g.filled]
    out = dict(n_ep=len(g), capture=100 * g.filled.mean())
    if len(fills):
        out.update(fill_bar_med=fills.fill_bar.median(),
                   fwd21_mean=100 * fills.fwd21.mean(),
                   fwd21_med=100 * fills.fwd21.median(),
                   disc_vs_s21_mean=100 * fills.disc_vs_s21.mean(),
                   mae_mean=100 * fills.mae.mean(),
                   mae_p10=100 * fills.mae.quantile(0.10),
                   score=g.filled.mean() * fills.fwd21.mean() * 100 + 100 * fills.mae.quantile(0.10))
    return pd.Series(out)


anchor_summary = anch.groupby("anchor").apply(agg_anchor)
anchor_by_cap = anch.groupby(["captured", "bucket", "anchor"]).apply(agg_anchor)
anchor_missed = anch[~anch.captured].groupby("anchor").apply(agg_anchor)


def agg_ctrl(g):
    fills = g[g.filled]
    out = dict(n=len(g), fill_rate=100 * g.filled.mean())
    if len(fills):
        out.update(fwd21_mean=100 * fills.fwd21.mean(), fwd21_med=100 * fills.fwd21.median(),
                   mae_mean=100 * fills.mae.mean(), mae_p10=100 * fills.mae.quantile(0.10))
    return pd.Series(out)


ctrl_summary = ctrl.groupby("anchor").apply(agg_ctrl)

anchor_summary.round(2).to_csv(os.path.join(OUT, "anchor_summary.csv"))
anchor_by_cap.round(2).to_csv(os.path.join(OUT, "anchor_by_group.csv"))
ctrl_summary.round(2).to_csv(os.path.join(OUT, "control_summary.csv"))

timing = ep.assign(year=ep.start_date.str[:4]).groupby(["year", "captured"]).size().unstack(fill_value=0)
timing.columns = ["missed", "captured"]
timing["missed_pct"] = (100 * timing.missed / timing.sum(axis=1)).round(1)
timing.to_csv(os.path.join(OUT, "timing_by_year.csv"))

big_missed = ep[(~ep.captured) & (ep.gain_pct >= 30)]
timing_bm = big_missed.start_date.str[:4].value_counts().sort_index()

pd.set_option("display.width", 250)
print("=== ANATOMY (median/%), by captured x bucket ===")
print(anat_summary.round(2).to_string())
print("\n=== ANCHORS all episodes ===")
print(anchor_summary.round(2).to_string())
print("\n=== ANCHORS missed-only ===")
print(anchor_missed.round(2).to_string())
print("\n=== ANCHORS by captured x bucket ===")
print(anchor_by_cap.round(2).to_string())
print("\n=== CONTROL (false-fill exposure), n_controls=%d ===" % n_ctrl)
print(ctrl_summary.round(2).to_string())
print("\n=== TIMING episode starts per year ===")
print(timing.to_string())
print("\n>=30%-missed starts by year:")
print(timing_bm.to_string())

json.dump(dict(n_episodes=len(ep), n_controls=n_ctrl,
               anchor_summary=anchor_summary.round(3).to_dict(),
               anchor_missed=anchor_missed.round(3).to_dict(),
               control_summary=ctrl_summary.round(3).to_dict()),
          open(os.path.join(OUT, "results.json"), "w"), indent=1)
print("\nartifacts written to", OUT)

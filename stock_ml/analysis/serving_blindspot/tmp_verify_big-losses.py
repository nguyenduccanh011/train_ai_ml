# -*- coding: utf-8 -*-
"""Adversarial verifier for the big-losses claims. Independent recompute:
- tail anatomy stats straight from trades_metrics.csv
- tail-cutter simulation re-implemented from scratch against ohlcv.db
- MSR/TIG bar-path spot checks
Prints everything; writes nothing else.
"""
import sqlite3
import numpy as np
import pandas as pd

AN = r"f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot"
DB = r"C:/Users/DUC CANH PC/Desktop/stock-serving/data/ohlcv.db"
pd.set_option("display.width", 250)

T = pd.read_csv(AN + r"/trades_metrics.csv", parse_dates=["entry_date", "exit_date", "signal_date"])
n = len(T)
net = T.pnl_pct.sum()
gp = T.loc[T.pnl_pct > 0, "pnl_pct"].sum()
print(f"[C1] closed trades={n} net={net:.2f}u gross_profit={gp:.2f}u")
for thr in (-0.10, -0.15, -0.20, -0.25, -0.30):
    B = T[T.pnl_pct <= thr]
    print(f"[C1] pnl<={thr:.0%}: n={len(B)} sum={B.pnl_pct.sum():.2f}u "
          f"= {B.pnl_pct.sum()/gp:.2%} gross, {B.pnl_pct.sum()/net:.2%} net")
print(f"[C1] min pnl={T.pnl_pct.min():.4f}")

# C2 worst singles
i0 = T.pnl_pct.idxmin()
print(f"[C2] worst: {T.loc[i0,'symbol']} {T.loc[i0,'entry_date'].date()}->{T.loc[i0,'exit_date'].date()} "
      f"pnl={T.loc[i0,'pnl_pct']:.4f} = {T.loc[i0,'pnl_pct']/net:.2%} of net")
w10 = T.nsmallest(10, "pnl_pct").pnl_pct.sum()
w20 = T.nsmallest(20, "pnl_pct").pnl_pct.sum()
print(f"[C2] worst10={w10:.2f}u ({w10/gp:.2%} gross, {w10/net:.2%} net); "
      f"worst20={w20:.2f}u ({w20/gp:.2%} gross, {w20/net:.2%} net)")
m30 = T[T.mae_pct <= -0.30]
print(f"[C2] mae<=-30%: n={len(m30)} min mae={T.mae_pct.min():.4f} "
      f"deepest={T.loc[T.mae_pct.idxmin(),'symbol']} {T.loc[T.mae_pct.idxmin(),'entry_date'].date()}; "
      f"all closed above trough: {bool((m30.pnl_pct > m30.mae_pct).all())}")

# C3 exit reasons + trail/bev arming
bl15 = T[T.pnl_pct <= -0.15]
bl10 = T[T.pnl_pct <= -0.10]
print(f"[C3] bl15 exit_reason={bl15.exit_reason.value_counts().to_dict()} "
      f"bl10 exit_reason={bl10.exit_reason.value_counts().to_dict()}")
print(f"[C3] bl15 max mfe={bl15.mfe_pct.max():.4f} n mfe>=0.27={ (bl15.mfe_pct>=0.27).sum() } "
      f"n mfe>=0.12={ (bl15.mfe_pct>=0.12).sum() }")

# C4 lateness / bounce
late = bl15.pnl_pct - bl15.mae_pct
print(f"[C4] lateness mean={late.mean():.4f} median={late.median():.4f} "
      f"share<=3pp={ (late<=0.03).mean():.1%}; holding median={bl15.holding_days.median():.0f}; "
      f"bounce>5% in 21b={ (bl15.post_ret21>0.05).mean():.1%}; post_ret21 median={bl15.post_ret21.median():.4f}")

# C5 crash-month concentration
xm = bl15.exit_date.dt.strftime("%Y-%m")
crash = xm.isin(["2020-03", "2022-06", "2022-10"])
print(f"[C5] crash months: {xm[crash].value_counts().to_dict()} total={crash.sum()}/{len(bl15)} "
      f"={crash.mean():.1%} sum={bl15.pnl_pct[crash].sum():.2f}u")

# C6 knife-fill profile
bl20 = T[T.pnl_pct <= -0.20]
print(f"[C6] touch_red bl15={bl15.touch_bar_red.mean():.1%} base={T.touch_bar_red.mean():.1%}; "
      f"hot_run bl15={bl15.hot_run.mean():.1%} base={T.hot_run.mean():.1%} bl20={bl20.hot_run.mean():.1%}; "
      f"mfe<3%={ (bl15.mfe_pct<0.03).mean():.1%}; wait>=20={ (bl15.wait_bars>=20).mean():.1%} "
      f"wait>=40 n={ (bl15.wait_bars>=40).sum() }; wait median={bl15.wait_bars.median():.0f}")
# hot_run definition cross-check vs pre_ret20>=0.12
print(f"[C6] hot_run==(pre_ret20>=0.12)? mismatches={ (T.hot_run != (T.pre_ret20>=0.12)).sum() }")

# C11 adjustment cleanliness
print(f"[C11] spans_bad_adjustment: bl15={int(bl15.spans_bad_adjustment.sum())} "
      f"bl10={int(bl10.spans_bad_adjustment.sum())} all={int(T.spans_bad_adjustment.sum())}")

# ================= independent tail-cutter sim =================
con = sqlite3.connect(DB)
bars = {}
for s in T.symbol.unique():
    df = pd.read_sql("select date, high, low, close from ohlcv where symbol=? order by date",
                     con, params=(s,))
    bars[s] = (df["date"].tolist(), {d: i for i, d in enumerate(df["date"])},
               df["high"].to_numpy(float), df["low"].to_numpy(float), df["close"].to_numpy(float))
con.close()

# calibration
ratios, ks, loc = [], [], []
for r in T.itertuples():
    dates, idx, hi, lo, cl = bars[r.symbol]
    e, x = r.entry_date.strftime("%Y-%m-%d"), r.exit_date.strftime("%Y-%m-%d")
    if e in idx and x in idx:
        ei, xi = idx[e], idx[x]
        loc.append((ei, xi))
        ratios.append(r.exit_price / cl[xi])
        ks.append(r.pnl_pct - (r.exit_price / r.entry_price - 1.0))
    else:
        loc.append(None)
ratios = np.array(ratios); ks = np.array(ks)
print(f"[C7] calib: n_missing={sum(1 for v in loc if v is None)} "
      f"slip median={np.median(ratios):.5f} std={ratios.std():.2e}; k median={np.median(ks):.5f} std={ks.std():.2e}")
SLIP = float(np.median(ratios));

RULES = ["stop_8", "stop_10", "stop_12", "tstop_15", "tstop_25", "bev_lock_8", "bev_lock_12"]
sim = {lab: np.full(n, np.nan) for lab in RULES}
simd = {lab: [None] * n for lab in RULES}

for t, r in enumerate(T.itertuples()):
    if loc[t] is None:
        continue
    ei, xi = loc[t]
    dates, idx, hi, lo, cl = bars[r.symbol]
    entry = r.entry_price
    k = r.pnl_pct - (r.exit_price / entry - 1.0)

    def fill(fi):
        return (cl[fi] * SLIP / entry - 1.0) + k

    for x_ in (0.08, 0.10, 0.12):
        lvl = entry * (1 - x_)
        hitv = np.nonzero(lo[ei + 1:xi] <= lvl)[0]
        if len(hitv):
            fi = ei + 1 + hitv[0] + 1
            if fi < xi:
                lab = f"stop_{int(x_*100)}"
                sim[lab][t] = fill(fi); simd[lab][t] = dates[fi]
    for nn in (15, 25):
        i = ei + nn
        if i < xi and cl[i] < entry and i + 1 < xi:
            lab = f"tstop_{nn}"
            sim[lab][t] = fill(i + 1); simd[lab][t] = dates[i + 1]
    for arm in (0.08, 0.12):
        armv = np.nonzero(hi[ei + 1:xi] / entry - 1.0 >= arm)[0]
        if len(armv):
            a = ei + 1 + armv[0]
            trg = np.nonzero(lo[a + 1:xi] < entry)[0]
            if len(trg):
                fi = a + 1 + trg[0] + 1
                if fi < xi:
                    lab = f"bev_lock_{int(arm*100)}"
                    sim[lab][t] = fill(fi); simd[lab][t] = dates[fi]

base = T.pnl_pct.to_numpy()
blm = base <= -0.15
for lab in RULES:
    s = sim[lab]
    trig = ~np.isnan(s)
    new = np.where(trig, s, base)
    d = new - base
    wk = trig & (base > 0) & (d < 0)
    bwk = trig & (base >= 0.15) & (d < 0)
    nbl = new[new <= -0.15]
    pre = trig & blm
    print(f"[C7-10] {lab:11s} trig={trig.sum():4d} delta={d.sum():+7.2f}u "
          f"winners_killed n={wk.sum():3d} base={base[wk].sum():+7.2f} sim={s[wk].sum():+7.2f} "
          f"| bigwin n={bwk.sum():3d} base={base[bwk].sum():+7.2f} sim={s[bwk].sum():+7.2f} d={d[bwk].sum():+7.2f} "
          f"| new_bl15 n={len(nbl)} sum={nbl.sum():+7.2f} "
          f"| bl15 preempted {pre.sum()}/{blm.sum()} their base mean={base[pre].mean() if pre.sum() else float('nan'):.4f} "
          f"sim mean={s[pre].mean() if pre.sum() else float('nan'):.4f}")
    # bucket decomposition
    dd = np.where(trig, d, 0.0)
    bks = {"bigwin>=15": base >= 0.15, "win": (base > 0) & (base < 0.15),
           "0..-15": (base <= 0) & (base > -0.15), "bigloss<=-15": blm}
    dec = {kk: round(float(dd[mm].sum()), 2) for kk, mm in bks.items()}
    othr = round(float(dd[~blm].sum()), 2)
    print(f"          decomposition {dec} | non-bigloss total={othr:+.2f}u")

# C8 stop-fill gap severity
s8 = sim["stop_8"][~np.isnan(sim["stop_8"])]
s10n = sim["stop_10"]; s12n = sim["stop_12"]
print(f"[C8] stop_8 fills: <=-12%: {(s8<=-0.12).sum()} <=-15%: {(s8<=-0.15).sum()} <=-20%: {(s8<=-0.20).sum()}")
for lab, want in (("stop_8", None), ("stop_10", None), ("stop_12", None)):
    s = sim[lab]; new = np.where(~np.isnan(s), s, base)
    nbl = new[new <= -0.15]
    print(f"[C8] {lab}: new bl15 n={len(nbl)} sum={nbl.sum():+.2f}u  (base 59 / {base[blm].sum():.2f}u)")

# named kills
def show(sym, ent, lab):
    m = (T.symbol == sym) & (T.entry_date == ent)
    if m.any():
        i = T.index[m][0]
        print(f"[C9] {lab} {sym} {ent}: base={base[i]:+.4f} sim={sim[lab][i]:+.4f} "
              f"simdate={simd[lab][i]} mae={T.mae_pct[i]:.4f} hold={T.holding_days[i]}")
show("CEO", "2021-10-08", "stop_8")
show("STB", "2025-04-04", "stop_8")
show("HVN", "2025-02-18", "stop_8")
show("VGI", "2023-11-17", "tstop_15")

# MSR + TIG bar paths
con = sqlite3.connect(DB)
tr = T[(T.symbol == "MSR") & (T.entry_date == "2025-04-03")].iloc[0]
q = pd.read_sql("select date, low, close from ohlcv where symbol='MSR' and date>='2025-04-03' and date<='2025-04-15' order by date", con)
print(f"[C8] MSR entry={tr.entry_price} base={tr.pnl_pct:.4f} exit={tr.exit_date.date()}")
print("[C8] MSR low/entry-1:", [(d, round(v / tr.entry_price - 1, 4)) for d, v in zip(q.date, q.low)])
print("[C8] MSR close/entry-1:", [(d, round(v / tr.entry_price - 1, 4)) for d, v in zip(q.date, q.close)])
# trough date within trade window
q2 = pd.read_sql("select date, low from ohlcv where symbol='MSR' and date>=? and date<=? order by date",
                 con, params=(tr.entry_date.strftime("%Y-%m-%d"), tr.exit_date.strftime("%Y-%m-%d")))
print(f"[C2] MSR trough date={q2.loc[q2.low.idxmin(),'date']} low_ret={q2.low.min()/tr.entry_price-1:.4f}")
con.close()

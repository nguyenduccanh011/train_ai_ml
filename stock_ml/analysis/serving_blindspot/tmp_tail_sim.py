"""Tail-cutter simulation on the 3,781 closed champion trades.

Rules simulated (all fills = NEXT-BAR CLOSE after breach, engine-style close_next):
  fixed stop -8/-10/-12  : breach when low[t] <= entry_price*(1-X) for t >= entry_idx+1;
                           fill at close[t+1]*(1-slip). Preempts engine exit only if fill
                           date < engine exit_date.
  time-stop 15/25        : at bar entry_idx+N, if close < entry_price -> fill close[t+1].
  breakeven_lock mfe 8/12: once running max(high)/entry-1 >= arm, floor at entry_price;
                           if a LATER low < entry_price -> fill close[t+1].
Sim pnl uses per-trade cost constant k = pnl_pct - (exit_price/entry_price - 1) so the
engine's exact cost treatment carries over; sim exit fill = close*(1-slip_adj) where
slip_adj is calibrated from exit_price vs store close on exit_date.
"""
import sqlite3
import numpy as np
import pandas as pd

AN = r"f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot"
DB = r"C:/Users/DUC CANH PC/Desktop/stock-serving/data/ohlcv.db"

T = pd.read_csv(AN + r"/trades_metrics.csv", parse_dates=["entry_date", "exit_date", "signal_date"])
T = T.reset_index(drop=True)
syms = sorted(T.symbol.unique())
con = sqlite3.connect(DB)
bars = {}
for s in syms:
    df = pd.read_sql("select date, open, high, low, close from ohlcv where symbol=? order by date", con, params=(s,))
    bars[s] = {
        "dates": df["date"].to_numpy(),
        "idx": {d: i for i, d in enumerate(df["date"])},
        "high": df["high"].to_numpy(float),
        "low": df["low"].to_numpy(float),
        "close": df["close"].to_numpy(float),
    }
con.close()
print(f"loaded {len(syms)} symbols")

# ---- calibrate exit fill vs store close on exit_date, and cost constant k ----
ratios, ks = [], []
skipped = 0
rows = []
for r in T.itertuples():
    b = bars[r.symbol]
    ed, xd = r.entry_date.strftime("%Y-%m-%d"), r.exit_date.strftime("%Y-%m-%d")
    if ed not in b["idx"] or xd not in b["idx"]:
        skipped += 1
        rows.append(None)
        continue
    ei, xi = b["idx"][ed], b["idx"][xd]
    rows.append((ei, xi))
    ratios.append(r.exit_price / b["close"][xi])
    ks.append(r.pnl_pct - (r.exit_price / r.entry_price - 1.0))
ratios = np.array(ratios); ks = np.array(ks)
print(f"skipped (missing bars): {skipped}")
print(f"exit_price / store_close(exit_date): mean={ratios.mean():.5f} median={np.median(ratios):.5f} "
      f"p5={np.percentile(ratios,5):.5f} p95={np.percentile(ratios,95):.5f}")
print(f"cost k = pnl - (exit/entry -1): mean={ks.mean():.5f} median={np.median(ks):.5f} std={ks.std():.5f}")
SLIP_ADJ = float(np.median(ratios))   # apply the same fill adjustment to simulated exits

STOPS = [0.08, 0.10, 0.12]
TSTOPS = [15, 25]
BEV = [0.08, 0.12]

results = {}
labels = ([f"stop_{int(x*100)}" for x in STOPS] + [f"tstop_{n}" for n in TSTOPS]
          + [f"bev_lock_{int(a*100)}" for a in BEV])
for lab in labels:
    results[lab] = np.full(len(T), np.nan)   # sim pnl (nan = unchanged)
sim_date = {lab: [None] * len(T) for lab in labels}

for t_i, r in enumerate(T.itertuples()):
    if rows[t_i] is None:
        continue
    ei, xi = rows[t_i]
    b = bars[r.symbol]
    lo, hi, cl = b["low"], b["high"], b["close"]
    entry = r.entry_price
    k = r.pnl_pct - (r.exit_price / entry - 1.0)

    def sim_fill(fill_i):
        p = cl[fill_i] * SLIP_ADJ
        return (p / entry - 1.0) + k

    # fixed stops
    for x in STOPS:
        lab = f"stop_{int(x*100)}"
        lvl = entry * (1.0 - x)
        for i in range(ei + 1, xi):          # breach must fill strictly before engine exit
            if lo[i] <= lvl:
                fi = i + 1
                if fi < xi:                   # fill date strictly before engine exit date
                    results[lab][t_i] = sim_fill(fi)
                    sim_date[lab][t_i] = b["dates"][fi]
                break
    # time stops
    for n in TSTOPS:
        lab = f"tstop_{n}"
        i = ei + n
        if i < xi and cl[i] < entry:
            fi = i + 1
            if fi < xi:
                results[lab][t_i] = sim_fill(fi)
                sim_date[lab][t_i] = b["dates"][fi]
    # breakeven lock
    for arm in BEV:
        lab = f"bev_lock_{int(arm*100)}"
        armed = False
        for i in range(ei + 1, xi):
            if not armed and hi[i] / entry - 1.0 >= arm:
                armed = True
                continue                      # arm bar itself can't trigger
            if armed and lo[i] < entry:
                fi = i + 1
                if fi < xi:
                    results[lab][t_i] = sim_fill(fi)
                    sim_date[lab][t_i] = b["dates"][fi]
                break

base = T.pnl_pct.to_numpy()
base_bl15 = base[base <= -0.15]
print(f"\nBASELINE: sum={base.sum():.2f}u  PF={base[base>0].sum()/-base[base<0].sum():.3f}  "
      f"bl15 n={len(base_bl15)} sum={base_bl15.sum():.2f}u  min={base.min():.4f}")

summary = []
for lab in labels:
    sim = results[lab]
    trig = ~np.isnan(sim)
    new = np.where(trig, sim, base)
    delta = new - base
    winners_killed = trig & (base > 0) & (delta < 0)
    bigwin_killed = trig & (base >= 0.15) & (delta < 0)
    bl15 = new[new <= -0.15]
    summary.append({
        "rule": lab,
        "n_trig": int(trig.sum()),
        "total_delta_u": round(float(delta.sum()), 2),
        "new_sum_u": round(float(new.sum()), 2),
        "new_PF": round(float(new[new > 0].sum() / -new[new < 0].sum()), 3),
        "bl15_n": len(bl15), "bl15_sum": round(float(bl15.sum()), 2),
        "new_min": round(float(new.min()), 4),
        "winners_killed_n": int(winners_killed.sum()),
        "winners_killed_u": round(float(delta[winners_killed].sum()), 2),
        "bigwin_killed_n": int(bigwin_killed.sum()),
        "bigwin_killed_u": round(float(delta[bigwin_killed].sum()), 2),
        "losers_saved_u": round(float(delta[trig & (base <= -0.10)].sum()), 2),
        "avg_stop_fill_pnl": round(float(np.nanmean(sim)), 4),
    })
S = pd.DataFrame(summary)
pd.set_option("display.width", 250)
print("\n", S.to_string(index=False))

# per-rule: what happens to the 59 big losses specifically
print("\nbig-loss (<=-15%) treatment per rule:")
bl_mask = base <= -0.15
for lab in labels:
    sim = results[lab]
    trig = ~np.isnan(sim) & bl_mask
    resid = np.where(~np.isnan(sim), sim, base)[bl_mask]
    print(f"  {lab:12s}: preempted {int(trig.sum())}/59; their new mean pnl "
          f"{np.nanmean(sim[trig]) if trig.sum() else float('nan'):.4f}; residual bl15 sum {resid[resid<=-0.15].sum():.2f}u")

# worst-10 after each rule
for lab in ["stop_8", "stop_10", "tstop_15", "bev_lock_12"]:
    sim = results[lab]
    new = np.where(~np.isnan(sim), sim, base)
    order = np.argsort(new)[:5]
    print(f"\n{lab} new worst5:")
    for i in order:
        print(f"  {T.symbol[i]} {T.entry_date[i].date()} base={base[i]:.3f} new={new[i]:.4f} "
              f"simdate={sim_date[lab][i]}")

# save per-trade sim for verification
out = T[["symbol", "entry_date", "exit_date", "pnl_pct", "mae_pct", "mfe_pct", "holding_days"]].copy()
for lab in labels:
    out[f"sim_{lab}"] = results[lab]
    out[f"simdate_{lab}"] = sim_date[lab]
out.to_csv(AN + r"/tmp_tail_sim_per_trade.csv", index=False)
print("\nwrote tmp_tail_sim_per_trade.csv")

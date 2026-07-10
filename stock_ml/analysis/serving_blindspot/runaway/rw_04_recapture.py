# -*- coding: utf-8 -*-
"""Kiem chung: sau khi limit runaway het han, champion co bat lai song bang core fill ke tiep?"""
import numpy as np
import pandas as pd

BASE = r"f:\PROJECTS\train_ai_ml\stock_ml\analysis\serving_blindspot"
uf = pd.read_csv(f"{BASE}/unfilled_signals.csv")
tr = pd.read_csv(f"{BASE}/trades_raw.csv")
tr["entry_signal_date"] = tr["entry_signal_date"].astype(str)
tr["entry_date"] = tr["entry_date"].astype(str)

u = uf[uf.drop_reason == "unfilled"]
co = u[u.window_end_close > u.signal_close]

core = {s: g.sort_values("entry_date") for s, g in tr.groupby("symbol")}
rows = []
for r in co.itertuples():
    g = core.get(r.symbol)
    if g is None:
        rows.append((np.nan, np.nan, np.nan)); continue
    nxt = g[g.entry_signal_date > r.signal_date]
    if len(nxt) == 0:
        rows.append((np.nan, np.nan, np.nan)); continue
    t0 = nxt.iloc[0]
    lag = (pd.Timestamp(t0.entry_date) - pd.Timestamp(r.signal_date)).days
    rows.append((lag, t0.pnl_pct, t0.entry_price / r.signal_close - 1))
d = pd.DataFrame(rows, columns=["lag_days", "next_pnl", "basis_vs_sigclose"])
have = d.lag_days.notna()
print(f"runaway cohort n={len(co)}; co core fill KE TIEP cung ma: {have.sum()} ({have.mean():.0%})")
print(f"  lag toi fill ke tiep (calendar days): med {d.lag_days.median():.0f}, p25/p75 "
      f"{d.lag_days.quantile(.25):.0f}/{d.lag_days.quantile(.75):.0f}")
w90 = d[d.lag_days <= 90]
print(f"  fill trong <=90d: {len(w90)} ({len(w90)/len(co):.0%}); pnl next-fill: mean "
      f"{w90.next_pnl.mean()*100:+.2f}% (champion pool {tr.pnl_pct.mean()*100:+.2f}%), "
      f"tong {w90.next_pnl.sum():+.1f}u")
print(f"  basis next-fill vs signal_close runaway: med {w90.basis_vs_sigclose.median()*100:+.1f}%")
# blocked premium tu rw_02
sim = pd.read_csv(f"{BASE}/runaway/rw_cohort_sim.csv")
print(f"\nblocked core trades (rw_01, exit champ): n={int(sim.blocked_n.sum())}, "
      f"foregone {sim.blocked_pnl.sum():+.1f}u, mean/trade "
      f"{sim.blocked_pnl.sum()/max(sim.blocked_n.sum(),1)*100:+.2f}% (pool {tr.pnl_pct.mean()*100:+.2f}%)")

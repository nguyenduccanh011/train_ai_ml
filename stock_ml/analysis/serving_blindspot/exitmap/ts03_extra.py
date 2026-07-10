# -*- coding: utf-8 -*-
"""TWOSIDED HEAD SCREEN — supplementary decision variants (verdict hardening).

V1 EXTEND-ONLY: keep the real exit machinery, but at the real decision bar, if
   pred_up > k*pred_dn keep holding; re-check every bar in the +20 post-exit
   extension; exit at first bar where pred_up < k*pred_dn (or ext end).
   Isolates the "sold-then-rallied rescue" direction only (never exits earlier).
V2 ABS-DOWNSIDE: exit early at first in-pos bar where pred_rem_mae (full) is
   below an absolute threshold q (quantile of the fold-train pred distribution
   is unavailable post-hoc -> use fixed grid on predicted downside magnitude).
   Isolates "downside forecast has IC, can it time exits alone?".
Same pnl convention as ts02 (close-fill minus per-trade implied cost).
"""
import numpy as np
import pandas as pd

EM = r"f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/exitmap"

D = pd.read_parquet(EM + "/ts_bars.parquet")
META = pd.read_parquet(EM + "/ts_trades_meta.parquet")
P = pd.read_parquet(EM + "/ts_preds.parquet")
D["date"] = pd.to_datetime(D["date"])
P["date"] = pd.to_datetime(P["date"])
META["entry_date"] = pd.to_datetime(META["entry_date"])

D2 = D.merge(META[["trade_id", "entry_price", "exit_price", "pnl_pct",
                   "exit_reason", "year_exit"]], on="trade_id", how="left")
D2["close_px"] = D2.entry_price * (1.0 + D2.gain)
D2 = D2.merge(P[["trade_id", "date"] + [c for c in P.columns if c.startswith("p_")]],
              on=["trade_id", "date"], how="left")
elig = META[(META.exit_reason != "open") & (META.entry_date >= "2022-01-01")]
TB = {tid: g.sort_values("date").reset_index(drop=True)
      for tid, g in D2[D2.trade_id.isin(elig.trade_id)].groupby("trade_id")}

rows = []
for sname in ["gatepos", "full"]:
    for tid, t in elig.set_index("trade_id").iterrows():
        g = TB.get(tid)
        if g is None or len(g) < 2:
            continue
        up = g[f"p_rem_mfe_{sname}"].to_numpy()
        dn = np.maximum(-g[f"p_rem_mae_{sname}"].to_numpy(), 0.0)
        dec_real = np.where(g.bars_vs_exit.to_numpy() == -1)[0]
        dec_real = int(dec_real[0]) if len(dec_real) else len(g) - 1
        cost = ((g.close_px.iloc[dec_real + 1] if dec_real + 1 < len(g)
                 else t.exit_price) / t.entry_price - 1.0 - t.pnl_pct)

        def pnl_at(jdec):
            jf = min(jdec + 1, len(g) - 1)
            return (g.close_px.iloc[jf] / t.entry_price - 1.0 - cost,
                    int(g.age.iloc[jf]))

        base = dict(set=sname, trade_id=tid, year_exit=int(t.year_exit),
                    pnl_act=t.pnl_pct, hold_act=pnl_at(dec_real)[1])
        # V1 extend-only
        for k in [0.75, 1.0, 1.5, 2.0]:
            j = dec_real
            while j < len(g) - 2:
                if np.isnan(up[j]) or up[j] < k * dn[j]:
                    break
                j += 1
            pnl, hold = pnl_at(j)
            rows.append({**base, "variant": f"extend_k{k}", "pnl_sim": pnl,
                         "hold_sim": hold, "moved": j != dec_real})
        # V2 absolute predicted-downside early exit
        for thr in [0.05, 0.08, 0.12]:
            inpos = g.in_pos.to_numpy()
            cand = np.where(inpos & ~np.isnan(dn) & (dn >= thr))[0]
            j = min(int(cand[0]), dec_real) if len(cand) else dec_real
            pnl, hold = pnl_at(j)
            rows.append({**base, "variant": f"absdn_{thr}", "pnl_sim": pnl,
                         "hold_sim": hold, "moved": j != dec_real})
S = pd.DataFrame(rows)

def table(df, yr):
    s = df[df.year_exit >= yr]
    out = []
    for (sname, var), g in s.groupby(["set", "variant"]):
        out.append(dict(set=sname, variant=var, n=len(g),
                        pnl_act=round(g.pnl_act.sum(), 2),
                        pnl_sim=round(g.pnl_sim.sum(), 2),
                        d_pnl=round(g.pnl_sim.sum() - g.pnl_act.sum(), 2),
                        psd_act=round(g.pnl_act.sum() / g.hold_act.sum(), 5),
                        psd_sim=round(g.pnl_sim.sum() / g.hold_sim.sum(), 5),
                        moved=int(g.moved.sum())))
    return pd.DataFrame(out)

print("=== supplementary variants, exit-year >= 2022 ===")
print(table(S, 2022).to_string(index=False))
print("\n=== supplementary variants, exit-year >= 2024 ===")
print(table(S, 2024).to_string(index=False))
S.to_parquet(EM + "/ts_sim_extra.parquet", index=False)

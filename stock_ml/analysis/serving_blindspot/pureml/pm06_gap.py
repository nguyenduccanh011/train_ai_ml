# -*- coding: utf-8 -*-
"""pureml step 6: phan ra composite + counterfactual gap t1058 vs gb_x08 (seed 42)."""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

SML = Path(r"f:\PROJECTS\train_ai_ml\stock_ml")
sys.path.insert(0, str(SML))
from src.evaluation.scoring import (  # noqa: E402
    calc_sortino, calc_mdd_per_symbol, calc_yearly_consistency, composite_score,
    SCORE_MDD_DIV, SCORE_MDD_POW, SCORE_PNL_W, SCORE_PNL_CAP,
)

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
OUT = r"f:\PROJECTS\train_ai_ml\stock_ml\analysis\serving_blindspot\pureml"

t = pd.read_csv(f"{OUT}\\pm_t1058_s42_trades.csv", parse_dates=["entry_date", "exit_date"])
g = pd.read_csv(r"f:\PROJECTS\train_ai_ml\stock_ml\analysis\serving_blindspot\exitmap\gbx08_s42_trades.csv",
                parse_dates=["entry_date", "exit_date"])

con = psycopg2.connect(**PG)
ns = pd.read_sql("select run_id, n_symbols, composite_score from leaderboard_runs where run_id in "
                 "('template/n2_sx_rr_h10_thr2p5-570ad8d5','template/gb_x08-32a8dfee')", con)
con.close()
print(ns.to_string(index=False))
NSYM = {r.run_id: int(r.n_symbols) for r in ns.itertuples()}


def as_trades(df):
    return [dict(symbol=r.symbol, pnl_pct=float(r.pnl_pct), holding_days=float(r.holding_days),
                 entry_date=str(r.entry_date.date()), exit_date=str(r.exit_date.date()))
            for r in df.itertuples()]


def decompose(df, n_symbols, label):
    tr = as_trades(df)
    n = len(df)
    avg_pnl = df.pnl_pct.mean()
    avg_hold = df.holding_days.mean()
    total_pnl = df.pnl_pct.sum()
    gains = df.pnl_pct[df.pnl_pct > 0].sum(); losses = -df.pnl_pct[df.pnl_pct <= 0].sum()
    pfv = gains / losses if losses > 0 else np.inf
    sortino = calc_sortino(tr)
    mdd = calc_mdd_per_symbol(tr)
    yr = calc_yearly_consistency(tr)
    norm_riskadj = float(np.tanh(sortino / 1.55))
    norm_avg_bar = float(np.tanh((avg_pnl / max(avg_hold, 1.0)) / 0.0015))
    norm_total = float(np.clip((total_pnl / max(n_symbols, 1)) / 1.70, 0.0, SCORE_PNL_CAP))
    norm_pf = float(1.0 - np.exp(-max(pfv - 1.0, 0.0) / 9.0))
    norm_mdd = float(np.clip((max(mdd, 0.0) / SCORE_MDD_DIV) ** SCORE_MDD_POW, 0.0, 1.0))
    norm_yr = float(max(yr - 0.35, 0.0) / 2.0)
    terms = dict(riskadj=0.15 * norm_riskadj, perbar=0.08 * norm_avg_bar,
                 total=SCORE_PNL_W * norm_total, pf=0.18 * norm_pf,
                 mdd=-0.15 * norm_mdd, yr=-0.07 * norm_yr)
    quality = sum(terms.values()) * 1000
    comp = composite_score(dict(trades=n, avg_pnl=avg_pnl, pf=pfv, avg_hold=avg_hold,
                                total_pnl=total_pnl, n_symbols=n_symbols), tr)
    print(f"\n[{label}] n={n} pnl={total_pnl:.1f} sortino={sortino:.2f} mdd_sym={mdd:.3f} yr_cv={yr:.2f}")
    print("  terms x1000:", {k: round(v * 1000, 1) for k, v in terms.items()},
          f"quality={quality:.1f} composite={comp:.1f}")
    return comp


c_t = decompose(t, NSYM.get("template/n2_sx_rr_h10_thr2p5-570ad8d5", 61), "t1058 as-is")
c_g = decompose(g, NSYM.get("template/gb_x08-32a8dfee", 61), "gb_x08 as-is")
ns_t = NSYM.get("template/n2_sx_rr_h10_thr2p5-570ad8d5", 61)

print("\n===== COUNTERFACTUALS on t1058 (first-order) =====")
# (i) force-exit protection proxy: clip moi trade tai -12% (downleg12-like tail cut)
for cap in (-0.12, -0.15):
    t2 = t.copy(); t2["pnl_pct"] = t2.pnl_pct.clip(lower=cap)
    # hold: trades bi cat se ngan hon; xap xi giu nguyen (bao thu)
    c2 = decompose(t2, ns_t, f"t1058 + stoploss clip {cap:.0%}")
    print(f"  -> dComposite = {c2 - c_t:+.1f}")

# (ii) trailing proxy: winners giu >120d ma pnl cuoi < 40% cua max? khong co MFE -> bo qua,
# thay bang: cap hold 90d cho LOSERS (force gate cung co tac dung nay) - khong tinh duoc pnl -> skip.

# (iii) entry regime: bo cac entry trong bear 2022-03-01..2022-11-15 (nonbull mask proxy)
t3 = t[~((t.entry_date >= "2022-03-01") & (t.entry_date <= "2022-11-15"))].copy()
c3 = decompose(t3, ns_t, "t1058 - bear-2022 entries (nonbull entry-gate proxy)")
print(f"  -> dComposite = {c3 - c_t:+.1f}")

# (i)+(iii) ket hop
t4 = t3.copy(); t4["pnl_pct"] = t4.pnl_pct.clip(lower=-0.12)
c4 = decompose(t4, ns_t, "t1058 + clip12 + no-bear-2022-entry")
print(f"  -> dComposite = {c4 - c_t:+.1f}")

# (iv) phan con lai cua gap = entry/label/feature (gb entry ensemble + velocity exit label)
print(f"\nGAP tong (s42, run gap): gb 735.0 - t1058 619.1 = 115.9")
print(f"GAP tong (multi-seed clone): 733.8 - 630.5 = 103.3")
print(f"(i) clip12 dong gop: {c4 - c_t:+.1f} (rieng clip12: xem tren)")
print("phan du sau (i)+(iii) ~ entry ensemble + velocity label + conv fill + trailing:")
print(f"  {c_g - c4:+.1f} diem so voi counterfactual tot nhat")

# exit-date clustering quanh dinh 2022-01 (ngach thang: ML exit bat dinh song)
ex = t[(t.exit_date >= "2021-11-01") & (t.exit_date <= "2022-03-31") & (t.holding_days > 200)]
print(f"\n== runner exits 2021-11..2022-03 (hold>200d): n={len(ex)} pnl_sum={ex.pnl_pct.sum():.1f} "
      f"(= {ex.pnl_pct.sum()/t.pnl_pct.sum():.0%} tong pnl)")

# concurrency / capacity starvation: so vi tri mo theo nam
days = pd.date_range(t.entry_date.min(), t.exit_date.max(), freq="B")
open_cnt = pd.Series(0, index=days)
for r in t.itertuples():
    open_cnt[(open_cnt.index >= r.entry_date) & (open_cnt.index <= r.exit_date)] += 1
print("\n== avg open positions theo nam (t1058):")
print(open_cnt.groupby(open_cnt.index.year).mean().round(1).to_string())
print("PM06_DONE")

# -*- coding: utf-8 -*-
"""FAMILY_CHAMPIONS step 4: phan ra composite + counterfactual gap vs gb_x08 (s42).

Usage: python fc04_gap.py <trades_csv> <label> <n_symbols>
Counterfactuals (first-order, gia dinh nhu pm06: clip khong tinh slippage gap-down,
khong doi hold):
  (i)   clip loser tai -12%/-15% (tail-cut proxy)
  (ii)  bo entry bear 2022-03-01..2022-11-15 (nonbull gate proxy)
  (iii) i+ii
  (iv)  winner-uplift: thay pnl winner bang pnl gb_x08 cung ma ±10d neu gb lon hon
        (proxy 'exit cho winner chay' — chi ap cho lenh trung ma, uoc luong tran fill/exit edge)
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SML = Path(r"f:\PROJECTS\train_ai_ml\stock_ml")
sys.path.insert(0, str(SML))
from src.evaluation.scoring import (  # noqa: E402
    calc_sortino, calc_mdd_per_symbol, calc_yearly_consistency, composite_score,
    SCORE_MDD_DIV, SCORE_MDD_POW, SCORE_PNL_W, SCORE_PNL_CAP,
)

CSV, LABEL, NSYM = sys.argv[1], sys.argv[2], int(sys.argv[3])
t = pd.read_csv(CSV, parse_dates=["entry_date", "exit_date"])
g = pd.read_csv(r"f:\PROJECTS\train_ai_ml\stock_ml\analysis\serving_blindspot\exitmap\gbx08_s42_trades.csv",
                parse_dates=["entry_date", "exit_date"])


def as_trades(df):
    return [dict(symbol=r.symbol, pnl_pct=float(r.pnl_pct), holding_days=float(r.holding_days),
                 entry_date=str(r.entry_date.date()), exit_date=str(r.exit_date.date()))
            for r in df.itertuples()]


def decompose(df, n_symbols, label):
    tr = as_trades(df)
    n = len(df)
    avg_pnl = df.pnl_pct.mean(); avg_hold = df.holding_days.mean(); total_pnl = df.pnl_pct.sum()
    gains = df.pnl_pct[df.pnl_pct > 0].sum(); losses = -df.pnl_pct[df.pnl_pct <= 0].sum()
    pfv = gains / losses if losses > 0 else np.inf
    sortino = calc_sortino(tr); mdd = calc_mdd_per_symbol(tr); yr = calc_yearly_consistency(tr)
    terms = dict(
        riskadj=0.15 * float(np.tanh(sortino / 1.55)),
        perbar=0.08 * float(np.tanh((avg_pnl / max(avg_hold, 1.0)) / 0.0015)),
        total=SCORE_PNL_W * float(np.clip((total_pnl / max(n_symbols, 1)) / 1.70, 0.0, SCORE_PNL_CAP)),
        pf=0.18 * float(1.0 - np.exp(-max(pfv - 1.0, 0.0) / 9.0)),
        mdd=-0.15 * float(np.clip((max(mdd, 0.0) / SCORE_MDD_DIV) ** SCORE_MDD_POW, 0.0, 1.0)),
        yr=-0.07 * float(max(yr - 0.35, 0.0) / 2.0))
    comp = composite_score(dict(trades=n, avg_pnl=avg_pnl, pf=pfv, avg_hold=avg_hold,
                                total_pnl=total_pnl, n_symbols=n_symbols), tr)
    print(f"\n[{label}] n={n} pnl={total_pnl:.1f} sortino={sortino:.2f} mdd_sym={mdd:.3f} yr_cv={yr:.2f}")
    print("  terms x1000:", {k: round(v * 1000, 1) for k, v in terms.items()}, f"composite={comp:.1f}")
    return comp


c0 = decompose(t, NSYM, f"{LABEL} as-is")
decompose(g, 61, "gb_x08 as-is")

print("\n===== COUNTERFACTUALS =====")
for cap in (-0.12, -0.15):
    t2 = t.copy(); t2["pnl_pct"] = t2.pnl_pct.clip(lower=cap)
    c = decompose(t2, NSYM, f"{LABEL} + clip {cap:.0%}")
    print(f"  -> d = {c - c0:+.1f}")

t3 = t[~((t.entry_date >= "2022-03-01") & (t.entry_date <= "2022-11-15"))].copy()
c3 = decompose(t3, NSYM, f"{LABEL} - bear2022 entries")
print(f"  -> d = {c3 - c0:+.1f}")

t4 = t3.copy(); t4["pnl_pct"] = t4.pnl_pct.clip(lower=-0.12)
c4 = decompose(t4, NSYM, f"{LABEL} + clip12 + no-bear2022")
print(f"  -> d = {c4 - c0:+.1f}")

# (iv) winner-uplift: doi voi moi trade, neu gb co trade cung ma entry ±10d pnl lon hon -> lay pnl gb
gj = {s: df for s, df in g.groupby("symbol")}


def uplift(row):
    sub = gj.get(row.symbol)
    if sub is None:
        return row.pnl_pct
    near = sub[abs(sub.entry_date - row.entry_date) <= pd.Timedelta(days=10)]
    if len(near) == 0:
        return row.pnl_pct
    return max(row.pnl_pct, near.pnl_pct.max())


t5 = t.copy(); t5["pnl_pct"] = t5.apply(uplift, axis=1)
c5 = decompose(t5, NSYM, f"{LABEL} + winner-uplift(gb near-entry)")
print(f"  -> d = {c5 - c0:+.1f}")
print("FC04_DONE")

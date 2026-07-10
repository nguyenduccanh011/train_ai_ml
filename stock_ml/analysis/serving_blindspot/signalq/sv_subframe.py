"""SALVAGE >=2022 sub-frame audit (reuse of tmp_prosecutor1.py charge-2 method).
Usage: python sv_subframe.py <variant_csv> <champ_csv> [label]
Prints entries>=cut composites for variant vs champ (cut in 2022/2023/2024) + per-year pnl.
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from stock_ml.src.evaluation.scoring import calc_metrics, calc_mdd_per_symbol, composite_score  # noqa: E402

D = Path(__file__).parent
var = pd.read_csv(sys.argv[1], parse_dates=["entry_date", "exit_date"])
champ = pd.read_csv(sys.argv[2], parse_dates=["entry_date", "exit_date"])
label = sys.argv[3] if len(sys.argv) > 3 else "variant"


def to_trades(df):
    return [dict(symbol=r.symbol, entry_date=str(r.entry_date.date()), pnl_pct=r.pnl_pct,
                 holding_days=r.holding_days) for r in df.itertuples()]


def full_metrics(df, n_sym):
    tr = to_trades(df)
    m = calc_metrics(tr)
    m["n_symbols"] = n_sym
    return m, tr


print(f"== {label} vs champ: full frame ==")
n_full = len(set(var.symbol) | set(champ.symbol))
for nm, df in [(label, var), ("champ", champ)]:
    m, t = full_metrics(df, n_full)
    print(f"  {nm}: tr={m['trades']} pnl={m['total_pnl']:.2f} pf={m['pf']:.3f} "
          f"mdd_sym={calc_mdd_per_symbol(t):.5f} comp={composite_score(m, t)}")

for cut in ["2022-01-01", "2023-01-01", "2024-01-01"]:
    v_sub = var[var.entry_date >= cut]
    c_sub = champ[champ.entry_date >= cut]
    n_sub = len(set(v_sub.symbol) | set(c_sub.symbol))
    m_v, t_v = full_metrics(v_sub, n_sub)
    m_c, t_c = full_metrics(c_sub, n_sub)
    cv, cc = composite_score(m_v, t_v), composite_score(m_c, t_c)
    print(f"entries >= {cut} (n_sym={n_sub}):")
    print(f"  {label:8s}: tr={m_v['trades']:4d} pnl={m_v['total_pnl']:7.2f} pf={m_v['pf']:.3f} "
          f"mdd_sym={calc_mdd_per_symbol(t_v):.5f} comp={cv:7.1f}")
    print(f"  {'champ':8s}: tr={m_c['trades']:4d} pnl={m_c['total_pnl']:7.2f} pf={m_c['pf']:.3f} "
          f"mdd_sym={calc_mdd_per_symbol(t_c):.5f} comp={cc:7.1f}   d_comp={cv-cc:+.1f} "
          f"d_pnl={m_v['total_pnl']-m_c['total_pnl']:+.2f}")

print("\nper-ENTRY-year delta pnl:")
vy = var.groupby(var.entry_date.dt.year).pnl_pct.sum()
cy = champ.groupby(champ.entry_date.dt.year).pnl_pct.sum()
for y in sorted(set(vy.index) | set(cy.index)):
    print(f"  {y}: {label} {vy.get(y, 0):7.2f} champ {cy.get(y, 0):7.2f} d {vy.get(y, 0)-cy.get(y, 0):+6.2f}")

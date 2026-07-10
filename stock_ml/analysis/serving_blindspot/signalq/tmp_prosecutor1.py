"""PROSECUTOR audit of st_dsb60_snr08 (2760) vs champion 2646 — charges 1,2,4,5.
Recomputes everything from the exported s42 trade frames + real composite_score.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

import stock_ml.src.evaluation.scoring as sc  # noqa: E402
from stock_ml.src.evaluation.scoring import calc_metrics, calc_mdd_per_symbol, calc_sortino, \
    calc_yearly_consistency, composite_score  # noqa: E402

D = Path(__file__).parent
stack = pd.read_csv(D / "st_dsb60_snr08_s42_trades.csv", parse_dates=["entry_date", "exit_date"])
champ = pd.read_csv(D / "st_champ2646_s42_trades.csv", parse_dates=["entry_date", "exit_date"])
dsb = pd.read_csv(D / "vx_dsb60_trades.csv", parse_dates=["entry_date", "exit_date"])

N_SYM = 61


def to_trades(df):
    return [dict(symbol=r.symbol, entry_date=str(r.entry_date.date()), pnl_pct=r.pnl_pct,
                 holding_days=r.holding_days) for r in df.itertuples()]


def full_metrics(df, n_sym=N_SYM):
    tr = to_trades(df)
    m = calc_metrics(tr)
    m["n_symbols"] = n_sym
    return m, tr


def decompose(df, n_sym=N_SYM):
    """Replicate composite_score term by term."""
    m, tr = full_metrics(df, n_sym)
    sortino = calc_sortino(tr)
    norm_riskadj = float(np.tanh(sortino / 1.55))
    mdd_sym = calc_mdd_per_symbol(tr)
    yr_cv = calc_yearly_consistency(tr)
    avg_per_bar = m["avg_pnl"] / max(m["avg_hold"], 1.0)
    norm_avg_bar = float(np.tanh(avg_per_bar / 0.0015))
    total_per_sym = m["total_pnl"] / max(n_sym, 1)
    norm_total = float(np.clip(total_per_sym / 1.70, 0.0, sc.SCORE_PNL_CAP))
    norm_pf = float(1.0 - np.exp(-max(m["pf"] - 1.0, 0.0) / 9.0))
    norm_mdd = float(np.clip((max(mdd_sym, 0.0) / sc.SCORE_MDD_DIV) ** sc.SCORE_MDD_POW, 0.0, 1.0))
    norm_yr = float(max(yr_cv - 0.35, 0.0) / 2.0)
    n_trades = m["trades"]
    k = (600.0 / 60.0) * n_sym
    confidence = 1.0 - np.exp(-n_trades / k)
    conf_mult = 0.32 + 0.68 * confidence
    terms = {
        "riskadj(0.15)": 0.15 * norm_riskadj,
        "avg_bar(0.08)": 0.08 * norm_avg_bar,
        "total(0.45)": sc.SCORE_PNL_W * norm_total,
        "pf(0.18)": 0.18 * norm_pf,
        "mdd(-0.15)": -0.15 * np.clip(norm_mdd, 0, 1),
        "yr(-0.07)": -0.07 * np.clip(norm_yr, 0, 1),
    }
    quality = sum(terms.values()) * 1000
    comp = round(quality * conf_mult, 1)
    real = composite_score(m, tr)
    return dict(comp=comp, real=real, conf_mult=conf_mult, quality=quality, terms=terms,
                sortino=sortino, mdd_sym=mdd_sym, yr_cv=yr_cv, metrics=m)


print("=" * 70)
print("CHARGE 1: composite reproduction + decomposition + mdd concentration")
print("=" * 70)
ds, dc = decompose(stack), decompose(champ)
for nm, d in [("stack", ds), ("champ", dc)]:
    print(f"{nm}: composite_real={d['real']} (leaderboard {'746.3' if nm=='stack' else '729.6'}) "
          f"quality={d['quality']:.1f} conf_mult={d['conf_mult']:.4f} "
          f"sortino={d['sortino']:.3f} mdd_sym={d['mdd_sym']:.5f} yr_cv={d['yr_cv']:.3f}")
print("\nterm contributions to composite (term*1000*conf_mult) and delta stack-champ:")
for k in ds["terms"]:
    s_c = ds["terms"][k] * 1000 * ds["conf_mult"]
    c_c = dc["terms"][k] * 1000 * dc["conf_mult"]
    print(f"  {k:15s} stack {s_c:8.2f}  champ {c_c:8.2f}  d {s_c - c_c:+7.2f}")
qs = ds["quality"] * ds["conf_mult"]
qc = dc["quality"] * dc["conf_mult"]
# conf_mult effect isolated: champ quality at stack conf
conf_effect = dc["quality"] * (ds["conf_mult"] - dc["conf_mult"])
print(f"  conf_mult uplift on champ quality: {conf_effect:+.2f}")
print(f"  total delta: {qs - qc:+.2f}")

# per-symbol mdd concentration
def per_sym_mdd(df):
    out = {}
    for sym, g in df.groupby("symbol"):
        g = g.sort_values("entry_date")
        eq = g.pnl_pct.cumsum()
        out[sym] = float((eq.cummax() - eq).max())
    return pd.Series(out)


ms, mc = per_sym_mdd(stack), per_sym_mdd(champ)
dm = (ms - mc.reindex(ms.index).fillna(0)).sort_values(ascending=False)
print(f"\nper-symbol mdd: stack mean={ms.mean():.5f} champ mean={mc.mean():.5f} d={ms.mean()-mc.mean():+.5f}")
print("top 10 symbol mdd increases (stack-champ):")
for sym, v in dm.head(10).items():
    print(f"  {sym:6s} champ {mc.get(sym, 0):.4f} -> stack {ms[sym]:.4f}  d {v:+.4f}")
top3 = dm.head(3).index
ms_ex = ms.drop(top3)
mc_ex = mc.drop(top3, errors="ignore")
print(f"excluding top-3 delta symbols: stack mean={ms_ex.mean():.5f} champ mean={mc_ex.mean():.5f} "
      f"(delta {ms_ex.mean()-mc_ex.mean():+.5f})")
n_worse = (dm > 0.01).sum()
print(f"symbols with mdd increase >1pp: {n_worse}/{len(dm)}; median delta {dm.median():+.5f}")

print("\nSTRESS: harsher mdd pricing")
for pow_, div_ in [(1.5, 0.40), (2.0, 0.40), (1.5, 0.30), (2.0, 0.30)]:
    old_p, old_d = sc.SCORE_MDD_POW, sc.SCORE_MDD_DIV
    sc.SCORE_MDD_POW, sc.SCORE_MDD_DIV = pow_, div_
    m_s, t_s = full_metrics(stack)
    m_c, t_c = full_metrics(champ)
    cs, cc = composite_score(m_s, t_s), composite_score(m_c, t_c)
    sc.SCORE_MDD_POW, sc.SCORE_MDD_DIV = old_p, old_d
    print(f"  POW={pow_} DIV={div_}: stack {cs:7.1f}  champ {cc:7.1f}  d {cs-cc:+6.1f}")

print()
print("=" * 70)
print("CHARGE 2: regime decay — modern-regime subsets + delta-trade anatomy")
print("=" * 70)
for cut in ["2022-01-01", "2023-01-01", "2024-01-01"]:
    s_sub = stack[stack.entry_date >= cut]
    c_sub = champ[champ.entry_date >= cut]
    n_sub = len(set(s_sub.symbol) | set(c_sub.symbol))
    m_s, t_s = full_metrics(s_sub, n_sub)
    m_c, t_c = full_metrics(c_sub, n_sub)
    cs, cc = composite_score(m_s, t_s), composite_score(m_c, t_c)
    print(f"entries >= {cut} (n_sym={n_sub}):")
    print(f"  stack: tr={m_s['trades']:4d} pnl={m_s['total_pnl']:7.2f} pf={m_s['pf']:.3f} "
          f"mdd_sym={calc_mdd_per_symbol(t_s):.5f} comp={cs:7.1f}")
    print(f"  champ: tr={m_c['trades']:4d} pnl={m_c['total_pnl']:7.2f} pf={m_c['pf']:.3f} "
          f"mdd_sym={calc_mdd_per_symbol(t_c):.5f} comp={cc:7.1f}   d_comp={cs-cc:+.1f} "
          f"d_pnl={m_s['total_pnl']-m_c['total_pnl']:+.2f}")

print("\nper-ENTRY-year delta pnl (verify claim):")
sy = stack.groupby(stack.entry_date.dt.year).pnl_pct.sum()
cy = champ.groupby(champ.entry_date.dt.year).pnl_pct.sum()
for y in sorted(set(sy.index) | set(cy.index)):
    print(f"  {y}: stack {sy.get(y,0):7.2f} champ {cy.get(y,0):7.2f} d {sy.get(y,0)-cy.get(y,0):+6.2f}")

# delta-trade anatomy by (symbol, entry_date)
stack["k"] = stack.symbol + "|" + stack.entry_date.astype(str)
champ["k"] = champ.symbol + "|" + champ.entry_date.astype(str)
dsb["k"] = dsb.symbol + "|" + dsb.entry_date.astype(str)
sk, ck = set(stack.k), set(champ.k)
added = stack[stack.k.isin(sk - ck)]
removed = champ[champ.k.isin(ck - sk)]
common = sk & ck
mg = stack[stack.k.isin(common)].merge(champ[champ.k.isin(common)], on="k", suffixes=("_s", "_c"))
changed = mg[mg.exit_date_s != mg.exit_date_c]
print(f"\ndelta trades: added={len(added)} removed={len(removed)} common={len(common)} "
      f"common-with-changed-exit={len(changed)}")
print(f"  added pnl={added.pnl_pct.sum():+.2f}  removed champ-pnl={removed.pnl_pct.sum():+.2f}  "
      f"changed-exit d_pnl={(changed.pnl_pct_s - changed.pnl_pct_c).sum():+.2f}")

for y0, y1 in [(2024, 2026), (2022, 2023)]:
    a = added[(added.entry_date.dt.year >= y0) & (added.entry_date.dt.year <= y1)]
    r = removed[(removed.entry_date.dt.year >= y0) & (removed.entry_date.dt.year <= y1)]
    ch = changed[(changed.entry_date_s.dt.year >= y0) & (changed.entry_date_s.dt.year <= y1)]
    print(f"\nentries {y0}-{y1}: added {len(a)} (pnl {a.pnl_pct.sum():+.2f}), "
          f"removed {len(r)} (champ pnl {r.pnl_pct.sum():+.2f}), "
          f"changed-exit {len(ch)} (d_pnl {(ch.pnl_pct_s - ch.pnl_pct_c).sum():+.2f})")
    print(f"  => net lever pnl {a.pnl_pct.sum() - r.pnl_pct.sum() + (ch.pnl_pct_s - ch.pnl_pct_c).sum():+.2f}")

print("\nbig-loss (<=-10%) counts by entry year:")
for nm, df in [("stack", stack), ("champ", champ)]:
    bl = df[df.pnl_pct <= -0.10]
    print(f"  {nm}: total {len(bl)}  by yr {bl.groupby(bl.entry_date.dt.year).size().to_dict()}")

print("\n2026-entry big losses (stack) anatomy:")
bl26 = stack[(stack.pnl_pct <= -0.10) & (stack.entry_date.dt.year == 2026)]
champ_by_k = champ.set_index("k")
dsb_by_k = dsb.set_index("k")
for r in bl26.itertuples():
    in_c = r.k in champ_by_k.index
    in_d = r.k in dsb_by_k.index
    cinfo = ""
    if in_c:
        c = champ_by_k.loc[r.k]
        cinfo = f"champ: exit {c.exit_date.date()} pnl {c.pnl_pct:+.3f} ({c.exit_reason})"
    else:
        cinfo = "NOT in champ (new entry)"
    dinfo = ""
    if in_d:
        d_ = dsb_by_k.loc[r.k]
        dinfo = f" | dsb60-only: exit {d_.exit_date.date()} pnl {d_.pnl_pct:+.3f} ({d_.exit_reason})"
    else:
        dinfo = " | NOT in dsb60-only"
    print(f"  {r.symbol} {r.entry_date.date()} -> {r.exit_date.date()} {r.pnl_pct:+.3f} ({r.exit_reason}) hold={r.holding_days:.0f} | {cinfo}{dinfo}")

print("\n2022-entry big losses added vs champ:")
bl22s = stack[(stack.pnl_pct <= -0.10) & (stack.entry_date.dt.year == 2022)]
bl22c = champ[(champ.pnl_pct <= -0.10) & (champ.entry_date.dt.year == 2022)]
new22 = bl22s[~bl22s.k.isin(set(bl22c.k))]
print(f"  stack 2022 big losses {len(bl22s)}, champ {len(bl22c)}, new-in-stack {len(new22)}")
for r in new22.itertuples():
    in_c = r.k in champ_by_k.index
    cinfo = (f"champ: exit {champ_by_k.loc[r.k].exit_date.date()} pnl {champ_by_k.loc[r.k].pnl_pct:+.3f}"
             if in_c else "NOT in champ (new entry)")
    print(f"  {r.symbol} {r.entry_date.date()} -> {r.exit_date.date()} {r.pnl_pct:+.3f} hold={r.holding_days:.0f} | {cinfo}")

print()
print("=" * 70)
print("CHARGE 4: guardrails on stack s42 frame")
print("=" * 70)
for nm, df in [("stack", stack), ("champ", champ)]:
    bw = df[df.pnl_pct >= 0.10]
    tot_pos = df[df.pnl_pct > 0].pnl_pct.sum()
    h20 = df[df.holding_days > 20]
    wr20 = (h20.pnl_pct > 0).mean() * 100
    net = df.groupby("symbol").pnl_pct.sum()
    print(f"{nm}: bigwin-rate={len(bw)/len(df)*100:.1f}% (n={len(bw)}) bigwin pnl={bw.pnl_pct.sum():.1f} "
          f"({bw.pnl_pct.sum()/df.pnl_pct.sum()*100:.1f}% of net pnl) | hold>20 n={len(h20)} WR={wr20:.1f}% | "
          f"symbols net-positive {(net>0).sum()}/{len(net)} = {(net>0).mean()*100:.1f}%")
neg_syms = stack.groupby("symbol").pnl_pct.sum()
print("net-negative symbols (stack):", {k: round(v, 3) for k, v in neg_syms[neg_syms <= 0].items()})
print(f"cohort shrink check: removed trades total {len(removed)}, by yr "
      f"{removed.groupby(removed.entry_date.dt.year).size().to_dict()}; removed pnl {removed.pnl_pct.sum():+.2f}")

# washout gate: recompute market-drop dates (zscore w5 lb60 thr -1.75) from duckdb closes
import duckdb  # noqa: E402

uni = sorted(set(stack.symbol) | set(champ.symbol))
con = duckdb.connect(str(REPO / "market_data" / "market.duckdb"), read_only=True)
px = con.execute(
    "SELECT symbol, date, close FROM ohlcv WHERE symbol IN ({}) AND date >= '2019-06-01'".format(
        ",".join("'" + s + "'" for s in uni))).df()
con.close()
piv = px.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
rets = piv.pct_change()
mret = rets.mean(axis=1)
roll = mret.rolling(5).sum()
mu = roll.rolling(60).mean()
sd = roll.rolling(60).std()
z = (roll - mu) / (sd + 1e-9)
drop_dates = pd.DatetimeIndex(z.index[z <= -1.75])
print(f"\nwashout (market-drop z<=-1.75) days computed: {len(drop_dates)} "
      f"({drop_dates.min().date()}..{drop_dates.max().date()})")
by_yr = pd.Series(1, index=drop_dates).groupby(drop_dates.year).size()
print("  drop days per year:", by_yr.to_dict())

def held_through_drops(df, label):
    """trades that stay OPEN >=3 drop days after a drop day occurs during the hold"""
    cnt = 0
    rows = []
    for r in df.itertuples():
        dd = drop_dates[(drop_dates >= r.entry_date) & (drop_dates <= r.exit_date)]
        # gate fires on drop day -> exit should come within a few bars; count trades
        # still held >=5 trading days after their FIRST in-trade drop day
        if len(dd) > 0:
            first = dd[0]
            bars_after = ((piv.index > first) & (piv.index <= r.exit_date)).sum()
            if bars_after >= 5:
                cnt += 1
                rows.append((r.symbol, str(r.entry_date.date()), str(r.exit_date.date()),
                             round(r.pnl_pct, 3), str(first.date()), int(bars_after)))
    print(f"{label}: trades holding >=5 bars past an in-trade washout day: {cnt}")
    for row in rows[:15]:
        print("   ", row)
    return cnt


held_through_drops(stack, "stack")
held_through_drops(champ, "champ")

print()
print("=" * 70)
print("CHARGE 5: shared mechanism / exit-mix / open trades / snr marginal")
print("=" * 70)
print("exit_reason mix (stack vs champ vs dsb60-only):")
mix = pd.DataFrame({"stack": stack.exit_reason.value_counts(),
                    "champ": champ.exit_reason.value_counts(),
                    "dsb60": dsb.exit_reason.value_counts()}).fillna(0).astype(int)
print(mix.to_string())
print("\nopen trades MTM:")
for nm, df in [("stack", stack), ("champ", champ)]:
    op = df[df.exit_reason == "open"]
    print(f"  {nm}: {len(op)} open, pnl {op.pnl_pct.sum():+.3f} ->",
          [(r.symbol, str(r.entry_date.date()), round(r.pnl_pct, 3)) for r in op.itertuples()])
    # composite without open trades
    cl = df[df.exit_reason != "open"]
    m, t = full_metrics(cl)
    print(f"     composite closed-only: {composite_score(m, t)}")

# snr marginal on the dsb60 base (s42): stack vs dsb60-only
dk = set(dsb.k)
sk2 = set(stack.k)
mg2 = stack.merge(dsb, on="k", suffixes=("_st", "_db"))
chg2 = mg2[mg2.exit_date_st != mg2.exit_date_db]
print(f"\nsnr marginal (stack vs dsb60-only, s42): added={len(sk2-dk)} removed={len(dk-sk2)} "
      f"changed-exit={len(chg2)} d_pnl_changed={(chg2.pnl_pct_st-chg2.pnl_pct_db).sum():+.2f}")
add2 = stack[stack.k.isin(sk2 - dk)]
rem2 = dsb[dsb.k.isin(dk - sk2)]
print(f"  added pnl {add2.pnl_pct.sum():+.2f} removed(dsb-side) pnl {rem2.pnl_pct.sum():+.2f} "
      f"=> net snr pnl on stack {add2.pnl_pct.sum()-rem2.pnl_pct.sum()+(chg2.pnl_pct_st-chg2.pnl_pct_db).sum():+.2f}")
print(f"  extended by snr: {len(chg2[chg2.exit_date_st > chg2.exit_date_db])}, "
      f"shortened {len(chg2[chg2.exit_date_st < chg2.exit_date_db])}")
if len(chg2):
    ext = chg2[chg2.exit_date_st > chg2.exit_date_db]
    print(f"  extended trades: peak-gain proxy — dsb-side pnl at exit min {ext.pnl_pct_db.min():+.3f}; "
          f"d_pnl sum {(ext.pnl_pct_st-ext.pnl_pct_db).sum():+.2f}")
print("\nDONE")

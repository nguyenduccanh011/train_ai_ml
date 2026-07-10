"""Gap decomposition: champion 2646 vs candidate 2669 (pyramid), seed 42.

1:1 trade match, add-unit PnL split, composite term decomposition,
pnl-per-unit / per-unit-day fairness, yearly add-delta.
Run from f:/PROJECTS/train_ai_ml.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "f:/PROJECTS/train_ai_ml")
from stock_ml.src.evaluation.scoring import (
    composite_score, calc_metrics, calc_sortino, calc_mdd_per_symbol,
    calc_yearly_consistency, SCORE_PNL_W, SCORE_PNL_CAP, SCORE_MDD_DIV, SCORE_MDD_POW,
)

BASE = "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/week0/best_trades"
champ = pd.read_csv(f"{BASE}/trades_n2_2643_wavestruct_la05_lamp02.csv")
cand = pd.read_csv(f"{BASE}/trades_w0_pyr_u10_r02.csv")

KEY = ["symbol", "entry_date", "exit_date"]
print("=== 1. TRADE MATCH ===")
print(f"champ rows={len(champ)}  cand rows={len(cand)}")
m = champ.merge(cand, on=KEY, suffixes=("_ch", "_ca"), how="outer", indicator=True)
print("merge indicator:", m["_merge"].value_counts().to_dict())
both = m[m["_merge"] == "both"]
same_entry_px = np.allclose(both["entry_price_ch"], both["entry_price_ca"])
same_exit_px = np.allclose(both["exit_price_ch"], both["exit_price_ca"])
same_hold = (both["holding_days_ch"] == both["holding_days_ca"]).all()
same_reason = (both["exit_reason_ch"] == both["exit_reason_ca"]).all()
print(f"identical entry_price={same_entry_px} exit_price={same_exit_px} holding_days={same_hold} exit_reason={same_reason}")

both = both.copy()
both["delta"] = both["pnl_pct_ca"] - both["pnl_pct_ch"]
adds = both[both["weight_ca"] > 1.0]
noadds = both[both["weight_ca"] <= 1.0]
print(f"\nn trades with add (weight_ca>1): {len(adds)}   without: {len(noadds)}")
print(f"max |delta| among no-add trades: {noadds['delta'].abs().max():.2e}  (should be ~0)")

d = adds["delta"]
pos, neg = d[d > 0], d[d < 0]
print(f"\nsum delta (total add-unit net pnl) = {d.sum():.4f}u")
print(f"winning adds: n={len(pos)}  sum=+{pos.sum():.4f}u  mean=+{pos.mean():.4f}")
print(f"losing  adds: n={len(neg)}  sum={neg.sum():.4f}u  mean={neg.mean():.4f}")
print(f"zero-delta adds: {len(d) - len(pos) - len(neg)}")
bw = adds.loc[d.idxmax()]
bl = adds.loc[d.idxmin()]
print(f"biggest winning add: {bw['symbol']} entry {bw['entry_date']} exit {bw['exit_date']} add_net=+{bw['delta']:.4f} (base pnl {bw['pnl_pct_ch']:.4f})")
print(f"biggest losing  add: {bl['symbol']} entry {bl['entry_date']} exit {bl['exit_date']} add_net={bl['delta']:.4f} (base pnl {bl['pnl_pct_ch']:.4f})")
# add trigger sanity: adds should have surged >=2% by bar 3 -> mostly winners
print(f"add-cohort base-trade winrate: {(adds['pnl_pct_ch']>0).mean()*100:.1f}%  vs no-add cohort: {(noadds['pnl_pct_ch']>0).mean()*100:.1f}%")
print(f"add-unit winrate (delta>0): {(d>0).mean()*100:.1f}%")

# ---- 2. composite decomposition -------------------------------------------
print("\n=== 2. COMPOSITE DECOMPOSITION ===")
N_SYMBOLS = 61  # leaderboard_runs.n_symbols for both configs (verified, seed-123 rows)

def to_trades(df):
    return [
        {"symbol": r.symbol, "entry_date": r.entry_date, "pnl_pct": r.pnl_pct,
         "holding_days": r.holding_days}
        for r in df.itertuples()
    ]

def terms(df, label):
    tr = to_trades(df)
    met = calc_metrics(tr)
    met["n_symbols"] = N_SYMBOLS
    sortino = calc_sortino(tr)
    mdd = calc_mdd_per_symbol(tr)
    yr = calc_yearly_consistency(tr)
    avg_pnl, pf, avg_hold = met["avg_pnl"], met["pf"], met["avg_hold"]
    total = float(df["pnl_pct"].sum())
    norm_riskadj = float(np.tanh(sortino / 1.55))
    norm_avg_bar = float(np.tanh((avg_pnl / max(avg_hold, 1.0)) / 0.0015))
    norm_total = float(np.clip((total / N_SYMBOLS) / 1.70, 0.0, SCORE_PNL_CAP))
    norm_pf = float(1.0 - np.exp(-max(pf - 1.0, 0.0) / 9.0))
    norm_mdd = float(np.clip((max(mdd, 0.0) / SCORE_MDD_DIV) ** SCORE_MDD_POW, 0.0, 1.0))
    norm_yr = float(max(yr - 0.35, 0.0) / 2.0)
    n = len(df)
    k = 10.0 * N_SYMBOLS
    conf = 1.0 - np.exp(-n / k)
    conf_mult = 0.32 + 0.68 * conf
    t = {
        "riskadj(0.15)": 0.15 * norm_riskadj,
        "avg_bar(0.08)": 0.08 * norm_avg_bar,
        f"total_pnl({SCORE_PNL_W})": SCORE_PNL_W * norm_total,
        "pf(0.18)": 0.18 * norm_pf,
        "mdd(-0.15)": -0.15 * np.clip(norm_mdd, 0, 1),
        "yr(-0.07)": -0.07 * np.clip(norm_yr, 0, 1),
    }
    quality = sum(t.values()) * 1000
    score = round(quality * conf_mult, 1)
    official = composite_score(met, tr)
    print(f"\n[{label}] total_pnl={total:.2f} sortino={sortino:.4f} mdd_sym={mdd:.4f} pf={pf:.3f} "
          f"avg_pnl={avg_pnl:.4f} avg_hold={avg_hold:.2f} yr_cv={yr:.4f}")
    print(f"  conf_mult={conf_mult:.6f}  quality={quality:.2f}  score(manual)={score}  score(official fn)={official}")
    for k2, v in t.items():
        print(f"  {k2:>16}: norm-weighted {v:+.6f}  -> pts {v*1000*conf_mult:+.2f}")
    return t, conf_mult, score, total, mdd

t_ch, cm_ch, s_ch, tot_ch, mdd_ch = terms(champ, "CHAMPION 2646")
t_ca, cm_ca, s_ca, tot_ca, mdd_ca = terms(cand, "CANDIDATE 2669")
print(f"\nDelta composite = {s_ca - s_ch:+.1f}  ({s_ch} -> {s_ca})")
print("Per-term contribution to delta (pts, conf_mult identical since same n_trades):")
for k2 in t_ch:
    dpts = (t_ca[k2] - t_ch[k2]) * 1000 * cm_ca
    print(f"  {k2:>16}: {dpts:+.2f} pts")

# ---- 3. per-unit fairness ---------------------------------------------------
print("\n=== 3. PNL PER UNIT DEPLOYED ===")
n_tr = len(champ)
n_add = len(adds)
units_ch = n_tr * 1.0
units_ca = n_tr * 1.0 + n_add * 1.0  # pyramid_add_units = 1.0
ppu_ch = tot_ch / units_ch
ppu_ca = tot_ca / units_ca
print(f"champion: {tot_ch:.2f}u / {units_ch:.0f} unit-entries = {ppu_ch:.5f} u/unit")
print(f"candidate: {tot_ca:.2f}u / {units_ca:.0f} unit-entries = {ppu_ca:.5f} u/unit  ({(ppu_ca/ppu_ch-1)*100:+.1f}%)")
print(f"add-unit-only pnl per add unit: {d.sum():.2f}/{n_add} = {d.sum()/n_add:.5f} u/unit")

# capital-days (unit-days): base unit deployed holding_days bars; add unit enters bar+3
K_BARS = 3
ud_base = champ["holding_days"].sum()
ud_add = (adds["holding_days_ca"] - K_BARS).clip(lower=0).sum()
print(f"\nunit-days champion = {ud_base}   candidate = {ud_base + ud_add} (base {ud_base} + adds {ud_add})")
print(f"pnl per unit-day: champion {tot_ch/ud_base*1000:.4f}e-3   candidate {tot_ca/(ud_base+ud_add)*1000:.4f}e-3  "
      f"({(tot_ca/(ud_base+ud_add))/(tot_ch/ud_base)-1:+.1%})")
print(f"add-unit-only pnl per unit-day: {d.sum()/ud_add*1000:.4f}e-3")

# ---- 4. yearly add-delta ----------------------------------------------------
print("\n=== 4. YEARLY DELTA OF ADD COMPONENT (by entry year) ===")
both["year"] = both["entry_date"].str[:4]
g = both.groupby("year").agg(
    delta_sum=("delta", "sum"),
    n_adds=("weight_ca", lambda w: int((w > 1).sum())),
    n_trades=("delta", "size"),
    champ_pnl=("pnl_pct_ch", "sum"),
)
g["delta_pos"] = both[both["delta"] > 0].groupby("year")["delta"].sum()
g["delta_neg"] = both[both["delta"] < 0].groupby("year")["delta"].sum()
g = g.fillna(0.0)
print(g.round(3).to_string())
tot_2021_2020 = g.loc[g.index.isin(["2020", "2021"]), "delta_sum"].sum()
print(f"\n2020+2021 share of add pnl: {tot_2021_2020:.2f}/{d.sum():.2f} = {tot_2021_2020/d.sum()*100:.1f}%")

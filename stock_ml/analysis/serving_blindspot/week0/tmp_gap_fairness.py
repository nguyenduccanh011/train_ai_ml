# tmp_gap_fairness.py — adversarial fairness audit of w0_pyr_u10_r02 vs champion 2646 (seed 42)
# Quantifies: (1) equal-weight per-unit metrics, (2) capital deployment ratio,
# (3) capital-normalized PnL/MDD, (5) fill-convention (close_same vs close_next) impact on adds.
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path("f:/PROJECTS/train_ai_ml")
BD = REPO / "stock_ml/analysis/serving_blindspot/week0/best_trades"
sys.path.insert(0, str(REPO))
from stock_ml.src.evaluation.scoring import (  # noqa: E402
    composite_score, calc_mdd_per_symbol, calc_sortino, calc_yearly_consistency,
)

SLIP = 0.0015
RTC = 2 * 0.0015 + 0.001  # 0.004
N_SYM = 61

champ = pd.read_csv(BD / "trades_n2_2643_wavestruct_la05_lamp02.csv", parse_dates=["entry_date", "exit_date"])
cand = pd.read_csv(BD / "trades_w0_pyr_u10_r02.csv", parse_dates=["entry_date", "exit_date"])
print(f"champ trades={len(champ)}  cand trades={len(cand)}")

m = champ.merge(cand, on=["symbol", "entry_date"], suffixes=("_ch", "_ca"), how="outer", indicator=True)
print("merge mismatch rows:", (m["_merge"] != "both").sum())
print("exit_date mismatches:", (m["exit_date_ch"] != m["exit_date_ca"]).sum())
w1 = m[m["weight_ca"] == 1.0]
w2 = m[m["weight_ca"] > 1.0]
print(f"weight==1: {len(w1)}  weight==2: {len(w2)}")
print("max |pnl diff| on weight-1 rows:", (w1["pnl_pct_ch"] - w1["pnl_pct_ca"]).abs().max())

# ---- decompose adds (a=1.0 -> add_net = cand_pnl - champ_pnl) ----
m["add_net"] = np.where(m["weight_ca"] > 1.0, m["pnl_pct_ca"] - m["pnl_pct_ch"], np.nan)
adds = m.loc[m["weight_ca"] > 1.0].copy()
print(f"\n=== ADD LEG (n={len(adds)}) ===")
print(f"add total pnl = {adds['add_net'].sum():.3f}u  (candidate total {cand['pnl_pct'].sum():.3f} - champ {champ['pnl_pct'].sum():.3f})")
print(f"add WR = {(adds['add_net']>0).mean():.4f}  mean = {adds['add_net'].mean():.5f}  median = {adds['add_net'].median():.5f}")
print(f"base leg of pyramided trades: WR={(adds['pnl_pct_ch']>0).mean():.4f} mean={adds['pnl_pct_ch'].mean():.5f}")

def stats(pnls, label):
    pnls = np.asarray(pnls, float)
    wr = (pnls > 0).mean()
    gp = pnls[pnls > 0].sum(); gl = -pnls[pnls < 0].sum()
    pf = gp / gl if gl > 0 else np.inf
    tr = [{"pnl_pct": float(p)} for p in pnls]
    so = calc_sortino(tr)
    print(f"{label:34s} n={len(pnls):5d} total={pnls.sum():8.2f} mean={pnls.mean():.5f} WR={wr:.4f} PF={pf:.3f} Sortino={so:.4f}")
    return dict(n=len(pnls), total=pnls.sum(), mean=pnls.mean(), wr=wr, pf=pf, sortino=so)

print("\n=== FRAME COMPARISON (per-trade vs per-unit) ===")
s_ch = stats(champ["pnl_pct"], "champion per-trade")
s_ca = stats(cand["pnl_pct"], "candidate per-TRADE (as scored)")
units = np.concatenate([champ["pnl_pct"].values, adds["add_net"].values])
s_eq = stats(units, "candidate per-UNIT (equal-weight)")

# ---- load per-symbol trading calendar + closes from signals csv ----
sig = pd.read_csv(BD / "signals_w0_pyr_u10_r02.csv", usecols=["symbol", "date", "close"], parse_dates=["date"])
cal = {s: g.sort_values("date").reset_index(drop=True) for s, g in sig.groupby("symbol")}
idx_of = {s: pd.Series(np.arange(len(g)), index=g["date"].values) for s, g in cal.items()}

# per trade: bar indices
def bar_ix(sym, d):
    return int(idx_of[sym].loc[np.datetime64(d)])

rows = []
for _, r in m.iterrows():
    sym = r["symbol"]
    ei = bar_ix(sym, r["entry_date"]); xi = bar_ix(sym, r["exit_date_ca"])
    rows.append((ei, xi))
m["ei"], m["xi"] = zip(*rows)

# ---- (5) causality / fill-convention check on adds ----
print("\n=== ADD FILL CHECK (trigger bar = ei+3, fill at SAME bar close; base entries fill close_NEXT) ===")
viol = 0; recon_err = []
alt_net = []  # counterfactual: add fills at close of ei+4 (engine's own close_next convention)
lost_adds = 0
for _, r in m[m["weight_ca"] > 1.0].iterrows():
    g = cal[r["symbol"]]
    ei, xi = int(r["ei"]), int(r["xi"])
    ab = ei + 3
    c3 = float(g["close"].iloc[ab])
    trig = c3 / r["entry_price_ch"] - 1.0
    if trig < 0.02 - 1e-12:
        viol += 1
    addfill = c3 * (1 + SLIP)
    recon = r["exit_price_ca"] / addfill - 1.0 - RTC
    recon_err.append(abs(recon - r["add_net"]))
    if ab + 1 < xi:  # next-bar fill still strictly before exit bar
        c4 = float(g["close"].iloc[ab + 1])
        alt_net.append(r["exit_price_ca"] / (c4 * (1 + SLIP)) - 1.0 - RTC)
    else:
        lost_adds += 1  # under close_next the add would fill AT the exit bar -> no add
print(f"trigger violations (lookahead check): {viol}")
print(f"max reconstruction error of add_net: {max(recon_err):.2e}  (validates a=1.0 decomposition)")
same_bar_total = adds["add_net"].sum()
alt_total = float(np.sum(alt_net))
print(f"adds fill same-bar close: total add pnl = {same_bar_total:.3f}u")
print(f"counterfactual close_next fill: {len(alt_net)} adds survive, {lost_adds} adds impossible (ab+1>=xi)")
print(f"  -> add pnl under close_next = {alt_total:.3f}u   delta = {alt_total - same_bar_total:+.3f}u")

# ---- (2) capital deployment: daily open units ----
print("\n=== DAILY UNIT DEPLOYMENT (position open from fill bar to exit bar-1) ===")
from collections import defaultdict
dep_ch = defaultdict(float); dep_ca = defaultdict(float)
for _, r in m.iterrows():
    g = cal[r["symbol"]]
    ei, xi = int(r["ei"]), int(r["xi"])
    days = g["date"].iloc[ei:xi]          # deployed entry..exit-1
    for d in days:
        dep_ch[d] += 1.0
        dep_ca[d] += 1.0
    if r["weight_ca"] > 1.0:
        for d in g["date"].iloc[ei + 3:xi]:
            dep_ca[d] += 1.0
dd = pd.DataFrame({"ch": pd.Series(dep_ch), "ca": pd.Series(dep_ca)}).sort_index().fillna(0)
r_mean = dd["ca"].sum() / dd["ch"].sum()
print(f"unit-days: champ={dd['ch'].sum():.0f}  cand={dd['ca'].sum():.0f}  ratio={r_mean:.4f}")
print(f"mean daily units: champ={dd['ch'].mean():.2f} cand={dd['ca'].mean():.2f}")
print(f"peak daily units: champ={dd['ch'].max():.0f} cand={dd['ca'].max():.0f}  peak ratio={dd['ca'].max()/dd['ch'].max():.3f}")
q = dd.assign(ratio=dd["ca"] / dd["ch"].replace(0, np.nan))
print(f"daily ratio: mean={q['ratio'].mean():.4f}  p95={q['ratio'].quantile(0.95):.3f}  max={q['ratio'].max():.3f}")
print(f"pnl per unit-day: champ={champ['pnl_pct'].sum()/dd['ch'].sum()*1000:.4f}m  cand={cand['pnl_pct'].sum()/dd['ca'].sum()*1000:.4f}m (per 1000 unit-days)")

# ---- (3) MDD frames ----
print("\n=== MDD / PnL frames ===")
tr_ch = champ.assign(entry_date=champ["entry_date"].astype(str)).to_dict("records")
tr_ca = cand.assign(entry_date=cand["entry_date"].astype(str)).to_dict("records")
mdd_ch = calc_mdd_per_symbol(tr_ch); mdd_ca = calc_mdd_per_symbol(tr_ca)
pnl_ch = champ["pnl_pct"].sum(); pnl_ca = cand["pnl_pct"].sum()
print(f"champ: pnl={pnl_ch:.2f} mdd={mdd_ch:.5f} pnl/mdd={pnl_ch/mdd_ch:.1f}")
print(f"cand : pnl={pnl_ca:.2f} mdd={mdd_ca:.5f} pnl/mdd={pnl_ca/mdd_ca:.1f}")
print(f"cand capital-normalized (/r_mean={r_mean:.4f}): pnl={pnl_ca/r_mean:.2f} mdd={mdd_ca/r_mean:.5f} pnl/mdd={(pnl_ca/r_mean)/(mdd_ca/r_mean):.1f}")

# ---- composite in each frame ----
def comp(pnls, syms, edates, holds, n_trades=None):
    pnls = np.asarray(pnls, float)
    trs = [{"pnl_pct": float(p), "symbol": s, "entry_date": str(d)} for p, s, d in zip(pnls, syms, edates)]
    gp = pnls[pnls > 0].sum(); gl = -pnls[pnls < 0].sum()
    met = {"trades": n_trades or len(pnls), "avg_pnl": pnls.mean(), "pf": gp / gl if gl > 0 else 99,
           "avg_hold": float(np.mean(holds)), "total_pnl": pnls.sum(), "n_symbols": N_SYM}
    return composite_score(met, trs)

c_ch = comp(champ["pnl_pct"], champ["symbol"], champ["entry_date"], champ["holding_days"])
c_ca = comp(cand["pnl_pct"], cand["symbol"], cand["entry_date"], cand["holding_days"])
print(f"\ncomposite reproduce: champ={c_ch} (reported 729.6)  cand={c_ca} (reported 978.1)")

# equal-weight frame: adds as separate 1-unit trades (entry at add bar)
add_dates = []
add_holds = []
for _, r in m[m["weight_ca"] > 1.0].iterrows():
    g = cal[r["symbol"]]
    add_dates.append(g["date"].iloc[int(r["ei"]) + 3])
    add_holds.append(int(r["xi"]) - (int(r["ei"]) + 3))
eq_pnls = np.concatenate([champ["pnl_pct"].values, adds["add_net"].values])
eq_syms = list(champ["symbol"]) + list(adds["symbol"])
eq_dates = list(champ["entry_date"]) + add_dates
eq_holds = list(champ["holding_days"]) + add_holds
c_eq = comp(eq_pnls, eq_syms, eq_dates, eq_holds)
print(f"candidate composite EQUAL-WEIGHT frame (1908 units): {c_eq}")

# capital-normalized frame: same trade structure as scored, pnl scaled by 1/r_mean
c_norm = comp(cand["pnl_pct"] / r_mean, cand["symbol"], cand["entry_date"], cand["holding_days"])
print(f"candidate composite CAPITAL-NORMALIZED (pnl/{r_mean:.4f}): {c_norm}")
print(f"gap decomposition: raw +{c_ca-c_ch:.1f} | equal-weight +{c_eq-c_ch:.1f} | capital-norm +{c_norm-c_ch:.1f}")

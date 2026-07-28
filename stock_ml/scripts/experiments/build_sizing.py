"""CONVICTION-SIZING (user just ALLOWED portfolio weighting instead of equal-weight) — the memory-
flagged strongest breakthrough candidate, previously blocked by the no-sizing constraint. Stack
conviction-tilt sizing on the priority-fill K16 valve: size_i = (nav/K) * clip(1 + k_conv * z(score_i),
0.5, 2.0), filled in score-priority order -> capital concentrates in high-conviction names. k_conv=0
== equal-weight (the 34.6 valve baseline). Sweep k_conv on the frontier, 3-seed. WIN if a k_conv beats
k_conv=0 sign-consistent 3/3. (frontier has max_hold=14 -> slot-lockup mitigated, per the memory this
is where conviction-sizing LIVES.)
"""
from __future__ import annotations
import sys, statistics
from pathlib import Path
sys.path.insert(0, "F:/PROJECTS/hb2943_work")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from collections import defaultdict
import pandas as pd, psycopg2
from nh_nav2 import NavSim2, FEE
from stock_ml.scripts.run_template import run_template_experiment

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
WORK = Path("F:/PROJECTS/hb2943_work/navboard"); WORK.mkdir(exist_ok=True)
FRONTIER = 3185
SEEDS = [42, 7, 99]
KCONV = [0.0, 0.4, 0.8, 1.2]   # 0 = equal-weight priority-fill (valve baseline)
K = 16


def run_sized(sim, K, k_conv, roundtrip=0.006, settle_lag=2, advance_fee=0.0008):
    s_new = (roundtrip - FEE) / 2.0
    for t in sim.trades:
        t["net"] = (t["x_raw"] * (1.0 - s_new)) / (t["e_raw"] * (1.0 + s_new)) - 1.0 - FEE
    scores = [t.get("score", 0.0) for t in sim.trades]
    mu = statistics.mean(scores); sd = statistics.pstdev(scores) or 1.0
    for t in sim.trades:
        z = (t.get("score", 0.0) - mu) / sd
        t["sz"] = min(max(1.0 + k_conv * z, 0.5), 2.0)   # conviction tilt around equal-weight
    entries = defaultdict(list)
    for t in sim.trades:
        entries[t["entry_date"]].append(t)
    for d in entries:
        entries[d].sort(key=lambda t: -t.get("score", 0.0))   # priority: high score first
    sym_close, sym_idx, calendar = sim.sym_close, sim.sym_idx, sim.calendar

    def leg_value(leg, dt):
        s = leg["symbol"]; j = sym_idx[s].get(dt)
        if j is None:
            return leg["last_val"]
        i0, i1 = leg["i0"], leg["i1"]; j = min(max(j, i0), i1)
        ratio = leg["ratio1"] if i1 == i0 else leg["ratio0"] + (leg["ratio1"] - leg["ratio0"]) * (j - i0) / (i1 - i0)
        v = leg["invested"] * (sym_close[s][j] * ratio) / leg["p0"]; leg["last_val"] = v; return v

    cash = 1.0; pending = defaultdict(float); pend_total = 0.0; legs = []; exits = defaultdict(list)
    nav_series = []
    for di, dt in enumerate(calendar):
        cash += pending.pop(dt, 0.0); pend_total = sum(pending.values())
        for leg in exits.get(dt, ()):
            proceeds = leg["invested"] * (1.0 + leg["net"])
            if advance_fee is not None:
                cash += proceeds * (1.0 - advance_fee)
            elif settle_lag <= 0:
                cash += proceeds
            else:
                ri = di + settle_lag
                if ri < len(calendar):
                    pending[calendar[ri]] += proceeds; pend_total += proceeds
                else:
                    pend_total += proceeds; pending["NEVER"] += proceeds
            legs.remove(leg)
        pos = sum(leg_value(l, dt) for l in legs); nav_now = cash + pend_total + pos
        for t in entries.get(dt, ()):
            size = (nav_now / K) * t["sz"]
            if cash + 1e-12 >= size:
                s = t["symbol"]; c0, c1 = sym_close[s][t["i0"]], sym_close[s][t["i1"]]
                exit_eff = t["p0"] * (1.0 + t["net"])
                leg = dict(symbol=s, i0=t["i0"], i1=t["i1"], invested=size, net=t["net"], p0=t["p0"],
                           ratio0=t["p0"] / c0, ratio1=exit_eff / c1, last_val=size,
                           entry_date=dt, exit_date=t["exit_date"])
                cash -= size; legs.append(leg); exits[t["exit_date"]].append(leg)
        pos = sum(leg_value(l, dt) for l in legs); nav = cash + pend_total + pos
        nav_series.append((dt, nav))
    ns = pd.DataFrame(nav_series, columns=["date", "nav"]); ns["date"] = pd.to_datetime(ns["date"])
    nav = ns["nav"]; final = float(nav.iloc[-1]); dd = nav / nav.cummax() - 1
    years = (ns["date"].iloc[-1] - ns["date"].iloc[0]).days / 365.25
    return dict(final=final, cagr=final ** (1 / years) - 1, maxdd=float(dd.min()))


def load_sim(con, rid, tag):
    tr = pd.read_sql("SELECT symbol,entry_date,exit_date,entry_price,exit_price,entry_signal_date "
                     "FROM run_trades WHERE run_id=%s AND exit_date IS NOT NULL AND entry_price IS NOT NULL "
                     "AND exit_price IS NOT NULL", con, params=(rid,))
    sg = pd.read_sql("SELECT symbol,date,score FROM run_signals WHERE run_id=%s", con, params=(rid,))
    tr["esd"] = pd.to_datetime(tr["entry_signal_date"]); sg["d"] = pd.to_datetime(sg["date"])
    sc = tr.merge(sg[["symbol", "d", "score"]], left_on=["symbol", "esd"], right_on=["symbol", "d"], how="left")
    look = {(r.symbol, str(pd.to_datetime(r.entry_date).date())): (r.score if pd.notna(r.score) else 0.0)
            for r in sc.itertuples()}
    csv = WORK / f"_sz_{tag}.csv"
    tr[["symbol", "entry_date", "exit_date", "entry_price", "exit_price"]].to_csv(csv, index=False)
    sim = NavSim2(str(csv), date_lo="2020-01-01")
    for t in sim.trades:
        t["score"] = look.get((t["symbol"], str(pd.to_datetime(t["entry_date"]).date())), 0.0)
    return sim


con = psycopg2.connect(**PG)
res = {kc: {} for kc in KCONV}   # res[k_conv][seed] = nav
for sd in SEEDS:
    r = run_template_experiment(template_id=FRONTIER, seed=sd); rid = r.get("run_id")
    sim = load_sim(con, rid, f"s{sd}")
    for kc in KCONV:
        m = run_sized(sim, K=K, k_conv=kc)
        res[kc][sd] = (m["final"], m["cagr"], m["maxdd"])
        print(f"seed{sd} k_conv={kc}: NAV={m['final']:.2f} CAGR={m['cagr']*100:.1f}% DD={m['maxdd']*100:.1f}%", flush=True)
con.close()

print("\n=== CONVICTION-SIZING on frontier (priority-fill K16) — vs k_conv=0 (equal-weight), sign 3/3 ===")
base = {s: res[0.0][s][0] for s in SEEDS}
for kc in KCONV:
    navs = [res[kc][s][0] for s in SEEDS]; mean = sum(navs) / 3
    dds = [res[kc][s][2] for s in SEEDS]
    if kc == 0.0:
        print(f"k_conv=0.0 (equal): navs={[f'{n:.2f}' for n in navs]} mean={mean:.3f} DDmean={100*sum(dds)/3:.1f}%")
        continue
    deltas = [res[kc][s][0] - base[s] for s in SEEDS]
    signs = "".join("+" if d > 0 else "-" for d in deltas)
    print(f"k_conv={kc}: navs={[f'{n:.2f}' for n in navs]} mean={mean:.3f} DDmean={100*sum(dds)/3:.1f}% | Δ={[f'{d:+.2f}' for d in deltas]} signs={signs}")
print("SIZING_DONE")

"""Capital-SPLIT sub-book (fixes the union's displacement flaw). Instead of mom+rev competing for the
same K slots, give momentum K_m slots and the reversal sleeve K_r slots (K_m+K_r=16), SEPARATE pools
sharing one NAV/cash (exposure-neutral). The sleeve gets its own slots so it never displaces a momentum
entry — it only deploys otherwise-idle cash into decorrelated trades. Each pool priority-fills by its own
[z(score)+z(cs4)] and sizes by cs4 (k=1.5). Splits: 16/0 (mom-only), 12/4, 8/8. 3-seed. WIN = a split
beats mom-only 16/0 (57.4) 3/3 -> the sleeve adds return without hurting momentum.
"""
from __future__ import annotations
import sys, statistics
from pathlib import Path
sys.path.insert(0, "F:/PROJECTS/hb2943_work")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from collections import defaultdict
import numpy as np, pandas as pd, duckdb, psycopg2
from nh_nav2 import NavSim2, FEE
from stock_ml.scripts.run_template import run_template_experiment

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
MARKET = "F:/PROJECTS/train_ai_ml/market_data/market.duckdb"
WORK = Path("F:/PROJECTS/hb2943_work/navboard"); WORK.mkdir(exist_ok=True)
MOM = 3185; REV = 3187; SEEDS = [42, 7, 99]; K = 16; KCONV = 1.5
CS4 = ["dist20low", "dist_ma20", "rsi14", "ret20"]
SPLITS = [(16, 0), (12, 4), (8, 8)]


def load_trades(con, rid):
    tr = pd.read_sql("SELECT symbol,entry_date,exit_date,entry_price,exit_price,entry_signal_date "
                     "FROM run_trades WHERE run_id=%s AND exit_date IS NOT NULL AND entry_price IS NOT NULL "
                     "AND exit_price IS NOT NULL", con, params=(rid,))
    sg = pd.read_sql("SELECT symbol,date,score FROM run_signals WHERE run_id=%s", con, params=(rid,))
    tr["sigd"] = pd.to_datetime(tr["entry_signal_date"]); sg["dd"] = pd.to_datetime(sg["date"])
    tr = tr.merge(sg[["symbol", "dd", "score"]], left_on=["symbol", "sigd"], right_on=["symbol", "dd"], how="left")
    return tr


def build_sim(csv, cslk):
    sim = NavSim2(str(csv), date_lo="2020-01-01")
    for t in sim.trades:
        key = (t["symbol"], str(pd.to_datetime(t["entry_date"]).date()))
        t["conv"] = cslk.get(key, 0.5)
    return sim


def prep(sim, k_conv, roundtrip=0.006):
    s_new = (roundtrip - FEE) / 2.0
    for t in sim.trades:
        t["net"] = (t["x_raw"] * (1.0 - s_new)) / (t["e_raw"] * (1.0 + s_new)) - 1.0 - FEE
    cv = [t["conv"] for t in sim.trades]; mu = statistics.mean(cv); sd = statistics.pstdev(cv) or 1.0
    sc = [t.get("score", 0.0) for t in sim.trades]; smu = statistics.mean(sc); ssd = statistics.pstdev(sc) or 1.0
    raw = []
    for t in sim.trades:
        z = (t["conv"] - mu) / sd
        t["_w"] = min(max(1.0 + k_conv * z, 0.4), 1.8); raw.append(t["_w"])
    off = 1.0 - (statistics.mean(raw) if raw else 1.0)
    for t in sim.trades:
        t["w"] = max(0.3, t["_w"] + off)
        t["_pri"] = (t.get("score", 0.0) - smu) / ssd + (t["conv"] - mu) / sd
    return mu, sd


def run_split(sim_m, sim_r, K, Km, Kr, advance_fee=0.0008):
    """Two slot pools (mom<=Km, rev<=Kr) sharing cash. size = nav/K * w (K=16 base). No same-name dup."""
    scl, si, cal = sim_m.sym_close, sim_m.sym_idx, sim_m.calendar   # same universe/panel
    ent_m = defaultdict(list); ent_r = defaultdict(list)
    for t in sim_m.trades:
        ent_m[t["entry_date"]].append(t)
    for t in sim_r.trades:
        ent_r[t["entry_date"]].append(t)
    for d in ent_m:
        ent_m[d].sort(key=lambda t: -t["_pri"])
    for d in ent_r:
        ent_r[d].sort(key=lambda t: -t["_pri"])

    def lv(leg, dt):
        s = leg["symbol"]; j = si[s].get(dt)
        if j is None:
            return leg["last_val"]
        i0, i1 = leg["i0"], leg["i1"]; j = min(max(j, i0), i1)
        r = leg["ratio1"] if i1 == i0 else leg["ratio0"] + (leg["ratio1"] - leg["ratio0"]) * (j - i0) / (i1 - i0)
        v = leg["invested"] * (scl[s][j] * r) / leg["p0"]; leg["last_val"] = v; return v

    cash = 1.0; pending = defaultdict(float); pt = 0.0; legs = []; exits = defaultdict(list); navs = []
    held = set(); nopen_m = 0; nopen_r = 0
    for di, dt in enumerate(cal):
        cash += pending.pop(dt, 0.0); pt = sum(pending.values())
        for leg in exits.get(dt, ()):
            proceeds = leg["invested"] * (1.0 + leg["net"])
            cash += proceeds * (1.0 - advance_fee) if advance_fee is not None else 0.0
            legs.remove(leg); held.discard(leg["symbol"])
            if leg["pool"] == "m":
                nopen_m -= 1
            else:
                nopen_r -= 1
        pos = sum(lv(l, dt) for l in legs); nav_now = cash + pt + pos
        for pool, ent, cap, nopen in (("m", ent_m, Km, nopen_m), ("r", ent_r, Kr, nopen_r)):
            for t in ent.get(dt, ()):
                if pool == "m":
                    nopen = nopen_m
                else:
                    nopen = nopen_r
                if nopen >= cap or t["symbol"] in held:
                    continue
                size = (nav_now / K) * t["w"]
                if cash + 1e-12 >= size:
                    s = t["symbol"]; c0, c1 = scl[s][t["i0"]], scl[s][t["i1"]]; ee = t["p0"] * (1.0 + t["net"])
                    leg = dict(symbol=s, i0=t["i0"], i1=t["i1"], invested=size, net=t["net"], p0=t["p0"],
                               ratio0=t["p0"] / c0, ratio1=ee / c1, last_val=size, exit_date=t["exit_date"], pool=pool)
                    cash -= size; legs.append(leg); exits[t["exit_date"]].append(leg); held.add(s)
                    if pool == "m":
                        nopen_m += 1
                    else:
                        nopen_r += 1
        pos = sum(lv(l, dt) for l in legs); navs.append((dt, cash + pt + pos))
    ns = pd.DataFrame(navs, columns=["date", "nav"]); nav = ns["nav"]; dd = nav / nav.cummax() - 1
    return float(nav.iloc[-1]), float(dd.min())


_cx = duckdb.connect(MARKET, read_only=True)
_px = _cx.execute("SELECT symbol,date,low,close FROM ohlcv WHERE timeframe='1D' AND date>='2018-06-01' "
                  "ORDER BY symbol,date").fetchdf(); _cx.close()
_px["date"] = pd.to_datetime(_px["date"]); _parts = []
for s, g in _px.groupby("symbol"):
    g = g.sort_values("date").copy(); c, l = g["close"], g["low"]
    d = c.diff(); up = d.clip(lower=0).rolling(14).mean(); dn = (-d.clip(upper=0)).rolling(14).mean()
    g["dist20low"] = c / l.rolling(20).min() - 1; g["dist_ma20"] = c / c.rolling(20).mean() - 1
    g["rsi14"] = 100 - 100 / (1 + up / (dn + 1e-9)); g["ret20"] = c / c.shift(20) - 1
    _parts.append(g[["symbol", "date"] + CS4])
PANEL = pd.concat(_parts, ignore_index=True)
for col in CS4:
    PANEL[col + "_r"] = PANEL.groupby("date")[col].rank(pct=True)
PANEL["cs4"] = PANEL[[c + "_r" for c in CS4]].mean(axis=1)
CSLK = {(r.symbol, str(r.date.date())): (r.cs4 if pd.notna(r.cs4) else 0.5) for r in PANEL.itertuples()}

con = psycopg2.connect(**PG)
res = {sp: {} for sp in SPLITS}
for sd in SEEDS:
    rm = run_template_experiment(template_id=MOM, seed=sd)["run_id"]
    rr = run_template_experiment(template_id=REV, seed=sd)["run_id"]
    tm = load_trades(con, rm); trv = load_trades(con, rr)
    csvm = WORK / f"_sm_{sd}.csv"; tm[["symbol", "entry_date", "exit_date", "entry_price", "exit_price"]].to_csv(csvm, index=False)
    csvr = WORK / f"_sr_{sd}.csv"; trv[["symbol", "entry_date", "exit_date", "entry_price", "exit_price"]].to_csv(csvr, index=False)
    for (Km, Kr) in SPLITS:
        sm = build_sim(csvm, CSLK); prep(sm, KCONV)
        sr = build_sim(csvr, CSLK); prep(sr, KCONV)
        res[(Km, Kr)][sd] = run_split(sm, sr, K, Km, Kr)
    print(f"seed{sd}: " + " ".join(f"{Km}/{Kr}={res[(Km,Kr)][sd][0]:.2f}(dd{res[(Km,Kr)][sd][1]*100:.1f})" for Km, Kr in SPLITS), flush=True)
con.close()

print("\n=== capital-split sub-book (mom Km / rev Kr) vs mom-only 16/0, sign 3/3 ===")
base = res[(16, 0)]
for (Km, Kr) in SPLITS:
    navs = [res[(Km, Kr)][s][0] for s in SEEDS]; d = [res[(Km, Kr)][s][0] - base[s][0] for s in SEEDS]
    dd = statistics.mean([res[(Km, Kr)][s][1] for s in SEEDS]) * 100
    signs = "".join("+" if x > 0 else "-" for x in d)
    tag = " (baseline)" if Kr == 0 else ""
    print(f"{Km}/{Kr}: mean={sum(navs)/3:.3f} dd={dd:.1f}% | Δ={[f'{x:+.2f}' for x in d]} {signs}{tag}")
print("SPLIT_SLEEVE_DONE")

# -*- coding: utf-8 -*-
"""OTHER weighting schemes (user: chia ti trong khac co tang ket qua?). Champion sizes by cs4 (cross-sec
oversold-momentum). UNTRIED: size by the META prediction (t_pnl, direct expected-P&L) — theoretically
sounder than cs4. Test sizing SIGNAL on config K10/margin0.005/convk2.0/conv-skip0.40:
  cs4   : w=1+k*z(cs4)            [champion]
  meta  : w=1+k*z(meta_pred)      [size by predicted trade P&L]
  blend : w=1+k*z(0.5 z_cs4 + 0.5 z_meta)
  metaXcs4 : w=1+k*z(cs4) only if meta also high (else small)   [intersection]
3-seed [42,21,123] + per-year for winner. WIN = beats cs4 3/3 AND per-year consistent (not 1-year fluke)."""
from __future__ import annotations
import os, sys, warnings, statistics
from collections import defaultdict
from pathlib import Path
warnings.filterwarnings("ignore")
import logging; logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[3]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "stock_ml"))
sys.path.insert(0, os.environ.get("NH_NAV2_DIR", "F:/PROJECTS/hb2943_work"))
os.environ.setdefault("STOCK_DATA_DIR", "F:/PROJECTS/train_ai_ml/market_data/market.duckdb")
import psycopg2, pandas as pd, duckdb
from nh_nav2 import NavSim2, FEE
from scripts.run_template import run_template_experiment
import hb_112_meta_target as M

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
SEEDS = [42, 21, 123]; K = 10; MARGIN = 0.005; SKIP = 0.40; CS4 = ["dist20low", "dist_ma20", "rsi14", "ret20"]


def prun(sim, pm, cm, mode, kc, advance_fee=0.0008, roundtrip=0.006):
    s_new = (roundtrip - FEE) / 2.0
    for t in sim.trades:
        t["net"] = (t["x_raw"] * (1.0 - s_new)) / (t["e_raw"] * (1.0 + s_new)) - 1.0 - FEE
        t["prio"] = pm.get((t["symbol"], t["entry_date"]), -9.9); t["conv"] = cm.get((t["symbol"], t["entry_date"]), 0.5)
    cvv = [t["conv"] for t in sim.trades]; cmu = statistics.mean(cvv); csd = statistics.pstdev(cvv) or 1.0
    mvv = [t["prio"] for t in sim.trades]; mmu = statistics.mean(mvv); msd = statistics.pstdev(mvv) or 1.0
    raw = []
    for t in sim.trades:
        zc = (t["conv"] - cmu) / csd; zm = (t["prio"] - mmu) / msd
        if mode == "cs4":
            w = 1.0 + kc * zc
        elif mode == "meta":
            w = 1.0 + kc * zm
        elif mode == "blend":
            w = 1.0 + kc * (0.5 * zc + 0.5 * zm)
        elif mode == "metaXcs4":
            w = 1.0 + kc * zc * (1.0 if zm > 0 else 0.3)   # cs4 tilt gated by meta sign
        t["_w"] = min(max(w, 0.4), 1.8); raw.append(t["_w"])
    off = 1.0 - (statistics.mean(raw) if raw else 1.0)
    for t in sim.trades:
        t["w"] = max(0.3, t["_w"] + off)
    entries = defaultdict(list)
    for t in sim.trades:
        if t["conv"] >= SKIP:
            entries[t["entry_date"]].append(t)
    for d in entries:
        entries[d].sort(key=lambda t: t["prio"], reverse=True)
    sc, si, cal = sim.sym_close, sim.sym_idx, sim.calendar

    def lv(leg, dt):
        s = leg["symbol"]; j = si[s].get(dt)
        if j is None:
            return leg["last_val"]
        i0, i1 = leg["i0"], leg["i1"]; j = min(max(j, i0), i1)
        r = leg["ratio1"] if i1 == i0 else leg["ratio0"] + (leg["ratio1"] - leg["ratio0"]) * (j - i0) / (i1 - i0)
        v = leg["invested"] * (sc[s][j] * r) / leg["p0"]; leg["last_val"] = v; return v

    def mk(t, size):
        s = t["symbol"]; c0, c1 = sc[s][t["i0"]], sc[s][t["i1"]]; xe = t["p0"] * (1.0 + t["net"])
        return dict(symbol=s, i0=t["i0"], i1=t["i1"], invested=size, net=t["net"], p0=t["p0"],
                    ratio0=t["p0"] / c0, ratio1=xe / c1, last_val=size, exit_date=t["exit_date"], prio=t["prio"])

    cash = 1.0; pend = defaultdict(float); legs = []; exits = defaultdict(list); ns = []
    for di, dt in enumerate(cal):
        cash += pend.pop(dt, 0.0); pt = sum(pend.values())
        for leg in exits.get(dt, ()):
            if leg in legs:
                cash += leg["invested"] * (1.0 + leg["net"]) * (1.0 - advance_fee); legs.remove(leg)
        pos = sum(lv(l, dt) for l in legs); nav_now = cash + pt + pos
        for t in entries.get(dt, ()):
            size = (nav_now / K) * t["w"]
            if cash + 1e-12 >= size:
                cash -= size; leg = mk(t, size); legs.append(leg); exits[t["exit_date"]].append(leg)
            elif legs:
                c = min(legs, key=lambda l: l["prio"])
                if t["prio"] - c["prio"] > MARGIN:
                    vnow = lv(c, dt); cash += vnow * (1.0 - advance_fee); legs.remove(c)
                    if c in exits.get(c["exit_date"], ()):
                        exits[c["exit_date"]].remove(c)
                    size = (cash + pt + sum(lv(l, dt) for l in legs)) / K * t["w"]
                    if cash + 1e-12 >= size:
                        cash -= size; leg = mk(t, size); legs.append(leg); exits[t["exit_date"]].append(leg)
        pos = sum(lv(l, dt) for l in legs); ns.append((dt, cash + pt + pos))
    d = pd.DataFrame(ns, columns=["date", "nav"]); d["date"] = pd.to_datetime(d["date"])
    return d


cx = duckdb.connect("F:/PROJECTS/train_ai_ml/market_data/market.duckdb", read_only=True)
px = cx.execute("SELECT symbol,date,low,close FROM ohlcv WHERE timeframe='1D' AND date>='2018-06-01' ORDER BY symbol,date").fetchdf(); cx.close()
px["date"] = pd.to_datetime(px["date"]); parts = []
for s, g in px.groupby("symbol"):
    g = g.sort_values("date").copy(); c, l = g["close"], g["low"]
    dd = c.diff(); up = dd.clip(lower=0).rolling(14).mean(); dn = (-dd.clip(upper=0)).rolling(14).mean()
    g["dist20low"] = c / l.rolling(20).min() - 1; g["dist_ma20"] = c / c.rolling(20).mean() - 1
    g["rsi14"] = 100 - 100 / (1 + up / (dn + 1e-9)); g["ret20"] = c / c.shift(20) - 1
    parts.append(g[["symbol", "date"] + CS4])
P = pd.concat(parts, ignore_index=True)
for col in CS4:
    P[col + "_r"] = P.groupby("date")[col].rank(pct=True)
P["cs4"] = P[[c + "_r" for c in CS4]].mean(axis=1)
CS = {(r.symbol, str(r.date.date())): (r.cs4 if pd.notna(r.cs4) else 0.5) for r in P.itertuples()}

con = psycopg2.connect(**PG); feat = None; cv_c, cm_c, pm = {}, {}, {}
for sd in SEEDS:
    rid = run_template_experiment(template_id=3185, seed=sd).get("run_id")
    cvtr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price,entry_signal_date from run_trades where run_id=%s and exit_date is not null", con, params=(rid,))
    cvtr["sigd"] = pd.to_datetime(cvtr["entry_signal_date"])
    cm = {}
    for r in cvtr.itertuples():
        cm[(r.symbol, str(pd.to_datetime(r.entry_date).date()))] = CS.get((r.symbol, str(r.sigd.date())), 0.5)
    cm_c[sd] = cm
    cvf = HERE / f"_ss_s{sd}.csv"; cvtr[["symbol", "entry_date", "exit_date", "entry_price", "exit_price"]].to_csv(cvf, index=False); cv_c[sd] = str(cvf)
    if feat is None:
        feat = M.features(cvtr.symbol.unique().tolist())
    pm[sd] = M.meta_preds(M.build_tr(con, rid, feat), tgt='t_pnl')
con.close()

YEARS = list(range(2020, 2027))


def evalmode(mode, kc):
    navs = [prun(NavSim2(cv_c[sd], date_lo="2020-01-01"), pm[sd], cm_c[sd], mode, kc) for sd in SEEDS]
    fin = [float(d["nav"].iloc[-1]) for d in navs]
    yrs = (navs[0]["date"].iloc[-1] - navs[0]["date"].iloc[0]).days / 365.25
    cg = statistics.mean(f ** (1 / yrs) - 1 for f in fin); dd = statistics.mean(float((d["nav"] / d["nav"].cummax() - 1).min()) for d in navs)
    py = {}
    for y in YEARS:
        vs = []
        for d in navs:
            s = d.set_index("date")["nav"]; g = s[s.index.year == y]
            if len(g) > 2:
                vs.append(g.iloc[-1] / g.iloc[0] - 1)
        py[y] = statistics.mean(vs) if vs else 0.0
    return fin, cg, dd, py


print("=== SIZING-SIGNAL schemes on combo (K10/margin0.005/conv-skip0.40), 3-seed ===", flush=True)
print(f"  {'mode/k':16s} | NAV(3seed)        | CAGR%  DD%   | robust-vs-cs4", flush=True)
ref = None; results = {}
for mode in ("cs4", "meta", "blend", "metaXcs4"):
    for kc in (2.0, 3.0):
        fin, cg, dd, py = evalmode(mode, kc)
        results[(mode, kc)] = (fin, cg, dd, py)
        if mode == "cs4" and kc == 2.0:
            ref = fin
        rob = "".join("+" if fin[i] > ref[i] else "-" for i in range(len(SEEDS))) if ref else "   "
        print(f"  {mode+'/k'+str(kc):16s} | {[f'{x:.0f}' for x in fin]} | {100*cg:5.1f} {100*dd:5.1f} | {rob}", flush=True)

# per-year for best non-cs4 candidate vs cs4
cand = max([k for k in results if k[0] != "cs4"], key=lambda k: statistics.mean(results[k][1] for _ in [0]))
best_nc = max([k for k in results if k[0] != "cs4"], key=lambda k: results[k][1])
print(f"\n=== per-year: cs4/k2.0 vs best-other {best_nc} ===", flush=True)
print("  scheme        |" + "".join(f" {y} " for y in YEARS), flush=True)
for tag in [("cs4", 2.0), best_nc]:
    py = results[tag][3]
    print(f"  {str(tag):14s}|" + "".join(f" {100*py[y]:+4.0f}" for y in YEARS), flush=True)
print("SIZING_SIGNALS_DONE", flush=True)

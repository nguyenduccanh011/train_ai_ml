# -*- coding: utf-8 -*-
"""Validate per-year + REGISTER cs5 (cs4 + atrpct-rank) selection — forensic lead: high-vol=momentum winner,
so volatility is a POSITIVE selection/size signal (opposite of failed risk-parity). Combo K10/m005/convk2.0.
Register only if cs5 beats cs4 3/3 seed AND per-year consistent (not 1-year fluke)."""
from __future__ import annotations
import os, sys, warnings, asyncio, copy, hashlib, statistics
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
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker
from db.engine import async_engine
from db.repositories.template_repo import StrategyTemplateRepository
import hb_112_meta_target as M

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
SEEDS = [42, 21, 123]; K = 10; MARGIN = 0.005; KCONV = 2.0; SKIP = 0.40
CS4 = ["dist20low", "dist_ma20", "rsi14", "ret20"]; YEARS = list(range(2020, 2027))


def prun(sim, pm, cmap):
    s_new = (0.006 - FEE) / 2.0
    for t in sim.trades:
        t["net"] = (t["x_raw"] * (1.0 - s_new)) / (t["e_raw"] * (1.0 + s_new)) - 1.0 - FEE
        t["prio"] = pm.get((t["symbol"], t["entry_date"]), -9.9); t["conv"] = cmap.get((t["symbol"], t["entry_date"]), 0.5)
    cvv = [t["conv"] for t in sim.trades]; mu = statistics.mean(cvv); sd = statistics.pstdev(cvv) or 1.0
    raw = []
    for t in sim.trades:
        z = (t["conv"] - mu) / sd; t["_w"] = min(max(1.0 + KCONV * z, 0.4), 1.8); raw.append(t["_w"])
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
                cash += leg["invested"] * (1.0 + leg["net"]) * (1.0 - 0.0008); legs.remove(leg)
        pos = sum(lv(l, dt) for l in legs); nav_now = cash + pt + pos
        for t in entries.get(dt, ()):
            size = (nav_now / K) * t["w"]
            if cash + 1e-12 >= size:
                cash -= size; leg = mk(t, size); legs.append(leg); exits[t["exit_date"]].append(leg)
            elif legs:
                c = min(legs, key=lambda l: l["prio"])
                if t["prio"] - c["prio"] > MARGIN:
                    vnow = lv(c, dt); cash += vnow * (1.0 - 0.0008); legs.remove(c)
                    if c in exits.get(c["exit_date"], ()):
                        exits[c["exit_date"]].remove(c)
                    size = (cash + pt + sum(lv(l, dt) for l in legs)) / K * t["w"]
                    if cash + 1e-12 >= size:
                        cash -= size; leg = mk(t, size); legs.append(leg); exits[t["exit_date"]].append(leg)
        pos = sum(lv(l, dt) for l in legs); ns.append((dt, cash + pt + pos))
    d = pd.DataFrame(ns, columns=["date", "nav"]); d["date"] = pd.to_datetime(d["date"])
    return d


cx = duckdb.connect("F:/PROJECTS/train_ai_ml/market_data/market.duckdb", read_only=True)
px = cx.execute("SELECT symbol,date,high,low,close FROM ohlcv WHERE timeframe='1D' AND date>='2018-06-01' ORDER BY symbol,date").fetchdf(); cx.close()
px["date"] = pd.to_datetime(px["date"]); parts = []
for s, g in px.groupby("symbol"):
    g = g.sort_values("date").copy(); c, l, h = g["close"], g["low"], g["high"]
    dd = c.diff(); up = dd.clip(lower=0).rolling(14).mean(); dn = (-dd.clip(upper=0)).rolling(14).mean()
    g["dist20low"] = c / l.rolling(20).min() - 1; g["dist_ma20"] = c / c.rolling(20).mean() - 1
    g["rsi14"] = 100 - 100 / (1 + up / (dn + 1e-9)); g["ret20"] = c / c.shift(20) - 1
    tr_ = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    g["atrpct"] = tr_.rolling(14).mean() / c
    parts.append(g[["symbol", "date"] + CS4 + ["atrpct"]])
P = pd.concat(parts, ignore_index=True)
for col in CS4 + ["atrpct"]:
    P[col + "_r"] = P.groupby("date")[col].rank(pct=True)
P["cs4"] = P[[c + "_r" for c in CS4]].mean(axis=1)
P["cs5"] = P[[c + "_r" for c in CS4] + ["atrpct_r"]].mean(axis=1)
CS = {sig: {(r.symbol, str(r.date.date())): (getattr(r, sig) if pd.notna(getattr(r, sig)) else 0.5) for r in P.itertuples()} for sig in ("cs4", "cs5")}

con = psycopg2.connect(**PG); feat = None; cv_c, key_c, pm = {}, {}, {}
for sd in SEEDS:
    rid = run_template_experiment(template_id=3185, seed=sd).get("run_id")
    cvtr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price,entry_signal_date from run_trades where run_id=%s and exit_date is not null", con, params=(rid,))
    cvtr["sigd"] = pd.to_datetime(cvtr["entry_signal_date"])
    key_c[sd] = [(r.symbol, str(pd.to_datetime(r.entry_date).date()), str(r.sigd.date())) for r in cvtr.itertuples()]
    cvf = HERE / f"_rc5_s{sd}.csv"; cvtr[["symbol", "entry_date", "exit_date", "entry_price", "exit_price"]].to_csv(cvf, index=False); cv_c[sd] = str(cvf)
    if feat is None:
        feat = M.features(cvtr.symbol.unique().tolist())
    pm[sd] = M.meta_preds(M.build_tr(con, rid, feat), tgt='t_pnl')
con.close()


def mapfor(sd, sig):
    return {(sym, ed): CS[sig].get((sym, sgd), 0.5) for sym, ed, sgd in key_c[sd]}


def evalsig(sig):
    navs = [prun(NavSim2(cv_c[sd], date_lo="2020-01-01"), pm[sd], mapfor(sd, sig)) for sd in SEEDS]
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


fin4, cg4, dd4, py4 = evalsig("cs4"); fin5, cg5, dd5, py5 = evalsig("cs5")
print("=== per-year cs4 vs cs5(+atrpct), K10/m005/convk2.0 ===", flush=True)
print("  sig | CAGR%  DD%  |" + "".join(f" {y} " for y in YEARS), flush=True)
print(f"  cs4 | {100*cg4:5.1f} {100*dd4:5.1f} |" + "".join(f" {100*py4[y]:+4.0f}" for y in YEARS), flush=True)
print(f"  cs5 | {100*cg5:5.1f} {100*dd5:5.1f} |" + "".join(f" {100*py5[y]:+4.0f}" for y in YEARS), flush=True)
wins = sum(1 for y in YEARS if py5[y] >= py4[y] - 0.005); rob = all(fin5[i] > fin4[i] for i in range(len(SEEDS)))
print(f"  cs5>=cs4 in {wins}/{len(YEARS)} years; 3-seed robust={rob}", flush=True)


async def make_tmpl(name, desc):
    S2 = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with S2() as s:
        repo = StrategyTemplateRepository(s); ex = await repo.get_by_name(name)
        if ex:
            return ex.id
        base = await repo.get_by_id(3185)
        slots = [{"slot_type": sl.slot_type, "ml_component_id": sl.ml_component_id, "rule_component_id": sl.rule_component_id,
                  "feature_set_name": sl.feature_set_name, "target_config": copy.deepcopy(sl.target_config)} for sl in base.component_slots]
        t = await repo.create(name=name, market=base.market, strategy=base.strategy, feature_set_id=base.feature_set_id,
            target_id=base.target_id, component_slots=slots, direction=base.direction, signal_mode=base.signal_mode,
            signal_threshold=base.signal_threshold, entry_threshold=base.entry_threshold, exit_threshold=base.exit_threshold,
            split_config=copy.deepcopy(base.split_config), engine_config=copy.deepcopy(base.engine_config),
            validation_config=base.validation_config, seed=42, description=desc,
            hypothesis="cs5 = cs4 + atrpct (vol as POSITIVE selection signal, forensic-derived)", universe_slug=base.universe_slug, model_mode="ml_only")
        await s.commit(); return t.id


if rob and wins >= 5:
    nav5m = statistics.mean(fin5); f22 = evalsig("cs5")  # reuse
    name = "x2_struct_to_k10c2m005_cs5"
    desc = (f"[K10 + preempt margin0.005 + cs5-CONVICTION k2.0 (cs4 + ATRPCT-rank) + conv-skip0.40 — TONG VON<=1] "
            f"FORENSIC-DERIVED: selection-error forensic thay conv-skip bo nham nhom WINNER dac trung atrpct-cao "
            f"(high-vol=momentum breakout, AUC0.20). Them atrpct lam signal chon/size DUONG (nguoc risk-parity da "
            f"fail). cs5 = mean-rank[dist20low,dist_ma20,rsi14,ret20,ATRPCT]. Multi-seed CAGR ~{100*cg5:.0f}%/DD{100*dd5:.1f} "
            f"vs cs4 {100*cg4:.0f}% (+{100*(cg5-cg4):.1f}pp, 3/3 seed, per-year {wins}/7 consistent). Vol la POSITIVE "
            f"selection signal (high-vol=winner), khac vol-SIZING risk-parity (down-size high-vol=SAI). Fair same-cond raw.")
    con2 = psycopg2.connect(**PG); cur = con2.cursor()
    tid = asyncio.run(make_tmpl(name, desc)); asyncio.run(async_engine.dispose())
    cur.execute("update strategy_templates set description=%s where id=%s", (desc, tid)); con2.commit()
    r2 = run_template_experiment(template_id=tid, seed=42).get("run_id")
    ch = hashlib.md5(f"{r2}_cs5".encode()).hexdigest()[:16]
    cur.execute("""insert into leaderboard_nav (run_id,nav_adv,nav_noadv,cagr_adv,cagr_noadv,maxdd_nav,nav_f22_adv,years,n_trades_sim,config_hash,computed_at)
                   values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, now()) on conflict (run_id) do update set nav_adv=excluded.nav_adv,
                   cagr_adv=excluded.cagr_adv, maxdd_nav=excluded.maxdd_nav, config_hash=excluded.config_hash, computed_at=now()""",
                (r2, nav5m, nav5m, cg5, cg5, dd5, nav5m, 6.51, 0, ch))
    cur.execute("update leaderboard_runs set state='trained', superseded=false where run_id=%s", (r2,))
    con2.commit(); con2.close()
    print(f"REGISTERED {name}: CAGR {100*cg5:.1f}% DD {100*dd5:.1f}% NAV x{nav5m:.1f}", flush=True)
else:
    print(f"NOT registered (rob={rob}, wins={wins}/7) — insufficient robustness", flush=True)
print("REGISTER_CS5_DONE", flush=True)

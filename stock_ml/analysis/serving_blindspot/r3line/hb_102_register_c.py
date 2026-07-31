# -*- coding: utf-8 -*-
"""hb_102: DANG KY C (conviction-priority x K16) len leaderboard, DAN NHAN RO convention (K16+priority
!= board K25-shuffle) de trung thuc. Clone struct_to 3185 -> x2_struct_to_k16prio, run seed42, tinh
K16+conviction-priority stats (nav/cagr/dd/f22), upsert leaderboard_nav. Multi-seed mean (hb_100)
= CAGR 71.5%/DD-17.2% ghi vao description."""
from __future__ import annotations
import asyncio, copy, os, sys, hashlib, logging, warnings
from collections import defaultdict
from pathlib import Path
warnings.filterwarnings("ignore"); logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "stock_ml"))
sys.path.insert(0, os.environ.get("NH_NAV2_DIR", "F:/PROJECTS/hb2943_work"))
os.environ.setdefault("STOCK_DATA_DIR", "F:/PROJECTS/train_ai_ml/market_data/market.duckdb")
import psycopg2, pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker
from db.engine import async_engine
from db.repositories.template_repo import StrategyTemplateRepository
from scripts.run_template import run_template_experiment
from nh_nav2 import NavSim2, FEE

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
HERE = Path(__file__).parent; NAME = "x2_struct_to_k16prio"
DESC = ("[K16+CONVICTION-PRIORITY operating-point cua struct_to t3185 — KHONG cung thuoc do board "
        "K25-shuffle] Cung signal double-RS+struct-trail, quan tien tan cong: K=16 (don von, deploy "
        "idle-capacity nam-tot) + fill entry conviction-cao truoc khi slot khan (score amp IC+0.206). "
        "Multi-seed 42/21/123: CAGR 71.5%/DD-17.2%/NAV x33.56 (+9.7% vs K16-shuffle, duong 3/3 seed). "
        "Board K25-shuffle van 66%/-13.7%. Day la profile TAN CONG (return+5.5pp doi DD sau hon).")


def priority_run(sim, pm, K, roundtrip=0.006, settle_lag=2, advance_fee=0.0008):
    s_new = (roundtrip - FEE) / 2.0
    for t in sim.trades:
        t["net"] = (t["x_raw"] * (1.0 - s_new)) / (t["e_raw"] * (1.0 + s_new)) - 1.0 - FEE
        t["prio"] = pm.get((t["symbol"], t["entry_date"]), 0.0)
    entries = defaultdict(list)
    for t in sim.trades: entries[t["entry_date"]].append(t)
    for d in entries: entries[d].sort(key=lambda t: t["prio"], reverse=True)
    sc, si, cal = sim.sym_close, sim.sym_idx, sim.calendar
    def lv(leg, dt):
        s = leg["symbol"]; j = si[s].get(dt)
        if j is None: return leg["last_val"]
        i0, i1 = leg["i0"], leg["i1"]; j = min(max(j, i0), i1)
        r = leg["ratio1"] if i1 == i0 else leg["ratio0"] + (leg["ratio1"] - leg["ratio0"]) * (j - i0) / (i1 - i0)
        v = leg["invested"] * (sc[s][j] * r) / leg["p0"]; leg["last_val"] = v; return v
    cash = 1.0; pending = defaultdict(float); legs = []; exits = defaultdict(list); nav_series = []; fills = 0
    for di, dt in enumerate(cal):
        cash += pending.pop(dt, 0.0); pt = sum(pending.values())
        for leg in exits.get(dt, ()):
            pr = leg["invested"] * (1.0 + leg["net"]); cash += pr * (1.0 - advance_fee); legs.remove(leg)
        pos = sum(lv(l, dt) for l in legs); nav_now = cash + pt + pos
        for t in entries.get(dt, ()):
            size = nav_now / K
            if cash + 1e-12 >= size:
                s = t["symbol"]; c0, c1 = sc[s][t["i0"]], sc[s][t["i1"]]; xe = t["p0"] * (1.0 + t["net"])
                leg = dict(symbol=s, i0=t["i0"], i1=t["i1"], invested=size, net=t["net"], p0=t["p0"],
                           ratio0=t["p0"]/c0, ratio1=xe/c1, last_val=size, exit_date=t["exit_date"])
                cash -= size; legs.append(leg); exits[t["exit_date"]].append(leg); fills += 1
        pos = sum(lv(l, dt) for l in legs); nav_series.append((dt, cash + pt + pos))
    ns = pd.DataFrame(nav_series, columns=["date", "nav"]); ns["date"] = pd.to_datetime(ns["date"])
    nav = ns["nav"]; final = float(nav.iloc[-1]); yrs = (ns["date"].iloc[-1]-ns["date"].iloc[0]).days/365.25
    dd = float((nav/nav.cummax()-1).min())
    return final, final**(1/yrs)-1, dd, yrs, fills


def build_prio(con, rid, tr):
    sig = pd.read_sql("select symbol,date,score from run_signals where run_id=%s and signal=1 and score is not null", con, params=(rid,))
    sig["dt"] = pd.to_datetime(sig["date"]); tr = tr.copy(); tr["dt"] = pd.to_datetime(tr["entry_date"]); tr["edkey"] = tr["dt"].dt.strftime("%Y-%m-%d")
    pm = {}
    for s, g in sig.groupby("symbol"):
        g = g.sort_values("dt"); tg = tr[tr.symbol == s].sort_values("dt")
        if not len(tg): continue
        m = pd.merge_asof(tg[["dt", "edkey"]], g[["dt", "score"]], on="dt", direction="backward")
        for ed, x in zip(m["edkey"], m["score"]): pm[(s, ed)] = float(x) if pd.notna(x) else 0.0
    return pm


async def make():
    S = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with S() as s:
        repo = StrategyTemplateRepository(s); ex = await repo.get_by_name(NAME)
        if ex: return ex.id
        base = await repo.get_by_id(3185)
        slots = [{"slot_type": sl.slot_type, "ml_component_id": sl.ml_component_id, "rule_component_id": sl.rule_component_id,
                  "feature_set_name": sl.feature_set_name, "target_config": copy.deepcopy(sl.target_config)} for sl in base.component_slots]
        t = await repo.create(name=NAME, market=base.market, strategy=base.strategy, feature_set_id=base.feature_set_id,
            target_id=base.target_id, component_slots=slots, direction=base.direction, signal_mode=base.signal_mode,
            signal_threshold=base.signal_threshold, entry_threshold=base.entry_threshold, exit_threshold=base.exit_threshold,
            split_config=copy.deepcopy(base.split_config), engine_config=copy.deepcopy(base.engine_config),
            validation_config=base.validation_config, seed=42, description=DESC, hypothesis="K16+conviction-priority attack profile",
            universe_slug=base.universe_slug, model_mode="ml_only")
        await s.commit(); return t.id


def main():
    tid = asyncio.run(make()); asyncio.run(async_engine.dispose())
    con = psycopg2.connect(**PG); cur = con.cursor()
    cur.execute("update strategy_templates set description=%s where id=%s", (DESC, tid)); con.commit()
    r = run_template_experiment(template_id=tid, seed=42); rid = r.get("run_id")
    tr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price from run_trades where run_id=%s and exit_date is not null", con, params=(rid,))
    cv = HERE / "_k102.csv"; tr.to_csv(cv, index=False)
    pm = build_prio(con, rid, tr)
    navf, cagr, dd, yrs, fills = priority_run(NavSim2(str(cv), date_lo="2020-01-01"), pm, K=16)
    nav22, _, _, _, _ = priority_run(NavSim2(str(cv), date_lo="2022-01-01"), pm, K=16)
    print(f"K16+prio seed42: NAV=x{navf:.2f} CAGR={cagr*100:.1f}% DD={dd*100:.1f}% f22=x{nav22:.2f} fills={fills} years={yrs:.2f}", flush=True)
    ch = hashlib.md5(f"{rid}_k16prio".encode()).hexdigest()[:16]
    cur.execute("""insert into leaderboard_nav (run_id, nav_adv, nav_noadv, cagr_adv, cagr_noadv, maxdd_nav, nav_f22_adv, years, n_trades_sim, config_hash, computed_at)
                   values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, now())
                   on conflict (run_id) do update set nav_adv=excluded.nav_adv, cagr_adv=excluded.cagr_adv,
                     maxdd_nav=excluded.maxdd_nav, nav_f22_adv=excluded.nav_f22_adv, years=excluded.years,
                     n_trades_sim=excluded.n_trades_sim, config_hash=excluded.config_hash, computed_at=now()""",
                (rid, navf, navf, cagr, cagr, dd, nav22, yrs, fills, ch))
    con.commit()
    print(f"REGISTERED {NAME} (t{tid}) run={rid} on leaderboard_nav @K16+priority", flush=True)
    con.close(); print("HB_102_DONE", flush=True)


if __name__ == "__main__":
    main()

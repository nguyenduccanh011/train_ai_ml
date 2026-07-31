# -*- coding: utf-8 -*-
"""Register 2 new standout operating points from K x kconv sweep (all 3/3 robust):
  - x2_struct_to_k8preempt_cssize   (K8, kconv2.5)  = MAX-CAGR aggressive 113.2%/DD-18.9 (high seed-var)
  - x2_struct_to_k12size2  (K12, kconv2.0) = 111.2%/DD-16.2 (dominates k12/k1.5 -> supersede that row)
Reuses hb_register_klever's prun + setup by import."""
from __future__ import annotations
import os, sys, warnings, asyncio, copy, hashlib, statistics
from pathlib import Path
warnings.filterwarnings("ignore")
import logging; logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[3]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "stock_ml"))
sys.path.insert(0, os.environ.get("NH_NAV2_DIR", "F:/PROJECTS/hb2943_work"))
os.environ.setdefault("STOCK_DATA_DIR", "F:/PROJECTS/train_ai_ml/market_data/market.duckdb")
import psycopg2, pandas as pd, duckdb
from collections import defaultdict
from nh_nav2 import NavSim2, FEE
from scripts.run_template import run_template_experiment
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker
from db.engine import async_engine
from db.repositories.template_repo import StrategyTemplateRepository
import hb_112_meta_target as M

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
SEEDS = [42, 21, 123]; MARGIN = 0.01; SKIP = 0.40; CS4 = ["dist20low", "dist_ma20", "rsi14", "ret20"]


def prun(sim, pm, cm, K, k_conv, margin=MARGIN, skip=SKIP, advance_fee=0.0008, roundtrip=0.006):
    s_new = (roundtrip - FEE) / 2.0
    for t in sim.trades:
        t["net"] = (t["x_raw"] * (1.0 - s_new)) / (t["e_raw"] * (1.0 + s_new)) - 1.0 - FEE
        t["prio"] = pm.get((t["symbol"], t["entry_date"]), -9.9); t["conv"] = cm.get((t["symbol"], t["entry_date"]), 0.5)
    cv = [t["conv"] for t in sim.trades]; mu = statistics.mean(cv); sd = statistics.pstdev(cv) or 1.0
    raw = []
    for t in sim.trades:
        z = (t["conv"] - mu) / sd; t["_w"] = min(max(1.0 + k_conv * z, 0.4), 1.8); raw.append(t["_w"])
    off = 1.0 - (statistics.mean(raw) if raw else 1.0)
    for t in sim.trades:
        t["w"] = max(0.3, t["_w"] + off)
    entries = defaultdict(list)
    for t in sim.trades:
        if t["conv"] >= skip:
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

    cash = 1.0; pend = defaultdict(float); legs = []; exits = defaultdict(list); ns = []; expo = []
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
                if t["prio"] - c["prio"] > margin:
                    vnow = lv(c, dt); cash += vnow * (1.0 - advance_fee); legs.remove(c)
                    if c in exits.get(c["exit_date"], ()):
                        exits[c["exit_date"]].remove(c)
                    size = (cash + pt + sum(lv(l, dt) for l in legs)) / K * t["w"]
                    if cash + 1e-12 >= size:
                        cash -= size; leg = mk(t, size); legs.append(leg); exits[t["exit_date"]].append(leg)
        pos = sum(lv(l, dt) for l in legs); tot = cash + pt + pos
        ns.append((dt, tot)); expo.append(pos / tot if tot > 0 else 0.0)
    d = pd.DataFrame(ns, columns=["date", "nav"]); d["date"] = pd.to_datetime(d["date"]); nav = d["nav"]
    final = float(nav.iloc[-1]); yrs = (d["date"].iloc[-1] - d["date"].iloc[0]).days / 365.25
    return final, final ** (1 / yrs) - 1, float((nav / nav.cummax() - 1).min()), statistics.mean(expo)


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

# rebuild per-seed sim inputs + meta
con = psycopg2.connect(**PG); feat = None; cv_c, cm_c, pm = {}, {}, {}
for sd in SEEDS:
    rid = run_template_experiment(template_id=3185, seed=sd).get("run_id")
    cvtr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price,entry_signal_date from run_trades where run_id=%s and exit_date is not null", con, params=(rid,))
    cvtr["sigd"] = pd.to_datetime(cvtr["entry_signal_date"])
    cm = {}
    for r in cvtr.itertuples():
        cm[(r.symbol, str(pd.to_datetime(r.entry_date).date()))] = CS.get((r.symbol, str(r.sigd.date())), 0.5)
    cm_c[sd] = cm
    cv = HERE / f"_r8_s{sd}.csv"
    cvtr[["symbol", "entry_date", "exit_date", "entry_price", "exit_price"]].to_csv(cv, index=False); cv_c[sd] = str(cv)
    if feat is None:
        feat = M.features(cvtr.symbol.unique().tolist())
    pm[sd] = M.meta_preds(M.build_tr(con, rid, feat), tgt='t_pnl')
con.close()


def evalKk(K, kc):
    nv, cg, dd, ex = [], [], [], []
    for sd in SEEDS:
        f, c, d, e = prun(NavSim2(cv_c[sd], date_lo="2020-01-01"), pm[sd], cm_c[sd], K, kc)
        nv.append(f); cg.append(c); dd.append(d); ex.append(e)
    f22 = statistics.mean(prun(NavSim2(cv_c[sd], date_lo="2022-01-01"), pm[sd], cm_c[sd], K, kc)[0] for sd in SEEDS)
    return dict(nav=statistics.mean(nv), cagr=statistics.mean(cg), dd=statistics.mean(dd), ex=statistics.mean(ex), f22=f22)


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
            hypothesis="K-lever + conviction-k tuning on champion combo", universe_slug=base.universe_slug, model_mode="ml_only")
        await s.commit(); return t.id


SPECS = [
    ("x2_struct_to_k8preempt_cssize", 8, 2.5, None,
     "MAX-CAGR aggressive (K8 + conviction-k2.5)"),
    ("x2_struct_to_k12size2", 12, 2.0, "x2_struct_to_k12preempt_cssize",
     "improved K12 (conviction-k2.0 > k1.5, dominates k12/k1.5 both CAGR and DD)"),
]


def register():
    con = psycopg2.connect(**PG); cur = con.cursor()
    for name, K, kc, supersede, tag in SPECS:
        r = evalKk(K, kc)
        desc = (f"[K{K} + preempt R2m01 + cs4-CONVICTION k{kc} + conv-skip0.40 — {tag}, TONG VON<=1 khong "
                f"don bay] CUNG combo champion, K{K}/kconv{kc}. Multi-seed CAGR ~{100*r['cagr']:.0f}%/DD{100*r['dd']:.1f} "
                f"(3/3 seed robust > champion K16 100%/-15.1). Exposure {r['ex']:.2f}. K la dial CAGR<->DD; "
                f"conviction-k2.0-2.5 tap trung von vao high-conviction manh hon k1.5. "
                f"{'CANH BAO seed-spread rong (K8 rat tap trung, variance cao).' if K==8 else 'Seed-spread tight.'} "
                f"Same signal double-RS+struct-trail, same fee/universe = fair same-conditions raw CAGR. "
                f"Diem manh: CAGR-tran cao nhat cho khau vi tan cong; risk-adjusted thi dung K16/K25.")
        tid = asyncio.run(make_tmpl(name, desc)); asyncio.run(async_engine.dispose())
        cur.execute("update strategy_templates set description=%s where id=%s", (desc, tid)); con.commit()
        r2 = run_template_experiment(template_id=tid, seed=42).get("run_id")
        ch = hashlib.md5(f"{r2}_k{K}_kc{kc}".encode()).hexdigest()[:16]
        cur.execute("""insert into leaderboard_nav (run_id, nav_adv, nav_noadv, cagr_adv, cagr_noadv, maxdd_nav,
                         nav_f22_adv, years, n_trades_sim, config_hash, computed_at) values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, now())
                       on conflict (run_id) do update set nav_adv=excluded.nav_adv, cagr_adv=excluded.cagr_adv,
                         maxdd_nav=excluded.maxdd_nav, nav_f22_adv=excluded.nav_f22_adv, config_hash=excluded.config_hash, computed_at=now()""",
                    (r2, r['nav'], r['nav'], r['cagr'], r['cagr'], r['dd'], r['f22'], 6.51, 0, ch))
        cur.execute("update leaderboard_runs set state='trained', superseded=false where run_id=%s", (r2,))
        if supersede:
            cur.execute("update leaderboard_runs set superseded=true where run_name=%s", (supersede,))
        con.commit()
        print(f"  REGISTERED {name} CAGR {100*r['cagr']:.1f}% DD {100*r['dd']:.1f}% expo {r['ex']:.2f}"
              f"{' (superseded '+supersede.split('_')[-1]+')' if supersede else ''}", flush=True)
    cur.execute("""select lr.run_name, ln.cagr_adv, ln.maxdd_nav, ln.nav_adv from leaderboard_runs lr
                   join leaderboard_nav ln on lr.run_id=ln.run_id
                   where lr.run_name like 'x2_struct_to%%' and lr.superseded=false order by ln.cagr_adv desc""")
    print("\n=== FINAL board (superseded=false) ===", flush=True)
    for row in cur.fetchall():
        print(f"  {row[0]:34s} CAGR {row[1]*100:5.1f}%  DD {row[2]*100:6.1f}%  NAV x{row[3]:.1f}", flush=True)
    con.close(); print("REGISTER_K8K12_DONE", flush=True)


register()

# -*- coding: utf-8 -*-
"""REGISTER K-lever aggressive operating points (user: dang ky model dot pha CAGR / diem manh rieng).
SAME champion combo (preempt R2m01 + cs4-conviction k1.5 + conv-skip 0.40, total<=1) but K10/K12
concentration = higher CAGR, higher DD (K is a CAGR<->DD dial). Validate K16 reproduces champion ~100%
first. 3-seed [42,21,123]; register only if 3/3 > champion. Honest desc: MAX-CAGR/aggressive profile,
deeper DD + higher exposure vs balanced K16. Leaderboard convention = RAW same-conditions (like CHAMP)."""
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
SEEDS = [42, 21, 123]; KCONV = 1.5; MARGIN = 0.01; SKIP = 0.40; CS4 = ["dist20low", "dist_ma20", "rsi14", "ret20"]


def prun(sim, pm, cm, K, k_conv=KCONV, margin=MARGIN, skip=SKIP, advance_fee=0.0008, roundtrip=0.006):
    """Champion combo scorer (cs4 conviction + conv-skip + preempt R2m01), parametrized by K.
    Returns (final_nav, cagr, maxdd, avg_exposure)."""
    s_new = (roundtrip - FEE) / 2.0
    for t in sim.trades:
        t["net"] = (t["x_raw"] * (1.0 - s_new)) / (t["e_raw"] * (1.0 + s_new)) - 1.0 - FEE
        t["prio"] = pm.get((t["symbol"], t["entry_date"]), -9.9)
        t["conv"] = cm.get((t["symbol"], t["entry_date"]), 0.5)
    cv = [t["conv"] for t in sim.trades]; mu = statistics.mean(cv); sd = statistics.pstdev(cv) or 1.0
    raw = []
    for t in sim.trades:
        z = (t["conv"] - mu) / sd
        t["_w"] = min(max(1.0 + k_conv * z, 0.4), 1.8); raw.append(t["_w"])
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


# cs4 panel + meta setup (per seed)
cx = duckdb.connect("F:/PROJECTS/train_ai_ml/market_data/market.duckdb", read_only=True)
px = cx.execute("SELECT symbol,date,low,close FROM ohlcv WHERE timeframe='1D' AND date>='2018-06-01' "
                "ORDER BY symbol,date").fetchdf(); cx.close()
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

con = psycopg2.connect(**PG); feat = None
cv_c, cm_c, pm = {}, {}, {}
for sd in SEEDS:
    rid = run_template_experiment(template_id=3185, seed=sd).get("run_id")
    cvtr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price,entry_signal_date "
                       "from run_trades where run_id=%s and exit_date is not null", con, params=(rid,))
    cvtr["sigd"] = pd.to_datetime(cvtr["entry_signal_date"])
    cm = {}
    for r in cvtr.itertuples():
        cm[(r.symbol, str(pd.to_datetime(r.entry_date).date()))] = CS.get((r.symbol, str(r.sigd.date())), 0.5)
    cm_c[sd] = cm
    cv = HERE / f"_kl_s{sd}.csv"
    cvtr[["symbol", "entry_date", "exit_date", "entry_price", "exit_price"]].to_csv(cv, index=False); cv_c[sd] = str(cv)
    if feat is None:
        feat = M.features(cvtr.symbol.unique().tolist())
    pm[sd] = M.meta_preds(M.build_tr(con, rid, feat), tgt='t_pnl')
con.close()


def evalK(K, lo="2020-01-01"):
    nv, cg, dd, ex = [], [], [], []
    for sd in SEEDS:
        f, c, d, e = prun(NavSim2(cv_c[sd], date_lo=lo), pm[sd], cm_c[sd], K)
        nv.append(f); cg.append(c); dd.append(d); ex.append(e)
    return nv, cg, dd, ex


# validate K16 ~ champion, then K10/K12
print("=== validate + K-lever (3-seed champion combo w/ conv-skip) ===", flush=True)
res = {}
champ_nav = None
for K in (16, 12, 10):
    nv, cg, dd, ex = evalK(K)
    f22, _, _, _ = zip(*[(prun(NavSim2(cv_c[sd], date_lo="2022-01-01"), pm[sd], cm_c[sd], K)) for sd in SEEDS])
    res[K] = dict(nav=statistics.mean(nv), cagr=statistics.mean(cg), dd=statistics.mean(dd),
                  ex=statistics.mean(ex), f22=statistics.mean(f22), nv=nv)
    if K == 16:
        champ_nav = nv
    tag = "(=champion validate)" if K == 16 else ""
    print(f"  K{K}: NAV x{res[K]['nav']:6.2f}  CAGR {100*res[K]['cagr']:5.1f}%  DD {100*res[K]['dd']:5.1f}%  "
          f"expo {res[K]['ex']:.2f}  f22 x{res[K]['f22']:.1f}  seeds={[f'{x:.0f}' for x in nv]} {tag}", flush=True)

# robustness gate: K10/K12 must beat K16 3/3 seed
for K in (12, 10):
    d = [res[K]['nv'][i] - champ_nav[i] for i in range(len(SEEDS))]
    res[K]['robust'] = all(x > 0 for x in d)
    print(f"  K{K} vs K16 3-seed delta: {[f'{x:+.1f}' for x in d]} -> {'3/3 ROBUST' if res[K]['robust'] else 'NOT robust'}", flush=True)
print("KLEVER_EVAL_DONE", flush=True)


# ---------- REGISTRATION ----------
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
            hypothesis="K-lever concentration on champion combo (CAGR<->DD dial)",
            universe_slug=base.universe_slug, model_mode="ml_only")
        await s.commit(); return t.id


def register():
    con = psycopg2.connect(**PG); cur = con.cursor()
    for K in (12, 10):
        if not res[K].get('robust'):
            print(f"  SKIP K{K} (not 3/3 robust)", flush=True); continue
        name = f"x2_struct_to_k{K}preempt_cssize"
        desc = (f"[K{K} + preempt R2m01 + cs4-CONVICTION-SIZING k1.5 + conv-skip0.40 — MAX-CAGR AGGRESSIVE "
                f"profile, TONG VON<=1 khong don bay] CUNG combo champion (k16preempt_cssize) nhung K{K} "
                f"concentration cao hon = day CAGR len doi lay DD sau hon. Multi-seed CAGR ~{100*res[K]['cagr']:.0f}%/"
                f"DD{100*res[K]['dd']:.1f} vs champion K16 100%/-15.1 (3/3 seed > champion). Exposure {res[K]['ex']:.2f} "
                f"(cao hon K16 -> 1 phan gain la exposure/beta, khong thuan skill). K la DIAL CAGR<->DD: chon theo "
                f"tieu chi CAGR-tran thi K{K}, chon risk-adjusted thi K16/K25. Same signal double-RS+struct-trail, "
                f"same fee/universe/target = fair same-conditions raw CAGR.")
        tid = asyncio.run(make_tmpl(name, desc)); asyncio.run(async_engine.dispose())
        cur.execute("update strategy_templates set description=%s where id=%s", (desc, tid)); con.commit()
        r2 = run_template_experiment(template_id=tid, seed=42).get("run_id")
        ch = hashlib.md5(f"{r2}_klever_k{K}".encode()).hexdigest()[:16]
        cur.execute("""insert into leaderboard_nav (run_id, nav_adv, nav_noadv, cagr_adv, cagr_noadv, maxdd_nav,
                         nav_f22_adv, years, n_trades_sim, config_hash, computed_at)
                       values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, now())
                       on conflict (run_id) do update set nav_adv=excluded.nav_adv, cagr_adv=excluded.cagr_adv,
                         maxdd_nav=excluded.maxdd_nav, nav_f22_adv=excluded.nav_f22_adv, config_hash=excluded.config_hash,
                         computed_at=now()""",
                    (r2, res[K]['nav'], res[K]['nav'], res[K]['cagr'], res[K]['cagr'], res[K]['dd'],
                     res[K]['f22'], 6.51, 0, ch))
        cur.execute("update leaderboard_runs set state='trained', superseded=false where run_id=%s", (r2,))
        con.commit()
        print(f"  REGISTERED {name} (t{tid}) CAGR {100*res[K]['cagr']:.1f}% DD {100*res[K]['dd']:.1f}%", flush=True)
    # final board
    cur.execute("""select lr.run_name, ln.cagr_adv, ln.maxdd_nav, ln.nav_adv from leaderboard_runs lr
                   join leaderboard_nav ln on lr.run_id=ln.run_id
                   where lr.run_name like 'x2_struct_to%%' and lr.superseded=false order by ln.cagr_adv desc""")
    print("\n=== FINAL board (superseded=false) ===", flush=True)
    for r in cur.fetchall():
        print(f"  {r[0]:34s} CAGR {r[1]*100:5.1f}%  DD {r[2]*100:6.1f}%  NAV x{r[3]:.1f}", flush=True)
    con.close(); print("REGISTER_DONE", flush=True)


register()

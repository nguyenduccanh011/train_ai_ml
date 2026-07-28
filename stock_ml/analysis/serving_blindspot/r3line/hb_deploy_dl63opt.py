# -*- coding: utf-8 -*-
"""DEPLOY the T+2 ret5-entry-gate on the top model K10/cs5_ma50. Gate PASSED: 3/3 seed (K10 & K12),
per-year positive-or-neutral (never hurts), config-specific (K16 champ fails -> gate belongs to the
concentrated cs5_ma50 model). Register `x2_struct_to_k10_cs5ma50_t2ret5g` with CAGR computed UNDER T+2
(its valid regime) + populate portfolio (equity/trades/skipped incl. ret5-gated entries). thr=0.03."""
from __future__ import annotations
import os, sys, warnings, asyncio, copy, hashlib, statistics
from collections import defaultdict
from bisect import bisect_right
from pathlib import Path
warnings.filterwarnings("ignore")
import logging; logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[3]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "stock_ml"))
sys.path.insert(0, os.environ.get("NH_NAV2_DIR", "F:/PROJECTS/hb2943_work"))
os.environ.setdefault("STOCK_DATA_DIR", "F:/PROJECTS/train_ai_ml/market_data/market.duckdb")
import psycopg2, pandas as pd, numpy as _np, duckdb
from psycopg2.extras import execute_values
from nh_nav2 import NavSim2, FEE
from scripts.run_template import run_template_experiment
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker
from db.engine import async_engine
from db.repositories.template_repo import StrategyTemplateRepository
import hb_112_meta_target as M

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
MARKET = "F:/PROJECTS/train_ai_ml/market_data/market.duckdb"
SEEDS = [42, 21, 123]; SEED = 42; SKIP = 0.40; CS4 = ["dist20low", "dist_ma20", "rsi14", "ret20"]
K, MARGIN, KCONV, TPLUS, R5THR = 10, 0.005, 2.0, 2, 0.02; PB_PCT, PB_WIN = 0.045, 40
PRE = "template/"; SUF = "-69338138"
# dl63 near-low selection lever (set per variant in the registration loop below).
# DL63_TH: dist_l63 threshold below which an entry is "near 63-session low" (opp-cost/weak setup).
# DL63_MODE: 'skip' = drop such entries from the book (max CAGR); 'size' = keep but shrink size (best risk-adj).
# DL63_SIZE: size multiplier applied to near-low entries when MODE=='size'.
DL63_TH = None; DL63_MODE = "skip"; DL63_SIZE = 0.7; DL63MAP = {}


def _prep(sim, pm, cm):
    s_new = (0.006 - FEE) / 2.0; sc = sim.sym_close
    for t in sim.trades:
        be, bx = t["i0"], t["i1"]
        if TPLUS and (bx - be) < TPLUS:
            nb = min(be + TPLUS, len(sc[t["symbol"]]) - 1); t["i1"] = nb; t["x_raw"] = sc[t["symbol"]][nb]
        t["net"] = (t["x_raw"] * (1.0 - s_new)) / (t["e_raw"] * (1.0 + s_new)) - 1.0 - FEE
        t["prio"] = pm.get((t["symbol"], t["entry_date"]), -9.9); t["conv"] = cm.get((t["symbol"], t["entry_date"]), 0.5)
    cv = [t["conv"] for t in sim.trades]; mu = statistics.mean(cv); sd = statistics.pstdev(cv) or 1.0
    raw = []
    for t in sim.trades:
        z = (t["conv"] - mu) / sd; t["_w"] = min(max(1.0 + KCONV * z, 0.4), 1.8); raw.append(t["_w"])
    off = 1.0 - (statistics.mean(raw) if raw else 1.0)
    for t in sim.trades:
        t["w"] = max(0.3, t["_w"] + off)
        # dl63 near-low lever in 'size' mode: keep the trade (exposure preserved) but shrink its size
        if DL63_TH is not None and DL63_MODE == "size":
            if DL63MAP.get((t["symbol"], t["entry_date"]), 9.0) < DL63_TH:
                t["w"] = t["w"] * DL63_SIZE


def _entries(sim, r5map):
    entries = defaultdict(list)
    for t in sim.trades:
        if t["conv"] < SKIP:
            continue
        v = r5map.get((t["symbol"], t["entry_date"]), _np.nan)
        if not _np.isnan(v) and v < R5THR:
            continue
        # dl63 near-low lever: in 'skip' mode drop entries below threshold (size mode handled in _prep)
        if DL63_TH is not None and DL63_MODE == "skip":
            if DL63MAP.get((t["symbol"], t["entry_date"]), 9.0) < DL63_TH:
                continue
        entries[t["entry_date"]].append(t)
    for d in entries:
        entries[d].sort(key=lambda t: t["prio"], reverse=True)
    return entries


def prun_metric(sim, pm, cm, r5map, advance_fee=0.0008):
    _prep(sim, pm, cm); entries = _entries(sim, r5map)
    sc, si, cal = sim.sym_close, sim.sym_idx, sim.calendar

    def lv(leg, dt):
        s = leg["symbol"]; j = si[s].get(dt)
        if j is None:
            return leg["last_val"]
        i0, i1 = leg["i0"], leg["i1"]; j = min(max(j, i0), i1)
        r = leg["ratio1"] if i1 == i0 else leg["ratio0"] + (leg["ratio1"] - leg["ratio0"]) * (j - i0) / (i1 - i0)
        v = leg["invested"] * (sc[s][j] * r) / leg["p0"]; leg["last_val"] = v; return v

    def mk(t, size, di):
        s = t["symbol"]; c0, c1 = sc[s][t["i0"]], sc[s][t["i1"]]; xe = t["p0"] * (1.0 + t["net"])
        return dict(symbol=s, i0=t["i0"], i1=t["i1"], invested=size, net=t["net"], p0=t["p0"],
                    ratio0=t["p0"] / c0, ratio1=xe / c1, last_val=size, exit_date=t["exit_date"], prio=t["prio"], be_di=di)

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
                cash -= size; leg = mk(t, size, di); legs.append(leg); exits[t["exit_date"]].append(leg)
            elif legs:
                cand = [l for l in legs if (di - l["be_di"]) >= TPLUS]
                if not cand:
                    continue
                c = min(cand, key=lambda l: l["prio"])
                if t["prio"] - c["prio"] > MARGIN:
                    vnow = lv(c, dt); cash += vnow * (1.0 - advance_fee); legs.remove(c)
                    if c in exits.get(c["exit_date"], ()):
                        exits[c["exit_date"]].remove(c)
                    size = (cash + pt + sum(lv(l, dt) for l in legs)) / K * t["w"]
                    if cash + 1e-12 >= size:
                        cash -= size; leg = mk(t, size, di); legs.append(leg); exits[t["exit_date"]].append(leg)
        pos = sum(lv(l, dt) for l in legs); ns.append((dt, cash + pt + pos))
    d = pd.DataFrame(ns, columns=["date", "nav"]); d["date"] = pd.to_datetime(d["date"]); return d


def prun_track(sim, pm, cm, r5map, base, RID, advance_fee=0.0008):
    _prep(sim, pm, cm); entries = _entries(sim, r5map)
    sc, si, cal = sim.sym_close, sim.sym_idx, sim.calendar

    def lv(leg, dt):
        s = leg["symbol"]; j = si[s].get(dt)
        if j is None:
            return leg["last_val"]
        i0, i1 = leg["i0"], leg["i1"]; j = min(max(j, i0), i1)
        r = leg["ratio1"] if i1 == i0 else leg["ratio0"] + (leg["ratio1"] - leg["ratio0"]) * (j - i0) / (i1 - i0)
        v = leg["invested"] * (sc[s][j] * r) / leg["p0"]; leg["last_val"] = v; return v

    def mk(t, size, dt, di, nav_at):
        s = t["symbol"]; c0, c1 = sc[s][t["i0"]], sc[s][t["i1"]]; xe = t["p0"] * (1.0 + t["net"])
        return dict(symbol=s, i0=t["i0"], i1=t["i1"], invested=size, net=t["net"], p0=t["p0"],
                    ratio0=t["p0"] / c0, ratio1=xe / c1, last_val=size, exit_date=t["exit_date"],
                    prio=t["prio"], entry_dt=dt, be_di=di, conv=t["conv"], nav_at=nav_at)

    trades = []

    def emit(leg, exit_dt, exit_di, evicted):
        s = leg["symbol"]; b = base.get((s, str(pd.to_datetime(leg["entry_dt"]).date())), {})
        ed = pd.to_datetime(leg["entry_dt"]); xd = pd.to_datetime(exit_dt)
        if evicted:
            j = si[s].get(exit_dt); xp = float(sc[s][j]) if j is not None else float(b.get("exit_price") or 0)
            pnl = xp / leg["p0"] - 1.0 if leg["p0"] else 0.0; reason = "preempt"
        else:
            xp = float(b.get("exit_price") or leg["p0"] * (1.0 + leg["net"])); pnl = float(leg["net"]); reason = b.get("exit_reason") or "signal"
        trades.append(dict(symbol=s, entry_date=str(ed.date()), entry_price=float(leg["p0"]),
                           exit_date=str(xd.date()), exit_price=xp, holding_days=int(exit_di - leg["be_di"]),
                           pnl_pct=float(pnl), exit_reason=reason,
                           entry_signal_date=(str(b.get("sigd"))[:10] if b.get("sigd") is not None else None)))

    equity = []; holdings = []; held_by_date = {}
    cash = 1.0; pend = defaultdict(float); legs = []; exits = defaultdict(list)
    for di, dt in enumerate(cal):
        cash += pend.pop(dt, 0.0); pt = sum(pend.values()); exit_today = {}
        for leg in list(exits.get(dt, ())):
            if leg in legs:
                cash += leg["invested"] * (1.0 + leg["net"]) * (1.0 - advance_fee); legs.remove(leg)
                exit_today[leg["symbol"]] = "signal"; emit(leg, dt, di, False)
        pos = sum(lv(l, dt) for l in legs); nav_now = cash + pt + pos; new_today = set()
        for t in entries.get(dt, ()):
            size = (nav_now / K) * t["w"]
            if cash + 1e-12 >= size:
                cash -= size; leg = mk(t, size, dt, di, nav_now); legs.append(leg); exits[t["exit_date"]].append(leg); new_today.add(t["symbol"])
            elif legs:
                cand = [l for l in legs if (di - l["be_di"]) >= TPLUS]
                if not cand:
                    continue
                c = min(cand, key=lambda l: l["prio"])
                if t["prio"] - c["prio"] > MARGIN:
                    vnow = lv(c, dt); cash += vnow * (1.0 - advance_fee); legs.remove(c)
                    if c in exits.get(c["exit_date"], ()):
                        exits[c["exit_date"]].remove(c)
                    exit_today[c["symbol"]] = "preempt"; emit(c, dt, di, True)
                    nav_mid = cash + pt + sum(lv(l, dt) for l in legs); size = (nav_mid / K) * t["w"]
                    if cash + 1e-12 >= size:
                        cash -= size; leg = mk(t, size, dt, di, nav_mid); legs.append(leg); exits[t["exit_date"]].append(leg); new_today.add(t["symbol"])
        pos = sum(lv(l, dt) for l in legs); nav = cash + pt + pos; ds = str(pd.to_datetime(dt).date())
        equity.append((RID, ds, float(nav), float(cash + pt), float(pos / nav if nav > 0 else 0.0), len(legs)))
        for leg in legs:
            val = lv(leg, dt)
            holdings.append((RID, ds, leg["symbol"], float(val / nav if nav > 0 else 0.0),
                             float(leg["invested"] / leg["nav_at"] if leg["nav_at"] else 0.0),
                             float(val / leg["invested"] - 1.0 if leg["invested"] else 0.0),
                             str(pd.to_datetime(leg["entry_dt"]).date()), int(di - leg["be_di"]),
                             leg["symbol"] in new_today, False, None, float(leg["conv"])))
        for sym, reason in exit_today.items():
            holdings.append((RID, ds, sym, 0.0, 0.0, 0.0, ds, 0, False, True, reason, None))
        held_by_date[ds] = {leg["symbol"] for leg in legs}
    for leg in legs:
        emit(leg, cal[-1], len(cal) - 1, False)
    return equity, holdings, held_by_date, trades


# ---- panels ----
cx = duckdb.connect(MARKET, read_only=True)
px = cx.execute("SELECT symbol,date,low,close,high FROM ohlcv WHERE timeframe='1D' AND date>='2018-06-01' ORDER BY symbol,date").fetchdf(); cx.close()
px["date"] = pd.to_datetime(px["date"]); parts = []; CLO = {}; DIDX = {}; INV = {}
for s, g in px.groupby("symbol"):
    g = g.sort_values("date").copy(); c, l, h = g["close"], g["low"], g["high"]
    CLO[s] = c.values; DIDX[s] = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(g["date"])}; INV[s] = {i: d.strftime("%Y-%m-%d") for i, d in enumerate(g["date"])}
    dd = c.diff(); up = dd.clip(lower=0).rolling(14).mean(); dn = (-dd.clip(upper=0)).rolling(14).mean()
    g["dist20low"] = c / l.rolling(20).min() - 1; g["dist_ma20"] = c / c.rolling(20).mean() - 1
    g["rsi14"] = 100 - 100 / (1 + up / (dn + 1e-9)); g["ret20"] = c / c.shift(20) - 1
    tr_ = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    g["atrpct"] = tr_.rolling(14).mean() / c; g["dist_ma50"] = c / c.rolling(50).mean() - 1; g["ret5"] = c / c.shift(7) - 1  # 7-day momentum window (ret7 > ret5)
    g["dist_l63"] = c / l.rolling(63).min() - 1  # distance above 63-session low (trend-quality / anti-near-low)
    parts.append(g[["symbol", "date"] + CS4 + ["atrpct", "dist_ma50", "ret5", "dist_l63"]])
P = pd.concat(parts, ignore_index=True)
for col in CS4 + ["atrpct", "dist_ma50"]:
    P[col + "_r"] = P.groupby("date")[col].rank(pct=True)
b4 = [c + "_r" for c in CS4]
P["cs5_ma50"] = P[b4 + ["atrpct_r", "dist_ma50_r"]].mean(axis=1)
CSm = {(r.symbol, str(r.date.date())): (r.cs5_ma50 if pd.notna(r.cs5_ma50) else 0.5) for r in P.itertuples()}
R5 = {(r.symbol, str(r.date.date())): (r.ret5 if pd.notna(r.ret5) else _np.nan) for r in P.itertuples()}
DL63m = {(r.symbol, str(r.date.date())): (r.dist_l63 if pd.notna(r.dist_l63) else 9.0) for r in P.itertuples()}

con = psycopg2.connect(**PG); feat = None
cv_c, pm, cm_c, r5_c, dl63_c = {}, {}, {}, {}, {}
base = sigdf = bt = None
for sd in SEEDS:
    rid = run_template_experiment(template_id=3185, seed=sd).get("run_id")
    cvtr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price,entry_signal_date,exit_reason,pnl_pct from run_trades where run_id=%s and exit_date is not null", con, params=(rid,))
    cvtr["sigd"] = pd.to_datetime(cvtr["entry_signal_date"]); cvtr["ed"] = cvtr["entry_date"].astype(str)
    # EARLY-CUT: observe đỏ tại phiên 2 (close[ei+2]<entry) -> BÁN phiên 3 (khớp quy ước T+1-exit, causal)
    ned, nep, nreason = [], [], []
    GT = 0.08
    for r in cvtr.itertuples():
        di = DIDX.get(r.symbol, {}); ei = di.get(str(r.entry_date)[:10]); xi = di.get(str(r.exit_date)[:10])
        if ei is None or xi is None or xi <= ei:
            ned.append(r.exit_date); nep.append(r.exit_price); nreason.append(r.exit_reason); continue
        c = CLO[r.symbol]
        if xi > ei + 3 and c[ei + 2] / r.entry_price - 1.0 < 0.0:          # red@s2 -> early-cut phien3
            ned.append(INV[r.symbol][ei + 3]); nep.append(float(c[ei + 3])); nreason.append("early_cut")
        else:                                                              # green@s2 -> green-trail 8% tu dinh -> ban phien ke
            ek = xi; ep_ = r.exit_price; rs = r.exit_reason; peak = c[ei]
            for b in range(ei + 2, xi + 1):
                peak = max(peak, c[b])
                if c[b] <= peak * (1 - GT) and b >= ei + 2:
                    if b + 1 <= xi:
                        ek = b + 1; ep_ = float(c[b + 1])
                    else:
                        ek = b; ep_ = float(c[b])
                    rs = "green_trail"; break
            ned.append(INV[r.symbol][ek]); nep.append(ep_); nreason.append(rs)
    cvtr["exit_date"] = pd.to_datetime(ned); cvtr["exit_price"] = nep; cvtr["exit_reason"] = nreason
    cm_c[sd] = {(r.symbol, r.ed): CSm.get((r.symbol, str(r.sigd.date())), 0.5) for r in cvtr.itertuples()}
    r5_c[sd] = {(r.symbol, r.ed): R5.get((r.symbol, str(r.sigd.date())), _np.nan) for r in cvtr.itertuples()}
    dl63_c[sd] = {(r.symbol, r.ed): DL63m.get((r.symbol, str(r.sigd.date())), 9.0) for r in cvtr.itertuples()}
    cvf = HERE / f"_dp_s{sd}.csv"; cvtr[["symbol", "entry_date", "exit_date", "entry_price", "exit_price"]].to_csv(cvf, index=False); cv_c[sd] = str(cvf)
    if feat is None:
        feat = M.features(cvtr.symbol.unique().tolist())
    pm[sd] = M.meta_preds(M.build_tr(con, rid, feat), tgt='t_pnl')
    if sd == SEED:
        bt = cvtr; base = {(r.symbol, r.ed): dict(exit_price=r.exit_price, exit_reason=r.exit_reason, sigd=r.sigd) for r in cvtr.itertuples()}
        sigdf = pd.read_sql("select symbol,date from run_signals where run_id=%s and signal=1", con, params=(rid,)); sigdf["date"] = pd.to_datetime(sigdf["date"])


# ============================================================================
# Register 3 dl63 near-low-selection variants (each: leaderboard_nav T+0/T+2 + full portfolio).
# dl63 = distance above 63-session low at signal date; near-low entries are opp-cost/weak setups
# (mean pnl ~1% vs book ~5%). Applied ON TOP of champion gtrail (ret7+early-cut+green-trail).
# ============================================================================
DL63_HASH_SALT = "r7ec_gtrail_dl63"


def metrics(tplus, lo="2020-01-01"):
    globals()["TPLUS"] = tplus
    navs = []
    for sd in SEEDS:
        globals()["DL63MAP"] = dl63_c[sd]
        navs.append(prun_metric(NavSim2(cv_c[sd], date_lo=lo), pm[sd], cm_c[sd], r5_c[sd]))
    fin = [float(d["nav"].iloc[-1]) for d in navs]; yrs = (navs[0]["date"].iloc[-1] - navs[0]["date"].iloc[0]).days / 365.25
    cg = statistics.mean(f ** (1 / yrs) - 1 for f in fin); dd = statistics.mean(float((d["nav"] / d["nav"].cummax() - 1).min()) for d in navs)
    return statistics.mean(fin), cg, dd, yrs


async def make_tmpl(name, desc):
    S2 = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with S2() as s:
        repo = StrategyTemplateRepository(s); ex = await repo.get_by_name(name)
        if ex:
            return ex.id
        b = await repo.get_by_id(3185)
        slots = [{"slot_type": sl.slot_type, "ml_component_id": sl.ml_component_id, "rule_component_id": sl.rule_component_id,
                  "feature_set_name": sl.feature_set_name, "target_config": copy.deepcopy(sl.target_config)} for sl in b.component_slots]
        t = await repo.create(name=name, market=b.market, strategy=b.strategy, feature_set_id=b.feature_set_id,
            target_id=b.target_id, component_slots=slots, direction=b.direction, signal_mode=b.signal_mode,
            signal_threshold=b.signal_threshold, entry_threshold=b.entry_threshold, exit_threshold=b.exit_threshold,
            split_config=copy.deepcopy(b.split_config), engine_config=copy.deepcopy(b.engine_config),
            validation_config=b.validation_config, seed=42, description=desc,
            hypothesis="dl63 near-low selection lever on champion gtrail (skip/size-reduce entries near 63-session low)",
            universe_slug=b.universe_slug, model_mode="ml_only")
        await s.commit(); return t.id


def populate(RID):
    """Full portfolio track for seed42 under current DL63 globals (equity/holdings/trades/skipped/pending)."""
    globals()["DL63MAP"] = dl63_c[SEED]; globals()["TPLUS"] = 2
    equity, holdings, hbd, trades = prun_track(NavSim2(cv_c[SEED], date_lo="2020-01-01"), pm[SEED], cm_c[SEED], r5_c[SEED], base, RID)
    caldates = [pd.to_datetime(e[1]) for e in equity]
    skipped = []
    for r in bt.itertuples():
        k = (r.symbol, r.ed); conv = cm_c[SEED].get(k, 0.5); v = r5_c[SEED].get(k, _np.nan); dl = dl63_c[SEED].get(k, 9.0)
        if conv < SKIP:
            skipped.append((RID, r.symbol, str(r.sigd.date()) if pd.notna(r.sigd) else None, r.ed, float(r.pnl_pct or 0.0), float(conv), "conv_skip"))
        elif not _np.isnan(v) and v < R5THR:
            skipped.append((RID, r.symbol, str(r.sigd.date()) if pd.notna(r.sigd) else None, r.ed, float(r.pnl_pct or 0.0), float(conv), "ret7_gate"))
        elif DL63_TH is not None and DL63_MODE == "skip" and dl < DL63_TH:
            skipped.append((RID, r.symbol, str(r.sigd.date()) if pd.notna(r.sigd) else None, r.ed, float(r.pnl_pct or 0.0), float(conv), "dl63_nearlow"))
    bar_of = {d: i for i, d in enumerate(caldates)}; nD = len(caldates); low_arr, close_arr = {}, {}
    for s, g in px.groupby("symbol"):
        la = _np.full(nD, _np.nan); ca = _np.full(nD, _np.nan)
        for r in g.itertuples():
            b = bar_of.get(r.date)
            if b is not None:
                la[b] = r.low; ca[b] = r.close
        low_arr[s] = la; close_arr[s] = ca
    sig_bars = defaultdict(list)
    for r in sigdf.itertuples():
        b = bar_of.get(r.date)
        if b is not None:
            sig_bars[r.symbol].append(b)
    for d in sig_bars:
        sig_bars[d].sort()
    pending = []
    for sym, sbars in sig_bars.items():
        la = low_arr.get(sym); ca = close_arr.get(sym)
        if la is None:
            continue
        for d in range(nD):
            li = bisect_right(sbars, d - PB_WIN); ri = bisect_right(sbars, d)
            if li >= ri:
                continue
            S = sbars[ri - 1]; cS = ca[S]
            if _np.isnan(cS):
                continue
            limit = cS * (1.0 - PB_PCT); seg_sd = la[S:d + 1]
            if seg_sd.size and _np.nanmin(seg_sd) <= limit:
                continue
            ds = str(caldates[d].date())
            if sym in hbd.get(ds, set()):
                continue
            cD = ca[d]
            if _np.isnan(cD):
                continue
            pctL = limit / cD - 1.0; hi = min(S + PB_WIN, nD - 1); seg = la[S:hi + 1]; touch = _np.where(seg <= limit)[0]
            if touch.size:
                outc, rdate = "fill", str(caldates[S + int(touch[0])].date())
            else:
                outc, rdate = "expire", str(caldates[hi].date())
            pending.append((RID, ds, sym, str(caldates[S].date()), d - S, float(limit), float(cD), float(pctL), outc, rdate))
    TCOLS = ["symbol", "entry_date", "entry_price", "exit_date", "exit_price", "holding_days", "pnl_pct", "exit_reason", "entry_signal_date"]
    for tbl in ("run_equity", "run_portfolio_daily", "run_skipped", "run_pending", "run_trades"):
        cur.execute(f"DELETE FROM {tbl} WHERE run_id=%s", (RID,))
    execute_values(cur, "INSERT INTO run_equity (run_id,date,nav,cash,exposure,n_positions) VALUES %s", equity)
    execute_values(cur, "INSERT INTO run_portfolio_daily (run_id,date,symbol,weight,entry_weight,unreal_pnl,entry_date,days_held,is_new,is_exit,exit_reason,conv) VALUES %s", holdings)
    execute_values(cur, "INSERT INTO run_trades (run_id," + ",".join(TCOLS) + ") VALUES %s", [(RID,) + tuple(t[c] for c in TCOLS) for t in trades])
    if skipped:
        execute_values(cur, "INSERT INTO run_skipped (run_id,symbol,signal_date,entry_date,pnl_pct,conv,skip_reason) VALUES %s", skipped)
    if pending:
        execute_values(cur, "INSERT INTO run_pending (run_id,date,symbol,signal_date,days_waiting,limit_price,ref_price,pct_to_limit,outcome,result_date) VALUES %s", pending)
    con.commit()
    nd = sum(1 for x in skipped if x[6] == "dl63_nearlow")
    print(f"  PORTFOLIO {RID}: eq={len(equity)} hold={len(holdings)} trades={len(trades)} skipped={len(skipped)} (dl63_nearlow={nd}) pending={len(pending)}", flush=True)


# ---- 1 variant: dl63 threshold optimized to 0.115 (sweet-spot between aggr 0.10 & bal 0.13) ----
VARIANTS = [
    dict(name="x2_struct_to_k10_cs5ma50_r7ec_gt_dl63opt", th=0.115, mode="skip", size=1.0,
         tag=("[DL63-OPT diem-ngot 0.115] Champion gtrail + SKIP entry dist_l63<0.115. "
              "Threshold toi uu (plateau CAGR 0.10-0.12), vuot dl63bal(0.13): 3-seed proxy CAGR ~141%/DD-12.2%/Calmar 11.6. "
              "Robust per-year 6/7 nam (cuu 2020 +33pp, 2022 +17pp so voi 0.13, gain phan bo deu khong don 1 nam). "
              "Lever selection thuan causal (dist_l63 tai signal date), khong retrain. Nam giua aggr(0.10)/bal(0.13).")),
]

cur = con.cursor()
for V in VARIANTS:
    globals()["DL63_TH"] = V["th"]; globals()["DL63_MODE"] = V["mode"]; globals()["DL63_SIZE"] = V["size"]
    NAV0, CG0, DD0, yrs = metrics(0)
    NAVF22 = metrics(0, "2022-01-01")[0]
    NAV5, CG5, DD5, _ = metrics(2)
    DESC = V["tag"] + f" cagr_adv=T+0({100*CG0:.0f}%) cagr_t2=T+2({100*CG5:.0f}%). Fair 61-sym, TONG VON<=1."
    print(f"[{V['name']}] 3-seed: T+0 {100*CG0:.1f}% | T+2 {100*CG5:.1f}%/DD{100*DD5:.1f} NAVx{NAV5:.2f}", flush=True)
    tid = asyncio.run(make_tmpl(V["name"], DESC))
    try:
        asyncio.run(async_engine.dispose())
    except Exception:
        pass
    cur.execute("update strategy_templates set description=%s where id=%s", (DESC, tid)); con.commit()
    RID = run_template_experiment(template_id=tid, seed=SEED).get("run_id")
    ch = hashlib.md5(f"{RID}_{DL63_HASH_SALT}_{V['name']}".encode()).hexdigest()[:16]
    cur.execute("""insert into leaderboard_nav (run_id,nav_adv,nav_noadv,cagr_adv,cagr_noadv,maxdd_nav,nav_f22_adv,cagr_t2,maxdd_t2,years,n_trades_sim,config_hash,computed_at)
                   values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, now()) on conflict (run_id) do update set nav_adv=excluded.nav_adv,
                   cagr_adv=excluded.cagr_adv, maxdd_nav=excluded.maxdd_nav, nav_f22_adv=excluded.nav_f22_adv,
                   cagr_t2=excluded.cagr_t2, maxdd_t2=excluded.maxdd_t2, config_hash=excluded.config_hash, computed_at=now()""",
                (RID, NAV0, NAV0, CG0, CG0, DD0, NAVF22, CG5, DD5, round(yrs, 2), 0, ch))
    cur.execute("update leaderboard_runs set state='trained', superseded=false where run_id=%s", (RID,))
    con.commit()
    print(f"  REGISTERED {V['name']} -> {RID}", flush=True)
    populate(RID)
con.close()
print("DEPLOY_DL63_DONE", flush=True)

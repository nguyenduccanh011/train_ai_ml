# -*- coding: utf-8 -*-
"""DIG early-cut branch + PRE×POST combo. Post-entry early behavior (e3) is the strongest signal (corr .65).
Unified engine (observe at session SIG, act at SIG+1, T+1-realistic): (a) EARLY-CUT red-early loser (cut_thr),
(b) conviction-conditional cut (only if conv<cut_conv_max — pre×post: cut low-conv, give high-conv room),
(c) EARLY-ADD to green-early confirmed winner if conv>=add_conv_min (pre-confirm × post-add, funded cash<=1).
All on ret7-gated K10/cs5_ma50, T+2, 3-seed. Base(no cut)=123.4%, registered cut=127.8%."""
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
import psycopg2, pandas as pd, numpy as np, duckdb
from nh_nav2 import NavSim2, FEE
from scripts.run_template import run_template_experiment
import hb_112_meta_target as M

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
SEEDS = [42, 21, 123]; SKIP = 0.40; CS4 = ["dist20low", "dist_ma20", "rsi14", "ret20"]
K, MARGIN, KCONV, TPLUS, R7THR = 10, 0.005, 2.0, 2, 0.02


def prun(sim, pm, cm, r7map, SIG=2, cut_thr=0.0, cut_conv_max=None, add_thr=None, add_conv_min=1.0, add_frac=1.0,
         advance_fee=0.0008, roundtrip=0.006):
    s_new = (roundtrip - FEE) / 2.0; sc, si, cal = sim.sym_close, sim.sym_idx, sim.calendar
    for t in sim.trades:
        be, bx = t["i0"], t["i1"]
        if (bx - be) < TPLUS:
            nb = min(be + TPLUS, len(sc[t["symbol"]]) - 1); t["i1"] = nb; t["x_raw"] = sc[t["symbol"]][nb]
        t["net"] = (t["x_raw"] * (1.0 - s_new)) / (t["e_raw"] * (1.0 + s_new)) - 1.0 - FEE
        t["prio"] = pm.get((t["symbol"], t["entry_date"]), -9.9); t["conv"] = cm.get((t["symbol"], t["entry_date"]), 0.5)
    cvv = [t["conv"] for t in sim.trades]; mu = statistics.mean(cvv); sd = statistics.pstdev(cvv) or 1.0
    raw = []
    for t in sim.trades:
        z = (t["conv"] - mu) / sd; t["_w"] = min(max(1.0 + KCONV * z, 0.4), 1.8); raw.append(t["_w"])
    off = 1.0 - (statistics.mean(raw) if raw else 1.0)
    for t in sim.trades:
        t["w"] = max(0.3, t["_w"] + off)
    entries = defaultdict(list)
    for t in sim.trades:
        if t["conv"] < SKIP:
            continue
        v = r7map.get((t["symbol"], t["entry_date"]), np.nan)
        if not np.isnan(v) and v < R7THR:
            continue
        entries[t["entry_date"]].append(t)
    for d in entries:
        entries[d].sort(key=lambda t: t["prio"], reverse=True)

    def lv(leg, dt):
        s = leg["symbol"]; j = si[s].get(dt)
        if j is None:
            return leg["last_val"]
        i0, i1 = leg["i0"], leg["i1"]; j = min(max(j, i0), i1)
        r = leg["ratio1"] if i1 == i0 else leg["ratio0"] + (leg["ratio1"] - leg["ratio0"]) * (j - i0) / (i1 - i0)
        v = leg["invested"] * (sc[s][j] * r) / leg["p0"]; leg["last_val"] = v; return v

    def netfrom(sym, i0, i1, p0):
        return (sc[sym][i1] * (1.0 - s_new)) / (p0 * (1.0 + s_new)) - 1.0 - FEE

    def mk(t, size, di, p0=None, i0=None):
        s = t["symbol"]; i0 = t["i0"] if i0 is None else i0; p0 = t["p0"] if p0 is None else p0; i1 = t["i1"]
        netv = t["net"] if (i0 == t["i0"] and p0 == t["p0"]) else netfrom(s, i0, i1, p0)
        c0, c1 = sc[s][i0], sc[s][i1]; xe = p0 * (1.0 + netv)
        return dict(symbol=s, i0=i0, i1=i1, invested=size, net=netv, p0=p0, ratio0=p0 / c0, ratio1=xe / c1,
                    last_val=size, exit_date=t["exit_date"], prio=t["prio"], be_di=di, conv=t["conv"], src=t, acted=False)

    cash = 1.0; pend = defaultdict(float); legs = []; exits = defaultdict(list); ns = []
    for di, dt in enumerate(cal):
        cash += pend.pop(dt, 0.0); pt = sum(pend.values())
        for leg in exits.get(dt, ()):
            if leg in legs:
                cash += leg["invested"] * (1.0 + leg["net"]) * (1.0 - advance_fee); legs.remove(leg)
        # POST-entry action at age SIG+1 (observe return at age SIG)
        for leg in list(legs):
            if leg["acted"] or leg.get("isadd"):
                continue
            age = di - leg["be_di"]
            if age != SIG + 1:
                continue
            bi = leg["be_di"]; s = leg["symbol"]
            if bi + SIG >= len(sc[s]):
                continue
            r_sig = sc[s][bi + SIG] / leg["p0"] - 1.0; jn = si[s].get(dt)
            if jn is None:
                continue
            if r_sig < cut_thr and (cut_conv_max is None or leg["conv"] < cut_conv_max):
                leg["acted"] = True; vnow = lv(leg, dt); cash += vnow * (1.0 - advance_fee); legs.remove(leg)
                if leg in exits.get(leg["exit_date"], ()):
                    exits[leg["exit_date"]].remove(leg)
            elif add_thr is not None and r_sig > add_thr and leg["conv"] >= add_conv_min:
                leg["acted"] = True; pos_ = sum(lv(l, dt) for l in legs); nav_ = cash + pt + pos_
                add_size = (nav_ / K) * leg["src"]["w"] * add_frac
                if cash + 1e-12 >= add_size:
                    cash -= add_size; al = mk(leg["src"], add_size, di, p0=sc[s][jn], i0=jn); al["isadd"] = True
                    legs.append(al); exits[al["exit_date"]].append(al)
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
    d = pd.DataFrame(ns, columns=["date", "nav"]); nav = d["nav"]; d["date"] = pd.to_datetime(d["date"])
    yrs = (d["date"].iloc[-1] - d["date"].iloc[0]).days / 365.25
    return float(nav.iloc[-1]) ** (1 / yrs) - 1, float((nav / nav.cummax() - 1).min())


cx = duckdb.connect("F:/PROJECTS/train_ai_ml/market_data/market.duckdb", read_only=True)
px = cx.execute("SELECT symbol,date,high,low,close FROM ohlcv WHERE timeframe='1D' AND date>='2018-06-01' ORDER BY symbol,date").fetchdf(); cx.close()
px["date"] = pd.to_datetime(px["date"]); parts = []
for s, g in px.groupby("symbol"):
    g = g.sort_values("date").copy(); c, l, h = g["close"], g["low"], g["high"]
    dd = c.diff(); up = dd.clip(lower=0).rolling(14).mean(); dn = (-dd.clip(upper=0)).rolling(14).mean()
    g["dist20low"] = c / l.rolling(20).min() - 1; g["dist_ma20"] = c / c.rolling(20).mean() - 1
    g["rsi14"] = 100 - 100 / (1 + up / (dn + 1e-9)); g["ret20"] = c / c.shift(20) - 1
    tr_ = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    g["atrpct"] = tr_.rolling(14).mean() / c; g["dist_ma50"] = c / c.rolling(50).mean() - 1; g["mw7"] = c / c.shift(7) - 1
    parts.append(g[["symbol", "date"] + CS4 + ["atrpct", "dist_ma50", "mw7"]])
P = pd.concat(parts, ignore_index=True)
for col in CS4 + ["atrpct", "dist_ma50"]:
    P[col + "_r"] = P.groupby("date")[col].rank(pct=True)
P["cs5_ma50"] = P[[c + "_r" for c in CS4] + ["atrpct_r", "dist_ma50_r"]].mean(axis=1)
CSm = {(r.symbol, str(r.date.date())): (r.cs5_ma50 if pd.notna(r.cs5_ma50) else 0.5) for r in P.itertuples()}
MW7 = {(r.symbol, str(r.date.date())): (r.mw7 if pd.notna(r.mw7) else np.nan) for r in P.itertuples()}

con = psycopg2.connect(**PG); feat = None; cv_c, pm, key_c = {}, {}, {}
for sd in SEEDS:
    rid = run_template_experiment(template_id=3185, seed=sd).get("run_id")
    cvtr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price,entry_signal_date from run_trades where run_id=%s and exit_date is not null", con, params=(rid,))
    cvtr["sigd"] = pd.to_datetime(cvtr["entry_signal_date"]); cvtr["ed"] = cvtr["entry_date"].astype(str)
    key_c[sd] = [(r.symbol, r.ed, str(r.sigd.date())) for r in cvtr.itertuples()]
    cvf = HERE / f"_pp_s{sd}.csv"; cvtr[["symbol", "entry_date", "exit_date", "entry_price", "exit_price"]].to_csv(cvf, index=False); cv_c[sd] = str(cvf)
    if feat is None:
        feat = M.features(cvtr.symbol.unique().tolist())
    pm[sd] = M.meta_preds(M.build_tr(con, rid, feat), tgt='t_pnl')
con.close()
CMS = {sd: {(sym, ed): CSm.get((sym, sgd), 0.5) for sym, ed, sgd in key_c[sd]} for sd in SEEDS}
MW = {sd: {(sym, ed): MW7.get((sym, sgd), np.nan) for sym, ed, sgd in key_c[sd]} for sd in SEEDS}


def ev(**kw):
    rr = [prun(NavSim2(cv_c[sd], date_lo="2020-01-01"), pm[sd], CMS[sd], MW[sd], **kw) for sd in SEEDS]
    return statistics.mean([c for c, d in rr]), statistics.mean([d for c, d in rr]), [c for c, d in rr]


base_cg, base_dd, base_s = ev(cut_thr=-99)  # no cut (thr impossible)
cut_cg, cut_dd, cut_s = ev(SIG=2, cut_thr=0.0)  # registered cut
print(f"=== PRE×POST dig (T+2, K10/cs5_ma50+ret7) — base {100*base_cg:.1f}% | registered-cut {100*cut_cg:.1f}% ===", flush=True)
print(f"  {'variant':44s} | CAGR%  DD%  | vs cut | 3/3cut", flush=True)


def row(lab, cg, dd, sv):
    sgn = sum(1 for i in range(3) if sv[i] > cut_s[i]); mk = "*" if (sgn == 3 and cg > cut_cg) else ("+" if sgn == 3 else " ")
    print(f"  {lab:44s} | {100*cg:5.1f} {100*dd:5.1f} | {100*(cg-cut_cg):+5.1f}{mk} | {sgn}/3", flush=True)


# (1) fine-tune cut sig/thr
for sig in (1, 2, 3):
    for thr in (0.0, -0.01, -0.02):
        cg, dd, sv = ev(SIG=sig, cut_thr=thr); row(f"[cut] sig{sig} thr{100*thr:+.0f}%", cg, dd, sv)
# (2) conviction-conditional cut
for cmax in (0.6, 0.7, 0.8):
    cg, dd, sv = ev(SIG=2, cut_thr=0.0, cut_conv_max=cmax); row(f"[cut|conv<{cmax}] sig2", cg, dd, sv)
# (3) early-ADD on green-early confirmed winner (+ cut on)
for athr in (0.02, 0.03):
    for amin in (0.5, 0.6):
        cg, dd, sv = ev(SIG=2, cut_thr=0.0, add_thr=athr, add_conv_min=amin, add_frac=1.0); row(f"[cut+add] green>{100*athr:.0f}% conv>={amin}", cg, dd, sv)
cg, dd, sv = ev(SIG=2, cut_thr=0.0, add_thr=0.02, add_conv_min=0.5, add_frac=0.5); row("[cut+add] green>2% conv>=0.5 frac0.5", cg, dd, sv)
# COMBINE: cut sớm (sig1) × chừa conviction cao
for cmax in (0.75, 0.8, 0.85, 0.9):
    cg, dd, sv = ev(SIG=1, cut_thr=0.0, cut_conv_max=cmax); row(f"[COMBO] sig1 | conv<{cmax}", cg, dd, sv)
print("PREPOST_DONE", flush=True)

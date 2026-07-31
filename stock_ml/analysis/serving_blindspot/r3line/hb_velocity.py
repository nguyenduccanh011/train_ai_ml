# -*- coding: utf-8 -*-
"""User: overshoot (magnitude) matters -> does SPEED (velocity/accel) or DURATION (sessions) add?
Early-cut uses LEVEL (red at session2). Test whether VELOCITY (falling fast / decelerating) beats level,
and whether time-to-profit / hold-length carries signal. Diagnostic corr/AUC of e1,e2,e3,velocity,accel vs
net; NAV cut variants (velocity/accel trigger, observe s2 / sell s3) vs registered level-cut 127.8%."""
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
from sklearn.metrics import roc_auc_score
import hb_112_meta_target as M

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
SEEDS = [42, 21, 123]; SKIP = 0.40; CS4 = ["dist20low", "dist_ma20", "rsi14", "ret20"]
K, MARGIN, KCONV, TPLUS, R7THR = 10, 0.005, 2.0, 2, 0.02


def prun(sim, pm, cm, r7map, advance_fee=0.0008, roundtrip=0.006):
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

    def mk(t, size, di):
        s = t["symbol"]; c0, c1 = sc[s][t["i0"]], sc[s][t["i1"]]; xe = t["p0"] * (1.0 + t["net"])
        return dict(symbol=s, i0=t["i0"], i1=t["i1"], invested=size, net=t["net"], p0=t["p0"], ratio0=t["p0"] / c0,
                    ratio1=xe / c1, last_val=size, exit_date=t["exit_date"], prio=t["prio"], be_di=di)

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
    d = pd.DataFrame(ns, columns=["date", "nav"]); nav = d["nav"]; d["date"] = pd.to_datetime(d["date"])
    yrs = (d["date"].iloc[-1] - d["date"].iloc[0]).days / 365.25
    return float(nav.iloc[-1]) ** (1 / yrs) - 1, float((nav / nav.cummax() - 1).min())


cx = duckdb.connect("F:/PROJECTS/train_ai_ml/market_data/market.duckdb", read_only=True)
px = cx.execute("SELECT symbol,date,high,low,close FROM ohlcv WHERE timeframe='1D' AND date>='2018-06-01' ORDER BY symbol,date").fetchdf(); cx.close()
px["date"] = pd.to_datetime(px["date"]); parts = []; CLO = {}; DIDX = {}; INV = {}
for s, g in px.groupby("symbol"):
    g = g.sort_values("date").copy(); c, l, h = g["close"], g["low"], g["high"]
    CLO[s] = c.values; DIDX[s] = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(g["date"])}; INV[s] = {i: d.strftime("%Y-%m-%d") for i, d in enumerate(g["date"])}
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

con = psycopg2.connect(**PG); feat = None; cv_c, pm, key_c, cvtrs = {}, {}, {}, {}
for sd in SEEDS:
    rid = run_template_experiment(template_id=3185, seed=sd).get("run_id")
    cvtr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price,entry_signal_date from run_trades where run_id=%s and exit_date is not null", con, params=(rid,))
    cvtr["sigd"] = pd.to_datetime(cvtr["entry_signal_date"]); cvtr["ed"] = cvtr["entry_date"].astype(str)
    key_c[sd] = [(r.symbol, r.ed, str(r.sigd.date())) for r in cvtr.itertuples()]
    cvf = HERE / f"_vel_s{sd}.csv"; cvtr[["symbol", "entry_date", "exit_date", "entry_price", "exit_price"]].to_csv(cvf, index=False); cv_c[sd] = str(cvf)
    cvtrs[sd] = cvtr
    if feat is None:
        feat = M.features(cvtr.symbol.unique().tolist())
    pm[sd] = M.meta_preds(M.build_tr(con, rid, feat), tgt='t_pnl')
con.close()
CMS = {sd: {(sym, ed): CSm.get((sym, sgd), 0.5) for sym, ed, sgd in key_c[sd]} for sd in SEEDS}
MW = {sd: {(sym, ed): MW7.get((sym, sgd), np.nan) for sym, ed, sgd in key_c[sd]} for sd in SEEDS}

# diagnostic seed42: e1/e2/e3, velocity, accel, sessions-to-+5% vs net
cv = cvtrs[42]; rows = []
for r in cv.itertuples():
    di = DIDX.get(r.symbol, {}); ei = di.get(r.ed); xi = di.get(str(r.exit_date)[:10])
    if ei is None or xi is None or ei + 3 >= len(CLO[r.symbol]):
        continue
    c = CLO[r.symbol]; ep = r.entry_price
    e1 = c[ei + 1] / ep - 1; e2 = c[ei + 2] / ep - 1; e3 = c[ei + 3] / ep - 1
    tt5 = next((k for k in range(1, min(xi - ei, 20) + 1) if ei + k < len(c) and c[ei + k] / ep - 1 >= 0.05), 99)
    rows.append(dict(e1=e1, e2=e2, e3=e3, vel=e2 - e1, accel=(e2 - e1) - e1, hold=xi - ei, tt5=tt5, net=r.exit_price / ep - 1))
D = pd.DataFrame(rows)
print(f"=== VELOCITY/DURATION diagnostic ({len(D)} lệnh seed42) ===", flush=True)
for f in ["e1", "e2", "e3", "vel", "accel", "tt5"]:
    a = roc_auc_score((D.net > 0).astype(int), D[f].values); a = max(a, 1 - a)
    print(f"  {f:6s} corr(,net)={D[f].corr(D.net):+.3f} | AUC={a:.3f}", flush=True)
print(f"  [số-phiên] hold vs net: corr={D.hold.corr(D.net):+.3f} (hold=outcome, không dùng entry được)", flush=True)
print(f"  [tới-lời-nhanh] tt5 (số phiên chạm +5%): net khi tt5<=3 phiên {100*D[D.tt5<=3].net.mean():+.1f}% (n={len(D[D.tt5<=3])}) vs tt5>3 {100*D[D.tt5>3].net.mean():+.1f}%", flush=True)


def cut_csv(sd, mode):
    cv = cvtrs[sd]; nd, npx = [], []
    for r in cv.itertuples():
        di = DIDX.get(r.symbol, {}); ei = di.get(str(r.entry_date)[:10]); xi = di.get(str(r.exit_date)[:10])
        if ei is None or xi is None or xi <= ei + 3:
            nd.append(r.exit_date); npx.append(r.exit_price); continue
        c = CLO[r.symbol]; ep = r.entry_price; e1 = c[ei + 1] / ep - 1; e2 = c[ei + 2] / ep - 1
        cut = False
        if mode == "level":
            cut = e2 < 0
        elif mode == "vel":
            cut = (e2 - e1) < 0                    # phiên 2 giảm so phiên 1 (mất đà)
        elif mode == "accel_down":
            cut = e1 < 0 and e2 < e1               # đang giảm VÀ gia tốc xuống
        elif mode == "level_or_vel":
            cut = (e2 < 0) or (e2 - e1 < -0.02)    # đỏ HOẶC rơi nhanh
        elif mode == "level_and_vel":
            cut = (e2 < 0) and (e2 < e1)           # đỏ VÀ vẫn đang giảm (chưa hồi)
        if cut:
            nd.append(INV[r.symbol][ei + 3]); npx.append(float(c[ei + 3]))
        else:
            nd.append(r.exit_date); npx.append(r.exit_price)
    out = cv[["symbol", "entry_date", "exit_date", "entry_price", "exit_price"]].copy()
    out["exit_date"] = pd.to_datetime(nd); out["exit_price"] = npx
    f = HERE / f"_vel_c_s{sd}.csv"; out.to_csv(f, index=False); return str(f)


def ev(mode):
    rr = [prun(NavSim2(cut_csv(sd, mode), date_lo="2020-01-01"), pm[sd], CMS[sd], MW[sd]) for sd in SEEDS]
    return statistics.mean([c for c, d in rr]), statistics.mean([d for c, d in rr]), [c for c, d in rr]


lcg, ldd, ls = ev("level")  # registered level-cut
print(f"\n=== NAV cut theo velocity vs level (registered level-cut = {100*lcg:.1f}%/DD{100*ldd:.1f}) ===", flush=True)
for mode in ["vel", "accel_down", "level_or_vel", "level_and_vel"]:
    cg, dd, sv = ev(mode); sgn = sum(1 for i in range(3) if sv[i] > ls[i]); mk = "*" if (sgn == 3 and cg > lcg) else ("+" if sgn == 3 else " ")
    print(f"  cut={mode:16s} CAGR {100*cg:5.1f}%{mk} ({100*(cg-lcg):+4.1f} vs level, {sgn}/3) DD {100*dd:5.1f}%", flush=True)
print("VELOCITY_DONE", flush=True)

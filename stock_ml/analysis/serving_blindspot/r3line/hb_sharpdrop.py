# -*- coding: utf-8 -*-
"""Case: strong SUSTAINED run (many green sessions) then SHARP drop from peak. Green-trail 8% already covers
it (green-early). Q: does it cover WELL, or do big-gain winners give back a lot despite the 8% trail (sharp
drop overshoots)? Diagnostic + refinements on top of registered r7ec_gtrail: (a) PROFIT-SCALED trail
(tighter when up more, protect big gains), (b) SHARP single-day drop cut. vs registered green-trail 131.7%."""
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
    cvtrs[sd] = cvtr
    if feat is None:
        feat = M.features(cvtr.symbol.unique().tolist())
    pm[sd] = M.meta_preds(M.build_tr(con, rid, feat), tgt='t_pnl')
con.close()
CMS = {sd: {(sym, ed): CSm.get((sym, sgd), 0.5) for sym, ed, sgd in key_c[sd]} for sd in SEEDS}
MW = {sd: {(sym, ed): MW7.get((sym, sgd), np.nan) for sym, ed, sgd in key_c[sd]} for sd in SEEDS}


def cut_csv(sd, trail_hi=None, trail_lo=0.08, gain_hi=0.15, sharp_day=None, sharp_up=0.05):
    """early-cut red@s2; green@s2 -> trailing: trail_lo mặc định, trail_hi (chặt hơn) khi peak-gain≥gain_hi;
    sharp_day: cắt nếu 1 phiên rớt ≥sharp_day trong khi đang lời ≥sharp_up."""
    cv = cvtrs[sd]; nd, npx = [], []
    for r in cv.itertuples():
        di = DIDX.get(r.symbol, {}); ei = di.get(str(r.entry_date)[:10]); xi = di.get(str(r.exit_date)[:10])
        if ei is None or xi is None or xi <= ei:
            nd.append(r.exit_date); npx.append(r.exit_price); continue
        c = CLO[r.symbol]; ep = r.entry_price; ek = xi; ep_ = r.exit_price
        if xi > ei + 3 and c[ei + 2] / ep - 1.0 < 0.0:
            ek = ei + 3; ep_ = float(c[ei + 3])
        else:
            peak = c[ei]
            for b in range(ei + 2, xi + 1):
                peak = max(peak, c[b]); pgain = peak / ep - 1.0
                tr = trail_hi if (trail_hi is not None and pgain >= gain_hi) else trail_lo
                sd_hit = sharp_day is not None and b - 1 >= ei and (c[b] / c[b - 1] - 1.0) <= -sharp_day and (c[b - 1] / ep - 1.0) >= sharp_up
                if (c[b] <= peak * (1 - tr) or sd_hit) and b >= ei + TPLUS:
                    if b + 1 <= xi:
                        ek = b + 1; ep_ = float(c[b + 1])
                    else:
                        ek = b; ep_ = float(c[b])
                    break
        nd.append(INV[r.symbol][ek]); npx.append(ep_)
    out = cv[["symbol", "entry_date", "exit_date", "entry_price", "exit_price"]].copy()
    out["exit_date"] = pd.to_datetime(nd); out["exit_price"] = npx
    f = HERE / f"_sh_s{sd}.csv"; out.to_csv(f, index=False); return str(f)


# diagnostic: cohort chạy-mạnh (MFE≥20%, đỉnh muộn ≥ngày8) — give-back còn lại dưới green-trail8%?
cv = cvtrs[42]; strong = 0; gb_raw = []; gb_gt = []
for r in cv.itertuples():
    di = DIDX.get(r.symbol, {}); ei = di.get(r.ed); xi = di.get(str(r.exit_date)[:10])
    if ei is None or xi is None or xi <= ei + 3:
        continue
    c = CLO[r.symbol]; ep = r.entry_price; seg = c[ei:xi + 1]; pk = int(np.argmax(seg)); mfe = c[ei + pk] / ep - 1
    if mfe < 0.20 or pk < 8 or c[ei + 2] / ep - 1 < 0:  # chạy mạnh, đỉnh muộn, green-early
        continue
    strong += 1; net_raw = r.exit_price / ep - 1; gb_raw.append(mfe - net_raw)
    # green-trail exit
    peak = c[ei]; net_gt = net_raw
    for b in range(ei + 2, xi + 1):
        peak = max(peak, c[b])
        if c[b] <= peak * 0.92 and b >= ei + 2:
            net_gt = (c[b + 1] if b + 1 <= xi else c[b]) / ep - 1; break
    gb_gt.append(mfe - net_gt)
print(f"=== Cohort CHẠY-MẠNH-rồi-crash (MFE≥20%, đỉnh≥ngày8, green-early): {strong} lệnh ===", flush=True)
print(f"  give-back TB: base exit {100*np.mean(gb_raw):.1f}% -> green-trail8% {100*np.mean(gb_gt):.1f}% (đã giảm {100*(np.mean(gb_raw)-np.mean(gb_gt)):.1f}pp)", flush=True)

# NAV refinements vs registered green-trail 8%
reg = [prun(NavSim2(cut_csv(sd, trail_hi=None, trail_lo=0.08), date_lo="2020-01-01"), pm[sd], CMS[sd], MW[sd]) for sd in SEEDS]
r0 = statistics.mean([c for c, d in reg])
print(f"\n=== NAV refinement vs registered green-trail8% ({100*r0:.1f}%/DD{100*statistics.mean([d for c,d in reg]):.1f}) ===", flush=True)
def row(lab, cg, dd, sv):
    sgn = sum(1 for i in range(3) if sv[i] > reg[i][0]); mk = "*" if (sgn == 3 and cg > r0) else ("+" if sgn == 3 else " ")
    print(f"  {lab:34s} | CAGR {100*cg:5.1f}%{mk} ({100*(cg-r0):+4.1f}, {sgn}/3) DD {100*dd:5.1f}%", flush=True)
for th, gh in [(0.05, 0.15), (0.06, 0.15), (0.05, 0.20), (0.06, 0.25)]:
    rr = [prun(NavSim2(cut_csv(sd, trail_hi=th, trail_lo=0.08, gain_hi=gh), date_lo="2020-01-01"), pm[sd], CMS[sd], MW[sd]) for sd in SEEDS]
    row(f"trail {int(100*th)}% khi lãi≥{int(100*gh)}% (else 8%)", statistics.mean([c for c, d in rr]), statistics.mean([d for c, d in rr]), [c for c, d in rr])
for sday in (0.06, 0.08):
    rr = [prun(NavSim2(cut_csv(sd, trail_lo=0.08, sharp_day=sday, sharp_up=0.05), date_lo="2020-01-01"), pm[sd], CMS[sd], MW[sd]) for sd in SEEDS]
    row(f"+ cắt rớt 1-phiên≥{int(100*sday)}% (lãi≥5%)", statistics.mean([c for c, d in rr]), statistics.mean([d for c, d in rr]), [c for c, d in rr])
print("SHARPDROP_DONE", flush=True)

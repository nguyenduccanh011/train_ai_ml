# -*- coding: utf-8 -*-
"""Verify finding #4 part 2: shipped vs causal meta-priority at the DEPLOY K=10 gtrail config."""
from __future__ import annotations
import os, sys, warnings, statistics
from collections import defaultdict
from pathlib import Path
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "stock_ml"))
sys.path.insert(0, os.environ.get("NH_NAV2_DIR", "F:/PROJECTS/hb2943_work"))
os.environ.setdefault("STOCK_DATA_DIR", "F:/PROJECTS/train_ai_ml/market_data/market.duckdb")
import psycopg2, duckdb, pandas as pd, numpy as _np
import hb_112_meta_target as M
from nh_nav2 import NavSim2, FEE

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
MARKET = "F:/PROJECTS/train_ai_ml/market_data/market.duckdb"
RID = "template/x2_struct_to-69338138"
CS4 = ["dist20low", "dist_ma20", "rsi14", "ret20"]
K, MARGIN, KCONV, R5THR = 10, 0.005, 2.0, 0.02
SKIP = 0.40
TPLUS = 2

# ---- panels (copied from hb_deploy_gt.py) ----
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
    g["atrpct"] = tr_.rolling(14).mean() / c; g["dist_ma50"] = c / c.rolling(50).mean() - 1; g["ret5"] = c / c.shift(7) - 1
    parts.append(g[["symbol", "date"] + CS4 + ["atrpct", "dist_ma50", "ret5"]])
P = pd.concat(parts, ignore_index=True)
for col in CS4 + ["atrpct", "dist_ma50"]:
    P[col + "_r"] = P.groupby("date")[col].rank(pct=True)
b4 = [c + "_r" for c in CS4]
P["cs5_ma50"] = P[b4 + ["atrpct_r", "dist_ma50_r"]].mean(axis=1)
CSm = {(r.symbol, str(r.date.date())): (r.cs5_ma50 if pd.notna(r.cs5_ma50) else 0.5) for r in P.itertuples()}
R5 = {(r.symbol, str(r.date.date())): (r.ret5 if pd.notna(r.ret5) else _np.nan) for r in P.itertuples()}

con = psycopg2.connect(**PG)
cvtr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price,entry_signal_date,exit_reason,pnl_pct from run_trades where run_id=%s and exit_date is not null", con, params=(RID,))
cvtr["sigd"] = pd.to_datetime(cvtr["entry_signal_date"]); cvtr["ed"] = cvtr["entry_date"].astype(str)
# early-cut + green-trail transformation (copied)
ned, nep, nreason = [], [], []
GT = 0.08
for r in cvtr.itertuples():
    di = DIDX.get(r.symbol, {}); ei = di.get(str(r.entry_date)[:10]); xi = di.get(str(r.exit_date)[:10])
    if ei is None or xi is None or xi <= ei:
        ned.append(r.exit_date); nep.append(r.exit_price); nreason.append(r.exit_reason); continue
    c = CLO[r.symbol]
    if xi > ei + 3 and c[ei + 2] / r.entry_price - 1.0 < 0.0:
        ned.append(INV[r.symbol][ei + 3]); nep.append(float(c[ei + 3])); nreason.append("early_cut")
    else:
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
cm = {(r.symbol, r.ed): CSm.get((r.symbol, str(r.sigd.date())), 0.5) for r in cvtr.itertuples()}
r5m = {(r.symbol, r.ed): R5.get((r.symbol, str(r.sigd.date())), _np.nan) for r in cvtr.itertuples()}
cvf = HERE / "_vf4b_trades.csv"
cvtr[["symbol", "entry_date", "exit_date", "entry_price", "exit_price"]].to_csv(cvf, index=False)

# meta preds shipped vs causal
feat = M.features(cvtr.symbol.unique().tolist())
tr = M.build_tr(con, RID, feat)
feat_lag = feat.copy().sort_values(["symbol", "date"])
feat_lag["date"] = feat_lag.groupby("symbol")["date"].shift(-1)
feat_lag = feat_lag.dropna(subset=["date"])
tr_c = M.build_tr(con, RID, feat_lag)
con.close()
pm_s = M.meta_preds(tr, tgt="t_pnl")
pm_c = M.meta_preds(tr_c, tgt="t_pnl")


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


def _entries(sim, r5map):
    entries = defaultdict(list)
    for t in sim.trades:
        if t["conv"] < SKIP:
            continue
        v = r5map.get((t["symbol"], t["entry_date"]), _np.nan)
        if not _np.isnan(v) and v < R5THR:
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


def stats(d):
    nav = d["nav"]; final = float(nav.iloc[-1]); yrs = (d["date"].iloc[-1] - d["date"].iloc[0]).days / 365.25
    return final, final ** (1 / yrs) - 1, float((nav / nav.cummax() - 1).min())


ds = prun_metric(NavSim2(str(cvf), date_lo="2020-01-01"), pm_s, cm, r5m)
dc = prun_metric(NavSim2(str(cvf), date_lo="2020-01-01"), pm_c, cm, r5m)
ns_, cgs, dds = stats(ds); nc_, cgc, ddc = stats(dc)
print(f"DEPLOY K10 gtrail T+2 replay (1 base run):")
print(f"  shipped meta: NAV x{ns_:.2f} CAGR {cgs*100:.1f}% DD {dds*100:.1f}%")
print(f"  causal  meta: NAV x{nc_:.2f} CAGR {cgc*100:.1f}% DD {ddc*100:.1f}%")
print(f"  delta CAGR {100*(cgs-cgc):+.2f}pp  delta DD {100*(dds-ddc):+.2f}pp")
# per-year NAV ratio
for d_, tag in ((ds, "shipped"), (dc, "causal")):
    d_["yr"] = d_["date"].dt.year
    ann = d_.groupby("yr")["nav"].agg(["first", "last"])
    print("  " + tag + " per-year: " + " ".join(f"{y}:{100*(r['last']/r['first']-1):+.0f}%" for y, r in ann.iterrows()))
print("VF4_PART2_DONE")

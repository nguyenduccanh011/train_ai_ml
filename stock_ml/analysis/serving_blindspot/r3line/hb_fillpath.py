# -*- coding: utf-8 -*-
"""Does the SIGNAL->FILL path (pullback quality) predict trade outcome / help sizing? Known AT fill (causal).
Features: fill_days (speed), drop_depth (how far below signal it filled), max_rise (rose before pulling back
= strength), overshoot (low below fill = falling-knife). Diagnostic corr/AUC + NAV (sizing tilt / filter).
Operating model K10/cs5_ma50+ret7, T+2, 3-seed. Register if 3/3-beats 127.8%."""
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


def prun(sim, pm, cm, r7map, fmap=None, fthr=None, fabove=True, smap=None, sk=0.0, advance_fee=0.0008, roundtrip=0.006):
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
        z = (t["conv"] - mu) / sd; w = 1.0 + KCONV * z
        if sk and smap is not None:
            sv = smap.get((t["symbol"], t["entry_date"]))
            if sv is not None:
                w += sk * sv
        t["_w"] = min(max(w, 0.4), 1.8); raw.append(t["_w"])
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
        if fmap is not None:
            fv = fmap.get((t["symbol"], t["entry_date"]))
            if fv is not None and ((fv < fthr) if fabove else (fv > fthr)):
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
px["date"] = pd.to_datetime(px["date"]); parts = []; CLO = {}; HI = {}; LO = {}; DIDX = {}
for s, g in px.groupby("symbol"):
    g = g.sort_values("date").copy(); c, l, h = g["close"], g["low"], g["high"]
    CLO[s] = c.values; HI[s] = h.values; LO[s] = l.values; DIDX[s] = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(g["date"])}
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

FEATS = ["fill_days", "drop_depth", "max_rise", "overshoot"]
con = psycopg2.connect(**PG); feat = None; cv_c, pm, key_c, FM, cvtr42 = {}, {}, {}, {f: {} for f in FEATS}, None
for sd in SEEDS:
    rid = run_template_experiment(template_id=3185, seed=sd).get("run_id")
    cvtr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price,entry_signal_date from run_trades where run_id=%s and exit_date is not null", con, params=(rid,))
    cvtr["sigd"] = pd.to_datetime(cvtr["entry_signal_date"]); cvtr["ed"] = cvtr["entry_date"].astype(str)
    key_c[sd] = [(r.symbol, r.ed, str(r.sigd.date())) for r in cvtr.itertuples()]
    # signal->fill path features
    for r in cvtr.itertuples():
        di = DIDX.get(r.symbol, {}); si_ = di.get(str(r.sigd.date())); fi = di.get(r.ed)
        if si_ is None or fi is None or fi < si_:
            continue
        sc0 = CLO[r.symbol][si_]; k = (r.symbol, r.ed)
        FM["fill_days"].setdefault(sd, {})[k] = fi - si_
        FM["drop_depth"].setdefault(sd, {})[k] = r.entry_price / sc0 - 1.0
        FM["max_rise"].setdefault(sd, {})[k] = (HI[r.symbol][si_:fi + 1].max() / sc0 - 1.0)
        FM["overshoot"].setdefault(sd, {})[k] = (r.entry_price - LO[r.symbol][si_:fi + 1].min()) / sc0  # low bao xa duoi fill
    cvf = HERE / f"_fp_s{sd}.csv"; cvtr[["symbol", "entry_date", "exit_date", "entry_price", "exit_price"]].to_csv(cvf, index=False); cv_c[sd] = str(cvf)
    if sd == 42:
        cvtr42 = cvtr
    if feat is None:
        feat = M.features(cvtr.symbol.unique().tolist())
    pm[sd] = M.meta_preds(M.build_tr(con, rid, feat), tgt='t_pnl')
con.close()
CMS = {sd: {(sym, ed): CSm.get((sym, sgd), 0.5) for sym, ed, sgd in key_c[sd]} for sd in SEEDS}
MW = {sd: {(sym, ed): MW7.get((sym, sgd), np.nan) for sym, ed, sgd in key_c[sd]} for sd in SEEDS}

# diagnostic seed42
rows = []
for r in cvtr42.itertuples():
    k = (r.symbol, r.ed); f = {ff: FM[ff][42].get(k, np.nan) for ff in FEATS}
    if any(np.isnan(v) for v in f.values()):
        continue
    f["net"] = r.exit_price / r.entry_price - 1.0; rows.append(f)
D = pd.DataFrame(rows)
print(f"=== SIGNAL->FILL path (chất lượng pullback) — {len(D)} lệnh, seed42 ===", flush=True)
for ff in FEATS:
    a = roc_auc_score((D.net > 0).astype(int), D[ff].values); a = max(a, 1 - a); dirn = "cao=>thắng" if roc_auc_score((D.net > 0).astype(int), D[ff].values) > 0.5 else "thấp=>thắng"
    print(f"  {ff:11s} corr(,net)={D[ff].corr(D.net):+.3f} | AUC={a:.3f} ({dirn}) | TB {D[ff].mean():+.3f}", flush=True)

# NAV: sizing tilt + filter theo max_rise (feature giàu tín hiệu nhất kỳ vọng)
def ev(**kw):
    rr = [prun(NavSim2(cv_c[sd], date_lo="2020-01-01"), pm[sd], CMS[sd], MW[sd], **kw) for sd in SEEDS]
    return statistics.mean([c for c, d in rr]), statistics.mean([d for c, d in rr]), [c for c, d in rr]


bcg, bdd, bs = ev()
print(f"\n=== NAV (base {100*bcg:.1f}%/DD{100*bdd:.1f}) ===", flush=True)
def row(lab, cg, dd, sv):
    sgn = sum(1 for i in range(3) if sv[i] > bs[i]); mk = "*" if (sgn == 3 and cg > bcg) else ("+" if sgn == 3 else " ")
    print(f"  {lab:40s} | {100*cg:5.1f} {100*dd:5.1f} | {100*(cg-bcg):+5.1f}{mk} | {sgn}/3", flush=True)
# sizing tilt by each feature (z-normalized within-run)
for ff in FEATS:
    for sk in (0.3, -0.3):
        smap = {sd: {k: v for k, v in FM[ff][sd].items()} for sd in SEEDS}
        # z-normalize per seed
        for sd in SEEDS:
            vals = np.array(list(smap[sd].values()), float); m, s = np.nanmean(vals), (np.nanstd(vals) or 1)
            smap[sd] = {k: (v - m) / s for k, v in smap[sd].items()}
        cg, dd, sv = ev(smap=smap[42] if False else None, sk=0)  # placeholder to keep base; real below
        break
    # sizing with per-seed maps
    smaps = {}
    for sd in SEEDS:
        vals = np.array(list(FM[ff][sd].values()), float); m, s = np.nanmean(vals), (np.nanstd(vals) or 1)
        smaps[sd] = {k: (v - m) / s for k, v in FM[ff][sd].items()}
    for sk in (0.3, -0.3):
        rr = [prun(NavSim2(cv_c[sd], date_lo="2020-01-01"), pm[sd], CMS[sd], MW[sd], smap=smaps[sd], sk=sk) for sd in SEEDS]
        cg = statistics.mean([c for c, d in rr]); dd = statistics.mean([d for c, d in rr]); sv = [c for c, d in rr]
        row(f"[sizing] w+={sk}*z({ff})", cg, dd, sv)
# FILTER: skip falling-knife (overshoot cao) / fast-crash (fill_days thấp)
print("  --- filter (bỏ hẳn lệnh) ---", flush=True)
allov = np.concatenate([[v for v in FM["overshoot"][sd].values()] for sd in SEEDS])
for p in (70, 80, 90):
    thr = float(np.nanpercentile(allov, p))
    rr = [prun(NavSim2(cv_c[sd], date_lo="2020-01-01"), pm[sd], CMS[sd], MW[sd], fmap=FM["overshoot"][sd], fthr=thr, fabove=False) for sd in SEEDS]
    cg = statistics.mean([c for c, d in rr]); dd = statistics.mean([d for c, d in rr]); sv = [c for c, d in rr]
    row(f"[filter] skip overshoot>top{100-p}%", cg, dd, sv)
# combine overshoot filter + fill_days (skip fast falling-knife)
allfd = np.concatenate([[v for v in FM["fill_days"][sd].values()] for sd in SEEDS])
thr = float(np.nanpercentile(allov, 80))
for fdmax in (2, 3):
    rr = []
    for sd in SEEDS:
        # skip if overshoot>thr AND fill_days<=fdmax (crash nhanh sâu)
        combo = {k: (1.0 if (FM["overshoot"][sd].get(k, 0) > thr and FM["fill_days"][sd].get(k, 99) <= fdmax) else 0.0) for k in FM["overshoot"][sd]}
        rr.append(prun(NavSim2(cv_c[sd], date_lo="2020-01-01"), pm[sd], CMS[sd], MW[sd], fmap=combo, fthr=0.5, fabove=False))
    cg = statistics.mean([c for c, d in rr]); dd = statistics.mean([d for c, d in rr]); sv = [c for c, d in rr]
    row(f"[filter] skip overshoot-top20% & fill≤{fdmax}p", cg, dd, sv)

# COMBINE overshoot-filter ON TOP of r7earlycut (early-cut exit) — DD trội hơn mà giữ CAGR?
def ecut_csv(sd):
    df = pd.read_csv(cv_c[sd]); nd, npx = [], []
    for r in df.itertuples():
        di = DIDX.get(r.symbol, {}); ei = di.get(str(r.entry_date)[:10]); xi = di.get(str(r.exit_date)[:10])
        if ei is not None and xi is not None and xi > ei + 3 and CLO[r.symbol][ei + 2] / r.entry_price - 1.0 < 0.0:
            inv = {v: kk for kk, v in DIDX[r.symbol].items()}; nd.append(inv[ei + 3]); npx.append(float(CLO[r.symbol][ei + 3]))
        else:
            nd.append(r.exit_date); npx.append(r.exit_price)
    df["exit_date"] = pd.to_datetime(nd); df["exit_price"] = npx; f = HERE / f"_fp_ec_s{sd}.csv"; df.to_csv(f, index=False); return str(f)


print("\n=== COMBINE: r7earlycut + overshoot-filter (DD trội mà giữ CAGR?) ===", flush=True)
ecg = [prun(NavSim2(ecut_csv(sd), date_lo="2020-01-01"), pm[sd], CMS[sd], MW[sd]) for sd in SEEDS]
e0 = statistics.mean([c for c, d in ecg])
print(f"  r7earlycut (đã ĐK): CAGR {100*e0:.1f}% DD {100*statistics.mean([d for c,d in ecg]):.1f}%", flush=True)
for p in (20, 10):
    thr = float(np.nanpercentile(allov, 100 - p))
    rr = [prun(NavSim2(ecut_csv(sd), date_lo="2020-01-01"), pm[sd], CMS[sd], MW[sd], fmap=FM["overshoot"][sd], fthr=thr, fabove=False) for sd in SEEDS]
    cg = statistics.mean([c for c, d in rr]); dd = statistics.mean([d for c, d in rr]); sgn = sum(1 for i in range(3) if rr[i][0] > ecg[i][0])
    ddsgn = sum(1 for i in range(3) if rr[i][1] > ecg[i][1])  # DD less negative
    print(f"  + skip overshoot-top{p}%: CAGR {100*cg:.1f}% ({100*(cg-e0):+.1f}pp,{sgn}/3) DD {100*dd:.1f}% (tốt hơn {ddsgn}/3)", flush=True)
print("FILLPATH_DONE", flush=True)

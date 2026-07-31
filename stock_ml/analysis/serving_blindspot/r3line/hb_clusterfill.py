# -*- coding: utf-8 -*-
"""User hypothesis (refined): base 3185 dùng PULLBACK-LIMIT -> khớp T+4+ (không phải T+1). Trong cửa sổ
chờ pullback, nếu THỊ TRƯỜNG quay đầu xấu diện rộng, nhiều limit bị kích CÙNG LÚC -> khớp đồng loạt vào
tape yếu -> lỗ cụm. Test:
 (1) fill_days distribution (xác nhận cửa sổ chờ nhiều phiên),
 (2) fill-clustering (breadth) -> lỗ tương quan,
 (3) market-return trong cửa sổ signal->fill dự báo lỗ,
 (4) filter causal: bỏ fill khi thị trường đã đổi xấu trong lúc chờ (regime break / mkt drop),
     đo standalone + có ADD gì hơn overshoot (osdef) không.
3-seed [42,21,123], T+2, close-accurate. Register nếu 3/3 vượt một slot family."""
from __future__ import annotations
import os, sys, warnings, statistics
from collections import defaultdict, Counter
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


def prun(sim, pm, cm, r7map, fmap=None, fthr=None, fabove=True, advance_fee=0.0008, roundtrip=0.006, want_nav=False):
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
    cg = float(nav.iloc[-1]) ** (1 / yrs) - 1; dd = float((nav / nav.cummax() - 1).min())
    if want_nav:
        return cg, dd, d
    return cg, dd


# ---- data ----
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

# ---- MARKET breadth index (robust) for regime / wait-window return ----
# median + clip ±15% (biên độ VN) => miễn nhiễm penny/split bad-tick làm index nổ vô cực
piv = px.pivot_table(index="date", columns="symbol", values="close").sort_index()
ewd = piv.pct_change().clip(-0.15, 0.15).median(axis=1)   # median daily return (breadth proxy, robust)
idx_s = (1.0 + ewd.fillna(0.0)).cumprod()                 # cumulative breadth index
ma50 = idx_s.rolling(50).mean(); ma20 = idx_s.rolling(20).mean()
IDX = {d.strftime("%Y-%m-%d"): float(v) for d, v in idx_s.items()}
EWD = {d.strftime("%Y-%m-%d"): (float(v) if pd.notna(v) else 0.0) for d, v in ewd.items()}
BELOW50 = {d.strftime("%Y-%m-%d"): (bool(v < m) if pd.notna(m) else False) for (d, v), (_, m) in zip(idx_s.items(), ma50.items())}
BELOW20 = {d.strftime("%Y-%m-%d"): (bool(v < m) if pd.notna(m) else False) for (d, v), (_, m) in zip(idx_s.items(), ma20.items())}

# ---- trades + per-trade features ----
FEATS = ["fill_days", "drop_depth", "overshoot"]
con = psycopg2.connect(**PG); feat = None
cv_c, pm, key_c, FM = {}, {}, {}, {f: {} for f in FEATS}
MWAIT = {sd: {} for sd in SEEDS}     # market return signal->fill (thị trường trong lúc chờ)
MFILL = {sd: {} for sd in SEEDS}     # market return on fill day
FB = {sd: {} for sd in SEEDS}        # fill breadth (số lệnh cùng entry_date, trước gating)
RBREAK = {sd: {} for sd in SEEDS}    # market < MA50 at fill (regime đã gãy)
IDIO = {sd: {} for sd in SEEDS}      # stock drop - market drop over wait (idiosyncratic dip?)
cvtr42 = None
for sd in SEEDS:
    rid = run_template_experiment(template_id=3185, seed=sd).get("run_id")
    cvtr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price,entry_signal_date from run_trades where run_id=%s and exit_date is not null", con, params=(rid,))
    cvtr["sigd"] = pd.to_datetime(cvtr["entry_signal_date"]); cvtr["ed"] = cvtr["entry_date"].astype(str)
    key_c[sd] = [(r.symbol, r.ed, str(r.sigd.date())) for r in cvtr.itertuples()]
    cnt = Counter(r.ed for r in cvtr.itertuples())
    for r in cvtr.itertuples():
        k = (r.symbol, r.ed); sgd = str(r.sigd.date())
        di = DIDX.get(r.symbol, {}); si_ = di.get(sgd); fi = di.get(r.ed)
        # market features (causal: index known at fill close)
        i_f = IDX.get(r.ed); i_s = IDX.get(sgd)
        mw = (i_f / i_s - 1.0) if (i_f and i_s) else np.nan
        MWAIT[sd][k] = mw
        MFILL[sd][k] = EWD.get(r.ed, np.nan)
        FB[sd][k] = cnt[r.ed]
        RBREAK[sd][k] = 1.0 if BELOW50.get(r.ed, False) else 0.0
        if si_ is not None and fi is not None and fi >= si_:
            FM["fill_days"][k] = fi - si_
            FM["drop_depth"][k] = r.entry_price / CLO[r.symbol][si_] - 1.0
            FM["overshoot"][k] = (r.entry_price - LO[r.symbol][si_:fi + 1].min()) / CLO[r.symbol][si_]
            IDIO[sd][k] = (r.entry_price / CLO[r.symbol][si_] - 1.0) - (mw if not np.isnan(mw) else 0.0)
    # store per-seed FM copies (FM was flat; rebuild per seed)
    cvf = HERE / f"_cf_s{sd}.csv"; cvtr[["symbol", "entry_date", "exit_date", "entry_price", "exit_price"]].to_csv(cvf, index=False); cv_c[sd] = str(cvf)
    if sd == 42:
        cvtr42 = cvtr.copy(); FM42 = {f: dict(FM[f]) for f in FM}
    FM = {f: {} for f in list(FM.keys())}  # reset for next seed (we captured seed42 already)
    if feat is None:
        feat = M.features(cvtr.symbol.unique().tolist())
    pm[sd] = M.meta_preds(M.build_tr(con, rid, feat), tgt='t_pnl')
con.close()
CMS = {sd: {(sym, ed): CSm.get((sym, sgd), 0.5) for sym, ed, sgd in key_c[sd]} for sd in SEEDS}
MW = {sd: {(sym, ed): MW7.get((sym, sgd), np.nan) for sym, ed, sgd in key_c[sd]} for sd in SEEDS}

# ======================= PART 1+2: DIAGNOSTIC (seed42) =======================
rows = []
for r in cvtr42.itertuples():
    k = (r.symbol, r.ed)
    fd = FM42["fill_days"].get(k); dd = FM42["drop_depth"].get(k)
    rows.append(dict(fill_days=fd, drop_depth=dd, mwait=MWAIT[42].get(k), mfill=MFILL[42].get(k),
                     fb=FB[42].get(k), rbreak=RBREAK[42].get(k), idio=IDIO[42].get(k),
                     net=r.exit_price / r.entry_price - 1.0))
D = pd.DataFrame(rows)
print(f"=== PART1: cửa sổ chờ pullback (fill_days) — {D.fill_days.notna().sum()} lệnh seed42 ===", flush=True)
fd = D.fill_days.dropna()
print(f"  fill_days: mean {fd.mean():.1f} | median {fd.median():.0f} | %(>1p) {100*(fd>1).mean():.0f}% | %(>=4p) {100*(fd>=4).mean():.0f}% | %(>=8p) {100*(fd>=8).mean():.0f}% | max {fd.max():.0f}", flush=True)
print(f"\n=== PART2: feature nào dự báo LỖ? (AUC>0.5 => cao thắng) ===", flush=True)
for ff in ["fill_days", "drop_depth", "mwait", "mfill", "fb", "rbreak", "idio"]:
    d2 = D[[ff, "net"]].dropna()
    if d2[ff].nunique() < 2:
        continue
    au = roc_auc_score((d2.net > 0).astype(int), d2[ff].values); au = max(au, 1 - au)
    print(f"  {ff:11s} corr(,net)={d2[ff].corr(d2.net):+.3f} | AUC={au:.3f} | TB={d2[ff].mean():+.4f}", flush=True)
# clustering: net theo breadth-of-fill quartile
print(f"\n  net trung bình theo BREADTH (số lệnh khớp cùng ngày):", flush=True)
D["fbq"] = pd.qcut(D.fb.rank(method="first"), 4, labels=["Q1-ít", "Q2", "Q3", "Q4-đông"])
for q, g in D.groupby("fbq"):
    print(f"    {q:8s} n={len(g):4d} | net TB {100*g.net.mean():+5.2f}% | winrate {100*(g.net>0).mean():4.1f}% | mwait TB {100*g.mwait.mean():+.2f}%", flush=True)
# market-turn split
print(f"\n  net theo THỊ TRƯỜNG trong lúc chờ (mwait) & regime tại fill:", flush=True)
for lab, m in [("mwait<-3% (tape rớt)", D.mwait < -0.03), ("mwait>=-3%", D.mwait >= -0.03),
               ("fill khi mkt<MA50 (regime gãy)", D.rbreak == 1.0), ("fill khi mkt>=MA50", D.rbreak == 0.0)]:
    g = D[m & D.net.notna()]
    print(f"    {lab:32s} n={len(g):4d} | net TB {100*g.net.mean():+5.2f}% | winrate {100*(g.net>0).mean():4.1f}%", flush=True)

# ======================= PART3: NAV filters (3-seed, causal) =======================
def ev(fmap_sd=None, fthr=None, fabove=True):
    rr = []
    for sd in SEEDS:
        fm = fmap_sd[sd] if fmap_sd is not None else None
        rr.append(prun(NavSim2(cv_c[sd], date_lo="2020-01-01"), pm[sd], CMS[sd], MW[sd], fmap=fm, fthr=fthr, fabove=fabove))
    return statistics.mean([c for c, _ in rr]), statistics.mean([d for _, d in rr]), [c for c, _ in rr]


bcg, bdd, bs = ev()
print(f"\n=== PART3: NAV base {100*bcg:.1f}%/DD{100*bdd:.1f}% (K10/cs5_ma50+ret7,T+2) ===", flush=True)


def row(lab, cg, dd, sv):
    sgn = sum(1 for i in range(3) if sv[i] > bs[i]); mk = "*" if (sgn == 3 and cg > bcg) else ("+" if sgn == 3 else " ")
    ddsgn = sum(1 for i in range(3) if False)  # not tracking per-seed dd here
    print(f"  {lab:44s} | {100*cg:6.1f} {100*dd:6.1f} | ΔCAGR {100*(cg-bcg):+5.1f}{mk} | {sgn}/3", flush=True)


# (a) skip when market dropped during wait (mwait < thr)
for thr in (-0.03, -0.05, -0.08):
    cg, dd, sv = ev(MWAIT, fthr=thr, fabove=True)
    row(f"[filter] skip mwait < {thr:+.2f} (tape rớt lúc chờ)", cg, dd, sv)
# (b) skip fills during regime break (mkt < MA50)
cg, dd, sv = ev(RBREAK, fthr=0.5, fabove=False)
row("[filter] skip fill khi mkt<MA50", cg, dd, sv)
# (c) skip fills on a down market day (mfill < thr)
for thr in (-0.01, -0.02):
    cg, dd, sv = ev(MFILL, fthr=thr, fabove=True)
    row(f"[filter] skip mfill < {thr:+.2f} (ngày khớp mkt đỏ)", cg, dd, sv)
# (d) breadth-conditioned: skip only when many co-fills AND market weak
allfb = np.concatenate([np.array(list(FB[sd].values()), float) for sd in SEEDS])
fbhi = float(np.nanpercentile(allfb, 75))
for mthr in (0.0, -0.03):
    gate = {sd: {k: (1.0 if (FB[sd].get(k, 0) >= fbhi and MWAIT[sd].get(k, 0) < mthr) else 0.0) for k in FB[sd]} for sd in SEEDS}
    cg, dd, sv = ev(gate, fthr=0.5, fabove=False)
    row(f"[filter] skip breadth>=Q4 & mwait<{mthr:+.2f}", cg, dd, sv)
print("CLUSTERFILL_DONE", flush=True)

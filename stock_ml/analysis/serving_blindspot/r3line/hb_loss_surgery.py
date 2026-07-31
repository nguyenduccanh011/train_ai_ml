# -*- coding: utf-8 -*-
"""LOSS-structure surgery on operating model (K10/cs5_ma50+ret7, T+2):
 (1) run-immediately vs red-early: does first-3-session return predict final net? (early-cut lever?)
 (2) heavy-loser profile: bottom-decile net -> MAE, early behavior, exit reason.
 (3) temporal clustering: per-month portfolio return, worst months, consecutive-loss streaks.
 (4) systemic days: on worst NAV-drop days, how many held positions red at once (crisis vs idiosyncratic).
 (5) STOP-LOSS NAV test: cut a leg early if low hits -X% from fill (attacks LEFT tail; take-profit failed on right)."""
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


def prun(sim, pm, cm, r7map, rec=None, advance_fee=0.0008, roundtrip=0.006):
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
        if rec is not None:
            rec.append(dict(symbol=t["symbol"], entry_date=t["entry_date"], exit_date=t["exit_date"], net=t["net"], p0=t["p0"]))
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
    return float(nav.iloc[-1]) ** (1 / yrs) - 1, float((nav / nav.cummax() - 1).min()), d


cx = duckdb.connect("F:/PROJECTS/train_ai_ml/market_data/market.duckdb", read_only=True)
px = cx.execute("SELECT symbol,date,high,low,close FROM ohlcv WHERE timeframe='1D' AND date>='2018-06-01' ORDER BY symbol,date").fetchdf(); cx.close()
px["date"] = pd.to_datetime(px["date"]); parts = []; CLO = {}; LO = {}; DIDX = {}
for s, g in px.groupby("symbol"):
    g = g.sort_values("date").copy(); c, l, h = g["close"], g["low"], g["high"]
    CLO[s] = c.values; LO[s] = l.values; DIDX[s] = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(g["date"])}
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

con = psycopg2.connect(**PG); feat = None; cv_c, pm, key_c, base_reason = {}, {}, {}, {}
for sd in SEEDS:
    rid = run_template_experiment(template_id=3185, seed=sd).get("run_id")
    cvtr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price,entry_signal_date,exit_reason from run_trades where run_id=%s and exit_date is not null", con, params=(rid,))
    cvtr["sigd"] = pd.to_datetime(cvtr["entry_signal_date"]); cvtr["ed"] = cvtr["entry_date"].astype(str)
    key_c[sd] = [(r.symbol, r.ed, str(r.sigd.date())) for r in cvtr.itertuples()]
    cvf = HERE / f"_ls_s{sd}.csv"; cvtr[["symbol", "entry_date", "exit_date", "entry_price", "exit_price"]].to_csv(cvf, index=False); cv_c[sd] = str(cvf)
    if sd == 42:
        base_reason = {(r.symbol, r.ed): r.exit_reason for r in cvtr.itertuples()}; cvtr42 = cvtr
    if feat is None:
        feat = M.features(cvtr.symbol.unique().tolist())
    pm[sd] = M.meta_preds(M.build_tr(con, rid, feat), tgt='t_pnl')
con.close()
CMS = {sd: {(sym, ed): CSm.get((sym, sgd), 0.5) for sym, ed, sgd in key_c[sd]} for sd in SEEDS}
MW = {sd: {(sym, ed): MW7.get((sym, sgd), np.nan) for sym, ed, sgd in key_c[sd]} for sd in SEEDS}

rec = []
_, _, NAVd = prun(NavSim2(cv_c[42], date_lo="2020-01-01"), pm[42], CMS[42], MW[42], rec=rec)
R = pd.DataFrame(rec)
e3, mae, hold = [], [], []
for r in R.itertuples():
    di = DIDX[r.symbol]; ei = di.get(r.entry_date); xi = di.get(r.exit_date)
    if ei is None or xi is None or xi <= ei:
        e3.append(np.nan); mae.append(np.nan); hold.append(0); continue
    k3 = min(ei + 3, xi); e3.append(CLO[r.symbol][k3] / r.p0 - 1.0)
    mae.append(LO[r.symbol][ei:xi + 1].min() / r.p0 - 1.0); hold.append(xi - ei)
R["e3"] = e3; R["mae"] = mae; R["hold"] = hold; R["reason"] = [base_reason.get((s, d), "?") for s, d in zip(R.symbol, R.entry_date)]

print(f"=== LOSS surgery (operating model seed42, {len(R)} filled legs) ===", flush=True)
print(f"(1) RUN-IMMEDIATELY: corr(ret 3 phiên đầu, net cuối) = {R.e3.corr(R.net):+.3f}", flush=True)
for lab, m in [("đỏ sớm (e3<0)", R.e3 < 0), ("xanh sớm (e3>0)", R.e3 > 0)]:
    g = R[m]; print(f"    {lab:16s}: n={len(g):4d} ({100*len(g)/len(R):3.0f}%) net cuối TB {100*g.net.mean():+5.2f}% | win {100*(g.net>0).mean():3.0f}%", flush=True)
print(f"\n(2) LỖ NẶNG (net ≤ p10): profile", flush=True)
p10 = R.net.quantile(0.10); hl = R[R.net <= p10]
print(f"    {len(hl)} lệnh, net TB {100*hl.net.mean():+.1f}%, MAE TB {100*hl.mae.mean():+.1f}%, e3 TB {100*hl.e3.mean():+.1f}%, hold TB {hl.hold.mean():.0f} phiên", flush=True)
print(f"    % lỗ-nặng đỏ ngay 3 phiên đầu: {100*(hl.e3<0).mean():.0f}% (vs toàn bộ {100*(R.e3<0).mean():.0f}%)", flush=True)
print(f"    exit_reason lỗ-nặng: {hl.reason.value_counts().head(5).to_dict()}", flush=True)

print(f"\n(3) CHUỖI LỖ theo THÁNG (portfolio):", flush=True)
mo = NAVd.set_index("date")["nav"].resample("M").last().pct_change().dropna()
worst = mo.nsmallest(6)
print(f"    6 tháng tệ nhất: {[(str(d.date())[:7], round(100*v,1)) for d, v in worst.items()]}", flush=True)
neg = (mo < 0).astype(int); streak = 0; mx = 0
for v in neg:
    streak = streak + 1 if v else 0; mx = max(mx, streak)
print(f"    tháng âm: {100*(mo<0).mean():.0f}% | chuỗi âm dài nhất: {mx} tháng liên tiếp", flush=True)

print(f"\n(4) VÙNG NHIỀU MÃ LỖ CÙNG LÚC (systemic): các lệnh lỗ theo tháng vào lệnh", flush=True)
R["emo"] = pd.to_datetime(R.entry_date).dt.strftime("%Y-%m"); bym = R.groupby("emo").agg(n=("net", "size"), loss=("net", lambda x: (x < 0).mean()), avg=("net", "mean"))
worstm = bym[bym.n >= 5].nlargest(6, "loss")
print("    6 tháng-vào có tỉ lệ lỗ cao nhất (n≥5):", flush=True)
for m, r in worstm.iterrows():
    print(f"      {m}: {int(r.n)} lệnh, {100*r.loss:.0f}% lỗ, net TB {100*r.avg:+.1f}%", flush=True)

print(f"\n(5) STOP-LOSS test (cắt sớm nếu low chạm -X% từ fill) — 3-seed T+2:", flush=True)


def stopcsv(sd, X):
    df = pd.read_csv(cv_c[sd])
    ne_d, ne_p = [], []
    for r in df.itertuples():
        di = DIDX.get(r.symbol, {}); ei = di.get(str(r.entry_date)[:10]); xi = di.get(str(r.exit_date)[:10])
        if ei is None or xi is None or xi <= ei:
            ne_d.append(r.exit_date); ne_p.append(r.exit_price); continue
        stop = r.entry_price * (1 - X); hit = None
        for k in range(ei + 1, xi + 1):
            if LO[r.symbol][k] <= stop:
                hit = k; break
        if hit is not None:
            inv = {v: kk for kk, v in DIDX[r.symbol].items()}; ne_d.append(inv[hit]); ne_p.append(stop)
        else:
            ne_d.append(r.exit_date); ne_p.append(r.exit_price)
    df["exit_date"] = ne_d; df["exit_price"] = ne_p; f = HERE / f"_ls_stop_s{sd}.csv"; df.to_csv(f, index=False); return str(f)


def earlycut_csv(sd, nday, thr):
    """thoát tại phiên `nday` nếu close[ei+nday]/entry-1 < thr (đỏ sớm)."""
    df = pd.read_csv(cv_c[sd]); ne_d, ne_p = [], []
    for r in df.itertuples():
        di = DIDX.get(r.symbol, {}); ei = di.get(str(r.entry_date)[:10]); xi = di.get(str(r.exit_date)[:10])
        if ei is None or xi is None or xi <= ei + nday:
            ne_d.append(r.exit_date); ne_p.append(r.exit_price); continue
        cpx = CLO[r.symbol][ei + nday]
        if cpx / r.entry_price - 1.0 < thr:
            inv = {v: kk for kk, v in DIDX[r.symbol].items()}; ne_d.append(inv[ei + nday]); ne_p.append(float(cpx))
        else:
            ne_d.append(r.exit_date); ne_p.append(r.exit_price)
    df["exit_date"] = ne_d; df["exit_price"] = ne_p; f = HERE / f"_ls_ec_s{sd}.csv"; df.to_csv(f, index=False); return str(f)


bcg = [prun(NavSim2(cv_c[sd], date_lo="2020-01-01"), pm[sd], CMS[sd], MW[sd])[:2] for sd in SEEDS]
bc0 = statistics.mean([c for c, d in bcg])
print(f"    base (no stop): CAGR {100*bc0:.1f}% DD {100*statistics.mean([d for c,d in bcg]):.1f}%", flush=True)
for X in (0.08, 0.12, 0.15):
    rr = [prun(NavSim2(stopcsv(sd, X), date_lo="2020-01-01"), pm[sd], CMS[sd], MW[sd])[:2] for sd in SEEDS]
    cg = statistics.mean([c for c, d in rr]); dd = statistics.mean([d for c, d in rr]); sgn = sum(1 for i in range(3) if rr[i][0] > bcg[i][0])
    print(f"    stop -{100*X:.0f}%: CAGR {100*cg:5.1f}% ({100*(cg-bc0):+4.1f}pp,{sgn}/3) DD {100*dd:5.1f}%", flush=True)
print(f"\n(6) EARLY-CUT test (thoát phiên N nếu đỏ < thr — lead từ finding #1) — 3-seed T+2:", flush=True)
for nday, thr in [(3, 0.0), (3, -0.02), (2, 0.0), (5, 0.0), (3, -0.04)]:
    rr = [prun(NavSim2(earlycut_csv(sd, nday, thr), date_lo="2020-01-01"), pm[sd], CMS[sd], MW[sd])[:2] for sd in SEEDS]
    cg = statistics.mean([c for c, d in rr]); dd = statistics.mean([d for c, d in rr]); sgn = sum(1 for i in range(3) if rr[i][0] > bcg[i][0])
    mk = "*" if (sgn == 3 and cg > bc0) else ("+" if sgn == 3 else " ")
    print(f"    cut@phiên{nday} nếu<{100*thr:+.0f}%: CAGR {100*cg:5.1f}%{mk} ({100*(cg-bc0):+4.1f}pp,{sgn}/3) DD {100*dd:5.1f}%", flush=True)
print("LOSS_SURGERY_DONE", flush=True)

# -*- coding: utf-8 -*-
"""RUNAWAY-CAPTURE (additive, OCO with pullback): the champion currently LETS the expire cohort (never
dips to the pullback limit) escape unfilled — they run +16.6% avg. Idea: for those escaping names, place a
BREAKOUT-confirmation entry (delay until price rises +CONF above signal close, i.e. 'recognize runaway'),
enter at MARKET there, and manage with the NORMAL champion exit (max_hold14 / trailing gb0.08 act+0.27 /
overext MA20*1.12). Pullback is untouched (if a name dips to limit it stays a pullback trade). Add these
runaway trades to the SAME K-slot book (score-ordered) and compare vs pullback-only base. 3-seed.
Question: do the escaping runaways, entered on causal breakout + normal exit, add positive DECORRELATED
return (esp. dead years 2024/2026)?  Fee roundtrip 0.6%."""
from __future__ import annotations
import os, sys, statistics
from pathlib import Path
import logging
for _n in ("sqlalchemy.engine", "sqlalchemy.engine.Engine", "sqlalchemy"):
    logging.getLogger(_n).setLevel(logging.WARNING)
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[3]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "stock_ml"))
os.environ.setdefault("STOCK_DATA_DIR", "F:/PROJECTS/train_ai_ml/market_data/market.duckdb")
import psycopg2, duckdb, numpy as np, pandas as pd
from scripts.run_template import run_template_experiment

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
MARKET = "F:/PROJECTS/train_ai_ml/market_data/market.duckdb"
COMBO_RID = "template/x2_struct_to_k16preempt_cssize-69338138"   # source of expire signal list
ROUNDTRIP = 0.006; SEEDS = [42, 21, 123]; WIN = 40
MH, ACT, GB, OVX = 14, 0.27, 0.08, 0.12

# ---- price arrays + MA20 ----
cx = duckdb.connect(MARKET, read_only=True)
px = cx.execute("SELECT symbol,date,high,low,close FROM ohlcv WHERE timeframe='1D' AND date>='2018-06-01' ORDER BY symbol,date").fetchdf(); cx.close()
px["date"] = pd.to_datetime(px["date"]); CA = {}; HI = {}; LO = {}; MA = {}; DT = {}; BO = {}
for s, g in px.groupby("symbol"):
    g = g.sort_values("date").reset_index(drop=True)
    CA[s] = g["close"].to_numpy(); HI[s] = g["high"].to_numpy(); LO[s] = g["low"].to_numpy()
    MA[s] = g["close"].rolling(20).mean().to_numpy(); DT[s] = g["date"].to_numpy(); BO[s] = {d: i for i, d in enumerate(g["date"])}
CAL = np.array(sorted(px["date"].unique()))


def normal_exit(sym, be):
    """Champion-style exit from entry bar be (entry=close[be]). Returns (exit_bar, exit_px)."""
    ca, hi = CA[sym], HI[sym]; ep = ca[be]; peak = ep; n = len(ca)
    for i in range(be + 1, min(be + MH, n - 1) + 1):
        peak = max(peak, hi[i])
        if peak >= ep * (1 + ACT) and ca[i] <= peak * (1 - GB):     # trailing
            return i, ca[i]
        m = MA[sym][i]
        if not np.isnan(m) and ca[i] > m * (1 + OVX):               # overextension
            return i, ca[i]
    j = min(be + MH, n - 1)
    return j, ca[j]                                                 # max_hold


def build_runaway_oco(sigs, conf, scoremap):
    """CAUSAL OCO over ALL signals: from bS walk forward; whichever triggers FIRST wins —
    dip (LOW<=limit, => pullback, skip) or breakout (CLOSE>=signal_close*(1+conf), => runaway).
    Runaway enters NEXT bar close + normal exit. INCLUDES breakout-then-collapse names (no survivor bias)."""
    out = []
    for sym, sd, limit in sigs:
        bS = BO.get(sym, {}).get(sd)
        if bS is None:
            continue
        ca, lo = CA[sym], LO[sym]; cS = ca[bS]
        if cS <= 0:
            continue
        thr = cS * (1 + conf); bc = None
        for i in range(bS + 1, min(bS + WIN, len(ca) - 1) + 1):
            if lo[i] <= limit:                         # dip first -> pullback owns it, skip
                break
            if ca[i] >= thr:                           # breakout first -> runaway
                bc = i; break
        if bc is None or bc + 1 >= len(ca):
            continue
        be = bc + 1                                    # enter NEXT bar close (causal)
        bx, xp = normal_exit(sym, be)
        out.append((DT[sym][be], DT[sym][bx], xp / ca[be] - 1 - ROUNDTRIP,
                    scoremap.get((sym, DT[sym][bS]), 0.0)))
    return out


def build_delayonly(expire_sigs, delay, scoremap):
    """CONTROL: enter EVERY escaping signal at a fixed delay (bS+delay), NO breakout condition +
    normal exit. If breakout-conf beats this, 'recognizing runaway' adds value beyond just adding expires."""
    out = []
    for sym, sd in expire_sigs:
        bS = BO.get(sym, {}).get(sd)
        if bS is None:
            continue
        be = bS + delay
        if be + 1 >= len(CA[sym]) or CA[sym][be] <= 0:
            continue
        bx, xp = normal_exit(sym, be)
        out.append((DT[sym][be], DT[sym][bx], xp / CA[sym][be] - 1 - ROUNDTRIP,
                    scoremap.get((sym, DT[sym][bS]), 0.0)))
    return out


def base_consistent(tr, sgm):
    """Pullback base trades but exit RE-COMPUTED with the SAME normal_exit as runaway (apples-to-apples)."""
    out = []
    for r in tr.itertuples():
        be = BO.get(r.symbol, {}).get(r.entry_date)
        if be is None or r.entry_price <= 0:
            continue
        bx, xp = normal_exit(r.symbol, be)
        out.append((r.entry_date, DT[r.symbol][bx], xp / CA[r.symbol][be] - 1 - ROUNDTRIP,
                    sgm.get((r.symbol, r.entry_date), 0.0)))
    return out


def navsim(trades, K):
    """score-ordered K-slot MTM NAV. trades: (entry_date,exit_date,ret,score)."""
    days = lambda a, b: (pd.Timestamp(a) - pd.Timestamp(b)).days
    df = pd.DataFrame(trades, columns=["entry_date", "exit_date", "ret", "score"]).sort_values(
        ["entry_date", "score"], ascending=[True, False])
    byE = {pd.Timestamp(d): g for d, g in df.groupby("entry_date")}
    legs = []; cash = 1.0; navs = []; exits = {}

    def mtm(dt):
        p = 0.0
        for lg in legs:
            span = max(1, days(lg["xd"], lg["ed"])); el = max(0, days(dt, lg["ed"]))
            p += lg["inv"] * (1 + lg["ret"] * min(1.0, el / span))
        return p
    for dt0 in CAL:
        dt = pd.Timestamp(dt0)
        for lg in exits.pop(dt, ()):
            if lg in legs:
                cash += lg["inv"] * (1 + lg["ret"]); legs.remove(lg)
        nav_now = cash + mtm(dt); g = byE.get(dt)
        if g is not None:
            for r in g.itertuples():
                if len(legs) >= K:
                    break
                size = nav_now / K
                if cash + 1e-12 >= size:
                    cash -= size; xd = pd.Timestamp(r.exit_date)
                    lg = dict(xd=xd, inv=size, ret=r.ret, ed=dt); legs.append(lg); exits.setdefault(xd, []).append(lg)
        navs.append((dt, cash + mtm(dt)))
    d = pd.DataFrame(navs, columns=["date", "nav"]); d = d[d["date"] >= "2020-01-01"]
    nav = d["nav"] / d["nav"].iloc[0]; yrs = (d["date"].iloc[-1] - d["date"].iloc[0]).days / 365.25
    return d.assign(nav=nav.values), float(nav.iloc[-1]), nav.iloc[-1] ** (1 / yrs) - 1, float((nav / nav.cummax() - 1).min())


def peryear(dnav):
    yv = dnav.set_index("date")["nav"]; out = []
    for y in range(2020, 2027):
        gg = yv[yv.index.year == y]
        out.append(f"{100*(gg.iloc[-1]/gg.iloc[0]-1):+4.0f}" if len(gg) > 2 else "   .")
    return " ".join(out)


# ---- expire signal list (from combo run_pending) with score ----
con = psycopg2.connect(**PG)
pend = pd.read_sql("SELECT DISTINCT symbol,signal_date,limit_price FROM run_pending WHERE run_id=%s",
                   con, params=(COMBO_RID,))
pend["signal_date"] = pd.to_datetime(pend["signal_date"])
EXP = [(r.symbol, r.signal_date, r.limit_price) for r in pend.itertuples()]
print(f"ALL signals (fill+expire, OCO universe): {len(EXP)}\n", flush=True)

CONFS = [0.03, 0.05, 0.08]
K = 16
# tags: base_real = pullback w/ real engine exit ; base_ce = pullback w/ approx normal_exit (apples-to-apples) ;
#       cNN = base_ce + runaway@confNN (both approx exit)
res = {"base_real": {}, "base_ce": {}, "rw_only": {}}
for c in CONFS:
    res[f"c{int(c*100):02d}"] = {}
for sd in SEEDS:
    rid = run_template_experiment(template_id=3185, seed=sd).get("run_id")
    tr = pd.read_sql("SELECT symbol,entry_date,exit_date,entry_price,exit_price FROM run_trades "
                     "WHERE run_id=%s AND exit_date IS NOT NULL", con, params=(rid,))
    sg = pd.read_sql("SELECT symbol,date,score FROM run_signals WHERE run_id=%s", con, params=(rid,))
    sg["date"] = pd.to_datetime(sg["date"]); sgm = {(r.symbol, r.date): r.score for r in sg.itertuples()}
    tr["entry_date"] = pd.to_datetime(tr["entry_date"]); tr["exit_date"] = pd.to_datetime(tr["exit_date"])
    base_real = [(r.entry_date, r.exit_date, r.exit_price / r.entry_price - 1 - ROUNDTRIP,
                  sgm.get((r.symbol, r.entry_date), 0.0)) for r in tr.itertuples() if r.entry_price > 0]
    base_ce = base_consistent(tr, sgm)
    dnav, navx, cg, dd = navsim(base_real, K); res["base_real"][sd] = (navx, cg, dd, peryear(dnav), len(base_real))
    dnav, navx, cg, dd = navsim(base_ce, K); res["base_ce"][sd] = (navx, cg, dd, peryear(dnav), len(base_ce))
    for c in CONFS:
        rw = build_runaway_oco(EXP, c, sgm)
        dnav2, navx2, cg2, dd2 = navsim(base_ce + rw, K)
        res[f"c{int(c*100):02d}"][sd] = (navx2, cg2, dd2, peryear(dnav2), len(rw))
    rw03 = build_runaway_oco(EXP, 0.03, sgm)
    dnav4, navx4, cg4, dd4 = navsim(rw03, K); res["rw_only"][sd] = (navx4, cg4, dd4, peryear(dnav4), len(rw03))
    print(f"seed{sd} done", flush=True)
con.close()

print("\n=== RUNAWAY-CAPTURE added to pullback base_ce (consistent approx exit), K=16, 3-seed ===", flush=True)
print(f"{'variant':9s} | {'NAVx':>6s} | {'CAGR':>6s} | {'DD':>6s} | Δ vs base_ce | per-year 2020..2026", flush=True)
for k in ["base_real", "base_ce", "rw_only"] + [f"c{int(c*100):02d}" for c in CONFS]:
    v = res[k]
    navx = statistics.mean(x[0] for x in v.values()); cg = statistics.mean(x[1] for x in v.values())
    dd = statistics.mean(x[2] for x in v.values())
    if k.startswith("c"):
        bce = res["base_ce"]; dsigns = "".join("+" if v[s][0] > bce[s][0] else "-" for s in SEEDS)
    else:
        dsigns = "—"
    print(f"{k:9s} | {navx:6.2f} | {100*cg:5.1f}% | {100*dd:5.1f}% | {dsigns:>12s} | {v[SEEDS[0]][3]}  (n~{v[SEEDS[0]][4]})", flush=True)
print("\n(WIN = beats base_ce 3/3 AND improves dead-year 2024/2026; base_real shown to gauge exit-approx bias)")
print("RUNAWAY_CAPTURE_DONE", flush=True)

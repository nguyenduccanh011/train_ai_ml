# -*- coding: utf-8 -*-
"""FAST-SLEEVE probe: the 'early orders' (buy at MARKET on signal, skip pullback) run as an INDEPENDENT
book with a FAST/sensitive exit (early TP / early SL / tight trailing / time-cap). Question: does this
sleeve have positive standalone risk-adjusted return, and is it DECORRELATED from the champion (works in
dead years 2024/2026)? If yes, a capital-split blend improves the whole. Signal universe = run_pending
(the pullback-eligible signals), entered at close_next instead of resting a limit. Score-ordered K-slot,
MTM NAV. Fee roundtrip 0.6%. Blend vs champion run_equity at 70/30 (exposure-neutral, no leverage)."""
from __future__ import annotations
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "stock_ml"))
import psycopg2, duckdb, numpy as np, pandas as pd

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
MARKET = "F:/PROJECTS/train_ai_ml/market_data/market.duckdb"
RID = "template/x2_struct_to_k16preempt_cssize-69338138"
ROUNDTRIP = 0.006

con = psycopg2.connect(**PG)
sig = pd.read_sql("SELECT DISTINCT symbol,signal_date FROM run_pending WHERE run_id=%s", con, params=(RID,))
sc = pd.read_sql("SELECT symbol,date,score FROM run_signals WHERE run_id=%s", con, params=(RID,))
eq = pd.read_sql("SELECT date,nav FROM run_equity WHERE run_id=%s ORDER BY date", con, params=(RID,)); con.close()
sig["signal_date"] = pd.to_datetime(sig["signal_date"]); sc["date"] = pd.to_datetime(sc["date"])
smap = {(r.symbol, r.date): r.score for r in sc.itertuples()}
sig["score"] = [smap.get((r.symbol, r.signal_date), 0.0) for r in sig.itertuples()]

cx = duckdb.connect(MARKET, read_only=True)
px = cx.execute("SELECT symbol,date,high,low,close FROM ohlcv WHERE timeframe='1D' AND date>='2018-06-01' ORDER BY symbol,date").fetchdf(); cx.close()
px["date"] = pd.to_datetime(px["date"]); CA = {}; HI = {}; LO = {}; DT = {}; BO = {}
for s, g in px.groupby("symbol"):
    g = g.sort_values("date").reset_index(drop=True)
    CA[s] = g["close"].to_numpy(); HI[s] = g["high"].to_numpy(); LO[s] = g["low"].to_numpy()
    DT[s] = g["date"].to_numpy(); BO[s] = {d: i for i, d in enumerate(g["date"])}


def fast_trade(sym, bS, tp, sl, mh, trail_act, trail_gb):
    """Enter at close[bS+1]; walk forward. Return (entry_bar, exit_bar, ret) or None.
    SL checked before TP (conservative). trail: after +trail_act peak, exit if drop trail_gb from peak."""
    ca, hi, lo = CA[sym], HI[sym], LO[sym]; be = bS + 1
    if be + 1 >= len(ca):
        return None
    ep = ca[be]
    if ep <= 0:
        return None
    peak = ep
    for i in range(be + 1, min(be + mh, len(ca) - 1) + 1):
        if sl is not None and lo[i] <= ep * (1 - sl):
            return be, i, (ep * (1 - sl)) / ep - 1 - ROUNDTRIP
        if tp is not None and hi[i] >= ep * (1 + tp):
            return be, i, tp - ROUNDTRIP
        if trail_act is not None:
            peak = max(peak, hi[i])
            if peak >= ep * (1 + trail_act) and ca[i] <= peak * (1 - trail_gb):
                return be, i, ca[i] / ep - 1 - ROUNDTRIP
    j = min(be + mh, len(ca) - 1)
    return be, j, ca[j] / ep - 1 - ROUNDTRIP


def build(cfg):
    """Return DataFrame of trades (entry_date, exit_date, ret, score)."""
    tp, sl, mh, ta, tg = cfg["tp"], cfg["sl"], cfg["mh"], cfg.get("ta"), cfg.get("tg")
    rows = []
    for r in sig.itertuples():
        bS = BO.get(r.symbol, {}).get(r.signal_date)
        if bS is None:
            continue
        ft = fast_trade(r.symbol, bS, tp, sl, mh, ta, tg)
        if ft is None:
            continue
        be, bx, ret = ft
        rows.append((DT[r.symbol][be], DT[r.symbol][bx], ret, r.score))
    d = pd.DataFrame(rows, columns=["entry_date", "exit_date", "ret", "score"])
    return d.sort_values(["entry_date", "score"], ascending=[True, False]).reset_index(drop=True)


CAL = np.array(sorted(px["date"].unique()))


def navsim(tr, K):
    """Score-ordered K-slot, MTM daily NAV. Each entry gets nav/K (equal weight). ret is final net
    (fee included); MTM path approximated linearly entry->exit for DD."""
    days = lambda a, b: (pd.Timestamp(a) - pd.Timestamp(b)).days
    byE = {pd.Timestamp(d): g for d, g in tr.groupby("entry_date")}
    open_legs = []            # dicts: exit_date, invested, ret, e_date
    cash = 1.0; navs = []
    exits = {}

    def mtm(dt):
        pos = 0.0
        for leg in open_legs:
            span = max(1, days(leg["exit_date"], leg["e_date"]))
            el = max(0, days(dt, leg["e_date"]))
            pos += leg["invested"] * (1 + leg["ret"] * min(1.0, el / span))
        return pos
    for dt0 in CAL:
        dt = pd.Timestamp(dt0)
        for leg in exits.pop(dt, ()):
            if leg in open_legs:
                cash += leg["invested"] * (1 + leg["ret"]); open_legs.remove(leg)
        nav_now = cash + mtm(dt)
        g = byE.get(dt)
        if g is not None:
            for r in g.itertuples():
                if len(open_legs) >= K:
                    break
                size = nav_now / K
                if cash + 1e-12 >= size:
                    cash -= size
                    ex = pd.Timestamp(r.exit_date)
                    leg = dict(exit_date=ex, invested=size, ret=r.ret, e_date=dt)
                    open_legs.append(leg); exits.setdefault(ex, []).append(leg)
        navs.append((dt, cash + mtm(dt)))
    d = pd.DataFrame(navs, columns=["date", "nav"]); d = d[d["date"] >= np.datetime64("2020-01-01")]
    nav = d["nav"] / d["nav"].iloc[0]
    yrs = (d["date"].iloc[-1] - d["date"].iloc[0]).days / 365.25
    cagr = nav.iloc[-1] ** (1 / yrs) - 1; dd = float((nav / nav.cummax() - 1).min())
    return d.assign(nav=nav.values), float(nav.iloc[-1]), cagr, dd


CONFS = [
    ("sl4_mh20",      dict(tp=None, sl=0.04, mh=20)),                       # cắt lỗ sớm, để chạy tới time-cap
    ("sl3_mh20",      dict(tp=None, sl=0.03, mh=20)),                       # cắt lỗ chặt hơn
    ("tp8_sl4_mh20",  dict(tp=0.08, sl=0.04, mh=20)),                       # chốt sớm + cắt sớm
    ("tp5_sl3_mh10",  dict(tp=0.05, sl=0.03, mh=10)),                       # rất nhanh
    ("trail_a5g4",    dict(tp=None, sl=0.06, mh=25, ta=0.05, tg=0.04)),     # nhạy: trail sát
    ("trail_a8g5",    dict(tp=None, sl=0.08, mh=30, ta=0.08, tg=0.05)),     # trail nới hơn
]

# champion daily equity (for blend)
eq["date"] = pd.to_datetime(eq["date"]); eq = eq[eq["date"] >= "2020-01-01"].sort_values("date")
champ = eq.set_index("date")["nav"]; champ = champ / champ.iloc[0]
cyrs = (champ.index[-1] - champ.index[0]).days / 365.25
c_cagr = champ.iloc[-1] ** (1 / cyrs) - 1; c_dd = float((champ / champ.cummax() - 1).min())
print(f"champion (blend ref): NAV x{champ.iloc[-1]:.2f}  CAGR {100*c_cagr:.1f}%  DD {100*c_dd:.1f}%\n", flush=True)

print("=== FAST-SLEEVE standalone (immediate entry + fast exit), score-ordered K-slot ===", flush=True)
print(f"{'config':14s} | {'K':>2s} | {'NAVx':>6s} | {'CAGR':>6s} | {'DD':>6s} | {'n_tr':>5s} | win% | per-year CAGR(2020..2026)", flush=True)
best = None
for name, cfg in CONFS:
    tr = build(cfg)
    win = 100 * (tr["ret"] > 0).mean()
    for K in (25, 16):
        dnav, navx, cagr, dd = navsim(tr, K)
        # per-year
        yv = dnav.set_index("date")["nav"]; ys = []
        for y in range(2020, 2027):
            gg = yv[yv.index.year == y]
            if len(gg) > 2:
                ys.append(f"{100*(gg.iloc[-1]/gg.iloc[0]-1):+4.0f}")
        print(f"{name:14s} | {K:>2d} | {navx:6.2f} | {100*cagr:5.1f}% | {100*dd:5.1f}% | {len(tr):5d} | {win:4.0f} | {' '.join(ys)}", flush=True)
        if K == 25 and (best is None or cagr > best[1]):
            best = (name, cagr, dnav)

# blend best sleeve with champion at 70/30 and 85/15 (exposure-neutral)
print("\n=== BLEND champion + best sleeve (fixed initial alloc, no leverage) ===", flush=True)
bname, _, bnav = best
sl = bnav.set_index("date")["nav"]; sl = sl.reindex(champ.index, method="ffill").fillna(1.0); sl = sl / sl.iloc[0]
for w in (1.0, 0.85, 0.70):
    bl = w * champ + (1 - w) * sl
    byrs = (bl.index[-1] - bl.index[0]).days / 365.25
    bc = bl.iloc[-1] ** (1 / byrs) - 1; bd = float((bl / bl.cummax() - 1).min())
    tag = "champ only" if w == 1.0 else f"{int(w*100)}/{int((1-w)*100)} champ/{bname}"
    print(f"  {tag:22s}: NAV x{bl.iloc[-1]:6.2f}  CAGR {100*bc:5.1f}%  DD {100*bd:5.1f}%", flush=True)
print("FASTSLEEVE_DONE", flush=True)

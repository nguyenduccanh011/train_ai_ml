# -*- coding: utf-8 -*-
"""MEAN-REVERSION SLEEVE (feasible direction): a SEPARATE book with its OWN capital + OWN target (buy-the-dip
in a long-term uptrend), decorrelated from the momentum champion. Prior notes: rule mean-rev is positive in
the flat/dead years (2023/24/26) where momentum dies; folding oversold INTO the momentum book FAILS (opposite
target) -> must be its own sleeve. Question: does a causal rule mean-rev sleeve on the SAME 61-symbol universe
give positive DECORRELATED returns (esp. dead years), and does an exposure-neutral capital-split blend with
the champion improve risk-adjusted return? Fee roundtrip 0.6%. If yes -> next step ML-label the sleeve."""
from __future__ import annotations
import sys, statistics
from pathlib import Path
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "stock_ml"))
import psycopg2, duckdb, numpy as np, pandas as pd

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
MARKET = "F:/PROJECTS/train_ai_ml/market_data/market.duckdb"
COMBO_RID = "template/x2_struct_to_k16preempt_cssize-69338138"
ROUNDTRIP = 0.006

con = psycopg2.connect(**PG)
uni = pd.read_sql("SELECT DISTINCT symbol FROM run_signals WHERE run_id=%s", con, params=(COMBO_RID,))
eq = pd.read_sql("SELECT date,nav FROM run_equity WHERE run_id=%s ORDER BY date", con, params=(COMBO_RID,)); con.close()
SYMS = set(uni["symbol"])
print(f"universe: {len(SYMS)} symbols\n", flush=True)

cx = duckdb.connect(MARKET, read_only=True)
syms_sql = ",".join(repr(s) for s in SYMS)
px = cx.execute(f"SELECT symbol,date,high,low,close FROM ohlcv WHERE timeframe='1D' AND symbol IN ({syms_sql}) "
                "AND date>='2017-06-01' ORDER BY symbol,date").fetchdf(); cx.close()
px["date"] = pd.to_datetime(px["date"])
CA = {}; HI = {}; LO = {}; DT = {}; BO = {}; SIG = {}
for s, g in px.groupby("symbol"):
    g = g.sort_values("date").reset_index(drop=True)
    c, h, l = g["close"], g["high"], g["low"]
    ma20 = c.rolling(20).mean(); ma100 = c.rolling(100).mean()
    dd = c.diff(); up = dd.clip(lower=0).rolling(14).mean(); dn = (-dd.clip(upper=0)).rolling(14).mean()
    rsi = 100 - 100 / (1 + up / (dn + 1e-9))
    dist20 = c / ma20 - 1
    # mean-rev BUY: long-term uptrend (above MA100) + short-term washed out (RSI<TH & below MA20 by >=drop)
    CA[s] = c.to_numpy(); HI[s] = h.to_numpy(); LO[s] = l.to_numpy(); DT[s] = g["date"].to_numpy()
    BO[s] = {d: i for i, d in enumerate(g["date"])}
    SIG[s] = dict(ma20=ma20.to_numpy(), ma100=ma100.to_numpy(), rsi=rsi.to_numpy(), dist20=dist20.to_numpy())
CAL = np.array(sorted(px["date"].unique()))


def gen_trades(rsi_th, drop, tp, sl, mh, exit_ma20):
    """Causal mean-rev trades. Entry at close[i+1] when uptrend+oversold at bar i. Exit: back to MA20 (if
    exit_ma20) OR +tp OR -sl OR max_hold mh. Priority score = -rsi (deeper oversold first)."""
    trades = []
    for s in SYMS:
        ca, hi, lo = CA[s], HI[s], LO[s]; S = SIG[s]; n = len(ca)
        ma20, ma100, rsi, dist20 = S["ma20"], S["ma100"], S["rsi"], S["dist20"]
        i = 100
        while i < n - 2:
            if (not np.isnan(ma100[i]) and ca[i] > ma100[i] and rsi[i] < rsi_th and dist20[i] <= -drop):
                be = i + 1; ep = ca[be]
                if ep <= 0:
                    i += 1; continue
                bx = None; xp = None
                for j in range(be + 1, min(be + mh, n - 1) + 1):
                    if lo[j] <= ep * (1 - sl):
                        bx, xp = j, ep * (1 - sl); break
                    if hi[j] >= ep * (1 + tp):
                        bx, xp = j, ep * (1 + tp); break
                    if exit_ma20 and not np.isnan(ma20[j]) and ca[j] >= ma20[j]:
                        bx, xp = j, ca[j]; break
                if bx is None:
                    bx = min(be + mh, n - 1); xp = ca[bx]
                trades.append((DT[s][be], DT[s][bx], xp / ep - 1 - ROUNDTRIP, -rsi[i]))
                i = bx + 1                                  # no overlap same symbol
            else:
                i += 1
    return trades


def navsim(trades, K):
    days = lambda a, b: (pd.Timestamp(a) - pd.Timestamp(b)).days
    df = pd.DataFrame(trades, columns=["ed", "xd", "ret", "score"]).sort_values(["ed", "score"], ascending=[True, False])
    byE = {pd.Timestamp(d): g for d, g in df.groupby("ed")}
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
                    cash -= size; xd = pd.Timestamp(r.xd)
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


# champion equity for blend + correlation
eq["date"] = pd.to_datetime(eq["date"]); eq = eq[eq["date"] >= "2020-01-01"].sort_values("date")
champ = eq.set_index("date")["nav"]; champ = champ / champ.iloc[0]
cyrs = (champ.index[-1] - champ.index[0]).days / 365.25
print(f"champion: NAV x{champ.iloc[-1]:.2f}  CAGR {100*(champ.iloc[-1]**(1/cyrs)-1):.1f}%  "
      f"DD {100*float((champ/champ.cummax()-1).min()):.1f}%\n", flush=True)

CONFS = [
    ("mr_rsi30_d05", dict(rsi_th=30, drop=0.05, tp=0.10, sl=0.06, mh=20, exit_ma20=True)),
    ("mr_rsi35_d03", dict(rsi_th=35, drop=0.03, tp=0.10, sl=0.06, mh=20, exit_ma20=True)),
    ("mr_rsi25_d08", dict(rsi_th=25, drop=0.08, tp=0.12, sl=0.08, mh=25, exit_ma20=True)),
    ("mr_rsi30_tp",  dict(rsi_th=30, drop=0.05, tp=0.08, sl=0.05, mh=15, exit_ma20=False)),
]
K = 16
print("=== MEAN-REV sleeve standalone (61-univ, K=16, buy-dip-in-uptrend) ===", flush=True)
print(f"{'config':14s} | {'NAVx':>6s} | {'CAGR':>6s} | {'DD':>6s} | {'n_tr':>5s} | win% | corr | per-year 2020..2026", flush=True)
best = None
for name, cfg in CONFS:
    tr = gen_trades(**cfg)
    if not tr:
        print(f"{name:14s} | (no trades)"); continue
    win = 100 * (pd.DataFrame(tr)[2] > 0).mean()
    dnav, navx, cagr, dd = navsim(tr, K)
    sl_daily = dnav.set_index("date")["nav"].reindex(champ.index, method="ffill").pct_change()
    corr = sl_daily.corr(champ.pct_change())
    print(f"{name:14s} | {navx:6.2f} | {100*cagr:5.1f}% | {100*dd:5.1f}% | {len(tr):5d} | {win:4.0f} | {corr:+.2f} | {peryear(dnav)}", flush=True)
    if best is None or cagr > best[1]:
        best = (name, cagr, dnav)

print("\n=== BLEND champion + best mean-rev sleeve (exposure-neutral, no leverage) ===", flush=True)
bname, _, bnav = best
sl = bnav.set_index("date")["nav"].reindex(champ.index, method="ffill").fillna(1.0); sl = sl / sl.iloc[0]
for w in (1.0, 0.85, 0.70, 0.5):
    bl = w * champ + (1 - w) * sl
    byrs = (bl.index[-1] - bl.index[0]).days / 365.25
    bc = bl.iloc[-1] ** (1 / byrs) - 1; bd = float((bl / bl.cummax() - 1).min())
    tag = "champ only" if w == 1.0 else f"{int(w*100)}/{int((1-w)*100)} champ/{bname}"
    print(f"  {tag:24s}: NAV x{bl.iloc[-1]:6.2f}  CAGR {100*bc:5.1f}%  DD {100*bd:5.1f}%", flush=True)
print("(WIN = sleeve positive in dead years 2024/2026 + low corr + blend raises CAGR-per-DD)")
print("MEANREV_SLEEVE_DONE", flush=True)

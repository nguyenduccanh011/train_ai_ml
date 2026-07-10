# -*- coding: utf-8 -*-
"""RUNAWAY AUTOPSY — Giai doan 1 (offline, khong dot run).

Cohort RUNAWAY thuan = tin hieu buy cua champion (serving frame, bundle 2643 wavestruct
= parity voi wheel engine) ma limit conviction-scaled (<=4.5%) KHONG khop trong 40 phien
VA gia chay len (window_end_close > signal_close).

Cost model tai lap tu engine (da verify rw_00_check.py, 20/20 trades):
  entry fill = px * 1.0015 ; exit fill = close * 0.9985 ; pnl = ratio - 1 - 0.004
  exit 'signal': sell-signal bar t (hold>=1) -> fill close[t+1]
Units: u = tong pnl fraction (1u = 100% tren 1 vi the), giong e_sim.py.
"""
import json
import sqlite3

import numpy as np
import pandas as pd

BASE = r"f:\PROJECTS\train_ai_ml\stock_ml\analysis\serving_blindspot"
OUT = BASE + r"\runaway"
DB = r"C:\Users\DUC CANH PC\Desktop\stock-serving\data\ohlcv.db"
SLIP_IN, SLIP_OUT, FEE = 1.0015, 0.9985, 0.004
WINDOW = 40

def net(entry_px, exit_px):
    return (exit_px * SLIP_OUT) / (entry_px * SLIP_IN) - 1.0 - FEE

# ---------- load ----------
sig = pd.read_csv(f"{BASE}/signals.csv")
uf = pd.read_csv(f"{BASE}/unfilled_signals.csv")
tr = pd.read_csv(f"{BASE}/trades_raw.csv")
con = sqlite3.connect(DB)
px = pd.read_sql("select symbol,date,open,high,low,close from ohlcv order by symbol,date", con)
con.close()

A = {}
for s, g in px.groupby("symbol"):
    g = g.reset_index(drop=True)
    A[s] = dict(dates=g.date.to_numpy(), idx={d: i for i, d in enumerate(g.date)},
                c=g.close.to_numpy(float), l=g.low.to_numpy(float), h=g.high.to_numpy(float))

# sell-signal bar indices per symbol (for champion signal-exit sim)
sell_idx = {}
for s, g in sig[sig.signal < 0].groupby("symbol"):
    a = A.get(s)
    if a is None:
        continue
    sell_idx[s] = np.array(sorted(a["idx"][d] for d in g.date if d in a["idx"]))

buy_total = int((sig.signal > 0).sum())

# ---------- Step A: cohort dinh luong ----------
u = uf[uf.drop_reason == "unfilled"].copy()
u["year"] = u.signal_date.str[:4].astype(int)
u["runaway"] = u.window_end_close > u.signal_close
co = u[u.runaway].copy()

def fwd_from_signal(row, k):
    a = A[row.symbol]
    i = a["idx"][row.signal_date]
    j = i + k
    return a["c"][j] / a["c"][i] - 1.0 if j < len(a["c"]) else np.nan

for k in (21, 40, 60):
    co[f"fwd{k}"] = co.apply(lambda r: fwd_from_signal(r, k), axis=1)
    u[f"fwd{k}"] = u.apply(lambda r: fwd_from_signal(r, k), axis=1)

print("=" * 70)
print("STEP A — COHORT RUNAWAY THUAN")
print(f"tong buy-signal (frame serving 2020-2026): {buy_total}")
print(f"unfilled (limit treo 40 bar khong khop):   {len(u)}  ({len(u)/buy_total:.1%})")
print(f"  trong do RUNAWAY (close_w40 > close_sig): {len(co)}  ({len(co)/buy_total:.1%} tong, {len(co)/len(u):.1%} unfilled)")
print(f"  non-runaway unfilled:                     {len(u)-len(co)}")
print("\nforward return tu CLOSE TIN HIEU (khong phi):")
for k in (21, 40, 60):
    print(f"  fwd{k}: runaway mean {co[f'fwd{k}'].mean():+.4f} med {co[f'fwd{k}'].median():+.4f}"
          f" | non-runaway mean {u.loc[~u.runaway, f'fwd{k}'].mean():+.4f}")
print("\nphan bo theo nam (runaway n / % unfilled nam do / fwd40 mean):")
yt = co.groupby("year").agg(n=("symbol", "size"), fwd40=("fwd40", "mean"))
yt["pct_unfilled"] = co.groupby("year").size() / u.groupby("year").size()
print(yt.round(4).to_string())

# ---------- Step B: at-market takeover sim (entry close[i+1]) + champion exit ----------
# sequential per symbol: skip cohort signal if takeover position of same symbol still open
tr["entry_signal_date"] = tr["entry_signal_date"].astype(str)
core_by_sym = {s: g.sort_values("entry_date").reset_index(drop=True) for s, g in tr.groupby("symbol")}

rows = []
open_until = {}  # symbol -> exit bar idx of last open takeover
co = co.sort_values(["symbol", "signal_date"])
for r in co.itertuples():
    a = A[r.symbol]
    i = a["idx"][r.signal_date]
    n = len(a["c"])
    if i + 1 >= n:
        continue
    if open_until.get(r.symbol, -1) >= i:
        rows.append(dict(symbol=r.symbol, signal_date=r.signal_date, skipped=True))
        continue
    ei = i + 1
    ec = a["c"][ei]
    # champion exit: first sell-signal bar t >= ei+1 (hold>=1) -> fill close[t+1]
    ss = sell_idx.get(r.symbol, np.array([], int))
    nxt = ss[ss >= ei + 1]
    if len(nxt):
        t = int(nxt[0]); xi = min(t + 1, n - 1); reason = "signal"
    else:
        xi = n - 1; reason = "end_of_data"
    xc = a["c"][xi]
    pnl_am = net(ec, xc)
    # channel (a): cung exit, entry tai limit (counterfactual khong xay ra)
    pnl_lim = net(r.limit_price, xc)
    # channel (c): ride exits tren cung entry at-market
    ride = {}
    for hz in (21, 40, 60):
        j = min(ei + hz, n - 1)
        ride[f"ride{hz}"] = net(ec, a["c"][j])
    # trailing 10% tu peak-close, arm ngay, fill close[t+1]
    peak = ec; xi_tr = None
    for t in range(ei + 1, n):
        peak = max(peak, a["c"][t])
        if a["c"][t] < peak * 0.90:
            xi_tr = min(t + 1, n - 1); break
    ride["trail10"] = net(ec, a["c"][xi_tr if xi_tr is not None else n - 1])
    # channel (b): core trades bi chan = entry_signal_date trong (signal_date, exit_date]
    cb = core_by_sym.get(r.symbol)
    blocked_pnl, blocked_n = 0.0, 0
    if cb is not None:
        x_date = a["dates"][xi]
        m = (cb.entry_signal_date > r.signal_date) & (cb.entry_signal_date <= x_date)
        blocked_n = int(m.sum()); blocked_pnl = float(cb.loc[m, "pnl_pct"].sum())
    open_until[r.symbol] = xi
    rows.append(dict(symbol=r.symbol, signal_date=r.signal_date, year=int(r.signal_date[:4]),
                     skipped=False, entry_date=a["dates"][ei], exit_date=a["dates"][xi],
                     hold=xi - ei, reason=reason, pnl_am=pnl_am, pnl_lim=pnl_lim,
                     blocked_n=blocked_n, blocked_pnl=blocked_pnl, **ride,
                     signal_close=r.signal_close, limit_price=r.limit_price))

sim = pd.DataFrame(rows)
taken = sim[~sim.skipped].copy()
taken["net_after_occ"] = taken.pnl_am - taken.blocked_pnl
print("\n" + "=" * 70)
print("STEP B — AT-MARKET TAKEOVER SIM (entry close[i+1], exit stack champion)")
print(f"cohort {len(co)} | taken {len(taken)} | skipped (slot takeover truoc do) {int(sim.skipped.sum())}")
print(f"pnl at-market (exit champion):  {taken.pnl_am.sum():+8.2f} u  (WR {(taken.pnl_am>0).mean():.2f}, mean {taken.pnl_am.mean()*100:+.2f}%, hold med {taken.hold.median():.0f})")
print(f"pnl neu fill tai LIMIT (ctf):   {taken.pnl_lim.sum():+8.2f} u")
print(f"  -> KENH (a) mat dem basis:    {(taken.pnl_lim - taken.pnl_am).sum():+8.2f} u")
print("  ride exits (cung entry at-market):")
for c in ("ride21", "ride40", "ride60", "trail10"):
    print(f"    {c:8s} {taken[c].sum():+8.2f} u   -> KENH (c) exit-mismatch vs champion-exit: {(taken[c]-taken.pnl_am).sum():+8.2f} u")
print(f"  -> KENH (b) occupancy: {int(taken.blocked_n.sum())} core trades bi chan, foregone {taken.blocked_pnl.sum():+8.2f} u")
print(f"NET sau occupancy (exit champion): {taken.net_after_occ.sum():+8.2f} u")
for c in ("ride21", "ride40", "ride60", "trail10"):
    print(f"NET sau occupancy ({c}):  {(taken[c] - taken.blocked_pnl).sum():+8.2f} u  (occupancy tinh theo exit champion — xem note)")
print("\nper-year (exit champion):")
py = taken.groupby("year").agg(n=("pnl_am", "size"), pnl_am=("pnl_am", "sum"),
                               blocked=("blocked_pnl", "sum"), net=("net_after_occ", "sum"))
print(py.round(2).to_string())

taken.to_csv(f"{OUT}/rw_cohort_sim.csv", index=False)
print(f"\nsaved {OUT}/rw_cohort_sim.csv")

# -*- coding: utf-8 -*-
"""RE-ENTRY GAP step 0: do phan tu-tai-mua + gap con lai cua cohort sold-then-rallied (663 lenh gb_x08).

1) Voi moi lenh rallied (post_max_c>=5%/20bar): tim trade MOI cung ma, fill trong <=10/20/40 bar
   sau exit (bar theo calendar cua chinh ma do). Dem n/% + u cua cac re-entry trade.
2) Nhom KHONG re-entry trong 40 bar = "bo lo hoan toan": can tren u neu vao lai qua pullback 4.5%
   chuan (signal gia dinh = bar dau tien sau exit; fill low<=limit trong 40 bar; exit = force-gate
   replication dl12/nonbull/lowbreadth nhu em02, suppress theo mkt_drop; cost model engine).
   Occupancy: tru pnl cac trade THAT cung ma bi hypothetical chan (entry roi vao holding window).
3) Chuan doan a/b/c/d cho nhom khong-re-entry (dung signals.csv frame 2643 lam proxy head,
   gate 2783 = upleg(6%) & close>=MA20, market_weak z<=-1.1 w5 lb60 UNION cumret5<=-4%,
   cooldown 4 bar sau exit LO, depth fill = eff_depth 2643 neu co, else 0.045).
Verify truoc khi tin: replicate pnl formula tren 20 trade thuc te.
"""
import json
import os

import duckdb
import numpy as np
import pandas as pd
import psycopg2

EM = r"f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/exitmap"
BASE = r"f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot"
W_RE = 40          # cua so re-entry (bar)
PB = 0.045         # pullback chuan
PB_WIN = 40
MIN_HOLD = 2
SLIP = 0.0015
RT_COST = 0.004    # 2*comm + tax

tr = pd.read_csv(os.path.join(EM, "gbx08_enriched2.csv"),
                 parse_dates=["entry_date", "exit_date", "entry_signal_date"])
print("trades:", len(tr), "pnl", round(tr.pnl_pct.sum(), 3), "| rallied:", int(tr.rallied.sum()))

pg = psycopg2.connect(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
cur = pg.cursor()
cur.execute("SELECT symbols_json FROM universe_versions WHERE universe_id=8 AND version=2")
uni_syms = [d["symbol"] for d in json.loads(cur.fetchone()[0])]
pg.close()

duck = duckdb.connect(r"f:/PROJECTS/train_ai_ml/market_data/market.duckdb", read_only=True)
bars = duck.execute(
    "SELECT symbol, date, open, high, low, close FROM ohlcv WHERE timeframe='1D' AND symbol IN ({}) "
    "ORDER BY symbol, date".format(",".join(f"'{s}'" for s in uni_syms))).df()
alls = duck.execute("SELECT symbol, date, close FROM ohlcv WHERE timeframe='1D' ORDER BY symbol, date").df()
duck.close()
bars["date"] = pd.to_datetime(bars["date"])
alls["date"] = pd.to_datetime(alls["date"])

# ---------- market series (EW tren universe 61 ma, nhu engine `bars` frame) ----------
piv = bars.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
rets = piv.pct_change()
mret = rets.replace([np.inf, -np.inf], np.nan).clip(-0.5, 0.5).mean(axis=1)
roll5 = mret.rolling(5).sum()
# entry market weak gate (engine _market_drop_dates zscore lb60 + cumret floor -4%)
mu60 = roll5.rolling(60).mean(); sd60 = roll5.rolling(60).std()
z_entry = (roll5 - mu60) / (sd60 + 1e-9)
mkt_weak = (z_entry <= -1.1) | (roll5 <= -0.04)
mkt_weak = mkt_weak.fillna(False)
# exit market drop (suppress signal exit) — thr -1.75 (cung cong thuc engine lb60)
mkt_drop = (z_entry * 0 + ((roll5 - mu60) / (sd60 + 1e-9) <= -1.75)).astype(bool).fillna(False)
# nonbull EW MA35 persist2 (em02)
lvl = (1.0 + mret.fillna(0.0)).cumprod()
ma35 = lvl.rolling(35, min_periods=35).mean()
nonbull = ((lvl < ma35).rolling(2, min_periods=2).sum() >= 2).where(ma35.notna(), True)
# lowbreadth full-duckdb pct_above_ma50 < 0.25 (em02)
pall = alls.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
ma50a = pall.rolling(50, min_periods=50).mean()
ind = (pall > ma50a)
breadth = ind.sum(axis=1) / ind.notna().sum(axis=1).clip(lower=1)
lowbreadth = (breadth < 0.25)


def causal_leg(close, pct):
    n = len(close)
    leg = np.zeros(n, dtype=np.int8)
    if n == 0:
        return leg
    direction, ext = 0, close[0]
    for i in range(1, n):
        p = close[i]
        if direction >= 0 and p > ext:
            ext = p; direction = 1
        elif direction <= 0 and p < ext:
            ext = p; direction = -1
        elif direction == 1 and p <= ext * (1.0 - pct):
            direction = -1; ext = p
        elif direction == -1 and p >= ext * (1.0 + pct):
            direction = 1; ext = p
        leg[i] = direction
    return leg


SYM = {}
for sym, g in bars.groupby("symbol"):
    g = g.reset_index(drop=True)
    c = g["close"].to_numpy(float)
    cs = pd.Series(c)
    ma20 = cs.rolling(20, min_periods=20).mean().to_numpy()
    below20 = c < ma20
    below20[np.isnan(ma20)] = False
    bma20p2 = (pd.Series(below20).rolling(2, min_periods=2).sum().to_numpy() >= 2)
    leg6 = causal_leg(c, 0.06)
    gate_open = (leg6 > 0) & (np.where(np.isnan(ma20), True, c >= np.nan_to_num(ma20)))
    SYM[sym] = dict(
        dates=pd.DatetimeIndex(g["date"]), close=c,
        high=g["high"].to_numpy(float), low=g["low"].to_numpy(float),
        leg12=causal_leg(c, 0.12), leg6=leg6, bma20p2=bma20p2, gate_open=gate_open,
        nonbull=nonbull.reindex(pd.DatetimeIndex(g["date"])).ffill().fillna(True).to_numpy(),
        lowb=lowbreadth.reindex(pd.DatetimeIndex(g["date"])).ffill().fillna(False).to_numpy(),
        weak=mkt_weak.reindex(pd.DatetimeIndex(g["date"])).ffill().fillna(False).to_numpy(),
        drop=mkt_drop.reindex(pd.DatetimeIndex(g["date"])).ffill().fillna(False).to_numpy(),
    )
print("symbols:", len(SYM))

# proxy signals (frame 2643) + eff_depth
sig = pd.read_csv(os.path.join(BASE, "signals.csv"), usecols=["symbol", "date", "signal"])
buys = sig[sig.signal > 0]
buy_dates = {s: set(pd.to_datetime(g.date)) for s, g in buys.groupby("symbol")}
dep = pd.read_parquet(os.path.join(BASE, "depths.parquet"))
dep["date"] = pd.to_datetime(dep["date"])
dep_map = {(r.symbol, r.date): r.eff_depth for r in dep.itertuples()}

# ---------- index hoa trades ----------
tr = tr.sort_values(["symbol", "entry_date"]).reset_index(drop=True)
ei_arr = np.full(len(tr), -1); xi_arr = np.full(len(tr), -1)
for k, t in tr.iterrows():
    s = SYM[t.symbol]
    ei_arr[k] = s["dates"].get_indexer([t.entry_date])[0]
    xi_arr[k] = s["dates"].get_indexer([t.exit_date])[0]
tr["ei"] = ei_arr; tr["xi"] = xi_arr
assert (tr.ei >= 0).all() and (tr.xi >= 0).all()

# verify cost model tren 20 trades: exit fill = close[xi]*(1-slip)? entry fill given.
chk = tr.head(20)
for _, t in chk.iterrows():
    s = SYM[t.symbol]
    if t.exit_reason == "signal":
        recon = t.exit_price / t.entry_price - 1.0 - RT_COST
        assert abs(recon - t.pnl_pct) < 1e-6, (t.symbol, recon, t.pnl_pct)
print("cost model verify OK (pnl = exit/entry - 1 - 0.004)")

# ---------- 1) do tu-tai-mua ----------
by_sym = {s: g.reset_index() for s, g in tr.groupby("symbol")}
rows = []
for _, t in tr[tr.rallied].iterrows():
    s = SYM[t.symbol]
    g = by_sym[t.symbol]
    n_bars = len(s["dates"])
    nxt = g[g.ei > t.xi]
    gap_fill = np.nan; gap_sig = np.nan; re_pnl = np.nan
    if len(nxt):
        r0 = nxt.iloc[0]
        gap_fill = r0.ei - t.xi
        si = s["dates"].get_indexer([r0.entry_signal_date])[0]
        gap_sig = si - t.xi if si >= 0 else np.nan
        re_pnl = r0.pnl_pct
    trunc = (t.xi + W_RE) >= n_bars  # cua so 40 bar bi cat cuoi du lieu
    rows.append(dict(symbol=t.symbol, xi=t.xi, exit_date=t.exit_date, year=t.year_exit,
                     pnl=t.pnl_pct, loss_exit=t.pnl_pct < 0, post_max_c=t.post_max_c,
                     gap_fill=gap_fill, gap_sig=gap_sig, re_pnl=re_pnl, trunc=trunc))
ra = pd.DataFrame(rows)
print("\n=== 1) TU-TAI-MUA (663 rallied) ===")
for w in (10, 20, 40):
    m = ra.gap_fill <= w
    print(f"  re-entry fill <={w} bar: n={int(m.sum())} ({m.mean()*100:.1f}%) | u re-entry = {ra.loc[m,'re_pnl'].sum():+.2f}")
m40 = ra.gap_fill <= 40
print("  u re-entry (<=40) theo nam:", {int(y): round(v, 2) for y, v in ra[m40].groupby("year").re_pnl.sum().items()})
print("  >=2022:", round(ra[m40 & (ra.year >= 2022)].re_pnl.sum(), 2),
      " | n>=2022:", int((m40 & (ra.year >= 2022)).sum()), "/", int((ra.year >= 2022).sum()))
missed = ra[~m40].copy()
print(f"  BO LO hoan toan (khong fill <=40 bar): n={len(missed)} ({len(missed)/len(ra)*100:.1f}%), trong do trunc={int(missed.trunc.sum())}")

# ---------- 2) can tren hypothetical re-entry pullback 4.5% cho nhom bo lo ----------
hrows = []
for _, t in missed.iterrows():
    s = SYM[t.symbol]
    n = len(s["dates"])
    s0 = int(t.xi) + 1
    if s0 >= n - 1:
        hrows.append(dict(symbol=t.symbol, year=t.year, status="no_data")); continue
    limit = s["close"][s0] * (1.0 - PB)
    fill_j = -1
    for j in range(s0 + 1, min(s0 + PB_WIN, n - 1) + 1):
        if s["low"][j] <= limit:
            fill_j = j; break
    if fill_j < 0:
        hrows.append(dict(symbol=t.symbol, year=t.year, status="no_fill")); continue
    entry_fill = limit * (1.0 + SLIP)
    # exit: force-gate replication, suppress khi mkt_drop; fill close[d+1]
    d_exit = -1
    for d in range(fill_j + MIN_HOLD, n - 1):
        fire = (s["leg12"][d] == -1) or (s["bma20p2"][d] and s["nonbull"][d]) or \
               (s["leg6"][d] == -1 and s["lowb"][d])
        if fire and not s["drop"][d]:
            d_exit = d; break
    if d_exit < 0:
        status = "open"; exit_fill = s["close"][n - 1] * (1.0 - SLIP); hold = n - 1 - fill_j; occ_end = n - 1
    else:
        status = "closed"; exit_fill = s["close"][d_exit + 1] * (1.0 - SLIP); hold = d_exit + 1 - fill_j; occ_end = d_exit + 1
    pnl_h = exit_fill / entry_fill - 1.0 - RT_COST
    # occupancy: trade THAT cung ma co entry fill trong [fill_j, occ_end] bi chan
    g = by_sym[t.symbol]
    blocked = g[(g.ei >= fill_j) & (g.ei <= occ_end)]
    occ_pnl = blocked.pnl_pct.sum(); occ_n = len(blocked)
    hrows.append(dict(symbol=t.symbol, year=t.year, status=status, gap_to_fill=fill_j - t.xi,
                      hold=hold, pnl_h=pnl_h, occ_n=occ_n, occ_pnl=occ_pnl, net=pnl_h - occ_pnl))
hy = pd.DataFrame(hrows)
print("\n=== 2) CAN TREN re-entry pullback 4.5% (nhom bo lo, oracle-signal bar exit+1) ===")
print("  status:", hy.status.value_counts().to_dict())
f = hy[hy.status.isin(["closed", "open"])]
print(f"  filled {len(f)}/{len(hy)} | pnl_h={f.pnl_h.sum():+.2f}u | occupancy blocked n={int(f.occ_n.sum())}, "
      f"pnl={f.occ_pnl.sum():+.2f}u | NET={f.net.sum():+.2f}u")
print("  NET theo nam:", {int(y): round(v, 2) for y, v in f.groupby("year").net.sum().items()})
print("  NET >=2022:", round(f[f.year >= 2022].net.sum(), 2), "| pnl_h >=2022:", round(f[f.year >= 2022].pnl_h.sum(), 2),
      "| occ >=2022:", round(f[f.year >= 2022].occ_pnl.sum(), 2))

# ---------- 3) chan doan a/b/c/d cho nhom bo lo ----------
drows = []
for _, t in missed.iterrows():
    s = SYM[t.symbol]
    n = len(s["dates"])
    lo = int(t.xi) + 1; hi = min(int(t.xi) + W_RE, n - 1)
    if lo > hi:
        drows.append(dict(symbol=t.symbol, year=t.year, cls="no_data")); continue
    win_dates = s["dates"][lo:hi + 1]
    gate_frac = float(s["gate_open"][lo:hi + 1].mean())
    bd = buy_dates.get(t.symbol)
    cool_end = int(t.xi) + 4 if t.loss_exit else int(t.xi)
    if bd is None:
        drows.append(dict(symbol=t.symbol, year=t.year, cls="no_proxy", gate_frac=gate_frac)); continue
    sig_idx = [lo + k for k, dt in enumerate(win_dates) if dt in bd]
    if not sig_idx:
        cls = "b1_gate_veto" if gate_frac < 0.10 else "a1_head_silent"
        drows.append(dict(symbol=t.symbol, year=t.year, cls=cls, gate_frac=gate_frac)); continue
    # co proxy signal: loc theo gate 2783 + market weak + cooldown
    viable = []; only_cool = True; any_after_gate = False
    for i2 in sig_idx:
        blocked_cool = i2 <= cool_end
        blocked_gate = (not s["gate_open"][i2]) or s["weak"][i2]
        if not blocked_gate:
            any_after_gate = True
        if not blocked_cool and not blocked_gate:
            viable.append(i2); only_cool = False
        elif blocked_gate:
            only_cool = False
    if not viable:
        cls = "d_cooldown" if (only_cool and t.loss_exit) else "b2_sig_gated"
        drows.append(dict(symbol=t.symbol, year=t.year, cls=cls, gate_frac=gate_frac)); continue
    # fill-sim tren signal viable dau tien -> het cua so
    filled = False
    for i2 in viable:
        depth = dep_map.get((t.symbol, s["dates"][i2]), PB)
        lim = s["close"][i2] * (1.0 - depth)
        for j in range(i2 + 1, min(i2 + PB_WIN, n - 1) + 1):
            if s["low"][j] <= lim:
                filled = True; break
        if filled:
            break
    cls = "e_head2783_silent" if filled else "c_pb_no_fill"
    drows.append(dict(symbol=t.symbol, year=t.year, cls=cls, gate_frac=gate_frac))
dg = pd.DataFrame(drows)
print("\n=== 3) CHAN DOAN nhom bo lo (proxy frame 2643, gate 2783 replicate) ===")
tab = dg.pivot_table(index="cls", columns="year", values="symbol", aggfunc="count", fill_value=0)
tab["ALL"] = tab.sum(axis=1)
print(tab.to_string())
print("\n  gate_frac (median) theo cls:", dg.groupby("cls").gate_frac.median().round(2).to_dict())

ra.to_csv(os.path.join(EM, "re00_rallied.csv"), index=False)
hy.to_csv(os.path.join(EM, "re00_hypo.csv"), index=False)
dg.to_csv(os.path.join(EM, "re00_diag.csv"), index=False)
print("\nsaved re00_rallied.csv / re00_hypo.csv / re00_diag.csv")

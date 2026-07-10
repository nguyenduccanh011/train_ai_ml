# -*- coding: utf-8 -*-
"""THIET KE 2-LO ALL-SIGNAL (de xuat user) — sim NAV, khung y het dl_01.

Sleeve A = trades that gb_x08 seed-42 (K=25) — lo pullback giu nguyen.
Sleeve B = trades np_atmkt (at-market MOI tin hieu, exit stack moi,
run leaderboard composite 498.6 / PF 3.08) — lo market vao ngay bar tin hieu,
ex-ante thuan (khong can biet truoc runaway hay khong).

Kich ban:
  BASE      : 100% NAV -> sleeve A (doi chieu ×14.44 / 50.67% / −15.31%)
  B_ONLY    : 100% NAV -> sleeve B K_B=10 (chan doan per-unit trong khung NAV)
  TL_90_10 / TL_80_20 : chia von tinh, 2 so rieng, khong rebalance, khong vay
  TL_MG     : A giu 100% NAV + B margin tran 50% NAV @15%/nam, lai theo ngay
              du no that (co che het DL_MG dl_01)
Sensitivity: K_B in {5,10,15} cho TL_80_20 va TL_MG.

Cost model giu nguyen: pnl_pct trong run_trades da net (slippage+fee+tax).
Loai lenh exit_reason='open' (chua dong tai cutoff).
Moi kich ban bao them CAGR >=2022 (neo NAV cuoi 2021 -> cuoi ky).
"""
import sqlite3
from collections import defaultdict
from datetime import datetime

import pandas as pd

BASE_DIR = "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot"
A_CSV = BASE_DIR + "/signalq/nicheloss/gb_x08_s42_trades.csv"
B_CSV = BASE_DIR + "/runaway/dl_sleeveB_npatmkt.csv"
DB_PATH = "C:/Users/DUC CANH PC/Desktop/stock-serving/data/ohlcv.db"
DATE_LO, DATE_HI = "2020-01-01", "2026-07-08"
MARGIN_RATE = 0.15
MARGIN_CAP_FRAC = 0.50

ta = pd.read_csv(A_CSV)
tb = pd.read_csv(B_CSV)
n0 = len(tb)
tb = tb[tb.exit_reason != "open"].copy()
print(f"sleeve B np_atmkt: {n0} lenh, loai {n0 - len(tb)} 'open' -> {len(tb)}")

# demand concurrency cua B (de doc K_B)
ev = []
for r in tb.itertuples():
    ev.append((r.entry_date, 1))
    ev.append((r.exit_date, -1))
ev.sort()
cur, days = 0, []
for _, d in ev:
    cur += d
    days.append(cur)
import statistics
print(f"B demand concurrency: mean {statistics.mean(days):.1f} / med {statistics.median(days):.0f} / max {max(days)}")

symbols = sorted(set(ta.symbol) | set(tb.symbol))
con = sqlite3.connect(DB_PATH)
px = pd.read_sql_query(
    "SELECT symbol, date, close FROM ohlcv WHERE symbol IN (%s) AND date>=? AND date<=?"
    % ",".join("?" * len(symbols)), con, params=symbols + [DATE_LO, DATE_HI])
con.close()

sym_close, sym_idx, sym_dates = {}, {}, {}
for s, g in px.groupby("symbol"):
    g = g.sort_values("date")
    sym_dates[s] = g["date"].tolist()
    sym_close[s] = g["close"].tolist()
    sym_idx[s] = {d: i for i, d in enumerate(sym_dates[s])}

calendar = sorted(set(px["date"]))

def build_trades(df):
    out, skipped = [], 0
    for r in df.itertuples():
        s = r.symbol
        if r.entry_date not in sym_idx.get(s, {}) or r.exit_date not in sym_idx.get(s, {}):
            skipped += 1
            continue
        out.append(dict(symbol=s, entry_date=r.entry_date, exit_date=r.exit_date,
                        i0=sym_idx[s][r.entry_date], i1=sym_idx[s][r.exit_date],
                        p0=float(r.entry_price), net=float(r.pnl_pct)))
    return out, skipped

trades_A, skA = build_trades(ta)
trades_B, skB = build_trades(tb)
print(f"sleeve A: {len(trades_A)} trades (skip {skA}); sleeve B: {len(trades_B)} trades (skip {skB})")

def by_date(trades):
    d = defaultdict(list)
    for t in trades:
        d[t["entry_date"]].append(t)
    for k in d:
        d[k].sort(key=lambda t: t["symbol"])
    return d

entries_A, entries_B = by_date(trades_A), by_date(trades_B)

def dparse(s):
    return datetime.strptime(s, "%Y-%m-%d").date()

def leg_value(leg, dt):
    s = leg["symbol"]
    j = sym_idx[s].get(dt)
    if j is None:
        return leg["last_val"]
    i0, i1 = leg["i0"], leg["i1"]
    j = min(max(j, i0), i1)
    if i1 == i0:
        ratio = leg["ratio1"]
    else:
        w = (j - i0) / (i1 - i0)
        ratio = leg["ratio0"] + (leg["ratio1"] - leg["ratio0"]) * w
    v = leg["invested"] * (sym_close[s][j] * ratio) / leg["p0"]
    leg["last_val"] = v
    return v

def make_leg(t, size):
    s = t["symbol"]
    c0, c1 = sym_close[s][t["i0"]], sym_close[s][t["i1"]]
    exit_eff = t["p0"] * (1.0 + t["net"])
    return dict(symbol=s, i0=t["i0"], i1=t["i1"], invested=size, net=t["net"],
                p0=t["p0"], ratio0=t["p0"] / c0, ratio1=exit_eff / c1, last_val=size)


def run_dual(name, wA, wB, K_A=25, K_B=10, margin_mode=False):
    cash_A, cash_B = wA, wB
    margin, interest_total = 0.0, 0.0
    legs_A, legs_B = [], []
    exits_A, exits_B = defaultdict(list), defaultdict(list)
    st = dict(a_skip=0, a_fill=0, b_skip=0, b_fill=0, peak_margin_pct=0.0,
              days_at_cap=0, dbl_legdays=0, b_legdays=0, min_navB=1e9)
    nav_series = []
    prev_day = None
    open_syms_A = defaultdict(int)

    for dt in calendar:
        d_obj = dparse(dt)
        if margin > 1e-12 and prev_day is not None:
            intr = margin * MARGIN_RATE / 365.0 * (d_obj - prev_day).days
            interest_total += intr
            pay = min(cash_B, intr)
            cash_B -= pay
            margin += intr - pay
        prev_day = d_obj

        for leg in exits_A.get(dt, ()):
            cash_A += leg["invested"] * (1.0 + leg["net"])
            legs_A.remove(leg)
            open_syms_A[leg["symbol"]] -= 1
        for leg in exits_B.get(dt, ()):
            proceeds = leg["invested"] * (1.0 + leg["net"])
            if margin_mode:
                pay = min(margin, proceeds)
                margin -= pay
                cash_B += proceeds - pay
            else:
                cash_B += proceeds
            legs_B.remove(leg)

        posA = sum(leg_value(l, dt) for l in legs_A)
        nav_A = cash_A + posA
        for t in entries_A.get(dt, ()):
            size = nav_A / K_A
            if wA > 0 and cash_A + 1e-12 >= size:
                leg = make_leg(t, size)
                cash_A -= size
                legs_A.append(leg)
                exits_A[t["exit_date"]].append(leg)
                open_syms_A[t["symbol"]] += 1
                st["a_fill"] += 1
            else:
                st["a_skip"] += 1

        posB = sum(leg_value(l, dt) for l in legs_B)
        if margin_mode:
            nav_tot = cash_A + posA + cash_B + posB - margin
            for t in entries_B.get(dt, ()):
                size = MARGIN_CAP_FRAC * nav_tot / K_B
                headroom = max(0.0, MARGIN_CAP_FRAC * nav_tot - margin)
                if len(legs_B) < K_B and cash_B + headroom + 1e-12 >= size:
                    from_cash = min(cash_B, size)
                    draw = size - from_cash
                    cash_B -= from_cash
                    margin += draw
                    leg = make_leg(t, size)
                    legs_B.append(leg)
                    exits_B[t["exit_date"]].append(leg)
                    st["b_fill"] += 1
                else:
                    st["b_skip"] += 1
        else:
            nav_B = cash_B + posB
            for t in entries_B.get(dt, ()):
                size = nav_B / K_B
                if wB > 0 and cash_B + 1e-12 >= size:
                    leg = make_leg(t, size)
                    cash_B -= size
                    legs_B.append(leg)
                    exits_B[t["exit_date"]].append(leg)
                    st["b_fill"] += 1
                else:
                    st["b_skip"] += 1

        posA = sum(leg_value(l, dt) for l in legs_A)
        posB = sum(leg_value(l, dt) for l in legs_B)
        nav = cash_A + posA + cash_B + posB - margin
        nav_series.append((dt, nav))
        for leg in legs_B:
            st["b_legdays"] += 1
            if open_syms_A.get(leg["symbol"], 0) > 0:
                st["dbl_legdays"] += 1
        if margin_mode:
            navB_net = cash_B + posB - margin
            st["min_navB"] = min(st["min_navB"], navB_net)
            if margin > 1e-9:
                st["peak_margin_pct"] = max(st["peak_margin_pct"], margin / nav * 100)
                if margin >= MARGIN_CAP_FRAC * nav - 1e-9:
                    st["days_at_cap"] += 1

    ns = pd.DataFrame(nav_series, columns=["date", "nav"])
    ns["date"] = pd.to_datetime(ns["date"])
    nav = ns["nav"]
    final = nav.iloc[-1]
    years = (ns["date"].iloc[-1] - ns["date"].iloc[0]).days / 365.25
    cagr = final ** (1 / years) - 1
    maxdd = (nav / nav.cummax() - 1).min()
    # CAGR >=2022: neo NAV phien cuoi 2021
    pre22 = ns[ns["date"].dt.year <= 2021]
    nav22_0 = pre22["nav"].iloc[-1]
    d22_0 = pre22["date"].iloc[-1]
    years22 = (ns["date"].iloc[-1] - d22_0).days / 365.25
    cagr22 = (final / nav22_0) ** (1 / years22) - 1
    ns2 = ns.set_index("date")
    yearly = {}
    for y, g in ns2.groupby(ns2.index.year):
        prev = ns2["nav"][ns2.index.year < y]
        start = prev.iloc[-1] if len(prev) else 1.0
        yearly[int(y)] = g["nav"].iloc[-1] / start - 1.0
    print(f"{name}: final={final:.3f} CAGR={cagr*100:.2f}% CAGR>=22={cagr22*100:.2f}% "
          f"MaxDD={maxdd*100:.2f}% A fill/skip={st['a_fill']}/{st['a_skip']} "
          f"B fill/skip={st['b_fill']}/{st['b_skip']} int={interest_total:.4f} "
          f"peakM={st['peak_margin_pct']:.1f}% cap_days={st['days_at_cap']} "
          f"dblexp={st['dbl_legdays']}/{st['b_legdays']} "
          f"minNavB={st['min_navB'] if margin_mode else float('nan'):.3f}")
    print("  yearly: " + " ".join(f"{y}:{v*100:+.1f}%" for y, v in sorted(yearly.items())))
    return dict(name=name, final=final, cagr=cagr, cagr_ge2022=cagr22, maxdd=maxdd,
                interest=interest_total, yearly=yearly, **st)


results = []
results.append(run_dual("BASE_gbx08_K25", 1.0, 0.0))
results.append(run_dual("B_ONLY_K10", 0.0, 1.0))
results.append(run_dual("TL_90_10", 0.9, 0.1))
results.append(run_dual("TL_80_20", 0.8, 0.2))
results.append(run_dual("TL_80_20_KB5", 0.8, 0.2, K_B=5))
results.append(run_dual("TL_80_20_KB15", 0.8, 0.2, K_B=15))
results.append(run_dual("TL_MG_KB10", 1.0, 0.0, K_B=10, margin_mode=True))
results.append(run_dual("TL_MG_KB5", 1.0, 0.0, K_B=5, margin_mode=True))
results.append(run_dual("TL_MG_KB15", 1.0, 0.0, K_B=15, margin_mode=True))

pd.DataFrame([{k: (str(v) if isinstance(v, dict) else v) for k, v in r.items()}
              for r in results]).to_csv(
    BASE_DIR + "/runaway/dl_twolot_metrics.csv", index=False)
print("saved dl_twolot_metrics.csv")

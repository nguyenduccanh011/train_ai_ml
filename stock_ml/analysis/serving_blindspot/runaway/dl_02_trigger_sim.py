# -*- coding: utf-8 -*-
"""DUAL-LAYER — BAN THUC THI DUOC (trigger cross-X) — thi hanh dieu kien khep an.

Tai dung framework dl_01_dual_sim.py, thay sleeve B ex-post (trail10) bang lenh
trigger cross-X tu rw_trigger_X05.csv / rw_trigger_X03.csv (rw_03_trigger.py):
  - exit scheme = CHAMPION (sell-head) — scheme DUY NHAT co trong CSV trigger
    (trail10 chi ton tai cho cohort ex-post, khong cho ban trigger).
  - Tai tao ngay: ei = idx(signal_date) + cross_lag + 1; xi = ei + hold.
    entry_fill = close[ei] * 1.0015; pnl_net = tw_pnl (da net slip+fee, rw_00 verify).
  - Dual-layer pool rieng => limit sleeve A KHONG bi huy: lay TOAN BO fires
    (add + cancel_fill) lam sleeve B. rw_03 da dedup sequential per-symbol
    (open_until), nen khong trung slot trong cung ma.
  - Loai lenh end_of_data: xi cham bar cuoi cua ma (exit cuong buc, chua dong).

Kich ban (het dl_01): BASE / DLT_90_10 / DLT_80_20 / DLT_MG (A 100% NAV +
B margin 50% @15%/nam lai theo ngay du no that). K_A=25, K_B=10, cost model giu nguyen.
"""
import sqlite3
from collections import defaultdict
from datetime import datetime

import pandas as pd

BASE_DIR = "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot"
A_CSV = BASE_DIR + "/signalq/nicheloss/gb_x08_s42_trades.csv"
DB_PATH = "C:/Users/DUC CANH PC/Desktop/stock-serving/data/ohlcv.db"
DATE_LO, DATE_HI = "2020-01-01", "2026-07-08"
MARGIN_RATE = 0.15
MARGIN_CAP_FRAC = 0.50
SLIP_IN = 1.0015

ta = pd.read_csv(A_CSV)

# ---------- tai tao sleeve B tu trigger CSV ----------
con = sqlite3.connect(DB_PATH)
px_all = pd.read_sql_query(
    "SELECT symbol, date, close FROM ohlcv ORDER BY symbol, date", con)
con.close()

FULL = {}
for s, g in px_all.groupby("symbol"):
    g = g.reset_index(drop=True)
    FULL[s] = dict(dates=g["date"].tolist(),
                   idx={d: i for i, d in enumerate(g["date"])},
                   c=g["close"].tolist())


def build_sleeveB(csv_name):
    d = pd.read_csv(BASE_DIR + "/runaway/" + csv_name)
    rows, dropped_eod, dropped_px = [], 0, 0
    for r in d.itertuples():
        a = FULL.get(r.symbol)
        if a is None or r.signal_date not in a["idx"]:
            dropped_px += 1
            continue
        i = a["idx"][r.signal_date]
        ei = i + int(r.cross_lag) + 1
        xi = ei + int(r.hold)
        n = len(a["c"])
        if xi >= n:
            dropped_px += 1
            continue
        if xi == n - 1:  # exit cuong buc tai bar cuoi cua ma = end_of_data
            dropped_eod += 1
            continue
        ed, xd = a["dates"][ei], a["dates"][xi]
        if xd > DATE_HI:
            dropped_eod += 1
            continue
        rows.append(dict(symbol=r.symbol, entry_date=ed, exit_date=xd,
                         entry_fill=a["c"][ei] * SLIP_IN, pnl_net=float(r.tw_pnl),
                         kind=r.kind))
    out = pd.DataFrame(rows)
    print(f"{csv_name}: {len(d)} fires -> {len(out)} lenh sleeve B "
          f"(loai end_of_data {dropped_eod}, thieu px {dropped_px}); "
          f"mean pnl {out.pnl_net.mean()*100:+.2f}%/lenh, WR {(out.pnl_net>0).mean():.2f}, "
          f"kind={out.kind.value_counts().to_dict()}")
    return out


# ---------- khung sim (het dl_01) ----------
def prep_market(tb):
    symbols = sorted(set(ta.symbol) | set(tb.symbol))
    con2 = sqlite3.connect(DB_PATH)
    px = pd.read_sql_query(
        "SELECT symbol, date, close FROM ohlcv WHERE symbol IN (%s) AND date>=? AND date<=?"
        % ",".join("?" * len(symbols)), con2, params=symbols + [DATE_LO, DATE_HI])
    con2.close()
    sym_close, sym_idx = {}, {}
    for s, g in px.groupby("symbol"):
        g = g.sort_values("date")
        sym_close[s] = g["close"].tolist()
        sym_idx[s] = {d: i for i, d in enumerate(g["date"].tolist())}
    return sym_close, sym_idx, sorted(set(px["date"]))


def build_trades(df, kind, sym_idx):
    out, skipped = [], 0
    for r in df.itertuples():
        s = r.symbol
        if r.entry_date not in sym_idx.get(s, {}) or r.exit_date not in sym_idx.get(s, {}):
            skipped += 1
            continue
        i0, i1 = sym_idx[s][r.entry_date], sym_idx[s][r.exit_date]
        if kind == "A":
            p0, netp = float(r.entry_price), float(r.pnl_pct)
        else:
            p0, netp = float(r.entry_fill), float(r.pnl_net)
        out.append(dict(symbol=s, entry_date=r.entry_date, exit_date=r.exit_date,
                        i0=i0, i1=i1, p0=p0, net=netp))
    return out, skipped


def dparse(s):
    return datetime.strptime(s, "%Y-%m-%d").date()


def run_all(tag, tb):
    sym_close, sym_idx, calendar = prep_market(tb)
    trades_A, skA = build_trades(ta, "A", sym_idx)
    trades_B, skB = build_trades(tb, "B", sym_idx)
    print(f"[{tag}] sleeve A: {len(trades_A)} trades (skip {skA}); "
          f"sleeve B: {len(trades_B)} trades (skip {skB})")

    def by_date(trades):
        d = defaultdict(list)
        for t in trades:
            d[t["entry_date"]].append(t)
        for k in d:
            d[k].sort(key=lambda t: t["symbol"])
        return d

    entries_A, entries_B = by_date(trades_A), by_date(trades_B)

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
        ns2 = ns.set_index("date")
        yearly = {}
        for y, g in ns2.groupby(ns2.index.year):
            prev = ns2["nav"][ns2.index.year < y]
            start = prev.iloc[-1] if len(prev) else 1.0
            yearly[int(y)] = g["nav"].iloc[-1] / start - 1.0
        print(f"{name}: final={final:.3f} CAGR={cagr*100:.2f}% MaxDD={maxdd*100:.2f}% "
              f"A fill/skip={st['a_fill']}/{st['a_skip']} B fill/skip={st['b_fill']}/{st['b_skip']} "
              f"int={interest_total:.4f} peakM={st['peak_margin_pct']:.1f}% cap_days={st['days_at_cap']} "
              f"dblexp={st['dbl_legdays']}/{st['b_legdays']}")
        print("  yearly: " + " ".join(f"{y}:{v*100:+.1f}%" for y, v in sorted(yearly.items())))
        return dict(name=name, final=final, cagr=cagr, maxdd=maxdd, interest=interest_total,
                    yearly=yearly, **st)

    results = []
    results.append(run_dual(f"BASE_{tag}", 1.0, 0.0))
    results.append(run_dual(f"DLT_90_10_{tag}", 0.9, 0.1))
    results.append(run_dual(f"DLT_80_20_{tag}", 0.8, 0.2))
    results.append(run_dual(f"DLT_MG_{tag}", 1.0, 0.0, K_B=10, margin_mode=True))
    return results


all_res = []
for csv_name, tag in (("rw_trigger_X05.csv", "X05"), ("rw_trigger_X03.csv", "X03")):
    tb = build_sleeveB(csv_name)
    all_res.extend(run_all(tag, tb))

pd.DataFrame([{k: (str(v) if isinstance(v, dict) else v) for k, v in r.items()}
              for r in all_res]).to_csv(
    BASE_DIR + "/runaway/dlt_trigger_metrics.csv", index=False)
print("saved dlt_trigger_metrics.csv")

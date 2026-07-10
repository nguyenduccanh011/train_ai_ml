# -*- coding: utf-8 -*-
"""na_navlib: ban sao ngu nghia r2_nav.py, tham so hoa de audit.

Khac r2_nav.py DUY NHAT o cac cong tac (mac dinh = hanh vi goc):
  - roundtrip: None = dung pnl_pct trong CSV (cost nhung san 0.6% eff);
               so R = tong roundtrip muc tieu -> tai tao net tu gia raw:
               fee=0.004 giu nguyen, slippage s=(R-0.004)/2 moi chieu.
  - settle_lag: so NGAY GIAO DICH tien ban bi treo (0 = goc, dung ngay).
               NAV van tinh ca receivable; chi entry khong duoc xai.
  - order_seed: None = sort alphabet (goc); int = shuffle thu tu entry moi ngay.
Sanity anchor: mac dinh phai tai lap dung tung chu so ket qua da cong bo.
"""
import sqlite3
import random
from collections import defaultdict

import pandas as pd

DB_PATH = "C:/Users/DUC CANH PC/Desktop/stock-serving/data/ohlcv.db"
DATE_HI = "2026-07-08"
S0 = 0.001   # slippage nhung trong entry_price/exit_price cua CSV
FEE = 0.004  # 2*commission + tax, tru them vao pnl_pct


def load_px(symbols, date_lo):
    con = sqlite3.connect(DB_PATH)
    px = pd.read_sql_query(
        "SELECT symbol, date, close FROM ohlcv WHERE symbol IN (%s) AND date>=? AND date<=?"
        % ",".join("?" * len(symbols)), con, params=list(symbols) + [date_lo, DATE_HI])
    con.close()
    sym_close, sym_idx, sym_dates = {}, {}, {}
    for s, g in px.groupby("symbol"):
        g = g.sort_values("date")
        sym_dates[s] = g["date"].tolist()
        sym_close[s] = g["close"].tolist()
        sym_idx[s] = {d: i for i, d in enumerate(sym_dates[s])}
    calendar = sorted(set(px["date"]))
    return sym_close, sym_idx, calendar


def run_sim(csv, K=25, date_lo="2020-01-01", roundtrip=None, settle_lag=0,
            order_seed=None, record_legs_window=None):
    ta = pd.read_csv(csv)
    ta = ta[ta.entry_date.astype(str).str[:10] >= date_lo]
    symbols = sorted(set(ta.symbol))
    sym_close, sym_idx, calendar = load_px(symbols, date_lo)
    cal_pos = {d: i for i, d in enumerate(calendar)}

    trades, skipped = [], 0
    for r in ta.itertuples():
        s = r.symbol
        ed, xd = str(r.entry_date)[:10], str(r.exit_date)[:10]
        if ed not in sym_idx.get(s, {}) or xd not in sym_idx.get(s, {}):
            skipped += 1
            continue
        if roundtrip is None:
            net = float(r.pnl_pct)
        else:
            s_new = (roundtrip - FEE) / 2.0
            e_raw = float(r.entry_price) / (1.0 + S0)
            x_raw = float(r.exit_price) / (1.0 - S0)
            net = (x_raw * (1.0 - s_new)) / (e_raw * (1.0 + s_new)) - 1.0 - FEE
        trades.append(dict(symbol=s, entry_date=ed, exit_date=xd,
                           i0=sym_idx[s][ed], i1=sym_idx[s][xd],
                           p0=float(r.entry_price), net=net))

    entries = defaultdict(list)
    for t in trades:
        entries[t["entry_date"]].append(t)
    rng = random.Random(order_seed) if order_seed is not None else None
    for d in entries:
        entries[d].sort(key=lambda t: t["symbol"])
        if rng is not None:
            rng.shuffle(entries[d])

    def leg_value(leg, dt):
        s = leg["symbol"]
        j = sym_idx[s].get(dt)
        if j is None:
            return leg["last_val"]
        i0, i1 = leg["i0"], leg["i1"]
        j = min(max(j, i0), i1)
        ratio = leg["ratio1"] if i1 == i0 else \
            leg["ratio0"] + (leg["ratio1"] - leg["ratio0"]) * (j - i0) / (i1 - i0)
        v = leg["invested"] * (sym_close[s][j] * ratio) / leg["p0"]
        leg["last_val"] = v
        return v

    cash = 1.0                 # tien XAI DUOC
    pending = defaultdict(float)  # release_date -> tien ban dang treo (settlement)
    pend_total = 0.0
    legs = []
    exits = defaultdict(list)
    skip_cash = fills = 0
    nav_series = []
    occ_days = idle_sum = 0
    total_buys = 0.0
    legs_snap = []

    for di, dt in enumerate(calendar):
        cash += pending.pop(dt, 0.0)
        pend_total = sum(pending.values())
        for leg in exits.get(dt, ()):
            proceeds = leg["invested"] * (1.0 + leg["net"])
            if settle_lag <= 0:
                cash += proceeds
            else:
                ri = di + settle_lag
                if ri < len(calendar):
                    pending[calendar[ri]] += proceeds
                    pend_total += proceeds
                else:
                    pend_total += proceeds
                    pending["NEVER"] += proceeds  # cuoi mau, van la receivable
            legs.remove(leg)
        pos = sum(leg_value(l, dt) for l in legs)
        nav_now = cash + pend_total + pos
        for t in entries.get(dt, ()):
            size = nav_now / K
            if cash + 1e-12 >= size:
                s = t["symbol"]
                c0, c1 = sym_close[s][t["i0"]], sym_close[s][t["i1"]]
                exit_eff = t["p0"] * (1.0 + t["net"])
                leg = dict(symbol=s, i0=t["i0"], i1=t["i1"], invested=size,
                           net=t["net"], p0=t["p0"], ratio0=t["p0"] / c0,
                           ratio1=exit_eff / c1, last_val=size,
                           entry_date=dt, exit_date=t["exit_date"])
                cash -= size
                legs.append(leg)
                exits[t["exit_date"]].append(leg)
                fills += 1
                total_buys += size
            else:
                skip_cash += 1
        pos = sum(leg_value(l, dt) for l in legs)
        nav = cash + pend_total + pos
        nav_series.append((dt, nav, len(legs), (cash + pend_total) / nav))
        occ_days += len(legs)
        idle_sum += (cash + pend_total) / nav
        if record_legs_window and record_legs_window[0] <= dt <= record_legs_window[1]:
            legs_snap.append((dt, [(l["symbol"], l["entry_date"], l["exit_date"],
                                    l["invested"], leg_value(l, dt)) for l in legs],
                              cash + pend_total))

    ns = pd.DataFrame(nav_series, columns=["date", "nav", "n_open", "cash_frac"])
    ns["date"] = pd.to_datetime(ns["date"])
    nav = ns["nav"]
    final = float(nav.iloc[-1])
    years = (ns["date"].iloc[-1] - ns["date"].iloc[0]).days / 365.25
    cagr = final ** (1 / years) - 1
    dd = nav / nav.cummax() - 1
    maxdd = float(dd.min())
    maxdd_date = ns["date"].iloc[dd.idxmin()].date()
    yearly = {}
    ns2 = ns.set_index("date")
    for y, g in ns2.groupby(ns2.index.year):
        prev = ns2["nav"][ns2.index.year < y]
        start = prev.iloc[-1] if len(prev) else 1.0
        yearly[int(y)] = float(g["nav"].iloc[-1] / start - 1.0)
    nd = len(calendar)
    turnover = total_buys / years / nav.mean()
    return dict(final=final, cagr=cagr, maxdd=maxdd, maxdd_date=str(maxdd_date),
                fills=fills, skip_cash=skip_cash, skipped_db=skipped,
                avg_open=occ_days / nd, idle=idle_sum / nd, turnover=turnover,
                yearly=yearly, ns=ns, legs_snap=legs_snap)


def fmt(m):
    return (f"final=x{m['final']:.2f} CAGR={m['cagr']*100:.2f}% MaxDD={m['maxdd']*100:.2f}%"
            f" ({m['maxdd_date']}) fills={m['fills']} skip_cash={m['skip_cash']}"
            f" turnover={m['turnover']:.1f}x/nam")

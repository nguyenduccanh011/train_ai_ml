# -*- coding: utf-8 -*-
"""Daily mark-to-market portfolio simulator for champion vs pyramid configs.

NAV0 = 1.0, fully compounding. Base position size f = 1/K of CURRENT NAV at entry.
Add leg (pyramid): on 3rd trading bar after entry (per-symbol calendar), same size
as the trade's base invested amount, funded cash-first then margin (if allowed).
Margin: cap 50% of current NAV, 15%/yr accrued daily (rate/365 * calendar days
between trading dates) on outstanding balance. Margin repaid first from exits.

Price paths anchored per trade: shares = invested/entry_price_csv; daily value uses
db close series with a scale ratio linearly interpolated (in bar index) between
ratio0 = entry_price_csv/close_db(entry_bar) and ratio1 = exit_eff/close_db(exit_bar),
where exit_eff = entry_price_csv*(1+net_pnl). This makes entry AND exit anchors match
CSV net pnl exactly (neutralizes raw-vs-adjusted drift); realized cash at exit =
invested*(1+net). Missing daily close -> carry last value.
"""
import sqlite3
import sys
from collections import defaultdict
from datetime import date, datetime

import pandas as pd

BASE_DIR = "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/week0"
CHAMP_CSV = BASE_DIR + "/best_trades/trades_n2_2643_wavestruct_la05_lamp02.csv"
CAND_CSV = BASE_DIR + "/best_trades/trades_w0_pyr_u10_r02.csv"
DB_PATH = "C:/Users/DUC CANH PC/Desktop/stock-serving/data/ohlcv.db"
DATE_LO, DATE_HI = "2020-01-01", "2026-07-08"
MARGIN_RATE = 0.15
MARGIN_CAP_FRAC = 0.50
ADD_SLIP = 1.0015  # informational; add pnl already net in add_net

# ---------------- data loading ----------------
champ = pd.read_csv(CHAMP_CSV)
cand = pd.read_csv(CAND_CSV)
key = ["symbol", "entry_date", "exit_date"]
m = champ.merge(cand[key + ["weight", "pnl_pct"]], on=key, suffixes=("", "_cand"))
assert len(m) == len(champ) == 1384, f"merge mismatch {len(m)}"
m["add_net"] = m["pnl_pct_cand"] - m["pnl_pct"]
m["has_add"] = m["weight_cand"] == 2.0 if "weight_cand" in m else m["weight"] == 2.0
# weight col name after merge: champ has weight, cand weight -> weight_cand
m["has_add"] = m["weight_cand"] == 2.0
n_adds_flag = int(m["has_add"].sum())
assert n_adds_flag == 524, f"expected 524 flagged adds, got {n_adds_flag}"

symbols = sorted(m["symbol"].unique())
con = sqlite3.connect(DB_PATH)
px = pd.read_sql_query(
    "SELECT symbol, date, close FROM ohlcv WHERE symbol IN (%s) AND date>=? AND date<=?"
    % ",".join("?" * len(symbols)),
    con, params=symbols + [DATE_LO, DATE_HI])
con.close()

sym_dates, sym_close, sym_idx = {}, {}, {}
for s, g in px.groupby("symbol"):
    g = g.sort_values("date")
    ds = g["date"].tolist()
    sym_dates[s] = ds
    sym_close[s] = g["close"].tolist()
    sym_idx[s] = {d: i for i, d in enumerate(ds)}

calendar = sorted(set(px["date"]))
cal_idx = {d: i for i, d in enumerate(calendar)}

def dparse(s):
    return datetime.strptime(s, "%Y-%m-%d").date()

# precompute trade structures
trades = []
skipped_data = []
for r in m.itertuples():
    s = r.symbol
    if r.entry_date not in sym_idx[s] or r.exit_date not in sym_idx[s]:
        skipped_data.append((s, r.entry_date, r.exit_date))
        continue
    i0 = sym_idx[s][r.entry_date]
    i1 = sym_idx[s][r.exit_date]
    add_i = i0 + 3
    has_add = bool(r.has_add) and add_i < i1 and add_i < len(sym_dates[s])
    trades.append(dict(
        symbol=s, entry_date=r.entry_date, exit_date=r.exit_date,
        entry_price=r.entry_price,
        i0=i0, i1=i1, net=r.pnl_pct, add_net=r.add_net,
        has_add=has_add, add_date=sym_dates[s][add_i] if has_add else None,
        add_i=add_i if has_add else None))
assert not skipped_data, f"trades with dates missing in db: {skipped_data[:5]}"
n_add_dropped = n_adds_flag - sum(t["has_add"] for t in trades)

entries_by_date = defaultdict(list)
for t in trades:
    entries_by_date[t["entry_date"]].append(t)
for d in entries_by_date:
    entries_by_date[d].sort(key=lambda t: t["symbol"])

# ---------------- simulator ----------------
def leg_value(leg, dt):
    """MTM value of a leg at trading date dt (carry last if no bar)."""
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


def run_config(K, base_frac_mult, do_adds, allow_margin):
    f = base_frac_mult / K
    cash, margin, interest_total = 1.0, 0.0, 0.0
    open_legs = []            # list of leg dicts
    exits_by_date = defaultdict(list)
    adds_by_date = defaultdict(list)
    stats = dict(base_skipped=0, adds_exec=0, adds_skipped=0,
                 peak_margin_pct=0.0, days_at_cap=0,
                 realized_check_maxerr=0.0)
    nav_series = []
    prev_cal_day = None
    trade_add_info = {}       # id(trade)->base invested (for add sizing)

    for dt in calendar:
        d_obj = dparse(dt)
        # 1) interest accrual on margin
        if margin > 1e-12 and prev_cal_day is not None:
            days = (d_obj - prev_cal_day).days
            intr = margin * MARGIN_RATE / 365.0 * days
            cash -= intr
            interest_total += intr
            if cash < 0:  # negative cash treated as extra margin draw
                margin += -cash
                cash = 0.0
        prev_cal_day = d_obj

        # 2) exits (repay margin first from inflow)
        for leg in exits_by_date.get(dt, ()):
            proceeds = leg["invested"] * (1.0 + leg["net"])
            # realized-pnl sanity: realized/invested == csv net by construction
            err = abs((proceeds - leg["invested"]) / leg["invested"] - leg["net"])
            stats["realized_check_maxerr"] = max(stats["realized_check_maxerr"], err)
            cash += proceeds
            open_legs.remove(leg)
        if margin > 1e-12 and cash > 0:
            pay = min(margin, cash)
            margin -= pay
            cash -= pay

        # NAV at this point (positions at today's closes)
        pos_val = sum(leg_value(l, dt) for l in open_legs)
        nav_now = cash + pos_val - margin
        over_cap = margin > MARGIN_CAP_FRAC * nav_now + 1e-12

        # 3) base entries
        for t in entries_by_date.get(dt, ()):
            size = f * nav_now
            if cash + 1e-12 >= size and not over_cap:
                s = t["symbol"]
                c0 = sym_close[s][t["i0"]]
                c1 = sym_close[s][t["i1"]]
                entry_price = float(t["entry_price"])
                exit_eff = entry_price * (1.0 + t["net"])
                leg = dict(symbol=s, i0=t["i0"], i1=t["i1"], invested=size,
                           net=t["net"], p0=entry_price,
                           ratio0=entry_price / c0, ratio1=exit_eff / c1,
                           last_val=size)
                cash -= size
                open_legs.append(leg)
                exits_by_date[t["exit_date"]].append(leg)
                if do_adds and t["has_add"]:
                    adds_by_date[t["add_date"]].append((t, size))
            else:
                stats["base_skipped"] += 1

        # 4) adds
        if do_adds:
            pend = sorted(adds_by_date.get(dt, ()), key=lambda x: x[0]["symbol"])
            for t, base_size in pend:
                req = base_size
                # recompute nav for cap check (cash changed by entries, nav same)
                nav_now2 = cash + sum(leg_value(l, dt) for l in open_legs) - margin
                headroom = max(0.0, MARGIN_CAP_FRAC * nav_now2 - margin) if allow_margin else 0.0
                if cash + headroom + 1e-12 >= req:
                    from_cash = min(cash, req)
                    draw = req - from_cash
                    cash -= from_cash
                    margin += draw
                    s = t["symbol"]
                    ia, i1 = t["add_i"], t["i1"]
                    ca = sym_close[s][ia]
                    c1 = sym_close[s][i1]
                    # add entry anchor: anchored close*1.0015 -> but net add_net is
                    # already per unit at that entry; anchor entry ratio 1 relative
                    # to itself: use p0 = anchored add price
                    entry_price = float(t["entry_price"])
                    # anchored close at add bar via linear ratio interp of BASE leg
                    exit_eff_base = entry_price * (1.0 + t["net"])
                    r0 = entry_price / sym_close[s][t["i0"]]
                    r1 = exit_eff_base / c1
                    w = (ia - t["i0"]) / (i1 - t["i0"])
                    ratio_at_add = r0 + (r1 - r0) * w
                    add_p0 = sym_close[s][ia] * ratio_at_add * ADD_SLIP
                    add_exit_eff = add_p0 * (1.0 + t["add_net"])
                    leg = dict(symbol=s, i0=ia, i1=i1, invested=req,
                               net=t["add_net"], p0=add_p0,
                               ratio0=add_p0 / ca, ratio1=add_exit_eff / c1,
                               last_val=req)
                    open_legs.append(leg)
                    exits_by_date[t["exit_date"]].append(leg)
                    stats["adds_exec"] += 1
                else:
                    stats["adds_skipped"] += 1

        # 5) end-of-day MTM
        pos_val = sum(leg_value(l, dt) for l in open_legs)
        nav = cash + pos_val - margin
        nav_series.append((dt, nav, margin, cash))
        if margin > 1e-9:
            mp = margin / nav * 100.0
            stats["peak_margin_pct"] = max(stats["peak_margin_pct"], mp)
            if margin >= MARGIN_CAP_FRAC * nav - 1e-9:
                stats["days_at_cap"] += 1

    ns = pd.DataFrame(nav_series, columns=["date", "nav", "margin", "cash"])
    ns["date"] = pd.to_datetime(ns["date"])
    stats["interest_total"] = interest_total
    return ns, stats


def metrics(ns, stats, K, name):
    nav = ns["nav"]
    final = nav.iloc[-1]
    years = (ns["date"].iloc[-1] - ns["date"].iloc[0]).days / 365.25
    cagr = final ** (1 / years) - 1
    peak = nav.cummax()
    dd = (nav / peak - 1.0)
    maxdd = dd.min()
    ret = nav.pct_change()
    worst_i = ret.idxmin()
    worst_day = ret.min()
    worst_date = ns["date"].iloc[worst_i].date() if pd.notna(worst_day) else None
    ns2 = ns.set_index("date")
    yearly = {}
    for y, g in ns2.groupby(ns2.index.year):
        prev = ns2["nav"][ns2.index.year < y]
        start = prev.iloc[-1] if len(prev) else 1.0
        yearly[int(y)] = g["nav"].iloc[-1] / start - 1.0
    big_moves = int((ret.abs() > 0.15).sum())
    profit = final - 1.0
    it = stats["interest_total"]
    return dict(
        name=name, K=K, final_nav=final, cagr=cagr, maxdd=maxdd,
        worst_day=worst_day, worst_date=str(worst_date),
        yearly=yearly, big_moves=big_moves,
        interest=it, interest_pct_nav0=it * 100,
        interest_pct_profit=(it / profit * 100) if profit > 0 else float("nan"),
        **stats)


CONFIGS = []
for K in (25, 30):
    CONFIGS += [
        (f"CHAMPION_K{K}", K, 1.0, False, False),
        (f"PYR_AGGR_MARGIN_K{K}", K, 1.0, True, True),
        (f"PYR_AGGR_CASHONLY_K{K}", K, 1.0, True, False),
        (f"PYR_DEFENSIVE_K{K}", K, 2.0 / 3.0, True, False),
    ]

results = []
nav_frames = {}
for name, K, bf, do_adds, allow_m in CONFIGS:
    ns, st = run_config(K, bf, do_adds, allow_m)
    res = metrics(ns, st, K, name)
    results.append(res)
    nav_frames[name] = ns
    print(f"{name}: final={res['final_nav']:.3f} CAGR={res['cagr']*100:.2f}% "
          f"MaxDD={res['maxdd']*100:.2f}% baseskip={res['base_skipped']} "
          f"adds={res['adds_exec']}/{res['adds_skipped']} "
          f"int={res['interest']:.4f} peakM={res['peak_margin_pct']:.1f}% "
          f"cap_days={res['days_at_cap']} worst_day={res['worst_day']*100:.2f}% "
          f"({res['worst_date']}) big15={res['big_moves']} "
          f"realized_err={res['realized_check_maxerr']:.2e}")
    print("  yearly: " + " ".join(f"{y}:{v*100:+.1f}%" for y, v in sorted(res["yearly"].items())))

print(f"\nadd flags dropped (add bar >= exit bar or missing): {n_add_dropped}")

# persist detail for report
out = pd.DataFrame([{k: (v if not isinstance(v, dict) else str({a: round(b, 4) for a, b in v.items()})) for k, v in r.items()} for r in results])
out.to_csv(BASE_DIR + "/portfolio_sim_metrics.csv", index=False)
print("saved metrics csv")

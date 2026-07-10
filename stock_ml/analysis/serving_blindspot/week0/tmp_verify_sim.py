# -*- coding: utf-8 -*-
"""Independent verification of portfolio_sim.py (adversarial re-run).

Reimplements the simulator with:
  - a daily margin ledger (debug) to independently recompute cumulative interest
  - parametrized interest rate (for +/-20% sensitivity)
  - parametrized add bar offset (for +/-1 bar sensitivity)
  - alternative leg valuation mode 'hold0' (ratio frozen at ratio0, realized pnl
    booked as a jump at exit) to quantify the look-ahead in the interp MTM path
Writes nothing except stdout.
"""
import sqlite3
from collections import defaultdict
from datetime import datetime

import pandas as pd

BASE_DIR = "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/week0"
CHAMP_CSV = BASE_DIR + "/best_trades/trades_n2_2643_wavestruct_la05_lamp02.csv"
CAND_CSV = BASE_DIR + "/best_trades/trades_w0_pyr_u10_r02.csv"
DB_PATH = "C:/Users/DUC CANH PC/Desktop/stock-serving/data/ohlcv.db"
DATE_LO, DATE_HI = "2020-01-01", "2026-07-08"
MARGIN_CAP_FRAC = 0.50
ADD_SLIP = 1.0015

champ = pd.read_csv(CHAMP_CSV)
cand = pd.read_csv(CAND_CSV)
key = ["symbol", "entry_date", "exit_date"]
m = champ.merge(cand[key + ["weight", "pnl_pct"]], on=key, suffixes=("", "_cand"))
assert len(m) == len(champ) == 1384, len(m)
m["add_net"] = m["pnl_pct_cand"] - m["pnl_pct"]
m["has_add"] = m["weight_cand"] == 2.0
n_flags = int(m["has_add"].sum())
print("flagged adds:", n_flags)

symbols = sorted(m["symbol"].unique())
con = sqlite3.connect(DB_PATH)
px = pd.read_sql_query(
    "SELECT symbol, date, close FROM ohlcv WHERE symbol IN (%s) AND date>=? AND date<=?"
    % ",".join("?" * len(symbols)), con, params=symbols + [DATE_LO, DATE_HI])
con.close()

sym_dates, sym_close, sym_idx = {}, {}, {}
for s, g in px.groupby("symbol"):
    g = g.sort_values("date")
    sym_dates[s] = g["date"].tolist()
    sym_close[s] = g["close"].tolist()
    sym_idx[s] = {d: i for i, d in enumerate(sym_dates[s])}
calendar = sorted(set(px["date"]))
print("calendar:", calendar[0], "->", calendar[-1], "n_days:", len(calendar))


def dparse(s):
    return datetime.strptime(s, "%Y-%m-%d").date()


def build_trades(add_offset):
    trades, dropped = [], 0
    for r in m.itertuples():
        s = r.symbol
        i0 = sym_idx[s][r.entry_date]
        i1 = sym_idx[s][r.exit_date]
        ai = i0 + add_offset
        ha = bool(r.has_add) and ai < i1 and ai < len(sym_dates[s])
        if bool(r.has_add) and not ha:
            dropped += 1
        trades.append(dict(symbol=s, entry_date=r.entry_date, exit_date=r.exit_date,
                           entry_price=r.entry_price, i0=i0, i1=i1, net=r.pnl_pct,
                           add_net=r.add_net, has_add=ha,
                           add_date=sym_dates[s][ai] if ha else None,
                           add_i=ai if ha else None))
    ebd = defaultdict(list)
    for t in trades:
        ebd[t["entry_date"]].append(t)
    for d in ebd:
        ebd[d].sort(key=lambda t: t["symbol"])
    return trades, ebd, dropped


def leg_value(leg, dt, mode):
    s = leg["symbol"]
    j = sym_idx[s].get(dt)
    if j is None:
        return leg["last_val"]
    i0, i1 = leg["i0"], leg["i1"]
    j = min(max(j, i0), i1)
    if mode == "hold0":
        ratio = leg["ratio0"]
    elif i1 == i0:
        ratio = leg["ratio1"]
    else:
        w = (j - i0) / (i1 - i0)
        ratio = leg["ratio0"] + (leg["ratio1"] - leg["ratio0"]) * w
    v = leg["invested"] * (sym_close[s][j] * ratio) / leg["p0"]
    leg["last_val"] = v
    return v


def run_config(K, bf, do_adds, allow_margin, rate=0.15, add_offset=3, mode="interp"):
    trades, entries_by_date, dropped = build_trades(add_offset)
    f = bf / K
    cash, margin, interest_total = 1.0, 0.0, 0.0
    open_legs = []
    exits_by_date = defaultdict(list)
    adds_by_date = defaultdict(list)
    st = dict(base_skipped=0, adds_exec=0, adds_skipped=0, adds_queued=0,
              peak_margin_pct=0.0, days_at_cap=0, err=0.0, flags_dropped=dropped)
    nav_series, ledger = [], []
    prev = None
    executed_bases = set()
    for dt in calendar:
        d = dparse(dt)
        intr_today, gap = 0.0, 0
        if margin > 1e-12 and prev is not None:
            gap = (d - prev).days
            intr_today = margin * rate / 365.0 * gap
            cash -= intr_today
            interest_total += intr_today
            if cash < 0:
                margin += -cash
                cash = 0.0
        prev = d
        for leg in exits_by_date.get(dt, ()):
            proceeds = leg["invested"] * (1.0 + leg["net"])
            st["err"] = max(st["err"], abs((proceeds - leg["invested"]) / leg["invested"] - leg["net"]))
            cash += proceeds
            open_legs.remove(leg)
        if margin > 1e-12 and cash > 0:
            pay = min(margin, cash)
            margin -= pay
            cash -= pay
        pos_val = sum(leg_value(l, dt, mode) for l in open_legs)
        nav_now = cash + pos_val - margin
        over_cap = margin > MARGIN_CAP_FRAC * nav_now + 1e-12
        for t in entries_by_date.get(dt, ()):
            size = f * nav_now
            if cash + 1e-12 >= size and not over_cap:
                s = t["symbol"]
                c0, c1 = sym_close[s][t["i0"]], sym_close[s][t["i1"]]
                ep = float(t["entry_price"])
                leg = dict(symbol=s, i0=t["i0"], i1=t["i1"], invested=size, net=t["net"],
                           p0=ep, ratio0=ep / c0, ratio1=ep * (1.0 + t["net"]) / c1,
                           last_val=size)
                cash -= size
                open_legs.append(leg)
                exits_by_date[t["exit_date"]].append(leg)
                executed_bases.add((t["symbol"], t["entry_date"], t["exit_date"]))
                if do_adds and t["has_add"]:
                    adds_by_date[t["add_date"]].append((t, size))
                    st["adds_queued"] += 1
            else:
                st["base_skipped"] += 1
        if do_adds:
            for t, base_size in sorted(adds_by_date.get(dt, ()), key=lambda x: x[0]["symbol"]):
                req = base_size
                nav2 = cash + sum(leg_value(l, dt, mode) for l in open_legs) - margin
                headroom = max(0.0, MARGIN_CAP_FRAC * nav2 - margin) if allow_margin else 0.0
                if cash + headroom + 1e-12 >= req:
                    from_cash = min(cash, req)
                    cash -= from_cash
                    margin += req - from_cash
                    s = t["symbol"]
                    ia, i1 = t["add_i"], t["i1"]
                    ca, c1 = sym_close[s][ia], sym_close[s][i1]
                    ep = float(t["entry_price"])
                    r0 = ep / sym_close[s][t["i0"]]
                    r1 = ep * (1.0 + t["net"]) / c1
                    w = (ia - t["i0"]) / (i1 - t["i0"])
                    add_p0 = ca * (r0 + (r1 - r0) * w) * ADD_SLIP
                    leg = dict(symbol=s, i0=ia, i1=i1, invested=req, net=t["add_net"],
                               p0=add_p0, ratio0=add_p0 / ca,
                               ratio1=add_p0 * (1.0 + t["add_net"]) / c1, last_val=req)
                    open_legs.append(leg)
                    exits_by_date[t["exit_date"]].append(leg)
                    st["adds_exec"] += 1
                else:
                    st["adds_skipped"] += 1
        pos_val = sum(leg_value(l, dt, mode) for l in open_legs)
        nav = cash + pos_val - margin
        nav_series.append((dt, nav, margin, cash))
        ledger.append(dict(date=dt, margin_eod=margin, intr=intr_today, gap=gap))
        if margin > 1e-9:
            mp = margin / nav * 100.0
            st["peak_margin_pct"] = max(st["peak_margin_pct"], mp)
            if margin >= MARGIN_CAP_FRAC * nav - 1e-9:
                st["days_at_cap"] += 1
    ns = pd.DataFrame(nav_series, columns=["date", "nav", "margin", "cash"])
    ns["date"] = pd.to_datetime(ns["date"])
    st["interest_total"] = interest_total
    st["executed_bases"] = executed_bases
    return ns, st, ledger


def metrics(ns):
    nav = ns["nav"]
    final = nav.iloc[-1]
    years = (ns["date"].iloc[-1] - ns["date"].iloc[0]).days / 365.25
    cagr = final ** (1 / years) - 1
    dd = (nav / nav.cummax() - 1.0).min()
    ret = nav.pct_change()
    wd = ret.min()
    wdate = ns["date"].iloc[ret.idxmin()].date()
    ns2 = ns.set_index("date")
    yearly = {}
    for y, g in ns2.groupby(ns2.index.year):
        prevnav = ns2["nav"][ns2.index.year < y]
        start = prevnav.iloc[-1] if len(prevnav) else 1.0
        yearly[int(y)] = g["nav"].iloc[-1] / start - 1.0
    big = int((ret.abs() > 0.15).sum())
    return final, cagr, dd, wd, str(wdate), yearly, big, years


CLAIMS = {
    "CHAMPION_K25":         (25, 1.0, False, False, 14.056, 50.05, -13.76, 0.0,   0,   0, 356),
    "PYR_AGGR_MARGIN_K25":  (25, 1.0, True,  True,  16.835, 54.26, -18.51, 10.96, 280, 0, 521),
    "PYR_AGGR_CASHONLY_K25":(25, 1.0, True,  False, 14.659, 51.02, -17.87, 0.0,   211, 84, 481),
    "PYR_DEFENSIVE_K25":    (25, 2/3, True,  False, 12.929, 48.14, -14.99, 0.0,   295, 82, 304),
    "CHAMPION_K30":         (30, 1.0, False, False, 12.274, 46.96, -13.35, 0.0,   0,   0, 257),
    "PYR_AGGR_MARGIN_K30":  (30, 1.0, True,  True,  14.473, 50.72, -16.11, 8.85,  317, 0, 444),
    "PYR_AGGR_CASHONLY_K30":(30, 1.0, True,  False, 13.524, 49.16, -16.35, 0.0,   242, 93, 402),
    "PYR_DEFENSIVE_K30":    (30, 2/3, True,  False, 11.206, 44.92, -13.94, 0.0,   344, 58, 238),
}

print("\n=== A) reproduction vs claimed ===")
repro = {}
for name, (K, bf, da, am, c_fn, c_cagr, c_dd, c_int, c_ae, c_as, c_bs) in CLAIMS.items():
    ns, st, led = run_config(K, bf, da, am)
    fn, cagr, dd, wd, wdate, yearly, big, years = metrics(ns)
    repro[name] = (ns, st, led, fn, cagr, dd, yearly)
    ok = (abs(fn - c_fn) < 5e-3 and abs(cagr * 100 - c_cagr) < 5e-2
          and abs(dd * 100 - c_dd) < 5e-2 and abs(st["interest_total"] * 100 - c_int) < 5e-2
          and st["adds_exec"] == c_ae and st["adds_skipped"] == c_as
          and st["base_skipped"] == c_bs)
    print(f"{name}: final={fn:.3f} (claim {c_fn}) cagr={cagr*100:.2f} (claim {c_cagr}) "
          f"dd={dd*100:.2f} (claim {c_dd}) int%NAV0={st['interest_total']*100:.2f} (claim {c_int}) "
          f"adds={st['adds_exec']}/{st['adds_skipped']} (claim {c_ae}/{c_as}) "
          f"baseskip={st['base_skipped']} (claim {c_bs}) queued={st['adds_queued']} "
          f"dropped_flags={st['flags_dropped']} peakM={st['peak_margin_pct']:.1f} cap={st['days_at_cap']} "
          f"worstday={wd*100:.2f}%@{wdate} big15={big} err={st['err']:.1e} MATCH={ok}")
    print("   yearly: " + " ".join(f"{y}:{v*100:+.1f}" for y, v in sorted(yearly.items())))
print(f"years used for CAGR: {years:.4f}")

print("\n=== B) independent interest recompute from daily ledger ===")
for name in ("PYR_AGGR_MARGIN_K25", "PYR_AGGR_MARGIN_K30"):
    ns, st, led, *_ = repro[name]
    rate = 0.15
    indep = 0.0
    for i in range(1, len(led)):
        mprev = led[i - 1]["margin_eod"]
        if mprev > 1e-12:
            gap = (dparse(led[i]["date"]) - dparse(led[i - 1]["date"])).days
            indep += mprev * rate / 365.0 * gap
    print(f"{name}: sim_accum={st['interest_total']:.10f} indep_from_ledger={indep:.10f} "
          f"absdiff={abs(indep - st['interest_total']):.2e} "
          f"pct_NAV0={indep*100:.3f} (claim {'10.96' if 'K25' in name else '8.85'})")
    prof = ns['nav'].iloc[-1] - 1.0
    print(f"   interest as % of profit = {st['interest_total']/prof*100:.3f}")

print("\n=== C) skip-count consistency / add accounting ===")
for K in (25, 30):
    ch = repro[f"CHAMPION_K{K}"][1]["base_skipped"]
    print(f"K={K}: champion baseskip={ch}")
print("champion K30 <= K25:", repro["CHAMPION_K30"][1]["base_skipped"] <= repro["CHAMPION_K25"][1]["base_skipped"])
for name in CLAIMS:
    st = repro[name][1]
    q = st["adds_queued"]
    print(f"{name}: queued={q} exec+skip={st['adds_exec']+st['adds_skipped']} "
          f"flags_on_skipped_or_dropped={n_flags - q} consistent={q == st['adds_exec']+st['adds_skipped']}")

print("\n=== D1) interest-rate sensitivity +/-20% (margin configs) ===")
for K in (25, 30):
    ch_fn = repro[f"CHAMPION_K{K}"][3]
    for rate in (0.12, 0.15, 0.18):
        ns, st, _ = run_config(K, 1.0, True, True, rate=rate)
        fn, cagr, dd, *_ = metrics(ns)
        print(f"PYR_AGGR_MARGIN_K{K} rate={rate}: final={fn:.3f} cagr={cagr*100:.2f} dd={dd*100:.2f} "
              f"int%NAV0={st['interest_total']*100:.2f} vs champ final={ch_fn:.3f} -> beats={fn > ch_fn}")

print("\n=== D2) add timing +/-1 bar (all pyramid configs) ===")
for K in (25, 30):
    ch_fn, ch_cagr, ch_dd = repro[f"CHAMPION_K{K}"][3], repro[f"CHAMPION_K{K}"][4], repro[f"CHAMPION_K{K}"][5]
    for name, bf, am in ((f"PYR_AGGR_MARGIN_K{K}", 1.0, True),
                         (f"PYR_AGGR_CASHONLY_K{K}", 1.0, False),
                         (f"PYR_DEFENSIVE_K{K}", 2/3, False)):
        for off in (2, 3, 4):
            ns, st, _ = run_config(K, bf, True, am, add_offset=off)
            fn, cagr, dd, *_ = metrics(ns)
            print(f"{name} addbar+{off}: final={fn:.3f} cagr={cagr*100:.2f} dd={dd*100:.2f} "
                  f"adds={st['adds_exec']}/{st['adds_skipped']} baseskip={st['base_skipped']} "
                  f"dropped={st['flags_dropped']} vs champ cagr={ch_cagr*100:.2f} dd={ch_dd*100:.2f}")

print("\n=== E) valuation-mode robustness (hold0 = no exit-anchored interp, quantifies MTM look-ahead) ===")
for name, (K, bf, da, am, *_rest) in CLAIMS.items():
    ns, st, _ = run_config(K, bf, da, am, mode="hold0")
    fn, cagr, dd, *_ = metrics(ns)
    fn0, cagr0, dd0 = repro[name][3], repro[name][4], repro[name][5]
    print(f"{name}: interp final={fn0:.3f} dd={dd0*100:.2f} | hold0 final={fn:.3f} dd={dd*100:.2f} "
          f"d_final={fn-fn0:+.3f} d_dd={100*(dd-dd0):+.2f}pt")

print("\n=== F) fairness: executed base-trade overlap at K=25 ===")
base_ch = repro["CHAMPION_K25"][1]["executed_bases"]
for name in ("PYR_AGGR_MARGIN_K25", "PYR_AGGR_CASHONLY_K25", "PYR_DEFENSIVE_K25"):
    b = repro[name][1]["executed_bases"]
    print(f"{name}: exec={len(b)} champ_exec={len(base_ch)} common={len(b & base_ch)} "
          f"only_champ={len(base_ch - b)} only_this={len(b - base_ch)}")

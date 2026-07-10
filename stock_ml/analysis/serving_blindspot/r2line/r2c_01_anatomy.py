# -*- coding: utf-8 -*-
"""r2c_01: giai phau r2b_oxtrail04 vs r2_c2_pb40snr.

1. Exit-reason mix hai he.
2. Paired diff theo (symbol, entry_date): cohort nao doi exit -> delta pnl/hold theo nam.
3. Yearly NAV returns (nh_nav2, K25, R0.6, lag2+adv 0.08%, mean 20 perm) full-frame
   -> nam nao oxtrail thua/thang c2 va gb; soi rieng 2022+ (f22).
Log: r2c_01_out.txt (tee).
"""
import sys
from collections import defaultdict

import pandas as pd

sys.path.insert(0, "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/r2line/na_audit")
from nh_nav2 import NavSim2, R2DIR, CSV_C2, CSV_GB  # noqa: E402

CSV_OX = f"{R2DIR}/r2b_oxtrail04_s42_trades.csv"


def run_series(sim, K=25, roundtrip=0.006, settle_lag=2, advance_fee=None, order_seed=None):
    """Ban sao NavSim2.run nhung tra ve nav series (de tinh yearly)."""
    import random
    FEE = 0.004
    S0 = 0.001  # noqa
    s_new = (roundtrip - FEE) / 2.0
    for t in sim.trades:
        t["net"] = (t["x_raw"] * (1.0 - s_new)) / (t["e_raw"] * (1.0 + s_new)) - 1.0 - FEE
    entries = defaultdict(list)
    for t in sim.trades:
        entries[t["entry_date"]].append(t)
    rng = random.Random(order_seed) if order_seed is not None else None
    for d in entries:
        entries[d].sort(key=lambda t: t["symbol"])
        if rng is not None:
            rng.shuffle(entries[d])
    sym_close, sym_idx, calendar = sim.sym_close, sim.sym_idx, sim.calendar

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

    cash, pend_total = 1.0, 0.0
    pending = defaultdict(float)
    legs, nav_series = [], []
    exits = defaultdict(list)
    for di, dt in enumerate(calendar):
        cash += pending.pop(dt, 0.0)
        pend_total = sum(v for k, v in pending.items() if k != "NEVER") + pending.get("NEVER", 0.0)
        for leg in exits.get(dt, ()):
            proceeds = leg["invested"] * (1.0 + leg["net"])
            if advance_fee is not None:
                cash += proceeds * (1.0 - advance_fee)
            elif settle_lag <= 0:
                cash += proceeds
            else:
                ri = di + settle_lag
                if ri < len(calendar):
                    pending[calendar[ri]] += proceeds
                    pend_total += proceeds
                else:
                    pend_total += proceeds
                    pending["NEVER"] += proceeds
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
        pos = sum(leg_value(l, dt) for l in legs)
        nav_series.append((dt, cash + pend_total + pos))
    ns = pd.DataFrame(nav_series, columns=["date", "nav"])
    ns["date"] = pd.to_datetime(ns["date"])
    return ns


def yearly_mean(csv, date_lo="2020-01-01", n=20, **kw):
    sim = NavSim2(csv, date_lo=date_lo)
    ys = defaultdict(list)
    for seed in range(n):
        ns = run_series(sim, order_seed=seed, **kw)
        ns["y"] = ns["date"].dt.year
        eoy = ns.groupby("y")["nav"].last()
        prev = 1.0
        for y, v in eoy.items():
            ys[y].append(v / prev - 1.0)
            prev = v
    return {y: (sum(v) / len(v)) for y, v in ys.items()}


def main():
    ox = pd.read_csv(CSV_OX)
    c2 = pd.read_csv(CSV_C2)
    for df in (ox, c2):
        df["entry_date"] = df["entry_date"].astype(str).str[:10]
        df["exit_date"] = df["exit_date"].astype(str).str[:10]
        df["ey"] = df["entry_date"].str[:4].astype(int)

    print("=== 1. EXIT MIX ===")
    for name, df in (("c2", c2), ("oxtrail04", ox)):
        g = df.groupby("exit_reason").agg(n=("pnl_pct", "size"), sum_pnl=("pnl_pct", "sum"),
                                          avg=("pnl_pct", "mean"), hold=("holding_days", "mean"))
        print(f"-- {name}: {len(df)} trades, sum_pnl={df.pnl_pct.sum():.1f}, "
              f"wr={(df.pnl_pct > 0).mean():.3f}, hold_med={df.holding_days.median():.0f}")
        print(g.round(2).to_string())

    print("\n=== 2. PAIRED DIFF (join symbol+entry_date) ===")
    m = c2.merge(ox, on=["symbol", "entry_date"], suffixes=("_c2", "_ox"), how="outer",
                 indicator=True)
    print(m["_merge"].value_counts().to_string())
    b = m[m._merge == "both"].copy()
    b["same"] = (b.exit_date_c2 == b.exit_date_ox) & \
        (abs(b.pnl_pct_c2 - b.pnl_pct_ox) < 1e-9)
    aff = b[~b.same].copy()
    print(f"\nidentical: {b.same.sum()}, affected: {len(aff)} "
          f"({len(aff) / len(b) * 100:.1f}%)")
    aff["d_pnl"] = aff.pnl_pct_ox - aff.pnl_pct_c2
    aff["d_hold"] = aff.holding_days_ox - aff.holding_days_c2
    print(f"affected: d_pnl sum={aff.d_pnl.sum():+.1f} mean={aff.d_pnl.mean():+.2f} "
          f"med={aff.d_pnl.median():+.2f}; d_hold mean={aff.d_hold.mean():+.1f}")
    print(f"win-share: d_pnl>0 {(aff.d_pnl > 0).mean():.2f}, "
          f"p10={aff.d_pnl.quantile(.1):+.1f} p90={aff.d_pnl.quantile(.9):+.1f}")
    print("\n-- affected theo exit_reason c2 -> ox:")
    g = aff.groupby(["exit_reason_c2", "exit_reason_ox"]).agg(
        n=("d_pnl", "size"), d_pnl_sum=("d_pnl", "sum"), d_pnl_avg=("d_pnl", "mean"),
        pnl_c2=("pnl_pct_c2", "mean"), pnl_ox=("pnl_pct_ox", "mean"),
        d_hold=("d_hold", "mean"))
    print(g.round(2).to_string())
    print("\n-- affected theo entry year:")
    gy = aff.groupby(aff.ey_c2).agg(n=("d_pnl", "size"), d_pnl_sum=("d_pnl", "sum"),
                                    d_pnl_avg=("d_pnl", "mean"))
    print(gy.round(2).to_string())
    # winner-vs-loser cohort trong affected
    aff["cohort"] = pd.cut(aff.pnl_pct_c2, [-100, 0, 10, 100],
                           labels=["c2_loser", "c2_win<10", "c2_win>10"])
    print("\n-- affected theo cohort pnl c2:")
    gc = aff.groupby("cohort", observed=True).agg(
        n=("d_pnl", "size"), d_pnl_sum=("d_pnl", "sum"), d_pnl_avg=("d_pnl", "mean"),
        d_hold=("d_hold", "mean"))
    print(gc.round(2).to_string())

    print("\n=== 3. YEARLY NAV (K25, R0.6, lag2+adv, mean 20 perm) ===")
    kw = dict(K=25, roundtrip=0.006, settle_lag=2, advance_fee=0.0008)
    rows = {}
    for name, csv in (("c2", CSV_C2), ("ox", CSV_OX), ("gb", CSV_GB)):
        rows[name] = yearly_mean(csv, "2020-01-01", 20, **kw)
    years = sorted(rows["c2"])
    print("year | c2 | ox | gb | ox-c2 | ox-gb")
    for y in years:
        c, o, g = rows["c2"].get(y), rows["ox"].get(y), rows["gb"].get(y)
        print(f"{y} | {c*100:+.1f} | {o*100:+.1f} | " +
              (f"{g*100:+.1f}" if g is not None else " n/a ") +
              f" | {(o-c)*100:+.1f} | " +
              (f"{(o-g)*100:+.1f}" if g is not None else " n/a "))
    print("\nDONE")


if __name__ == "__main__":
    main()

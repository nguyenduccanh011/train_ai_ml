# -*- coding: utf-8 -*-
"""nh_nav2: SIM NAV CHUAN v2 — thay khung do cu (r2_nav.py) cho moi so sanh R2-vs-gb.

Ba sua so voi khung cu (theo NAV_FRAMEWORK_AUDIT.md):
  (a) settle_lag: tien ban ve sau N phien lam viec (VN: T+2, tien ve chieu T+2
      -> mac dinh settle_lag=2; cau hinh 0/1/2). Tien treo van tinh vao NAV
      (receivable), chi khong xai duoc de entry.
  (b) advance_fee: ung truoc tien ban — tien ban xai NGAY nhung tra phi
      advance_fee tren proceeds moi vong (mac dinh 0.08%/vong nhu audit,
      ~0.0375%/ngay x ~2 ngay). advance_fee=None = khong ung (dung settle_lag).
  (c) shuffle-mean: tie-break tin hieu cung ngay xao 20 permutation
      (shuffle_stats), bao mean±sd — KHONG dung alphabet don le.

Cost: roundtrip R tham so (mac dinh 0.006 = cost nhung thuc te; sweep 0.6/0.7/0.9%).
Tai tao net tu gia raw y het na_navlib: fee=0.004 giu, slippage s=(R-0.004)/2 moi chieu.

VERIFY (chay truc tiep file nay): settle_lag=0, advance=None, alphabet, R=0.006
phai tai lap DUNG 7 anchor cu (c2 K22 x18.03 ...).
"""
import random
import sqlite3
import statistics
from collections import defaultdict

import pandas as pd

DB_PATH = "C:/Users/DUC CANH PC/Desktop/stock-serving/data/ohlcv.db"
DATE_HI = "2026-07-08"
S0 = 0.001   # slippage nhung trong entry_price/exit_price cua CSV
FEE = 0.004  # 2*commission + tax


class NavSim2:
    """Load trades + gia MOT lan cho (csv, date_lo); run() nhieu cau hinh nhanh."""

    def __init__(self, csv, date_lo="2020-01-01"):
        self.date_lo = date_lo
        ta = pd.read_csv(csv)
        ta = ta[ta.entry_date.astype(str).str[:10] >= date_lo]
        symbols = sorted(set(ta.symbol))
        con = sqlite3.connect(DB_PATH)
        px = pd.read_sql_query(
            "SELECT symbol, date, close FROM ohlcv WHERE symbol IN (%s) AND date>=? AND date<=?"
            % ",".join("?" * len(symbols)), con,
            params=list(symbols) + [date_lo, DATE_HI])
        con.close()
        self.sym_close, self.sym_idx = {}, {}
        for s, g in px.groupby("symbol"):
            g = g.sort_values("date")
            dates = g["date"].tolist()
            self.sym_close[s] = g["close"].tolist()
            self.sym_idx[s] = {d: i for i, d in enumerate(dates)}
        self.calendar = sorted(set(px["date"]))

        self.trades, self.skipped_db = [], 0
        for r in ta.itertuples():
            s = r.symbol
            ed, xd = str(r.entry_date)[:10], str(r.exit_date)[:10]
            if ed not in self.sym_idx.get(s, {}) or xd not in self.sym_idx.get(s, {}):
                self.skipped_db += 1
                continue
            self.trades.append(dict(
                symbol=s, entry_date=ed, exit_date=xd,
                i0=self.sym_idx[s][ed], i1=self.sym_idx[s][xd],
                p0=float(r.entry_price),
                e_raw=float(r.entry_price) / (1.0 + S0),
                x_raw=float(r.exit_price) / (1.0 - S0)))

    def run(self, K=25, roundtrip=0.006, settle_lag=2, advance_fee=None,
            order_seed=None):
        s_new = (roundtrip - FEE) / 2.0
        for t in self.trades:
            t["net"] = (t["x_raw"] * (1.0 - s_new)) / (t["e_raw"] * (1.0 + s_new)) \
                - 1.0 - FEE

        entries = defaultdict(list)
        for t in self.trades:
            entries[t["entry_date"]].append(t)
        rng = random.Random(order_seed) if order_seed is not None else None
        for d in entries:
            entries[d].sort(key=lambda t: t["symbol"])
            if rng is not None:
                rng.shuffle(entries[d])

        sym_close, sym_idx, calendar = self.sym_close, self.sym_idx, self.calendar

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

        cash = 1.0
        pending = defaultdict(float)   # release_date -> tien ban dang treo
        pend_total = 0.0
        legs = []
        exits = defaultdict(list)
        skip_cash = fills = 0
        nav_series = []
        occ_days = idle_sum = 0
        total_buys = 0.0

        for di, dt in enumerate(calendar):
            cash += pending.pop(dt, 0.0)
            pend_total = sum(pending.values())
            for leg in exits.get(dt, ()):
                proceeds = leg["invested"] * (1.0 + leg["net"])
                if advance_fee is not None:
                    # ung truoc tien ban: xai ngay, tra phi tren proceeds
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
            nav_series.append((dt, nav))
            occ_days += len(legs)
            idle_sum += (cash + pend_total) / nav

        ns = pd.DataFrame(nav_series, columns=["date", "nav"])
        ns["date"] = pd.to_datetime(ns["date"])
        nav = ns["nav"]
        final = float(nav.iloc[-1])
        years = (ns["date"].iloc[-1] - ns["date"].iloc[0]).days / 365.25
        dd = nav / nav.cummax() - 1
        return dict(final=final, cagr=final ** (1 / years) - 1,
                    maxdd=float(dd.min()),
                    maxdd_date=str(ns["date"].iloc[dd.idxmin()].date()),
                    fills=fills, skip_cash=skip_cash,
                    avg_open=occ_days / len(calendar),
                    idle=idle_sum / len(calendar),
                    turnover=total_buys / years / nav.mean())


def shuffle_stats(sim, K=25, roundtrip=0.006, settle_lag=2, advance_fee=None,
                  n=20):
    """CHUAN v2: 20 permutation tie-break -> mean±sd (pstdev)."""
    navs, dds = [], []
    for seed in range(n):
        m = sim.run(K=K, roundtrip=roundtrip, settle_lag=settle_lag,
                    advance_fee=advance_fee, order_seed=seed)
        navs.append(m["final"])
        dds.append(m["maxdd"])
    return dict(mean=statistics.mean(navs), sd=statistics.pstdev(navs),
                min=min(navs), max=max(navs),
                dd_mean=statistics.mean(dds), dd_worst=min(dds))


R2DIR = "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/r2line"
CSV_C2 = f"{R2DIR}/r2_c2_pb40snr_s42_trades.csv"
CSV_BASE = f"{R2DIR}/r2_base_s42_trades.csv"
CSV_GB = "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/exitmap/gbx08_enriched2.csv"


if __name__ == "__main__":
    # VERIFY: che do legacy (lag0, khong ung, alphabet, R0.6) phai ra dung anchor cu
    anchors = [  # (csv, date_lo, K, final_ky_vong, maxdd_ky_vong hoac None)
        ("c2  K22 full", CSV_C2, "2020-01-01", 22, 18.03, -15.41),
        ("c2  K23 full", CSV_C2, "2020-01-01", 23, 17.14, -14.77),
        ("c2  K25 full", CSV_C2, "2020-01-01", 25, 16.34, -13.63),
        ("base K20 full", CSV_BASE, "2020-01-01", 20, 17.70, None),
        ("base K25 full", CSV_BASE, "2020-01-01", 25, 14.78, None),
        ("gb  K25 full", CSV_GB, "2020-01-01", 25, 14.44, -15.31),
        ("c2  K25 f22 ", CSV_C2, "2022-01-01", 25, 3.80, None),
        ("c2  K22 f22 ", CSV_C2, "2022-01-01", 22, 3.92, None),
        ("gb  K25 f22 ", CSV_GB, "2022-01-01", 25, 3.50, None),
    ]
    sims = {}
    n_ok = 0
    for name, csv, lo, K, exp_f, exp_dd in anchors:
        key = (csv, lo)
        if key not in sims:
            sims[key] = NavSim2(csv, date_lo=lo)
        m = sims[key].run(K=K, roundtrip=0.006, settle_lag=0, advance_fee=None,
                          order_seed=None)
        ok = abs(m["final"] - exp_f) < 0.005 and \
            (exp_dd is None or abs(m["maxdd"] * 100 - exp_dd) < 0.01)
        n_ok += ok
        print(f"{name}: x{m['final']:.2f} (ky vong x{exp_f:.2f})"
              + (f" DD {m['maxdd']*100:.2f}% (ky vong {exp_dd:.2f}%)" if exp_dd else "")
              + ("  OK" if ok else "  ***FAIL***"))
    print(f"\n{n_ok}/{len(anchors)} anchor khop.")

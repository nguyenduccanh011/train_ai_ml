# -*- coding: utf-8 -*-
"""pr3_lib: NavSim2 mở rộng — trả về NAV series (để phân rã năm/tháng/episode)
và biến thể entry-lag. Copy body run() từ nh_nav2 (thước chuẩn) + trả series;
VERIFY: final phải khớp NavSim2.run gốc."""
import sys
from collections import defaultdict

import pandas as pd

sys.path.insert(0, "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/r2line/na_audit")
from nh_nav2 import NavSim2  # noqa: E402

import random  # noqa: E402


class NavSim2S(NavSim2):
    """run_series: y hệt run() nhưng trả thêm nav series; entry_lag: dời entry
    sang phiên kế tiếp (fill close ngày +1, giữ exit) — test độ nhạy thực thi trễ."""

    def apply_entry_lag(self, lag=1):
        """Dời entry +lag phiên (theo lịch của symbol). Trade có i0+lag >= i1 bị bỏ.
        Giá entry mới = close raw ngày mới; slippage giữ nguyên cấu trúc."""
        S0 = 0.001
        kept, dropped = [], 0
        for t in self.trades:
            s = t["symbol"]
            n_i0 = t["i0"] + lag
            if n_i0 >= t["i1"]:
                dropped += 1
                continue
            c_new = self.sym_close[s][n_i0]
            t2 = dict(t)
            t2["i0"] = n_i0
            t2["e_raw"] = c_new
            t2["p0"] = c_new * (1.0 + S0)
            # entry_date mới = date tại n_i0
            inv = {v: k for k, v in self.sym_idx[s].items()}
            t2["entry_date"] = inv[n_i0]
            kept.append(t2)
        self.trades = kept
        return dropped

    def run_series(self, K=25, roundtrip=0.006, settle_lag=2, advance_fee=None,
                   order_seed=None):
        s_new = (roundtrip - 4e-3) / 2.0
        FEE = 0.004
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
        pending = defaultdict(float)
        pend_total = 0.0
        legs = []
        exits = defaultdict(list)
        skip_cash = fills = 0
        nav_series = []
        for di, dt in enumerate(calendar):
            cash += pending.pop(dt, 0.0)
            pend_total = sum(pending.values())
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
                    fills += 1
                else:
                    skip_cash += 1
            pos = sum(leg_value(l, dt) for l in legs)
            nav = cash + pend_total + pos
            nav_series.append((dt, nav))
        ns = pd.DataFrame(nav_series, columns=["date", "nav"])
        ns["date"] = pd.to_datetime(ns["date"])
        return ns


def yearly_returns(ns):
    """Yearly return từ NAV series (calendar year, chain từ nav cuối năm trước)."""
    ns = ns.copy()
    ns["y"] = ns["date"].dt.year
    out = {}
    prev = 1.0
    for y, g in ns.groupby("y"):
        end = float(g["nav"].iloc[-1])
        out[y] = end / prev - 1.0
        prev = end
    return out


if __name__ == "__main__":
    # VERIFY: run_series final khớp run() gốc trên p42, 3 seed
    R2 = "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/r2line"
    sim = NavSim2S(f"{R2}/r2c_oxt04_p42_s42_trades.csv")
    for seed in (0, 7, 19):
        a = sim.run(K=25, advance_fee=0.0008, order_seed=seed)["final"]
        b = float(NavSim2S.run_series(sim, K=25, advance_fee=0.0008,
                                      order_seed=seed)["nav"].iloc[-1])
        print(f"seed {seed}: run x{a:.4f} vs series x{b:.4f}  "
              + ("OK" if abs(a - b) < 1e-9 else "***FAIL***"))

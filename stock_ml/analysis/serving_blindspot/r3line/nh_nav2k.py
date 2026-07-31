# -*- coding: utf-8 -*-
"""nh_nav2k: NavSim2 + DYNAMIC regime-K. Subclass override run(): size = nav/K_of(dt) voi
K_of = k_bull neu VNINDEX>MA(ma) else k_bear. Test regime x concentration (tap trung bull,
gian bear -> giam DD). Copy y run() cua nh_nav2, chi doi 1 dong size."""
from __future__ import annotations
import os, sys, random
from collections import defaultdict
import pandas as pd
sys.path.insert(0, os.environ.get("NH_NAV2_DIR", "F:/PROJECTS/hb2943_work"))
from nh_nav2 import NavSim2, FEE, S0, DATE_HI  # noqa: E402

VNI_CSV = "F:/PROJECTS/train_ai_ml/portable_data/vn_stock_ai_dataset_cleaned/context_features/symbol=VNINDEX/timeframe=1D/data.csv"


def _load_vni_regime(ma=200):
    v = pd.read_csv(VNI_CSV)
    v["d"] = pd.to_datetime(v["timestamp"]).dt.tz_localize(None).dt.normalize()
    v = v.drop_duplicates("d", keep="last").set_index("d")["close"].astype(float).sort_index()
    bull = (v > v.rolling(ma, min_periods=ma // 2).mean())
    return {d.strftime("%Y-%m-%d"): bool(b) for d, b in bull.items()}


class NavSim2K(NavSim2):
    def set_k(self, k_bull, k_bear, ma=200):
        self.k_bull, self.k_bear, self._reg = k_bull, k_bear, _load_vni_regime(ma)

    def _k_of(self, dt):
        return self.k_bull if self._reg.get(dt, True) else self.k_bear

    def run(self, K=25, roundtrip=0.006, settle_lag=2, advance_fee=None, order_seed=None):
        s_new = (roundtrip - FEE) / 2.0
        for t in self.trades:
            t["net"] = (t["x_raw"] * (1.0 - s_new)) / (t["e_raw"] * (1.0 + s_new)) - 1.0 - FEE
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
            s = leg["symbol"]; j = sym_idx[s].get(dt)
            if j is None:
                return leg["last_val"]
            i0, i1 = leg["i0"], leg["i1"]; j = min(max(j, i0), i1)
            ratio = leg["ratio1"] if i1 == i0 else leg["ratio0"] + (leg["ratio1"] - leg["ratio0"]) * (j - i0) / (i1 - i0)
            v = leg["invested"] * (sym_close[s][j] * ratio) / leg["p0"]; leg["last_val"] = v
            return v

        cash = 1.0; pending = defaultdict(float); pend_total = 0.0; legs = []; exits = defaultdict(list)
        nav_series = []
        for di, dt in enumerate(calendar):
            cash += pending.pop(dt, 0.0); pend_total = sum(pending.values())
            for leg in exits.get(dt, ()):
                proceeds = leg["invested"] * (1.0 + leg["net"])
                if advance_fee is not None:
                    cash += proceeds * (1.0 - advance_fee)
                elif settle_lag <= 0:
                    cash += proceeds
                else:
                    ri = di + settle_lag
                    if ri < len(calendar):
                        pending[calendar[ri]] += proceeds; pend_total += proceeds
                    else:
                        pend_total += proceeds; pending["NEVER"] += proceeds
                legs.remove(leg)
            pos = sum(leg_value(l, dt) for l in legs); nav_now = cash + pend_total + pos
            k_dt = self._k_of(dt)
            for t in entries.get(dt, ()):
                size = nav_now / k_dt
                if cash + 1e-12 >= size:
                    s = t["symbol"]; c0, c1 = sym_close[s][t["i0"]], sym_close[s][t["i1"]]
                    exit_eff = t["p0"] * (1.0 + t["net"])
                    leg = dict(symbol=s, i0=t["i0"], i1=t["i1"], invested=size, net=t["net"], p0=t["p0"],
                               ratio0=t["p0"] / c0, ratio1=exit_eff / c1, last_val=size, entry_date=dt, exit_date=t["exit_date"])
                    cash -= size; legs.append(leg); exits[t["exit_date"]].append(leg)
            pos = sum(leg_value(l, dt) for l in legs); nav = cash + pend_total + pos
            nav_series.append((dt, nav))
        ns = pd.DataFrame(nav_series, columns=["date", "nav"]); ns["date"] = pd.to_datetime(ns["date"])
        nav = ns["nav"]; final = float(nav.iloc[-1]); years = (ns["date"].iloc[-1] - ns["date"].iloc[0]).days / 365.25
        dd = nav / nav.cummax() - 1
        return dict(final=final, cagr=final ** (1 / years) - 1, maxdd=float(dd.min()))


def kstats(sim, n=20, **kw):
    import statistics as st
    finals = [sim.run(order_seed=sd, **kw)["final"] for sd in range(n)]
    dds = [sim.run(order_seed=sd, **kw)["maxdd"] for sd in range(n)]
    return st.mean(finals), min(dds)

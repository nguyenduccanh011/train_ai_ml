"""K-slot daily portfolio sim: conviction sizing + preemption + T+2 + advance-fee.

Verbatim port of serving/portfolio/core.py::_run_sim (fullest variant: preempt +
trade emit + holdings). NAV math is golden-guarded byte-exact — do not "clean up"
arithmetic or iteration order here.
"""

from __future__ import annotations

from collections import defaultdict

import pandas as pd

from stock_ml.portfolio.constants import ADVANCE_FEE, PortfolioConstants


def run_sim(
    legs_src, sym_close, sym_idx, calendar, C: PortfolioConstants, paused=None, riskoff=None
):
    """legs: dicts with symbol/entry_date/exit_date/i0/i1/p0/net/prio/conv/w/reason (+hv).
    paused: optional set of dates where NEW fills are suspended (crash-pause valve).
    riskoff: optional set of dates where HELD legs are force-exited (sell-down valve);
             new fills are also skipped on those days, but resume the very next day.
    Returns equity df, holdings rows, trades df, held_by_date."""
    tplus, K, MARGIN = C.tplus, C.k, C.margin

    def lv(leg, dt):
        s = leg["symbol"]
        j = sym_idx[s].get(dt)
        if j is None:
            return leg["last_val"]
        i0, i1 = leg["i0"], leg["i1"]
        j = min(max(j, i0), i1)
        r = (
            leg["ratio1"]
            if i1 == i0
            else leg["ratio0"] + (leg["ratio1"] - leg["ratio0"]) * (j - i0) / (i1 - i0)
        )
        v = leg["invested"] * (sym_close[s][j] * r) / leg["p0"]
        leg["last_val"] = v
        return v

    def mk(t, size, dt, di, nav_at):
        s = t["symbol"]
        c0, c1 = sym_close[s][t["i0"]], sym_close[s][t["i1"]]
        xe = t["p0"] * (1.0 + t["net"])
        return dict(
            symbol=s,
            i0=t["i0"],
            i1=t["i1"],
            invested=size,
            net=t["net"],
            p0=t["p0"],
            ratio0=t["p0"] / c0,
            ratio1=xe / c1,
            last_val=size,
            exit_date=t["exit_date"],
            prio=t["prio"],
            entry_dt=dt,
            be_di=di,
            conv=t["conv"],
            nav_at=nav_at,
            reason=t["reason"],
            hv=t.get("hv", False),
        )

    entries = defaultdict(list)
    for t in legs_src:
        entries[t["entry_date"]].append(t)
    for d in entries:
        entries[d].sort(key=lambda t: t["prio"], reverse=True)

    trades = []

    def emit(leg, exit_dt, exit_di, evicted, reason_override=None):
        if evicted:
            j = sym_idx[leg["symbol"]].get(exit_dt)
            xp = (
                float(sym_close[leg["symbol"]][j])
                if j is not None
                else leg["p0"] * (1.0 + leg["net"])
            )
            pnl = xp / leg["p0"] - 1.0 if leg["p0"] else 0.0
            reason = reason_override or "preempt"
        else:
            xp = leg["p0"] * (1.0 + leg["net"])
            pnl = float(leg["net"])
            reason = leg["reason"]
        trades.append(
            dict(
                symbol=leg["symbol"],
                entry_date=leg["entry_dt"],
                entry_price=float(leg["p0"]),
                exit_date=exit_dt,
                exit_price=xp,
                holding_days=int(exit_di - leg["be_di"]),
                pnl_pct=float(pnl),
                exit_reason=reason,
                conv=float(leg["conv"]),
                prio=float(leg["prio"]),
            )
        )

    equity, holdings, held_by_date = [], [], {}
    cash = 1.0
    pend = defaultdict(float)
    legs = []
    exits = defaultdict(list)
    for di, dt in enumerate(calendar):
        cash += pend.pop(dt, 0.0)
        pt = sum(pend.values())
        # coarse display reason per symbol exiting today -> run_portfolio_daily is_exit rows
        # (the real per-trade reason lives in run_trades_overlay; this tab marks "sold today").
        exit_today = {}
        for leg in list(exits.get(dt, ())):
            if leg in legs:
                cash += leg["invested"] * (1.0 + leg["net"]) * (1.0 - ADVANCE_FEE)
                legs.remove(leg)
                exit_today[leg["symbol"]] = "signal"
                emit(leg, dt, di, False)
        # risk-off sell-down (default off): force-exit sellable held legs on market-stress
        # days; fills skip today but resume next day (V-bottom entries stay monetized).
        if riskoff is not None and dt in riskoff:
            for l in list(legs):
                if (di - l["be_di"]) < tplus:
                    continue
                if C.riskoff_scope == "losers" and lv(l, dt) >= l["invested"]:
                    continue
                vnow = lv(l, dt)
                cash += vnow * (1.0 - ADVANCE_FEE)
                legs.remove(l)
                if l in exits.get(l["exit_date"], ()):
                    exits[l["exit_date"]].remove(l)
                exit_today[l["symbol"]] = "risk_off"
                emit(l, dt, di, True, reason_override="risk_off")
        pos = sum(lv(l, dt) for l in legs)
        nav_now = cash + pt + pos
        new_today = set()
        for t in entries.get(dt, ()):
            # DD valves (default off): market crash-pause + high-vol book cap
            if paused is not None and dt in paused:
                continue
            if riskoff is not None and dt in riskoff:
                continue
            if (
                C.vol_cap_q is not None
                and t.get("hv")
                and sum(1 for l in legs if l.get("hv")) >= C.vol_cap_max
            ):
                continue
            if (
                C.max_expo is not None
                and nav_now > 0
                and (nav_now - cash - pt) >= C.max_expo * nav_now
            ):
                continue
            size = (nav_now / K) * t["w"]
            if cash + 1e-12 >= size:
                cash -= size
                leg = mk(t, size, dt, di, nav_now)
                legs.append(leg)
                exits[t["exit_date"]].append(leg)
                new_today.add(t["symbol"])
            elif legs:
                cand = [l for l in legs if (di - l["be_di"]) >= tplus]
                if not cand:
                    continue
                c = min(cand, key=lambda l: l["prio"])
                if t["prio"] - c["prio"] > MARGIN:
                    vnow = lv(c, dt)
                    cash += vnow * (1.0 - ADVANCE_FEE)
                    legs.remove(c)
                    if c in exits.get(c["exit_date"], ()):
                        exits[c["exit_date"]].remove(c)
                    exit_today[c["symbol"]] = "preempt"
                    emit(c, dt, di, True)
                    nav_mid = cash + pt + sum(lv(l, dt) for l in legs)
                    size = (nav_mid / K) * t["w"]
                    if cash + 1e-12 >= size:
                        cash -= size
                        leg = mk(t, size, dt, di, nav_mid)
                        legs.append(leg)
                        exits[t["exit_date"]].append(leg)
                        new_today.add(t["symbol"])
        pos = sum(lv(l, dt) for l in legs)
        nav = cash + pt + pos
        equity.append(
            (dt, float(nav), float(cash + pt), float(pos / nav if nav > 0 else 0.0), len(legs))
        )
        for leg in legs:
            val = lv(leg, dt)
            holdings.append(
                (
                    dt,
                    leg["symbol"],
                    float(val / nav if nav > 0 else 0.0),  # weight (drifts)
                    float(leg["invested"] / leg["nav_at"] if leg["nav_at"] else 0.0),  # entry_weight
                    float(val / leg["invested"] - 1.0 if leg["invested"] else 0.0),  # unreal_pnl
                    leg["entry_dt"],
                    int(di - leg["be_di"]),
                    leg["symbol"] in new_today,
                    False,  # is_exit
                    None,  # exit_reason
                    float(leg["conv"]),
                )
            )
        for sym, reason in exit_today.items():
            holdings.append((dt, sym, 0.0, 0.0, 0.0, dt, 0, False, True, reason, None))
        held_by_date[dt] = {leg["symbol"] for leg in legs}
    for leg in legs:
        emit(leg, calendar[-1], len(calendar) - 1, False)
    eq = pd.DataFrame(equity, columns=["date", "nav", "cash", "exposure", "n_positions"])
    eq["date"] = pd.to_datetime(eq["date"])
    return eq, holdings, pd.DataFrame(trades), held_by_date

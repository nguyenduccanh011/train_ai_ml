"""run_portfolio: the ONE Stage-2 overlay pipeline (base -> rewrite -> gates ->
meta-priority -> K-slot sim). Backtest and serving call THIS with different
PortfolioContext; logic is identical by construction.

Golden-guarded byte-exact vs the champion production replay
(stock_ml/tests/test_portfolio_golden.py) — any numeric drift here is a bug.
"""
from __future__ import annotations

import statistics

import numpy as np
import pandas as pd

from stock_ml.portfolio.constants import FEE, ROUNDTRIP, S0, PortfolioConstants
from stock_ml.portfolio.context import PortfolioContext
from stock_ml.portfolio.gates import os_threshold, overshoot_map, skip_by_year_map, skip_for
from stock_ml.portfolio.panel import build_market_panel, build_price_panel
from stock_ml.portfolio.priority import meta_priority
from stock_ml.portfolio.rewrite import rewrite
from stock_ml.portfolio.sim import run_sim

# exit_reason values only the OVERLAY produces. Their presence in the input means
# someone fed OUTPUT trades back in -> overlay would run twice (the 151%->73% trap).
_OVERLAY_EXIT_REASONS = {"preempt", "green_trail", "early_cut"}


def run_portfolio(base_trades: pd.DataFrame, signals: pd.DataFrame, *,
                  ctx: PortfolioContext, C: PortfolioConstants | None = None) -> dict:
    """Full Stage-2 on BASE (engine-level) trades. Returns metrics + equity/holdings/trades/skipped."""
    C = C or PortfolioConstants()
    bad = set(base_trades["exit_reason"].dropna().unique()) & _OVERLAY_EXIT_REASONS
    if bad:
        raise ValueError(
            f"base_trades contain overlay-level exit_reason {sorted(bad)} — this is OUTPUT "
            "of a previous overlay run, not engine BASE; running the overlay twice is invalid "
            "(see PORTFOLIO_LAYER_UNIFICATION §1). Feed run_template_experiment BASE trades."
        )
    base = base_trades.copy()
    base["entry_date"] = pd.to_datetime(base["entry_date"]); base["exit_date"] = pd.to_datetime(base["exit_date"])
    base["sigd"] = pd.to_datetime(base["entry_signal_date"]); base["ed"] = base["entry_date"].dt.strftime("%Y-%m-%d")
    closed = base[base.exit_date.notna()].copy()

    syms = sorted(closed.symbol.unique().tolist())
    pm = meta_priority(closed, signals, ctx.meta_frame(syms))
    CLO, LO, DIDX, INV, CSm, R5 = build_market_panel(ctx.market_frame(C.market_start), ret_win=C.ret_win)
    rw = rewrite(closed, CLO, DIDX, INV, C.gt, C.ec_check_bar) if C.rewrite_on else closed.copy()

    cm = {(r.symbol, r.ed): CSm.get((r.symbol, str(r.sigd.date())), 0.5) for r in rw.itertuples()}
    r5 = {(r.symbol, r.ed): R5.get((r.symbol, str(r.sigd.date())), np.nan) for r in rw.itertuples()}
    osm = overshoot_map(rw, DIDX, LO, CLO)
    osthr = os_threshold(osm, C.os_pct)
    cv = [cm.get((r.symbol, r.ed), 0.5) for r in rw.itertuples()]
    # stat_mode="full": KNOWN non-causal full-period stats (mu/sd + the `off` recenter below):
    # champion parity (design doc §6). stat_mode="causal": expanding per-year, mirrors SKIP.
    mu = statistics.mean(cv) if cv else 0.5; sd = statistics.pstdev(cv) or 1.0
    mu_by = sd_by = osthr_by = None
    if C.stat_mode == "causal":
        yr_conv = [(int(r.ed[:4]), cm.get((r.symbol, r.ed), 0.5)) for r in rw.itertuples()]
        mu_by, sd_by, osthr_by = {}, {}, {}
        yr_os = [(int(k[1][:4]), v) for k, v in osm.items()]
        for y in sorted({y for y, _ in yr_conv}):
            past = [c for yy, c in yr_conv if yy < y]
            if len(past) >= 30:
                mu_by[y] = statistics.mean(past); sd_by[y] = statistics.pstdev(past) or 1.0
            past_os = [v for yy, v in yr_os if yy < y]
            if len(past_os) >= 30:
                osthr_by[y] = float(np.nanpercentile(past_os, C.os_pct))

    skip_by_year = skip_by_year_map(rw, cm, C)

    # price panel for NAV + build sim legs (T+2 extend, net, conviction weight, gates)
    symbols = sorted(rw.symbol.unique().tolist())
    sym_close, sym_idx, calendar = build_price_panel(ctx.price_frame(symbols, C.date_lo))
    s_new = (ROUNDTRIP - FEE) / 2.0
    legs_src, skipped = [], []
    raw_w = []
    for r in rw.itertuples():
        s = r.symbol; ed = r.ed; xd = str(r.exit_date)[:10]
        if ed not in sym_idx.get(s, {}) or xd not in sym_idx.get(s, {}):
            continue
        i0, oi1 = sym_idx[s][ed], sym_idx[s][xd]
        e_raw = float(r.entry_price) / (1.0 + S0)
        # match NavSim2 + hb_deploy _prep exactly: extended (T+2 min-hold) legs mark at the raw
        # ohlcv close of the extended bar (NO slippage-unwind); un-extended legs use the REWRITTEN
        # exit_price (market panel-derived) / (1-S0). Mixing the two stores here would drift net.
        if C.tplus and (oi1 - i0) < C.tplus:
            i1 = min(i0 + C.tplus, len(sym_close[s]) - 1); x_raw = float(sym_close[s][i1])
        else:
            i1 = oi1; x_raw = float(r.exit_price) / (1.0 - S0)
        net = (x_raw * (1.0 - s_new)) / (e_raw * (1.0 + s_new)) - 1.0 - FEE
        conv = cm.get((s, ed), 0.5); prio = pm.get((s, ed), -9.9)
        if mu_by is None:
            _w = min(max(1.0 + C.kconv * ((conv - mu) / sd), 0.4), 1.8)
        else:
            _y = int(ed[:4])
            _w = (1.0 if _y not in mu_by
                  else min(max(1.0 + C.kconv * ((conv - mu_by[_y]) / sd_by[_y]), 0.4), 1.8))
        raw_w.append(_w)
        legs_src.append(dict(symbol=s, entry_date=ed, exit_date=xd, i0=i0, i1=i1, p0=float(r.entry_price),
                             net=net, prio=prio, conv=conv, _w=_w, reason=r.exit_reason))
    if mu_by is None:
        off = 1.0 - (statistics.mean(raw_w) if raw_w else 1.0)
        off_by = None
    else:
        yr_w = [(int(l["entry_date"][:4]), l["_w"]) for l in legs_src]
        off = 0.0
        off_by = {}
        for y in sorted({y for y, _ in yr_w}):
            past = [w for yy, w in yr_w if yy < y]
            off_by[y] = (1.0 - statistics.mean(past)) if len(past) >= 30 else 0.0
    gated = []
    for leg in legs_src:
        leg["w"] = max(0.3, leg["_w"] + (off if off_by is None else off_by.get(int(leg["entry_date"][:4]), 0.0)))
        conv = leg["conv"]; key = (leg["symbol"], leg["entry_date"])
        if conv < skip_for(skip_by_year, C, leg["entry_date"]):
            skipped.append((leg["symbol"], leg["entry_date"], "conv_skip")); continue
        v = r5.get(key, np.nan)
        if not np.isnan(v) and v < C.r5thr:
            skipped.append((leg["symbol"], leg["entry_date"], "ret7_gate")); continue
        ov = osm.get(key)
        thr = osthr if osthr_by is None else osthr_by.get(int(leg["entry_date"][:4]), np.inf)
        if ov is not None and ov > thr:
            skipped.append((leg["symbol"], leg["entry_date"], "overshoot_fallknife")); continue
        gated.append(leg)

    eq, holdings, trades, hbd = run_sim(gated, sym_close, sym_idx, calendar, C)
    nav = eq["nav"]; fin = float(nav.iloc[-1]); yrs = (eq["date"].iloc[-1] - eq["date"].iloc[0]).days / 365.25
    cagr = fin ** (1 / yrs) - 1 if yrs > 0 else 0.0; dd = float((nav / nav.cummax() - 1).min())
    return dict(nav=fin, cagr=cagr, maxdd=dd, years=yrs, osthr=osthr, conv_mu=mu, conv_sd=sd,
                equity=eq, holdings=holdings, trades=trades, skipped=skipped, rewritten=rw,
                n_base=len(closed), n_gated=len(gated), n_skipped=len(skipped))

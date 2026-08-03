"""run_portfolio: the ONE Stage-2 overlay pipeline (base -> rewrite -> gates ->
meta-priority -> K-slot sim). Backtest and serving call THIS with different
PortfolioContext; logic is identical by construction.

Golden-guarded byte-exact vs the champion production replay
(stock_ml/tests/test_portfolio_golden.py) — any numeric drift here is a bug.
"""

from __future__ import annotations

import bisect
import statistics
from collections import defaultdict

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


def _bundle_key(C: PortfolioConstants) -> tuple:
    """Identity of a panel bundle: the (ctx-implicit) market panel derives ONLY from these."""
    return (C.market_start, C.ret_win, C.liqcol_adv10_ty is not None or C.w_liq_full_ty is not None)


def build_panel_bundle(ctx: PortfolioContext, C: PortfolioConstants) -> dict:
    """Market-panel-derived structures (cross-sectional conviction rank + traded-value history)
    that depend ONLY on ctx + (market_start, ret_win, liq-flags) — NOT on any run's trades. A
    batch scoring many runs under ONE C builds this ONCE and passes it to
    run_portfolio(..., bundle=...), skipping the ~60s/run rank rebuild. Byte-identical to the
    inline path (run_portfolio with bundle=None reproduces exactly this)."""
    mf = ctx.market_frame(C.market_start)
    CLO, LO, DIDX, INV, CSm, R5 = build_market_panel(mf, ret_win=C.ret_win)
    # liq-collapse veto data (default off): per-symbol traded-value history
    _tv = None
    if C.liqcol_adv10_ty is not None or C.w_liq_full_ty is not None:
        if "volume" not in mf.columns:
            raise ValueError("liqcol filter needs ctx.market_frame to include a volume column")
        _m = mf[["symbol", "date", "close", "volume"]].copy()
        _m["date"] = pd.to_datetime(_m["date"])
        _m["tvv"] = _m["close"].astype(float) * _m["volume"].astype(float)
        _tv = {
            s: (g["date"].to_numpy(), g["tvv"].to_numpy())
            for s, g in _m.sort_values(["symbol", "date"]).groupby("symbol")
        }
    return {"key": _bundle_key(C), "CLO": CLO, "LO": LO, "DIDX": DIDX, "INV": INV,
            "CSm": CSm, "R5": R5, "_tv": _tv}  # fmt: skip


def _compute_pending(signals, CLO, LO, DIDX, caldates, held_by_date, pull_pct, pull_win):
    """CAUSAL resting-pullback order book (PANEL plane) — rows for run_pending.

    For each date, the BUY signals (signal==1) still WAITING to fill: a resting limit at
    close[signal]*(1-pull_pct), within pull_win trading bars of the signal, not yet touched, and
    not currently held. low/close come from the market panel (bundle CLO/LO/DIDX) — the SAME plane
    conviction is ranked on. A signal on a symbol ABSENT from the panel yields NO row (off-panel;
    declared-universe contract — see docs/refactor/PORTFOLIO_WRITE_UNIFICATION_IMPL.md §3).

    Verbatim port of hb_portfolio_fix.compute_pending; hb read the FULL market duckdb ad-hoc, this
    reuses the already-built panel arrays (byte-identical for panel symbols, drops off-panel).
    Returns list of 9-tuples matching run_pending columns (date, symbol, signal_date, days_waiting,
    limit_price, ref_price, pct_to_limit, outcome, result_date). NAV/holdings/trades are untouched."""
    nD = len(caldates)
    ds_list = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in caldates]
    bar_of = {ds: i for i, ds in enumerate(ds_list)}
    buy = signals[signals["signal"] == 1] if "signal" in getattr(signals, "columns", []) else signals
    sig_bars: dict = defaultdict(list)
    for r in buy.itertuples():
        b = bar_of.get(pd.Timestamp(r.date).strftime("%Y-%m-%d"))
        if b is not None:
            sig_bars[r.symbol].append(b)
    # per-signal-symbol low/close aligned to the equity calendar bars, sourced from the panel
    low_arr, close_arr = {}, {}
    for sym in sig_bars:
        sig_bars[sym].sort()
        didx = DIDX.get(sym)
        if didx is None:
            continue  # off-panel -> no pending row (contract)
        clo, lo = CLO[sym], LO[sym]
        la = np.full(nD, np.nan)
        ca = np.full(nD, np.nan)
        for b, ds in enumerate(ds_list):
            j = didx.get(ds)
            if j is not None:
                la[b] = lo[j]
                ca[b] = clo[j]
        low_arr[sym] = la
        close_arr[sym] = ca
    out = []
    for sym, sbars in sig_bars.items():
        la = low_arr.get(sym)
        ca = close_arr.get(sym)
        if la is None:
            continue
        for d in range(nD):
            li = bisect.bisect_right(sbars, d - pull_win)
            ri = bisect.bisect_right(sbars, d)
            if li >= ri:
                continue
            S = sbars[ri - 1]
            cS = ca[S]
            if np.isnan(cS):
                continue
            limit = cS * (1.0 - pull_pct)
            seg_sd = la[S : d + 1]
            if seg_sd.size and np.nanmin(seg_sd) <= limit:
                continue  # limit already touched between signal and d -> filled, not pending
            ds = ds_list[d]
            if sym in held_by_date.get(ds, ()):
                continue
            cD = ca[d]
            if np.isnan(cD):
                continue
            pctL = limit / cD - 1.0
            hi = min(S + pull_win, nD - 1)
            touch = np.where(la[S : hi + 1] <= limit)[0]
            if touch.size:
                outc, rdate = "fill", ds_list[S + int(touch[0])]
            else:
                outc, rdate = "expire", ds_list[hi]
            out.append(
                (ds, sym, ds_list[S], d - S, float(limit), float(cD), float(pctL), outc, rdate)
            )
    return out


def run_portfolio(
    base_trades: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    ctx: PortfolioContext,
    C: PortfolioConstants | None = None,
    bundle: dict | None = None,
    emit_pending: bool = False,
) -> dict:
    """Full Stage-2 on BASE (engine-level) trades. Returns metrics + equity/holdings/trades/skipped.

    ``bundle`` (from build_panel_bundle) reuses a prebuilt market panel across many runs sharing
    one C; None (default) builds it inline — byte-identical, golden-safe.
    ``emit_pending`` (default False): also compute the resting-pullback order book (run_pending).
    OFF by default so the golden + interactive sandbox path skip the O(symbols×days) pending scan;
    only the DB persist path turns it on."""
    C = C or PortfolioConstants()
    bad = set(base_trades["exit_reason"].dropna().unique()) & _OVERLAY_EXIT_REASONS
    if bad:
        raise ValueError(
            f"base_trades contain overlay-level exit_reason {sorted(bad)} — this is OUTPUT "
            "of a previous overlay run, not engine BASE; running the overlay twice is invalid "
            "(see PORTFOLIO_LAYER_UNIFICATION §1). Feed run_template_experiment BASE trades."
        )
    base = base_trades.copy()
    base["entry_date"] = pd.to_datetime(base["entry_date"])
    base["exit_date"] = pd.to_datetime(base["exit_date"])
    base["sigd"] = pd.to_datetime(base["entry_signal_date"])
    base["ed"] = base["entry_date"].dt.strftime("%Y-%m-%d")
    closed = base[base.exit_date.notna()].copy()

    syms = sorted(closed.symbol.unique().tolist())
    pm = meta_priority(closed, signals, ctx.meta_frame(syms))
    if bundle is None:
        bundle = build_panel_bundle(ctx, C)
    elif bundle["key"] != _bundle_key(C):
        raise ValueError(
            f"panel bundle key mismatch: bundle built for {bundle['key']} but this run's C needs "
            f"{_bundle_key(C)} (market_start/ret_win/liq-flags differ) — rebuild the bundle."
        )
    CLO, LO, DIDX, INV, CSm, R5, _tv = (
        bundle["CLO"], bundle["LO"], bundle["DIDX"], bundle["INV"],
        bundle["CSm"], bundle["R5"], bundle["_tv"],
    )  # fmt: skip
    rw = rewrite(closed, CLO, DIDX, INV, C.gt, C.ec_check_bar) if C.rewrite_on else closed.copy()

    cm = {(r.symbol, r.ed): CSm.get((r.symbol, str(r.sigd.date())), 0.5) for r in rw.itertuples()}
    r5 = {(r.symbol, r.ed): R5.get((r.symbol, str(r.sigd.date())), np.nan) for r in rw.itertuples()}
    osm = overshoot_map(rw, DIDX, LO, CLO)
    osthr = os_threshold(osm, C.os_pct)
    cv = [cm.get((r.symbol, r.ed), 0.5) for r in rw.itertuples()]
    # stat_mode="full": KNOWN non-causal full-period stats (mu/sd + the `off` recenter below):
    # champion parity (design doc §6). stat_mode="causal": expanding per-year, mirrors SKIP.
    mu = statistics.mean(cv) if cv else 0.5
    sd = statistics.pstdev(cv) or 1.0
    mu_by = sd_by = osthr_by = None
    if C.stat_mode == "causal":
        yr_conv = [(int(r.ed[:4]), cm.get((r.symbol, r.ed), 0.5)) for r in rw.itertuples()]
        mu_by, sd_by, osthr_by = {}, {}, {}
        yr_os = [(int(k[1][:4]), v) for k, v in osm.items()]
        for y in sorted({y for y, _ in yr_conv}):
            past = [c for yy, c in yr_conv if yy < y]
            if len(past) >= 30:
                mu_by[y] = statistics.mean(past)
                sd_by[y] = statistics.pstdev(past) or 1.0
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
    # DATA-COVERAGE QC (does NOT change NAV). Two DISTINCT kinds of conviction miss — keep apart
    # (CONVICTION_UNIVERSE_UNIFICATION §3B): (1) symbol ENTIRELY absent from the panel = a data gap
    # → fail-loud under C.strict_panel; (2) symbol present but its signal-date bar is missing
    # (halt/suspension) = intrinsic, keep the neutral default, NEVER raise.
    _conv_syms = {k[0] for k in CSm}
    n_offpanel = n_halt = n_prio_miss = 0
    _offpanel_syms: set = set()
    for r in rw.itertuples():
        s = r.symbol
        ed = r.ed
        xd = str(r.exit_date)[:10]
        if ed not in sym_idx.get(s, {}) or xd not in sym_idx.get(s, {}):
            continue
        if (s, str(r.sigd.date())) not in CSm:
            if s in _conv_syms:
                n_halt += 1
            else:
                n_offpanel += 1
                _offpanel_syms.add(s)
        if (s, ed) not in pm:
            n_prio_miss += 1
        i0, oi1 = sym_idx[s][ed], sym_idx[s][xd]
        e_raw = float(r.entry_price) / (1.0 + S0)
        # match NavSim2 + hb_deploy _prep exactly: extended (T+2 min-hold) legs mark at the raw
        # ohlcv close of the extended bar (NO slippage-unwind); un-extended legs use the REWRITTEN
        # exit_price (market panel-derived) / (1-S0). Mixing the two stores here would drift net.
        if C.tplus and (oi1 - i0) < C.tplus:
            i1 = min(i0 + C.tplus, len(sym_close[s]) - 1)
            x_raw = float(sym_close[s][i1])
        else:
            i1 = oi1
            x_raw = float(r.exit_price) / (1.0 - S0)
        net = (x_raw * (1.0 - s_new)) / (e_raw * (1.0 + s_new)) - 1.0 - FEE
        conv = cm.get((s, ed), 0.5)
        prio = pm.get((s, ed), -9.9)
        if mu_by is None:
            _w = min(max(1.0 + C.kconv * ((conv - mu) / sd), 0.4), 1.8)
        else:
            _y = int(ed[:4])
            _w = (
                1.0
                if _y not in mu_by
                else min(max(1.0 + C.kconv * ((conv - mu_by[_y]) / sd_by[_y]), 0.4), 1.8)
            )
        raw_w.append(_w)
        legs_src.append(
            dict(
                symbol=s,
                entry_date=ed,
                exit_date=xd,
                i0=i0,
                i1=i1,
                p0=float(r.entry_price),
                net=net,
                prio=prio,
                conv=conv,
                sigd=str(r.sigd.date()),
                _w=_w,
                reason=r.exit_reason,
            )
        )
    # fail-loud data-contract guard: refuse to fabricate conviction for symbols the panel does
    # not cover (only symbol-ABSENT; halt/date-absent of a present symbol is fine). Off by default.
    if C.strict_panel and n_offpanel:
        raise ValueError(
            f"strict_panel: {n_offpanel} trades on {len(_offpanel_syms)} symbol(s) ABSENT from the "
            f"conviction panel — refusing to size on a fake neutral 0.5 (fail-loud). Prepare panel "
            f"data for: {sorted(_offpanel_syms)[:20]}. See CONVICTION_UNIVERSE_UNIFICATION §5.2."
        )
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
    # slow regime valve (default off): shrink NEW fills' weight while market breadth is weak
    weak_breadth_dates = None
    if C.regime_w_scale is not None:
        # market breadth = fraction of panel symbols above their own 50-bar MA (causal)
        above, total = {}, {}
        for s, a in CLO.items():
            ma = pd.Series(a).rolling(50, min_periods=50).mean().to_numpy()
            inv = INV[s]
            for i in range(len(a)):
                if np.isfinite(ma[i]):
                    d = inv[i]
                    total[d] = total.get(d, 0) + 1
                    if a[i] > ma[i]:
                        above[d] = above.get(d, 0) + 1
        weak_breadth_dates = frozenset(
            d for d, n in total.items() if n >= 30 and above.get(d, 0) / n < C.regime_breadth_thr
        )

    # sizing-experiment data (default off)
    _vol20 = None
    if C.w_invvol is not None:
        _vol20 = {s: pd.Series(a).pct_change().rolling(20).std().to_numpy() for s, a in CLO.items()}

    def _adv10_of(leg):
        arr = _tv.get(leg["symbol"]) if _tv is not None else None
        if arr is None:
            return None
        dts, tvs = arr
        i = int(np.searchsorted(dts, np.datetime64(pd.Timestamp(leg["entry_date"]))))
        h = tvs[:i]
        return float(h[-10:].mean()) if len(h) >= 10 else None

    def _skip(leg, reason):
        # run_skipped row (opportunity-cost ledger): symbol, signal_date, entry_date, base pnl,
        # conviction, reason. pnl_pct = the base leg's net return the portfolio forwent.
        skipped.append(
            (leg["symbol"], leg["sigd"], leg["entry_date"], float(leg["net"]), float(leg["conv"]), reason)
        )

    gated = []
    for leg in legs_src:
        leg["w"] = max(
            0.3,
            leg["_w"] + (off if off_by is None else off_by.get(int(leg["entry_date"][:4]), 0.0)),
        )
        if weak_breadth_dates is not None and leg["entry_date"] in weak_breadth_dates:
            leg["w"] *= C.regime_w_scale
        if _vol20 is not None:
            v = _vol20.get(leg["symbol"])
            i = DIDX.get(leg["symbol"], {}).get(leg["entry_date"])
            if v is not None and i is not None and i < len(v) and np.isfinite(v[i]) and v[i] > 0:
                leg["w"] *= float(np.clip(0.025 / v[i], 1.0 / C.w_invvol, C.w_invvol))
        if C.w_liq_full_ty is not None:
            a10 = _adv10_of(leg)
            if a10 is not None:
                leg["w"] *= float(np.clip(a10 / (C.w_liq_full_ty * 1e6), 0.3, 1.0))
        conv = leg["conv"]
        key = (leg["symbol"], leg["entry_date"])
        if conv < skip_for(skip_by_year, C, leg["entry_date"]):
            _skip(leg, "conv_skip")
            continue
        v = r5.get(key, np.nan)
        if not np.isnan(v) and v < C.r5thr:
            _skip(leg, "ret7_gate")
            continue
        ov = osm.get(key)
        thr = osthr if osthr_by is None else osthr_by.get(int(leg["entry_date"][:4]), np.inf)
        if ov is not None and ov > thr:
            _skip(leg, "overshoot_fallknife")
            continue
        if _tv is not None and C.liqcol_adv10_ty is not None:
            arr = _tv.get(leg["symbol"])
            if arr is not None:
                _dts, _tvs = arr
                _i = int(np.searchsorted(_dts, np.datetime64(pd.Timestamp(leg["entry_date"]))))
                _h = _tvs[:_i]
                if (
                    len(_h) >= 20
                    and float(_h[-10:].mean()) < C.liqcol_adv10_ty * 1e6
                    and float(_h[-252:].mean()) >= C.liqcol_adv252_ty * 1e6
                ):
                    _skip(leg, "liq_collapse")
                    continue
        gated.append(leg)

    # DD-signature valves (loss forensic 2026-07-29). Both OFF by default — this whole
    # block is skipped and run_sim receives paused=None -> byte-identical to the golden.
    paused = riskoff = None
    if C.crash_pause_ret5 is not None or C.vol_cap_q is not None or C.riskoff_ret5 is not None:
        vol_by_sym = {
            s: pd.Series(a).pct_change().rolling(20).std().to_numpy() for s, a in CLO.items()
        }
        by_date: dict = {}
        for s, dmap in DIDX.items():
            for d, i in dmap.items():
                by_date.setdefault(d, []).append((s, i))
    if C.crash_pause_ret5 is not None or C.riskoff_ret5 is not None:
        # EW market daily return (full market panel, clipped like the engine proxy)
        mr = {}
        for d, lst in by_date.items():
            rs = [CLO[s][i] / CLO[s][i - 1] - 1.0 for s, i in lst if i > 0 and CLO[s][i - 1] > 0]
            if rs:
                mr[d] = float(np.mean(np.clip(rs, -0.5, 0.5)))
        mrs = pd.Series(mr).sort_index()
        r5s = (1.0 + mrs).rolling(5).apply(np.prod, raw=True) - 1.0
        if C.crash_pause_ret5 is not None:
            paused = frozenset(r5s.index[r5s <= C.crash_pause_ret5])
        if C.riskoff_ret5 is not None:
            riskoff = frozenset(r5s.index[r5s <= C.riskoff_ret5])
    if C.vol_cap_q is not None:
        # high-vol flag: leg's vol20 >= same-day cross-sectional quantile q (causal)
        thr_by_date: dict = {}
        for leg in gated:
            d = leg["entry_date"]
            if d not in thr_by_date:
                vals = [vol_by_sym[s][i] for s, i in by_date.get(d, ()) if i >= 20]
                vals = [v for v in vals if np.isfinite(v)]
                thr_by_date[d] = float(np.quantile(vals, C.vol_cap_q)) if vals else np.inf
            i = DIDX.get(leg["symbol"], {}).get(d)
            v = vol_by_sym.get(leg["symbol"])
            leg["hv"] = bool(
                i is not None
                and v is not None
                and i < len(v)
                and np.isfinite(v[i])
                and v[i] >= thr_by_date[d]
            )

    eq, holdings, trades, hbd = run_sim(
        gated, sym_close, sym_idx, calendar, C, paused=paused, riskoff=riskoff
    )
    nav = eq["nav"]
    fin = float(nav.iloc[-1])
    yrs = (eq["date"].iloc[-1] - eq["date"].iloc[0]).days / 365.25
    cagr = fin ** (1 / yrs) - 1 if yrs > 0 else 0.0
    dd = float((nav / nav.cummax() - 1).min())
    # resting-pullback order book (display-only; OFF by default -> golden byte-identical).
    # None = NOT computed (emit_pending False); [] = computed-but-empty. The writer relies on this
    # sentinel to refuse a detail persist that would DELETE run_pending without re-inserting.
    pending = (
        _compute_pending(signals, CLO, LO, DIDX, list(eq["date"]), hbd, C.pull_pct, C.pull_win)
        if emit_pending
        else None
    )
    return dict(
        nav=fin,
        cagr=cagr,
        maxdd=dd,
        years=yrs,
        osthr=osthr,
        conv_mu=mu,
        conv_sd=sd,
        equity=eq,
        holdings=holdings,
        trades=trades,
        skipped=skipped,
        held_by_date=hbd,
        pending=pending,
        rewritten=rw,
        n_base=len(closed),
        n_gated=len(gated),
        n_skipped=len(skipped),
        # data-coverage QC (see legs loop). Two kinds of conviction miss kept apart:
        #   offpanel = symbol ABSENT from panel (data gap; raises under strict_panel)
        #   halt     = symbol present but signal-date bar missing (halt/suspension; intrinsic)
        conv_miss_frac=((n_offpanel + n_halt) / len(legs_src)) if legs_src else 0.0,
        offpanel_frac=(n_offpanel / len(legs_src)) if legs_src else 0.0,
        halt_frac=(n_halt / len(legs_src)) if legs_src else 0.0,
        prio_miss_frac=(n_prio_miss / len(legs_src)) if legs_src else 0.0,
        n_offpanel=n_offpanel,
        n_halt=n_halt,
    )

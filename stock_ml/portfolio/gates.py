"""Gates: overshoot fall-knife map/threshold + causal per-year conviction SKIP.

Overshoot decides AT FILL-DATE using low[signal..fill] = closed bars only ->
CAUSAL (design doc §9.1; do NOT re-flag as leak). Threshold is in-sample p90 —
known non-causal label, parity-preserved (doc §6).

SKIP causal mode: each year's gate uses PAST convictions only (expanding), so
adding a future fold never changes an earlier year's gate.
"""
from __future__ import annotations

import statistics

import numpy as np

from stock_ml.portfolio.constants import PortfolioConstants


def overshoot_map(rw, DIDX, LO, CLO) -> dict:
    """{(symbol, entry_date): (entry_price - min(low[signal..fill])) / close[signal]}"""
    osm = {}
    for r in rw.itertuples():
        di = DIDX.get(r.symbol, {}); si_ = di.get(str(r.sigd.date())); fi = di.get(r.ed)
        if si_ is not None and fi is not None and fi >= si_:
            osm[(r.symbol, r.ed)] = (r.entry_price - LO[r.symbol][si_:fi + 1].min()) / CLO[r.symbol][si_]
    return osm


def os_threshold(osm: dict, os_pct: float) -> float:
    return float(np.nanpercentile([v for v in osm.values()], os_pct)) if osm else np.inf


def skip_by_year_map(rw, cm: dict, C: PortfolioConstants):
    """Panel-adaptive conviction GATE. cs5_ma50 is a CROSS-SECTIONAL rank, so a wider
    panel COMPRESSES the distribution and inflates every conv past a fixed skip.
    Raise the gate by the amount the panel inflated the mean:
    skip = base + max(0, conv_mu - mu_ref) * gain.  Returns None when gate is off."""
    if C.skip_gain <= 0 or C.skip_mode == "off":
        return None
    cv = [cm.get((r.symbol, r.ed), 0.5) for r in rw.itertuples()]
    mu = statistics.mean(cv) if cv else 0.5
    if C.skip_mode == "fixed":
        return {"_all": C.skip + max(0.0, mu - C.skip_mu_ref) * C.skip_gain}
    # causal: per-year gate from PAST convictions only (expanding)
    _ty = [(int(str(r.ed)[:4]), cm.get((r.symbol, r.ed), 0.5)) for r in rw.itertuples()]
    skip_by_year = {}
    for _y in sorted({y for y, _ in _ty}):
        _past = [c for (yy, c) in _ty if yy < _y]
        skip_by_year[_y] = (C.skip if len(_past) < 30
                            else C.skip + max(0.0, statistics.mean(_past) - C.skip_mu_ref) * C.skip_gain)
    return skip_by_year


def skip_for(skip_by_year, C: PortfolioConstants, entry_date: str) -> float:
    if skip_by_year is None:
        return C.skip
    if "_all" in skip_by_year:
        return skip_by_year["_all"]
    return skip_by_year.get(int(str(entry_date)[:4]), C.skip)

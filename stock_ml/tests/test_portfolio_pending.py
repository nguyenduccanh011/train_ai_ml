"""Unit test for the resting-pullback order book (_compute_pending).

Deterministic, no DB / golden store needed: hand-built panel arrays + signals with a
by-hand expected pending book. Guards the ported hb_portfolio_fix.compute_pending logic
(limit = close[signal]*(1-pull_pct), pull_win window, already-touched skip, held skip,
fill vs expire outcome) independent of the slow golden replay.
"""

from __future__ import annotations

import pandas as pd

from stock_ml.portfolio.api import _compute_pending


def _norm(rows):
    """pct_to_limit (col 7) carries float noise (90/100-1 != -0.1 exactly) — round for compare."""
    return [r[:6] + (round(r[6], 10),) + r[7:] for r in rows]

# 5-bar calendar; panel index aligns 1:1 with the calendar for simplicity.
CALDATES = [pd.Timestamp(f"2024-01-0{i + 1}") for i in range(5)]
DS = [d.strftime("%Y-%m-%d") for d in CALDATES]
DIDX = {"AAA": {ds: i for i, ds in enumerate(DS)}}
# close flat at 100 -> limit = 90 (pull_pct=0.1). low pierces 90 first at bar 2 (88).
CLO = {"AAA": [100.0, 100.0, 100.0, 100.0, 100.0]}
LO = {"AAA": [95.0, 92.0, 88.0, 85.0, 99.0]}
SIG = pd.DataFrame({"symbol": ["AAA"], "date": [pd.Timestamp("2024-01-01")], "signal": [1]})


def _rows(held):
    return _compute_pending(SIG, CLO, LO, DIDX, CALDATES, held, pull_pct=0.1, pull_win=3)


def test_pending_fill_and_touch_skip():
    # signal at bar 0. limit 90. bars 0,1 rest (low 95,92 > 90); bar 2 low 88 <= 90 -> filled,
    # not pending; bars 3,4 fall outside the pull_win=3 window -> no signal in range.
    rows = _rows({})
    assert _norm(rows) == [
        (DS[0], "AAA", DS[0], 0, 90.0, 100.0, -0.1, "fill", DS[2]),
        (DS[1], "AAA", DS[0], 1, 90.0, 100.0, -0.1, "fill", DS[2]),
    ]


def test_pending_held_skip():
    # holding AAA on bar 1 removes that day's pending row (only bar 0 remains).
    rows = _rows({DS[1]: {"AAA"}})
    assert _norm(rows) == [(DS[0], "AAA", DS[0], 0, 90.0, 100.0, -0.1, "fill", DS[2])]


def test_pending_expire_outcome():
    # low never reaches the limit -> outcome 'expire' at the window end (bar S+pull_win=3).
    lo = {"AAA": [99.0, 99.0, 99.0, 99.0, 99.0]}
    rows = _compute_pending(SIG, CLO, lo, DIDX, CALDATES, {}, pull_pct=0.1, pull_win=3)
    assert [r[7] for r in rows] == ["expire", "expire", "expire"]
    assert all(r[8] == DS[3] for r in rows)


def test_pending_offpanel_dropped():
    # a signal on a symbol absent from the panel (DIDX) yields no row (contract).
    sig = pd.DataFrame({"symbol": ["ZZZ"], "date": [pd.Timestamp("2024-01-01")], "signal": [1]})
    assert _compute_pending(sig, CLO, LO, DIDX, CALDATES, {}, pull_pct=0.1, pull_win=3) == []


def test_pending_only_buy_signals():
    # signal!=1 rows are ignored (resting orders are buy-side only).
    sig = pd.DataFrame(
        {"symbol": ["AAA", "AAA"], "date": [pd.Timestamp("2024-01-01")] * 2, "signal": [0, 0]}
    )
    assert _compute_pending(sig, CLO, LO, DIDX, CALDATES, {}, pull_pct=0.1, pull_win=3) == []

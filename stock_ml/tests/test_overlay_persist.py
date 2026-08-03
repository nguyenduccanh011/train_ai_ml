"""Shape guards for the unified overlay writer path (no DB needed).

- run_sim now emits run_portfolio_daily-shaped holdings (11 fields) incl. is_exit rows.
- overlay_persist._detail_rows maps the wheel result dict 1:1 onto each table's INSERT arity.
Golden covers NAV/trades byte-parity but NOT holdings/skipped shape — this pins that contract.
"""

from __future__ import annotations

import pandas as pd
import pytest

from stock_ml.db.overlay_persist import _INSERTS, _detail_rows, persist_overlay
from stock_ml.portfolio.constants import PortfolioConstants
from stock_ml.portfolio.sim import run_sim


def _insert_arity(sql: str) -> int:
    """Number of columns in the '(a,b,c) VALUES %s' INSERT column list."""
    cols = sql[sql.index("(") + 1 : sql.index(")")]
    return cols.count(",") + 1


def test_run_sim_holdings_11_fields_with_exit_row():
    cal = ["2024-01-01", "2024-01-02", "2024-01-03"]
    sym_close = {"AAA": [10.0, 11.0, 12.0]}
    sym_idx = {"AAA": {d: i for i, d in enumerate(cal)}}
    leg = dict(
        symbol="AAA", entry_date="2024-01-01", exit_date="2024-01-02", i0=0, i1=1, p0=10.0,
        net=0.1, prio=1.0, conv=0.6, w=1.0, reason="max_hold",
    )
    C = PortfolioConstants(tplus=0)
    eq, holdings, trades, hbd = run_sim([leg], sym_close, sym_idx, cal, C)
    # every holdings row carries the full run_portfolio_daily tuple (11 fields)
    assert holdings and all(len(h) == 11 for h in holdings)
    # the scheduled exit on day 2 produces an is_exit row (idx 8) with a coarse reason (idx 9)
    exit_rows = [h for h in holdings if h[8] is True]
    assert exit_rows and exit_rows[0][1] == "AAA" and exit_rows[0][9] == "signal"
    # a normal held row on day 1 is is_exit=False with entry_weight populated (idx 3)
    held = [h for h in holdings if h[8] is False]
    assert held and held[0][3] is not None
    assert list(eq.columns) == ["date", "nav", "cash", "exposure", "n_positions"]


def test_detail_rows_arity_matches_inserts():
    eq = pd.DataFrame(
        [("2024-01-01", 1.05, 0.2, 0.8, 3)],
        columns=["date", "nav", "cash", "exposure", "n_positions"],
    )
    eq["date"] = pd.to_datetime(eq["date"])
    trades = pd.DataFrame(
        [("AAA", "2024-01-01", 10.0, "2024-01-05", 11.0, 4, 0.1, "max_hold", 0.6, 1.2)],
        columns=["symbol", "entry_date", "entry_price", "exit_date", "exit_price",
                 "holding_days", "pnl_pct", "exit_reason", "conv", "prio"],
    )
    result = dict(
        cagr=0.5, maxdd=-0.1, conv_miss_frac=0.0, offpanel_frac=0.0,
        equity=eq,
        holdings=[("2024-01-01", "AAA", 0.3, 0.28, 0.05, "2024-01-01", 0, True, False, None, 0.6)],
        trades=trades,
        skipped=[("BBB", "2024-01-02", "2024-01-03", -0.02, 0.3, "conv_skip")],
        pending=[("2024-01-04", "CCC", "2024-01-02", 2, 9.0, 10.0, -0.1, "fill", "2024-01-06")],
    )
    rows = _detail_rows("run/x", result)
    for tbl, sql in _INSERTS.items():
        want = _insert_arity(sql)
        assert rows[tbl], f"{tbl} produced no rows"
        assert all(len(r) == want for r in rows[tbl]), f"{tbl}: row arity != {want} cols"
        assert rows[tbl][0][0] == "run/x", f"{tbl}: run_id not prefixed"


def test_persist_detail_refuses_uncomputed_pending():
    # detail=True with pending=None (emit_pending was False) must raise BEFORE any DELETE, so it
    # can't silently wipe run_pending. No DB touched: the guard fires before conn is used.
    result = dict(
        cagr=0.5, maxdd=-0.1, conv_miss_frac=0.0, offpanel_frac=0.0,
        equity=None, holdings=[], trades=None, skipped=[], pending=None,
    )
    with pytest.raises(ValueError, match="emit_pending=True"):
        persist_overlay(object(), "run/x", result, PortfolioConstants(), "h", "fp", "n", detail=True)

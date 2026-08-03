"""persist_overlay: the ONE writer that maps a stock_ml.portfolio.run_portfolio result dict
straight to the overlay DB tables — the unified replacement for the ad-hoc hb_portfolio_* scripts
(PORTFOLIO_WRITE_UNIFICATION_IMPL.md §4).

The wheel now emits every output already shaped to its target table (equity/holdings/trades/
skipped/pending), so this is a thin 1:1 mapper: no engine, no data reload. Lives in stock_ml.db
(outside the wheel) because it touches the research DB.

Transaction: this function only DELETE+INSERT+UPSERT on the given ``conn`` — it does NOT commit.
The CALLER owns the transaction (commit on success, rollback on error) so a batch loop keeps its
per-run rollback semantics. ``detail`` tiers the write:
  detail=False  -> only UPSERT leaderboard_nav metrics (board-wide scoring; keeps the DB lean).
  detail=True   -> also repopulate the 5 detail tables (pinned / deploy-tier runs with a full tab).
When detail=True the caller must have run run_portfolio(..., emit_pending=True) so result["pending"]
is populated (else run_pending lands empty).
"""

from __future__ import annotations

import dataclasses
import json

import pandas as pd
from psycopg2.extras import execute_values

from stock_ml.portfolio import overlay_key


def config_of(C) -> dict:
    """Every PortfolioConstants field, JSON-ready — the row must be re-runnable from itself.
    A hash proves two books came from the same strategy but cannot say WHICH strategy, and a
    number nobody can reproduce is a number nobody can check."""
    return {k: v for k, v in dataclasses.asdict(C).items()}


def reference_key() -> str:
    """Imported lazily-ish via a wrapper: stock_ml.db.overlay_scoring pulls in duckdb + the panel
    artifact check, which the persist path does not otherwise need."""
    from stock_ml.db.overlay_scoring import reference_key as _rk

    return _rk()

_UPSERT_NAV = """
INSERT INTO leaderboard_nav
    (run_id, cagr_overlay, maxdd_overlay, overlay_k, overlay_note, overlay_config_hash,
     overlay_panel_fp, conv_miss_frac, offpanel_frac, computed_at)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, now())
ON CONFLICT (run_id) DO UPDATE SET
    cagr_overlay = EXCLUDED.cagr_overlay, maxdd_overlay = EXCLUDED.maxdd_overlay,
    overlay_k = EXCLUDED.overlay_k, overlay_note = EXCLUDED.overlay_note,
    overlay_config_hash = EXCLUDED.overlay_config_hash, overlay_panel_fp = EXCLUDED.overlay_panel_fp,
    conv_miss_frac = EXCLUDED.conv_miss_frac, offpanel_frac = EXCLUDED.offpanel_frac,
    computed_at = now()
"""

# (table, column-list) per detail table. Column order MUST match the row builders below + the ORM
# (stock_ml/db/models/portfolio.py). run_trades stays BASE-only (0033) — overlay trades go to
# run_trades_overlay.
_DETAIL_TABLES = ("run_equity", "run_portfolio_daily", "run_trades_overlay", "run_skipped", "run_pending")
_INSERTS = {
    "run_equity": (
        "INSERT INTO run_equity (run_id,overlay_key,date,nav,cash,exposure,n_positions) VALUES %s"
    ),
    "run_portfolio_daily": (
        "INSERT INTO run_portfolio_daily (run_id,overlay_key,date,symbol,weight,entry_weight,"
        "unreal_pnl,entry_date,days_held,is_new,is_exit,exit_reason,conv) VALUES %s"
    ),
    "run_trades_overlay": (
        "INSERT INTO run_trades_overlay (run_id,overlay_key,symbol,entry_date,entry_price,exit_date,"
        "exit_price,holding_days,pnl_pct,exit_reason,conv,prio) VALUES %s"
    ),
    "run_skipped": (
        "INSERT INTO run_skipped (run_id,overlay_key,symbol,signal_date,entry_date,pnl_pct,conv,"
        "skip_reason) VALUES %s"
    ),
    "run_pending": (
        "INSERT INTO run_pending (run_id,overlay_key,date,symbol,signal_date,days_waiting,"
        "limit_price,ref_price,pct_to_limit,outcome,result_date) VALUES %s"
    ),
}

_UPSERT_OVERLAY = """
INSERT INTO run_overlay
    (run_id, overlay_key, label, config, base_run, source_run, cagr, maxdd, nav, years,
     n_trades, k, scoring_hash, panel_fp, conv_miss_frac, offpanel_frac, has_detail, computed_at)
VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, now())
ON CONFLICT (run_id, overlay_key) DO UPDATE SET
    label = COALESCE(EXCLUDED.label, run_overlay.label),
    config = EXCLUDED.config, base_run = EXCLUDED.base_run,
    source_run = COALESCE(EXCLUDED.source_run, run_overlay.source_run),
    cagr = EXCLUDED.cagr, maxdd = EXCLUDED.maxdd, nav = EXCLUDED.nav, years = EXCLUDED.years,
    n_trades = EXCLUDED.n_trades, k = EXCLUDED.k,
    scoring_hash = EXCLUDED.scoring_hash, panel_fp = EXCLUDED.panel_fp,
    conv_miss_frac = EXCLUDED.conv_miss_frac, offpanel_frac = EXCLUDED.offpanel_frac,
    -- a metrics-only re-score must NOT claim the (now stale) book is fresh, but it must not
    -- forget one that IS there either: only a detail write may flip this true.
    has_detail = run_overlay.has_detail OR EXCLUDED.has_detail,
    computed_at = now()
"""


def _detail_rows(run_id: str, overlay_key: str, result: dict) -> dict:
    """Build the ((run_id, overlay_key)-prefixed) INSERT rows for each detail table."""
    pre = (run_id, overlay_key)
    eq = result["equity"]
    equity = [
        (*pre, str(pd.Timestamp(r.date).date()), float(r.nav), float(r.cash),
         float(r.exposure), int(r.n_positions))
        for r in eq.itertuples()
    ]  # fmt: skip
    holdings = [(*pre, *h) for h in result["holdings"]]
    tr = result["trades"]
    trades = (
        [
            (*pre, t.symbol, t.entry_date, float(t.entry_price), t.exit_date, float(t.exit_price),
             int(t.holding_days), float(t.pnl_pct), t.exit_reason, float(t.conv), float(t.prio))
            for t in tr.itertuples()
        ]
        if len(tr)
        else []
    )  # fmt: skip
    skipped = [(*pre, *s) for s in result["skipped"]]
    pending = [(*pre, *p) for p in result["pending"]]
    return {
        "run_equity": equity,
        "run_portfolio_daily": holdings,
        "run_trades_overlay": trades,
        "run_skipped": skipped,
        "run_pending": pending,
    }


def persist_overlay(
    conn,
    run_id: str,
    result: dict,
    C,
    cfg_hash: str,
    panel_fp: str,
    note: str,
    *,
    detail: bool,
    label: str | None = None,
    base_run: str | None = None,
    source_run: str | None = None,
) -> str:
    """Write one run's overlay result under its STRATEGY key. Returns the overlay_key.

    Identity (OVERLAY_IDENTITY_RESTRUCTURE §3.1): the book is keyed by ``(run_id, overlay_key)``
    where ``overlay_key = overlay_key(C)`` covers the config ALONE. So scoring a second strategy
    on the same run adds a second book instead of deleting the first — the defect this replaces.
    ``cfg_hash`` is the scoring identity (config + panel) and is recorded, not keyed on.

    ``leaderboard_nav`` is written ONLY for the board reference strategy. The leaderboard ranks
    SIGNAL quality under one deploy-style policy (register_overlay docstring); letting a K=6 tier
    score overwrite that row is how the board silently became a mix of policies.

    Does NOT commit — the caller owns the transaction.
    """
    if detail and result.get("pending") is None:
        raise ValueError(
            f"persist_overlay(detail=True) for {run_id} but result['pending'] is None — the run "
            "was scored with emit_pending=False, so writing detail would DELETE run_pending "
            "without re-inserting. Call run_portfolio/score_overlay with emit_pending=True."
        )
    key = overlay_key(C)
    cur = conn.cursor()
    if detail:
        rows = _detail_rows(run_id, key, result)
        for tbl in _DETAIL_TABLES:
            cur.execute(f"DELETE FROM {tbl} WHERE run_id=%s AND overlay_key=%s", (run_id, key))
            if rows[tbl]:
                execute_values(cur, _INSERTS[tbl], rows[tbl])
    cur.execute(
        _UPSERT_OVERLAY,
        (run_id, key, label, json.dumps(config_of(C)), base_run, source_run,
         result["cagr"], result["maxdd"], result["nav"], result["years"],
         len(result.get("trades", ())), C.k, cfg_hash, panel_fp,
         result["conv_miss_frac"], result["offpanel_frac"], detail),
    )  # fmt: skip
    if key == reference_key():
        cur.execute(
            _UPSERT_NAV,
            (run_id, result["cagr"], result["maxdd"], C.k, note, cfg_hash,
             panel_fp, result["conv_miss_frac"], result["offpanel_frac"]),
        )  # fmt: skip
    return key

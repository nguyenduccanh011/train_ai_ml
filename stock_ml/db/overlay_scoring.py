"""Overlay scoring service — load a run's BASE trades + signals from Postgres, run the
Stage-2 overlay (stock_ml.portfolio.run_portfolio), return metrics.

Shared by the batch registrar (scripts/ops/register_overlay.py) and the live sandbox API
endpoint, so both surfaces score through the ONE unified engine. Lives in stock_ml.db
(outside the stock_ml_core wheel) because it touches the research DB; run_portfolio itself
stays wheel-pure (all data injected via PortfolioContext).
"""

from __future__ import annotations

import hashlib
import json
import os

import pandas as pd

from stock_ml.portfolio import DuckDBContext, PortfolioConstants, run_portfolio
from stock_ml.portfolio.context import PortfolioContext

# Data context defaults: the pinned serving snapshot (same store the champion golden uses).
# Override via env so batch/API/tests can point elsewhere without code changes.
_MARKET_DB = os.environ.get(
    "OVERLAY_MARKET_DB",
    "C:/Users/DUC CANH PC/Desktop/stock-serving/market_data/market_golden_pin_20260729.duckdb",
)
_OHLCV_DB = os.environ.get(
    "OVERLAY_OHLCV_DB",
    "C:/Users/DUC CANH PC/Desktop/stock-serving/data/ohlcv_golden_pin_20260729.db",
)
_DATE_HI = os.environ.get("OVERLAY_DATE_HI", "2026-07-08")


def pg_dsn() -> str:
    """psycopg2 DSN from DATABASE_URL (strips the async +asyncpg driver suffix)."""
    return os.environ.get(
        "DATABASE_URL", "postgresql://stockml:stockml_dev@localhost:5433/stockml"
    ).replace("postgresql+asyncpg://", "postgresql://")


def default_context() -> DuckDBContext:
    """DuckDBContext (market panel + NAV marks) from the OVERLAY_* env / pinned defaults."""
    return DuckDBContext(_MARKET_DB, _OHLCV_DB, date_hi=_DATE_HI)

# Board reference config: every research run scored under the SAME deploy-style policy so
# the leaderboard ranks SIGNAL quality, not portfolio policy. K=10 (legacy default), causal,
# T+2, plus a FLAT 1-tỷ-VND ADV floor: liqcol_adv10_ty=1.0 with liqcol_adv252_ty=0.0 turns
# the collapse-veto (adv10<X AND adv252>=Y) into a plain min-ADV floor (Y=0 → always true).
# This is NOT a change to PortfolioConstants defaults (those stay golden-locked); it is
# applied explicitly here and to per-strategy configs that don't override the floor.
REFERENCE_CONSTANTS = dict(
    k=10, tplus=2, stat_mode="causal", liqcol_adv10_ty=1.0, liqcol_adv252_ty=0.0
)

# Fields that define the overlay identity → overlay_config_hash (32-char md5, matches the
# leaderboard_nav.overlay_config_hash column). Every knob the sandbox may expose (K, the
# liquidity floor, the entry-start window date_lo, gates, sizing) is included so a different
# config yields a different hash → correct cache key. Keep in sync with the sandbox inputs.
_HASH_FIELDS = (
    "k", "tplus", "stat_mode", "skip", "skip_mode", "skip_gain", "skip_mu_ref",
    "kconv", "r5thr", "gt", "os_pct", "date_lo", "market_start", "rewrite_on",
    "ec_check_bar", "ret_win", "liqcol_adv10_ty", "liqcol_adv252_ty", "w_invvol",
    "w_liq_full_ty", "max_expo", "vol_cap_q", "crash_pause_ret5", "riskoff_ret5",
    "regime_w_scale",
)  # fmt: skip


def reference_config(**overrides) -> PortfolioConstants:
    """Board reference PortfolioConstants; pass overrides (e.g. k=6, liqcol_adv10_ty=5.0)
    for the sandbox / per-strategy configs."""
    cfg = dict(REFERENCE_CONSTANTS)
    cfg.update(overrides)
    return PortfolioConstants(**cfg)


def overlay_config_hash(C: PortfolioConstants) -> str:
    """Stable 32-char identity of an overlay config (→ cache key + traceability)."""
    d = {k: getattr(C, k) for k in _HASH_FIELDS}
    return hashlib.md5(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()


def load_run_frames(conn, run_id: str, base_run: str | None = None):
    """(base_trades, signals) for a run. base_run lets an OUTPUT-only run borrow BASE
    trades from its engine parent (run_trades = BASE-only invariant, PORTFOLIO_REGISTRATION_
    PIPELINE §1). Signals always come from the scored run itself."""
    base = pd.read_sql(
        "SELECT symbol, entry_date, exit_date, entry_signal_date, entry_price, exit_price, "
        "exit_reason FROM run_trades WHERE run_id=%s",
        conn,
        params=(base_run or run_id,),
    )
    sig = pd.read_sql(
        "SELECT symbol, date, signal, score, exit_score FROM run_signals WHERE run_id=%s",
        conn,
        params=(run_id,),
    )
    return base, sig


def score_overlay(
    conn, run_id: str, ctx: PortfolioContext, C: PortfolioConstants, *, base_run: str | None = None
) -> dict:
    """Load frames + run the Stage-2 overlay. Returns run_portfolio's dict (nav/cagr/maxdd/…)."""
    base, sig = load_run_frames(conn, run_id, base_run)
    if len(base) < 5:
        raise ValueError(f"run {run_id}: only {len(base)} base trades (need >=5)")
    return run_portfolio(base, sig, ctx=ctx, C=C)

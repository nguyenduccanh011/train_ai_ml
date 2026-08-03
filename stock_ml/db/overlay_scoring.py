"""Overlay scoring service — load a run's BASE trades + signals from Postgres, run the
Stage-2 overlay (stock_ml.portfolio.run_portfolio), return metrics.

Shared by the batch registrar (scripts/ops/register_overlay.py) and the live sandbox API
endpoint, so both surfaces score through the ONE unified engine. Lives in stock_ml.db
(outside the stock_ml_core wheel) because it touches the research DB; run_portfolio itself
stays wheel-pure (all data injected via PortfolioContext).
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

from stock_ml.portfolio import (
    DuckDBContext,
    PortfolioConstants,
    overlay_key,
    run_portfolio,
    scoring_hash,
)
from stock_ml.portfolio.context import PortfolioContext
from stock_ml.src.data.universe_resolver import is_nonstock

# Data context defaults: the SERVING-DECLARED market panel artifact (SERVING_PANEL_TASKS.md T1) —
# market.duckdb holds exactly the point-in-time full-market universe serving deploys on (1477
# stock-only symbols, ICB 8995/8985 excluded), with a sibling manifest {market.duckdb}.panel.json
# declaring its identity (n_symbols + symbols_sha256). The board scores against THIS so the
# backtest≡serving band is comparable. NAV marks come from the live ohlcv.db (full coverage to the
# artifact date). Override via env so tests/replays can point elsewhere without code changes.
_MARKET_DB = os.environ.get(
    "OVERLAY_MARKET_DB",
    "C:/Users/DUC CANH PC/Desktop/stock-serving/market_data/market.duckdb",
)
_OHLCV_DB = os.environ.get(
    "OVERLAY_OHLCV_DB",
    "C:/Users/DUC CANH PC/Desktop/stock-serving/data/ohlcv.db",
)
_DATE_HI = os.environ.get("OVERLAY_DATE_HI", "2026-07-31")


def pg_dsn() -> str:
    """psycopg2 DSN from DATABASE_URL (strips the async +asyncpg driver suffix)."""
    return os.environ.get(
        "DATABASE_URL", "postgresql://stockml:stockml_dev@localhost:5433/stockml"
    ).replace("postgresql+asyncpg://", "postgresql://")


@functools.lru_cache(maxsize=4)
def panel_identity(market_db: str = _MARKET_DB) -> dict:
    """Fail-loud guard + identity of the declared market panel artifact.

    The store MUST match its sibling manifest `{market_db}.panel.json` (written by the serving
    panel build) on n_symbols + symbols_sha256 — this guarantees the board scores on the SAME
    declared universe serving deploys on (single-declared-artifact principle, CONVICTION_UNIVERSE_
    UNIFICATION §5.1). Returns {n_symbols, symbols_sha256, panel_fingerprint, filter, max_date};
    ``panel_fingerprint`` (rows:max_date:Σclose:Σvolume) feeds overlay_config_hash so a re-declared
    panel forces a board re-score. Memoized per store path (immutable within a process)."""
    import duckdb

    manifest_path = Path(market_db + ".panel.json")
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"declared panel manifest missing: {manifest_path} — the overlay board requires the "
            f"serving-declared artifact (SERVING_PANEL_TASKS.md T1). Point OVERLAY_MARKET_DB at "
            f"the market.duckdb whose sibling .panel.json declares its identity."
        )
    manifest = json.loads(manifest_path.read_text())
    cx = duckdb.connect(market_db, read_only=True)
    syms = [
        r[0]
        for r in cx.execute(
            "SELECT DISTINCT symbol FROM ohlcv WHERE timeframe='1D' ORDER BY symbol"
        ).fetchall()
    ]
    rows, max_date, sum_close, sum_vol = cx.execute(
        "SELECT count(*), max(date), sum(close), sum(volume) FROM ohlcv WHERE timeframe='1D'"
    ).fetchone()
    cx.close()
    sha = hashlib.sha256("\n".join(syms).encode()).hexdigest()  # syms already sorted by the query
    if sha != manifest["symbols_sha256"] or len(syms) != manifest["n_symbols"]:
        raise ValueError(
            f"panel identity drift: store {market_db} has {len(syms)} symbols "
            f"sha={sha[:12]} but manifest declares {manifest['n_symbols']} "
            f"sha={manifest['symbols_sha256'][:12]}. Re-declare the panel (serving panel build) "
            f"or repoint OVERLAY_MARKET_DB — the board must score the DECLARED universe."
        )
    return {
        "n_symbols": manifest["n_symbols"],
        "symbols_sha256": manifest["symbols_sha256"],
        "panel_fingerprint": f"{rows}:{max_date}:{round(float(sum_close), 2)}:{int(sum_vol)}",
        "filter": manifest.get("filter"),
        "max_date": str(max_date),
    }


def panel_fingerprint() -> str:
    """rows:max_date:Σclose:Σvolume identity of the declared panel (for config_hash + traceability)."""
    return panel_identity(_MARKET_DB)["panel_fingerprint"]


def default_context() -> DuckDBContext:
    """DuckDBContext (declared market panel + live NAV marks). Asserts the panel matches its
    manifest (fail-loud) before returning — a drifted/undeclared store aborts here."""
    panel_identity(_MARKET_DB)  # fail-loud on artifact drift (memoized)
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

def reference_config(**overrides) -> PortfolioConstants:
    """Board reference PortfolioConstants; pass overrides (e.g. k=6, liqcol_adv10_ty=5.0)
    for the sandbox / per-strategy configs."""
    cfg = dict(REFERENCE_CONSTANTS)
    cfg.update(overrides)
    return PortfolioConstants(**cfg)


def overlay_config_hash(C: PortfolioConstants, panel_fp: str | None = None) -> str:
    """Back-compat alias. The identity now lives in the WHEEL (stock_ml.portfolio.identity) so
    serving computes the SAME key for its tiers.yaml strategies without a side agreement.

    Prefer the explicit names at call sites — they are two different questions:
      ``overlay_key(C)``            WHICH strategy  -> primary key of a book, panel-independent
      ``scoring_hash(C, panel_fp)`` WHICH scoring   -> idempotency, moves when the panel is re-declared
    See docs/refactor/OVERLAY_IDENTITY_RESTRUCTURE.md §3.1.
    """
    return overlay_key(C) if panel_fp is None else scoring_hash(C, panel_fp)


def reference_key() -> str:
    """overlay_key of the BOARD REFERENCE config — the one strategy leaderboard_nav may hold.
    Every other strategy lives in run_overlay only, so a per-strategy score can no longer
    overwrite the board's signal-quality ranking."""
    return overlay_key(reference_config())


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
    # Drop ETF/fund-cert legs (ICB 8995/8985) the pre-fix universe_resolver let into some bases —
    # ONE chokepoint for BOTH the board (score_overlay) and the deploy-tier scorer, so both size on
    # the SAME stock-only leg set (else a CCQ leg is sized on a fake neutral 0.5 conviction). Mirrors
    # serving core._drop_non_stock_trades. Forward: the universe re-pin regenerates a clean base.
    # (PORTFOLIO_WRITE_UNIFICATION_IMPL §5.1)
    base = base[~base["symbol"].map(is_nonstock)].reset_index(drop=True)
    return base, sig


def score_overlay(
    conn,
    run_id: str,
    ctx: PortfolioContext,
    C: PortfolioConstants,
    *,
    base_run: str | None = None,
    bundle: dict | None = None,
    emit_pending: bool = False,
) -> dict:
    """Load frames + run the Stage-2 overlay. Returns run_portfolio's dict (nav/cagr/maxdd/…).
    Pass ``bundle`` (build_panel_bundle) to reuse a prebuilt market panel across a batch.
    ``emit_pending`` forwards to run_portfolio (set when the caller will persist the detail tab)."""
    base, sig = load_run_frames(conn, run_id, base_run)
    if len(base) < 5:
        raise ValueError(f"run {run_id}: only {len(base)} base trades (need >=5)")
    return run_portfolio(base, sig, ctx=ctx, C=C, bundle=bundle, emit_pending=emit_pending)

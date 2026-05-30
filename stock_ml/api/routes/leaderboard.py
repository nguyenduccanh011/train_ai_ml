"""Leaderboard endpoints"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings
from ..middleware.rate_limiter import get_limiter

logger = logging.getLogger(__name__)
router = APIRouter(tags=["leaderboard"])
limiter = get_limiter()


# ------------------------------------------------------------------ #
# DB dependency — only imported when db_enabled=True
# ------------------------------------------------------------------ #


def _get_db_dep():
    from stock_ml.db.engine import get_db

    return Depends(get_db)


def _calculate_composite_score(model: dict) -> float:
    """Calculate composite score using official system formula.

    Official weights (from src/evaluation/scoring.py):
    + norm_sharpe:   0.18  (sharpe normalized by tanh)
    + norm_avg_pnl:  0.28  (average pnl per trade)
    + norm_pf:       0.26  (profit factor)
    - norm_mdd:      0.12  (max drawdown penalty)
    - norm_yr_cv:    0.05  (yearly consistency penalty)
    - norm_hold:     0.01  (hold days penalty)
    * confidence:    sqrt(n_trades / 1000) shrinkage
    * pnl_scale:     0.18 * tanh(total_pnl / 18000) bonus

    Returns score in 0-1000+ scale.
    """
    import math

    # Extract metrics
    n_trades = model.get("n_trades", 0)
    if n_trades == 0:
        return 0.0

    avg_pnl = model.get("avg_pnl", 0)
    pf = model.get("profit_factor", 0)
    max_loss = model.get("max_loss", 0)
    sharpe = model.get("sharpe", 0)
    avg_hold = model.get("avg_hold_days", 0)
    pnl_pct = model.get("pnl_pct", 0)

    # Normalize components using tanh and exp transforms
    norm_sharpe = math.tanh(sharpe / 0.55)
    norm_avg = math.tanh(avg_pnl / 12.0)
    norm_pf = 1.0 - math.exp(-max(pf - 1.0, 0.0) / 9.0)
    norm_mdd = 1.0 - math.exp(-max(abs(max_loss), 0.0) / 35.0)
    norm_yr = max(0.0, 0.0 - 0.35) / 2.0  # yr_consistency unavailable for test data
    norm_hold = max(avg_hold - 50.0, 0.0) / 25.0

    # Quality score
    quality_score = (
        0.18 * norm_sharpe
        + 0.28 * norm_avg
        + 0.26 * norm_pf
        - 0.12 * min(max(norm_mdd, 0), 1)
        - 0.05 * min(max(norm_yr, 0), 1)
        - 0.01 * min(max(norm_hold, 0), 1)
    ) * 1000

    # Confidence shrinkage based on trade count
    confidence = min(1.0, math.sqrt(n_trades / 1000.0))

    # Total PnL scale bonus (compressed)
    total_pnl = pnl_pct  # Approximate for single timeframe
    pnl_scale = math.tanh(max(total_pnl, 0.0) / 18000.0)

    # Final score
    final_score = quality_score * confidence + 0.18 * pnl_scale * 1000
    return round(final_score, 1)


def _make_test_data() -> dict:
    """Return test leaderboard data using official system scoring formula.

    Official weights (from src/evaluation/scoring.py):
    - Sharpe: 18% (risk-adjusted per-trade return)
    - AvgPnL: 28% (per-trade expectancy)
    - ProfitFactor: 26% (win/loss ratio efficiency)
    - MDD: -12% penalty (per-position risk)
    - YearlyCV: -5% penalty (consistency across years)
    - HoldDays: -1% penalty (overtrading)
    - Trade confidence: sqrt(n_trades / 1000) shrinkage
    - PnL scale: 18% bonus for large total_pnl
    """
    models = [
        {
            "name": "Technical Rules (MACD+MA20)",
            "market": "vn_stock",
            "timeframe": "1D",
            "year": 2026,
            "model_type": "rule_based",
            "n_trades": 74,
            "pnl_pct": -41.2,
            "win_rate": 0.405,
            "profit_factor": 0.833,
            "avg_pnl": -0.00557,
            "max_win": 0.2198,
            "max_loss": -0.1155,
            "avg_hold_days": 15.7,
            "max_drawdown": 0.412,
            "sharpe": -0.5,
            "notes": "Rule-based (MACD crossover + MA20 filter). Simpler signal, fewer false entries.",
            "audit_status": "PASS",
            "date_run": "2026-05-30",
        },
        {
            "name": "ML Regression (LightGBM)",
            "market": "vn_stock",
            "timeframe": "1D",
            "year": 2026,
            "model_type": "lightgbm",
            "n_trades": 210,
            "pnl_pct": -118.68,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_pnl": -0.565,
            "max_win": 0.0,
            "max_loss": -5.0,
            "avg_hold_days": 14.2,
            "max_drawdown": 1.1868,
            "sharpe": -8.5,
            "notes": "ML model: 20 seeds, leading_v2 features (36), purged KFold. Overfitting detected.",
            "audit_status": "PASS",
            "date_run": "2026-05-29",
        },
    ]

    # Calculate composite scores using official formula
    for model in models:
        model["composite_score"] = _calculate_composite_score(model)

    # Sort by composite score descending
    models.sort(key=lambda m: m["composite_score"], reverse=True)

    # Assign ranks
    for i, model in enumerate(models, 1):
        model["rank"] = i

    return {
        "generated_at": "2026-05-31T00:00:00.000000",
        "models": models,
        "summary": {
            "total_models": len(models),
            "markets": ["vn_stock"],
            "best_model": models[0]["name"],
            "best_score": models[0]["composite_score"],
            "worst_score": models[-1]["composite_score"],
            "note": f"Official system formula: Sharpe 18%, AvgPnL 28%, PF 26%, MDD -12%, YearlyCV -5%, HoldDays -1%. {models[0]['name']}: {models[0]['composite_score']}, {models[1]['name']}: {models[1]['composite_score']}.",
        },
    }


def _make_empty_response() -> dict:
    return {"models": [], "summary": {"total_models": 0, "best_model": None, "best_pnl": None}}


async def _leaderboard_from_db(
    session: AsyncSession,
    market: str | None = None,
    state: str | None = None,
    entry_model: str | None = None,
    limit: int = 200,
) -> dict:
    from stock_ml.db.adapters.leaderboard_adapter import model_to_row
    from stock_ml.db.repositories.run_repo import LeaderboardRunRepository

    repo = LeaderboardRunRepository(session)
    models = await repo.list_ranked(
        market=market, state=state, entry_model=entry_model, limit=limit
    )
    rows = [model_to_row(m).model_dump(mode="json") for m in models]

    best_model = rows[0].get("run_name") if rows else None
    best_score = rows[0].get("composite_score") if rows else None

    return {
        "models": rows,
        "summary": {"total_models": len(rows), "best_model": best_model, "best_pnl": best_score},
    }


def _leaderboard_from_file(limit: int = 200) -> dict:
    # Try multiple locations for leaderboard.json
    paths_to_try = [
        settings.results_dir / "leaderboard.json",
        settings.data_dir / "leaderboard.json",
        settings._root_dir / "stock_ml" / "dashboard" / "leaderboard.json",  # Dashboard static file
    ]

    for path in paths_to_try:
        if path.exists():
            try:
                with open(path, encoding="utf-8-sig") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}, using test data")
                return _make_test_data()

    # Fallback to test data
    logger.info("No leaderboard.json found, returning test data")
    return _make_test_data()


# ------------------------------------------------------------------ #
# Routes
# ------------------------------------------------------------------ #


@router.get("/leaderboard")
@limiter.limit("100/minute")
async def get_leaderboard(
    request: Request,
    market: str | None = None,
    state: str | None = None,
    entry_model: str | None = None,
    limit: int = 200,
) -> dict:
    """Get leaderboard ranked by composite_score DESC.

    Query params: market, state (trained|pinned|retired), entry_model, limit.
    """
    if limit < 1 or limit > 5000:
        raise HTTPException(status_code=400, detail="'limit' must be between 1 and 5000")

    try:
        if settings.db_enabled:
            from stock_ml.db.engine import AsyncSessionLocal

            async with AsyncSessionLocal() as session:
                return await _leaderboard_from_db(
                    session, market=market, state=state, entry_model=entry_model, limit=limit
                )
        return _leaderboard_from_file(limit)

    except Exception as e:
        logger.error(f"Error loading leaderboard: {e}")
        raise HTTPException(status_code=500, detail="Failed to load leaderboard") from e


@router.get("/leaderboard/top/{n}")
@limiter.limit("100/minute")
async def get_top_models(request: Request, n: int = 10) -> dict:
    """Get top N models by composite_score."""
    if n < 1 or n > 1000:
        raise HTTPException(status_code=400, detail="Parameter 'n' must be between 1 and 1000")

    try:
        if settings.db_enabled:
            from stock_ml.db.engine import AsyncSessionLocal

            async with AsyncSessionLocal() as session:
                result = await _leaderboard_from_db(session, limit=n)
                return {"models": result["models"]}

        data = _leaderboard_from_file()
        models = data.get("models", [])
        sorted_models = sorted(models, key=lambda x: x.get("composite_score", 0), reverse=True)
        return {"models": sorted_models[:n]}

    except Exception as e:
        logger.error(f"Error getting top models: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve top models") from e


@router.get("/leaderboard/{run_id:path}")
@limiter.limit("100/minute")
async def get_run(request: Request, run_id: str) -> dict:
    """Get a single run by run_id."""
    if not settings.db_enabled:
        raise HTTPException(status_code=501, detail="DB not enabled; enable DB_ENABLED=true")

    try:
        from stock_ml.db.adapters.leaderboard_adapter import model_to_row
        from stock_ml.db.engine import AsyncSessionLocal
        from stock_ml.db.repositories.run_repo import LeaderboardRunRepository

        async with AsyncSessionLocal() as session:
            repo = LeaderboardRunRepository(session)
            model = await repo.get_by_run_id(run_id)
            if model is None:
                raise HTTPException(status_code=404, detail="Run not found")
            return model_to_row(model).model_dump(mode="json")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching run {run_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch run") from e

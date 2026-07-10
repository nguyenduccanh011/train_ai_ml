"""Phase 0.5: DB write-path integration tests.

Guards the schema-drift bug where ``LeaderboardRow`` lacked ``model_mode`` /
``signal_mode`` while ``row_to_model`` read them, crashing the upsert path with
``AttributeError``. Also asserts a run persists into ``leaderboard_runs`` plus
the four child tables (trades/signals/symbol_stats/yearly_stats) with matching
row counts — the end-to-end persist contract P3 depends on.

Run with the repo root on PYTHONPATH so ``stock_ml`` resolves as a namespace
package, e.g. ``PYTHONPATH=<repo_root> pytest stock_ml/tests/leaderboard``.
"""

from __future__ import annotations

import asyncio
from datetime import date

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

import stock_ml.db.models  # noqa: F401  — register every table on Base.metadata
from stock_ml.db.adapters.leaderboard_adapter import model_to_row, row_to_model
from stock_ml.db.base import Base
from stock_ml.db.models.signal import RunSignalModel
from stock_ml.db.models.symbol_stat import RunSymbolStatModel
from stock_ml.db.models.trade import RunTradeModel
from stock_ml.db.models.yearly_stat import RunYearlyStatModel
from stock_ml.db.repositories.run_repo import LeaderboardRunRepository
from stock_ml.db.repositories.signal_repo import RunSignalRepository
from stock_ml.db.repositories.symbol_stat_repo import RunSymbolStatRepository
from stock_ml.db.repositories.trade_repo import RunTradeRepository
from stock_ml.db.repositories.yearly_stat_repo import RunYearlyStatRepository
from stock_ml.src.leaderboard.schema import (
    Artifacts,
    CacheKeys,
    CostProfile,
    LeaderboardRow,
    LifecycleState,
    TargetConfig,
)


def _make_row(**overrides) -> LeaderboardRow:
    base = dict(
        run_id="template/persist_test-deadbeef",
        bundle="template",
        run_name="persist_test",
        config_hash="deadbeefcafe1234",
        generated_at="2026-06-01T10:00:00",
        strategy="macd_ma20",
        feature_set="leading_v2",
        entry_model="rule",
        model_mode="rule_only",
        signal_mode="entry_first",
        direction="long",
        target=TargetConfig(type="forward_return", forward_window=5),
        trades=10,
        wr=0.5,
        avg_pnl=0.01,
        total_pnl=0.1,
        pnl_pct=10.0,
        pf=1.5,
        avg_hold=5.0,
        max_win=0.2,
        max_loss=-0.1,
        sharpe=0.3,
        max_drawdown=0.5,
        mdd_per_symbol=0.2,
        yearly_consistency=1.0,
        composite_score=42.0,
        n_symbols=3,
        first_test_year=2020,
        last_test_year=2024,
        fairness_group_key="template_vn_stock",
        cost_profile=CostProfile(commission=0.0025, tax=0.001, slippage=0.0015),
        cache_keys=CacheKeys(),
        artifacts=Artifacts(),
        state=LifecycleState.trained,
        universe_slug="vn30",
        universe_version=1,
    )
    base.update(overrides)
    return LeaderboardRow(**base)


def test_row_to_model_no_attribute_error():
    """row_to_model must read model_mode/signal_mode without AttributeError."""
    row = _make_row(model_mode="hybrid_ml_entry_rule_exit", signal_mode="entry_first")
    model = row_to_model(row)
    assert model.model_mode == "hybrid_ml_entry_rule_exit"
    assert model.signal_mode == "entry_first"
    assert model.direction == "long"
    # Symmetric: ORM -> Row round-trips the same values.
    back = model_to_row(model)
    assert back.model_mode == "hybrid_ml_entry_rule_exit"
    assert back.signal_mode == "entry_first"


async def _persist_and_read() -> dict:
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        Session = async_sessionmaker(engine, expire_on_commit=False)
        run_id = "template/persist_test-deadbeef"
        async with Session() as session:
            await LeaderboardRunRepository(session).upsert(
                _make_row(), template_config_hash="deadbeefcafe1234"
            )
            await RunTradeRepository(session).bulk_insert(
                run_id,
                [
                    {"symbol": "AAA", "entry_date": date(2020, 1, 2), "pnl_pct": 0.05},
                    {"symbol": "BBB", "entry_date": date(2020, 1, 3), "pnl_pct": -0.02},
                ],
            )
            await RunSignalRepository(session).bulk_insert(
                run_id,
                [
                    {"symbol": "AAA", "date": date(2020, 1, 2), "signal": 1, "score": 0.7},
                    {"symbol": "BBB", "date": date(2020, 1, 3), "signal": -1, "score": 0.1},
                ],
            )
            await RunSymbolStatRepository(session).bulk_insert(
                run_id,
                [
                    {"symbol": "AAA", "trades": 1, "total_pnl": 0.05},
                    {"symbol": "BBB", "trades": 1, "total_pnl": -0.02},
                ],
            )
            await RunYearlyStatRepository(session).bulk_insert(
                run_id, [{"year": 2020, "trades": 2, "total_pnl": 0.03}]
            )
            await session.commit()

        async with Session() as session:
            fetched = await LeaderboardRunRepository(session).get_by_run_id(run_id)

            async def _count(model) -> int:
                res = await session.execute(
                    select(func.count()).select_from(model).where(model.run_id == run_id)
                )
                return res.scalar_one()

            return {
                "row": model_to_row(fetched) if fetched else None,
                "model_mode": fetched.model_mode if fetched else None,
                "signal_mode": fetched.signal_mode if fetched else None,
                "direction": fetched.direction if fetched else None,
                "universe_slug": fetched.universe_slug if fetched else None,
                "trades": await _count(RunTradeModel),
                "signals": await _count(RunSignalModel),
                "symbol_stats": await _count(RunSymbolStatModel),
                "yearly_stats": await _count(RunYearlyStatModel),
            }
    finally:
        await engine.dispose()


def test_run_persists_to_leaderboard_and_child_tables():
    """A run upserts to leaderboard_runs and all four child tables with matching counts."""
    result = asyncio.run(_persist_and_read())

    assert result["row"] is not None, "run missing from leaderboard_runs after upsert"
    assert result["model_mode"] == "rule_only"
    assert result["signal_mode"] == "entry_first"
    assert result["direction"] == "long"
    assert result["universe_slug"] == "vn30"
    # Row counts must match what was inserted in-memory.
    assert result["trades"] == 2
    assert result["signals"] == 2
    assert result["symbol_stats"] == 2
    assert result["yearly_stats"] == 1
    # Round-tripped DTO carries the component composition forward.
    assert result["row"].model_mode == "rule_only"
    assert result["row"].signal_mode == "entry_first"

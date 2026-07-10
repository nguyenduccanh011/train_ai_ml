"""Phase 3: seed_features populates feature_def + feature_set with correct counts."""

from __future__ import annotations

import asyncio

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

import stock_ml.db.models  # noqa: F401  — register all tables on Base.metadata
from stock_ml.db.base import Base
from stock_ml.db.models.feature import (
    FeatureDefModel,
    FeatureSetMemberModel,
    FeatureSetModel,
)
from stock_ml.scripts.seed_features import FEATURES, seed_with_session
from stock_ml.src.features.dsl.engine import extract_deps, infer_kind
from stock_ml.src.features.dsl.parser import parse


def test_all_feature_expressions_parse_and_classify():
    for name, expr in FEATURES.items():
        node = parse(expr)  # must not raise
        kind = infer_kind(node)
        assert kind in {"per_symbol", "cross_sectional", "market"}, name
        refs, _ = extract_deps(node)
        for r in refs:
            assert r in FEATURES, f"{name} references unknown feature #{r}"
    assert infer_kind(parse(FEATURES["momentum_rank"])) == "cross_sectional"
    assert infer_kind(parse(FEATURES["market_trend"])) == "market"
    assert infer_kind(parse(FEATURES["rsi_14"])) == "per_symbol"


async def _seed_and_count() -> dict:
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as session:
        await seed_with_session(session)
        # Idempotency: a second run must not duplicate rows.
        await seed_with_session(session)

        n_features = (await session.execute(select(func.count(FeatureDefModel.id)))).scalar()
        set_counts = {}
        sets = (await session.execute(select(FeatureSetModel))).scalars().all()
        for fs in sets:
            cnt = (
                await session.execute(
                    select(func.count(FeatureSetMemberModel.id)).where(
                        FeatureSetMemberModel.feature_set_id == fs.id
                    )
                )
            ).scalar()
            set_counts[fs.name] = cnt
    await engine.dispose()
    return {"features": n_features, "sets": set_counts}


def test_seed_counts():
    result = asyncio.run(_seed_and_count())
    assert result["features"] == len(FEATURES)
    # True counts from the actual feature lists (legacy docstrings under-counted by 1).
    assert result["sets"]["basic_v1"] == 8
    assert result["sets"]["leading_v2"] == 37
    assert result["sets"]["leading_v3"] == 56
    assert result["sets"]["leading_deriv"] == 37
    assert result["sets"]["leading_v4"] == 56

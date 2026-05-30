"""Repository for leaderboard_runs table."""

from __future__ import annotations

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from stock_ml.db.adapters.leaderboard_adapter import row_to_model
from stock_ml.db.models.run import LeaderboardRunModel
from stock_ml.src.leaderboard.schema import LeaderboardRow


class LeaderboardRunRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_run_id(self, run_id: str) -> LeaderboardRunModel | None:
        result = await self._session.execute(
            select(LeaderboardRunModel).where(LeaderboardRunModel.run_id == run_id)
        )
        return result.scalar_one_or_none()

    async def upsert(self, row: LeaderboardRow) -> LeaderboardRunModel:
        """Insert or update a run. Sets parent_run_id when superseding an older run."""
        # Find any older run with same bundle+run_name that isn't this one
        existing = await self._session.execute(
            select(LeaderboardRunModel)
            .where(
                LeaderboardRunModel.bundle == row.bundle,
                LeaderboardRunModel.run_name == row.run_name,
                LeaderboardRunModel.run_id != row.run_id,
                LeaderboardRunModel.superseded.is_(False),
            )
            .order_by(LeaderboardRunModel.generated_at.desc())
            .limit(1)
        )
        old = existing.scalar_one_or_none()
        parent_run_id = old.run_id if old else None

        model = row_to_model(row, parent_run_id=parent_run_id)
        values = {c.name: getattr(model, c.name) for c in model.__table__.columns if c.name != "id"}

        stmt = insert(LeaderboardRunModel).values(**values)
        stmt = stmt.on_conflict_do_update(
            index_elements=["run_id"],
            set_={k: stmt.excluded[k] for k in values if k != "run_id"},
        )
        await self._session.execute(stmt)
        await self._session.flush()

        if old:
            await self.mark_superseded(row.bundle, row.run_name, except_run_id=row.run_id)

        return await self.get_by_run_id(row.run_id)  # type: ignore[return-value]

    async def mark_superseded(self, bundle: str, run_name: str, except_run_id: str) -> int:
        result = await self._session.execute(
            update(LeaderboardRunModel)
            .where(
                LeaderboardRunModel.bundle == bundle,
                LeaderboardRunModel.run_name == run_name,
                LeaderboardRunModel.run_id != except_run_id,
            )
            .values(superseded=True)
        )
        return result.rowcount  # type: ignore[return-value]

    async def list_ranked(
        self,
        *,
        market: str | None = None,
        strategy: str | None = None,
        feature_set: str | None = None,
        entry_model: str | None = None,
        state: str | None = None,
        timeframe: str | None = None,
        superseded: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[LeaderboardRunModel]:
        q = select(LeaderboardRunModel).where(LeaderboardRunModel.superseded.is_(superseded))
        if market:
            q = q.where(LeaderboardRunModel.market == market)
        if strategy:
            q = q.where(LeaderboardRunModel.strategy == strategy)
        if feature_set:
            q = q.where(LeaderboardRunModel.feature_set == feature_set)
        if entry_model:
            q = q.where(LeaderboardRunModel.entry_model == entry_model)
        if state:
            q = q.where(LeaderboardRunModel.state == state)
        if timeframe:
            q = q.where(LeaderboardRunModel.timeframe == timeframe)
        q = q.order_by(LeaderboardRunModel.composite_score.desc()).limit(limit).offset(offset)
        result = await self._session.execute(q)
        return list(result.scalars().all())

    async def get_by_fairness_group(self, key: str) -> list[LeaderboardRunModel]:
        result = await self._session.execute(
            select(LeaderboardRunModel)
            .where(LeaderboardRunModel.fairness_group_key == key)
            .order_by(LeaderboardRunModel.composite_score.desc())
        )
        return list(result.scalars().all())

    async def get_version_chain(self, run_id: str) -> list[LeaderboardRunModel]:
        """Walk parent_run_id chain (newest first)."""
        chain: list[LeaderboardRunModel] = []
        current_id: str | None = run_id
        seen: set[str] = set()
        while current_id and current_id not in seen:
            seen.add(current_id)
            row = await self.get_by_run_id(current_id)
            if row is None:
                break
            chain.append(row)
            current_id = row.parent_run_id
        return chain

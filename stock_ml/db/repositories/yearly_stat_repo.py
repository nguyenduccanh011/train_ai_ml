"""Repository for run_yearly_stats table."""

from __future__ import annotations

from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from stock_ml.db.models.yearly_stat import RunYearlyStatModel


class RunYearlyStatRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def bulk_insert(self, run_id: str, stats: list[dict[str, Any]]) -> int:
        """Idempotent bulk insert. Skips rows that already exist (by run_id+year)."""
        if not stats:
            return 0
        rows = [{"run_id": run_id, **s} for s in stats]

        dialect_name = self._session.bind.dialect.name if self._session.bind else None
        if dialect_name == "postgresql":
            stmt = pg_insert(RunYearlyStatModel).values(rows)
        else:  # SQLite or others
            stmt = sqlite_insert(RunYearlyStatModel).values(rows)

        stmt = stmt.on_conflict_do_nothing()
        result = await self._session.execute(stmt)
        return result.rowcount  # type: ignore[return-value]

    async def get_by_run_id(self, run_id: str) -> list[RunYearlyStatModel]:
        """Get yearly stats for a run."""
        result = await self._session.execute(
            select(RunYearlyStatModel).where(RunYearlyStatModel.run_id == run_id)
        )
        return list(result.scalars().all())

    async def delete_by_run_id(self, run_id: str) -> int:
        """Delete all yearly stats for a run."""
        result = await self._session.execute(
            delete(RunYearlyStatModel).where(RunYearlyStatModel.run_id == run_id)
        )
        return result.rowcount  # type: ignore[return-value]

"""Repository for run_symbol_stats table."""

from __future__ import annotations

from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from stock_ml.db.models.symbol_stat import RunSymbolStatModel


class RunSymbolStatRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def bulk_insert(self, run_id: str, stats: list[dict[str, Any]]) -> int:
        """Idempotent bulk insert. Skips rows that already exist (by run_id+symbol)."""
        if not stats:
            return 0
        rows = [{"run_id": run_id, **s} for s in stats]

        dialect_name = self._session.bind.dialect.name if self._session.bind else None
        if dialect_name == "postgresql":
            stmt = pg_insert(RunSymbolStatModel).values(rows)
        else:  # SQLite or others
            stmt = sqlite_insert(RunSymbolStatModel).values(rows)

        stmt = stmt.on_conflict_do_nothing()
        result = await self._session.execute(stmt)
        return result.rowcount  # type: ignore[return-value]

    async def get_by_run_id(self, run_id: str) -> list[RunSymbolStatModel]:
        """Get symbol stats for a run."""
        result = await self._session.execute(
            select(RunSymbolStatModel).where(RunSymbolStatModel.run_id == run_id)
        )
        return list(result.scalars().all())

    async def delete_by_run_id(self, run_id: str) -> int:
        """Delete all symbol stats for a run."""
        result = await self._session.execute(
            delete(RunSymbolStatModel).where(RunSymbolStatModel.run_id == run_id)
        )
        return result.rowcount  # type: ignore[return-value]

"""Repository for run_trades table."""
from __future__ import annotations

from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from stock_ml.db.models.trade import RunTradeModel


class RunTradeRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def bulk_insert(self, run_id: str, trades: list[dict[str, Any]]) -> int:
        """Idempotent bulk insert. Skips rows that already exist (by run_id+symbol+entry_date)."""
        if not trades:
            return 0
        rows = [{"run_id": run_id, **t} for t in trades]
        stmt = insert(RunTradeModel).values(rows)
        stmt = stmt.on_conflict_do_nothing()
        result = await self._session.execute(stmt)
        return result.rowcount  # type: ignore[return-value]

    async def get_by_run_id(self, run_id: str) -> list[RunTradeModel]:
        result = await self._session.execute(
            select(RunTradeModel).where(RunTradeModel.run_id == run_id)
        )
        return list(result.scalars().all())

    async def delete_by_run_id(self, run_id: str) -> int:
        result = await self._session.execute(
            delete(RunTradeModel).where(RunTradeModel.run_id == run_id)
        )
        return result.rowcount  # type: ignore[return-value]

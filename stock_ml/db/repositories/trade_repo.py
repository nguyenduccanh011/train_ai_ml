"""Repository for run_trades table."""

from __future__ import annotations

from datetime import date
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from stock_ml.db.models.trade import RunTradeModel

_DATE_FIELDS = ("entry_date", "exit_date", "entry_signal_date")


def _to_date(value: Any) -> date | None:
    """Coerce an ISO date string to a date. asyncpg rejects str for DATE columns."""
    if value is None or value == "":
        return None
    if isinstance(value, str):
        return date.fromisoformat(value[:10])
    return value


class RunTradeRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def bulk_insert(self, run_id: str, trades: list[dict[str, Any]]) -> int:
        """Idempotent bulk insert. Skips rows that already exist (by run_id+symbol+entry_date)."""
        if not trades:
            return 0
        rows = []
        for t in trades:
            row = {"run_id": run_id, **t}
            for field in _DATE_FIELDS:
                if field in row:
                    row[field] = _to_date(row[field])
            rows.append(row)

        # Use dialect-agnostic insert: detect DB type and use appropriate insert
        dialect_name = self._session.bind.dialect.name if self._session.bind else None
        insert_fn = pg_insert if dialect_name == "postgresql" else sqlite_insert

        # Chunk to stay under asyncpg's 32767 bind-parameter cap.
        chunk_size = 2000
        total = 0
        for start in range(0, len(rows), chunk_size):
            chunk = rows[start : start + chunk_size]
            stmt = insert_fn(RunTradeModel).values(chunk).on_conflict_do_nothing()
            result = await self._session.execute(stmt)
            total += result.rowcount or 0
        return total

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

"""Repository for run_signals table."""

from __future__ import annotations

from datetime import date
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from stock_ml.db.models.signal import RunSignalModel


def _to_date(value: Any) -> date | None:
    """Coerce an ISO date string to a date. asyncpg rejects str for DATE columns."""
    if value is None or value == "":
        return None
    if isinstance(value, str):
        return date.fromisoformat(value[:10])
    return value


class RunSignalRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def bulk_insert(self, run_id: str, signals: list[dict[str, Any]]) -> int:
        """Idempotent bulk insert. Skips rows that already exist (by run_id+symbol+date)."""
        if not signals:
            return 0
        rows = []
        for s in signals:
            row = {"run_id": run_id, **s}
            if "date" in row:
                row["date"] = _to_date(row["date"])
            rows.append(row)

        dialect_name = self._session.bind.dialect.name if self._session.bind else None
        insert_fn = pg_insert if dialect_name == "postgresql" else sqlite_insert

        # Chunk to stay under asyncpg's 32767 bind-parameter cap. run_signals inserts 7 cols/row
        # (run_id, symbol, date, signal, score, exit_score, + on-conflict target) -> 4681 max; 4000
        # keeps a margin while 2x fewer round-trips than the old 2000 (persist is the bottleneck).
        chunk_size = 4000
        total = 0
        for start in range(0, len(rows), chunk_size):
            chunk = rows[start : start + chunk_size]
            stmt = insert_fn(RunSignalModel).values(chunk).on_conflict_do_nothing()
            result = await self._session.execute(stmt)
            total += result.rowcount or 0
        return total

    async def get_by_run_id(
        self, run_id: str, symbol: str | None = None, limit: int = 2000
    ) -> list[RunSignalModel]:
        """Get signals for a run, optionally filtered by symbol."""
        query = select(RunSignalModel).where(RunSignalModel.run_id == run_id)
        if symbol:
            query = query.where(RunSignalModel.symbol == symbol)
        query = query.limit(limit)
        result = await self._session.execute(query)
        return list(result.scalars().all())

    async def delete_by_run_id(self, run_id: str) -> int:
        """Delete all signals for a run."""
        result = await self._session.execute(
            delete(RunSignalModel).where(RunSignalModel.run_id == run_id)
        )
        return result.rowcount  # type: ignore[return-value]

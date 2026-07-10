"""FastAPI dependency injection for database sessions."""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from stock_ml.db.engine import AsyncSessionLocal


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency injection for AsyncSession.

    Usage:
        @app.get("/items")
        async def get_items(session: AsyncSession = Depends(get_db)):
            ...
    """
    async with AsyncSessionLocal() as session:
        yield session

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine


def _get_settings():
    from stock_ml.api.config import settings
    return settings


def _make_async_url(url: str) -> str:
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url


def _make_sync_url(url: str) -> str:
    if url.startswith("postgresql+asyncpg://"):
        return url.replace("postgresql+asyncpg://", "postgresql://", 1)
    return url


def _build_engines():
    s = _get_settings()
    async_url = _make_async_url(s.database_url)
    sync_url = _make_sync_url(s.database_url)

    _async_engine = create_async_engine(
        async_url,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        echo=s.debug,
    )
    _async_session = async_sessionmaker(_async_engine, expire_on_commit=False)
    _sync_engine = create_engine(sync_url, pool_pre_ping=True)
    return _async_engine, _async_session, _sync_engine


async_engine, AsyncSessionLocal, sync_engine = _build_engines()


async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session

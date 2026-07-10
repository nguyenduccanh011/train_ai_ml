from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

REPO_ROOT = Path(__file__).resolve().parents[2]


def _get_settings():
    from stock_ml.api.config import settings

    return settings


def _make_async_url(url: str) -> str:
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    if url.startswith("sqlite://"):
        # Preserve the original slash count (sqlite:///rel vs sqlite:////abs).
        # Replacing the 2-slash prefix with a 3-slash one would inject an extra
        # leading slash before a Windows drive (sqlite+aiosqlite:////C:\...) → broken.
        return url.replace("sqlite://", "sqlite+aiosqlite://", 1)
    return url


def _make_sync_url(url: str) -> str:
    if url.startswith("postgresql+asyncpg://"):
        return url.replace("postgresql+asyncpg://", "postgresql://", 1)
    if url.startswith("sqlite+aiosqlite://"):
        return url.replace("sqlite+aiosqlite://", "sqlite://", 1)
    return url


def _is_sqlite(url: str) -> bool:
    return "sqlite" in url


def _build_engines():
    s = _get_settings()
    db_url = s.database_url

    # Resolve relative SQLite paths to absolute
    if db_url.startswith("sqlite:///") and not db_url.startswith("sqlite:////"):
        rel_path = db_url[len("sqlite:///") :]
        abs_path = REPO_ROOT / rel_path
        db_url = f"sqlite:///{abs_path}"

    async_url = _make_async_url(db_url)
    sync_url = _make_sync_url(db_url)
    is_sqlite = _is_sqlite(db_url)

    if is_sqlite:
        _async_engine = create_async_engine(
            async_url,
            echo=s.debug,
            connect_args={"timeout": 30},
        )
        _async_session = async_sessionmaker(_async_engine, expire_on_commit=False)
        _sync_engine = create_engine(
            sync_url,
            connect_args={"timeout": 30, "check_same_thread": False},
        )

        @event.listens_for(_sync_engine, "connect")
        def _sqlite_pragma(dbapi_conn, connection_record):
            dbapi_conn.execute("PRAGMA journal_mode=WAL")
            dbapi_conn.execute("PRAGMA foreign_keys=ON")
    else:
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

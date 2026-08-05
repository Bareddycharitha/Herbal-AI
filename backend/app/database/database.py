"""
Database Engine and Session Factory

Creates the async SQLAlchemy engine and session factory for PostgreSQL.
"""

from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from backend.app.config import settings
from backend.app.database.base import Base


def create_engine() -> AsyncEngine:
    """Create and return the async SQLAlchemy engine."""
    engine = create_async_engine(
        settings.database_url,
        echo=False,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
        pool_recycle=300,
    )
    return engine


def create_session_factory(engine: AsyncEngine | None = None) -> async_sessionmaker[AsyncSession]:
    """Create and return an async session factory."""
    if engine is None:
        engine = create_engine()
    return async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
        autocommit=False,
    )


async def init_db() -> None:
    """Initialize database tables."""
    engine = create_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await engine.dispose()


async def shutdown_db() -> None:
    """Dispose of the database engine."""
    engine = create_engine()
    await engine.dispose()


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Async generator that yields a database session.

    Usage:
        async with get_session() as session:
            ...
    """
    session_factory = create_session_factory()
    async with session_factory() as session:
        yield session

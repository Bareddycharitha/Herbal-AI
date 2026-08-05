"""
Database Session Dependency

FastAPI dependency for async database sessions.
"""

from typing import AsyncGenerator

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.database.database import create_session_factory


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that yields an async database session.

    Usage:
        async def some_endpoint(db: AsyncSession = Depends(get_db_session)):
            ...
    """
    session_factory = create_session_factory()
    async with session_factory() as session:
        yield session

"""
User Repository

Repository layer for user database operations.
The API must NEVER directly access SQLAlchemy.
"""

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select, update, delete, func
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.models.user import User
from backend.app.schemas.user import UserCreate, UserUpdate, UserRole


class UserRepository:
    """Repository for user database operations."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    async def create(self, user_create: UserCreate) -> User:
        """Create a new user."""
        now = datetime.now(timezone.utc)
        user = User(
            email=user_create.email,
            full_name=user_create.full_name,
            role=user_create.role.value,
            hashed_password=user_create.password,
            is_active=True,
            created_at=now,
            updated_at=now,
        )
        self._session.add(user)
        await self._session.flush()
        await self._session.refresh(user)
        return user

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def get_by_id(self, user_id: int) -> Optional[User]:
        """Get user by ID."""
        result = await self._session.get(User, user_id)
        return result

    async def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        stmt = select(User).where(func.lower(User.email) == email.lower())
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_users(
        self, skip: int = 0, limit: int = 100
    ) -> list[User]:
        """List users with pagination."""
        stmt = select(User).offset(skip).limit(limit)
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    async def update(
        self, user_id: int, updates: dict
    ) -> Optional[User]:
        """Update user fields. Returns the updated user or None."""
        # Check for email conflict
        if "email" in updates and updates["email"] is not None:
            existing = await self.get_by_email(updates["email"])
            if existing and existing.id != user_id:
                return None

        # Handle role type conversion
        if "role" in updates and updates["role"] is not None:
            if isinstance(updates["role"], UserRole):
                updates = {**updates, "role": updates["role"].value}
            elif isinstance(updates["role"], str):
                updates = {**updates, "role": updates["role"].lower()}

        now = datetime.now(timezone.utc)
        updates["updated_at"] = now

        stmt = (
            update(User)
            .where(User.id == user_id)
            .values(**updates)
            .returning(User)
        )
        result = await self._session.execute(stmt)
        updated_user = result.scalar_one_or_none()
        return updated_user

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    async def delete(self, user_id: int) -> bool:
        """Delete user. Returns True if deleted, False if not found."""
        user = await self.get_by_id(user_id)
        if not user:
            return False
        await self._session.delete(user)
        await self._session.flush()
        return True

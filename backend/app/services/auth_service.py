"""
Auth Service

Business logic layer for authentication and user management.
"""

from datetime import datetime, timezone
from typing import Optional

from backend.app.config import settings
from backend.app.models.user import User
from backend.app.repositories.user_repository import UserRepository
from backend.app.schemas.user import (
    UserCreate,
    UserUpdate,
    UserRole,
)
from backend.app.utils.auth import (
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
    verify_token,
)
from backend.app.exceptions import (
    AuthenticationError,
    ValidationError,
    NotFoundError,
    AuthorizationError,
)


class AuthService:
    """Service layer for authentication and user management."""

    def __init__(self, repository: UserRepository) -> None:
        self._repository = repository

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    async def authenticate_user(
        self, email: str, password: str
    ) -> Optional[User]:
        """Authenticate user with email and password."""
        user = await self._repository.get_by_email(email)
        if not user:
            return None
        if not user.is_active:
            return None
        if not verify_password(password, user.hashed_password):
            return None
        return user

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    async def register_user(self, user_create: UserCreate) -> User:
        """Register a new user."""
        # Check if email already exists
        existing = await self._repository.get_by_email(user_create.email)
        if existing:
            raise ValidationError(
                message="Email already registered",
                field="email",
            )
        return await self._repository.create(user_create)

    # ------------------------------------------------------------------
    # User CRUD
    # ------------------------------------------------------------------

    async def get_user_by_id(self, user_id: int) -> Optional[User]:
        """Get user by ID."""
        return await self._repository.get_by_id(user_id)

    async def list_users(
        self, skip: int = 0, limit: int = 100
    ) -> list[User]:
        """List all users with pagination."""
        return await self._repository.list_users(skip=skip, limit=limit)

    async def update_user(
        self, user_id: int, updates: dict
    ) -> Optional[User]:
        """Update user fields. Returns updated user or None if email conflict."""
        updated = await self._repository.update(user_id, updates)
        return updated

    async def delete_user(self, user_id: int) -> bool:
        """Delete a user. Returns True if deleted, False if not found."""
        return await self._repository.delete(user_id)

    # ------------------------------------------------------------------
    # Password Management
    # ------------------------------------------------------------------

    async def change_password(
        self, user_id: int, current_password: str, new_password: str
    ) -> None:
        """Change user password after verifying current password."""
        user = await self._repository.get_by_id(user_id)
        if not user:
            raise NotFoundError("User", str(user_id))
        if not verify_password(current_password, user.hashed_password):
            raise AuthenticationError("Current password is incorrect")

        new_hashed = hash_password(new_password)
        now = datetime.now(timezone.utc)
        await self._repository.update(user_id, {
            "hashed_password": new_hashed,
            "updated_at": now,
        })

    # ------------------------------------------------------------------
    # Token Management
    # ------------------------------------------------------------------

    def create_tokens(self, user: User) -> tuple[str, str]:
        """Create access and refresh tokens for user."""
        access_token = create_access_token(
            user.id, user.email, user.role
        )
        refresh_token = create_refresh_token(
            user.id, user.email, user.role
        )
        return access_token, refresh_token

    async def refresh_access_token(
        self, refresh_token: str
    ) -> Optional[str]:
        """Create new access token from refresh token."""
        payload = verify_token(refresh_token, "refresh")
        if not payload:
            return None

        user = await self._repository.get_by_id(payload.sub)
        if not user or not user.is_active:
            return None

        return create_access_token(user.id, user.email, user.role)

    async def refresh_tokens(
        self, refresh_token: str
    ) -> Optional[tuple[str, str]]:
        """Refresh both access and refresh tokens."""
        new_access = await self.refresh_access_token(refresh_token)
        if not new_access:
            return None

        # Get user for new refresh token rotation
        payload = verify_token(refresh_token, "refresh")
        if payload:
            user = await self._repository.get_by_id(payload.sub)
            if user:
                new_refresh = create_refresh_token(
                    user.id, user.email, user.role
                )
                return new_access, new_refresh

        return new_access, refresh_token

    # ------------------------------------------------------------------
    # Token Validation
    # ------------------------------------------------------------------

    def get_user_from_token(self, token: str) -> Optional[User]:
        """Get user from access token."""
        payload = verify_token(token, "access")
        if not payload:
            return None
        # Synchronous get_by_id needed here; we use the session directly
        # This is a limitation of the sync context in utils/auth.py
        # The actual implementation uses the repository via async
        return None  # Handled by async get_user_from_token below

    async def get_user_from_token_async(self, token: str) -> Optional[User]:
        """Async version: get user from access token."""
        payload = verify_token(token, "access")
        if not payload:
            return None
        return await self._repository.get_by_id(payload.sub)

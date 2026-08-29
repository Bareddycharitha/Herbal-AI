"""
Auth Service

Business logic layer for authentication and user management.
Uses Clerk for authentication and Supabase PostgreSQL for
profile persistence. No Supabase Auth is used.
"""

from datetime import datetime, timezone
from typing import Optional

from backend.app.config import settings
from backend.app.repositories.profile_repository import ProfileRepository
from backend.app.schemas.user import (
    UserCreate,
    UserRole,
)
from backend.app.utils.auth import verify_clerk_token
from backend.app.exceptions import (
    AuthenticationError,
    ValidationError,
    NotFoundError,
    AuthorizationError,
)
from backend.app.utils.logging import get_logger

logger = get_logger(__name__)


class AuthService:
    """Service layer for authentication and user management.

    Delegates authentication to Clerk. Manages profiles,
    authorization, and user metadata in Supabase PostgreSQL.
    """

    def __init__(self, repository: ProfileRepository) -> None:
        self._repository = repository

    # ------------------------------------------------------------------
    # Profile Management
    # ------------------------------------------------------------------

    async def ensure_profile(
        self,
        clerk_user_id: str,
        email: str,
        full_name: str | None = None,
    ) -> dict:
        """Ensure a profile exists for the Clerk user.

        If the profile does not exist, it is created automatically
        with the default role of 'user'. This is called on first
        successful authentication.

        Args:
            clerk_user_id: Clerk user ID from the `sub` claim.
            email: User email address.
            full_name: Optional full name.

        Returns:
            The existing or newly created profile dict.
        """
        profile = self._repository.get_profile_by_clerk_id(clerk_user_id)
        if profile is not None:
            return profile

        # Auto-create profile on first authentication
        profile = self._repository.create_profile(
            clerk_user_id=clerk_user_id,
            email=email,
            full_name=full_name,
            role=UserRole.USER,
        )
        logger.info(
            "Auto-created profile for new Clerk user",
            clerk_user_id=clerk_user_id,
            email=email,
        )
        return profile

    # ------------------------------------------------------------------
    # User Lookup
    # ------------------------------------------------------------------

    async def get_user_from_token(
        self, access_token: str
    ) -> Optional[dict]:
        """Get user profile from a Clerk JWT access token.

        Verifies the token, then looks up the profile in the
        database by Clerk user ID. If the profile does not exist,
        it is created automatically.

        Args:
            access_token: Clerk JWT access token.

        Returns:
            User profile dict if the token is valid,
            None otherwise.
        """
        payload = verify_clerk_token(access_token)
        if not payload:
            return None

        return await self.get_user_from_verified_token(payload)

    async def get_user_from_verified_token(
        self, payload: dict
    ) -> Optional[dict]:
        """Get user profile from a verified Clerk token payload.

        Looks up the profile in the database by Clerk user ID.
        If the profile does not exist, it is created automatically.

        Args:
            payload: Verified Clerk token payload.

        Returns:
            User profile dict if the token is valid,
            None otherwise.
        """
        profile = self._repository.get_profile_by_clerk_id(payload["id"])
        if profile is not None:
            return profile

        # Auto-create profile on first authentication
        try:
            profile = await self.ensure_profile(
                clerk_user_id=payload["id"],
                email=payload.get("email", ""),
                full_name=payload.get("full_name"),
            )
            return profile
        except Exception:
            # If profile creation fails, return None to indicate authentication failure
            return None

    async def get_user_by_clerk_id(
        self, clerk_user_id: str
    ) -> Optional[dict]:
        """Get user profile by Clerk user ID."""
        return self._repository.get_profile_by_clerk_id(clerk_user_id)

    # ------------------------------------------------------------------
    # User CRUD
    # ------------------------------------------------------------------

    async def list_users(
        self, skip: int = 0, limit: int = 100
    ) -> list[dict]:
        """List all profiles with pagination."""
        return self._repository.list_profiles(skip=skip, limit=limit)

    async def update_user(
        self, clerk_user_id: str, updates: dict[str, object]
    ) -> Optional[dict]:
        """Update user profile fields. Returns updated profile or None."""
        return self._repository.update_profile(clerk_user_id, updates)

    async def delete_user(self, clerk_user_id: str) -> bool:
        """Delete a user profile. Returns True if deleted, False if not found."""
        profile = self._repository.get_profile_by_clerk_id(clerk_user_id)
        if not profile:
            return False

        return self._repository.delete_profile(clerk_user_id)

    # ------------------------------------------------------------------
    # Role Management
    # ------------------------------------------------------------------

    async def update_user_role(
        self, clerk_user_id: str, role: UserRole
    ) -> Optional[dict]:
        """Update a user's role. Returns updated profile or None."""
        return self._repository.update_profile(
            clerk_user_id, {"role": role.value}
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_user_response(self, profile: dict) -> dict:
        """Build a user response dict from a profile record."""
        return {
            "id": profile.get("clerk_user_id", ""),
            "email": profile.get("email", ""),
            "full_name": profile.get("full_name"),
            "role": profile.get("role", "user"),
            "is_active": profile.get("is_active", True),
            "created_at": profile.get("created_at"),
        }

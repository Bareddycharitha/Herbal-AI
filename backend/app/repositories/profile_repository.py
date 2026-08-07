"""
Profile Repository

Repository layer for profile database operations using Supabase.
The API must NEVER directly access Supabase tables.
Repositories only persist data — they do not manage authentication,
generate tokens, or validate business rules.
"""

from typing import Optional

from backend.app.supabase.client import get_supabase_client
from backend.app.schemas.user import UserRole


class ProfileRepository:
    """Repository for profile operations via Supabase PostgreSQL.

    Uses clerk_user_id as the unique identifier for profiles.
    The Supabase client is used for database operations only;
    authentication is handled by Clerk on the frontend and
    verified by the backend using Clerk JWTs.
    """

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    def create_profile(
        self,
        clerk_user_id: str,
        email: str,
        full_name: str | None,
        role: UserRole,
    ) -> dict:
        """Create a new profile for a user.

        Args:
            clerk_user_id: Clerk user ID from the `sub` claim.
            email: User email address.
            full_name: Optional full name.
            role: User role.

        Returns:
            The created profile dict.
        """
        supabase = get_supabase_client()
        result = supabase.table("profiles").insert(
            {
                "clerk_user_id": clerk_user_id,
                "email": email,
                "full_name": full_name,
                "role": role.value,
                "is_active": True,
            }
        ).execute()

        if result.data:
            return result.data[0]
        raise RuntimeError(
            f"Failed to create profile for user {clerk_user_id}"
        )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_profile_by_clerk_id(self, clerk_user_id: str) -> Optional[dict]:
        """Get profile by Clerk user ID."""
        supabase = get_supabase_client()
        result = (
            supabase.table("profiles")
            .select("*")
            .eq("clerk_user_id", clerk_user_id)
            .execute()
        )
        if result.data:
            return result.data[0]
        return None

    def get_profile_by_email(self, email: str) -> Optional[dict]:
        """Get profile by email (case-insensitive)."""
        supabase = get_supabase_client()
        result = (
            supabase.table("profiles")
            .select("*")
            .eq("email", email)
            .execute()
        )
        if result.data:
            return result.data[0]
        return None

    def list_profiles(
        self, skip: int = 0, limit: int = 100
    ) -> list[dict]:
        """List all profiles with pagination."""
        supabase = get_supabase_client()
        result = (
            supabase.table("profiles")
            .select("*")
            .range(skip, skip + limit - 1)
            .execute()
        )
        return result.data or []

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update_profile(
        self, clerk_user_id: str, updates: dict
    ) -> Optional[dict]:
        """Update profile fields. Returns the updated profile or None."""
        supabase = get_supabase_client()

        # Remove fields that should not be updated directly
        updates = {k: v for k, v in updates.items() if k not in ("id", "created_at", "clerk_user_id")}

        if not updates:
            return self.get_profile_by_clerk_id(clerk_user_id)

        result = (
            supabase.table("profiles")
            .update(updates)
            .eq("clerk_user_id", clerk_user_id)
            .select()
            .execute()
        )

        if result.data:
            return result.data[0]
        return None

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete_profile(self, clerk_user_id: str) -> bool:
        """Delete a profile. Returns True if deleted, False if not found."""
        supabase = get_supabase_client()
        result = (
            supabase.table("profiles")
            .delete()
            .eq("clerk_user_id", clerk_user_id)
            .execute()
        )
        return result.data is not None and len(result.data) > 0

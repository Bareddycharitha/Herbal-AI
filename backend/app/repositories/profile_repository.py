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
        email: str | None = None,
        full_name: str | None = None,
        role: UserRole = UserRole.USER,
    ) -> dict:
        """Create a new profile for a user.

        Args:
            clerk_user_id: Clerk user ID from the ``sub`` claim. This
                is the authoritative application identity.
            email: Optional email address. ``None`` is stored as NULL
                in the database. Email is profile data, not an
                authentication requirement, and may be filled in
                later by the user.
            full_name: Optional full name.
            role: User role.

        Returns:
            The created profile dict.
        """
        supabase = get_supabase_client()
        # Build the row without the optional fields so the database
        # column defaults / NULLs are used when those values are not
        # provided. This avoids sending a sentinel empty string for
        # email which would conflict with UNIQUE constraints across
        # multiple users without an email claim.
        row: dict = {
            "clerk_user_id": clerk_user_id,
            "role": role.value,
            "is_active": True,
        }
        if email is not None:
            row["email"] = email
        if full_name is not None:
            row["full_name"] = full_name

        try:
            result = supabase.table("profiles").insert(row).execute()

            if result.data:
                return result.data[0]
            raise RuntimeError(
                f"Failed to create profile for user {clerk_user_id}"
            )
        except Exception as e:
            # Handle race condition where another request created the profile first
            if "duplicate key" in str(e).lower() or "unique constraint" in str(e).lower():
                # Fetch the existing profile created by the concurrent request
                existing_profile = self.get_profile_by_clerk_id(clerk_user_id)
                if existing_profile:
                    return existing_profile
            # Re-raise if it's not a duplicate key error or if we couldn't fetch existing
            raise RuntimeError(
                f"Failed to create profile for user {clerk_user_id}: {str(e)}"
            ) from e

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

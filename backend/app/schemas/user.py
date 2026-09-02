"""
User Pydantic Schemas

Request and response schemas for user authentication and management.
Uses Clerk user IDs (from the `sub` claim in Clerk JWTs) as the
authoritative application identity. Email is optional profile data
that may be present in the Clerk JWT or filled in later by the user.
"""

from datetime import datetime
from typing import Any, Optional

from enum import Enum

from pydantic import BaseModel, EmailStr, Field, ConfigDict, model_validator


class UserRole(str, Enum):
    """User roles for RBAC."""

    USER = "user"
    ADMIN = "admin"
    RESEARCHER = "researcher"


class UserBase(BaseModel):
    """Base user schema.

    The application's authoritative identity is the Clerk user ID
    (``id`` / ``sub`` claim). Email is optional profile data and is
    not required for authentication or for the existence of a user
    record.
    """

    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    role: UserRole = UserRole.USER


class UserCreate(UserBase):
    """Schema for user registration.

    Note: Clerk handles registration. This schema is used
    for initial profile creation after first Clerk authentication.
    """

    pass


class UserUpdate(BaseModel):
    """Schema for user updates."""

    full_name: Optional[str] = None
    email: Optional[EmailStr] = None
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None


class UserResponse(UserBase):
    """User schema for API responses.

    ``id`` is the Clerk user ID (``sub`` claim) and is the authoritative
    application identity. ``email`` is optional and may be ``None`` if
    the Clerk token does not include an email claim.

    The database column is named ``clerk_user_id``; the model accepts
    both ``id`` and ``clerk_user_id`` keys on input (so the raw
    repository row can be passed directly via ``model_validate``).
    """

    id: str  # Clerk user ID from the `sub` claim
    is_active: bool
    created_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    @model_validator(mode="before")
    @classmethod
    def _accept_clerk_user_id_key(cls, data: Any) -> Any:
        """Allow the raw Supabase row's ``clerk_user_id`` key to be
        mapped onto the ``id`` field without an explicit alias.

        The repository returns rows with the DB column name
        (``clerk_user_id``); the schema exposes that value as
        ``id``. Without this validator, ``UserResponse.model_validate``
        would reject the row because the required ``id`` field is
        missing.
        """
        if isinstance(data, dict) and "id" not in data and "clerk_user_id" in data:
            # Copy the dict to avoid mutating the caller's data.
            new_data = dict(data)
            new_data["id"] = new_data.pop("clerk_user_id")
            return new_data
        return data


class UserProfileResponse(UserResponse):
    """Extended user profile response with additional fields."""

    pass

"""
User Pydantic Schemas

Request and response schemas for user authentication and management.
Uses Clerk user IDs (from the `sub` claim in Clerk JWTs).
"""

from datetime import datetime
from typing import Optional

from enum import Enum

from pydantic import BaseModel, EmailStr, Field, ConfigDict


class UserRole(str, Enum):
    """User roles for RBAC."""

    USER = "user"
    ADMIN = "admin"
    RESEARCHER = "researcher"


class UserBase(BaseModel):
    """Base user schema."""

    email: EmailStr
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
    """User schema for API responses."""

    id: str  # Clerk user ID from the `sub` claim
    is_active: bool
    created_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class UserProfileResponse(UserResponse):
    """Extended user profile response with additional fields."""

    pass

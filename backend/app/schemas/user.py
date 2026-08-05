"""
User Pydantic Schemas

Request and response schemas for user authentication and management.
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
    """Schema for user registration."""
    password: str = Field(..., min_length=8, max_length=100)


class UserUpdate(BaseModel):
    """Schema for user updates."""
    full_name: Optional[str] = None
    email: Optional[EmailStr] = None
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None


class UserInDB(UserBase):
    """User schema as stored in database."""
    id: Optional[int] = None
    hashed_password: str
    is_active: bool = True
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class UserResponse(UserBase):
    """User schema for API responses."""
    id: int
    is_active: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class Token(BaseModel):
    """Token response schema."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds


class TokenPayload(BaseModel):
    """JWT token payload."""
    sub: int  # user_id
    email: str
    role: UserRole
    exp: int
    iat: int
    type: str  # "access" or "refresh"


class LoginRequest(BaseModel):
    """Login request schema."""
    email: EmailStr
    password: str


class RefreshTokenRequest(BaseModel):
    """Refresh token request schema."""
    refresh_token: str


class PasswordChangeRequest(BaseModel):
    """Password change request schema."""
    current_password: str
    new_password: str = Field(..., min_length=8, max_length=100)

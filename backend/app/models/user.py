"""
User Models and Schemas

Pydantic models for user authentication and authorization.
"""

from datetime import datetime
from typing import Optional, List
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


# ==========================================================
# In-Memory User Store (Replace with Database in Production)
# ==========================================================

class InMemoryUserStore:
    """
    Simple in-memory user store for development/testing.
    Replace with SQLAlchemy + PostgreSQL in production.
    """

    def __init__(self):
        self._users: dict[int, UserInDB] = {}
        self._email_index: dict[str, int] = {}
        self._next_id = 1

    def create_user(self, user: UserInDB) -> UserInDB:
        """Create a new user."""
        user.id = self._next_id
        self._next_id += 1
        self._users[user.id] = user
        self._email_index[user.email.lower()] = user.id
        return user

    def get_by_id(self, user_id: int) -> Optional[UserInDB]:
        """Get user by ID."""
        return self._users.get(user_id)

    def get_by_email(self, email: str) -> Optional[UserInDB]:
        """Get user by email."""
        user_id = self._email_index.get(email.lower())
        if user_id:
            return self._users.get(user_id)
        return None

    def update_user(self, user_id: int, updates: dict) -> Optional[UserInDB]:
        """Update user fields."""
        user = self._users.get(user_id)
        if not user:
            return None

        # Handle email change
        if "email" in updates and updates["email"] != user.email:
            old_email = user.email.lower()
            new_email = updates["email"].lower()
            if new_email in self._email_index:
                return None  # Email already exists
            del self._email_index[old_email]
            self._email_index[new_email] = user_id

        # Apply updates
        user_data = user.model_dump()
        user_data.update(updates)
        user_data["updated_at"] = datetime.utcnow()
        updated_user = UserInDB(**user_data)
        self._users[user_id] = updated_user
        return updated_user

    def delete_user(self, user_id: int) -> bool:
        """Delete user."""
        user = self._users.pop(user_id, None)
        if user:
            self._email_index.pop(user.email.lower(), None)
            return True
        return False

    def list_users(self, skip: int = 0, limit: int = 100) -> List[UserInDB]:
        """List users with pagination."""
        users = list(self._users.values())
        return users[skip:skip + limit]

    def create_user_tokens(self, user: UserInDB) -> tuple[str, str]:
        """Create access and refresh tokens for user."""
        from backend.app.utils.auth import create_access_token, create_refresh_token
        access_token = create_access_token(user.id, user.email, user.role)
        refresh_token = create_refresh_token(user.id, user.email, user.role)
        return access_token, refresh_token


# Global user store instance
user_store = InMemoryUserStore()
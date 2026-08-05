"""
Authentication Utilities

Password hashing and JWT token management.
"""

import secrets
from datetime import datetime, timedelta
from typing import Optional, TYPE_CHECKING

from passlib.context import CryptContext
from jose import jwt, JWTError

from backend.app.config import settings
from backend.app.models.user import UserRole, TokenPayload, user_store


# Avoid circular imports
if TYPE_CHECKING:
    from backend.app.models.user import UserInDB


# Password hashing
# Use argon2 which is more secure and doesn't have bcrypt's 72-byte limit
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")


def hash_password(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


# JWT Configuration
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7


def get_secret_key() -> str:
    """Get secret key from settings at runtime."""
    from backend.app.config import settings
    return settings.secret_key


def create_access_token(
    user_id: int,
    email: str,
    role: UserRole,
    expires_delta: Optional[timedelta] = None
) -> str:
    """Create JWT access token."""
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode = {
        "sub": str(user_id),
        "email": email,
        "role": role.value,
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access",
    }
    return jwt.encode(to_encode, get_secret_key(), algorithm=ALGORITHM)


def create_refresh_token(user_id: int, email: str, role: UserRole) -> str:
    """Create JWT refresh token."""
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)

    to_encode = {
        "sub": str(user_id),
        "email": email,
        "role": role.value,
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "refresh",
    }
    return jwt.encode(to_encode, get_secret_key(), algorithm=ALGORITHM)


def decode_token(token: str) -> Optional[TokenPayload]:
    """Decode and validate JWT token."""
    try:
        payload = jwt.decode(token, get_secret_key(), algorithms=[ALGORITHM])
        return TokenPayload(**payload)
    except JWTError:
        return None


def verify_token(token: str, token_type: str = "access") -> Optional[TokenPayload]:
    """Verify token and check type."""
    payload = decode_token(token)
    if not payload:
        return None
    if payload.type != token_type:
        return None
    return payload


# ==========================================================
# Authentication Functions
# ==========================================================

def authenticate_user(email: str, password: str) -> "Optional[UserInDB]":
    """Authenticate user with email and password."""
    from backend.app.models.user import user_store

    user = user_store.get_by_email(email)
    if not user:
        return None
    if not user.is_active:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def create_user_tokens(user) -> tuple[str, str]:
    """Create access and refresh tokens for user."""
    access_token = create_access_token(user.id, user.email, user.role)
    refresh_token = create_refresh_token(user.id, user.email, user.role)
    return access_token, refresh_token


def get_user_from_token(token: str) -> "Optional[UserInDB]":
    """Get user from access token."""
    from backend.app.models.user import user_store

    payload = verify_token(token, "access")
    if not payload:
        return None
    return user_store.get_by_id(payload.sub)


def refresh_access_token(refresh_token: str) -> Optional[str]:
    """Create new access token from refresh token."""
    from backend.app.models.user import user_store

    payload = verify_token(refresh_token, "refresh")
    if not payload:
        return None

    user = user_store.get_by_id(payload.sub)
    if not user or not user.is_active:
        return None

    return create_access_token(user.id, user.email, user.role)


def generate_reset_token(email: str) -> str:
    """Generate password reset token (short-lived)."""
    expire = datetime.utcnow() + timedelta(hours=1)
    to_encode = {
        "email": email,
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "reset",
    }
    return jwt.encode(to_encode, get_secret_key(), algorithm=ALGORITHM)


def verify_reset_token(token: str) -> Optional[str]:
    """Verify password reset token and return email."""
    try:
        payload = jwt.decode(token, get_secret_key(), algorithms=[ALGORITHM])
        if payload.get("type") != "reset":
            return None
        return payload.get("email")
    except JWTError:
        return None


def generate_secure_token(length: int = 32) -> str:
    """Generate cryptographically secure random token."""
    return secrets.token_urlsafe(length)
"""
Authentication Utilities

Password hashing and JWT token management.
Pure utilities — no database access.
"""

import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional, TYPE_CHECKING

from passlib.context import CryptContext
from jose import jwt, JWTError

from backend.app.config import settings
from backend.app.schemas.user import UserRole, TokenPayload


# Avoid circular imports
if TYPE_CHECKING:
    from backend.app.models.user import User


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
    return settings.secret_key


def create_access_token(
    user_id: int,
    email: str,
    role: UserRole,
    expires_delta: Optional[timedelta] = None
) -> str:
    """Create JWT access token."""
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=ACCESS_TOKEN_EXPIRE_MINUTES
        )

    to_encode = {
        "sub": str(user_id),
        "email": email,
        "role": role.value,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "type": "access",
    }
    return jwt.encode(to_encode, get_secret_key(), algorithm=ALGORITHM)


def create_refresh_token(user_id: int, email: str, role: UserRole) -> str:
    """Create JWT refresh token."""
    expire = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)

    to_encode = {
        "sub": str(user_id),
        "email": email,
        "role": role.value,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
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


def generate_reset_token(email: str) -> str:
    """Generate password reset token (short-lived)."""
    expire = datetime.now(timezone.utc) + timedelta(hours=1)
    to_encode = {
        "email": email,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
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

"""
Authentication Dependencies

FastAPI dependencies for authentication and authorization.
"""

from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

from backend.app.utils.auth import (
    get_user_from_token,
    verify_token,
    refresh_access_token,
)
from backend.app.models.user import UserInDB, UserRole, user_store
from backend.app.exceptions import AuthenticationError, AuthorizationError


# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


async def get_current_user(token: str = Depends(oauth2_scheme)) -> UserInDB:
    """
    Get current authenticated user from access token.

    Raises:
        AuthenticationError: If token is invalid or user not found
    """
    user = get_user_from_token(token)
    if not user:
        raise AuthenticationError("Invalid or expired token")
    if not user.is_active:
        raise AuthenticationError("User account is deactivated")
    return user


async def get_current_active_user(
    current_user: UserInDB = Depends(get_current_user)
) -> UserInDB:
    """
    Get current active user (alias for get_current_user).
    """
    if not current_user.is_active:
        raise AuthenticationError("Inactive user")
    return current_user


async def get_optional_user(token: Optional[str] = Depends(oauth2_scheme)) -> Optional[UserInDB]:
    """
    Get current user if token provided, otherwise None.
    Useful for endpoints that work with or without authentication.
    """
    if not token:
        return None
    return get_user_from_token(token)


def require_role(*roles: UserRole):
    """
    Dependency factory for role-based access control.

    Usage:
        @router.get("/admin", dependencies=[Depends(require_role(UserRole.ADMIN))])
    """
    async def role_checker(current_user: UserInDB = Depends(get_current_user)) -> UserInDB:
        if current_user.role not in roles:
            raise AuthorizationError(f"Requires one of roles: {[r.value for r in roles]}")
        return current_user
    return role_checker


# Convenience dependencies
require_admin = require_role(UserRole.ADMIN)
require_researcher = require_role(UserRole.RESEARCHER, UserRole.ADMIN)
require_user = require_role(UserRole.USER, UserRole.RESEARCHER, UserRole.ADMIN)


async def get_refresh_token_user(refresh_token: str) -> UserInDB:
    """
    Get user from refresh token for token refresh endpoint.

    Raises:
        AuthenticationError: If refresh token is invalid
    """
    from backend.app.utils.auth import verify_token

    payload = verify_token(refresh_token, "refresh")
    if not payload:
        raise AuthenticationError("Invalid or expired refresh token")

    user = user_store.get_by_id(payload.sub)
    if not user or not user.is_active:
        raise AuthenticationError("User not found or inactive")

    return user
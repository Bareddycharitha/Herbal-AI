"""
Authentication Dependencies

FastAPI dependencies for authentication and authorization.
Uses Clerk JWT verification for protected endpoints.
"""

from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from backend.app.services.auth_service import AuthService
from backend.app.schemas.user import UserRole, UserResponse
from backend.app.exceptions import AuthenticationError, AuthorizationError
from backend.app.utils.auth import verify_clerk_token


# OAuth2 scheme — Clerk tokens are Bearer tokens
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/me")


def get_auth_service() -> AuthService:
    """Dependency: get AuthService instance."""
    from backend.app.repositories.profile_repository import ProfileRepository

    repository = ProfileRepository()
    return AuthService(repository)


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    auth_service: AuthService = Depends(get_auth_service),
) -> UserResponse:
    """
    Get current authenticated user from Clerk JWT.

    Raises:
        AuthenticationError: If token is invalid or user not found
    """
    payload = verify_clerk_token(token)
    if not payload:
        raise AuthenticationError("Invalid or expired token")

    user = await auth_service.get_user_from_verified_token(payload)
    if not user:
        raise AuthenticationError("Invalid or expired token")
    if not user.get("is_active", True):
        raise AuthenticationError("User account is deactivated")

    return UserResponse.model_validate(user)


async def get_current_active_user(
    current_user: UserResponse = Depends(get_current_user),
) -> UserResponse:
    """
    Get current active user (alias for get_current_user).
    """
    if not current_user.is_active:
        raise AuthenticationError("Inactive user")
    return current_user


async def get_optional_user(
    token: Optional[str] = Depends(oauth2_scheme),
    auth_service: AuthService = Depends(get_auth_service),
) -> Optional[UserResponse]:
    """
    Get current user if token provided, otherwise None.
    Useful for endpoints that work with or without authentication.
    """
    if not token:
        return None

    payload = verify_clerk_token(token)
    if not payload:
        return None

    user = await auth_service.get_user_from_verified_token(payload)
    if not user or not user.get("is_active", True):
        return None

    return UserResponse.model_validate(user)


def require_role(*roles: str):
    """
    Dependency factory for role-based access control.

    Usage:
        @router.get("/admin", dependencies=[Depends(require_role("admin"))])
    """

    async def role_checker(
        current_user: UserResponse = Depends(get_current_user),
    ) -> UserResponse:
        if current_user.role not in roles:
            raise AuthorizationError(
                f"Requires one of roles: {roles}"
            )
        return current_user

    return role_checker


# Convenience dependencies
require_admin = require_role("admin")
require_researcher = require_role("researcher", "admin")
require_user = require_role("user", "researcher", "admin")

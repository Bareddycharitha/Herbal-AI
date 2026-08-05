"""
Authentication Dependencies

FastAPI dependencies for authentication and authorization.
Uses AuthService via dependency injection — no global state.
"""

from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.database.session import get_db_session
from backend.app.repositories.user_repository import UserRepository
from backend.app.services.auth_service import AuthService
from backend.app.schemas.user import UserRole, UserResponse
from backend.app.exceptions import AuthenticationError, AuthorizationError


# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


def get_user_repository(
    db: AsyncSession = Depends(get_db_session),
) -> UserRepository:
    """Dependency: get UserRepository instance."""
    return UserRepository(db)


def get_auth_service(
    repository: UserRepository = Depends(get_user_repository),
) -> AuthService:
    """Dependency: get AuthService instance."""
    return AuthService(repository)


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    auth_service: AuthService = Depends(get_auth_service),
) -> UserResponse:
    """
    Get current authenticated user from access token.

    Raises:
        AuthenticationError: If token is invalid or user not found
    """
    from backend.app.utils.auth import verify_token

    payload = verify_token(token, "access")
    if not payload:
        raise AuthenticationError("Invalid or expired token")

    user = await auth_service.get_user_by_id(payload.sub)
    if not user:
        raise AuthenticationError("Invalid or expired token")
    if not user.is_active:
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

    from backend.app.utils.auth import verify_token
    payload = verify_token(token, "access")
    if not payload:
        return None

    user = await auth_service.get_user_by_id(payload.sub)
    if not user or not user.is_active:
        return None

    return UserResponse.model_validate(user)


def require_role(*roles: UserRole):
    """
    Dependency factory for role-based access control.

    Usage:
        @router.get("/admin", dependencies=[Depends(require_role(UserRole.ADMIN))])
    """
    async def role_checker(
        current_user: UserResponse = Depends(get_current_user),
    ) -> UserResponse:
        if current_user.role not in [r.value for r in roles]:
            raise AuthorizationError(
                f"Requires one of roles: {[r.value for r in roles]}"
            )
        return current_user
    return role_checker


# Convenience dependencies
require_admin = require_role(UserRole.ADMIN)
require_researcher = require_role(UserRole.RESEARCHER, UserRole.ADMIN)
require_user = require_role(UserRole.USER, UserRole.RESEARCHER, UserRole.ADMIN)


async def get_refresh_token_user(
    refresh_token: str,
    auth_service: AuthService = Depends(get_auth_service),
) -> UserResponse:
    """
    Get user from refresh token for token refresh endpoint.

    Raises:
        AuthenticationError: If refresh token is invalid
    """
    from backend.app.utils.auth import verify_token

    payload = verify_token(refresh_token, "refresh")
    if not payload:
        raise AuthenticationError("Invalid or expired refresh token")

    user = await auth_service.get_user_by_id(payload.sub)
    if not user or not user.is_active:
        raise AuthenticationError("User not found or inactive")

    return UserResponse.model_validate(user)

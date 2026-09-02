"""
Authentication Dependencies

FastAPI dependencies for authentication and authorization.
Uses Clerk JWT verification for protected endpoints.
"""

from typing import Optional

from fastapi import Depends, Request, status
from fastapi.security import OAuth2PasswordBearer

from backend.app.services.auth_service import AuthService
from backend.app.schemas.user import UserRole, UserResponse
from backend.app.exceptions import (
    AuthenticationError,
    AuthorizationError,
    ServiceUnavailableError,
)
from backend.app.utils.auth import verify_clerk_token, TokenError
from backend.app.utils.logging import get_logger


logger = get_logger(__name__)


# OAuth2 scheme — Clerk tokens are Bearer tokens.
# ``auto_error=False`` so the dependency function below can produce
# its own diagnostic log when the Authorization header is missing,
# instead of FastAPI silently returning a generic 401.
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/api/v1/auth/me",
    auto_error=False,
)


def get_auth_service() -> AuthService:
    """Dependency: get AuthService instance."""
    from backend.app.repositories.profile_repository import ProfileRepository

    repository = ProfileRepository()
    return AuthService(repository)


async def get_current_user(
    request: Request,
    token: Optional[str] = Depends(oauth2_scheme),
    auth_service: AuthService = Depends(get_auth_service),
) -> UserResponse:
    """
    Get current authenticated user from Clerk JWT.

    Raises:
        AuthenticationError: If the token is missing, invalid, the
            profile cannot be resolved, or the user is deactivated.
    """
    # Step 0: token-present check. With ``auto_error=False`` on the
    # OAuth2 scheme, FastAPI will NOT reject a missing Authorization
    # header for us — we have to do it ourselves. Logging this case
    # separately from "verification failed" lets us tell the two
    # failure modes apart in the backend log.
    if not token:
        auth_header = request.headers.get("Authorization")
        logger.warning(
            "Auth rejected: no Bearer token in Authorization header",
            has_auth_header=bool(auth_header),
            auth_header_prefix=auth_header.split(" ")[0] if auth_header else None,
        )
        raise AuthenticationError("Invalid or expired token")

    # Step 1: verify the Clerk JWT signature + standard claims.
    # ``verify_clerk_token`` raises ``TokenError`` with a specific
    # ``reason`` code (e.g. ``token_expired``, ``no_matching_jwks_key``).
    # We catch it to surface the cause in the 401 response body, so the
    # frontend's console log can tell the user (or operator) exactly
    # why the token was rejected, instead of a generic "Invalid".
    try:
        payload = verify_clerk_token(token)
    except TokenError as exc:
        logger.warning(
            "Auth rejected: Clerk token verification failed",
            reason=exc.reason,
            detail=exc.message,
        )
        raise AuthenticationError(
            "Invalid or expired token",
            details={
                "reason": exc.reason,
                "detail": exc.message,
            },
        ) from exc

    if not payload:
        # Defensive: ``verify_clerk_token`` should always raise rather
        # than return None, but if a future change introduces a None
        # return we still want a clean 401 instead of a crash.
        logger.warning(
            "Auth rejected: Clerk token verification returned no payload",
            token_prefix=token[:20] + "..." if len(token) > 20 else token,
        )
        raise AuthenticationError("Invalid or expired token")

    # Step 2: resolve the application profile for the verified identity.
    # A failure here is a *different* problem from a bad token: the JWT
    # is valid but the profile lookup or auto-create did not produce a
    # user. Distinguishing these in the log makes it obvious whether to
    # look at Clerk or at Supabase when a 401 comes back.
    try:
        user = await auth_service.get_user_from_verified_token(payload)
    except Exception as exc:
        logger.error(
            "Auth rejected: profile lookup raised",
            user_id=payload.get("id"),
            error=str(exc),
            error_type=type(exc).__name__,
        )
        # Surface as 503, not 401 — auth itself was fine, the database
        # is the problem. The frontend toast maps 503 to a clearer
        # 'try again later' message.
        raise ServiceUnavailableError(
            "Profile service is temporarily unavailable"
        ) from exc

    if not user:
        logger.warning(
            "Auth rejected: no profile for verified token",
            user_id=payload.get("id"),
        )
        raise AuthenticationError("Invalid or expired token")
    if not user.get("is_active", True):
        logger.warning(
            "Auth rejected: user account deactivated",
            user_id=payload.get("id"),
        )
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

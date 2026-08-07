"""
Authentication API Endpoints

User profile management using Clerk Authentication.
Clerk handles signup, login, logout, email verification,
password reset, Google OAuth, session validation, and JWT
issuance. The backend verifies Clerk JWTs and manages
the application profile in Supabase PostgreSQL.
"""

from fastapi import APIRouter, Depends

from backend.app.schemas.user import UserResponse, UserUpdate, UserRole
from backend.app.services.auth_service import AuthService
from backend.app.dependencies import (
    get_auth_service,
    get_current_active_user,
)
from backend.app.exceptions import (
    ValidationError,
    AuthorizationError,
    NotFoundError,
)

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"]
)


# ==========================================================
# Current User Profile
# ==========================================================

@router.get("/me", response_model=UserResponse)
async def get_current_user_profile(
    current_user: UserResponse = Depends(get_current_active_user),
):
    """Get the current authenticated user's profile."""
    return current_user


@router.patch("/me", response_model=UserResponse)
async def update_current_user(
    user_update: UserUpdate,
    auth_service: AuthService = Depends(get_auth_service),
    current_user: UserResponse = Depends(get_current_active_user),
):
    """
    Update current user's profile.

    - **full_name**: Update full name
    - **email**: Update email (must be unique)
    """
    updates = user_update.model_dump(exclude_unset=True)

    # Prevent role escalation for non-admins
    if "role" in updates and current_user.role != "admin":
        raise AuthorizationError("Cannot change own role")

    # Convert role enum to string if needed
    if "role" in updates and isinstance(updates["role"], UserRole):
        updates["role"] = updates["role"].value

    updated_user = await auth_service.update_user(current_user.id, updates)
    if not updated_user:
        raise ValidationError("Update failed")

    return UserResponse.model_validate(updated_user)


# ==========================================================
# Admin Endpoints
# ==========================================================

@router.get("/users", response_model=list[UserResponse])
async def list_users(
    skip: int = 0,
    limit: int = 100,
    auth_service: AuthService = Depends(get_auth_service),
    current_user: UserResponse = Depends(get_current_active_user),
):
    """List all users (admin only)."""
    if current_user.role != "admin":
        raise AuthorizationError("Admin access required")

    users = await auth_service.list_users(skip=skip, limit=limit)
    return [UserResponse.model_validate(u) for u in users]


@router.get("/users/{clerk_user_id}", response_model=UserResponse)
async def get_user(
    clerk_user_id: str,
    auth_service: AuthService = Depends(get_auth_service),
    current_user: UserResponse = Depends(get_current_active_user),
):
    """Get user by Clerk user ID (admin only)."""
    if current_user.role != "admin":
        raise AuthorizationError("Admin access required")

    user = await auth_service.get_user_by_clerk_id(clerk_user_id)
    if not user:
        raise NotFoundError("User", clerk_user_id)

    return UserResponse.model_validate(user)


@router.patch("/users/{clerk_user_id}", response_model=UserResponse)
async def update_user(
    clerk_user_id: str,
    user_update: UserUpdate,
    auth_service: AuthService = Depends(get_auth_service),
    current_user: UserResponse = Depends(get_current_active_user),
):
    """Update user by Clerk user ID (admin only)."""
    if current_user.role != "admin":
        raise AuthorizationError("Admin access required")

    updates = user_update.model_dump(exclude_unset=True)

    # Convert role enum to string if needed
    if "role" in updates and isinstance(updates["role"], UserRole):
        updates["role"] = updates["role"].value

    updated_user = await auth_service.update_user(clerk_user_id, updates)
    if not updated_user:
        raise NotFoundError("User", clerk_user_id)

    return UserResponse.model_validate(updated_user)


@router.delete("/users/{clerk_user_id}", status_code=204)
async def delete_user(
    clerk_user_id: str,
    auth_service: AuthService = Depends(get_auth_service),
    current_user: UserResponse = Depends(get_current_active_user),
):
    """Delete user by Clerk user ID (admin only)."""
    if current_user.role != "admin":
        raise AuthorizationError("Admin access required")

    if clerk_user_id == current_user.id:
        raise ValidationError("Cannot delete yourself")

    success = await auth_service.delete_user(clerk_user_id)
    if not success:
        raise NotFoundError("User", clerk_user_id)


# ==========================================================
# Token Validation (for internal use)
# ==========================================================

@router.post("/validate")
async def validate_token(
    token: str,
    auth_service: AuthService = Depends(get_auth_service),
):
    """
    Validate a Clerk JWT and return user info.
    Useful for API gateways or microservices.
    """
    user = await auth_service.get_user_from_token(token)
    if not user:
        from backend.app.exceptions import AuthenticationError
        raise AuthenticationError("Invalid token")

    return {
        "valid": True,
        "user_id": user.get("id"),
        "email": user.get("email"),
        "role": user.get("role"),
    }

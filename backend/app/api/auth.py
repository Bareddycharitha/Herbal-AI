"""
Authentication API Endpoints

User registration, login, token refresh, and profile management.
"""

from datetime import timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status, Response
from fastapi.security import OAuth2PasswordRequestForm

from backend.app.config import settings
from backend.app.models.user import (
    UserCreate,
    UserResponse,
    UserUpdate,
    Token,
    LoginRequest,
    RefreshTokenRequest,
    PasswordChangeRequest,
    UserRole,
    user_store,
)
from backend.app.utils.auth import (
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
    authenticate_user,
    create_user_tokens,
    get_user_from_token,
    refresh_access_token,
    ACCESS_TOKEN_EXPIRE_MINUTES,
)
from backend.app.dependencies import get_current_user, get_current_active_user
from backend.app.exceptions import (
    AuthenticationError,
    ValidationError,
    ConfigurationError,
)

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"]
)


# ==========================================================
# User Registration
# ==========================================================

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserCreate):
    """
    Register a new user.

    - **email**: Valid email address (unique)
    - **password**: Minimum 8 characters
    - **full_name**: Optional full name
    - **role**: Optional role (default: user)
    """
    # Check if email already exists
    if user_store.get_by_email(user_data.email):
        raise ValidationError(
            message="Email already registered",
            field="email",
        )

    # Hash password
    hashed_password = hash_password(user_data.password)

    # Create user
    from backend.app.models.user import UserInDB
    from datetime import datetime

    now = datetime.utcnow()
    user = UserInDB(
        email=user_data.email,
        full_name=user_data.full_name,
        role=user_data.role,
        hashed_password=hashed_password,
        is_active=True,
        created_at=now,
        updated_at=now,
    )

    created_user = user_store.create_user(user)

    return UserResponse.model_validate(created_user)


# ==========================================================
# User Login
# ==========================================================

@router.post("/login", response_model=Token)
async def login(
    response: Response,
    form_data: OAuth2PasswordRequestForm = Depends(),
):
    """
    Login with email and password.

    Returns access token and refresh token.
    Sets access token in HttpOnly cookie for browser clients.
    """
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise AuthenticationError("Incorrect email or password")

    access_token, refresh_token = create_user_tokens(user)

    # Set cookies for browser clients
    if hasattr(settings, 'cookie_secure') and settings.cookie_secure:
        response.set_cookie(
            key="access_token",
            value=access_token,
            httponly=True,
            secure=True,
            samesite="lax",
            max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        )
        response.set_cookie(
            key="refresh_token",
            value=refresh_token,
            httponly=True,
            secure=True,
            samesite="lax",
            max_age=7 * 24 * 60 * 60,  # 7 days
        )

    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


# Alternative JSON-based login
@router.post("/login/json", response_model=Token)
async def login_json(response: Response, credentials: LoginRequest):
    """Login with JSON body (alternative to form data)."""
    user = authenticate_user(credentials.email, credentials.password)
    if not user:
        raise AuthenticationError("Incorrect email or password")

    access_token, refresh_token = create_user_tokens(user)

    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


# ==========================================================
# Token Refresh
# ==========================================================

@router.post("/refresh", response_model=Token)
async def refresh_token(response: Response, token_data: RefreshTokenRequest):
    """
    Refresh access token using refresh token.
    """
    new_access_token = refresh_access_token(token_data.refresh_token)
    if not new_access_token:
        raise AuthenticationError("Invalid or expired refresh token")

    # Get user for new refresh token rotation
    from backend.app.utils.auth import verify_token
    payload = verify_token(token_data.refresh_token, "refresh")
    if payload:
        from backend.app.models.user import user_store
        user = user_store.get_by_id(payload.sub)
        if user:
            new_refresh_token = create_refresh_token(user.id, user.email, user.role)
        else:
            new_refresh_token = token_data.refresh_token
    else:
        new_refresh_token = token_data.refresh_token

    return Token(
        access_token=new_access_token,
        refresh_token=new_refresh_token,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


# ==========================================================
# User Profile
# ==========================================================

@router.get("/me", response_model=UserResponse)
async def get_current_user_profile(current_user = Depends(get_current_active_user)):
    """Get current user's profile."""
    return UserResponse.model_validate(current_user)


@router.patch("/me", response_model=UserResponse)
async def update_current_user(
    user_update: UserUpdate,
    current_user = Depends(get_current_active_user),
):
    """
    Update current user's profile.

    - **full_name**: Update full name
    - **email**: Update email (must be unique)
    """
    updates = user_update.model_dump(exclude_unset=True)

    # Prevent role escalation
    if "role" in updates and current_user.role != "admin":
        from backend.app.exceptions import AuthorizationError
        raise AuthorizationError("Cannot change own role")

    updated_user = user_store.update_user(current_user.id, updates)
    if not updated_user:
        raise ValidationError("Email already in use")

    return UserResponse.model_validate(updated_user)


@router.delete("/me", status_code=status.HTTP_204_NO_CONTENT)
async def delete_current_user(current_user = Depends(get_current_active_user)):
    """Delete current user's account."""
    user_store.delete_user(current_user.id)


# ==========================================================
# Password Management
# ==========================================================

@router.post("/change-password")
async def change_password(
    password_data: PasswordChangeRequest,
    current_user = Depends(get_current_active_user),
):
    """
    Change current user's password.

    - **current_password**: Current password for verification
    - **new_password**: New password (min 8 characters)
    """
    if not verify_password(password_data.current_password, current_user.hashed_password):
        raise AuthenticationError("Current password is incorrect")

    from backend.app.utils.auth import hash_password
    from backend.app.models.user import user_store
    from datetime import datetime

    new_hashed = hash_password(password_data.new_password)
    user_store.update_user(current_user.id, {
        "hashed_password": new_hashed,
        "updated_at": datetime.utcnow(),
    })

    return {"message": "Password changed successfully"}


# ==========================================================
# Admin Endpoints
# ==========================================================

@router.get("/users", response_model=list[UserResponse])
async def list_users(
    skip: int = 0,
    limit: int = 100,
    current_user = Depends(get_current_user),
):
    """
    List all users (admin only).
    """
    if current_user.role != "admin":
        from backend.app.exceptions import AuthorizationError
        raise AuthorizationError("Admin access required")

    users = user_store.list_users(skip=skip, limit=limit)
    return [UserResponse.model_validate(u) for u in users]


@router.get("/users/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: int,
    current_user = Depends(get_current_user),
):
    """
    Get user by ID (admin only).
    """
    if current_user.role != "admin":
        from backend.app.exceptions import AuthorizationError
        raise AuthorizationError("Admin access required")

    user = user_store.get_by_id(user_id)
    if not user:
        from backend.app.exceptions import NotFoundError
        raise NotFoundError("User", str(user_id))

    return UserResponse.model_validate(user)


@router.patch("/users/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: int,
    user_update: UserUpdate,
    current_user = Depends(get_current_user),
):
    """
    Update user by ID (admin only).
    """
    if current_user.role != "admin":
        from backend.app.exceptions import AuthorizationError
        raise AuthorizationError("Admin access required")

    updates = user_update.model_dump(exclude_unset=True)
    updated_user = user_store.update_user(user_id, updates)

    if not updated_user:
        from backend.app.exceptions import NotFoundError
        raise NotFoundError("User", str(user_id))

    return UserResponse.model_validate(updated_user)


@router.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id: int,
    current_user = Depends(get_current_user),
):
    """
    Delete user by ID (admin only).
    """
    if current_user.role != "admin":
        from backend.app.exceptions import AuthorizationError
        raise AuthorizationError("Admin access required")

    if user_id == current_user.id:
        raise ValidationError("Cannot delete yourself")

    if not user_store.delete_user(user_id):
        from backend.app.exceptions import NotFoundError
        raise NotFoundError("User", str(user_id))


# ==========================================================
# Token Validation (for internal use)
# ==========================================================

@router.post("/validate")
async def validate_token(token: str):
    """
    Validate access token and return user info.
    Useful for API gateways or microservices.
    """
    user = get_user_from_token(token)
    if not user:
        raise AuthenticationError("Invalid token")

    return {
        "valid": True,
        "user_id": user.id,
        "email": user.email,
        "role": user.role.value,
    }
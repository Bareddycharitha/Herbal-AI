"""
Authentication API Endpoints

User registration, login, token refresh, and profile management.
Thin controllers — all business logic delegated to AuthService.
"""

from fastapi import APIRouter, Depends, Response
from fastapi.security import OAuth2PasswordRequestForm

from backend.app.config import settings
from backend.app.schemas.user import (
    UserCreate,
    UserResponse,
    UserUpdate,
    Token,
    LoginRequest,
    RefreshTokenRequest,
    PasswordChangeRequest,
    UserRole,
)
from backend.app.services.auth_service import AuthService
from backend.app.dependencies import (
    get_auth_service,
    get_current_user,
    get_current_active_user,
)
from backend.app.exceptions import (
    AuthenticationError,
    ValidationError,
    AuthorizationError,
    NotFoundError,
)
from backend.app.utils.auth import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
)

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"]
)


# ==========================================================
# User Registration
# ==========================================================

@router.post("/register", response_model=UserResponse, status_code=201)
async def register(
    user_data: UserCreate,
    auth_service: AuthService = Depends(get_auth_service),
):
    """
    Register a new user.

    - **email**: Valid email address (unique)
    - **password**: Minimum 8 characters
    - **full_name**: Optional full name
    - **role**: Optional role (default: user)
    """
    created_user = await auth_service.register_user(user_data)
    return UserResponse.model_validate(created_user)


# ==========================================================
# User Login
# ==========================================================

@router.post("/login", response_model=Token)
async def login(
    response: Response,
    form_data: OAuth2PasswordRequestForm = Depends(),
    auth_service: AuthService = Depends(get_auth_service),
):
    """
    Login with email and password.

    Returns access token and refresh token.
    Sets access token in HttpOnly cookie for browser clients.
    """
    user = await auth_service.authenticate_user(
        form_data.username, form_data.password
    )
    if not user:
        raise AuthenticationError("Incorrect email or password")

    access_token, refresh_token = auth_service.create_tokens(user)

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
async def login_json(
    response: Response,
    credentials: LoginRequest,
    auth_service: AuthService = Depends(get_auth_service),
):
    """Login with JSON body (alternative to form data)."""
    user = await auth_service.authenticate_user(
        credentials.email, credentials.password
    )
    if not user:
        raise AuthenticationError("Incorrect email or password")

    access_token, refresh_token = auth_service.create_tokens(user)

    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


# ==========================================================
# Token Refresh
# ==========================================================

@router.post("/refresh", response_model=Token)
async def refresh_token(
    response: Response,
    token_data: RefreshTokenRequest,
    auth_service: AuthService = Depends(get_auth_service),
):
    """
    Refresh access token using refresh token.
    """
    result = await auth_service.refresh_tokens(token_data.refresh_token)
    if not result:
        raise AuthenticationError("Invalid or expired refresh token")

    new_access_token, new_refresh_token = result

    return Token(
        access_token=new_access_token,
        refresh_token=new_refresh_token,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


# ==========================================================
# User Profile
# ==========================================================

@router.get("/me", response_model=UserResponse)
async def get_current_user_profile(
    current_user: UserResponse = Depends(get_current_active_user),
):
    """Get current user's profile."""
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

    # Prevent role escalation
    if "role" in updates and current_user.role != "admin":
        raise AuthorizationError("Cannot change own role")

    # Convert role enum to string if needed
    if "role" in updates and isinstance(updates["role"], UserRole):
        updates["role"] = updates["role"].value

    updated_user = await auth_service.update_user(current_user.id, updates)
    if not updated_user:
        raise ValidationError("Email already in use")

    return UserResponse.model_validate(updated_user)


@router.delete("/me", status_code=204)
async def delete_current_user(
    auth_service: AuthService = Depends(get_auth_service),
    current_user: UserResponse = Depends(get_current_active_user),
):
    """Delete current user's account."""
    await auth_service.delete_user(current_user.id)


# ==========================================================
# Password Management
# ==========================================================

@router.post("/change-password")
async def change_password(
    password_data: PasswordChangeRequest,
    auth_service: AuthService = Depends(get_auth_service),
    current_user: UserResponse = Depends(get_current_active_user),
):
    """
    Change current user's password.

    - **current_password**: Current password for verification
    - **new_password**: New password (min 8 characters)
    """
    await auth_service.change_password(
        current_user.id,
        password_data.current_password,
        password_data.new_password,
    )

    return {"message": "Password changed successfully"}


# ==========================================================
# Admin Endpoints
# ==========================================================

@router.get("/users", response_model=list[UserResponse])
async def list_users(
    skip: int = 0,
    limit: int = 100,
    auth_service: AuthService = Depends(get_auth_service),
    current_user: UserResponse = Depends(get_current_user),
):
    """
    List all users (admin only).
    """
    if current_user.role != "admin":
        raise AuthorizationError("Admin access required")

    users = await auth_service.list_users(skip=skip, limit=limit)
    return [UserResponse.model_validate(u) for u in users]


@router.get("/users/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: int,
    auth_service: AuthService = Depends(get_auth_service),
    current_user: UserResponse = Depends(get_current_user),
):
    """
    Get user by ID (admin only).
    """
    if current_user.role != "admin":
        raise AuthorizationError("Admin access required")

    user = await auth_service.get_user_by_id(user_id)
    if not user:
        raise NotFoundError("User", str(user_id))

    return UserResponse.model_validate(user)


@router.patch("/users/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: int,
    user_update: UserUpdate,
    auth_service: AuthService = Depends(get_auth_service),
    current_user: UserResponse = Depends(get_current_user),
):
    """
    Update user by ID (admin only).
    """
    if current_user.role != "admin":
        raise AuthorizationError("Admin access required")

    updates = user_update.model_dump(exclude_unset=True)

    # Convert role enum to string if needed
    if "role" in updates and isinstance(updates["role"], UserRole):
        updates["role"] = updates["role"].value

    updated_user = await auth_service.update_user(user_id, updates)

    if not updated_user:
        raise NotFoundError("User", str(user_id))

    return UserResponse.model_validate(updated_user)


@router.delete("/users/{user_id}", status_code=204)
async def delete_user(
    user_id: int,
    auth_service: AuthService = Depends(get_auth_service),
    current_user: UserResponse = Depends(get_current_user),
):
    """
    Delete user by ID (admin only).
    """
    if current_user.role != "admin":
        raise AuthorizationError("Admin access required")

    if user_id == current_user.id:
        raise ValidationError("Cannot delete yourself")

    success = await auth_service.delete_user(user_id)
    if not success:
        raise NotFoundError("User", str(user_id))


# ==========================================================
# Token Validation (for internal use)
# ==========================================================

@router.post("/validate")
async def validate_token(
    token: str,
    auth_service: AuthService = Depends(get_auth_service),
):
    """
    Validate access token and return user info.
    Useful for API gateways or microservices.
    """
    from backend.app.utils.auth import verify_token
    payload = verify_token(token, "access")
    if not payload:
        raise AuthenticationError("Invalid token")

    user = await auth_service.get_user_by_id(payload.sub)
    if not user:
        raise AuthenticationError("Invalid token")

    return {
        "valid": True,
        "user_id": user.id,
        "email": user.email,
        "role": user.role.value,
    }

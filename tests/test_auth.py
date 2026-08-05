"""
Tests for Authentication API Endpoints
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from datetime import datetime, timedelta

from backend.app.models.user import (
    UserCreate,
    UserInDB,
    UserRole,
    Token,
    user_store,
)
from backend.app.utils.auth import (
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
    verify_token,
    authenticate_user,
    user_store as auth_user_store,
)


class TestAuthUtils:
    """Tests for authentication utilities."""

    def test_hash_password(self):
        """Test password hashing."""
        password = "testpassword123"
        hashed = hash_password(password)
        assert hashed != password
        assert hashed.startswith("$argon2")  # argon2

    def test_verify_password(self):
        """Test password verification."""
        password = "testpassword123"
        hashed = hash_password(password)
        assert verify_password(password, hashed) is True
        assert verify_password("wrongpassword", hashed) is False

    def test_create_access_token(self):
        """Test access token creation."""
        token = create_access_token(1, "test@example.com", UserRole.USER)
        assert isinstance(token, str)
        assert len(token) > 0

        # Verify token can be decoded
        payload = verify_token(token, "access")
        assert payload is not None
        assert payload.sub == 1
        assert payload.email == "test@example.com"
        assert payload.role == UserRole.USER
        assert payload.type == "access"

    def test_create_refresh_token(self):
        """Test refresh token creation."""
        token = create_refresh_token(1, "test@example.com", UserRole.USER)
        assert isinstance(token, str)

        payload = verify_token(token, "refresh")
        assert payload is not None
        assert payload.type == "refresh"

    def test_verify_token_invalid(self):
        """Test invalid token verification."""
        assert verify_token("invalid.token", "access") is None
        assert verify_token("invalid.token", "refresh") is None


class TestUserStore:
    """Tests for in-memory user store."""

    def setup_method(self):
        """Clear user store before each test."""
        auth_user_store._users.clear()
        auth_user_store._email_index.clear()
        auth_user_store._next_id = 1

    def _create_test_user(self, email="test@example.com", **kwargs):
        """Helper to create a test user via the store."""
        user = UserInDB(
            email=email,
            full_name=kwargs.get("full_name", "Test User"),
            role=kwargs.get("role", UserRole.USER),
            hashed_password=hash_password(kwargs.get("password", "password123")),
            is_active=kwargs.get("is_active", True),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        return auth_user_store.create_user(user)

    def test_create_user(self):
        """Test user creation."""
        user = self._create_test_user()
        assert user.id == 1
        assert user.email == "test@example.com"

    def test_get_by_email(self):
        """Test getting user by email."""
        self._create_test_user()

        found = auth_user_store.get_by_email("test@example.com")
        assert found is not None
        assert found.email == "test@example.com"

        not_found = auth_user_store.get_by_email("nonexistent@example.com")
        assert not_found is None

    def test_get_by_id(self):
        """Test getting user by ID."""
        user = self._create_test_user()

        found = auth_user_store.get_by_id(user.id)
        assert found is not None
        assert found.id == user.id

        not_found = auth_user_store.get_by_id(999)
        assert not_found is None

    def test_update_user(self):
        """Test user update."""
        user = self._create_test_user()

        updated = auth_user_store.update_user(user.id, {"full_name": "Updated Name"})
        assert updated is not None
        assert updated.full_name == "Updated Name"

    def test_delete_user(self):
        """Test user deletion."""
        user = self._create_test_user()

        deleted = auth_user_store.delete_user(user.id)
        assert deleted is True

        not_found = auth_user_store.get_by_id(user.id)
        assert not_found is None

    def test_duplicate_email(self):
        """Test duplicate email handling."""
        self._create_test_user(email="test@example.com", full_name="User 1")
        self._create_test_user(email="test@example.com", full_name="User 2")

        # Only one user should exist with that email
        found = auth_user_store.get_by_email("test@example.com")
        assert found is not None


class TestAuthenticateUser:
    """Tests for user authentication."""

    def setup_method(self):
        """Clear user store before each test."""
        auth_user_store._users.clear()
        auth_user_store._email_index.clear()
        auth_user_store._next_id = 1

    def _create_test_user(self, email="test@example.com", **kwargs):
        """Helper to create a test user via the store."""
        user = UserInDB(
            email=email,
            full_name=kwargs.get("full_name", "Test User"),
            role=kwargs.get("role", UserRole.USER),
            hashed_password=hash_password(kwargs.get("password", "password123")),
            is_active=kwargs.get("is_active", True),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        return auth_user_store.create_user(user)

    def test_authenticate_valid_user(self):
        """Test valid user authentication."""
        password = "password123"
        self._create_test_user(password=password)

        authenticated = authenticate_user("test@example.com", password)
        assert authenticated is not None
        assert authenticated.email == "test@example.com"

    def test_authenticate_wrong_password(self):
        """Test authentication with wrong password."""
        self._create_test_user()

        authenticated = authenticate_user("test@example.com", "wrongpassword")
        assert authenticated is None

    def test_authenticate_nonexistent_user(self):
        """Test authentication for nonexistent user."""
        authenticated = authenticate_user("nonexistent@example.com", "password123")
        assert authenticated is None

    def test_authenticate_inactive_user(self):
        """Test authentication for inactive user."""
        self._create_test_user(is_active=False)

        authenticated = authenticate_user("test@example.com", "password123")
        assert authenticated is None


class TestTokenCreation:
    """Tests for token creation functions."""

    def setup_method(self):
        """Clear user store before each test."""
        auth_user_store._users.clear()
        auth_user_store._email_index.clear()
        auth_user_store._next_id = 1

    def _create_test_user(self, email="test@example.com", **kwargs):
        """Helper to create a test user via the store."""
        user = UserInDB(
            email=email,
            full_name=kwargs.get("full_name", "Test User"),
            role=kwargs.get("role", UserRole.USER),
            hashed_password=hash_password(kwargs.get("password", "password123")),
            is_active=kwargs.get("is_active", True),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        return auth_user_store.create_user(user)

    def test_create_user_tokens(self):
        """Test token creation for user."""
        user = self._create_test_user()

        access_token, refresh_token = auth_user_store.create_user_tokens(user)

        assert isinstance(access_token, str)
        assert isinstance(refresh_token, str)

        # Verify tokens
        access_payload = verify_token(access_token, "access")
        refresh_payload = verify_token(refresh_token, "refresh")

        assert access_payload.sub == user.id
        assert refresh_payload.sub == user.id
        assert access_payload.type == "access"
        assert refresh_payload.type == "refresh"


# ==========================================================
# API Endpoint Tests
# ==========================================================

class TestAuthEndpoints:
    """Tests for authentication API endpoints."""

    @pytest.fixture
    def test_client(self):
        """Create FastAPI test client."""
        from backend.app.main import app
        with TestClient(app) as client:
            yield client

    @pytest.fixture
    def test_user(self):
        """Create test user."""
        auth_user_store._users.clear()
        auth_user_store._email_index.clear()
        auth_user_store._next_id = 1

        user = UserInDB(
            email="test@example.com",
            full_name="Test User",
            role=UserRole.USER,
            hashed_password=hash_password("password123"),
            is_active=True,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        return auth_user_store.create_user(user)

    def test_register_success(self, test_client):
        """Test successful user registration."""
        response = test_client.post(
            "/api/v1/auth/register",
            json={
                "email": "newuser@example.com",
                "password": "password123",
                "full_name": "New User",
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["email"] == "newuser@example.com"
        assert data["full_name"] == "New User"
        assert "id" in data

    def test_register_duplicate_email(self, test_client, test_user):
        """Test registration with duplicate email."""
        response = test_client.post(
            "/api/v1/auth/register",
            json={
                "email": "test@example.com",
                "password": "password123",
            },
        )
        assert response.status_code == 400
        data = response.json()
        assert data["error"]["code"] == "VALIDATION_ERROR"

    def test_register_invalid_email(self, test_client):
        """Test registration with invalid email."""
        response = test_client.post(
            "/api/v1/auth/register",
            json={
                "email": "invalid-email",
                "password": "password123",
            },
        )
        assert response.status_code == 422

    def test_register_short_password(self, test_client):
        """Test registration with short password."""
        response = test_client.post(
            "/api/v1/auth/register",
            json={
                "email": "newuser@example.com",
                "password": "short",
            },
        )
        assert response.status_code == 422

    def test_login_success(self, test_client, test_user):
        """Test successful login."""
        response = test_client.post(
            "/api/v1/auth/login",
            data={"username": "test@example.com", "password": "password123"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"

    def test_login_invalid_credentials(self, test_client):
        """Test login with invalid credentials."""
        response = test_client.post(
            "/api/v1/auth/login",
            data={"username": "test@example.com", "password": "wrongpassword"},
        )
        assert response.status_code == 401

    def test_login_nonexistent_user(self, test_client):
        """Test login for nonexistent user."""
        response = test_client.post(
            "/api/v1/auth/login",
            data={"username": "nonexistent@example.com", "password": "password123"},
        )
        assert response.status_code == 401

    def test_login_json(self, test_client, test_user):
        """Test JSON-based login."""
        response = test_client.post(
            "/api/v1/auth/login/json",
            json={"email": "test@example.com", "password": "password123"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data

    def test_refresh_token(self, test_client, test_user):
        """Test token refresh."""
        # First login to get refresh token
        login_response = test_client.post(
            "/api/v1/auth/login",
            data={"username": "test@example.com", "password": "password123"},
        )
        refresh_token = login_response.json()["refresh_token"]

        # Refresh token
        response = test_client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": refresh_token},
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data

    def test_refresh_invalid_token(self, test_client):
        """Test refresh with invalid token."""
        response = test_client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": "invalid.token"},
        )
        assert response.status_code == 401

    def test_get_profile(self, test_client, test_user):
        """Test getting user profile."""
        # Login first
        login_response = test_client.post(
            "/api/v1/auth/login",
            data={"username": "test@example.com", "password": "password123"},
        )
        access_token = login_response.json()["access_token"]

        # Get profile
        response = test_client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["email"] == "test@example.com"
        assert data["full_name"] == "Test User"

    def test_get_profile_unauthorized(self, test_client):
        """Test getting profile without token."""
        response = test_client.get("/api/v1/auth/me")
        assert response.status_code == 401

    def test_update_profile(self, test_client, test_user):
        """Test updating user profile."""
        login_response = test_client.post(
            "/api/v1/auth/login",
            data={"username": "test@example.com", "password": "password123"},
        )
        access_token = login_response.json()["access_token"]

        response = test_client.patch(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {access_token}"},
            json={"full_name": "Updated Name"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["full_name"] == "Updated Name"

    def test_change_password(self, test_client, test_user):
        """Test password change."""
        login_response = test_client.post(
            "/api/v1/auth/login",
            data={"username": "test@example.com", "password": "password123"},
        )
        access_token = login_response.json()["access_token"]

        response = test_client.post(
            "/api/v1/auth/change-password",
            headers={"Authorization": f"Bearer {access_token}"},
            json={
                "current_password": "password123",
                "new_password": "newpassword123",
            },
        )
        assert response.status_code == 200

        # Verify new password works
        login_response = test_client.post(
            "/api/v1/auth/login",
            data={"username": "test@example.com", "password": "newpassword123"},
        )
        assert login_response.status_code == 200

    def test_change_password_wrong_current(self, test_client, test_user):
        """Test password change with wrong current password."""
        login_response = test_client.post(
            "/api/v1/auth/login",
            data={"username": "test@example.com", "password": "password123"},
        )
        access_token = login_response.json()["access_token"]

        response = test_client.post(
            "/api/v1/auth/change-password",
            headers={"Authorization": f"Bearer {access_token}"},
            json={
                "current_password": "wrongpassword",
                "new_password": "newpassword123",
            },
        )
        assert response.status_code == 401
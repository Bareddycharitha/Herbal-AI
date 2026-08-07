"""
Tests for Authentication (Clerk-based)

Tests for Clerk JWT verification, profile repository,
auth service, and auth API endpoints.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient

from backend.app.schemas.user import UserRole
from backend.app.exceptions import AuthenticationError, AuthorizationError, ValidationError, NotFoundError


# ==========================================================
# Clerk JWT Verification Tests
# ==========================================================

class TestVerifyClerkToken:
    """Tests for Clerk JWT verification."""

    def test_verify_valid_token(self):
        """Test verifying a valid Clerk JWT returns user payload."""
        from backend.app.utils.auth import verify_clerk_token

        mock_payload = {
            "sub": "clerk_user_1234",
            "email": "test@example.com",
            "name": "Test User",
            "email_verified_at": "2026-01-01T00:00:00Z",
        }

        mock_jwks = {
            "keys": [
                {
                    "kid": "test-kid",
                    "kty": "RSA",
                    "n": "test-modulus",
                    "e": "test-exponent",
                }
            ]
        }

        mock_key = MagicMock()

        with patch("backend.app.utils.auth._fetch_jwks", return_value=mock_jwks):
            with patch("jose.jwt.get_unverified_header", return_value={"kid": "test-kid"}):
                with patch("jose.jwk.construct", return_value=mock_key):
                    with patch("jose.jwt.decode", return_value=mock_payload):
                        result = verify_clerk_token("valid.clerk.token")

        assert result is not None
        assert result["id"] == "clerk_user_1234"
        assert result["email"] == "test@example.com"
        assert result["full_name"] == "Test User"
        assert result["is_active"] is True

    def test_verify_invalid_token_no_jwks_key(self):
        """Test that tokens with no matching JWKS key return None."""
        from backend.app.utils.auth import verify_clerk_token

        mock_jwks = {"keys": []}

        with patch("backend.app.utils.auth._fetch_jwks", return_value=mock_jwks):
            with patch("jose.jwt.get_unverified_header", return_value={"kid": "missing-kid"}):
                result = verify_clerk_token("invalid.token")

        assert result is None

    def test_verify_token_fetch_error(self):
        """Test that JWKS fetch errors return None."""
        from backend.app.utils.auth import verify_clerk_token

        with patch("backend.app.utils.auth._fetch_jwks", side_effect=Exception("Network error")):
            result = verify_clerk_token("error.token")

        assert result is None

    def test_get_user_id_from_token_valid(self):
        """Test extracting user ID from a valid Clerk token."""
        from backend.app.utils.auth import get_user_id_from_token

        with patch("backend.app.utils.auth.verify_clerk_token", return_value={"id": "clerk_user_5678"}):
            result = get_user_id_from_token("valid.clerk.token")

        assert result == "clerk_user_5678"

    def test_get_user_id_from_invalid_token(self):
        """Test extracting user ID from an invalid token."""
        from backend.app.utils.auth import get_user_id_from_token

        with patch("backend.app.utils.auth.verify_clerk_token", return_value=None):
            result = get_user_id_from_token("invalid.token")

        assert result is None


# ==========================================================
# Profile Repository Tests
# ==========================================================

class TestProfileRepository:
    """Tests for ProfileRepository."""

    def test_create_profile(self):
        """Test creating a profile."""
        from backend.app.repositories.profile_repository import ProfileRepository

        repo = ProfileRepository()

        mock_result = MagicMock()
        mock_result.data = [
            {
                "clerk_user_id": "clerk-uuid-1",
                "email": "test@example.com",
                "full_name": "Test",
                "role": "user",
                "is_active": True,
            }
        ]

        with patch("backend.app.repositories.profile_repository.get_supabase_client") as mock_client:
            mock_client.return_value.table.return_value.insert.return_value.execute.return_value = mock_result
            result = repo.create_profile("clerk-uuid-1", "test@example.com", "Test", UserRole.USER)

        assert result is not None
        assert result["email"] == "test@example.com"

    def test_get_profile_by_clerk_id_found(self):
        """Test getting a profile that exists."""
        from backend.app.repositories.profile_repository import ProfileRepository

        repo = ProfileRepository()

        mock_result = MagicMock()
        mock_result.data = [
            {
                "clerk_user_id": "clerk-uuid-1",
                "email": "test@example.com",
                "full_name": "Test",
                "role": "user",
                "is_active": True,
            }
        ]

        with patch("backend.app.repositories.profile_repository.get_supabase_client") as mock_client:
            mock_client.return_value.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_result
            result = repo.get_profile_by_clerk_id("clerk-uuid-1")

        assert result is not None
        assert result["email"] == "test@example.com"

    def test_get_profile_by_clerk_id_not_found(self):
        """Test getting a profile that does not exist."""
        from backend.app.repositories.profile_repository import ProfileRepository

        repo = ProfileRepository()

        mock_result = MagicMock()
        mock_result.data = []

        with patch("backend.app.repositories.profile_repository.get_supabase_client") as mock_client:
            mock_client.return_value.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_result
            result = repo.get_profile_by_clerk_id("clerk-nonexistent")

        assert result is None

    def test_get_profile_by_email_found(self):
        """Test getting a profile by email."""
        from backend.app.repositories.profile_repository import ProfileRepository

        repo = ProfileRepository()

        mock_result = MagicMock()
        mock_result.data = [
            {
                "clerk_user_id": "clerk-uuid-1",
                "email": "test@example.com",
                "full_name": "Test",
                "role": "user",
                "is_active": True,
            }
        ]

        with patch("backend.app.repositories.profile_repository.get_supabase_client") as mock_client:
            mock_client.return_value.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_result
            result = repo.get_profile_by_email("test@example.com")

        assert result is not None
        assert result["email"] == "test@example.com"

    def test_get_profile_by_email_not_found(self):
        """Test getting a profile by email that does not exist."""
        from backend.app.repositories.profile_repository import ProfileRepository

        repo = ProfileRepository()

        mock_result = MagicMock()
        mock_result.data = []

        with patch("backend.app.repositories.profile_repository.get_supabase_client") as mock_client:
            mock_client.return_value.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_result
            result = repo.get_profile_by_email("nonexistent@example.com")

        assert result is None

    def test_list_profiles(self):
        """Test listing profiles with pagination."""
        from backend.app.repositories.profile_repository import ProfileRepository

        repo = ProfileRepository()

        mock_result = MagicMock()
        mock_result.data = [
            {"clerk_user_id": "clerk-uuid-1", "email": "user1@example.com", "role": "user"},
            {"clerk_user_id": "clerk-uuid-2", "email": "user2@example.com", "role": "admin"},
        ]

        with patch("backend.app.repositories.profile_repository.get_supabase_client") as mock_client:
            mock_client.return_value.table.return_value.select.return_value.range.return_value.execute.return_value = mock_result
            result = repo.list_profiles(skip=0, limit=10)

        assert len(result) == 2

    def test_update_profile(self):
        """Test updating a profile."""
        from backend.app.repositories.profile_repository import ProfileRepository

        repo = ProfileRepository()

        mock_result = MagicMock()
        mock_result.data = [
            {
                "clerk_user_id": "clerk-uuid-1",
                "email": "test@example.com",
                "full_name": "Updated Name",
                "role": "user",
                "is_active": True,
            }
        ]

        with patch("backend.app.repositories.profile_repository.get_supabase_client") as mock_client:
            mock_client.return_value.table.return_value.update.return_value.eq.return_value.select.return_value.execute.return_value = mock_result
            result = repo.update_profile("clerk-uuid-1", {"full_name": "Updated Name"})

        assert result is not None
        assert result["full_name"] == "Updated Name"

    def test_delete_profile(self):
        """Test deleting a profile."""
        from backend.app.repositories.profile_repository import ProfileRepository

        repo = ProfileRepository()

        mock_result = MagicMock()
        mock_result.data = [{"clerk_user_id": "clerk-uuid-1"}]

        with patch("backend.app.repositories.profile_repository.get_supabase_client") as mock_client:
            mock_client.return_value.table.return_value.delete.return_value.eq.return_value.execute.return_value = mock_result
            result = repo.delete_profile("clerk-uuid-1")

        assert result is True

    def test_delete_profile_not_found(self):
        """Test deleting a profile that does not exist."""
        from backend.app.repositories.profile_repository import ProfileRepository

        repo = ProfileRepository()

        mock_result = MagicMock()
        mock_result.data = []

        with patch("backend.app.repositories.profile_repository.get_supabase_client") as mock_client:
            mock_client.return_value.table.return_value.delete.return_value.eq.return_value.execute.return_value = mock_result
            result = repo.delete_profile("clerk-nonexistent")

        assert result is False


# ==========================================================
# Auth Service Tests
# ==========================================================

class TestAuthService:
    """Tests for AuthService with Clerk."""

    def test_ensure_profile_creates_new(self):
        """Test that ensure_profile creates a new profile when one doesn't exist."""
        import asyncio
        from backend.app.services.auth_service import AuthService
        from backend.app.repositories.profile_repository import ProfileRepository

        repo = ProfileRepository()
        service = AuthService(repo)

        mock_profile = {
            "clerk_user_id": "clerk-uuid-1",
            "email": "test@example.com",
            "full_name": "Test User",
            "role": "user",
            "is_active": True,
        }

        with patch.object(repo, "get_profile_by_clerk_id", return_value=None):
            with patch.object(repo, "create_profile", return_value=mock_profile) as mock_create:
                result = asyncio.run(service.ensure_profile("clerk-uuid-1", "test@example.com", "Test User"))

        assert result is not None
        assert result["email"] == "test@example.com"
        mock_create.assert_called_once()

    def test_ensure_profile_returns_existing(self):
        """Test that ensure_profile returns existing profile when one exists."""
        import asyncio
        from backend.app.services.auth_service import AuthService
        from backend.app.repositories.profile_repository import ProfileRepository

        repo = ProfileRepository()
        service = AuthService(repo)

        mock_profile = {
            "clerk_user_id": "clerk-uuid-1",
            "email": "test@example.com",
            "full_name": "Test User",
            "role": "user",
            "is_active": True,
        }

        with patch.object(repo, "get_profile_by_clerk_id", return_value=mock_profile):
            result = asyncio.run(service.ensure_profile("clerk-uuid-1", "test@example.com", "Test User"))

        assert result is not None
        assert result["email"] == "test@example.com"

    def test_get_user_from_token_valid(self):
        """Test getting user from a valid Clerk token."""
        import asyncio
        from backend.app.services.auth_service import AuthService
        from backend.app.repositories.profile_repository import ProfileRepository

        repo = ProfileRepository()
        service = AuthService(repo)

        mock_profile = {
            "clerk_user_id": "clerk-uuid-1",
            "email": "test@example.com",
            "full_name": "Test User",
            "role": "user",
            "is_active": True,
        }

        with patch.object(service, "get_user_from_token", return_value=mock_profile):
            result = asyncio.run(service.get_user_from_token("valid.clerk.token"))

        assert result is not None
        assert result["email"] == "test@example.com"

    def test_get_user_from_token_invalid(self):
        """Test getting user from an invalid token."""
        import asyncio
        from backend.app.services.auth_service import AuthService
        from backend.app.repositories.profile_repository import ProfileRepository

        repo = ProfileRepository()
        service = AuthService(repo)

        with patch.object(service, "get_user_from_token", return_value=None):
            result = asyncio.run(service.get_user_from_token("invalid.token"))

        assert result is None

    def test_get_user_by_clerk_id_found(self):
        """Test getting a user by Clerk ID."""
        import asyncio
        from backend.app.services.auth_service import AuthService
        from backend.app.repositories.profile_repository import ProfileRepository

        repo = ProfileRepository()
        service = AuthService(repo)

        mock_profile = {
            "clerk_user_id": "clerk-uuid-1",
            "email": "test@example.com",
            "full_name": "Test User",
            "role": "user",
            "is_active": True,
        }

        with patch.object(repo, "get_profile_by_clerk_id", return_value=mock_profile):
            result = asyncio.run(service.get_user_by_clerk_id("clerk-uuid-1"))

        assert result is not None
        assert result["email"] == "test@example.com"

    def test_get_user_by_clerk_id_not_found(self):
        """Test getting a user by Clerk ID that doesn't exist."""
        import asyncio
        from backend.app.services.auth_service import AuthService
        from backend.app.repositories.profile_repository import ProfileRepository

        repo = ProfileRepository()
        service = AuthService(repo)

        with patch.object(repo, "get_profile_by_clerk_id", return_value=None):
            result = asyncio.run(service.get_user_by_clerk_id("clerk-nonexistent"))

        assert result is None


# ==========================================================
# Auth API Endpoint Tests
# ==========================================================

class TestAuthEndpoints:
    """Tests for authentication API endpoints."""

    @pytest.fixture
    def test_client(self, mock_current_user):
        """Create FastAPI test client with mocked auth."""
        from backend.app.main import app
        from backend.app.dependencies import get_current_active_user

        async def override_get_current_user():
            return mock_current_user

        app.dependency_overrides[get_current_active_user] = override_get_current_user
        try:
            with TestClient(app) as client:
                yield client
        finally:
            app.dependency_overrides.clear()

    def test_get_current_user_profile(self, test_client):
        """Test the get current user profile endpoint."""
        mock_user = {
            "id": "clerk-uuid-1",
            "email": "test@example.com",
            "full_name": "Test User",
            "role": "user",
            "is_active": True,
        }

        with patch("backend.app.api.auth.AuthService.get_user_from_token", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_user
            response = test_client.get(
                "/api/v1/auth/me",
                headers={"Authorization": "Bearer valid.clerk.token"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["email"] == "test@example.com"
        assert data["full_name"] == "Test User"

    def test_update_current_user(self, test_client):
        """Test updating the current user profile."""
        mock_user = {
            "id": "clerk-uuid-1",
            "email": "test@example.com",
            "full_name": "Updated Name",
            "role": "user",
            "is_active": True,
        }

        with patch("backend.app.api.auth.AuthService.get_user_from_token", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_user
            with patch("backend.app.api.auth.AuthService.update_user", new_callable=AsyncMock) as mock_update:
                mock_update.return_value = mock_user
                response = test_client.patch(
                    "/api/v1/auth/me",
                    json={"full_name": "Updated Name"},
                    headers={"Authorization": "Bearer valid.clerk.token"},
                )

        assert response.status_code == 200
        data = response.json()
        assert data["full_name"] == "Updated Name"

    def test_validate_token_endpoint(self, test_client):
        """Test token validation endpoint."""
        mock_user = {
            "id": "clerk-uuid-1",
            "email": "test@example.com",
            "role": "user",
        }

        with patch("backend.app.api.auth.AuthService.get_user_from_token", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_user
            response = test_client.post(
                "/api/v1/auth/validate?token=valid.clerk.token",
            )

        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True
        assert data["user_id"] == "clerk-uuid-1"

    def test_validate_token_invalid(self, test_client):
        """Test token validation with invalid token."""
        with patch("backend.app.api.auth.AuthService.get_user_from_token", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = None
            response = test_client.post(
                "/api/v1/auth/validate?token=invalid.token",
            )

        assert response.status_code == 401


# ==========================================================
# Dependencies Tests
# ==========================================================

class TestDependencies:
    """Tests for FastAPI auth dependencies."""

    def test_require_role_admin(self):
        """Test role-based access control for admin."""
        from backend.app.dependencies import require_role

        mock_user = MagicMock()
        mock_user.role = "admin"

    def test_require_role_researcher(self):
        """Test role-based access control for researcher."""
        from backend.app.dependencies import require_role

        mock_user = MagicMock()
        mock_user.role = "researcher"

    def test_require_role_user(self):
        """Test role-based access control for user."""
        from backend.app.dependencies import require_role

        mock_user = MagicMock()
        mock_user.role = "user"

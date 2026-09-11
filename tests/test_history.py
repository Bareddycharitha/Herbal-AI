from unittest.mock import AsyncMock, MagicMock
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.dependencies import get_current_active_user
from backend.app.schemas.user import UserResponse
from backend.app.api.history import get_history_service

client = TestClient(app)

def test_history_unauthorized():
    """Unauthenticated request to history endpoint should return 401."""
    # Ensure overrides are clear
    app.dependency_overrides.clear()
    response = client.get("/api/v1/history/")
    assert response.status_code == 401

def test_history_authorized_get():
    """Authenticated request should retrieve prediction history."""
    mock_service = MagicMock()
    mock_service.get_user_history = AsyncMock(return_value=[
        {
            "id": "11111111-2222-3333-4444-555555555555",
            "prediction": "Eczema",
            "confidence": 88.5,
            "disease_information": {"description": "Skin inflammation"},
            "recommended_herbs": [{"name": "Aloe Vera"}],
            "ai_summary": "Test AI summary",
            "created_at": "2026-09-10T09:00:00Z",
        }
    ])

    app.dependency_overrides[get_current_active_user] = lambda: UserResponse(
        id="user_clerk_123",
        email="test@example.com",
        full_name="Test User",
        is_active=True,
        role="user",
    )
    app.dependency_overrides[get_history_service] = lambda: mock_service

    try:
        response = client.get(
            "/api/v1/history/",
            headers={"Authorization": "Bearer mock-test-token"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["history"]) == 1
        assert data["history"][0]["prediction"] == "Eczema"
        assert data["history"][0]["ai_summary"] == "Test AI summary"
    finally:
        app.dependency_overrides.clear()

def test_history_delete_item():
    """Authenticated request can delete a specific history record."""
    mock_service = MagicMock()
    mock_service.delete_item = AsyncMock(return_value=True)

    app.dependency_overrides[get_current_active_user] = lambda: UserResponse(
        id="user_clerk_123",
        email="test@example.com",
        full_name="Test User",
        is_active=True,
        role="user",
    )
    app.dependency_overrides[get_history_service] = lambda: mock_service

    try:
        response = client.delete(
            "/api/v1/history/11111111-2222-3333-4444-555555555555",
            headers={"Authorization": "Bearer mock-test-token"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
    finally:
        app.dependency_overrides.clear()

def test_history_clear_all():
    """Authenticated request can clear all history records."""
    mock_service = MagicMock()
    mock_service.clear_history = AsyncMock(return_value=True)

    app.dependency_overrides[get_current_active_user] = lambda: UserResponse(
        id="user_clerk_123",
        email="test@example.com",
        full_name="Test User",
        is_active=True,
        role="user",
    )
    app.dependency_overrides[get_history_service] = lambda: mock_service

    try:
        response = client.delete(
            "/api/v1/history/",
            headers={"Authorization": "Bearer mock-test-token"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
    finally:
        app.dependency_overrides.clear()

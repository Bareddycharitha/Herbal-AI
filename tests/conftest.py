"""
Pytest Configuration and Shared Fixtures

Provides common fixtures for unit and integration tests.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
sys.path.insert(0, str(PROJECT_ROOT / "ai"))


# ==========================================================
# Environment Setup
# ==========================================================

@pytest.fixture(scope="session", autouse=True)
def setup_test_env():
    """Set up test environment variables."""
    os.environ["ENVIRONMENT"] = "test"
    # Don't set DEBUG/LOG_LEVEL here - let tests control their own settings
    os.environ["OPENROUTER_API_KEY"] = "test-openrouter-key"
    os.environ["OPENROUTER_MODEL"] = "google/gemini-flash-1.5"
    # Disable rate limiting in tests
    os.environ["RATE_LIMIT_REQUESTS_PER_MINUTE"] = "1000"
    os.environ["RATE_LIMIT_BURST"] = "100"
    yield
    # Cleanup if needed


# ==========================================================
# Test Data Fixtures
# ==========================================================

@pytest.fixture
def sample_image():
    """Create a sample test image."""
    img = Image.new("RGB", (256, 256), color="red")
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        img.save(f, "JPEG")
        yield f.name
    # Cleanup
    try:
        Path(f.name).unlink(missing_ok=True)
    except Exception:
        pass


@pytest.fixture
def sample_image_bytes():
    """Create sample image bytes for upload testing."""
    img = Image.new("RGB", (256, 256), color="blue")
    from io import BytesIO
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def invalid_file_bytes():
    """Create invalid file bytes (text file)."""
    return b"This is not an image file"


@pytest.fixture
def large_file_bytes():
    """Create file bytes exceeding size limit."""
    return b"x" * (11 * 1024 * 1024)  # 11 MB


# ==========================================================
# Mock Fixtures
# ==========================================================

@pytest.fixture(autouse=True)
def mock_classifier():
    """Mock universal classifier (autouse so lifespan never instantiates the real one)."""
    with patch("backend.app.services.universal_classifier.get_classifier") as mock_get, \
         patch("backend.app.services.universal_classifier.init_classifier") as mock_init:
        classifier = MagicMock()
        classifier.predict.return_value = {
            "class": "Skin",
            "confidence": 85.5,
            "is_ood": False,
            "ood_scores": {"energy": 0.1, "msp": 0.9, "entropy": 0.5, "combined": 0.3},
            "top_predictions": [
                {"class": "Skin", "confidence": 85.5},
                {"class": "Medicinal", "confidence": 10.2},
                {"class": "Other", "confidence": 4.3},
            ],
        }
        # ``/ready`` reads these attributes; return plain JSON-safe values
        # so the response can be serialised.
        from ai.utils.model_loader import ModelLoadStatus
        classifier.model_load_status = ModelLoadStatus(
            model_name="universal_classifier",
            checkpoint_path="ai/image_classifier/checkpoints/best_model.pth",
            loaded=True,
            error=None,
        )
        mock_get.return_value = classifier
        yield classifier


@pytest.fixture
def mock_skin_pipeline():
    """Mock skin disease recommendation pipeline."""
    with patch("ai.recommendation.recommendation_engine.get_recommendation") as mock:
        mock.return_value = {
            "success": True,
            "prediction": {"disease": "Acne", "confidence": 85.5, "confidence_level": "high"},
            "message": "Prediction confidence is high.",
            "top_predictions": [
                {"disease": "Acne", "confidence": 85.5},
                {"disease": "Eczema", "confidence": 10.2},
            ],
            "gradcam_image": "/results/gradcam_0_85.5.jpg",
            "disease_information": {
                "description": "Acne is a common skin condition...",
                "symptoms": ["pimples", "blackheads", "inflammation"],
                "causes": ["hormones", "bacteria", "excess oil"],
                "prevention": ["clean face regularly", "avoid touching face"],
            },
            "recommended_herbs": [
                {"name": "Tea Tree", "efficacy": 8, "weight": 0.8},
                {"name": "Aloe Vera", "efficacy": 7, "weight": 0.7},
            ],
            "herb_details": {},
            "ai_summary": "Based on the analysis...",
            "binary_stage": {"used": True, "is_healthy": False, "confidence": 92.0},
            "ood_scores": {"energy": 0.1, "msp": 0.9, "entropy": 0.5, "combined": 0.3},
            "is_ood": False,
        }
        yield mock


@pytest.fixture
def mock_herb_pipeline():
    """Mock herb identification pipeline."""
    with patch("ai.recommendation.herb_recommendation_engine.get_herb_recommendation") as mock:
        mock.return_value = {
            "success": True,
            "prediction": {"herb": "Neem", "confidence": 92.0, "is_confident": True},
            "message": "Medicinal plant identified successfully.",
            "top_predictions": [
                {"class": "Neem", "confidence": 92.0},
                {"class": "Tulsi", "confidence": 5.0},
            ],
            "herb_information": {
                "name": "Neem",
                "botanical_name": "Azadirachta indica",
                "family": "Meliaceae",
                "benefits": ["antibacterial", "antifungal", "anti-inflammatory"],
                "preparation_method": "Paste or decoction",
                "side_effects": ["possible skin irritation"],
                "contraindications": ["pregnancy"],
            },
            "ai_summary": "Neem is a medicinal plant...",
            "ood_scores": {"energy": 0.1, "msp": 0.9, "entropy": 0.5, "combined": 0.3},
            "is_ood": False,
        }
        yield mock


@pytest.fixture
def mock_openrouter_client():
    """Mock OpenRouter client and lazy-initialized engines."""
    with patch("ai.llm.openrouter_client.get_openrouter_client") as mock_openrouter, \
         patch("backend.app.api.summary.get_summary_engine") as mock_summary_engine, \
         patch("backend.app.api.chat.get_chatbot") as mock_chatbot, \
         patch("ai.llm.summary_engine.SummaryEngine.generate_summary") as mock_generate_summary, \
         patch("ai.llm.summary_engine.SummaryEngine.generate_herb_summary") as mock_generate_herb_summary, \
         patch("ai.llm.summary_engine.SummaryEngine._generate_async") as mock_generate_async, \
         patch("ai.llm.chatbot_engine.ChatbotEngine.ask") as mock_chatbot_ask:

        client = AsyncMock()
        client.generate.return_value = {
            "success": True,
            "response": "This is a generated summary.",
            "cached": False,
        }
        client.health_check.return_value = True
        mock_openrouter.return_value = client

        # Mock summary engine instance methods
        mock_generate_summary.return_value = "This is a generated summary."
        mock_generate_herb_summary.return_value = "This is a generated herb summary."

        # Mock the *private* async method that the API actually calls.
        # Returns a coroutine-friendly value.
        async def _fake_generate_async(*args, **kwargs):
            return {
                "success": True,
                "response": "This is a generated summary.",
                "cached": False,
            }
        mock_generate_async.side_effect = _fake_generate_async

        # Mock chatbot engine instance method
        mock_chatbot_ask.return_value = "This is a chat response."

        # Mock summary engine factory
        summary_engine = AsyncMock()
        summary_engine.generate_summary.return_value = "This is a generated summary."
        summary_engine.generate_summary_async.return_value = "This is a generated summary."
        summary_engine.generate_herb_summary.return_value = "This is a generated herb summary."
        summary_engine.generate_herb_summary_async.return_value = "This is a generated herb summary."
        # Provide a sync-returning _generate_async on the engine mock too,
        # so a coroutine that returns it directly is also fine.
        async def _fake_engine_generate(*args, **kwargs):
            return {
                "success": True,
                "response": "This is a generated summary.",
                "cached": False,
            }
        summary_engine._generate_async.side_effect = _fake_engine_generate
        mock_summary_engine.return_value = summary_engine

        # Mock chatbot engine factory
        chatbot_engine = AsyncMock()
        chatbot_engine.ask.return_value = "This is a chat response."
        chatbot_engine.ask_async.return_value = "This is a chat response."
        mock_chatbot.return_value = chatbot_engine

        yield client


# ==========================================================
# FastAPI Test Client
# ==========================================================

@pytest.fixture
def mock_current_user():
    """Mock current authenticated user (Clerk profile)."""
    from backend.app.schemas.user import UserResponse, UserRole
    from datetime import datetime

    return UserResponse(
        id="clerk_user_1234",
        email="test@example.com",
        full_name="Test User",
        role=UserRole.ADMIN,
        is_active=True,
        created_at=datetime(2026, 1, 1),
    )


@pytest.fixture
def test_client(mock_current_user):
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


# ==========================================================
# Knowledge Base Fixtures
# ==========================================================

@pytest.fixture
def temp_disease_kb():
    """Create temporary disease knowledge base for testing."""
    import json
    data = {
        "diseases": [
            {
                "label": "Acne",
                "description": "Acne is a common skin condition...",
                "symptoms": ["pimples", "blackheads"],
                "causes": ["hormones", "bacteria"],
                "prevention": ["clean face regularly"],
                "self_care": ["use gentle cleanser"],
                "when_to_consult_doctor": "If severe",
                "medical_disclaimer": "Not medical advice",
                "recommended_herbs": [
                    {"name": "Tea Tree", "efficacy": 8, "weight": 0.8}
                ],
            }
        ]
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        yield f.name
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def temp_herbal_kb():
    """Create temporary herbal knowledge base for testing."""
    import json
    data = {
        "Tea Tree": {
            "name": "Tea Tree",
            "botanical_name": "Melaleuca alternifolia",
            "family": "Myrtaceae",
            "active_compounds": ["terpinen-4-ol"],
            "phytochemicals": ["terpenes"],
            "benefits": ["antimicrobial", "anti-inflammatory"],
            "preparation_method": "Topical application",
            "side_effects": ["skin irritation"],
            "contraindications": ["pregnancy"],
            "research_papers": [],
            "skin_types": ["oily", "acne-prone"],
            "evidence_level": "moderate",
        }
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        yield f.name
    Path(f.name).unlink(missing_ok=True)


# ==========================================================
# Settings Fixture
# ==========================================================

@pytest.fixture
def test_settings():
    """Create test settings."""
    from backend.app.config import Settings
    return Settings(
        environment="test",
        debug=True,
        log_level="DEBUG",
        log_format="text",
        cors_origins=["http://localhost:3000"],
        max_upload_size_mb=10,
        clerk_publishable_key="pk_test_123",
        clerk_secret_key="sk_test_456",
        clerk_jwks_url="https://test.clerk.accounts.dev/.well-known/jwks.json",
    )
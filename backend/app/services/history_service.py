"""
History Service

Business logic for managing patient history records.
"""

from typing import Any, Dict, List, Optional
import structlog

from backend.app.repositories.history_repository import HistoryRepository
from backend.app.repositories.profile_repository import ProfileRepository

logger = structlog.get_logger(__name__)


class HistoryService:
    """Service to handle user analysis history."""

    def __init__(self):
        self.history_repo = HistoryRepository()
        self.profile_repo = ProfileRepository()

    def _get_or_create_profile_uuid(
        self,
        clerk_user_id: str,
        user_email: Optional[str] = None,
        user_name: Optional[str] = None,
    ) -> Optional[str]:
        """Ensure profile exists and return its internal UUID."""
        try:
            profile = self.profile_repo.get_profile_by_clerk_id(clerk_user_id)
            if not profile:
                profile = self.profile_repo.create_profile(
                    clerk_user_id=clerk_user_id,
                    email=user_email,
                    full_name=user_name or "Herbal-AI User",
                )
            return profile.get("id")
        except Exception as e:
            logger.error("Failed to resolve profile UUID", clerk_user_id=clerk_user_id, error=str(e))
            return None

    async def record_prediction(
        self,
        profile_id: str,
        prediction: Any,
        confidence: Optional[float] = None,
        confidence_level: Optional[str] = None,
        top_predictions: Optional[List[Dict[str, Any]]] = None,
        disease_information: Optional[Dict[str, Any]] = None,
        recommended_herbs: Optional[List[Dict[str, Any]]] = None,
        image_path: Optional[str] = None,
        ai_summary: Optional[str] = None,
        prediction_id: Optional[str] = None,
        user_email: Optional[str] = None,
        user_name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Record prediction into history, creating profile row if missing."""
        try:
            profile_uuid = self._get_or_create_profile_uuid(profile_id, user_email, user_name)
            if not profile_uuid:
                logger.error("Cannot save history: profile UUID missing", profile_id=profile_id)
                return None

            return self.history_repo.create_prediction_history(
                profile_uuid=profile_uuid,
                prediction=prediction,
                confidence=float(confidence or 0.0),
                confidence_level=confidence_level,
                top_predictions=top_predictions,
                disease_information=disease_information,
                recommended_herbs=recommended_herbs,
                image_path=image_path,
                ai_summary=ai_summary,
                prediction_id=prediction_id,
            )
        except Exception as e:
            logger.error("Error in record_prediction", profile_id=profile_id, error=str(e))
            return None

    async def get_user_history(
        self,
        profile_id: str,
        limit: int = 50,
        skip: int = 0,
    ) -> List[Dict[str, Any]]:
        """Get history for a user."""
        profile_uuid = self._get_or_create_profile_uuid(profile_id)
        if not profile_uuid:
            return []
        return self.history_repo.get_user_history(profile_uuid, limit=limit, skip=skip)

    async def get_history_detail(
        self,
        profile_id: str,
        history_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get a single historical prediction."""
        profile_uuid = self._get_or_create_profile_uuid(profile_id)
        if not profile_uuid:
            return None
        return self.history_repo.get_history_by_id(profile_uuid, history_id)

    async def update_summary(
        self,
        profile_id: str,
        history_id_or_pred_id: str,
        ai_summary: str,
    ) -> bool:
        """Update the summary of a historical analysis."""
        profile_uuid = self._get_or_create_profile_uuid(profile_id)
        if not profile_uuid:
            return False
        return self.history_repo.update_ai_summary(profile_uuid, history_id_or_pred_id, ai_summary)

    async def delete_item(
        self,
        profile_id: str,
        history_id: str,
    ) -> bool:
        """Delete a history item."""
        profile_uuid = self._get_or_create_profile_uuid(profile_id)
        if not profile_uuid:
            return False
        return self.history_repo.delete_history_item(profile_uuid, history_id)

    async def clear_history(
        self,
        profile_id: str,
    ) -> bool:
        """Clear all history for a user."""
        profile_uuid = self._get_or_create_profile_uuid(profile_id)
        if not profile_uuid:
            return False
        return self.history_repo.clear_user_history(profile_uuid)

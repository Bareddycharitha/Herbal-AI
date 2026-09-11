"""
History Repository

Repository layer for user prediction and analysis history in Supabase PostgreSQL.
"""

from typing import Any, Dict, List, Optional
import json
import structlog
from backend.app.supabase.client import get_supabase_client

logger = structlog.get_logger(__name__)


class HistoryRepository:
    """Repository for managing prediction history in Supabase."""

    @staticmethod
    def create_prediction_history(
        profile_uuid: str,
        prediction: Any,
        confidence: float,
        disease_information: Optional[Dict[str, Any]] = None,
        recommended_herbs: Optional[List[Dict[str, Any]]] = None,
        prediction_id: Optional[str] = None,
        top_predictions: Optional[List[Dict[str, Any]]] = None,
        image_path: Optional[str] = None,
        ai_summary: Optional[str] = None,
        confidence_level: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Insert a prediction history record for a user."""
        try:
            supabase = get_supabase_client()
            
            # Format prediction as clean string / JSON
            pred_val = prediction if isinstance(prediction, str) else json.dumps(prediction)
            
            # Bundle metadata into disease_information JSON to ensure full persistence
            disease_info_payload = dict(disease_information or {})
            if top_predictions:
                disease_info_payload["top_predictions"] = top_predictions
            if image_path:
                disease_info_payload["image_path"] = image_path
            if ai_summary:
                disease_info_payload["ai_summary"] = ai_summary
            if confidence_level:
                disease_info_payload["confidence_level"] = confidence_level
            if prediction_id:
                disease_info_payload["prediction_id"] = prediction_id

            row = {
                "profile_id": profile_uuid,
                "prediction": pred_val,
                "confidence": float(confidence or 0.0),
                "disease_information": disease_info_payload,
                "recommended_herbs": recommended_herbs or [],
            }

            result = supabase.table("prediction_history").insert(row).execute()
            if result.data and len(result.data) > 0:
                logger.info("Saved prediction history", profile_uuid=profile_uuid, id=result.data[0].get("id"))
                return HistoryRepository._format_row(result.data[0])
            return None
        except Exception as e:
            logger.error("Failed to save prediction history", profile_uuid=profile_uuid, error=str(e))
            return None

    @staticmethod
    def get_user_history(
        profile_uuid: str,
        limit: int = 50,
        skip: int = 0,
    ) -> List[Dict[str, Any]]:
        """Fetch historical predictions for a user UUID, sorted newest first."""
        try:
            supabase = get_supabase_client()
            result = (
                supabase.table("prediction_history")
                .select("*")
                .eq("profile_id", profile_uuid)
                .order("created_at", desc=True)
                .range(skip, skip + limit - 1)
                .execute()
            )
            return [HistoryRepository._format_row(row) for row in (result.data or [])]
        except Exception as e:
            logger.error("Failed to fetch user history", profile_uuid=profile_uuid, error=str(e))
            return []

    @staticmethod
    def get_history_by_id(profile_uuid: str, history_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a specific history record belonging to the user."""
        try:
            supabase = get_supabase_client()
            result = (
                supabase.table("prediction_history")
                .select("*")
                .eq("id", history_id)
                .eq("profile_id", profile_uuid)
                .limit(1)
                .execute()
            )
            if result.data and len(result.data) > 0:
                return HistoryRepository._format_row(result.data[0])
            return None
        except Exception as e:
            logger.error("Failed to fetch history by id", profile_uuid=profile_uuid, history_id=history_id, error=str(e))
            return None

    @staticmethod
    def update_ai_summary(profile_uuid: str, history_id_or_pred_id: str, ai_summary: str) -> bool:
        """Update the AI summary on an existing prediction history record."""
        try:
            supabase = get_supabase_client()
            # Check by primary key id first
            record = HistoryRepository.get_history_by_id(profile_uuid, history_id_or_pred_id)
            target_id = None
            if record:
                target_id = record["id"]
                disease_info = record.get("disease_information") or {}
            else:
                # Search by prediction_id in disease_information JSON
                res = (
                    supabase.table("prediction_history")
                    .select("*")
                    .eq("profile_id", profile_uuid)
                    .order("created_at", desc=True)
                    .limit(20)
                    .execute()
                )
                for item in (res.data or []):
                    d_info = item.get("disease_information") or {}
                    if isinstance(d_info, dict) and d_info.get("prediction_id") == history_id_or_pred_id:
                        target_id = item["id"]
                        disease_info = d_info
                        break

            if not target_id:
                return False

            disease_info["ai_summary"] = ai_summary
            result = (
                supabase.table("prediction_history")
                .update({"disease_information": disease_info})
                .eq("id", target_id)
                .execute()
            )
            return bool(result.data)
        except Exception as e:
            logger.error("Failed to update AI summary in history", id=history_id_or_pred_id, error=str(e))
            return False

    @staticmethod
    def delete_history_item(profile_uuid: str, history_id: str) -> bool:
        """Delete a single history entry for a user."""
        try:
            supabase = get_supabase_client()
            result = (
                supabase.table("prediction_history")
                .delete()
                .eq("id", history_id)
                .eq("profile_id", profile_uuid)
                .execute()
            )
            return bool(result.data)
        except Exception as e:
            logger.error("Failed to delete history item", profile_uuid=profile_uuid, history_id=history_id, error=str(e))
            return False

    @staticmethod
    def clear_user_history(profile_uuid: str) -> bool:
        """Clear all prediction history for a user."""
        try:
            supabase = get_supabase_client()
            supabase.table("prediction_history").delete().eq("profile_id", profile_uuid).execute()
            return True
        except Exception as e:
            logger.error("Failed to clear user history", profile_uuid=profile_uuid, error=str(e))
            return False

    @staticmethod
    def _format_row(row: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize database row for API consumers."""
        d_info = row.get("disease_information") or {}
        if isinstance(d_info, str):
            try:
                d_info = json.loads(d_info)
            except Exception:
                d_info = {}

        pred = row.get("prediction")
        if isinstance(pred, str) and pred.startswith("{"):
            try:
                pred = json.loads(pred)
            except Exception:
                pass

        return {
            "id": row.get("id"),
            "prediction_id": d_info.get("prediction_id") or row.get("id"),
            "prediction": pred,
            "confidence": row.get("confidence"),
            "confidence_level": d_info.get("confidence_level") or ("High" if (row.get("confidence") or 0) >= 70 else "Medium"),
            "top_predictions": d_info.get("top_predictions") or [],
            "image_path": d_info.get("image_path"),
            "disease_information": d_info,
            "recommended_herbs": row.get("recommended_herbs") or [],
            "ai_summary": d_info.get("ai_summary"),
            "created_at": row.get("created_at"),
        }

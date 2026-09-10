"""
History API Endpoints

Endpoints for retrieving and managing user analysis history.
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query

from backend.app.dependencies import get_current_active_user
from backend.app.schemas.user import UserResponse
from backend.app.services.history_service import HistoryService
from backend.app.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(
    prefix="/history",
    tags=["History"],
)

_history_service = None


def get_history_service() -> HistoryService:
    global _history_service
    if _history_service is None:
        _history_service = HistoryService()
    return _history_service


@router.get("/", response_model=Dict[str, Any])
async def list_user_history(
    limit: int = Query(50, ge=1, le=100),
    skip: int = Query(0, ge=0),
    service: HistoryService = Depends(get_history_service),
    current_user: UserResponse = Depends(get_current_active_user),
):
    """Retrieve the authenticated user's prediction history, sorted newest first."""
    items = await service.get_user_history(
        profile_id=current_user.id,
        limit=limit,
        skip=skip,
    )
    return {
        "success": True,
        "history": items,
        "total": len(items),
    }


@router.get("/{history_id}", response_model=Dict[str, Any])
async def get_history_item(
    history_id: str,
    service: HistoryService = Depends(get_history_service),
    current_user: UserResponse = Depends(get_current_active_user),
):
    """Retrieve a specific prediction record from history."""
    item = await service.get_history_detail(
        profile_id=current_user.id,
        history_id=history_id,
    )
    if not item:
        raise HTTPException(status_code=404, detail="History record not found")
    return {
        "success": True,
        "history": item,
    }


@router.delete("/{history_id}")
async def delete_history_item(
    history_id: str,
    service: HistoryService = Depends(get_history_service),
    current_user: UserResponse = Depends(get_current_active_user),
):
    """Delete a single history entry."""
    success = await service.delete_item(
        profile_id=current_user.id,
        history_id=history_id,
    )
    if not success:
        raise HTTPException(status_code=404, detail="History record not found or could not be deleted")
    return {"success": True, "message": "History entry deleted"}


@router.delete("/")
async def clear_history(
    service: HistoryService = Depends(get_history_service),
    current_user: UserResponse = Depends(get_current_active_user),
):
    """Clear all analysis history for the current user."""
    success = await service.clear_history(profile_id=current_user.id)
    return {"success": success, "message": "All history cleared"}

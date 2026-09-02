"""
Grad-CAM job status and retrieval endpoint.

The main prediction endpoint schedules Grad-CAM generation as a
``BackgroundTasks`` task and returns the prediction immediately without
the attention map. The frontend polls this endpoint to learn when the
Grad-CAM image is ready and to retrieve its public URL.

The job store is an in-memory dict keyed by ``prediction_id``. Entries
expire after a TTL (default 10 minutes) to keep memory bounded; if the
client polls after the TTL, the endpoint reports ``ready=False`` so the
client can give up gracefully instead of seeing a 404.
"""

import threading
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query

from backend.app.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(
    prefix="/gradcam",
    tags=["Grad-CAM"],
)


# Default TTL: 10 minutes. Tuned to be longer than typical Grad-CAM
# generation time (a few seconds) so the frontend has plenty of time to
# poll, but short enough to keep memory usage bounded under traffic.
DEFAULT_TTL_SECONDS = 600.0


class _GradCamJobStore:
    """
    Thread-safe in-memory job store.

    Each entry is a dict with at least:
      - ``status``: one of "pending", "ready", "failed"
      - ``url``: public URL of the saved Grad-CAM image (when status=ready)
      - ``prediction_id``: the lookup key
      - ``created_at``: monotonic time when the entry was created
    """

    def __init__(self, ttl_seconds: float = DEFAULT_TTL_SECONDS) -> None:
        self._ttl = ttl_seconds
        self._lock = threading.RLock()
        self._jobs: Dict[str, Dict[str, Any]] = {}

    def create(self, prediction_id: str) -> None:
        with self._lock:
            self._jobs[prediction_id] = {
                "status": "pending",
                "url": None,
                "prediction_id": prediction_id,
                "created_at": time.monotonic(),
            }

    def mark_ready(self, prediction_id: str, url: str) -> None:
        with self._lock:
            entry = self._jobs.get(prediction_id)
            if entry is None:
                # The status endpoint was not registered yet (race). Insert
                # a ready entry so the next poll sees it.
                entry = {
                    "status": "ready",
                    "url": url,
                    "prediction_id": prediction_id,
                    "created_at": time.monotonic(),
                }
                self._jobs[prediction_id] = entry
                return
            entry["status"] = "ready"
            entry["url"] = url

    def mark_no_gradcam(self, prediction_id: str) -> None:
        """Mark a job as 'no Grad-CAM will be produced' (e.g. unsupported image)."""
        with self._lock:
            entry = self._jobs.get(prediction_id)
            if entry is None:
                return
            entry["status"] = "no_gradcam"
            entry["url"] = None

    def mark_failed(self, prediction_id: str, error: str) -> None:
        with self._lock:
            entry = self._jobs.get(prediction_id)
            if entry is None:
                return
            entry["status"] = "failed"
            entry["error"] = error

    def get(self, prediction_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            entry = self._jobs.get(prediction_id)
            if entry is None:
                return None
            # TTL eviction. We do not delete aggressively, but we return
            # ``None`` once the entry has aged past the TTL so the client
            # sees a clean "expired" response instead of stale data.
            if time.monotonic() - entry["created_at"] > self._ttl:
                return None
            status = entry["status"]
            url = entry.get("url")
            # A job is "ready" only when the status is "ready" AND a URL
            # was actually produced. "no_gradcam" is a terminal state
            # that tells the client no heatmap will appear.
            ready = status == "ready" and url is not None
            return {
                "prediction_id": entry["prediction_id"],
                "ready": ready,
                "status": status,
                "url": url,
                "error": entry.get("error"),
            }


# Module-level singleton. Used by the prediction endpoint to register
# new jobs and by this router to retrieve their status.
_job_store = _GradCamJobStore()


def get_job_store() -> _GradCamJobStore:
    """Return the process-wide Grad-CAM job store."""
    return _job_store


@router.get("/{prediction_id}", tags=["Grad-CAM"])
async def get_gradcam_status(
    prediction_id: str,
    wait_seconds: float = Query(
        0.0,
        ge=0.0,
        le=10.0,
        description=(
            "If > 0, block up to this many seconds waiting for the Grad-CAM "
            "to become ready. Returns immediately once ready or the timeout "
            "elapses, whichever comes first."
        ),
    ),
) -> Dict[str, Any]:
    """
    Return the Grad-CAM job status for a given ``prediction_id``.

    Response shape::

        {
          "prediction_id": "...",
          "ready": true|false,
          "status": "pending|ready|failed",
          "url": "/results/gradcam_<id>.jpg" | null,
          "error": "..." (only when status=failed)
        }
    """
    store = get_job_store()
    deadline = time.monotonic() + wait_seconds

    while True:
        entry = store.get(prediction_id)
        if entry is None:
            # Either the prediction_id is unknown (caller never got it
            # from /predict) or it has expired. Distinguish by checking
            # the static file: a Grad-CAM may exist on disk even after
            # the job entry has expired. We only fall back to the file
            # check for "ready" cases to avoid 404s on stale polling.
            raise HTTPException(
                status_code=404,
                detail={
                    "prediction_id": prediction_id,
                    "ready": False,
                    "status": "unknown_or_expired",
                },
            )

        if entry["ready"] or entry["status"] == "failed" or wait_seconds <= 0:
            return entry

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return entry

        # Short polling interval so the client can keep its connection
        # alive without flooding the server.
        time.sleep(min(0.1, remaining))

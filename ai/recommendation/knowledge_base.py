"""
Knowledge Base with Caching

In-memory cached knowledge base with hot-reload support for development.
"""

import json
import threading
import time
from pathlib import Path
from typing import Any, Optional

import structlog

logger = structlog.get_logger(__name__)


class KnowledgeBase:
    """
    Thread-safe knowledge base with file-watching for hot reload.

    Features:
    - In-memory caching for fast lookups
    - File modification detection for hot reload (dev mode)
    - Thread-safe operations
    """

    def __init__(self, json_path: str | Path, hot_reload: bool = False):
        self.json_path = Path(json_path)
        self.hot_reload = hot_reload
        self._lock = threading.RLock()
        self._data: dict[str, Any] = {}
        self._diseases: list[dict[str, Any]] = []
        self._last_modified: float = 0
        self._load()

        if hot_reload:
            self._start_watcher()

    def _load(self) -> None:
        """Load knowledge base from JSON file."""
        with self._lock:
            try:
                if self.json_path.exists():
                    stat = self.json_path.stat()
                    self._last_modified = stat.st_mtime

                    with open(self.json_path, "r", encoding="utf-8") as f:
                        self._data = json.load(f)

                    self._diseases = self._data.get("diseases", [])
                    logger.info(
                        "Knowledge base loaded",
                        path=str(self.json_path),
                        disease_count=len(self._diseases),
                    )
                else:
                    self._data = {"diseases": []}
                    self._diseases = []
                    logger.warning("Knowledge base file not found, initializing empty", path=str(self.json_path))
            except Exception as e:
                logger.error("Failed to load knowledge base", path=str(self.json_path), error=str(e))
                self._data = {"diseases": []}
                self._diseases = []

    def _check_reload(self) -> bool:
        """Check if file has been modified and reload if needed."""
        if not self.hot_reload:
            return False

        try:
            stat = self.json_path.stat()
            if stat.st_mtime > self._last_modified:
                logger.info("Knowledge base file changed, reloading", path=str(self.json_path))
                self._load()
                return True
        except Exception as e:
            logger.warning("Failed to check knowledge base modification", error=str(e))
        return False

    def _start_watcher(self) -> None:
        """Start background thread for hot reload (dev mode only)."""
        def watcher():
            while True:
                time.sleep(2)  # Check every 2 seconds
                self._check_reload()

        thread = threading.Thread(target=watcher, daemon=True)
        thread.start()
        logger.info("Knowledge base hot-reload enabled", path=str(self.json_path))

    # ======================================================
    # Disease Lookup
    # ======================================================

    def get_disease(self, disease_name: str) -> Optional[dict[str, Any]]:
        """Get disease by name (case-insensitive)."""
        self._check_reload()

        with self._lock:
            disease_name = disease_name.strip().lower()

            for disease in self._diseases:
                if disease["label"].lower() == disease_name:
                    return disease

            # No match found — return None so callers can handle the
            # "not found" case explicitly (e.g. fallback to a generic
            # response at the recommendation-engine level).
            return None

    # ======================================================
    # Herbal Recommendations
    # ======================================================

    def get_recommendations(self, disease_name: str) -> list[dict[str, Any]]:
        """Get herbal recommendations for a disease."""
        disease = self.get_disease(disease_name)
        if disease is None:
            return []
        return disease.get("recommended_herbs", [])

    # ======================================================
    # Disease Information
    # ======================================================

    def get_disease_information(self, disease_name: str) -> dict[str, Any]:
        """Get formatted disease information."""
        disease = self.get_disease(disease_name)

        if disease is None:
            return {}

        return {
            "description": disease.get("description", ""),
            "symptoms": disease.get("symptoms", []),
            "self_care": disease.get("self_care", []),
            "when_to_consult_doctor": disease.get("when_to_consult_doctor", ""),
            "medical_disclaimer": disease.get("medical_disclaimer", ""),
            "causes": disease.get("causes", []),
            "prevention": disease.get("prevention", []),
        }

    # ======================================================
    # Utility Methods
    # ======================================================

    def list_diseases(self) -> list[str]:
        """List all disease labels."""
        self._check_reload()
        with self._lock:
            return [d["label"] for d in self._diseases]

    def reload(self) -> None:
        """Force reload knowledge base."""
        self._load()

    @property
    def disease_count(self) -> int:
        """Get number of diseases in knowledge base."""
        return len(self._diseases)
"""
Herb Knowledge Base with Caching

In-memory cached herbal knowledge base with hot-reload support.
"""

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

import structlog

logger = structlog.get_logger(__name__)


def load_class_to_kb_mapping(json_path: str | Path) -> dict[str, Optional[str]]:
    """Load class name to knowledge base key mapping."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Failed to load class-to-KB mapping", path=str(json_path), error=str(e))
        return {}


class HerbKnowledgeBase:
    """
    Thread-safe herbal knowledge base with file-watching for hot reload.

    Features:
    - In-memory caching for fast lookups
    - File modification detection for hot reload (dev mode)
    - Thread-safe operations
    """

    def __init__(self, json_path: str | Path = None, hot_reload: bool = False):
        if json_path is None:
            json_path = (
                Path(__file__).resolve().parent.parent
                / "datasets"
                / "knowledge_base"
                / "herbal_knowledge_base.json"
            )
        self.json_path = Path(json_path)
        self.hot_reload = hot_reload
        self._lock = threading.RLock()
        self._data: dict[str, Any] = {}
        self._last_modified: float = 0

        # Load class-to-KB mapping
        class_mapping_path = self.json_path.parent / "class_to_kb_mapping.json"
        self._class_to_kb = load_class_to_kb_mapping(class_mapping_path)

        self._load()

        if hot_reload:
            self._start_watcher()

    def _load(self) -> None:
        """Load herbal knowledge base from JSON file."""
        with self._lock:
            try:
                stat = self.json_path.stat()
                self._last_modified = stat.st_mtime

                with open(self.json_path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)

                logger.info(
                    "Herb knowledge base loaded",
                    path=str(self.json_path),
                    herb_count=len(self._data),
                )
            except Exception as e:
                logger.error("Failed to load herb knowledge base", path=str(self.json_path), error=str(e))
                raise

    def _check_reload(self) -> bool:
        """Check if file has been modified and reload if needed."""
        if not self.hot_reload:
            return False

        try:
            stat = self.json_path.stat()
            if stat.st_mtime > self._last_modified:
                logger.info("Herb knowledge base file changed, reloading", path=str(self.json_path))
                self._load()
                return True
        except Exception as e:
            logger.warning("Failed to check herb knowledge base modification", error=str(e))
        return False

    def _start_watcher(self) -> None:
        """Start background thread for hot reload (dev mode only)."""
        def watcher():
            while True:
                time.sleep(2)  # Check every 2 seconds
                self._check_reload()

        thread = threading.Thread(target=watcher, daemon=True)
        thread.start()
        logger.info("Herb knowledge base hot-reload enabled", path=str(self.json_path))

    # =====================================================
    # Get Herb Details
    # =====================================================

    def get_herb(self, herb_name: str) -> Optional[dict[str, Any]]:
        """Get herb details by name (with class-to-KB mapping support)."""
        self._check_reload()

        with self._lock:
            # Try direct lookup first
            herb = self._data.get(herb_name)

            # If not found, try class-to-KB mapping
            if herb is None:
                kb_key = self._class_to_kb.get(herb_name)
                if kb_key:
                    herb = self._data.get(kb_key)

            if herb is None:
                return None

            return {
                "name": herb.get("name", ""),
                "botanical_name": herb.get("botanical_name", ""),
                "family": herb.get("family", ""),
                "active_compounds": herb.get("active_compounds", []),
                "phytochemicals": herb.get("phytochemicals", []),
                "benefits": herb.get("benefits", []),
                "preparation_method": herb.get("preparation_method", ""),
                "side_effects": herb.get("side_effects", []),
                "contraindications": herb.get("contraindications", []),
                "research_papers": herb.get("research_papers", []),
                "skin_types": herb.get("skin_types", []),
                "evidence_level": herb.get("evidence_level", ""),
            }

    def exists(self, herb_name: str) -> bool:
        """Check if herb exists in knowledge base."""
        self._check_reload()
        with self._lock:
            return herb_name in self._data

    # =====================================================
    # Utility Methods
    # =====================================================

    def list_herbs(self) -> list[str]:
        """List all herb names."""
        self._check_reload()
        with self._lock:
            return list(self._data.keys())

    def reload(self) -> None:
        """Force reload knowledge base."""
        self._load()

    @property
    def herb_count(self) -> int:
        """Get number of herbs in knowledge base."""
        return len(self._data)


# Global instance with hot-reload in development
HOT_RELOAD = os.getenv("ENVIRONMENT", "development") == "development"
herb_knowledge = HerbKnowledgeBase(hot_reload=HOT_RELOAD)
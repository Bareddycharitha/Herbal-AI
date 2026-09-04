"""
File Upload Validation

Secure file validation utilities for uploaded images including
MIME type verification, magic bytes checking, and size limits.
"""

# IMPORTANT: do NOT import python-magic at module-load time. The ``magic``
# package is a thin ctypes wrapper around libmagic and on Windows it
# segfaults (Windows access violation) for many installs — that segfault
# cannot be caught by a ``try/except`` because it is a *native* crash,
# not a Python exception, and it kills the entire process during import.
# Instead, probe the package in a subprocess once at module import time
# (see ``_magic_available`` below) and import it lazily inside
# ``FileValidator.__init__`` if the probe says it is safe.
import importlib
import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Optional

from backend.app.config import settings
from backend.app.exceptions import (
    FileSizeError,
    FileTypeError,
    FileValidationError,
)


# Magic bytes for common image formats
MAGIC_BYTES = {
    "image/jpeg": [b"\xff\xd8\xff"],
    "image/png": [b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a"],
    "image/jpg": [b"\xff\xd8\xff"],  # Same as JPEG
}

# Additional magic bytes for python-magic fallback detection
# Some systems may return different MIME types
MAGIC_BYTES_FALLBACK = {
    b"\xff\xd8\xff": "image/jpeg",
    b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a": "image/png",
}

# Maximum file sizes
MAX_FILE_SIZE_BYTES = settings.max_upload_size_mb * 1024 * 1024


def _probe_magic_available() -> bool:
    """
    Probe whether ``python-magic`` can be imported and used without a
    native crash.

    Returns True only if the import AND the ``Magic(mime=True)``
    constructor both succeed. Returns False if the package is missing,
    the import fails, the constructor fails, or the subprocess is
    killed by a Windows access violation.

    The probe runs in a *subprocess* so a native crash in the child
    cannot kill the parent process.
    """
    if importlib.util.find_spec("magic") is None:
        return False
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import magic; magic.Magic(mime=True); print('OK')"],
            capture_output=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return False
    return result.returncode == 0 and b"OK" in result.stdout


# Run the probe once at module import time. The result is cached.
_MAGIC_AVAILABLE: bool = _probe_magic_available()


class FileValidator:
    """
    Secure file validator for uploaded images.

    Validates:
    - File size
    - MIME type (via python-magic, if available)
    - Magic bytes (file signature)
    - File extension
    """

    def __init__(
        self,
        max_size_bytes: int = MAX_FILE_SIZE_BYTES,
        allowed_mime_types: Optional[list[str]] = None,
        allowed_extensions: Optional[list[str]] = None,
    ):
        self.max_size_bytes = max_size_bytes
        self.allowed_mime_types = allowed_mime_types or settings.allowed_mime_types
        self.allowed_extensions = allowed_extensions or settings.allowed_extensions
        self._magic = None
        if _MAGIC_AVAILABLE:
            # We already proved in a subprocess that this is safe. If
            # somehow the import still fails in the parent (e.g. very
            # different Python builds), fall back to None silently.
            try:
                import magic  # noqa: PLC0415
                self._magic = magic.Magic(mime=True)
            except Exception:
                self._magic = None

    def validate_size(self, file_size: int) -> None:
        """Validate file size."""
        if file_size > self.max_size_bytes:
            raise FileSizeError(
                max_size_mb=settings.max_upload_size_mb,
                actual_size_mb=file_size / (1024 * 1024),
            )

    def validate_extension(self, filename: str) -> None:
        """Validate file extension."""
        suffix = Path(filename).suffix.lower()
        if suffix not in self.allowed_extensions:
            raise FileValidationError(
                message=f"File extension '{suffix}' is not allowed. Allowed: {', '.join(self.allowed_extensions)}",
                details={
                    "allowed_extensions": self.allowed_extensions,
                    "detected_extension": suffix,
                },
            )

    def validate_mime_type(self, mime_type: str) -> None:
        """Validate MIME type."""
        if mime_type not in self.allowed_mime_types:
            raise FileTypeError(
                allowed_types=self.allowed_mime_types,
                detected_type=mime_type,
            )

    def validate_magic_bytes(self, content: bytes) -> str:
        """
        Validate file magic bytes and return detected MIME type.

        Uses signature matching first, then python-magic for verification.
        """
        if len(content) < 12:
            raise FileValidationError(
                message="File too small to determine type",
                details={"min_bytes": 12},
            )

        # First, try to detect by magic bytes signatures
        detected_mime = None
        for mime, signatures in MAGIC_BYTES.items():
            if any(content.startswith(sig) for sig in signatures):
                detected_mime = mime
                break

        # If no signature match, use python-magic as fallback if available
        if detected_mime is None and self._magic is not None:
            try:
                detected_mime = self._magic.from_buffer(content)
            except Exception:
                detected_mime = None

        if detected_mime is None:
            # Check fallback dict
            for sig, mime in MAGIC_BYTES_FALLBACK.items():
                if content.startswith(sig):
                    detected_mime = mime
                    break

        if detected_mime in ("application/octet-stream", "application/x-empty", "", None):
            raise FileValidationError(
                message="File signature not recognized as valid image format",
                details={"detected_by_magic": detected_mime},
            )

        # Verify against allowed types
        if detected_mime not in self.allowed_mime_types:
            raise FileTypeError(
                allowed_types=self.allowed_mime_types,
                detected_type=detected_mime,
            )

        return detected_mime

    def validate(self, content: bytes, filename: str, declared_mime: Optional[str] = None) -> str:
        """
        Perform complete file validation.

        Args:
            content: File content bytes
            filename: Original filename
            declared_mime: MIME type from upload (optional, will be verified)

        Returns:
            Validated MIME type

        Raises:
            FileValidationError: If any validation fails
        """
        # 1. Size check
        self.validate_size(len(content))

        # 2. Extension check
        self.validate_extension(filename)

        # 3. Magic bytes + MIME detection
        detected_mime = self.validate_magic_bytes(content)

        # 4. Cross-check with declared MIME if provided
        if declared_mime and declared_mime != detected_mime:
            raise FileValidationError(
                message="Declared MIME type does not match detected type",
                details={
                    "declared_mime": declared_mime,
                    "detected_mime": detected_mime,
                },
            )

        return detected_mime


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent path traversal and injection.

    - Removes path traversal sequences (..)
    - Removes path separators
    - Removes dangerous characters (but preserves dots for extensions)
    - Limits length
    """
    # Remove path traversal sequences and separators first
    name = filename.replace("..", "").replace("/", "").replace("\\", "")

    # Remove dangerous characters (but keep dots for extensions)
    dangerous_chars = [':', '*', '?', '"', '<', '>', '|', '\0']
    for char in dangerous_chars:
        name = name.replace(char, '')

    # Handle edge case: only dots or empty
    if not name or name.strip('.') == '':
        name = 'upload'

    # Limit length
    max_len = 255
    if len(name) > max_len:
        # Try to preserve extension
        if '.' in name and name.rfind('.') > len(name) - 20:  # Extension within last 20 chars
            parts = name.rsplit('.', 1)
            stem = parts[0][:max_len - len(parts[1]) - 1]
            name = stem + '.' + parts[1]
        else:
            name = name[:max_len]

    # Ensure it's not empty
    if not name:
        name = 'upload'

    return name


def generate_secure_temp_path(suffix: str = ".tmp") -> Path:
    """
    Generate a secure temporary file path.

    Uses system temp directory with random name.
    """
    import tempfile
    import secrets

    # Create a random filename
    random_name = secrets.token_urlsafe(16) + suffix
    temp_dir = Path(tempfile.gettempdir())
    return temp_dir / random_name


# Global validator instance
file_validator = FileValidator()
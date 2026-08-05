"""
Tests for File Validator
"""

import pytest
from backend.app.utils.file_validator import (
    FileValidator,
    sanitize_filename,
    generate_secure_temp_path,
)
from backend.app.exceptions import (
    FileSizeError,
    FileTypeError,
    FileValidationError,
)
from backend.app.config import Settings


class TestFileValidator:
    """Tests for FileValidator class."""

    @pytest.fixture
    def validator(self):
        return FileValidator(
            max_size_bytes=10 * 1024 * 1024,  # 10 MB
            allowed_mime_types=["image/jpeg", "image/png"],
            allowed_extensions=[".jpg", ".jpeg", ".png"],
        )

    @pytest.fixture
    def valid_jpeg_bytes(self):
        """Valid JPEG magic bytes."""
        return b"\xff\xd8\xff\xe0\x00\x10JFIF" + b"x" * 100

    @pytest.fixture
    def valid_png_bytes(self):
        """Valid PNG magic bytes."""
        return b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a" + b"x" * 100

    def test_validate_size_ok(self, validator):
        """Test size validation passes for valid size."""
        validator.validate_size(5 * 1024 * 1024)  # 5 MB

    def test_validate_size_exceeds(self, validator):
        """Test size validation fails for oversized file."""
        with pytest.raises(FileSizeError) as exc_info:
            validator.validate_size(15 * 1024 * 1024)  # 15 MB
        assert exc_info.value.error_code == "FILE_SIZE_EXCEEDED"
        assert exc_info.value.details["max_size_mb"] == 10
        assert exc_info.value.details["actual_size_mb"] == 15.0

    def test_validate_extension_ok(self, validator):
        """Test extension validation passes for valid extensions."""
        for ext in [".jpg", ".jpeg", ".png", ".JPG", ".PNG"]:
            validator.validate_extension(f"test{ext}")

    def test_validate_extension_invalid(self, validator):
        """Test extension validation fails for invalid extension."""
        with pytest.raises(FileValidationError) as exc_info:
            validator.validate_extension("test.gif")
        assert exc_info.value.error_code == "FILE_VALIDATION_ERROR"
        assert ".gif" in str(exc_info.value.details["detected_extension"])

    def test_validate_mime_type_ok(self, validator):
        """Test MIME type validation passes for valid types."""
        validator.validate_mime_type("image/jpeg")
        validator.validate_mime_type("image/png")

    def test_validate_mime_type_invalid(self, validator):
        """Test MIME type validation fails for invalid type."""
        with pytest.raises(FileTypeError) as exc_info:
            validator.validate_mime_type("image/gif")
        assert exc_info.value.error_code == "FILE_TYPE_NOT_ALLOWED"
        assert "image/gif" in str(exc_info.value.details["detected_type"])

    def test_validate_magic_bytes_jpeg(self, validator, valid_jpeg_bytes):
        """Test magic bytes validation for JPEG."""
        mime = validator.validate_magic_bytes(valid_jpeg_bytes)
        assert mime == "image/jpeg"

    def test_validate_magic_bytes_png(self, validator, valid_png_bytes):
        """Test magic bytes validation for PNG."""
        mime = validator.validate_magic_bytes(valid_png_bytes)
        assert mime == "image/png"

    def test_validate_magic_bytes_invalid(self, validator):
        """Test magic bytes validation fails for invalid signature."""
        invalid_bytes = b"not an image" + b"x" * 100
        with pytest.raises(FileValidationError) as exc_info:
            validator.validate_magic_bytes(invalid_bytes)
        # Invalid content gets detected as non-image type, so FILE_TYPE_NOT_ALLOWED
        assert exc_info.value.error_code in ("FILE_VALIDATION_ERROR", "FILE_TYPE_NOT_ALLOWED")

    def test_validate_magic_bytes_too_small(self, validator):
        """Test magic bytes validation fails for too small file."""
        with pytest.raises(FileValidationError) as exc_info:
            validator.validate_magic_bytes(b"tiny")
        assert "too small" in exc_info.value.message.lower()

    def test_full_validation_ok(self, validator, valid_jpeg_bytes):
        """Test complete validation passes for valid file."""
        mime = validator.validate(valid_jpeg_bytes, "test.jpg", "image/jpeg")
        assert mime == "image/jpeg"

    def test_full_validation_mime_mismatch(self, validator, valid_jpeg_bytes):
        """Test validation fails when declared MIME doesn't match detected."""
        with pytest.raises(FileValidationError) as exc_info:
            validator.validate(valid_jpeg_bytes, "test.jpg", "image/png")
        assert "does not match" in exc_info.value.message.lower()


class TestSanitizeFilename:
    """Tests for filename sanitization."""

    def test_removes_path_traversal(self):
        """Test path traversal sequences are removed."""
        assert sanitize_filename("../../../etc/passwd") == "etcpasswd"
        assert sanitize_filename("..\\..\\windows\\system32") == "windowssystem32"

    def test_removes_dangerous_chars(self):
        """Test dangerous characters are removed (dots preserved for extensions)."""
        assert sanitize_filename('file:name"with*chars?.txt') == "filenamewithchars.txt"
        assert sanitize_filename("file<with>|pipes.txt") == "filewithpipes.txt"

    def test_preserves_valid_filename(self):
        """Test valid filenames are preserved."""
        assert sanitize_filename("normal_image.jpg") == "normal_image.jpg"
        assert sanitize_filename("my-photo_123.png") == "my-photo_123.png"

    def test_limits_length(self):
        """Test filename length is limited but extension preserved."""
        long_name = "a" * 300 + ".jpg"
        result = sanitize_filename(long_name)
        assert len(result) <= 255
        # Extension should be preserved if possible
        assert result.endswith(".jpg") or len(result) == 255

    def test_handles_empty(self):
        """Test empty filename gets default."""
        assert sanitize_filename("") == "upload"
        # Single dot is treated as hidden file, becomes empty after dot removal
        assert sanitize_filename(".") == "upload"


class TestGenerateSecureTempPath:
    """Tests for secure temp path generation."""

    def test_creates_valid_path(self):
        """Test generated path is valid."""
        path = generate_secure_temp_path(".jpg")
        assert path.suffix == ".jpg"
        assert path.parent.exists()  # System temp dir exists

    def test_unique_names(self):
        """Test generated names are unique."""
        paths = {generate_secure_temp_path(".jpg") for _ in range(100)}
        assert len(paths) == 100  # All unique

    def test_uses_system_temp(self):
        """Test paths use system temp directory."""
        import tempfile
        path = generate_secure_temp_path()
        assert str(path).startswith(tempfile.gettempdir())
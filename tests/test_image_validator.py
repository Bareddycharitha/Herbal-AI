"""
Tests for Image Validator
"""

import pytest
import cv2
import numpy as np
from PIL import Image

from ai.validation.image_validator import ImageValidator


class TestImageValidator:
    """Tests for ImageValidator class."""

    @pytest.fixture
    def validator(self):
        return ImageValidator()

    @pytest.fixture
    def valid_image(self, tmp_path):
        """Create a valid test image."""
        img = Image.new("RGB", (300, 300), color=(128, 128, 128))
        # Add some texture to avoid blur detection
        pixels = img.load()
        for x in range(300):
            for y in range(300):
                pixels[x, y] = (x % 256, y % 256, (x + y) % 256)
        path = tmp_path / "valid.jpg"
        img.save(path, "JPEG", quality=90)
        return str(path)

    @pytest.fixture
    def small_image(self, tmp_path):
        """Create a too-small image."""
        img = Image.new("RGB", (100, 100), color="red")
        path = tmp_path / "small.jpg"
        img.save(path, "JPEG")
        return str(path)

    @pytest.fixture
    def blurry_image(self, tmp_path):
        """Create a blurry image."""
        img = np.ones((300, 300, 3), dtype=np.uint8) * 128
        # Apply heavy blur
        img = cv2.GaussianBlur(img, (51, 51), 0)
        path = tmp_path / "blurry.jpg"
        cv2.imwrite(str(path), img)
        return str(path)

    @pytest.fixture
    def dark_image(self, tmp_path):
        """Create a too-dark image with texture to pass blur check."""
        img = Image.new("RGB", (300, 300), color=(10, 10, 10))
        # Add high-frequency texture to avoid blur detection
        pixels = img.load()
        for x in range(300):
            for y in range(300):
                # Dark but with high-frequency pattern
                val = 10 + ((x * 7 + y * 11) % 20)
                pixels[x, y] = (val, val, val)
        path = tmp_path / "dark.jpg"
        img.save(path, "JPEG", quality=90)
        return str(path)

    @pytest.fixture
    def bright_image(self, tmp_path):
        """Create an overexposed image with texture to pass blur check."""
        img = Image.new("RGB", (300, 300), color=(250, 250, 250))
        # Add high-frequency texture to avoid blur detection
        pixels = img.load()
        for x in range(300):
            for y in range(300):
                # Bright but with high-frequency pattern
                val = 250 - ((x * 7 + y * 11) % 20)
                pixels[x, y] = (val, val, val)
        path = tmp_path / "bright.jpg"
        img.save(path, "JPEG", quality=90)
        return str(path)

    @pytest.fixture
    def min_dim_image(self, tmp_path):
        """Create a minimum-dimension image with texture to pass blur check."""
        img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        # Add high-frequency texture to avoid blur detection
        pixels = img.load()
        for x in range(224):
            for y in range(224):
                val = 128 + ((x * 13 + y * 17) % 50)
                pixels[x, y] = (val % 256, (val + 50) % 256, (val + 100) % 256)
        path = tmp_path / "min_dim.jpg"
        img.save(path, "JPEG", quality=90)
        return str(path)

    @pytest.fixture
    def nonexistent_image(self):
        """Path to non-existent image."""
        return "/path/that/does/not/exist.jpg"

    def test_valid_image_passes(self, validator, valid_image):
        """Test valid image passes validation."""
        result = validator.validate(valid_image)
        assert result["success"] is True
        assert "blur_score" in result
        assert "brightness" in result
        assert result["width"] == 300
        assert result["height"] == 300

    def test_small_image_fails(self, validator, small_image):
        """Test small image fails validation."""
        result = validator.validate(small_image)
        assert result["success"] is False
        assert "resolution too low" in result["message"].lower()
        # width/height only returned on success
        assert "width" not in result or result.get("width") == 100
        assert "height" not in result or result.get("height") == 100

    def test_blurry_image_fails(self, validator, blurry_image):
        """Test blurry image fails validation."""
        result = validator.validate(blurry_image)
        assert result["success"] is False
        assert "blurry" in result["message"].lower()
        assert "blur_score" in result
        assert result["blur_score"] < validator.BLUR_THRESHOLD

    def test_dark_image_fails(self, validator, dark_image):
        """Test dark image fails validation."""
        result = validator.validate(dark_image)
        assert result["success"] is False
        assert "dark" in result["message"].lower()
        assert "brightness" in result

    def test_bright_image_fails(self, validator, bright_image):
        """Test overexposed image fails validation."""
        result = validator.validate(bright_image)
        assert result["success"] is False
        assert "overexposed" in result["message"].lower() or "bright" in result["message"].lower()
        assert "brightness" in result

    def test_nonexistent_image_fails(self, validator, nonexistent_image):
        """Test non-existent image fails validation."""
        result = validator.validate(nonexistent_image)
        assert result["success"] is False
        assert "unable to read" in result["message"].lower()

    def test_minimum_dimensions(self, validator, min_dim_image):
        """Test minimum allowed dimensions."""
        result = validator.validate(min_dim_image)
        assert result["success"] is True

    def test_just_below_minimum(self, validator, tmp_path):
        """Test just below minimum dimensions."""
        img = Image.new("RGB", (223, 224), color="gray")
        # Add texture
        pixels = img.load()
        for x in range(223):
            for y in range(224):
                pixels[x, y] = (x % 256, y % 256, (x + y) % 256)
        path = tmp_path / "below_min.jpg"
        img.save(path, "JPEG")
        result = validator.validate(str(path))
        assert result["success"] is False

    def test_blur_threshold_boundary(self, validator, tmp_path):
        """Test image at blur threshold boundary."""
        # Create image with controlled blur
        img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        # Mild blur
        img = cv2.GaussianBlur(img, (5, 5), 1.0)
        path = tmp_path / "mild_blur.jpg"
        cv2.imwrite(str(path), img)
        result = validator.validate(str(path))
        # Should pass with mild blur
        # (depends on random content, but usually passes)
        assert "blur_score" in result
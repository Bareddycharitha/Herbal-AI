"""
Basic Image Validator

Validates image quality: resolution, blur, brightness.
"""

import cv2
import numpy as np


class ImageValidator:

    MIN_WIDTH = 224
    MIN_HEIGHT = 224

    # Blur threshold (Laplacian variance) - calibrated for skin/leaf images
    # Higher = stricter (requires sharper images)
    BLUR_THRESHOLD = 30.0

    # Brightness range (0-255)
    MIN_BRIGHTNESS = 20
    MAX_BRIGHTNESS = 240

    def validate(self, image_path):

        image = cv2.imread(image_path)

        if image is None:
            return {
                "success": False,
                "message": "Unable to read the uploaded image."
            }

        height, width = image.shape[:2]

        if width < self.MIN_WIDTH or height < self.MIN_HEIGHT:
            return {
                "success": False,
                "message": f"Image resolution too low ({width}x{height}). Minimum: {self.MIN_WIDTH}x{self.MIN_HEIGHT}."
            }

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Blur detection using Laplacian variance
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

        if blur_score < self.BLUR_THRESHOLD:
            return {
                "success": False,
                "message": f"Image is too blurry (score: {blur_score:.1f}). Please upload a clearer image.",
                "blur_score": round(blur_score, 2),
                "threshold": self.BLUR_THRESHOLD,
            }

        brightness = np.mean(gray)

        if brightness < self.MIN_BRIGHTNESS:
            return {
                "success": False,
                "message": f"Image is too dark (brightness: {brightness:.1f}).",
                "brightness": round(brightness, 2),
            }

        if brightness > self.MAX_BRIGHTNESS:
            return {
                "success": False,
                "message": f"Image is overexposed (brightness: {brightness:.1f}).",
                "brightness": round(brightness, 2),
            }

        return {
            "success": True,
            "message": "Image validation passed.",
            "blur_score": round(blur_score, 2),
            "brightness": round(brightness, 2),
            "width": width,
            "height": height,
        }
import cv2
import numpy as np


class MedicalImageValidator:

    def __init__(self):

        self.MIN_SKIN_RATIO = 0.18
        self.MAX_EDGE_RATIO = 0.25

    def validate(self, image_path):

        image = cv2.imread(image_path)

        if image is None:

            return {
                "success": False,
                "message": "Unable to read image."
            }

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower = np.array([0, 20, 50], dtype=np.uint8)
        upper = np.array([25, 255, 255], dtype=np.uint8)

        skin_mask = cv2.inRange(
            hsv,
            lower,
            upper
        )

        skin_ratio = np.count_nonzero(
            skin_mask
        ) / skin_mask.size

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        edges = cv2.Canny(
            gray,
            100,
            200
        )

        edge_ratio = np.count_nonzero(
            edges
        ) / edges.size

        if skin_ratio < self.MIN_SKIN_RATIO:

            return {

                "success": False,

                "message":
                "The uploaded image does not appear to contain sufficient visible skin.",

                "skin_ratio":
                round(skin_ratio * 100, 2),

                "edge_ratio":
                round(edge_ratio * 100, 2)
            }

        if edge_ratio > self.MAX_EDGE_RATIO:

            return {

                "success": False,

                "message":
                "The uploaded image appears to contain excessive background or unrelated objects.",

                "skin_ratio":
                round(skin_ratio * 100, 2),

                "edge_ratio":
                round(edge_ratio * 100, 2)
            }

        return {

            "success": True,

            "message":
            "Medical image validation passed.",

            "skin_ratio":
            round(skin_ratio * 100, 2),

            "edge_ratio":
            round(edge_ratio * 100, 2)
        }
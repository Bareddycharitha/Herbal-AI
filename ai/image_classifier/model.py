import torch.nn as nn
import timm

from .config import (
    MODEL_NAME,
    NUM_CLASSES,
    PRETRAINED,
)


class UniversalImageClassifier(nn.Module):
    """
    Universal Image Classifier

    Classes:
        0 -> Skin
        1 -> Medicinal
        2 -> Other
    """

    def __init__(self):

        super().__init__()

        # Load pretrained EfficientNetV2
        self.backbone = timm.create_model(
            MODEL_NAME,
            pretrained=PRETRAINED,
            num_classes=NUM_CLASSES,
        )

    def forward(self, x):
        return self.backbone(x)


def build_model():
    """
    Returns a Universal Image Classifier.
    """

    return UniversalImageClassifier()
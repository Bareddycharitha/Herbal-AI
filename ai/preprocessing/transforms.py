import albumentations as A
from albumentations.pytorch import ToTensorV2

from ai.config import IMAGE_SIZE

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def get_train_transforms():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),

        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),

        # Equivalent conservative geometry without ShiftScaleRotate warning.
        A.Affine(
            translate_percent=(-0.05, 0.05),
            scale=(0.90, 1.10),
            rotate=(-15, 15),
            border_mode=0,
            p=0.4,
        ),

        # Preserve medically relevant color information.
        A.RandomBrightnessContrast(
            brightness_limit=0.10,
            contrast_limit=0.10,
            p=0.3,
        ),

        A.GaussNoise(
            std_range=(0.01, 0.04),
            p=0.10,
        ),

        A.GaussianBlur(
            blur_limit=(3, 3),
            p=0.10,
        ),

        A.Normalize(
            mean=MEAN,
            std=STD,
        ),

        ToTensorV2(),
    ])


def get_valid_transforms():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),

        A.Normalize(
            mean=MEAN,
            std=STD,
        ),

        ToTensorV2(),
    ])

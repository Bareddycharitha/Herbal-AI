import albumentations as A
from albumentations.pytorch import ToTensorV2

from ai.config import IMAGE_SIZE

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def get_train_transforms():

    return A.Compose([

        A.Resize(IMAGE_SIZE, IMAGE_SIZE),

        # -----------------------------
        # Geometric Augmentations
        # -----------------------------

        A.HorizontalFlip(p=0.5),

        A.Rotate(
            limit=20,
            border_mode=0,
            p=0.5
        ),

        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.10,
            rotate_limit=10,
            border_mode=0,
            p=0.4
        ),

        # -----------------------------
        # Lighting Augmentations
        # -----------------------------

        A.RandomBrightnessContrast(
            brightness_limit=0.25,
            contrast_limit=0.25,
            p=0.5
        ),

        A.HueSaturationValue(
            hue_shift_limit=8,
            sat_shift_limit=15,
            val_shift_limit=10,
            p=0.4
        ),

        A.RandomGamma(
            gamma_limit=(80, 120),
            p=0.3
        ),

        # -----------------------------
        # Camera Noise
        # -----------------------------

        A.GaussNoise(
            std_range=(0.02, 0.08),
            p=0.25
        ),

        A.GaussianBlur(
            blur_limit=(3, 5),
            p=0.2
        ),

        # -----------------------------
        # Contrast Enhancement
        # -----------------------------

        A.CLAHE(
            clip_limit=2,
            tile_grid_size=(8, 8),
            p=0.3
        ),

        # -----------------------------
        # Partial Occlusion
        # -----------------------------

        A.CoarseDropout(
            num_holes_range=(1, 4),
            hole_height_range=(16, 32),
            hole_width_range=(16, 32),
            fill=0,
            p=0.2
        ),

        # -----------------------------
        # Normalize
        # -----------------------------

        A.Normalize(
            mean=MEAN,
            std=STD
        ),

        ToTensorV2()

    ])


def get_valid_transforms():

    return A.Compose([

        A.Resize(
            IMAGE_SIZE,
            IMAGE_SIZE
        ),

        A.Normalize(
            mean=MEAN,
            std=STD
        ),

        ToTensorV2()

    ])
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
import numpy as np
from PIL import Image

from .config import IMAGE_SIZE

# ==========================================================
# ImageNet Statistics
# ==========================================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ==========================================================
# Custom Augmentations
# ==========================================================

class RandomErasing:
    """Random Erasing (Cutout) augmentation."""

    def __init__(self, p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value

    def __call__(self, img):
        if random.random() > self.p:
            return img

        if isinstance(img, torch.Tensor):
            # Tensor input (C, H, W)
            c, h, w = img.shape
        else:
            # PIL Image
            w, h = img.size
            c = 3

        area = h * w
        target_area = random.uniform(*self.scale) * area
        aspect_ratio = random.uniform(*self.ratio)

        new_h = int(round((target_area * aspect_ratio) ** 0.5))
        new_w = int(round((target_area / aspect_ratio) ** 0.5))

        if new_h < h and new_w < w:
            top = random.randint(0, h - new_h)
            left = random.randint(0, w - new_w)

            if isinstance(img, torch.Tensor):
                img[:, top:top+new_h, left:left+new_w] = self.value
            else:
                # For PIL, we need to convert to tensor first or use ImageDraw
                pass  # Skip for PIL, use tensor version

        return img


class GaussianBlur:
    """Gaussian Blur augmentation."""

    def __init__(self, p=0.5, kernel_size=21, sigma=(0.1, 2.0)):
        self.p = p
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, img):
        if random.random() > self.p:
            return img

        if isinstance(img, torch.Tensor):
            # Use torchvision functional
            sigma = random.uniform(*self.sigma)
            return TF.gaussian_blur(img, kernel_size=self.kernel_size, sigma=(sigma, sigma))
        else:
            # For PIL
            from PIL import ImageFilter
            radius = random.uniform(*self.sigma)
            return img.filter(ImageFilter.GaussianBlur(radius=radius))


class Solarize:
    """Solarize augmentation."""

    def __init__(self, p=0.2, threshold=128):
        self.p = p
        self.threshold = threshold

    def __call__(self, img):
        if random.random() > self.p:
            return img

        if isinstance(img, torch.Tensor):
            # Apply solarize to tensor
            return torch.where(img > self.threshold/255.0, 1.0 - img, img)
        else:
            from PIL import ImageOps
            return ImageOps.solarize(img, threshold=self.threshold)


class Posterize:
    """Posterize augmentation."""

    def __init__(self, p=0.2, bits=4):
        self.p = p
        self.bits = bits

    def __call__(self, img):
        if random.random() > self.p:
            return img

        if isinstance(img, torch.Tensor):
            # Quantize tensor
            levels = 2 ** self.bits
            return torch.round(img * (levels - 1)) / (levels - 1)
        else:
            from PIL import ImageOps
            return ImageOps.posterize(img, bits=self.bits)


class AutoAugmentPolicy:
    """
    Simplified AutoAugment policy for medical images.
    """

    def __init__(self):
        self.policies = [
            # (op1, prob1, mag1, op2, prob2, mag2)
            ('Rotate', 0.7, 5, 'Color', 0.3, 0.9),
            ('ShearX', 0.5, 0.2, 'Brightness', 0.5, 0.9),
            ('TranslateX', 0.6, 0.1, 'Contrast', 0.4, 0.9),
            ('Rotate', 0.7, 5, 'Sharpness', 0.3, 0.9),
            ('ShearY', 0.5, 0.2, 'Color', 0.5, 0.9),
        ]

    def __call__(self, img):
        policy = random.choice(self.policies)
        for op_name, prob, mag in zip(policy[::2], policy[1::2], policy[2::2]):
            if random.random() < prob:
                img = self._apply_op(img, op_name, mag)
        return img

    def _apply_op(self, img, op_name, magnitude):
        """Apply augmentation operation."""
        if isinstance(img, torch.Tensor):
            img = TF.to_pil_image(img)
            to_tensor = True
        else:
            to_tensor = False

        # Apply operation
        if op_name == 'Rotate':
            img = img.rotate(magnitude * 30)
        elif op_name == 'ShearX':
            img = TF.affine(img, angle=0, translate=(0, 0), scale=1, shear=(magnitude, 0))
        elif op_name == 'ShearY':
            img = TF.affine(img, angle=0, translate=(0, 0), scale=1, shear=(0, magnitude))
        elif op_name == 'TranslateX':
            w, h = img.size
            img = TF.affine(img, angle=0, translate=(int(magnitude * w), 0), scale=1, shear=(0, 0))
        elif op_name == 'TranslateY':
            w, h = img.size
            img = TF.affine(img, angle=0, translate=(0, int(magnitude * h)), scale=1, shear=(0, 0))
        elif op_name == 'Color':
            img = TF.adjust_saturation(img, magnitude)
        elif op_name == 'Brightness':
            img = TF.adjust_brightness(img, magnitude)
        elif op_name == 'Contrast':
            img = TF.adjust_contrast(img, magnitude)
        elif op_name == 'Sharpness':
            img = TF.adjust_sharpness(img, magnitude)

        if to_tensor:
            img = TF.to_tensor(img)

        return img


# ==========================================================
# MixUp & CutMix
# ==========================================================

def mixup_data(x, y, alpha=1.0):
    """
    MixUp augmentation.

    Args:
        x: Input batch [B, C, H, W]
        y: Labels [B]
        alpha: Beta distribution parameter

    Returns:
        mixed_x, y_a, y_b, lam
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute MixUp loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def cutmix_data(x, y, alpha=1.0):
    """
    CutMix augmentation.

    Args:
        x: Input batch [B, C, H, W]
        y: Labels [B]
        alpha: Beta distribution parameter

    Returns:
        mixed_x, y_a, y_b, lam
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    # Get random bounding box
    h, w = x.size(2), x.size(3)
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)

    cx = np.random.randint(w)
    cy = np.random.randint(h)

    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)

    x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]

    # Adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))

    y_a, y_b = y, y[index]

    return x, y_a, y_b, lam


# ==========================================================
# Training Transform
# ==========================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomRotation(degrees=15),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05,
    ),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05),
        scale=(0.95, 1.05),
    ),

    # Advanced augmentations (applied with some probability)
    transforms.RandomApply([
        GaussianBlur(p=1.0, kernel_size=21, sigma=(0.1, 2.0)),
    ], p=0.3),

    transforms.RandomApply([
        Solarize(p=1.0, threshold=128),
    ], p=0.1),

    transforms.RandomApply([
        Posterize(p=1.0, bits=4),
    ], p=0.1),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    ),

    # Random Erasing (Cutout) - applied after normalization
    transforms.RandomErasing(
        p=0.3,
        scale=(0.02, 0.2),
        ratio=(0.3, 3.3),
        value=0,
    ),
])


# ==========================================================
# Validation / Test Transform
# ==========================================================

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    ),
])


# ==========================================================
# TTA Transforms (for inference)
# ==========================================================

def get_tta_transforms():
    """
    Returns list of transforms for Test-Time Augmentation.
    """
    base_transform = test_transform

    tta_transforms = [
        base_transform,  # Original
        transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]),
        transforms.Compose([
            transforms.Resize((int(IMAGE_SIZE * 1.1), int(IMAGE_SIZE * 1.1))),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]),
        transforms.Compose([
            transforms.Resize((int(IMAGE_SIZE * 0.9), int(IMAGE_SIZE * 0.9))),
            transforms.Lambda(lambda img: transforms.functional.pad(img, (11, 11, 12, 12))),  # left, top, right, bottom
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]),
        transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]),
    ]

    return tta_transforms
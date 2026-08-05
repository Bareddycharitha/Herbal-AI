"""
Extract a balanced 'Other Objects' dataset from COCO 2017.

Author: Herbal-AI
"""

import json
import random
import shutil
from pathlib import Path
from collections import defaultdict

# =====================================================
# CONFIGURATION
# =====================================================

random.seed(42)

COCO_ROOT = Path("ai/datasets/coco")

COCO_IMAGES = COCO_ROOT / "train2017"
COCO_ANNOTATIONS = COCO_ROOT / "annotations" / "instances_train2017.json"

OUTPUT_DIR = Path("ai/datasets/OtherObjects")

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

MAX_IMAGES_PER_CATEGORY = 300

CATEGORIES = [
    "laptop",
    "cell phone",
    "keyboard",
    "mouse",
    "tv",
    "book",
    "bottle",
    "cup",
    "chair",
    "couch",
    "car",
    "bicycle",
    "dog",
    "cat",
    "bird",
]

# =====================================================
# FUNCTIONS
# =====================================================

def create_output_folders():
    """Create fresh train/val/test folders."""

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    for split in ["train", "val", "test"]:
        (OUTPUT_DIR / split).mkdir(parents=True, exist_ok=True)


def load_coco():
    """Load COCO annotation file."""

    print("\nLoading COCO annotations...")

    with open(COCO_ANNOTATIONS, "r") as f:
        coco = json.load(f)

    print("COCO annotations loaded.\n")

    return coco


def get_category_lookup(coco):
    """Return category name -> category id."""

    return {
        category["name"]: category["id"]
        for category in coco["categories"]
    }


def get_image_lookup(coco):
    """Return image id -> filename."""

    return {
        image["id"]: image["file_name"]
        for image in coco["images"]
    }


def collect_images(coco, category_lookup):
    """
    Collect image ids for selected categories.
    """

    category_images = defaultdict(set)

    for annotation in coco["annotations"]:

        category_id = annotation["category_id"]

        for name in CATEGORIES:

            if category_lookup[name] == category_id:

                category_images[name].add(annotation["image_id"])

    return category_images


def sample_images(category_images):
    """
    Randomly sample images from each category.
    """

    selected_images = set()

    print("=" * 55)
    print("Category Summary")
    print("=" * 55)

    for category in CATEGORIES:

        images = list(category_images[category])

        random.shuffle(images)

        images = images[:MAX_IMAGES_PER_CATEGORY]

        selected_images.update(images)

        print(f"{category:<20} {len(images):>5}")

    print("=" * 55)

    return list(selected_images)


def split_dataset(images):

    random.shuffle(images)

    total = len(images)

    train_end = int(total * TRAIN_SPLIT)
    val_end = train_end + int(total * VAL_SPLIT)

    train = images[:train_end]
    val = images[train_end:val_end]
    test = images[val_end:]

    return train, val, test


def copy_images(image_ids, split, image_lookup):

    destination = OUTPUT_DIR / split

    copied = 0

    for image_id in image_ids:

        filename = image_lookup.get(image_id)

        if filename is None:
            continue

        source = COCO_IMAGES / filename

        if source.exists():

            shutil.copy2(source, destination / filename)

            copied += 1

    print(f"{split:<8}: {copied} images copied")


# =====================================================
# MAIN
# =====================================================

def main():

    create_output_folders()

    coco = load_coco()

    category_lookup = get_category_lookup(coco)

    image_lookup = get_image_lookup(coco)

    category_images = collect_images(coco, category_lookup)

    selected_images = sample_images(category_images)

    print(f"\nTotal unique images: {len(selected_images)}")

    train, val, test = split_dataset(selected_images)

    print("\nDataset Split")
    print("-" * 30)
    print(f"Train : {len(train)}")
    print(f"Val   : {len(val)}")
    print(f"Test  : {len(test)}")
    print("-" * 30)

    print("\nCopying images...\n")

    copy_images(train, "train", image_lookup)
    copy_images(val, "val", image_lookup)
    copy_images(test, "test", image_lookup)

    print("\nDataset created successfully!")
    print(f"\nSaved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
import os
from pathlib import Path
from PIL import Image
from collections import Counter
import matplotlib.pyplot as plt

# ======================================================
# Dataset Paths (Automatically Detect Project Location)
# ======================================================

BASE_DIR = Path(__file__).resolve().parents[2]

TRAIN_PATH = BASE_DIR / "ai" / "datasets" / "SkinDisease" / "train"
TEST_PATH = BASE_DIR / "ai" / "datasets" / "SkinDisease" / "test"

# ======================================================
# Analyze Dataset
# ======================================================

def analyze_dataset(dataset_path):

    if not dataset_path.exists():
        print(f"\n❌ Dataset not found:\n{dataset_path}")
        return None

    total_images = 0
    class_counts = Counter()
    image_sizes = Counter()
    image_formats = Counter()
    corrupted = []

    classes = sorted(os.listdir(dataset_path))

    print("=" * 70)
    print(f"Analyzing Dataset : {dataset_path}")
    print("=" * 70)

    for cls in classes:

        class_path = dataset_path / cls

        if not class_path.is_dir():
            continue

        images = list(class_path.iterdir())

        class_counts[cls] = len(images)
        total_images += len(images)

        for img_path in images:

            try:
                with Image.open(img_path) as image:
                    image_sizes[image.size] += 1
                    image_formats[image.format] += 1

            except Exception:
                corrupted.append(str(img_path))

    print(f"\nTotal Classes : {len(class_counts)}")
    print(f"Total Images  : {total_images}")

    print("\n" + "=" * 70)
    print("Images Per Class")
    print("=" * 70)

    for cls, count in class_counts.items():
        print(f"{cls:<30} {count}")

    print("\n" + "=" * 70)
    print("Image Formats")
    print("=" * 70)
    print(image_formats)

    print("\n" + "=" * 70)
    print("Top Image Sizes")
    print("=" * 70)

    for size, count in image_sizes.most_common(10):
        print(f"{size} : {count}")

    print("\n" + "=" * 70)
    print("Corrupted Images")
    print("=" * 70)
    print(len(corrupted))

    if corrupted:
        print("\nFirst 10 Corrupted Images:")
        for img in corrupted[:10]:
            print(img)

    return class_counts


# ======================================================
# Main
# ======================================================

if __name__ == "__main__":

    train_counts = analyze_dataset(TRAIN_PATH)

    if train_counts:

        plt.figure(figsize=(16, 7))

        plt.bar(train_counts.keys(), train_counts.values())

        plt.xticks(rotation=90)

        plt.xlabel("Disease Classes")

        plt.ylabel("Number of Images")

        plt.title("Skin Disease Training Dataset Distribution")

        plt.tight_layout()

        plt.show()
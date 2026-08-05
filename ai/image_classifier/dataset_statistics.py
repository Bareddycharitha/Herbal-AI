from pathlib import Path
from PIL import Image

# ==========================================================
# CONFIGURATION
# ==========================================================

DATASETS = {
    "SkinDisease": Path("ai/datasets/SkinDisease"),
    "Medicinal_Leaf": Path("ai/datasets/Medicinal_Leaf_dataset"),
    "Medicinal_Plant": Path("ai/datasets/Medicinal_plant_dataset"),
    "OtherObjects": Path("ai/datasets/OtherObjects"),
}

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}

# ==========================================================


def is_image(file: Path):
    return file.suffix.lower() in IMAGE_EXTENSIONS


def check_image(image_path: Path):
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def analyze_dataset(name: str, dataset_path: Path):

    print("\n" + "=" * 80)
    print(f"{name}")
    print("=" * 80)

    if not dataset_path.exists():
        print("Dataset not found.")
        return

    total_images = 0
    corrupted = 0

    for split in ["train", "val", "test"]:

        split_path = dataset_path / split

        if not split_path.exists():
            print(f"\n{split.upper()} : Not Found")
            continue

        print(f"\n{split.upper()}")

        class_dirs = sorted(
            [d for d in split_path.iterdir() if d.is_dir()]
        )

        split_total = 0

        for class_dir in class_dirs:

            images = [
                img
                for img in class_dir.iterdir()
                if img.is_file() and is_image(img)
            ]

            bad = 0

            for img in images:
                if not check_image(img):
                    bad += 1

            corrupted += bad
            count = len(images)

            split_total += count
            total_images += count

            print(f"{class_dir.name:<25} {count:>5}")

        print("-" * 40)
        print(f"Total {split:<5}: {split_total}")

    print("\n" + "-" * 40)
    print(f"Total Images : {total_images}")
    print(f"Corrupted    : {corrupted}")


def main():

    for name, path in DATASETS.items():
        analyze_dataset(name, path)


if __name__ == "__main__":
    main()
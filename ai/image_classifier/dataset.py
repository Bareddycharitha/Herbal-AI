from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


# ==========================================================
# Class Mapping
# ==========================================================

LABEL_MAPPING = {
    "skin": 0,
    "medicinal": 1,
    "other": 2,
}

LABEL_NAMES = {
    0: "Skin",
    1: "Medicinal",
    2: "Other",
}

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ==========================================================
# Dataset
# ==========================================================

class UniversalImageDataset(Dataset):

    def __init__(
        self,
        dataset_dirs,
        transform=None,
    ):

        self.dataset_dirs = [Path(d) for d in dataset_dirs]
        self.transform = transform

        self.samples = []

        self._collect_images()

        print(f"Loaded {len(self.samples)} images.")

    # ------------------------------------------------------

    def _collect_images(self):

        for dataset_dir in self.dataset_dirs:

            dataset_name = dataset_dir.parent.name.lower()

            # ----------------------------------------------
            # Assign label based on dataset
            # ----------------------------------------------

            if "skin" in dataset_name:
                label = LABEL_MAPPING["skin"]

            elif "medicinal_leaf" in dataset_name:
                label = LABEL_MAPPING["medicinal"]

            elif "medicinal_plant" in dataset_name:
                label = LABEL_MAPPING["medicinal"]

            elif "otherobjects" in dataset_name:
                label = LABEL_MAPPING["other"]

            else:
                print(f"Skipping unknown dataset : {dataset_name}")
                continue

            # ----------------------------------------------
            # CASE 1:
            # Images are directly inside dataset folder
            # Example:
            # OtherObjects/test/*.jpg
            # ----------------------------------------------

            direct_images = False

            for item in dataset_dir.iterdir():

                if (
                    item.is_file()
                    and item.suffix.lower() in IMAGE_EXTENSIONS
                ):
                    self.samples.append((item, label))
                    direct_images = True

            if direct_images:
                continue

            # ----------------------------------------------
            # CASE 2:
            # Images are inside class folders
            # Example:
            # SkinDisease/train/acne/*.jpg
            # ----------------------------------------------

            for class_dir in sorted(dataset_dir.iterdir()):

                if not class_dir.is_dir():
                    continue

                for image_path in class_dir.iterdir():

                    if (
                        image_path.is_file()
                        and image_path.suffix.lower() in IMAGE_EXTENSIONS
                    ):
                        self.samples.append((image_path, label))

    # ------------------------------------------------------

    def __len__(self):
        return len(self.samples)

    # ------------------------------------------------------

    def __getitem__(self, index):

        image_path, label = self.samples[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
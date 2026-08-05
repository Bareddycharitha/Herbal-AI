from ai.config import TRAIN_DIR
from ai.preprocessing.dataset import create_dataloaders
from ai.models.efficientnet import build_model


def main():

    train_loader, val_loader, class_names = create_dataloaders(TRAIN_DIR)

    model = build_model()

    print("=" * 50)
    print("Pipeline Loaded Successfully")
    print("=" * 50)

    print(f"Number of Classes : {len(class_names)}")
    print(f"Classes : {class_names}")

    images, labels = next(iter(train_loader))

    print(f"Image Batch Shape : {images.shape}")
    print(f"Label Batch Shape : {labels.shape}")

    print(type(model).__name__)


if __name__ == "__main__":
    main()
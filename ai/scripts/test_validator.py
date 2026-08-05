from pathlib import Path

from ai.validation.image_validator import ImageValidator

validator = ImageValidator()

image_path = Path(
    r"E:\Projects\Herbal-AI\ai\datasets\Medicinal_Leaf_dataset\Beans\1680.jpg"
)

result = validator.validate(str(image_path))

print(result)
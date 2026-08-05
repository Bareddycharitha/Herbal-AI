import timm
import torch.nn as nn

from ai.config import MODEL_NAME, PRETRAINED, NUM_CLASSES


def build_model(num_classes=None):
    """
    Build EfficientNetV2 model with custom classifier head.

    Args:
        num_classes: Number of output classes. If None, uses NUM_CLASSES from config.

    Returns:
        Model with modified classifier head
    """
    if num_classes is None:
        num_classes = NUM_CLASSES

    model = timm.create_model(
        MODEL_NAME,
        pretrained=PRETRAINED,
        num_classes=num_classes,  # timm handles this automatically for some models
    )

    # For EfficientNetV2, the classifier is typically model.classifier
    # But timm's create_model with num_classes should handle it
    # Let's verify and ensure proper head replacement

    if hasattr(model, 'classifier'):
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )
    elif hasattr(model, 'head'):
        # Some models use 'head'
        in_features = model.head.in_features
        model.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )
    elif hasattr(model, 'fc'):
        # ResNet style
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )
    else:
        # Try to find the final linear layer
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and module.out_features == 1000:
                # This is likely the classifier
                in_features = module.in_features
                parent_name = name.rsplit('.', 1)[0]
                parent = dict(model.named_modules())[parent_name]
                child_name = name.rsplit('.', 1)[1]
                setattr(parent, child_name, nn.Sequential(
                    nn.Dropout(0.3),
                    nn.Linear(in_features, num_classes)
                ))
                break

    return model


def build_binary_model():
    """Build model for binary classification (Healthy vs Diseased)."""
    return build_model(num_classes=2)
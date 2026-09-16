"""
Herbal-AI : Phase 4.3
MLflow Model Registry Retrieval + Inference Verification

This verification:
    - loads registered Version 1 models from DagsHub MLflow
    - downloads the registered model artifacts
    - reconstructs the serialized PyTorch models
    - runs a local forward pass
    - verifies expected output dimensions

This script does NOT:
    - train models
    - modify checkpoints
    - modify DVC
    - register new models
    - modify registry metadata
"""

import sys
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch


# ------------------------------------------------------------------------------
# Project root
# ------------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# ------------------------------------------------------------------------------
# Existing MLflow configuration
# ------------------------------------------------------------------------------

from ai.mlflow_tracking import configure_mlflow


# ------------------------------------------------------------------------------
# Registry configuration
# ------------------------------------------------------------------------------

MODELS = {
    "Universal Classifier": {
        "name": "UniversalClassifier",
        "version": "1",
        "image_size": 224,
        "expected_classes": 3,
    },
    "Herb Classifier": {
        "name": "HerbClassifier",
        "version": "1",
        "image_size": 256,
        "expected_classes": 40,
    },
    "Skin Disease Classifier": {
        "name": "SkinDiseaseClassifier",
        "version": "1",
        "image_size": 320,
        "expected_classes": 22,
    },
}


# ------------------------------------------------------------------------------
# Verification
# ------------------------------------------------------------------------------


def verify_model(display_name, config):
    """Retrieve and verify one registered model."""

    print("\n" + "=" * 70)
    print(f"VERIFYING: {display_name}")
    print("=" * 70)

    model_name = config["name"]
    version = config["version"]
    image_size = config["image_size"]
    expected_classes = config["expected_classes"]

    model_uri = f"models:/{model_name}/{version}"

    print(f"Model Name       : {model_name}")
    print(f"Version          : {version}")
    print(f"Model URI        : {model_uri}")
    print(f"Image Size       : {image_size}")
    print(f"Expected Classes : {expected_classes}")

    # --------------------------------------------------------------------------
    # 1. Registry retrieval
    # --------------------------------------------------------------------------

    print("\n[1/5] Registry retrieval .................", end=" ")

    try:
        model = mlflow.pytorch.load_model(
            model_uri=model_uri,
            map_location="cpu",
        )

        print("PASS")

    except Exception as exc:
        print("FAIL")
        print(f"\nRegistry retrieval error:\n{exc}")
        return False

    # --------------------------------------------------------------------------
    # 2. Model object
    # --------------------------------------------------------------------------

    print("[2/5] Model deserialization .............", end=" ")

    if model is None:
        print("FAIL")
        return False

    print("PASS")

    # --------------------------------------------------------------------------
    # 3. Eval mode
    # --------------------------------------------------------------------------

    print("[3/5] Eval mode ........................", end=" ")

    model.eval()

    if model.training:
        print("FAIL")
        return False

    print("PASS")

    # --------------------------------------------------------------------------
    # 4. Forward pass
    # --------------------------------------------------------------------------

    print("[4/5] Forward pass ......................", end=" ")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)

    dummy_input = torch.randn(
        1,
        3,
        image_size,
        image_size,
        device=device,
    )

    try:
        with torch.no_grad():
            output = model(dummy_input)

    except Exception as exc:
        print("FAIL")
        print(f"\nForward-pass error:\n{exc}")
        return False

    print("PASS")

    # --------------------------------------------------------------------------
    # 5. Output shape
    # --------------------------------------------------------------------------

    print("[5/5] Output shape ......................", end=" ")

    expected_shape = (1, expected_classes)

    if tuple(output.shape) != expected_shape:
        print("FAIL")
        print(
            f"\nExpected output shape: {expected_shape}"
        )
        print(
            f"Actual output shape  : {tuple(output.shape)}"
        )
        return False

    print("PASS")

    # --------------------------------------------------------------------------
    # Details
    # --------------------------------------------------------------------------

    print("\nRegistry model details:")
    print(f"  Output shape : {tuple(output.shape)}")
    print(f"  Output dtype : {output.dtype}")
    print(f"  Device       : {output.device}")
    print(f"  Eval mode    : {model.training is False}")

    return True


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------


def main():

    print("#" * 75)
    print("HERBAL-AI : PHASE 4.3")
    print("MLFLOW MODEL REGISTRY RETRIEVAL + INFERENCE VERIFICATION")
    print("#" * 75)

    print("\nThis verification performs:")
    print("  - no training")
    print("  - no model registration")
    print("  - no checkpoint modification")
    print("  - no DVC modification")

    # --------------------------------------------------------------------------
    # Configure MLflow
    # --------------------------------------------------------------------------

    configure_mlflow()

    print("\nMLflow Tracking URI:")
    print(f"  {mlflow.get_tracking_uri()}")

    print("\nPyTorch version:")
    print(f"  {torch.__version__}")

    print("\nCUDA available:")
    print(f"  {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print("GPU:")
        print(f"  {torch.cuda.get_device_name(0)}")

    # --------------------------------------------------------------------------
    # Verify all models
    # --------------------------------------------------------------------------

    results = {}

    for display_name, config in MODELS.items():

        results[display_name] = verify_model(
            display_name,
            config,
        )

    # --------------------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------------------

    print("\n" + "#" * 75)
    print("PHASE 4.3 VERIFICATION SUMMARY")
    print("#" * 75)

    for display_name, passed in results.items():

        status = "PASS" if passed else "FAIL"

        print(
            f"{display_name:<32}: {status}"
        )

    print("\n" + "-" * 75)

    if all(results.values()):

        print(
            "RESULT: ALL REGISTERED MODELS PASSED "
            "RETRIEVAL + INFERENCE VERIFICATION"
        )

        print("-" * 75)

        print("\nPhase 4.3 is COMPLETE.")
        print(
            "All three Model Registry Version 1 artifacts "
            "can be retrieved and executed."
        )

    else:

        print(
            "RESULT: ONE OR MORE REGISTERED MODELS FAILED "
            "VERIFICATION"
        )

        print("-" * 75)

        print(
            "\nPhase 4.3 is NOT complete."
        )

        sys.exit(1)


if __name__ == "__main__":
    main()
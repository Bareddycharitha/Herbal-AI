"""
Herbal-AI
Phase 4.1 - Local Model Load Verification

Purpose:
    Verify that all three existing trained models can be:
    1. Reconstructed using the existing model architecture
    2. Loaded from their existing best_model.pth checkpoint
    3. Switched to evaluation mode
    4. Used for a successful forward pass
    5. Verified against the expected output dimensions

Models:
    - Universal Classifier -> 3 classes
    - Herb Classifier      -> 40 classes
    - Skin Disease         -> 22 classes

IMPORTANT:
    This script does NOT:
    - train models
    - modify checkpoints
    - register models in MLflow
    - modify DVC
    - create test images
    - create random project artifacts
"""

from pathlib import Path
import sys

import torch


# ==============================================================================
# PROJECT ROOT
# ==============================================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# ==============================================================================
# EXISTING MODEL BUILDERS AND CONFIGURATION
# ==============================================================================

# ------------------------------------------------------------------------------
# Universal Classifier
# ------------------------------------------------------------------------------

from ai.image_classifier.model import (
    build_model as build_universal_model,
)

from ai.image_classifier.config import (
    BEST_MODEL_PATH as UNIVERSAL_CHECKPOINT,
    NUM_CLASSES as UNIVERSAL_NUM_CLASSES,
    IMAGE_SIZE as UNIVERSAL_IMAGE_SIZE,
    DEVICE as UNIVERSAL_DEVICE,
)


# ------------------------------------------------------------------------------
# Herb Classifier
# ------------------------------------------------------------------------------

from ai.herb.model import (
    build_model as build_herb_model,
)

from ai.herb.config import (
    BEST_MODEL_PATH as HERB_CHECKPOINT,
    IMAGE_SIZE as HERB_IMAGE_SIZE,
    DEVICE as HERB_DEVICE,
)


# Existing trained Herb model has 40 classes.
HERB_NUM_CLASSES = 40


# ------------------------------------------------------------------------------
# Skin Disease Classifier
# ------------------------------------------------------------------------------

from ai.models.efficientnet import (
    build_model as build_skin_model,
)

from ai.config import (
    BEST_MODEL_PATH as SKIN_CHECKPOINT,
    NUM_CLASSES as SKIN_NUM_CLASSES,
    IMAGE_SIZE as SKIN_IMAGE_SIZE,
    DEVICE as SKIN_DEVICE,
)


# ==============================================================================
# CHECKPOINT HELPERS
# ==============================================================================


def extract_state_dict(checkpoint):
    """
    Extract model state dictionary from supported checkpoint formats.
    """

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]

    return checkpoint


def normalize_herb_state_dict(state_dict):
    """
    Normalize older Herb checkpoint key names to the current architecture.

    Existing project compatibility:

        model.classifier.* -> classifier.*
        model.*            -> backbone.*
    """

    normalized = {}

    for key, value in state_dict.items():

        if key.startswith("model.classifier."):

            new_key = key.replace(
                "model.classifier.",
                "classifier.",
                1,
            )

        elif key.startswith("model."):

            new_key = key.replace(
                "model.",
                "backbone.",
                1,
            )

        else:

            new_key = key

        normalized[new_key] = value

    return normalized


# ==============================================================================
# STATE DICT VALIDATION
# ==============================================================================


def validate_state_dict(model, state_dict):
    """
    Validate checkpoint keys and tensor shapes before loading.

    Returns:
        missing_keys
        unexpected_keys
        shape_mismatches
    """

    model_state = model.state_dict()

    missing_keys = []
    unexpected_keys = []
    shape_mismatches = []

    checkpoint_keys = set(state_dict.keys())
    model_keys = set(model_state.keys())

    for key in sorted(model_keys - checkpoint_keys):
        missing_keys.append(key)

    for key in sorted(checkpoint_keys - model_keys):
        unexpected_keys.append(key)

    for key in sorted(model_keys & checkpoint_keys):

        model_shape = tuple(model_state[key].shape)
        checkpoint_shape = tuple(state_dict[key].shape)

        if model_shape != checkpoint_shape:

            shape_mismatches.append(
                (
                    key,
                    model_shape,
                    checkpoint_shape,
                )
            )

    return (
        missing_keys,
        unexpected_keys,
        shape_mismatches,
    )


# ==============================================================================
# GENERIC MODEL VERIFICATION
# ==============================================================================


def verify_model(
    model_name,
    checkpoint_path,
    device,
    image_size,
    expected_classes,
    build_function,
    build_kwargs=None,
    normalize_function=None,
):
    """
    Verify one trained model locally.
    """

    print("\n" + "=" * 75)
    print(f"VERIFYING: {model_name}")
    print("=" * 75)

    print(f"Checkpoint        : {checkpoint_path}")
    print(f"Device            : {device}")
    print(f"Image Size        : {image_size}")
    print(f"Expected Classes  : {expected_classes}")

    # --------------------------------------------------------------------------
    # 1. Check checkpoint exists
    # --------------------------------------------------------------------------

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():

        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print("\n[1/5] Checkpoint exists ................. PASS")

    # --------------------------------------------------------------------------
    # 2. Reconstruct model
    # --------------------------------------------------------------------------

    if build_kwargs is None:
        build_kwargs = {}

    model = build_function(**build_kwargs)

    model = model.to(device)

    print("[2/5] Model architecture reconstructed .. PASS")

    # --------------------------------------------------------------------------
    # 3. Load checkpoint
    # --------------------------------------------------------------------------

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    state_dict = extract_state_dict(checkpoint)

    if not isinstance(state_dict, dict):

        raise TypeError("Checkpoint does not contain a valid state dictionary.")

    if normalize_function is not None:

        state_dict = normalize_function(state_dict)

    # --------------------------------------------------------------------------
    # 4. Validate keys and shapes
    # --------------------------------------------------------------------------

    (
        missing_keys,
        unexpected_keys,
        shape_mismatches,
    ) = validate_state_dict(
        model,
        state_dict,
    )

    if missing_keys:

        print("\nMissing checkpoint keys:")

        for key in missing_keys[:20]:
            print(f"  - {key}")

        if len(missing_keys) > 20:
            print(f"  ... and {len(missing_keys) - 20} more")

        raise RuntimeError(
            f"{model_name}: checkpoint is missing " f"{len(missing_keys)} model keys."
        )

    if unexpected_keys:

        print("\nUnexpected checkpoint keys:")

        for key in unexpected_keys[:20]:
            print(f"  - {key}")

        if len(unexpected_keys) > 20:
            print(f"  ... and {len(unexpected_keys) - 20} more")

        raise RuntimeError(
            f"{model_name}: checkpoint contains "
            f"{len(unexpected_keys)} unexpected keys."
        )

    if shape_mismatches:

        print("\nShape mismatches:")

        for (
            key,
            model_shape,
            checkpoint_shape,
        ) in shape_mismatches[:20]:

            print(
                f"  - {key}: " f"model={model_shape}, " f"checkpoint={checkpoint_shape}"
            )

        raise RuntimeError(
            f"{model_name}: checkpoint contains "
            f"{len(shape_mismatches)} shape mismatches."
        )

    print("[3/5] Checkpoint keys/shapes match ...... PASS")

    # --------------------------------------------------------------------------
    # 5. Load state dictionary
    # --------------------------------------------------------------------------

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.eval()

    print("[4/5] Weights loaded + eval mode ........ PASS")

    # --------------------------------------------------------------------------
    # 6. Forward pass
    # --------------------------------------------------------------------------

    dummy_input = torch.randn(
        1,
        3,
        image_size,
        image_size,
        device=device,
    )

    with torch.inference_mode():

        output = model(dummy_input)

    if not isinstance(output, torch.Tensor):

        raise TypeError(f"{model_name}: model output is not a torch.Tensor.")

    if output.ndim != 2:

        raise RuntimeError(
            f"{model_name}: expected 2D output "
            f"(batch, classes), "
            f"got shape {tuple(output.shape)}"
        )

    if output.shape[0] != 1:

        raise RuntimeError(
            f"{model_name}: expected batch dimension 1, " f"got {output.shape[0]}"
        )

    actual_classes = output.shape[1]

    if actual_classes != expected_classes:

        raise RuntimeError(
            f"{model_name}: expected "
            f"{expected_classes} output classes, "
            f"got {actual_classes}"
        )

    print("[5/5] Forward pass ..................... PASS")

    print("\nModel verification details:")
    print(f"  Output shape : {tuple(output.shape)}")
    print(f"  Output dtype : {output.dtype}")
    print(f"  Eval mode    : {not model.training}")

    print(f"\n>>> {model_name}: PASS")

    return True


# ==============================================================================
# MAIN
# ==============================================================================


def main():

    print("\n" + "#" * 75)
    print("HERBAL-AI : PHASE 4.1")
    print("LOCAL MODEL-LOAD VERIFICATION")
    print("#" * 75)

    print("\nThis verification performs " "NO training and NO MLflow registration.")

    print(
        "\nPyTorch version :",
        torch.__version__,
    )

    print(
        "CUDA available  :",
        torch.cuda.is_available(),
    )

    if torch.cuda.is_available():

        print(
            "GPU             :",
            torch.cuda.get_device_name(0),
        )

    results = {}

    # ==========================================================================
    # UNIVERSAL
    # ==========================================================================

    try:

        results["Universal Classifier"] = verify_model(
            model_name="Universal Classifier",
            checkpoint_path=UNIVERSAL_CHECKPOINT,
            device=UNIVERSAL_DEVICE,
            image_size=UNIVERSAL_IMAGE_SIZE,
            expected_classes=UNIVERSAL_NUM_CLASSES,
            build_function=build_universal_model,
        )

    except Exception as exc:

        results["Universal Classifier"] = False

        print("\n>>> Universal Classifier: FAIL")

        print(f"Reason: {type(exc).__name__}: {exc}")

    # ==========================================================================
    # HERB
    # ==========================================================================

    try:

        results["Herb Classifier"] = verify_model(
            model_name="Herb Classifier",
            checkpoint_path=HERB_CHECKPOINT,
            device=HERB_DEVICE,
            image_size=HERB_IMAGE_SIZE,
            expected_classes=HERB_NUM_CLASSES,
            build_function=build_herb_model,
            build_kwargs={
                "num_classes": HERB_NUM_CLASSES,
            },
            normalize_function=normalize_herb_state_dict,
        )

    except Exception as exc:

        results["Herb Classifier"] = False

        print("\n>>> Herb Classifier: FAIL")

        print(f"Reason: {type(exc).__name__}: {exc}")

    # ==========================================================================
    # SKIN
    # ==========================================================================

    try:

        results["Skin Disease Classifier"] = verify_model(
            model_name="Skin Disease Classifier",
            checkpoint_path=SKIN_CHECKPOINT,
            device=SKIN_DEVICE,
            image_size=SKIN_IMAGE_SIZE,
            expected_classes=SKIN_NUM_CLASSES,
            build_function=build_skin_model,
            build_kwargs={
                "num_classes": SKIN_NUM_CLASSES,
            },
        )

    except Exception as exc:

        results["Skin Disease Classifier"] = False

        print("\n>>> Skin Disease Classifier: FAIL")

        print(f"Reason: {type(exc).__name__}: {exc}")

    # ==========================================================================
    # FINAL SUMMARY
    # ==========================================================================

    print("\n" + "#" * 75)
    print("PHASE 4.1 VERIFICATION SUMMARY")
    print("#" * 75)

    for model_name, status in results.items():

        print(f"{model_name:<30} : " f"{'PASS' if status else 'FAIL'}")

    all_passed = all(results.values())

    print("\n" + "-" * 75)

    if all_passed:

        print("RESULT: ALL THREE MODELS PASSED " "LOCAL VERIFICATION")

        print("-" * 75)

        print("\nPhase 4.1 is COMPLETE.")

        print("Safe to proceed to Phase 4.2:")

        print("MLflow Model Logging + Model Registry")

    else:

        print("RESULT: ONE OR MORE MODELS FAILED")

        print("-" * 75)

        print("\nDO NOT register the models yet.")

        print("Fix the failed model verification first.")

    print("\n" + "#" * 75)


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    main()

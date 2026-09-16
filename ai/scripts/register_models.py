"""
Herbal-AI : Phase 4.2
MLflow Model Logging + Model Registry

Purpose:
    Log the existing, verified Herbal-AI models to MLflow
    and register them in the MLflow Model Registry.

This script performs:
    1. Existing checkpoint loading
    2. MLflow PyTorch model logging
    3. Model Registry registration
    4. Model-version metadata tagging

This script does NOT:
    - train models
    - modify checkpoints
    - modify DVC
    - modify datasets
"""

import sys
from pathlib import Path

from huggingface_hub import model_info

import mlflow
import torch

# ------------------------------------------------------------------------------
# Project root
# ------------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# ------------------------------------------------------------------------------
# Existing project imports
# ------------------------------------------------------------------------------

from ai.mlflow_tracking import (
    configure_mlflow,
    get_git_commit,
    register_model,
    set_model_version_tags,
)

from ai.image_classifier.config import (
    BEST_MODEL_PATH as UNIVERSAL_CHECKPOINT,
    IMAGE_SIZE as UNIVERSAL_IMAGE_SIZE,
    NUM_CLASSES as UNIVERSAL_NUM_CLASSES,
    DEVICE as UNIVERSAL_DEVICE,
)

from ai.image_classifier.config import (
    BEST_MODEL_PATH as UNIVERSAL_CHECKPOINT,
    IMAGE_SIZE as UNIVERSAL_IMAGE_SIZE,
    NUM_CLASSES as UNIVERSAL_NUM_CLASSES,
)

from ai.image_classifier.model import (
    build_model as build_universal_model,
)

from ai.herb.config import (
    BEST_MODEL_PATH as HERB_CHECKPOINT,
    IMAGE_SIZE as HERB_IMAGE_SIZE,
    DEVICE as HERB_DEVICE,
)

from ai.herb.model import (
    build_model as build_herb_model,
)

from ai.config import (
    BEST_MODEL_PATH as SKIN_CHECKPOINT,
    IMAGE_SIZE as SKIN_IMAGE_SIZE,
    NUM_CLASSES as SKIN_NUM_CLASSES,
    DEVICE as SKIN_DEVICE,
)

from ai.models.efficientnet import (
    build_model as build_skin_model,
)


# ------------------------------------------------------------------------------
# Registry configuration
# ------------------------------------------------------------------------------

DATASET_VERSION = "v1"
REGISTRY_STAGE = "production_candidate"

MODELS = {
    "universal": {
        "registry_name": "UniversalClassifier",
        "checkpoint": UNIVERSAL_CHECKPOINT,
        "image_size": UNIVERSAL_IMAGE_SIZE,
        "num_classes": UNIVERSAL_NUM_CLASSES,
        "device": UNIVERSAL_DEVICE,
        "build_model": build_universal_model,

        # Existing evaluation run
        "evaluation_run_id": "1fc0a9c6065d42458c5d673ab467c571",

        # Existing training experiment
        "experiment": "Herbal-AI / Universal Classifier",

        "metrics": {
            "test_accuracy": 0.9932,
            "skin_accuracy": 0.9948,
            "medicinal_accuracy": 0.9993,
            "other_accuracy": 0.9674,
            "skin_to_other_errors": 7,
            "medicinal_to_other_errors": 1,
            "other_to_skin_errors": None,
            "other_to_medicinal_errors": None,
            "high_confidence_error_rate": None,
        },
    },

    "herb": {
        "registry_name": "HerbClassifier",
        "checkpoint": HERB_CHECKPOINT,
        "image_size": HERB_IMAGE_SIZE,
        "num_classes": 40,
        "device": HERB_DEVICE,
        "build_model": build_herb_model,

        # Existing evaluation run
        "evaluation_run_id": "aca98c9d7341488082359d89367d2b51",

        "experiment": "Herbal-AI / Herb Classifier",

        "metrics": {
            "test_accuracy": 0.9984,
            "test_precision": 0.9985,
            "test_recall": 0.9984,
            "test_f1_score": 0.9984,
        },
    },

    "skin": {
        "registry_name": "SkinDiseaseClassifier",
        "checkpoint": SKIN_CHECKPOINT,
        "image_size": SKIN_IMAGE_SIZE,
        "num_classes": SKIN_NUM_CLASSES,
        "device": SKIN_DEVICE,
        "build_model": build_skin_model,

        # Existing evaluation run
        "evaluation_run_id": "7930f0ec39a14aa88f45c1974ae1913c",

        "experiment": "Herbal-AI / Skin Disease Classifier",

        "metrics": {
            "test_accuracy": 0.7924,
            "test_macro_precision": 0.7601,
            "test_macro_recall": 0.7724,
            "test_macro_f1": 0.7639,
            "test_weighted_precision": 0.7976,
            "test_weighted_recall": 0.7924,
            "test_weighted_f1": 0.7930,
        },
    },
}


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------


def extract_state_dict(checkpoint):
    """Extract a model state dictionary from an existing checkpoint."""

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]

        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]

        # Raw state_dict checkpoint
        if checkpoint and all(
            isinstance(value, torch.Tensor)
            for value in checkpoint.values()
        ):
            return checkpoint

    raise ValueError(
        "Could not find a valid model state dictionary in checkpoint."
    )


def normalize_herb_state_dict(state_dict):
    """
    Preserve the existing Herb checkpoint compatibility logic.

    Older Herb checkpoints may contain:
        model.classifier.*
        model.*

    Current model expects:
        classifier.*
        backbone.*
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


def load_checkpoint_model(model_type, config):
    """Load an existing checkpoint into its existing architecture."""

    checkpoint_path = Path(config["checkpoint"])

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"{model_type} checkpoint not found: {checkpoint_path}"
        )

    print(f"\nLoading {model_type} model")
    print(f"Checkpoint : {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    state_dict = extract_state_dict(checkpoint)

    if model_type == "herb":
        state_dict = normalize_herb_state_dict(state_dict)

    build_model = config["build_model"]

    if model_type == "universal":
        model = build_model()

    else:
        model = build_model(
            num_classes=config["num_classes"]
        )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.eval()

    print("Architecture : PASS")
    print("Weights      : PASS")
    print("Eval mode    : PASS")

    return model


def log_and_register_model(model_type, config, git_commit):
    """Log an existing model and register it."""

    registry_name = config["registry_name"]

    print("\n" + "=" * 70)
    print(f"REGISTERING: {registry_name}")
    print("=" * 70)

    model = load_checkpoint_model(
        model_type,
        config,
    )

    # --------------------------------------------------------------------------
    # Start dedicated MLflow model-packaging run
    # --------------------------------------------------------------------------

    configure_mlflow()

    mlflow.set_experiment(
        config["experiment"]
    )

    with mlflow.start_run(
        run_name=f"{registry_name} Model Packaging",
        tags={
            "project": "Herbal-AI",
            "model_type": model_type,
            "stage": "model_packaging",
            "dataset_version": DATASET_VERSION,
            "registry_stage": REGISTRY_STAGE,
        },
    ) as run:

        run_id = run.info.run_id

        print(f"MLflow Run ID : {run_id}")

        # ----------------------------------------------------------------------
        # Parameters
        # ----------------------------------------------------------------------

        mlflow.log_params(
            {
                "model_name": (
                    "tf_efficientnetv2_s"
                    if model_type != "universal"
                    else "universal_classifier"
                ),
                "image_size": config["image_size"],
                "num_classes": config["num_classes"],
                "dataset_version": DATASET_VERSION,
                "git_commit": git_commit,
                "source_evaluation_run": config["evaluation_run_id"],
            }
        )

        # ----------------------------------------------------------------------
        # Evaluation metrics
        # ----------------------------------------------------------------------

        metrics = {}

        for key, value in config["metrics"].items():

            if value is not None:
                metrics[key] = float(value)

        if metrics:
            mlflow.log_metrics(metrics)

        # ----------------------------------------------------------------------
        # Model signature / input example
        # ----------------------------------------------------------------------

        input_example = torch.randn(
            1,
            3,
            config["image_size"],
            config["image_size"],
        )

        # ----------------------------------------------------------------------
        # Log PyTorch model
        # ----------------------------------------------------------------------

        print("Logging PyTorch model to MLflow...")

        model_info = mlflow.pytorch.log_model(
            pytorch_model=model,
            name="model",
            input_example=input_example,
            serialization_format="pickle",
        )
        

        model_uri = model_info.model_uri

        print(f"Model URI    : {model_uri}")

        # ----------------------------------------------------------------------
        # Register model
        # ----------------------------------------------------------------------

        print(
            f"Registering as: {registry_name}"
        )

        model_version = register_model(
            model_uri=model_uri,
            model_name=registry_name,
        )

        print(
            f"Registered Version : {model_version.version}"
        )

        # ----------------------------------------------------------------------
        # Model-version metadata
        # ----------------------------------------------------------------------

        set_model_version_tags(
            model_name=registry_name,
            version=model_version.version,
            tags={
                "project": "Herbal-AI",
                "model_type": model_type,
                "dataset_version": DATASET_VERSION,
                "git_commit": git_commit,
                "source_packaging_run": run_id,
                "source_evaluation_run": config["evaluation_run_id"],
                "stage": REGISTRY_STAGE,
                "image_size": config["image_size"],
                "num_classes": config["num_classes"],
            },
        )

        print("Metadata tags   : PASS")

    print(
        f"\n>>> {registry_name} Version "
        f"{model_version.version}: SUCCESS"
    )


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------


def main():

    print("#" * 75)
    print("HERBAL-AI : PHASE 4.2")
    print("MLFLOW MODEL LOGGING + MODEL REGISTRY")
    print("#" * 75)

    print("\nThis process:")
    print("  - uses existing trained checkpoints")
    print("  - performs no training")
    print("  - does not modify checkpoints")
    print("  - does not modify DVC")

    git_commit = get_git_commit()

    print(f"\nGit Commit : {git_commit}")
    print(f"Dataset    : {DATASET_VERSION}")
    print(f"Registry Stage : {REGISTRY_STAGE}")

    # --------------------------------------------------------------------------
    # Register each model
    # --------------------------------------------------------------------------

    for model_type, config in MODELS.items():

        log_and_register_model(
            model_type=model_type,
            config=config,
            git_commit=git_commit,
        )

    print("\n" + "#" * 75)
    print("PHASE 4.2 MODEL REGISTRATION COMPLETE")
    print("#" * 75)

    print("\nRegistered Models:")
    print("  1. UniversalClassifier")
    print("  2. HerbClassifier")
    print("  3. SkinDiseaseClassifier")

    print("\nAll models are marked:")
    print(f"  {REGISTRY_STAGE}")

    print("\nNo model has been promoted to Production.")


if __name__ == "__main__":
    main()
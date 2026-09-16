"""
Reusable MLflow tracking utilities for Herbal-AI.

Tracking is backend-independent:
- Local MLflow by default
- Remote MLflow/DagsHub can be configured later
"""

import os
import subprocess
from pathlib import Path

import mlflow
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_TRACKING_DB = PROJECT_ROOT / "mlflow.db"

load_dotenv()

TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{DEFAULT_TRACKING_DB.resolve().as_posix()}",
)


EXPERIMENTS = {
    "universal": "Herbal-AI / Universal Classifier",
    "herb": "Herbal-AI / Herb Classifier",
    "skin": "Herbal-AI / Skin Disease Classifier",
}


# ==============================================================================
# MLflow Configuration
# ==============================================================================


def configure_mlflow():
    """Configure MLflow tracking URI."""

    mlflow.set_tracking_uri(TRACKING_URI)


# ==============================================================================
# Training Tracking
# ==============================================================================


def start_training_run(model_type, run_name=None, params=None):
    """Start an MLflow training run."""

    configure_mlflow()

    if model_type not in EXPERIMENTS:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Expected one of: {list(EXPERIMENTS.keys())}"
        )

    mlflow.set_experiment(EXPERIMENTS[model_type])

    run = mlflow.start_run(
        run_name=run_name,
        tags={
            "project": "Herbal-AI",
            "model_type": model_type,
            "stage": "training",
        },
    )

    if params:
        mlflow.log_params(params)

    return run


# ==============================================================================
# Evaluation Tracking
# ==============================================================================


def start_evaluation_run(model_type, run_name=None):
    """Start an MLflow evaluation run."""

    configure_mlflow()

    if model_type not in EXPERIMENTS:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Expected one of: {list(EXPERIMENTS.keys())}"
        )

    mlflow.set_experiment(EXPERIMENTS[model_type])

    return mlflow.start_run(
        run_name=run_name,
        tags={
            "project": "Herbal-AI",
            "model_type": model_type,
            "stage": "evaluation",
        },
    )


# ==============================================================================
# Metrics
# ==============================================================================


def log_epoch_metrics(metrics, epoch):
    """Log training metrics for a particular epoch."""

    if mlflow.active_run() is None:
        return

    cleaned = {}

    for key, value in metrics.items():
        if value is None:
            continue

        try:
            cleaned[key] = float(value)
        except (TypeError, ValueError):
            continue

    if cleaned:
        mlflow.log_metrics(cleaned, step=epoch)


def log_final_metrics(metrics):
    """Log final evaluation metrics."""

    if mlflow.active_run() is None:
        return

    cleaned = {}

    for key, value in metrics.items():
        if value is None:
            continue

        try:
            cleaned[key] = float(value)
        except (TypeError, ValueError):
            continue

    if cleaned:
        mlflow.log_metrics(cleaned)


# ==============================================================================
# Artifacts
# ==============================================================================


def log_artifact_if_exists(path, artifact_path=None):
    """Log an artifact only if the file exists."""

    if mlflow.active_run() is None:
        return

    path = Path(path)

    if path.exists():
        mlflow.log_artifact(
            str(path),
            artifact_path=artifact_path,
        )


def log_artifacts_if_exist(paths, artifact_path=None):
    """Log multiple artifacts if they exist."""

    for path in paths:
        log_artifact_if_exists(
            path,
            artifact_path=artifact_path,
        )


# ==============================================================================
# Tags
# ==============================================================================


def set_tags(tags):
    """Set MLflow run tags."""

    if mlflow.active_run() is None:
        return

    mlflow.set_tags(tags)


# ==============================================================================
# Git Metadata
# ==============================================================================


def get_git_commit():
    """
    Return the current Git commit hash.

    Returns:
        str: Current commit hash, or 'unknown' if unavailable.
    """

    try:
        result = subprocess.run(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )

        commit = result.stdout.strip()

        if commit:
            return commit

    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        OSError,
    ):
        pass

    return "unknown"


# ==============================================================================
# Model Registry
# ==============================================================================


def register_model(model_uri, model_name):
    """
    Register an MLflow model in the Model Registry.

    Args:
        model_uri: MLflow model URI, for example:
            runs:/<run_id>/model
        model_name: Registered model name.

    Returns:
        ModelVersion object returned by MLflow.
    """

    configure_mlflow()

    if not model_uri:
        raise ValueError("model_uri must not be empty.")

    if not model_name:
        raise ValueError("model_name must not be empty.")

    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=model_name,
    )

    return model_version


def set_model_version_tags(model_name, version, tags):
    """
    Set metadata tags on a registered model version.

    Args:
        model_name: Registered model name.
        version: Registered model version number.
        tags: Dictionary containing model-version metadata.
    """

    configure_mlflow()

    if not model_name:
        raise ValueError("model_name must not be empty.")

    if version is None:
        raise ValueError("version must not be None.")

    if not tags:
        return

    cleaned_tags = {}

    for key, value in tags.items():

        if value is None:
            continue

        cleaned_tags[str(key)] = str(value)

    if cleaned_tags:
        for key, value in cleaned_tags.items():
            mlflow.set_model_version_tag(
                name=model_name,
                version=str(version),
                key=key,
                value=value,
            )


# ==============================================================================
# Run Management
# ==============================================================================


def end_run(status="FINISHED"):
    """Safely end the active MLflow run."""

    if mlflow.active_run() is not None:
        mlflow.end_run(status=status)
"""
Reusable MLflow tracking utilities for Herbal-AI.

Tracking is backend-independent:
- Local MLflow by default
- Remote MLflow/DagsHub can be configured later
"""

import os
from pathlib import Path

import mlflow


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_TRACKING_DB = PROJECT_ROOT / "mlflow.db"

TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{DEFAULT_TRACKING_DB.resolve().as_posix()}",
)


EXPERIMENTS = {
    "universal": "Herbal-AI / Universal Classifier",
    "herb": "Herbal-AI / Herb Classifier",
    "skin": "Herbal-AI / Skin Disease Classifier",
}


def configure_mlflow():
    """Configure MLflow tracking URI."""
    mlflow.set_tracking_uri(TRACKING_URI)


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


def set_tags(tags):
    """Set MLflow tags."""

    if mlflow.active_run() is None:
        return

    mlflow.set_tags(tags)


def end_run(status="FINISHED"):
    """Safely end the active MLflow run."""

    if mlflow.active_run() is not None:
        mlflow.end_run(status=status)
from pathlib import Path
import torch

# ==========================================================
# PROJECT PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = PROJECT_ROOT / "datasets"

SKIN_DATASET = DATASET_ROOT / "SkinDisease"
MEDICINAL_LEAF_DATASET = DATASET_ROOT / "Medicinal_Leaf_dataset"
MEDICINAL_PLANT_DATASET = DATASET_ROOT / "Medicinal_plant_dataset"
OTHER_OBJECTS_DATASET = DATASET_ROOT / "OtherObjects"

# ==========================================================
# TRAINING DATASETS
# ==========================================================

TRAIN_DIRS = [
    SKIN_DATASET / "train",
    MEDICINAL_LEAF_DATASET / "train",
    MEDICINAL_PLANT_DATASET / "train",
    OTHER_OBJECTS_DATASET / "train",
]

VAL_DIRS = [
    SKIN_DATASET / "val",
    MEDICINAL_LEAF_DATASET / "val",
    MEDICINAL_PLANT_DATASET / "val",
    OTHER_OBJECTS_DATASET / "val",
]

TEST_DIRS = [
    SKIN_DATASET / "test",
    MEDICINAL_LEAF_DATASET / "test",
    MEDICINAL_PLANT_DATASET / "test",
    OTHER_OBJECTS_DATASET / "test",
]

# ==========================================================
# OUTPUT DIRECTORIES
# ==========================================================

CHECKPOINT_DIR = PROJECT_ROOT / "image_classifier" / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "image_classifier" / "results"
LOG_DIR = PROJECT_ROOT / "image_classifier" / "logs"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# DEVICE
# ==========================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================================
# MODEL
# ==========================================================

MODEL_NAME = "tf_efficientnetv2_s"

NUM_CLASSES = 3

IMAGE_SIZE = 224

PRETRAINED = True

# ==========================================================
# TRAINING
# ==========================================================

BATCH_SIZE = 32

EPOCHS = 20

LEARNING_RATE = 3e-4

WEIGHT_DECAY = 1e-4

NUM_WORKERS = 2

PIN_MEMORY = torch.cuda.is_available()

# ==========================================================
# LOSS FUNCTION
# ==========================================================

# Label Smoothing
LABEL_SMOOTHING = 0.1

# Focal Loss
USE_FOCAL_LOSS = True
FOCAL_GAMMA = 2.0

# Class Weights (auto-computed if None)
CLASS_WEIGHTS = None

# ==========================================================
# OUTLIER EXPOSURE (OE)
# ==========================================================

USE_OE = True
OE_DATASET_DIR = DATASET_ROOT / "OE_Dataset"  # Additional diverse images for OE
OE_RATIO = 0.5  # Ratio of OE samples per batch
OE_LOSS_WEIGHT = 0.5

# ==========================================================
# HARD NEGATIVE MINING
# ==========================================================

USE_HARD_NEGATIVES = True
HARD_NEGATIVE_DIR = DATASET_ROOT / "HardNegatives"  # Challenging "Other" images
HARD_NEGATIVE_RATIO = 0.3

# ==========================================================
# EARLY STOPPING
# ==========================================================

PATIENCE = 5

MIN_DELTA = 0.001

# ==========================================================
# RANDOMNESS
# ==========================================================

RANDOM_SEED = 42

# ==========================================================
# MIXED PRECISION
# ==========================================================

USE_AMP = torch.cuda.is_available()

# ==========================================================
# CALIBRATION
# ==========================================================

CALIBRATE_AFTER_TRAINING = True
CALIBRATION_LR = 0.01
CALIBRATION_MAX_ITER = 100

# ==========================================================
# CHECKPOINT FILES
# ==========================================================

BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pth"

LAST_MODEL_PATH = CHECKPOINT_DIR / "last_model.pth"

# ==========================================================
# OOD Detection Thresholds (for 3-class universal classifier)
# ==========================================================
# Energy score: E(x) = -logsumexp(logits). For 3 classes, energy
# ranges from very negative (confident ID) to positive (uncertain OOD).
# Threshold 0.5 catches very anomalous inputs with strongly negative logits.
OOD_ENERGY_THRESHOLD = 0.5
# MSP score: 1 - max_softmax. Higher = more OOD. 0.5 means max_prob < 0.5.
OOD_MSP_THRESHOLD = 0.5
# Entropy: -sum(p*log(p)). Max for 3 classes = log(3) ≈ 1.099.
# Threshold must be below log(3) to be achievable.
# Using 0.9 (≈82% of max entropy) to catch moderately uncertain predictions.
OOD_ENTROPY_THRESHOLD = 0.9
# Calibration file
CALIBRATION_PATH = CHECKPOINT_DIR / "temperature_scale.pth"

TRAIN_HISTORY = RESULTS_DIR / "training_history.csv"

# ==========================================================
# ENSEMBLE (DISABLED FOR PRODUCTION INFERENCE - SINGLE MODEL ONLY)
# ==========================================================

# Ensemble training is supported but ensemble inference is disabled for production.
# Only single model inference is used regardless of these settings.
ENSEMBLE_SIZE = 1  # Number of models to ensemble (kept for training compatibility)
ENSEMBLE_SEEDS = [42]  # Seeds for ensemble models (kept for training compatibility)
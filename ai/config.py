import torch
import tempfile
from pathlib import Path

# ===========================
# Dataset
# ===========================

IMAGE_SIZE = 320
NUM_CLASSES = 22  # 22 skin diseases (excluding Unknown_Normal)

# ===========================
# Training
# ===========================

BATCH_SIZE = 16
EPOCHS = 25

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 5e-4

RANDOM_SEED = 42
VAL_SPLIT = 0.2

# ===========================
# Hardware
# ===========================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

NUM_WORKERS = 2
PIN_MEMORY = True

# ===========================
# Model
# ===========================

MODEL_NAME = "tf_efficientnetv2_s"
PRETRAINED = True

# ===========================
# Dataset Paths
# ===========================

BASE_DIR = Path(__file__).resolve().parent

TRAIN_DIR = BASE_DIR / "datasets" / "SkinDisease" / "train"
TEST_DIR = BASE_DIR / "datasets" / "SkinDisease" / "test"

# ===========================
# Output Paths
# ===========================

CHECKPOINT_DIR = BASE_DIR / "checkpoints"
RESULTS_DIR = BASE_DIR / "results"

# Runtime output directory for Grad-CAM images
GRADCAM_DIR = (
    Path(tempfile.gettempdir())
    / "herbal_ai_gradcam"
)

HISTORY_FILE = (
    RESULTS_DIR
    / "training_history.csv"
)

BEST_MODEL_PATH = (
    CHECKPOINT_DIR
    / "best_model.pth"
)

LAST_MODEL_PATH = (
    CHECKPOINT_DIR
    / "last_model.pth"
)

# ===========================
# Early Stopping
# ===========================

PATIENCE = 7
MIN_DELTA = 0.001

# ===========================
# Loss Function
# ===========================

LABEL_SMOOTHING = 0.05

USE_FOCAL_LOSS = False
FOCAL_GAMMA = 2.0

# Standard balanced class weights
# Automatically calculated from training data.
CLASS_WEIGHTS = None

# ===========================
# Two-Stage Training
# ===========================

USE_TWO_STAGE = False

STAGE1_EPOCHS = 10
STAGE2_EPOCHS = 20

# ===========================
# Outlier Exposure
# ===========================

USE_OE = False

OE_DATASET_DIR = (
    BASE_DIR
    / "datasets"
    / "OE_Dataset"
)

OE_RATIO = 0.3
OE_LOSS_WEIGHT = 0.5

# ===========================
# Mixed Precision
# ===========================

USE_AMP = torch.cuda.is_available()

# ===========================
# Calibration
# ===========================

CALIBRATE_AFTER_TRAINING = True

CALIBRATION_LR = 0.01
CALIBRATION_MAX_ITER = 100

# ===========================
# Ensemble
# ===========================

ENSEMBLE_SIZE = 3
ENSEMBLE_SEEDS = [42, 123, 456]

# ===========================
# Prediction Confidence
# ===========================

HIGH_CONFIDENCE_THRESHOLD = 70.0
MEDIUM_CONFIDENCE_THRESHOLD = 40.0

# ===========================
# OOD Detection
# ===========================

OOD_ENERGY_THRESHOLD = 0.5
OOD_MSP_THRESHOLD = 0.5
OOD_ENTROPY_THRESHOLD = 1.5

# ===========================
# Grad-CAM
# ===========================

GRADCAM_TARGET_LAYER = "conv_head"
USE_GRADCAM_PLUS = True
"""
Configuration for Herb Identification Model
"""

from pathlib import Path
import torch

# ==============================================================================
# BASE PATHS
# ==============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset
DATASET_DIR = BASE_DIR / "datasets" / "Medicinal_plant_dataset"

# ==============================================================================
# DATA SPLIT
# ==============================================================================

TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10

RANDOM_SEED = 42

# ==============================================================================
# OUTPUT DIRECTORIES
# ==============================================================================

HERB_DIR = BASE_DIR / "herb"

CHECKPOINT_DIR = HERB_DIR / "checkpoints"
RESULTS_DIR = HERB_DIR / "results"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# IMAGE SETTINGS
# ==============================================================================

IMAGE_SIZE = 256

# ==============================================================================
# MODEL SETTINGS
# ==============================================================================

MODEL_NAME = "tf_efficientnetv2_s"
PRETRAINED = True

# Alternative higher capacity models
# MODEL_NAME = "tf_efficientnetv2_m"
# MODEL_NAME = "convnext_base.fb_in1k"

# ==============================================================================
# LOSS FUNCTION
# ==============================================================================

# Label Smoothing
LABEL_SMOOTHING = 0.1

# Metric Learning Loss
USE_ARCFACE = False  # Disabled to match existing checkpoint
ARCFACE_MARGIN = 0.5
ARCFACE_SCALE = 30.0

# Focal Loss (alternative)
USE_FOCAL_LOSS = False
FOCAL_GAMMA = 2.0

# Class Weights
CLASS_WEIGHTS = None  # Auto-computed

# ==============================================================================
# TRAINING SETTINGS
# ==============================================================================

BATCH_SIZE = 32
EPOCHS = 30  # Increased

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4

# ==============================================================================
# LEARNING RATE SCHEDULER
# ==============================================================================

LR_SCHEDULER = "cosine"  # Options: "cosine", "plateau", "step"
COSINE_T_MAX = EPOCHS
COSINE_ETA_MIN = 1e-6

# ==============================================================================
# EARLY STOPPING
# ==============================================================================

PATIENCE = 7
MIN_DELTA = 0.001

# ==============================================================================
# DEVICE
# ==============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_WORKERS = 2
PIN_MEMORY = torch.cuda.is_available()

# ==============================================================================
# MIXED PRECISION
# ==============================================================================

USE_AMP = torch.cuda.is_available()

# ==============================================================================
# OUTLIER EXPOSURE
# ==============================================================================

USE_OE = True
OE_DATASET_DIR = BASE_DIR / "datasets" / "OE_Dataset"
OE_RATIO = 0.3
OE_LOSS_WEIGHT = 0.5

# ==============================================================================
# BINARY LEAF DETECTOR
# ==============================================================================

USE_LEAF_DETECTOR = False  # Disabled by default; auto-enabled if checkpoint exists
LEAF_DETECTOR_MODEL = "mobilenetv3_small_100"  # Lightweight
LEAF_DETECTOR_PATH = CHECKPOINT_DIR / "leaf_detector.pth"
LEAF_DETECTOR_THRESHOLD = 0.7

# ==============================================================================
# CHECKPOINT FILES
# ==============================================================================

BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pth"
LAST_MODEL_PATH = CHECKPOINT_DIR / "last_model.pth"

CLASS_MAPPING_PATH = CHECKPOINT_DIR / "class_mapping.json"

TRAINING_CONFIG_PATH = CHECKPOINT_DIR / "training_config.json"

MODEL_INFO_PATH = CHECKPOINT_DIR / "model_info.json"

METRICS_PATH = CHECKPOINT_DIR / "metrics.json"

# ==============================================================================
# TRAINING HISTORY
# ==============================================================================

HISTORY_FILE = RESULTS_DIR / "training_history.csv"

# ==============================================================================
# INFERENCE
# ==============================================================================

TOP_K = 3

CONFIDENCE_THRESHOLD = 0.60  # Increased from 0.50

# OOD Detection thresholds
OOD_ENERGY_THRESHOLD = 0.5
OOD_MSP_THRESHOLD = 0.5
OOD_ENTROPY_THRESHOLD = 1.5

# ==============================================================================
# CALIBRATION
# ==============================================================================

CALIBRATE_AFTER_TRAINING = True
CALIBRATION_LR = 0.01
CALIBRATION_MAX_ITER = 100

# ==============================================================================
# ENSEMBLE
# ==============================================================================

ENSEMBLE_SIZE = 3
ENSEMBLE_SEEDS = [42, 123, 456]

# ==============================================================================
# CHECKPOINT SETTINGS
# ==============================================================================

SAVE_EVERY = 5

RESUME_TRAINING = False
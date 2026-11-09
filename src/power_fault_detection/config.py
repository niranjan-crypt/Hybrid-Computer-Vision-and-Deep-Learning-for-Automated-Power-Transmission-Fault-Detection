"""Central configuration module for Power Transmission Fault Detection."""

import os
import numpy as np

# ----------------- Dataset Paths -----------------
# Default placeholder directories (can be overridden via environment variables)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_DIR = os.getenv("DATASET_DIR", os.path.join(BASE_DIR, "Dataset_root"))
TRAIN_VAL_DIR = os.getenv("TRAIN_VAL_DIR", os.path.join(BASE_DIR, "Dataset_root_final"))
TEST_DIR = os.getenv("TEST_DIR", os.path.join(BASE_DIR, "Test_final"))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", os.path.join(BASE_DIR, "outputs"))

# ----------------- Image Parameters -----------------
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
CHANNELS = 3
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)

# ----------------- Deep Learning Hyperparameters -----------------
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
L2_REGULARIZATION = 5e-4
VALIDATION_SPLIT = 0.2
DROPOUT_RATE = 0.5
RANDOM_SEED = 42

# ----------------- Classical CV Feature Parameters -----------------
GLCM_DISTANCES = [1]
GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
GLCM_LEVELS = 256

LBP_POINTS = 8
LBP_RADIUS = 1
LBP_METHOD = "uniform"

HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_BLOCK_NORM = "L2-Hys"
HOG_FEATURE_LIMIT = 50

COLOR_HIST_BINS = 16
CANNY_LOW_THRESHOLD = 100
CANNY_HIGH_THRESHOLD = 200

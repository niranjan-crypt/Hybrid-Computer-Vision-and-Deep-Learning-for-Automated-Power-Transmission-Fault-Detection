"""Classical computer vision feature extraction algorithms."""

from typing import Tuple, List, Union
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog
from power_fault_detection.config import (
    GLCM_DISTANCES, GLCM_ANGLES, GLCM_LEVELS,
    LBP_POINTS, LBP_RADIUS, LBP_METHOD,
    HOG_ORIENTATIONS, HOG_PIXELS_PER_CELL, HOG_CELLS_PER_BLOCK, HOG_BLOCK_NORM, HOG_FEATURE_LIMIT,
    COLOR_HIST_BINS, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD
)


def extract_glcm_features(
    gray_image: np.ndarray,
    multidirectional: bool = True
) -> Tuple[List[float], np.ndarray]:
    """
    Extracts Gray-Level Co-occurrence Matrix (GLCM) statistical texture descriptors.

    Computes:
    - Contrast
    - Energy
    - Homogeneity
    - Correlation (if multidirectional)
    - Angular Second Moment (ASM) (if multidirectional)

    Returns:
        feature_vector: List of floating-point statistical metrics.
        glcm_matrix: Raw GLCM array.
    """
    angles = GLCM_ANGLES if multidirectional else [0]
    glcm = graycomatrix(
        gray_image,
        distances=GLCM_DISTANCES,
        angles=angles,
        levels=GLCM_LEVELS,
        symmetric=True,
        normed=True
    )

    if multidirectional:
        contrast = float(np.mean(graycoprops(glcm, 'contrast')))
        energy = float(np.mean(graycoprops(glcm, 'energy')))
        homogeneity = float(np.mean(graycoprops(glcm, 'homogeneity')))
        correlation = float(np.mean(graycoprops(glcm, 'correlation')))
        asm = float(np.mean(graycoprops(glcm, 'ASM')))
        features = [contrast, energy, homogeneity, correlation, asm]
    else:
        contrast = float(graycoprops(glcm, 'contrast')[0, 0])
        energy = float(graycoprops(glcm, 'energy')[0, 0])
        homogeneity = float(graycoprops(glcm, 'homogeneity')[0, 0])
        features = [contrast, energy, homogeneity]

    return features, glcm

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


def extract_lbp_features(
    gray_image: np.ndarray,
    p: int = LBP_POINTS,
    r: int = LBP_RADIUS,
    method: str = LBP_METHOD
) -> Tuple[List[float], np.ndarray]:
    """
    Extracts Local Binary Pattern (LBP) rotation-invariant uniform texture features.

    Returns:
        hist: Normalized histogram of LBP codes.
        lbp_image: Spatial 2D LBP transformed matrix.
    """
    lbp = local_binary_pattern(gray_image, P=p, R=r, method=method)
    num_bins = p + 2 if method == "uniform" else 256
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, num_bins + 1), range=(0, num_bins))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)

    return hist.tolist(), lbp


def extract_hog_features(
    gray_image: np.ndarray,
    limit: int = HOG_FEATURE_LIMIT,
    visualize: bool = True
) -> Tuple[List[float], Union[np.ndarray, None]]:
    """
    Extracts Histogram of Oriented Gradients (HOG) structural edge and shape descriptors.

    Returns:
        features: Leading subset of normalized HOG descriptors.
        hog_image: Visual gradient map (if visualize=True).
    """
    if visualize:
        features, hog_image = hog(
            gray_image,
            orientations=HOG_ORIENTATIONS,
            pixels_per_cell=HOG_PIXELS_PER_CELL,
            cells_per_block=HOG_CELLS_PER_BLOCK,
            block_norm=HOG_BLOCK_NORM,
            visualize=True
        )
        return features[:limit].tolist(), hog_image
    else:
        features = hog(
            gray_image,
            orientations=HOG_ORIENTATIONS,
            pixels_per_cell=HOG_PIXELS_PER_CELL,
            cells_per_block=HOG_CELLS_PER_BLOCK,
            block_norm=HOG_BLOCK_NORM,
            visualize=False
        )
        return features[:limit].tolist(), None


def extract_color_histogram(
    bgr_image: np.ndarray,
    bins: int = COLOR_HIST_BINS
) -> List[float]:
    """
    Extracts joint HSV color channel histogram feature vectors.

    Returns:
        hist: L1-normalized 1D concatenated color distribution vector.
    """
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [bins], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [bins], [0, 256])

    hist = np.concatenate([hist_h, hist_s, hist_v]).ravel()
    hist = hist / (hist.sum() + 1e-6)
    return hist.tolist()


def extract_edge_features(
    gray_image: np.ndarray,
    low_thresh: int = CANNY_LOW_THRESHOLD,
    high_thresh: int = CANNY_HIGH_THRESHOLD
) -> Tuple[List[float], np.ndarray]:
    """
    Computes Canny structural edge density across the input image.

    Returns:
        edge_density: Normalized ratio of edge pixels to total image pixels.
        edge_map: Binary 2D Canny edge response map.
    """
    edges = cv2.Canny(gray_image, low_thresh, high_thresh)
    edge_density = float(edges.sum() / (edges.shape[0] * edges.shape[1] * 255.0))
    return [edge_density], edges

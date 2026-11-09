"""Bird nest and foreign object detection using LBP texture anomaly and edge analysis."""

from typing import Tuple, List, Dict, Any
import numpy as np
import cv2
from skimage.feature import local_binary_pattern


def compute_lbp_diff_mask(
    good_img: np.ndarray,
    bad_img: np.ndarray,
    radius: int = 2,
    n_points: int = 16,
    threshold: int = 30
) -> np.ndarray:
    """
    Computes texture difference mask using Local Binary Patterns (LBP)
    between a healthy reference transmission structure and the inspected target.
    """
    gray_good = cv2.cvtColor(good_img, cv2.COLOR_BGR2GRAY)
    gray_bad = cv2.cvtColor(bad_img, cv2.COLOR_BGR2GRAY)

    # Ensure identical dimensions
    if gray_good.shape != gray_bad.shape:
        gray_good = cv2.resize(gray_good, (gray_bad.shape[1], gray_bad.shape[0]))

    lbp_good = local_binary_pattern(gray_good, n_points, radius, method="uniform")
    lbp_bad = local_binary_pattern(gray_bad, n_points, radius, method="uniform")

    diff = cv2.absdiff(lbp_good.astype(np.uint8), lbp_bad.astype(np.uint8))
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    mask = cv2.medianBlur(mask, 7)
    return mask


def detect_bird_nest_faults(
    good_reference_img: np.ndarray,
    target_img: np.ndarray,
    min_cluster_area: int = 150
) -> Dict[str, Any]:
    """
    Detects foreign vegetation and bird nest clusters on electrical transmission towers.
    """
    mask = compute_lbp_diff_mask(good_reference_img, target_img)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    nest_boxes: List[Tuple[int, int, int, int]] = []
    total_nest_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_cluster_area:
            x, y, w, h = cv2.boundingRect(cnt)
            nest_boxes.append((x, y, w, h))
            total_nest_area += area

    return {
        "mask": mask,
        "nest_boxes": nest_boxes,
        "anomaly_area": total_nest_area,
        "is_faulty": len(nest_boxes) > 0
    }

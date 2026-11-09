"""Rust and corrosion fault detection using HSV color segmentation and morphological filtering."""

from typing import Tuple, List, Dict, Any
import numpy as np
import cv2


def create_rust_mask(bgr_image: np.ndarray, median_kernel: int = 9) -> np.ndarray:
    """
    Generates a binary segmentation mask for rust/corrosion defects.

    Combines two HSV color ranges:
    - Primary rust (orange/brown hue: 5-25)
    - Dark oxidized rust (red/brown hue: 0-10)
    Followed by median blur filtering to remove speckle noise.
    """
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

    # Range 1: Orange/Brown rust tones
    lower_rust = np.array([5, 60, 60], dtype=np.uint8)
    upper_rust = np.array([25, 255, 255], dtype=np.uint8)
    mask1 = cv2.inRange(hsv, lower_rust, upper_rust)

    # Range 2: Deep reddish rust tones
    lower_red = np.array([0, 40, 40], dtype=np.uint8)
    upper_red = np.array([10, 255, 255], dtype=np.uint8)
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    combined_mask = cv2.bitwise_or(mask1, mask2)
    filtered_mask = cv2.medianBlur(combined_mask, median_kernel)
    return filtered_mask


def detect_rust_faults(
    bgr_image: np.ndarray,
    min_area: int = 50
) -> Dict[str, Any]:
    """
    Detects rust areas, calculates total rust severity percentage,
    and returns bounding bounding boxes for identified defect regions.
    """
    mask = create_rust_mask(bgr_image)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes: List[Tuple[int, int, int, int]] = []
    rust_pixel_count = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            bounding_boxes.append((x, y, w, h))
            rust_pixel_count += area

    total_pixels = bgr_image.shape[0] * bgr_image.shape[1]
    rust_percentage = (rust_pixel_count / total_pixels) * 100.0

    return {
        "mask": mask,
        "bounding_boxes": bounding_boxes,
        "rust_percentage": round(rust_percentage, 2),
        "is_faulty": len(bounding_boxes) > 0
    }

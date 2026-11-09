"""Insulator glass disc segmentation and missing/broken disc defect detection."""

from typing import List, Tuple
import numpy as np
import cv2


def detect_insulator_discs(
    bgr_image: np.ndarray,
    dp: float = 1.2,
    min_dist: int = 40,
    param1: int = 80,
    param2: int = 30,
    min_radius: int = 20,
    max_radius: int = 50
) -> List[Tuple[int, int, int]]:
    """
    Detects circular insulator glass discs in transmission strings using Hough Circle Transform.

    Args:
        bgr_image: Input BGR image.
        dp: Inverse ratio of the accumulator resolution to the image resolution.
        min_dist: Minimum distance between the centers of detected circles.
        param1: Higher threshold passed to Canny edge detector.
        param2: Accumulator threshold for circle centers at the detection stage.
        min_radius: Minimum disc radius.
        max_radius: Maximum disc radius.

    Returns:
        List of tuples: (x_center, y_center, radius), sorted vertically.
    """
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 2)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius
    )

    if circles is not None:
        disc_list = np.round(circles[0, :]).astype("int")
        # Sort vertically by Y coordinate to follow string sequence
        sorted_discs = sorted(disc_list, key=lambda c: c[1])
        return [(int(x), int(y), int(r)) for x, y, r in sorted_discs]

    return []

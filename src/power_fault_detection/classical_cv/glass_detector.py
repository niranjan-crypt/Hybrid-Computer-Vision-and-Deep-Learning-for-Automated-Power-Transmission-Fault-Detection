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


def detect_missing_or_broken_discs(
    good_reference_img: np.ndarray,
    target_img: np.ndarray,
    circles: List[Tuple[int, int, int]],
    match_threshold: float = 0.5
) -> List[Tuple[int, int, int]]:
    """
    Identifies broken, shattered, or missing glass discs by performing
    normalized cross-correlation template matching against a healthy reference disc.

    Args:
        good_reference_img: Intact reference insulator disc image.
        target_img: Insulator string image under inspection.
        circles: Detected disc coordinates (x, y, radius).
        match_threshold: Correlation coefficient threshold below which a disc is flagged defective.

    Returns:
        List of flagged defective discs: (x, y, pad_radius).
    """
    defects = []
    h_ref, w_ref = good_reference_img.shape[:2]
    h_tgt, w_tgt = target_img.shape[:2]

    for (x, y, r) in circles:
        pad = int(r * 1.1)
        y1, y2 = max(y - pad, 0), min(y + pad, h_ref)
        x1, x2 = max(x - pad, 0), min(x + pad, w_ref)
        template = good_reference_img[y1:y2, x1:x2]

        ty1, ty2 = max(y - pad, 0), min(y + pad, h_tgt)
        tx1, tx2 = max(x - pad, 0), min(x + pad, w_tgt)
        search_region = target_img[ty1:ty2, tx1:tx2]

        if (
            template.shape[:2] != search_region.shape[:2]
            or 0 in template.shape
            or 0 in search_region.shape
        ):
            continue

        res = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)

        if max_val < match_threshold:
            defects.append((x, y, pad))

    return defects

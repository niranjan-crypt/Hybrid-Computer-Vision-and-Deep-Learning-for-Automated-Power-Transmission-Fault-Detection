"""Unified fault detection pipeline executing glass, rust, and nest inspection algorithms."""

import os
import sys
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from power_fault_detection.config import DATASET_DIR
from power_fault_detection.classical_cv.glass_detector import (
    detect_insulator_discs,
    detect_missing_or_broken_discs
)
from power_fault_detection.classical_cv.rust_detector import detect_rust_faults
from power_fault_detection.classical_cv.nest_detector import detect_bird_nest_faults


def run_insulator_inspection(good_img_path: str, bad_img_path: str):
    good_img = cv2.imread(good_img_path)
    bad_img = cv2.imread(bad_img_path)
    if good_img is None or bad_img is None:
        print("Error reading insulator images.")
        return

    circles = detect_insulator_discs(good_img)
    defects = detect_missing_or_broken_discs(good_img, bad_img, circles)
    print(f"[Insulator] Detected {len(circles)} discs. Flagged {len(defects)} defective/missing discs.")


def run_rust_inspection(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        print("Error reading rust image.")
        return

    result = detect_rust_faults(img)
    print(f"[Rust] Fault detected: {result['is_faulty']}, Rust coverage: {result['rust_percentage']}%")


def run_nest_inspection(good_img_path: str, target_img_path: str):
    good_img = cv2.imread(good_img_path)
    tgt_img = cv2.imread(target_img_path)
    if good_img is None or tgt_img is None:
        print("Error reading nest images.")
        return

    result = detect_bird_nest_faults(good_img, tgt_img)
    print(f"[Bird Nest] Fault detected: {result['is_faulty']}, Anomaly area: {result['anomaly_area']} px")


if __name__ == "__main__":
    print("Power Transmission Infrastructure Inspection Runner")
    print("Provide paths to run specific inspection pipelines.")

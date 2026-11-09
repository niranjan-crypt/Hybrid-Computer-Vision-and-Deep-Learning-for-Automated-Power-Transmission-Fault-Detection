"""Visualizes handcrafted feature extraction on sample images from each fault class."""

import os
import sys
import random
import cv2

# Add src to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from power_fault_detection.config import DATASET_DIR
from power_fault_detection.classical_cv.features import (
    extract_glcm_features,
    extract_lbp_features,
    extract_hog_features,
    extract_color_histogram,
    extract_edge_features,
)
from power_fault_detection.utils.visualization import plot_feature_gallery


def main(dataset_path: str = DATASET_DIR):
    print(f"Scanning dataset at: {dataset_path}")
    if not os.path.exists(dataset_path):
        print(f"Dataset path not found: {dataset_path}. Please update config or pass path.")
        return

    for root, _, files in os.walk(dataset_path):
        img_files = [f for f in files if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        if not img_files:
            continue

        sample_file = random.choice(img_files)
        label = os.path.basename(root)
        img_path = os.path.join(root, sample_file)

        bgr = cv2.imread(img_path)
        if bgr is None:
            continue

        resized = cv2.resize(bgr, (128, 128))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        glcm_feat, glcm_matrix = extract_glcm_features(gray, multidirectional=False)
        lbp_feat, lbp_img = extract_lbp_features(gray)
        hog_feat, hog_img = extract_hog_features(gray, visualize=True)
        color_feat = extract_color_histogram(resized)
        edge_feat, edge_img = extract_edge_features(gray)

        print(f"Visualizing features for class: {label} (Image: {sample_file})")
        plot_feature_gallery(
            original_rgb=rgb,
            lbp_image=lbp_img,
            hog_image=hog_img,
            glcm_matrix=glcm_matrix,
            edge_image=edge_img,
            color_hist=color_feat,
            title=f"Feature Visualization - {label}"
        )


if __name__ == "__main__":
    main()

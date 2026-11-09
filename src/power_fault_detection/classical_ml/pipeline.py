"""End-to-end dataset feature extraction and classical ML training pipeline."""

import os
from typing import Tuple, Dict, Any, List
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from power_fault_detection.config import RANDOM_SEED
from power_fault_detection.classical_cv.features import extract_composite_features
from power_fault_detection.classical_ml.models import get_classical_classifiers
from power_fault_detection.utils.metrics import compute_one_vs_rest_metrics


def build_dataset_feature_matrix(
    dataset_path: str,
    target_size: Tuple[int, int] = (128, 128)
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extracts handcrafted visual feature vectors across all images in a categorized directory.

    Returns:
        X: Feature matrix of shape (n_samples, n_features).
        y: Ground truth labels of shape (n_samples,).
        classes: Unique sorted class names.
    """
    X_list, y_list = [], []

    for root, _, files in os.walk(dataset_path):
        img_files = [f for f in files if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        if not img_files:
            continue

        label = os.path.basename(root)
        for fname in img_files:
            img_path = os.path.join(root, fname)
            img = cv2.imread(img_path)
            if img is None:
                continue

            features = extract_composite_features(img, target_size=target_size)
            X_list.append(features)
            y_list.append(label)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list)
    classes = sorted(list(set(y_list)))

    return X, y, classes


def train_and_benchmark_models(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = RANDOM_SEED
) -> Dict[str, Dict[str, Any]]:
    """Trains and compares all classical ML estimators on extracted features."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    models = get_classical_classifiers(random_state=random_state)
    results = {}
    classes = sorted(list(np.unique(y)))

    for name, clf in models.items():
        print(f"\n--- Training {name} ---")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred, labels=classes)
        ovr_metrics = compute_one_vs_rest_metrics(cm, classes)
        report = classification_report(y_test, y_pred, target_names=classes, zero_division=0)

        results[name] = {
            "model": clf,
            "accuracy": acc,
            "confusion_matrix": cm,
            "one_vs_rest": ovr_metrics,
            "classification_report": report
        }

    return results

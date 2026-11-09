"""Runs handcrafted feature extraction and benchmarks classical ML classifiers."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from power_fault_detection.config import DATASET_DIR
from power_fault_detection.classical_ml.pipeline import (
    build_dataset_feature_matrix,
    train_and_benchmark_models
)
from power_fault_detection.utils.visualization import plot_confusion_matrix


def main(dataset_path: str = DATASET_DIR):
    print(f"Loading and extracting features from: {dataset_path}")
    if not os.path.exists(dataset_path):
        print(f"Directory not found: {dataset_path}")
        return

    X, y, class_names = build_dataset_feature_matrix(dataset_path)
    print(f"Extracted {X.shape[0]} samples with {X.shape[1]} features across {len(class_names)} classes.")

    results = train_and_benchmark_models(X, y)

    for name, res in results.items():
        print(f"\n================ {name} ================")
        print(f"Test Accuracy: {res['accuracy'] * 100:.2f}%\n")
        print("Confusion Matrix:\n", res['confusion_matrix'])
        print("\nOne-vs-Rest TP, TN, FP, FN breakdown:")
        for cls, metrics in res['one_vs_rest'].items():
            print(f"  {cls} -> TP: {metrics['TP']}, TN: {metrics['TN']}, FP: {metrics['FP']}, FN: {metrics['FN']}")
        print(f"\nClassification Report:\n{res['classification_report']}")

        plot_confusion_matrix(
            cm=res['confusion_matrix'],
            class_names=class_names,
            title=f"Confusion Matrix: {name}"
        )


if __name__ == "__main__":
    main()

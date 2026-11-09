"""Deep learning model evaluation, confusion matrix extraction, and metric reporting."""

import os
from typing import List, Dict, Any, Optional
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from power_fault_detection.config import OUTPUT_DIR
from power_fault_detection.utils.metrics import compute_one_vs_rest_metrics
from power_fault_detection.utils.visualization import plot_confusion_matrix


def evaluate_cnn_model(
    model: tf.keras.Model,
    name: str,
    data_generator,
    class_names: List[str],
    data_type: str = "Testing",
    plot_cm: bool = True,
    save_dir: str = OUTPUT_DIR
) -> Dict[str, Any]:
    """
    Evaluates a trained CNN model on evaluation/test data generators.

    Computes:
    - Step-based batched prediction alignment
    - Multi-class Confusion Matrix
    - One-vs-Rest TP, TN, FP, FN breakdown
    - Precision, Recall, and F1-Score classification report
    - Confusion Matrix visualization saved to disk
    """
    print(f"\n====== Evaluating {name} on {data_type} Data ======")
    data_generator.reset()

    steps = (
        data_generator.n // data_generator.batch_size
        + (data_generator.n % data_generator.batch_size > 0)
    )

    y_true = data_generator.classes
    y_pred_probs = model.predict(data_generator, steps=steps, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = y_true[:len(y_pred)]

    cm = confusion_matrix(y_true, y_pred)
    ovr_metrics = compute_one_vs_rest_metrics(cm, class_names)
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)

    print("\nConfusion Matrix:\n", cm)
    print("\nTP, TN, FP, FN for each class:")
    for cls_name, m in ovr_metrics.items():
        print(f"  {cls_name} -> TP: {m['TP']}, TN: {m['TN']}, FP: {m['FP']}, FN: {m['FN']}")

    print(f"\nClassification Report:\n{report}")

    if plot_cm:
        save_plot_path = os.path.join(save_dir, f"{name}_{data_type.lower()}_confusion_matrix.png")
        plot_confusion_matrix(
            cm=cm,
            class_names=class_names,
            title=f"Confusion Matrix: {name} ({data_type})",
            save_path=save_plot_path,
            show_plot=False
        )
        print(f"Saved confusion matrix plot to: {save_plot_path}")

    return {
        "confusion_matrix": cm,
        "one_vs_rest": ovr_metrics,
        "classification_report": report
    }

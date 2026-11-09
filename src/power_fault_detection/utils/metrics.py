"""Metrics calculation utilities for multi-class classification evaluation."""

from typing import Dict, List, Any
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support


def compute_one_vs_rest_metrics(cm: np.ndarray, class_names: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Computes per-class One-vs-Rest True Positives (TP), True Negatives (TN),
    False Positives (FP), and False Negatives (FN) from a confusion matrix.
    """
    metrics = {}
    total_samples = np.sum(cm)

    for i, name in enumerate(class_names):
        try:
            tp = int(cm[i, i])
            fp = int(np.sum(cm[:, i]) - tp)
            fn = int(np.sum(cm[i, :]) - tp)
            tn = int(total_samples - tp - fp - fn)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            metrics[name] = {
                "TP": tp,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "Precision": round(precision, 4),
                "Recall": round(recall, 4),
                "Specificity": round(specificity, 4),
                "F1_Score": round(f1, 4),
            }
        except IndexError:
            metrics[name] = {"Error": "Class index out of bounds"}

    return metrics


def generate_evaluation_summary(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]) -> str:
    """Generates a formatted classification report string."""
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    summary = f"\nOverall Accuracy: {acc:.4f}\n\nClassification Report:\n{report}"
    return summary

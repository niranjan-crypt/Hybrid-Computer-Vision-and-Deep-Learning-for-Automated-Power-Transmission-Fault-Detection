"""Visualization utilities for model evaluations and feature maps."""

import os
from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
import cv2


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> plt.Figure:
    """Plots and optionally saves a colored confusion matrix with numeric annotations."""
    fig = plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title, fontsize=14, pad=15)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right', fontsize=10)
    plt.yticks(tick_marks, class_names, fontsize=10)

    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(
            j, i, format(cm[i, j], 'd'),
            ha="center", va="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=11
        )

    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_feature_gallery(
    original_rgb: np.ndarray,
    lbp_image: np.ndarray,
    hog_image: np.ndarray,
    glcm_matrix: np.ndarray,
    edge_image: np.ndarray,
    color_hist: List[float],
    title: str = "Feature Visualization",
    save_path: Optional[str] = None
) -> None:
    """Visualizes extracted handcrafted visual features across a 6-panel subplot gallery."""
    fig, axs = plt.subplots(1, 6, figsize=(22, 4))

    axs[0].imshow(original_rgb)
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    axs[1].imshow(lbp_image, cmap="gray")
    axs[1].set_title("LBP Texture")
    axs[1].axis("off")

    axs[2].imshow(hog_image, cmap="gray")
    axs[2].set_title("HOG Gradients")
    axs[2].axis("off")

    axs[3].imshow(glcm_matrix[:, :, 0, 0], cmap="inferno")
    axs[3].set_title("GLCM Co-occurrence")
    axs[3].axis("off")

    axs[4].imshow(edge_image, cmap="gray")
    axs[4].set_title("Canny Edges")
    axs[4].axis("off")

    axs[5].bar(range(len(color_hist)), color_hist, color='steelblue')
    axs[5].set_title("HSV Color Histogram")
    axs[5].set_xlim(0, len(color_hist))

    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

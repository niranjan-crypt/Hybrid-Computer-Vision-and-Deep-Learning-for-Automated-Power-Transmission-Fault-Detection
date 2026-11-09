"""End-to-end deep learning benchmark runner for power transmission fault classification."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from power_fault_detection.config import (
    TRAIN_VAL_DIR, TEST_DIR, IMG_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE
)
from power_fault_detection.deep_learning.data_loader import get_data_generators
from power_fault_detection.deep_learning.models.alexnet import build_alexnet
from power_fault_detection.deep_learning.models.vgg16 import build_vgg16
from power_fault_detection.deep_learning.models.googlenet import build_googlenet
from power_fault_detection.deep_learning.models.resnet50 import build_resnet50
from power_fault_detection.deep_learning.trainer import train_cnn_model
from power_fault_detection.deep_learning.evaluator import evaluate_cnn_model


def main():
    print("=" * 60)
    print("Automated Power Transmission Fault Detection - Deep Learning Pipeline")
    print("=" * 60)

    train_data, val_data, test_data = get_data_generators(
        train_val_dir=TRAIN_VAL_DIR,
        test_dir=TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    num_classes = train_data.num_classes
    class_names = list(train_data.class_indices.keys())
    print(f"Detected {num_classes} Fault Classes: {class_names}")

    # Build the 4 model architectures
    models = {
        "AlexNet": build_alexnet(num_classes=num_classes),
        "VGG16": build_vgg16(num_classes=num_classes),
        "GoogLeNet": build_googlenet(num_classes=num_classes),
        "ResNet50": build_resnet50(num_classes=num_classes)
    }

    # Train and evaluate each architecture
    eval_target = test_data if test_data is not None else val_data
    target_name = "Testing" if test_data is not None else "Validation"

    for name, model in models.items():
        train_cnn_model(
            model=model,
            name=name,
            train_data=train_data,
            val_data=val_data,
            epochs=EPOCHS,
            lr=LEARNING_RATE
        )

        evaluate_cnn_model(
            model=model,
            name=name,
            data_generator=eval_target,
            class_names=class_names,
            data_type=target_name
        )

    print("\nAll Deep Learning models trained and benchmarked successfully!")


if __name__ == "__main__":
    main()

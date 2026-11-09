"""Data pipeline and image augmentations using Keras ImageDataGenerator."""

from typing import Tuple, Optional
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, DirectoryIterator
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from power_fault_detection.config import (
    IMG_SIZE, BATCH_SIZE, VALIDATION_SPLIT, TRAIN_VAL_DIR, TEST_DIR
)


def get_data_generators(
    train_val_dir: str = TRAIN_VAL_DIR,
    test_dir: Optional[str] = TEST_DIR,
    target_size: Tuple[int, int] = IMG_SIZE,
    batch_size: int = BATCH_SIZE,
    val_split: float = VALIDATION_SPLIT
) -> Tuple[DirectoryIterator, DirectoryIterator, Optional[DirectoryIterator]]:
    """
    Constructs high-throughput image generators for training, validation, and testing.

    Training pipeline applies data augmentations (flips, rotations, zoom, shear)
    to improve generalization and prevent overfitting.
    Validation and test sets apply only canonical preprocessing and keep shuffle=False.
    """
    train_val_datagen = ImageDataGenerator(
        preprocessing_function=vgg_preprocess,
        validation_split=val_split,
        horizontal_flip=True,
        rotation_range=10,
        zoom_range=0.2,
        shear_range=0.2
    )

    train_generator = train_val_datagen.flow_from_directory(
        train_val_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    val_generator = train_val_datagen.flow_from_directory(
        train_val_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    test_generator = None
    if test_dir and os.path.exists(test_dir):
        test_datagen = ImageDataGenerator(preprocessing_function=vgg_preprocess)
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )

    return train_generator, val_generator, test_generator

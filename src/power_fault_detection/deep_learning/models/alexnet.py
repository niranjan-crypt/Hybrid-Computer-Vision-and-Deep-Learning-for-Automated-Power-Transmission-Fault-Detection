"""AlexNet convolutional neural network implemented from scratch."""

from typing import Tuple
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from power_fault_detection.config import INPUT_SHAPE, L2_REGULARIZATION, DROPOUT_RATE


def build_alexnet(
    input_shape: Tuple[int, int, int] = INPUT_SHAPE,
    num_classes: int = 2,
    l2_reg: float = L2_REGULARIZATION,
    dropout_rate: float = DROPOUT_RATE
) -> Sequential:
    """
    Constructs an AlexNet architecture tailored for power transmission fault detection.

    Consists of 5 Convolutional layers with Batch Normalization, Max Pooling,
    and 3 Fully Connected layers regularized via L2 weight decay and Dropout.
    """
    model = Sequential(name="AlexNet_Custom")

    # Layer 1: Conv -> BN -> Pool
    model.add(Conv2D(96, (11, 11), strides=(4, 4), activation='relu', input_shape=input_shape, kernel_regularizer=l2(l2_reg)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3), strides=(2, 2)))

    # Layer 2: Conv -> BN -> Pool
    model.add(Conv2D(256, (5, 5), padding="same", activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3), strides=(2, 2)))

    # Layers 3-5: Three consecutive Conv layers
    model.add(Conv2D(384, (3, 3), padding="same", activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(Conv2D(384, (3, 3), padding="same", activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(Conv2D(256, (3, 3), padding="same", activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(MaxPooling2D((3, 3), strides=(2, 2)))

    # Classifier Head
    model.add(Flatten())
    model.add(Dense(4096, activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(4096, activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation='softmax', name="predictions"))

    return model

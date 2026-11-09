"""GoogLeNet (InceptionV3) Transfer Learning model architecture."""

from typing import Tuple
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from power_fault_detection.config import INPUT_SHAPE, L2_REGULARIZATION, DROPOUT_RATE


def build_googlenet(
    input_shape: Tuple[int, int, int] = INPUT_SHAPE,
    num_classes: int = 2,
    l2_reg: float = L2_REGULARIZATION,
    dropout_rate: float = DROPOUT_RATE
) -> Model:
    """
    Constructs a transfer learning model utilizing the multi-scale InceptionV3 (GoogLeNet) backbone.

    Multi-scale convolutional kernels capture both fine-grained surface defects (e.g. rust pits)
    and large-scale macro anomalies (e.g. bird nests, missing insulator discs).
    """
    base = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)

    x = GlobalAveragePooling2D(name="gap")(base.output)
    x = Dense(512, activation='relu', kernel_regularizer=l2(l2_reg), name="dense_512")(x)
    x = Dropout(dropout_rate, name="dropout_1")(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(l2_reg), name="dense_256")(x)
    x = Dropout(dropout_rate, name="dropout_2")(x)
    predictions = Dense(num_classes, activation='softmax', name="predictions")(x)

    model = Model(inputs=base.input, outputs=predictions, name="GoogLeNet_InceptionV3")
    return model

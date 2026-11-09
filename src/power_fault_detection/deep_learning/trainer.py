"""Deep learning model compilation, layer freezing, and training pipeline."""

import os
from typing import Optional
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from power_fault_detection.config import EPOCHS, LEARNING_RATE, OUTPUT_DIR


def train_cnn_model(
    model: tf.keras.Model,
    name: str,
    train_data,
    val_data,
    epochs: int = EPOCHS,
    lr: float = LEARNING_RATE,
    save_dir: str = OUTPUT_DIR
) -> tf.keras.callbacks.History:
    """
    Trains a CNN architecture with transfer learning layer-freezing protocols.

    For pre-trained transfer learning models (VGG16, GoogLeNet, ResNet50),
    the base feature extractor is frozen, training only top projection layers.
    AlexNet is trained end-to-end from scratch.
    """
    print(f"\n================ Training {name} ================")

    if name in ["VGG16", "GoogLeNet", "ResNet50"]:
        print(f"Freezing pre-trained backbone layers for {name} (training top 6 layers)")
        for layer in model.layers[:-6]:
            layer.trainable = False

    optimizer = Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"{name}_model.h5")

    callbacks = [
        ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1)
    ]

    history = model.fit(
        train_data,
        epochs=epochs,
        validation_data=val_data,
        callbacks=callbacks,
        verbose=1
    )

    # Save final weights
    model.save(model_path)
    print(f"Successfully saved {name} to: {model_path}")
    return history

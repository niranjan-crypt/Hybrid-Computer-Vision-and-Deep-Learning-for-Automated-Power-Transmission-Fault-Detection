import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# ---------------- PARAMETERS ----------------
TRAIN_VAL_DIR = r"C:\Users\Gaura\Downloads\Dataset_root_final" # Folder with class subfolders (train+val)
TEST_DIR = r"C:\Users\Gaura\Downloads\Test_final"  # Separate test folder
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.0001

# ---------------- DATA GENERATORS ----------------
# Data augmentation for training
train_val_datagen = ImageDataGenerator(
    preprocessing_function=vgg_preprocess,
    validation_split=0.2,  # 20% for validation
    horizontal_flip=True,
    rotation_range=10,
    zoom_range=0.2,
    shear_range=0.2
)

# Generator for training subset
train_data = train_val_datagen.flow_from_directory(
    TRAIN_VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

# Generator for validation subset
val_data = train_val_datagen.flow_from_directory(
    TRAIN_VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False  # Important for consistent evaluation
)

# Generator for testing data (no augmentation, only preprocessing)
test_datagen = ImageDataGenerator(preprocessing_function=vgg_preprocess)
test_data = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

NUM_CLASSES = train_data.num_classes
CLASS_NAMES = list(train_data.class_indices.keys())
print(f"Detected {NUM_CLASSES} classes: {CLASS_NAMES}")

# ---------------- MODEL ARCHITECTURES ----------------
def build_alexnet(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
    model = Sequential()
    model.add(tf.keras.layers.Conv2D(96, (11,11), strides=(4,4), activation='relu', input_shape=input_shape, kernel_regularizer=l2(0.0005)))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((3,3), strides=(2,2)))
    model.add(tf.keras.layers.Conv2D(256, (5,5), padding="same", activation='relu', kernel_regularizer=l2(0.0005)))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((3,3), strides=(2,2)))
    model.add(tf.keras.layers.Conv2D(384, (3,3), padding="same", activation='relu', kernel_regularizer=l2(0.0005)))
    model.add(tf.keras.layers.Conv2D(384, (3,3), padding="same", activation='relu', kernel_regularizer=l2(0.0005)))
    model.add(tf.keras.layers.Conv2D(256, (3,3), padding="same", activation='relu', kernel_regularizer=l2(0.0005)))
    model.add(tf.keras.layers.MaxPooling2D((3,3), strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu', kernel_regularizer=l2(0.0005)))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu', kernel_regularizer=l2(0.0005)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def build_vgg16(num_classes=NUM_CLASSES):
    base = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.0005))(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.0005))(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=base.input, outputs=predictions)

def build_googlenet(num_classes=NUM_CLASSES):
    base = InceptionV3(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.0005))(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.0005))(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=base.input, outputs=predictions)

def build_resnet50(num_classes=NUM_CLASSES):
    base = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.0005))(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.0005))(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=base.input, outputs=predictions)

MODELS = {
    "AlexNet": build_alexnet(),
    "VGG16": build_vgg16(),
    "GoogLeNet": build_googlenet(),
    "ResNet50": build_resnet50(),
}

# ---------------- TRAINING AND EVALUATION ----------------
def train_model(model, name):
    print(f"\n====== Training {name} ======")
    if name in ["VGG16", "GoogLeNet", "ResNet50"]:
        for layer in model.layers[:-6]:
            layer.trainable = False
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_data, epochs=EPOCHS, validation_data=val_data, verbose=1)
    model.save(f"{name}_model.h5")
    print(f"{name} model saved!")

def evaluate_model(model, name, data, data_type="Testing"):
    print(f"\n====== Evaluating {name} on {data_type} Data ======")
    data.reset()
    steps = data.n // data.batch_size + (data.n % data.batch_size > 0)
    y_true = data.classes
    y_pred_probs = model.predict(data, steps=steps, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = y_true[:len(y_pred)]
    
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:\n", cm)
    
    print("\nTP, TN, FP, FN for each class:")
    for i, class_name in enumerate(CLASS_NAMES):
        TP = cm[i,i]
        FP = np.sum(cm[:,i]) - TP
        FN = np.sum(cm[i,:]) - TP
        TN = np.sum(cm) - TP - FP - FN
        print(f"{class_name} -> TP:{TP} TN:{TN} FP:{FP} FN:{FN}")
    
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))
    
    plt.figure(figsize=(10,8))
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for {name} ({data_type})')
    plt.colorbar()
    tick_marks = np.arange(len(CLASS_NAMES))
    plt.xticks(tick_marks, CLASS_NAMES, rotation=45, ha='right')
    plt.yticks(tick_marks, CLASS_NAMES)
    
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, cm[i,j], ha='center', va='center', color='white' if cm[i,j]>thresh else 'black')
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

# ---------------- RUN ALL MODELS ----------------
for name, model in MODELS.items():
    train_model(model, name)
    evaluate_model(model, name, test_data, data_type="Testing")

print("\nAll models trained and evaluated successfully!")

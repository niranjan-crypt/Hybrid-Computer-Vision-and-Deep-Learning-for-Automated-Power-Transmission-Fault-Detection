import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# --- 1. Parameters & Data Loading ---
# Parameters
# NOTE: Replace the placeholder with your actual data directory path
data_dir = r"C:\Users\Gaura\Downloads\Dataset_root (2)\Dataset_root"
img_size = 224
batch_size = 32
epochs = 20 
learning_rate = 0.0001 

# Data generators to automatically split the data
data_gen = ImageDataGenerator(
    preprocessing_function=vgg_preprocess, 
    validation_split=0.2, 
    horizontal_flip=True,
    rotation_range=10,
    zoom_range=0.2,
    shear_range=0.2 
)

# Training data (80% of the dataset)
train_data = data_gen.flow_from_directory(
    data_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

# Validation data generator (renamed from test_data_gen)
validation_data_gen = ImageDataGenerator(
    preprocessing_function=vgg_preprocess,
    validation_split=0.2 
)

# Validation data (the 20% validation subset, renamed from test_data)
validation_data = validation_data_gen.flow_from_directory(
    data_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False # Crucial for confusion matrix to match predictions to true labels
)

num_classes = train_data.num_classes
class_names = list(train_data.class_indices.keys())
print(f"Detected {num_classes} classes: {class_names}")

# --- 2. Model Architectures ---
def build_alexnet(input_shape=(224, 224, 3), num_classes=2):
    model = Sequential()
    # 1. Convolutional Layer: 96 filters, (11,11) kernel, stride 4
    model.add(tf.keras.layers.Conv2D(96, (11, 11), strides=(4, 4), activation='relu', input_shape=input_shape, kernel_regularizer=l2(0.0005)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2)))
    
    # 2. Convolutional Layer: 256 filters, (5,5) kernel, stride 1 (padded)
    model.add(tf.keras.layers.Conv2D(256, (5, 5), padding="same", activation='relu', kernel_regularizer=l2(0.0005)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2)))
    
    # 3. Convolutional Layer: 384 filters, (3,3) kernel, stride 1 (padded)
    model.add(tf.keras.layers.Conv2D(384, (3, 3), padding="same", activation='relu', kernel_regularizer=l2(0.0005)))
    
    # 4. Convolutional Layer: 384 filters, (3,3) kernel, stride 1 (padded)
    model.add(tf.keras.layers.Conv2D(384, (3, 3), padding="same", activation='relu', kernel_regularizer=l2(0.0005)))
    
    # 5. Convolutional Layer: 256 filters, (3,3) kernel, stride 1 (padded)
    model.add(tf.keras.layers.Conv2D(256, (3, 3), padding="same", activation='relu', kernel_regularizer=l2(0.0005)))
    model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2)))
    
    # Flatten and Fully Connected Layers
    model.add(Flatten())
    
    # 6. Dense Layer: 4096 neurons
    model.add(Dense(4096, activation='relu', kernel_regularizer=l2(0.0005)))
    model.add(Dropout(0.5))
    
    # 7. Dense Layer: 4096 neurons
    model.add(Dense(4096, activation='relu', kernel_regularizer=l2(0.0005)))
    model.add(Dropout(0.5))
    
    # 8. Output Layer
    model.add(Dense(num_classes, activation='softmax'))
    return model


def build_vgg16(num_classes):
    base = VGG16(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
    
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.0005))(x) 
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.0005))(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs=base.input, outputs=predictions)


def build_googlenet(num_classes):
    base = InceptionV3(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
    
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.0005))(x) 
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.0005))(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs=base.input, outputs=predictions)


def build_resnet50(num_classes):
    base = ResNet50(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
    
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.0005))(x) 
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.0005))(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs=base.input, outputs=predictions)


# Dictionary of models for convenience
models = {
    "AlexNet": build_alexnet(input_shape=(img_size, img_size, 3), num_classes=num_classes),
    "VGG16": build_vgg16(num_classes),
    "GoogLeNet": build_googlenet(num_classes),
    "ResNet50": build_resnet50(num_classes),
}


# --- 3. Training and Evaluation Loop ---
# Changed parameter name from test_data to validation_data
def train_and_evaluate(model, name, train_data, validation_data, epochs, lr=learning_rate):
    print(f"================ Training {name} ================ ")
    
    # Freeze base models if transfer learning is used
    if name in ["VGG16", "GoogLeNet", "ResNet50"]:
        for layer in model.layers[:-6]: 
            layer.trainable = False
    elif name == "AlexNet":
        pass
            
    # Compile the model - Using Adam with a specified learning rate
    optimizer = Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Train the model
    history = model.fit(
        train_data,
        epochs=epochs,
        validation_data=validation_data, # Changed variable name
        verbose=1
    )
    
    # Save the trained model
    model.save(f"{name}_power_fault_model.h5")
    print(f"\n{name} training complete and model saved!\n")
    
    # Evaluation
    evaluate_model(model, name, validation_data) # Changed variable name


# Changed parameter name from test_data to validation_data
def evaluate_model(model, name, validation_data):
    print(f"================ Evaluating {name} on Validation Data ================ ") # Changed print output
    
    # Reset validation data generator to ensure predictions are in order
    validation_data.reset()
    
    # Calculate steps for prediction
    steps = validation_data.n // validation_data.batch_size + (validation_data.n % validation_data.batch_size > 0)

    # Get true labels and make predictions
    y_true = validation_data.classes
    y_pred_probs = model.predict(validation_data, steps=steps, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Ensure y_true and y_pred have the same length (important due to generator batching)
    y_true = y_true[:len(y_pred)]
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    print("\nConfusion Matrix:")
    print(cm)
    
    # Calculate and display TP, TN, FP, FN for each class
    print("\nTP, TN, FP, FN Matrix:")
    for i, class_name in enumerate(class_names):
        try:
            TP = cm[i, i]
            FP = np.sum(cm[:, i]) - TP
            FN = np.sum(cm[i, :]) - TP
            TN = np.sum(cm) - TP - FP - FN
            print(f"    Class: {class_name}")
            print(f"      TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
        except IndexError:
             print(f"    Class: {class_name} - Error in calculating metrics, possibly missing class in predictions.")


    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    
    # Plot the confusion matrix using matplotlib
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for {name}')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    
    # --- MODIFICATION FOR X-AXIS ALIGNMENT ---
    # Changing rotation from 45 to 60 and setting horizontalalignment to 'right'
    plt.xticks(tick_marks, class_names, rotation=90, ha='right') 
    plt.yticks(tick_marks, class_names)
    # ------------------------------------------
    
    # Add text annotations for each cell
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                  ha="center", va="center",
                  color="white" if cm[i, j] > thresh else "black")
                  
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()


# Run the training and evaluation for each model
for name, model in models.items():
    train_and_evaluate(model, name, train_data, validation_data, epochs) # Changed variable name


print("\nAll models have been trained and evaluated.")


print("\n================ Data Information ================")
total_data_points = train_data.n + validation_data.n # Changed variable name
print(f"Total number of data points: {total_data_points}")
print(f"Number of training data points: {train_data.n}")
print(f"Number of validation data points: {validation_data.n}") # Changed print output
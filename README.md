# Hybrid Computer Vision and Deep Learning for Automated Power Transmission Fault Detection

An automated inspection and fault diagnosis system for power transmission infrastructure using a hybrid approach combining classical computer vision, handcrafted feature engineering, machine learning classifiers, and deep convolutional neural networks (CNNs).

---

## 📌 Project Overview

Power transmission lines and electrical grid components (e.g., insulator strings, towers, and lines) face environmental and physical degradation, including:
- **Broken / Missing Insulator Glass Discs**
- **Rust / Corrosion on Metallic Hardware**
- **Foreign Object Infestation (e.g., Bird Nests)**

This project explores and benchmarks two complementary paradigms:
1. **Classical Computer Vision & Machine Learning**: Domain-specific feature extraction (GLCM, LBP, HOG, Color Histograms, Canny edges) coupled with traditional ML classifiers (SVM, Random Forest, KNN, Logistic Regression).
2. **Deep Learning & Transfer Learning**: End-to-end multi-class classification utilizing four architectures: AlexNet (from scratch), VGG16, GoogLeNet (InceptionV3), and ResNet50.

---

## 🛠️ Repository Structure

```
├── deep_lrn_ninj.py    # Complete Deep Learning pipeline (Train, Val, Test split, 4 CNNs, Evaluation)
├── cv deeplrn.py       # Deep Learning training & validation script
├── 1.py                # Feature extraction & multi-feature visualization (Color, LBP, HOG, GLCM, Canny)
├── 2.py                # Feature extraction & ML classifier training (SVM, RF, LR, KNN)
├── 3.py                # Feature extraction & model evaluation metrics
├── final.txt           # Unified pipeline for classical feature extraction and ML classification
├── glass.txt           # Insulator disc detection via Hough Circles & Template Matching
├── rust.txt            # Rust detection using HSV color thresholding & morphological filtering
├── nest.txt            # Bird nest detection via edge, contour, and texture analysis
├── all3codes.txt       # Combined script for glass disc, rust, and nest fault detection
├── .gitignore          # Git ignore file
└── README.md           # Project documentation
```

---

## 🔬 Methodology

### 1. Classical Computer Vision & Defect Detection
- **Insulator Glass Disc Detection**: Utilizes Gaussian blur followed by the **Hough Circle Transform** (`cv2.HoughCircles`) to segment individual insulator discs. Defective/missing discs are identified via **Normalized Cross-Correlation Template Matching** (`cv2.matchTemplate`).
- **Rust Detection**: Converts images to the **HSV color space** to segment rust and corrosion regions using color threshold masks combined with morphological filtering.
- **Bird Nest Detection**: Uses Local Binary Patterns (LBP) difference masking and Canny edge analysis to identify anomalous texture clusters on transmission towers.

### 2. Handcrafted Feature Extraction
- **GLCM (Gray-Level Co-occurrence Matrix)**: Extracts contrast, energy, homogeneity, correlation, and angular second moment (ASM).
- **LBP (Local Binary Patterns)**: Captures micro-texture distributions.
- **HOG (Histogram of Oriented Gradients)**: Encodes structural shape and directional edge gradients.
- **HSV Color Histograms**: Quantifies color distributions across Hue, Saturation, and Value channels.
- **Canny Edge Density**: Computes structural edge presence across the image.

### 3. Machine Learning Classifiers
- Support Vector Classifier (SVC with RBF kernel)
- Random Forest Classifier
- Logistic Regression
- K-Nearest Neighbors (KNN)

### 4. Deep Learning Architectures
- **AlexNet**: Custom implementation built from scratch using 5 convolutional layers, batch normalization, max pooling, and dense dropout layers.
- **VGG16**: Transfer learning using ImageNet pre-trained weights, fine-tuned with a custom classification head (GlobalAveragePooling2D, Dense, Dropout).
- **GoogLeNet (InceptionV3)**: Multi-scale inception modules pre-trained on ImageNet.
- **ResNet50**: Deep residual network utilizing shortcut connections to mitigate vanishing gradients.

---

## 📊 Evaluation & Metrics

The deep learning pipeline comprehensively evaluates each model on test data:
- **Confusion Matrix & Heatmap Visualization**
- **One-vs-Rest Per-Class Analysis**: True Positives (TP), True Negatives (TN), False Positives (FP), False Negatives (FN)
- **Classification Report**: Precision, Recall, F1-Score, and Support

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- TensorFlow 2.x
- OpenCV (`opencv-python`)
- scikit-image
- scikit-learn
- NumPy, Matplotlib

```bash
pip install tensorflow opencv-python scikit-image scikit-learn numpy matplotlib
```

### Running the Deep Learning Pipeline
Update the dataset paths in `deep_lrn_ninj.py`:
```python
TRAIN_VAL_DIR = "path/to/Dataset_root_final"
TEST_DIR = "path/to/Test_final"
```
Run the script:
```bash
python deep_lrn_ninj.py
```

### Running Classical CV & ML
```bash
python 1.py    # To visualize extracted features
python 2.py    # To train and evaluate classical ML models
```

---

## 📄 License
This repository is released under the MIT License.

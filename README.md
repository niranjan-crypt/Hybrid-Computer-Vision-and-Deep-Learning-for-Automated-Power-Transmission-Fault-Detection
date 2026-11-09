# Hybrid Computer Vision and Deep Learning for Automated Power Transmission Fault Detection

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

An automated inspection, detection, and diagnostic system for electrical power transmission infrastructure utilizing a hybrid framework that bridges domain-specific handcrafted computer vision algorithms, classical machine learning classifiers, and deep convolutional neural networks (CNNs).

---

## 📌 Project Overview

Transmission line maintenance and grid resilience require continuous inspection of critical components. This repository provides automated fault diagnosis for three key degradation categories:
1. **Broken / Shattered / Missing Insulator Glass Discs**
2. **Surface Rust and Atmospheric Corrosion on Steel Towers & Hardware**
3. **Foreign Object Infestation (e.g., Bird Nests and Vegetation on Pylons)**

The project implements two complementary diagnostic pipelines:
- **Classical Computer Vision & Machine Learning**: Handcrafted texture (GLCM, LBP), structural gradient (HOG), color distribution (HSV histograms), and edge density (Canny) features paired with SVM, Random Forest, Logistic Regression, and KNN classifiers.
- **Deep Convolutional Neural Networks & Transfer Learning**: Benchmark across 4 CNN architectures (AlexNet trained from scratch, VGG16, GoogLeNet/InceptionV3, and ResNet50) with automated data augmentation, layer freezing, and One-vs-Rest confusion matrix analytics.

---

## 🏗️ Modular Architecture & Repository Structure

```
├── requirements.txt                         # Python package dependencies
├── .gitignore                               # Git ignore specification
├── README.md                                # Comprehensive project documentation
│
├── src/                                     # Core modular package
│   └── power_fault_detection/
│       ├── __init__.py
│       ├── config.py                        # Centralized hyperparameters & paths
│       │
│       ├── classical_cv/                    # Classical CV & Defect Detectors
│       │   ├── __init__.py
│       │   ├── features.py                  # GLCM, LBP, HOG, HSV Hist, Canny
│       │   ├── glass_detector.py            # Hough Circles & Template Matching
│       │   ├── rust_detector.py             # HSV Color Masking & Morphology
│       │   └── nest_detector.py             # LBP Texture Anomaly Detection
│       │
│       ├── classical_ml/                    # Classical Machine Learning
│       │   ├── __init__.py
│       │   ├── models.py                    # SVM, Random Forest, KNN, Logistic Regression
│       │   └── pipeline.py                  # Feature matrix extraction & model training
│       │
│       ├── deep_learning/                   # Deep Learning Pipelines
│       │   ├── __init__.py
│       │   ├── data_loader.py               # ImageDataGenerator with augmentations
│       │   ├── trainer.py                   # Layer freezing, Adam optimizer & training
│       │   ├── evaluator.py                 # Multi-class CM, One-vs-Rest TP/TN/FP/FN
│       │   └── models/                      # CNN Architectures
│       │       ├── __init__.py
│       │       ├── alexnet.py               # Custom AlexNet from scratch
│       │       ├── vgg16.py                 # VGG16 Transfer Learning
│       │       ├── googlenet.py             # InceptionV3 Transfer Learning
│       │       └── resnet50.py              # ResNet50 Residual Transfer Learning
│       │
│       └── utils/                           # Common Utilities
│           ├── __init__.py
│           ├── metrics.py                   # TP, TN, FP, FN, Precision, Recall, F1
│           └── visualization.py             # CM heatmaps & 6-panel feature galleries
│
├── scripts/                                 # Executable CLI Runner Scripts
│   ├── run_classical_features.py            # Visualizes extracted handcrafted features
│   ├── run_fault_detection.py               # Inspects glass discs, rust, and bird nests
│   ├── run_classical_ml.py                  # Trains & evaluates classical ML classifiers
│   └── run_deep_learning.py                 # Trains & benchmarks the 4 DL models
│
└── legacy/                                  # Archived original prototype scripts
    ├── 1.py
    ├── 2.py
    ├── 3.py
    ├── cv deeplrn.py
    ├── deep_lrn_ninj.py
    ├── final.txt
    ├── glass.txt
    ├── nest.txt
    ├── rust.txt
    └── all3codes.txt
```

---

## 🔬 Methodology & Components

### 1. Classical Defect Inspection Algorithms
- **Insulator Glass Discs**: Identified using Gaussian blur and Hough Circle Transform (`cv2.HOUGH_GRADIENT`). Defective or missing units are localized via Normalized Cross-Correlation Template Matching (`cv2.TM_CCOEFF_NORMED`) against standard intact discs.
- **Surface Rust Detection**: Formulates a dual-band HSV color segmentation mask capturing both standard orange/brown rust `[5, 60, 60] - [25, 255, 255]` and oxidized dark reddish rust `[0, 40, 40] - [10, 255, 255]`. Noise is removed via median filtering.
- **Bird Nest Detection**: Employs Local Binary Pattern (LBP) difference masking between healthy transmission structures and inspection targets to detect foreign anomalous clusters.

### 2. Handcrafted Feature Representation
The classical ML pipeline fuses five distinct visual modalities into a single compact feature vector:
1. **GLCM Texture**: Contrast, Energy, Homogeneity, Correlation, Angular Second Moment (ASM).
2. **LBP Micro-texture**: 10-bin rotation-invariant uniform distribution.
3. **HOG Shape/Gradient**: 50 directional gradient histogram bins.
4. **HSV Color Histogram**: 48 joint color bins (16 bins per channel).
5. **Canny Edge Density**: Ratio of edge boundary pixels to total image area.

### 3. Deep Learning Architectures
| Architecture | Learning Paradigm | Total Layers | Key Characteristics |
| :--- | :--- | :--- | :--- |
| **AlexNet** | Trained From Scratch | 8 | 5 Convolutions, BatchNormalization, L2 Reg ($5 \times 10^{-4}$), 50% Dropout |
| **VGG16** | Transfer Learning | 16 | ImageNet pre-trained base, Frozen backbone, GlobalAveragePooling2D, Dense Head |
| **GoogLeNet** | Transfer Learning | InceptionV3 | Multi-scale parallel kernels ($1\times1, 3\times3, 5\times5$), Multi-receptive fields |
| **ResNet50** | Transfer Learning | 50 | Residual shortcut connections ($F(x) + x$), Mitigates vanishing gradient |

---

## 📊 Evaluation & Metrics

The framework evaluates all models using both aggregate and fine-grained class statistics:
- **One-vs-Rest Confusion Matrix Analysis**:
  $$TP = CM[i, i]$$
  $$FP = \sum_{k} CM[k, i] - TP$$
  $$FN = \sum_{k} CM[i, k] - TP$$
  $$TN = \text{Total} - TP - FP - FN$$
- **Classification Report**: Precision, Recall, Specificity, Macro/Weighted F1-Scores.
- **Visual Confusion Matrices**: Auto-generated annotated heatmaps saved to the `outputs/` folder.

---

## 🚀 Installation & Usage

### 1. Setup Environment
```bash
git clone https://github.com/niranjan-crypt/Hybrid-Computer-Vision-and-Deep-Learning-for-Automated-Power-Transmission-Fault-Detection.git
cd Hybrid-Computer-Vision-and-Deep-Learning-for-Automated-Power-Transmission-Fault-Detection
pip install -r requirements.txt
```

### 2. Configure Dataset Paths
Export environment variables or adjust defaults in `src/power_fault_detection/config.py`:
```bash
export TRAIN_VAL_DIR="/path/to/Dataset_root_final"
export TEST_DIR="/path/to/Test_final"
export DATASET_DIR="/path/to/Dataset_root"
```

### 3. Run Pipeline Scripts
```bash
# Visualize handcrafted feature extraction on sample images
python scripts/run_classical_features.py

# Run classical CV defect inspection (glass discs, rust, bird nest)
python scripts/run_fault_detection.py

# Train and benchmark Classical ML models (SVM, RF, KNN, Logistic Regression)
python scripts/run_classical_ml.py

# Train and benchmark Deep Learning models (AlexNet, VGG16, GoogLeNet, ResNet50)
python scripts/run_deep_learning.py
```

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog
import random

# ------------------ Step 1: Dataset path ------------------
dataset_path = r"C:\Users\Gaura\Downloads\Dataset_root"  # change this to your dataset folder

# ------------------ Step 2: Feature extraction functions ------------------
def extract_glcm_features(gray):
    glcm = graycomatrix(gray, [1], [0], 256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    return [contrast, energy, homogeneity], glcm

def extract_lbp_features(gray):
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist.tolist(), lbp

def extract_hog_features(gray):
    features, hog_image = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True)
    return features[:50].tolist(), hog_image

def extract_color_histogram(img, bins=16):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [bins], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [bins], [0, 256])
    hist = np.concatenate([hist_h, hist_s, hist_v]).ravel()
    hist = hist / (hist.sum() + 1e-6)
    return hist.tolist()

def extract_edge_features(gray):
    edges = cv2.Canny(gray, 100, 200)
    edge_density = edges.sum() / (edges.shape[0] * edges.shape[1])
    return [edge_density], edges

# ------------------ Step 3: Take 1 random image from each class ------------------
X, y = [], []

for root, dirs, files in os.walk(dataset_path):
    img_files = [f for f in files if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    if img_files:
        file = random.choice(img_files)  # pick a random image each run
        label = os.path.basename(root)
        path = os.path.join(root, file)

        img = cv2.imread(path)
        img = cv2.resize(img, (128, 128))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        glcm_feat, glcm_matrix = extract_glcm_features(gray)
        lbp_feat, lbp_image = extract_lbp_features(gray)
        hog_feat, hog_image = extract_hog_features(gray)
        color_feat = extract_color_histogram(img)
        edge_feat, edge_image = extract_edge_features(gray)

        features = glcm_feat + lbp_feat + hog_feat + color_feat + edge_feat
        X.append(features)
        y.append(label)

        # ------------------ Step 3.1: Visualization ------------------
        fig, axs = plt.subplots(1, 6, figsize=(20, 4))
        axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axs[0].set_title("Original (Color)")
        axs[0].axis("off")

        axs[1].imshow(lbp_image, cmap="gray")
        axs[1].set_title("LBP Features")
        axs[1].axis("off")

        axs[2].imshow(hog_image, cmap="gray")
        axs[2].set_title("HOG Features")
        axs[2].axis("off")

        axs[3].imshow(glcm_matrix[:, :, 0, 0], cmap="inferno")
        axs[3].set_title("GLCM Heatmap")
        axs[3].axis("off")

        axs[4].imshow(edge_image, cmap="gray")
        axs[4].set_title("Edge Features")
        axs[4].axis("off")

        axs[5].bar(range(len(color_feat)), color_feat)
        axs[5].set_title("Color Histogram")
        axs[5].axis("off")

        plt.suptitle(f"Feature Visualization - {label}")
        plt.show()

X = np.array(X)
y = np.array(y)

print("Feature extraction complete. Shape of X:", X.shape)
print("Labels extracted:", y)

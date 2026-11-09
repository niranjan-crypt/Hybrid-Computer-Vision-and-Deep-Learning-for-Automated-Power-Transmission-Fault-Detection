import os
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score
import matplotlib.pyplot as plt

# ------------------ Dataset path ------------------
dataset_path = r"C:\Users\Gaura\Downloads\Dataset_root"

# ------------------ Feature extraction ------------------
def extract_glcm_features(gray):
    glcm = graycomatrix(gray, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], 256, symmetric=True, normed=True)
    contrast = np.mean(graycoprops(glcm, 'contrast'))
    energy = np.mean(graycoprops(glcm, 'energy'))
    homogeneity = np.mean(graycoprops(glcm, 'homogeneity'))
    correlation = np.mean(graycoprops(glcm, 'correlation'))
    asm = np.mean(graycoprops(glcm, 'ASM'))
    return [contrast, energy, homogeneity, correlation, asm]

def extract_lbp_features(gray):
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
    hist = hist.astype("float") / (hist.sum() + 1e-6)
    return hist.tolist()

def extract_hog_features(gray):
    features, _ = hog(gray, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2),
                      block_norm='L2-Hys', visualize=True)
    return features[:50].tolist()

def extract_color_histogram(img, bins=16):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv],[0],None,[bins],[0,180])
    hist_s = cv2.calcHist([hsv],[1],None,[bins],[0,256])
    hist_v = cv2.calcHist([hsv],[2],None,[bins],[0,256])
    hist = np.concatenate([hist_h, hist_s, hist_v]).ravel()
    hist = hist / (hist.sum() + 1e-6)
    return hist.tolist()

def extract_edge_features(gray):
    edges = cv2.Canny(gray, 100, 200)
    edge_density = edges.sum() / (edges.shape[0]*edges.shape[1])
    return [edge_density]

# ------------------ Extract features for all images ------------------
X, y = [], []

for root, dirs, files in os.walk(dataset_path):
    img_files = [f for f in files if f.lower().endswith((".jpg",".png",".jpeg"))]
    for file in img_files:
        path = os.path.join(root, file)
        label = os.path.basename(root)

        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.resize(img, (128,128))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        features = (
            extract_glcm_features(gray) +
            extract_lbp_features(gray) +
            extract_hog_features(gray) +
            extract_color_histogram(img) +
            extract_edge_features(gray)
        )
        X.append(features)
        y.append(label)

X = np.array(X)
y = np.array(y)

if len(X)==0:
    raise ValueError("No images found. Check dataset path.")

# ------------------ Train-Test Split ------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ------------------ Train & Evaluate Models ------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='linear'),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

accuracies = {}
precisions = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    accuracies[name] = acc
    precisions[name] = prec
    print(f"\n{name} Accuracy: {acc:.4f}")
    print(f"{name} Precision (macro): {prec:.4f}")
    print(classification_report(y_test, y_pred))

# ------------------ Plot Comparison ------------------
plt.figure(figsize=(10,5))
width = 0.35
x = np.arange(len(models))

plt.bar(x - width/2, list(accuracies.values()), width, label='Accuracy', color='skyblue')
plt.bar(x + width/2, list(precisions.values()), width, label='Precision', color='orange')
plt.xticks(x, list(models.keys()))
plt.ylabel("Score")
plt.ylim(0,1)
plt.title("ML Model Comparison (Accuracy & Precision)")
plt.legend()
plt.show()

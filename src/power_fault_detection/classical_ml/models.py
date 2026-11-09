"""Factory definitions for classical machine learning classifiers."""

from typing import Dict, Any
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from power_fault_detection.config import RANDOM_SEED


def get_classical_classifiers(random_state: int = RANDOM_SEED) -> Dict[str, Any]:
    """
    Initializes standard supervised ML classifiers for tabular visual feature vectors.

    Returns:
        Dictionary mapping model names to initialized Scikit-Learn estimators.
    """
    return {
        "Support Vector Machine (SVM)": SVC(
            kernel="rbf",
            probability=True,
            random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=random_state
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=random_state
        ),
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(
            n_neighbors=5
        )
    }

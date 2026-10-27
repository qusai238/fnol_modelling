"""Baseline model training helpers."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def train_baseline_classifier(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    """Train a simple logistic regression baseline classifier."""
    model = LogisticRegression(max_iter=1000, n_jobs=None)
    model.fit(X, y)
    return model


def evaluate_auc(model: LogisticRegression, X: np.ndarray, y: np.ndarray) -> float:
    """Compute ROC AUC for binary classification."""
    probs = model.predict_proba(X)[:, 1]
    return float(roc_auc_score(y, probs))

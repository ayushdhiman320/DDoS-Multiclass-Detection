from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_curve


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro"))


def per_class_metrics(y_true, y_pred, labels):
    p, r, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        label: {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, label in enumerate(labels)
    }


def focal_like_sample_weights(y, gamma: float = 2.0):
    values, counts = np.unique(y, return_counts=True)
    freq = {v: c / len(y) for v, c in zip(values, counts)}
    weights = np.array([(1.0 - freq[v]) ** gamma for v in y], dtype=float)
    return weights / np.mean(weights)


def optimize_binary_threshold(y_true_binary: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true_binary, y_score)
    idx = int(np.argmax(tpr - fpr))
    return float(thresholds[idx])

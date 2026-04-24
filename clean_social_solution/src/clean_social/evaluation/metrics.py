from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

LABEL_ORDER = ["Negative", "Neutral", "Positive"]


def compute_classification_metrics(
    y_true: list[str],
    y_pred: list[str],
    score_matrix: np.ndarray | None = None,
) -> dict[str, Any]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABEL_ORDER).tolist(),
    }

    if score_matrix is None:
        metrics["roc_auc_ovr"] = None
        return metrics

    y_true_binarized = label_binarize(y_true, classes=LABEL_ORDER)
    try:
        metrics["roc_auc_ovr"] = float(
            roc_auc_score(y_true_binarized, score_matrix, multi_class="ovr", average="macro")
        )
    except Exception:
        metrics["roc_auc_ovr"] = None

    return metrics

from __future__ import annotations

import numpy as np

import sklearn.metrics


def wmae(y_true: np.ndarray, y_pred: np.ndarray, weight: np.ndarray | None = None) -> float:
    return sklearn.metrics.mean_absolute_error(y_true, y_pred, sample_weight=weight)





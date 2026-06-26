"""Simple split conformal prediction for regression."""

from __future__ import annotations

import math

import numpy as np


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")

    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        raise ValueError("Cannot calibrate conformal prediction with an empty score array.")

    quantile_level = math.ceil((scores.size + 1) * (1 - alpha)) / scores.size
    quantile_level = min(1.0, quantile_level)

    try:
        return float(np.quantile(scores, quantile_level, method="higher"))
    except TypeError:
        return float(np.quantile(scores, quantile_level, interpolation="higher"))


def calibrate_conformal_regression(y_true, y_pred, alpha: float = 0.1) -> tuple[float, np.ndarray]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")

    residuals = np.abs(y_true - y_pred)
    q_hat = _conformal_quantile(residuals, alpha)
    return q_hat, residuals


def make_prediction_intervals(y_pred, q_hat: float) -> tuple[np.ndarray, np.ndarray]:
    y_pred = np.asarray(y_pred, dtype=float)
    return y_pred - q_hat, y_pred + q_hat


def conformal_coverage(y_true, lower_bound, upper_bound) -> float:
    y_true = np.asarray(y_true, dtype=float)
    lower_bound = np.asarray(lower_bound, dtype=float)
    upper_bound = np.asarray(upper_bound, dtype=float)
    return float(np.mean((y_true >= lower_bound) & (y_true <= upper_bound)))

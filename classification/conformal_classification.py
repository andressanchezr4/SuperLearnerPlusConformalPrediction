"""Simple split conformal prediction sets for classification."""

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


def calibrate_conformal_classification(y_true, probabilities, alpha: float = 0.1) -> tuple[float, np.ndarray]:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be a 2D array.")
    if y_true.shape[0] != probabilities.shape[0]:
        raise ValueError("y_true and probabilities must have the same number of rows.")

    true_class_probabilities = probabilities[np.arange(y_true.shape[0]), y_true]
    scores = 1.0 - true_class_probabilities
    q_hat = _conformal_quantile(scores, alpha)
    return q_hat, scores


def make_prediction_sets(probabilities, q_hat: float) -> list[list[int]]:
    probabilities = np.asarray(probabilities, dtype=float)
    include_mask = (1.0 - probabilities) <= q_hat
    return [np.flatnonzero(row_mask).astype(int).tolist() for row_mask in include_mask]


def conformal_coverage(y_true, prediction_sets: list[list[int]]) -> float:
    y_true = np.asarray(y_true, dtype=int)
    covered = [int(label) in prediction_set for label, prediction_set in zip(y_true, prediction_sets)]
    return float(np.mean(covered))

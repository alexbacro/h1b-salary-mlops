"""Evaluation metrics shared by the baseline, AutoML training, and monitoring.

All functions take wages in dollars (not logs) so results read in business terms.
"""

from __future__ import annotations

import numpy as np


def mae(y_true, y_pred) -> float:
    """Mean absolute error in dollars."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def mape(y_true, y_pred) -> float:
    """Mean absolute percentage error, as a fraction (0.12 = 12%)."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred) / y_true))


def log_rmse(y_true, y_pred) -> float:
    """RMSE on log wages. This is the loss the model is trained on."""
    diff = np.log(np.asarray(y_true, dtype=float)) - np.log(np.asarray(y_pred, dtype=float))
    return float(np.sqrt(np.mean(diff**2)))


def regression_report(y_true, y_pred) -> dict[str, float]:
    return {"mae": mae(y_true, y_pred), "mape": mape(y_true, y_pred), "log_rmse": log_rmse(y_true, y_pred)}

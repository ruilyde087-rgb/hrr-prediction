"""
Evaluation metrics for regression tasks.
"""

from typing import Dict
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute common regression metrics.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        Dictionary with keys: r2, mae, rmse, mape.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    # Mean Absolute Percentage Error (avoid division by zero)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else float("inf")

    return {
        "r2": float(r2),
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
    }

"""Evaluation metrics used in the C-MAPSS / PHM08 literature."""
import numpy as np


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def nasa_score(y_true, y_pred) -> float:
    """Asymmetric PHM08 scoring function (lower is better).

    d = predicted - true. A late prediction (d > 0: we think the engine lives
    longer than it does) is penalised more than an early one, because it
    means maintenance happens *after* the failure.
    """
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)))


def interval_coverage(y_true, lower, upper) -> float:
    """Fraction of true values that fall inside the predicted interval."""
    y_true = np.asarray(y_true, float)
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def conformal_halfwidth(residuals, alpha: float = 0.1) -> float:
    """Split conformal prediction (Vovk et al.; Angelopoulos & Bates, 2021).

    From absolute errors |y - y_hat| on held-out calibration data, return q such
    that [y_hat - q, y_hat + q] contains the truth with probability >= 1 - alpha,
    assuming calibration and test points are exchangeable. No model assumption.
    """
    r = np.sort(np.abs(np.asarray(residuals, float)))
    n = len(r)
    k = min(n - 1, int(np.ceil((n + 1) * (1 - alpha))) - 1)
    return float(r[k])

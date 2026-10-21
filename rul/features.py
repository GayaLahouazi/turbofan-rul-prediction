"""Hand-crafted window features for the classical (non-deep) baseline."""
import numpy as np


def window_features(x: np.ndarray) -> np.ndarray:
    """x: (n, window, n_sensors) -> (n, 4 * n_sensors).

    For every sensor: last value, mean, standard deviation and linear slope
    over the window. The slope captures the degradation trend.
    """
    n, w, _ = x.shape
    t = np.arange(w, dtype=np.float32)
    t = (t - t.mean()) / ((t - t.mean()) ** 2).sum()  # least-squares slope weights
    last = x[:, -1, :]
    mean = x.mean(axis=1)
    std = x.std(axis=1)
    slope = np.einsum("t,nts->ns", t, x)
    return np.concatenate([last, mean, std, slope], axis=1)

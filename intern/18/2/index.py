import numpy as np


def mse(y: np.ndarray) -> float:
    """Compute the mean squared error of a vector."""
    return np.square(y-np.average(y)).mean()


def weighted_mse(y_left: np.ndarray, y_right: np.ndarray) -> float:
    """Compute the weighted mean squared error of two vectors."""
    y_left_n = y_left.shape[0]
    y_right_n = y_right.shape[0]
    return (mse(y_left) * y_left_n + mse(y_right) * y_right_n) / (y_left_n+y_right_n)

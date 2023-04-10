from __future__ import annotations

import numpy as np


def mse(y: np.ndarray) -> float:
    """Compute the mean squared error of a vector."""
    return np.square(y-np.average(y)).mean()


def weighted_mse(y_left: np.ndarray, y_right: np.ndarray) -> float:
    """Compute the weighted mean squared error of two vectors."""
    y_left_n = y_left.shape[0]
    y_right_n = y_right.shape[0]
    return (mse(y_left) * y_left_n + mse(y_right) * y_right_n) / (y_left_n+y_right_n)


def split(X: np.ndarray, y: np.ndarray, feature: int) -> float:
    """Find the best split for a node (one feature)"""
    values = X[:, feature]

    min_weighted = mse(values)
    best_threshold = values[0]

    for idx, value in enumerate(values):
        if y[idx + 1:].shape[0] == 0 or y[:idx + 1].shape[0] == 0:
            pass
        weighted = weighted_mse(y[:idx + 1], y[idx + 1:])
        if weighted < min_weighted:
            min_weighted = weighted
            best_threshold = value

    return best_threshold


def best_split(X: np.ndarray, y: np.ndarray) -> tuple[int, float]:
    """Find the best split for a node (one feature)"""
    n_features = X.shape[1]
    values = [split(X, y, features) for features in range(n_features)]
    best_feature = np.argmin(values)

    return best_feature, values[best_feature]


if __name__ == "__main__":
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [
                 16, 17, 18], [19, 20, 21], [22, 23, 24], [25, 26, 27], [28, 29, 30]])
    y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    t = best_split(X, y)
    print(t)

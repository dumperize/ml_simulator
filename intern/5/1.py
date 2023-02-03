import numpy as np


def turnover_error(y_true: np.array, y_pred: np.array) -> float:
    """Asymmetric error, """
    return np.mean(
        np.where(
            y_true - y_pred < 0, np.abs(y_true - y_pred), 2 * np.abs(y_true - y_pred)
        )
    )

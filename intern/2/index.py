"""Module providing numpy Function """
import numpy as np


def smape(y_true: np.array, y_pred: np.array) -> float:
    """Calc sMAPE"""
    abs_sum = np.abs(y_true) + np.abs(y_pred)
    return np.mean(2 * np.abs(y_true - y_pred) / np.where(abs_sum == 0, 1, abs_sum))

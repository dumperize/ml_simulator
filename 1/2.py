import numpy as np


def ltv_error(y_true: np.array, y_pred: np.array) -> float:
    estimate = []
    for t, p in zip(y_true, y_pred):
        diff = t-p
        res = (diff / t) ** 2
        if diff < 0:
            estimate.append(res * 2)
        else:
            estimate.append(res)
    return sum(estimate)

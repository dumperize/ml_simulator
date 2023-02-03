"""Module providing numpy Function """
import numpy as np


def smape(y_true: np.array, y_pred: np.array) -> float:
    """Calc sMAPE"""
    abs_sum = np.abs(y_true) + np.abs(y_pred)
    return np.mean(2 * np.abs(y_true - y_pred) / np.where(abs_sum == 0, 1, abs_sum))


# import numpy as np


# def smape(y_true: np.array, y_pred: np.array) -> float:
#     abs_sum = np.abs(y_true) + np.abs(y_pred)
#     return np.mean(2 * np.abs(y_true - y_pred) / np.where(abs_sum == 0, 1, abs_sum))

print("y_true: 2 \n")
for i in range(8):
    print(
          'y_pred: {point}: {smape:.2f}'.format(point=2+i, smape=smape(np.array([2]), np.array([2+i]))),
        " | ",
          'y_pred: {point}: {smape:.2f}'.format(point=2-i, smape=smape(np.array([2]), np.array([2-i]))))
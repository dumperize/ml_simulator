"""Solution for boosting uncertainty problem"""
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd
import math
from sklearn.ensemble import GradientBoostingRegressor


@dataclass
class PredictionDict:
    pred: np.ndarray = field(default_factory=lambda: np.array([]))
    uncertainty: np.ndarray = field(default_factory=lambda: np.array([]))
    pred_virt: np.ndarray = field(default_factory=lambda: np.array([]))
    lcb: np.ndarray = field(default_factory=lambda: np.array([]))
    ucb: np.ndarray = field(default_factory=lambda: np.array([]))


def virtual_ensemble_iterations(
    model: GradientBoostingRegressor, k: int = 20
) -> List[int]:
    """ virtual_ensemble_iterations """
    n = model.n_estimators
    first = math.floor(n / 2) - 1
    iterations = [first]

    while iterations[-1] + k < n:
        iterations.append(iterations[-1] + k)

    return iterations


def virtual_ensemble_predict(
    model: GradientBoostingRegressor, X: np.ndarray, k: int = 20
) -> np.ndarray:
    """ virtual_ensemble_predict """
    cols = virtual_ensemble_iterations(model, k)

    stage_preds = []

    for i, pred in enumerate(model.staged_predict(X)):
        if i in cols:
            stage_preds.append(pred)

    return np.array(stage_preds).T


def predict_with_uncertainty(
    model: GradientBoostingRegressor, X: np.ndarray, k: int = 20
) -> PredictionDict:
    """ predict_with_uncertainty """
    pred = virtual_ensemble_predict(model, X, k)
    uncertainty = np.var(pred, axis=1)
    pred_virt = np.average(pred, axis=1)
    lcb = pred_virt - 3 * np.sqrt(uncertainty)
    ucb = pred_virt + 3 * np.sqrt(uncertainty)

    prediction_dict = PredictionDict(
        pred=pred,
        uncertainty=uncertainty,
        pred_virt=pred_virt,
        ucb=ucb,
        lcb=lcb
    )
    return prediction_dict


if __name__ == "__main__":
    df_path = "../datasets/data_train_sql.csv"
    df = pd.read_csv(df_path, parse_dates=["monday"])

    y = df.pop("y")

    df.drop("product_name", axis=1, inplace=True)
    groups = df.pop("monday")

    X = df

    model = GradientBoostingRegressor(
        learning_rate=0.3737890991405422,
        random_state=427,
        n_estimators=201,
        subsample=0.6290569858604921,
        max_depth=1)
    model.fit(X, y)
    print(predict_with_uncertainty(model=model, X=X))

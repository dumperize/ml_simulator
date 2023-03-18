"""Baseline for Kaggle AB."""

from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from tqdm import tqdm


def prepare_dataset(DATA_PATH: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare dataset.
    Load data, split into X and y, one-hot encode categorical

    Parameters
    ----------
    DATA_PATH: str :
        path to the dataset

    Returns
    -------
    Tuple[np.ndarray, np.ndarray] :
        X and y
    """
    df = pd.read_csv(DATA_PATH)
    df = df.drop(["ID"], axis=1)
    y = df.pop("y").values

    # select only numeric columns
    X_num = df.select_dtypes(include="number")

    # select only categorical columns and one-hot encode them
    X_cat = df.select_dtypes(exclude="number")
    X_cat = pd.get_dummies(X_cat)

    # combine numeric and categorical
    X = pd.concat([X_num, X_cat], axis=1)
    X = X.fillna(0).values

    return X, y


def cross_val_score(
    model: Callable,
    X: np.ndarray,
    y: np.ndarray,
    cv: int,
    params_list: List[Dict],
    scoring: Callable,
    random_state: int = 0,
    show_progress: bool = False,
) -> np.ndarray:
    """
    Cross-validation score.

    Parameters
    ----------
    model: Callable :
        model to train (e.g. RandomForestRegressor)
    X: np.ndarray :

    y: np.ndarray :

    cv :
        number of folds fo cross-validation

    params_list: List[Dict] :
        list of model parameters

    scoring: Callable :
        scoring function (e.g. r2_score)

    random_state: int :
        (Default value = 0)
        random state for cross-validation

    show_progress: bool :
        (Default value = False)

    Returns
    -------
    np.ndarray :
        cross-validation scores [n_models x n_folds]

    """
    # YOR CODE HERE
    result_by_param = []

    kf = KFold(n_splits=cv, random_state=random_state, shuffle=True)

    y = np.log1p(y)
    for param in params_list:
        metrics = []
        model.set_params(max_depth=param['max_depth'])

        for (train_index, test_index) in tqdm(kf.split(X), disable=not show_progress):
            X_train = X[train_index]
            X_test = X[test_index]
            y_train = y[train_index]
            y_test = y[test_index]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = scoring(np.expm1(y_test), np.expm1(y_pred))

            metrics.append(score)
        result_by_param.append(metrics)
    return np.matrix(result_by_param)


def compare_models(
    cv: int,
    model: Callable,
    params_list: List[Dict],
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 0,
    show_progress: bool = False,
) -> List[Dict]:
    """Compare models with Cross val.

    Parameters
    ----------
    cv: int :
        number of folds fo cross-validation

    model: Callable :
        model to train (e.g. RandomForestRegressor)

    params_list: List[Dict] :
        list of model parameters

    X: np.ndarray :

    y: np.ndarray :

    random_state: int :
        (Default value = 0)
        random state for cross-validation

    show_progress: bool :
        (Default value = False)

    Returns
    -------
    List[Dict] :
        list of dicts with model comparison results
        {
            model_index,
            avg_score,
            effect_sign
        }
    """
    # YOR CODE HERE
    result = []

    kfold_res = cross_val_score(
            model,
            X=X,
            y=y,
            cv=cv,
            params_list=params_list,
            scoring=r2_score,
            random_state=random_state,
            show_progress=show_progress,
            )

    baseline_mean = np.mean(kfold_res[0])
    for idx, res in enumerate(kfold_res[1:]):
        res_mean = np.mean(res)
        sign = 0 if baseline_mean == res_mean else ( -1 if baseline_mean > res_mean else 1)

        result.append({
            "model_index": idx + 1,
            "avg_score": res_mean,
            "effect_sign": sign
        })
    result.sort(key=lambda x: x['avg_score'], reverse=True)
    return result


def run() -> None:
    """Run."""

    data_path = "train.csv.zip"
    random_state = 42
    cv = 5
    params_list = [
        {'max_depth': 10}, {'max_depth': 2}, {'max_depth': 3}, {'max_depth': 12}, {'max_depth': 15}
    ]

    X, y = prepare_dataset(data_path)
    model = RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=random_state)

    result = compare_models(
        cv=cv,
        model=model,
        params_list=params_list,
        X=X,
        y=y,
        random_state=random_state,
        show_progress=False,
    )
    df = pd.DataFrame(result)
    df = df.set_index('model_index')
    print("KFold")
    print(df)


if __name__ == "__main__":
    run()

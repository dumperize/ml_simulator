"""Baseline for Kaggle AB."""

from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, RepeatedKFold
from tqdm import tqdm
from scipy import stats


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
    cv: Union[int, Tuple[int, int]],
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

    cv Union[int, Tuple[int, int]]:
        (Default value = 5)
        number of folds or (n_folds, n_repeats)
        if int, then KFold is used
        if tuple, then RepeatedKFold is used

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

    if type(cv) is tuple:
        kf = RepeatedKFold(
            n_splits=cv[0], n_repeats=cv[1], random_state=random_state)
    else:
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
    return np.array(result_by_param)


def compare_models(
    cv: Union[int, Tuple[int, int]],
    model: Callable,
    params_list: List[Dict],
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 0,
    alpha: float = 0.05,
    show_progress: bool = False,
) -> List[Dict]:
    """Compare models with Cross val.

    Parameters
    ----------
    cv: Union[int, Tuple[int, int]] :
        (Default value = 5)
        number of folds or (n_folds, n_repeats)
        if int, then KFold is used
        if tuple, then RepeatedKFold is used

    model: Callable :
        model to train (e.g. RandomForestRegressor)

    params_list: List[Dict] :
        list of model parameters

    X: np.ndarray :

    y: np.ndarray :

    random_state: int :
        (Default value = 0)
        random state for cross-validation

    alpha: float :
        (Default value = 0.05)
        significance level for t-test

    show_progress: bool :
        (Default value = False)

    Returns
    -------
    List[Dict] :
        list of dicts with model comparison results
        {
            model_index,
            avg_score,
            p_value,
            effect_sign
        }
    """
    # YOR CODE HERE
    result = []

    cross_val = cross_val_score(
        model,
        X=X,
        y=y,
        cv=cv,
        params_list=params_list,
        scoring=r2_score,
        random_state=random_state,
        show_progress=show_progress,
    )

    baseline = cross_val[0]
    baseline_mean = np.mean(baseline)

    for idx, res in enumerate(cross_val[1:]):
        ttest = stats.ttest_rel(res, baseline)

        res_mean = np.mean(res)

        effect_sign = 0
        if ttest.pvalue < alpha and res_mean > baseline_mean:
            effect_sign = 1
        if ttest.pvalue < alpha and res_mean < baseline_mean:
            effect_sign = -1

        result.append({
            "model_index": idx + 1,
            "avg_score": res_mean,
            "p_value": ttest.pvalue,
            "effect_sign": effect_sign
        })
    result.sort(key=lambda x: x['avg_score'], reverse=True)
    return result


def run() -> None:
    """Run."""

    data_path = "train.csv.zip"
    random_state = 42
    cv = 5
    repeat = 2
    params_list = [
        {"max_depth": 10},  # baseline
        {"max_depth": 2},
        {"max_depth": 3},
        {"max_depth": 4},
        {"max_depth": 5},
        {"max_depth": 9},
        {"max_depth": 11},
        {"max_depth": 12},
        {"max_depth": 15},
    ]

    X, y = prepare_dataset(data_path)
    model = RandomForestRegressor(
        n_estimators=50, n_jobs=-1, random_state=random_state)

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

    result = compare_models(
        cv=(cv, repeat),
        model=model,
        params_list=params_list,
        X=X,
        y=y,
        random_state=random_state,
        show_progress=False,
    )
    df = pd.DataFrame(result)
    df = df.set_index('model_index')
    print("RepeatedKFold")
    print(df)


if __name__ == "__main__":
    run()

"""Solution for Kaggle AB2."""
from typing import Tuple

import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import r2_score
from tqdm import tqdm


class SequentialForwardSelector:
    """
    Sequential forward selection.
    Algorithm selects features one by one, each time adding the feature that
    improves the model the most.

    Parameters
    ----------
    model: estimator :
        ML model, e.g. LinearRegression
    cv: cross-validation generator :
        cross-validation generator, e.g. KFold, RepeatedKFold
    max_features: int :
        maximum number of features to select
    verbose: int :
        (Default value = 0)
        verbosity level

    Attributes
    ----------
    n_features_: int :
        number of features in the dataset
    selected_features_: List[int] :
        list of selected features, ordered by index
    n_selected_features_: int :
        number of selected features
    """

    def __init__(
        self,
        model,
        cv,
        max_features: int = 10,
        verbose: int = 0,
    ) -> None:
        """Initialize SequentialForwardSelector."""
        self.model = model
        self.cv = cv
        self.max_features = max_features
        self.verbose = verbose

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the model.

        Parameters
        ----------
        X: np.ndarray :
            features
        y: np.ndarray :
            target
        """
        self.features_ = range(X.shape[1])

        includes = set()
        excludes = set(x for x in self.features_)
        while len(includes) < self.max_features:
            scores = {}
            for k in tqdm(excludes, disable=self.verbose == 0):
                evaluate_res = []
                for i, (train_index, test_index) in enumerate(self.cv.split(X)):
                    rows = includes.copy()
                    rows.add(k)
                    X_train = X[train_index, :][:, list(rows)]
                    y_train = y[train_index]
                    self.model.fit(X_train, y_train)

                    X_test = X[test_index, :][:, list(rows)]
                    y_pred = self.model.predict(X_test)
                    y_true = y[test_index]
                    res = r2_score(y_true, y_pred)
                    evaluate_res.append(res)
                scores[k] = np.average(evaluate_res)
            max_el = max(scores, key=scores.get)
            includes.add(max_el)
            excludes.remove(max_el)

        self.selected_features_ = sorted(list(includes))

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Reduce the dataset to selected features.

        Parameters
        ----------
        X: np.ndarray :
            features

        Returns
        -------
        X: np.ndarray :
            reduced dataset
        """
        return X[:, self.selected_features_]

    @property
    def n_features_(self):
        """Number of selected features."""
        return len(self.features_)

    @property
    def n_selected_features_(self):
        """Number of selected features."""
        return len(self.selected_features_)


def generate_dataset(
    n_samples: int = 10_000,
    n_features: int = 50,
    n_informative: int = 10,
    random_state: int = 42,
) -> Tuple:
    """
    Generate dataset.

    Parameters
    ----------
    n_samples: int :
        (Default value = 10_000)
        number of samples
    n_features: int :
        (Default value = 50)
        number of features
    n_informative: int :
        (Default value = 10)
        number of informative features, other features are noise
    random_state: int :
        (Default value = 42)
        random state for reproducibility

    Returns
    -------
    X: np.ndarray :
        features
    y: np.ndarray :
        target

    """
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=100,
        random_state=random_state,
        n_informative=n_informative,
        bias=100,
        shuffle=True,
    )
    return X, y


def run() -> None:
    """Run."""
    random_state = 42
    n_samples = 10_000
    n_features = 50
    n_informative = 5
    max_features = 10
    n_splits = 3
    n_repeats = 10

    # generate data
    X, y = generate_dataset(n_samples, n_features, n_informative, random_state)

    # define model and cross-validation
    model = LinearRegression()
    cv = RepeatedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )

    # baseline
    scores = cross_val_score(model, X, y, scoring="r2", cv=cv, n_jobs=-1)
    print(f"Baseline features count: {X.shape[1]}")
    print(f"Baseline R2 score: {scores.mean():.4f}")

    selector = SequentialForwardSelector(model, cv, max_features, verbose=1)
    selector.fit(X, y)
    X_transformed = selector.transform(X)
    scores = cross_val_score(model, X_transformed, y,
                             scoring="r2", cv=cv, n_jobs=-1)

    print(f"Features: {selector.selected_features_}")
    print(f"Features count: {selector.n_selected_features_}")
    print(f"Mean R2 score: {scores.mean():.4f}")


if __name__ == "__main__":
    run()

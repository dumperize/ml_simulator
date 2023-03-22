"""Solution for Kaggle AB2."""
from typing import Tuple

import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel


class LassoSelector:
    """
    Lasso selector.
    Select features using Linear Regression with Lasso regularization.

    Parameters
    ----------
    cv: cross-validation generator :
        cross-validation
    alphas: List[float] :
        list of alphas for Lasso
    random_state: int :
        random state for reproducibility

    Attributes
    ----------
    selected_features: List[int] :
        list of selected features
    """

    def __init__(self, cv, alphas, random_state=42):
        """Initialize LassoSelector."""
        self.cv = cv
        self.alphas = alphas
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit model.

        Parameters
        ----------
        X: np.ndarray :
            features
        y: np.ndarray :
            target

        Returns
        -------
        None

        """
        self.features_ = list(range(X.shape[1]))

        reg = LassoCV(alphas=self.alphas, cv=self.cv,
                      random_state=self.random_state)
        select_reg = SelectFromModel(reg).fit(X, y)
        params = select_reg.get_support()

        self.selected_features_ = [
            y for x, y in zip(params, self.features_) if x]

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Reduce features to selected features.

        Parameters
        ----------
        X: np.ndarray :
            features

        Returns
        -------
        X: np.ndarray :
            reduced features

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
    Generate datasets.

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
    n_splits = 3
    n_repeats = 10
    alphas = [2, 10]

    # generate data
    X, y = generate_dataset(n_samples, n_features, n_informative, random_state)

    # define model and cross-validation
    cv = RepeatedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )
    print(f"Baseline features count: {X.shape[1]}")

    # lasso selector
    selector = LassoSelector(cv, alphas, random_state)
    selector.fit(X, y)

    # show scores
    print(f"Features: {selector.selected_features_}")
    print(f"Features count: {selector.n_selected_features_}")


if __name__ == "__main__":
    run()

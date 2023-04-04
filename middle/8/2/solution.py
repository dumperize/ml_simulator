from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection._split import _BaseKFold
from sklearn.model_selection import TimeSeriesSplit


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """ mape """
    return (1 / y_true.shape[0]) * np.sum(np.abs(y_true-y_pred) / np.abs(y_true))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """ smape """
    z = np.abs(y_true) + np.abs(y_pred)
    z = np.where(z == 0, 0.00000001, z)
    return 1/y_true.shape[0] * np.sum(2 * np.abs(y_pred-y_true) / z)


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """ wape """
    return np.sum(np.abs(y_true-y_pred)) / np.sum(np.abs(y_true))


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """ bias """
    return np.sum(y_pred-y_true) / np.sum(np.abs(y_true))


class GroupTimeSeriesSplit(_BaseKFold):
    """Time Series cross-validator variant with non-overlapping groups.
    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals according to a
    third-party provided group.
    In each split, test indices must be higher than before, and thus shuffling
    in cross validator is inappropriate.
    This cross-validation object is a variation of :class:`KFold`.
    In the kth split, it returns first k folds as train set and the
    (k+1)th fold as test set.
    The same group will not appear in two different folds (the number of
    distinct groups has to be at least equal to the number of folds).
    Note that unlike standard cross-validation methods, successive
    training sets are supersets of those that come before them.
    Parameters
    ----------
    n_splits : int, default=5
        Number of splits. Must be at least 2.
    max_train_size : int, default=None
        Maximum groups for a single training set.
    test_size : int, default=None
        Number of groups in test
    gap : int, default=0
        Number of groups between train and test sets
    Examples
    --------
    >>> import numpy as np
    >>> groups = np.array(['a', 'a', 'a', 'a', 'a', 'a',\
                    'b', 'b', 'b', 'b', 'b',\
                    'c', 'c', 'c', 'c',\
                    'd', 'd', 'd',
                    'e', 'e', 'e'])
    >>> splitter = GroupTimeSeriesSplit(n_splits=3, max_train_size=2, gap=1)
    >>> for i, (train_idx, test_idx) in enumerate(
    ...     splitter.split(groups, groups=groups)):
    ...     print(f"Split: {i + 1}")
    ...     print(f"Train idx: {train_idx}, test idx: {test_idx}")
    ...     print(f"Train groups: {groups[train_idx]},
                    test groups: {groups[test_idx]}\n")
    Split: 1
    Train idx: [0 1 2 3 4 5], test idx: [11 12 13 14]
    Train groups: ['a' 'a' 'a' 'a' 'a' 'a'], test groups: ['c' 'c' 'c' 'c']

    Split: 2
    Train idx: [ 0  1  2  3  4  5  6  7  8  9 10], test idx: [15 16 17]
    Train groups: ['a' 'a' 'a' 'a' 'a' 'a' 'b' 'b' 'b' 'b' 'b'],
    test groups: ['d' 'd' 'd']

    Split: 3
    Train idx: [ 6  7  8  9 10 11 12 13 14], test idx: [18 19 20]
    Train groups: ['b' 'b' 'b' 'b' 'b' 'c' 'c' 'c' 'c'],
    test groups: ['e' 'e' 'e']
    """

    def __init__(self, n_splits=5, max_train_size=None, test_size=None, gap=0):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.max_train_size = max_train_size
        self.test_size = test_size
        self.gap = gap

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        groups = pd.Series(groups)
        groups_unique = pd.unique(groups)
        n_samples = groups_unique.shape[0]
        n_splits = self.n_splits
        n_folds = n_splits + 1
        gap = self.gap
        test_size = (
            self.test_size if self.test_size is not None else n_samples // n_folds
        )

        # Make sure we have enough samples for the given split parameters
        if n_folds > n_samples:
            raise ValueError(
                f"Cannot have number of folds={n_folds} greater"
                f" than the number of samples={n_samples}."
            )
        if n_samples - gap - (test_size * n_splits) <= 0:
            raise ValueError(
                f"Too many splits={n_splits} for number of samples"
                f"={n_samples} with test_size={test_size} and gap={gap}."
            )

        indices = np.arange(n_samples)
        test_starts = range(n_samples - n_splits *
                            test_size, n_samples, test_size)

        for test_start in test_starts:
            train_end = test_start - gap
            if self.max_train_size and self.max_train_size < train_end:
                train_idx = groups_unique[indices[train_end -
                                                  self.max_train_size: train_end]]
                test_idx = groups_unique[indices[test_start: test_start + test_size]]
                yield (
                    groups.index[groups.isin(train_idx)],
                    groups.index[groups.isin(test_idx)],
                )
            else:
                train_idx = groups_unique[indices[:train_end]]
                test_idx = groups_unique[indices[test_start: test_start + test_size]]
                yield (
                    groups.index[groups.isin(train_idx)],
                    groups.index[groups.isin(test_idx)],
                )


def best_model() -> Any:
    model = GradientBoostingRegressor(
        learning_rate=0.3737890991405422,
        random_state=427,
        n_estimators=201,
        subsample=0.6290569858604921,
        max_depth=1)
    return model

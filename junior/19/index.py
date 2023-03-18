from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


@dataclass
class BaseSelector(ABC):
    """ BaseSelector abstract class """
    threshold: float = 0.5

    @abstractmethod
    def fit(self, X, y):
        pass

    def transform(self, X):
        return X[self.high_corr_features]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    @property
    def n_features_(self):
        return len(self.high_corr_features)

    @property
    def original_features_(self):
        return self.original_features

    @property
    def selected_features_(self):
        return self.high_corr_features


@dataclass
class PearsonSelector(BaseSelector):
    """ PearsonSelector """
    def fit(self, X, y) -> PearsonSelector:
        """ Correlation between features and target """
        corr = pd.concat([X, y], axis=1).corr(method="pearson")
        corr_target = corr.iloc[:-1, -1]

        self.original_features = X.columns.tolist()
        self.high_corr_features = corr_target[
            abs(corr_target) >= self.threshold
        ].index.tolist()

        return self


@dataclass
class SpearmanSelector(BaseSelector):
    """ SpearmanSelector """
    def fit(self, X, y) -> SpearmanSelector:
        """ Correlation between features and target """
        corr = pd.concat([X, y], axis=1).corr(method="spearman")
        corr_target = corr.iloc[:-1, -1]
        self.original_features = X.columns.tolist()
        self.high_corr_features = corr_target[
            abs(corr_target) >= self.threshold
        ].index.tolist()

        return self


@dataclass
class VarianceSelector(BaseSelector):
    """ VarianceSelector """
    def fit(self, X, y=None) -> VarianceSelector:
        """ Correlation between features and target """
        variances = np.var(X, axis=0)
        self.original_features = X.columns.tolist()
        self.high_corr_features = X.columns[variances >
                                            self.threshold].tolist()
        return self

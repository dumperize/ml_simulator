from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import json


class NpEncoder(json.JSONEncoder):
    """converter type class"""

    def default(self, obj):
        """default method"""
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj


@dataclass
class Node:
    """Decision tree node."""
    feature: int = field(default=None)
    threshold: float = field(default=None)
    n_samples: int = field(default=None)
    value: int = field(default=None)
    mse: float = field(default=None)

    left: Node = None
    right: Node = None


@dataclass
class DecisionTreeRegressor:
    """Decision tree regressor."""
    max_depth: int
    min_samples_split: int = 2

    def fit(self, X: np.ndarray, y: np.ndarray) -> DecisionTreeRegressor:
        """Build a decision tree regressor from the training set (X, y)."""
        self.n_features_ = X.shape[1]
        self.tree_ = self._split_node(X, y)
        return self

    def _mse(self, y: np.ndarray) -> float:
        """Compute the mse criterion for a given set of target values."""
        return np.square(y-np.average(y)).mean()

    def _weighted_mse(self, y_left: np.ndarray, y_right: np.ndarray) -> float:
        """Compute the weithed mse criterion for a two given sets of target values"""
        y_left_n = y_left.shape[0]
        y_right_n = y_right.shape[0]
        return (self._mse(y_left) * y_left_n + self._mse(y_right) * y_right_n) / (y_left_n+y_right_n)

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> tuple[int, float]:
        """Find the best split for a node."""
        n_features = X.shape[1]
        values = []

        min_weighted = None
        best_threshold = None
        best_feature = None

        for feature in range(n_features):
            values_feature = X[:, feature]
            for idx, value in enumerate(values_feature):
                cond = values_feature <= value
                left = y[cond]
                right = y[~cond]
                if left.shape[0] < self.min_samples_split or right.shape[0] < self.min_samples_split:
                    continue
                weighted = self._weighted_mse(left, right)

                if min_weighted is None or weighted < min_weighted:
                    min_weighted = weighted
                    best_threshold = value
                    best_feature = feature

        return best_feature, best_threshold

    def _split_node(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """Split a node and return the resulting left and right child nodes."""
        if depth >= self.max_depth:
            return Node(
                n_samples=X.shape[0],
                value=round(np.average(y)),
                mse=self._mse(y),
            )

        best_idx, best_threshold = self._best_split(X, y)

        if best_idx is None:
            return Node(
                n_samples=X.shape[0],
                value=round(np.average(y)),
                mse=self._mse(y),
            )

        cond = X[:, best_idx] <= best_threshold

        return Node(
            feature=best_idx,
            threshold=best_threshold,
            n_samples=X.shape[0],
            value=round(np.average(y)),
            mse=self._mse(y),
            left=self._split_node(X[cond], y[cond], depth=depth+1),
            right=self._split_node(X[~cond], y[~cond], depth=depth+1)
        )

    def as_json(self) -> str:
        """Return the decision tree as a JSON string."""
        return json.dumps(self._as_json(self.tree_),  cls=NpEncoder)

    def _as_json(self, node: Node) -> str:
        """Return the decision tree as a JSON string. Execute recursively."""
        params = {}
        params_node = {}

        if node.feature and node.threshold:
            params["feature"] = node.feature
            params["threshold"] = node.threshold
        else:
            params["value"] = node.value

        if node.left:
            params_node["left"] = self._as_json(node.left)
        if node.right:
            params_node["right"] = self._as_json(node.right)
        return {
            **params,
            "n_samples": node.n_samples,
            "mse": round(node.mse, 2),
            **params_node,
        }


if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv('../dataset.csv')
    X = df.drop("delay_days", axis=1).to_numpy()
    y = df["delay_days"].to_numpy()
    model = DecisionTreeRegressor(max_depth=3, min_samples_split=2)
    model.fit(X, y)
    print(model.as_json())

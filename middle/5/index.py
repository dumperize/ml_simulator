from __future__ import annotations
from dataclasses import dataclass, field
import pandas as pd
import numpy as np


@dataclass
class Node:
    """Uplift tree node."""
    n_items: int = field(default=None)
    ATE: float = field(default=None)
    split_feat: int = field(default=None)
    split_threshold: float = field(default=None)

    left: Node = None
    right: Node = None


@dataclass
class UpliftTreeRegressor:
    """

    Parameters
    ----------
    max_depth : np.int :
        maximum depth of tree    
    min_samples_leaf : int :
        minimum count of samples in leaf    
    min_samples_leaf_treated : int :
        minimum count of treated samples in leaf
    min_samples_leaf_control : int :    
        minimum count of control samples in leaf
    Returns
    -------

    """

    max_depth: int = 3,
    min_samples_leaf: int = 1000,
    min_samples_leaf_treated: int = 300,
    min_samples_leaf_control: int = 300,
    tree: Node = None

    def threshold_options(self, column_values: np.ndarray):
        """threshold_options."""
        unique_values = np.unique(column_values)

        if len(unique_values) > 10:
            percentiles = np.percentile(
                column_values, [3, 5, 10, 20, 30, 50, 70, 80, 90, 95, 97]
            )
        else:
            percentiles = np.percentile(unique_values, [10, 50, 90])

        return np.unique(percentiles)

    def ate(self, treatment: np.ndarray, y: np.ndarray):
        """ate."""
        sumT = np.sum(treatment)
        sumIT = np.sum(1 - treatment)
        # print(sumT, sumIT)
        TY = np.sum(treatment * y)
        ITY = np.sum((1-treatment) * y)
        # print(TY, ITY)
        return TY / sumT - ITY / sumIT

    def find_best_feat_split(self, X: np.ndarray, treatment: np.ndarray, y: np.ndarray):
        """find_best_feat_split."""
        n_feat = X.shape[1]

        max_delta_delta_p = None
        best_feat = None
        best_treshold = None

        for feat in range(n_feat):
            column = X[:, feat]
            treshhold_options = self.threshold_options(column)
            for threshold in treshhold_options:
                cond = column <= threshold
                treatment_left = treatment[cond]
                treatment_right = treatment[~cond]
                y_left = y[cond]
                y_right = y[~cond]

                size_exp_group_left = treatment_left[treatment_left == 1].shape[0]
                size_cntr_group_left = treatment_left[treatment_left == 0].shape[0]
                size_exp_group_right = treatment_right[treatment_right == 1].shape[0]
                size_cntr_group_right = treatment_right[treatment_right == 0].shape[0]

                if y_left.shape[0] <= self.min_samples_leaf:
                    continue
                if y_right.shape[0] <= self.min_samples_leaf:
                    continue
                if size_exp_group_left <= self.min_samples_leaf_treated:
                    continue
                if size_exp_group_right <= self.min_samples_leaf_treated:
                    continue
                if size_cntr_group_left <= self.min_samples_leaf_control:
                    continue
                if size_cntr_group_right <= self.min_samples_leaf_control:
                    continue

                ate_left = self.ate(treatment_left, y_left)
                ate_right = self.ate(treatment_right, y_right)
                delta_delta_p = np.absolute(ate_left - ate_right)

                if (max_delta_delta_p is None or max_delta_delta_p < delta_delta_p):
                    max_delta_delta_p = delta_delta_p
                    best_feat = feat
                    best_treshold = threshold
        return max_delta_delta_p, best_feat, best_treshold

    def build_node(self, X: np.ndarray, treatment: np.ndarray, y: np.ndarray, depth=0):
        """build_node."""
        if depth > self.max_depth:
            return Node(
                n_items=X.shape[0],
                ATE=self.ate(treatment, y),
            )

        delta_delta_p, feat, treshold = self.find_best_feat_split(
            X, treatment, y)
        if delta_delta_p is None:
            return Node(
                n_items=X.shape[0],
                ATE=self.ate(treatment, y),
            )

        cond = X[:, feat] <= treshold
        treatment_left = treatment[cond]
        treatment_right = treatment[~cond]
        y_left = y[cond]
        y_right = y[~cond]
        X_left = X[cond]
        X_right = X[~cond]
        return Node(
            n_items=X.shape[0],
            ATE=self.ate(treatment, y),
            split_feat=feat,
            split_threshold=treshold,
            left=self.build_node(X_left, treatment_left, y_left, depth+1),
            right=self.build_node(X_right, treatment_right, y_right, depth + 1)
        )

    def fit(self, X: np.ndarray, treatment: np.ndarray, y: np.ndarray) -> UpliftTreeRegressor:
        """Fit model."""

        self.tree = self.build_node(X, treatment, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicts for X."""
        predictions = []
        for x in X:
            node = self.tree
            while node.left:
                if x[node.split_feat] <= node.split_threshold:
                    node = node.left
                else:
                    node = node.right
            predictions.append(node.ATE)

        return np.array(predictions)


if __name__ == "__main__":
    df = pd.read_csv("data.csv")
    y = df['target'].to_numpy()
    treatment = df['treatment'].to_numpy()
    X = df.iloc[:, 1:-2].to_numpy()
    model = UpliftTreeRegressor(
        max_depth=3,
        min_samples_leaf=6000,
        min_samples_leaf_treated=2500,
        min_samples_leaf_control=2500)

    print(model.fit(X, treatment, y))
    print(model.tree)

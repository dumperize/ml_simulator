from typing import Tuple

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample

def roc_auc_ci(
    classifier: ClassifierMixin,
    X: np.ndarray,
    y: np.ndarray,
    conf: float = 0.95,
    n_bootstraps: int = 10_000,
) -> Tuple[float, float]:
    """Returns confidence bounds of the ROC-AUC"""
    n = len(y) // 2

    def roc_auc():
        bootstrap_data = resample(np.c_[X,y], n_samples=n)
        prediction = classifier.predict(bootstrap_data[:,:-1])
        return roc_auc_score(bootstrap_data[:,-1], prediction)

    stats = [roc_auc() for i in range(n_bootstraps)]

    p = ((1.0-conf)/2.0) * 100
    lower = max(0.0, np.percentile(stats, p))
    p = (conf+((1.0-conf)/2.0)) * 100
    upper = min(1.0, np.percentile(stats, p))

    return lower,upper


# from sklearn.dummy import DummyClassifier

# X = np.array([-1, 1, 1, 1])
# y = np.array([0, 1, 1, 1])
# dummy_clf = DummyClassifier(strategy="most_frequent")
# dummy_clf.fit(X,y)
# print(roc_auc_ci(dummy_clf, X, y))


# from sklearn.datasets import load_breast_cancer
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import roc_auc_score
# X, y = load_breast_cancer(return_X_y=True)
# clf = LogisticRegression(solver="liblinear", random_state=0).fit(X, y)
# clf.fit(X,y)

# print(roc_auc_ci(clf, X, y))

# # n = 1000
# # B = 1000
# # values = np.random.normal(90, 20, n)
# # quantile = np.quantile(values, 0.9)
# # bootstrap_quantiles = np.quantile(np.random.choice(values, (B, n), True), 0.9, axis=1)

# print(quantile)
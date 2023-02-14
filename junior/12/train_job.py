import os
from typing import Any
from typing import Tuple

import fire
import mlflow
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay

IDENTIFIER = f'antifraud-{os.environ.get("KCHECKER_USER_USERNAME", "default")}'
TRACKING_URI = os.environ.get("TRACKING_URI")

mlflow.set_tracking_uri(TRACKING_URI)

def recall_at_precision(
    true_labels: np.ndarray,
    pred_scores: np.ndarray,
    min_precision: float = 0.95,
) -> float:
    """Compute recall at precision

    Args:
        true_labels (np.ndarray): True labels
        pred_scores (np.ndarray): Target scores
        min_precision (float, optional): Min precision for recall. Defaults to 0.95.

    Returns:
        float: Metric value
    """

    precision, recall, _ = precision_recall_curve(true_labels, pred_scores)
    metric = next(filter(lambda x: x[0] > min_precision, zip(precision, recall)))[1]
    return metric


def recall_at_specificity(
    true_labels: np.ndarray,
    pred_scores: np.ndarray,
    min_specificity: float = 0.95,
) -> float:
    """Compute recall at specificity

    Args:
        true_labels (np.ndarray): True labels
        pred_scores (np.ndarray): Target scores
        min_specificity (float, optional): Min specificity for recall. Defaults to 0.95.

    Returns:
        float: Metric value
    """

    fpr, tpr, _ = roc_curve(true_labels, pred_scores)
    metric = next(filter(lambda x: 1 - x[0] < min_specificity, zip(fpr, tpr)))[1]
    return metric


def curves(true_labels: np.ndarray, pred_scores: np.ndarray) -> Tuple[np.ndarray]:
    """Return ROC and FPR curves

    Args:
        true_labels (np.ndarray): True labels
        pred_scores (np.ndarray): Target scores

    Returns:
        Tuple[np.ndarray]: ROC and FPR curves
    """

    def fig2numpy(fig: Any) -> np.ndarray:
        fig.canvas.draw()
        img = fig.canvas.buffer_rgba()
        img = np.asarray(img)
        return img

    pr_curve = PrecisionRecallDisplay.from_predictions(true_labels, pred_scores)
    pr_curve = fig2numpy(pr_curve.figure_)

    roc_curve = RocCurveDisplay.from_predictions(true_labels, pred_scores)
    roc_curve = fig2numpy(roc_curve.figure_)

    return pr_curve, roc_curve


def job(
    train_path: str = "",
    test_path: str = "",
    target: str = "target",
):
    """Model training job

    Args:
        train_path (str): Train dataset path
        test_path (str): Test dataset path
        target (str): Target column name
    """
    mlflow.set_experiment(IDENTIFIER)

    mlflow.start_run()
    mlflow.set_tag("task_type", "anti-fraud")
    mlflow.set_tag("framework", "sklearn")

    train_dataset = pd.read_csv(train_path)
    test_dataset = pd.read_csv(test_path)

    model = LogisticRegression()
    X = train_dataset.drop(columns=target, axis=1)
    model.fit(X, train_dataset[target])

    test_targets = test_dataset[target]
    pred_scores = model.predict(test_dataset.drop(columns=target, axis=1))

    mlflow.log_params({
        "features": "['" + "', '".join(X.columns) + "']",
        "target": target,
        "model_type": model.__class__.__name__,
    })

    recall_precision = recall_at_precision(true_labels=test_targets, pred_scores=pred_scores)
    recall_specificity = recall_at_specificity(true_labels=test_targets, pred_scores=pred_scores)
    roc_auc = roc_auc_score(test_targets, pred_scores)
    pr_curve, roc_curve = curves(test_targets, pred_scores)

    pr_file_path = 'pr.png'
    plt.imsave(pr_file_path, pr_curve)

    roc_file_path = 'roc.png'
    plt.imsave(roc_file_path, pr_curve)

    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("recall_precision_95", recall_precision)
    mlflow.log_metric("recall_specificity_95", recall_specificity)
    mlflow.log_artifact(train_path, 'data')
    mlflow.log_artifact(test_path, 'data')
    mlflow.log_artifact(pr_file_path, 'metrics')
    mlflow.log_artifact(roc_file_path, 'metrics')

    mlflow.sklearn.log_model(model, IDENTIFIER, registered_model_name=IDENTIFIER)
    mlflow.end_run()


if __name__ == "__main__":
    fire.Fire(job)

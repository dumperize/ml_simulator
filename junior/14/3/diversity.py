"""Template for user."""
from typing import Tuple

import numpy as np
from sklearn.neighbors import KernelDensity


def kde_uniqueness(embeddings: np.ndarray) -> np.ndarray:
    """Estimate uniqueness of each item in item embeddings group. Based on KDE.

    Parameters
    ----------
    embeddings: np.ndarray :
        embeddings group 

    Returns
    -------
    np.ndarray
        uniqueness estimates

    """
    kde = KernelDensity(kernel='gaussian').fit(embeddings)
    res = 1 / np.exp(kde.score_samples(embeddings))
    return res


def group_diversity(embeddings: np.ndarray, threshold: float) -> Tuple[bool, float]:
    """Calculate group diversity based on kde uniqueness.

    Parameters
    ----------
    item_embedding_space: np.ndarray :
        item embeddings for estimate uniqueness    

    Returns
    -------
    Tuple[bool, float]
        reject
        group diverstity

    """
    diverstity = np.mean(kde_uniqueness(embeddings=embeddings))
    return diverstity > threshold, diverstity

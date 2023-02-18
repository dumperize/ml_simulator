import numpy as np


def triplet_loss(
    anchor: np.ndarray, positive: np.ndarray, negative: np.ndarray, margin: float = 5.0
) -> float:
    """
    Computes the triplet loss using numpy.
    Using Euclidean distance as metric function.

    Args:
        anchor (np.ndarray): Embedding vectors of
            the anchor objects in the triplet (shape: (N, M))
        positive (np.ndarray): Embedding vectors of
            the positive objects in the triplet (shape: (N, M))
        negative (np.ndarray): Embedding vectors of
            the negative objects in the triplet (shape: (N, M))
        margin (float): Margin to enforce dissimilar samples to be farther apart than

    Returns:
        float: The triplet loss
    """
    dist_ap = np.linalg.norm(anchor-positive, axis=1)
    dist_an = np.linalg.norm(anchor-negative, axis=1)

    concat = np.vstack((dist_ap - dist_an + margin, np.zeros(dist_ap.shape[0])))
    loss_arr = np.amax(concat, axis=0)
    return np.mean(loss_arr)

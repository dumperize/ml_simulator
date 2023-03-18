import numpy as np


def contrastive_loss(
    x1: np.ndarray, x2: np.ndarray, y: np.ndarray, margin: float = 5.0
) -> float:
    """
    Computes the contrastive loss using numpy.
    Using Euclidean distance as metric function.

    Args:
        x1 (np.ndarray): Embedding vectors of the
            first objects in the pair (shape: (N, M))
        x2 (np.ndarray): Embedding vectors of the
            second objects in the pair (shape: (N, M))
        y (np.ndarray): Ground truthlabels (1 for similar, 0 for dissimilar)
            (shape: (N,))
        margin (float): Margin to enforce dissimilar samples to be farther apart than

    Returns:
        float: The contrastive loss
    """
    euclidian_dist = np.linalg.norm(x1-x2, axis=1)
    concat = np.vstack(
        (margin - euclidian_dist, np.zeros(euclidian_dist.shape[0])))
    loss_arr = y * euclidian_dist**2 + (1-y) * np.amax(concat, axis=0)**2
    return np.mean(loss_arr)

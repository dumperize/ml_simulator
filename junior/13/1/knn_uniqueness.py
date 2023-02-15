'import module'
import numpy as np
from sklearn.neighbors import NearestNeighbors


def knn_uniqueness(embeddings: np.ndarray, num_neighbors: int) -> np.ndarray:
    """Estimate uniqueness of each item in item embeddings group. Based on knn.

    Parameters
    ----------
    embeddings: np.ndarray :
        embeddings group 
    num_neighbors: int :
        number of neighbors to estimate uniqueness    

    Returns
    -------
    np.ndarray
        uniqueness estimates

    """
    neigh = NearestNeighbors(n_neighbors=num_neighbors)
    neigh.fit(embeddings)
    dist, _ = neigh.kneighbors(embeddings)
    mean_dist = np.mean(dist, axis=1)
    return mean_dist

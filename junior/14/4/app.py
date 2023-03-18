from typing import Tuple

import os
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi_utils.tasks import repeat_every
from sklearn.neighbors import KernelDensity


DIVERSITY_THRESHOLD = 10

app = FastAPI()
embeddings = {}


@app.on_event("startup")
@repeat_every(seconds=10)
def load_embeddings() -> dict:
    """Load embeddings from file."""

    # Load new embeddings each 10 seconds
    path = os.path.join(os.path.dirname(__file__), "embeddings.npy")
    embeddings_raw = np.load(path, allow_pickle=True).item()
    for item_id, embedding in embeddings_raw.items():
        embeddings[item_id] = embedding

    return {}


@app.get("/uniqueness/")
def uniqueness(item_ids: str) -> dict:
    """Calculate uniqueness of each product"""

    # Parse item IDs
    item_ids = [int(item) for item in item_ids.split(",")]

    # Calculate uniqueness
    item_embeddings = [embeddings.get(item_id) for item_id in item_ids]
    kde = kde_uniqueness(item_embeddings)
    item_uniqueness = {key: value for key, value in zip(item_ids, kde)}
    return item_uniqueness


@app.get("/diversity/")
def diversity(item_ids: str) -> dict:
    """Calculate diversity of group of products"""

    # Parse item IDs
    item_ids = [int(item) for item in item_ids.split(",")]
    item_embeddings = [embeddings.get(item_id) for item_id in item_ids]

    # Calculate diversity
    reject, diversity = group_diversity(item_embeddings, DIVERSITY_THRESHOLD)

    return {"reject": reject, "diversity": diversity}


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
    # Fit a kernel density estimator to the item embedding space
    kde = KernelDensity().fit(embeddings)

    uniqueness = []
    for item in embeddings:
        uniqueness.append(1 / np.exp(kde.score_samples([item])[0]))

    return np.array(uniqueness)


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
    diversity = np.sum(kde_uniqueness(embeddings)) / len(embeddings)
    reject = True if diversity >= threshold else False
    return reject, diversity


def main() -> None:
    """Run application"""
    uvicorn.run("main:app", host="localhost", port=5000)


if __name__ == "__main__":
    main()

"""Solution for Similar Items task"""
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
from scipy.spatial.distance import cosine
from itertools import combinations


class TupleComparator(tuple):
    def __lt__(self, other):
        return self[1] > other[1]


class SimilarItems:
    """Similar items class"""

    @staticmethod
    def similarity(embeddings: Dict[int, np.ndarray]) -> Dict[Tuple[int, int], float]:
        """Calculate pairwise similarities between each item
        in embedding.

        Args:
            embeddings (Dict[int, np.ndarray]): Items embeddings.

        Returns:
            Tuple[List[str], Dict[Tuple[int, int], float]]:
            List of all items + Pairwise similarities dict
            Keys are in form of (i, j) - combinations pairs of item_ids
            with i < j.
            Round each value to 8 decimal places.
        """
        comb = combinations(embeddings.keys(), 2)

        return {x: round(1 - cosine(embeddings[x[0]], embeddings[x[1]]), 8) for x in comb}

    @staticmethod
    def knn(
        sim: Dict[Tuple[int, int], float], top: int
    ) -> Dict[int, List[Tuple[int, float]]]:
        """Return closest neighbors for each item.

        Args:
            sim (Dict[Tuple[int, int], float]): <similarity> method output.
            top (int): Number of top neighbors to consider.

        Returns:
            Dict[int, List[Tuple[int, float]]]: Dict with top closest neighbors
            for each item.
        """
        knn_dict = {}
        for x, y in sim.keys():
            if not knn_dict.get(x):
                knn_dict[x] = []
            if not knn_dict.get(y):
                knn_dict[y] = []
            knn_dict[x].append((y, sim.get((x, y))))
            knn_dict[y].append((x, sim.get((x, y))))
        return {
            point: sorted(knn_dict[point], key=TupleComparator)[:top]
            for point in knn_dict
        }

    @staticmethod
    def knn_price(
        knn_dict: Dict[int, List[Tuple[int, float]]],
        prices: Dict[int, float],
    ) -> Dict[int, float]:
        """Calculate weighted average prices for each item.
        Weights should be positive numbers in [0, 2] interval.

        Args:
            knn_dict (Dict[int, List[Tuple[int, float]]]): <knn> method output.
            prices (Dict[int, float]): Price dict for each item.

        Returns:
            Dict[int, float]: New prices dict, rounded to 2 decimal places.
        """
        knn_price_dict = {}
        for x in knn_dict:
            values = []
            weights = []
            for key, value in knn_dict[x]:
                values.append(prices[key])
                weights.append(value + 1)
            knn_price_dict[x] = round(np.average(values, weights=weights), 2)
        return knn_price_dict

    @staticmethod
    def transform(
        embeddings: Dict[int, np.ndarray],
        prices: Dict[int, float],
        top: int,
    ) -> Dict[int, float]:
        """Transforming input embeddings into a dictionary
        with weighted average prices for each item.

        Args:
            embeddings (Dict[int, np.ndarray]): Items embeddings.
            prices (Dict[int, float]): Price dict for each item.
            top (int): Number of top neighbors to consider.

        Returns:
            Dict[int, float]: Dict with weighted average prices for each item.
        """
        sim = SimilarItems.similarity(embeddings=embeddings)
        knn = SimilarItems.knn(sim, top)
        knn_price_dict = SimilarItems.knn_price(knn, prices=prices)
        return knn_price_dict

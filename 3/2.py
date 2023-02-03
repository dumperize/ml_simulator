from typing import List, Tuple

import numpy as np
import math


def discounted_cumulative_gain(relevance: List[float], k: int, method: str = "standard") -> float:
    """Discounted Cumulative Gain

    Parameters
    ----------
    relevance : `List[float]`
        Video relevance list
    k : `int`
        Count relevance to compute
    method : `str`, optional
        Metric implementation method, takes the values​​
        `standard` - adds weight to the denominator
        `industry` - adds weights to the numerator and denominator
        `raise ValueError` - for any value

    Returns
    -------
    score : `float`
        Metric score
    """
    relevance = relevance[:k]

    def standart(value: Tuple[int, float]):
        position, x = value
        return x / math.log(position+2, 2)

    def industry(value: Tuple[int, float]):
        position, x = value
        return (2**x - 1) / math.log(position+2, 2)

    if method == 'standard':
        return sum(map(standart, enumerate(relevance)))
    if method == 'industry':
        return sum(map(industry, enumerate(relevance)))

    raise ValueError


# relevance = [0.99, 0.94, 0.88, 0.74, 0.71, 0.68]
# k = 5
# method = 'standard'
# print(discounted_cumulative_gain(relevance, k, method))

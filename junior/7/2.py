from typing import List, Tuple

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

    def standard(value: Tuple[int, float]):
        position, x = value
        return x / math.log(position+2, 2)

    def industry(value: Tuple[int, float]):
        position, x = value
        return (2**x - 1) / math.log(position+2, 2)

    if method == 'standard':
        return sum(map(standard, enumerate(relevance)))
    if method == 'industry':
        return sum(map(industry, enumerate(relevance)))

    raise ValueError

from typing import List, Tuple

import math


def discounted_cumulative_gain(
    relevance: List[float], k: int, method: str = "standard"
) -> float:
    """discounted cumulative gain"""
    relevance = relevance[:k]

    def standard(value: Tuple[int, float]):
        position, x = value
        return x / math.log(position + 2, 2)

    def industry(value: Tuple[int, float]):
        position, x = value
        return (2**x - 1) / math.log(position + 2, 2)

    if method == "standard":
        return sum(map(standard, enumerate(relevance)))
    if method == "industry":
        return sum(map(industry, enumerate(relevance)))

    raise ValueError


def normalized_dcg(relevance: List[float], k: int, method: str = "standard") -> float:
    """normalized DCG"""
    return discounted_cumulative_gain(
        relevance, k, method
    ) / discounted_cumulative_gain(sorted(relevance, reverse=True), k, method)


def avg_ndcg(
    list_relevances: List[List[float]], k: int, method: str = "standard"
) -> float:
    """avarage nDCG

    Parameters
    ----------
    list_relevances : `List[List[float]]`
        Video relevance matrix for various queries
    k : `int`
        Count relevance to compute
    method : `str`, optional
        Metric implementation method, takes the values ​​\
        `standard` - adds weight to the denominator\
        `industry` - adds weights to the numerator and denominator\
        `raise ValueError` - for any value

    Returns
    -------
    score : `float`
        Metric score
    """

    return sum(
        map(lambda l: normalized_dcg(l, k, method=method), list_relevances)
    ) / len(list_relevances)

from typing import List, Tuple

import math


def discounted_cumulative_gain(relevance: List[float], k: int, method: str = "standard") -> float:
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


def normalized_dcg(relevance: List[float], k: int, method: str = "standard") -> float:
    return discounted_cumulative_gain(relevance, k, method) / discounted_cumulative_gain(sorted(relevance, reverse=True), k, method)


def avg_ndcg(list_relevances: List[List[float]], k: int, method: str = 'standard') -> float:
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

    return sum(map(lambda l: normalized_dcg(l, k, method=method), list_relevances)) / len(list_relevances)


# list_relevances = [
#     [0.99, 0.94, 0.88, 0.89, 0.72, 0.65],
#     [0.99, 0.92, 0.93, 0.74, 0.61, 0.68],
#     [0.99, 0.96, 0.81, 0.73, 0.76, 0.6]
# ]
# k = 5
# method = 'standard'
# print(avg_ndcg(list_relevances, k, method))

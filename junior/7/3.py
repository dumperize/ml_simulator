from typing import List, Tuple

import math


def discounted_cumulative_gain(
    relevance: List[float], k: int, method: str = "standard"
) -> float:
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
    return discounted_cumulative_gain(
        relevance, k, method
    ) / discounted_cumulative_gain(sorted(relevance, reverse=True), k, method)

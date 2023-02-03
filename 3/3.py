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


# relevance = [0.99, 0.94, 0.74, 0.88, 0.71, 0.68]
# k = 5
# method = 'standard'
# print(normalized_dcg(relevance, k, method))

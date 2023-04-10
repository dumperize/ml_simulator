from typing import List, Tuple
from scipy import stats


def ttest(
    control: List[float],
    experiment: List[float],
    alpha: float = 0.05,
) -> Tuple[float, bool]:
    """Two-sample t-test for the means of two independent samples"""
    _, p_value = stats.ttest_ind(control, experiment)
    result = p_value < alpha

    return p_value, bool(result)


if __name__ == "__main__":
    import numpy as np
    rng = np.random.default_rng()
    rvs1 = stats.norm.rvs(loc=5, scale=10, size=500, random_state=rng)
    rvs2 = stats.norm.rvs(loc=5, scale=10, size=500, random_state=rng)
    print(ttest(rvs1, rvs2))

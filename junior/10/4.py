from typing import Tuple
from scipy import stats
import numpy as np


def cpc_sample(
    n_samples: int, conversion_rate: float, reward_avg: float, reward_std: float
) -> np.ndarray:
    """Sample data."""
    cvr = stats.binom.rvs(1, conversion_rate, size=n_samples)
    cpa = stats.norm.rvs(loc=reward_avg, scale=reward_std, size=n_samples)
    return cvr * cpa


def t_test(cpc_a: np.ndarray, cpc_b: np.ndarray, alpha=0.05) -> Tuple[bool, float]:
    """Perform t-test.

    Parameters
    ----------
    cpc_a: np.ndarray :
        first samples
    cpc_b: np.ndarray :
        second samples
    alpha :
         (Default value = 0.05)

    Returns
    -------
    Tuple[bool, float] :
        True if difference is significant, False otherwise
        p-value
    """
    ttest = stats.ttest_ind(cpc_a, cpc_b)
    return (bool(ttest.pvalue < alpha), float(ttest.pvalue))


def ab_test(
    n_simulations: int,
    n_samples: int,
    conversion_rate: float,
    mde: float,
    reward_avg: float,
    reward_std: float,
    alpha: float = 0.05,
) -> float:
    """Do the A/B test (simulation)."""

    type_2_errors = np.zeros(n_simulations)
    for i in range(n_simulations):
        # Generate one cpc sample with the given conversion_rate, reward_avg, and reward_std
        # Generate another cpc sample with the given conversion_rate * mde, reward_avg, and reward_std
        # Check t-test and save type 2 error
        cpv1 = cpc_sample(n_samples, conversion_rate, reward_avg, reward_std)
        cpv2 = cpc_sample(n_samples, conversion_rate * (1 + mde), reward_avg, reward_std)
        t = t_test(cpv1, cpv2, alpha)
        type_2_errors[i] = int(not t[0])

    # Calculate the type 2 errors rate
    return sum(type_2_errors) / len(type_2_errors)

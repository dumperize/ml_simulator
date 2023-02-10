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


def aa_test(
    n_simulations: int,
    n_samples: int,
    conversion_rate: float,
    reward_avg: float,
    reward_std: float,
    alpha: float = 0.05,
) -> float:
    """Do the A/A test (simulation)."""

    type_1_errors = np.zeros(n_simulations)
    for i in range(n_simulations):
        # Generate two cpc samples with the same cvr, reward_avg, and reward_std
        # Check t-test and save type 1 error
        cpv1 = cpc_sample(n_samples, conversion_rate, reward_avg, reward_std)
        cpv2 = cpc_sample(n_samples, conversion_rate, reward_avg, reward_std)
        t = t_test(cpv1, cpv2, alpha)
        type_1_errors[i] = int(t[0])
    # Calculate the type 1 errors rate
    return sum(type_1_errors) / len(type_1_errors)

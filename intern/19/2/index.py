from typing import List
from typing import Tuple

import numpy as np
from scipy.stats import ttest_ind


def ttest(
    control: List[float],
    experiment: List[float],
    alpha: float = 0.05,
) -> Tuple[float, bool]:
    """Two-sample t-test for the means of two independent samples"""
    _, p_value = ttest_ind(control, experiment)
    result = p_value < alpha

    return p_value, bool(result)

def quantile_ttest(
    control: List[float],
    experiment: List[float],
    alpha: float = 0.05,
    quantile: float = 0.95,
    n_bootstraps: int = 1000,
) -> Tuple[float, bool]:
    """
    Boostrapped t-test for quantiles of two samples.
    """
    bootstrapped_control_quantile = []
    bootstrapped_experiment_quantile = []
    
    for _ in range(n_bootstraps):
        bootstrapped_control = np.random.choice(control, size=len(control), replace=True)
        bootstrapped_experiment = np.random.choice(experiment, size=len(experiment), replace=True)

        bootstrapped_control_quantile.append(np.quantile(bootstrapped_control, quantile))
        bootstrapped_experiment_quantile.append(np.quantile(bootstrapped_experiment, quantile))

    
    return ttest(bootstrapped_control_quantile, bootstrapped_experiment_quantile, alpha)

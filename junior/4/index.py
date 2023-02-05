import pandas as pd
import math
from scipy import stats


def count_lin_reg(x: pd.DataFrame) -> pd.Series:
    """Calc lin regression and koef determination"""
    res = stats.linregress(x["price"], (x["qty"] + 1).apply(math.log))
    return pd.Series({"elasticity": res.rvalue**2})


def elasticity_df(df: pd.DataFrame) -> pd.DataFrame:
    """Calc elasticity"""
    return df.groupby("sku", as_index=False).apply(count_lin_reg)

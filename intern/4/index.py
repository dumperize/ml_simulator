import pandas as pd


def fillna_with_mean(df: pd.DataFrame, target: str, group: str) -> pd.DataFrame:
    """Fill nan with mean value by group"""

    values = { target: df.groupby(group)[target].transform('mean')}
    return df.fillna(values)

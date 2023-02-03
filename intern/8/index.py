import pandas as pd


def limit_gmv(df: pd.DataFrame) -> pd.DataFrame:
    """Limit product by stock"""

    count_product = ((df["gmv"] / df["price"]).astype(int)).clip(upper=df["stock"])
    return df.assign(gmv=count_product * df["price"])

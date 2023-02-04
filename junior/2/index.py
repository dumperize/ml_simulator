import numpy as np
import pandas as pd

convert_fn = {"med": "median", "max": "max", "min": "min", "avg": "mean"}


def new_price(X):
    """Get aggregation price of competitors"""
    type_agg = X['agg'].iloc[0]
    if type_agg == 'rnk':
        return pd.Series({'comp_price': X[X['rank'] == X['rank'].min()]['comp_price'].values[0]})
    else:
        return pd.Series({'comp_price': X['comp_price'].agg(convert_fn[type_agg])})


def agg_comp_price(X: pd.DataFrame) -> pd.DataFrame:
    """Calc dinamic price"""
    full = (
        X.groupby(["sku", "agg", "base_price"], as_index=False)
        .apply(new_price)
    )

    koef = full['comp_price'] / full['base_price']
    full['new_price'] = np.where((koef >= 0.8) & (
        koef <= 1.2), full['comp_price'], full['base_price'])
    return full.sort_values(by=['sku']).reset_index(drop=True)

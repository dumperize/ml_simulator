"""Metrics."""

from typing import Any, Dict, Union, List
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import pyspark.sql as ps


def not_null(df: ps.DataFrame, column_name: str):
    """Not null element"""

    from pyspark.sql.functions import isnan
    col = df.select(column_name)[0]
    return ~isnan(col) & col.isNotNull()


@dataclass
class Metric:
    """Base class for Metric"""

    def __call__(self, df: Union[pd.DataFrame, ps.DataFrame]) -> Dict[str, Any]:
        if isinstance(df, pd.DataFrame):
            return self._call_pandas(df)

        if isinstance(df, ps.DataFrame):
            return self._call_pyspark(df)

        msg = (
            f"Not supported type of arg 'df': {type(df)}. "
            "Supported types: pandas.DataFrame, "
            "pyspark.sql.dataframe.DataFrame"
        )
        raise NotImplementedError(msg)

    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        return {}

    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:
        return {}


@dataclass
class CountTotal(Metric):
    """Total number of rows in DataFrame"""

    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        return {"total": len(df)}

    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:
        return {"total": df.count()}


@dataclass
class CountZeros(Metric):
    """Number of zeros in choosen column"""

    column: str

    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        n = len(df)
        k = sum(df[self.column] == 0)
        return {"total": n, "count": k, "delta": k / n}

    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:
        from pyspark.sql.functions import col, count

        n = df.count()
        k = df.filter(col(self.column) == 0).count()
        return {"total": n, "count": k, "delta": k / n}


@dataclass
class CountNull(Metric):
    """Number of empty values in choosen columns"""

    columns: List[str]
    aggregation: str = "any"  # either "all", or "any"

    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        n = len(df)
        is_null = df[self.columns].isnull()
        if self.aggregation == "any":
            mask = is_null.any(axis=1)
        else:
            mask = is_null.all(axis=1)
        k = len(df[mask])
        return {"total": n, "count": k, "delta": k / n}

    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:
        from pyspark.sql.functions import col, isnan, when, count

        n = df.count()

        c = df.select(self.columns[0])[0]
        mask = c.isNull() | isnan(c)
        if self.aggregation == "any":
            for column in self.columns[1:]:
                c = df.select(column)[0]
                mask = mask | (c.isNull() | isnan(c))
        else:
            for column in self.columns[1:]:
                c = df.select(column)[0]
                mask = mask & (c.isNull() | isnan(c))
        k = df.where(mask).count()

        return {"total": n, "count": k, "delta": k / n}


@dataclass
class CountDuplicates(Metric):
    """Number of duplicates in choosen columns"""

    columns: List[str]

    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        n = len(df)
        k = sum(df[self.columns].duplicated())
        return {"total": n, "count": k, "delta": k / n}

    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:
        from pyspark.sql.functions import sum
        n = df.count()

        df_group = df.groupBy(self.columns).count().filter("count > 1")
        df_sum = df_group.select(sum(df_group["count"]))
        k = df_sum.collect()[0][0] - df_group.count()

        return {"total": n, "count": k, "delta": k / n}


@dataclass
class CountValue(Metric):
    """Number of values in choosen column"""

    column: str
    value: Union[str, int, float]

    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        n = len(df)
        k = sum(df[self.column] == self.value)
        return {"total": n, "count": k, "delta": k / n}

    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:
        from pyspark.sql.functions import col, count
        n = df.count()

        k = df.filter(df[self.column] == self.value).count()
        return {"total": n, "count": k, "delta": k / n}


@dataclass
class CountBelowValue(Metric):
    """Number of values below threshold"""

    column: str
    value: float
    strict: bool = False

    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        n = len(df)
        k = sum(df[self.column] <
                self.value if self.strict else df[self.column] <= self.value)

        return {"total": n, "count": k, "delta": k / n}

    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:
        from pyspark.sql.functions import col, count
        n = df.count()

        k = df.filter(
            df[self.column] < self.value if self.strict else df[self.column] <= self.value).count()
        return {"total": n, "count": k, "delta": k / n}


@dataclass
class CountBelowColumn(Metric):
    """Count how often column X below Y"""

    column_x: str
    column_y: str
    strict: bool = False

    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        n = len(df)
        k = sum(df[self.column_x] < df[self.column_y]
                if self.strict else df[self.column_x] <= df[self.column_y])

        return {"total": n, "count": k, "delta": k / n}

    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:
        n = df.count()

        condition = df[self.column_x] < df[self.column_y] if self.strict else df[self.column_x] <= df[self.column_y]
        k = df.filter(not_null(df, self.column_x) & not_null(
            df, self.column_y) & condition).count()
        return {"total": n, "count": k, "delta": k / n}


@dataclass
class CountRatioBelow(Metric):
    """Count how often X / Y below Z"""

    column_x: str
    column_y: str
    column_z: str
    strict: bool = False

    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        n = len(df)
        ratio = df[self.column_x]/df[self.column_y]
        k = sum(ratio < df[self.column_z]
                if self.strict else ratio <= df[self.column_z])
        return {"total": n, "count": k, "delta": k / n}

    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:
        n = df.count()

        ratio = df[self.column_x]/df[self.column_y]
        condition = ratio < df[self.column_z] if self.strict else ratio <= df[self.column_z]

        k = df.filter(not_null(df, self.column_x) & not_null(
            df, self.column_y) & not_null(df, self.column_z) & condition).count()

        return {"total": n, "count": k, "delta": k / n}


@dataclass
class CountCB(Metric):
    """Calculate lower/upper bounds for N%-confidence interval"""

    column: str
    conf: float = 0.95

    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        step = (1 - self.conf) / 2
        lcb, ucb = df[self.column].quantile([step, 1 - step])
        return {"lcb": lcb, "ucb": ucb}

    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:
        from pyspark.sql.functions import percentile_approx
        step = (1 - self.conf) / 2
        df = df.select(percentile_approx(
            self.column, [step, 1 - step]).alias("quantiles"))
        value = df.collect()[0][0]

        return {"lcb": value[0], "ucb": value[1]}


@dataclass
class CountLag(Metric):
    """A lag between latest date and today"""

    column: str
    fmt: str = "%Y-%m-%d"

    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        a = datetime.today()
        b = datetime.strptime(max(df[self.column]), self.fmt)
        lag = a - b
        return {
            "today": a.strftime(self.fmt),
            "last_day": b.strftime(self.fmt),
            "lag": lag.days
        }

    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:
        from pyspark.sql.functions import max

        a = datetime.today()
        maximum = df.select(max(self.column)).collect()[0][0]
        b = datetime.strptime(maximum, self.fmt)
        lag = a - b
        return {
            "today": a.strftime(self.fmt),
            "last_day": b.strftime(self.fmt),
            "lag": lag.days
        }

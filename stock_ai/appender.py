"""DataFrame附加器"""
import pandas as pd


def append_year(df, **kwargs):
    """附加年信息

    Args:
        df (:class:`~pandas.DataFrame`): 原始数据。

    Examples:
        >>> appender.append_year(data_processor.load_stock_daily('601398')).head()
        date
        2006-10-27    2006
        2006-10-30    2006
        2006-10-31    2006
        2006-11-01    2006
        2006-11-02    2006
        Name: col, dtype: int64

    Returns:
        :class:`~pandas.Series`: 计算后的列。
    """
    return _create_series(df.index.year, df=df, dtype='int64')


def _create_series(data, df, **kwargs):
    """

    Args:
        data: 参考 :func:`~pandas.Series` 中同名参数。
        df (:class:`~pandas.DataFrame`): 原始数据。
        **kwargs:

    Returns:

    """
    name = kwargs.pop('name', None)
    dtype = kwargs.pop('dtype', None)
    copy = kwargs.pop('copy', None)
    fastpath = kwargs.pop('fastpath', None)
    return pd.Series(data,
                     index=df.index,
                     name=name,
                     dtype=dtype,
                     copy=copy,
                     fastpath=fastpath)


def append_month(df, **kwargs):
    """附加月信息

    Args:
        df (:class:`~pandas.DataFrame`): 原始数据。

    Examples:
        >>> appender.append_month(data_processor.load_stock_daily('601398')).head()
        date
        2006-10-27    10
        2006-10-30    10
        2006-10-31    10
        2006-11-01    11
        2006-11-02    11
        Name: col, dtype: int64

    Returns:
        :class:`~pandas.Series`: 计算后的列。
    """
    return _create_series(df.index.month, df=df, dtype='int64')

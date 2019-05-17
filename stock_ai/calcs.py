"""数据计算器"""
import pandas as pd


def calc_year(df, **kwargs):
    """计算年信息

    Args:
        df (:class:`~pandas.DataFrame`): 原始数据。

    Examples:
        >>> appender.calc_year(data_processor.load_stock_daily('601398')).head()
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


def is_trade_suspension(df, **kwargs):
    """返回是否为停牌

    根据 `df` 中指定列 `close` 来确定是否为停牌日。（如果数据为 NaN 则为True，否则为False）

    Args:
        df (:class:`~pandas.DataFrame`): 原始数据。
        colname (str): 用来判断的列名。默认为 ``close``。

    Examples:
        >>> df_stock = data_processor.load_stock_daily('601398')
        >>> df_index = data_processor.load_index_daily('399300')
        >>> df = wrapper.dataframe_merge(df_stock, df_index)
        >>> is_sus = calcs.is_trade_suspension(df)
        >>> is_sus['2012-02-23'] # 2012-01-06 召开2012年度第1次临时股东大会，停牌一天 ，2012-02-23
        True
        >>> is_sus['2012-02-24']
        False

    Returns:
        :class:`~pandas.Series`: 如果是停牌数据则为True，否则为False。
    """
    colname = kwargs.pop('colname', 'close')
    return df[colname].isna()


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


def calc_month(df, **kwargs):
    """计算月信息

    Args:
        df (:class:`~pandas.DataFrame`): 原始数据。

    Examples:
        >>> appender.calc_month(data_processor.load_stock_daily('601398')).head()
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

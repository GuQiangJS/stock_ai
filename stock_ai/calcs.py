"""数据计算器"""
import pandas as pd
from QUANTAXIS.QAIndicator import indicators


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

    根据 `df` 中指定参数 `colname` 找到指定列，采用 :func:`pandas.Series.isna` 判断。
    ``True`` 表示是停牌日。

    See Also:
        :func:`pandas.Series.isna`

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


def tech_ma(df: pd.DataFrame, **kwargs) -> pd.Series:
    """计算移动均线

    Args:
        days (int): 天数。默认值为5。
    """
    days = kwargs.pop('days', 5)
    return indicators.MA(df['close'], days)


def tech_ewm(df: pd.DataFrame, **kwargs) -> pd.Series:
    """计算指数均线

    Args:
        days (int): 天数。默认值为5。
    """
    days = kwargs.pop('days', 5)
    indicators.QA_indicator_MA()
    return indicators.EMA(df['close'], days)


def tech_macd(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """计算MACD

    Args:
        short (int): 默认值12。
        long (int): 默认值26。
        mid (int): 默认值9。
        columns (str): 取得列。默认为 ``MACD``。实际包含 ``['DIF','DEA','MACD']``

    Examples:
        >>> df_stock = data_processor.load_stock_daily('601398')
        >>> df_index = data_processor.load_index_daily('399300')
        >>> df = wrapper.dataframe_merge(df_stock, df_index)
        >>> calcs.tech_macd(df).tail()
                         DIF       DEA      MACD
        date
        2019-01-18 -0.013661 -0.029889  0.032456
        2019-01-21  0.002288 -0.023454  0.051484
        2019-01-22  0.009972 -0.016769  0.053482
        2019-01-23  0.019867 -0.009441  0.058617
        2019-01-24  0.028989 -0.001755  0.061488

    Returns:
        :class:`pandas.Series` 或者 :class:`pandas.DataFrame`
    """
    short = kwargs.pop('short', 12)
    long = kwargs.pop('long', 26)
    mid = kwargs.pop('mid', 9)
    columns = kwargs.pop('columns', 'MACD')
    return indicators.QA_indicator_MACD(df, short, long, mid)[columns]


def tech_boll(df: pd.DataFrame, **kwargs) -> pd.Series:
    """计算布林线

    See Also:
        https://zh.wikipedia.org/wiki/%E5%B8%83%E6%9E%97%E5%B8%A6

    Args:
        df:
        window (int): 窗口期。默认值为 20。
        n (int): 标准差。默认值为 2。

    Returns:

    """
    N = kwargs.pop('N', 20)
    K = kwargs.pop('K', 2)
    return indicators.QA_indicator_BOLL(df, N, K)['BOLL']


def daily_return(df: pd.DataFrame, **kwargs):
    """计算日收益

    Args:
        data (:class:`pandas.Series` 或 :class:`pandas.DataFrame`): 待计算的数据。

    Return:
        :class:`pandas.Series` or :class:`pandas.DataFrame`:
    """
    return df[:-1].values / df[1:] - 1


def cum_return(data, **kwargs):
    """计算累计收益

    Args:
        data (:class:`pandas.Series` 或 :class:`pandas.DataFrame`): 待计算的数据。

    Return:
        :class:`pandas.Series` or :class:`pandas.DataFrame`:
    """
    return data / data.iloc[0]


def sharpe_ratio(r=None, rf=None, r_std: float = None) :
    """计算 `夏普比率`_
    Args:
        r (:class:`pandas.DataFrame` or :class:`pandas.Series` or float): 收益数据表或均值(`float`)。
        rf (:class:`pandas.DataFrame` or :class:`pandas.Series` or float): 无风险收益率表或均值( `float` )。
        r_std: 参数 `r` 的标准差。如果 `r` 传入的是 :class:`pandas.DataFrame` or :class:`pandas.Series` 则无需传入此参数。
    Returns:
        float: 计算后的夏普比率。

    .. _夏普比率:
        https://zh.wikipedia.org/wiki/%E8%AF%81%E5%88%B8%E6%8A%95%E8%B5%84%E5%9F%BA%E9%87%91#%E5%A4%8F%E6%99%AE%E6%AF%94%E7%8E%87
    """
    # 夏普比率是回报与风险的比率。公式为：
    # （Rp - Rf） / ？p
    # 其中：
    #
    # Rp = 投资者投资组合的预期回报率
    # Rf = 无风险回报率
    # ？p = 投资组合的标准差，风险度量

    r_mean = r
    rf_mean = rf
    rf_std = r_std
    if isinstance(r, pd.DataFrame) or isinstance(r,pd.Series):
        r_mean = r.mean()
        rf_std = r.std()
    if isinstance(rf, pd.DataFrame) or isinstance(rf,pd.Series):
        rf_mean = rf.mean()

    result = (r_mean - rf_mean) / rf_std
    return result if isinstance(result, float) else result[0]
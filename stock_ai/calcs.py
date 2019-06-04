"""数据计算器"""
import pandas as pd
from QUANTAXIS.QAIndicator import indicators
from QUANTAXIS.QAIndicator import talib_indicators
from QUANTAXIS.QAIndicator import talib_series
import numpy as np
from sklearn.preprocessing import OneHotEncoder


def calc_year(df, **kwargs):
    """计算年信息

    Args:
        df (:class:`~pandas.DataFrame`): 原始数据。

    Examples:
        >>> from stock_ai import calcs
        >>> dates = pd.date_range('20130101', periods=3, freq='Y')
        >>> df = pd.DataFrame([1, 3, 5], index=dates, columns=list('A'))
        >>> df
                    A
        2013-12-31  1
        2014-12-31  3
        2015-12-31  5
        >>> calcs.calc_year(df).values
        array([2013, 2014, 2015], dtype=int64)

    Returns:
        :class:`~pandas.Series`: 计算后的列。
    """
    return _create_series(df.index.year, df=df, dtype='int64')


def trans_onehot(data):
    """转换 OneHot 编码

     设定参数 ``sparse=False``，调用 :py:class:`sklearn.preprocessing.OneHotEncoder`。

    Args:
        data: 一维数组或多维数组，可以为普通list,tuple，或np.array。

    Examples:
        >>> from stock_ai import calcs
        >>> import numpy as np
        >>> calcs.trans_onehot([[2016], [2017], [2018], [2018], [2019]])
        array([[1., 0., 0., 0.],
               [0., 1., 0., 0.],
               [0., 0., 1., 0.],
               [0., 0., 1., 0.],
               [0., 0., 0., 1.]])
        >>> calcs.trans_onehot([2016, 2017, 2018, 2018, 2019])
        array([[1., 0., 0., 0.],
               [0., 1., 0., 0.],
               [0., 0., 1., 0.],
               [0., 0., 1., 0.],
               [0., 0., 0., 1.]])

    Returns:
        (2-d array): 参考 :py:func:`sklearn.preprocessing.OneHotEncoder.fit_transform` 的返回值。

    """
    n = np.array(data)
    if len(n.shape) == 1:
        #如果是一维数组，则转换为二维数组。因为OneHotEncoder只接受二维数组。
        n = np.reshape(n, (n.shape[0], 1))
    return OneHotEncoder(sparse=False).fit_transform(n)


def fillna(df, **kwargs):
    """:func:`pandas.DataFraem.fillna`

    Args:
        df (:class:`~pandas.DataFrame`): 原始数据。
    """
    return df.fillna(**kwargs)


def dropna(df, **kwargs):
    """:func:`pandas.DataFraem.dropna`

    Args:
        df (:class:`~pandas.DataFrame`): 原始数据。

    Returns:

    """
    return df.dropna(**kwargs)


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
        >>> from stock_ai import calcs
        >>> dates = pd.date_range('20130101', periods=3, freq='M')
        >>> df = pd.DataFrame([1, 3, 5], index=dates, columns=list('A'))
        >>> df
                    A
        2013-01-31  1
        2013-02-28  3
        2013-03-31  5
        >>> calcs.calc_month(df).values
        array([1, 2, 3], dtype=int64)

    Returns:
        :class:`~pandas.Series`: 计算后的列。
    """
    return _create_series(df.index.month, df=df, dtype='int64')


def tech_ma(df: pd.DataFrame, **kwargs) -> pd.Series:
    """计算移动均线

    Args:
        days (int): 天数。默认值为5。

    Examples:
        >>> df_stock = data_processor.load_stock_daily('601398')
        >>> calcs.tech_ma(df).tail()
        date
        2019-01-18    5.262
        2019-01-21    5.312
        2019-01-22    5.344
        2019-01-23    5.384
        2019-01-24    5.426
        Name: MA, dtype: float64
    """
    days = kwargs.pop('days', 5)
    ma = indicators.MA(df['close'], days)
    return _create_series(ma, df, name='MA', dtype=ma.dtype)


def _STD(Series, N):
    return pd.Series.rolling(Series, N).std()


def fft(df: pd.DataFrame, **kwargs):
    num = kwargs.pop('num', 3)
    close_fft = np.fft.fft(np.asarray(df['close'].tolist()))
    fft_df = pd.DataFrame({'fft': close_fft})
    fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
    fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))
    fft_list = np.asarray(fft_df['fft'].tolist())
    fft_list_m10 = np.copy(fft_list)
    fft_list_m10[num:-num] = 0
    return _create_series(np.fft.ifft(fft_list_m10),
                          df=df,
                          name='fft',
                          dtype=close_fft.dtype)


def tech_ema(df: pd.DataFrame, **kwargs) -> pd.Series:
    """计算指数移动均线

    Args:
        days (int): 天数。默认值为5。

    Examples:
        >>> df_stock = data_processor.load_stock_daily('601398')
        >>> calcs.tech_ema(df).tail()
        date
        2019-01-18    5.287964
        2019-01-21    5.341976
        2019-01-22    5.357984
        2019-01-23    5.385323
        2019-01-24    5.410215
        Name: EMA, dtype: float64
    """
    days = kwargs.pop('days', 5)
    ema = indicators.EMA(df['close'], days)
    return _create_series(ema, df=df, dtype=ema.dtype, name='EMA')


def tech_bbands(df: pd.DataFrame, **kwargs):
    """布林带

    Args:
        df: 日线数据。
        timeperiod (int): 窗口期。默认为 ``5``。
        nbdevup (int): 上轨标准差。默认为 ``2``。
        nbdevdn (int): 下轨标准差。默认为 ``2``。

    Examples:
        >>> df_stock = data_processor.load_stock_daily('601398')
        >>> calcs.tech_bbands(df).tail()
                     BOLL        UB        LB
        date
        2019-01-18  5.262  5.409919  5.114081
        2019-01-21  5.312  5.514188  5.109812
        2019-01-22  5.344  5.531403  5.156597
        2019-01-23  5.384  5.543750  5.224250
        2019-01-24  5.426  5.493231  5.358769

    Returns:
        :class:`pandas.DataFrame`
    """
    timeperiod = kwargs.pop('timeperiod', 5)
    nbdevup = kwargs.pop('nbdevup', 2)
    nbdevdn = kwargs.pop('nbdevdn', 2)
    # matype = kwargs.pop('matype', 0)

    C = df['close']
    boll = tech_ma(df, days=timeperiod)
    UB = boll + nbdevup * _STD(C, timeperiod)
    LB = boll - nbdevdn * _STD(C, timeperiod)
    DICT = {'BOLL': boll, 'UB': UB, 'LB': LB}

    return pd.DataFrame(DICT)


def tech_macd(df: pd.DataFrame, **kwargs):
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


def pct_change(df, **kwargs):
    """包装 :func:`pandas.DataFrame.pct_change` 方法

    Args:
        data (:class:`pandas.Series` 或 :class:`pandas.DataFrame`): 待计算的数据。

    Return:
        :class:`pandas.Series` or :class:`pandas.DataFrame`:
    """
    return df.pct_change()


def daily_return(df, **kwargs):
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


def kurtosis(data, **kwargs):
    """计算峰度

    Args:
        data (:class:`pandas.Series` 或 :class:`pandas.DataFrame`): 待计算的数据。

    Return:
        :class:`pandas.Series` or :class:`pandas.DataFrame`:
    """
    return data.kurtosis()


def skew(data, **kwargs):
    """计算偏度

    > 若数据分布是对称的，偏度为0；
    > 若偏度>0,则可认为分布为右偏，即分布有一条长尾在右；
    > 若偏度<0，则可认为分布为左偏，即分布有一条长尾在左；
    > 同时偏度的绝对值越大，说明分布的偏移程度越严重。

    Args:
        data (:class:`pandas.Series` 或 :class:`pandas.DataFrame`): 待计算的数据。

    Return:
        :class:`pandas.Series` or :class:`pandas.DataFrame`:
    """
    return data.skew()


def sharpe_ratio(r=None, rf=None, r_std: float = None):
    """计算 `夏普比率`_

    夏普指数代表投资人每多承担一分风险，可以拿到几分报酬；
    若为正值，代表基金报酬率高过波动风险；
    若为负值，代表基金操作风险大过于报酬率。
    这样一来，每个投资组合都可以计算Sharpe Ratio,即投资回报与多冒风险的比例，这个比例越高，投资组合越佳。

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
    if isinstance(r, pd.DataFrame) or isinstance(r, pd.Series):
        r_mean = r.mean()
        rf_std = r.std()
    if isinstance(rf, pd.DataFrame) or isinstance(rf, pd.Series):
        rf_mean = rf.mean()

    result = (r_mean - rf_mean) / rf_std
    return result if isinstance(result, float) else result[0]


def drop_column(data: pd.DataFrame, **kwargs):
    """将数据源中的部分列丢弃，参考 :func:`pandas.DataFrame.drop`"""
    return data.drop(**kwargs)

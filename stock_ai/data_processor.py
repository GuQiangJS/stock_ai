"""数据读写器"""

from stock_ai import util
import datetime
import QUANTAXIS as QA
import socket
import pandas as pd
import os
import tushare as ts

_csv_encoding = 'utf-8'  #CSV文件默认字符编码


def _save_dataframe_to_csv(data, path, **kwargs):
    encoding = kwargs.pop('encoding', _csv_encoding)
    os.makedirs(path, exist_ok=True)
    data.to_csv(path, encoding=encoding)


def save_stock_daily_csv(data, path, **kwargs):
    """保存股票日线数据至csv文件

    Args:
        data (pd.DataFrame): 数据表。
        path (str): 待保存的文件完整路径。
        encoding (str): 文件编码。默认为 ``utf-8``。参考 :py:func:`pandas.DataFrame.to_csv` 中同名参数。
    """
    _save_dataframe_to_csv(data, path, kwargs)


def load_stock_daily(code, **kwargs):
    """读取股票日线数据（在线或从数据库）

        Args:
            code (str): 股票代码。
            start (str, optional): 开始日期。数据格式为%Y-%m-%d。默认为 ``1990-01-01``。
            end (str, optional): 结束日期。数据格式为%Y-%m-%d。默认为 :func:`datetime.now`。
            online (bool, optional): 是否获取在线数据。默认为 ``False``。
            fq (str, optional): 是否取复权数据。默认为 ``qfq``。
                - ``qfq`` : 前复权
                - ``hfq`` : 后复权
                - ``bfq`` or ``None`` : 不复权
            columns (tuple, optional): 获取的列集合。默认为
                ``['open', 'high', 'low', 'close', 'volume', 'amount']``

        Examples:
            >>> data_processor.load_stock_daily('601398').head()
                        open  high   low  close      volume        amount
            date
            2006-10-27  3.40  3.44  3.26   3.28  25825396.0  8.725310e+09
            2006-10-30  3.27  3.32  3.25   3.29   3519210.0  1.153128e+09
            2006-10-31  3.28  3.33  3.28   3.30   2301262.0  7.610508e+08
            2006-11-01  3.30  3.31  3.28   3.30   1328924.0  4.372648e+08
            2006-11-02  3.30  3.30  3.25   3.28   1751554.0  5.733994e+08

        Returns:
            :py:class:`pandas.DataFrame`: 股票日线数据。

        """
    start = kwargs.pop('start', '1990-01-01')
    end = kwargs.pop('end', util.date2str(datetime.datetime.now()))
    online = kwargs.pop('online', False)
    fq = kwargs.pop('fq', 'qfq')
    columns = kwargs.pop('columns',
                         ['open', 'high', 'low', 'close', 'volume', 'amount'])
    if online:
        return load_stock_daily_online(code, start, end, columns)
    else:
        return load_stock_daily_mongodb(code, start, end, fq, columns)


def load_stock_daily_online(code, start, end, columns, times=5):
    """读取股票在线日线数据

    Notes:
        在线数据只取tdx的数据。

    Args:
        code: 参考 :func:`load_stock_daily` 中参数 ``code`` 的说明。
        start: 参考 :func:`load_stock_daily` 中参数 ``start`` 的说明。
        end: 参考 :func:`load_stock_daily` 中参数 ``end`` 的说明。
        fq: 参考 :func:`load_stock_daily` 中参数 ``fq`` 的说明。
        columns: 参考 :func:`load_stock_daily` 中参数 ``columns`` 的说明。
        times (int, optional): 重试次数。默认为 ``5``。
    Returns:
        :py:class:`pandas.DataFrame`: 股票日线数据。
    """
    retries = 0
    while True:
        try:
            df = QA.QAFetch.QATdx.QA_fetch_get_stock_day(code, start, end)
            # 原始列名['open', 'close', 'high', 'low', 'vol', 'amount', 'code', 'date', 'date_stamp']
            df = df.rename(columns={'vol': 'volume'})[columns]
            df.index = pd.to_datetime(df.index)
            return df
        except socket.timeout:
            if retries < times:
                retries = retries + 1
                continue
            raise


def load_stock_daily_csv(path, **kwargs):
    """从CSV文件读取股票日线数据

    Args:
        path (str): csv文件所在的完整路径。
        index_col (int, sequence or bool, optional): 索引列。参考 :py:func:`pandas.read_csv` 中同名参数。
        encoding (str): 文件编码。默认为 `utf-8`。参考 :py:func:`pandas.read_csv` 中同名参数。
        parse_dates (str): 文件编码。默认为 `utf-8`。参考 :py:func:`pandas.read_csv` 中同名参数。
        float_precision (str): 浮点类型转换。默认为 `round_trip`。参考 :py:func:`pandas.read_csv` 中同名参数。
    Returns:
        :py:class:`pandas.DataFrame`: 股票日线数据。
    """
    index_col = kwargs.pop('index_col', 'date')
    encoding = kwargs.pop('encoding', _csv_encoding)
    parse_dates = kwargs.pop('parse_dates', ['date'])
    float_precision = kwargs.pop('float_precision', 'round_trip')
    return pd.read_csv(path,
                       index_col=index_col,
                       encoding=encoding,
                       parse_dates=parse_dates,
                       float_precision=float_precision)


def load_stock_daily_mongodb(code, start, end, fq, columns):
    """读取股票本地日线数据

    Args:
        code: 参考 :func:`load_stock_daily` 中参数 ``code`` 的说明。
        start: 参考 :func:`load_stock_daily` 中参数 ``start`` 的说明。
        end: 参考 :func:`load_stock_daily` 中参数 ``end`` 的说明。
        fq: 参考 :func:`load_stock_daily` 中参数 ``fq`` 的说明。
        columns: 参考 :func:`load_stock_daily` 中参数 ``columns`` 的说明。

    Returns:
        :py:class:`pandas.DataFrame`: 股票日线数据。

    """
    d = QA.QA_fetch_stock_day_adv(code, start=start, end=end)
    if not d:
        return pd.DataFrame()
    if fq == 'qfq':
        df = d.to_qfq()
    elif fq == 'hfq':
        df = d.to_hfq()
    else:
        df = d.data
    # 原始列名['open', 'high', 'low', 'close', 'volume', 'amount', 'preclose', 'adj']
    df = df.reset_index().drop(columns=['code']).set_index('date')
    return df[columns]


def load_index_daily(code, **kwargs):
    """读取指数日线数据（在线或从数据库）

        Args:
            code (str): 指数代码。
            start (str, optional): 开始日期。数据格式为%Y-%m-%d。默认为 ``1990-01-01``。
            end (str, optional): 结束日期。数据格式为%Y-%m-%d。默认为 :func:`datetime.now`。
            online (bool, optional): 是否获取在线数据。默认为 ``False``。
            columns (tuple, optional): 获取的列集合。默认为
                ``['open', 'high', 'low', 'close', 'up_count', 'down_count', 'volume','amount']``

        Returns:
            :py:class:`pandas.DataFrame`: 股票日线数据。

        """
    start = kwargs.pop('start', '1990-01-01')
    end = kwargs.pop('end', util.date2str(datetime.datetime.now()))
    online = kwargs.pop('online', False)
    columns = kwargs.pop('columns', [
        'open', 'high', 'low', 'close', 'up_count', 'down_count', 'volume',
        'amount'
    ])
    if online:
        return load_index_daily_online(code, start, end, columns)
    else:
        return load_index_daily_mongodb(code, start, end, columns)


def load_index_daily_online(code, start, end, columns, times=5):
    """读取指数在线日线数据

    Args:
        start (str):  参考 :func:`load_index_daily` 中参数 ``start`` 的说明。
        end (str): 参考 :func:`load_index_daily` 中参数 ``end`` 的说明。
        columns: 参考 :func:`load_index_daily` 中参数 ``columns`` 的说明。原始的列名中volume是vol，为了统一，内部会修改为volume。
        code (str): 参考 :func:`load_index_daily` 中参数 ``code`` 的说明。
        times (int, optional): 重试次数。默认为 ``5``。
    Returns:
        :py:class:`pandas.DataFrame`: 股票日线数据。
    """
    retries = 0
    while True:
        try:
            df = QA.QAFetch.QATdx.QA_fetch_get_index_day(code, start, end)
            # 原始列名['open', 'close', 'high', 'low', 'vol', 'amount', 'up_count', 'down_count', 'date', 'code', 'date_stamp'] pylint: disable=C0301
            df = df.rename(columns={'vol': 'volume'})[columns]
            return df
        except socket.timeout:
            if retries < times:
                retries = retries + 1
                continue
            raise


def load_index_daily_mongodb(code, start, end, columns):
    """读取指数本地日线数据

    Args:
        start (str):  参考 :func:`load_index_daily` 中参数 ``start`` 的说明。
        end (str): 参考 :func:`load_index_daily` 中参数 ``end`` 的说明。
        columns: 参考 :func:`load_index_daily` 中参数 ``columns`` 的说明。
        code (str): 参考 :func:`load_index_daily` 中参数 ``code`` 的说明。
    Returns:
        :py:class:`pandas.DataFrame`: 指数日线数据。
    """
    d = QA.QA_fetch_index_day_adv(code, start, end)
    df = d.data.reset_index().drop(columns=['code']).set_index('date')
    return df[columns]


def load_stock_list(online=False):
    """获取股票列表

    Args:
        online (bool): 是否获取在线数据。默认为 ``False``。

    Returns:
        :py:class:`pandas.DataFrame`: 股票列表。
    """
    return load_stock_list_online() if online else load_stock_list_mongodb()


def load_stock_list_online():
    """从在线数据中获取股票列表

    Notes:
        在线数据只取tdx的数据。

    Examples:

        >>> data_processor.load_stock_list_mongodb().head()
                    code  decimal_point  name    pre_close       sec sse  volunit
            code
            000001  000001              2  平安银行   648.020000  stock_cn  sz      100
            000002  000002              2  万 科Ａ    25.410000  stock_cn  sz      100
            000004  000004              2  国农科技    16.299999  stock_cn  sz      100
            000005  000005              2  世纪星源  3778.008125  stock_cn  sz      100
            000006  000006              2  深振业Ａ     5.270000  stock_cn  sz      100

    """
    df = QA.QAFetch.QATdx.QA_fetch_get_stock_list()
    df.index = df.index.droplevel(1)
    return df


def load_stock_list_mongodb():
    """从本地数据库中获取股票列表

    Returns:
        :py:class:`pandas.DataFrame`: 股票列表。

    Examples:

        >>> data_processor.load_stock_list_mongodb().head()
                      code  volunit  decimal_point  name    pre_close sse       sec
        code   sse
        000001 sz   000001      100              2  平安银行  1096.050000  sz  stock_cn
        000002 sz   000002      100              2  万 科Ａ    28.049999  sz  stock_cn
        000004 sz   000004      100              2  国农科技    21.410000  sz  stock_cn
        000005 sz   000005      100              2  世纪星源  4738.003750  sz  stock_cn
        000006 sz   000006      100              2  深振业Ａ     5.680000  sz  stock_cn

    """
    return QA.QA_fetch_stock_list_adv()


def load_stock_info_mongodb(code):
    """从数据库中获取股票信息

    Args:
        code (str): 股票代码

    Examples:
        >>> data_processor.load_stock_info_mongodb('601398').dtypes
        baoliu2               float64
        bgu                   float64
        changqifuzhai         float64
        code                   object
        cunhuo                float64
        faqirenfarengu        float64
        farengu               float64
        gudingzichan          float64
        gudongrenshu          float64
        guojiagu              float64
        hgu                   float64
        industry                int64
        ipo_date                int64
        jinglirun             float64
        jingyingxianjinliu    float64
        jingzichan            float64
        lirunzonghe           float64
        liudongfuzhai         float64
        liudongzichan         float64
        liutongguben          float64
        market                  int64
        meigujingzichan       float64
        province                int64
        shuihoulirun          float64
        touzishouyu           float64
        updated_date            int64
        weifenpeilirun        float64
        wuxingzichan          float64
        yingshouzhangkuan     float64
        yingyelirun           float64
        zhigonggu             float64
        zhuyinglirun          float64
        zhuyingshouru         float64
        zibengongjijin        float64
        zongguben             float64
        zongxianjinliu        float64
        zongzichan            float64
        dtype: object

    Returns:
        :py:class:`pandas.DataFrame`: 股票信息。

    """
    return QA.QA_fetch_stock_info(code)


def load_stock_info_online(code):
    """在线获取股票信息

    Notes:
        在线数据只取tdx的数据。

    Args:
        code (str): 股票代码

    Examples:
        >>> data_processor.load_stock_info_online('601398').dtypes
        market                  int64
        code                   object
        liutongguben          float64
        province                int64
        industry                int64
        updated_date            int64
        ipo_date                int64
        zongguben             float64
        guojiagu              float64
        faqirenfarengu        float64
        farengu               float64
        bgu                   float64
        hgu                   float64
        zhigonggu             float64
        zongzichan            float64
        liudongzichan         float64
        gudingzichan          float64
        wuxingzichan          float64
        gudongrenshu          float64
        liudongfuzhai         float64
        changqifuzhai         float64
        zibengongjijin        float64
        jingzichan            float64
        zhuyingshouru         float64
        zhuyinglirun          float64
        yingshouzhangkuan     float64
        yingyelirun           float64
        touzishouyu           float64
        jingyingxianjinliu    float64
        zongxianjinliu        float64
        cunhuo                float64
        lirunzonghe           float64
        shuihoulirun          float64
        jinglirun             float64
        weifenpeilirun        float64
        meigujingzichan       float64
        baoliu2               float64
        dtype: object

    Returns:
        :py:class:`pandas.DataFrame`: 股票信息。

    """
    return QA.QAFetch.QATdx.QA_fetch_get_stock_info(code)


def load_stock_block(**kwargs):
    """获取股票板块信息

    Examples:

        >>> df = data_processor.load_stock_block().loc['601398'].head()
        >>> df
               blockname    code type source
        code
        601398       环渤海  601398   gn    tdx
        601398       含H股  601398   gn    tdx
        601398     沪深300  601398   yb    tdx
        601398      融资融券  601398   yb    tdx
        601398       环渤海  601398   yb    tdx
    Args:
        online (bool, optional): 是否获取在线数据。默认为 ``False``。
    Returns:
        :py:class:`pandas.DataFrame`: 股票板块信息。
    """
    online = kwargs.pop('online', False)
    if online:
        return load_stock_block_online()
    else:
        return load_stock_block_mongodb()


def load_stock_block_mongodb():
    """从数据库获取股票板块信息

    Examples:

        >>> df = data_processor.load_stock_block_mongodb().loc['601398'].head()
        >>> set(df['type'].values)
        {'yb', 'zs', 'thshy', 'dy', 'zjhhy', 'fg', 'gn'}
        >>> set(df['source'].values)
        {'tdx', 'ths'}

    Returns:
        :py:class:`pandas.DataFrame`: 股票板块信息。返回数据内容参见 :py:func:`load_stock_block` 。

    """
    return QA.QA_fetch_stock_block()


def load_stock_block_online():
    """在线获取股票板块信息

    Notes:
        在线数据只取tdx的数据。

    Examples:

        >>> df = data_processor.load_stock_block_online().loc['601398'].head()
        >>> set(df['type'].values)
        {{'gn', 'zs', 'fg', 'yb'}
        >>> set(df['source'].values)
        {'tdx'}

    Returns:
        :py:class:`pandas.DataFrame`: 股票板块信息。返回数据内容参见 :py:func:`load_stock_block`

    """
    return QA.QAFetch.QATdx.QA_fetch_get_stock_block()


def load_deposit_rate_online(**kwargs):
    """在线获取历史存款利率

    See Also:
        http://tushare.org/macro.html#id2

    Examples:
        >>> df = data_processor.load_deposit_rate_online()
        >>> df.head()
        .                          rate
        date       deposit_type
        2015-10-24 定活两便(定期)        NaN
                   定期存款整存整取(半年)   1.30
                   定期存款整存整取(二年)   2.10
                   定期存款整存整取(三个月)  1.10
                   定期存款整存整取(三年)   2.75
        >>> df.xs('定期存款整存整取(一年)',axis=0,level=1).head()
                    rate
        date
        2015-10-24  1.50
        2015-08-26  1.75
        2015-06-28  2.00
        2015-05-11  2.25
        2015-03-01  2.50
        >>> df.index.get_level_values(1).unique().values
        ['定活两便(定期)' '定期存款整存整取(半年)' '定期存款整存整取(二年)' '定期存款整存整取(三个月)' '定期存款整存整取(三年)'
         '定期存款整存整取(五年)' '定期存款整存整取(一年)' '活期存款(不定期)' '零存整取、整存零取、存本取息定期存款(三年)'
         '零存整取、整存零取、存本取息定期存款(五年)' '零存整取、整存零取、存本取息定期存款(一年)' '通知存款(七天)' '通知存款(一天)'
         '协定存款(定期)']

    Returns:
        :py:class:`pandas.DataFrame`: 历史存款利率。
    """
    df = ts.get_deposit_rate().set_index(['date', 'deposit_type'])
    df['rate'] = pd.to_numeric(df['rate'], errors='coerce', downcast='float')
    return df


def load_loan_rate_online(**kwargs):
    """在线获取历史贷款利率

    See Also:
        http://tushare.org/macro.html#id3

    Examples:
        >>> df = data_processor.load_loan_rate_online()
        >>> df.head()
                                           rate
        date       loan_type
        2015-10-24 短期贷款(六个月以内)   4.35
                   短期贷款(六个月至一年)  4.35
                   中长期贷款(三至五年)   4.75
                   中长期贷款(五年以上)   4.90
                   中长期贷款(一至三年)   4.75
        >>> df.xs('短期贷款(六个月以内)',axis=0,level=1).head()
                    rate
        date
        2015-10-24  4.35
        2015-08-26  4.60
        2015-06-28  4.85
        2015-05-11  5.10
        2015-03-01  5.35
        >>> df.index.get_level_values(1).unique().values
        ['短期贷款(六个月以内)' '短期贷款(六个月至一年)' '中长期贷款(三至五年)' '中长期贷款(五年以上)' '中长期贷款(一至三年)'
         '贴现(贴现)' '优惠贷款(扶贫贴息贷款)' '优惠贷款(老少边穷发展经济贷款)' '优惠贷款(民政部门福利工厂贷款)'
         '优惠贷款(民族贸易及民族用品生产贷款)' '优惠贷款(贫困县办工业贷款)' '个人住房商业贷款(六个月以内)'
         '个人住房商业贷款(六个月至一年)' '个人住房商业贷款(三至五年)' '个人住房商业贷款(五年以上)' '个人住房商业贷款(一至三年)'
         '个人住房公积金贷款(五年以上)' '个人住房公积金贷款(五年以下)' '再贴现(再贴现率)' '优惠贷款(成套和高技术含量)'
         '优惠贷款(低技术含量和一般产品)' '优惠贷款(广深珠高速公路贷款)' '优惠贷款(特区、开发区差别利率贷款（五年以内）)'
         '优惠贷款(特区、开发区差别利率贷款（五年以上）)' '优惠贷款(银行系统印制企业基建储备贷款)' '优惠贷款(中国进出口银行出口卖方信贷)'
         '流动资产贷款(流动资产贷款利率)' '个人住房商业贷款(一至六个月（含六个月）)' '罚息(挤占挪用贷款)' '罚息(逾期贷款)'
         '特种贷款(特种贷款利率)']
    Returns:
        :py:class:`pandas.DataFrame`: 历史存款利率。
    """
    df = ts.get_loan_rate().set_index(['date', 'loan_type'])
    df['rate'] = pd.to_numeric(df['rate'], errors='coerce', downcast='float')
    return df


def load_cpi_online(**kwargs):
    """在线获取居民消费价格指数(CPI)

    Examples:
        >>> data_processor.load_cpi_online().head()
        .                cpi
        month
        2019.4   102.540001
        2019.3   102.279999
        2019.2   101.489998
        2019.1   101.739998
        2018.12  101.860001

    Returns:
        :py:class:`pandas.DataFrame`: 居民消费价格指数(CPI)。

    """
    df = ts.get_cpi().set_index('month')
    df['cpi'] = pd.to_numeric(df['cpi'], errors='coerce', downcast='float')
    return df


def load_money_supply_online(**kwargs):
    """在线获取货币供应量（月）

    Examples:
        >>> data_processor.load_money_supply_online().head()
        .                  m2  m2_yoy           m1  ...  sd_yoy          rests  rests_yoy
        month                                      ...
        2019.4   1884670.375     8.5  540614.6250  ...     NaN  202474.453125        NaN
        2019.3   1889412.125     8.6  547575.5625  ...     NaN  200214.984375        NaN
        2019.2   1867427.500     8.0  527190.5000  ...     NaN  209785.796875        NaN
        2019.1   1865935.375     8.4  545638.4375  ...     NaN  204300.187500        NaN
        2018.12  1826744.250     8.1  551685.9375  ...     NaN  213190.828125        NaN

    Returns:
        :py:class:`pandas.DataFrame`: 货币供应量。
        
        列名：
            * month :统计时间
            * m2 :货币和准货币（广义货币M2）(亿元)
            * m2_yoy:货币和准货币（广义货币M2）同比增长(%)
            * m1:货币(狭义货币M1)(亿元)
            * m1_yoy:货币(狭义货币M1)同比增长(%)
            * m0:流通中现金(M0)(亿元)
            * m0_yoy:流通中现金(M0)同比增长(%)
            * cd:活期存款(亿元)
            * cd_yoy:活期存款同比增长(%)
            * qm:准货币(亿元)
            * qm_yoy:准货币同比增长(%)
            * ftd:定期存款(亿元)
            * ftd_yoy:定期存款同比增长(%)
            * sd:储蓄存款(亿元)
            * sd_yoy:储蓄存款同比增长(%)
            * rests:其他存款(亿元)
            * rests_yoy:其他存款同比增长(%)
    """
    df = ts.get_money_supply().set_index('month')
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce', downcast='float')
    return df


def load_money_supply_year_online(**kwargs):
    """在线获取货币供应量（年底余额）

    Examples: data_processor.load_money_supply_year_online().head()
        >>>
        .               m2            m1  ...            sd          rests
        year                             ...
        2017  1690235.250  543790.12500  ...  649341.50000  176907.406250
        2016  1550066.750  486557.18750  ...  603504.18750  152015.593750
        2015  1392278.125  400953.40625  ...  552073.50000  151010.500000
        2014  1228374.750  348056.40625  ...  508878.09375  107384.601562
        2013  1106525.000  337291.09375  ...  467031.09375   69506.203125

    Returns:
        :py:class:`pandas.DataFrame`: 货币供应量。

        列名：
            * year :统计年度
            * m2 :货币和准货币(亿元)
            * m1:货币(亿元)
            * m0:流通中现金(亿元)
            * cd:活期存款(亿元)
            * qm:准货币(亿元)
            * ftd:定期存款(亿元)
            * sd:储蓄存款(亿元)
            * rests:其他存款(亿元)
    """
    df = ts.get_money_supply_bal().set_index('year')
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce', downcast='float')
    return df

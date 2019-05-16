"""数据读写器"""

from stock_ai import util
import datetime
import QUANTAXIS as QA
import socket
import pandas as pd
import os

_csv_encoding = 'utf-8'  #CSV文件默认字符编码


def _save_dataframe_to_csv(data, path, **kwargs):
    encoding = kwargs.pop('encoding', _csv_encoding)
    os.makedirs(path, exist_ok=True)
    data.to_csv(path, encoding=encoding)


def save_stock_daily_csv(data, path, **kwargs):
    """保存股票日线数据至csv文件。

    Args:
        data (pd.DataFrame): 数据表。
        path (str): 待保存的文件完整路径。
        encoding (str): 文件编码。默认为`utf-8`。参考 :py:func:`pandas.DataFrame.to_csv` 中同名参数。
    """
    _save_dataframe_to_csv(data, path, kwargs)


def load_stock_daily(code, **kwargs):
    """读取股票日线数据（在线或从数据库）

        Args:
            code (str): 股票代码。
            start (str, optional): 开始日期。数据格式为%Y-%m-%d。默认为`1990-01-01`。
            end (str, optional): 结束日期。数据格式为%Y-%m-%d。默认为`当天`。
            online (bool, optional): 是否获取在线数据。默认为 `False`。
            fq (str, optional): 是否取复权数据。默认为 `qfq`。
                * `qfq` - 前复权
                * `hfq` - 后复权
                * `bfq` or `None` - 不复权
            columns (tuple, optional): 获取的列集合。默认为
                ['open', 'high', 'low', 'close', 'volume', 'amount']

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
        code: 参考 :func:`load_stock_daily` 中参数 `start` 的说明。
        start: 参考 :func:`load_stock_daily` 中参数 `start` 的说明。
        end: 参考 :func:`load_stock_daily` 中参数 `start` 的说明。
        fq: 参考 :func:`load_stock_daily` 中参数 `start` 的说明。
        columns: 参考 :func:`load_stock_daily` 中参数 `start` 的说明。
        times (int, optional): 重试次数。默认为 `5`。
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
        index_col (int, sequence or bool, optional): 索引列。参考
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
        code: 参考 :func:`load_stock_daily` 中参数 `start` 的说明。
        start: 参考 :func:`load_stock_daily` 中参数 `start` 的说明。
        end: 参考 :func:`load_stock_daily` 中参数 `start` 的说明。
        fq: 参考 :func:`load_stock_daily` 中参数 `start` 的说明。
        columns: 参考 :func:`load_stock_daily` 中参数 `start` 的说明。

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
            start (str, optional): 开始日期。数据格式为%Y-%m-%d。默认为`1990-01-01`。
            end (str, optional): 结束日期。数据格式为%Y-%m-%d。默认为`当天`。
            online (bool, optional): 是否获取在线数据。默认为 `False`。
            columns (tuple, optional): 获取的列集合。默认为
                ['open', 'high', 'low', 'close', 'up_count', 'down_count', 'volume','amount']

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
        start (str):  参考 :func:`load_index_daily` 中参数 `start` 的说明。
        end (str): 参考 :func:`load_index_daily` 中参数 `end` 的说明。
        columns: 参考 :func:`load_index_daily` 中参数 `columns` 的说明。原始的列名中volume是vol，为了统一，内部会修改为volume。
        code (str): 参考 :func:`load_index_daily` 中参数 `code` 的说明。
        times (int, optional): 重试次数。默认为 `5`。
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
        start (str):  参考 :func:`load_index_daily` 中参数 `start` 的说明。
        end (str): 参考 :func:`load_index_daily` 中参数 `end` 的说明。
        columns: 参考 :func:`load_index_daily` 中参数 `columns` 的说明。
        code (str): 参考 :func:`load_index_daily` 中参数 `code` 的说明。
    Returns:
        :py:class:`pandas.DataFrame`: 指数日线数据。
    """
    d = QA.QA_fetch_index_day_adv(code, start, end)
    df = d.data.reset_index().drop(columns=['code']).set_index('date')
    return df[columns]


def load_stock_list(online=False):
    """获取股票列表

    Args:
        online (bool): 是否获取在线数据。默认为False。

    Returns:
        :py:class:`pandas.DataFrame`: 股票列表。
    """
    return load_stock_list_online() if online else load_stock_list_mongodb()


def load_stock_list_online():
    """从在线数据中获取股票列表。

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
    """从本地数据库中获取股票列表。

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
        online (bool, optional): 是否获取在线数据。默认为 `False`。
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
        :py:class:`pandas.DataFrame`: 股票板块信息。返回数据内容参见 :py:func:`load_stock_block`

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

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

    Args:
        start: See parameter ``start`` in :func:`fetch_stock`.
        end:
        columns:
        code (str): 股票代码。
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
        code:
        start:
        end:
        fq:
        columns:

    Returns:

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

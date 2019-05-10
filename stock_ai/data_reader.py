from stock_ai import util
import datetime
import QUANTAXIS as QA
import socket
import pandas as pd


class DailyReader(object):

    def fetch_stock(self, code, **kwargs):
        """

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
            **kwargs:

        Returns:

        """
        start = kwargs.pop('start', '1990-01-01')
        end = kwargs.pop('end', util.date2str(datetime.datetime.now()))
        online = kwargs.pop('online', False)
        fq = kwargs.pop('fq', 'qfq')
        columns = kwargs.pop(
            'columns', ['open', 'high', 'low', 'close', 'volume', 'amount'])
        if online:
            return self._fetch_stock_online(code, start, end, columns)
        else:
            return self._fetch_stock_mongodb(code, start, end, fq, columns)

    def _fetch_stock_online(self, code, start, end, columns, times=5):
        """读取股票在线日线数据

        Args:
            start: 
            end: 
            columns: 
            code (str): 股票代码。
            times (int, optional): 重试次数。默认为 `5`。
        """
        retries = 0
        while True:
            try:
                df = QA.QAFetch.QATdx.QA_fetch_get_stock_day(code, start, end)
                # 原始列名['open', 'close', 'high', 'low', 'vol', 'amount', 'code', 'date', 'date_stamp']
                df = df.rename(columns={'vol': 'volume'})[columns]
                return df
            except socket.timeout:
                if retries < times:
                    retries = retries + 1
                    continue
                raise

    def _fetch_stock_mongodb(self, code, start, end, fq, columns):
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

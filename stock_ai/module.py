from . import data_processor as dp
from . import util
from . import calcs
import numpy as np
import pandas as pd
import datetime


class Stock():
    """股票类型定义"""

    def __init__(self, code: str):
        self._code = code

    @property
    def code(self):
        """股票代码。

        Returns:
            str:
        """
        return self._code

    def get_daily(self, start=None, end=None):
        """获取日线数据。未实现该方法。在派生类中实现。"""
        pass


class StockCN(Stock):
    """中国股票类型定义"""

    def __init__(self, code, **kwargs):
        """构造函数

        Args:
            code (str): 股票代码。
            getinfo_online (bool): 是否从在线获取股票信息数据。默认为 False。
            getdaily_online (bool): 是否从在线获取股票日线数据。默认为 False。
            getblock_online (bool): 是否从在线获取股票日线数据。默认为 False。
        """
        super(StockCN, self).__init__(code)
        self.getinfo_online = kwargs.pop('getinfo_online', False)
        self.getdaily_online = kwargs.pop('getdaily_online', False)
        self.getblock_online = kwargs.pop('getblock_online', False)
        self._info = pd.DataFrame()
        self._simple_info = pd.Series()
        self._daily = pd.DataFrame()
        self._block = pd.DataFrame()

    def _get_simple_info(self) -> pd.Series:
        """根据 getinfo_online 属性确定是否从在线获取数据"""
        lst = dp.load_stock_list(self.getinfo_online)
        return lst.loc[self.code]

    def _get_info(self) -> pd.DataFrame:
        """根据 getinfo_online 属性确定是否从在线获取数据"""
        if self.getinfo_online:
            return dp.load_stock_info_mongodb(self.code)
        else:
            return dp.load_stock_info_online(self.code)

    @property
    def simple_info(self):
        """当前股票简单信息。

        获取数据来源参考 :py:attr:`_get_info`。

        Returns:
            :class:`~pandas.Series`: 当前股票简单信息。
        """
        if self._simple_info.empty:
            self._simple_info = self._get_simple_info()
        return self._simple_info

    @property
    def info(self):
        """当前股票信息。

        获取数据来源参考 :py:attr:`_get_info`。

        Returns:
            :class:`~pandas.DataFrame`: 当前股票信息。
        """
        if self._info.empty:
            self._info = self._get_info()
        return self._info

    @property
    def name(self):
        """股票名称。

        Returns:
            str: 股票名称。
        """
        return self.info['name'][
            0] if 'name' in self.info.columns else self.simple_info['name']

    @property
    def exchange(self):
        """所属交易所。

        Returns:
            str: 所属交易所。（返回``sh``或``sz``）
        """
        return self.simple_info['sse']

    @property
    def ipo_date(self):
        """当前股票ipo日期。

        数据获取参考 :py:attr:`info`。

        Returns:
            datetime.datetime: 当前股票ipo日期。
        """
        return util.str2date(util.int2str(self.info['ipo_date'][0]))

    def get_cum_returns(self, **kwargs):
        """获取指定日期间的累计收益。

        Args:
            start (str or datetime.datetime): 开始时间。参考 :func:`get_daily`。
            end (str or datetime.datetime): 开始时间。参考 :func:`get_daily`。
            column (str): 计算列。默认为 close。
        """
        start = kwargs.pop('start', None)
        end = kwargs.pop('end', None)
        column = kwargs.pop('column', 'close')
        daily = self.get_daily(start, end).copy()
        return calcs.cum_return(daily[column])

    def get_sharpe_ratio(self, **kwargs):
        """计算指定时间段的夏普比率。

        Args:
            start (str or datetime.datetime): 开始时间。参考 :func:`get_daily`。
            end (str or datetime.datetime): 开始时间。参考 :func:`get_daily`。
            如果传入字符串，需要满足 :func:`stock_ai.util.str2date` 方法要求。
            rf (float): 无风险利率。默认为 1.0。
            split (str): 分割规则。包含 year,none。none为不分割。默认为 year。
            column (str): 计算列。默认为 close。

        Examples:
            >>> StockCN('601398').get_sharpe_ratio()
                          0
            2006   3.269645
            2007   3.744622
            2008   3.136494
            2009   4.293783
            2010  10.313356
            2011  22.193504
            2012  15.875993
            2013  25.002890
            2014   6.332456
            2015  10.798641
            2016  14.663031
            2017   6.170423
            2018   9.429071
            2019  44.551702

        Returns:
            :class:`~pandas.DataFrame`: 夏普比率。
            数据参考 :func:`stock_ai.calcs import sharpe_ratio`
        """
        start = kwargs.pop('start', None)
        end = kwargs.pop('end', None)
        rf = kwargs.pop('rf', 1.0)
        column = kwargs.pop('column', 'close')
        split = kwargs.pop('split', 'year')
        daily = self.get_daily(start, end).copy()
        result = {}
        if split and split == 'year':
            daily['year'] = calcs.calc_year(daily)
            for year in daily['year'].unique():
                df = daily[daily['year'] == year]
                result[year] = calcs.sharpe_ratio(df[column], rf)
        elif not split:
            return calcs.sharpe_ratio(daily[column], rf)
        return pd.DataFrame.from_dict(result, orient='index')

    def get_daily(self, start=None, end=None):
        """获取日线数据。

        当前实例会缓存所有的日线数据。再从缓存中提取指定日期的数据作为返回值。

        Args:
            start (str): 开始日期。
            end (str): 结束日期。

        Returns:
            :class:`~pandas.DataFrame`: 日线数据。
            如果 `start` 和 `end` 相同，则返回一日数据。
            数据参考 :func:`stock_ai.data_processor.load_stock_daily`

        """
        if self._daily.empty:
            self._daily = dp.load_stock_daily(self.code,
                                              online=self.getdaily_online)
        if start and end:
            return self._daily.loc[slice(pd.Timestamp(start),
                                         pd.Timestamp(end))]
        elif start:
            return self._daily.loc[pd.Timestamp(start):]
        elif end:
            return self._daily.loc[:pd.Timestamp(end)]
        else:
            return self._daily

    @property
    def related_codes(self):
        """获取同板块其他股票代码

        根据板块 :func:`stock_ai.data_processor.load_stock_block` 中 'type'=='thshy' 的板块获取。

        Examples:
            >>> self.related_codes
            ('601169', '601988', '600926', '601939', '600036', '601328', '601229', '601818', '601288', '600919', '600016', '601009', '601998', '601128', '601166', '002142', '600000', '000001', '600015', '601997', '002839', '002807', '600908', '603323', '601838')

        Returns:
            tuple(str): 代码列表。
        """
        b = 'blockname'
        thshy = self.block[self.block['type'] == 'thshy'][b]
        if not thshy.empty:
            codes = self._block[(self._block[b] == thshy.values[0]) &
                                (self._block['code'] != self.code)]
            return tuple(codes['code'].unique())
        return ()

    @property
    def block(self):
        """股票所属板块。

        Returns:
            :class:`~pandas.DataFrame`: 股票板块信息。
            数据参考 :func:`stock_ai.data_processor.load_stock_block`
        """
        if self._block.empty:
            self._block = dp.load_stock_block(online=self.getblock_online)
        return self._block.loc[self.code]


class Index():
    """指数类型定义"""

    def __init__(self, code: str):
        self._code = code

    @property
    def code(self):
        """指数代码。

        Returns:
            str:
        """
        return self._code

    def get_daily(self, start=None, end=None):
        """获取日线数据。未实现该方法。在派生类中实现。"""
        pass


class IndexCN(Index):
    """中国指数类型定义"""

    def __init__(self, code, **kwargs):
        """构造函数

        Args:
            code (str): 指数代码。
            getdaily_online (bool): 是否从在线获取股票日线数据。默认为 False。
        """
        super(IndexCN, self).__init__(code)
        self.getdaily_online = kwargs.pop('getdaily_online', False)
        self._daily = pd.DataFrame()

    def get_daily(self, start=None, end=None):
        """获取日线数据。

        当前实例会缓存所有的日线数据。再从缓存中提取指定日期的数据作为返回值。

        Args:
            start (str): 开始日期。
            end (str): 结束日期。

        Returns:
            :class:`~pandas.DataFrame`: 日线数据。
            如果 `start` 和 `end` 相同，则返回一日数据。
            数据参考 :func:`stock_ai.data_processor.load_index_daily`

        """
        if self._daily.empty:
            self._daily = dp.load_index_daily(self.code,
                                              online=self.getdaily_online)
        if start and end:
            return self._daily.loc[slice(pd.Timestamp(start),
                                         pd.Timestamp(end))]
        elif start:
            return self._daily.loc[pd.Timestamp(start):]
        elif end:
            return self._daily.loc[:pd.Timestamp(end)]
        else:
            return self._daily

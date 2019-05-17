from . import data_processor as dp
from . import util
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

import datetime

import pytest

from stock_ai import util
from stock_ai.module import StockCN
from test import is_travis


@pytest.mark.skipif(is_travis, reason="Skipping this test on Travis CI.")
def test_get_info_mongodb():
    """测试默认数据"""
    s = StockCN('601398')
    _test_info(s)


def _test_info(stock: StockCN):
    """测试默认数据"""
    assert stock.code == '601398'
    assert stock.name == '工商银行'
    assert stock.exchange == 'sh'
    assert util.date2str(stock.ipo_date) == '2006-10-27'
    assert isinstance(stock.ipo_date, datetime.datetime)


def test_stockcn_construct():
    """测试构造函数"""
    s = StockCN('601398')
    _test_info(s)
    assert not s.getblock_online
    assert not s.getdaily_online
    assert not s.getinfo_online


def test_get_info_online():
    s = StockCN('601398', getinfo_online=True)
    _test_info(s)


def test_getdaily_online():
    s = StockCN('601398', getdaily_online=True)
    d1 = s.get_daily()
    assert not d1.empty
    d2 = s.get_daily(start='2010-01-04')
    assert d1.iloc[-1].equals(d2.iloc[-1])
    assert not d2.empty
    d3 = s.get_daily(end='2010-01-04')
    assert not d3.empty
    assert d1.iloc[0].equals(d3.iloc[0])
    assert d2.iloc[0].equals(d3.iloc[-1])
    d4 = s.get_daily(start='2010-01-04', end='2010-01-04')
    assert not d4.empty
    d4 = s.get_daily(start='2010-01-01', end='2010-01-01')
    assert d4.empty


def test_getblock_online():
    s = StockCN('601398', getblock_online=is_travis)
    print(s.block)
    assert not s.block.empty


def test_getrelated_codes():
    s = StockCN('601398', getblock_online=is_travis)
    print(s.related_codes)
    assert len(s.related_codes) > 0
    assert '601398' not in s.related_codes
    assert '601939' in s.related_codes


def test_get_sharpe_ratio():
    print(StockCN('601398', getdaily_online=is_travis).get_sharpe_ratio())

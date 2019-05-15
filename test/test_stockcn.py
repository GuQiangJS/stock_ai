from stock_ai.module import StockCN
import pytest
import os
import datetime
from stock_ai import util


@pytest.mark.skipif("TRAVIS" in os.environ and os.environ["TRAVIS"] == "true",
                    reason="Skipping this test on Travis CI.")
def test_get_info_mongodb():
    s = StockCN('601398')
    assert not s.info.empty


@pytest.mark.skipif("TRAVIS" in os.environ and os.environ["TRAVIS"] == "true",
                    reason="Skipping this test on Travis CI.")
def test_get_info_mongodb():
    s = StockCN('601398')
    assert s.ipo_date
    assert isinstance(s.ipo_date, datetime.datetime)
    assert util.date2str(s.ipo_date) == '2006-10-27'


def test_stockcn_construct():
    s = StockCN('601398')
    assert not s.getinfo_online
    s.getinfo_online = True
    assert s.getinfo_online


def test_get_info_online():
    s = StockCN('601398', getinfo_online=True)
    assert s.getinfo_online
    assert not s.info.empty


def test_get_info_online():
    s = StockCN('601398', getinfo_online=True)
    assert s.ipo_date
    assert isinstance(s.ipo_date, datetime.datetime)
    assert util.date2str(s.ipo_date) == '2006-10-27'


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
    d4 = s.get_daily(start='2010-01-04',end='2010-01-04')
    assert not d4.empty
    d4 = s.get_daily(start='2010-01-01',end='2010-01-01')
    assert d4.empty

def test_getblock_online():
    s = StockCN('601398', getblock_online=True)
    b = s.get_block()
    assert not b.empty
    print(b.head())

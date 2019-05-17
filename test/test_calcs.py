import pytest
import pandas as pd
from stock_ai import data_processor
from stock_ai import calcs
from stock_ai import wrapper
import numpy as np
from test import get_index_daily
from test import get_stock_daily


def _assert(func, col):
    df = get_stock_daily().copy()
    df['col'] = func(df)
    print(df['col'].head())
    assert np.array_equal(df['col'], col)


def test_append_year():
    _assert(calcs.calc_year, get_stock_daily().index.year)


def test_append_month():
    _assert(calcs.calc_month, get_stock_daily().index.month)


def test_is_trade_suspension():
    df = wrapper.dataframe_merge(get_stock_daily(), get_index_daily())
    is_sus = calcs.is_trade_suspension(df)
    assert not is_sus['2012-02-22']
    assert is_sus['2012-02-23']
    assert not is_sus['2012-02-24']
    print(is_sus['2012-02-23'])


def test_rolling_mean():
    df = wrapper.dataframe_merge(get_stock_daily(), get_index_daily())
    days = 7
    rm = calcs.tech_ma(df, days=days)
    print(rm.tail())
    rm1 = df['close'].rolling(window=days).mean()
    pd.Series.equals(rm, rm1)

def test_macd_series():
    df = wrapper.dataframe_merge(get_stock_daily(), get_index_daily())
    macd = calcs.tech_macd(df)
    print(macd.tail())


def test_daily_return():
    df = wrapper.dataframe_merge(get_stock_daily(), get_index_daily())
    dr = calcs.daily_return(df)
    print(dr.tail())

def test_sharpe_ratio():
    df = wrapper.dataframe_merge(get_stock_daily(), get_index_daily())
    dr = calcs.sharpe_ratio(df['close'].dropna(),1.0)
    print(dr)
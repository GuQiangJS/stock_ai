import pytest
import pandas as pd
from stock_ai import data_processor
from stock_ai import calcs
from stock_ai import wrapper
from stock_ai.util import str2date
import numpy as np
from test import get_index_daily
from test import get_stock_daily
from test import get_deposit_rate
import logging
import test


def _assert(func, col):
    df = get_stock_daily().copy()
    df['col'] = func(df)
    print(df['col'].head())
    assert np.array_equal(df['col'], col)


def test_tech_bbands():
    df = get_stock_daily().copy()
    print(calcs.tech_bbands(df).tail())


def test_tech_ema():
    df = get_stock_daily().copy()
    print(calcs.tech_ema(df).tail())


def test_tech_ma():
    df = get_stock_daily().copy()
    print(calcs.tech_ma(df).tail())


def test_fft():
    df = get_stock_daily().copy()
    fft = calcs.fft(df)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(14, 7), dpi=100)
    plt.plot(np.fft.ifft(fft))
    plt.plot(df['close'])
    plt.show()
    print(fft)


def test_append_year():
    _assert(calcs.calc_year, get_stock_daily().index.year)


def test_calc_year():
    dates = pd.date_range('20130101', periods=3, freq='Y')
    df = pd.DataFrame([1, 3, 5], index=dates, columns=list('A'))
    dx=pd.DatetimeIndex(['2013-12-31', '2014-12-31', '2015-12-31'], dtype='datetime64[ns]')
    dx.equals(df.index)
    print(df)
    print(calcs.calc_year(df).values)

def test_append_month():
    _assert(calcs.calc_month, get_stock_daily().index.month)


def test_is_trade_suspension():
    df = data_processor.merge({
            test.stock_code: get_stock_daily(),
            test.index_code: get_index_daily()
        },how='right')
    is_sus = calcs.is_trade_suspension(df)
    assert not is_sus['2012-02-22']
    assert is_sus['2012-02-23']
    assert not is_sus['2012-02-24']
    print(is_sus['2012-02-23'])


def test_rolling_mean():
    df = test.merged_dataframe()
    days = 7
    rm = calcs.tech_ma(df, days=days)
    print(rm.tail())
    rm1 = df['close'].rolling(window=days).mean()
    pd.Series.equals(rm, rm1)


def test_macd_series():
    df = test.merged_dataframe()
    macd = calcs.tech_macd(df)
    print(macd.tail())


def test_daily_return():
    df = test.merged_dataframe()
    dr = calcs.daily_return(df['close'])
    print(dr.tail())


def test_cum_return():
    df = test.merged_dataframe()
    dr = calcs.cum_return(df['close'].dropna())
    print(dr.tail())


def test_sharpe_ratio():
    rate = get_deposit_rate().sort_index()
    df_s = get_stock_daily().index[0].date()
    df_e = get_stock_daily().index[-1].date()
    for i in range(len(rate.index)):
        s = str2date(rate.index[i]).date()
        e = str2date(rate.index[i + 1]).date() if rate[i] != rate[-1] else df_e
        if s >= df_s and e <= df_e:
            df = data_processor.merge({
                test.index_code: get_index_daily().loc[s:e],
                test.stock_code: get_stock_daily().loc[s:e]
            })
            k = '{0}~{1}'.format(s, e)
            v = calcs.sharpe_ratio(df['close'].dropna(), rate[i])
            print('{0}:{1}'.format(k, v))


def test_kurtosis():
    logging.debug("测试峰度")
    for i in range(2010, 2018):
        s = '{0}-01-01'.format(i)
        e = '{0}-12-31'.format(i)
        df = data_processor.merge({
            test.index_code: get_index_daily().loc[s:e],
            test.stock_code: get_stock_daily().loc[s:e]
        })
        v = calcs.kurtosis(df['close'].dropna())
        print('{0}:{1}'.format(i, v))


def test_skew():
    logging.debug("测试偏度")
    for i in range(2010, 2018):
        s = '{0}-01-01'.format(i)
        e = '{0}-12-31'.format(i)
        df = data_processor.merge({
            test.index_code: get_index_daily().loc[s:e],
            test.stock_code: get_stock_daily().loc[s:e]
        })
        v = calcs.skew(df['close'].dropna())
        print('{0}:{1}'.format(i, v))

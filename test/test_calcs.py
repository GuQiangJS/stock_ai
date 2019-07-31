import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

import test
from stock_ai import calcs
from stock_ai import data_processor
from stock_ai.util import str2date
from test import get_deposit_rate
from test import get_index_daily
from test import get_stock_daily


def _assert(func, col):
    df = get_stock_daily().copy()
    df['col'] = func(df)
    print(df['col'].head())
    assert np.array_equal(df['col'], col)


def test_onehot_encode():
    lst1 = [[2016], [2017], [2018], [2018], [2019]]
    n1 = OneHotEncoder().fit_transform(lst1).toarray()
    lst2 = [2016, 2017, 2018, 2018, 2019]
    lst3 = np.reshape(lst2, (len(lst2), 1))
    n2 = OneHotEncoder().fit_transform(lst3).toarray()
    assert np.array_equal(n1, n2)
    n4 = calcs.trans_onehot(lst1)
    n5 = calcs.trans_onehot(lst2)
    assert np.array_equal(n1, n4)
    assert np.array_equal(n1, n5)
    print(n4)
    print(n5)


def test_reshape():
    lst = [2016, 2017, 2018, 2019]
    np_arr = np.array(lst)
    print(np_arr.shape)
    n = np.reshape(np_arr, (4, 1))
    print(n.shape)
    print(n)


def test_calc_year():
    dates = pd.date_range('20130101', periods=3, freq='Y')
    df = pd.DataFrame([1, 3, 5], index=dates, columns=list('A'))
    dx = pd.DatetimeIndex(['2013-12-31', '2014-12-31', '2015-12-31'],
                          dtype='datetime64[ns]')
    dx.equals(df.index)
    print(df)
    print(calcs.calc_year(df).values)


def test_append_month():
    _assert(calcs.calc_month, get_stock_daily().index.month)


def test_is_trade_suspension():
    """测试是否停牌的计算"""
    df = data_processor.merge(
        {
            test.stock_code: get_stock_daily(),
            test.index_code: get_index_daily()
        },
        how='right')
    is_sus = calcs.is_trade_suspension(df)
    assert not is_sus['2012-02-22']
    assert is_sus['2012-02-23']
    assert not is_sus['2012-02-24']
    print(is_sus['2012-02-23'])


def test_daily_return():
    df = test.merged_dataframe()
    dr = calcs.calc_daily_return(df['close'])
    print(dr.tail())


def test_cum_return():
    df = test.merged_dataframe()
    dr = calcs.calc_cum_return(df['close'].dropna())
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
                test.index_code:
                    get_index_daily().loc[s:e],
                test.stock_code:
                    get_stock_daily().loc[s:e]
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

import test
from stock_ai import data_processor
import pytest
import os
import numpy as np
import pprint


def test_load_stock_daily_csv():
    code = '601398'
    df_csv = test.read_stock_daily_from_csv(code)
    assert not df_csv.empty
    first_line = df_csv.iloc[0]
    last_line = df_csv.iloc[-1]
    assert 1.9921332735028636 == first_line['open']
    assert 484066272.0 == last_line['amount']


def test_load_stock_daily_online():
    code = '601398'
    end = '2019-01-24'
    online_columns = ['open', 'high', 'low', 'close', 'volume', 'amount']
    df_online1 = data_processor.load_stock_daily_online(code,
                                                        '1990-01-01',
                                                        end=end,
                                                        columns=online_columns)
    assert df_online1 is not None
    assert not df_online1.empty
    df_online2 = data_processor.load_stock_daily(code, end=end, online=True)
    assert df_online2 is not None
    assert not df_online2.empty
    assert df_online1.equals(df_online2)
    print(df_online2.head())


@pytest.mark.skipif("TRAVIS" in os.environ and os.environ["TRAVIS"] == "true",
                    reason="Skipping this test on Travis CI.")
def test_load_stock_list_mongo():
    """测试从数据库中读取股票列表"""
    df = data_processor.load_stock_list_mongodb()
    assert not df.empty
    for name in [
            'code', 'decimal_point', 'name', 'pre_close', 'sec', 'sse',
            'volunit'
    ]:
        assert name in df.columns


def test_load_stock_list_online():
    """测试在线数据中读取股票列表"""
    df = data_processor.load_stock_list_online()
    print(df.head())
    assert not df.empty
    for name in [
            'code', 'decimal_point', 'name', 'pre_close', 'sec', 'sse',
            'volunit'
    ]:
        assert name in df.columns


@pytest.mark.skipif("TRAVIS" in os.environ and os.environ["TRAVIS"] == "true",
                    reason="Skipping this test on Travis CI.")
def test_load_stock_list():
    df1 = data_processor.load_stock_list_mongodb()
    df2 = data_processor.load_stock_list(online=False)
    df1.equals(df2)
    df1 = data_processor.load_stock_list_online()
    df2 = data_processor.load_stock_list(online=True)
    df1.equals(df2)


def test_load_stock_info_online():
    code = '601398'
    df = data_processor.load_stock_info_online(code)
    print(df.dtypes)


@pytest.mark.skipif("TRAVIS" in os.environ and os.environ["TRAVIS"] == "true",
                    reason="Skipping this test on Travis CI.")
def test_load_stock_info_mongo():
    code = '601398'
    df = data_processor.load_stock_info_mongodb(code)
    print(df.dtypes)


def test_load_stock_block_online():
    df = data_processor.load_stock_block_online()
    d1 = df.loc['601398']
    print(set(d1['type'].values))
    print(set(d1['source'].values))
    print(df.loc['601398'].head())


@pytest.mark.skipif("TRAVIS" in os.environ and os.environ["TRAVIS"] == "true",
                    reason="Skipping this test on Travis CI.")
def test_load_stock_block_mongodb():
    df = data_processor.load_stock_block_mongodb()
    d1 = df.loc['601398']
    print(set(d1['type'].values))
    print(set(d1['source'].values))
    print(df.loc['601398'].head())


def test_load_deposit_rate_online():
    """测试在线获取存款利率"""
    df = data_processor.load_deposit_rate_online()
    print(df.head())
    print(df.index.get_level_values(1).unique().values)
    print(df.xs('定期存款整存整取(一年)', axis=0, level=1).head())
    assert str(df.loc['2015-03-01', '定期存款整存整取(一年)']['rate']) == str(2.5)


def test_load_loan_rate_online():
    """测试在线获取贷款利率"""
    df = data_processor.load_loan_rate_online()
    print(df.head())
    print(df.index.get_level_values(1).unique().values)
    print(df.xs('短期贷款(六个月以内)', axis=0, level=1).head())
    assert str(df.loc['2014-11-22', '短期贷款(六个月以内)']['rate']) == str(5.6)


def test_load_cpi_online():
    """测试在线获取居民消费价格指数(CPI)"""
    df = data_processor.load_cpi_online()
    print(df.head())
    assert str(df.loc['2014.11']['cpi']) == str(101.44)


def test_load_money_supply_online():
    """测试在线获取货币供应量（月）"""
    df = data_processor.load_money_supply_online()
    print(df.head())
    print(df.dtypes)
    v = df.loc['2019.2']
    assert str(v['m2']) == str(1867427.5)
    assert str(v['m1']) == str(527190.5)


def test_load_money_supply_year_online():
    """测试在线获取货币供应量（年）"""
    df = data_processor.load_money_supply_year_online()
    print(df.head())
    print(df.dtypes)
    v = df.loc['2013']
    assert v['m2'] == 1106525

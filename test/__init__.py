import os
import pandas as pd
from stock_ai import data_processor

test_stocks = ['601398']


def _current_path():
    return os.path.abspath(os.path.dirname(__file__))


def _get_stock_daily_filepath(code):
    """根据股票代码，获取股票日线数据文件保存路径"""
    return os.path.join(_current_path(), 'data', 'daily_stock',
                        "{0}.csv".format(code))


def read_stock_daily_from_csv(code)->pd.DataFrame:
    filepath = _get_stock_daily_filepath(code)
    if not os.path.exists(filepath):
        raise FileNotFoundError(filepath)
    return data_processor.load_stock_daily_csv(filepath)


def init_stock_daily_data():
    """在本地构建默认的测试用数据"""
    for code in test_stocks:
        filepath = _get_stock_daily_filepath(code)
        if not os.path.exists(filepath):
            df = data_processor.load_stock_daily_online(code)
            data_processor.save_stock_daily_csv(df, filepath)


if __name__ == '__main__':
    init_stock_daily_data()

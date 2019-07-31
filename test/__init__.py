import logging
import os
import warnings

import pandas as pd

from stock_ai import data_processor

warnings.filterwarnings("ignore")

is_travis = "TRAVIS" in os.environ and os.environ["TRAVIS"] == "true"
stock_code = '601398'
index_code = '399300'

__cache = {}  # 日线数据缓存缓存


def merged_dataframe() -> dict:
    return data_processor.merge({
        index_code: get_index_daily(),
        stock_code: get_stock_daily()
    })


def get_deposit_rate(n='dq1y') -> pd.Series:
    """获取存款利率"""
    if n == 'hq':
        n = '活期存款(不定期)'
    elif n == 'dq1y':
        n = '定期存款整存整取(一年)'
    else:
        raise ValueError
    if __cache and 'deposit_rate' in __cache:
        r = __cache['deposit_rate']
    else:
        r = data_processor.load_deposit_rate_online()
        __cache['deposit_rate'] = r
        logging.debug("Load deposit_rate_online.")
    return r.xs(n, axis=0, level=1)['rate']


def get_stock_daily(code: str = stock_code) -> pd.DataFrame:
    if __cache and code in __cache:
        return __cache[code]
    df = data_processor.load_stock_daily(code, online=is_travis, fq=None)
    logging.debug("Load Daily:" + code)
    __cache[code] = df
    return df


def get_index_daily(code: str = index_code) -> pd.DataFrame:
    if __cache and code in __cache:
        return __cache[code]
    df = data_processor.load_index_daily(code, online=is_travis)
    logging.debug("Load Daily:" + code)
    __cache[code] = df
    return df


def test_keras():
    import keras
    keras.models.model_from_json()

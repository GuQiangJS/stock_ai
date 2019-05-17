import os
from stock_ai import data_processor
import pandas as pd
import logging

is_travis = "TRAVIS" in os.environ and os.environ["TRAVIS"] == "true"
stock_code = '601398'
index_code = '399300'

_daily_dfs = {}  #日线数据缓存缓存


def get_stock_daily(code: str = stock_code) -> pd.DataFrame:
    if _daily_dfs and code in _daily_dfs:
        return _daily_dfs[code]
    df = data_processor.load_stock_daily(code, online=is_travis, fq=None)
    logging.debug("Load Daily:" + code)
    _daily_dfs[code] = df
    return df


def get_index_daily(code: str = index_code) -> pd.DataFrame:
    if _daily_dfs and code in _daily_dfs:
        return _daily_dfs[code]
    df = data_processor.load_index_daily(code, online=is_travis)
    logging.debug("Load Daily:" + code)
    _daily_dfs[code] = df
    return df

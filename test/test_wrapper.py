import pytest
from stock_ai import wrapper
from stock_ai import data_processor
import numpy as np
import pandas as pd
from stock_ai import appender


def test_stock_index_merge_online():
    """测试默认的stock_index_merge。不包含任何参数"""
    df_1 = data_processor.load_stock_daily('601398')
    df_2 = data_processor.load_index_daily('399300')
    df = wrapper.stock_index_merge(df_1, df_2)
    assert np.isnan(df.iloc[0]['open'])
    assert not np.isnan(df.loc[df_1.index[0]]['open'])
    _assert_columns_in_dataframe(df, df_1.columns)
    for col in df_2.columns:
        assert col + '_index' if col in df_1 else col in df.columns

def _assert_columns_in_dataframe(df: pd.DataFrame, cols):
    for col in cols:
        assert col in df.columns


def test_stock_index_merge_online_appendfuncs():
    """测试stock_index_merge。附加列。"""
    funcs = {'year': appender.append_year,
             'month': appender.append_month}
    df_1 = data_processor.load_stock_daily('601398')
    df_2 = data_processor.load_index_daily('399300')
    df = wrapper.stock_index_merge(df_1, df_2, append_funcs=funcs)
    _assert_columns_in_dataframe(df, funcs.keys())
    print(df[funcs.keys()].head())

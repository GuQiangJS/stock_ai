import pytest
from stock_ai import wrapper
from stock_ai import data_processor
import numpy as np
import pandas as pd
from stock_ai import calcs
from test import get_stock_daily
from test import get_index_daily


def test_dataframe_merge():
    """测试默认的stock_index_merge。不包含任何参数"""
    df = wrapper.dataframe_merge(get_stock_daily(), get_index_daily())
    assert np.isnan(df.iloc[0]['open'])
    assert not np.isnan(df.loc[get_stock_daily().index[0]]['open'])
    _assert_columns_in_dataframe(df, get_stock_daily().columns)
    for col in get_index_daily().columns:
        assert col + '_index' if col in get_stock_daily() else col in df.columns


def _assert_columns_in_dataframe(df: pd.DataFrame, cols):
    for col in cols:
        assert col in df.columns


def test_stock_index_merge_appendfuncs():
    """测试stock_index_merge。附加列。"""
    funcs = {'year': calcs.calc_year, 'month': calcs.calc_month}
    df = wrapper.dataframe_merge(get_stock_daily(),
                                 get_index_daily(),
                                 append_funcs=funcs)
    _assert_columns_in_dataframe(df, funcs.keys())
    print(df[funcs.keys()].head())


def test_stock_index_merge_appendfuncs_has_params():

    def app(df, **kwargs):
        v = kwargs.pop('v', '123')
        return pd.Series(v, index=df.index)

    new_col = 'cv'
    new_value='123'
    df = wrapper.dataframe_merge(get_stock_daily(),
                                 get_index_daily(),
                                 append_funcs={new_col: [app, {
                                     'v': new_value
                                 }]})
    print(df[new_col].head())
    v = df[new_col].unique()
    assert len(v) == 1
    assert v[0] == new_value

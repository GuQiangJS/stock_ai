import pytest
import pandas as pd
from stock_ai import data_processor
from stock_ai import appender
import numpy as np

test_df = pd.DataFrame()


def setup_module(module):
    global test_df
    print('Load Data...')
    test_df = data_processor.load_stock_daily('601398')

def _assert(func,col):
    df = test_df.copy()
    df['col'] = func(df)
    print(df['col'].head())
    assert np.array_equal(df['col'], col)

def test_append_year():
    _assert(appender.append_year,test_df.index.year)

def test_append_month():
    _assert(appender.append_month,test_df.index.month)

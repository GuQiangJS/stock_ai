import pytest
from stock_ai import preprocessing
import pandas as pd
from stock_ai import calcs
import numpy as np
from test import get_stock_daily
from test import is_travis


def test_preprocessing__xy_split_1():
    arr = [i for i in range(2, 8)]
    window = len(arr) - 2
    days = 2
    assert window == 4
    assert days == 2
    cols = ['c']

    x, y = preprocessing._xy_split_1([pd.DataFrame(arr, columns=cols)],
                                     window,
                                     days,
                                     x_cols=cols,
                                     y_cols=cols)

    print(x)
    print(y)
    print(type(x))
    print(type(y))
    print(type(x[0]))
    print(type(y[0]))

    assert np.array_equal(np.arange(1.0, 3, 0.5), x[0]['c'].to_numpy())
    assert np.array_equal(np.array([1.2, 1.4]), y[0]['c'].to_numpy())


def test_preprocessing__xy_split_2():
    arr = [i for i in range(2, 8)]
    window = len(arr) - 2
    days = 2
    assert window == 4
    assert days == 2
    cols = ['c']

    x, y = preprocessing._xy_split_2([pd.DataFrame(arr, columns=cols)],
                                     window,
                                     days,
                                     x_cols=cols,
                                     y_cols=cols)

    print(x)
    print(y)
    print(type(x))
    print(type(y))
    print(type(x[0]))
    print(type(y[0]))

    assert np.array_equal(np.arange(1.0, 3, 0.5), x[0]['c'].to_numpy())
    assert np.array_equal(np.array([3.0, 3.5]), y[0]['c'].to_numpy())


def test_P1():
    arr = [i for i in range(2, 8)]
    window = len(arr) - 2
    days = 2
    columns = ['close']
    p = preprocessing.P1(window=window, days=days)
    df = pd.DataFrame(arr, columns=columns)
    result_df = p.trans_to_df(df)
    result_np = p.trans_to_numpy(df)
    assert len(result_df) == len(result_np)
    assert (len(columns), window) == p.x_shape
    assert days == p.y_shape
    for i in range(len(result_df)):
        print(result_np[i].shape)
        print(result_df[i].to_numpy())
        assert np.array_equal(result_np[i], result_df[i].to_numpy())


def test_P1_in_real_data():
    df = get_stock_daily()
    p = preprocessing.P1(x_cols=df.columns)
    x_df, y_df = p.trans_to_df(df)
    x_np, y_np = p.trans_to_numpy(df)
    for i in range(len(x_df)):
        assert np.array_equal(x_df[i].to_numpy(), x_np[i])
        assert np.array_equal(y_df[i].to_numpy(), y_np[i])
        assert np.array_equal(y_np[i].shape, (len(p.y_cols), p.days))
        assert np.array_equal(x_np[i].shape, (len(p.x_cols), p.window))

import pytest
from test import get_stock_daily
from test import get_index_daily
from stock_ai.data_processor import merge
import test
from stock_ai import calcs
from stock_ai import preprocessing
import h5py
import os
import numpy as np
from keras import layers
from keras.models import Sequential
from stock_ai.ploter import plot_keras_history
from keras.callbacks import EarlyStopping


# @pytest.mark.skip()
def test_train():
    data_file = 'data.h5'

    x, y, x_col, y_col = _get_data(data_file)
    # if not os.path.exists(data_file):
    #     _write_data({
    #         'x': x,
    #         'y': y,
    #         'x_col': np.string_(x_col),
    #         'y_col': y_col
    #     }, data_file)

    train_size = (int)(y.shape[0] * 0.8)
    train_x = x[:train_size]
    train_y = np.reshape(y[:train_size], len(y[:train_size]))
    test_x = x[train_size:]
    test_y = np.reshape(y[train_size:], len(y[train_size:]))

    model = Sequential()
    model.layers.append(layers.LSTM(512, input_shape=train_x[0].shape,return_sequences=True))
    model.layers.append(layers.LSTM(256,return_sequences=True))
    model.layers.append(layers.LSTM(128))
    model.layers.append(layers.Dense(1))

    model.compile(optimizer='rmsprop', loss='mse', metrics=["mae", "acc"])

    history = model.fit(
        train_x,
        train_y,
        batch_size=128,
        epochs=1000,
        validation_data=(test_x, test_y),
        # callbacks=[EarlyStopping(monitor='val_loss')]
    )
    print('x_col:{}'.format(x_col))
    print('y_col:{}'.format(y_col))
    model.summary()
    plot_keras_history(history)


# @pytest.mark.skip()
def test_data():
    """测试读写x，y的值，是否一致"""
    data_file = 'data.h5'

    x1, y1, x_col, y_col = _get_data()
    if not os.path.exists(data_file):
        _write_data(
            {
                'x': x1,
                'y': y1,
                'x_col': np.string_(x_col),
                'y_col': y_col
            }, data_file)
    x2, y2, x_col, y_col = _get_data(data_file)

    print(x1)
    print(x2)
    print(y1)
    print(y2)
    np.array_equal(x1, x2)
    np.array_equal(y1, y2)


def test_print_merge_data():
    df_stock = get_stock_daily()
    df_index = get_index_daily()
    df = merge({
        test.stock_code: df_stock,
        test.index_code: df_index
    },
               append_funcs=_get_append_funcs())
    cols = [col for col in df.columns if 'onehot' in col or '_return_' in col]

    for col in cols:
        print(df[col].describe())


def _get_append_funcs():
    return {
        'is_suspension': calcs.is_trade_suspension,
        'year': calcs.calc_year,
        'month': calcs.calc_month,
        'year_onehot': [calcs.trans_onehot, {
            'column': 'year'
        }],
        'month_onehot': [calcs.trans_onehot, {
            'column': 'month'
        }],
        'daily_return_close': [calcs.calc_daily_return, {
            'column': 'close'
        }],
        'sum_return_close': [calcs.calc_cum_return, {
            'column': 'close'
        }],
        'ema_5': [calcs.tech_ema, {
            'days': 5
        }],
        'ema_10': [calcs.tech_ema, {
            'days': 10
        }],
        'ema_15': [calcs.tech_ema, {
            'days': 15
        }],
        'daily_return_ema_5': [calcs.calc_daily_return, {
            'column': 'ema_5'
        }],
        'daily_return_ema_10': [calcs.calc_daily_return, {
            'column': 'ema_10'
        }],
        'daily_return_ema_15': [calcs.calc_daily_return, {
            'column': 'ema_15'
        }],
        'dropna': [calcs.dropna, {
            'replace': True
        }]
    }


def _get_data(f=None):
    if f and os.path.exists(f):
        g = h5py.File(f, 'r')
        x = np.array(g.get('x'))
        y = np.array(g.get('y'))
        x_col = np.array(g.get('x_col'))
        y_col = np.array(g.get('y_col'))
        g.close()
        return x, y, x_col, y_col

    df_stock = get_stock_daily()
    df_index = get_index_daily()
    df = merge({
        test.stock_code: df_stock,
        test.index_code: df_index
    },
               append_funcs=_get_append_funcs())
    cols = [
        col for col in df.columns
        if 'onehot' in col or '_return_' in col or col == 'is_suspension'
    ]

    print(df[cols])
    print(df[cols].dtypes)

    p1 = preprocessing.P1(x_cols=cols, y_cols=['daily_return_close'],window=30)
    x, y = p1.trans_to_numpy(df, norm_func=None)

    return x, y, cols, 'daily_return_close'


def _write_data(d, file):
    g = h5py.File(file, 'w')
    for k, v in d.items():
        g.create_dataset(k, data=v)
    g.close()

import os
from stock_ai import data_processor
import pandas as pd
import logging
import warnings
import numpy as np
from keras.preprocessing.sequence import TimeseriesGenerator
from stock_ai import util
from stock_ai import calcs
import sklearn.preprocessing

warnings.filterwarnings("ignore")
__cache = {}  # 日线数据缓存缓存

is_travis = "TRAVIS" in os.environ and os.environ["TRAVIS"] == "true"
stock_code = '601398'
index_code = '399300'


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
    df=df.drop(columns=['down_count','up_count'])
    logging.debug("Load Daily:" + code)
    __cache[code] = df
    return df


def get_data():
    df = data_processor.merge({
        index_code: get_index_daily(),
        stock_code: get_stock_daily()
    })
    df.fillna(method='ffill', inplace=True)
    df.dropna(inplace=True)
    return df

def get_data_regression(y_col='close601398', **kwargs):
    """回归测试使用的数据源"""
    test_split = kwargs.pop('test_split', 0.1)
    assert 0 <= test_split < 1

    df = get_data()

    # cols_X=[col for col in df.columns if col != y_col and '601398' in col]
    cols_X=['open601398', 'high601398', 'low601398','close601398']
    print(cols_X)
    x_np = df[cols_X].to_numpy()

    y_np = df[y_col].values

    round_Y = kwargs.pop('round_Y', 4)
    y_np = np.round(y_np, round_Y)

    length = kwargs.pop('length', 5)#x轴天数
    sampling_rate = kwargs.pop('sampling_rate', 1)
    batch_size = kwargs.pop('batch_size', 1)

    data_gen = TimeseriesGenerator(x_np, y_np,
                                   length=length, sampling_rate=sampling_rate,
                                   batch_size=batch_size)

    x = []
    y = []

    round_X=kwargs.pop('round_Y', 4)
    scaler = kwargs.pop('scaler', sklearn.preprocessing.MinMaxScaler())

    for d in data_gen:
        x.append(np.round(d[0][0],round_X))
        y.append(d[1][0])

    x_train = np.array(x[:int(len(x) * (1 - test_split))])
    y_train = np.array(y[:int(len(x) * (1 - test_split))])
    x_test = np.array(x[int(len(x) * (1 - test_split)):])
    y_test = np.array(y[int(len(x) * (1 - test_split)):])
    return (x_train, y_train), (x_test, y_test)

def get_data_classification(y_col='close601398', **kwargs):
    """分类测试使用的数据源"""
    test_split = kwargs.pop('test_split', 0.1)
    assert 0 <= test_split < 1

    df = get_data()
    x_np = df.to_numpy()

    if kwargs.pop('pct', True):
        y_np = df[y_col].pct_change().values
    else:
        y_np = df[y_col].values

    round_Y = kwargs.pop('round_Y', 2)
    y_np = np.round(y_np, round_Y)

    length = kwargs.pop('length', 5)
    sampling_rate = kwargs.pop('sampling_rate', 1)
    batch_size = kwargs.pop('batch_size', 1)

    data_gen = TimeseriesGenerator(x_np, y_np,
                                   length=length, sampling_rate=sampling_rate,
                                   batch_size=batch_size)

    x = []
    y = []

    scaler = kwargs.pop('scaler', sklearn.preprocessing.MinMaxScaler())

    for d in data_gen:
        x.append(scaler.fit_transform(d[0][0]))
        y.append(d[1][0])

    onehot = kwargs.pop('onehot', True)
    if onehot:
        y = calcs.trans_onehot(y)

    x_train = np.array(x[:int(len(x) * (1 - test_split))])
    y_train = np.array(y[:int(len(x) * (1 - test_split))])
    x_test = np.array(x[int(len(x) * (1 - test_split)):])
    y_test = np.array(y[int(len(x) * (1 - test_split)):])
    return (x_train, y_train), (x_test, y_test)


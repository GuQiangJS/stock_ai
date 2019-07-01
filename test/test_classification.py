import pytest
from test import get_stock_daily
from test import get_index_daily
import test
from stock_ai import calcs
import stock_ai.preprocessing
import h5py
import os
import numpy as np
from keras import layers
from keras.models import Sequential
from stock_ai.ploter import plot_keras_history
from keras.callbacks import EarlyStopping
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from stock_ai.module import StockCN


# @pytest.mark.skip()
def test_train():
    plt.figure(figsize=(100, 80))

    data_file = 'data.h5'

    x, y, x_col, y_col = _get_data(None)
    # if not os.path.exists(data_file):
    #     _write_data({
    #         'x': x,
    #         'y': y,
    #         'x_col': np.string_(x_col),
    #         'y_col': y_col
    #     }, data_file)

    train_size = (int)(y.shape[0] * 0.95)
    train_x = x[:train_size]
    train_y = y[:train_size]
    test_x = x[train_size:]
    test_y = y[train_size:]

    print(train_x[0])
    print(train_x[1])
    print(train_y[0])

    model = Sequential()
    # model.layers.append(
    #     layers.LSTM(512, input_shape=x[0].shape, return_sequences=True))
    # model.layers.append(layers.LSTM(256, return_sequences=True))
    # model.layers.append(layers.LSTM(128, return_sequences=True))
    # model.layers.append(layers.LSTM(64, return_sequences=True))
    # model.layers.append(layers.LSTM(32))
    model.layers.append(layers.LSTM(256, input_shape=x[0].shape))
    model.layers.append(layers.Dense(y[0].size))

    model.compile(optimizer='rmsprop',
                  loss='mae',
                  metrics=['mae', 'acc'])

    history = model.fit(train_x,
                        train_y,
                        batch_size=128,
                        epochs=1000,
                        shuffle=True,
                        validation_split=0.1,
                        verbose=1,
                        # callbacks=[EarlyStopping(monitor='acc')]
                        )
    print('x_col:{}'.format(x_col))
    print('y_col:{}'.format(y_col))
    model.summary()
    plot_keras_history(history)

    pred_y = model.predict(test_x)
    print(pred_y)
    # plot_pred_compare_real(test_y, pred_y, title="\n".join(x_col))


def plot_pred_compare_real(real, pred, **kwargs):
    plots = []
    d = pd.DataFrame({'real': real, 'pred': pred.reshape(pred.shape[0])})
    # d['c']=abs(d['pred']-d['real'])/d['real']
    # print(d)
    ax = sns.lineplot(data=d)
    title = kwargs.pop('title', None)
    if title:
        ax.set_title(title, fontsize='x-small')

    fig, axs = plt.subplots(nrows=2)
    sns.lineplot(data=d, ax=axs[0])
    c=pd.DataFrame((d['pred'] - d['real']) / d['real'])
    sns.lineplot(data=c,ax=axs[1])
    print(c.describe())
    plt.tight_layout()
    plt.show()


# @pytest.mark.skip()
def test_plot_data_df():
    df, col_x, col_y = _get_data_df()
    print(df[col_x].head())
    print(col_x)

def _get_data(f=None, **kwargs):
    if f and os.path.exists(f):
        g = h5py.File(f, 'r')
        x = np.array(g.get('x'))
        y = np.array(g.get('y'))
        x_col = np.array(g.get('x_col'))
        y_col = np.array(g.get('y_col'))
        g.close()
        return x, y, x_col, y_col

    df, col_x, col_y = _get_data_df()

    x = []
    y = []

    df = _norm(df)

    window = kwargs.pop('window', 5)
    days = kwargs.pop('days', 1)

    p1 = stock_ai.preprocessing.P1(x_cols=col_x,
                                   y_cols=col_y,
                                   window=window,
                                   days=days)
    x, y = p1.trans_to_df(data=df, norm_func=None)
    x=np.array([r.to_numpy() for r in x])
    y=calcs.trans_onehot(np.round(y,1))

    # decimals=kwargs.pop('decimals',3)
    # x=np.around(x,decimals=decimals)
    # y=np.around(y,decimals=decimals)

    return x, y, col_x, col_y


def test_get_data_df():
    df, x, y = _get_data_df()
    # print(df[df['need_is_sus']==1][x])
    # print(df[x])
    # print(df[x].describe())
    print(df[x].max())
    print(df[x].min())
    print(df[x].std())
    df = _norm(df)
    print(df[x].max())
    print(df[x].min())
    print(df[x].std())


def test_get_data():
    df = _get_data_df()[0]
    window = 5
    days = 1
    x, y, x_col, y_col = _get_data(window=window, days=days)
    y = np.around(y, decimals=2)
    x = np.around(x, decimals=2)
    print(x.shape)
    print(y.shape)
    print(x_col)
    print(y_col)
    print(x[0], y[0])
    print(df[y_col][:window + days])
    print(x[-1], y[-1])
    print(df[y_col][(window + days) * -1:])


def _norm(df):
    x = df.values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    return pd.DataFrame(x_scaled, columns=df.columns, index=df.index)


def _get_data_df(**kwargs):
    df_stock = get_stock_daily()
    df_index = get_index_daily()
    # df_601939 = get_stock_daily('601939')  #建设银行
    # df_601328 = get_stock_daily('601328')  #交通银行
    # df_600036 = get_stock_daily('600036')  #招商银行
    # df_601988 = get_stock_daily('601988')  #中国银行
    # df_601288 = get_stock_daily('601288')  #农业银行

    s_601398=StockCN('601398')

    df = df_index.join(df_stock, how='left', lsuffix=test.index_code)

    other_code=[]

    for code in ['601339','601328','600036','601988','601288']:
        s=StockCN(code)
        if s.ipo_date<=s_601398.ipo_date:
            other_code.append(code)
            df = get_stock_daily(code).join(df, how='left', lsuffix=code)

    df['need_is_sus'] = calcs.is_trade_suspension(df)
    df.fillna(method='ffill', inplace=True)

    df[['need_BOLL_5_2','need_UB_5_2','need_LB_5_2']]=calcs.indicators.QA_indicator_BOLL(df,N=5,P=2)
    df[['need_BOLL_10_2','need_UB_10_2','need_LB_10_2']]=calcs.indicators.QA_indicator_BOLL(df,N=10,P=2)
    df[['need_BOLL_15_2','need_UB_105_2','need_LB_15_2']]=calcs.indicators.QA_indicator_BOLL(df,N=15,P=2)
    df[['need_MTM_10_5','need_MTMMA_10_5']]=calcs.indicators.QA_indicator_MTM(df,N=10,M=5)
    df[['need_MTM_15_10','need_MTMMA_15_10']]=calcs.indicators.QA_indicator_MTM(df,N=15,M=10)
    df[['need_DIF_5_15_10','need_DEA_5_15_10','need_MACD_5_15_10']]=calcs.indicators.QA_indicator_MACD(df,short=5,long=15,mid=10)

    for code in ['',test.index_code]+other_code:
        df['need_close'+code]=df['close'+code]
        df['need_volume'+code]=df['volume'+code]

    df.dropna(inplace=True)

    col_y = 'need_close'
    col_x = [col for col in df.columns if 'need_' in col]

    # df = merge(
    #     {
    #         test.stock_code: df_stock,
    #         test.index_code: df_index,
    #         # '601939': df_601939,
    #         # '601328': df_601328,
    #         # '600036': df_600036,
    #         # '601988': df_601988,
    #         # '601288': df_601288
    #     },
    #     append_funcs=_get_append_funcs())
    # col_x, col_y = _get_columns(df)
    # print(df[col_x])
    return df, col_x, col_y


#
# def _get_data(f=None):
#     if f and os.path.exists(f):
#         g = h5py.File(f, 'r')
#         x = np.array(g.get('x'))
#         y = np.array(g.get('y'))
#         x_col = np.array(g.get('x_col'))
#         y_col = np.array(g.get('y_col'))
#         g.close()
#         return x, y, x_col, y_col
#
#     df, col_x, col_y = _get_data_df()
#
#     p1 = preprocessing.P1(x_cols=col_x, y_cols=col_y, window=5)
#     x, y = p1.trans_to_numpy(df, norm_func=None)
#
#     return x, y, col_x, col_y
#
#
# def _write_data(d, file):
#     g = h5py.File(file, 'w')
#     for k, v in d.items():
#         g.create_dataset(k, data=v)
#     g.close()

# from stock_ai import keras
# from test import get_stock_daily
# from test import get_index_daily
# from stock_ai import ploter

# def test_data():
#     m = keras.StockSequentialModel(stock='601398')
#     # print(m._data)
#     # print(m._data.shape)
#     # print(m._data.dtypes)
#     print(m.train_x.shape)
#     print(m.train_y.shape)
#     print(m.compile())
#     print(m.fit())
#     print(m.model.summary())

import keras
import test
import stock_ai.wrapper
import stock_ai.calcs
import numpy as np
import stock_ai.preprocessing
import logging
from stock_ai import ploter
from stock_ai.module import StockCN

def test_keras():
    df_stock = test.get_stock_daily()
    df_index = test.get_index_daily()
    df = stock_ai.wrapper.dataframe_merge(df_stock, df_index)

    stock=StockCN('601398')
    columns = [
        'close',
        'close_index',
        'volume',
        'volume_index',
        'year',
        'month',
        # 'is_trade_sus'
    ]
    for code in stock.related_codes:
        stock_code=StockCN(code)
        if stock_code.ipo_date>stock.ipo_date:
            continue
        df_code=test.get_stock_daily()
        df['{0}_colse'.format(code)]=df_code['close']
        df['{0}_volume'.format(code)]=df_code['volume']
        columns.append('{0}_colse'.format(code))
        columns.append('{0}_volume'.format(code))
    #附加列
    df['year'] = stock_ai.calcs.calc_year(df)  #年份
    df['year'] = df['year'] - df['year'][0]
    df['month'] = stock_ai.calcs.calc_month(df)  #月份
    df['month'] = df['month'] / 10.0
    # df['is_trade_sus'] = stock_ai.calcs.is_trade_suspension(df)  #是否停牌
    # df['is_trade_sus'] = np.where(df['is_trade_sus'], 1, 0)
    #附加列完成

    df.fillna(method='ffill', inplace=True)
    df.dropna(inplace=True)

    #正则化

    #正则化结束

    train_len = int(df.shape[0] * 0.9)  # 训练集/验证集拆分
    train_df = df[columns][:train_len]  # 训练集DataFrame，未经过数据预处理
    val_df = df[columns][train_len:]  # 验证集DataFrame，未经过数据预处理

    window = 3
    days = 1
    p = stock_ai.preprocessing.P1(x_cols=columns, window=window, days=days)

    train_x_df, train_y_df = p.trans_to_df(train_df)
    logging.debug(train_x_df[0])
    train_x_np = np.array([r.to_numpy() for r in train_x_df])
    train_y_np = np.array([r.to_numpy()[0] for r in train_y_df])
    val_x_df, val_y_df = p.trans_to_df(val_df)
    val_x_np = np.array([r.to_numpy() for r in val_x_df])
    val_y_np = np.array([r.to_numpy()[0] for r in val_y_df])

    input_shape = train_x_np[0].shape
    output_unit = days

    print(train_x_df[0])
    print(train_y_df[0])

    # 期望输入数据尺寸: (batch_size, timesteps, data_dim)
    model = keras.Sequential()
    model.add(
        keras.layers.LSTM(32, return_sequences=True,
                          input_shape=input_shape))  # 返回维度为 32 的向量序列
    model.add(keras.layers.LSTM(32, return_sequences=True))  # 返回维度为 32 的向量序列
    model.add(keras.layers.LSTM(32))  # 返回维度为 32 的单个向量
    model.add(keras.layers.Dense(output_unit, activation='softmax'))

    model.compile(loss='mse', optimizer='rmsprop', metrics=["mae", "acc"])

    logging.debug(train_x_np[0])

    h = model.fit(train_x_np,
                  train_y_np,
                  batch_size=128,
                  epochs=50,
                  validation_data=(val_x_np, val_y_np))
    model.summary()
    ploter.plot_keras_history(h)

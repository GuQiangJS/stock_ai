# """keras相关包装"""
#
# import keras
# from keras import layers
# from stock_ai.module import StockCN
# from stock_ai.module import IndexCN
# from keras.callbacks import EarlyStopping
# from stock_ai import wrapper
# from stock_ai import calcs
# import pandas as pd
# from copy import deepcopy
# from stock_ai import preprocessing
# from abc import ABCMeta
# import numpy as np
#
#
# class Model(metaclass=ABCMeta):
#     pass
#
#
# class SequentialModel(Model):
#
#     def __init__(self, **kwargs):
#         self._model = keras.models.Sequential()
#
#     @property
#     def model(self):
#         """ :class:`keras.models.Sequential` 模型实例
#
#         Returns:
#             :class:`keras.models.Sequential`
#         """
#         return self._model
#
#     def model_from_config(self, config, **kwargs):
#         """参考 :func:`keras.models.model_from_config`"""
#         custom_objects = kwargs.pop('custom_objects', None)
#         self._model = keras.models.model_from_config(config, custom_objects)
#
#     def model_from_json(self, json_string, **kwargs):
#         """参考 :func:`keras.models.model_from_config`"""
#         custom_objects = kwargs.pop('custom_objects', None)
#         self._model = keras.models.model_from_json(json_string, custom_objects)
#
#     def save_model(self, filepath, **kwargs):
#         """参考 :func:`keras.models.Sequential.save`"""
#         overwrite = kwargs.pop('overwrite', True)
#         include_optimizer = kwargs.pop('include_optimizer', True)
#         self._model.save(filepath, overwrite, include_optimizer)
#
#     def compile(self, optimizer, **kwargs):
#         """参考 :func:`keras.models.Sequential.compile`"""
#         self.model.compile(optimizer, **kwargs)
#
#     def fit(self, **kwargs):
#         """参考 :func:`keras.models.Sequential.fit`"""
#         return self.model.fit(**kwargs)
#
#     def evaluate(self, **kwargs):
#         """参考 :func:`keras.models.Sequential.evaluate`"""
#         return self.model.evaluate(**kwargs)
#
#     def predict(self, **kwargs):
#         """参考 :func:`keras.models.Sequential.predict`"""
#         return self.model.predict(**kwargs)
#
#
# class StockSequentialModel(SequentialModel):
#     """包装 :class:`keras.models.Sequential`"""
#
#     def __init__(self, **kwargs):
#         """构造函数
#
#         Args:
#             stock (str or stock_ai.module.Stock, 可选): 股票代码或股票实例。默认为 ``None``。
#             index (str or stock_ai.module.Stock, 可选): 指数代码或股票实例。默认为 ``399300``。
#             train_size (float, 可选): 训练集/测试集拆分比率。默认为 ``0.9``。
#             window (int, 可选): 计算天数。默认为 3。
#             days (int, 可选): 预测天数。默认为 1。
#             layers (tuple(dict)): 层定义集合。集合中的每一项代表一层的定义。默认为 ::
#
#                     (
#                         {'type': 'lstm' ,'units': 128},
#                         {'type': 'dense' ,'units': days}
#                     )
#
#                 層定義包含 `type`:(dense或lstm) 用來定義層的類型。
#                 層定義中其他屬性參見 `Dense`_ 和 `LSTM`_ 構造函數參數定義。
#             append_funcs (collections.OrderedDict, 可选): 附加其他列数据时的计算方法字典。参考 :func:`stock_ai.wrapper.dataframe_merge` 中的同名参数。
#                 默认为 ::
#
#                     {
#                         'type': 'lstm',
#                         'month': stock_ai.calcs.calc_month,
#                         'is_trade_suspension': stock_ai.calcs.is_trade_suspension
#                     }
#
#             x_cols (tuple(str), 可选): 训练数据集从原始数据源中获取的列名集合。默认为所有。
#             y_cols (tuple(str), 可选)): 验证数据集从原始数据源中获取的列名。默认为 ``['close']``。
#
#         See Also:
#             * :func:`stock_ai.wrapper.dataframe_merge`
#             * :mod:`stock_ai.calcs`
#
#         .. _Dense:
#         https://keras.io/zh/layers/core/#dense
#         .. _LSTM:
#         https://keras.io/zh/layers/recurrent/#lstm
#         """
#         super(StockSequentialModel, self).__init__()
#         self._stock = kwargs.pop('stock', None)
#         self._index = kwargs.pop('index', '399300')
#         self._history = None
#         self._train_size = kwargs.pop('train_size', 0.9)
#         self._append_funcs = kwargs.pop(
#             'append_funcs', {
#                 'year': calcs.calc_year,
#                 'month': calcs.calc_month,
#                 'is_trade_suspension': calcs.is_trade_suspension
#             })
#         # # 从完整数据集中取哪些列作为训练数据集
#         # if not self._x_cols:
#         #     for append_column in self._append_funcs.keys():
#         #         if append_column not in self._x_cols:
#         #             self._x_cols.append(append_column)
#         # 完整数据
#         self._data = wrapper.dataframe_merge(self.stock.get_daily(),
#                                              self.index.get_daily(),
#                                              append_funcs=self._append_funcs)
#
#         self._data = self._data.fillna(method='ffill')
#         self._data = self._data.dropna()
#         self._x_cols = kwargs.pop('x_cols', self._data.columns)
#         self._y_cols = kwargs.pop('y_cols', ['close'])
#
#         _train_len = int(self._data.shape[0] * self._train_size)  #训练集/验证集拆分
#         self._train_df = self._data[
#             self._x_cols][:_train_len]  #训练集DataFrame，未经过数据预处理
#         raise ValueError #_val_df取值
#         self._val_df = self._data[self._y_cols][
#             _train_len:]  #验证集DataFrame，未经过数据预处理
#         self.window = kwargs.pop('window', 3)
#         self.days = kwargs.pop('days', 1)
#         self.prep = kwargs.pop(
#             'prep_func',
#             preprocessing.P1(self.window,
#                              self.days,
#                              x_cols=self._x_cols,
#                              y_cols=self._y_cols))
#         self._layers = kwargs.pop(
#             'layers', self._default_layers(self.prep.x_shape, self.days))
#
#         self._train_x = []
#         self._train_y = []
#         self._val_x = []
#         self._val_y = []
#
#     @staticmethod
#     def _norm_func(data: pd.DataFrame) -> pd.DataFrame:
#         df = data.copy()
#         for col in df.columns:
#             if col in ['year', 'month']:
#                 continue
#             elif col in ['is_trade_suspension']:
#                 df[col] = np.where(df[col], 1, 0)
#             else:
#                 df[col] = calcs.cum_return(df[col])
#         return df
#
#     def _default_layers(self, first_layer_shape, last_layer_shape):
#         return ({
#             'type': 'lstm',
#             'units': 128,
#             'input_shape': first_layer_shape
#         }, {
#             'type': 'dense',
#             'units': last_layer_shape
#         })
#
#     def __append_layers(self, ls):
#         self.model.layers.clear()
#
#         for layer in ls:
#             t = layer.pop('type')
#             if t == 'dense':
#                 # https://keras.io/zh/layers/core/
#                 self.model.add(layers.Dense.from_config(layer))
#             elif t == 'lstm':
#                 # https://keras.io/zh/layers/recurrent/#lstm
#                 self.model.add(layers.LSTM.from_config(layer))
#             elif t == 'dropout':
#                 # https://keras.io/zh/layers/recurrent/#Dropout
#                 self.model.add(layers.Dropout.from_config(layer))
#             elif t == 'cudnnlstm':
#                 # https://keras.io/zh/layers/recurrent/#Dropout
#                 self.model.add(layers.CuDNNLSTM.from_config(layer))
#
#     @property
#     def stock(self):
#         """当前模型使用的主要股票对象
#
#         Returns:
#             :class:`stock_ai.module.Stock`:
#         """
#         if self._stock and isinstance(self._stock, str):
#             self._stock = StockCN(self._stock)
#         return self._stock
#
#     @property
#     def index(self):
#         """当前模型使用的主要股票指数
#
#         Returns:
#             :class:`stock_ai.module.Index`:
#         """
#         if self._index and isinstance(self._index, str):
#             self._index = IndexCN(self._index)
#         return self._index
#
#     def compile(self, **kwargs):
#         """配置训练模型
#
#         See Also:
#             :func:`keras.models.Sequential.compile`
#
#         Args:
#             optimizer (str or :mod:`keras.optimizers` 中的优化器实例): 参考 :func:`keras.models.Sequential.compile` 中同名参数。默认为 :class:`keras.optimizers.RMSprop`。
#             loss (str or :mod:`keras.losses` 中的优化器实例): 参考 :func:`keras.models.Sequential.compile` 中同名参数。默认为 :class:`keras.losses.mean_squared_error`。
#             loss : 参考 :func:`keras.models.Sequential.compile` 中同名参数。默认为 ``["mae", "acc"]``。
#             kwargs : 参考 :func:`keras.models.Sequential.compile` 中同名参数。
#         """
#         self.__append_layers(deepcopy(self._layers))
#
#         optimizer = kwargs.pop('optimizer', 'rmsprop')
#         loss = kwargs.pop('loss', 'mse')
#         metrics = kwargs.pop('metrics', ["mae", "acc"])
#         self.model.compile(optimizer, loss=loss, metrics=metrics, **kwargs)
#
#     @property
#     def train_df(self):
#         """训练用数据集
#
#         直接按照 :attr:`_train_size` 对 :attr:`_data` 拆分后的前部分数据。
#
#         Returns:
#             :class:`pandas.DataFrame`:
#         """
#         return self._train_df
#
#     @property
#     def val_df(self):
#         """测试用数据集
#
#         直接按照 :attr:`_train_size` 对 :attr:`_data` 拆分后的剩余部分数据。
#
#         Returns:
#             :class:`pandas.DataFrame`:
#         """
#         return self._val_df
#
#     @property
#     def train_x(self):
#         """训练集训练用Numpy数组"""
#
#         if not self._train_x:
#             self._train_x, self._train_y = self.prep.trans_to_numpy(
#                 self.train_df, norm_func=StockSequentialModel._norm_func)
#         return self._train_x
#
#     @property
#     def train_y(self):
#         """训练集验证用Numpy数组"""
#         # return self._train_df[self._y_cols].to_numpy()
#
#         if not self._train_y:
#             self._train_x, self._train_y = self.prep.trans_to_numpy(
#                 self.train_df, norm_func=StockSequentialModel._norm_func)
#         return self._train_y
#
#     @property
#     def val_y(self):
#         """验证集验证用Numpy数组"""
#         if not self._val_y:
#             self._val_x, self._val_y = self.prep.trans_to_numpy(
#                 self.val_df, norm_func=StockSequentialModel._norm_func)
#         return self._val_y
#
#     @property
#     def val_x(self):
#         """验证集训练用Numpy数组"""
#         if not self._val_x:
#             self._val_x, self._val_y = self.prep.trans_to_numpy(
#                 self.val_df, norm_func=StockSequentialModel._norm_func)
#         return self._val_x
#
#     @property
#     def history(self):
#         """参考 :func:`keras.models.Sequential.fit` 返回值。"""
#         return self._history
#
#     def fit(self, **kwargs):
#         """训练模型
#
#         Args:
#             x: 训练数据的 Numpy 数组。参考 :func:`keras.models.Sequential.fit` 中同名参数。默认为 :attr:`train_x`。
#             y: 目标（标签）数据的 Numpy 数组。参考 :func:`keras.models.Sequential.fit` 中同名参数。默认为 :attr:`train_y`。
#             batch_size (int): 参考 :func:`keras.models.Sequential.fit` 中同名参数。默认为 ``128``。
#             epochs (int): 参考 :func:`keras.models.Sequential.fit` 中同名参数。默认为 ``10``。
#             callbacks:  参考 :func:`keras.models.Sequential.fit` 中同名参数。默认为 ``[EarlyStopping(monitor='val_loss')]``。
#             kwargs : 参考 :func:`keras.models.Sequential.fit` 中其他参数定义。
#
#         Returns:
#             :class:`keras.callbacks.History`: 参考 :func:`keras.models.Sequential.fit` 返回值。
#
#         See Also:
#             :func:`keras.models.Sequential.fit`
#         """
#         batch_size = kwargs.pop('batch_size', 128)
#         epochs = kwargs.pop('epochs', 10)
#         callbacks = kwargs.pop('callbacks', [EarlyStopping(monitor='val_loss')])
#         x = kwargs.pop('x', self.train_x)
#         y = kwargs.pop('y', self.train_y)
#         self._history = self.model.fit(x=x,
#                                        y=y,
#                                        batch_size=batch_size,
#                                        epochs=epochs,
#                                        callbacks=callbacks,
#                                        **kwargs)
#         return self._history
#
#     def evaluate(self, **kwargs):
#         """在测试模式，返回误差值和评估标准值
#
#         Args:
#             x: 训练数据的 Numpy 数组。参考 :func:`keras.models.Sequential.evaluate` 中同名参数。默认为 :attr:`val_x`。
#             y: 目标（标签）数据的 Numpy 数组。参考 :func:`keras.models.Sequential.evaluate` 中同名参数。默认为 :attr:`val_y`。
#             batch_size (int): 参考 :func:`keras.models.Sequential.evaluate` 中同名参数。默认为 ``128``。
#             kwargs : 参考 :func:`keras.models.Sequential.evaluate` 中其他参数定义。
#
#         Returns:
#             参考 :func:`keras.models.Sequential.evaluate` 的返回值定义。
#
#         """
#         batch_size = kwargs.pop('batch_size', 128)
#         x = kwargs.pop('x', self.val_x)
#         y = kwargs.pop('y', self.val_y)
#         return self.model.evaluate(x=x, y=y, batch_size=batch_size, **kwargs)
#
#     def predict(self, x, **kwargs):
#         """输入样本生成输出预测
#
#         Args:
#             x: 训练数据的 Numpy 数组。参考 :func:`keras.models.Sequential.predict` 中同名参数。默认为 :attr:`val_x`。
#             batch_size (int): 参考 :func:`keras.models.Sequential.predict` 中同名参数。默认为 ``128``。
#             kwargs : 参考 :func:`keras.models.Sequential.predict` 中其他参数定义。
#
#         Returns:
#             参考 :func:`keras.models.Sequential.predict` 的返回值定义。
#
#         """
#
#         batch_size = kwargs.pop('batch_size', 128)
#         return self.model.predict(x, batch_size=batch_size, **kwargs)

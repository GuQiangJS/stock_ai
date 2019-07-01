"""数据预处理

主要负责将DataFrame拆分为 :func:`keras.models.Sequential.evaluate` 或 :func:`keras.models.Sequential.fit` 或 :func:`keras.models.Sequential.predict` 中需要的 ``x`` 和 ``y``。

"""

from abc import ABCMeta
from copy import deepcopy
import numpy as np


class P(metaclass=ABCMeta):

    @property
    def x_shape(self):
        pass

    @property
    def y_shape(self):
        pass

    def trans_to_df(self, data):
        pass

    def trans_to_numpy(self, data):
        pass


def xy_split_1(dfs,
               window,
               days,
               x_cols=None,
               y_cols=['close'],
               skip=0,
               **kwargs):
    """拆分数据

    返回的 `x` 和 `y` 结果。
    与 `xy_split_2` 的区别在于。本函数拆分的结果集y的开始值是 **相对于结果集x的最后一条数据** 的。

    Args:
        dfs ([:class:`pandas.DataFrame`]): 待拆分的数据集。
        norm_func: 数据构造器。接受参数类型为 :class:`pandas.DataFrame`，返回类型为 :class:`pandas.DataFrame` 的静态方法即可。默认为 None。
        window (int): 窗口期。
        days (int): 结果期。
        skip (int): 窗口期与结果期之间需要跳过的日期数量。默认为0。
        x_cols (tuple(str)): x 取列集合。如果为None，则取df的所有列。
        y_cols (tuple(str)): y 取列集合。默认为 ``['close']``。

    Returns:
        [[:class:`pandas.DataFrame` 或者 :class:`pandas.Series`],[:class:`pandas.DataFrame` 或者 :class:`pandas.Series`]]: 按照 window,days 拆分后的集合。
        分别表示为 [x,y]。所使用的的列分别由 `x_cols` 和 `y_cols` 指定。

        根据 `x_cols` 和 `y_cols` 确定返回类型，如果是单独字符串则返回值中
        的元素类型为 :class:`pandas.Series`，如果是字符串数组或元组时返回值
        中的元素类型为 :class:`pandas.DataFrame`。

        **Y 返回的结果是相对于返回集合的前一条数据做的`norm`处理。**

    See Also:
        * :py:func:`xy_split_2`

    Examples:

        >>> import pandas as pd
        >>> window=4
        >>> days=1
        >>> skip=0
        >>> df=pd.DataFrame(data={'c':[c for c in range(0,window+days+skip)]})
        >>> df['c'].values
        array([0, 1, 2, 3, 4], dtype=int64)
        >>> x, y = xy_split_1([df],window=window,days=days,skip=skip, y_cols=['c'])
        >>> x
        [   c
        0  0
        1  1
        2  2
        3  3]
        >>> y
        [   c
        4  4]
        >>> skip=2
        >>> df=pd.DataFrame(data={'c':[c for c in range(0,window+days+skip)]})
        >>> df['c'].values
        array([0, 1, 2, 3, 4, 5, 6], dtype=int64)
        >>> x, y = xy_split_1([df],window=window,days=days,skip=skip, y_cols=['c'])
        >>> x
        [   c
        0  0
        1  1
        2  2
        3  3]
        >>> y
        [   c
        6  6]

    """
    X = []
    Y = []
    norm_func = kwargs.pop('norm_func', None)
    for df in dfs:
        if not x_cols:
            x_cols = df.columns
        df_tmp = df.copy()
        if norm_func:
            n = {k: deepcopy(v) for k, v in kwargs.items()}
            X.append(norm_func(df_tmp, **n)[x_cols][:window])
            Y.append(norm_func(df_tmp[-1 - days:], **n)[y_cols][1:1 + days])
        else:
            X.append(df_tmp[x_cols][:window])
            Y.append(df_tmp[-1 - days:][y_cols][1:1 + days])
    return X, Y


#
# def xy_split_2(dfs, window, days, x_cols, y_cols=['close'], **kwargs):
#     """拆分数据
#
#     返回的 `train` 和 `test` 结果。
#
#     * 与 `xy_split_1` 的区别在于。本函数拆分的结果集y的开始值是**相对于结果集x的第一条数据**的。
#
#     Args:
#         dfs ([:class:`pandas.DataFrame`]): 待拆分的数据集。
#         norm_func: 数据构造器。接受参数类型为 :class:`pandas.DataFrame`，返回类型为 :class:`pandas.DataFrame` 的静态方法即可。默认为 :func:`stock_ai.calcs.calc_cum_return`。
#         window (int): 窗口期
#         days (int): 结果期
#         x_cols (tuple(str)): x 取列集合。
#         y_cols (tuple(str)): y 取列集合。默认为 ``['close']``。
#
#     Returns:
#         [[:class:`pandas.DataFrame` 或者 :class:`pandas.Series`],[:class:`pandas.DataFrame` 或者 :class:`pandas.Series`]]: 按照 window,days 拆分后的集合。
#         分别表示为 [x,y]。所使用的的列分别由 `x_cols` 和 `y_cols` 指定。
#
#         根据 `x_cols` 和 `y_cols` 确定返回类型，如果是单独字符串则返回值中
#         的元素类型为 :class:`pandas.Series`，如果是字符串数组或元组时返回值
#         中的元素类型为 :class:`pandas.DataFrame`。
#
#         **Y 返回的结果是相对于返回集合的第一条数据做的`norm`处理。**
#
#     See Also:
#         * :py:func:`xy_split_1`
#
#     Examples:
#
#         >>> arr = [i for i in range(2, 8)]
#         >>> window = len(arr) - 2
#         >>> days = 2
#         >>> window, days
#         4, 2
#         >>> x, y = _xy_split_2([pd.DataFrame(arr, columns=['c'])],
#         ...                              window, days, col_name='c')
#         >>> x
#         [      c
#             0  1.0
#             1  1.5
#             2  2.0
#             3  2.5]
#         >>> y
#         [4    3.0
#         5    3.5
#         Name: c, dtype: float64]
#         >>> type(x[0])
#         <class 'pandas.core.frame.DataFrame'>
#         >>> type(y[0])
#         <class 'pandas.core.series.DataFrame'>
#
#     """
#     X = []
#     Y = []
#     norm_func = kwargs.pop('norm_func', calc_cum_return)
#     for df in dfs:
#         df_tmp = df.copy()
#         if norm_func:
#             df_tmp = norm_func(df_tmp,
#                                **{k: deepcopy(v) for k, v in kwargs.items()})
#         X.append(df_tmp[x_cols][:window])
#         Y.append(df_tmp[-1 - days:][y_cols][1:1 + days])
#     return X, Y


class P1(P):

    def __init__(self,
                 window=3,
                 days=1,
                 skip=0,
                 x_cols=['close'],
                 y_cols=['close']):
        self.window = window
        self.days = days
        self.x_cols = x_cols
        self.y_cols = y_cols
        self.skip = skip

    @property
    def x_shape(self):
        """x数据的数据维度

        Returns:
            (int,int): 返回值为 (:attr"`window`,:attr:`x_cols`)。
        """
        return (self.window, len(self.x_cols))

    @property
    def y_shape(self):
        """y数据的数据维度

        Returns:
            (int,int): 返回值为 (:attr"`days`,:attr:`y_cols`)。
        """
        return (self.days, len(self.y_cols))

    def trans_to_df(self, data, **kwargs):
        """拆分数据源为x,y格式，使用 :func:`xy_split_1`。

        Args:
            data (:class:`pandas.DataFrame`): 数据源。
            norm_func: 参考 :func:`xy_split_1`中同名参数。

        Returns:
            参考 :func:`xy_split_1`

        """
        x = []
        df_tmp = data.copy()
        batch_size = self.window + self.skip + self.days
        for i in range(df_tmp.shape[0]):
            if i + batch_size > df_tmp.shape[0]:
                break
            x.append(df_tmp[i:i + batch_size])  # 当前取出需要分割为X，Y的批次数据
        x, y = xy_split_1(x,
                          window=self.window,
                          days=self.days,
                          skip=self.skip,
                          x_cols=self.x_cols,
                          y_cols=self.y_cols,
                          **kwargs)
        return (x, y)

    def trans_to_numpy(self, data, **kwargs):
        x, y = self.trans_to_df(data, **kwargs)
        return (np.array([r.to_numpy() for r in x]),
                np.array([r.to_numpy() for r in y]))

"""数据包装器"""

import pandas as pd


def dataframe_merge(df1, df2, **kwargs):
    """股票日线数据与指数日线数据合并

    Args:
        df1 (:class:`~pandas.DataFrame`): 股票数据。
        df2 (:class:`~pandas.DataFrame`): 指数数据。
        copy (bool): 是否使用copy后的数据。参考 :meth:`pandas.DataFrame.copy` 方法。默认为 True。
        rsuffix (str): 默认为 ``_index``。参考 :meth:`pandas.DataFrame.join` 中同名参数。
        how (str): 默认为 ``right``。参考 :meth:`pandas.DataFrame.join` 中同名参数。
        append_funcs (collections.OrderedDict): 附加其他列数据时的计算方法字典。

            - key值为需要附加的列名。
            - value值为可以为方法名称或者是[方法名,参数字典](详见 :mod:`.stock_ai.appender`)。

            关于附加列名的赋值方式：
                - 如果方法返回的是 :class:`pandas.Series`，用key值来赋值。
                - 如果方法返回的是 :class:`pandas.DataFrame`，会用返回的 `DataFrame` 中的列名+Key值赋值。

    See Also:
        :meth:`pandas.DataFrame.join`

    Examples:
        使用默认参数合并。

        >>> df_1 = data_processor.load_stock_daily('601398')
        >>> df_2 = data_processor.load_index_daily('399300')
        >>> wrapper.dataframe_merge(df_1, df_2).head()
        .            open  high  low  ...  down_count  volume_index  amount_index
        date                         ...
        2005-01-04   NaN   NaN  NaN  ...           0       74128.0  4.431976e+09
        2005-01-05   NaN   NaN  NaN  ...           0       71191.0  4.529207e+09
        2005-01-06   NaN   NaN  NaN  ...           0       62880.0  3.921015e+09
        2005-01-07   NaN   NaN  NaN  ...           0       72986.0  4.737468e+09
        2005-01-10   NaN   NaN  NaN  ...           0       57916.0  3.762931e+09

        附加列只传递方法名（无参数）

        >>> funcs = {'year': stock_ai.appender.calc_year}
        >>> df = wrapper.dataframe_merge(df_1, df_2, append_funcs=funcs)
        >>> df['year'].head()
        date
        2005-01-04    2005
        2005-01-05    2005
        2005-01-06    2005
        2005-01-07    2005
        2005-01-10    2005
        Name: year, dtype: int64

        附加列时传递参数的方式

        >>> def app(df, **kwargs):
        >>>    v = kwargs.pop('v', '123')
        >>>    return pd.Series(v, index=df.index)
        >>> new_col = 'cv'
        >>> new_value='123'
        >>> funcs={new_col: [app, {'v': new_value}]}
        >>> df = wrapper.dataframe_merge(df_1, df_2, append_funcs=funcs)
        >>> df[new_col].head()
        date
        2005-01-04    123
        2005-01-05    123
        2005-01-06    123
        2005-01-07    123
        2005-01-10    123
        Name: cv, dtype: object

    Returns:
        :class:`~pandas.DataFrame`: 合并后的数据。
    """
    how = kwargs.pop('rsuffix', 'right')
    rsuffix = kwargs.pop('rsuffix', '_index')
    append_funcs = kwargs.pop('append_funcs', {})
    df = df1.copy() if kwargs.pop('copy', True) else df1
    df = df.join(df2, rsuffix=rsuffix, how=how)
    for k, v in append_funcs.items():
        new_value = None
        if isinstance(v, (list, tuple)) and len(v) > 1:
            new_value = v[0](df, **v[1])
        else:
            new_value = v(df)
        if isinstance(new_value, pd.Series):
            new_value = new_value.to_frame(k)
        elif isinstance(new_value,pd.DataFrame):
            new_value=new_value.add_suffix('_' + k)
        else:
            raise NotImplementedError
        df = df.join(new_value)
    return df

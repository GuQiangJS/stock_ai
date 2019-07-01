import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import keras
import pandas as pd

sns.set()
matplotlib.rcParams['font.family'] = 'sans-serif',
matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def __plot_keras_history(d: dict, title, ylabel, xlabel):
    """训练历史可视化"""

    return sns.lineplot(data=pd.DataFrame.from_dict(d))
    #
    # plt.title(title)
    # plt.ylabel(ylabel)
    # plt.xlabel(xlabel)
    # plt.legend(d.keys(), loc='upper left')
    # for n, v in d.items():
    #     # plt.plot(v)
    # return plt


def plot_keras_history(his: keras.callbacks.History, **kwargs):
    """训练历史可视化

    Args:
        his (:class:`keras.callbacks.History`):
        show (bool): 是否显示。默认为True。

    Returns:
        :class:`matplotlib.pyplot`:
    """
    def _a(d:dict,name:str,k,v):
        if name not in d.keys():
            d[name]={}
        d[name][k]=v

    show = kwargs.pop('show', True)
    plots = []
    # # 绘制训练 & 验证的准确率值
    # acc = {}
    for n in his.history.keys():
        if 'val_' not in n:
            d={}
            _a(d, 'data', 'Train', his.history[n])
            if 'val_'+n in his.history.keys():
                _a(d, 'data', 'Test', his.history['val_'+n])
            d['ylabel'] = 'Value'
            d['xlabel'] = 'Epochs'
            d['title']=n
            plots.append(d)

    if plots:
        figSize_w=kwargs.pop('figSize_w',10)
        figSize_h=kwargs.pop('figSize_h',5)
        fig, axs = plt.subplots(nrows=len(plots),figsize=(figSize_w,len(plots)*figSize_h))
        for plot in plots:
            ax=sns.lineplot(data=pd.DataFrame.from_dict(plot['data']),
                         ax=axs[plots.index(plot)])
            ax.set_xlabel(plot['xlabel'])
            ax.set_ylabel(plot['ylabel'])
            ax.set_title(plot['title'])

    plt.tight_layout()
    if show:
        plt.show()
    return plt


def plot_daily_return_histogram(data, **kwargs):
    """绘制直方图

    Args:
        data (:class:`pandas.Series`): 日收益数据。
        bins (int): 参见 :func:`pandas.Series.hist` 中的同名参数。默认为 50.
        show (bool): 是否显示。默认为True。
        title (str): 标题。默认为 '日收益直方图'。
        suptitle (str): 标题。
        show_mean (bool): 绘制均值。默认为True。
        show_std (bool): 绘制标准差。默认为True。

    Returns:

    """
    bins = kwargs.pop('bins', 50)
    show = kwargs.pop('show', True)
    title = kwargs.pop('title', '日收益直方图')
    suptitle = kwargs.pop('suptitle', None)
    show_mean = kwargs.pop('show_mean', True)
    show_std = kwargs.pop('show_std', True)
    data.hist(bins=bins)

    if show_mean:
        mean = data.mean()
        plt.axvline(mean, color='g')
    if show_std:
        std = data.std()
        plt.axvline(std, color='r', linestyle='dashed')
        plt.axvline(-std, color='r', linestyle='dashed')
    if title:
        plt.title(title)
    if suptitle:
        plt.suptitle(suptitle)
    if show:
        plt.show()
    return plt

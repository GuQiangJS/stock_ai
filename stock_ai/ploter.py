import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import keras

sns.set()
matplotlib.rcParams['font.family'] = 'sans-serif',
matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def __plot_keras_history(d: dict, title, ylabel, xlabel):
    """训练历史可视化"""
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend(d.keys(), loc='upper left')
    for v in d.values():
        plt.plot(v)
    return plt


def plot_keras_history(his: keras.callbacks.History, **kwargs):
    """训练历史可视化

    Args:
        his (:class:`keras.callbacks.History`):
        show (bool): 是否显示。默认为True。

    Returns:
        :class:`matplotlib.pyplot`:
    """
    show = kwargs.pop('show', True)
    result = []
    # 绘制训练 & 验证的准确率值
    acc = {}
    if 'acc' in his.history:
        acc['训练'] = his.history['acc']
    if 'val_acc' in his.history:
        acc['测试'] = his.history['val_acc']
    if acc:
        plt = __plot_keras_history(acc, '模型准确性', '准确性', '轮次')
        result.append(plt)
        if show:
            plt.show()
    loss = {}
    if 'loss' in his.history:
        loss['训练'] = his.history['loss']
    if 'val_loss' in his.history:
        loss['测试'] = his.history['val_loss']
    if loss:
        plt = __plot_keras_history(loss, '模型损失值', '损失值', '轮次')
        result.append(plt)
        if show:
            plt.show()

    return result


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

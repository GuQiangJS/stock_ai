from stock_ai.data_processor import load_stock_block
import logging
import pandas as pd
from stock_ai.module import StockCN
from stock_ai.util import str2date


def sharpe_ratio(**kwargs):
    """计算指定行业的夏普比率

    Examples:
        >>> sharpe_ratio()
        .         601988     601398     600036  ...     600000     000001     600015
        2005        NaN        NaN   7.862071  ...   0.065330   5.211949   2.752302
        2006   5.423067   3.269645   2.508599  ...   1.674587   1.714323   2.387494
        2007   5.520200   3.744622   2.646470  ...   2.711249   2.535364   2.678983
        2008   2.678578   3.136494   2.646753  ...   2.194612   2.218519   2.306832
        2009   4.409181   4.293783   4.531678  ...   3.496624   3.730579   6.084146
        2010   7.478264  10.313356  12.802671  ...   9.754310   6.521051  12.330933
        2011  13.690974  22.193504  11.786194  ...  10.525052  16.241037  10.762463
        2012  20.270051  15.875993  10.438670  ...   9.520852   9.550869   6.414574
        2013  18.892670  25.002890  11.693822  ...   8.823424   7.255290   9.936727
        2014   4.025824   6.332456   6.430317  ...   5.487892   5.728920   5.893492
        2015   8.052619  10.798641   9.794130  ...   8.867595   6.183538   8.009029
        2016  14.122428  14.663031  11.673594  ...  25.512644  19.262027  13.585451
        2017   9.030946   6.170423   5.266668  ...  22.523839   5.819815  21.213616
        2018  11.248927   9.429071  16.268102  ...   9.185897   7.025021   9.874763

    Args:
        start: 开始日期。默认为 '2005-01-01'
        end: 结束日期。默认为 '2018-12-31'
        block: 板块名称。默认为 '银行'。
        值参考 :func:stock_ai.data_processor.load_stock_block` 返回值中的 'blockname' 列。
        n: 最少统计 n 年的数据。数据少于 n 年，不包含在统计结果中。默认为 end-start 中取年数。

    Returns:
        :class:`~pandas.DataFrame`: 板块夏普比率报表。

    """
    start = kwargs.pop('start', '2005-01-01')
    end = kwargs.pop('end', '2018-12-31')
    block = kwargs.pop('block', '银行')
    blocks = load_stock_block()
    blocks = blocks[blocks['blockname'] == block]
    logging.debug('show_all_sharpe_ratio - ' + block)
    result = pd.DataFrame()
    n = kwargs.pop('n', (str2date(end).year - str2date(start).year))
    for stock in blocks['code'].unique():
        stock = StockCN(code=stock)
        logging.debug(stock)
        df = stock.get_sharpe_ratio(start=start,
                                    end=end).rename(columns={0: stock.code})
        if df.shape[0] >= n:
            result = result.join(df, how='outer')
    return result

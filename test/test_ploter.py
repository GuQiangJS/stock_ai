import pytest
from test import get_stock_daily
from test import get_index_daily
from test import is_travis
from stock_ai import calcs
from stock_ai import ploter


@pytest.mark.skipif(condition=is_travis,
                    reason="Skipping this test on Travis CI.")
def test_plot_daily_return_histogram():
    close = calcs.calc_daily_return(get_stock_daily()['close'])
    ploter.plot_daily_return_histogram(close, bin=100)
    print('Kurtosis:{}'.format(close.kurtosis()))

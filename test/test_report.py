from stock_ai.report import sharpe_ratio
from stock_ai.module import StockCN
import pytest
from test import is_travis


@pytest.mark.skipif(is_travis, reason="Skipping this test on Travis CI.")
def test_sharpe_ratio():
    df = sharpe_ratio()
    print(df)
    df_describe = df.describe()
    print(df_describe)
    print(df_describe.T.sort_values('mean', ascending=False))
    print(df_describe.T.sort_values('std'))
    for code in df.columns:
        stock = StockCN(code)
        print('{0}:{1}.'.format(
            code,
            stock.get_cum_returns(start='2005-01-01', end='2018-12-31')[-1]))

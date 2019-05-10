import test
import os
from stock_ai.data_reader import DailyReader


def test_read_from_csv():
    for code in test.test_stocks:
        filepath = test._get_stock_daily_filepath(code)
        if os.path.exists(filepath):
            df = DailyReader().fetch_stock(code, online=False)
            df_csv = test.read_stock_daily_from_csv(code)
            assert df.equals(df_csv)

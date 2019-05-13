import test
import os
import pytest
from stock_ai import data_processor
from pandas.testing import assert_frame_equal
from pandas.testing import assert_index_equal


@pytest.mark.skipif("TRAVIS" in os.environ and os.environ["TRAVIS"] == "true",
                    reason="Skipping this test on Travis CI.")
def test_read_from_csv():
    for code in test.test_stocks:
        filepath = test._get_stock_daily_filepath(code)
        if os.path.exists(filepath):
            df = data_processor.load_stock_daily(code, online=False)
            df_csv = test.read_stock_daily_from_csv(code)
            assert_frame_equal(df, df_csv)
            assert_index_equal(df.index, df_csv.index)

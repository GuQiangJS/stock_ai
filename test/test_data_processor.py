import test
from stock_ai import data_processor
from pandas.testing import assert_frame_equal


def test_load_stock_daily_csv():
    code = '601398'
    df_csv = test.read_stock_daily_from_csv(code)
    assert not df_csv.empty
    first_line = df_csv.iloc[0]
    last_line = df_csv.iloc[-1]
    assert 1.9921332735028636 == first_line['open']
    assert 484066272.0 == last_line['amount']


def test_load_stock_daily_online():
    code = '601398'
    end = '2019-01-24'
    online_columns = ['open', 'high', 'low', 'close', 'volume', 'amount']
    df_online1 = data_processor.load_stock_daily_online(code,
                                                        '1990-01-01',
                                                        end=end,
                                                        columns=online_columns)
    assert not df_online1.empty
    df_online2 = data_processor.load_stock_daily(code, end=end, online=True)
    assert not df_online2.empty
    assert df_online1.equals(df_online2)
    assert assert_frame_equal(df_online1, df_online2)

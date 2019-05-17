from stock_ai.module import IndexCN
import pytest
from stock_ai.util import date2str
import numpy as np
import os
from test import is_travis

def test_getdaily_online():
    d1 = _test_index_daily(IndexCN('399300', getdaily_online=True))
    d2 = _test_index_daily(IndexCN('000300', getdaily_online=True))
    assert np.array_equal(d1.columns.values, d2.columns.values)
    assert len(d1) == len(d2)


@pytest.mark.skipif(is_travis, reason="Skipping this test on Travis CI.")
def test_getdaily_mongodb():
    _test_index_daily(IndexCN('399300'))


def _test_index_daily(index: IndexCN):
    ipo_date = '2005-01-04'
    d = index.get_daily()
    assert not d.empty
    assert date2str(d.index[0]) == ipo_date
    return d


def test_construct():
    assert not IndexCN('399300').getdaily_online

from QUANTAXIS.QAUtil import QA_util_format_date2str as date2str
from QUANTAXIS.QAUtil import QA_util_date_int2str as int2str
import datetime


def str2date(str_date):
    """将时间格式字符串转换为`datetime.datetime`类型。

    Args:
        str_date (str): 日期字符串。格式为 `%Y-%m-%d`。

    Returns:
        datetime.datetime:

    See Also:
        `date2str`。

    """
    return datetime.datetime.strptime(str_date, '%Y-%m-%d')

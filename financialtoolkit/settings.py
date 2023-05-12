"""Settings Module"""
__docformat__ = "numpy"

from typing import Final

import pkg_resources

_BALANCE_SHEET_FORMAT_LOCATION: Final[str] = pkg_resources.resource_stream(
    __name__, "normalization/balance.csv"
).name
_INCOME_FORMAT_LOCATION: Final[str] = pkg_resources.resource_stream(
    __name__, "normalization/income.csv"
).name
_CASH_FLOW_FORMAT_LOCATION: Final[str] = pkg_resources.resource_stream(
    __name__, "normalization/cash.csv"
).name

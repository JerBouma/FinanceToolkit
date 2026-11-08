"""Risk Helpers Module"""

__docformat__ = "google"

from collections.abc import Mapping

import numpy as np
import pandas as pd

from financetoolkit.utilities.statistics_model import WITHIN_PERIOD_TRANSLATION


def determine_within_historical_data(
    period: str,
    historical_data: Mapping[str, pd.DataFrame],
    intraday_period: str | None,
) -> pd.DataFrame:
    """
    Reshapes the historical data of a period into a (period, observation) multi-index, the
    format the "within period" calculations of the Risk, Performance and Econometrics
    controllers operate on. The intraday and daily periods nest the intraday bars, the other
    periods the daily observations; only the data the period nests is read, so a
    LazyDict (financetoolkit.helpers) retrieves nothing else.

    Args:
        period (str): The period, one of the keys of WITHIN_PERIOD_TRANSLATION.
        historical_data (Mapping[str, pd.DataFrame]): The historical data with a PeriodIndex
            under the "daily" and "intraday" keys.
        intraday_period (str | None): The intraday frequency (e.g. "1min", "1hour") used to
            look up the outer resampling symbol of the intraday period.

    Raises:
        KeyError: For the intraday period without an intraday_period, and for the daily
            period without intraday data, which have no within period data.

    Returns:
        pd.DataFrame: The historical data with a (period, observation) multi-index.
    """
    if period not in WITHIN_PERIOD_TRANSLATION or (
        period == "intraday" and not intraday_period
    ):
        raise KeyError(period)

    intraday_historical_data = (
        historical_data["intraday"]
        if period in ("intraday", "daily")
        else pd.DataFrame()
    )

    # Without intraday observations there is nothing to nest inside a single day, so the daily period would collapse to one observation per group.  # noqa: E501
    if period == "daily" and intraday_historical_data.empty:
        raise KeyError(period)

    # The intraday entry maps each intraday_period to a symbol; every other entry is the
    # symbol itself.
    symbol = WITHIN_PERIOD_TRANSLATION[period]
    period_symbol = (
        symbol[intraday_period]
        if isinstance(symbol, dict) and intraday_period is not None
        else symbol
    )
    if not isinstance(period_symbol, str):
        raise TypeError(f"No period symbol resolved for {period}.")

    source_data = (
        intraday_historical_data
        if not intraday_historical_data.empty
        else historical_data["daily"]
    )
    dates = pd.DatetimeIndex(source_data.to_timestamp().index)

    # The daily period nests intraday observations inside each day, so like the intraday period it needs a minute-level inner index rather than a daily one.  # noqa: E501
    inner_freq = "min" if period in ("intraday", "daily") else "D"
    period_data = source_data.copy()
    period_data.index = pd.MultiIndex.from_arrays(
        [dates.to_period(period_symbol), dates.to_period(inner_freq)]
    )

    return period_data


def as_float(value: float | np.ndarray | pd.Series | pd.DataFrame) -> float:
    """
    Reduces a single-value estimate to a float. The Student-t and Extreme Value Theory
    Value at Risk and Conditional Value at Risk return a length one array or Series when
    given a Series, which a rolling apply does not accept as its return value.

    Args:
        value (float | np.ndarray | pd.Series | pd.DataFrame): The estimate to reduce.

    Returns:
        float: The estimate as a float.
    """
    return float(np.asarray(value, dtype=float).reshape(-1)[0])

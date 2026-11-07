"""Models Helpers Module"""

__docformat__ = "google"


from collections.abc import Callable

import pandas as pd

from financetoolkit.utilities.logger_model import get_logger

logger = get_logger()

# pylint: disable=protected-access

PERIOD_TRANSLATION: dict[str, str] = {
    "weekly": "W",
    "monthly": "M",
    "quarterly": "Q",
    "yearly": "Y",
}


def determine_within_historical_data(
    period: str, daily_historical_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Nests the daily historical data under the week, month, quarter or year it falls in, the
    "within period" data the models estimate a beta per period from.

    Args:
        period (str): The period (weekly, monthly, quarterly or yearly).
        daily_historical_data (pd.DataFrame): The daily historical data, with a PeriodIndex.

    Raises:
        KeyError: If the period is not weekly, monthly, quarterly or yearly.

    Returns:
        pd.DataFrame: The daily data with the period and the day as the index levels.
    """
    symbol = PERIOD_TRANSLATION[period]
    dates = pd.DatetimeIndex(daily_historical_data.to_timestamp().index)

    period_data = daily_historical_data.copy()
    period_data.index = pd.MultiIndex.from_arrays(
        [dates.to_period(symbol), dates.to_period("D")]
    )

    return period_data


def rates_per_ticker(
    value: float | list | dict[str, float] | None,
    tickers: list[str],
    name: str,
    default: Callable[[], pd.Series] | None = None,
) -> dict[str, float]:
    """
    A rate per ticker from what a valuation method is given: one number for every ticker, a
    list in the order of the tickers or a dictionary by ticker. Where no rate is given (None,
    or a ticker missing from the dictionary), the rate comes from `default`, which returns the
    company's own rate per ticker; it is only called when needed, since it may load data.

    Args:
        value (float | list | dict[str, float] | None): The rate(s) given.
        tickers (list[str]): The tickers, in order.
        name (str): The name of the rate, for the error messages.
        default (Callable[[], pd.Series] | None, optional): Returns the rate per ticker for
            the tickers without a given rate. Defaults to None, which requires every rate
            to be given.

    Raises:
        ValueError: If a list does not have a rate for every ticker, or a rate is missing
            while there is no default.

    Returns:
        dict[str, float]: The rate per ticker (NaN where the default has no value).
    """
    if isinstance(value, list):
        if len(value) != len(tickers):
            raise ValueError(
                f"The {name} list must have a value for each of the {len(tickers)} tickers."
            )
        return {
            ticker: float(rate) for ticker, rate in zip(tickers, value, strict=True)
        }

    given = (
        {ticker: value.get(ticker) for ticker in tickers}
        if isinstance(value, dict)
        else dict.fromkeys(tickers, value)
    )
    missing = [ticker for ticker, rate in given.items() if rate is None]

    if missing and default is None:
        raise ValueError(f"No {name} given for: {', '.join(missing)}.")

    calculated = default().reindex(tickers) if missing and default else pd.Series()

    return {
        ticker: float(calculated[ticker]) if rate is None else float(rate)
        for ticker, rate in given.items()
    }


def exceeds_growth(
    ticker: str, rate: float, growth: float, rate_name: str = "discount rate"
) -> bool:
    """
    Whether a ticker's discount rate exceeds the growth rate it keeps forever, which the
    terminal value of a perpetuity needs to be finite. A ticker where it doesn't is logged
    and left out, so the other tickers are still valued.

    Args:
        ticker (str): The ticker, for the log.
        rate (float): The discount rate.
        growth (float): The growth rate it keeps forever.
        rate_name (str, optional): The name of the discount rate, for the log. Defaults to
            "discount rate".

    Returns:
        bool: True when the rate exceeds the growth or either is NaN (the valuation is then
        NaN by itself), False otherwise.
    """
    if pd.isna(rate) or pd.isna(growth) or rate > growth:
        return True

    logger.warning(
        "%s is left out: its %s (%.4f) does not exceed the growth rate it keeps forever "
        "(%.4f), so its value is infinite. Pass a higher rate or a lower growth rate.",
        ticker,
        rate_name,
        rate,
        growth,
    )
    return False

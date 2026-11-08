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
    given = _given_rates(value, tickers, name, default is not None)
    missing = [ticker for ticker, rate in given.items() if rate is None]
    calculated = default().reindex(tickers) if missing and default else pd.Series()

    return {
        ticker: float(calculated[ticker]) if rate is None else float(rate)
        for ticker, rate in given.items()
    }


def rates_per_period(
    value: float | list | dict[str, float] | None,
    tickers: list[str],
    periods: pd.Index,
    name: str,
    default: Callable[[], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """
    A rate per ticker and period, for valuations that value every period: a given rate (one
    number, a list in the order of the tickers or a dictionary by ticker) is the same in
    every period, while the `default` gives each period the company's own rate of that
    period, so that no period is valued with a rate that was only known later.

    Args:
        value (float | list | dict[str, float] | None): The rate(s) given.
        tickers (list[str]): The tickers, in order.
        periods (pd.Index): The periods.
        name (str): The name of the rate, for the error messages.
        default (Callable[[], pd.DataFrame] | None, optional): Returns the rate per ticker
            (rows) and period (columns) for the tickers without a given rate. Defaults to
            None, which requires every rate to be given.

    Raises:
        ValueError: If a list does not have a rate for every ticker, or a rate is missing
            while there is no default.

    Returns:
        pd.DataFrame: The rate per ticker (rows) and period (columns), NaN where the default
        has no value.
    """
    given = _given_rates(value, tickers, name, default is not None)
    missing = [ticker for ticker, rate in given.items() if rate is None]
    calculated = (
        default().reindex(index=tickers, columns=periods)
        if missing and default
        else pd.DataFrame(index=tickers, columns=periods, dtype=float)
    )

    return pd.DataFrame(
        [
            (
                calculated.loc[ticker].to_numpy(dtype=float)
                if rate is None
                else [float(rate)] * len(periods)
            )
            for ticker, rate in given.items()
        ],
        index=tickers,
        columns=periods,
    )


def _given_rates(
    value: float | list | dict[str, float] | None,
    tickers: list[str],
    name: str,
    has_default: bool,
) -> dict[str, float | None]:
    """The rate given per ticker, None where it is not given."""
    if isinstance(value, list):
        if len(value) != len(tickers):
            raise ValueError(
                f"The {name} list must have a value for each of the {len(tickers)} tickers."
            )
        return dict(zip(tickers, value, strict=True))

    given = (
        {ticker: value.get(ticker) for ticker in tickers}
        if isinstance(value, dict)
        else dict.fromkeys(tickers, value)
    )
    missing = [ticker for ticker, rate in given.items() if rate is None]

    if missing and not has_default:
        raise ValueError(f"No {name} given for: {', '.join(missing)}.")

    return given


def exceeds_growth_per_period(
    rates: pd.DataFrame, growth: pd.DataFrame, rate_name: str = "discount rate"
) -> pd.DataFrame:
    """
    Per ticker (rows) and period (columns), whether the discount rate exceeds the growth
    rate kept forever, as `exceeds_growth` checks for one rate. The periods where it
    doesn't are logged per ticker and left out (NaN), so the other periods are still valued.

    Args:
        rates (pd.DataFrame): The discount rate per ticker and period.
        growth (pd.DataFrame): The growth rate kept forever, on the same ticker and periods.
        rate_name (str, optional): The name of the discount rate, for the log. Defaults to
            "discount rate".

    Returns:
        pd.DataFrame: True where the rate exceeds the growth or either is NaN, False otherwise.
    """
    exceeds = rates.isna() | growth.isna() | (rates > growth)

    for ticker in exceeds.index[~exceeds.all(axis=1)]:
        left_out = exceeds.columns[~exceeds.loc[ticker]]
        logger.warning(
            "%s is left out in %d period(s) (%s): its %s does not exceed the growth rate it "
            "keeps forever there, so its value is infinite. Pass a higher rate or a lower "
            "growth rate.",
            ticker,
            len(left_out),
            ", ".join(str(period) for period in left_out),
            rate_name,
        )

    return exceeds


def trailing_within_periods(
    within: pd.DataFrame | pd.Series, trailing: int
) -> pd.DataFrame | pd.Series:
    """
    The "within period" data (period, day) with every period holding the days of the
    `trailing` periods that end with it, so that a statistic per period, such as a beta,
    covers that trailing span: with quarterly periods and trailing=4, a year of days.

    Args:
        within (pd.DataFrame | pd.Series): Data with the period and the day as index levels.
        trailing (int): The number of periods each period covers.

    Returns:
        pd.DataFrame | pd.Series: The same data, with each period's rows covering the trailing
        periods (calendar periods, so a gap in the data is not filled from earlier periods).
    """
    period_level = within.index.get_level_values(0)
    return pd.concat(
        {
            period: within[
                (period_level > period - trailing) & (period_level <= period)
            ].droplevel(0)
            for period in period_level.unique()
        }
    )


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

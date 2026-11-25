"""Risk Helpers Module"""

__docformat__ = "google"

from collections.abc import Mapping

import numpy as np
import pandas as pd

from financetoolkit.utilities.error_model import check_period, validate_window
from financetoolkit.utilities.logger_model import get_logger
from financetoolkit.utilities.statistics_model import (
    PERIOD_TRANSLATION,
    PERIODS_PER_YEAR,
    VOLATILITY_WINDOW_TRANSLATION,
    WITHIN_PERIOD_TRANSLATION,
    convert_annualized_rate_to_period,
    to_period_index,
)

logger = get_logger()

# The day of the week of a Saturday and a Sunday in pandas (Monday is 0).
SATURDAY = 5
SUNDAY = 6

# The share of the dates that Saturdays and Sundays must each exceed for daily dates to be calendar days rather than trading days: one in seven (about 14%) for a cryptocurrency, none or a fraction of a percent for a stock exchange.  # noqa: E501
WEEKEND_DAY_SHARE = 0.1


def check_rolling(
    rolling: int | None,
    historical_data: Mapping[str, pd.DataFrame],
    period: str,
    minimum: int = 2,
) -> None:
    """
    Check the rolling window of a Risk or Performance method: a whole number of at least
    minimum, the smallest window the measure has a value for (two for a spread, ratio or
    regression, more for the higher moments and the fractal measures). A window at least as
    long as the periods with a return is allowed but gives at most one value (NaN throughout
    when longer), which is logged once, naming the tickers (and the Benchmark) whose own returns are fewer
    when the other tickers have enough.

    Args:
        rolling (int | None): The rolling window, None for none.
        historical_data (Mapping[str, pd.DataFrame]): The historical data per period, with
            the daily data under "daily" and the intraday data under "intraday".
        period (str): The period the window counts.
        minimum (int, optional): The smallest window the measure has a value for. Defaults
            to 2.

    Raises:
        InvalidArgumentError: If rolling is not a whole number of at least minimum.
    """
    validate_window("rolling", rolling, minimum)

    if rolling is None:
        return

    # The window counts returns, so the periods are those with a return: the first period
    # of the data has a price but no return (a range starting on 2019-12-31 holds four
    # years and three yearly returns). Each ticker is rolled over its own returns, so a
    # ticker listed later is counted on its own, without the Benchmark's longer history.
    period_data = historical_data.get(period)
    if period_data is None:
        return

    # The Benchmark is named with the tickers when its own returns are fewer (a benchmark listed later than they are), since the betas and other measures against it are then empty too.  # noqa: E501
    returns = period_data.get("Return", period_data)
    if isinstance(returns, pd.DataFrame) and returns.columns.nlevels == 1:
        counts = returns.notna().sum()
        tickers = counts.drop("Benchmark", errors="ignore")
        ticker_counts = tickers if not tickers.empty else counts
    else:
        counts = ticker_counts = pd.Series({period: len(returns.dropna(how="all"))})

    short = counts[counts <= rolling]
    if short.empty:
        return

    if (ticker_counts <= rolling).all():
        logger.warning(
            "The rolling window of %s %s periods is at least as long as the %s %s returns in the "
            "data, so it gives at most one value (NaN throughout when longer). Use a shorter "
            "rolling window or a longer date range.",
            rolling,
            period,
            int(ticker_counts.max()),
            period,
        )
    else:
        logger.warning(
            "The rolling window of %s %s periods is at least as long as the %s returns of %s, so "
            "it gives at most one value for them (NaN throughout when longer). Use a shorter "
            "rolling window or an earlier start date.",
            rolling,
            period,
            period,
            ", ".join(f"{ticker} ({count})" for ticker, count in short.items()),
        )


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
        InvalidArgumentError: For a period that is not a key of WITHIN_PERIOD_TRANSLATION.
        KeyError: For the intraday period without an intraday_period, and for the daily
            period without intraday data, which have no within period data.

    Returns:
        pd.DataFrame: The historical data with a (period, observation) multi-index.
    """
    # An unknown period is a mistake in the call; a known one without data is missing data.
    check_period(period, list(WITHIN_PERIOD_TRANSLATION))
    if period == "intraday" and not intraday_period:
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


def mask_non_trading_days(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    The `column` prices of a historical data frame on the dates each ticker traded.

    On a date only another ticker or the benchmark traded, a ticker's last price is
    carried forward as a flat bar without a Return (see
    `historical_model.fill_non_trading_days`). A measure on the prices themselves (a daily
    range, a price change, the number of observations) would count it as a day without a
    move, so the price is NaN there, as its Return is. A ticker's first price has no
    Return by construction and is kept, as is everything after its last Return.

    Args:
        frame (pd.DataFrame): The historical data of one period, with "Return" and `column`.
        column (str): The price column, e.g. "Close".

    Returns:
        pd.DataFrame: The prices, NaN on the dates the ticker did not trade.
    """
    prices = frame[column]
    has_return = frame["Return"].reindex(columns=prices.columns).notna()
    after_first_price = prices.notna().cumsum() > 1
    up_to_last_return = has_return[::-1].cummax()[::-1]

    return prices.mask(~has_return & after_first_price & up_to_last_return)


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


def trades_on_weekends(dates: pd.PeriodIndex) -> bool:
    """
    Whether the daily dates are calendar days rather than trading days, as those of a
    cryptocurrency that trades on every day of the week: Saturdays and Sundays each make
    up more than 10% of the dates (one in seven of the dates of a cryptocurrency). The
    trading days of an exchange that opens from Sunday to Thursday, which has no Saturdays,
    or of a stock with a one-off Saturday session stay trading days, about 252 a year.

    Args:
        dates (pd.PeriodIndex): The daily dates with an observation.

    Returns:
        bool: True when Saturdays and Sundays are each more than 10% of the dates.
    """
    if len(dates) == 0:
        return False

    day_of_week = np.asarray(dates.dayofweek)

    return bool(
        (day_of_week == SATURDAY).mean() > WEEKEND_DAY_SHARE
        and (day_of_week == SUNDAY).mean() > WEEKEND_DAY_SHARE
    )


def get_observations_per_year(returns: pd.Series | pd.DataFrame) -> float | pd.Series:
    """
    The number of daily returns in a year per column, to scale a daily Variance or
    Volatility to a longer period: 252 for a column that trades on weekdays only, and 365
    for a column that also has returns on a Saturday or Sunday, such as a cryptocurrency
    that trades every calendar day. Counting the returns of each year instead would give
    a stock between 250 and 253 depending on its holidays, so a ticker on an exchange's
    trading days keeps the 252 convention.

    Args:
        returns (pd.Series | pd.DataFrame): The daily returns with a daily PeriodIndex,
            NaN on the dates a ticker has no return.

    Returns:
        float | pd.Series: The observations per year, a float for a Series and indexed by
        column for a DataFrame.
    """
    # A range without data can lose its PeriodIndex on the way (an empty frame gets a
    # RangeIndex), and without returns there are no weekend dates either.
    dates = (
        to_period_index(returns.index)
        if len(returns.index)
        else pd.PeriodIndex([], freq="D")
    )

    def per_year(column_returns: pd.Series) -> float:
        return (
            365.0
            if trades_on_weekends(dates[column_returns.notna().to_numpy()])
            else 252.0
        )

    if isinstance(returns, pd.Series):
        return per_year(returns)

    return pd.Series(
        {column: per_year(returns[column]) for column in returns.columns},
        dtype=float,
    )


def get_days_per_period(
    returns: pd.Series | pd.DataFrame,
    period: str,
    groups: pd.Series | np.ndarray | None = None,
) -> float | pd.Series:
    """
    The number of daily returns in one period of each column, the multiplier that scales
    a daily Variance to the period (and its square root a daily Volatility): the
    observations per year (see `get_observations_per_year`) divided by the periods per
    year, so 252 / 52 for a weekly figure of a stock and 365 / 52 for one of a
    cryptocurrency.

    Args:
        returns (pd.Series | pd.DataFrame): The daily returns (or another daily estimator
            term) with a daily DatetimeIndex or PeriodIndex.
        period (str): The period to scale to: weekly, monthly, quarterly or yearly.
        groups (pd.Series | np.ndarray | None, optional): Explicit group labels used instead
            of calendar periods, for data without a date index. Its rows are not dates, so
            the trading day convention of 252 days a year applies. Defaults to None.

    Returns:
        float | pd.Series: The daily returns per period, a float for a Series or for
        `groups`, otherwise indexed by column.
    """
    if groups is not None:
        return VOLATILITY_WINDOW_TRANSLATION[period]

    return get_observations_per_year(returns) / PERIODS_PER_YEAR[period]


def get_observed_span(
    returns: pd.Series | pd.DataFrame, by_period: bool = False
) -> pd.Series | pd.DataFrame:
    """
    Whether each row lies between a column's first and last return. A drawdown measure
    compounds missing returns as 0, which is right for a gap inside the history but would
    add a drawdown of exactly 0 (or a frozen drawdown) for every row before a ticker's
    first return (a later listing) or after its last return (a delisting), diluting an
    average or a quantile over the rows.

    Args:
        returns (pd.Series | pd.DataFrame): The returns.
        by_period (bool, optional): Whether the span restarts in every period of a
            (period, date) multi-index. Defaults to False.

    Returns:
        pd.Series | pd.DataFrame: True on the rows from a column's first to its last return.
    """
    observed = returns.notna()

    if by_period:
        started = observed.groupby(level=0, sort=False).cummax()
        ended = observed[::-1].groupby(level=0, sort=False).cummax()[::-1]
    else:
        started = observed.cummax()
        ended = observed[::-1].cummax()[::-1]

    return started & ended


def align_annualized_rate(
    annualized_rate: pd.Series, dates: pd.PeriodIndex
) -> pd.Series:
    """
    The annualized risk free rate on each of the dates: a date without a quote (a weekend or
    a holiday of the Treasury market) takes the most recent earlier quote, and the dates
    before the first quote take the first quote.

    Args:
        annualized_rate (pd.Series): The annualized risk free rate with a daily PeriodIndex.
        dates (pd.PeriodIndex): The daily dates to align the rate to.

    Returns:
        pd.Series: The annualized risk free rate per date.
    """
    rate = annualized_rate.sort_index()
    rate = rate[~rate.index.duplicated(keep="last")]

    return rate.reindex(rate.index.union(dates)).ffill().bfill().reindex(dates)


def get_calendar_days_between(dates: pd.PeriodIndex) -> np.ndarray:
    """
    The number of calendar days since the date before, 1 for the first date.

    Args:
        dates (pd.PeriodIndex): The daily dates, in order.

    Returns:
        np.ndarray: The calendar days since the date before, per date.
    """
    ordinals = dates.asfreq("D").asi8

    return (
        np.diff(ordinals, prepend=ordinals[0] - 1).astype(float)
        if len(dates)
        else np.array([])
    )


def get_risk_free_rate_per_date(
    annualized_rate: pd.Series, returns: pd.DataFrame
) -> pd.Series:
    """
    The risk free return earned on each date of the daily returns since the date before, from
    the annualized risk free rate (such as a Treasury yield) quoted on that date, the rate
    the daily returns are compared with. The longer periods compound the annualized rate
    over the calendar days they span instead (see `compound_risk_free_rate`).

    Dates that are trading days (no returns on a Saturday or Sunday) follow the trading day
    convention of 252 days a year, so a year of trading days earns the annualized rate.
    Once a ticker trades on weekends as well, such as a cryptocurrency, the dates are
    calendar days and the rate accrues over the calendar days since the date before, on
    365 days a year.

    The formula is as follows:

        - Trading Days: Risk Free Rate per Date = (1 + Annualized Rate)^(1 / 252) - 1
        - Calendar Days: Risk Free Rate per Date = (1 + Annualized Rate)^(Days Since the Date Before / 365) - 1

    Args:
        annualized_rate (pd.Series): The annualized risk free rate with a daily PeriodIndex, see
            `align_annualized_rate`.
        returns (pd.DataFrame): The daily returns with a daily PeriodIndex, one column per ticker.

    Returns:
        pd.Series: The risk free return per date of the returns, as a decimal.
    """
    dates = returns.index
    rate = align_annualized_rate(annualized_rate, dates)

    if trades_on_weekends(dates[returns.notna().any(axis=1).to_numpy()]):
        return (1 + rate) ** (get_calendar_days_between(dates) / 365) - 1

    return (1 + rate) ** (1 / 252) - 1


def get_risk_free_rate_per_observation(
    annualized_rate: pd.Series, returns: pd.DataFrame
) -> pd.DataFrame:
    """
    The risk free return each ticker earns over the period of each of its returns, from the
    annualized risk free rate quoted on the date of the return.

    A ticker that trades on trading days earns a trading day of the rate with every return,
    on 252 trading days a year, whichever other tickers the data holds: its measures use
    its own trading days. A ticker that also trades on weekends, such as a cryptocurrency,
    earns the rate over the calendar days since its previous return, on 365 days a year,
    so it earns the annualized rate over a year rather than about 365 / 252 times it.

    The formula is as follows:

        - Trading Days: Risk Free Rate per Observation = (1 + Annualized Rate)^(1 / 252) - 1
        - Weekends Included: Risk Free Rate per Observation = (1 + Annualized Rate)^(Days Since the Previous
        Observation / 365) - 1

    Args:
        annualized_rate (pd.Series): The annualized risk free rate with a daily PeriodIndex, see
            `align_annualized_rate`.
        returns (pd.DataFrame): The daily returns with a daily PeriodIndex, one column per ticker,
            NaN on the dates a ticker has no return.

    Returns:
        pd.DataFrame: The risk free return per observation, shaped as the returns and NaN where
        the returns are NaN. The first observation of a ticker trading on weekends earns one
        calendar day.
    """
    rate = align_annualized_rate(annualized_rate, returns.index)
    risk_free_rate = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)

    for position in range(returns.shape[1]):
        observed = returns.iloc[:, position].notna().to_numpy()
        dates = returns.index[observed]
        exponent = (
            get_calendar_days_between(dates) / 365
            if trades_on_weekends(dates)
            else 1 / 252
        )
        risk_free_rate.iloc[observed, position] = (
            (1 + rate[observed]) ** exponent - 1
        ).to_numpy()

    return risk_free_rate


def compound_risk_free_rate(
    annualized_rate: pd.Series, returns: pd.DataFrame, period: str
) -> pd.Series:
    """
    The risk free return of each period: the annualized risk free rate compounded over the
    calendar days between the dates in the period that hold a return, each date earning the
    rate quoted on it over the days since the date before on 365 days a year. The return of
    a period runs from the last date of the period before to the last date of the period, so
    a full year earns the annualized rate whether its dates are the trading days of a stock
    or every calendar day of a cryptocurrency, and the rate of a period does not depend on
    which tickers the data holds. A period the data covers in part, such as the current year,
    earns the risk free rate of the days it covers rather than of a full period.

    The formula is as follows:

        - Period Risk Free Rate = PRODUCT((1 + Annualized Rate)^(Days Since the Date Before / 365)) - 1,
          over the dates in the period

    Args:
        annualized_rate (pd.Series): The annualized risk free rate with a daily PeriodIndex, see
            `align_annualized_rate`.
        returns (pd.DataFrame): The daily returns with a daily PeriodIndex, one column per ticker.
            The first date with a return accrues from the date before it (the date of the
            first price), or over one day when it is the first date.
        period (str): The period to compound to (weekly, monthly, quarterly or yearly).

    Returns:
        pd.Series: The risk free return per period, as a decimal.
    """
    observed = returns.notna().any(axis=1).to_numpy()
    positions = np.flatnonzero(observed)

    if not len(positions):
        return pd.Series(dtype=float)

    ordinals = returns.index.asfreq("D").asi8
    first = positions[0]
    days_since_date_before = np.diff(
        ordinals[positions],
        prepend=ordinals[first - 1] if first > 0 else ordinals[first] - 1,
    ).astype(float)

    dates = returns.index[positions]
    rate = align_annualized_rate(annualized_rate, dates)
    growth = (1 + rate) ** (days_since_date_before / 365)

    return growth.groupby(dates.asfreq(PERIOD_TRANSLATION[period])).prod() - 1


def get_risk_free_rate_per_ticker(
    annualized_rate_data: Mapping[str, pd.Series],
    historical_data: Mapping[str, pd.DataFrame],
    period: str,
) -> pd.DataFrame:
    """
    The risk free return per period of each ticker (and the Benchmark), the rate subtracted
    from its returns of that period, on the ticker's own calendar.

    For period="daily" this is the risk free return over each of a ticker's returns: a
    trading day of the rate for a ticker that trades on trading days, and the calendar days
    since its previous return for a ticker that also trades on weekends (see
    `get_risk_free_rate_per_observation`). The other periods compound the annualized rate
    over the calendar days between the ticker's own dates with a return in each period,
    starting from the date of its first price, as the return of a period compounds its
    daily returns (see `compound_risk_free_rate`). A full year therefore earns the
    annualized rate, and a period the ticker covers in part (the current year, or its first
    period) earns the rate of the days it covers. The rate of a ticker does not depend on
    the other tickers: the weekends a cryptocurrency adds to the dates neither split a
    stock's Monday return from the Friday before nor move a weekend at the turn of a month
    out of the month of the stock's next trading day. Without daily returns the annualized
    rate of the period is converted with the periods in a year instead, the same for every
    ticker.

    Args:
        annualized_rate_data (Mapping[str, pd.Series]): The annualized risk free rate per
            period, with a PeriodIndex.
        historical_data (Mapping[str, pd.DataFrame]): The historical data per period, with
            a "Return" column per ticker (and an "Adj Close" column for the daily period).
        period (str): "daily", "weekly", "monthly", "quarterly" or "yearly".

    Returns:
        pd.DataFrame: The risk free return per period and ticker, as a decimal.

    Raises:
        KeyError: For any other period, as a dictionary of the periods would.
    """
    if period not in PERIODS_PER_YEAR:
        raise KeyError(period)

    daily_historical_data = historical_data["daily"]

    if daily_historical_data.empty or "Return" not in daily_historical_data:
        risk_free_rate = convert_annualized_rate_to_period(
            annualized_rate_data[period], period
        )
        return pd.DataFrame(
            {
                column: risk_free_rate
                for column in historical_data[period]["Return"].columns
            }
        )

    daily_returns = daily_historical_data["Return"]
    annualized_rate = annualized_rate_data["daily"]

    if period == "daily":
        return get_risk_free_rate_per_observation(annualized_rate, daily_returns)

    risk_free_rate = {}
    for column, returns in daily_returns.items():
        observed = returns.notna()
        # The first return of a ticker accrues from the date of its first price, or
        # without prices from the date before it.
        first_price_date = (
            daily_historical_data["Adj Close"][column].first_valid_index()
            if "Adj Close" in daily_historical_data
            else observed.shift(-1, fill_value=False).idxmax()
        )
        own_dates = observed | (returns.index == first_price_date)
        risk_free_rate[column] = compound_risk_free_rate(
            annualized_rate, returns[own_dates].to_frame(), period
        )

    # The union of the tickers' periods keeps the order they are first met in (pandas only
    # sorts a DatetimeIndex here, not a PeriodIndex), so a first ticker listed later would put
    # the earlier periods of the others at the end and break every positional rolling window.  # noqa: E501
    return pd.concat(risk_free_rate, axis=1).sort_index()


def compound_into_shared_dates(returns: pd.DataFrame) -> pd.DataFrame:
    """
    The returns on the dates every column has a return, each the return since the previous
    such date. A date only some columns traded on (a holiday on one exchange, or a weekend
    for a cryptocurrency next to stocks) is left out and its returns are compounded into
    the next shared date, so a Monday pairs the three-day return of a stock with the
    three-day return of a cryptocurrency rather than with its Sunday-to-Monday return.
    The returns before the first shared date (before a later listing) are left out.

    The formula is as follows:

        - Return on a Shared Date = PRODUCT(1 + Return since the previous Shared Date) - 1

    Args:
        returns (pd.DataFrame): The returns, one column per asset, NaN where an asset has
            no return.

    Returns:
        pd.DataFrame: The compounded returns on the shared dates.
    """
    shared = returns.notna().all(axis=1).to_numpy()

    if not shared.any():
        return returns.iloc[:0]

    # Each date belongs to the first shared date on or after it. The dates before the first shared date start no period with a return of every column, and those after the last one have no shared date to go into.  # noqa: E501
    group = np.cumsum(shared[::-1])[::-1]
    within = (np.arange(len(shared)) >= np.argmax(shared)) & (group > 0)

    # The group numbers count down over time, so the groups keep the order of the dates
    # rather than being sorted by their number.
    growth = (1 + returns.fillna(0))[within]
    compounded = growth.groupby(group[within], sort=False).prod() - 1

    return compounded.set_axis(returns.index[shared])


def average_two_day_spreads(spreads: pd.Series, rolling: int | None, name) -> pd.Series:
    """
    The mean of the two-day spreads per period, or over the last `rolling` periods, for
    the Corwin-Schultz and Abdi-Ranaldo spreads.
    """
    if spreads.empty:
        return pd.Series(dtype=float, name=name)

    grouped = spreads.groupby(level=0)

    if not rolling:
        return grouped.mean().rename(name)

    # The mean of every two-day spread in the window, not the mean of the period means.
    full_range = pd.period_range(
        spreads.index.min(), spreads.index.max(), freq=spreads.index.freq
    )
    totals = grouped.sum().reindex(full_range).rolling(rolling).sum()
    counts = grouped.count().reindex(full_range).rolling(rolling).sum()

    return (totals / counts).rename(name)


def get_corwin_schultz_pairs(
    high_prices: pd.Series,
    low_prices: pd.Series,
    close_prices: pd.Series,
    period: str,
) -> pd.DataFrame:
    """
    The β and γ of Corwin and Schultz (2012) of each pair of consecutive trading days within
    a period, which both the Corwin-Schultz spread and the Bekker-Parkinson volatility take.

    The second day of a pair is adjusted for the overnight return as in section III.B of the
    paper: when its Low is above the previous Close, its High and Low are lowered by the gap,
    and when its High is below the previous Close, they are raised by the gap.

    Args:
        high_prices (pd.Series): The daily High prices.
        low_prices (pd.Series): The daily Low prices.
        close_prices (pd.Series): The daily Close prices.
        period (str): The period both days of a pair fall within: weekly, monthly, quarterly
            or yearly.

    Returns:
        pd.DataFrame: The columns Beta and Gamma, one row per pair, with the period of the
        pair as the index.
    """
    prices = pd.concat([high_prices, low_prices, close_prices], axis=1)
    # A bad bar (a price of 0 or below) has no logarithm, and a day without a High, Low or Close is not a trading day of the asset, so both leave the pairs.  # noqa: E501
    prices = prices.where(prices > 0).dropna()
    periods = to_period_index(prices.index).asfreq(PERIOD_TRANSLATION[period])

    high, low, close = (prices.iloc[:, number].to_numpy() for number in range(3))

    # The overnight adjustment of the second day: only one of the two gaps can be positive.
    shift = np.maximum(close[:-1] - high[1:], 0) - np.maximum(low[1:] - close[:-1], 0)
    second_high = high[1:] + shift
    second_low = low[1:] + shift

    beta = np.log(high[:-1] / low[:-1]) ** 2 + np.log(second_high / second_low) ** 2
    gamma = (
        np.log(np.maximum(high[:-1], second_high) / np.minimum(low[:-1], second_low))
        ** 2
    )

    same_period = np.asarray(periods[1:] == periods[:-1])

    return pd.DataFrame(
        {"Beta": beta[same_period], "Gamma": gamma[same_period]},
        index=periods[1:][same_period],
    )


def sum_per_period(values: pd.DataFrame, rolling: int | None) -> pd.DataFrame:
    """
    The sums of the observations per period, or over the last `rolling` periods, for the
    estimators that follow from sums (a regression slope, a covariance, a mean).

    A rolling window spans `rolling` periods of the calendar: a period without observations
    is a row of its own, so a window that holds one is NaN.

    Args:
        values (pd.DataFrame): The observations, one row each, with their period as the
            index.
        rolling (int | None): The number of periods summed together, or None for one
            period.

    Returns:
        pd.DataFrame: The sums with the periods as the index.
    """
    sums = values.groupby(level=0).sum()

    if not rolling or sums.empty:
        return sums

    full_range = pd.period_range(
        sums.index.min(), sums.index.max(), freq=sums.index.freq
    )

    return sums.reindex(full_range).rolling(rolling).sum()


def get_slope_through_origin(
    dependent: pd.Series,
    regressor: pd.Series,
    period: str,
    rolling: int | None,
    minimum_observations: int = 2,
) -> pd.Series:
    """
    The slope of a least squares regression without an intercept, y_t = λ * x_t + ε_t, of the
    observations within each period (or the last `rolling` periods): λ = Σ(x_t * y_t) / Σ(x_t^2).

    This is the regression form of the price impact coefficients (Kyle's, Amihud's and
    Hasbrouck's lambda) of López de Prado (2018), chapter 19.4.

    Args:
        dependent (pd.Series): The dependent variable y_t, with a daily DatetimeIndex or
            PeriodIndex.
        regressor (pd.Series): The regressor x_t on the same dates.
        period (str): The period each regression runs within: weekly, monthly, quarterly or
            yearly.
        rolling (int | None): Regress on the observations of the last `rolling` periods
            instead of a single period; a window with a period without observations is NaN.
        minimum_observations (int, optional): The fewest observations a regression runs on.
            Defaults to 2.

    Returns:
        pd.Series: The slope per period, NaN for a period with fewer than
        `minimum_observations` observations or with a regressor of 0 throughout.
    """
    data = pd.concat([dependent, regressor], axis=1).dropna()
    periods = to_period_index(data.index).asfreq(PERIOD_TRANSLATION[period])
    y, x = data.iloc[:, 0].to_numpy(), data.iloc[:, 1].to_numpy()

    sums = sum_per_period(
        pd.DataFrame({"XY": x * y, "XX": x * x, "Count": 1.0}, index=periods), rolling
    )

    if sums.empty:
        return pd.Series(dtype=float, name=dependent.name)

    estimable = (sums["Count"] >= minimum_observations) & (sums["XX"] > 0)

    return (sums["XY"] / sums["XX"].where(estimable)).rename(dependent.name)


def get_paired_windows(
    returns: pd.Series,
    market_returns: pd.Series,
    period: str,
    rolling: int | None,
):
    """
    Yields the returns of an asset and of the market within each period (or within the
    last `rolling` periods), on the dates both have a return.

    A date only one of the two traded on (a weekend of a cryptocurrency next to a stock
    market index, or a holiday on one exchange) is compounded into the next shared date
    (see `compound_into_shared_dates`), so each pair keeps the trading days it shares and
    the other tickers of the Toolkit do not change it. The windows
    run from the period in which the pair first spans `rolling` periods to the last period
    of `returns`, so a period without shared returns yields empty arrays.

    Args:
        returns (pd.Series): The daily returns of the asset.
        market_returns (pd.Series): The daily returns of the market (the system).
        period (str): The period of each window: weekly, monthly, quarterly or yearly.
        rolling (int | None): The number of periods in each window, None for one.

    Yields:
        tuple[pd.Period, np.ndarray, np.ndarray]: The last period of the window and the
        returns of the asset and of the market within it.
    """
    aligned = compound_into_shared_dates(
        pd.concat([returns, market_returns], axis=1, keys=["Asset", "Market"])
    )

    if aligned.empty:
        return

    periods = to_period_index(aligned.index).asfreq(PERIOD_TRANSLATION[period])
    last_period = (
        to_period_index(returns.index).asfreq(PERIOD_TRANSLATION[period]).max()
    )
    window = rolling if rolling else 1
    asset = aligned["Asset"].to_numpy()
    market = aligned["Market"].to_numpy()

    for end in pd.period_range(
        periods.min() + window - 1, max(periods.max(), last_period), freq=periods.freq
    ):
        rows = np.asarray((periods > end - window) & (periods <= end))
        yield end, asset[rows], market[rows]

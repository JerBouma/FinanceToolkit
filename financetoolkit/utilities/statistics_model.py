"""Statistics Module"""

__docformat__ = "google"


import difflib
import functools
import inspect
from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd

from financetoolkit.utilities import logger_model
from financetoolkit.utilities.error_model import (
    InvalidArgumentError,
    check_period,
    validate_window,
)

logger = logger_model.get_logger()


# Period to pandas frequency, shared so every period calculation agrees.
PERIOD_TRANSLATION = {
    "weekly": "W",
    "monthly": "M",
    "quarterly": "Q",
    "yearly": "Y",
}

# The periods "within period" calculations group observations by. The intraday period maps
# each intraday interval to the period its bars are grouped in; the daily period nests the
# intraday bars of a single day, so it is only available when intraday data was fetched.
WITHIN_PERIOD_TRANSLATION: dict[str, str | dict[str, str]] = {
    "intraday": {
        "1min": "h",
        "5min": "h",
        "15min": "D",
        "30min": "D",
        "1hour": "D",
    },
    "daily": "D",
    **PERIOD_TRANSLATION,
}

# Scales a daily Variance or Volatility by the trading days in a period.
VOLATILITY_WINDOW_TRANSLATION = {
    "weekly": 252 / 52,
    "monthly": 252 / 12,
    "quarterly": 252 / 4,
    "yearly": 252,
}

# The number of observations of each period within a single year.
PERIODS_PER_YEAR = {
    "daily": 252,
    "weekly": 52,
    "monthly": 12,
    "quarterly": 4,
    "yearly": 1,
}


# Short forms of country names, matched in lowercase, and the names the sources use.
COUNTRY_ALIASES = {
    "us": "United States",
    "u.s.": "United States",
    "usa": "United States",
    "russian federation": "Russia",
    "eswatini": "Swaziland",
    "congo dr": "Democratic Republic of the Congo",
    "u.s.a.": "United States",
    "america": "United States",
    "united states of america": "United States",
    "uk": "United Kingdom",
    "u.k.": "United Kingdom",
    "britain": "United Kingdom",
    "great britain": "United Kingdom",
    "england": "United Kingdom",
    "eurozone": "Euro Area",
    "euro zone": "Euro Area",
    "euro-zone": "Euro Area",
    "ea": "Euro Area",
    "eu": "European Union",
    "korea": "South Korea",
    "republic of korea": "South Korea",
    "holland": "Netherlands",
    "the netherlands": "Netherlands",
    "uae": "United Arab Emirates",
    "czechia": "Czech Republic",
    "turkiye": "Turkey",
    "türkiye": "Turkey",
    # The official names of countries the Global Macro Database names differently.
    "north macedonia": "Macedonia",
    "macao": "Macau",
    "cabo verde": "Cape Verde",
    "são tomé and príncipe": "Sao Tome and Principe",
}


# The members of the euro area, whose monetary statistics are published for the area.
EURO_AREA_MEMBERS = frozenset(
    {
        "Austria",
        "Belgium",
        "Croatia",
        "Cyprus",
        "Estonia",
        "Finland",
        "France",
        "Germany",
        "Greece",
        "Ireland",
        "Italy",
        "Latvia",
        "Lithuania",
        "Luxembourg",
        "Malta",
        "Netherlands",
        "Portugal",
        "Slovakia",
        "Slovenia",
        "Spain",
    }
)


def validate_period(period: str) -> None:
    """
    Raises when a period is not one of the periods returns and prices are grouped by.

    Args:
        period (str): The period, weekly, monthly, quarterly or yearly.

    Raises:
        InvalidArgumentError: If the period is not weekly, monthly, quarterly or yearly.
    """
    check_period(period, list(PERIOD_TRANSLATION))


def newey_west_long_run_variance(residuals: np.ndarray, lags: int) -> float:
    """
    Newey-West (1987) HAC long-run variance estimator with Bartlett kernel weights, for an
    already-demeaned series. It corrects the KPSS and Phillips-Perron statistics and the
    Diebold-Mariano loss differential for serial correlation.

    - lambda^2 = gamma_0 + 2 * SUM_{l=1}^{lags} (1 - l / (lags + 1)) * gamma_l

    where gamma_l is the l-th sample autocovariance of `residuals`.

    For more information about the method, see the following paper:

    - Newey, W.K. and West, K.D. (1987). "A Simple, Positive Semi-Definite,
    Heteroskedasticity and Autocorrelation Consistent Covariance Matrix."
    Econometrica, 55(3), 703-708.

    Args:
        residuals (np.ndarray): The demeaned series.
        lags (int): The number of autocovariances weighted in.

    Returns:
        float: The long-run variance.
    """
    number_of_observations = len(residuals)
    variance = float(np.sum(residuals**2) / number_of_observations)

    for lag in range(1, lags + 1):
        weight = 1 - lag / (lags + 1)
        autocovariance = (
            np.sum(residuals[lag:] * residuals[:-lag]) / number_of_observations
        )
        variance += 2 * weight * autocovariance

    return variance


def euro_area_hint(countries: Iterable[object], dataset: pd.DataFrame) -> str:
    """
    A pointer to the euro area for euro members without data of their own, such as the
    money supply or an overnight rate, which are published for the area as a whole.

    Args:
        countries (Iterable[object]): The countries without data.
        dataset (pd.DataFrame): The indicator, to check whether it has the euro area.

    Returns:
        str: The hint, or an empty string when it does not apply.
    """
    members = [
        str(country) for country in countries if str(country) in EURO_AREA_MEMBERS
    ]

    return (
        f" For {', '.join(members)}, use 'Euro Area', which this indicator covers."
        if members and "Euro Area" in map(str, dataset.columns.get_level_values(-1))
        else ""
    )


def normalize_country_name(country: object, known: Iterable[object]) -> object:
    """
    The name a source uses for a country written as a short form or in any casing, e.g.
    "USA" or "united states" for "United States". A name that matches nothing known is
    returned as it was given.

    Args:
        country (object): The country as given.
        known (Iterable[object]): The country names available.

    Returns:
        object: The matching known name, or the country as given.
    """
    by_lowercase = {str(name).lower(): name for name in known}
    text = str(country).strip()

    return by_lowercase.get(COUNTRY_ALIASES.get(text.lower(), text).lower(), country)


def convert_annualized_rate_to_period(
    annualized_rate: pd.Series | pd.DataFrame | float, period: str
) -> pd.Series | pd.DataFrame | float:
    """
    Converts an annualized rate, such as a Treasury yield, into the equivalent rate for
    a single period of the given frequency.

    Rates like the risk-free rate are quoted on an annual basis. Subtracting them from a
    daily, weekly, monthly or quarterly return without conversion mixes two different
    time scales and makes every excess return wrong by roughly the annual rate. The
    conversion is geometric so that compounding the result over a full year reproduces
    the original annualized rate.

    The formula is as follows:

        Period Rate = (1 + Annualized Rate)^(1 / Periods per Year) - 1

    Args:
        annualized_rate (pd.Series | pd.DataFrame | float): the annualized rate to convert.
        period (str): the period to convert the rate to. Must be one of daily, weekly,
            monthly, quarterly or yearly.

    Raises:
        InvalidArgumentError: If the period is not one of the supported frequencies.

    Returns:
        pd.Series | pd.DataFrame | float: the rate expressed per single period. A yearly
        period returns the rate unchanged.
    """
    check_period(period, list(PERIODS_PER_YEAR))

    return (1 + annualized_rate) ** (1 / PERIODS_PER_YEAR[period]) - 1


# The periods a missing base period warning names per series before it counts the rest.
SKIPPED_BASE_PERIODS_SHOWN = 3


def warn_about_missing_base_periods(
    levels: pd.Series | pd.DataFrame,
    lag: int,
    start_date: str | None = None,
    end_date: str | None = None,
    columns: Iterable | None = None,
) -> None:
    """
    Logs a warning naming the periods whose change over lag periods is empty because the
    period it compares with (the base period) has no value, although the series has
    values before it: a month a statistics office did not publish leaves the change of the
    same month a year later empty, which would otherwise disappear without a reason.

    Args:
        levels (pd.Series | pd.DataFrame): The values the change is taken of, a column per
            series, on consecutive periods (rows).
        lag (int): The number of periods (rows) the change compares with.
        start_date (str | None, optional): The first period to name. Defaults to None.
        end_date (str | None, optional): The last period to name. Defaults to None.
        columns (Iterable | None, optional): The series to name. Defaults to None, which
            names every column.
    """
    frame = levels.to_frame() if isinstance(levels, pd.Series) else levels
    if (
        frame.empty
        or isinstance(frame.columns, pd.MultiIndex)
        or not frame.index.is_monotonic_increasing
        or frame.index.has_duplicates
    ):
        return

    has_value = frame.notna()
    # The base period has no value but an earlier period has, so it is a gap in the series
    # rather than the start of it.
    base_is_gap = (~has_value & has_value.cummax()).shift(lag, fill_value=False)
    missing = (has_value & base_is_gap).loc[start_date:end_date]
    base_periods = pd.Series(frame.index, index=frame.index).shift(lag)

    for column in frame.columns if columns is None else columns:
        if column not in missing.columns or not missing[column].any():
            continue
        periods = missing.index[missing[column].to_numpy()]
        shown = periods[:SKIPPED_BASE_PERIODS_SHOWN]
        more = (
            f" and {len(periods) - len(shown)} more"
            if len(periods) > len(shown)
            else ""
        )
        logger.warning(
            "No change for %s in %s%s: %s no value.",
            column,
            ", ".join(map(str, shown)),
            more,
            (
                f"its base period {base_periods[shown[0]]} has"
                if len(periods) == 1
                else "the base periods "
                + ", ".join(str(base_periods[period]) for period in shown)
                + " have"
            ),
        )


def finalize_dataset(
    dataset: pd.Series | pd.DataFrame,
    start_date: str | None,
    end_date: str | None,
    default_rounding: int | None,
    growth: bool = False,
    lag: int | list[int] = 1,
    rounding: int | None = None,
    standardize: bool = False,
    axis: str = "columns",
    row_slice: bool = False,
    compound: bool = False,
    apply_slice: bool = True,
    rolling: int | None = None,
    trailing: int | None = None,
    dropna: bool = False,
    countries: list[str] | str | None = None,
    indicator_name: str = "",
) -> pd.Series | pd.DataFrame:
    """
    Shared post-processing for every Finance Toolkit metric, used across the Ratios,
    Risk, Performance, Models, Technicals, Economics and Fixed Income modules: replacing
    infinities (from a zero denominator) with NaN, optional rolling-window smoothing,
    optional trailing-window summation, growth conversion, Z-Score standardization
    (applied on top of the growth values when growth is also requested), rounding, date
    range slicing, optional dropping of all-NaN rows and optional country filtering.

    Growth is computed on the full dataset before the date range is applied, since it
    needs history from before the display window to calculate the correct value for the
    first rows within it. Standardization is applied after, so the Z-Scores are measured
    against the mean and standard deviation of the window shown.

    Args:
        dataset (pd.Series | pd.DataFrame): The raw metric values.
        start_date (str | None): The start date to slice the results to.
        end_date (str | None): The end date to slice the results to.
        default_rounding (int | None): The rounding to fall back to when rounding is None.
        growth (bool, optional): Whether to return the growth of the metric instead of
            the actual values. Defaults to False.
        lag (int | list[int], optional): The number of periods to lag the growth data.
            Defaults to 1.
        compound (bool, optional): Whether the growth is the compound annual growth rate
            over the lag instead of the total change. Defaults to False.
        rounding (int | None, optional): The number of decimals to round the results to.
            Defaults to None, which falls back to default_rounding.
        standardize (bool, optional): Whether to standardize (Z-Score) the result. When
            combined with growth, the growth values are standardized instead of the raw
            values. Defaults to False.
        axis (str, optional): The axis growth and standardization are computed over. Use
            "columns" (default) when each row is an entity observed over time (e.g. ratios
            indexed by ticker with periods as columns). Use "rows" or "index" when each
            column is the series to observe over time (e.g. risk, performance, technical
            or economic indicators indexed by date with tickers or countries as columns).
            Defaults to "columns".
        row_slice (bool, optional): Whether to slice the date range by row index
            (dataset.loc[start_date:end_date]) instead of by column (the default,
            dataset.loc[:, start_date:end_date]). Defaults to False.
        apply_slice (bool, optional): Whether to slice the date range at all. Set to
            False when the caller already applies its own (e.g. conditional) date range
            slicing before calling this function. Defaults to True.
        rolling (int | None, optional): The rolling window size to use for smoothing the
            data (simple moving average) before growth/standardization. Defaults to None.
        trailing (int | None, optional): The trailing window size to use for summing the
            data over trailing periods (e.g. a trailing-4-quarter sum) before
            growth/standardization. Defaults to None.
        dropna (bool, optional): Whether to drop rows that are entirely NaN after growth
            and standardization have been applied; with periods as the index only those
            before the first and after the last period with data, so a skipped period stays
            an empty row. Defaults to False.
        countries (list[str] | str | None, optional): A list of countries or a single
            country to include in the results. Defaults to None.
        indicator_name (str, optional): The human-readable name of the indicator, used in
            the missing-country warning message. Defaults to "".

    Returns:
        pd.Series | pd.DataFrame: The processed metric values; an empty Series (a metric
        that could not be calculated) is returned as it is.
    """
    # Explicitly compare to None so that rounding=0 is honoured rather than treated as "not supplied", and so that rounding=None disables rounding altogether.  # noqa: E501
    rounding = rounding if rounding is not None else default_rounding

    # A metric without data (an empty Series from a calculation that could not be done) has
    # no periods to slice and is returned as it is, already reported where it failed.
    if isinstance(dataset, pd.Series) and dataset.empty:
        return dataset

    # A zero denominator (no equity, no drawdown, no losses, a beta of 0) leaves the
    # metric undefined: NaN like any other missing value, also so that one such period
    # does not turn the growth or Z-Scores of the whole series into NaN.
    dataset = drop_non_finite(dataset)

    # A month, quarter or year a source skipped (a month a statistics office did not
    # publish) becomes an empty row, so the periods stay consecutive and a growth rate
    # never compares the wrong ones: November with September when October is missing.
    # Days and weeks are left as they are, as markets close on weekends and holidays.
    calendar_periods = isinstance(dataset.index, pd.PeriodIndex) and (
        dataset.index.freqstr[0] in "MQY"
    )
    if (
        axis == "rows"
        and calendar_periods
        and len(dataset.index)
        and dataset.index.is_monotonic_increasing
        and not dataset.index.has_duplicates
    ):
        dataset = dataset.reindex(
            pd.period_range(
                dataset.index[0],
                dataset.index[-1],
                freq=dataset.index.freq,
                name=dataset.index.name,
            )
        )

    validate_window("rolling", rolling, 1)
    validate_window("trailing", trailing, 1)
    # A window longer than the data leaves every period without a complete window, which is
    # easily mistaken for missing data, so it is said once.
    for name, window in (("rolling", rolling), ("trailing", trailing)):
        if window and len(dataset.index) and window > len(dataset.index):
            logger.warning(
                "The %s window of %s periods is longer than the %s periods of data, so no "
                "period has a complete window and the result is empty. Pass a smaller %s "
                "window or an earlier start_date.",
                name,
                window,
                len(dataset.index),
                name,
            )

    if rolling:
        dataset = dataset.rolling(window=rolling).mean()

    if trailing:
        dataset = dataset.rolling(window=trailing).sum()

    levels = dataset

    if growth:
        dataset = calculate_growth(
            dataset=dataset, lag=lag, rounding=rounding, axis=axis, compound=compound
        )

    unsliced = dataset

    if apply_slice:
        had_data = not dataset.empty
        # Where the data runs is where it has values: periods of only NaN would point to a
        # window that holds nothing either.
        has_value = dataset.notna()
        if isinstance(has_value, pd.DataFrame):
            has_value = has_value.any(axis="columns" if row_slice else "index")
        periods = (dataset.index if row_slice else dataset.columns)[
            has_value.to_numpy()
        ]
        available = (
            f" The data runs from {periods.min()} to {periods.max()}."
            if had_data and len(periods)
            else ""
        )
        dataset = (
            dataset.loc[start_date:end_date]
            if row_slice
            else dataset.loc[:, start_date:end_date]
        )

        # Data exists, just not in the window, e.g. a weekend, a range not published yet
        # or a historical dataset that ends long ago; where it does run says what to ask.
        if had_data and dataset.empty:
            logger.warning(
                "There are no %sobservations between %s and %s.%s",
                f"{indicator_name} " if indicator_name else "",
                start_date,
                end_date,
                available,
            )

    # The Z-Score describes the values shown: their mean and standard deviation are those
    # of the window, not of the history before it.
    if standardize:
        dataset = calculate_standardization(
            dataset=dataset, rounding=rounding, axis=axis
        )
    elif not growth:
        dataset = apply_rounding(dataset, rounding)

    if countries is not None and not isinstance(countries, str | list | tuple):
        raise TypeError(
            "The countries must be a country name or a list of country names, such as "
            f"'Japan' or ['Japan', 'Germany'], not a {type(countries).__name__} ({countries!r})."
        )

    if countries:
        # Economic indicators are indexed by date with countries as columns.
        if isinstance(countries, str):
            # "United States,Japan" selects both, as the sources accept it.
            countries = [
                country.strip() for country in countries.split(",") if country.strip()
            ]
        # Common short forms and any casing resolve to the names the sources use, e.g.
        # "USA" or "united states" to "United States".
        # A column per country and item (e.g. a maturity) is selected by its country.
        known_countries = (
            dataset.columns.get_level_values(0).unique()
            if isinstance(dataset.columns, pd.MultiIndex)
            else dataset.columns
        )
        # Two spellings of one country ("USA" and "United States") select it once, as a
        # repeated column would be counted twice by anything that aggregates the result.
        selected_countries: list[object] = list(
            dict.fromkeys(
                normalize_country_name(country, known_countries)
                for country in countries
            )
        )
        missing_countries = [
            country for country in selected_countries if country not in dataset.columns
        ]
        if missing_countries:
            suggestions = [
                (
                    f"'{country}' (did you mean '{close[0]}'?)"
                    if (
                        close := difflib.get_close_matches(
                            str(country),
                            [str(column) for column in known_countries],
                            n=1,
                            cutoff=0.75,
                        )
                    )
                    else f"'{country}'"
                )
                for country in missing_countries
            ]
            logger.warning(
                f"The following countries are not available for {indicator_name}: "
                f"{', '.join(suggestions)}{euro_area_hint(missing_countries, dataset)}"
            )
        dataset = dataset[
            [
                country
                for country in selected_countries
                if country not in missing_countries
            ]
        ]

        # Countries that are there but without a value in the window: say where their
        # data ends (a series that stopped years ago) rather than answer with nothing.
        if not dataset.empty and bool(dataset.isna().all().all()):
            available_countries = list(dataset.columns)
            last = (
                unsliced[available_countries].last_valid_index()
                if row_slice and set(available_countries) <= set(unsliced.columns)
                else None
            )
            logger.warning(
                "There are no %sobservations for %s between %s and %s. %s%s",
                f"{indicator_name} " if indicator_name else "",
                ", ".join(map(str, available_countries)),
                start_date,
                end_date,
                (
                    f"The last one is from {last}."
                    if last is not None
                    else "The source has no data for them."
                ),
                euro_area_hint(available_countries, unsliced),
            )

    # A change that is empty because its base period has no value (a month the source
    # skipped, a year back) is named for the series and window shown. Monthly data dated
    # on the first day of each month (the US Census Bureau's) counts as months too.
    if (
        growth
        and axis == "rows"
        and isinstance(lag, int)
        and isinstance(dataset, pd.DataFrame)
        and (
            calendar_periods
            or (
                isinstance(levels.index, pd.PeriodIndex | pd.DatetimeIndex)
                and len(levels.index)
                and bool((levels.index.day == 1).all())
            )
        )
    ):
        warn_about_missing_base_periods(
            levels,
            lag,
            start_date if apply_slice and row_slice else None,
            end_date if apply_slice and row_slice else None,
            columns=dataset.columns,
        )

    # After the country selection, so a row that is empty for every selected country goes
    # too, such as a month only an unselected country has published yet. Months, quarters
    # and years keep the empty rows between the first and the last one with data: a month
    # the source skipped is shown as missing rather than left out.
    if dropna and calendar_periods:
        with_data = dataset.notna().any(axis=1)
        dataset = (
            dataset.loc[with_data.idxmax() : with_data.iloc[::-1].idxmax()]
            if with_data.any()
            else dataset.iloc[0:0]
        )
    elif dropna:
        dataset = dataset.dropna(how="all", axis=0)

    return dataset


def to_period_index(index: pd.Index) -> pd.PeriodIndex:
    """
    Returns the given index as a PeriodIndex, which is what every period based
    calculation in the package expects its historical data to be indexed by.

    Historical data is period indexed from the moment it is collected, so this is a
    check rather than a conversion: a frame that lost its PeriodIndex somewhere along
    the way (for example through a reset_index) would otherwise fail deep inside a
    calculation with an unhelpful AttributeError on `asfreq`.

    Args:
        index (pd.Index): The index of a historical dataset.

    Returns:
        pd.PeriodIndex: The same index, narrowed to a PeriodIndex.

    Raises:
        TypeError: If the index is not a PeriodIndex.
    """
    if not isinstance(index, pd.PeriodIndex):
        raise TypeError(
            f"Expected a PeriodIndex, got {type(index).__name__}. Historical data must "
            "be indexed by periods for period based calculations."
        )

    return index


def to_multi_index(index: pd.Index) -> pd.MultiIndex:
    """
    Returns the given index as a MultiIndex, for the places that reshape a frame
    (an unstack, a concat with keys) and then operate on the resulting levels.

    Args:
        index (pd.Index): The index or columns of a reshaped dataset.

    Returns:
        pd.MultiIndex: The same index, narrowed to a MultiIndex.

    Raises:
        TypeError: If the index is not a MultiIndex.
    """
    if not isinstance(index, pd.MultiIndex):
        raise TypeError(
            f"Expected a MultiIndex but received {type(index).__name__}; the "
            "reshaping step before this point did not produce the expected levels."
        )

    return index


def to_datetime_index(index: pd.Index) -> pd.DatetimeIndex:
    """
    Returns the given index as a DatetimeIndex, the counterpart of `to_period_index`
    for the within-period helpers that nest each observation under its period.

    Args:
        index (pd.Index): The index of a historical dataset.

    Returns:
        pd.DatetimeIndex: The same index, narrowed to a DatetimeIndex.

    Raises:
        TypeError: If the index is not a DatetimeIndex.
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError(
            f"Expected a DatetimeIndex, got {type(index).__name__}. Within-period "
            "calculations start from timestamp indexed data."
        )

    return index


def by_own_trading_days(
    function: Callable | None = None,
    *,
    compound: bool = False,
    rolling_only: bool = False,
) -> Callable:
    """
    Calculate a measure on the trading days of each column.

    Tickers listed on different exchanges, and the benchmark, share one set of dates, so a
    ticker has no Return on the dates only another one traded, such as a holiday on its own
    exchange (see `historical_model.fill_non_trading_days`). Measures that skip missing
    values (a mean, a standard deviation, a quantile) are unaffected, but a measure that
    counts or walks through rows (a drawdown, a duration, a rolling window or a recursion)
    would count those dates as periods without a move, or lose every rolling window that
    holds one. When a column has dates without a value inside its history, the decorated
    function is therefore called per column on its own rows, together with the same rows of
    every argument on the same dates (the risk free rate, the benchmark), and the results
    are put back on the original dates, NaN where the column did not trade. Columns without
    such dates, the usual case for tickers on one exchange, are calculated at once as before.

    With compound=True, for a measure against the benchmark, the rows are the dates both
    traded and the returns on them are those since the previous date both traded: the
    returns of the dates only one of them traded are compounded into the next shared date.
    Pairing each one's own daily return instead would compare, after a holiday on one
    exchange, a two-day move with a one-day move.

    With rolling_only=True, for a measure that is either one value over the whole sample or a
    rolling series, only a call with a window (a `window` argument other than None) is
    calculated per column: a whole-sample measure skips the missing values itself.

    Args:
        function (Callable | None): A function taking the returns (or prices), a Series or
            DataFrame with one column per asset, as its first argument.
        compound (bool, optional): Whether to compound the returns of the left out dates
            into the next date kept, for returns measured against each other. Defaults to False.
        rolling_only (bool, optional): Whether to calculate per column only when the function
            is called with a window. Defaults to False.

    Returns:
        Callable: The function, calculated per column when the columns trade on different dates.
    """
    if function is None:
        return functools.partial(
            by_own_trading_days, compound=compound, rolling_only=rolling_only
        )

    # The returns are the first parameter, passed by position or by name.
    signature = inspect.signature(function)
    returns_name = next(iter(signature.parameters))

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        if (
            rolling_only
            and signature.bind(*args, **kwargs).arguments.get("window") is None
        ):
            return function(*args, **kwargs)
        if args:
            returns, args = args[0], args[1:]
        else:
            returns = kwargs.pop(returns_name)
        if not isinstance(returns, pd.Series | pd.DataFrame):
            return function(returns, *args, **kwargs)

        frame = returns.to_frame() if isinstance(returns, pd.Series) else returns
        index = frame.index
        values = (*args, *kwargs.values())

        def on_index(value) -> bool:
            return isinstance(value, pd.Series | pd.DataFrame) and value.index.equals(
                index
            )

        # A Series on the same dates (the benchmark, the risk free rate) is needed on
        # every date the column is calculated on.
        partners = [
            value
            for value in values
            if isinstance(value, pd.Series) and on_index(value)
        ]

        def own_rows(column) -> tuple[np.ndarray, np.ndarray]:
            has_value = frame[column].notna()
            for partner in partners:
                has_value &= partner.notna()
            started = has_value.cummax()
            ended = has_value[::-1].cummax()[::-1]
            # Rows before the first and after the last value stay, as without other tickers.
            return (has_value | ~started | ~ended).to_numpy(), has_value.to_numpy()

        rows = {column: own_rows(column) for column in frame.columns}

        if all(keep.all() for keep, _ in rows.values()):
            return function(returns, *args, **kwargs)

        def take(value, column, keep, has_value):
            if isinstance(value, pd.DataFrame) and column in value.columns:
                value = value[[column]]
            if not on_index(value):
                return value
            if compound and isinstance(value, pd.Series | pd.DataFrame):
                # Each kept row with the left out rows before it, within its period.
                group = keep[::-1].cumsum()[::-1]
                keys = (
                    [group]
                    if index.nlevels == 1
                    else [index.get_level_values(0), group]
                )
                compounded = (1 + value.fillna(0)).groupby(keys).transform("prod") - 1
                value = value.mask(
                    pd.Series(has_value, index=index), compounded, axis=0
                )
            return value[keep]

        results = {}
        for column, (keep, has_value) in rows.items():
            column_returns = take(frame, column, keep, has_value)
            results[column] = function(
                (
                    column_returns[column]
                    if isinstance(returns, pd.Series)
                    else column_returns
                ),
                *(take(value, column, keep, has_value) for value in args),
                **{
                    key: take(value, column, keep, has_value)
                    for key, value in kwargs.items()
                },
            )

        first = next(iter(results.values()))
        # A result on the dates of the returns goes back on all of them, NaN where the
        # column was not calculated; a value per column or per period is combined as it is.
        on_dates = (
            isinstance(first, pd.Series | pd.DataFrame)
            and first.index.nlevels == index.nlevels
            and first.index.isin(index).all()
        )

        if isinstance(returns, pd.Series):
            return first.reindex(index) if on_dates else first

        first_column = frame.columns[0]
        if isinstance(first, pd.DataFrame) and any(
            first_column in first.columns.get_level_values(level)
            for level in range(first.columns.nlevels)
        ):
            combined = pd.concat(results.values(), axis=1)
            return combined.reindex(index) if on_dates else combined

        if isinstance(first, pd.Series | pd.DataFrame):
            return pd.concat(results.values())

        return pd.Series(results)

    return wrapper


def apply_rounding(
    dataset: pd.Series | pd.DataFrame, rounding: int | None
) -> pd.Series | pd.DataFrame:
    """
    Rounds a dataset to the given number of decimals, leaving it untouched when no
    rounding is requested. Calling pd.DataFrame.round(None) raises a TypeError, so
    rounding=None (documented as "no rounding") has to be handled explicitly.

    Args:
        dataset (pd.Series | pd.DataFrame): the dataset to round.
        rounding (int | None): the number of decimals to round to, or None to skip rounding.

    Returns:
        pd.Series | pd.DataFrame: the rounded dataset, or the dataset as-is when rounding is None.
    """
    if rounding is None:
        return dataset

    # A small negative number rounds to -0.0, which prints as "-0"; adding 0.0 turns it
    # into 0.0 and leaves every other number as it is. Numeric columns round as pandas
    # rounds them (integers too, for a negative rounding). Numbers in a column of mixed
    # values, such as the statistics of a regression, are rounded one by one, and a number
    # that would round to zero keeps its first significant digits instead, so a p-value
    # of 0.00001 doesn't show as 0. Text, dates and booleans are left as they are.
    def round_value(value: object) -> object:
        if not isinstance(value, float | np.floating) or isinstance(value, bool):
            return value
        value = float(value)
        rounded_value = round(value, rounding) + 0.0
        if rounded_value == 0 and value != 0 and np.isfinite(value) and rounding > 0:
            return float(f"{value:.{rounding}g}")
        return rounded_value

    if isinstance(dataset, pd.Series):
        if pd.api.types.is_object_dtype(dataset):
            return dataset.map(round_value)
        if pd.api.types.is_bool_dtype(dataset) or not pd.api.types.is_numeric_dtype(
            dataset
        ):
            return dataset
        rounded_series = dataset.round(rounding)
        return (
            rounded_series + 0.0
            if pd.api.types.is_float_dtype(rounded_series)
            else rounded_series
        )

    def is_number_dtype(dtype: object) -> bool:
        return pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(
            dtype
        )

    number_columns = dataset.dtypes.map(is_number_dtype).to_numpy(dtype=bool)
    object_columns = dataset.dtypes.map(pd.api.types.is_object_dtype).to_numpy(
        dtype=bool
    )
    if (
        number_columns.all()
        and dataset.dtypes.map(pd.api.types.is_float_dtype).to_numpy(dtype=bool).all()
    ):
        return dataset.round(rounding) + 0.0
    if not number_columns.any() and not object_columns.any():
        return dataset

    rounded = dataset.copy()
    for position in np.flatnonzero(number_columns):
        column = rounded.iloc[:, position].round(rounding)
        rounded.isetitem(
            position,
            column + 0.0 if pd.api.types.is_float_dtype(column) else column,
        )
    for position in np.flatnonzero(object_columns):
        rounded.isetitem(position, rounded.iloc[:, position].map(round_value))

    return rounded


def bounded_ffill(
    dataset: pd.Series | pd.DataFrame, axis: str = "columns"
) -> pd.Series | pd.DataFrame:
    """
    Forward-fills only a single missing observation, and only when a later valid
    observation exists to bound the gap — never past the last real data point, and
    never across a longer run of missing periods. Used ahead of pct_change so a data
    gap surfaces as a NaN growth/return rather than a fabricated flat one.

    Args:
        dataset (pd.Series | pd.DataFrame): The data to fill.
        axis (str, optional): The axis that holds the periods of a DataFrame, "columns" or
            "rows" (also "index"). Ignored for a Series. Defaults to "columns".

    Returns:
        pd.Series | pd.DataFrame: The data with each bounded single-period gap filled.
    """
    # pandas accepts the "rows"/"index"/"columns" aliases the callers use, but its stubs
    # only spell out the numeric form. A Series has a single axis.
    axis_number = (
        0 if axis in ("rows", "index") or isinstance(dataset, pd.Series) else 1
    )
    previous = dataset.shift(1, axis=axis_number)

    # A gap of one observation has a value on both sides. The first observation of a longer
    # run has a value before it but not after it, so it stays empty as the rest of the run.
    single_gap = (
        dataset.isna() & previous.notna() & dataset.shift(-1, axis=axis_number).notna()
    )

    return dataset.mask(single_gap, previous)


def calculate_growth(
    dataset: pd.Series | pd.DataFrame,
    lag: int | list[int] = 1,
    rounding: int | None = 4,
    axis: str = "columns",
    compound: bool = False,
) -> pd.Series | pd.DataFrame:
    """
    Calculates growth for a given dataset. Defaults to a lag of 1 (i.e. 1 year or 1 quarter).

    The formula is as follows:

        Growth = (Value - Previous Value) / |Previous Value|

    Dividing by the absolute previous value keeps the sign of the growth equal to the sign of
    the change: a loss narrowing from -100 to -50 is a growth of 0.5 and a profit of 20 after a
    loss of -50 a growth of 1.4, where dividing by the signed previous value would report both
    as a decline. For a positive previous value it equals the percentage change. Growth from a
    previous value of 0 is undefined and reported as missing.

    A missing period has no growth, and neither has the period after it: its previous value
    is missing, and comparing with an older value would report a change over several periods
    as the change of one.

    With compound=True the growth is the compound annual growth rate (CAGR) over the lag:

        CAGR = (Value / Previous Value) ^ (1 / Years) - 1

    where Years is the lag divided by the number of periods in a year (1 for yearly, 4 for
    quarterly data), so a lag of 5 on yearly data or 20 on quarterly data gives the five-year
    CAGR. A compound rate is only defined when both values are positive: from or to a loss
    (or zero) it is reported as missing, as there is no constant yearly rate that turns one
    into the other.

    Args:
        dataset (pd.Series | pd.DataFrame): the dataset to calculate the growth values for.
        lag (int | list[int]): the lag or lags to use for the calculation. A list returns one row or
            column per lag. Defaults to 1.
        rounding (int | None): the number of decimals to round the result to, or None for no rounding.
            Defaults to 4.
        axis (str): the axis to compute the change over. Use "columns" when each row is an entity
            observed over time and "rows" when each column is. Defaults to "columns".
        compound (bool): whether to return the compound annual growth rate over the lag instead of
            the total change. Defaults to False.

    Returns:
        pd.Series | pd.DataFrame: the period over period growth of the dataset.

    Raises:
        InvalidArgumentError: When a lag is not a positive whole number (a ValueError): a lag
            of 0 compares a period with itself and a negative lag with a later period.
        ValueError: When compound=True and the periods are not yearly, quarterly, monthly or
            weekly.
    """
    lags = lag if isinstance(lag, list) else [lag]
    if not lags or any(value is None for value in lags):
        raise InvalidArgumentError(
            f"The lag is a whole number of at least 1 or a list of them, not {lag!r}."
        )
    for value in lags:
        validate_window("lag", value, 1)

    # No data (a source that could not be reached, which is logged there) has no growth.
    if dataset.empty:
        return dataset

    # A lag reaching back as far as the data leaves no period with an earlier one to
    # compare with, so the growth is empty rather than wrong; said once, as it is easily
    # mistaken for missing data. Only the periods with a value count, as a rolling window
    # leaves its first periods NaN.
    periods = (
        dataset.dropna(axis=1, how="all").shape[1]
        if axis == "columns" and isinstance(dataset, pd.DataFrame)
        else len(dataset.dropna(how="all"))
    )
    if periods and min(lags) >= periods:
        logger.warning(
            "The lag of %s periods is at least as long as the %s periods of data, so there "
            "is no earlier period to compare with and the growth is empty. Pass a smaller "
            "lag or an earlier start_date.",
            min(lags),
            periods,
        )

    years_per_period = 1 / periods_per_year(dataset, axis) if compound else 1.0

    def change(data, periods, axis):
        if compound:
            return calculate_compound_growth(
                data, periods, axis, years=periods * years_per_period
            )
        return calculate_change_over_absolute_base(data, periods, axis)

    if isinstance(lag, list):
        # Each row (axis="columns") or column (axis="rows") is an entity, and every lag
        # becomes an extra index level after it: (entity, "Lag 1"), (entity, "Lag 2").
        lag_dict = {f"Lag {lag_value}": lag_value for lag_value in lag}
        entities = dataset.index if axis == "columns" else dataset.columns
        new_index = pd.MultiIndex.from_tuples(
            [
                (*entity, lag_key) if isinstance(entity, tuple) else (entity, lag_key)
                for entity in entities
                for lag_key in lag_dict
            ]
        )

        growth_per_lag = pd.concat(
            {
                lag_key: change(dataset, lag_value, axis)
                for lag_key, lag_value in lag_dict.items()
            },
            axis=0 if axis == "columns" else 1,
        )

        # The lag is the first level after the concat and moves to the last, so the
        # labels match new_index, which also orders the lags per entity.
        stacked_axis = (
            growth_per_lag.index if axis == "columns" else growth_per_lag.columns
        )
        stacked_axis = stacked_axis.reorder_levels(
            list(range(1, stacked_axis.nlevels)) + [0]
        )

        if axis == "columns":
            dataset_lag = growth_per_lag.set_axis(stacked_axis, axis=0).reindex(
                new_index
            )
        else:
            dataset_lag = growth_per_lag.set_axis(stacked_axis, axis=1).reindex(
                new_index, axis=1
            )

        return apply_rounding(drop_non_finite(dataset_lag.astype(np.float64)), rounding)

    return apply_rounding(
        drop_non_finite(change(dataset, lag, axis)),
        rounding,
    )


def shift_periods(
    dataset: pd.Series | pd.DataFrame, periods: int = 1, axis: str = "columns"
) -> pd.Series | pd.DataFrame:
    """
    Shifts a dataset by a number of periods, so each value lines up with the value that
    many periods earlier. On yearly, quarterly, monthly or weekly periods (a PeriodIndex)
    the shift counts calendar periods: a period missing from the data (2018, 2019, 2021)
    leaves the value after it without a previous value, rather than comparing 2021 with
    2019 as if they were a year apart. Other axes, daily periods among them (markets
    close on weekends and holidays), shift by position.

    Args:
        dataset (pd.Series | pd.DataFrame): the dataset to shift.
        periods (int): the number of periods to shift by. Defaults to 1.
        axis (str): the axis that holds the periods of a DataFrame ("columns" or "rows").
            Defaults to "columns".

    Returns:
        pd.Series | pd.DataFrame: the dataset shifted, with the same labels.
    """
    axis_number = (
        0 if isinstance(dataset, pd.Series) or axis in ("rows", "index") else 1
    )
    labels = dataset.axes[axis_number]

    if (
        isinstance(labels, pd.PeriodIndex)
        and len(labels) > 0
        and labels.is_unique
        and not labels.hasnans
        and labels.freqstr.upper().startswith(("Y", "A", "Q", "M", "W"))
    ):
        every_period = pd.period_range(labels.min(), labels.max(), freq=labels.freq)

        return (
            dataset.reindex(every_period, axis=axis_number)
            .shift(periods, axis=axis_number)
            .reindex(labels, axis=axis_number)
        )

    return dataset.shift(periods, axis=axis_number)


def calculate_change_over_absolute_base(
    dataset: pd.Series | pd.DataFrame, periods: int = 1, axis: str = "columns"
) -> pd.Series | pd.DataFrame:
    """
    Calculates the change over a number of periods relative to the absolute value at the
    start, (Value - Previous Value) / |Previous Value|, so a negative base does not flip
    the sign of the change.

    Args:
        dataset (pd.Series | pd.DataFrame): the dataset to calculate the change for.
        periods (int): the number of periods to compare against. Defaults to 1.
        axis (str): the axis to compute the change over for a DataFrame ("columns" or
            "rows"). Defaults to "columns".

    Returns:
        pd.Series | pd.DataFrame: the change relative to the absolute previous value.
    """
    previous = shift_periods(dataset, periods, axis)

    return (dataset - previous) / previous.abs()


def periods_per_year(dataset: pd.Series | pd.DataFrame, axis: str = "columns") -> int:
    """
    Returns the number of periods in a year of a dataset's periods, from the frequency of
    its PeriodIndex.

    Args:
        dataset (pd.Series | pd.DataFrame): the dataset whose periods to count.
        axis (str): the axis that holds the periods ("columns" or "rows"); a Series always
            has them in its index. Defaults to "columns".

    Returns:
        int: 1 for yearly, 4 for quarterly, 12 for monthly and 52 for weekly periods.

    Raises:
        ValueError: When the periods are not yearly, quarterly, monthly or weekly.
    """
    dates = (
        dataset.columns
        if isinstance(dataset, pd.DataFrame) and axis == "columns"
        else dataset.index
    )
    frequency = getattr(dates, "freqstr", None)

    # The frequency's first letter names its unit, e.g. "Y-DEC", "A", "Q-DEC", "QE", "M" or "W-SUN".
    for prefix, count in (("Y", 1), ("A", 1), ("Q", 4), ("M", 12), ("W", 52)):
        if frequency and frequency.upper().startswith(prefix):
            return count

    raise ValueError(
        "The compound annual growth rate needs yearly, quarterly, monthly or weekly periods, "
        f"not {frequency or 'periods without a frequency'}."
    )


def calculate_compound_growth(
    dataset: pd.Series | pd.DataFrame,
    periods: int = 1,
    axis: str = "columns",
    years: float = 1.0,
) -> pd.Series | pd.DataFrame:
    """
    Calculates the compound annual growth rate over a number of periods,
    (Value / Previous Value) ^ (1 / Years) - 1, where both values are positive.

    Args:
        dataset (pd.Series | pd.DataFrame): the dataset to calculate the growth for.
        periods (int): the number of periods to compare against. Defaults to 1.
        axis (str): the axis to compute the growth over for a DataFrame ("columns" or
            "rows"). Defaults to "columns".
        years (float): the number of years the periods span. Defaults to 1.0.

    Returns:
        pd.Series | pd.DataFrame: the compound annual growth rate, missing where either value
            is zero or negative.
    """
    previous = shift_periods(dataset, periods, axis)

    # A compound rate between a loss and a profit (or from zero) does not exist.
    ratio = (dataset / previous).where((dataset > 0) & (previous > 0))

    return ratio ** (1 / years) - 1


def drop_non_finite(
    values: float | pd.Series | pd.DataFrame,
) -> float | pd.Series | pd.DataFrame:
    """
    Replace positive and negative infinities with NaN.

    A division by zero (growth from a value of 0, a ratio with a zero denominator) or the
    logarithm of zero is undefined, not infinite, so it is reported as missing. An infinity
    would otherwise pass through every later step: one infinite value turns the mean and
    standard deviation of a Z-Score into NaN for the whole series.

    Args:
        values (float | pd.Series | pd.DataFrame): The values to clean.

    Returns:
        float | pd.Series | pd.DataFrame: The values with any infinity replaced by NaN.
    """
    if isinstance(values, pd.Series | pd.DataFrame):
        return values.replace([np.inf, -np.inf], np.nan)

    return values if np.isfinite(values) else np.nan


def calculate_standardization(
    dataset: pd.Series | pd.DataFrame,
    rounding: int | None = 4,
    axis: str = "columns",
) -> pd.Series | pd.DataFrame:
    """
    Calculates the Z-Score (standard score) for a given dataset, i.e. how many standard
    deviations each value lies from the mean of its own series.

    The formula is as follows:

        Z-Score = (value - mean) / standard deviation

    Args:
        dataset (pd.Series | pd.DataFrame): the dataset to standardize.
        rounding (int | None, optional): the number of decimals to round the results to. Defaults to 4.
        axis (str, optional): the axis the mean and standard deviation are computed over. Use
            "columns" (default) when each row is an entity observed over time (e.g. ratios indexed
            by ticker with periods as columns) so the Z-Score is computed per row across its own
            history. Use "rows" when each column is the series to standardize (e.g. economic
            indicators indexed by date with countries as columns).

    Returns:
        pd.Series | pd.DataFrame: the standardized (Z-Score) dataset.
    """
    # An infinity (a division by zero upstream) is undefined rather than a value, and left
    # in it would make the mean and standard deviation of its whole series NaN.
    dataset = drop_non_finite(dataset)

    if isinstance(dataset, pd.Series):
        return apply_rounding((dataset - dataset.mean()) / dataset.std(), rounding)

    if axis == "columns":
        return apply_rounding(
            dataset.sub(dataset.mean(axis=1), axis=0).div(dataset.std(axis=1), axis=0),
            rounding,
        )

    return apply_rounding(
        dataset.sub(dataset.mean(axis=0), axis=1).div(dataset.std(axis=0), axis=1),
        rounding,
    )

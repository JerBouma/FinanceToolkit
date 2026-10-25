"""Statistics Module"""

__docformat__ = "google"


import difflib
from collections.abc import Iterable

import numpy as np
import pandas as pd

from financetoolkit.utilities import logger_model

logger = logger_model.get_logger()

# pylint: disable=comparison-with-itself,too-many-locals

# Period to pandas frequency, shared so every period calculation agrees.
PERIOD_TRANSLATION = {
    "weekly": "W",
    "monthly": "M",
    "quarterly": "Q",
    "yearly": "Y",
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
        ValueError: If the period is not one of the supported frequencies.

    Returns:
        pd.Series | pd.DataFrame | float: the rate expressed per single period. A yearly
        period returns the rate unchanged.
    """
    if period not in PERIODS_PER_YEAR:
        raise ValueError(
            f"Period {period} is not valid. It should be one of "
            f"{', '.join(PERIODS_PER_YEAR)}."
        )

    return (1 + annualized_rate) ** (1 / PERIODS_PER_YEAR[period]) - 1


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
    apply_slice: bool = True,
    rolling: int | None = None,
    trailing: int | None = None,
    dropna: bool = False,
    countries: list[str] | str | None = None,
    indicator_name: str = "",
) -> pd.Series | pd.DataFrame:
    """
    Shared post-processing for every Finance Toolkit metric, used across the Ratios,
    Risk, Performance, Models, Technicals, Economics and Fixed Income modules: optional
    rolling-window smoothing, optional trailing-window summation, growth conversion,
    Z-Score standardization (applied on top of the growth values when growth is also
    requested), rounding, date range slicing, optional dropping of all-NaN rows and
    optional country filtering.

    Growth and standardization are always computed on the full dataset before the date
    range is applied, since growth needs history from before the display window to
    calculate the correct value for the first rows within it.

    Args:
        dataset (pd.Series | pd.DataFrame): The raw metric values.
        start_date (str | None): The start date to slice the results to.
        end_date (str | None): The end date to slice the results to.
        default_rounding (int | None): The rounding to fall back to when rounding is None.
        growth (bool, optional): Whether to return the growth of the metric instead of
            the actual values. Defaults to False.
        lag (int | list[int], optional): The number of periods to lag the growth data.
            Defaults to 1.
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
            and standardization have been applied. Defaults to False.
        countries (list[str] | str | None, optional): A list of countries or a single
            country to include in the results. Defaults to None.
        indicator_name (str, optional): The human-readable name of the indicator, used in
            the missing-country warning message. Defaults to "".

    Returns:
        pd.Series | pd.DataFrame: The processed metric values.
    """
    # Explicitly compare to None so that rounding=0 is honoured rather than treated as "not supplied", and so that rounding=None disables rounding altogether.  # noqa: E501
    rounding = rounding if rounding is not None else default_rounding

    if rolling:
        dataset = dataset.rolling(window=rolling).mean()

    if trailing:
        dataset = dataset.rolling(window=trailing).sum()

    if growth:
        dataset = calculate_growth(
            dataset=dataset, lag=lag, rounding=rounding, axis=axis
        )

    if standardize:
        dataset = calculate_standardization(
            dataset=dataset, rounding=rounding, axis=axis
        )
    elif not growth:
        dataset = apply_rounding(dataset, rounding)

    unsliced = dataset

    if apply_slice:
        had_data = not dataset.empty
        periods = dataset.index if row_slice else dataset.columns
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

    if countries is not None and not isinstance(countries, str | list | tuple):
        raise TypeError(
            "The countries must be a country name or a list of country names, such as "
            f"'Japan' or ['Japan', 'Germany'], not a {type(countries).__name__} ({countries!r})."
        )

    if countries:
        # Economic indicators are indexed by date with countries as columns.
        if isinstance(countries, str):
            countries = [countries]
        # Common short forms and any casing resolve to the names the sources use, e.g.
        # "USA" or "united states" to "United States".
        selected_countries: list[object] = [
            normalize_country_name(country, dataset.columns) for country in countries
        ]
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
                            [str(column) for column in dataset.columns],
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

    # After the country selection, so a row that is empty for every selected country goes
    # too, such as a month only an unselected country has published yet.
    if dropna:
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

    rounded = dataset.round(rounding)

    # A small negative number rounds to -0.0, which prints as "-0"; adding 0.0 turns it
    # into 0.0 and leaves every other number as it is.
    if isinstance(rounded, pd.Series):
        return rounded + 0.0 if pd.api.types.is_float_dtype(rounded) else rounded

    float_columns = rounded.dtypes.map(pd.api.types.is_float_dtype).to_numpy(dtype=bool)

    if float_columns.any():
        rounded.iloc[:, float_columns] = rounded.iloc[:, float_columns] + 0.0

    return rounded


def bounded_ffill(
    dataset: pd.Series | pd.DataFrame, axis: str = "columns"
) -> pd.Series | pd.DataFrame:
    """
    Forward-fills only a single missing observation, and only when a later valid
    observation exists to bound the gap — never past the last real data point, and
    never across a longer run of missing periods. Used ahead of pct_change so a data
    gap surfaces as a NaN growth/return rather than a fabricated flat one.
    """
    if isinstance(dataset, pd.DataFrame):
        # pandas accepts the "rows"/"index"/"columns" aliases the callers use, but its
        # stubs only spell out the numeric form.
        axis_number = 0 if axis in ("rows", "index") else 1
        filled = dataset.ffill(axis=axis_number, limit=1)
        has_future_data = dataset.bfill(axis=axis_number).notna()
    else:
        filled = dataset.ffill(limit=1)
        has_future_data = dataset.bfill().notna()

    return filled.where(has_future_data, dataset)


def calculate_growth(
    dataset: pd.Series | pd.DataFrame,
    lag: int | list[int] = 1,
    rounding: int | None = 4,
    axis: str = "columns",
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

    Args:
        dataset (pd.Series | pd.DataFrame): the dataset to calculate the growth values for.
        lag (int | list[int]): the lag or lags to use for the calculation. A list returns one row or
            column per lag. Defaults to 1.
        rounding (int | None): the number of decimals to round the result to, or None for no rounding.
            Defaults to 4.
        axis (str): the axis to compute the change over. Use "columns" when each row is an entity
            observed over time and "rows" when each column is. Defaults to "columns".

    Returns:
        pd.Series | pd.DataFrame: the period over period growth of the dataset.
    """
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

        filled = bounded_ffill(dataset, axis=axis)
        growth_per_lag = pd.concat(
            {
                lag_key: calculate_change_over_absolute_base(filled, lag_value, axis)
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

        return apply_rounding(
            drop_infinite_growth(dataset_lag.astype(np.float64)), rounding
        )

    # The forward fill has to run along the same axis as the pct_change, since a statement or ratio DataFrame is indexed by ticker with the periods as columns, so filling along the default axis would carry the previous ticker's value into the gap.  # noqa: E501
    dataset = bounded_ffill(dataset, axis=axis)

    return apply_rounding(
        drop_infinite_growth(calculate_change_over_absolute_base(dataset, lag, axis)),
        rounding,
    )


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
    if isinstance(dataset, pd.DataFrame):
        axis_number = 0 if axis in ("rows", "index") else 1
        previous = dataset.shift(periods, axis=axis_number)
    else:
        previous = dataset.shift(periods)

    return (dataset - previous) / previous.abs()


def drop_infinite_growth(growth: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """
    Growth from a value of 0 is undefined, not infinite, so it is reported as missing.

    Args:
        growth (pd.Series | pd.DataFrame): The growth values.

    Returns:
        pd.Series | pd.DataFrame: The growth values with infinities replaced by NaN.
    """
    return growth.replace([np.inf, -np.inf], np.nan)


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

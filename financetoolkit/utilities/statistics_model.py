"""Statistics Module"""

__docformat__ = "google"

import warnings

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

    if dropna:
        dataset = dataset.dropna(how="all", axis=0)

    if apply_slice:
        dataset = (
            dataset.loc[start_date:end_date]
            if row_slice
            else dataset.loc[:, start_date:end_date]
        )

    if countries:
        # Economic indicators are indexed by date with countries as columns.
        if isinstance(countries, str):
            countries = [countries]
        missing_countries = [
            country for country in countries if country not in dataset.columns
        ]
        if missing_countries:
            logger.warning(
                f"The following countries are not available for {indicator_name}: {missing_countries}"
            )
        dataset = dataset[
            [country for country in countries if country not in missing_countries]
        ]

    return dataset


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
    return dataset if rounding is None else dataset.round(rounding)


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
        filled = dataset.ffill(axis=axis, limit=1)
        has_future_data = dataset.bfill(axis=axis).notna()
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
    # pandas 2.1 warns about pct_change fill even though the code handles it.
    warnings.simplefilter(action="ignore", category=FutureWarning)

    if isinstance(lag, list):
        new_index = []
        lag_dict = {f"Lag {lag_value}": lag_value for lag_value in lag}

        if axis == "columns":
            for old_index in dataset.index:
                for lag_value in lag_dict:
                    new_index.append(
                        (*old_index, lag_value)
                        if isinstance(old_index, tuple)
                        else (old_index, lag_value)
                    )

            dataset_lag = pd.DataFrame(
                index=pd.MultiIndex.from_tuples(new_index),
                columns=dataset.columns,
                dtype=np.float64,
            )

            for new_index in dataset_lag.index:
                lag_key = new_index[-1]
                other_indices = new_index[:-1]
                if len(other_indices) == 1:
                    other_indices = other_indices[0]

                dataset_lag.loc[new_index] = (
                    bounded_ffill(dataset.loc[other_indices])
                    .pct_change(periods=lag_dict[lag_key])  # type: ignore
                    .to_numpy()
                    .reshape(-1)
                )
        else:
            for old_index in dataset.columns:
                for lag_value in lag_dict:
                    new_index.append(
                        (*old_index, lag_value)
                        if isinstance(old_index, tuple)
                        else (old_index, lag_value)
                    )

            dataset_lag = pd.DataFrame(
                columns=pd.MultiIndex.from_tuples(new_index),
                index=dataset.index,
                dtype=np.float64,
            )

            for new_index in dataset_lag.columns:
                lag_key = new_index[-1]
                other_indices = new_index[:-1]
                if len(other_indices) == 1:
                    other_indices = other_indices[0]

                dataset_lag.loc[:, new_index] = (
                    bounded_ffill(dataset.loc[:, other_indices])
                    .pct_change(periods=lag_dict[lag_key])  # type: ignore
                    .to_numpy()
                    .reshape(-1)
                )

        return apply_rounding(dataset_lag, rounding)

    # The forward fill has to run along the same axis as the pct_change, since a statement or ratio DataFrame is indexed by ticker with the periods as columns, so filling along the default axis would carry the previous ticker's value into the gap.  # noqa: E501
    dataset = bounded_ffill(dataset, axis=axis)

    return apply_rounding(dataset.pct_change(periods=lag, axis=axis), rounding)


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

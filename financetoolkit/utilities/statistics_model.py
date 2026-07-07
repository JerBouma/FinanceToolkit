"""Statistics Module"""

__docformat__ = "google"

import warnings

import numpy as np
import pandas as pd

from financetoolkit.utilities import logger_model

logger = logger_model.get_logger()

# pylint: disable=comparison-with-itself,too-many-locals

# This is used to translate a period to the corresponding Pandas frequency string
# which is required to resample daily historical data to a lower frequency. This is
# shared across historical_model.py, risk_model.py and performance_model.py so that
# every period-based calculation (Return, Variance, Volatility, ...) agrees on the
# same frequency mapping.
PERIOD_TRANSLATION = {
    "weekly": "W",
    "monthly": "M",
    "quarterly": "Q",
    "yearly": "Y",
}

# This is used to scale a daily Variance or Volatility to the corresponding
# period by multiplying it with the number of trading days within that period.
VOLATILITY_WINDOW_TRANSLATION = {
    "weekly": 252 / 52,
    "monthly": 252 / 12,
    "quarterly": 252 / 4,
    "yearly": 252,
}


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
    rounding = rounding if rounding else default_rounding

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
        dataset = dataset.round(rounding)

    if dropna:
        dataset = dataset.dropna(how="all", axis=0)

    if apply_slice:
        dataset = (
            dataset.loc[start_date:end_date]
            if row_slice
            else dataset.loc[:, start_date:end_date]
        )

    if countries:
        # Solely meant for economic indicators, which are indexed by date with countries as columns. If the
        # user requests a country that is not available, this function can log a warning and return
        # the available countries instead of raising an error.
        if isinstance(countries, str):
            countries = [countries]
        missing_countries = [
            country for country in countries if country not in dataset.columns
        ]
        if missing_countries:
            logger.warning(
                f"The following countries are not available for {indicator_name}: {missing_countries}"
            )
        dataset = dataset[list(set(countries) - set(missing_countries))]

    return dataset


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
        lag (int | str): the lag to use for the calculation. Defaults to 1.

    Returns:
        pd.Series | pd.DataFrame: _description_
    """
    # With Pandas 2.1, pct_change will no longer automatically forward fill
    # given that this has been solved within the code already but the warning
    # still appears, this is a temporary fix to ignore the warning
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

                dataset_lag.loc[new_index] = (
                    dataset.loc[other_indices]
                    .ffill()
                    .pct_change(periods=lag_dict[lag_key])  # type: ignore
                    .to_numpy()
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

                dataset_lag.loc[:, new_index] = (
                    dataset.loc[:, other_indices]
                    .ffill()
                    .pct_change(periods=lag_dict[lag_key])  # type: ignore
                    .to_numpy()
                )

        return dataset_lag.round(rounding)

    return dataset.ffill().pct_change(periods=lag, axis=axis).round(rounding)


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
        return ((dataset - dataset.mean()) / dataset.std()).round(rounding)

    if axis == "columns":
        return (
            dataset.sub(dataset.mean(axis=1), axis=0)
            .div(dataset.std(axis=1), axis=0)
            .round(rounding)
        )

    return (
        dataset.sub(dataset.mean(axis=0), axis=1)
        .div(dataset.std(axis=0), axis=1)
        .round(rounding)
    )

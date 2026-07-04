"""Helpers Module"""

__docformat__ = "google"

import contextlib
import inspect
import re
import warnings
from functools import wraps

import numpy as np
import pandas as pd
import requests

from financetoolkit.utilities import logger_model

logger = logger_model.get_logger()

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/136.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}

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


def get_request(
    url: str,
    timeout: int = 60,
    extra_headers: dict | None = None,
) -> requests.Response:
    """
    Make an HTTP GET request with automatic SSL fallback for corporate proxies
    and environments with self-signed certificates.

    Args:
        url (str): The URL to request.
        timeout (int): Request timeout in seconds.
        extra_headers (dict | None): Additional headers merged on top of the default HEADERS,
            e.g. {"Authorization": "Bearer <token>"}. Defaults to None.

    Returns:
        requests.Response: The HTTP response object.

    Raises:
        requests.exceptions.RequestException: If the request fails even without SSL verification.
    """
    headers = {**HEADERS, **(extra_headers or {})}
    try:
        response = requests.get(url, headers=headers, timeout=timeout, verify=True)
        response.raise_for_status()
        return response
    except requests.exceptions.SSLError:
        logger.warning(
            "SSL certificate verification failed for %s. Retrying without verification. "
            "This is common in corporate networks with self-signed certificates.",
            url,
        )
        response = requests.get(
            url, headers=headers, timeout=timeout, verify=False  # noqa
        )
        response.raise_for_status()
        return response


# pylint: disable=comparison-with-itself,too-many-locals,protected-access


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


def combine_dataframes(dataset_dictionary: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Combine the dataframes from different companies of the same financial statement,
    e.g. the balance sheet statement, into a single dataframe.

    Args:
        dataset_dictionary (dict[str, pd.DataFrame]): A dictionary containing the
        dataframes for each company. It should have the structure key: ticker,
        value: dataframe.

    Returns:
        pd.DataFrame: A pandas DataFrame with the combined financial statements.
    """
    combined_df = pd.concat(dict(dataset_dictionary), axis=0)

    return combined_df.sort_index(level=0, sort_remaining=False)


def equal_length(dataset1: pd.Series, dataset2: pd.Series) -> pd.Series:
    """
    Equalize the length of two datasets by adding zeros to the beginning of the shorter dataset.

    Args:
        dataset1 (pd.Series): The first dataset to be equalized.
        dataset2 (pd.Series): The second dataset to be equalized.

    Returns:
        pd.Series, pd.Series: The equalized datasets.
    """
    if int(dataset1.columns[0]) > int(dataset2.columns[0]):
        for value in range(
            int(dataset1.columns[0]) - 1, int(dataset2.columns[0]) - 1, -1
        ):
            dataset1.insert(0, value, 0.0)
        dataset1 = dataset1.sort_index()
    elif int(dataset1.columns[0]) < int(dataset2.columns[0]):
        for value in range(
            int(dataset2.columns[0]) - 1, int(dataset1.columns[0]) - 1, -1
        ):
            dataset2.insert(0, value, 0.0)
        dataset2 = dataset2.sort_index()

    return dataset1, dataset2


def convert_isin_to_ticker(isin_code: str) -> str:
    """
    Converts an ISIN code to a ticker symbol using Yahoo Finance search.

    Args:
        isin_code (str): The ISIN code to convert.

    Returns:
        str: The corresponding ticker symbol if found, otherwise the original ISIN code.
    """
    if bool(re.match("^([A-Z]{2})([A-Z0-9]{9})([0-9])$", isin_code)):
        try:
            response = get_request(
                f"https://query2.finance.yahoo.com/v1/finance/search?q={isin_code}",
                timeout=60,
            )

            data = response.json()

            if data.get("quotes"):
                symbol = data["quotes"][0]["symbol"]
                logger.info("Converted ISIN %s to ticker %s", isin_code, symbol)

                return symbol

            logger.warning(
                "Could not find a ticker for ISIN %s. Returning ISIN.", isin_code
            )
            return isin_code

        except requests.exceptions.RequestException as e:
            logger.warning(
                "Request failed for ISIN %s: %s. Returning ISIN.", isin_code, e
            )
            return isin_code
        except (KeyError, ValueError, IndexError):
            logger.warning(
                "Could not parse response for ISIN %s. Returning ISIN.", isin_code
            )
            return isin_code
    else:
        # If it's not a valid ISIN format, return the original input
        return isin_code


def enrich_historical_data(
    historical_data: pd.DataFrame,
    start: str | None = None,
    end: str | None = None,
    return_column: str = "Adj Close",
):
    """
    Retrieves enriched historical stock data for the given ticker(s) from Yahoo! Finance API for
    a specified period. It calculates the following:

        - Return: The return for the given period.
        - Cumulative Return: The cumulative return for the given period.

    The return is calculated as the percentage change in the given return column.

    The cumulative return is calculated as the cumulative product of the percentage change in the given
    return column.

    Note that Volatility, Excess Return and Excess Volatility are intentionally not calculated here
    anymore. These are available as dedicated, reusable calculations in the Risk module
    (e.g. risk_model.get_volatility, risk_model.get_excess_volatility) and the Performance module
    (e.g. performance_model.get_excess_return) instead.

    Args:
        historical_data (pd.DataFrame): A pandas DataFrame object containing the historical stock data
        for the given ticker(s).
        start (str, optional): A string representing the start date of the period to retrieve data for
            in 'YYYY-MM-DD' format. Defaults to None.
        end (str, optional): A string representing the end date of the period to retrieve data for
            in 'YYYY-MM-DD' format. Defaults to None.
        return_column (str, optional): A string representing the column to use for return calculations.

    Returns:
        pd.DataFrame: A pandas DataFrame object containing the enriched historical stock data for the given ticker(s).
    """

    historical_data["Return"] = historical_data[return_column].ffill().pct_change()

    historical_data["Cumulative Return"] = 1

    adjusted_return = historical_data.loc[start:end, "Return"].copy()

    with contextlib.suppress(IndexError):
        adjusted_return.iloc[0] = 0

    historical_data["Cumulative Return"] = pd.Series(np.nan).astype(float)

    historical_data.loc[start:end, "Cumulative Return"] = (
        1.0 + adjusted_return.fillna(0)
    ).cumprod()

    return historical_data


def handle_portfolio(func):
    """
    A decorator that processes the result of a function to handle portfolio data.
    This decorator checks if "Portfolio" is in the `self._tickers` attribute and, if so,
    calculates the weighted average of the result DataFrame using `self._portfolio_weights`
    and appends it as a new row or column named "Portfolio".

    Args:
        func (function): The function to be decorated.

    Returns:
        function: The wrapped function with additional portfolio handling logic.

    Notes:
        - The decorated function should have a `self` parameter as the first argument.
        - The decorated function should return a DataFrame.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # Call the original function
        result = func(self, *args, **kwargs)

        # Check if "Portfolio" is in self._tickers
        if (
            isinstance(self._tickers, list | str)
            and "Portfolio" in self._tickers
            and isinstance(result, pd.DataFrame)
        ):
            sig = inspect.signature(func)
            bound_args = sig.bind(self, *args, **kwargs)
            bound_args.apply_defaults()

            # Merge defaults with kwargs, without overriding explicitly passed values
            for key, value in bound_args.arguments.items():
                if key not in kwargs:
                    kwargs[key] = value

            # Get the rounding parameter from kwargs or use a default value
            rounding = kwargs.get("rounding", self._rounding)
            lag = kwargs.get("lag", 1)
            growth = kwargs.get("growth", False)
            period = kwargs.get("period")

            if rounding is None:
                rounding = self._rounding
            if period is None:
                period = "quarterly" if getattr(self, "_quarterly", False) else "yearly"

            # Select the appropriate portfolio weights
            weights = self._portfolio_weights.get(period, pd.DataFrame())

            # Exclude "Benchmark" from the weighted average calculation
            result_without_benchmark = (
                result.drop(columns=["Benchmark"])
                if "Benchmark" in result.columns
                else result
            )

            # Calculate the weighted average for each column
            if isinstance(result.columns, pd.PeriodIndex) and not isinstance(
                result.columns, pd.MultiIndex
            ):
                weights = weights.loc[result_without_benchmark.columns, :].T

                weighted_averages = round(
                    (result_without_benchmark * weights).sum(axis=0)
                    / weights.sum(axis=0),
                    rounding,
                )

                # Append the weighted averages as a new row
                result.loc["Portfolio"] = weighted_averages
            elif isinstance(result.index, pd.PeriodIndex) and not isinstance(
                result.columns, pd.MultiIndex
            ):
                weights = weights.loc[result.index, :]

                weighted_averages = round(
                    (result_without_benchmark * weights).sum(axis=1)
                    / weights.sum(axis=1),
                    rounding,
                )

                # Append the weighted averages as a new row
                result["Portfolio"] = weighted_averages

            if growth and isinstance(lag, list):
                logger.warning(
                    "Calculating multiple lags for the portfolio data is not currently available. \n"
                    "If desired, please reach out via https://github.com/JerBouma/FinanceToolkit/issues"
                )

        return result

    return wrapper


def filter_columns(
    result: pd.DataFrame | pd.Series | dict | object,
    show_columns: list[str] | None,
) -> pd.DataFrame | pd.Series | dict | object:
    """Filter a Finance Toolkit result to only include the specified columns.

    Works on pd.DataFrame, dicts of pd.DataFrame (multi-ticker financial
    statements), and passes through pd.Series, scalars, and any other type
    unchanged.  When *show_columns* is None the result is returned unmodified.

    Args:
        result: The value returned by a controller ``get_*`` method.
        show_columns: Column names to keep.  For MultiIndex DataFrames the
            first index level is used for matching.  Invalid names are logged
            as warnings; if none of the requested columns exist the original
            result is returned unchanged.

    Returns:
        The filtered result, or *result* unchanged when filtering cannot be
        applied or *show_columns* is None.
    """
    if show_columns is None:
        return result

    if isinstance(result, pd.DataFrame):
        return _filter_dataframe_columns(result, show_columns)

    if isinstance(result, dict):
        return {
            key: (
                _filter_dataframe_columns(value, show_columns)
                if isinstance(value, pd.DataFrame)
                else value
            )
            for key, value in result.items()
        }

    return result


def _filter_dataframe_columns(
    df: pd.DataFrame,
    show_columns: list[str],
) -> pd.DataFrame:
    """Internal helper: filter a single DataFrame to *show_columns*.

    Resolution order:
        1. MultiIndex *columns* — filter by first column level (e.g. OHLCV type in
        historical data where columns are ``(metric, ticker)``).
        2. Flat *columns* — filter columns whose string representation appears in
        *show_columns*.
        3. MultiIndex *index* (fallback) — filter by the last index level (e.g.
        financial-statement line items in multi-ticker data where the row index
        is ``(ticker, line_item)``).
        4. Flat *index* (fallback) — filter by the index values whose string
        representation appears in *show_columns* (e.g. single-ticker income
        statement where rows are individual line items).

    If none of the above yield any matches the original DataFrame is returned
    unchanged and a warning is logged.
    """
    if df.empty:
        return df

    # MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        available = [str(c) for c in df.columns.get_level_values(0).unique()]
        valid = [c for c in show_columns if c in available]
        invalid = [c for c in show_columns if c not in available]
        for col in invalid:
            logger.warning("Column '%s' not found. Valid columns: %s", col, available)
        if valid:
            mask = df.columns.get_level_values(0).isin(valid)
            return df.loc[:, mask]
        return df

    # Flat columns
    available_cols = [str(c) for c in df.columns]
    col_map = {str(c): c for c in df.columns}
    valid_cols = [c for c in show_columns if c in available_cols]

    if valid_cols:
        return df[[col_map[c] for c in valid_cols]]

    # Row-index fallback (financial statements)
    if isinstance(df.index, pd.MultiIndex):
        level_values = df.index.get_level_values(-1)
        available_idx = [str(v) for v in level_values.unique()]
        idx_map = {str(v): v for v in level_values.unique()}
        valid_idx = [c for c in show_columns if c in available_idx]
        if valid_idx:
            mask = level_values.isin([idx_map[c] for c in valid_idx])
            filtered = df[mask]
            # When the filter reduces the last index level to one unique value
            # (e.g. show_columns=['Revenue'] on a multi-ticker statement), that
            # level repeats the same label in every row — drop it so the result
            # is indexed by ticker alone.
            if len(filtered.index.get_level_values(-1).unique()) == 1:
                filtered = filtered.copy()
                filtered.index = filtered.index.droplevel(-1)
            return filtered
    else:
        available_idx = [str(v) for v in df.index.unique()]
        idx_map = {str(v): v for v in df.index.unique()}
        valid_idx = [c for c in show_columns if c in available_idx]
        if valid_idx:
            filtered = df.loc[[idx_map[c] for c in valid_idx]]
            # When only one metric row remains the index label is known from the
            # filter — squeeze to a Series so the caller gets a clean period →
            # value mapping without the redundant metric label.
            if len(filtered) == 1:
                return filtered.squeeze()
            return filtered

    all_available = available_cols + (
        available_idx
        if not isinstance(df.index, pd.MultiIndex)
        else [str(v) for v in df.index.get_level_values(-1).unique()]
    )
    logger.warning(
        "show_columns %s not matched in columns or index. Available: %s",
        show_columns,
        all_available,
    )
    return df

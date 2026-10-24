"""Portfolio Model"""

import os

import numpy as np
import pandas as pd

from financetoolkit import currencies_model
from financetoolkit.portfolio import helpers
from financetoolkit.utilities import logger_model

logger = logger_model.get_logger()

# pylint: disable=too-many-locals

# Matches currency codes (EUR, USD) in Yahoo Finance notation, e.g. EURUSD=X.
CURRENCY_CODE_LENGTH = 3


def read_portfolio_dataset(
    excel_location: list,
    adjust_duplicates: bool,
    date_column: list[str],
    date_format_options: list[str],
    name_columns: list[str],
    ticker_columns: list[str],
    price_columns: list[str],
    volume_columns: list[str],
    column_mapping: dict[str, str],
    currency_columns: list[str] | str | None = None,
    costs_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, str, str, str, str, str, str, str]:
    """
    Read and preprocess a portfolio dataset from Excel files.

    This function reads one or more Excel or CSV files from the specified locations, preprocesses
    the data by renaming columns, handling duplicates, and parsing dates. It returns a combined
    portfolio dataset along with column names and processing details.

    Args:
        excel_location (list): A list of file paths to Excel or CSV files containing portfolio data.
        adjust_duplicates (bool): If True, duplicate entries are adjusted by aggregating values.
        date_column (list[str]): A list of column names that represent date information.
        date_format_options (list[str]): The format string for parsing dates.
        name_columns (list[str]): A list of column names representing transaction descriptions.
        ticker_columns (list[str]): A list of column names representing asset tickers.
        price_columns (list[str]): A list of column names representing asset prices.
        volume_columns (list[str]): A list of column names representing transaction volumes.
        column_mapping (dict[str, str]): A dictionary mapping original column names to standardized names.
        currency_columns (list[str] | str | None, optional): Column name(s) representing currency codes
            or a single column. Defaults to None.
        costs_columns (list[str] | None, optional): Column name(s) representing transaction costs. Defaults to None.

    Returns:
        tuple[pd.DataFrame, str, str, str, str, str, str | None]:
            A tuple containing:
            - A DataFrame with the processed portfolio dataset.
            - The name of the selected date column.
            - The name of the selected transaction description column(s).
            - The name of the selected ticker column.
            - The name of the selected price column.
            - The name of the selected volume column.
            - The name of the selected currency column, if applicable.
            - The name of the selected costs column, if applicable.

    Raises:
        ValueError: If no valid Excel or CSV files are found at the specified 'excel_location'.
        ValueError: If no date columns are found in the dataset.
        ValueError: If no ticker columns are found in the dataset.
        ValueError: If no name columns are found in the dataset.
        ValueError: If no price columns are found in the dataset.
        ValueError: If no volume columns are found in the dataset.
    """
    combined_portfolio_dataset = pd.DataFrame()
    additional_files = []
    spreadsheet_files = []

    # A folder stands for the spreadsheets in it, which are read after the files given directly.
    for file in excel_location:
        if file.split(".")[-1] not in ["xlsx", "xls", "csv"]:
            for sub_file in os.listdir(file):
                if sub_file.endswith(("xlsx", "xls", "csv")):
                    additional_files.append(f"{file}/{sub_file}")
        else:
            spreadsheet_files.append(file)

    excel_location = spreadsheet_files + additional_files

    logger.info("Reading Portfolio Files")
    for file in excel_location:
        portfolio_dataset = helpers.read_excel(file)

        (
            portfolio_dataset,
            selected_date_column,
            selected_name_column,
            selected_ticker_column,
            selected_price_column,
            selected_volume_column,
            selected_currency_column,
            selected_costs_column,
        ) = format_portfolio_dataset(
            dataset=portfolio_dataset,
            date_columns=date_column,
            date_format_options=date_format_options,
            name_columns=name_columns,
            tickers_columns=ticker_columns,
            price_columns=price_columns,
            volume_columns=volume_columns,
            column_mapping=column_mapping,
            currency_columns=currency_columns,
            costs_columns=costs_columns,
        )

        duplicate_mask = portfolio_dataset.duplicated(keep=False)

        if duplicate_mask.any() and adjust_duplicates:
            logger.info(
                "The same transaction was bought and/or sold on the same day in %s. "
                "These entries will be merged together.",
                file,
            )

            number_columns = list(portfolio_dataset.select_dtypes(np.number).columns)

            # Summing prices would falsely indicate a higher investment than made.
            number_columns.remove(selected_price_column)

            # Each group of identical rows collapses onto its first occurrence with the summable columns multiplied by how often the row appears.  # noqa: E501
            row_identity = portfolio_dataset.astype(str).agg("|".join, axis=1)
            occurrences = row_identity.map(row_identity.value_counts())

            merged_rows = portfolio_dataset[
                duplicate_mask & ~portfolio_dataset.duplicated(keep="first")
            ].copy()
            merged_rows[number_columns] = merged_rows[number_columns].mul(
                occurrences.loc[merged_rows.index], axis=0
            )

            portfolio_dataset = pd.concat(
                [portfolio_dataset[~duplicate_mask], merged_rows]
            )

        combined_portfolio_dataset = pd.concat(
            [combined_portfolio_dataset, portfolio_dataset]
        )

    if combined_portfolio_dataset.duplicated().any() and adjust_duplicates:
        logger.warning(
            "Found duplicates in the combination of datasets. This is usually due to overlapping periods. "
            "The duplicates will be removed from the datasets to prevent counting the same transaction twice."
        )
        combined_portfolio_dataset = combined_portfolio_dataset.drop_duplicates()

    combined_portfolio_dataset = combined_portfolio_dataset.sort_values(
        by=selected_date_column, ascending=False, kind="stable"
    )

    return (
        combined_portfolio_dataset,
        selected_date_column,
        selected_name_column,
        selected_ticker_column,
        selected_price_column,
        selected_volume_column,
        selected_currency_column,
        selected_costs_column,
    )


def _first_column(dataset: pd.DataFrame, names: list[str]) -> str | None:
    """The first of the names that is a column of the dataset, or None when none is."""
    return next((name for name in names if name in dataset.columns), None)


def format_portfolio_dataset(
    dataset: pd.DataFrame,
    date_columns: list[str],
    date_format_options: list[str],
    name_columns: list[str],
    tickers_columns: list[str],
    price_columns: list[str],
    volume_columns: list[str],
    column_mapping: dict[str, str],
    currency_columns: list[str] | str | None = None,
    costs_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, str, str, str, str, str, str, str]:
    """
    Format and preprocess a raw portfolio dataset for analysis.

    This function takes a raw portfolio dataset and performs various preprocessing tasks, including:
        - Parsing the date column and converting it to datetime.
        - Formatting description columns as categories.
        - Handling numeric values (e.g., prices, volumes, costs) and applying appropriate types.
        - Managing optional columns for currency codes and costs.
    The function returns the formatted dataset along with column names used for the various data fields.

    Args:
        dataset (pd.DataFrame): The raw portfolio dataset to be formatted.
        date_columns (list[str]): A list of column names representing date information.
        date_format_options (list[str]): A list of format strings for parsing dates.
        name_columns (list[str]): A list of column names representing transaction descriptions.
        tickers_columns (list[str]): A list of column names representing asset tickers.
        price_columns (list[str]): A list of column names representing asset prices.
        volume_columns (list[str]): A list of column names representing transaction volumes.
        column_mapping (dict[str, str]): A dictionary mapping original column names to standardized names.
        currency_columns (list[str] | str | None, optional): Column name(s) representing currency codes,
            or a single currency column. Defaults to None.
        costs_columns (list[str] | None, optional): Column name(s) representing transaction costs,
            or None if not applicable.

    Returns:
        tuple[pd.DataFrame, str, str, str, str, str, str | None]:
            A tuple containing:
            - A DataFrame with the formatted portfolio dataset.
            - The name of the selected date column.
            - The name of the selected transaction description column.
            - The name of the selected ticker column.
            - The name of the selected price column.
            - The name of the selected volume column.
            - The name of the selected currency column, if processed.
            - The name of the selected costs column, if processed.

    Raises:
        ValueError: If no date columns are found in the portfolio dataset.
        ValueError: If no ticker columns are found in the portfolio dataset.
        ValueError: If no name columns are found in the portfolio dataset.
        ValueError: If no price columns are found in the portfolio dataset.
        ValueError: If no volume columns are found in the portfolio dataset.
        ValueError: If cost or income columns are specified, but none are found in the dataset.
        ValueError: If the currency column contains values other than 3-letter currency codes.
        ValueError: If the provided currency code is not a 3-letter code.
    """
    # Clean trailing spaces and case if applicable; the column names are matched case-insensitively so that a DataFrame handed in directly behaves the same as a spreadsheet read from disk.  # noqa: E501
    dataset = dataset.copy()
    dataset.columns = [str(column).strip().lower() for column in dataset.columns]

    date_columns = [column.lower() for column in date_columns]
    date_column_first = _first_column(dataset, date_columns)

    if date_column_first is None:
        raise ValueError(
            "No date columns found in the portfolio dataset. Please ensure in your dataset "
            f"there is a column named one of the following: {date_columns}"
        )

    dataset = dataset.set_index(date_column_first)

    for date_format in date_format_options:
        # Each date format is tried in turn until one parses the column.
        try:
            dataset.index = pd.to_datetime(dataset.index, format=date_format).to_period(
                freq="D"
            )
            break
        except ValueError:
            continue

    if dataset.index.dtype != pd.PeriodDtype("D"):
        raise ValueError(
            "The date column could not be formatted to a datetime object. Please check the format of the "
            f"date column and whether it is correct. The options are: {date_format_options}"
        )

    dataset = dataset.reset_index()

    tickers_columns = [column.lower() for column in tickers_columns]
    tickers_column_first = _first_column(dataset, tickers_columns)

    if tickers_column_first is None:
        raise ValueError(
            "No ticker columns found in the portfolio dataset. Please ensure in your dataset "
            f"there is a column named one of the following: {tickers_columns}"
        )
    dataset[tickers_column_first] = dataset[tickers_column_first].astype("category")

    # Without a name column the ticker doubles as the name, in a column of its own.
    name_column_first = _first_column(dataset, [c.lower() for c in name_columns])
    if name_column_first in (None, tickers_column_first):
        name_column_first = tickers_column_first
        name_column_name = "TEMP " + name_column_first
    else:
        name_column_name = name_column_first
    dataset[name_column_name] = dataset[name_column_first].astype("category")

    price_columns = [column.lower() for column in price_columns]
    price_column_first = _first_column(dataset, price_columns)

    if price_column_first is None:
        raise ValueError(
            "No price columns found in the portfolio dataset. Please ensure in your dataset "
            f"there is a column named one of the following: {price_columns}"
        )
    dataset[price_column_first] = dataset[price_column_first].apply(
        helpers.convert_to_float
    )

    volume_columns = [column.lower() for column in volume_columns]
    volume_column_first = _first_column(dataset, volume_columns)

    if volume_column_first is None:
        raise ValueError(
            "No volume columns found in the portfolio dataset. Please ensure in your dataset "
            f"there is a column named one of the following: {volume_columns}"
        )
    dataset[volume_column_first] = dataset[volume_column_first].apply(
        helpers.convert_to_float
    )

    # An empty price or volume cell converts to NaN without raising, which would travel all the way into the invested amount and the weights as a missing number, so the transaction is reported instead of being carried along silently.  # noqa: E501
    incomplete_rows = dataset[
        dataset[price_column_first].isna() | dataset[volume_column_first].isna()
    ]

    if not incomplete_rows.empty:
        reported_columns = [
            date_column_first,
            tickers_column_first,
            price_column_first,
            volume_column_first,
        ]

        raise ValueError(
            "The portfolio dataset contains transactions without a price or a volume. "
            "Please complete or remove the following row(s):\n"
            f"{incomplete_rows[reported_columns].to_string()}"
        )

    if costs_columns:
        costs_columns = [column.lower() for column in costs_columns]
        costs_column_first = _first_column(dataset, costs_columns)

        if costs_column_first is None:
            logger.warning(
                "No costs columns found in the portfolio dataset. Please ensure in your dataset "
                "there is a column named one of the following: %s. "
                "Setting the costs to zero for now.",
                costs_columns,
            )
            costs_column_first = "TEMP Costs"
            dataset[costs_column_first] = 0.0
        else:
            dataset[costs_column_first] = dataset[costs_column_first].apply(
                helpers.convert_to_float
            )
            dataset[costs_column_first] = dataset[costs_column_first].fillna(0)
    else:
        costs_column_first = None

    if isinstance(currency_columns, list):
        currency_columns = [column.lower() for column in currency_columns]
        currency_column_first = _first_column(dataset, currency_columns)

        if currency_column_first is None:
            logger.warning(
                "No currency columns found in the portfolio dataset. Please ensure in your dataset "
                "there is a column named one of the following: %s. "
                "Setting the currency to EUR for now.",
                currency_columns,
            )
            currency_column_first = "TEMP Currency"
            dataset[currency_column_first] = "EUR"
        else:
            if (dataset[currency_column_first].str.len() != CURRENCY_CODE_LENGTH).any():
                raise ValueError(
                    "Currency column must contain 3-letter currency codes only (e.g. EUR, USD or JPY)."
                )

            # Capitals, except for GBp, ZAc and ILA: GBp is pence and GBP is pounds, so writing pence in capitals would make every price a hundred times too large.  # noqa: E501
            dataset[currency_column_first] = (
                dataset[currency_column_first].astype(str).map(normalize_currency)
            )

            # Prevents the currency column being dropped with DEGIRO's Unnamed columns.
            dataset = dataset.rename(columns={currency_column_first: "currency"})
            currency_column_first = "currency"

    elif isinstance(currency_columns, str):
        if len(currency_columns) != CURRENCY_CODE_LENGTH:
            raise ValueError(
                "Currency must be a 3-letter currency code (e.g. EUR, USD or JPY)."
            )
        # A single currency code applies to every transaction, so it becomes a column of its own.
        currency_column_first = "TEMP Currency"
        dataset[currency_column_first] = normalize_currency(currency_columns)
    else:
        currency_column_first = None

    # Rename what is relevant and drop the rest, so any file type still works.
    dataset = dataset.rename(
        columns={
            date_column_first: column_mapping["date"],
            name_column_name: column_mapping["name"],
            tickers_column_first: column_mapping["identifier"],
            price_column_first: column_mapping["price"],
            volume_column_first: column_mapping["volume"],
            currency_column_first: column_mapping["currency"],
            costs_column_first: column_mapping["costs"],
        }
    )

    # Drop the other columns, so any file type can be added and still work.
    dataset = dataset[column_mapping.values()]

    return (
        dataset,
        column_mapping["date"],
        column_mapping["name"],
        column_mapping["identifier"],
        column_mapping["price"],
        column_mapping["volume"],
        column_mapping["currency"],
        column_mapping["costs"],
    )


# Exchanges that quote in a fractional unit: London in pence (GBp, which brokers also write
# as GBX), Johannesburg in cents (ZAc) and Tel Aviv in agorot (ILA). Exchange rates are only
# published between major units, so these are converted through the pound, rand and shekel.
MINOR_CURRENCY_UNITS: dict[str, tuple[str, int]] = {
    **currencies_model.MINOR_CURRENCY_UNITS,
    "GBX": ("GBP", 100),
    "ZAC": ("ZAR", 100),  # codespell:ignore zar
}


def normalize_currency(currency: str) -> str:
    """
    A currency code written in capitals, except for the fractional units (GBp, ZAc, ILA)
    whose case tells them apart from the major currency: GBp is pence, GBP is pounds.

    Args:
        currency (str): The currency code, e.g. "usd", "GBp" or "GBX".

    Returns:
        str: The normalized currency code, e.g. "USD", "GBp" or "GBX".
    """
    currency = str(currency).strip()
    return (
        currency
        if currency in currencies_model.MINOR_CURRENCY_UNITS
        else currency.upper()
    )


def get_price_unit(currency: str) -> tuple[str, int]:
    """
    The major currency a price quoted in a currency belongs to and how many of the quoted
    units make up one of it: ("GBP", 100) for a price in pence (GBp or GBX), ("ZAR", 100)
    for South African cents, ("ILS", 100) for agorot and (currency, 1) otherwise.

    Args:
        currency (str): The currency the price is quoted in, e.g. "USD" or "GBp".

    Returns:
        tuple[str, int]: The major currency and the number of quoted units in one of it.
    """
    currency = str(currency).strip()
    if currency in MINOR_CURRENCY_UNITS:
        return MINOR_CURRENCY_UNITS[currency]
    return MINOR_CURRENCY_UNITS.get(currency.upper(), (currency.upper(), 1))


def get_exchange_rate_ticker(base_currency: str, currency: str) -> str | None:
    """
    The exchange rate that converts a price quoted in a currency into the base currency,
    as the Yahoo Finance symbol BASE + QUOTE + "=X" between the major currencies: USDGBP=X,
    the pounds per dollar, for a price in pounds or pence that is converted to dollars.

    Args:
        base_currency (str): The currency to convert to, e.g. "USD".
        currency (str): The currency the price is quoted in, e.g. "GBp".

    Returns:
        str | None: The exchange rate ticker, or None when both are the same currency (a
        price in pence and one in pounds differ by a factor of 100 but need no exchange
        rate).
    """
    base, _ = get_price_unit(base_currency)
    quote, _ = get_price_unit(currency)
    return None if base == quote else f"{base}{quote}=X"


def get_exchange_rate_tickers(
    currencies: dict[str, str], base_currency: str
) -> list[str]:
    """
    The exchange rates needed to convert prices quoted in the currencies to the base
    currency, see `get_exchange_rate_ticker`.

    Args:
        currencies (dict[str, str]): The currency each ticker is quoted in.
        base_currency (str): The currency to convert to, e.g. "USD".

    Returns:
        list[str]: The exchange rate tickers, e.g. ["USDEUR=X", "USDGBP=X"].
    """
    return sorted(
        {
            rate_ticker
            for currency in currencies.values()
            if (rate_ticker := get_exchange_rate_ticker(base_currency, currency))
        }
    )


def get_base_currency(currencies: dict[str, str], weights: dict[str, float]) -> str:
    """
    The currency most of a portfolio trades in: the major currency with the largest sum of
    weights, the first one listed when two are equal.

    Args:
        currencies (dict[str, str]): The currency each ticker is quoted in.
        weights (dict[str, float]): The weight per ticker.

    Returns:
        str: The major currency, e.g. "USD". USD when no currency is known.
    """
    totals: dict[str, float] = {}
    for ticker, weight in weights.items():
        currency = currencies.get(ticker)
        if currency:
            major, _ = get_price_unit(currency)
            totals[major] = totals.get(major, 0.0) + abs(float(weight))
    return max(totals, key=lambda currency: totals[currency]) if totals else "USD"


def _align_exchange_rate(rate: pd.Series, index: pd.Index) -> pd.Series:
    """
    A daily exchange rate on the dates of a price series: the rate of the same day, or the
    last one before it when the rate has no quote that day (exchanges and the currency
    market close on different holidays). For weekly to yearly prices, which hold the last
    price of each period, it is the last rate of each period.
    """
    rate = rate.dropna()
    if (
        isinstance(index, pd.PeriodIndex)
        and isinstance(rate.index, pd.PeriodIndex)
        and rate.index.freq != index.freq
    ):
        rate = rate.groupby(rate.index.asfreq(index.freq)).last()
    return rate.reindex(rate.index.union(index)).sort_index().ffill().reindex(index)


def convert_prices(
    prices: pd.DataFrame,
    currencies: dict[str, str],
    base_currency: str,
    exchange_rates: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Prices in the currency each ticker is quoted in, converted to one base currency. A price
    is divided by the number of quoted units in its major currency (100 for pence) and by
    the exchange rate BASE/QUOTE of the same day (e.g. USDGBP=X, the pounds per dollar),
    and multiplied by the units of the base currency. Values in money, such as dividends,
    convert the same way.

    Args:
        prices (pd.DataFrame): The prices, one column per ticker, on a daily (or weekly to
            yearly) date index.
        currencies (dict[str, str]): The currency each ticker is quoted in. A ticker without
            one is taken to be quoted in the base currency.
        base_currency (str): The currency to convert to, e.g. "USD".
        exchange_rates (pd.DataFrame | None, optional): The daily exchange rates, one column
            per ticker of `get_exchange_rate_ticker` (e.g. USDGBP=X). Defaults to None, for
            prices that need no exchange rate.

    Returns:
        pd.DataFrame: The prices in the base currency.

    Raises:
        ValueError: When a ticker needs an exchange rate that is not available.
    """
    _, base_units = get_price_unit(base_currency)
    converted = prices.astype(float)

    for ticker in prices.columns:
        currency = currencies.get(ticker) or base_currency
        _, units = get_price_unit(currency)
        rate_ticker = get_exchange_rate_ticker(base_currency, currency)

        rate: pd.Series | float = 1.0
        if rate_ticker is not None:
            if (
                exchange_rates is None
                or rate_ticker not in exchange_rates.columns
                or exchange_rates[rate_ticker].isna().all()
            ):
                raise ValueError(
                    f"{ticker} is quoted in {currency} and cannot be converted to "
                    f"{base_currency} without the exchange rate {rate_ticker}."
                )
            rate = _align_exchange_rate(exchange_rates[rate_ticker], prices.index)

        if units != base_units or rate_ticker is not None:
            converted[ticker] = converted[ticker] / units * base_units / rate

    return converted


def convert_historical_data(
    historical_data: pd.DataFrame,
    conversions: dict[str, tuple[str, str]],
    exchange_rates: pd.DataFrame | None,
    columns: list[str],
) -> pd.DataFrame:
    """
    Historical data (columns such as Open and Adj Close, each with one column per ticker)
    with the money columns of some tickers converted to another currency, after which
    their Return and Cumulative Return are calculated again from the converted Adj Close,
    so they are the returns in that currency.

    Args:
        historical_data (pd.DataFrame): The historical data, with the column names on the
            first level of the columns and the tickers on the second.
        conversions (dict[str, tuple[str, str]]): Per ticker the currency its data is
            quoted in and the currency to convert it to, e.g. {"VOD.L": ("GBp", "USD")}.
        exchange_rates (pd.DataFrame | None): The daily exchange rates, see
            `convert_prices`.
        columns (list[str]): The columns to convert, e.g. Open, High, Low, Close, Adj Close
            and Dividends.

    Returns:
        pd.DataFrame: The converted historical data.
    """
    historical_data = historical_data.copy()
    available = historical_data.columns.get_level_values(0).unique()

    for ticker, (currency, target) in conversions.items():
        if ("Adj Close", ticker) not in historical_data.columns:
            continue
        for column in columns:
            if (column, ticker) in historical_data.columns:
                historical_data[(column, ticker)] = convert_prices(
                    historical_data[column][[ticker]],
                    {ticker: currency},
                    target,
                    exchange_rates,
                )[ticker]

        adjusted_close = historical_data[("Adj Close", ticker)]
        if "Return" in available:
            historical_data[("Return", ticker)] = adjusted_close.pct_change()
        first = adjusted_close.first_valid_index()
        if "Cumulative Return" in available and first is not None:
            historical_data[("Cumulative Return", ticker)] = (
                adjusted_close / adjusted_close.loc[first]
            )

    return historical_data


def align_benchmark_data(benchmark_data: pd.DataFrame, dates: pd.Index) -> pd.DataFrame:
    """
    Daily benchmark data aligned to the dates of the portfolio's historical data.

    Prices (Open, High, Low, Close, Adj Close and any other level) on a date without a benchmark
    quote, such as a holiday of the benchmark's exchange, are carried forward from the previous
    quote. Flows (Volume and Dividends) on a benchmark date that is missing from `dates` are
    added to the first following date, so no dividend is lost. Return and Cumulative Return are
    calculated again from the aligned Adj Close: a carried-forward price is a return of zero.

    Filling a missing date with the next quote instead would copy that day's price and return
    onto the earlier date, counting the return twice and pricing a day with information from
    the future.

    Args:
        benchmark_data (pd.DataFrame): The daily historical data of one benchmark, one column per
            field (e.g. Close, Adj Close, Dividends, Return).
        dates (pd.Index): The dates to align to, e.g. those of the portfolio's historical data.

    Returns:
        pd.DataFrame: The benchmark data on `dates`, with the columns of `benchmark_data`.
    """
    if benchmark_data.empty:
        return benchmark_data.reindex(dates)

    benchmark_data = benchmark_data.sort_index()
    benchmark_data = benchmark_data[~benchmark_data.index.duplicated(keep="first")]
    dates = pd.Index(dates)

    flow_columns = [
        column for column in ["Volume", "Dividends"] if column in benchmark_data
    ]
    return_columns = [
        column for column in ["Return", "Cumulative Return"] if column in benchmark_data
    ]
    level_columns = [
        column
        for column in benchmark_data.columns
        if column not in flow_columns + return_columns
    ]

    aligned = (
        benchmark_data[level_columns]
        .reindex(dates.union(benchmark_data.index))
        .ffill()
        .reindex(dates)
    )

    # Every benchmark date maps to the first portfolio date on or after it.
    locations = dates.get_indexer(benchmark_data.index, method="bfill")
    matched = locations != -1
    for column in flow_columns:
        aligned[column] = (
            benchmark_data[column]
            .astype(float)
            .fillna(0.0)[matched]
            .groupby(dates[locations[matched]])
            .sum()
            .reindex(dates, fill_value=0.0)
        )

    if "Adj Close" in aligned:
        adjusted_close = aligned["Adj Close"]
        first = adjusted_close.first_valid_index()
        if "Return" in return_columns:
            returns = adjusted_close.pct_change()
            if first is not None:
                returns.loc[first] = 0.0
            aligned["Return"] = returns
        if "Cumulative Return" in return_columns and first is not None:
            aligned["Cumulative Return"] = adjusted_close / adjusted_close.loc[first]

    return aligned.reindex(columns=benchmark_data.columns)

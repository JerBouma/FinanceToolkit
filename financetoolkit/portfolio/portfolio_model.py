"""Portfolio Model"""

import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from financetoolkit.portfolio import helpers
from financetoolkit.utilities import logger_model

logger = logger_model.get_logger()

# pylint: disable=too-many-locals

# Matches up with currency codes EUR, USD, JPY etc. This is used for
# Yahoo Finance's notation of currencies. E.g. EURUSD=X
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
) -> tuple[pd.DataFrame, str, str, str, str, str, str]:
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
    for file in excel_location:
        if file.split(".")[-1] not in ["xlsx", "xls", "csv"]:
            excel_location.remove(file)

            for sub_file in os.listdir(file):
                if sub_file.endswith(("xlsx", "xls", "csv")):
                    additional_files.append(f"{file}/{sub_file}")

    excel_location = excel_location + additional_files

    for file in (
        tqdm(excel_location, desc="Reading Portfolio Files")
        if len(excel_location) > 1
        else excel_location
    ):
        portfolio_dataset = helpers.read_excel(file)
        portfolio_dataset.columns = portfolio_dataset.columns.str.lower()

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

        if portfolio_dataset.duplicated().any() and adjust_duplicates:
            logger.info(
                "The same transaction was bought and/or sold on the same day in %s. "
                "These entries will be merged together.",
                file,
            )
            duplicates = portfolio_dataset[portfolio_dataset.duplicated()]
            originals = portfolio_dataset[portfolio_dataset.duplicated(keep="first")]

            number_columns = list(
                duplicates.select_dtypes(np.number).columns.intersection(
                    originals.select_dtypes(np.number).columns
                )
            )

            # It shouldn't add together the prices as this falsely indicates a higher investment
            # than actually made and result in false return calculations.
            number_columns.remove(selected_price_column)  # type: ignore

            duplicates.loc[:, number_columns] = duplicates.loc[:, number_columns].add(
                originals[number_columns], fill_value=0
            )

            portfolio_dataset = pd.concat(
                [portfolio_dataset, duplicates]
            ).drop_duplicates(keep=False)

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
        by=selected_date_column, ascending=False  # type: ignore
    )

    return (
        combined_portfolio_dataset,
        selected_date_column,  # type: ignore
        selected_name_column,  # type: ignore
        selected_ticker_column,  # type: ignore
        selected_price_column,  # type: ignore
        selected_volume_column,  # type: ignore
        selected_currency_column,  # type: ignore
        selected_costs_column,  # type: ignore
    )


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
    # Clean trailing spaces if applicable
    dataset.columns = [column.strip() for column in dataset.columns]

    date_columns = [column.lower() for column in date_columns]
    date_column_match = [column for column in date_columns if column in dataset.columns]

    if not date_column_match:
        raise ValueError(
            "No date columns found in the portfolio dataset. Please ensure in your dataset "
            f"there is a column named one of the following: {date_columns}"
        )
    date_column_first = date_column_match[0]

    dataset = dataset.set_index(date_column_first)

    for date_format in date_format_options:
        # An attempt is made to format the date column to a datetime object. If this fails, the next format is tried.
        # This is done to ensure that the date column is correctly formatted.
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
    tickers_columns_match = [
        column for column in tickers_columns if column in dataset.columns
    ]

    if not tickers_columns_match:
        raise ValueError(
            "No ticker columns found in the portfolio dataset. Please ensure in your dataset "
            f"there is a column named one of the following: {tickers_columns}"
        )
    tickers_column_first = tickers_columns_match[0]
    dataset[tickers_column_first] = dataset[tickers_column_first].astype("category")

    name_columns = [column.lower() for column in name_columns]
    name_columns_match = [
        column for column in name_columns if column in dataset.columns
    ]

    if not name_columns_match:
        name_columns_match = tickers_columns_match

    name_column_first = name_columns_match[0]
    name_column_name = (
        "TEMP " + name_column_first
        if name_columns_match == tickers_columns_match
        else name_column_first
    )
    dataset[name_column_name] = dataset[name_column_first].astype("category")

    price_columns = [column.lower() for column in price_columns]
    price_columns_match = [
        column for column in price_columns if column in dataset.columns
    ]

    if not price_columns_match:
        raise ValueError(
            "No price columns found in the portfolio dataset. Please ensure in your dataset "
            f"there is a column named one of the following: {price_columns}"
        )
    price_column_first = price_columns_match[0]
    dataset[price_column_first] = dataset[price_column_first].apply(
        helpers.convert_to_float
    )

    volume_columns = [column.lower() for column in volume_columns]
    volume_columns_match = [
        column for column in volume_columns if column in dataset.columns
    ]

    if not volume_columns_match:
        raise ValueError(
            "No volume columns found in the portfolio dataset. Please ensure in your dataset "
            f"there is a column named one of the following: {volume_columns}"
        )
    volume_column_first = volume_columns_match[0]
    dataset[volume_column_first] = dataset[volume_column_first].apply(
        helpers.convert_to_float
    )

    if costs_columns:
        costs_columns = [column.lower() for column in costs_columns]
        costs_columns_match = [
            column for column in costs_columns if column in dataset.columns
        ]

        if not costs_columns_match:
            logger.warning(
                "No costs columns found in the portfolio dataset. Please ensure in your dataset "
                "there is a column named one of the following: %s. "
                "Setting the costs to zero for now.",
                costs_columns,
            )
            costs_column_first = "TEMP Costs"
            dataset[costs_column_first] = 0.0
        else:
            costs_column_first = costs_columns_match[0]
            dataset[costs_column_first] = dataset[costs_column_first].apply(
                helpers.convert_to_float
            )
            dataset[costs_column_first] = dataset[costs_column_first].fillna(0)
    else:
        costs_column_first = None

    if isinstance(currency_columns, list):
        currency_columns = [column.lower() for column in currency_columns]
        currency_columns_match = [
            column for column in currency_columns if column in dataset.columns
        ]

        if not currency_columns_match:
            logger.warning(
                "No currency columns found in the portfolio dataset. Please ensure in your dataset "
                "there is a column named one of the following: %s. "
                "Setting the currency to EUR for now.",
                currency_columns,
            )
            currency_column_first = "TEMP Currency"
            dataset[currency_column_first] = "EUR"
        else:
            currency_column_first = currency_columns_match[0]

            if dataset[currency_column_first].str.len().max() != CURRENCY_CODE_LENGTH:
                raise ValueError(
                    "Currency column must contain 3-letter currency codes only (e.g. EUR, USD or JPY)."
                )

            dataset[currency_column_first] = dataset[currency_column_first].astype(
                "category"
            )
            dataset[currency_column_first] = dataset[currency_column_first].str.upper()

            # This is mostly done given that Unnamed columns could exist in the dataset, specifically in
            # the DEGIRO dataset and are automatically dropped. This prevents this column from being dropped.
            dataset = dataset.rename(columns={currency_column_first: "currency"})
            currency_column_first = "currency"

    elif isinstance(currency_columns, str):
        if len(currency_columns) != CURRENCY_CODE_LENGTH:
            raise ValueError(
                "Currency must be a 3-letter currency code (e.g. EUR, USD or JPY)."
            )
        currency_column_first = currency_columns.upper()
    else:
        currency_column_first = None

    # Rename all columns that are relevant and drop the others. This is done so that any type of file
    # can be added to your portfolio and it will still work.
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

    # Drop out any other columns, this is done so that any type of file can be added to your portfolio
    # and it will still work.
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

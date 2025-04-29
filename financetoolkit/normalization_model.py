"""Normalization Module"""

__docformat__ = "google"

import shutil
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd

from financetoolkit.utilities import cache_model, logger_model

logger = logger_model.get_logger()

# pylint: disable=too-many-locals, broad-exception-caught


def initialize_statements_and_normalization(
    balance: pd.DataFrame,
    income: pd.DataFrame,
    cash: pd.DataFrame,
    format_location: str,
    reverse_dates: bool,
    use_cached_data: bool | str,
    cached_data_location: str,
    start_date: str,
    end_date: str,
    quarterly: bool,
):
    """
    Initializes financial statements by applying normalization, date conversion,
    and potentially loading from cache. Also loads normalization format files.

    Args:
        balance (pd.DataFrame): Raw balance sheet data.
        income (pd.DataFrame): Raw income statement data.
        cash (pd.DataFrame): Raw cash flow statement data.
        format_location (str): Path to normalization file directory.
        reverse_dates (bool): Whether to reverse the order of dates.
        use_cached_data (bool): Whether to load statements from cache if input is empty.
        cached_data_location (str): Path to the cache directory.
        start_date (str): Start date for filtering.
        end_date (str): End date for filtering.
        quarterly (bool): Whether the data is quarterly.

    Returns:
        tuple: A tuple containing the processed statements (balance, income, cash, statistics)
               and the loaded normalization formats.
    """
    norm_file_names = [
        "balance",
        "balance_yf",
        "income",
        "income_yf",
        "cash",
        "cash_yf",
        "statistics",
    ]
    norm_formats = {
        name: read_normalization_file(name, format_location) for name in norm_file_names
    }
    fmp_balance_sheet_statement_generic = norm_formats["balance"]
    yf_balance_sheet_statement_generic = norm_formats["balance_yf"]
    fmp_income_statement_generic = norm_formats["income"]
    yf_income_statement_generic = norm_formats["income_yf"]
    fmp_cash_flow_statement_generic = norm_formats["cash"]
    yf_cash_flow_statement_generic = norm_formats["cash_yf"]
    fmp_statistics_statement_generic = norm_formats["statistics"]

    def _process_or_load_statement(
        statement_df: pd.DataFrame,
        statement_format: pd.Series,
        cache_file_name: str,
        statement_name: str,
    ) -> pd.DataFrame:
        """Processes a statement or loads it from cache."""
        if not statement_df.empty:
            try:
                processed_statement = convert_financial_statements(
                    financial_statements=statement_df,
                    statement_format=statement_format,
                    adjust_financial_statements=False,
                    reverse_dates=reverse_dates,
                )

                # Ensure DataFrame type
                if not isinstance(processed_statement, pd.DataFrame):
                    logger.error(
                        "convert_financial_statements for %s did not return a DataFrame (got %s). Returning empty.",
                        statement_name,
                        type(processed_statement),
                    )
                    return pd.DataFrame()

                processed_statement = convert_date_label(
                    financial_statement=processed_statement,
                    start_date=start_date,
                    end_date=end_date,
                    quarter=quarterly,
                )
                return processed_statement
            except Exception as e:
                logger.error(
                    "Error processing %s statement: %s. Returning empty DataFrame.",
                    statement_name,
                    e,
                )
                return pd.DataFrame()
        elif use_cached_data:
            try:
                cached_data = cache_model.load_cached_data(
                    cached_data_location=cached_data_location,
                    file_name=cache_file_name,
                )
                if not isinstance(cached_data, pd.DataFrame):
                    logger.warning(
                        "Cached data %s for %s is not a DataFrame (type: %s). Returning empty.",
                        cache_file_name,
                        statement_name,
                        type(cached_data),
                    )
                    return pd.DataFrame()
                return cached_data
            except FileNotFoundError:
                logger.info(
                    "Cache file %s not found for %s.", cache_file_name, statement_name
                )
                return pd.DataFrame()
            except Exception as e:
                logger.error(
                    "Failed to load %s from cache for %s: %s. Returning empty DataFrame.",
                    cache_file_name,
                    statement_name,
                    e,
                )
                return pd.DataFrame()
        else:
            return pd.DataFrame()

    balance_sheet_statement = _process_or_load_statement(
        balance,
        fmp_balance_sheet_statement_generic,
        "balance_sheet_statement.pickle",
        "balance sheet",
    )
    income_statement = _process_or_load_statement(
        income, fmp_income_statement_generic, "income_statement.pickle", "income"
    )
    cash_flow_statement = _process_or_load_statement(
        cash, fmp_cash_flow_statement_generic, "cash_flow_statement.pickle", "cash flow"
    )

    statistics_statement = pd.DataFrame()
    if use_cached_data:
        cache_file_name = "statistics_statement.pickle"
        try:
            cached_stats = cache_model.load_cached_data(
                cached_data_location=cached_data_location,
                file_name=cache_file_name,
            )
            if isinstance(cached_stats, pd.DataFrame):
                statistics_statement = cached_stats
            else:
                logger.warning(
                    "Cached statistics data (%s) is not a DataFrame (type: %s). Returning empty.",
                    cache_file_name,
                    type(cached_stats),
                )
        except FileNotFoundError:
            logger.info("Cache file %s not found.", cache_file_name)
        except Exception as e:
            logger.error("Failed to load %s from cache: %s.", cache_file_name, e)

    return (
        balance_sheet_statement,
        income_statement,
        cash_flow_statement,
        statistics_statement,
        fmp_balance_sheet_statement_generic,
        yf_balance_sheet_statement_generic,
        fmp_income_statement_generic,
        yf_income_statement_generic,
        fmp_cash_flow_statement_generic,
        yf_cash_flow_statement_generic,
        fmp_statistics_statement_generic,
    )


def read_normalization_file(statement: str, format_location: str = ""):
    """
    This function copies the normalization files as found inside the environment and saves them
    to a custom location defined by the user. This is designed to make it possible to edit these files
    and supply them to the financial statement functions and within the Analyzer class.

    Args:
        statement (string): the type of statement you wish to read (balance, income, or cash).
        format_location (string): the location to read the file from.
    Returns:
        A pandas Series containing the line items for the desired statement.
    """
    if statement not in [
        "balance_yf",
        "balance",
        "income_yf",
        "income",
        "cash_yf",
        "cash",
        "statistics_yf",
        "statistics",
    ]:
        raise ValueError(
            "Please provide a valid statement type (balance, income, cash or statistics)."
        )

    if format_location:
        file_location = f"{format_location}/{statement}.csv"
    else:
        file_location = resources.files(__package__).joinpath(  # type: ignore
            f"normalization/{statement}.csv"
        )

    try:
        return pd.read_csv(file_location, index_col=[0]).iloc[:, 0]
    except FileNotFoundError:
        return pd.Series()


def convert_financial_statements(
    financial_statements: pd.DataFrame,
    statement_format: pd.DataFrame = pd.DataFrame(),
    adjust_financial_statements: bool = True,
    reverse_dates: bool = False,
):
    """
    Converts financial statements (balance, income, or cash flow statements) based on custom input
    and returns a DataFrame containing the data with the correct naming.

    Args:
        financial_statements (pd.DataFrame): DataFrame containing the financial statement data.
        format (pd.DataFrame): Optional DataFrame containing the names of the financial statement line items to include
                            in the output. Rows should contain the original name of the line item, and columns should
                            contain the desired name for that line item.

    Returns:
        pd.DataFrame: A DataFrame containing the financial statement data. If only one ticker is provided, the
                    returned DataFrame will have a single column containing the data for that company.
    """
    naming = {}

    if statement_format.empty:
        # If not format is provided, simply use the original financial statements
        return financial_statements

    for name in financial_statements.index.unique(level=1):
        try:
            if name in statement_format.to_numpy():
                naming[name] = name
            else:
                naming[name] = statement_format.loc[name]
        except KeyError:
            continue

    if adjust_financial_statements:
        # Add missing columns if applicable. Fill these with NaN.
        for name in statement_format.index:
            for ticker in financial_statements.index.unique(level=0):
                if name not in financial_statements.loc[ticker].index:
                    financial_statements.loc[(ticker, name), :] = 0

        # Given that all the columns are now present, it is possible to
        # simply overwrite the naming variable with the original statement format
        naming = statement_format

    # Select only the columns it could trace back to the format
    financial_statements = financial_statements.loc[:, list(naming.keys()), :]
    financial_statements = financial_statements.rename(index=naming)

    if reverse_dates:
        financial_statements = financial_statements[financial_statements.columns[::-1]]

    return financial_statements.sort_index(level=0, sort_remaining=False)


def convert_date_label(
    financial_statement: pd.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
    quarter: bool = False,
):
    """
    This function converts the date labels of the financial statements to a PeriodIndex. This is meant to be used
    throughout the entire Toolkit. This function is relevant for custom datasets that do not have a PeriodIndex.

    As an example, if quarter = False, the date labels will be converted to a PeriodIndex with a frequency of 'Y'
    which would convert 2022-12-31 to 2022. If quarter = True, the date labels will be converted to a PeriodIndex
    with a frequency of 'Q' which would convert 2022-12-31 to 2022Q4.

    Args:
        financial_statements (pd.DataFrame): DataFrame containing the financial statement data.
        start_date (str): The start date of the financial statement data.
        end_date (str): The end date of the financial statement data.
        quarter (bool): Whether to convert the date labels to a PeriodIndex with a frequency of 'Q' or 'Y'.

    Returns:
        pd.DataFrame: A DataFrame containing the financial statement data with a PeriodIndex.
    """
    financial_statement = financial_statement.sort_index(axis=1).truncate(
        before=start_date, after=end_date, axis=1
    )

    if financial_statement.columns.duplicated().any():
        # This happens in the rare case that a company has two financial statements for the same period.
        # Browsing through the data has shown that these financial statements are equal therefore
        # one of the columns can be dropped.
        financial_statement = financial_statement.loc[
            :, ~financial_statement.columns.duplicated()
        ]

    try:
        financial_statement = financial_statement.astype(np.float64)
    except ValueError as error:
        logger.error(
            "Not able to convert DataFrame to float64 due to %s. This could result in"
            "issues when values are zero and is predominantly relevant for "
            "ratio calculations.",
            error,
        )

    if quarter:
        financial_statement.columns = pd.PeriodIndex(
            financial_statement.columns, freq="Q"
        )
    else:
        financial_statement.columns = pd.PeriodIndex(
            financial_statement.columns, freq="Y"
        )

    return financial_statement


def copy_normalization_files(
    format_location: str = "",
    save_location: str | Path = Path(Path.home(), "Downloads"),
):
    """
    This function copies the normalization files as found inside the environment and saves them
    to a custom location defined by the user. This is designed to make it possible to edit these files
    and supply them to the financial statement functions and within the Analyzer class.

    Args:
        format_location (string): the location to read the file from.
        save_location (string): the location you wish to save the files to.
    Returns:
        Three csv files saved to the desired location.
    """
    logger.info(
        "Files are being saved to %s. Please see the following: "
        "https://www.jeroenbouma.com/projects/financetoolkit/external-datasets "
        "to understand how to work with these files. In essence, all it requires is "
        "to match up the rows in your dataframe with the normalization format.",
        save_location,
    )
    for statement in ["balance", "income", "cash", "statistics"]:
        if format_location:
            file_location = f"{format_location}/{statement}.csv"
        else:
            file_location = resources.files(__package__).joinpath(  # type: ignore
                f"normalization/{statement}.csv"
            )

        shutil.copyfile(file_location, Path(save_location, f"{statement}.csv"))

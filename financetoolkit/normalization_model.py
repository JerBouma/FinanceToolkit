"""Normalization Module"""
__docformat__ = "google"

import shutil
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd


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
    if statement not in ["balance", "income", "cash", "statistics"]:
        raise ValueError(
            "Please provide a valid statement type (balance, income, cash or statistics)."
        )

    if format_location:
        file_location = f"{format_location}/{statement}.csv"
    else:
        file_location = resources.files(__package__).joinpath(  # type: ignore
            f"normalization/{statement}.csv"
        )

    return pd.read_csv(file_location, index_col=[0]).iloc[:, 0]


def convert_financial_statements(
    financial_statements: pd.DataFrame,
    statement_format: pd.DataFrame = pd.DataFrame(),
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
        raise ValueError("Please provide a non-empty format DataFrame.")

    for name in financial_statements.index.unique(level=1):
        try:
            if name in statement_format.to_numpy():
                naming[name] = name
            else:
                naming[name] = statement_format.loc[name]
        except KeyError:
            continue

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
        print(
            f"Not able to convert DataFrame to float64 due to {error}. This could result in"
            "issues when values are zero and is predominantly relevant for "
            "ratio calculations."
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
    print(
        f"Files are being saved to {save_location}. Please see the following: "
        "https://www.jeroenbouma.com/projects/financetoolkit/external-datasets "
        "to understand how to work with these files. In essence, all it requires is "
        "to match up the rows in your dataframe with the normalization format."
    )
    for statement in ["balance", "income", "cash", "statistics"]:
        if format_location:
            file_location = f"{format_location}/{statement}.csv"
        else:
            file_location = resources.files(__package__).joinpath(  # type: ignore
                f"normalization/{statement}.csv"
            )

        shutil.copyfile(file_location, Path(save_location, f"{statement}.csv"))

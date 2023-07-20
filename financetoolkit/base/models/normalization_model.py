"""Normalization Module"""
__docformat__ = "numpy"

import shutil
from pathlib import Path

import pandas as pd
import pkg_resources


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
    if statement not in ["balance", "income", "cash"]:
        raise ValueError(
            "Please provide a valid statement type (balance, income, or cash)."
        )

    if format_location:
        file_location = f"{format_location}/{statement}.csv"
    else:
        file_location = pkg_resources.resource_stream(
            __name__, f"normalization/{statement}.csv"
        ).name

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
    print(f"Files are being saved to {save_location}")
    for statement in ["balance", "income", "cash"]:
        if format_location:
            file_location = f"{format_location}/{statement}.csv"
        else:
            file_location = pkg_resources.resource_stream(
                __name__, f"normalization/{statement}.csv"
            ).name

        shutil.copyfile(file_location, Path(save_location, f"{statement}.csv"))

"""Normalization Module"""
__docformat__ = "numpy"

import shutil
from pathlib import Path

import pandas as pd

from financialtoolkit.settings import (
    _BALANCE_SHEET_FORMAT_LOCATION,
    _CASH_FLOW_FORMAT_LOCATION,
    _INCOME_FORMAT_LOCATION,
)


def copy_normalization_files(save_location: str = Path(Path.home(), "Downloads")):
    """
    This function copies the normalization files as found inside the environment and saves them
    to a custom location defined by the user. This is designed to make it possible to edit these files
    and supply them to the financial statement functions and within the Analyzer class.

    Args:
        save_location (string): the location you wish to save the files to.
    Returns:
        Three csv files saved to the desired location.
    """
    print(f"Files are being saved to {save_location}")
    shutil.copyfile(_BALANCE_SHEET_FORMAT_LOCATION, Path(save_location, "balance.csv"))
    shutil.copyfile(_INCOME_FORMAT_LOCATION, Path(save_location, "income.csv"))
    shutil.copyfile(_CASH_FLOW_FORMAT_LOCATION, Path(save_location, "cash.csv"))


def convert_financial_statement(
    financial_statement: pd.DataFrame,
    statement_format: pd.DataFrame = pd.DataFrame(),
    reverse_dates: bool = False,
):
    """
    Converts financial statements (balance, income, or cash flow statements) based on custom input
    and returns a DataFrame containing the data with the correct naming.

    Args:
        financial_statement (pd.DataFrame): DataFrame containing the financial statement data.
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

    for name in financial_statement.index:
        try:
            naming[name] = statement_format.loc[name]
        except KeyError:
            continue

    # Select only the columns it could trace back to the format
    financial_statement = financial_statement.loc[list(naming.keys())]
    financial_statement = financial_statement.rename(index=naming)

    if reverse_dates:
        financial_statement = financial_statement[financial_statement.columns[::-1]]

    return financial_statement

"""Helpers Module"""
__docformat__ = "numpy"

import pandas as pd


def combine_dataframes(tickers: str | list[str], *args) -> pd.DataFrame:
    """
    Combine the dataframes from different companies of the same financial statement,
    e.g. the balance sheet statement, into a single dataframe.

    Args:
        **args: A dictionary of the different financial statements.

    Returns:
        pd.DataFrame: A pandas DataFrame with the combined financial statements.
    """
    ticker_list = tickers if isinstance(tickers, list) else [tickers]
    combined = zip(ticker_list, args)

    return pd.concat(dict(combined), axis=0)

"""Holdings Model: the prices and sectors of a fund's holdings and of the indices it tracks."""

from collections.abc import Callable
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from financetoolkit.toolkit_controller import Toolkit


def get_prices(
    tickers: list[str],
    start: str,
    end: str,
    other_toolkit: Callable[[list[str], str, str], "Toolkit"],
) -> pd.DataFrame:
    """
    Retrieve the daily close of other tickers, such as a fund's holdings or an index.

    The traded (split-adjusted) close is used, as an index level is a price index.

    Args:
        tickers (list[str]): The tickers.
        start (str): The start date (YYYY-MM-DD).
        end (str): The end date (YYYY-MM-DD).
        other_toolkit (Callable[[list[str], str, str], Toolkit]): Builds a Toolkit of other
            tickers between two dates, sharing the API key, cache and source of the caller's.

    Returns:
        pd.DataFrame: The daily close per ticker (columns), without rows when there are no prices.
    """
    historical = other_toolkit(tickers, start, end).get_historical_data(period="daily")
    if historical.empty:
        return pd.DataFrame(columns=tickers)
    close = historical["Close"]
    return close.to_frame(tickers[0]) if isinstance(close, pd.Series) else close


def get_sectors(
    tickers: list[str],
    start: str,
    end: str,
    other_toolkit: Callable[[list[str], str, str], "Toolkit"],
) -> pd.Series:
    """
    Retrieve the sector of each of the tickers from their company profiles.

    Args:
        tickers (list[str]): The tickers.
        start (str): The start date (YYYY-MM-DD) of the Toolkit the profiles come from.
        end (str): The end date (YYYY-MM-DD) of the Toolkit the profiles come from.
        other_toolkit (Callable[[list[str], str, str], Toolkit]): Builds a Toolkit of other
            tickers between two dates, sharing the API key, cache and source of the caller's.

    Returns:
        pd.Series: The sector per ticker, empty when the profiles hold none.
    """
    profile = other_toolkit(tickers, start, end).get_profile()
    if isinstance(profile, pd.Series):
        return pd.Series({tickers[0]: profile.get("Sector")})
    if profile.empty or "Sector" not in profile.index:
        return pd.Series(dtype=object)
    return profile.loc["Sector"]

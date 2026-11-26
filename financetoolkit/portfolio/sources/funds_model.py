"""Funds Model: the prices of other assets and the weightings of funds the portfolio analyses use."""

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
    Retrieve the daily adjusted close of other assets, such as the holdings of a benchmark.

    Args:
        tickers (list[str]): The tickers.
        start (str): The start date (YYYY-MM-DD).
        end (str): The end date (YYYY-MM-DD).
        other_toolkit (Callable[[list[str], str, str], Toolkit]): Builds a Toolkit of other
            tickers between two dates, sharing the API key, cache and source of the caller's.

    Returns:
        pd.DataFrame: The daily adjusted close per ticker (columns), without rows when there
            are no prices.
    """
    historical = other_toolkit(tickers, start, end).get_historical_data(period="daily")
    if historical.empty:
        return pd.DataFrame(columns=tickers)
    adjusted = historical["Adj Close"]
    if isinstance(adjusted, pd.Series):
        adjusted = adjusted.to_frame(tickers[0])
    return adjusted


def get_fund_weights(
    kind: str,
    tickers: list[str],
    api_key: str,
    own_tickers: list[str],
    own_fund_data: Callable[[str], pd.DataFrame | pd.Series | None],
    start: str,
    end: str,
    other_toolkit: Callable[[list[str], str, str], "Toolkit"],
) -> pd.DataFrame:
    """
    Retrieve the sector, country or holding weights of funds.

    The data of the caller's own tickers is read from its Toolkit, so it is retrieved once;
    other funds (a benchmark) come from a Toolkit of their own.

    Args:
        kind (str): The weights to retrieve. Options are: "sector" (sector weightings),
            "country" (country weightings), "holdings" (the weight of each holding).
        tickers (list[str]): The funds.
        api_key (str): The Financial Modeling Prep API key; without one there are no weights.
        own_tickers (list[str]): The tickers of the caller's Toolkit.
        own_fund_data (Callable[[str], pd.DataFrame | pd.Series | None]): Returns a fund
            dataset ("sector_weightings", "country_weightings" or "holdings") of the caller's
            own tickers.
        start (str): The start date (YYYY-MM-DD) of the Toolkit of other funds.
        end (str): The end date (YYYY-MM-DD) of the Toolkit of other funds.
        other_toolkit (Callable[[list[str], str, str], Toolkit]): Builds a Toolkit of other
            tickers between two dates, sharing the API key, cache and source of the caller's.

    Returns:
        pd.DataFrame: The weights per category (rows) and fund (columns), empty when there
            are none.
    """
    # Without an API key there are no fund weightings (or company profiles), which the method
    # using them reports once, rather than every request.
    if not api_key:
        return pd.DataFrame()
    fund_data = (
        own_fund_data
        if set(tickers) <= set(own_tickers)
        else other_toolkit(tickers, start, end)._collect_fund_data  # noqa: SLF001
    )
    if kind == "sector":
        result = fund_data("sector_weightings")
    elif kind == "country":
        result = fund_data("country_weightings")
    else:
        holdings = fund_data("holdings")
        if holdings is None or holdings.empty:
            return pd.DataFrame()
        # One fund comes back indexed by holding, several by fund and holding.
        if not isinstance(holdings.index, pd.MultiIndex):
            holdings = pd.concat({tickers[0]: holdings})
        weights = pd.to_numeric(holdings["Weight"], errors="coerce")
        result = pd.DataFrame(
            {
                fund: weights.loc[fund].groupby(level=0).sum()
                for fund in holdings.index.get_level_values(0).unique()
            }
        )
    if result is None:
        return pd.DataFrame()
    if isinstance(result, pd.Series):
        result = result.to_frame(tickers[0])
    return result

"""Market Model: the S&P 500 and corporate bond yields the models measure companies against."""

from collections.abc import Callable
from typing import TYPE_CHECKING

import pandas as pd

from financetoolkit.fmp_model import get_sp500_market_value

if TYPE_CHECKING:
    from financetoolkit.toolkit_controller import Toolkit


def get_sp500_data(
    start: str,
    end: str,
    other_toolkit: Callable[[list[str], str, str], "Toolkit"],
    api_key: str,
    user_subscription: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Retrieve the daily price and total return index of the S&P 500 and the latest total
    market value of its companies.

    The price index scales the total market value of the S&P 500 companies, known for the
    latest date only, back in time; the total return index is the market return.

    Args:
        start (str): The start date (YYYY-MM-DD).
        end (str): The end date (YYYY-MM-DD).
        other_toolkit (Callable[[list[str], str, str], Toolkit]): Builds a Toolkit of other
            tickers between two dates, sharing the API key, cache and source of the caller's.
        api_key (str): The Financial Modeling Prep API key; without one the market value is empty.
        user_subscription (str): The Financial Modeling Prep plan.

    Returns:
        tuple[pd.DataFrame, pd.Series]: The daily "Price Index" and "Total Return Index"
            closes, and the market value of the S&P 500 companies.
    """
    historical = other_toolkit(["^GSPC", "^SP500TR"], start, end).get_historical_data(
        period="daily"
    )
    if historical.empty:
        index_data = pd.DataFrame(columns=["Price Index", "Total Return Index"])
    else:
        index_data = (
            historical["Close"]
            .reindex(columns=["^GSPC", "^SP500TR"])
            .set_axis(["Price Index", "Total Return Index"], axis=1)
        )
        index_data.index = index_data.index.to_timestamp()
    market_value = (
        get_sp500_market_value(api_key, user_subscription)
        if api_key
        else pd.Series(dtype=float)
    )
    return index_data, market_value


def get_corporate_bond_yield(start: str, end: str, fred_api_key: str) -> pd.Series:
    """
    Retrieve the daily yield of Moody's Aaa corporate bonds from FRED.

    Args:
        start (str): The start date (YYYY-MM-DD).
        end (str): The end date (YYYY-MM-DD).
        fred_api_key (str): The FRED API key; without one a ValueError is raised.

    Returns:
        pd.Series: The daily Aaa corporate bond yield.
    """
    # Imported when first used, as only the Graham intrinsic value needs it.
    from financetoolkit.fixedincome.sources import fred_model  # noqa: PLC0415
    from financetoolkit.helpers import require_fred_api_key  # noqa: PLC0415

    require_fred_api_key(
        fred_api_key, "the Moody's Aaa corporate bond yield (or pass a bond_yield)"
    )
    return fred_model.get_moodys_corporate_bond_yields(
        "daily", start, end, fred_api_key
    )["Aaa"]

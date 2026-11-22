"""Cboe Global Markets Model"""

__docformat__ = "google"

import io

import pandas as pd

from financetoolkit.cache import policy_model
from financetoolkit.economics.helpers import collect_cached_data, require_columns
from financetoolkit.utilities.requests_model import get_request

# Cboe publishes the full daily history of each of its volatility indices as one file,
# updated after every close.
INDEX_URL = (
    "https://cdn.cboe.com/api/global/us_indices/daily_prices/{symbol}_History.csv"
)


def get_index(symbol: str) -> pd.Series:
    """
    Retrieves the daily closing values of a Cboe index, e.g. "VIX" from 1990.

    Args:
        symbol (str): The index, e.g. "VIX", "VIX3M" or "VVIX".

    Returns:
        pd.Series: The closing values as published (index points), indexed by day.

    Raises:
        ValueError: When the file misses the date or closing value.
    """
    description = f"Cboe {symbol} index"

    def fetch() -> pd.DataFrame:
        response = get_request(INDEX_URL.format(symbol=symbol), timeout=60)

        if not response.text.strip():
            return pd.DataFrame()

        data = pd.read_csv(io.StringIO(response.text))
        data.columns = [str(column).strip().upper() for column in data.columns]

        # Indices with only a closing value name the column after themselves.
        value_column = "CLOSE" if "CLOSE" in data.columns else symbol.upper()
        require_columns(data, {"DATE", value_column}, description)

        values = pd.to_numeric(data[value_column], errors="coerce")
        values.index = pd.PeriodIndex(
            pd.to_datetime(data["DATE"], format="%m/%d/%Y"), freq="D"
        )

        return values.dropna().sort_index().to_frame(symbol)

    index = collect_cached_data(
        source=policy_model.CBOE,
        dataset="index",
        entity=symbol,
        fetch=fetch,
        description=description,
    )

    return index[symbol] if not index.empty else pd.Series(dtype=float)

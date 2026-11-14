"""STOXX Model"""

__docformat__ = "google"

import io

import pandas as pd

from financetoolkit.cache import policy_model
from financetoolkit.economics.helpers import collect_cached_data, require_columns
from financetoolkit.utilities.requests_model import get_request

# STOXX publishes the full daily history of each index as one file, updated daily.
INDEX_URL = (
    "https://www.stoxx.com/document/Indices/Current/HistoricalData/h_{symbol}.txt"
)

# The VSTOXX sub-indices: the implied volatility of EURO STOXX 50 options with a fixed time
# to expiry, from 1999.
VSTOXX_MATURITIES = {
    "v6i1": "1M",
    "v6i2": "2M",
    "v6i3": "3M",
    "v6i4": "6M",
    "v6i5": "9M",
    "v6i6": "12M",
    "v6i7": "18M",
    "v6i8": "24M",
}


def get_index(symbol: str) -> pd.Series:
    """
    Retrieves the daily values of a STOXX index, e.g. "v6i1" (VSTOXX 1 month) from 1999.

    Args:
        symbol (str): The index symbol in lower case, e.g. "v6i1" or "v2tx".

    Returns:
        pd.Series: The values as published (index points), indexed by day.

    Raises:
        ValueError: When the file misses the date or value.
    """
    description = f"STOXX {symbol.upper()} index"

    def fetch() -> pd.DataFrame:
        response = get_request(INDEX_URL.format(symbol=symbol), timeout=60)

        # A discontinued index is answered with an empty file.
        if not response.text.strip():
            return pd.DataFrame()

        data = pd.read_csv(io.StringIO(response.text), sep=";")
        data.columns = [str(column).strip() for column in data.columns]
        require_columns(data, {"Date", "Indexvalue"}, description)

        values = pd.to_numeric(data["Indexvalue"], errors="coerce")
        values.index = pd.PeriodIndex(
            pd.to_datetime(data["Date"], format="%d.%m.%Y"), freq="D"
        )

        return values.dropna().sort_index().to_frame(symbol)

    index = collect_cached_data(
        source=policy_model.STOXX,
        dataset="index",
        entity=symbol,
        fetch=fetch,
        description=description,
    )

    return index[symbol] if not index.empty else pd.Series(dtype=float)


def get_vstoxx_term_structure() -> pd.DataFrame:
    """
    Retrieves the VSTOXX term structure, the implied volatility of EURO STOXX 50 options
    with 1 to 24 months to expiry, daily from 1999.

    Returns:
        pd.DataFrame: The implied volatility in index points (17.7 for 17.7%), indexed by
        day with a column per maturity.
    """
    columns = {
        maturity: get_index(symbol) for symbol, maturity in VSTOXX_MATURITIES.items()
    }
    columns = {
        maturity: values for maturity, values in columns.items() if not values.empty
    }

    return pd.DataFrame(columns).sort_index() if columns else pd.DataFrame()

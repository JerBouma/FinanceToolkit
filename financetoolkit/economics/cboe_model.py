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


# Cboe's delayed quotes (15 minutes) of every option listed on a US equity or index, with
# the implied volatility and greeks Cboe computes. Indices are prefixed with an
# underscore ("_SPX"), equities are not ("AAPL").
OPTION_CHAIN_URL = (
    "https://cdn.cboe.com/api/global/delayed_quotes/options/{symbol}.json"
)

# An OCC option symbol: the root ("SPX", "SPXW" or "AAPL"), the expiry (YYMMDD), C or P
# and the strike times 1,000 in eight digits.
OPTION_SYMBOL = r"^(?P<root>[A-Z.]+)(?P<expiry>\d{6})(?P<kind>[CP])(?P<strike>\d{8})$"


def get_option_chain(ticker: str) -> pd.DataFrame:
    """
    Retrieves every listed option of a US equity or index from Cboe's delayed quotes.

    Args:
        ticker (str): The ticker, e.g. "AAPL", or an index as "^SPX".

    Returns:
        pd.DataFrame: A row per option with its "Contract Symbol", "Expiration" (YYYY-MM-DD),
        "Put" (whether it is a put), "Strike", "Bid", "Ask", "Last Price", "Change",
        "Percent Change", "Volume", "Open Interest", "Implied Volatility" (a decimal),
        "Delta", "Gamma", "Vega", "Theta", "Rho" and "Underlying Price".

    Raises:
        ValueError: When the response has no options, which means its layout changed.
    """
    symbol = f"_{ticker[1:]}" if ticker.startswith("^") else ticker
    description = f"Cboe option chain of {ticker}"

    def fetch() -> pd.DataFrame:
        response = get_request(OPTION_CHAIN_URL.format(symbol=symbol), timeout=120)
        data = response.json().get("data") or {}
        options = pd.DataFrame(data.get("options") or [])

        if options.empty:
            return options

        require_columns(options, {"option", "bid", "ask", "iv"}, description)
        parts = options["option"].str.extract(OPTION_SYMBOL)
        chain = pd.DataFrame(
            {
                "Contract Symbol": options["option"],
                "Expiration": pd.to_datetime(
                    parts["expiry"], format="%y%m%d"
                ).dt.strftime("%Y-%m-%d"),
                "Put": parts["kind"] == "P",
                "Strike": pd.to_numeric(parts["strike"], errors="coerce") / 1000,
                "Bid": options["bid"],
                "Ask": options["ask"],
                "Last Price": options.get("last_trade_price"),
                "Change": options.get("change"),
                "Percent Change": options.get("percent_change"),
                "Volume": options.get("volume"),
                "Open Interest": options.get("open_interest"),
                "Implied Volatility": options["iv"],
                "Delta": options.get("delta"),
                "Gamma": options.get("gamma"),
                "Vega": options.get("vega"),
                "Theta": options.get("theta"),
                "Rho": options.get("rho"),
                "Underlying Price": data.get("current_price"),
            }
        )

        return chain.dropna(subset=["Expiration", "Strike"]).reset_index(drop=True)

    return collect_cached_data(
        source=policy_model.CBOE,
        dataset="option_chain",
        entity=symbol,
        fetch=fetch,
        description=description,
    )

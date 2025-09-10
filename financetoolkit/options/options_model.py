"""Options Model"""

import pandas as pd
import yfinance as yf


def get_option_expiry_dates(ticker: str) -> list[str]:
    """
    Retrieve available option expiry dates for a given ticker symbol.

    Args:
        ticker (str): The ticker symbol for which to fetch option expiry dates.

    Returns:
        list[str]: A list of option expiry dates in 'YYYY-MM-DD' format.
    """
    return yf.Ticker(ticker).options


def get_option_chains(
    tickers: list[str], expiration_date: str, put_option: bool = False
) -> pd.DataFrame:
    """
    Retrieve option chains (calls or puts) for a list of tickers and a specific expiration date.

    Args:
        tickers (list[str]): List of ticker symbols.
        expiration_date (str): The expiration date for the options (format: 'YYYY-MM-DD').
        put_option (bool, optional): If True, fetch put options; otherwise, fetch call options. Defaults to False.

    Returns:
        pd.DataFrame: A DataFrame containing the option chains for the specified tickers and expiration date.
    """
    result_dict = {}

    for ticker in tickers:
        option_chain = yf.Ticker(ticker).option_chain(expiration_date)
        options_df = option_chain.puts if put_option else option_chain.calls

        options_df = options_df.rename(
            columns={
                "contractSymbol": "Contract Symbol",
                "strike": "Strike",
                "currency": "Currency",
                "lastPrice": "Last Price",
                "change": "Change",
                "percentChange": "Percent Change",
                "volume": "Volume",
                "openInterest": "Open Interest",
                "bid": "Bid",
                "ask": "Ask",
                "contractSize": "Contract Size",
                "expiration": "Expiration",
                "lastTradeDate": "Last Trade Date",
                "impliedVolatility": "Implied Volatility",
                "inTheMoney": "In The Money",
            }
        )

        if "Contract Size" in options_df.columns:
            options_df = options_df.drop(columns="Contract Size")

        options_df = options_df.set_index("Strike")
        result_dict[ticker] = options_df

    result_final = pd.concat(result_dict)
    if "Last Trade Date" in result_final.columns:
        result_final["Last Trade Date"] = pd.to_datetime(
            result_final["Last Trade Date"], unit="s", errors="coerce"
        ).dt.strftime("%Y-%m-%d")

    result_final.index.names = ["Ticker", "Strike Price"]

    return result_final

"""Options Model"""
__doc__ = "google"

import pandas as pd
import requests


def get_option_expiry_dates(ticker: str):
    """
    Get the option expiry dates for a given ticker.

    Args:
        ticker (str): Ticker symbol.

    Returns:
        pd.DataFrame: Option expiry dates.
    """

    response = requests.get(
        f"https://query1.finance.yahoo.com/v7/finance/options/{ticker}",
        timeout=60,
        headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit"
            "/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"
        },
    )

    result = response.json()

    timestamps = pd.Series(result["optionChain"]["result"][0]["expirationDates"])
    dates = pd.to_datetime(timestamps, unit="s").dt.strftime("%Y-%m-%d")

    ticker_result = pd.concat([timestamps, dates], axis=1)
    ticker_result.columns = ["Timestamp", "Date"]

    ticker_result = ticker_result.set_index("Date")

    return ticker_result


def get_option_chains(tickers: list[str], expiration_date: str):
    """
    Get the option chains for a given list of tickers and expiration date.

    Args:
        tickers (list[str]): List of ticker symbols.
        expiration_date (str): Option expiration date.

    Returns:
        pd.DataFrame: Option chains.
    """
    result_dict = {}

    for ticker in tickers:
        response = requests.get(
            f"https://query1.finance.yahoo.com/v7/finance/options/{ticker}?date={expiration_date}",
            timeout=60,
            headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit"
                "/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"
            },
        )

        result = response.json()

        ticker_result = pd.DataFrame(
            result["optionChain"]["result"][0]["options"][0]["calls"]
        )

        if ticker_result.empty:
            continue

        ticker_result = ticker_result.rename(
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

        ticker_result = ticker_result.drop(columns="Contract Size")

        result_dict[ticker] = ticker_result.set_index("Strike")

    result_final = pd.concat(result_dict)

    result_final["Expiration"] = pd.to_datetime(
        result_final["Expiration"], unit="s"
    ).dt.strftime("%Y-%m-%d")
    result_final["Last Trade Date"] = pd.to_datetime(
        result_final["Last Trade Date"], unit="s"
    ).dt.strftime("%Y-%m-%d")

    result_final.index.names = ["Ticker", "Strike Price"]

    return result_final

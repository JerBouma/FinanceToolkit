"""Validation Module"""

__docformat__ = "google"

import re
from collections import Counter

from financetoolkit.utilities import logger_model
from financetoolkit.utilities.requests_model import convert_isin_to_ticker

logger = logger_model.get_logger()

RISK_FREE_RATE_OPTIONS = ["13w", "5y", "10y", "30y"]
ENFORCE_SOURCE_OPTIONS = [None, "FinancialModelingPrep", "YahooFinance"]
INTRADAY_PERIOD_OPTIONS = ["1min", "5min", "15min", "30min", "1hour"]


def validate_toolkit_parameters(
    tickers: list | str | None,
    start_date: str | None,
    end_date: str | None,
    risk_free_rate: str,
    enforce_source: str | None,
    api_key: str,
    intraday_period: str | None,
    benchmark_ticker: str | None,
) -> list[str]:
    """
    Validates the user-facing Toolkit parameters and normalizes the tickers, split out
    of `Toolkit.__init__` so that the input checking is testable in isolation from the
    (data-collecting) initialisation itself. Everything here either raises a clear
    error on invalid input or is a pure normalisation; no state is touched.

    The ticker normalisation upper-cases plain tickers (leaving the special
    "Portfolio" entry as is), converts ISIN codes to tickers, removes duplicates
    while preserving the original order (deduplicating through a set would make the
    column order of every output depend on the hash seed) and removes the benchmark
    ticker from the list, warning about both removals.

    Args:
        tickers (list | str | None): The ticker(s) the Toolkit is initialised with.
        start_date (str | None): The start date, formatted as YYYY-MM-DD.
        end_date (str | None): The end date, formatted as YYYY-MM-DD.
        risk_free_rate (str): The risk free rate duration (13w, 5y, 10y or 30y).
        enforce_source (str | None): The enforced data source, if any.
        api_key (str): The FinancialModelingPrep API key, only used to check that an
        enforced FinancialModelingPrep source actually has one.
        intraday_period (str | None): The intraday period, if any.
        benchmark_ticker (str | None): The benchmark ticker, if any.

    Returns:
        list[str]: The normalized ticker list.

    Raises:
        ValueError: If any of the parameters is invalid.
        TypeError: If tickers is neither a string nor a list of strings.
    """
    if start_date and re.match(r"^\d{4}-\d{2}-\d{2}$", start_date) is None:
        raise ValueError("Please input a valid start date (%Y-%m-%d) like '2010-01-01'")
    if end_date and re.match(r"^\d{4}-\d{2}-\d{2}$", end_date) is None:
        raise ValueError("Please input a valid end date (%Y-%m-%d) like '2020-01-01'")
    if start_date and end_date and start_date > end_date:
        raise ValueError(
            f"Please ensure the start date {start_date} is before the end date {end_date}"
        )

    if risk_free_rate not in RISK_FREE_RATE_OPTIONS:
        raise ValueError("Please select a valid risk free rate (13w, 5y, 10y or 30y)")

    if enforce_source not in ENFORCE_SOURCE_OPTIONS:
        raise ValueError(
            "Please select either FinancialModelingPrep or YahooFinance as the "
            "enforced source."
        )
    if enforce_source == "FinancialModelingPrep" and not api_key:
        raise ValueError(
            "Please input an API key from FinancialModelingPrep if you wish to use "
            "historical data from FinancialModelingPrep."
        )

    if intraday_period and intraday_period not in INTRADAY_PERIOD_OPTIONS:
        raise ValueError(
            "Please select a valid intraday period (1min, 5min, 15min, 30min or 1hour)"
        )

    if isinstance(tickers, str):
        tickers = [tickers.upper()]
    elif isinstance(tickers, list):
        tickers = [
            ticker.upper() if ticker != "Portfolio" else ticker for ticker in tickers
        ]
    elif tickers is None:
        raise ValueError("Please input a ticker or a list of tickers.")
    else:
        raise TypeError("Tickers must be a string or a list of strings.")

    # Check whether the ticker is in ISIN format and if so convert it to a ticker
    ticker_list = [convert_isin_to_ticker(ticker) for ticker in tickers]

    # Take out duplicate tickers if applicable; deduplicating through a set would make the ticker order, and therefore the column order of every single output, depend on the hash seed and change between runs.  # noqa: E501
    deduplicated_tickers = list(dict.fromkeys(ticker_list))

    if len(deduplicated_tickers) != len(ticker_list):
        duplicate_tickers = [
            ticker for ticker, count in Counter(ticker_list).items() if count > 1
        ]
        logger.warning(
            "Found duplicate tickers, duplicate entries of the following tickers are removed: %s",
            ", ".join(duplicate_tickers),
        )
        ticker_list = deduplicated_tickers

    if benchmark_ticker in ticker_list:
        logger.warning(
            "Please note that the benchmark ticker (%s) is also "
            "included in the tickers. Therefore, this ticker will be removed from the "
            "tickers list. If this is not desired, please set the benchmark_ticker to None.",
            benchmark_ticker,
        )
        ticker_list.remove(benchmark_ticker)

    return ticker_list

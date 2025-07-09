"""Fundamentals Model"""

import importlib.util
import threading
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from financetoolkit import fmp_model, normalization_model, yfinance_model
from financetoolkit.utilities import error_model, logger_model

# Check if yfinance is installed
yf_spec = importlib.util.find_spec("yfinance")
ENABLE_YFINANCE = yf_spec is not None


logger = logger_model.get_logger()

# pylint: disable=too-many-locals


def collect_financial_statements(
    tickers: str | list[str],
    statement: str = "",
    api_key: str = "",
    quarter: bool = False,
    start_date: str | None = None,
    end_date: str | None = None,
    rounding: int | None = 4,
    fmp_statement_format: pd.DataFrame = pd.DataFrame(),
    fmp_statistics_format: pd.DataFrame = pd.DataFrame(),
    yf_statement_format: pd.DataFrame = pd.DataFrame(),
    sleep_timer: bool = True,
    progress_bar: bool = True,
    user_subscription: str = "Free",
    enforce_source: str | None = None,
) -> pd.DataFrame:
    """
    Retrieves financial statements (balance, income, or cash flow statements) for one or multiple companies,
    and returns DataFrames containing the data.

    Args:
        tickers (str | list[str]): A single ticker or a list of company tickers.
        statement (str): The type of financial statement to retrieve. Must be "balance", "income", or "cashflow".
        api_key (str): API key for FinancialModelingPrep. Required if enforce_source is "FinancialModelingPrep" or None.
        quarter (bool): Whether to retrieve quarterly data. Defaults to False (annual data).
        start_date (str | None): The start date to filter data with (YYYY-MM-DD). Defaults to None.
        end_date (str | None): The end date to filter data with (YYYY-MM-DD). Defaults to None.
        rounding (int | None): The number of decimals to round the final financial statement data to. Defaults to 4.
        fmp_statement_format (pd.DataFrame): Optional DataFrame defining the desired format for FMP statement data.
                                             Defaults to an empty DataFrame.
        fmp_statistics_format (pd.DataFrame): Optional DataFrame defining the desired format for FMP
                                              statistics data.
                                              Defaults to an empty DataFrame.
        yf_statement_format (pd.DataFrame): Optional DataFrame defining the desired format for Yahoo
                                            Finance statement data.
                                            Defaults to an empty DataFrame.
        yf_statistics_format (pd.DataFrame): Optional DataFrame defining the desired format for Yahoo Finance
                                            statistics data.
                                             Defaults to an empty DataFrame.
        sleep_timer (bool): Whether to pause execution temporarily if the FMP API rate limit is reached. Defaults to True.
        progress_bar (bool): Whether to display a progress bar during data retrieval. Defaults to True.
        user_subscription (str): The FMP subscription plan ("Free", "Starter", etc.). Defaults to "Free".
        enforce_source (str): Specifies the data source to use ("FinancialModelingPrep" or "YahooFinance").
                              If "FinancialModelingPrep", only FMP is used. If "YahooFinance", only Yahoo Finance is used.
                              If None (or any other value), FMP is tried first, and Yahoo Finance is used as a fallback.
                              Defaults to "FinancialModelingPrep".

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, list[str]]:
            - financial_statement_total (pd.DataFrame): A DataFrame containing the formatted and rounded financial
              statement data, indexed by ticker, with columns representing periods (annual or quarterly).
            - financial_statement_statistics (pd.DataFrame): A DataFrame containing the raw data used for statistics
              and normalization, indexed by ticker, with columns representing periods.
            - no_data (list[str]): A list of tickers for which no data could be retrieved from any source.
    """

    def worker(ticker, financial_statement_dict, enforce_source):
        financial_statement_data = pd.DataFrame()
        attempted_fmp = False

        if api_key and enforce_source in [None, "FinancialModelingPrep"]:
            financial_statement_data = fmp_model.get_financial_statement(
                ticker=ticker,
                statement=statement,
                api_key=api_key,
                quarter=quarter,
                start_date=start_date,
                end_date=end_date,
                sleep_timer=sleep_timer,
                user_subscription=user_subscription,
            )

            financial_statement_dict["FinancialModelingPrep"][
                ticker
            ] = financial_statement_data

            if not financial_statement_data.empty:
                fmp_tickers.append(ticker)

            attempted_fmp = True

        if enforce_source != "FinancialModelingPrep" and financial_statement_data.empty:
            if ENABLE_YFINANCE:
                financial_statement_data = yfinance_model.get_financial_statement(
                    ticker=ticker,
                    statement=statement,
                    quarter=quarter,
                    fallback=attempted_fmp,
                )

                financial_statement_dict["YahooFinance"][
                    ticker
                ] = financial_statement_data

            if not financial_statement_data.empty:
                yf_tickers.append(ticker)

        if financial_statement_data.empty:
            no_data.append(ticker)

    if isinstance(tickers, str):
        ticker_list = [tickers]
    elif isinstance(tickers, list):
        ticker_list = tickers
    else:
        raise ValueError(f"Type for the tickers ({type(tickers)}) variable is invalid.")

    if statement not in ["balance", "income", "cashflow"]:
        raise ValueError(
            "Please choose either 'balance', 'income', or "
            "cashflow' for the statement parameter."
        )

    if not api_key and enforce_source == "FinancialModelingPrep":
        raise ValueError(
            "Please enter an API key from FinancialModelingPrep. "
            "For more information, look here: https://www.jeroenbouma.com/fmp"
        )

    ticker_list_iterator = (
        tqdm(ticker_list, desc=f"Obtaining {statement} data")
        if progress_bar
        else ticker_list
    )

    financial_statement_dict: dict[str, pd.DataFrame] = {
        "FinancialModelingPrep": {},
        "YahooFinance": {},
    }
    fmp_tickers: list[str] = []
    yf_tickers: list[str] = []
    no_data: list[str] = []
    threads = []

    for ticker in ticker_list_iterator:
        # Introduce a sleep timer to prevent rate limit errors
        time.sleep(0.1)

        thread = threading.Thread(
            target=worker,
            args=(ticker, financial_statement_dict, enforce_source),
        )
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    fmp_financial_statements_total = pd.DataFrame()
    yf_financial_statements_total = pd.DataFrame()
    fmp_financial_statement_statistics = pd.DataFrame()

    for source, _ in financial_statement_dict.items():
        financial_statement_dict[source] = error_model.check_for_error_messages(
            dataset_dictionary=financial_statement_dict[source],
            user_subscription=user_subscription,
        )

        if source == "FinancialModelingPrep" and financial_statement_dict[source]:
            fmp_financial_statements = pd.concat(
                financial_statement_dict[source], axis=0
            )

            fmp_financial_statement_statistics = (
                normalization_model.convert_financial_statements(
                    financial_statements=fmp_financial_statements,
                    statement_format=fmp_statistics_format,
                    reverse_dates=True,
                )
            )

            fmp_financial_statements_total = (
                normalization_model.convert_financial_statements(
                    financial_statements=fmp_financial_statements,
                    statement_format=fmp_statement_format,
                    reverse_dates=True,
                )
            )
        elif source == "YahooFinance" and financial_statement_dict[source]:
            yf_financial_statements = pd.concat(
                financial_statement_dict[source], axis=0
            )

            yf_financial_statements_total = (
                (
                    normalization_model.convert_financial_statements(
                        financial_statements=yf_financial_statements,
                        statement_format=yf_statement_format,
                        reverse_dates=True,
                    )
                )
                if not yf_financial_statements.empty
                else pd.DataFrame()
            )

    if fmp_tickers and yf_tickers:
        logger.info(
            "The following tickers acquired %s data from FinancialModelingPrep: %s",
            statement,
            ", ".join(fmp_tickers),
        )
        logger.info(
            "The following tickers acquired %s data from YahooFinance: %s",
            statement,
            ", ".join(yf_tickers),
        )

    if yf_tickers and not fmp_tickers and enforce_source == "FinancialModelingPrep":
        logger.info(
            "No data found using FinancialModelingPrep, this is usually due to Bandwidth "
            "API limits or usage of the Free plan. Therefore data was retrieved from YahooFinance instead for: %s",
            ", ".join(yf_tickers),
        )

    if no_data:
        if not ENABLE_YFINANCE:
            logger.info(
                "Due to a missing optional dependency (yfinance) and your current FinancialModelingPrep plan, "
                "data for the following tickers could not be acquired: %s\n"
                "Enable this functionality by using:\033[1m pip install yfinance \033[0m",
                ", ".join(no_data),
            )
        else:
            logger.info(
                "No %s data found for the following tickers: %s",
                statement,
                ", ".join(no_data),
            )

    financial_statement_total = pd.concat(
        [fmp_financial_statements_total, yf_financial_statements_total], axis=0
    )

    financial_statement_statistics = fmp_financial_statement_statistics

    try:
        financial_statement_total = financial_statement_total.astype(np.float64)
    except ValueError as error:
        logger.error(
            "Not able to convert DataFrame to float64 due to %s. This could result in"
            "issues when values are zero and is predominantly relevant for "
            "ratio calculations.",
            error,
        )

    if quarter:
        financial_statement_statistics.columns = pd.PeriodIndex(
            financial_statement_statistics.columns, freq="Q"
        )
        financial_statement_total.columns = pd.PeriodIndex(
            financial_statement_total.columns, freq="Q"
        )
    else:
        financial_statement_statistics.columns = pd.PeriodIndex(
            financial_statement_statistics.columns, freq="Y"
        )

        financial_statement_total.columns = pd.PeriodIndex(
            financial_statement_total.columns, freq="Y"
        )

    financial_statement_statistics = financial_statement_statistics.sort_index(
        axis=1
    ).truncate(before=start_date, after=end_date, axis=1)

    financial_statement_total = financial_statement_total.sort_index(axis=1).truncate(
        before=start_date, after=end_date, axis=1
    )

    # In the case there are columns that have no data over the entire period,
    # these are dropped automatically
    financial_statement_total = financial_statement_total.dropna(axis=1, how="all")

    # Round the financial statement total if rounding is specified
    financial_statement_total = financial_statement_total.round(rounding)

    return (
        financial_statement_total,
        financial_statement_statistics,
        no_data,
    )

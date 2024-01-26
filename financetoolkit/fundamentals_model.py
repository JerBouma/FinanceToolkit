"""Fundamentals Module"""
__docformat__ = "google"


import threading
import time

import numpy as np
import pandas as pd

from financetoolkit import helpers
from financetoolkit.normalization_model import (
    convert_financial_statements,
)

# pylint: disable=no-member,too-many-locals,too-many-lines

try:
    from tqdm import tqdm

    ENABLE_TQDM = True
except ImportError:
    ENABLE_TQDM = False


def get_financial_statements(
    tickers: str | list[str],
    statement: str = "",
    api_key: str = "",
    quarter: bool = False,
    start_date: str | None = None,
    end_date: str | None = None,
    rounding: int | None = 4,
    statement_format: pd.DataFrame = pd.DataFrame(),
    statistics_format: pd.DataFrame = pd.DataFrame(),
    sleep_timer: bool = True,
    progress_bar: bool = True,
) -> pd.DataFrame:
    """
    Retrieves financial statements (balance, income, or cash flow statements) for one or multiple companies,
    and returns a DataFrame containing the data.

    Args:
        tickers (List[str]): List of company tickers.
        statement (str): The type of financial statement to retrieve. Can be "balance", "income", or "cash-flow".
        api_key (str): API key for the financial data provider.
        quarter (bool): Whether to retrieve quarterly data. Defaults to False (annual data).
        start_date (str): The start date to filter data with.
        end_date (str): The end date to filter data with.
        statement_format (pd.DataFrame): Optional DataFrame containing the names of the financial
            statement line items to include in the output. Rows should contain the original name
            of the line item, and columns should contain the desired name for that line item.
        sleep_timer (bool): Whether to set a sleep timer when the rate limit is reached. Note that this only works
        if you have a Premium subscription (Starter or higher) from FinancialModelingPrep. Defaults to False.
        progress_bar (bool): Whether to show a progress bar when retrieving data over 10 tickers. Defaults to True.

    Returns:
        pd.DataFrame: A DataFrame containing the financial statement data. If only one ticker is provided, the
                      returned DataFrame will have a single column containing the data for that ticker. If multiple
                      tickers are provided, the returned DataFrame will have multiple columns, one for each ticker,
                      with the ticker symbol as the column name.
    """

    def worker(ticker, financial_statement_dict):
        url = (
            f"https://financialmodelingprep.com/api/v3/{location}/"
            f"{ticker}?period={period}&apikey={api_key}"
        )
        financial_statement = helpers.get_financial_data(
            url=url,
            sleep_timer=sleep_timer,
        )

        try:
            financial_statement = financial_statement.drop("symbol", axis=1)

            # One day is deducted from the date because it could be that
            # the date is reported as 2023-07-01 while the data is about the
            # second quarter of 2023. This usually happens when companies
            # have a different financial year than the calendar year. It doesn't
            # matter for others that are correctly reporting since 2023-06-31
            # minus one day is still 2023
            financial_statement["date"] = pd.to_datetime(
                financial_statement["date"]
            ) - pd.offsets.Day(1)

            if quarter:
                financial_statement["date"] = financial_statement["date"].dt.to_period(
                    "Q"
                )
            else:
                financial_statement["date"] = pd.to_datetime(
                    financial_statement["calendarYear"].astype(str)
                ).dt.to_period("Y")

            financial_statement = financial_statement.set_index("date").T

            if financial_statement.columns.duplicated().any():
                # This happens in the rare case that a company has two financial statements for the same period.
                # Browsing through the data has shown that these financial statements are equal therefore
                # one of the columns can be dropped.
                financial_statement = financial_statement.loc[
                    :, ~financial_statement.columns.duplicated()
                ]

            financial_statement_dict[ticker] = financial_statement
        except KeyError:
            no_data.append(ticker)
            financial_statement_dict[ticker] = financial_statement

    if isinstance(tickers, str):
        ticker_list = [tickers]
    elif isinstance(tickers, list):
        ticker_list = tickers
    else:
        raise ValueError(f"Type for the tickers ({type(tickers)}) variable is invalid.")

    if statement == "balance":
        location = "balance-sheet-statement"
    elif statement == "income":
        location = "income-statement"
    elif statement == "cashflow":
        location = "cash-flow-statement"
    else:
        raise ValueError(
            "Please choose either 'balance', 'income', or "
            "cashflow' for the statement parameter."
        )

    if not api_key:
        raise ValueError(
            "Please enter an API key from FinancialModelingPrep. "
            "For more information, look here: https://www.jeroenbouma.com/fmp"
        )

    period = "quarter" if quarter else "annual"

    ticker_list_iterator = (
        tqdm(ticker_list, desc=f"Obtaining {statement} data")
        if ENABLE_TQDM & progress_bar
        else ticker_list
    )

    financial_statement_dict: dict[str, pd.DataFrame] = {}
    no_data: list[str] = []
    threads = []

    for ticker in ticker_list_iterator:
        # Introduce a sleep timer to prevent rate limit errors
        time.sleep(0.1)

        thread = threading.Thread(
            target=worker,
            args=(ticker, financial_statement_dict),
        )
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    # Checks if any errors are in the dataset and if this is the case, reports them
    financial_statement_dict = helpers.check_for_error_messages(
        financial_statement_dict
    )

    if no_data:
        print(f"No data found for the following tickers: {', '.join(no_data)}")

    if financial_statement_dict:
        financial_statement_total = pd.concat(financial_statement_dict, axis=0)

        financial_statement_statistics = convert_financial_statements(
            financial_statement_total, statistics_format, True
        )

        financial_statement_total = convert_financial_statements(
            financial_statement_total, statement_format, True
        )

        try:
            financial_statement_total = financial_statement_total.astype(np.float64)
        except ValueError as error:
            print(
                f"Not able to convert DataFrame to float64 due to {error}. This could result in"
                "issues when values are zero and is predominantly relevant for "
                "ratio calculations."
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

        financial_statement_total = financial_statement_total.sort_index(
            axis=1
        ).truncate(before=start_date, after=end_date, axis=1)

        financial_statement_total = financial_statement_total.round(rounding)

        # In the case there are columns that have no data over the entire period,
        # these are dropped automatically
        financial_statement_total = financial_statement_total.dropna(axis=1, how="all")

        return (
            financial_statement_total,
            financial_statement_statistics,
            no_data,
        )

    return pd.DataFrame(), pd.DataFrame(), no_data


def get_revenue_segmentation(
    tickers: str | list[str],
    method: str = "",
    api_key: str = "",
    quarter: bool = False,
    start_date: str | None = None,
    end_date: str | None = None,
    sleep_timer: bool = False,
    progress_bar: bool = True,
) -> pd.DataFrame:
    """
    Retrieves financial statements (balance, income, or cash flow statements) for one or multiple companies,
    and returns a DataFrame containing the data.

    Args:
        tickers (List[str]): List of company tickers.
        statement (str): The type of financial statement to retrieve. Can be "balance", "income", or "cash-flow".
        api_key (str): API key for the financial data provider.
        quarter (bool): Whether to retrieve quarterly data. Defaults to False (annual data).
        start_date (str): The start date to filter data with.
        end_date (str): The end date to filter data with.
        statement_format (pd.DataFrame): Optional DataFrame containing the names of the financial
            statement line items to include in the output. Rows should contain the original name
            of the line item, and columns should contain the desired name for that line item.
        sleep_timer (bool): Whether to set a sleep timer when the rate limit is reached. Note that this only works
        if you have a Premium subscription (Starter or higher) from FinancialModelingPrep. Defaults to False.
        progress_bar (bool): Whether to show a progress bar when retrieving data over 10 tickers. Defaults to True.

    Returns:
        pd.DataFrame: A DataFrame containing the financial statement data. If only one ticker is provided, the
                      returned DataFrame will have a single column containing the data for that ticker. If multiple
                      tickers are provided, the returned DataFrame will have multiple columns, one for each ticker,
                      with the ticker symbol as the column name.
    """

    def worker(ticker, revenue_segmentation_dict):
        url = (
            f"https://financialmodelingprep.com/api/v4/{location}"
            f"?symbol={ticker}&period={period}&structure=flat&apikey={api_key}"
        )
        revenue_segmentation_json = helpers.get_financial_data(
            url=url,
            sleep_timer=sleep_timer,
            raw=True,
        )

        try:
            revenue_segmentation = pd.DataFrame()

            for period_data in revenue_segmentation_json:
                revenue_segmentation = pd.concat(
                    [revenue_segmentation, pd.DataFrame(period_data)], axis=1
                )

            if quarter:
                revenue_segmentation.columns = pd.PeriodIndex(
                    revenue_segmentation.columns, freq="Q"
                )
            else:
                revenue_segmentation.columns = pd.PeriodIndex(
                    revenue_segmentation.columns, freq="Y"
                )

            if revenue_segmentation.columns.duplicated().any():
                # This happens in the rare case that a company has two financial statements for the same period.
                # Browsing through the data has shown that these financial statements are equal therefore
                # one of the columns can be dropped.
                revenue_segmentation = revenue_segmentation.loc[
                    :, ~revenue_segmentation.columns.duplicated()
                ]

            revenue_segmentation = revenue_segmentation.rename(index=naming)
            revenue_segmentation.index = [
                index.lower().title() for index in revenue_segmentation.index
            ]

            # This groups items that have the same naming convention
            revenue_segmentation = revenue_segmentation.groupby(level=0).sum()

            revenue_segmentation_dict[ticker] = revenue_segmentation
        except (KeyError, ValueError):
            no_data.append(ticker)
            revenue_segmentation_dict[ticker] = revenue_segmentation

    if isinstance(tickers, str):
        ticker_list = [tickers]
    elif isinstance(tickers, list):
        ticker_list = tickers
    else:
        raise ValueError(f"Type for the tickers ({type(tickers)}) variable is invalid.")

    if method == "geographic":
        location = "revenue-geographic-segmentation"
        naming = {
            "U S": "United States",
            "U": "United States",
            "C N": "China",
            "N O": "North America",
            "Non Us": "Non-US",
            "Americas Segment": "Americas",
            "Europe Segment": "Europe",
            "Greater China Segment": "China",
            "Japan Segment": "Japan",
            "Rest of Asia Pacific Segment": "Asia Pacific",
            "Asia-Pacific": "Asia Pacific",
            "J P": "Japan",
            "North America Segment": "North America",
            "TAIWAN, PROVINCE OF CHINA": "Taiwan",
            "Segment, Geographical, Rest of the World, Excluding United States and United Kingdom": "Rest Of The World",
            "D E": "Germany",
        }
    elif method == "product":
        location = "revenue-product-segmentation"
        naming = {}
    else:
        raise ValueError(
            "Please choose either 'balance', 'income', or "
            "cashflow' for the statement parameter."
        )

    if not api_key:
        raise ValueError(
            "Please enter an API key from FinancialModelingPrep. "
            "For more information, look here: https://www.jeroenbouma.com/fmp"
        )

    period = "quarter" if quarter else "annual"

    revenue_segmentation_dict: dict = {}
    no_data: list[str] = []
    threads = []

    ticker_list_iterator = (
        tqdm(ticker_list, desc=f"Obtaining {method} segmentation data")
        if ENABLE_TQDM & progress_bar
        else ticker_list
    )

    for ticker in ticker_list_iterator:
        # Introduce a sleep timer to prevent rate limit errors
        time.sleep(0.1)

        thread = threading.Thread(
            target=worker,
            args=(ticker, revenue_segmentation_dict),
        )
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    # Checks if any errors are in the dataset and if this is the case, reports them
    revenue_segmentation_dict = helpers.check_for_error_messages(
        dataset_dictionary=revenue_segmentation_dict,
        subscription_type="Professional or Enterprise",
    )

    if no_data:
        print(f"No data found for the following tickers: {', '.join(no_data)}")

    if revenue_segmentation_dict:
        revenue_segmentation_total = pd.concat(revenue_segmentation_dict, axis=0)

        try:
            revenue_segmentation_total = revenue_segmentation_total.astype(np.float64)
        except ValueError as error:
            print(
                f"Not able to convert DataFrame to float64 due to {error}. This could result in"
                "issues when values are zero and is predominantly relevant for "
                "ratio calculations."
            )

        revenue_segmentation_total = revenue_segmentation_total.sort_index(
            axis=1
        ).truncate(before=start_date, after=end_date, axis=1)

        if quarter:
            revenue_segmentation_total.columns = pd.PeriodIndex(
                revenue_segmentation_total.columns, freq="Q"
            )
        else:
            revenue_segmentation_total.columns = pd.PeriodIndex(
                revenue_segmentation_total.columns, freq="Y"
            )

        # Check whether the rows sum to zero, if so with the current start and end date there is no data
        # for those rows and thus they can be dropped out of the dataset to clean it up
        revenue_segmentation_total = revenue_segmentation_total[
            revenue_segmentation_total.sum(axis=1) != 0
        ]

        return (
            revenue_segmentation_total,
            no_data,
        )

    return pd.DataFrame(), no_data


def get_analyst_estimates(
    tickers: str | list[str],
    api_key: str = "",
    quarter: bool = False,
    start_date: str | None = None,
    rounding: int | None = 4,
    sleep_timer: bool = False,
    progress_bar: bool = True,
) -> pd.DataFrame:
    """
    Retrieves analyst estimates for one or multiple companies, and returns a DataFrame containing the data.

    This data contains the following estimates:
        - Estimated Revenue Low
        - Estimated Revenue High
        - Estimated Revenue Average
        - Estimated EBITDA Low
        - Estimated EBITDA High
        - Estimated EBITDA Average
        - Estimated EBIT Low
        - Estimated EBIT High
        - Estimated EBIT Average
        - Estimated Net Income Low
        - Estimated Net Income High
        - Estimated Net Income Average
        - Estimated SGA Expense Low
        - Estimated SGA Expense High
        - Estimated SGA Expense Average
        - Estimated EPS Low
        - Estimated EPS High
        - Estimated EPS Average
        - Number of Analysts

    Args:
        tickers (List[str]): List of company tickers.
        api_key (str): API key for the financial data provider.
        quarter (bool): Whether to retrieve quarterly data. Defaults to False (annual data).
        start_date (str): The start date to filter data with.
        sleep_timer (bool): Whether to set a sleep timer when the rate limit is reached. Note that this only works
        if you have a Premium subscription (Starter or higher) from FinancialModelingPrep. Defaults to False.
        progress_bar (bool): Whether to show a progress bar when retrieving data over 10 tickers. Defaults to True.

    Returns:
        pd.DataFrame: A DataFrame containing the analyst estimates for all provided tickers.
    """

    def worker(ticker, analyst_estimates_dict):
        url = (
            "https://financialmodelingprep.com/api/v3/analyst-estimates/"
            f"{ticker}?period={period}&apikey={api_key}"
        )
        analyst_estimates = helpers.get_financial_data(url=url, sleep_timer=sleep_timer)

        try:
            analyst_estimates = analyst_estimates.drop("symbol", axis=1)

            if quarter:
                analyst_estimates["date"] = pd.to_datetime(
                    analyst_estimates["date"]
                ).dt.to_period("Q")
            else:
                analyst_estimates["date"] = pd.to_datetime(
                    analyst_estimates["date"].astype(str)
                ).dt.to_period("Y")

            analyst_estimates = analyst_estimates.set_index("date").T

            if analyst_estimates.columns.duplicated().any():
                # This happens in the rare case that a company has two financial statements for the same period.
                # Browsing through the data has shown that these financial statements are equal therefore
                # one of the columns can be dropped.
                analyst_estimates = analyst_estimates.loc[
                    :, ~analyst_estimates.columns.duplicated()
                ]

            analyst_estimates.loc["Number of Analysts", :] = (
                analyst_estimates.loc["numberAnalystEstimatedRevenue", :]
                + analyst_estimates.loc["numberAnalystsEstimatedEps", :]
            ) // 2
            analyst_estimates = analyst_estimates.drop(
                ["numberAnalystEstimatedRevenue", "numberAnalystsEstimatedEps"], axis=0
            )

            analyst_estimates_dict[ticker] = analyst_estimates.rename(index=naming)
        except KeyError:
            no_data.append(ticker)
            analyst_estimates_dict[ticker] = analyst_estimates

    naming: dict = {
        "estimatedRevenueLow": "Estimated Revenue Low",
        "estimatedRevenueHigh": "Estimated Revenue High",
        "estimatedRevenueAvg": "Estimated Revenue Average",
        "estimatedEbitdaLow": "Estimated EBITDA Low",
        "estimatedEbitdaHigh": "Estimated EBITDA High",
        "estimatedEbitdaAvg": "Estimated EBITDA Average",
        "estimatedEbitLow": "Estimated EBIT Low",
        "estimatedEbitHigh": "Estimated EBIT High",
        "estimatedEbitAvg": "Estimated EBIT Average",
        "estimatedNetIncomeLow": "Estimated Net Income Low",
        "estimatedNetIncomeHigh": "Estimated Net Income High",
        "estimatedNetIncomeAvg": "Estimated Net Income Average",
        "estimatedSgaExpenseLow": "Estimated SGA Expense Low",
        "estimatedSgaExpenseHigh": "Estimated SGA Expense High",
        "estimatedSgaExpenseAvg": "Estimated SGA Expense Average",
        "estimatedEpsLow": "Estimated EPS Low",
        "estimatedEpsHigh": "Estimated EPS High",
        "estimatedEpsAvg": "Estimated EPS Average",
    }

    if isinstance(tickers, str):
        ticker_list = [tickers]
    elif isinstance(tickers, list):
        ticker_list = tickers
    else:
        raise ValueError(f"Type for the tickers ({type(tickers)}) variable is invalid.")

    if not api_key:
        raise ValueError(
            "Please enter an API key from FinancialModelingPrep. "
            "For more information, look here: https://www.jeroenbouma.com/fmp"
        )

    period = "quarter" if quarter else "annual"

    analyst_estimates_dict: dict = {}
    no_data: list[str] = []
    threads = []

    ticker_list_iterator = (
        tqdm(ticker_list, desc="Obtaining analyst estimates")
        if ENABLE_TQDM & progress_bar
        else ticker_list
    )

    for ticker in ticker_list_iterator:
        # Introduce a sleep timer to prevent rate limit errors
        time.sleep(0.1)

        thread = threading.Thread(
            target=worker,
            args=(ticker, analyst_estimates_dict),
        )
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    # Checks if any errors are in the dataset and if this is the case, reports them
    analyst_estimates_dict = helpers.check_for_error_messages(analyst_estimates_dict)

    if no_data:
        print(f"No data found for the following tickers: {', '.join(no_data)}")

    if analyst_estimates_dict:
        analyst_estimates_total = pd.concat(analyst_estimates_dict, axis=0)

        try:
            analyst_estimates_total = analyst_estimates_total.astype(np.float64)
            analyst_estimates_total.loc[:, "Number of Analysts", :].fillna(0).astype(
                int
            )
        except ValueError as error:
            print(
                f"Not able to convert DataFrame to float64 and int due to {error}. This could result in"
                "issues when values are zero and is predominantly relevant for "
                "ratio calculations."
            )

        analyst_estimates_total.columns = pd.PeriodIndex(
            analyst_estimates_total.columns, freq="Q" if quarter else "Y"
        )

        analyst_estimates_total = analyst_estimates_total.sort_index(axis=1).truncate(
            before=start_date, axis=1
        )

        analyst_estimates_total = analyst_estimates_total.round(rounding)

        if quarter:
            analyst_estimates_total.columns = pd.PeriodIndex(
                analyst_estimates_total.columns, freq="Q"
            )
        else:
            analyst_estimates_total.columns = pd.PeriodIndex(
                analyst_estimates_total.columns, freq="Y"
            )

        return (
            analyst_estimates_total,
            no_data,
        )

    return pd.DataFrame(), no_data


def get_profile(
    tickers: list[str] | str,
    api_key: str,
    progress_bar: bool = True,
    report_missing: bool = True,
) -> pd.DataFrame:
    """
    Gives information about the profile of a company which includes i.a. beta, company description, industry and sector.

    Args:
        ticker (list or string): the company ticker (for example: "AAPL")
        api_key (string): the API Key obtained from
        https://www.jeroenbouma.com/fmp
        progress_bar (bool): Whether to show a progress bar. Defaults to True.

    Returns:
        pd.DataFrame: the profile data.
    """

    def worker(ticker, profile_dict):
        url = f"https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey={api_key}"
        profile_data = helpers.get_financial_data(url=url)

        if profile_data.empty:
            no_data.append(ticker)
            profile_dict[ticker] = profile_data
        else:
            profile_dict[ticker] = profile_data.T

    naming: dict = {
        "symbol": "Symbol",
        "price": "Price",
        "beta": "Beta",
        "volAvg": "Average Volume",
        "mktCap": "Market Capitalization",
        "lastDiv": "Last Dividend",
        "range": "Range",
        "changes": "Changes",
        "companyName": "Company Name",
        "currency": "Currency",
        "cik": "CIK",
        "isin": "ISIN",
        "cusip": "CUSIP",
        "exchange": "Exchange",
        "exchangeShortName": "Exchange Short Name",
        "industry": "Industry",
        "website": "Website",
        "description": "Description",
        "ceo": "CEO",
        "sector": "Sector",
        "country": "Country",
        "fullTimeEmployees": "Full Time Employees",
        "phone": "Phone",
        "address": "Address",
        "city": "City",
        "state": "State",
        "zip": "ZIP Code",
        "dcfDiff": "DCF Difference",
        "dcf": "DCF",
        "ipoDate": "IPO Date",
    }

    if isinstance(tickers, str):
        ticker_list = [tickers]
    elif isinstance(tickers, list):
        ticker_list = tickers
    else:
        raise ValueError(f"Type for the tickers ({type(tickers)}) variable is invalid.")

    profile_dict: dict[str, pd.DataFrame] = {}
    no_data: list[str] = []
    threads = []

    ticker_list_iterator = (
        tqdm(ticker_list, desc="Obtaining company profiles")
        if ENABLE_TQDM & progress_bar
        else ticker_list
    )

    for ticker in ticker_list_iterator:
        # Introduce a sleep timer to prevent rate limit errors
        time.sleep(0.1)

        thread = threading.Thread(
            target=worker,
            args=(ticker, profile_dict),
        )
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    # Checks if any errors are in the dataset and if this is the case, reports them
    profile_dict = helpers.check_for_error_messages(profile_dict)

    if no_data and report_missing:
        print(f"No data found for the following tickers: {', '.join(no_data)}")

    if profile_dict:
        try:
            profile_dataframe = pd.concat(profile_dict)[0].unstack(level=0)
            profile_dataframe = profile_dataframe.rename(index=naming)
            profile_dataframe = profile_dataframe.drop(
                [
                    "image",
                    "defaultImage",
                    "isEtf",
                    "isActivelyTrading",
                    "isAdr",
                    "isFund",
                ],
                axis=0,
            )
        except (ValueError, KeyError):
            return pd.DataFrame(), no_data

        return profile_dataframe, no_data

    return pd.DataFrame(), no_data


def get_quote(
    tickers: list[str] | str, api_key: str, progress_bar: bool = True
) -> pd.DataFrame:
    """
    Gives information about the quote of a company which includes i.a. high/low close prices,
    price-to-earning ratio and shares outstanding.

    Args:
        ticker (list or string): the company ticker (for example: "AMD")
        api_key (string): the API Key obtained from
        https://www.jeroenbouma.com/fmp
        progress_bar (bool): Whether to show a progress bar. Defaults to True.

    Returns:
        pd.DataFrame: the quote data.
    """

    def worker(ticker, quote_dict):
        url = (
            f"https://financialmodelingprep.com/api/v3/quote/{ticker}?apikey={api_key}"
        )
        quote_data = helpers.get_financial_data(url=url)

        if quote_data.empty:
            no_data.append(ticker)
            quote_dict[ticker] = quote_data
        else:
            quote_dict[ticker] = quote_data.T

    naming: dict = {
        "symbol": "Symbol",
        "name": "Name",
        "price": "Price",
        "changesPercentage": "Changes Percentage",
        "change": "Change",
        "dayLow": "Day Low",
        "dayHigh": "Day High",
        "yearHigh": "Year High",
        "yearLow": "Year Low",
        "marketCap": "Market Capitalization",
        "priceAvg50": "Price Average 50 Days",
        "priceAvg200": "Price Average 200 Days",
        "exchange": "Exchange",
        "volume": "Volume",
        "avgVolume": "Average Volume",
        "open": "Open",
        "previousClose": "Previous Close",
        "eps": "EPS",
        "pe": "PE",
        "earningsAnnouncement": "Earnings Announcement",
        "sharesOutstanding": "Shares Outstanding",
        "timestamp": "Timestamp",
    }

    if isinstance(tickers, str):
        ticker_list = [tickers]
    elif isinstance(tickers, list):
        ticker_list = tickers
    else:
        raise ValueError(f"Type for the tickers ({type(tickers)}) variable is invalid.")

    quote_dict: dict[str, pd.DataFrame] = {}
    no_data: list[str] = []
    threads = []

    ticker_list_iterator = (
        tqdm(ticker_list, desc="Obtaining company quotes")
        if ENABLE_TQDM & progress_bar
        else ticker_list
    )

    for ticker in ticker_list_iterator:
        # Introduce a sleep timer to prevent rate limit errors
        time.sleep(0.1)

        thread = threading.Thread(
            target=worker,
            args=(ticker, quote_dict),
        )
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    # Checks if any errors are in the dataset and if this is the case, reports them
    quote_dict = helpers.check_for_error_messages(quote_dict)

    if no_data:
        print(f"No data found for the following tickers: {', '.join(no_data)}")

    if quote_dict:
        quote_dataframe = pd.concat(quote_dict)[0].unstack(level=0)
        quote_dataframe = quote_dataframe.rename(index=naming)

    return quote_dataframe, no_data


def get_rating(tickers: list[str] | str, api_key: str, progress_bar: bool = True):
    """
    Gives information about the rating of a company which includes i.a. the company rating and
    recommendation as well as ratings based on a variety of ratios.

    Args:
        ticker (list or string): the company ticker (for example: "MSFT")
        api_key (string): the API Key obtained from
        https://www.jeroenbouma.com/fmp
        progress_bar (bool): Whether to show a progress bar. Defaults to True.

    Returns:
        pd.DataFrame: the rating data.
    """

    def worker(ticker, ratings_dict):
        url = f"https://financialmodelingprep.com/api/v3/historical-rating/{ticker}?l&apikey={api_key}"
        ratings = helpers.get_financial_data(url=url)

        try:
            ratings = ratings.drop("symbol", axis=1).sort_values(
                by="date", ascending=True
            )

            ratings = ratings.set_index("date")

            ratings = ratings.rename(
                columns={
                    "rating": "Rating",
                    "ratingScore": "Rating Score",
                    "ratingRecommendation": "Rating Recommendation",
                    "ratingDetailsDCFScore": "DCF Score",
                    "ratingDetailsDCFRecommendation": "DCF Recommendation",
                    "ratingDetailsROEScore": "ROE Score",
                    "ratingDetailsROERecommendation": "ROE Recommendation",
                    "ratingDetailsROAScore": "ROA Score",
                    "ratingDetailsROARecommendation": "ROA Recommendation",
                    "ratingDetailsDEScore": "DE Score",
                    "ratingDetailsDERecommendation": "DE Recommendation",
                    "ratingDetailsPEScore": "PE Score",
                    "ratingDetailsPERecommendation": "PE Recommendation",
                    "ratingDetailsPBScore": "PB Score",
                    "ratingDetailsPBRecommendation": "PB Recommendation",
                }
            )

            ratings_dict[ticker] = ratings
        except (KeyError, ValueError):
            no_data.append(ticker)
            ratings_dict[ticker] = ratings

    if isinstance(tickers, str):
        ticker_list = [tickers]
    elif isinstance(tickers, list):
        ticker_list = tickers
    else:
        raise ValueError(f"Type for the tickers ({type(tickers)}) variable is invalid.")

    ratings_dict: dict[str, pd.DataFrame] = {}
    no_data: list[str] = []
    threads = []

    ticker_list_iterator = (
        tqdm(ticker_list, desc="Obtaining company ratings")
        if ENABLE_TQDM & progress_bar
        else ticker_list
    )

    for ticker in ticker_list_iterator:
        # Introduce a sleep timer to prevent rate limit errors
        time.sleep(0.1)

        thread = threading.Thread(
            target=worker,
            args=(ticker, ratings_dict),
        )
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    # Checks if any errors are in the dataset and if this is the case, reports them
    ratings_dict = helpers.check_for_error_messages(ratings_dict)

    if no_data:
        print(f"No data found for the following tickers: {', '.join(no_data)}")

    if ratings_dict:
        ratings_dataframe = pd.concat(ratings_dict, axis=0).dropna()

        if len(ticker_list) == 1:
            ratings_dataframe = ratings_dataframe.loc[ticker_list[0]]

        return ratings_dataframe, no_data

    return pd.DataFrame(), no_data


def get_earnings_calendar(
    tickers: list[str] | str,
    api_key: str,
    start_date: str | None = None,
    end_date: str | None = None,
    actual_dates: bool = True,
    sleep_timer: bool = False,
    progress_bar: bool = True,
):
    """
    Obtains Earnings Calendar which shows the expected earnings and EPS for a company.

    Args:
        ticker (list or string): the company ticker (for example: "MSFT")
        api_key (string): the API Key obtained from
        https://www.jeroenbouma.com/fmp
        start_date (str): The start date to filter data with.
        end_date (str): The end date to filter data with.
        actual_dates (bool): Whether to retrieve actual dates. Defaults to False (converted to quarterly). This is the
        default because the actual date refers to the corresponding quarter.
        sleep_timer (bool): Whether to set a sleep timer when the rate limit is reached. Note that this only works
        if you have a Premium subscription (Starter or higher) from FinancialModelingPrep. Defaults to False.
        progress_bar (bool): Whether to show a progress bar when retrieving data over 10 tickers. Defaults to True.

    Returns:
        pd.DataFrame: the earnings calendar data.
    """

    def worker(ticker, earnings_calendar_dict):
        url = (
            "https://financialmodelingprep.com/api/v3/historical/earning_calendar/"
            f"{ticker}?apikey={api_key}"
        )
        earnings_calendar = helpers.get_financial_data(url=url, sleep_timer=sleep_timer)

        try:
            earnings_calendar["date"] = (
                pd.to_datetime(earnings_calendar["date"])
                if actual_dates
                else pd.to_datetime(earnings_calendar["date"]).dt.to_period("Q")
            )

            earnings_calendar = earnings_calendar.set_index("date").sort_index()

            if earnings_calendar.columns.duplicated().any():
                # This happens in the rare case that a company has two financial statements for the same period.
                # Browsing through the data has shown that these financial statements are equal therefore
                # one of the columns can be dropped.
                earnings_calendar = earnings_calendar.loc[
                    :, ~earnings_calendar.columns.duplicated()
                ]

            earnings_calendar = earnings_calendar.drop(["updatedFromDate"], axis=1)
            earnings_calendar = earnings_calendar.rename(columns=naming)

            earnings_calendar = earnings_calendar.sort_index(axis=0).truncate(
                before=start_date, after=end_date, axis=0
            )

            earnings_calendar_dict[ticker] = earnings_calendar[naming.values()]
        except KeyError:
            no_data.append(ticker)
            earnings_calendar_dict[ticker] = earnings_calendar

    naming: dict = {
        "eps": "EPS",
        "epsEstimated": "Estimated EPS",
        "revenue": "Revenue",
        "revenueEstimated": "Estimated Revenue",
        "fiscalDateEnding": "Fiscal Date Ending",
        "time": "Time",
    }

    if isinstance(tickers, str):
        ticker_list = [tickers]
    elif isinstance(tickers, list):
        ticker_list = tickers
    else:
        raise ValueError(f"Type for the tickers ({type(tickers)}) variable is invalid.")

    if not api_key:
        raise ValueError(
            "Please enter an API key from FinancialModelingPrep. "
            "For more information, look here: https://www.jeroenbouma.com/fmp"
        )

    earnings_calendar_dict: dict = {}
    no_data: list[str] = []
    threads = []

    ticker_list_iterator = (
        tqdm(ticker_list, desc="Obtaining earnings calendars")
        if ENABLE_TQDM & progress_bar
        else ticker_list
    )

    for ticker in ticker_list_iterator:
        # Introduce a sleep timer to prevent rate limit errors
        time.sleep(0.1)

        thread = threading.Thread(
            target=worker,
            args=(ticker, earnings_calendar_dict),
        )
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    # Checks if any errors are in the dataset and if this is the case, reports them
    earnings_calendar_dict = helpers.check_for_error_messages(earnings_calendar_dict)

    if no_data:
        print(f"No data found for the following tickers: {', '.join(no_data)}")

    if earnings_calendar_dict:
        earnings_calendar_total = pd.concat(earnings_calendar_dict, axis=0)

        return (
            earnings_calendar_total,
            no_data,
        )

    return pd.DataFrame(), no_data


def get_dividend_calendar(
    tickers: list[str] | str,
    api_key: str,
    start_date: str | None = None,
    end_date: str | None = None,
    sleep_timer: bool = False,
    progress_bar: bool = True,
):
    """
    Obtains Dividend Calendar which shows the dividends and related dates.

    Args:
        ticker (list or string): the company ticker (for example: "MSFT")
        api_key (string): the API Key obtained from
        https://www.jeroenbouma.com/fmp
        start_date (str): The start date to filter data with.
        end_date (str): The end date to filter data with.
        sleep_timer (bool): Whether to set a sleep timer when the rate limit is reached. Note that this only works
        if you have a Premium subscription (Starter or higher) from FinancialModelingPrep. Defaults to False.
        progress_bar (bool): Whether to show a progress bar when retrieving data over 10 tickers. Defaults to True.

    Returns:
        pd.DataFrame: the earnings calendar data.
    """

    def worker(ticker, dividend_calendar_dict):
        url = (
            "https://financialmodelingprep.com/api/v3/historical-price-full/stock_dividend/"
            f"{ticker}?apikey={api_key}&from={start_date}&to={end_date}"
        )
        dividend_calendar = helpers.get_financial_data(
            url=url, sleep_timer=sleep_timer, raw=True
        )

        try:
            dividend_calendar = pd.DataFrame(dividend_calendar["historical"])

            if "date" not in dividend_calendar.columns:
                no_data.append(ticker)
            else:
                dividend_calendar = dividend_calendar.set_index("date")

                dividend_calendar.index = pd.to_datetime(dividend_calendar.index)
                dividend_calendar.index = dividend_calendar.index.to_period(freq="D")

                dividend_calendar = dividend_calendar.sort_index()

                dividend_calendar = dividend_calendar.drop(["label"], axis=1)
                dividend_calendar = dividend_calendar.rename(columns=naming)

                dividend_calendar = dividend_calendar.sort_index(axis=0).truncate(
                    before=start_date, after=end_date, axis=0
                )

                dividend_calendar_dict[ticker] = dividend_calendar[naming.values()]
        except KeyError:
            no_data.append(ticker)
            dividend_calendar_dict[ticker] = dividend_calendar

    naming: dict = {
        "adjDividend": "Adj Dividend",
        "dividend": "Dividend",
        "recordDate": "Record Date",
        "paymentDate": "Payment Date",
        "declarationDate": "Declaration Date",
    }

    if isinstance(tickers, str):
        ticker_list = [tickers]
    elif isinstance(tickers, list):
        ticker_list = tickers
    else:
        raise ValueError(f"Type for the tickers ({type(tickers)}) variable is invalid.")

    if not api_key:
        raise ValueError(
            "Please enter an API key from FinancialModelingPrep. "
            "For more information, look here: https://www.jeroenbouma.com/fmp"
        )

    dividend_calendar_dict: dict = {}
    no_data: list[str] = []
    threads = []

    ticker_list_iterator = (
        tqdm(ticker_list, desc="Obtaining dividend calendars")
        if ENABLE_TQDM & progress_bar
        else ticker_list
    )

    for ticker in ticker_list_iterator:
        # Introduce a sleep timer to prevent rate limit errors
        time.sleep(0.1)

        thread = threading.Thread(
            target=worker,
            args=(ticker, dividend_calendar_dict),
        )
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    # Checks if any errors are in the dataset and if this is the case, reports them
    dividend_calendar_dict = helpers.check_for_error_messages(dividend_calendar_dict)

    if no_data:
        print(f"No data found for the following tickers: {', '.join(no_data)}")

    if dividend_calendar_dict:
        dividend_calendar_total = pd.concat(dividend_calendar_dict, axis=0)

        return (
            dividend_calendar_total,
            no_data,
        )

    return pd.DataFrame(), no_data


def get_esg_scores(
    tickers: list[str] | str,
    api_key: str,
    quarter: bool = False,
    start_date: str | None = None,
    end_date: str | None = None,
    sleep_timer: bool = False,
    progress_bar: bool = True,
):
    """
    Obtains the ESG Scores for a selection of companies.

    Args:
        ticker (list or string): the company ticker (for example: "MSFT")
        api_key (string): the API Key obtained from
        https://www.jeroenbouma.com/fmp
        start_date (str): The start date to filter data with.
        end_date (str): The end date to filter data with.
        sleep_timer (bool): Whether to set a sleep timer when the rate limit is reached. Note that this only works
        if you have a Premium subscription (Starter or higher) from FinancialModelingPrep. Defaults to False.
        progress_bar (bool): Whether to show a progress bar when retrieving data over 10 tickers. Defaults to True.

    Returns:
        pd.DataFrame: the earnings calendar data.
    """

    def worker(ticker, esg_scores_dict):
        url = (
            "https://financialmodelingprep.com/api/v4/esg-environmental-social-governance-data?"
            f"symbol={ticker}&apikey={api_key}"
        )
        esg_scores = helpers.get_financial_data(url=url, sleep_timer=sleep_timer)

        try:
            if "date" not in esg_scores.columns:
                no_data.append(ticker)
            else:
                # One day is deducted from the date because it could be that
                # the date is reported as 2023-07-01 while the data is about the
                # second quarter of 2023. This usually happens when companies
                # have a different financial year than the calendar year. It doesn't
                # matter for others that are correctly reporting since 2023-06-31
                # minus one day is still 2023 Q2.
                esg_scores["date"] = pd.to_datetime(
                    esg_scores["date"]
                ) - pd.offsets.Day(1)

                esg_scores = esg_scores.set_index("date")
                esg_scores.index = esg_scores.index.to_period(
                    freq="Q" if quarter else "Y"
                )

                esg_scores = esg_scores.sort_index()
                esg_scores = esg_scores.rename(columns=naming)

                esg_scores = esg_scores.sort_index(axis=0).truncate(
                    before=start_date, after=end_date, axis=0
                )

                esg_scores = esg_scores[~esg_scores.index.duplicated()]

                esg_scores_dict[ticker] = esg_scores[naming.values()]
        except KeyError:
            no_data.append(ticker)
            esg_scores_dict[ticker] = esg_scores

    naming: dict = {
        "environmentalScore": "Environmental Score",
        "socialScore": "Social Score",
        "governanceScore": "Governance Score",
        "ESGScore": "ESG Score",
    }

    if isinstance(tickers, str):
        ticker_list = [tickers]
    elif isinstance(tickers, list):
        ticker_list = tickers
    else:
        raise ValueError(f"Type for the tickers ({type(tickers)}) variable is invalid.")

    if not api_key:
        raise ValueError(
            "Please enter an API key from FinancialModelingPrep. "
            "For more information, look here: https://www.jeroenbouma.com/fmp"
        )

    esg_scores_dict: dict = {}
    no_data: list[str] = []
    threads = []

    ticker_list_iterator = (
        tqdm(ticker_list, desc="Obtaining ESG scores")
        if ENABLE_TQDM & progress_bar
        else ticker_list
    )

    for ticker in ticker_list_iterator:
        # Introduce a sleep timer to prevent rate limit errors
        time.sleep(0.1)

        thread = threading.Thread(
            target=worker,
            args=(ticker, esg_scores_dict),
        )
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    # Checks if any errors are in the dataset and if this is the case, reports them
    esg_scores_dict = helpers.check_for_error_messages(esg_scores_dict)

    if no_data:
        print(f"No data found for the following tickers: {', '.join(no_data)}")

    if esg_scores_dict:
        esg_scores_total = pd.concat(esg_scores_dict, axis=0).unstack(level=0)

        return (
            esg_scores_total,
            no_data,
        )

    return pd.DataFrame(), no_data

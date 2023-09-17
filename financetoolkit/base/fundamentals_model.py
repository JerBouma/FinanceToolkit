"""Fundamentals Module"""
__docformat__ = "google"


import time
from io import StringIO

import numpy as np
import pandas as pd
import requests

# pylint: disable=no-member,too-many-locals

try:
    from tqdm import tqdm

    ENABLE_TQDM = True
except ImportError:
    ENABLE_TQDM = False

from financetoolkit.base.normalization_model import (
    convert_financial_statements,
)


def get_financial_data(
    ticker: str,
    url: str,
    sleep_timer: bool = False,
    subscription_type: str = "Premium",
    raw: bool = False,
) -> pd.DataFrame:
    """
    Collects the financial data from the FinancialModelingPrep API. This is a
    separate function to properly segregate the different types of errors that can occur.

    Args:
        ticker (str): The company ticker (for example: "AAPL")
        url (str): The url to retrieve the data from.
        sleep_timer (bool): Whether to set a sleep timer when the rate limit is reached. Note that this only works
        if you have a Premium subscription (Starter or higher) from FinancialModelingPrep. Defaults to False.
        raw (bool): Whether to return the raw JSON data. Defaults to False.

    Returns:
        pd.DataFrame: A DataFrame containing the financial data.
    """
    while True:
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()

            if raw:
                return response.json()

            json_io = StringIO(response.text)

            financial_data = pd.read_json(json_io)

            return financial_data

        except requests.exceptions.HTTPError:
            if (
                "not available under your current subscription"
                in response.json()["Error Message"]
            ):
                print(
                    f"The requested data for {ticker} is part of the {subscription_type} Subscription from "
                    "FinancialModelingPrep. If you wish to access this data, consider upgrading "
                    "your plan.\nYou can get 15% off by using the following affiliate link which "
                    "also supports the project: https://site.financialmodelingprep.com/developer/docs/pricing/jeroen/"
                )
                return pd.DataFrame(columns=["ERROR_MESSAGE"])

            if "Limit Reach" in response.json()["Error Message"]:
                print(
                    "You have reached the rate limit of your subscription, set sleep_timer in the Toolkit class "
                    "to True to continue collecting data (Premium only). If you use the Free plan, consider upgrading "
                    "your plan.\nYou can get 15% off by using the following affiliate link which also supports the "
                    "project: https://site.financialmodelingprep.com/developer/docs/pricing/jeroen/"
                )

                if sleep_timer:
                    time.sleep(60)
                else:
                    return pd.DataFrame(columns=["ERROR_MESSAGE"])
            if (
                "Free plan is limited to US stocks only"
                in response.json()["Error Message"]
            ):
                print(
                    f"The Free plan is limited to US stocks only (therefore, {ticker} is unavailable), consider "
                    "upgrading your plan to Starter or higher.\nYou can get 15% off by using the following affiliate "
                    "link which also supports the project: "
                    "https://site.financialmodelingprep.com/developer/docs/pricing/jeroen/"
                )
                return pd.DataFrame(columns=["ERROR_MESSAGE"])

            if "Invalid API KEY." in response.json()["Error Message"]:
                print(
                    "You have entered an invalid API key from FinancialModelingPrep. Obtain your API key for free "
                    "and get 15% off the Premium plans by using the following affiliate link.\nThis also supports "
                    "the project: https://site.financialmodelingprep.com/developer/docs/pricing/jeroen/"
                )
                return pd.DataFrame(columns=["ERROR_MESSAGE"])

            print(
                "This is an undefined error, please report to the author at https://github.com/JerBouma/FinanceToolkit"
                f"\n{response.json()['Error Message']}"
            )

            return pd.DataFrame(columns=["ERROR_MESSAGE"])


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
            "For more information, look here: https://site.financialmodelingprep.com/developer/docs/pricing/jeroen/"
        )

    period = "quarter" if quarter else "annual"

    financial_statement_dict: dict = {}
    invalid_tickers = []

    ticker_list_iterator = (
        tqdm(ticker_list, desc=f"Obtaining {statement} data")
        if ENABLE_TQDM & progress_bar
        else ticker_list
    )

    for ticker in ticker_list_iterator:
        url = (
            f"https://financialmodelingprep.com/api/v3/{location}/"
            f"{ticker}?period={period}&apikey={api_key}"
        )
        financial_statement = get_financial_data(
            ticker=ticker,
            url=url,
            sleep_timer=sleep_timer,
        )

        if "ERROR_MESSAGE" in financial_statement:
            invalid_tickers.append(ticker)
            continue

        try:
            financial_statement = financial_statement.drop("symbol", axis=1)
        except KeyError:
            print(f"No financial statement data found for {ticker}")
            invalid_tickers.append(ticker)
            continue

        if quarter:
            financial_statement["date"] = pd.to_datetime(
                financial_statement["date"]
            ).dt.to_period("Q")
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
                "issues when values are zero and is predominently relevant for "
                "ratio calculations."
            )

        financial_statement_statistics = financial_statement_statistics.sort_index(
            axis=1
        ).truncate(before=start_date, after=end_date, axis=1)

        financial_statement_total = financial_statement_total.sort_index(
            axis=1
        ).truncate(before=start_date, after=end_date, axis=1)

        financial_statement_total = financial_statement_total.round(rounding)

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

        return (
            financial_statement_total,
            financial_statement_statistics,
            invalid_tickers,
        )

    return pd.DataFrame(), pd.DataFrame(), invalid_tickers


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
            "For more information, look here: https://site.financialmodelingprep.com/developer/docs/pricing/jeroen/"
        )

    period = "quarter" if quarter else "annual"

    revenue_segmentation_dict: dict = {}
    invalid_tickers = []

    ticker_list_iterator = (
        tqdm(ticker_list, desc=f"Obtaining {method} segmentation data")
        if ENABLE_TQDM & progress_bar
        else ticker_list
    )

    for ticker in ticker_list_iterator:
        url = (
            f"https://financialmodelingprep.com/api/v4/{location}"
            f"?symbol={ticker}&period={period}&structure=flat&apikey={api_key}"
        )
        revenue_segmentation_json = get_financial_data(
            ticker=ticker,
            url=url,
            sleep_timer=sleep_timer,
            subscription_type="Professional or Enterprise",
            raw=True,
        )

        if "ERROR_MESSAGE" in revenue_segmentation_json:
            invalid_tickers.append(ticker)
            continue

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

    if revenue_segmentation_dict:
        revenue_segmentation_total = pd.concat(revenue_segmentation_dict, axis=0)

        try:
            revenue_segmentation_total = revenue_segmentation_total.astype(np.float64)
        except ValueError as error:
            print(
                f"Not able to convert DataFrame to float64 due to {error}. This could result in"
                "issues when values are zero and is predominently relevant for "
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
            invalid_tickers,
        )

    return pd.DataFrame(), invalid_tickers


def get_analyst_estimates(
    tickers: str | list[str],
    api_key: str = "",
    quarter: bool = False,
    start_date: str | None = None,
    end_date: str | None = None,
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
        end_date (str): The end date to filter data with.
        sleep_timer (bool): Whether to set a sleep timer when the rate limit is reached. Note that this only works
        if you have a Premium subscription (Starter or higher) from FinancialModelingPrep. Defaults to False.
        progress_bar (bool): Whether to show a progress bar when retrieving data over 10 tickers. Defaults to True.

    Returns:
        pd.DataFrame: A DataFrame containing the analyst estimates for all provided tickers.
    """
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
            "For more information, look here: https://site.financialmodelingprep.com/developer/docs/pricing/jeroen/"
        )

    period = "quarter" if quarter else "annual"

    analyst_estimates_dict: dict = {}
    invalid_tickers = []

    ticker_list_iterator = (
        tqdm(ticker_list, desc="Obtaining analyst estimates")
        if ENABLE_TQDM & progress_bar
        else ticker_list
    )

    for ticker in ticker_list_iterator:
        url = (
            "https://financialmodelingprep.com/api/v3/analyst-estimates/"
            f"{ticker}?period={period}&apikey={api_key}"
        )
        analyst_estimates = get_financial_data(
            ticker=ticker, url=url, sleep_timer=sleep_timer
        )

        if "ERROR_MESSAGE" in analyst_estimates:
            invalid_tickers.append(ticker)
            continue

        try:
            analyst_estimates = analyst_estimates.drop("symbol", axis=1)
        except KeyError:
            print(f"No analyst estimates data found for {ticker}")
            invalid_tickers.append(ticker)
            continue

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
                "issues when values are zero and is predominently relevant for "
                "ratio calculations."
            )

        analyst_estimates_total = analyst_estimates_total.sort_index(axis=1).truncate(
            before=start_date, after=end_date, axis=1
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
            invalid_tickers,
        )

    return pd.DataFrame(), invalid_tickers


def get_profile(tickers: list[str] | str, api_key: str) -> pd.DataFrame:
    """
    Gives information about the profile of a company which includes i.a. beta, company description, industry and sector.

    Args:
        ticker (list or string): the company ticker (for example: "AAPL")
        api_key (string): the API Key obtained from https://site.financialmodelingprep.com/developer/docs/pricing/jeroen/

    Returns:
        pd.DataFrame: the profile data.
    """

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

    profile_dataframe: pd.DataFrame = pd.DataFrame()

    invalid_tickers = []
    for ticker in ticker_list:
        try:
            url = f"https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey={api_key}"
            profile_dataframe[ticker] = get_financial_data(ticker=ticker, url=url).T
        except ValueError:
            print(f"No profile data found for {ticker}")
            invalid_tickers.append(ticker)

    if not profile_dataframe.empty:
        profile_dataframe = profile_dataframe.rename(index=naming)
        profile_dataframe = profile_dataframe.drop(
            ["image", "defaultImage", "isEtf", "isActivelyTrading", "isAdr", "isFund"],
            axis=0,
        )

    return profile_dataframe, invalid_tickers


def get_quote(tickers: list[str] | str, api_key: str) -> pd.DataFrame:
    """
    Gives information about the quote of a company which includes i.a. high/low close prices,
    price-to-earning ratio and shares outstanding.

    Args:
        ticker (list or string): the company ticker (for example: "AMD")
        api_key (string): the API Key obtained from https://site.financialmodelingprep.com/developer/docs/pricing/jeroen/

    Returns:
        pd.DataFrame: the quote data.
    """
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

    quote_dataframe: pd.DataFrame = pd.DataFrame()

    invalid_tickers = []
    for ticker in ticker_list:
        try:
            url = f"https://financialmodelingprep.com/api/v3/quote/{ticker}?apikey={api_key}"
            quote_dataframe[ticker] = get_financial_data(ticker=ticker, url=url).T
        except ValueError:
            print(f"No quote data found for {ticker}")
            invalid_tickers.append(ticker)
        except Exception as error:
            raise ValueError(error) from error

    quote_dataframe = quote_dataframe.rename(index=naming)

    return quote_dataframe, invalid_tickers


def get_rating(tickers: list[str] | str, api_key: str):
    """
    Gives information about the rating of a company which includes i.a. the company rating and
    recommendation as well as ratings based on a variety of ratios.

    Args:
        ticker (list or string): the company ticker (for example: "MSFT")
        api_key (string): the API Key obtained from https://site.financialmodelingprep.com/developer/docs/pricing/jeroen/

    Returns:
        pd.DataFrame: the rating data.
    """
    if isinstance(tickers, str):
        ticker_list = [tickers]
    elif isinstance(tickers, list):
        ticker_list = tickers
    else:
        raise ValueError(f"Type for the tickers ({type(tickers)}) variable is invalid.")

    ratings_dict = {}
    invalid_tickers = []
    for ticker in ticker_list:
        try:
            url = f"https://financialmodelingprep.com/api/v3/historical-rating/{ticker}?l&apikey={api_key}"
            ratings = get_financial_data(ticker=ticker, url=url)

        except ValueError:
            print(f"No rating data found for {ticker}")
            invalid_tickers.append(ticker)
            break
        except Exception as error:
            raise ValueError(error) from error

        try:
            ratings = ratings.drop("symbol", axis=1).sort_values(
                by="date", ascending=True
            )
        except KeyError:
            print(f"No rating data found for {ticker}")
            invalid_tickers.append(ticker)
            continue

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

    if not ratings.empty:
        ratings_dataframe = pd.concat(ratings_dict, axis=0).dropna()

        if len(ticker_list) == 1:
            ratings_dataframe = ratings_dataframe.loc[ticker_list[0]]

        return ratings_dataframe, invalid_tickers
    return pd.DataFrame(), invalid_tickers


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
        api_key (string): the API Key obtained from https://site.financialmodelingprep.com/developer/docs/pricing/jeroen/
        start_date (str): The start date to filter data with.
        end_date (str): The end date to filter data with.
        actual_dates (bool): Whether to retrieve actual dates. Defaults to False (converted to quarterly). This is the
        default because the actual date refers to the corresponding quarter.
        sleep_timer (bool): Whether to set a sleep timer when the rate limit is reached. Note that this only works
        if you have a Premium subscription (Starter or higher) from FinancialModelingPrep. Defaults to False.
        progress_bar (bool): Whether to show a progress bar when retrieving data over 10 tickers. Defaults to True.

    Returns:
        pd.DataFrame: the rating data.
    """
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
            "For more information, look here: https://site.financialmodelingprep.com/developer/docs/pricing/jeroen/"
        )

    earnings_calendar_dict: dict = {}
    invalid_tickers = []

    ticker_list_iterator = (
        tqdm(ticker_list, desc="Obtaining earnings calendars")
        if (ENABLE_TQDM & progress_bar)
        else ticker_list
    )

    for ticker in ticker_list_iterator:
        url = (
            "https://financialmodelingprep.com/api/v3/historical/earning_calendar/"
            f"{ticker}?apikey={api_key}"
        )
        earnings_calendar = get_financial_data(
            ticker=ticker, url=url, sleep_timer=sleep_timer
        )

        if "ERROR_MESSAGE" in earnings_calendar:
            invalid_tickers.append(ticker)
            continue

        try:
            earnings_calendar = earnings_calendar.drop("symbol", axis=1)
        except KeyError:
            print(f"No earnings calendar found for {ticker}")
            invalid_tickers.append(ticker)
            continue

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

    if earnings_calendar_dict:
        earnings_calendar_total = pd.concat(earnings_calendar_dict, axis=0)

        return (
            earnings_calendar_total,
            invalid_tickers,
        )

    return pd.DataFrame(), invalid_tickers

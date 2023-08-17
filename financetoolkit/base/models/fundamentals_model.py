"""Fundamentals Module"""
__docformat__ = "google"


import time

import numpy as np
import pandas as pd
import requests

# pylint: disable=no-member,too-many-locals

try:
    from tqdm import tqdm

    ENABLE_TQDM = True
except ImportError:
    ENABLE_TQDM = False

from financetoolkit.base.models.normalization_model import (
    convert_financial_statements,
)

PROGRESS_BAR_LIMIT = 10


def get_financial_statements(
    tickers: str | list[str],
    statement: str = "",
    api_key: str = "",
    quarter: bool = False,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 100,
    rounding: int = 4,
    statement_format: pd.DataFrame = pd.DataFrame(),
    statistics_format: pd.DataFrame = pd.DataFrame(),
    sleep_timer: bool = False,
    progress_bar: bool = True,
):
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
        financial_statement = get_financial_statement_data(
            ticker=ticker,
            location=location,
            period=period,
            api_key=api_key,
            limit=limit,
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


def get_financial_statement_data(
    ticker: str,
    location: str,
    period: str,
    api_key: str,
    limit: int,
    sleep_timer: bool = False,
) -> pd.DataFrame:
    """
    Collects the financial statement data from the FinancialModelingPrep API. This is a
    separate function to properly segregate the different types of errors that can occur.

    Args:
        ticker (str): The company ticker (for example: "AAPL")
        location (str): The location of the financial statement (for example: "balance-sheet-statement")
        period (str): The period of the financial statement (for example: "annual")
        api_key (str): The API Key obtained from FinancialModelingPrep
        limit (int): The limit for the years of data
        sleep_timer (bool): Whether to set a sleep timer when the rate limit is reached. Note that this only works
        if you have a Premium subscription (Starter or higher) from FinancialModelingPrep. Defaults to False.

    Returns:
        pd.DataFrame: A DataFrame containing the financial statement data.
    """
    while True:
        try:
            response = requests.get(
                f"https://financialmodelingprep.com/api/v3/{location}/"
                f"{ticker}?period={period}&apikey={api_key}&limit={limit}",
                timeout=60,
            )
            response.raise_for_status()
            financial_statement = pd.read_json(response.text)

            return financial_statement

        except requests.exceptions.HTTPError:
            if (
                "not available under your current subscription"
                in response.json()["Error Message"]
            ):
                print(
                    f"The requested data for {ticker} is part of the Premium Subscription from "
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

            print(
                "This is an undefined error, please report to the author at https://github.com/JerBouma/FinanceToolkit"
                f"\n{response.json()['Error Message']}"
            )

            return pd.DataFrame(columns=["ERROR_MESSAGE"])


def get_analyst_estimates(
    tickers: str | list[str],
    api_key: str = "",
    quarter: bool = False,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 100,
    rounding: int = 4,
    sleep_timer: bool = False,
    progress_bar: bool = True,
):
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
        analyst_estimates = get_analyst_estimates_data(
            ticker=ticker,
            period=period,
            api_key=api_key,
            limit=limit,
            sleep_timer=sleep_timer,
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


def get_analyst_estimates_data(
    ticker: str,
    period: str,
    api_key: str,
    limit: int,
    sleep_timer: bool = False,
) -> pd.DataFrame:
    """
    Collects the analyst estimates data from the FinancialModelingPrep API. This is a
    separate function to properly segregate the different types of errors that can occur.

    Args:
        ticker (str): The company ticker (for example: "AAPL")
        period (str): The period (for example: "annual")
        api_key (str): The API Key obtained from FinancialModelingPrep
        limit (int): The limit for the years of data
        sleep_timer (bool): Whether to set a sleep timer when the rate limit is reached. Note that this only works
        if you have a Premium subscription (Starter or higher) from FinancialModelingPrep. Defaults to False.

    Returns:
        pd.DataFrame: A DataFrame containing the analyst estimates data.
    """
    while True:
        try:
            response = requests.get(
                f"https://financialmodelingprep.com/api/v3/analyst-estimates/"
                f"{ticker}?period={period}&apikey={api_key}&limit={limit}",
                timeout=60,
            )
            response.raise_for_status()
            analyst_estimates = pd.read_json(response.text)

            return analyst_estimates

        except requests.exceptions.HTTPError:
            if (
                "not available under your current subscription"
                in response.json()["Error Message"]
            ):
                print(
                    f"The requested data for {ticker} is part of the Premium Subscription from "
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

            print(
                "This is an undefined error, please report to the author at https://github.com/JerBouma/FinanceToolkit"
                f"\n{response.json()['Error Message']}"
            )

            return pd.DataFrame(columns=["ERROR_MESSAGE"])


def get_profile(tickers: list[str] | str, api_key: str):
    """
    Description
    ----
    Gives information about the profile of a company which includes
    i.a. beta, company description, industry and sector.
    Input
    ----
    ticker (list or string)
        The company ticker (for example: "AAPL")
    api_key (string)
        The API Key obtained from https://site.financialmodelingprep.com/developer/docs/pricing/jeroen/
    Output
    ----
    data (dataframe)
        Data with variables in rows and the period in columns.
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
            profile_dataframe[ticker] = pd.read_json(
                f"https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey={api_key}"
            ).T
        except ValueError:
            print(f"No profile data found for {ticker}")
            invalid_tickers.append(ticker)
        except Exception as error:
            raise ValueError(error) from error

    profile_dataframe = profile_dataframe.rename(index=naming)
    profile_dataframe = profile_dataframe.drop(
        ["image", "defaultImage", "isEtf", "isActivelyTrading", "isAdr", "isFund"],
        axis=0,
    )

    return profile_dataframe, invalid_tickers


def get_quote(tickers: list[str] | str, api_key: str):
    """
    Description
    ----
    Gives information about the quote of a company which includes i.a.
    high/low close prices, price-to-earning ratio and shares outstanding.
    Input
    ----
    ticker (list or string)
        The company ticker (for example: "AMD")
    api_key (string)
        The API Key obtained from https://site.financialmodelingprep.com/developer/docs/pricing/jeroen/
    Output
    ----
    data (dataframe)
        Data with variables in rows and the period in columns.
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
            quote_dataframe[ticker] = pd.read_json(
                f"https://financialmodelingprep.com/api/v3/quote/{ticker}?apikey={api_key}"
            ).T
        except ValueError:
            print(f"No quote data found for {ticker}")
            invalid_tickers.append(ticker)
        except Exception as error:
            raise ValueError(error) from error

    quote_dataframe = quote_dataframe.rename(index=naming)

    return quote_dataframe, invalid_tickers


def get_rating(tickers: list[str] | str, api_key: str, limit: int = 100):
    """
    Description
    ----
    Gives information about the rating of a company which includes i.a. the company
    rating and recommendation as well as ratings based on a variety of ratios.
    Input
    ----
    ticker (list or string)
       The company ticker (for example: "MSFT")
    api_key (string)
       The API Key obtained from https://site.financialmodelingprep.com/developer/docs/pricing/jeroen/
    Output
    ----
    data (dataframe)
       Data with variables in rows and the period in columns..
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
            ratings = pd.read_json(
                f"https://financialmodelingprep.com/api/v3/historical-rating/{ticker}?limit={limit}&apikey={api_key}"
            )
        except ValueError:
            print(f"No rating data found for {ticker}")
            invalid_tickers.append(ticker)
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

    ratings_dataframe = pd.concat(ratings_dict, axis=0).dropna()

    if len(ticker_list) == 1:
        ratings_dataframe = ratings_dataframe.loc[ticker_list[0]]

    return ratings_dataframe, invalid_tickers

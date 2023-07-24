"""Fundamentals Module"""
__docformat__ = "numpy"


import numpy as np
import pandas as pd
import requests

from financetoolkit.base.models.normalization_model import (
    convert_financial_statements,
)

# pylint: disable=no-member,too-many-locals


def get_financial_statements(
    tickers: str | list[str],
    statement: str = "",
    api_key: str = "",
    quarter: bool = False,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 100,
    statement_format: pd.DataFrame = pd.DataFrame(),
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
            "For more information, look here: https://site.financialmodelingprep.com/developer/docs/"
        )

    period = "quarter" if quarter else "annual"

    financial_statement_dict: dict = {}
    invalid_tickers = []
    for ticker in ticker_list:
        try:
            response = requests.get(
                f"https://financialmodelingprep.com/api/v3/{location}/"
                f"{ticker}?period={period}&apikey={api_key}&limit={limit}",
                timeout=60,
            )
            response.raise_for_status()
            financial_statement = pd.read_json(response.text)
        except requests.exceptions.HTTPError:
            print(f"{response.json()['Error Message']} (ticker: {ticker})")
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
            ).dt.to_period("M")
        else:
            financial_statement["date"] = pd.to_datetime(
                financial_statement["date"]
            ).dt.year

        financial_statement = financial_statement.set_index("date").T

        financial_statement_dict[ticker] = financial_statement

    if financial_statement_dict:
        financial_statement_total = pd.concat(financial_statement_dict, axis=0)

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

        financial_statement_total = financial_statement_total.sort_index(
            axis=1
        ).truncate(before=start_date, after=end_date, axis=1)

        if quarter:
            financial_statement_total.columns = pd.PeriodIndex(
                financial_statement_total.columns, freq="M"
            )
        else:
            financial_statement_total.columns = pd.PeriodIndex(
                financial_statement_total.columns, freq="Y"
            )

        return financial_statement_total, invalid_tickers

    return pd.DataFrame(), invalid_tickers


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
        The API Key obtained from https://financialmodelingprep.com/developer/docs/
    Output
    ----
    data (dataframe)
        Data with variables in rows and the period in columns.
    """
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
            print(f"No historical data found for {ticker}")
            invalid_tickers.append(ticker)
        except Exception as error:
            raise ValueError(error) from error

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
        The API Key obtained from https://financialmodelingprep.com/developer/docs/
    Output
    ----
    data (dataframe)
        Data with variables in rows and the period in columns.
    """
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
            print(f"No historical data found for {ticker}")
            invalid_tickers.append(ticker)
        except Exception as error:
            raise ValueError(error) from error

    return quote_dataframe, invalid_tickers


def get_enterprise(
    tickers: list[str] | str,
    api_key: str,
    quarter: bool = False,
    limit: int = 100,
):
    """
    Description
    ----
    Gives information about the enterprise value of a company which includes
    i.a. market capitalisation, Cash & Cash Equivalents, total debt and enterprise value.
    Input
    ----
    ticker (list or string)
        The company ticker (for example: "TSLA")
    api_key (string)
        The API Key obtained from https://financialmodelingprep.com/developer/docs/
    period (string)
        Data period, this can be "annual" or "quarter".
    limit (integer)
        The limit for the years of data
    Output
    ----
    data (dataframe)
        Data with variables in rows and the period in columns.
    """
    if isinstance(tickers, str):
        ticker_list = [tickers]
    elif isinstance(tickers, list):
        ticker_list = tickers
    else:
        raise ValueError(f"Type for the tickers ({type(tickers)}) variable is invalid.")

    enterprise_value_dict = {}

    period = "quarter" if quarter else "annual"

    invalid_tickers = []
    for ticker in ticker_list:
        try:
            response = requests.get(
                f"https://financialmodelingprep.com/api/v3/enterprise-values/{ticker}"
                f"?period={period}&limit={limit}&apikey={api_key}",
                timeout=60,
            )
            response.raise_for_status()
            enterprise_values = pd.read_json(response.text)
        except requests.exceptions.HTTPError:
            print(f"{response.json()['Error Message']} (ticker: {ticker})")
            invalid_tickers.append(ticker)
            continue

        try:
            enterprise_values = enterprise_values.drop("symbol", axis=1).sort_values(
                by="date", ascending=True
            )
        except KeyError:
            print(f"No historical data found for {ticker}.")
            invalid_tickers.append(ticker)
            continue

        if quarter:
            enterprise_values["date"] = pd.PeriodIndex(
                enterprise_values["date"], freq="M"
            )
        else:
            enterprise_values["date"] = pd.PeriodIndex(
                enterprise_values["date"], freq="Y"
            )

        enterprise_values = enterprise_values.set_index("date")

        enterprise_values = enterprise_values.rename(
            columns={
                "stockPrice": "Stock Price",
                "numberOfShares": "Number of Shares",
                "marketCapitalization": "Market Capitalization",
                "minusCashAndCashEquivalents": "Cash and Cash Equivalents",
                "addTotalDebt": "Total Debt",
                "enterpriseValue": "Enterprise Value",
            }
        )

        enterprise_value_dict[ticker] = enterprise_values

    if enterprise_value_dict:
        enterprise_dataframe = pd.concat(enterprise_value_dict, axis=0).dropna()

        if len(ticker_list) == 1:
            enterprise_dataframe = enterprise_dataframe.loc[ticker_list[0]]

        return enterprise_dataframe, invalid_tickers

    return pd.DataFrame(), invalid_tickers


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
       The API Key obtained from https://financialmodelingprep.com/developer/docs/
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
            print(f"No historical data found for {ticker}")
            invalid_tickers.append(ticker)
        except Exception as error:
            raise ValueError(error) from error

        ratings = ratings.drop("symbol", axis=1).sort_values(by="date", ascending=True)
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

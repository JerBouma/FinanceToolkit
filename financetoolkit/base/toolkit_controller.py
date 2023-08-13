"""Toolkit Module"""
__docformat__ = "google"


import re

import pandas as pd

from financetoolkit.base.helpers import calculate_growth as _calculate_growth
from financetoolkit.base.models.fundamentals_model import (
    get_enterprise as _get_enterprise,
    get_financial_statements as _get_financial_statements,
    get_profile as _get_profile,
    get_quote as _get_quote,
    get_rating as _get_rating,
)
from financetoolkit.base.models.historical_model import (
    convert_daily_to_quarterly as _convert_daily_to_quarterly,
    convert_daily_to_yearly as _convert_daily_to_yearly,
    get_historical_data as _get_historical_data,
)
from financetoolkit.base.models.normalization_model import (
    convert_financial_statements as _convert_financial_statements,
    copy_normalization_files as _copy_normalization_files,
    read_normalization_file as _read_normalization_file,
)
from financetoolkit.base.models_controller import Models
from financetoolkit.base.ratios_controller import Ratios

# pylint: disable=too-many-instance-attributes,too-many-lines,line-too-long
# ruff: noqa: E501


class Toolkit:
    """
    The Finance Toolkit is an open-source toolkit in which
    all relevant financial ratios (50+), indicators and performance measurements
    are written down in the most simplistic way allowing for complete transparency
    of the calculation method. This allows you to not have to rely on metrics
    from other providers and, given a financial statement, allow for efficient manual
    calculations. This leads to one uniform method of calculation being applied that
    is available and understood by everyone.
    """

    def __init__(
        self,
        tickers: list | str,
        api_key: str = "",
        historical: pd.DataFrame = pd.DataFrame(),
        balance: pd.DataFrame = pd.DataFrame(),
        income: pd.DataFrame = pd.DataFrame(),
        cash: pd.DataFrame = pd.DataFrame(),
        custom_ratios: dict | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        quarterly: bool = False,
        format_location: str = "",
        reverse_dates: bool = False,
        remove_invalid_tickers: bool = True,
    ):
        """
        Initializes an Toolkit object with a ticker or a list of tickers. The way the Toolkit is initialized will define how the data is collected. For example, if you enable the quarterly flag, you will be able to collect quarterly data. Next to that, you can define the start and end date to specify a specific range. Another options is to define the custom ratios you want to calculate. This can be done by passing a dictionary.

        See for more information on all of this, the following link: https://www.jeroenbouma.com/projects/financetoolkit#how-to-guides-for-the-financetoolkit

        Args:
        tickers (str or list): A string or a list of strings containing the company ticker(s).
        api_key (str): An API key from FinancialModelingPrep.
        historical (pd.DataFrame): A DataFrame containing historical data.
        balance (pd.DataFrame): A DataFrame containing balance sheet data.
        income (pd.DataFrame): A DataFrame containing income statement data.
        cash (pd.DataFrame): A DataFrame containing cash flow statement data.
        format_location (str): A string containing the location of the normalization files.
        reverse_dates (bool): A boolean indicating whether to reverse the dates in the financial statements.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["MSFT", "AAPL"], quarterly=True, start_date="2020-01-01", api_key=FMP_KEY)
        ```

        """
        if isinstance(tickers, str):
            self._tickers = [tickers.upper()]
        elif isinstance(tickers, list):
            self._tickers = [ticker.upper() for ticker in tickers]
        else:
            raise TypeError("Tickers must be a string or a list of strings.")

        if start_date and re.match(r"^\d{4}-\d{2}-\d{2}$", start_date) is None:
            raise ValueError(
                "Please input a valid start date (%Y-%m-%d) like '2010-01-01'"
            )
        if end_date and re.match(r"^\d{4}-\d{2}-\d{2}$", end_date) is None:
            raise ValueError(
                "Please input a valid end date (%Y-%m-%d) like '2020-01-01'"
            )
        if start_date and end_date and start_date > end_date:
            raise ValueError(
                f"Please ensure the start date {start_date} is before the end date {end_date}"
            )

        self._api_key = api_key
        self._start_date = start_date
        self._end_date = end_date
        self._quarterly = quarterly
        self._remove_invalid_tickers = remove_invalid_tickers
        self._invalid_tickers: list = []

        if self._api_key:
            # Initialization of FinancialModelingPrep Variables
            self._profile: pd.DataFrame = pd.DataFrame()
            self._quote: pd.DataFrame = pd.DataFrame()
            self._enterprise: pd.DataFrame = pd.DataFrame()
            self._rating: pd.DataFrame = pd.DataFrame()

        # Initialization of Historical Variables
        self._daily_historical_data: pd.DataFrame = (
            historical if not historical.empty else pd.DataFrame()
        )
        self._weekly_historical_data: pd.DataFrame = pd.DataFrame()
        self._monthly_historical_data: pd.DataFrame = pd.DataFrame()
        self._quarterly_historical_data: pd.DataFrame = (
            _convert_daily_to_quarterly(self._daily_historical_data)
            if not historical.empty
            else pd.DataFrame()
        )
        self._yearly_historical_data: pd.DataFrame = (
            _convert_daily_to_yearly(self._daily_historical_data)
            if not historical.empty
            else pd.DataFrame()
        )

        # Initialization of Normalization Variables
        self._balance_sheet_statement_generic: pd.DataFrame = _read_normalization_file(
            "balance", format_location
        )
        self._income_statement_generic: pd.DataFrame = _read_normalization_file(
            "income", format_location
        )
        self._cash_flow_statement_generic: pd.DataFrame = _read_normalization_file(
            "cash", format_location
        )

        self._statistics_statement_generic: pd.DataFrame = _read_normalization_file(
            "statistics", format_location
        )

        # Initialization of Financial Statements
        self._balance_sheet_statement: pd.DataFrame = (
            _convert_financial_statements(
                balance, self._balance_sheet_statement_generic, reverse_dates
            )
            if not balance.empty
            else pd.DataFrame()
        )
        self._balance_sheet_statement_growth: pd.DataFrame = pd.DataFrame()

        self._income_statement: pd.DataFrame = (
            _convert_financial_statements(
                income, self._income_statement_generic, reverse_dates
            )
            if not income.empty
            else pd.DataFrame()
        )
        self._income_statement_growth: pd.DataFrame = pd.DataFrame()

        self._cash_flow_statement: pd.DataFrame = (
            _convert_financial_statements(
                cash, self._cash_flow_statement_generic, reverse_dates
            )
            if not cash.empty
            else pd.DataFrame()
        )
        self._cash_flow_statement_growth: pd.DataFrame = pd.DataFrame()

        self._statistics_statement: pd.DataFrame = pd.DataFrame()
        self._custom_ratios: dict | None = custom_ratios

    @property
    def ratios(self) -> Ratios:
        """
        This gives access to the Ratios module. The Ratios Module contains over 50+ ratios that can be used to analyse companies. These ratios are divided into 5 categories: profitability, liquidity, solvency, efficiency and valuation. Each ratio is calculated using the data from the Toolkit module.

        Also see the following link for more information on all of this: https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        profitability_ratios = toolkit.ratios.collect_profitability_ratios()

        profitability_ratios.loc['AAPL']
        ```

        Which returns:

        |                                             |     2018 |     2019 |     2020 |     2021 |     2022 |
        |:--------------------------------------------|---------:|---------:|---------:|---------:|---------:|
        | Gross Margin                                | 0.383437 | 0.378178 | 0.382332 | 0.417794 | 0.433096 |
        | Operating Margin                            | 0.26694  | 0.24572  | 0.241473 | 0.297824 | 0.302887 |
        | Net Profit Margin                           | 0.224142 | 0.212381 | 0.209136 | 0.258818 | 0.253096 |
        | Interest Burden Ratio                       | 1.02828  | 1.02827  | 1.01211  | 1.00237  | 0.997204 |
        | Income Before Tax Profit Margin             | 0.274489 | 0.252666 | 0.244398 | 0.298529 | 0.30204  |
        | Effective Tax Rate                          | 0.183422 | 0.159438 | 0.144282 | 0.133023 | 0.162045 |
        | Return on Assets (ROA)                      | 0.162775 | 0.16323  | 0.177256 | 0.269742 | 0.282924 |
        | Return on Equity (ROE)                      | 0.555601 | 0.610645 | 0.878664 | 1.50071  | 1.96959  |
        | Return on Invested Capital (ROIC)           | 0.269858 | 0.293721 | 0.344126 | 0.503852 | 0.562645 |
        | Return on Capital Employed (ROCE)           | 0.305968 | 0.297739 | 0.320207 | 0.495972 | 0.613937 |
        | Return on Tangible Assets                   | 0.555601 | 0.610645 | 0.878664 | 1.50071  | 1.96959  |
        | Income Quality Ratio                        | 1.30073  | 1.25581  | 1.4052   | 1.09884  | 1.22392  |
        | Net Income per EBT                          | 0.816578 | 0.840562 | 0.855718 | 0.866977 | 0.837955 |
        | Free Cash Flow to Operating Cash Flow Ratio | 0.828073 | 0.848756 | 0.909401 | 0.893452 | 0.912338 |
        | EBT to EBIT Ratio                           | 0.957448 | 0.948408 | 0.958936 | 0.976353 | 0.975982 |
        | EBIT to Revenue                             | 0.286688 | 0.26641  | 0.254864 | 0.305759 | 0.309473 |

        """
        empty_data: list = []

        if not self._api_key and (
            self._balance_sheet_statement.empty
            or self._income_statement.empty
            or self._cash_flow_statement.empty
        ):
            raise ValueError(
                "The ratios property requires manual addition of balance, income "
                "and cash flow statements or an API key from FinancialModelPrep "
                "within the Toolkit class."
            )
        if self._balance_sheet_statement.empty:
            empty_data.append("Balance Sheet Statement")
            self.get_balance_sheet_statement()
        if self._income_statement.empty:
            empty_data.append("Income Statement")
            self.get_income_statement()
        if self._cash_flow_statement.empty:
            empty_data.append("Cash Flow Statement")
            self.get_cash_flow_statement()

        if (
            self._balance_sheet_statement.empty
            and self._income_statement.empty
            and self._cash_flow_statement.empty
        ):
            raise ValueError(
                "The datasets could not be populated and therefore the Ratios class cannot be initialized."
            )

        if not self._start_date:
            self._start_date = (
                f"{self._balance_sheet_statement.columns[0].year - 5}-01-01"
            )
        if not self._end_date:
            self._end_date = (
                f"{self._balance_sheet_statement.columns[-1].year + 5}-01-01"
            )

        if self._quarterly:
            if (
                self._quarterly_historical_data.empty
                and not self._balance_sheet_statement.empty
            ):
                self.get_historical_data(period="quarterly")
        elif not self._quarterly:
            if (
                self._yearly_historical_data.empty
                and not self._balance_sheet_statement.empty
            ):
                self.get_historical_data(period="yearly")
        else:
            raise ValueError("Invalid value for the quarterly parameter.")

        if empty_data:
            print(
                "The following data was not provided within the Toolkit class and "
                f"thus was retrieved from FinancialModelingPrep: {', '.join(empty_data)}."
            )

        return Ratios(
            self._tickers,
            self._quarterly_historical_data
            if self._quarterly
            else self._yearly_historical_data,
            self._balance_sheet_statement,
            self._income_statement,
            self._cash_flow_statement,
            self._custom_ratios,
        )

    @property
    def models(self) -> Models:
        """
        Gives access to the Models module. The Models module is meant to execute well-known models such as DUPONT and the Discounted Cash Flow (DCF) model. These models are also directly related to the data retrieved from the Toolkit module.

        Also see the following link for more information on all of this: https://www.jeroenbouma.com/projects/financetoolkit/docs/models

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["TSLA", "AMZN"], api_key=FMP_KEY, quarterly=True, start_date='2022-12-31')

        dupont_analysis = toolkit.models.get_extended_dupont_analysis()

        dupont_analysis.loc['AMZN']
        ```

        Which returns:

        |                         |      2022Q2 |    2022Q3 |      2022Q4 |    2023Q1 |    2023Q2 |
        |:------------------------|------------:|----------:|------------:|----------:|----------:|
        | Interest Burden Ratio   |  -1.24465   | 0.858552  | -2.88409    | 1.20243   | 1.01681   |
        | Tax Burden Ratio        |  -0.611396  | 1.13743   |  0.101571   | 0.640291  | 0.878792  |
        | Operating Profit Margin |  -0.0219823 | 0.0231391 | -0.00636042 | 0.0323498 | 0.0562125 |
        | Asset Turnover          | nan         | 0.299735  |  0.3349     | 0.274759  | 0.285319  |
        | Equity Multiplier       | nan         | 3.15403   |  3.14263    | 3.08433   | 2.91521   |
        | Return on Equity        | nan         | 0.0213618 |  0.00196098 | 0.0211066 | 0.0417791 |
        """
        empty_data: list = []

        if not self._api_key and (
            self._balance_sheet_statement.empty
            or self._income_statement.empty
            or self._cash_flow_statement.empty
        ):
            raise ValueError(
                "The model property requires manual addition of balance, income "
                "and cash flow statements or an API key from FinancialModelPrep "
                "within the Toolkit class."
            )

        if self._balance_sheet_statement.empty:
            empty_data.append("Balance Sheet Statement")
            self.get_balance_sheet_statement()
        if self._income_statement.empty:
            empty_data.append("Income Statement")
            self.get_income_statement()
        if self._cash_flow_statement.empty:
            empty_data.append("Cash Flow Statement")
            self.get_cash_flow_statement()

        if (
            self._balance_sheet_statement.empty
            and self._income_statement.empty
            and self._cash_flow_statement.empty
        ):
            raise ValueError(
                "The datasets could not be populated and therefore the Models class cannot be initialized."
            )

        if empty_data:
            print(
                "The following data was not provided within the Toolkit class and "
                f"thus was retrieved from FinancialModelingPrep: {', '.join(empty_data)}."
            )

        return Models(
            self._tickers,
            self._balance_sheet_statement,
            self._income_statement,
            self._cash_flow_statement,
        )

    def get_profile(self):
        """
        Returns a pandas dataframe containing the company profile information for the specified tickers.

        Raises:
            ValueError: If an API key is not defined for FinancialModelingPrep.

        Returns:
            pandas.DataFrame: The company profile information for the specified tickers.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["MSFT", "AAPL"], api_key=FMP_KEY)

        toolkit.get_profile()
        ```

        Which returns:

        |                   | MSFT                                                   | AAPL                                                   |
        |:------------------|:-------------------------------------------------------|:-------------------------------------------------------|
        | symbol            | MSFT                                                   | AAPL                                                   |
        | price             | 323.538                                                | 178.185                                                |
        | beta              | 0.903706                                               | 1.286802                                               |
        | volAvg            | 28691370                                               | 56642391                                               |
        | mktCap            | 2405663496324                                          | 2802618483625                                          |
        | lastDiv           | 2.7199999999999998                                     | 0.96                                                   |
        | range             | 213.43-366.78                                          | 124.17-198.23                                          |
        | changes           | -6.572                                                 | -0.665                                                 |
        | companyName       | Microsoft Corporation                                  | Apple Inc.                                             |
        | currency          | USD                                                    | USD                                                    |
        | cik               | 789019                                                 | 320193                                                 |
        | isin              | US5949181045                                           | US0378331005                                           |
        | cusip             | 594918104                                              | 37833100                                               |
        | exchange          | NASDAQ Global Select                                   | NASDAQ Global Select                                   |
        | exchangeShortName | NASDAQ                                                 | NASDAQ                                                 |
        | industry          | Software—Infrastructure                                | Consumer Electronics                                   |
        | website           | https://www.microsoft.com                              | https://www.apple.com                                  |
        | ceo               | Mr. Satya  Nadella                                     | Mr. Timothy D. Cook                                    |
        | sector            | Technology                                             | Technology                                             |
        | country           | US                                                     | US                                                     |
        | fullTimeEmployees | 221000                                                 | 164000                                                 |
        | phone             | 425 882 8080                                           | 408 996 1010                                           |
        | address           | One Microsoft Way                                      | One Apple Park Way                                     |
        | city              | Redmond                                                | Cupertino                                              |
        | state             | WA                                                     | CA                                                     |
        | zip               | 98052-6399                                             | 95014                                                  |
        | dcfDiff           | 4.56584                                                | 4.15176                                                |
        | dcf               | 243.594                                                | 150.082                                                |
        | image             | https://financialmodelingprep.com/image-stock/MSFT.png | https://financialmodelingprep.com/image-stock/AAPL.png |
        | ipoDate           | 1986-03-13                                             | 1980-12-12                                             |
        | defaultImage      | False                                                  | False                                                  |
        | isEtf             | False                                                  | False                                                  |
        | isActivelyTrading | True                                                   | True                                                   |
        | isAdr             | False                                                  | False                                                  |
        | isFund            | False                                                  | False                                                  |
        """
        if not self._api_key:
            return print(
                "Please define an API key from FinancialModelingPrep to use this functionality."
            )

        if self._profile.empty:
            self._profile, self._invalid_tickers = _get_profile(
                self._tickers, self._api_key
            )

        if self._remove_invalid_tickers:
            self._tickers = [
                ticker
                for ticker in self._tickers
                if ticker not in self._invalid_tickers
            ]

        return self._profile

    def get_quote(self):
        """
        Returns a pandas dataframe containing the stock quote information for the specified tickers.

        Raises:
            ValueError: If an API key is not defined for FinancialModelingPrep.

        Returns:
            pandas.DataFrame: The stock quote information for the specified tickers.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["TSLA", "AAPL"], api_key=FMP_KEY)

        toolkit.get_quote()
        ```

        Which returns:

        |                      | TSLA                         | AAPL                         |
        |:---------------------|:-----------------------------|:-----------------------------|
        | symbol               | TSLA                         | AAPL                         |
        | name                 | Tesla, Inc.                  | Apple Inc.                   |
        | price                | 249.35                       | 178.23                       |
        | changesPercentage    | -0.8352                      | -0.3467                      |
        | change               | -2.1                         | -0.62                        |
        | dayLow               | 245.02                       | 177.58                       |
        | dayHigh              | 250.91                       | 179.74                       |
        | yearHigh             | 314.66666                    | 198.23                       |
        | yearLow              | 101.81                       | 124.17                       |
        | marketCap            | 790314801062                 | 2803326275144                |
        | priceAvg50           | 254.011                      | 187.227                      |
        | priceAvg200          | 195.875                      | 160.1889                     |
        | exchange             | NASDAQ                       | NASDAQ                       |
        | volume               | 44241191                     | 29828689                     |
        | avgVolume            | 134442424                    | 56642391                     |
        | open                 | 247.45                       | 179.69                       |
        | previousClose        | 251.45                       | 178.85                       |
        | eps                  | 3.08                         | 5.89                         |
        | pe                   | 80.96                        | 30.26                        |
        | earningsAnnouncement | 2023-10-17T00:00:00.000+0000 | 2023-10-25T00:00:00.000+0000 |
        | sharesOutstanding    | 3169499904                   | 15728700416                  |
        | timestamp            | 2023-08-08 15:19:10          | 2023-08-08 15:19:10          |
        """
        if not self._api_key:
            return print(
                "Please define an API key from FinancialModelingPrep to use this functionality."
            )

        if self._quote.empty:
            self._quote, self._invalid_tickers = _get_quote(
                self._tickers, self._api_key
            )

        if self._remove_invalid_tickers:
            self._tickers = [
                ticker
                for ticker in self._tickers
                if ticker not in self._invalid_tickers
            ]

        return self._quote

    def get_enterprise(self, limit: int = 100):
        """
        Returns a pandas dataframe containing the enterprise value information for the specified tickers.

        Args:
            quarter (str): The quarter for which the enterprise value is required. Defaults to False.
            limit (str): The number of results to return. Defaults to 100.

        Raises:
            ValueError: If an API key is not defined for FinancialModelingPrep.

        Returns:
            pandas.DataFrame: The enterprise value information for the specified tickers.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["MU", "MSFT"], api_key=FMP_KEY)

        enterprise = toolkit.get_enterprise()

        enterprise.loc['MSFT'].tail()
        ```

        Which returns:

        | date   |   Stock Price |   Number of Shares |   Market Capitalization |   Cash and Cash Equivalents |   Total Debt |   Enterprise Value |
        |:-------|--------------:|-------------------:|------------------------:|----------------------------:|-------------:|-------------------:|
        | 2019   |        135.68 |          7.673e+09 |             1.04107e+12 |                  1.1356e+10 |   7.8366e+10 |        1.10808e+12 |
        | 2020   |        203.51 |          7.61e+09  |             1.54871e+12 |                  1.3576e+10 |   7.0998e+10 |        1.60613e+12 |
        | 2021   |        270.9  |          7.547e+09 |             2.04448e+12 |                  1.4224e+10 |   6.7775e+10 |        2.09803e+12 |
        | 2022   |        256.83 |          7.496e+09 |             1.9252e+12  |                  1.3931e+10 |   6.127e+10  |        1.97254e+12 |
        | 2023   |        340.54 |          7.446e+09 |             2.53566e+12 |                  3.4704e+10 |   5.9965e+10 |        2.56092e+12 |
        """
        if not self._api_key:
            return print(
                "Please define an API key from FinancialModelingPrep to use this functionality."
            )

        if self._enterprise.empty:
            self._enterprise, self._invalid_tickers = _get_enterprise(
                self._tickers,
                self._api_key,
                self._quarterly,
                limit,
            )

        if self._remove_invalid_tickers:
            self._tickers = [
                ticker
                for ticker in self._tickers
                if ticker not in self._invalid_tickers
            ]

        return self._enterprise

    def get_rating(self, limit: int = 100):
        """
        Returns a pandas dataframe containing the stock rating information for the specified tickers.

        Args:
            limit (int): The number of results to return. Defaults to 100.

        Raises:
            ValueError: If an API key is not defined for FinancialModelingPrep.

        Returns:
            pandas.DataFrame: The stock rating information for the specified tickers.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "TSLA"], api_key=FMP_KEY)

        rating = toolkit.get_rating()

        rating.loc['AMZN', 'Rating Recommendation'].tail()
        ```

        Which returns:

        | date                | Rating Recommendation   |
        |:--------------------|:------------------------|
        | 2023-08-01 00:00:00 | Strong Buy              |
        | 2023-08-02 00:00:00 | Strong Buy              |
        | 2023-08-03 00:00:00 | Strong Buy              |
        | 2023-08-04 00:00:00 | Strong Buy              |
        | 2023-08-07 00:00:00 | Strong Buy              |

        """
        if not self._api_key:
            return print(
                "Please define an API key from FinancialModelingPrep to use this functionality."
            )

        if self._rating.empty:
            self._rating, self._invalid_tickers = _get_rating(
                self._tickers, self._api_key, limit
            )

        if self._remove_invalid_tickers:
            self._tickers = [
                ticker
                for ticker in self._tickers
                if ticker not in self._invalid_tickers
            ]

        return self._rating

    def get_historical_data(
        self, period: str = "daily", return_column: str = "Adj Close"
    ):
        """
        Returns a pandas dataframe containing the historical data for the specified tickers.

        Args:
            start (str): The start date for the historical data. Defaults to None.
            end (str): The end date for the historical data. Defaults to None.
            period (str): The interval at which the historical data should be
            returned - daily, weekly, monthly, quarterly, or yearly.
            Defaults to "daily".

        Raises:
            ValueError: If an invalid value is specified for period.

        Returns:
            pandas.DataFrame: The historical data for the specified tickers.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit("AAPL", api_key=FMP_KEY)

        toolkit.get_historical_data(period="yearly")
        ```

        Which returns:

        | Date   |   Adj Close |    Close |     High |      Low |     Open |      Volume |
        |:-------|------------:|---------:|---------:|---------:|---------:|------------:|
        | 2013   |     17.6127 |  20.0364 |  20.0457 |  19.7857 |  19.7918 | 2.23084e+08 |
        | 2014   |     24.7674 |  27.595  |  28.2825 |  27.5525 |  28.205  | 1.65614e+08 |
        | 2015   |     24.0209 |  26.315  |  26.7575 |  26.205  |  26.7525 | 1.63649e+08 |
        | 2016   |     27.0189 |  28.955  |  29.3    |  28.8575 |  29.1625 | 1.22345e+08 |
        | 2017   |     40.1134 |  42.3075 |  42.6475 |  42.305  |  42.63   | 1.04e+08    |
        | 2018   |     37.9512 |  39.435  |  39.84   |  39.12   |  39.6325 | 1.40014e+08 |
        | 2019   |     71.7117 |  73.4125 |  73.42   |  72.38   |  72.4825 | 1.00806e+08 |
        | 2020   |    130.735  | 132.69   | 134.74   | 131.72   | 134.08   | 9.91166e+07 |
        | 2021   |    176.033  | 177.57   | 179.23   | 177.26   | 178.09   | 6.40623e+07 |
        | 2022   |    129.553  | 129.93   | 129.95   | 127.43   | 128.41   | 7.70342e+07 |
        | 2023   |    178.1    | 178.1    | 179.74   | 177.58   | 179.69   | 3.03427e+07 |

        """
        if period == "daily":
            self._daily_historical_data, self._invalid_tickers = _get_historical_data(
                self._tickers,
                self._start_date,
                self._end_date,
                interval="1d",
                return_column=return_column,
            )

            if self._remove_invalid_tickers:
                self._tickers = [
                    ticker
                    for ticker in self._tickers
                    if ticker not in self._invalid_tickers
                ]

            self._daily_historical_data = self._daily_historical_data.sort_index()

            if len(self._tickers) == 1:
                return self._daily_historical_data.xs(
                    self._tickers[0], level=1, axis="columns"
                )

            return self._daily_historical_data

        if period == "weekly":
            self._weekly_historical_data, self._invalid_tickers = _get_historical_data(
                self._tickers,
                self._start_date,
                self._end_date,
                interval="1wk",
                return_column=return_column,
            )

            if self._remove_invalid_tickers:
                self._tickers = [
                    ticker
                    for ticker in self._tickers
                    if ticker not in self._invalid_tickers
                ]

            self._weekly_historical_data = self._weekly_historical_data.sort_index()

            if len(self._tickers) == 1:
                return self._weekly_historical_data.xs(
                    self._tickers[0], level=1, axis="columns"
                )

            return self._weekly_historical_data

        if period == "monthly":
            self._monthly_historical_data, self._invalid_tickers = _get_historical_data(
                self._tickers,
                self._start_date,
                self._end_date,
                interval="1mo",
                return_column=return_column,
            )

            if self._remove_invalid_tickers:
                self._tickers = [
                    ticker
                    for ticker in self._tickers
                    if ticker not in self._invalid_tickers
                ]

            self._monthly_historical_data = self._monthly_historical_data.sort_index()

            if len(self._tickers) == 1:
                return self._monthly_historical_data.xs(
                    self._tickers[0], level=1, axis="columns"
                )

            return self._monthly_historical_data

        if period == "quarterly":
            if self._daily_historical_data.empty:
                (
                    self._daily_historical_data,
                    self._invalid_tickers,
                ) = _get_historical_data(
                    self._tickers,
                    self._start_date,
                    self._end_date,
                    interval="1d",
                    return_column=return_column,
                )

            self._quarterly_historical_data = _convert_daily_to_quarterly(
                self._daily_historical_data
            )

            if self._remove_invalid_tickers:
                self._tickers = [
                    ticker
                    for ticker in self._tickers
                    if ticker not in self._invalid_tickers
                ]

            if len(self._tickers) == 1:
                return self._quarterly_historical_data.xs(
                    self._tickers[0], level=1, axis="columns"
                )

            return self._quarterly_historical_data

        if period == "yearly":
            if self._daily_historical_data.empty:
                (
                    self._daily_historical_data,
                    self._invalid_tickers,
                ) = _get_historical_data(
                    self._tickers,
                    self._start_date,
                    self._end_date,
                    interval="1d",
                    return_column=return_column,
                )

            self._yearly_historical_data = _convert_daily_to_yearly(
                self._daily_historical_data
            )

            if self._remove_invalid_tickers:
                self._tickers = [
                    ticker
                    for ticker in self._tickers
                    if ticker not in self._invalid_tickers
                ]

            self._yearly_historical_data = self._yearly_historical_data.sort_index()

            if len(self._tickers) == 1:
                return self._yearly_historical_data.xs(
                    self._tickers[0], level=1, axis="columns"
                )

            return self._yearly_historical_data

        raise ValueError(
            "Please choose from daily, weekly, monthly, quarterly or yearly as period."
        )

    def get_balance_sheet_statement(
        self,
        limit: int = 100,
        overwrite: bool = False,
        rounding: int = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ):
        """
        Retrieves the balance sheet statement financial data for the company(s) from the specified source.

        Args:
            limit (int): Defines the maximum years or quarters to obtain.
            overwrite (bool): Defines whether to overwrite the existing data.
            rounding (int): Defines the number of decimal places to round the data to.
            growth (bool): Defines whether to return the growth of the data.
            lag (int | str): Defines the number of periods to lag the growth data by.

        Returns:
            pd.DataFrame: A pandas DataFrame with the retrieved balance sheet statement data.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["MSFT", "MU"], api_key=FMP_KEY, quarterly=True, start_date='2022-05-01')

        balance_sheet_statements = toolkit.get_balance_sheet_statement()

        balance_sheet_statements.loc['MU']
        ```

        Which returns:

        |                                          |      2022Q2 |      2022Q3 |      2022Q4 |      2023Q1 |      2023Q2 |
        |:-----------------------------------------|------------:|------------:|------------:|------------:|------------:|
        | Cash and Cash Equivalents                |  9.157e+09  |  8.262e+09  |  9.574e+09  |  9.798e+09  |  9.298e+09  |
        | Short Term Investments                   |  1.07e+09   |  1.069e+09  |  1.007e+09  |  1.02e+09   |  1.054e+09  |
        | Cash and Short Term Investments          |  1.0227e+10 |  9.331e+09  |  1.0581e+10 |  1.0818e+10 |  1.0352e+10 |
        | Accounts Receivable                      |  6.229e+09  |  5.13e+09   |  3.318e+09  |  2.278e+09  |  2.429e+09  |
        | Inventory                                |  5.629e+09  |  6.663e+09  |  8.359e+09  |  8.129e+09  |  8.238e+09  |
        | Other Current Assets                     |  6.08e+08   |  6.44e+08   |  6.63e+08   |  6.73e+08   |  7.15e+08   |
        | Total Current Assets                     |  2.2708e+10 |  2.1781e+10 |  2.2921e+10 |  2.1898e+10 |  2.1734e+10 |
        | Property, Plant and Equipment            |  3.7355e+10 |  3.9227e+10 |  4.0028e+10 |  3.9758e+10 |  3.9382e+10 |
        | Goodwill                                 |  1.228e+09  |  1.228e+09  |  1.228e+09  |  1.228e+09  |  1.252e+09  |
        | Intangible Assets                        |  4.15e+08   |  4.21e+08   |  4.28e+08   |  4.1e+08    |  4.1e+08    |
        | Long Term Investments                    |  1.646e+09  |  1.647e+09  |  1.426e+09  |  1.212e+09  |  9.73e+08   |
        | Tax Assets                               |  6.82e+08   |  7.02e+08   |  6.72e+08   |  6.97e+08   |  7.08e+08   |
        | Other Fixed Assets                       |  1.262e+09  |  1.277e+09  |  1.171e+09  |  1.317e+09  |  1.221e+09  |
        | Fixed Assets                             |  4.2588e+10 |  4.4502e+10 |  4.4953e+10 |  4.4622e+10 |  4.3946e+10 |
        | Other Assets                             |  0          |  0          |  0          |  0          |  0          |
        | Total Assets                             |  6.5296e+10 |  6.6283e+10 |  6.7874e+10 |  6.652e+10  |  6.568e+10  |
        | Accounts Payable                         |  2.019e+09  |  2.142e+09  |  1.789e+09  |  1.689e+09  |  1.64e+09   |
        | Short Term Debt                          |  1.07e+08   |  1.03e+08   |  1.71e+08   |  2.37e+08   |  2.59e+08   |
        | Tax Payables                             |  3.82e+08   |  4.2e+08    |  4.19e+08   |  2.41e+08   |  1.48e+08   |
        | Deferred Revenue                         |  0          |  0          |  0          |  0          | -1.64e+09   |
        | Other Current Liabilities                |  4.883e+09  |  5.294e+09  |  4.565e+09  |  3.329e+09  |  4.845e+09  |
        | Total Current Liabilities                |  7.009e+09  |  7.539e+09  |  6.525e+09  |  5.255e+09  |  5.104e+09  |
        | Long Term Debt                           |  7.485e+09  |  7.413e+09  |  1.0719e+10 |  1.2647e+10 |  1.3589e+10 |
        | Deferred Revenue Non Current             |  6.63e+08   |  5.89e+08   |  5.16e+08   |  5.29e+08   |  6.32e+08   |
        | Deferred Tax Liabilities                 |  0          |  0          |  0          |  0          |  0          |
        | Other Non Current Liabilities            |  8.58e+08   |  8.35e+08   |  8.08e+08   |  8.32e+08   |  9.5e+08    |
        | Total Non Current Liabilities            |  9.006e+09  |  8.837e+09  |  1.2043e+10 |  1.4008e+10 |  1.5171e+10 |
        | Other Liabilities                        |  0          |  0          |  0          |  0          |  0          |
        | Capital Lease Obligations                |  6.29e+08   |  6.1e+08    |  6.25e+08   |  6.1e+08    |  6.03e+08   |
        | Total Liabilities                        |  1.6015e+10 |  1.6376e+10 |  1.8568e+10 |  1.9263e+10 |  2.0275e+10 |
        | Preferred Stock                          |  0          |  0          |  0          |  0          |  0          |
        | Common Stock                             |  1.22e+08   |  1.23e+08   |  1.23e+08   |  1.23e+08   |  1.24e+08   |
        | Retained Earnings                        |  4.5916e+10 |  4.7274e+10 |  4.6873e+10 |  4.4426e+10 |  4.2391e+10 |
        | Accumulated Other Comprehensive Income   | -3.64e+08   | -5.6e+08    | -4.73e+08   | -3.73e+08   | -3.4e+08    |
        | Other Total Shareholder Equity           |  3.607e+09  |  3.07e+09   |  2.783e+09  |  3.081e+09  |  3.23e+09   |
        | Total Shareholder Equity                 |  4.9281e+10 |  4.9907e+10 |  4.9306e+10 |  4.7257e+10 |  4.5405e+10 |
        | Total Equity                             |  4.9281e+10 |  4.9907e+10 |  4.9306e+10 |  4.7257e+10 |  4.5405e+10 |
        | Total Liabilities and Shareholder Equity |  6.5296e+10 |  6.6283e+10 |  6.7874e+10 |  6.652e+10  |  6.568e+10  |
        | Minority Interest                        |  0          |  0          |  0          |  0          |  0          |
        | Total Liabilities and Equity             |  6.5296e+10 |  6.6283e+10 |  6.7874e+10 |  6.652e+10  |  6.568e+10  |
        | Total Investments                        |  2.716e+09  |  2.716e+09  |  2.433e+09  |  2.232e+09  |  2.027e+09  |
        | Total Debt                               |  7.592e+09  |  7.516e+09  |  1.089e+10  |  1.2884e+10 |  1.3848e+10 |
        | Net Debt                                 | -1.565e+09  | -7.46e+08   |  1.316e+09  |  3.086e+09  |  4.55e+09   |
        """
        if not self._api_key and self._balance_sheet_statement.empty:
            raise ValueError(
                "Please define an API key from FinancialModelingPrep to use this functionality."
            )

        if self._balance_sheet_statement.empty or overwrite:
            (
                self._balance_sheet_statement,
                self._statistics_statement,
                self._invalid_tickers,
            ) = _get_financial_statements(
                self._tickers,
                "balance",
                self._api_key,
                self._quarterly,
                self._start_date,
                self._end_date,
                limit,
                rounding,
                self._balance_sheet_statement_generic,
                self._statistics_statement_generic,
            )

        if self._remove_invalid_tickers:
            self._tickers = [
                ticker
                for ticker in self._tickers
                if ticker not in self._invalid_tickers
            ]

        if growth:
            self._balance_sheet_statement_growth = _calculate_growth(
                self._balance_sheet_statement, lag=lag, rounding=rounding
            )

        if len(self._tickers) == 1:
            return (
                self._balance_sheet_statement_growth.loc[self._tickers[0]]
                if growth
                else self._balance_sheet_statement.loc[self._tickers[0]]
            )

        return (
            self._balance_sheet_statement_growth
            if growth
            else self._balance_sheet_statement
        )

    def get_income_statement(
        self,
        limit: int = 100,
        overwrite: bool = False,
        rounding: int = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ):
        """
        Retrieves the income statement financial data for the company(s) from the specified source.

        Args:
            limit (int): Defines the maximum years or quarters to obtain.
            overwrite (bool): Defines whether to overwrite the existing data.
            rounding (int): Defines the number of decimal places to round the data to.
            growth (bool): Defines whether to return the growth of the data.
            lag (int | str): Defines the number of periods to lag the growth data by.

        Returns:
            pd.DataFrame: A pandas DataFrame with the retrieved income statement data.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["TSLA", "MU"], api_key=FMP_KEY, quarterly=True, start_date='2022-05-01')

        income_sheet_statements = toolkit.get_income_statement()

        income_sheet_statements.loc['TSLA']
        ```

        Which returns:

        |                                              |     2022Q2 |      2022Q3 |      2022Q4 |      2023Q1 |     2023Q2 |
        |:---------------------------------------------|-----------:|------------:|------------:|------------:|-----------:|
        | Revenue                                      | 1.6934e+10 |  2.1454e+10 |  2.4318e+10 |  2.3329e+10 | 2.4927e+10 |
        | Cost of Goods Sold                           | 1.27e+10   |  1.6072e+10 |  1.8541e+10 |  1.8818e+10 | 2.0394e+10 |
        | Gross Profit                                 | 4.234e+09  |  5.382e+09  |  5.777e+09  |  4.511e+09  | 4.533e+09  |
        | Gross Profit Ratio                           | 0.25003    |  0.250862   |  0.237561   |  0.193364   | 0.181851   |
        | Research and Development Expenses            | 6.67e+08   |  7.33e+08   |  8.1e+08    |  7.71e+08   | 9.43e+08   |
        | General and Administrative Expenses          | 0          |  0          |  0          |  0          | 0          |
        | Selling and Marketing Expenses               | 0          |  0          |  0          |  0          | 0          |
        | Selling, General and Administrative Expenses | 9.61e+08   |  9.61e+08   |  1.032e+09  |  1.076e+09  | 1.191e+09  |
        | Other Expenses                               | 2.8e+07    | -8.5e+07    | -4.2e+07    | -4.8e+07    | 3.28e+08   |
        | Operating Expenses                           | 1.628e+09  |  1.694e+09  |  1.842e+09  |  1.847e+09  | 2.134e+09  |
        | Cost and Expenses                            | 1.4328e+10 |  1.7766e+10 |  2.0383e+10 |  2.0665e+10 | 2.2528e+10 |
        | Interest Income                              | 2.6e+07    |  8.6e+07    |  1.57e+08   |  2.13e+08   | 2.38e+08   |
        | Interest Expense                             | 4.4e+07    |  5.3e+07    |  3.3e+07    |  2.9e+07    | 2.8e+07    |
        | Depreciation and Amortization                | 1.118e+09  |  9.57e+08   |  1.138e+09  |  1.211e+09  | 1.72e+09   |
        | EBITDA                                       | 3.582e+09  |  4.645e+09  |  5.039e+09  |  3.875e+09  | 4.119e+09  |
        | EBITDA Ratio                                 | 0.211527   |  0.21651    |  0.207213   |  0.166102   | 0.165243   |
        | Operating Income                             | 2.464e+09  |  3.688e+09  |  3.901e+09  |  2.664e+09  | 2.399e+09  |
        | Operating Income Ratio                       | 0.145506   |  0.171903   |  0.160416   |  0.114193   | 0.096241   |
        | Total Other Income                           | 1e+07      | -5.2e+07    |  8.2e+07    |  1.36e+08   | 5.38e+08   |
        | Income Before Tax                            | 2.474e+09  |  3.636e+09  |  3.983e+09  |  2.8e+09    | 2.937e+09  |
        | Income Before Tax Ratio                      | 0.146097   |  0.169479   |  0.163788   |  0.120022   | 0.117824   |
        | Income Tax Expense                           | 2.05e+08   |  3.05e+08   |  2.76e+08   |  2.61e+08   | 3.23e+08   |
        | Net Income                                   | 2.259e+09  |  3.292e+09  |  3.687e+09  |  2.513e+09  | 2.703e+09  |
        | Net Income Ratio                             | 0.1334     |  0.153445   |  0.151616   |  0.10772    | 0.108437   |
        | EPS                                          | 0.73       |  1.05       |  1.18       |  0.8        | 0.85       |
        | EPS Diluted                                  | 0.65       |  0.95       |  1.07       |  0.73       | 0.78       |
        | Weighted Average Shares                      | 3.111e+09  |  3.146e+09  |  3.16e+09   |  3.166e+09  | 3.171e+09  |
        | Weighted Average Shares Diluted              | 3.465e+09  |  3.468e+09  |  3.471e+09  |  3.468e+09  | 3.478e+09  |
        """
        if not self._api_key and self._income_statement.empty:
            raise ValueError(
                "Please define an API key from FinancialModelingPrep to use this functionality."
            )

        if self._income_statement.empty or overwrite:
            (
                self._income_statement,
                self._statistics_statement,
                self._invalid_tickers,
            ) = _get_financial_statements(
                self._tickers,
                "income",
                self._api_key,
                self._quarterly,
                self._start_date,
                self._end_date,
                limit,
                rounding,
                self._income_statement_generic,
                self._statistics_statement_generic,
            )

        if self._remove_invalid_tickers:
            self._tickers = [
                ticker
                for ticker in self._tickers
                if ticker not in self._invalid_tickers
            ]

        if growth:
            self._income_statement_growth = _calculate_growth(
                self._income_statement, lag=lag, rounding=rounding
            )

        if len(self._tickers) == 1:
            return (
                self._income_statement_growth.loc[self._tickers[0]]
                if growth
                else self._income_statement.loc[self._tickers[0]]
            )

        return self._income_statement_growth if growth else self._income_statement

    def get_cash_flow_statement(
        self,
        limit: int = 100,
        overwrite: bool = False,
        rounding: int = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ):
        """
        Retrieves the cash flow statement financial data for the company(s) from the specified source.

        Args:
            limit (int): Defines the maximum years or quarters to obtain.
            overwrite (bool): Defines whether to overwrite the existing data.
            rounding (int): Defines the number of decimal places to round the data to.
            growth (bool): Defines whether to return the growth of the data.
            lag (int | str): Defines the number of periods to lag the growth data by.

        Returns:
            pd.DataFrame: A pandas DataFrame with the retrieved cash flow statement data.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["MU", "AMZN"], api_key=FMP_KEY, quarterly=True, start_date='2022-09-01')

        cash_flow_statements = toolkit.get_cash_flow_statement()

        cash_flow_statements.loc['AMZN']
        ```

        Which returns:

        |                               |      2022Q3 |      2022Q4 |      2023Q1 |      2023Q2 |
        |:------------------------------|------------:|------------:|------------:|------------:|
        | Net Income                    |  2.872e+09  |  2.78e+08   |  3.172e+09  |  6.75e+09   |
        | Depreciation and Amortization |  1.0204e+10 |  1.2685e+10 |  1.1123e+10 |  1.1589e+10 |
        | Deferred Income Tax           | -8.25e+08   | -3.367e+09  | -4.72e+08   | -2.744e+09  |
        | Stock Based Compensation      |  5.556e+09  |  5.606e+09  |  4.748e+09  |  7.127e+09  |
        | Change in Working Capital     | -5.254e+09  |  1.0526e+10 | -1.4317e+10 | -6.293e+09  |
        | Accounts Receivables          | -4.794e+09  | -8.788e+09  |  1.521e+09  | -5.167e+09  |
        | Inventory                     |  7.32e+08   |  3.18e+09   |  3.71e+08   | -2.373e+09  |
        | Accounts Payables             | -1.226e+09  |  9.852e+09  | -1.1264e+10 |  3.029e+09  |
        | Other Working Capital         |  3.4e+07    |  6.282e+09  | -4.945e+09  | -1.782e+09  |
        | Other Non Cash Items          | -1.149e+09  |  3.445e+09  |  5.34e+08   |  4.7e+07    |
        | Cash Flow from Operations     |  1.1404e+10 |  2.9173e+10 |  4.788e+09  |  1.6476e+10 |
        | Property, Plant and Equipment | -1.6378e+10 | -1.6592e+10 | -1.4207e+10 | -1.1455e+10 |
        | Acquisitions                  | -8.85e+08   | -8.31e+08   | -3.513e+09  | -3.16e+08   |
        | Purchases of Investments      | -2.39e+08   | -2.33e+08   | -3.38e+08   | -4.96e+08   |
        | Sales of Investments          |  5.57e+08   |  5.683e+09  |  1.115e+09  |  1.551e+09  |
        | Other Investing Activities    |  1.337e+09  |  1.152e+09  |  1.137e+09  |  1.043e+09  |
        | Cash Flow from Investing      | -1.5608e+10 | -1.0821e+10 | -1.5806e+10 | -9.673e+09  |
        | Debt Repayment                | -9.429e+09  | -1.8756e+10 | -6.369e+09  | -1.0861e+10 |
        | Common Stock Issued           |  0          |  0          |  0          |  0          |
        | Common Stock Purchased        |  0          |  6e+09      |  0          |  0          |
        | Dividends Paid                |  0          |  0          |  0          |  0          |
        | Other Financing Activities    |  1.2445e+10 |  1.2842e+10 |  1.2723e+10 |  4.322e+09  |
        | Cash Flow from Financing      |  3.016e+09  |  8.6e+07    |  6.354e+09  | -6.539e+09  |
        | Forex Changes on Cash         | -1.334e+09  |  6.37e+08   |  1.45e+08   |  6.9e+07    |
        | Net Change in Cash            | -2.522e+09  |  1.9075e+10 | -4.519e+09  |  3.33e+08   |
        | Cash End of Period            |  3.5178e+10 |  5.4253e+10 |  4.9734e+10 |  5.0067e+10 |
        | Cash Beginning of Period      |  3.77e+10   |  3.5178e+10 |  5.4253e+10 |  4.9734e+10 |
        | Operating Cash Flow           |  1.1404e+10 |  2.9173e+10 |  4.788e+09  |  1.6476e+10 |
        | Capital Expenditure           | -1.6378e+10 | -1.6592e+10 | -1.4207e+10 | -1.1455e+10 |
        | Free Cash Flow                | -4.974e+09  |  1.2581e+10 | -9.419e+09  |  5.021e+09  |
        """
        if not self._api_key and self._cash_flow_statement.empty:
            raise ValueError(
                "Please define an API key from FinancialModelingPrep to use this functionality."
            )

        if self._cash_flow_statement.empty or overwrite:
            (
                self._cash_flow_statement,
                self._statistics_statement,
                self._invalid_tickers,
            ) = _get_financial_statements(
                self._tickers,
                "cashflow",
                self._api_key,
                self._quarterly,
                self._start_date,
                self._end_date,
                limit,
                rounding,
                self._cash_flow_statement_generic,
                self._statistics_statement_generic,
            )

        if self._remove_invalid_tickers:
            self._tickers = [
                ticker
                for ticker in self._tickers
                if ticker not in self._invalid_tickers
            ]

        if growth:
            self._cash_flow_statement_growth = _calculate_growth(
                self._cash_flow_statement, lag=lag, rounding=rounding
            )

        if len(self._tickers) == 1:
            return (
                self._cash_flow_statement_growth.loc[self._tickers[0]]
                if growth
                else self._cash_flow_statement.loc[self._tickers[0]]
            )

        return self._cash_flow_statement_growth if growth else self._cash_flow_statement

    def get_statistics_statement(
        self,
        limit: int = 100,
        overwrite: bool = False,
    ):
        """
        Retrieves the balance, cash and income statistics for the company(s) from the specified source.

        Note that this also obtains the balance sheet statement at the same time given that it's the same
        API call. This is done to reduce the number of API calls to FinancialModelingPrep.

        Args:
            limit (int): Defines the maximum years or quarters to obtain.
            overwrite (bool): Defines whether to overwrite the existing data.

        Returns:
            pd.DataFrame: A pandas DataFrame with the retrieved statistics statement data.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit("TSLA", api_key=FMP_KEY, quarterly=True, start_date='2023-05-01')

        toolkit.get_statistics_statement()
        ```

        Which returns:

        |                   | 2023Q2                                                                                            |
        |:------------------|:--------------------------------------------------------------------------------------------------|
        | Reported Currency | USD                                                                                               |
        | CIK ID            | 1318605                                                                                           |
        | Filling Date      | 2023-07-24                                                                                        |
        | Accepted Date     | 2023-07-21 18:08:29                                                                               |
        | Calendar Year     | 2023                                                                                              |
        | Period            | Q2                                                                                                |
        | SEC Link          | https://www.sec.gov/Archives/edgar/data/1318605/000095017023033872/0000950170-23-033872-index.htm |
        | Document Link     | https://www.sec.gov/Archives/edgar/data/1318605/000095017023033872/tsla-20230630.htm              |

        """
        if not self._api_key and self._statistics_statement.empty:
            raise ValueError(
                "Please define an API key from FinancialModelingPrep to use this functionality."
            )

        if self._statistics_statement.empty or overwrite:
            (
                self._balance_sheet_statement,
                self._statistics_statement,
                self._invalid_tickers,
            ) = _get_financial_statements(
                self._tickers,
                "balance",
                self._api_key,
                self._quarterly,
                self._start_date,
                self._end_date,
                limit,
                self._balance_sheet_statement_generic,
                self._statistics_statement_generic,
            )

        if self._remove_invalid_tickers:
            self._tickers = [
                ticker
                for ticker in self._tickers
                if ticker not in self._invalid_tickers
            ]

        if len(self._tickers) == 1:
            return self._statistics_statement.loc[self._tickers[0]]

        return self._statistics_statement

    def get_normalization_files(self, path: str = ""):
        """
        Copies the normalization files to a folder based on path. By default, this is the path
        of the 'Downloads' folder.

        Args:
            path (str, optional): The path where to save the files to.

        Returns:
            Three csv files saved to the desired location.
        """
        if path:
            _copy_normalization_files(path)
        else:
            _copy_normalization_files()

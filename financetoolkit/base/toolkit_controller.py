"""Toolkit Module"""
__docformat__ = "numpy"


import re

import pandas as pd

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

# pylint: disable=too-many-instance-attributes


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
        start_date: str | None = None,
        end_date: str | None = None,
        quarterly: bool = False,
        format_location: str = "",
        reverse_dates: bool = False,
        remove_invalid_tickers: bool = True,
    ):
        """
        Initializes an Toolkit object with a ticker or a list of tickers.

        Args:
        tickers (str or list): A string or a list of strings containing the company ticker(s).
        api_key (str): An API key from FinancialModelingPrep.
        historical (pd.DataFrame): A DataFrame containing historical data.
        balance (pd.DataFrame): A DataFrame containing balance sheet data.
        income (pd.DataFrame): A DataFrame containing income statement data.
        cash (pd.DataFrame): A DataFrame containing cash flow statement data.
        format_location (str): A string containing the location of the normalization files.
        reverse_dates (bool): A boolean indicating whether to reverse the dates in the financial statements.
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

        # Initialization of Financial Statements
        self._balance_sheet_statement: pd.DataFrame = (
            _convert_financial_statements(
                balance, self._balance_sheet_statement_generic, reverse_dates
            )
            if not balance.empty
            else pd.DataFrame()
        )
        self._income_statement: pd.DataFrame = (
            _convert_financial_statements(
                income, self._income_statement_generic, reverse_dates
            )
            if not income.empty
            else pd.DataFrame()
        )
        self._cash_flow_statement: pd.DataFrame = (
            _convert_financial_statements(
                cash, self._cash_flow_statement_generic, reverse_dates
            )
            if not cash.empty
            else pd.DataFrame()
        )

    @property
    def ratios(self) -> Ratios:
        """
        Gives access to financial ratios.
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
        )

    @property
    def models(self) -> Models:
        """
        Gives access to financial models.
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

    def get_historical_data(self, period: str = "daily"):
        """
        Returns a pandas dataframe containing the historical data for the specified tickers.

        Args:
            start (str): The start date for the historical data. Defaults to None.
            end (str): The end date for the historical data. Defaults to None.
            period (str): The interval at which the historical data should be
            returned - daily, weekly, monthly, or yearly. Defaults to "daily".

        Raises:
            ValueError: If an invalid value is specified for period.

        Returns:
            pandas.DataFrame: The historical data for the specified tickers.
        """
        if period == "daily":
            self._daily_historical_data, self._invalid_tickers = _get_historical_data(
                self._tickers, self._start_date, self._end_date, interval="1d"
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
                self._tickers, self._start_date, self._end_date, interval="1wk"
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
                self._tickers, self._start_date, self._end_date, interval="1mo"
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
                    self._tickers, self._start_date, self._end_date, interval="1d"
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
                    self._tickers, self._start_date, self._end_date, interval="1d"
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
    ):
        """
        Retrieves the balance sheet statement financial data for the company(s) from the specified source.

        Args:
            limit (int): Defines the maximum years or quarters to obtain.

        Returns:
            pd.DataFrame: A pandas DataFrame with the retrieved balance sheet statement data.
        """
        if not self._api_key and self._balance_sheet_statement.empty:
            raise ValueError(
                "Please define an API key from FinancialModelingPrep to use this functionality."
            )

        if self._balance_sheet_statement.empty or overwrite:
            (
                self._balance_sheet_statement,
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
            )

        if self._remove_invalid_tickers:
            self._tickers = [
                ticker
                for ticker in self._tickers
                if ticker not in self._invalid_tickers
            ]

        if len(self._tickers) == 1:
            return self._balance_sheet_statement.loc[self._tickers[0]]

        return self._balance_sheet_statement

    def get_income_statement(
        self,
        limit: int = 100,
        overwrite: bool = False,
    ):
        """
        Retrieves the income statement financial data for the company(s) from the specified source.

        Args:
            limit (int): Defines the maximum years or quarters to obtain.

        Returns:
            pd.DataFrame: A pandas DataFrame with the retrieved income statement data.
        """
        if not self._api_key and self._income_statement.empty:
            raise ValueError(
                "Please define an API key from FinancialModelingPrep to use this functionality."
            )

        if self._income_statement.empty or overwrite:
            self._income_statement, self._invalid_tickers = _get_financial_statements(
                self._tickers,
                "income",
                self._api_key,
                self._quarterly,
                self._start_date,
                self._end_date,
                limit,
                self._income_statement_generic,
            )

        if self._remove_invalid_tickers:
            self._tickers = [
                ticker
                for ticker in self._tickers
                if ticker not in self._invalid_tickers
            ]

        if len(self._tickers) == 1:
            return self._income_statement.loc[self._tickers[0]]

        return self._income_statement

    def get_cash_flow_statement(
        self,
        limit: int = 100,
        overwrite: bool = False,
    ):
        """
        Retrieves the cash flow statement financial data for the company(s) from the specified source.

        Args:
            limit (int): Defines the maximum years or quarters to obtain.

        Returns:
            pd.DataFrame: A pandas DataFrame with the retrieved cash flow statement data.
        """
        if not self._api_key and self._cash_flow_statement.empty:
            raise ValueError(
                "Please define an API key from FinancialModelingPrep to use this functionality."
            )

        if self._cash_flow_statement.empty or overwrite:
            (
                self._cash_flow_statement,
                self._invalid_tickers,
            ) = _get_financial_statements(
                self._tickers,
                "cashflow",
                self._api_key,
                self._quarterly,
                self._start_date,
                self._end_date,
                limit,
                self._cash_flow_statement_generic,
            )

        if self._remove_invalid_tickers:
            self._tickers = [
                ticker
                for ticker in self._tickers
                if ticker not in self._invalid_tickers
            ]

        if len(self._tickers) == 1:
            return self._cash_flow_statement.loc[self._tickers[0]]

        return self._cash_flow_statement

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

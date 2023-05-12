"""Analyzer Module"""
__docformat__ = "numpy"

import pandas as pd

from financialtoolkit.fundamentals import (
    get_enterprise as _get_enterprise,
    get_financial_statements as _get_financial_statements,
    get_profile as _get_profile,
    get_quote as _get_quote,
    get_rating as _get_rating,
)
from financialtoolkit.historical import (
    convert_daily_to_yearly as _convert_daily_to_yearly,
    get_historical_data as _get_historical_data,
)
from financialtoolkit.normalization import (
    convert_financial_statement as _convert_financial_statement,
    copy_normalization_files as _copy_normalization_files,
)
from financialtoolkit.ratios.overview import (
    get_efficiency_ratios as _get_efficiency_ratios,
    get_liquidity_ratios as _get_liquidity_ratios,
    get_profitability_ratios as _get_profitability_ratios,
    get_solvency_ratios as _get_solvency_ratios,
    get_valuation_ratios as _get_valuation_ratios,
)
from financialtoolkit.settings import (
    _BALANCE_SHEET_FORMAT_LOCATION,
    _CASH_FLOW_FORMAT_LOCATION,
    _INCOME_FORMAT_LOCATION,
)

# pylint: disable=too-many-instance-attributes

# TODO: change financial statement location to one variable
# TODO: add financial statement (3 variables) to initalization
# TODO: create ratios class that takes financial statements as input
# TODO: create model class that takes financial statements as input
# TODO: add historical data parameter (?)
# TODO: make the analyzer (Toolkit) the only option for users
# TODO: explain the other method is still valid but not recommended
# TODO: exclude source variable if financial statement variables are provided
# TODO: create uniform dataframes with company, item x year (see balance sheet and ratios difference)
# TODO: make it easy to convert a DataFrame to the prefered format

class Analyzer:
    """
    Analyzer Class
    """

    def __init__(
        self,
        tickers,
        api_key=None,
        source="FinancialModelingPrep",
        balance_sheet_format_location: str = "",
        income_statement_format_location: str = "",
        cash_flow_statement_format_location: str = "",
    ):
        """
        Initializes an Analyzer object with a ticker or a list of tickers.

        Args:
        tickers (str or list): A string or a list of strings containing the company ticker(s).
        api_key (str): An API key from FinancialModelingPrep.
        source (str): The source of the data. Either FinancialModelingPrep or Custom.
        balance_sheet_format_location (str): The location of the balance sheet format file.
        income_statement_format_location (str): The location of the income statement format file.
        cash_flow_statement_format_location (str): The location of the cash flow statement format file.
        """
        self._source = source
        self._api_key = api_key

        self._balance_sheet_format_location = (
            balance_sheet_format_location
            if balance_sheet_format_location
            else _BALANCE_SHEET_FORMAT_LOCATION
        )

        self._income_statement_format_location = (
            income_statement_format_location
            if income_statement_format_location
            else _INCOME_FORMAT_LOCATION
        )

        self._cash_flow_statement_format_location = (
            cash_flow_statement_format_location
            if cash_flow_statement_format_location
            else _CASH_FLOW_FORMAT_LOCATION
        )

        if isinstance(tickers, str):
            self.tickers = [tickers]
        elif isinstance(tickers, list):
            self.tickers = tickers
        else:
            raise TypeError("Tickers must be a string or a list of strings.")

        if self._source not in ["FinancialModelingPrep", "Custom"]:
            raise ValueError(
                "Please select either FinancialModelingPrep or Custom as the source."
            )
        if self._source == "Custom" and len(tickers) > 1:
            raise ValueError("The source Custom currently only supports one ticker.")

        # Initialization of Basic Variables
        self._profile: pd.DataFrame = pd.DataFrame()
        self._quote: pd.DataFrame = pd.DataFrame()
        self._enterprise: pd.DataFrame = pd.DataFrame()
        self._rating: pd.DataFrame = pd.DataFrame()

        # Initialization of Historical Variables
        self._daily_historical_data: pd.DataFrame = pd.DataFrame()
        self._weekly_historical_data: pd.DataFrame = pd.DataFrame()
        self._monthly_historical_data: pd.DataFrame = pd.DataFrame()
        self._yearly_historical_data: pd.DataFrame = pd.DataFrame()

        # Initialization of Fundamentals Variables
        self._balance_sheet_statement: pd.DataFrame = pd.DataFrame()
        self._income_statement: pd.DataFrame = pd.DataFrame()
        self._cash_flow_statement: pd.DataFrame = pd.DataFrame()
        self._efficiency_ratios: pd.DataFrame = pd.DataFrame()
        self._liquidity_ratios: pd.DataFrame = pd.DataFrame()
        self._profitability_ratios: pd.DataFrame = pd.DataFrame()
        self._solvency_ratios: pd.DataFrame = pd.DataFrame()
        self._valuation_ratios: pd.DataFrame = pd.DataFrame()

        try:
            # Initialization of Normalization Variables
            self._balance_sheet_statement_generic: pd.DataFrame = pd.read_csv(
                self._balance_sheet_format_location, index_col=self._source
            )["Generic"]
            self._income_statement_generic: pd.DataFrame = pd.read_csv(
                self._income_statement_format_location, index_col=self._source
            )["Generic"]
            self._cash_flow_statement_generic: pd.DataFrame = pd.read_csv(
                self._cash_flow_statement_format_location, index_col=self._source
            )["Generic"]
        except KeyError as exc:
            raise ValueError(
                f"Source {self._source} is not an option. Please ensure "
                "this source and 'Generic' is included as columns "
                "in the normalization files."
            ) from exc

    def get_profile(self):
        """
        Returns a pandas dataframe containing the company profile information for the specified tickers.

        Raises:
            ValueError: If an API key is not defined for FinancialModelingPrep.

        Returns:
            pandas.DataFrame: The company profile information for the specified tickers.
        """
        if not self._api_key and self._source == "FinancialModelingPrep":
            return print(
                "Please define an API key from FinancialModelingPrep to use this functionality."
            )

        if self._profile.empty:
            self._profile = _get_profile(self.tickers, self._api_key)

        return self._profile

    def get_quote(self):
        """
        Returns a pandas dataframe containing the stock quote information for the specified tickers.

        Raises:
            ValueError: If an API key is not defined for FinancialModelingPrep.

        Returns:
            pandas.DataFrame: The stock quote information for the specified tickers.
        """
        if not self._api_key and self._source == "FinancialModelingPrep":
            return print(
                "Please define an API key from FinancialModelingPrep to use this functionality."
            )

        if self._quote.empty:
            self._quote = _get_quote(self.tickers, self._api_key)

        return self._quote

    def get_enterprise(self, quarter: str = False, limit: str = 100):
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
        if not self._api_key and self._source == "FinancialModelingPrep":
            return print(
                "Please define an API key from FinancialModelingPrep to use this functionality."
            )

        if self._enterprise.empty:
            self._enterprise = _get_enterprise(
                self.tickers, self._api_key, quarter, limit
            )

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
        if not self._api_key and self._source == "FinancialModelingPrep":
            return print(
                "Please define an API key from FinancialModelingPrep to use this functionality."
            )

        if self._rating.empty:
            self._rating = _get_rating(self.tickers, self._api_key, limit)

        return self._rating

    def get_historical_data(self, start=None, end=None, period: str = "daily"):
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
            self._daily_historical_data = _get_historical_data(
                self.tickers, start, end, interval="1d"
            )
            return self._daily_historical_data

        if period == "weekly":
            self._weekly_historical_data = _get_historical_data(
                self.tickers, start, end, interval="1wk"
            )
            return self._weekly_historical_data

        if period == "monthly":
            self._monthly_historical_data = _get_historical_data(
                self.tickers, start, end, interval="1mo"
            )
            return self._monthly_historical_data

        if period == "yearly":
            if self._daily_historical_data.empty:
                self._daily_historical_data = _get_historical_data(
                    self.tickers, start, end, interval="1d"
                )

            self._yearly_historical_data = _convert_daily_to_yearly(
                self._daily_historical_data
            )
            return self._yearly_historical_data

        raise ValueError(
            "Please choose from daily, weekly, monthly or yearly as period."
        )

    def get_balance_sheet_statement(
        self,
        quarter=False,
        limit: int = 100,
        custom_balance_sheet: pd.DataFrame = pd.DataFrame(),
    ):
        """
        Retrieves the balance sheet statement financial data for the company(s) from the specified source.

        Args:
            quarter (bool, optional): Flag to retrieve quarterly or annual data. Defaults to False.

        Returns:
            pd.DataFrame: A pandas DataFrame with the retrieved balance sheet statement data.
        """
        if not self._api_key and self._source == "FinancialModelingPrep":
            raise ValueError(
                "Please define an API key from FinancialModelingPrep to use this functionality "
                "or change the source to custom accompanied with a custom balance sheet."
            )

        if self._source == "Custom" and not custom_balance_sheet.empty:
            self._balance_sheet_statement = _convert_financial_statement(
                custom_balance_sheet, self._balance_sheet_statement_generic
            )
        elif self._balance_sheet_statement.empty:
            self._balance_sheet_statement = _get_financial_statements(
                self.tickers,
                "balance",
                self._api_key,
                quarter,
                limit,
                self._balance_sheet_statement_generic,
            )

        return self._balance_sheet_statement

    def get_income_statement(
        self,
        quarter=False,
        limit: int = 100,
        custom_income_statement: pd.DataFrame = pd.DataFrame(),
    ):
        """
        Retrieves the income statement financial data for the company(s) from the specified source.

        Args:
            quarter (bool, optional): Flag to retrieve quarterly or annual data. Defaults to False.

        Returns:
            pd.DataFrame: A pandas DataFrame with the retrieved income statement data.
        """
        if not self._api_key and self._source == "FinancialModelingPrep":
            raise ValueError(
                "Please define an API key from FinancialModelingPrep to use this functionality "
                "or change the source to custom accompanied with a custom income statement."
            )

        if self._source == "Custom" and not custom_income_statement.empty:
            self._income_statement = _convert_financial_statement(
                custom_income_statement, self._income_statement_generic
            )
        elif self._income_statement.empty:
            self._income_statement = _get_financial_statements(
                self.tickers,
                "income",
                self._api_key,
                quarter,
                limit,
                self._income_statement_generic,
            )

        return self._income_statement

    def get_cash_flow_statement(
        self,
        quarter=False,
        limit: int = 100,
        custom_cash_flow_statement: pd.DataFrame = pd.DataFrame(),
    ):
        """
        Retrieves the cash flow statement financial data for the company(s) from the specified source.

        Args:
            quarter (bool, optional): Flag to retrieve quarterly or annual data. Defaults to False.

        Returns:
            pd.DataFrame: A pandas DataFrame with the retrieved cash flow statement data.
        """
        if not self._api_key and self._source == "FinancialModelingPrep":
            raise ValueError(
                "Please define an API key from FinancialModelingPrep to use this functionality "
                "or change the source to custom accompanied with a custom cash flow statement."
            )

        if self._source == "Custom" and not custom_cash_flow_statement.empty:
            self._cash_flow_statement = _convert_financial_statement(
                custom_cash_flow_statement, self._cash_flow_statement_generic
            )
        elif self._cash_flow_statement.empty:
            self._cash_flow_statement = _get_financial_statements(
                self.tickers,
                "cashflow",
                self._api_key,
                quarter,
                limit,
                self._cash_flow_statement_generic,
            )

        return self._cash_flow_statement

    def get_efficiency_ratios(
        self,
        custom_balance_sheet: pd.DataFrame = pd.DataFrame(),
        custom_income_statement: pd.DataFrame = pd.DataFrame(),
    ):
        """
        Retrieves the cash flow statement financial data for the company(s) from the specified source.

        Args:
            quarter (bool, optional): Flag to retrieve quarterly or annual data. Defaults to False.

        Returns:
            pd.DataFrame: A pandas DataFrame with the retrieved cash flow statement data.
        """
        if not self._api_key and self._source == "FinancialModelingPrep":
            raise ValueError(
                "Please define an API key from FinancialModelingPrep to use this functionality "
                "or change the source to custom accompanied with a custom balance and income statement."
            )

        if self._balance_sheet_statement.empty:
            self._balance_sheet_statement = self.get_balance_sheet_statement(
                custom_balance_sheet=custom_balance_sheet
            )
        if self._income_statement.empty:
            self._income_statement = self.get_income_statement(
                custom_income_statement=custom_income_statement
            )

        if self._efficiency_ratios.empty:
            self._efficiency_ratios = _get_efficiency_ratios(
                self.tickers, self._balance_sheet_statement, self._income_statement
            )

        return self._efficiency_ratios

    def get_liquidity_ratios(
        self,
        custom_balance_sheet: pd.DataFrame = pd.DataFrame(),
        custom_income_statement: pd.DataFrame = pd.DataFrame(),
        custom_cash_flow_statement: pd.DataFrame = pd.DataFrame(),
    ):
        """
        Retrieves the cash flow statement financial data for the company(s) from the specified source.

        Args:
            quarter (bool, optional): Flag to retrieve quarterly or annual data. Defaults to False.

        Returns:
            pd.DataFrame: A pandas DataFrame with the retrieved cash flow statement data.
        """
        if not self._api_key and self._source == "FinancialModelingPrep":
            raise ValueError(
                "Please define an API key from FinancialModelingPrep to use this functionality "
                "or change the source to custom accompanied with a custom balance, income "
                "and cash flow statement."
            )

        if self._balance_sheet_statement.empty:
            self._balance_sheet_statement = self.get_balance_sheet_statement(
                custom_balance_sheet=custom_balance_sheet
            )
        if self._income_statement.empty:
            self._income_statement = self.get_income_statement(
                custom_income_statement=custom_income_statement
            )
        if self._cash_flow_statement.empty:
            self._cash_flow_statement = self.get_cash_flow_statement(
                custom_cash_flow_statement=custom_cash_flow_statement
            )

        if self._liquidity_ratios.empty:
            self._liquidity_ratios = _get_liquidity_ratios(
                self.tickers,
                self._balance_sheet_statement,
                self._income_statement,
                self._cash_flow_statement,
            )

        return self._liquidity_ratios

    def get_profitability_ratios(
        self,
        custom_balance_sheet: pd.DataFrame = pd.DataFrame(),
        custom_income_statement: pd.DataFrame = pd.DataFrame(),
        custom_cash_flow_statement: pd.DataFrame = pd.DataFrame(),
    ):
        """
        Retrieves the cash flow statement financial data for the company(s) from the specified source.

        Args:
            quarter (bool, optional): Flag to retrieve quarterly or annual data. Defaults to False.

        Returns:
            pd.DataFrame: A pandas DataFrame with the retrieved cash flow statement data.
        """
        if not self._api_key and self._source == "FinancialModelingPrep":
            raise ValueError(
                "Please define an API key from FinancialModelingPrep to use this functionality "
                "or change the source to custom accompanied with a custom balance, income "
                "and cash flow statement."
            )

        if self._balance_sheet_statement.empty:
            self._balance_sheet_statement = self.get_balance_sheet_statement(
                custom_balance_sheet=custom_balance_sheet
            )
        if self._income_statement.empty:
            self._income_statement = self.get_income_statement(
                custom_income_statement=custom_income_statement
            )
        if self._cash_flow_statement.empty:
            self._cash_flow_statement = self.get_cash_flow_statement(
                custom_cash_flow_statement=custom_cash_flow_statement
            )

        if self._profitability_ratios.empty:
            self._profitability_ratios = _get_profitability_ratios(
                self.tickers,
                self._balance_sheet_statement,
                self._income_statement,
                self._cash_flow_statement,
            )

        return self._profitability_ratios

    def get_solvency_ratios(
        self,
        custom_balance_sheet: pd.DataFrame = pd.DataFrame(),
        custom_income_statement: pd.DataFrame = pd.DataFrame(),
    ):
        """
        Retrieves the cash flow statement financial data for the company(s) from the specified source.

        Args:
            quarter (bool, optional): Flag to retrieve quarterly or annual data. Defaults to False.

        Returns:
            pd.DataFrame: A pandas DataFrame with the retrieved cash flow statement data.
        """
        if not self._api_key and self._source == "FinancialModelingPrep":
            raise ValueError(
                "Please define an API key from FinancialModelingPrep to use this functionality "
                "or change the source to custom accompanied with a custom balance and income statement."
            )

        if self._balance_sheet_statement.empty:
            self._balance_sheet_statement = self.get_balance_sheet_statement(
                custom_balance_sheet=custom_balance_sheet
            )
        if self._income_statement.empty:
            self._income_statement = self.get_income_statement(
                custom_income_statement=custom_income_statement
            )

        if self._solvency_ratios.empty:
            self._solvency_ratios = _get_solvency_ratios(
                self.tickers,
                self._balance_sheet_statement,
                self._income_statement,
            )

        return self._solvency_ratios

    def get_valuation_ratios(
        self,
        custom_balance_sheet: pd.DataFrame = pd.DataFrame(),
        custom_income_statement: pd.DataFrame = pd.DataFrame(),
        custom_cash_flow_statement: pd.DataFrame = pd.DataFrame(),
    ):
        """
        Retrieves the cash flow statement financial data for the company(s) from the specified source.

        Args:
            quarter (bool, optional): Flag to retrieve quarterly or annual data. Defaults to False.

        Returns:
            pd.DataFrame: A pandas DataFrame with the retrieved cash flow statement data.
        """
        if not self._api_key and self._source == "FinancialModelingPrep":
            raise ValueError(
                "Please define an API key from FinancialModelingPrep to use this functionality "
                "or change the source to custom accompanied with a custom balance, income "
                "and cash flow statement."
            )

        if self._yearly_historical_data.empty:
            self._yearly_historical_data = self.get_historical_data(period="yearly")

        if self._balance_sheet_statement.empty:
            self._balance_sheet_statement = self.get_balance_sheet_statement(
                custom_balance_sheet=custom_balance_sheet
            )
        if self._income_statement.empty:
            self._income_statement = self.get_income_statement(
                custom_income_statement=custom_income_statement
            )
        if self._cash_flow_statement.empty:
            self._cash_flow_statement = self.get_cash_flow_statement(
                custom_cash_flow_statement=custom_cash_flow_statement
            )

        if self._valuation_ratios.empty:
            self._valuation_ratios = _get_valuation_ratios(
                self.tickers,
                self._yearly_historical_data,
                self._balance_sheet_statement,
                self._income_statement,
                self._cash_flow_statement,
            )

        return self._valuation_ratios

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

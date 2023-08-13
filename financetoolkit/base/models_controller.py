"""Models Module"""
__docformat__ = "google"

import pandas as pd

from financetoolkit.base.helpers import calculate_growth, handle_errors
from financetoolkit.models.dupont import (
    get_dupont_analysis,
    get_extended_dupont_analysis,
)

# pylint: disable=too-many-instance-attributes


class Models:
    """
    Models Controller Class
    """

    def __init__(
        self,
        tickers: str | list[str],
        balance: pd.DataFrame,
        income: pd.DataFrame,
        cash: pd.DataFrame,
    ):
        """
        Initializes the Models Controller Class.
        """
        self._tickers = tickers
        self._balance_sheet_statement = balance
        self._income_statement = income
        self._cash_flow_statement = cash

        # Initialization of Model Variables
        self._dupont_analysis: pd.DataFrame = pd.DataFrame()
        self._dupont_analysis_growth: pd.DataFrame = pd.DataFrame()
        self._extended_dupont_analysis: pd.DataFrame = pd.DataFrame()
        self._extended_dupont_analysis_growth: pd.DataFrame = pd.DataFrame()

    @handle_errors
    def get_dupont_analysis(
        self,
        rounding: int = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
        overwrite: bool = False,
    ) -> pd.DataFrame:
        """
        Perform a Dupont analysis to breakdown the return on equity (ROE) into its components.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.models.get_dupont_analysis()
        ```
        """
        if self._dupont_analysis.empty or overwrite:
            self._dupont_analysis = get_dupont_analysis(
                self._income_statement.loc[:, "Net Income", :],
                self._income_statement.loc[:, "Revenue", :],
                self._balance_sheet_statement.loc[:, "Total Assets", :].shift(axis=1),
                self._balance_sheet_statement.loc[:, "Total Assets", :],
                self._balance_sheet_statement.loc[:, "Total Equity", :].shift(axis=1),
                self._balance_sheet_statement.loc[:, "Total Equity", :],
            )

        if (self._dupont_analysis_growth.empty or overwrite) and growth:
            self._dupont_analysis_growth = calculate_growth(
                self._dupont_analysis, lag=lag, rounding=rounding
            )

        if len(self._tickers) == 1:
            return (
                self._dupont_analysis_growth.droplevel(level=0)
                if growth
                else self._dupont_analysis.droplevel(level=0)
            )

        return self._dupont_analysis_growth if growth else self._dupont_analysis

    @handle_errors
    def get_extended_dupont_analysis(
        self,
        rounding: int = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
        overwrite: bool = False,
    ) -> pd.DataFrame:
        """
        Perform am Extended Dupont analysis to breakdown the return on equity (ROE) into its components.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.models.get_extended_dupont_analysis()
        ```
        """
        if self._extended_dupont_analysis.empty or overwrite:
            self._extended_dupont_analysis = get_extended_dupont_analysis(
                self._income_statement.loc[:, "Income Before Tax", :],
                self._income_statement.loc[:, "Operating Income", :],
                self._income_statement.loc[:, "Net Income", :],
                self._income_statement.loc[:, "Revenue", :],
                self._balance_sheet_statement.loc[:, "Total Assets", :].shift(axis=1),
                self._balance_sheet_statement.loc[:, "Total Assets", :],
                self._balance_sheet_statement.loc[:, "Total Equity", :].shift(axis=1),
                self._balance_sheet_statement.loc[:, "Total Equity", :],
            )

        if (self._extended_dupont_analysis_growth.empty or overwrite) and growth:
            self._extended_dupont_analysis_growth = calculate_growth(
                self._extended_dupont_analysis, lag=lag, rounding=rounding
            )

        if len(self._tickers) == 1:
            return (
                self._extended_dupont_analysis_growth.droplevel(level=0)
                if growth
                else self._extended_dupont_analysis.droplevel(level=0)
            )

        return (
            self._extended_dupont_analysis_growth
            if growth
            else self._extended_dupont_analysis
        )

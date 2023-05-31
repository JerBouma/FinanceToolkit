"""Models Module"""
__docformat__ = "numpy"

import pandas as pd

from financetoolkit.base.helpers import handle_errors
from financetoolkit.models.dupont import (
    get_dupont_analysis,
    get_extended_dupont_analysis,
)


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
        self._tickers = tickers
        self._balance_sheet_statement = balance
        self._income_statement = income
        self._cash_flow_statement = cash

        # Initialization of Model Variables
        self._dupont_analysis: pd.DataFrame = pd.DataFrame()
        self._extended_dupont_analysis: pd.DataFrame = pd.DataFrame()

    @handle_errors
    def get_dupont_analysis(self) -> pd.DataFrame:
        """
        Perform a Dupont analysis to breakdown the return on equity (ROE) into its components.
        """
        if self._dupont_analysis.empty:
            self._dupont_analysis = get_dupont_analysis(
                self._income_statement.loc[:, "Net Income", :],
                self._income_statement.loc[:, "Revenue", :],
                self._balance_sheet_statement.loc[:, "Total Assets", :].shift(axis=1),
                self._balance_sheet_statement.loc[:, "Total Assets", :],
                self._balance_sheet_statement.loc[:, "Total Equity", :].shift(axis=1),
                self._balance_sheet_statement.loc[:, "Total Equity", :],
            )

        if len(self._tickers) == 1:
            return self._dupont_analysis.droplevel(level=0)

        return self._dupont_analysis

    @handle_errors
    def get_extended_dupont_analysis(self) -> pd.DataFrame:
        """
        Perform am Extended Dupont analysis to breakdown the return on equity (ROE) into its components.
        """
        if self._extended_dupont_analysis.empty:
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

        if len(self._tickers) == 1:
            return self._extended_dupont_analysis.droplevel(level=0)

        return self._extended_dupont_analysis

"""Models Module"""
__docformat__ = "numpy"

import pandas as pd

from financialtoolkit.models.dupont import (
    get_dupont_analysis,
    get_extended_dupont_analysis,
)


class Models:
    """
    Models Module
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

    def get_dupont_analysis(self) -> pd.DataFrame:
        """
        Perform a Dupont analysis to breakdown the return on equity (ROE) into its components.
        """
        dupont_analysis_dict: dict = {}

        if self._dupont_analysis.empty:
            for ticker in self._tickers:
                bst = self._balance_sheet_statement.loc[ticker]
                ist = self._income_statement.loc[ticker]

                dupont_analysis_dict[ticker] = get_dupont_analysis(
                    ist.loc["Net Income"],
                    ist.loc["Revenue"],
                    bst.loc["Total Assets"],
                    bst.loc["Total Equity"],
                )

            self._dupont_analysis = pd.concat(dupont_analysis_dict, axis=0)

        return self._dupont_analysis

    def get_extended_dupont_analyis(self) -> pd.DataFrame:
        """
        Perform am Extended Dupont analysis to breakdown the return on equity (ROE) into its components.
        """

        extended_dupont_analysis_dict: dict = {}

        if self._extended_dupont_analysis.empty:
            for ticker in self._tickers:
                bst = self._balance_sheet_statement.loc[ticker]
                ist = self._income_statement.loc[ticker]

                extended_dupont_analysis_dict[ticker] = get_extended_dupont_analysis(
                    ist.loc["Income Before Tax"],
                    ist.loc["Operating Income"],
                    ist.loc["Net Income"],
                    ist.loc["Revenue"],
                    bst.loc["Total Assets"],
                    bst.loc["Total Equity"],
                )

            self._extended_dupont_analysis = pd.concat(
                extended_dupont_analysis_dict, axis=0
            )

        return self._extended_dupont_analysis

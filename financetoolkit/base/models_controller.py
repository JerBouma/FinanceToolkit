"""Models Module"""
__docformat__ = "google"

import pandas as pd

from financetoolkit.base.helpers import calculate_growth, handle_errors
from financetoolkit.models.dupont import (
    get_dupont_analysis,
    get_extended_dupont_analysis,
)
from financetoolkit.models.enterprise import get_enterprise_value_breakdown

# pylint: disable=too-many-instance-attributes


class Models:
    """
    Models Controller Class
    """

    def __init__(
        self,
        tickers: str | list[str],
        historical: pd.DataFrame,
        balance: pd.DataFrame,
        income: pd.DataFrame,
        cash: pd.DataFrame,
        rounding: int | None = 4,
    ):
        """
        Initializes the Models Controller Class.
        """
        self._tickers = tickers
        self._historical_data: pd.DataFrame = historical
        self._balance_sheet_statement: pd.DataFrame = balance
        self._income_statement: pd.DataFrame = income
        self._cash_flow_statement: pd.DataFrame = cash
        self._rounding = rounding

        # Initialization of Model Variables
        self._dupont_analysis: pd.DataFrame = pd.DataFrame()
        self._dupont_analysis_growth: pd.DataFrame = pd.DataFrame()
        self._extended_dupont_analysis: pd.DataFrame = pd.DataFrame()
        self._extended_dupont_analysis_growth: pd.DataFrame = pd.DataFrame()
        self._enterprise_value_breakdown: pd.DataFrame = pd.DataFrame()
        self._enterprise_value_breakdown_growth: pd.DataFrame = pd.DataFrame()
        self._weighted_average_cost_of_capital: pd.DataFrame = pd.DataFrame()

    @handle_errors
    def get_dupont_analysis(
        self,
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ) -> pd.DataFrame:
        """
        Perform a Dupont analysis to breakdown the return on equity (ROE) into its components.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the values. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.models.get_dupont_analysis()
        ```
        """
        self._dupont_analysis = get_dupont_analysis(
            self._income_statement.loc[:, "Net Income", :],
            self._income_statement.loc[:, "Revenue", :],
            self._balance_sheet_statement.loc[:, "Total Assets", :].shift(axis=1),
            self._balance_sheet_statement.loc[:, "Total Assets", :],
            self._balance_sheet_statement.loc[:, "Total Equity", :].shift(axis=1),
            self._balance_sheet_statement.loc[:, "Total Equity", :],
        )

        if growth:
            self._dupont_analysis_growth = calculate_growth(
                self._dupont_analysis,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        self._dupont_analysis = self._dupont_analysis.round(
            rounding if rounding else self._rounding
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
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ) -> pd.DataFrame:
        """
        Perform am Extended Dupont analysis to breakdown the return on equity (ROE) into its components.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the values. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.models.get_extended_dupont_analysis()
        ```
        """
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

        if growth:
            self._extended_dupont_analysis_growth = calculate_growth(
                self._extended_dupont_analysis,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        self._extended_dupont_analysis = self._extended_dupont_analysis.round(
            rounding if rounding else self._rounding
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

    def get_enterprise_value_breakdown(
        self,
        diluted: bool = True,
        rounding: int | None = 4,
        growth: bool = False,
        lag: int | list[int] = 1,
    ) -> pd.DataFrame:
        """
        The Enterprise Value breakdown corresponds to the following components:
            - Share Price: given for each quarter or year.
            - Market cap: The total value of a company's outstanding common and preferred shares
            - Debt: The sum of long-term and short-term debt
            - Preferred equity: The value of preferred shares
            - Minority interest: The equity value of a subsidiary with less than 50% ownership.
            - Cash and cash equivalents: The total amount of cash, certificates of
            deposit, drafts, money orders, commercial paper, marketable securities, money market
            funds, short-term government bonds, or Treasury bills a company possesses.

        This is displayed in one DataFrame for each company and includes the option to show
        growth values as well.

        Args:
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the values. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        Returns:
            pd.DataFrame: the Enterprise Value breakdown.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.models.get_enterprise_value_breakdown()
        ```
        """
        average_shares = (
            self._income_statement.loc[:, "Weighted Average Shares Diluted", :]
            if diluted
            else self._income_statement.loc[:, "Weighted Average Shares", :]
        )

        years = self._cash_flow_statement.columns
        begin, end = str(years[0]), str(years[-1])

        share_prices = self._historical_data.loc[begin:end, "Adj Close"].T  # type: ignore

        self._enterprise_value_breakdown = get_enterprise_value_breakdown(
            share_price=share_prices,
            shares_outstanding=average_shares,
            total_debt=self._balance_sheet_statement.loc[:, "Total Debt", :],
            minority_interest=self._balance_sheet_statement.loc[
                :, "Minority Interest", :
            ],
            preferred_equity=self._balance_sheet_statement.loc[:, "Preferred Stock", :],
            cash_and_cash_equivalents=self._balance_sheet_statement.loc[
                :, "Cash and Cash Equivalents", :
            ],
        )

        if growth:
            self._enterprise_value_breakdown_growth = calculate_growth(
                self._enterprise_value_breakdown,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        self._enterprise_value_breakdown = self._enterprise_value_breakdown.round(
            rounding
        )

        if len(self._tickers) == 1:
            return (
                self._enterprise_value_breakdown_growth.droplevel(level=0)
                if growth
                else self._enterprise_value_breakdown.droplevel(level=0)
            )

        return (
            self._enterprise_value_breakdown_growth
            if growth
            else self._enterprise_value_breakdown
        )

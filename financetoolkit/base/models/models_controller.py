"""Models Module"""
__docformat__ = "google"

import pandas as pd

from financetoolkit.base.helpers import calculate_growth, handle_errors
from financetoolkit.models.dupont import (
    get_dupont_analysis,
    get_extended_dupont_analysis,
)
from financetoolkit.models.enterprise import get_enterprise_value_breakdown
from financetoolkit.models.intrinsic import get_intrinsic_value
from financetoolkit.models.wacc import get_weighted_average_cost_of_capital
from financetoolkit.performance.performance import get_beta

# pylint: disable=too-many-instance-attributes,too-many-locals


class Models:
    """
    Models Controller Class
    """

    def __init__(
        self,
        tickers: str | list[str],
        daily_historical: pd.DataFrame,
        period_historical: pd.DataFrame,
        risk_free_rate: pd.DataFrame,
        balance: pd.DataFrame,
        income: pd.DataFrame,
        cash: pd.DataFrame,
        quarterly: bool = False,
        rounding: int | None = 4,
    ):
        """
        Initializes the Models Controller Class.
        """
        self._tickers = tickers
        self._benchmark_name = "Benchmark"
        self._balance_sheet_statement: pd.DataFrame = balance
        self._income_statement: pd.DataFrame = income
        self._cash_flow_statement: pd.DataFrame = cash
        self._rounding = rounding

        # Initialization of Historical Data
        self._daily_historical: pd.DataFrame = daily_historical
        self._historical_data: pd.DataFrame = period_historical
        self._risk_free_rate: pd.DataFrame = risk_free_rate

        self._within_historical_data = self._daily_historical.groupby(
            pd.Grouper(freq="Q" if quarterly else "Y")
        ).apply(lambda x: x)

        # Initialization of Model Variables
        self._dupont_analysis: pd.DataFrame = pd.DataFrame()
        self._dupont_analysis_growth: pd.DataFrame = pd.DataFrame()
        self._extended_dupont_analysis: pd.DataFrame = pd.DataFrame()
        self._extended_dupont_analysis_growth: pd.DataFrame = pd.DataFrame()
        self._enterprise_value_breakdown: pd.DataFrame = pd.DataFrame()
        self._enterprise_value_breakdown_growth: pd.DataFrame = pd.DataFrame()
        self._weighted_average_cost_of_capital: pd.DataFrame = pd.DataFrame()
        self._weighted_average_cost_of_capital_growth: pd.DataFrame = pd.DataFrame()
        self._intrinsic_values: pd.DataFrame = pd.DataFrame()

    @handle_errors
    def get_dupont_analysis(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
    ) -> pd.DataFrame:
        """
        Perform a Dupont analysis to breakdown the return on equity (ROE) into its components.

        The Dupont analysis is a method used to dissect and understand the factors that drive a company's
        return on equity (ROE). It breaks down the ROE into three key components: Profit Margin, Asset
        Turnover, and Financial Leverage.

        The formula is as follows:

            - Profit Margin = Net Income / Revenue
            - Asset Turnover = Revenue / Average Total Assets
            - Financial Leverage = Average Total Assets / Average Total Equity
            - ROE = Profit Margin * Asset Turnover * Financial Leverage

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the values. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        Returns:
            pd.DataFrame: DataFrame containing Dupont analysis results, including Profit Margin, Asset
                          Turnover, Financial Leverage, and the calculated ROE values.

        Notes:
            - The Profit Margin is the ratio of Net Income to Total Revenue, indicating the percentage of
            revenue that translates into profit.
            - Asset Turnover measures the efficiency of a company's use of its assets to generate sales
            revenue.
            - Financial Leverage represents the use of debt to finance a company's operations, which can
            amplify returns as well as risks.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        dupont_analysis = toolkit.models.get_dupont_analysis()
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
                axis="index",
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
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
    ) -> pd.DataFrame:
        """
        Perform an Extended Dupont analysis to breakdown the return on equity (ROE) into its components,
        while considering additional financial metrics.

        The Extended Dupont analysis is an advanced method used to break down the return on equity (ROE)
        into multiple components, providing a more detailed insight into the factors influencing a
        company's profitability. It considers additional metrics such as Return on Assets (ROA),
        Total Asset Turnover, Financial Leverage, and more.

        The formula is as follows:

            - Profit Margin = Net Income / Revenue
            - Asset Turnover = Revenue / Average Total Assets
            - Financial Leverage = Average Total Assets / Average Total Equity
            - ROA = Net Income / Average Total Assets
            - Total Asset Turnover = Revenue / Average Total Assets
            - ROE = Profit Margin * Asset Turnover * Financial Leverage * ROA * Total Asset Turnover

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the values. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        Returns:
            pd.DataFrame: DataFrame containing Extended Dupont analysis results, including Profit Margin, Asset Turnover,
                        Financial Leverage, ROA, Total Asset Turnover, and the calculated ROE values.

        Notes:
            - The Profit Margin is the ratio of Net Income to Total Revenue, indicating the percentage of
            revenue that translates into profit.
            - Asset Turnover measures the efficiency of a company's use of its assets to generate
            sales revenue.
            - Financial Leverage represents the use of debt to finance a company's operations, which can
            amplify returns as well as risks.
            - Return on Assets (ROA) measures the efficiency of a company's use of its assets to
            generate profit.
            - Total Asset Turnover considers all assets, including both equity and debt financing.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        extended_dupont_analysis = toolkit.models.get_extended_dupont_analysis()
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
                axis="columns",
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

    @handle_errors
    def get_enterprise_value_breakdown(
        self,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
    ) -> pd.DataFrame:
        """
        Calculate the Enterprise Value (EV) breakdown, providing a detailed view of its components.

        The Enterprise Value breakdown includes the following components for each quarter or year:

            - Share Price: The market price per share of the company's stock.
            - Market Capitalization (Market Cap): The total value of a company's outstanding common and preferred shares.
            - Debt: The sum of long-term and short-term debt on the company's balance sheet.
            - Preferred Equity: The value of preferred shares, if applicable.
            - Minority Interest: The equity value of a subsidiary with less than 50% ownership.
            - Cash and Cash Equivalents: The total amount of liquid assets including cash, marketable securities,
            and short-term investments.

        The Enterprise Value is calculated as the sum of Market Cap, Debt, Preferred Equity,
        Minority Interest, minus Cash and Cash Equivalents.

        This breakdown is displayed in a DataFrame for each company and includes the option to show growth values as well.

        Args:
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the values. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        Returns:
            pd.DataFrame: DataFrame containing the Enterprise Value breakdown, including the calculated components.

        Notes:
        - All the inputs must be in the same currency and unit for accurate calculations.
        - The Enterprise Value is an important metric used for valuation and investment analysis.
        - A positive Enterprise Value indicates that the company is financed primarily by equity and has excess cash.
        - A negative Enterprise Value may indicate financial distress or unusual financial situations.
        - Understanding the Enterprise Value breakdown can provide insights into the sources of a
        company's value and potential risks.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        enterprise_value_breakdown = toolkit.models.get_enterprise_value_breakdown()
        ```
        """
        average_shares = (
            self._income_statement.loc[:, "Weighted Average Shares Diluted", :]
            if diluted
            else self._income_statement.loc[:, "Weighted Average Shares", :]
        )

        years = self._cash_flow_statement.columns
        begin, end = str(years[0]), str(years[-1])
        share_prices = self._historical_data.loc[begin:end, "Adj Close"][
            self._tickers
        ].T

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
            rounding if rounding else self._rounding
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

    @handle_errors
    def get_weighted_average_cost_of_capital(
        self,
        show_full_results: bool = True,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
    ) -> pd.DataFrame:
        """
        The Weighted Average Cost of Capital (WACC) is a financial metric used to estimate
        the cost of capital for a company. It represents the average rate of return a company
        must pay to its investors for using their capital. WACC takes into account the cost of
        both equity and debt, weighted by their respective proportions in the company's capital structure.

        The formula is as follows:

            Market Value of Equity = Share Price * Total Shares Outstanding

            Market Value of Debt = Total Debt

            Total Market Value = Market Value of Equity + Market Value of Debt

            Cost of Equity = Risk Free Rate + Beta * (Benchmark Return - Risk Free Rate)

            Cost of Debt = Interest Expense / Total Debt

            WACC = (Market Value of Equity / Total Market Value) * Cost of Equity +
            (Market Value of Debt / Total Market Value) * Cost of Debt * (1 - Corporate Tax Rate)

        Cost of Equity (Re): The cost of equity represents the return required by the company's shareholders or
        equity investors. It is the cost of raising funds by selling equity (such as common stock). The cost
        of equity is often estimated using methods like the Capital Asset Pricing Model (CAPM) or
        the Dividend Discount Model (DDM).

        Cost of Debt (Rd): The cost of debt is the interest rate the company pays on its outstanding debt.
        It is the cost of raising funds through borrowing, such as issuing bonds or taking loans.
        The cost of debt is typically based on the prevailing interest rates in the market and the
        company's creditworthiness.

        Corporate Tax Rate (Tc): The corporate tax rate is the percentage of a company's profits that
        is paid in taxes. It is used to calculate the tax shield on interest payments. Interest expenses
        on debt reduce taxable income, and the tax shield represents the tax savings resulting from these deductions.

        Market Value of Equity (E): The market value of equity is the total value of the company's outstanding
        shares of common stock. It is calculated by multiplying the current stock price by the number of shares
        outstanding.

        Market Value of Debt (D): The market value of debt is the total value of the company's outstanding
        debt obligations, such as bonds and loans. It represents the current market price of the debt instruments.

        Total Market Value of Capital (V): The total market value of capital is the sum of the market value of
        equity and the market value of debt (V = E + D). It represents the total value of the company's financing,
        both through equity and debt.

        Args:
            show_full_results (bool, optional): Whether to show the full results or just the WACC values.
            Defaults to True.
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the values. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.

        Returns:
            pd.DataFrame: DataFrame containing Dupont analysis results, including Profit Margin, Asset
                          Turnover, Financial Leverage, and the calculated ROE values.

        Notes:
        - The Cost of Equity is approximated with the Capital Asset Pricing Model (CAPM).
        - The Market Value of Debt is approximated as the Total Debt.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.models.get_weighted_average_cost_of_capital()
        ```
        """
        average_shares = (
            self._income_statement.loc[:, "Weighted Average Shares Diluted", :]
            if diluted
            else self._income_statement.loc[:, "Weighted Average Shares", :]
        )

        years = self._cash_flow_statement.columns
        begin, end = str(years[0]), str(years[-1])

        share_prices = self._historical_data.loc[begin:end, "Adj Close"][
            self._tickers
        ].T

        risk_free_rate = self._risk_free_rate.loc[begin:end, "Adj Close"]

        returns_within = self._within_historical_data.loc[begin:end, "Return"][
            self._tickers
        ]
        benchmark_returns_within = self._within_historical_data.loc[
            begin:end, "Return"
        ][self._benchmark_name]

        beta = get_beta(returns_within, benchmark_returns_within)

        benchmark_returns = self._historical_data.loc[begin:end, "Return"][
            self._benchmark_name
        ]

        self._weighted_average_cost_of_capital = get_weighted_average_cost_of_capital(
            share_price=share_prices,
            total_shares_outstanding=average_shares,
            interest_expense=self._income_statement.loc[:, "Interest Expense", :],
            total_debt=self._balance_sheet_statement.loc[:, "Total Debt", :],
            risk_free_rate=risk_free_rate,
            beta=beta,
            benchmark_returns=benchmark_returns,
            income_tax_expense=self._income_statement.loc[:, "Income Tax Expense", :],
            income_before_tax=self._income_statement.loc[:, "Income Before Tax", :],
        )

        if growth:
            self._weighted_average_cost_of_capital_growth = calculate_growth(
                self._weighted_average_cost_of_capital,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="columns",
            )

        self._weighted_average_cost_of_capital = (
            self._weighted_average_cost_of_capital.round(
                rounding if rounding else self._rounding
            )
        )

        if len(self._tickers) == 1:
            return (
                self._weighted_average_cost_of_capital_growth.droplevel(level=0)
                if growth
                else self._weighted_average_cost_of_capital.droplevel(level=0)
            )

        if show_full_results:
            return (
                self._weighted_average_cost_of_capital_growth
                if growth
                else self._weighted_average_cost_of_capital
            )

        return (
            self._weighted_average_cost_of_capital_growth.loc[
                :, "Weighted Average Cost of Capital", :
            ]
            if growth
            else self._weighted_average_cost_of_capital.loc[
                :, "Weighted Average Cost of Capital", :
            ]
        )

    def get_intrinsic_valuation(
        self,
        growth_rate: float | list | dict[str, float],
        perpetual_growth_rate: float | list | dict[str, float],
        weighted_average_cost_of_capital: float | list | dict[str, float],
        periods: int = 5,
        cash_flow_type: str = "Free Cash Flow",
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Intrinsic value is a fundamental concept in finance and investing that represents the
        true worth or value of an asset, security, or investment, independent of its current
        market price or prevailing market sentiment. It is a concept often associated with
        the value investing philosophy, made famous by legendary investors like Benjamin Graham
        and Warren Buffett. Understanding intrinsic value is crucial for investors looking to make
        informed decisions about where to allocate their capital.

        This functionality uses DCF, or Discounted Cash Flow which is a widely used financial
        valuation method that allows investors and analysts to estimate the intrinsic value of
        an investment or business based on its expected future cash flows. It is a fundamental
        tool in finance and investment analysis, providing a systematic way to assess the
        present value of future cash flows while considering the time value of money.

        The formula is as follows:

            Cash Flow Projection_t = Cash Flow_t-1 * (1 + Growth Rate)

            Terminal Value = Last Cash Flow Projection * (1 + Perpetual Growth Rate) /
            (Weighted Average Cost of Capital - Perpetual Growth Rate)

            Enterprise Value = Sum of Present Value of Cash Flow Projections + Terminal Value

            Equity Value = Enterprise Value - Total Debt + Cash and Cash Equivalents

            Intrinsic Value = Equity Value / Total Shares Outstanding

        Args:
            growth_rate (float, list or dict): The growth rate to use for the cash flow projections. Can be one number
            to use for all tickers, or a list or dict that contains a growth rate for each ticker.
            perpetual_growth_rate (float, list or dict): The perpetual growth rate to use for the terminal value.
            Can be one number to use for all tickers, or a list or dict that contains a perpetual growth rate for
            each ticker.
            weighted_average_cost_of_capital (float, list or dict): The weighted average cost of capital to use for
            the terminal value. Can be one number to use for all tickers, or a list or dict that contains a
            weighted average cost of capital for each ticker.
            periods (int, optional): The number of periods to use for the cash flow projections. Defaults to 5.
            cash_flow_type (str, optional): The type of cash flow to use for the cash flow projections.
            Defaults to "Free Cash Flow". Other options are "Operating Cash Flow", "Change in Working Capital",
            and "Capital Expenditure".
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.

        Returns:
            pd.DataFrame: DataFrame containing the intrinsic value for each ticker.

        Notes:
        - The results are highly dependent on the input. Therefore, think carefully about each input parameter to
        ensure the results are accurate (given your beliefs)

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.models.get_intrinsic_valuation(0.05, 0.025, 0.094)
        ```
        """
        if cash_flow_type not in [
            "Free Cash Flow",
            "Operating Cash Flow",
            "Change in Working Capital",
            "Capital Expenditure",
        ]:
            raise ValueError(
                "Invalid cash flow type. Please choose from the following: 'Free Cash Flow', 'Operating Cash Flow',"
                "'Change in Working Capital', 'Capital Expenditure'"
            )

        if isinstance(growth_rate, list):
            if len(growth_rate) != len(self._tickers):
                raise ValueError(
                    "The length of the growth rate list must match the number of tickers."
                )
            growth_rate_dict = {
                ticker: growth_rate[i] for i, ticker in enumerate(self._tickers)
            }
        elif isinstance(growth_rate, dict):
            if len(growth_rate) != len(self._tickers):
                raise ValueError(
                    "The length of the growth rate dict must match the number of tickers."
                )
            growth_rate_dict = growth_rate
        else:
            growth_rate_dict = {}

        if isinstance(perpetual_growth_rate, list):
            if len(perpetual_growth_rate) != len(self._tickers):
                raise ValueError(
                    "The length of the perpetual growth rate list must match the number of tickers."
                )
            perpetual_growth_rate_dict = {
                ticker: perpetual_growth_rate[i]
                for i, ticker in enumerate(self._tickers)
            }
        elif isinstance(perpetual_growth_rate, dict):
            if len(perpetual_growth_rate) != len(self._tickers):
                raise ValueError(
                    "The length of the perpetual growth rate dict must match the number of tickers."
                )
            perpetual_growth_rate_dict = perpetual_growth_rate
        else:
            perpetual_growth_rate_dict = {}

        if isinstance(weighted_average_cost_of_capital, list):
            if len(weighted_average_cost_of_capital) != len(self._tickers):
                raise ValueError(
                    "The length of weighted average cost of capital list must match the number of tickers."
                )
            wacc_dict = {
                ticker: weighted_average_cost_of_capital[i]
                for i, ticker in enumerate(self._tickers)
            }
        elif isinstance(weighted_average_cost_of_capital, dict):
            if len(weighted_average_cost_of_capital) != len(self._tickers):
                raise ValueError(
                    "The length of weighted average cost of capital dict must match the number of tickers."
                )
            wacc_dict = weighted_average_cost_of_capital
        else:
            wacc_dict = {}

        intrinsic_values_dict = {}
        for ticker in self._tickers:
            intrinsic_values_dict[ticker] = get_intrinsic_value(
                cash_flow=self._cash_flow_statement.loc[ticker, cash_flow_type]
                .dropna()
                .iloc[-1],
                growth_rate=growth_rate_dict[ticker]
                if growth_rate_dict
                else growth_rate,
                perpetual_growth_rate=perpetual_growth_rate_dict[ticker]
                if perpetual_growth_rate_dict
                else perpetual_growth_rate,
                weighted_average_cost_of_capital=wacc_dict[ticker]
                if wacc_dict
                else weighted_average_cost_of_capital,
                cash_and_cash_equivalents=self._balance_sheet_statement.loc[
                    ticker, "Cash and Cash Equivalents"
                ]
                .dropna()
                .iloc[-1],
                total_debt=self._balance_sheet_statement.loc[ticker, "Total Debt"]
                .dropna()
                .iloc[-1],
                shares_outstanding=self._income_statement.loc[
                    ticker, "Weighted Average Shares Diluted"
                ]
                .dropna()
                .iloc[-1],
                periods=periods,
            )
            intrinsic_values_dict[ticker] = intrinsic_values_dict[ticker].round(
                rounding if rounding else self._rounding
            )

        self._intrinsic_values = pd.concat(intrinsic_values_dict)

        return self._intrinsic_values

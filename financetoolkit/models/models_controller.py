"""Models Module"""

__docformat__ = "google"

import pandas as pd

from financetoolkit.helpers import calculate_growth
from financetoolkit.models import (
    altman_model,
    dupont_model,
    enterprise_model,
    growth_model,
    helpers,
    intrinsic_model,
    piotroski_model,
    wacc_model,
)
from financetoolkit.performance.performance_model import get_beta
from financetoolkit.ratios import liquidity_model, valuation_model
from financetoolkit.utilities.error_model import handle_errors

# pylint: disable=too-many-instance-attributes,too-many-locals,too-many-lines


class Models:
    """
    The Models module is meant to execute well-known models such
    as DUPONT and the Discounted Cash Flow (DCF) model. These models
    are also directly related to the data retrieved from the Toolkit module.

    """

    def __init__(
        self,
        tickers: str | list[str],
        historical_data: pd.DataFrame,
        risk_free_rate_data: pd.DataFrame,
        balance: pd.DataFrame,
        income: pd.DataFrame,
        cash: pd.DataFrame,
        quarterly: bool = False,
        rounding: int | None = 4,
    ):
        """
        Initializes the Models Controller Class.

        Args:
            tickers (str | list[str]): The ticker(s) to use for the models.
            daily_historical (pd.DataFrame): The daily historical data.
            period_historical (pd.DataFrame): The period historical data.
            risk_free_rate (pd.DataFrame): The risk free rate data.
            balance (pd.DataFrame): The balance sheet data.
            income (pd.DataFrame): The income statement data.
            cash (pd.DataFrame): The cash flow statement data.
            quarterly (bool, optional): Whether to use quarterly or yearly data. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["TSLA", "AMZN"], api_key="FINANCIAL_MODELING_PREP_KEY",
        quarterly=True, start_date='2022-12-31')

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
        self._tickers = tickers
        self._benchmark_name = "Benchmark"
        self._balance_sheet_statement: pd.DataFrame = balance
        self._income_statement: pd.DataFrame = income
        self._cash_flow_statement: pd.DataFrame = cash
        self._quarterly = quarterly
        self._rounding = rounding

        # Historical Data
        self._historical_data = historical_data
        self._risk_free_rate_data = risk_free_rate_data

        # Within Period Calculations
        daily_historical_data = self._historical_data["daily"].copy()

        daily_historical_data.index = pd.DatetimeIndex(
            daily_historical_data.to_timestamp().index
        )

        self._within_historical_data = helpers.determine_within_historical_data(
            daily_historical_data=daily_historical_data,
        )

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
        trailing: int | None = None,
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
            trailing (int | None, optional): The trailing period to use for the calculation. Defaults to None.

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

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        dupont_analysis = toolkit.models.get_dupont_analysis()
        ```
        """
        if trailing:
            self._dupont_analysis = dupont_model.get_dupont_analysis(
                self._income_statement.loc[:, "Net Income", :]
                .T.rolling(trailing)
                .sum()
                .T,
                self._income_statement.loc[:, "Revenue", :].T.rolling(trailing).sum().T,
                self._balance_sheet_statement.loc[:, "Total Assets", :]
                .T.rolling(trailing)
                .mean()
                .T,
                self._balance_sheet_statement.loc[:, "Total Equity", :]
                .T.rolling(trailing)
                .mean()
                .T,
            )
        else:
            self._dupont_analysis = dupont_model.get_dupont_analysis(
                self._income_statement.loc[:, "Net Income", :],
                self._income_statement.loc[:, "Revenue", :],
                self._balance_sheet_statement.loc[:, "Total Assets", :]
                .T.rolling(2)
                .mean()
                .T,
                self._balance_sheet_statement.loc[:, "Total Equity", :]
                .T.rolling(2)
                .mean()
                .T,
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
        trailing: int | None = None,
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
            trailing (int | None, optional): The trailing period to use for the calculation. Defaults to None.

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

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        extended_dupont_analysis = toolkit.models.get_extended_dupont_analysis()
        ```
        """
        if trailing:
            self._extended_dupont_analysis = dupont_model.get_extended_dupont_analysis(
                self._income_statement.loc[:, "Income Before Tax", :]
                .T.rolling(trailing)
                .sum()
                .T,
                self._income_statement.loc[:, "Operating Income", :]
                .T.rolling(trailing)
                .sum()
                .T,
                self._income_statement.loc[:, "Net Income", :]
                .T.rolling(trailing)
                .sum()
                .T,
                self._income_statement.loc[:, "Revenue", :].T.rolling(trailing).sum().T,
                self._balance_sheet_statement.loc[:, "Total Assets", :]
                .T.rolling(trailing)
                .mean()
                .T,
                self._balance_sheet_statement.loc[:, "Total Equity", :]
                .T.rolling(trailing)
                .mean()
                .T,
            )
        else:
            self._extended_dupont_analysis = dupont_model.get_extended_dupont_analysis(
                self._income_statement.loc[:, "Income Before Tax", :],
                self._income_statement.loc[:, "Operating Income", :],
                self._income_statement.loc[:, "Net Income", :],
                self._income_statement.loc[:, "Revenue", :],
                self._balance_sheet_statement.loc[:, "Total Assets", :]
                .T.rolling(2)
                .mean()
                .T,
                self._balance_sheet_statement.loc[:, "Total Equity", :]
                .T.rolling(2)
                .mean()
                .T,
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

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

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
        share_prices = (
            self._historical_data["quarterly" if self._quarterly else "yearly"]
            .loc[begin:end, "Adj Close"][self._tickers]
            .T
        )

        self._enterprise_value_breakdown = (
            enterprise_model.get_enterprise_value_breakdown(
                share_price=share_prices,
                shares_outstanding=average_shares,
                total_debt=self._balance_sheet_statement.loc[:, "Total Debt", :],
                minority_interest=self._balance_sheet_statement.loc[
                    :, "Minority Interest", :
                ],
                preferred_equity=self._balance_sheet_statement.loc[
                    :, "Preferred Stock", :
                ],
                cash_and_cash_equivalents=self._balance_sheet_statement.loc[
                    :, "Cash and Cash Equivalents", :
                ],
            )
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

            - Market Value of Equity = Share Price * Total Shares Outstanding
            - Market Value of Debt = Total Debt
            - Total Market Value = Market Value of Equity + Market Value of Debt
            - Cost of Equity = Risk Free Rate + Beta * (Benchmark Return — Risk Free Rate)
            - Cost of Debt = Interest Expense / Total Debt
            - WACC = (Market Value of Equity / Total Market Value) * Cost of Equity +
            (Market Value of Debt / Total Market Value) * Cost of Debt * (1 — Corporate Tax Rate)

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
            pd.DataFrame: DataFrame containing the WACC values.

        Notes:
        - The Cost of Equity is approximated with the Capital Asset Pricing Model (CAPM).
        - The Market Value of Debt is approximated as the Total Debt.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

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

        share_prices = (
            self._historical_data["quarterly" if self._quarterly else "yearly"]
            .loc[begin:end, "Adj Close"][self._tickers]
            .T
        )

        risk_free_rate = self._risk_free_rate_data[
            "quarterly" if self._quarterly else "yearly"
        ].loc[begin:end, "Adj Close"]

        returns_within = self._within_historical_data[
            "quarterly" if self._quarterly else "yearly"
        ].loc[begin:end, "Return"][self._tickers]
        benchmark_returns_within = self._within_historical_data[
            "quarterly" if self._quarterly else "yearly"
        ].loc[begin:end, "Return"][self._benchmark_name]

        beta = get_beta(returns_within, benchmark_returns_within)

        benchmark_returns = self._historical_data[
            "quarterly" if self._quarterly else "yearly"
        ].loc[begin:end, "Return"][self._benchmark_name]

        self._weighted_average_cost_of_capital = (
            wacc_model.get_weighted_average_cost_of_capital(
                share_price=share_prices,
                total_shares_outstanding=average_shares,
                interest_expense=self._income_statement.loc[:, "Interest Expense", :],
                total_debt=self._balance_sheet_statement.loc[:, "Total Debt", :],
                risk_free_rate=risk_free_rate,
                beta=beta,
                benchmark_returns=benchmark_returns,
                income_tax_expense=self._income_statement.loc[
                    :, "Income Tax Expense", :
                ],
                income_before_tax=self._income_statement.loc[:, "Income Before Tax", :],
            )
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

            - Cash Flow Projection_t = Cash Flow_t-1 * (1 + Growth Rate)
            - Terminal Value = Last Cash Flow Projection * (1 + Perpetual Growth Rate) /
            (Weighted Average Cost of Capital — Perpetual Growth Rate)
            - Enterprise Value = Sum of Present Value of Cash Flow Projections + Terminal Value
            - Equity Value = Enterprise Value — Total Debt + Cash and Cash Equivalents
            - Intrinsic Value = Equity Value / Total Shares Outstanding

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

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

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
            perpetual_growth_rate_float = perpetual_growth_rate_dict.get(
                ticker, perpetual_growth_rate
            )
            growth_rate_float = growth_rate_dict.get(ticker, growth_rate)
            weighted_average_cost_of_capital_float = wacc_dict.get(
                ticker, weighted_average_cost_of_capital
            )
            intrinsic_values_dict[ticker] = intrinsic_model.get_intrinsic_value(
                cash_flow=self._cash_flow_statement.loc[ticker, cash_flow_type]
                .dropna()
                .iloc[-1],
                growth_rate=growth_rate_float,
                perpetual_growth_rate=perpetual_growth_rate_float,
                weighted_average_cost_of_capital=weighted_average_cost_of_capital_float,
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

    def get_gorden_growth_model(
        self,
        rate_of_return: float,
        growth_rate: float,
        project_periods: int = 5,
        rounding: int | None = None,
    ):
        """
        The Gordon Growth Model, also known as the Dividend Discount Model (DDM) with Constant Growth,
        is a method used to estimate the intrinsic value of a stock based on its expected future dividends.
        The model assumes that dividends will grow at a constant rate indefinitely.

        The formula is as follows:

        - Intrinsic Value = (Dividends Per Share * (1 + Growth Rate)) / (Rate of Return — Growth Rate)

        The formula essentially discounts the future expected dividends to their present value, taking into account
        the required rate of return and the growth rate. The numerator represents the expected dividend in the
        next period. The denominator represents the required rate of return minus the growth rate.

        Investors often use the Gordon Growth Model to compare the intrinsic value of a stock with its current
        market price. If the intrinsic value is higher than the market price, some investors may interpret it
        as an indication that the stock is undervalued.

        It's important to note that the Gordon Growth Model is based on several assumptions, including the
        assumption of constant growth in dividends. It is most applicable to mature companies with stable and
        predictable dividend growth. If a company's dividend growth is expected to fluctuate or if it does
        not pay dividends, alternative valuation models may be more appropriate.

        The assumption of constant growth of dividends is often unrealistic. In reality, dividends may fluctuate
        or even be suspended. Therefore, the Gordon Growth Model should be used with caution and in conjunction
        with other valuation methods.

        Args:
            rate_of_return (float): The required rate of return.
            growth_rate (float): The growth rate of the dividends.
            project_periods (int, optional): The number of periods to project the the stock price. Defaults to 5.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.

        Returns:
            pd.DataFrame: DataFrame containing the intrinsic value for each ticker over time.

        Notes:
        - The results are highly dependent on the input. Therefore, think carefully about each input parameter to
        ensure the results are accurate (given your beliefs)

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "MSFT"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.models.get_gorden_growth_model(0.20, 0.05)
        ```
        """
        dividends_per_share = self._historical_data[
            "quarterly" if self._quarterly else "yearly"
        ]["Dividends"]

        gorden_growth_model: dict[str, dict[str, float]] = {}
        previous_period = dividends_per_share.index[0]

        periods = pd.period_range(
            start=dividends_per_share.index[0],
            periods=project_periods + len(dividends_per_share.index),
            freq="Q" if self._quarterly else "Y",
        )

        for ticker in self._tickers:
            gorden_growth_model[ticker] = {}

            for period in periods:
                previous_period_location = periods.get_loc(previous_period)
                period_location = periods.get_loc(period)
                distance = period_location - previous_period_location

                if (period_location + 1) < len(dividends_per_share.index):
                    previous_period = period

                dividends_per_share_value = (
                    dividends_per_share.loc[period, ticker]
                    if period != dividends_per_share.index[-1]
                    and period in dividends_per_share.index
                    else dividends_per_share.loc[previous_period, ticker]
                    * (1 + growth_rate) ** distance
                )

                gorden_growth_model[ticker][period] = (
                    intrinsic_model.get_gorden_growth_model(
                        dividends_per_share=dividends_per_share_value,
                        rate_of_return=rate_of_return,
                        growth_rate=growth_rate,
                    )
                )

        gorden_growth_model_df = pd.DataFrame(gorden_growth_model)

        gorden_growth_model_df = gorden_growth_model_df.round(
            rounding if rounding else self._rounding
        )

        return gorden_growth_model_df

    @handle_errors
    def get_altman_z_score(
        self,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
    ) -> pd.DataFrame:
        """
        Calculates the Altman Z-Score, a financial metric used to predict the likelihood of a company going bankrupt.
        The Altman Z-Score is calculated using several financial ratios, including working capital to total assets,
        retained earnings to total assets, earnings before interest and taxes (EBIT) to total assets, market value
        of equity to book value of total liabilities, and sales to total assets.

        The formula is as follows:

            - Working Capital to Total Assets = Working Capital / Total Assets
            - Retained Earnings to Total Assets = Retained Earnings / Total Assets
            - EBIT to Total Assets = EBIT / Total Assets
            - Market Value to Total Liabilities = Market Value of Equity / Total Liabilities
            - Sales to Total Assets = Sales / Total Assets
            - Altman Z-Score = 1.2 * Working Capital to Total Assets + 1.4 * Retained Earnings to Total Assets +
            3.3 * EBIT to Total Assets + 0.6 * Market Value to Total Liabilities + 1.0 * Sales to Total Assets

        The Altman Z-Score can be interpreted as follows:

            - A Z-Score of less than 1.81 indicates a high likelihood of bankruptcy.
            - A Z-Score between 1.81 and 2.99 indicates a gray area.
            - A Z-Score of greater than 2.99 indicates a low likelihood of bankruptcy.

        Args:
            diluted (bool, optional): Whether to use diluted shares outstanding in the calculation. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to None.
            growth (bool, optional): Whether to calculate the growth of the values. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.

        Returns:
            pd.DataFrame: DataFrame containing the Altman Z-Score and its components.

        Notes:
            - The Altman Z-Score is a financial metric used to predict the likelihood of a company going bankrupt.
            - The Z-Score is calculated using several financial ratios, including working capital to total assets,
            retained earnings to total assets, earnings before interest and taxes (EBIT) to total assets, market value
            of equity to book value of total liabilities, and sales to total assets.
            - A Z-Score of less than 1.81 indicates a high likelihood of bankruptcy, while a Z-Score of greater than 2.99
            indicates a low likelihood of bankruptcy.
            - The Z-Score is most effective when used to analyze manufacturing companies with assets of
            $1 million or more.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        altman_z_score = toolkit.models.get_altman_z_score()
        ```
        """
        altman_z_score = {}

        working_capital = liquidity_model.get_working_capital(
            self._balance_sheet_statement.loc[:, "Total Current Assets", :],
            self._balance_sheet_statement.loc[:, "Total Current Liabilities", :],
        )

        altman_z_score["Working Capital to Total Assets"] = (
            altman_model.get_working_capital_to_total_assets_ratio(
                working_capital=working_capital,
                total_assets=self._balance_sheet_statement.loc[:, "Total Assets", :],
            )
        )

        altman_z_score["Retained Earnings to Total Assets"] = (
            altman_model.get_retained_earnings_to_total_assets_ratio(
                retained_earnings=self._balance_sheet_statement.loc[
                    :, "Retained Earnings", :
                ],
                total_assets=self._balance_sheet_statement.loc[:, "Total Assets", :],
            )
        )

        altman_z_score["EBIT to Total Assets"] = (
            altman_model.get_earnings_before_interest_and_taxes_to_total_assets_ratio(
                ebit=(
                    self._income_statement.loc[:, "Net Income", :]
                    + self._income_statement.loc[:, "Income Tax Expense", :]
                    + self._income_statement.loc[:, "Interest Expense", :]
                ),
                total_assets=self._balance_sheet_statement.loc[:, "Total Assets", :],
            )
        )

        years = self._balance_sheet_statement.columns
        begin, end = str(years[0]), str(years[-1])

        share_prices = (
            self._historical_data["quarterly" if self._quarterly else "yearly"]
            .loc[begin:end, "Adj Close"][self._tickers]
            .T
        )

        average_shares = (
            self._income_statement.loc[:, "Weighted Average Shares Diluted", :]
            if diluted
            else self._income_statement.loc[:, "Weighted Average Shares", :]
        )

        market_cap = valuation_model.get_market_cap(
            share_price=share_prices, total_shares_outstanding=average_shares
        )

        altman_z_score["Market Value to Total Liabilities"] = (
            altman_model.get_market_value_of_equity_to_book_value_of_total_liabilities_ratio(
                market_value_of_equity=market_cap,
                total_liabilities=self._balance_sheet_statement.loc[
                    :, "Total Liabilities", :
                ],
            )
        )

        altman_z_score["Sales to Total Assets"] = (
            altman_model.get_sales_to_total_assets_ratio(
                sales=self._income_statement.loc[:, "Revenue", :],
                total_assets=self._balance_sheet_statement.loc[:, "Total Assets", :],
            )
        )

        altman_z_score["Altman Z-Score"] = altman_model.get_altman_z_score(
            altman_z_score["Working Capital to Total Assets"],
            altman_z_score["Retained Earnings to Total Assets"],
            altman_z_score["EBIT to Total Assets"],
            altman_z_score["Market Value to Total Liabilities"],
            altman_z_score["Sales to Total Assets"],
        )

        altman_results = (
            pd.concat(altman_z_score)
            .dropna(axis=1, how="all")
            .swaplevel(0, 1)
            .reindex(self._tickers, level=0)
        )

        if growth:
            return calculate_growth(
                altman_results,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="columns",
            )

        altman_results = altman_results.round(rounding if rounding else self._rounding)

        return altman_results

    @handle_errors
    def get_piotroski_score(self) -> pd.DataFrame:
        """
        Calculate the Piotroski Score, a comprehensive financial assessment tool that helps investors and analysts
        evaluate a company's financial health and fundamental strength.

        The Piotroski Score was developed by Joseph Piotroski and is based on a set of nine fundamental
        financial criteria. Each criterion is assigned a score of 0 or 1, and the scores are then summed to
        calculate the Piotroski Score.

        The nine criteria are categorized into three groups:

        1. Profitability:
            - Return on Assets (ROA) Criteria: Measures the profitability of the company.
            - Operating Cash Flow Criteria: Evaluates the company's ability to generate cash from its operations.
            - Change in ROA Criteria: Assesses the trend in ROA over time.
            - Accruals Criteria: Examines the quality of earnings.

        2. Leverage, Liquidity, and Operating Efficiency:
            - Change in Leverage Criteria: Analyzes changes in the company's leverage (debt).
            - Change in Current Ratio Criteria: Evaluates changes in the current ratio.
            - Number of Shares Criteria: Assesses the issuance of common shares.

        3. Operating Efficiency and Asset Utilization:
            - Gross Margin Criteria: Examines the company's gross margin, a measure of profitability.
            - Asset Turnover Ratio Criteria: Evaluates the efficiency of asset utilization and sales generation.

        The Piotroski Score is calculated by summing the scores assigned to each of the nine criteria.
        The maximum possible score is 9, indicating the highest financial strength, while the minimum score is 0,
        suggesting potential financial weaknesses.

        Note that the Piostroski Score has been developed many decades ago and that it is important to
        always compare the same sectors. E.g. it could be that it is quite normal that a firm issues shares
        each year which nets a lower score even though it is a normal practice in that sector.

        Please see Piotroski, Joseph D. "Value Investing: The Use of Historical Financial Statement
        Information to Separate Winners from Losers." Journal of Accounting Research, Vol. 38, No.
        3, 1999, pp. 1-41.

        Args:
            growth (bool, optional): Whether to calculate the growth of the values. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.

        Returns:
            pd.DataFrame: DataFrame containing the Piotroski F-Score and its components.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA", "MSFT"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.models.get_piotroski_score()
        ```
        """
        piotroski_score = {}

        piotroski_score["Return on Assets Criteria"] = (
            piotroski_model.get_return_on_assets_criteria(
                net_income=self._income_statement.loc[:, "Net Income", :],
                average_total_assets=self._balance_sheet_statement.loc[
                    :, "Total Assets", :
                ]
                .T.rolling(2)
                .mean()
                .T,
            )
        )

        piotroski_score["Operating Cashflow Criteria"] = (
            piotroski_model.get_operating_cashflow_criteria(
                operating_cashflow=self._cash_flow_statement.loc[
                    :, "Operating Cash Flow", :
                ],
            )
        )

        piotroski_score["Change in Return on Assets Criteria"] = (
            piotroski_model.get_change_in_return_on_asset_criteria(
                net_income=self._income_statement.loc[:, "Net Income", :],
                average_total_assets=self._balance_sheet_statement.loc[
                    :, "Total Assets", :
                ]
                .T.rolling(2)
                .mean()
                .T,
            )
        )

        piotroski_score["Accruals Criteria"] = piotroski_model.get_accruals_criteria(
            net_income=self._income_statement.loc[:, "Net Income", :],
            average_total_assets=self._balance_sheet_statement.loc[:, "Total Assets", :]
            .T.rolling(2)
            .mean()
            .T,
            operating_cashflow=self._cash_flow_statement.loc[
                :, "Operating Cash Flow", :
            ],
            total_assets=self._balance_sheet_statement.loc[:, "Total Assets", :],
        )

        piotroski_score["Change in Leverage Criteria"] = (
            piotroski_model.get_change_in_leverage_criteria(
                total_debt=self._balance_sheet_statement.loc[:, "Total Debt", :],
                total_assets=self._balance_sheet_statement.loc[:, "Total Assets", :],
            )
        )

        piotroski_score["Change in Current Ratio Criteria"] = (
            piotroski_model.get_change_in_current_ratio_criteria(
                current_assets=self._balance_sheet_statement.loc[
                    :, "Total Current Assets", :
                ],
                current_liabilities=self._balance_sheet_statement.loc[
                    :, "Total Current Liabilities", :
                ],
            )
        )

        piotroski_score["Number of Shares Criteria"] = (
            piotroski_model.get_number_of_shares_criteria(
                common_stock_issued=self._cash_flow_statement.loc[
                    :, "Common Stock Issued", :
                ],
            )
        )

        piotroski_score["Gross Margin Criteria"] = (
            piotroski_model.get_gross_margin_criteria(
                revenue=self._income_statement.loc[:, "Revenue", :],
                cost_of_goods_sold=self._income_statement.loc[
                    :, "Cost of Goods Sold", :
                ],
            )
        )

        piotroski_score["Asset Turnover Criteria"] = (
            piotroski_model.get_asset_turnover_ratio_criteria(
                sales=self._income_statement.loc[:, "Revenue", :],
                average_total_assets=self._balance_sheet_statement.loc[
                    :, "Total Assets", :
                ]
                .T.rolling(2)
                .mean()
                .T,
            )
        )

        piotroski_score["Piotroski Score"] = piotroski_model.get_piotroski_score(
            piotroski_score["Return on Assets Criteria"],
            piotroski_score["Operating Cashflow Criteria"],
            piotroski_score["Change in Return on Assets Criteria"],
            piotroski_score["Accruals Criteria"],
            piotroski_score["Change in Leverage Criteria"],
            piotroski_score["Change in Current Ratio Criteria"],
            piotroski_score["Number of Shares Criteria"],
            piotroski_score["Gross Margin Criteria"],
            piotroski_score["Asset Turnover Criteria"],
        )

        piotroski_results = (
            pd.concat(piotroski_score)
            .dropna(axis=1, how="all")
            .swaplevel(0, 1)
            .reindex(self._tickers, level=0)
        )

        # The first column is taken out because calculating the change of the
        # first date will always result in NaN which means that any criteria
        # looking at the change over time will return a 0. This is a meaningless
        # result for the analysis
        piotroski_results = piotroski_results[piotroski_results.columns[1:]]

        return piotroski_results

    @handle_errors
    def get_present_value_of_growth_opportunities(
        self,
        calculate_daily: bool = False,
        diluted: bool = True,
        include_dividends: bool = False,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
    ) -> pd.DataFrame:
        """
        The Present Value of Growth Opportunities (PVGO) is a financial metric that represents the
        present value of a company's future growth opportunities. It is calculated as the difference
        between the company's current stock price and the discounted value of its future cash flows.

        The formula is as follows:

            - PVGO = Stock Price — Earnings Per Share / Weighted Average Cost of Capital

        Args:
            calculate_daily (bool, optional): Whether to calculate the PVGO using daily historical data.
            Defaults to False.
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            include_dividends (bool, optional): Whether to include dividends in the calculation.
            Defaults to False.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the values. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.

        Returns:
            pd.DataFrame: DataFrame containing the PVGO values.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.models.get_present_value_of_growth_opportunities()
        ```
        """
        wacc = self.get_weighted_average_cost_of_capital(show_full_results=False)

        average_shares = (
            self._income_statement.loc[:, "Weighted Average Shares Diluted", :]
            if diluted
            else self._income_statement.loc[:, "Weighted Average Shares", :]
        )

        dividends = (
            self._cash_flow_statement.loc[:, "Preferred Dividends Paid", :]
            if include_dividends
            else 0
        )

        earnings_per_share = valuation_model.get_earnings_per_share(
            net_income=self._income_statement.loc[:, "Net Income", :],
            preferred_dividends=dividends,
            average_outstanding_shares=average_shares,
        )

        historical_prices = (
            self._historical_data["daily"]["Adj Close"]
            if calculate_daily
            else self._historical_data["quarterly" if self._quarterly else "yearly"][
                "Adj Close"
            ]
        )

        pvgo = growth_model.get_present_value_of_growth_opportunities(
            weighted_average_cost_of_capital=wacc,
            earnings_per_share=earnings_per_share,
            close_prices=historical_prices[self._tickers],
            calculate_daily=calculate_daily,
        )

        if growth:
            return calculate_growth(
                pvgo,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        pvgo = pvgo.round(rounding if rounding else self._rounding)

        # When there is no data found for any ticker, drop the row
        pvgo = pvgo.dropna(how="all", axis=0)

        return pvgo

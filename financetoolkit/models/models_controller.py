"""Models Module"""

__docformat__ = "google"

import pandas as pd

from financetoolkit.models import (
    altman_model,
    beneish_model,
    dupont_model,
    enterprise_model,
    eva_model,
    fulmer_model,
    grover_model,
    growth_model,
    helpers,
    intrinsic_model,
    ohlson_model,
    piotroski_model,
    springate_model,
    wacc_model,
    zmijewski_model,
)
from financetoolkit.performance.performance_model import get_beta
from financetoolkit.ratios import liquidity_model, profitability_model, valuation_model
from financetoolkit.utilities.dataframe_model import filter_columns
from financetoolkit.utilities.error_model import handle_errors
from financetoolkit.utilities.statistics_model import (
    calculate_growth,
    calculate_standardization,
    finalize_dataset,
)

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
        start_date: str | None = None,
        end_date: str | None = None,
    ):
        """
        Initializes the Models Controller Class.

        Args:
            tickers (str | list[str]): The ticker(s) to use for the models.
            historical_data (pd.DataFrame): The historical data containing all periods.
            risk_free_rate_data (pd.DataFrame): The risk free rate data.
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
        self._start_date: str | None = start_date
        self._end_date: str | None = end_date

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
        standardize: bool = False,
        trailing: int | None = None,
        show_columns: list[str] | None = None,
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

        Also known as: DuPont, ROE decomposition, three-factor DuPont.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the values. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
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

        dupont_analysis.loc["AAPL"]
        ```

        Which returns:

        |                   |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:------------------|-------:|-------:|-------:|-------:|-------:|
        | Net Profit Margin | 0.2588 | 0.2531 | 0.2531 | 0.2397 | 0.2692 |
        | Asset Turnover    | 1.0841 | 1.1206 | 1.0868 | 1.0899 | 1.1493 |
        | Equity Multiplier | 5.255  | 6.1862 | 6.252  | 6.0251 | 5.5418 |
        | Return on Equity  | 1.4744 | 1.7546 | 1.7195 | 1.5741 | 1.7142 |
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
                dataset=self._dupont_analysis,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="columns",
            )

        self._dupont_analysis = self._dupont_analysis.round(
            rounding if rounding else self._rounding
        )

        if standardize:
            standardize_rounding = rounding if rounding else self._rounding
            if growth:
                self._dupont_analysis_growth = calculate_standardization(
                    dataset=self._dupont_analysis_growth,
                    rounding=standardize_rounding,
                    axis="columns",
                )
            else:
                self._dupont_analysis = calculate_standardization(
                    dataset=self._dupont_analysis,
                    rounding=standardize_rounding,
                    axis="columns",
                )

        if len(self._tickers) == 1:
            result = (
                self._dupont_analysis_growth.droplevel(level=0)
                if growth
                else self._dupont_analysis.droplevel(level=0)
            )
            return filter_columns(
                result.loc[:, self._start_date : self._end_date], show_columns
            )

        result = self._dupont_analysis_growth if growth else self._dupont_analysis
        return filter_columns(
            result.loc[:, self._start_date : self._end_date], show_columns
        )

    @handle_errors
    def get_extended_dupont_analysis(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
        show_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Perform an Extended Dupont analysis to breakdown the return on equity (ROE) into its components,
        while considering additional financial metrics.

        The Extended Dupont analysis splits the three-factor decomposition's Net Profit Margin
        into three separate drivers — the Tax Burden, the Interest Burden and the Operating
        Profit Margin — so that the effect of taxation, of financing costs and of operating
        performance on the return on equity (ROE) can be read separately.

        The formula is as follows:

            - Interest Burden Ratio = Income Before Tax / Operating Income
            - Tax Burden Ratio = Net Income / Income Before Tax
            - Operating Profit Margin = Operating Income / Revenue
            - Asset Turnover = Revenue / Average Total Assets
            - Equity Multiplier = Average Total Assets / Average Total Equity
            - ROE = Interest Burden Ratio * Tax Burden Ratio * Operating Profit Margin *
            Asset Turnover * Equity Multiplier

        Also known as: extended DuPont, five-factor DuPont, ROE breakdown.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the values. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int | None, optional): The trailing period to use for the calculation. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame containing Extended Dupont analysis results, including the Interest
                        Burden Ratio, Tax Burden Ratio, Operating Profit Margin, Asset Turnover, Equity
                        Multiplier, and the calculated ROE values.

        Notes:
            - The Interest Burden Ratio (Income Before Tax / Operating Income) measures how much of
            operating profit survives the cost of debt financing. It equals 1 for a company with no
            net interest expense and falls as interest costs rise.
            - The Tax Burden Ratio (Net Income / Income Before Tax) measures how much of pre-tax
            profit survives taxation, i.e. it equals (1 - Effective Tax Rate).
            - The Operating Profit Margin measures operating performance before financing and tax
            effects.
            - Asset Turnover measures the efficiency of a company's use of its assets to generate
            sales revenue.
            - The Equity Multiplier represents the use of debt to finance a company's operations,
            which can amplify returns as well as risks.
            - Multiplying the first three components back together reproduces the Net Profit Margin
            of the three-factor `get_dupont_analysis`, so both decompositions resolve to the same ROE.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        extended_dupont_analysis = toolkit.models.get_extended_dupont_analysis()

        extended_dupont_analysis.loc["AAPL"]
        ```

        Which returns:

        |                         |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:------------------------|-------:|-------:|-------:|-------:|-------:|
        | Interest Burden Ratio   | 0.9976 | 1.0028 | 1.005  | 0.9978 | 1.0024 |
        | Tax Burden Ratio        | 0.869  | 0.8356 | 0.8486 | 0.7607 | 0.8419 |
        | Operating Profit Margin | 0.2985 | 0.302  | 0.2967 | 0.3158 | 0.3189 |
        | Asset Turnover          | 1.0841 | 1.1206 | 1.0868 | 1.0899 | 1.1493 |
        | Equity Multiplier       | 5.255  | 6.1862 | 6.252  | 6.0251 | 5.5418 |
        | Return on Equity        | 1.4744 | 1.7546 | 1.7195 | 1.5741 | 1.7142 |
        """
        if trailing:
            self._extended_dupont_analysis = dupont_model.get_extended_dupont_analysis(
                operating_income=self._income_statement.loc[:, "Operating Income", :]
                .T.rolling(trailing)
                .sum()
                .T,
                income_before_tax=self._income_statement.loc[:, "Income Before Tax", :]
                .T.rolling(trailing)
                .sum()
                .T,
                net_income=self._income_statement.loc[:, "Net Income", :]
                .T.rolling(trailing)
                .sum()
                .T,
                total_revenue=self._income_statement.loc[:, "Revenue", :]
                .T.rolling(trailing)
                .sum()
                .T,
                average_total_assets=self._balance_sheet_statement.loc[
                    :, "Total Assets", :
                ]
                .T.rolling(trailing)
                .mean()
                .T,
                average_total_equity=self._balance_sheet_statement.loc[
                    :, "Total Equity", :
                ]
                .T.rolling(trailing)
                .mean()
                .T,
            )
        else:
            self._extended_dupont_analysis = dupont_model.get_extended_dupont_analysis(
                operating_income=self._income_statement.loc[:, "Operating Income", :],
                income_before_tax=self._income_statement.loc[:, "Income Before Tax", :],
                net_income=self._income_statement.loc[:, "Net Income", :],
                total_revenue=self._income_statement.loc[:, "Revenue", :],
                average_total_assets=self._balance_sheet_statement.loc[
                    :, "Total Assets", :
                ]
                .T.rolling(2)
                .mean()
                .T,
                average_total_equity=self._balance_sheet_statement.loc[
                    :, "Total Equity", :
                ]
                .T.rolling(2)
                .mean()
                .T,
            )

        if growth:
            self._extended_dupont_analysis_growth = calculate_growth(
                dataset=self._extended_dupont_analysis,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="columns",
            )

        self._extended_dupont_analysis = self._extended_dupont_analysis.round(
            rounding if rounding else self._rounding
        )

        if standardize:
            standardize_rounding = rounding if rounding else self._rounding
            if growth:
                self._extended_dupont_analysis_growth = calculate_standardization(
                    dataset=self._extended_dupont_analysis_growth,
                    rounding=standardize_rounding,
                    axis="columns",
                )
            else:
                self._extended_dupont_analysis = calculate_standardization(
                    dataset=self._extended_dupont_analysis,
                    rounding=standardize_rounding,
                    axis="columns",
                )

        if len(self._tickers) == 1:
            result = (
                self._extended_dupont_analysis_growth.droplevel(level=0)
                if growth
                else self._extended_dupont_analysis.droplevel(level=0)
            )
            return filter_columns(
                result.loc[:, self._start_date : self._end_date], show_columns
            )

        result = (
            self._extended_dupont_analysis_growth
            if growth
            else self._extended_dupont_analysis
        )
        return filter_columns(
            result.loc[:, self._start_date : self._end_date], show_columns
        )

    @handle_errors
    def get_enterprise_value_breakdown(
        self,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Enterprise Value (EV) breakdown, providing a detailed view of its components.

        The Enterprise Value breakdown includes the following components for each quarter or year:

            - Share Price: The market price per share of the company's stock.
            - Market Capitalization (Market Cap): The total value of a company's outstanding common shares,
            i.e. the share price multiplied by the shares outstanding. Preferred shares are excluded here and
            enter as their own component below, so that they are counted once rather than twice.
            - Debt: The sum of long-term and short-term debt on the company's balance sheet.
            - Preferred Equity: The value of preferred shares, if applicable.
            - Minority Interest: The equity value of a subsidiary with less than 50% ownership.
            - Cash and Cash Equivalents: The total amount of liquid assets including cash, marketable securities,
            and short-term investments.

        The Enterprise Value is calculated as the sum of Market Cap, Debt, Preferred Equity,
        Minority Interest, minus Cash and Cash Equivalents.

        This breakdown is displayed in a DataFrame for each company and includes the option to show growth values as well.

        Also known as: EV breakdown, enterprise value components, EV bridge.

        Args:
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the values. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.DataFrame: DataFrame containing the Enterprise Value breakdown, including the calculated components.

        Notes:
        - All the inputs must be in the same currency and unit for accurate calculations.
        - The Enterprise Value is an important metric used for valuation and investment analysis.
        It represents the cost of acquiring the entire business: the equity is bought at its market
        value, the debt (and any preferred equity and minority interest) is assumed, and the acquired
        cash reduces the effective price, which is why cash is subtracted rather than added.
        - Enterprise Value is positive for essentially every going concern. It only turns negative
        when a company's cash exceeds its market capitalization plus its debt, which is a sign of a
        cash-rich balance sheet priced below its net cash — not of financial distress. A distressed,
        heavily indebted company shows the opposite: an Enterprise Value far above its market
        capitalization.
        - Understanding the Enterprise Value breakdown can provide insights into the sources of a
        company's value and potential risks.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        enterprise_value_breakdown = toolkit.models.get_enterprise_value_breakdown()

        enterprise_value_breakdown.loc["AAPL"]
        ```

        Which returns:

        |                           |          2021 |          2022 |          2023 |          2024 |          2025 |
        |:--------------------------|--------------:|--------------:|--------------:|--------------:|--------------:|
        | Share Price               | 177.57        | 129.93        | 192.53        | 250.42        | 271.86        |
        | Market Capitalization     |   2.9947e+12  |   2.12121e+12 |   3.04439e+12 |   3.8585e+12  |   4.07918e+12 |
        | Total Debt                |   1.36522e+11 |   1.3248e+11  |   1.2393e+11  |   1.19059e+11 |   1.12377e+11 |
        | Minority Interest         |   0           |   0           |   0           |   0           |   0           |
        | Preferred Equity          |   0           |   0           |   0           |   0           |   0           |
        | Cash and Cash Equivalents |   3.494e+10   |   2.3646e+10  |   2.9965e+10  |   2.9943e+10  |   3.5934e+10  |
        | Enterprise Value          |   3.09629e+12 |   2.23005e+12 |   3.13835e+12 |   3.94761e+12 |   4.15562e+12 |
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
                dataset=self._enterprise_value_breakdown,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        self._enterprise_value_breakdown = self._enterprise_value_breakdown.round(
            rounding if rounding else self._rounding
        )

        if standardize:
            standardize_rounding = rounding if rounding else self._rounding
            if growth:
                self._enterprise_value_breakdown_growth = calculate_standardization(
                    dataset=self._enterprise_value_breakdown_growth,
                    rounding=standardize_rounding,
                    axis="columns",
                )
            else:
                self._enterprise_value_breakdown = calculate_standardization(
                    dataset=self._enterprise_value_breakdown,
                    rounding=standardize_rounding,
                    axis="columns",
                )

        if len(self._tickers) == 1:
            result = (
                self._enterprise_value_breakdown_growth.droplevel(level=0)
                if growth
                else self._enterprise_value_breakdown.droplevel(level=0)
            )
            return filter_columns(
                result.loc[:, self._start_date : self._end_date], show_columns
            )

        result = (
            self._enterprise_value_breakdown_growth
            if growth
            else self._enterprise_value_breakdown
        )
        return filter_columns(
            result.loc[:, self._start_date : self._end_date], show_columns
        )

    @handle_errors
    def get_tobins_q_ratio(
        self,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
        show_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate Tobin's Q Ratio, a valuation metric developed by economist James Tobin that
        compares the market value of a company to the cost of replacing its assets.

        The formula is as follows:

            - Market Value of Equity = Share Price * Total Shares Outstanding
            - Tobin's Q Ratio = (Market Value of Equity + Total Liabilities) / Total Assets

        Tobin's Q Ratio can be interpreted as follows:

            - A Q ratio greater than 1 indicates that the market values the company above the
            cost of replacing its assets, which can reflect growth expectations, unrecognized
            intangible value, or overvaluation.
            - A Q ratio less than 1 indicates that the market values the company below the
            cost of replacing its assets, which can reflect undervaluation or declining growth
            prospects.

        Also known as: Tobin's Q, Q ratio.

        Args:
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to None.
            growth (bool, optional): Whether to calculate the growth of the values. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int | None, optional): The trailing period to use for the calculation. Defaults to None.
            show_columns (list[str] | None, optional): List of columns to show in the results. If None, all
                columns will be shown. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame containing the Market Value of Equity, Total Liabilities,
            Total Assets and Tobin's Q Ratio.

        Notes:
        - This implementation approximates the market value of debt with the book value of
        Total Liabilities, and the replacement cost of assets with the book value of Total
        Assets, consistent with the simplifications used in the Weighted Average Cost of
        Capital calculation elsewhere in this module.

        References:
        - Tobin, James. "A General Equilibrium Approach to Monetary Theory." Journal of Money,
        Credit and Banking, Vol. 1, No. 1, 1969, pp. 15-29.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "MSFT"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.models.get_tobins_q_ratio().loc["AAPL"]
        ```

        Which returns:

        |                        |        2021 |        2022 |
        |:-----------------------|------------:|------------:|
        | Market Value of Equity | 2.94327e+12 | 2.09689e+12 |
        | Total Liabilities      | 2.87912e+11 | 3.02083e+11 |
        | Total Assets           | 3.51002e+11 | 3.52755e+11 |
        | Tobin's Q Ratio        | 9.2056      | 6.8007      |
        """
        tobins_q_ratio = {}

        average_shares = (
            self._income_statement.loc[:, "Weighted Average Shares Diluted", :]
            if diluted
            else self._income_statement.loc[:, "Weighted Average Shares", :]
        )

        years = self._balance_sheet_statement.columns
        begin, end = str(years[0]), str(years[-1])

        share_prices = (
            self._historical_data["quarterly" if self._quarterly else "yearly"]
            .loc[begin:end, "Adj Close"][self._tickers]
            .T
        )

        total_liabilities = (
            self._balance_sheet_statement.loc[:, "Total Liabilities", :]
            .T.rolling(trailing)
            .mean()
            .T
            if trailing
            else self._balance_sheet_statement.loc[:, "Total Liabilities", :]
        )
        total_assets = (
            self._balance_sheet_statement.loc[:, "Total Assets", :]
            .T.rolling(trailing)
            .mean()
            .T
            if trailing
            else self._balance_sheet_statement.loc[:, "Total Assets", :]
        )

        market_cap = valuation_model.get_market_cap(
            share_price=share_prices,
            total_shares_outstanding=(
                average_shares.T.rolling(trailing).mean().T
                if trailing
                else average_shares
            ),
        )

        tobins_q_ratio["Market Value of Equity"] = market_cap
        tobins_q_ratio["Total Liabilities"] = total_liabilities
        tobins_q_ratio["Total Assets"] = total_assets
        tobins_q_ratio["Tobin's Q Ratio"] = enterprise_model.get_tobins_q_ratio(
            market_value_of_equity=market_cap,
            total_liabilities=total_liabilities,
            total_assets=total_assets,
        )

        tobins_q_results = (
            pd.concat(tobins_q_ratio)
            .dropna(axis=1, how="all")
            .swaplevel(0, 1)
            .reindex(self._tickers, level=0)
        )

        tobins_q_results = finalize_dataset(
            dataset=tobins_q_results,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="columns",
        )

        return filter_columns(tobins_q_results, show_columns)

    @handle_errors
    def get_weighted_average_cost_of_capital(
        self,
        show_full_results: bool = True,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
        show_columns: list[str] | None = None,
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

        Also known as: WACC, blended cost of capital, discount rate.

        Args:
            show_full_results (bool, optional): Whether to show the full results or just the WACC values.
            Defaults to True.
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the values. Defaults to False.
            lag (int | str, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int | None, optional): The trailing period to use for the calculation. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame containing the WACC values.

        Notes:
        - The Cost of Equity is approximated with the Capital Asset Pricing Model (CAPM).
        - The Market Value of Debt is approximated as the Total Debt.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.models.get_weighted_average_cost_of_capital().loc["AAPL"]
        ```

        Which returns:

        |                                  |        2021 |         2022 |        2023 |        2024 |        2025 |
        |:---------------------------------|------------:|-------------:|------------:|------------:|------------:|
        | Market Value Equity              | 2.9947e+12  |  2.12121e+12 | 3.04439e+12 | 3.8585e+12  | 4.07918e+12 |
        | Market Value Debt                | 1.36522e+11 |  1.3248e+11  | 1.2393e+11  | 1.19059e+11 | 1.12377e+11 |
        | Cost of Equity                   | 0.3494      | -0.2646      | 0.2633      | 0.2266      | 0.1938      |
        | Cost of Debt                     | 0.0194      |  0.0221      | 0.0317      | 0           | 0           |
        | Corporate Tax Rate               | 0.133       |  0.162       | 0.1472      | 0.2409      | 0.1561      |
        | Weighted Average Cost of Capital | 0.3349      | -0.248       | 0.2541      | 0.2198      | 0.1886      |
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

        if trailing:
            self._weighted_average_cost_of_capital = (
                wacc_model.get_weighted_average_cost_of_capital(
                    share_price=share_prices,
                    total_shares_outstanding=average_shares.T.rolling(trailing)
                    .mean()
                    .T,
                    interest_expense=self._income_statement.loc[
                        :, "Interest Expense", :
                    ]
                    .T.rolling(trailing)
                    .sum()
                    .T,
                    total_debt=self._balance_sheet_statement.loc[:, "Total Debt", :]
                    .T.rolling(trailing)
                    .mean()
                    .T,
                    risk_free_rate=risk_free_rate,
                    beta=beta,
                    benchmark_returns=benchmark_returns,
                    income_tax_expense=self._income_statement.loc[
                        :, "Income Tax Expense", :
                    ]
                    .T.rolling(trailing)
                    .sum()
                    .T,
                    income_before_tax=self._income_statement.loc[
                        :, "Income Before Tax", :
                    ]
                    .T.rolling(trailing)
                    .sum()
                    .T,
                )
            )
        else:
            self._weighted_average_cost_of_capital = (
                wacc_model.get_weighted_average_cost_of_capital(
                    share_price=share_prices,
                    total_shares_outstanding=average_shares,
                    interest_expense=self._income_statement.loc[
                        :, "Interest Expense", :
                    ],
                    total_debt=self._balance_sheet_statement.loc[:, "Total Debt", :],
                    risk_free_rate=risk_free_rate,
                    beta=beta,
                    benchmark_returns=benchmark_returns,
                    income_tax_expense=self._income_statement.loc[
                        :, "Income Tax Expense", :
                    ],
                    income_before_tax=self._income_statement.loc[
                        :, "Income Before Tax", :
                    ],
                )
            )

        if growth:
            self._weighted_average_cost_of_capital_growth = calculate_growth(
                dataset=self._weighted_average_cost_of_capital,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="columns",
            )

        self._weighted_average_cost_of_capital = (
            self._weighted_average_cost_of_capital.round(
                rounding if rounding else self._rounding
            )
        )

        if standardize:
            standardize_rounding = rounding if rounding else self._rounding
            if growth:
                self._weighted_average_cost_of_capital_growth = (
                    calculate_standardization(
                        dataset=self._weighted_average_cost_of_capital_growth,
                        rounding=standardize_rounding,
                        axis="columns",
                    )
                )
            else:
                self._weighted_average_cost_of_capital = calculate_standardization(
                    dataset=self._weighted_average_cost_of_capital,
                    rounding=standardize_rounding,
                    axis="columns",
                )

        if len(self._tickers) == 1 and show_full_results:
            result = (
                self._weighted_average_cost_of_capital_growth.droplevel(level=0)
                if growth
                else self._weighted_average_cost_of_capital.droplevel(level=0)
            )
            return filter_columns(
                result.loc[:, self._start_date : self._end_date], show_columns
            )

        if show_full_results:
            result = (
                self._weighted_average_cost_of_capital_growth
                if growth
                else self._weighted_average_cost_of_capital
            )
            return filter_columns(
                result.loc[:, self._start_date : self._end_date], show_columns
            )

        result = (
            self._weighted_average_cost_of_capital_growth.loc[
                :, "Weighted Average Cost of Capital", :
            ]
            if growth
            else self._weighted_average_cost_of_capital.loc[
                :, "Weighted Average Cost of Capital", :
            ]
        )
        return filter_columns(
            result.loc[:, self._start_date : self._end_date], show_columns
        )

    @handle_errors
    def get_economic_value_added(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
        show_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Economic Value Added (EVA) is a measure of a company's financial performance that
        represents the value created in excess of the required return of the company's capital
        providers. It captures whether a company is generating returns above its true cost of
        capital, which distinguishes it from purely accounting-based measures such as Net Income.

        The formula is as follows:

            - NOPAT = EBIT * (1 - Effective Tax Rate)
            - Invested Capital = Total Equity + Total Debt
            - EVA = NOPAT - (Weighted Average Cost of Capital * Invested Capital)

        Also known as: EVA, economic profit.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to None.
            growth (bool, optional): Whether to calculate the growth of the values. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int | None, optional): The trailing period to use for the calculation. Defaults to None.
            show_columns (list[str] | None, optional): List of columns to show in the results. If None, all
                columns will be shown. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame containing the EVA and its components.

        Notes:
        - A positive EVA indicates that the company is generating returns in excess of its cost of
        capital, i.e. it is creating value for its capital providers. A negative EVA indicates the
        company is destroying value.
        - EBIT is approximated as Net Income + Income Tax Expense + Interest Expense, consistent
        with the Altman Z-Score calculation elsewhere in this module.
        - Invested Capital is the sum of the two-period average of Total Equity and the two-period
        average of Total Debt, i.e. the capital employed over the course of the period rather than
        its closing balance, consistent with the Return on Invested Capital calculation in the
        Ratios module.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "MSFT"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.models.get_economic_value_added().loc["AAPL"]
        ```

        Which returns:

        |                                   |        2021 |          2022 |
        |:----------------------------------|-------------:|-------------:|
        | Net Operating Profit After Taxes  |  9.69732e+10 |  1.02259e+11 |
        | Invested Capital                  |  1.93614e+11 |  1.91382e+11 |
        | Weighted Average Cost of Capital  |       0.3598 |      -0.2326 |
        | Economic Value Added              |  2.73107e+10 |  1.46775e+11 |
        """
        eva = {}

        net_income = (
            self._income_statement.loc[:, "Net Income", :].T.rolling(trailing).sum().T
            if trailing
            else self._income_statement.loc[:, "Net Income", :]
        )
        income_tax_expense = (
            self._income_statement.loc[:, "Income Tax Expense", :]
            .T.rolling(trailing)
            .sum()
            .T
            if trailing
            else self._income_statement.loc[:, "Income Tax Expense", :]
        )
        interest_expense = (
            self._income_statement.loc[:, "Interest Expense", :]
            .T.rolling(trailing)
            .sum()
            .T
            if trailing
            else self._income_statement.loc[:, "Interest Expense", :]
        )
        income_before_tax = (
            self._income_statement.loc[:, "Income Before Tax", :]
            .T.rolling(trailing)
            .sum()
            .T
            if trailing
            else self._income_statement.loc[:, "Income Before Tax", :]
        )
        total_equity = (
            self._balance_sheet_statement.loc[:, "Total Equity", :]
            .T.rolling(trailing)
            .mean()
            .T
            if trailing
            else self._balance_sheet_statement.loc[:, "Total Equity", :]
            .T.rolling(2)
            .mean()
            .T
        )
        total_debt = (
            self._balance_sheet_statement.loc[:, "Total Debt", :]
            .T.rolling(trailing)
            .mean()
            .T
            if trailing
            else self._balance_sheet_statement.loc[:, "Total Debt", :]
            .T.rolling(2)
            .mean()
            .T
        )

        ebit = net_income + income_tax_expense + interest_expense
        effective_tax_rate = profitability_model.get_effective_tax_rate(
            income_tax_expense=income_tax_expense,
            income_before_tax=income_before_tax,
        )

        eva["Net Operating Profit After Taxes"] = (
            eva_model.get_net_operating_profit_after_taxes(
                ebit=ebit,
                effective_tax_rate=effective_tax_rate,
            )
        )

        eva["Invested Capital"] = eva_model.get_invested_capital(
            total_equity=total_equity,
            total_debt=total_debt,
        )

        eva["Weighted Average Cost of Capital"] = (
            self.get_weighted_average_cost_of_capital(
                show_full_results=False, trailing=trailing
            )
        )

        eva["Economic Value Added"] = eva_model.get_economic_value_added(
            net_operating_profit_after_taxes=eva["Net Operating Profit After Taxes"],
            weighted_average_cost_of_capital=eva["Weighted Average Cost of Capital"],
            invested_capital=eva["Invested Capital"],
        )

        eva_results = (
            pd.concat(eva)
            .dropna(axis=1, how="all")
            .swaplevel(0, 1)
            .reindex(self._tickers, level=0)
        )

        eva_results = finalize_dataset(
            dataset=eva_results,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="columns",
        )

        return filter_columns(eva_results, show_columns)

    @handle_errors
    def get_market_value_added(
        self,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
        show_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Market Value Added (MVA) is a measure of a company's financial performance that
        represents the difference between the current market value of a company (both its
        equity and its debt) and the total capital that has historically been invested in it.
        It is the market-priced counterpart to Economic Value Added (EVA): where EVA measures
        a single period's excess return over the cost of capital, MVA reflects the market's
        cumulative, forward-looking verdict on all of a company's expected future EVA.

        The formula is as follows:

            - Market Value of Equity = Share Price * Total Shares Outstanding
            - Invested Capital = Total Equity + Total Debt
            - MVA = (Market Value of Equity + Market Value of Debt) - Invested Capital

        Also known as: MVA, market value added.

        Args:
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to None.
            growth (bool, optional): Whether to calculate the growth of the values. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int | None, optional): The trailing period to use for the calculation. Defaults to None.
            show_columns (list[str] | None, optional): List of columns to show in the results. If None, all
                columns will be shown. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame containing the MVA and its components.

        Notes:
        - A positive MVA indicates that the market believes management has created value in
        excess of the capital invested in the company. A negative MVA indicates the market
        values the company below the capital that has historically been invested in it.
        - The Market Value of Debt is approximated as the closing book value of Total Debt,
        the same simplification used in the Weighted Average Cost of Capital calculation
        elsewhere in this module.
        - Invested Capital is the sum of the two-period average of Total Equity and the
        two-period average of Total Debt, i.e. the capital that was employed *during* the
        period, matching the Economic Value Added calculation elsewhere in this module. Note
        that this leaves the two sides of the MVA measured on slightly different bases: the
        market value is a closing (point-in-time) figure while the invested capital is an
        average over the period.

        References:
        - Stern, Joel M., G. Bennett Stewart, and Donald H. Chew. "The EVA Financial
        Management System." Journal of Applied Corporate Finance, Vol. 8, No. 2, 1995, pp. 32-46.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "MSFT"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.models.get_market_value_added().loc["AAPL"]
        ```

        Which returns:

        |                        |        2021 |        2022 |        2023 |        2024 |        2025 |
        |:-----------------------|------------:|------------:|------------:|------------:|------------:|
        | Market Value of Equity | 2.92522e+12 | 2.08399e+12 | 3.00786e+12 | 3.83091e+12 | 4.06822e+12 |
        | Market Value of Debt   | 1.36522e+11 | 1.3248e+11  | 1.2393e+11  | 1.19059e+11 | 1.12377e+11 |
        | Invested Capital       | 1.93614e+11 | 1.91382e+11 | 1.84614e+11 | 1.81042e+11 | 1.8106e+11  |
        | Market Value Added     | 2.86813e+12 | 2.02509e+12 | 2.94718e+12 | 3.76893e+12 | 3.99954e+12 |
        """
        mva = {}

        years = self._cash_flow_statement.columns
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

        total_equity = (
            self._balance_sheet_statement.loc[:, "Total Equity", :]
            .T.rolling(trailing)
            .mean()
            .T
            if trailing
            else self._balance_sheet_statement.loc[:, "Total Equity", :]
            .T.rolling(2)
            .mean()
            .T
        )
        total_debt = (
            self._balance_sheet_statement.loc[:, "Total Debt", :]
            .T.rolling(trailing)
            .mean()
            .T
            if trailing
            else self._balance_sheet_statement.loc[:, "Total Debt", :]
            .T.rolling(2)
            .mean()
            .T
        )

        market_cap = valuation_model.get_market_cap(
            share_price=share_prices,
            total_shares_outstanding=(
                average_shares.T.rolling(trailing).mean().T
                if trailing
                else average_shares
            ),
        )

        # A market value is a point-in-time quantity, so the closing balance of Total Debt is
        # used as its proxy — not the averaged balance that Invested Capital is built from,
        # which would otherwise cancel the debt out of the Market Value Added entirely
        market_value_of_debt = (
            self._balance_sheet_statement.loc[:, "Total Debt", :]
            .T.rolling(trailing)
            .mean()
            .T
            if trailing
            else self._balance_sheet_statement.loc[:, "Total Debt", :]
        )

        mva["Market Value of Equity"] = market_cap
        mva["Market Value of Debt"] = market_value_of_debt

        mva["Invested Capital"] = eva_model.get_invested_capital(
            total_equity=total_equity,
            total_debt=total_debt,
        )

        mva["Market Value Added"] = eva_model.get_market_value_added(
            market_value_of_equity=mva["Market Value of Equity"],
            market_value_of_debt=mva["Market Value of Debt"],
            invested_capital=mva["Invested Capital"],
        )

        mva_results = (
            pd.concat(mva)
            .dropna(axis=1, how="all")
            .swaplevel(0, 1)
            .reindex(self._tickers, level=0)
        )

        mva_results = finalize_dataset(
            dataset=mva_results,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="columns",
        )

        return filter_columns(mva_results, show_columns)

    def get_intrinsic_valuation(
        self,
        growth_rate: float | list | dict[str, float],
        perpetual_growth_rate: float | list | dict[str, float],
        weighted_average_cost_of_capital: float | list | dict[str, float],
        periods: int = 5,
        cash_flow_type: str = "Free Cash Flow",
        trailing: int | None = None,
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

        Also known as: DCF, discounted cash flow, fair value, intrinsic value.

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
            trailing (int | None, optional): The number of trailing periods to sum for the base cash flow.
            When set, uses the sum of the last N periods instead of only the most recent period. Defaults to None.
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

        toolkit.models.get_intrinsic_valuation(0.05, 0.025, 0.094).loc["AAPL"]
        ```

        Which returns:

        |                      |   Periods = 5 |
        |:---------------------|--------------:|
        | Terminal Value       |   1.87255e+12 |
        | Cash Flow Projection |   1.9986e+12  |
        | Enterprise Value     |   1.58232e+12 |
        | Equity Value         |   1.50588e+12 |
        | Intrinsic Value      | 100.36        |
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
            cash_flow_series = self._cash_flow_statement.loc[
                ticker, cash_flow_type
            ].dropna()

            intrinsic_values_dict[ticker] = intrinsic_model.get_intrinsic_value(
                cash_flow=(
                    cash_flow_series.iloc[-trailing:].sum()
                    if trailing
                    else cash_flow_series.iloc[-1]
                ),
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

        return self._intrinsic_values.loc[:, self._start_date :]

    @handle_errors
    def get_free_cash_flow_to_firm(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
        show_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Free Cash Flow to the Firm (FCFF) is the cash flow available to all providers of
        capital, both debt and equity holders, after the company has paid all of its
        operating expenses and invested in the assets needed to sustain its operations.
        Because it is measured before any financing cash flows, FCFF is capital-structure
        neutral, making it the cash flow base typically discounted at the Weighted Average
        Cost of Capital (WACC) when valuing the Enterprise Value of a company directly.

        The formula is as follows:

            - NOPAT = EBIT * (1 - Effective Tax Rate)
            - FCFF = NOPAT + Depreciation and Amortization - Capital Expenditure -
            Change in Net Working Capital

        Also known as: FCFF, unlevered free cash flow.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to None.
            growth (bool, optional): Whether to calculate the growth of the values. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int | None, optional): The trailing period to use for the calculation. Defaults to None.
            show_columns (list[str] | None, optional): List of columns to show in the results. If None, all
                columns will be shown. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame containing the FCFF and its components.

        Notes:
        - EBIT is approximated as Net Income + Income Tax Expense + Interest Expense,
        consistent with the Altman Z-Score and Economic Value Added calculations elsewhere
        in this module.
        - The Capital Expenditure and Change in Working Capital line items from the cash flow
        statement are stored using a cash-flow-impact sign convention (a use of cash is
        negative). This method negates them internally so that they represent positive
        magnitudes (amount spent / amount of the increase), matching the standard academic
        FCFF formula. See `get_free_cash_flow_to_firm` in the Intrinsic module for details.
        - FCFF can be used together with the Weighted Average Cost of Capital in
        `get_intrinsic_valuation` (set `cash_flow_type` appropriately, or supply this
        result as a custom base cash flow) as an alternative to the reported Free Cash Flow.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "MSFT"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.models.get_free_cash_flow_to_firm().loc["AAPL"]
        ```

        Which returns:

        |                                   |        2021 |          2022 |
        |:----------------------------------|-------------:|-------------:|
        | Net Operating Profit After Taxes  |  9.69732e+10 |  1.02259e+11 |
        | Depreciation and Amortization     |  1.1284e+10  |  1.1104e+10  |
        | Capital Expenditure               |  1.1085e+10  |  1.0708e+10  |
        | Change in Net Working Capital     |  4.911e+09   | -1.2e+09     |
        | Free Cash Flow to Firm            |  9.22612e+10 |  1.03855e+11 |
        """
        fcff = {}

        net_income = (
            self._income_statement.loc[:, "Net Income", :].T.rolling(trailing).sum().T
            if trailing
            else self._income_statement.loc[:, "Net Income", :]
        )
        income_tax_expense = (
            self._income_statement.loc[:, "Income Tax Expense", :]
            .T.rolling(trailing)
            .sum()
            .T
            if trailing
            else self._income_statement.loc[:, "Income Tax Expense", :]
        )
        interest_expense = (
            self._income_statement.loc[:, "Interest Expense", :]
            .T.rolling(trailing)
            .sum()
            .T
            if trailing
            else self._income_statement.loc[:, "Interest Expense", :]
        )
        income_before_tax = (
            self._income_statement.loc[:, "Income Before Tax", :]
            .T.rolling(trailing)
            .sum()
            .T
            if trailing
            else self._income_statement.loc[:, "Income Before Tax", :]
        )
        depreciation_and_amortization = (
            self._income_statement.loc[:, "Depreciation and Amortization", :]
            .T.rolling(trailing)
            .sum()
            .T
            if trailing
            else self._income_statement.loc[:, "Depreciation and Amortization", :]
        )
        capital_expenditure = -1 * (
            self._cash_flow_statement.loc[:, "Capital Expenditure", :]
            .T.rolling(trailing)
            .sum()
            .T
            if trailing
            else self._cash_flow_statement.loc[:, "Capital Expenditure", :]
        )
        change_in_net_working_capital = -1 * (
            self._cash_flow_statement.loc[:, "Change in Working Capital", :]
            .T.rolling(trailing)
            .sum()
            .T
            if trailing
            else self._cash_flow_statement.loc[:, "Change in Working Capital", :]
        )

        ebit = net_income + income_tax_expense + interest_expense
        effective_tax_rate = profitability_model.get_effective_tax_rate(
            income_tax_expense=income_tax_expense,
            income_before_tax=income_before_tax,
        )

        fcff["Net Operating Profit After Taxes"] = (
            eva_model.get_net_operating_profit_after_taxes(
                ebit=ebit,
                effective_tax_rate=effective_tax_rate,
            )
        )
        fcff["Depreciation and Amortization"] = depreciation_and_amortization
        fcff["Capital Expenditure"] = capital_expenditure
        fcff["Change in Net Working Capital"] = change_in_net_working_capital

        fcff["Free Cash Flow to Firm"] = intrinsic_model.get_free_cash_flow_to_firm(
            net_operating_profit_after_taxes=fcff["Net Operating Profit After Taxes"],
            depreciation_and_amortization=fcff["Depreciation and Amortization"],
            capital_expenditure=fcff["Capital Expenditure"],
            change_in_net_working_capital=fcff["Change in Net Working Capital"],
        )

        fcff_results = (
            pd.concat(fcff)
            .dropna(axis=1, how="all")
            .swaplevel(0, 1)
            .reindex(self._tickers, level=0)
        )

        fcff_results = finalize_dataset(
            dataset=fcff_results,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="columns",
        )

        return filter_columns(fcff_results, show_columns)

    @handle_errors
    def get_free_cash_flow_to_equity(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
        show_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Free Cash Flow to Equity (FCFE) is the cash flow available to a company's common
        equity holders after all operating expenses, reinvestment needs, and net payments to
        (or from) debt holders have been accounted for. Unlike FCFF, FCFE is a levered cash
        flow measure and should be discounted at the Cost of Equity, not the Weighted Average
        Cost of Capital, when used to value equity directly.

        The formula is as follows:

            - FCFE = Net Income + Depreciation and Amortization - Capital Expenditure -
            Change in Net Working Capital + Net Borrowing

        Also known as: FCFE, levered free cash flow.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to None.
            growth (bool, optional): Whether to calculate the growth of the values. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int | None, optional): The trailing period to use for the calculation. Defaults to None.
            show_columns (list[str] | None, optional): List of columns to show in the results. If None, all
                columns will be shown. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame containing the FCFE and its components.

        Notes:
        - The Capital Expenditure and Change in Working Capital line items from the cash flow
        statement are stored using a cash-flow-impact sign convention (a use of cash is
        negative). This method negates them internally so that they represent positive
        magnitudes, matching the standard academic FCFE formula. Net Debt Issued (used as
        Net Borrowing) is already reported using the cash-flow-impact convention (a positive
        value means the company was a net borrower) and is therefore used as-is.
        - FCFE is the natural cash-flow-based counterpart to Residual Income
        (`get_residual_income`) for equity-only valuation, in the same way FCFF is the
        counterpart to Economic Value Added for firm-level valuation.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "MSFT"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.models.get_free_cash_flow_to_equity().loc["AAPL"]
        ```

        Which returns:

        |                                   |        2021 |          2022 |
        |:----------------------------------|-------------:|-------------:|
        | Net Income                        |  9.4680e+10  |  9.9803e+10  |
        | Depreciation and Amortization     |  1.1284e+10  |  1.1104e+10  |
        | Capital Expenditure               |  1.1085e+10  |  1.0708e+10  |
        | Change in Net Working Capital     |  4.911e+09   | -1.2e+09     |
        | Net Borrowing                     |  1.1643e+10  | -4.078e+09   |
        | Free Cash Flow to Equity          |  1.01611e+11 |  9.7321e+10  |
        """
        fcfe = {}

        net_income = (
            self._income_statement.loc[:, "Net Income", :].T.rolling(trailing).sum().T
            if trailing
            else self._income_statement.loc[:, "Net Income", :]
        )
        depreciation_and_amortization = (
            self._income_statement.loc[:, "Depreciation and Amortization", :]
            .T.rolling(trailing)
            .sum()
            .T
            if trailing
            else self._income_statement.loc[:, "Depreciation and Amortization", :]
        )
        capital_expenditure = -1 * (
            self._cash_flow_statement.loc[:, "Capital Expenditure", :]
            .T.rolling(trailing)
            .sum()
            .T
            if trailing
            else self._cash_flow_statement.loc[:, "Capital Expenditure", :]
        )
        change_in_net_working_capital = -1 * (
            self._cash_flow_statement.loc[:, "Change in Working Capital", :]
            .T.rolling(trailing)
            .sum()
            .T
            if trailing
            else self._cash_flow_statement.loc[:, "Change in Working Capital", :]
        )
        net_borrowing = (
            self._cash_flow_statement.loc[:, "Net Debt Issued", :]
            .T.rolling(trailing)
            .sum()
            .T
            if trailing
            else self._cash_flow_statement.loc[:, "Net Debt Issued", :]
        )

        fcfe["Net Income"] = net_income
        fcfe["Depreciation and Amortization"] = depreciation_and_amortization
        fcfe["Capital Expenditure"] = capital_expenditure
        fcfe["Change in Net Working Capital"] = change_in_net_working_capital
        fcfe["Net Borrowing"] = net_borrowing

        fcfe["Free Cash Flow to Equity"] = intrinsic_model.get_free_cash_flow_to_equity(
            net_income=fcfe["Net Income"],
            depreciation_and_amortization=fcfe["Depreciation and Amortization"],
            capital_expenditure=fcfe["Capital Expenditure"],
            change_in_net_working_capital=fcfe["Change in Net Working Capital"],
            net_borrowing=fcfe["Net Borrowing"],
        )

        fcfe_results = (
            pd.concat(fcfe)
            .dropna(axis=1, how="all")
            .swaplevel(0, 1)
            .reindex(self._tickers, level=0)
        )

        fcfe_results = finalize_dataset(
            dataset=fcfe_results,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="columns",
        )

        return filter_columns(fcfe_results, show_columns)

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

        Also known as: Gordon Growth Model, DDM, dividend discount model, GGM.

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
        - Each historical period is valued off that period's *actual* Dividends per Share, and only the
        periods beyond the last one available are projected forward at the given growth rate. The first
        period of the historical window will therefore often be understated, since it only covers the part
        of the year that falls inside the requested date range and so captures only part of the year's
        dividends. Use a start_date at least one full period before the first period you intend to read.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "MSFT"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.models.get_gorden_growth_model(0.20, 0.05)
        ```

        Which returns:

        |      |   AAPL |    MSFT |
        |:-----|-------:|--------:|
        | 2021 | 1.54   |  8.26   |
        | 2022 | 6.37   | 17.78   |
        | 2023 | 6.65   | 19.53   |
        | 2024 | 6.93   | 21.56   |
        | 2025 | 7.21   | 23.8    |
        | 2026 | 7.5705 | 24.99   |
        | 2027 | 7.949  | 26.2395 |
        | 2028 | 8.3465 | 27.5515 |
        | 2029 | 8.7638 | 28.929  |
        | 2030 | 9.202  | 30.3755 |
        | 2031 | 9.6621 | 31.8943 |
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

        return gorden_growth_model_df.loc[self._start_date :]

    def get_two_stage_dividend_discount_model(
        self,
        rate_of_return: float | list | dict[str, float],
        high_growth_rate: float | list | dict[str, float],
        stable_growth_rate: float | list | dict[str, float],
        high_growth_periods: int = 5,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        The Two-Stage Dividend Discount Model extends the (single-stage) Gordon Growth Model
        to companies that are not expected to grow at a constant rate forever. It explicitly
        projects and discounts dividends over an initial high-growth phase, and then values
        everything from the end of that phase onward as a perpetuity growing at a lower,
        more sustainable stable rate (a Gordon Growth terminal value).

        The formula is as follows:

            - Dividend Projection_t = Dividends Per Share * (1 + High Growth Rate)^t
            - High-Growth Phase Present Value = Sum of Dividend Projection_t / (1 + Rate of
            Return)^t, for t = 1, ..., High Growth Periods
            - Terminal Value = Last Dividend Projection * (1 + Stable Growth Rate) /
            (Rate of Return - Stable Growth Rate)
            - Intrinsic Value = High-Growth Phase Present Value +
            (Terminal Value / (1 + Rate of Return)^High Growth Periods)

        Also known as: two-stage DDM, two-stage dividend discount model.

        Args:
            rate_of_return (float, list or dict): The required rate of return (discount rate)
            used to discount both phases. Can be one number to use for all tickers, or a list
            or dict that contains a rate of return for each ticker.
            high_growth_rate (float, list or dict): The constant growth rate applied to
            dividends during the explicit high-growth phase. Can be one number to use for all
            tickers, or a list or dict that contains a high growth rate for each ticker.
            stable_growth_rate (float, list or dict): The perpetual (terminal) growth rate
            applied to dividends from the end of the high-growth phase onward. Can be one
            number to use for all tickers, or a list or dict that contains a stable growth
            rate for each ticker.
            high_growth_periods (int, optional): The number of periods in the explicit
            high-growth phase. Defaults to 5.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.

        Returns:
            pd.DataFrame: DataFrame containing the intrinsic value and its components for
            each ticker.

        Notes:
        - The results are highly dependent on the input. Therefore, think carefully about
        each input parameter to ensure the results are accurate (given your beliefs).
        - The Rate of Return must be greater than the Stable Growth Rate, otherwise the
        Terminal Value formula divides by a non-positive number.
        - The base dividend used for the projection is the most recent available Dividends
        Per Share for each ticker. For companies that do not pay a dividend, this model is
        not meaningful (the projections and resulting intrinsic value will be zero).

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "MSFT"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.models.get_two_stage_dividend_discount_model(0.10, 0.12, 0.03).loc["AAPL"]
        ```

        Which returns:

        |                                   |   High-Growth Periods = 5 |
        |:----------------------------------|---------------------------:|
        | Final High-Growth Dividend        |                     1.6037 |
        | High-Growth Phase Present Value   |                     4.8043 |
        | Terminal Value                    |                    23.5978 |
        | Terminal Value Present Value      |                    14.6523 |
        | Intrinsic Value                   |                    19.4566 |
        """
        dividends_per_share = self._historical_data[
            "quarterly" if self._quarterly else "yearly"
        ]["Dividends"]

        if isinstance(rate_of_return, list):
            if len(rate_of_return) != len(self._tickers):
                raise ValueError(
                    "The length of the rate of return list must match the number of tickers."
                )
            rate_of_return_dict = {
                ticker: rate_of_return[i] for i, ticker in enumerate(self._tickers)
            }
        elif isinstance(rate_of_return, dict):
            if len(rate_of_return) != len(self._tickers):
                raise ValueError(
                    "The length of the rate of return dict must match the number of tickers."
                )
            rate_of_return_dict = rate_of_return
        else:
            rate_of_return_dict = {}

        if isinstance(high_growth_rate, list):
            if len(high_growth_rate) != len(self._tickers):
                raise ValueError(
                    "The length of the high growth rate list must match the number of tickers."
                )
            high_growth_rate_dict = {
                ticker: high_growth_rate[i] for i, ticker in enumerate(self._tickers)
            }
        elif isinstance(high_growth_rate, dict):
            if len(high_growth_rate) != len(self._tickers):
                raise ValueError(
                    "The length of the high growth rate dict must match the number of tickers."
                )
            high_growth_rate_dict = high_growth_rate
        else:
            high_growth_rate_dict = {}

        if isinstance(stable_growth_rate, list):
            if len(stable_growth_rate) != len(self._tickers):
                raise ValueError(
                    "The length of the stable growth rate list must match the number of tickers."
                )
            stable_growth_rate_dict = {
                ticker: stable_growth_rate[i] for i, ticker in enumerate(self._tickers)
            }
        elif isinstance(stable_growth_rate, dict):
            if len(stable_growth_rate) != len(self._tickers):
                raise ValueError(
                    "The length of the stable growth rate dict must match the number of tickers."
                )
            stable_growth_rate_dict = stable_growth_rate
        else:
            stable_growth_rate_dict = {}

        two_stage_ddm_dict = {}
        for ticker in self._tickers:
            rate_of_return_float = rate_of_return_dict.get(ticker, rate_of_return)
            high_growth_rate_float = high_growth_rate_dict.get(ticker, high_growth_rate)
            stable_growth_rate_float = stable_growth_rate_dict.get(
                ticker, stable_growth_rate
            )
            dividends_per_share_series = dividends_per_share[ticker].dropna()
            base_dividend = (
                dividends_per_share_series.iloc[-1]
                if not dividends_per_share_series.empty
                else 0.0
            )

            two_stage_ddm_dict[ticker] = (
                intrinsic_model.get_two_stage_dividend_discount_model(
                    dividends_per_share=base_dividend,
                    rate_of_return=rate_of_return_float,
                    high_growth_rate=high_growth_rate_float,
                    stable_growth_rate=stable_growth_rate_float,
                    high_growth_periods=high_growth_periods,
                )
            )
            two_stage_ddm_dict[ticker] = two_stage_ddm_dict[ticker].round(
                rounding if rounding else self._rounding
            )

        two_stage_ddm_results = pd.concat(two_stage_ddm_dict)

        return two_stage_ddm_results.loc[:, self._start_date :]

    @handle_errors
    def get_residual_income(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
        show_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Residual Income is a measure of the profit a company generates in excess of the
        return required by its equity holders. It is the equity-side counterpart to Economic
        Value Added (EVA), and underpins the Residual Income Model, an alternative equity
        valuation lens to a traditional Discounted Cash Flow (DCF) that is particularly
        useful when a company's free cash flows are negative or unpredictable but its
        accounting earnings are more stable.

        The formula is as follows:

            - Residual Income = Net Income - (Cost of Equity * Book Value of Equity)

        Also known as: RI, economic profit (equity variant), abnormal earnings.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to None.
            growth (bool, optional): Whether to calculate the growth of the values. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int | None, optional): The trailing period to use for the calculation. Defaults to None.
            show_columns (list[str] | None, optional): List of columns to show in the results. If None, all
                columns will be shown. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame containing the Residual Income and its components.

        Notes:
        - A positive Residual Income indicates that the company generated more profit than
        equity holders required given the capital they have invested. A negative Residual
        Income indicates the company failed to earn its equity holders' required return, even
        if it still reported a positive Net Income.
        - The Cost of Equity is approximated with the Capital Asset Pricing Model (CAPM),
        consistent with the Weighted Average Cost of Capital calculation elsewhere in this
        module.
        - The Book Value of Equity is approximated as the average of Total Shareholder Equity
        minus Preferred Stock over the current and prior period, excluding preferred equity
        since Residual Income (like Net Income) belongs to common equity holders only.

        References:
        - Ohlson, James A. "Earnings, Book Values, and Dividends in Equity Valuation."
        Contemporary Accounting Research, Vol. 11, No. 2, 1995, pp. 661-687.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "MSFT"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.models.get_residual_income().loc["AAPL"]
        ```

        Which returns:

        |                          |        2021 |          2022 |
        |:-------------------------|-------------:|-------------:|
        | Net Income               |  9.4680e+10  |  9.9803e+10  |
        | Cost of Equity           |       0.3757 |      -0.2485 |
        | Book Value of Equity     |  6.4215e+10  |  5.6881e+10  |
        | Residual Income          |  7.0554e+10  |  1.1394e+11  |
        """
        residual_income = {}

        net_income = (
            self._income_statement.loc[:, "Net Income", :].T.rolling(trailing).sum().T
            if trailing
            else self._income_statement.loc[:, "Net Income", :]
        )
        total_shareholder_equity = (
            self._balance_sheet_statement.loc[:, "Total Shareholder Equity", :]
            .T.rolling(trailing)
            .mean()
            .T
            if trailing
            else self._balance_sheet_statement.loc[:, "Total Shareholder Equity", :]
            .T.rolling(2)
            .mean()
            .T
        )
        preferred_stock = (
            self._balance_sheet_statement.loc[:, "Preferred Stock", :]
            .T.rolling(trailing)
            .mean()
            .T
            if trailing
            else self._balance_sheet_statement.loc[:, "Preferred Stock", :]
            .T.rolling(2)
            .mean()
            .T
        )

        years = self._balance_sheet_statement.columns
        begin, end = str(years[0]), str(years[-1])

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

        cost_of_equity = wacc_model.get_cost_of_equity(
            risk_free_rate=risk_free_rate,
            beta=beta,
            benchmark_returns=benchmark_returns,
        ).T

        residual_income["Net Income"] = net_income
        residual_income["Cost of Equity"] = cost_of_equity
        residual_income["Book Value of Equity"] = (
            total_shareholder_equity - preferred_stock
        )

        residual_income["Residual Income"] = intrinsic_model.get_residual_income(
            net_income=residual_income["Net Income"],
            cost_of_equity=residual_income["Cost of Equity"],
            book_value_of_equity=residual_income["Book Value of Equity"],
        )

        residual_income_results = (
            pd.concat(residual_income)
            .dropna(axis=1, how="all")
            .swaplevel(0, 1)
            .reindex(self._tickers, level=0)
        )

        residual_income_results = finalize_dataset(
            dataset=residual_income_results,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="columns",
        )

        return filter_columns(residual_income_results, show_columns)

    @handle_errors
    def get_altman_z_score(
        self,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
        show_columns: list[str] | None = None,
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

        Also known as: Altman Z-score, bankruptcy prediction, financial distress score.

        Args:
            diluted (bool, optional): Whether to use diluted shares outstanding in the calculation. Defaults to True.
            rounding (int, optional): The number of decimals to round the results to. Defaults to None.
            growth (bool, optional): Whether to calculate the growth of the values. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int | None, optional): The trailing period to use for the calculation. Defaults to None.

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

        altman_z_score.loc["AAPL"]
        ```

        Which returns:

        |                                   |    2021 |    2022 |    2023 |    2024 |    2025 |
        |:----------------------------------|--------:|--------:|--------:|--------:|--------:|
        | Working Capital to Total Assets   |  0.0267 | -0.0527 | -0.0049 | -0.0641 | -0.0492 |
        | Retained Earnings to Total Assets |  0.0158 | -0.0087 | -0.0006 | -0.0525 | -0.0397 |
        | EBIT to Total Assets              |  0.3187 |  0.3459 |  0.3337 |  0.3383 |  0.3695 |
        | Market Value to Total Liabilities | 10.4015 |  7.022  | 10.4821 | 12.5264 | 14.2874 |
        | Sales to Total Assets             |  1.0422 |  1.1179 |  1.0871 |  1.0714 |  1.1584 |
        | Altman Z-Score                    |  8.3888 |  6.3973 |  8.4709 |  9.5533 | 10.8355 |
        """
        altman_z_score = {}

        current_assets = (
            self._balance_sheet_statement.loc[:, "Total Current Assets", :]
            .T.rolling(trailing)
            .mean()
            .T
            if trailing
            else self._balance_sheet_statement.loc[:, "Total Current Assets", :]
        )
        current_liabilities = (
            self._balance_sheet_statement.loc[:, "Total Current Liabilities", :]
            .T.rolling(trailing)
            .mean()
            .T
            if trailing
            else self._balance_sheet_statement.loc[:, "Total Current Liabilities", :]
        )
        total_assets = (
            self._balance_sheet_statement.loc[:, "Total Assets", :]
            .T.rolling(trailing)
            .mean()
            .T
            if trailing
            else self._balance_sheet_statement.loc[:, "Total Assets", :]
        )
        retained_earnings = (
            self._balance_sheet_statement.loc[:, "Retained Earnings", :]
            .T.rolling(trailing)
            .mean()
            .T
            if trailing
            else self._balance_sheet_statement.loc[:, "Retained Earnings", :]
        )
        total_liabilities = (
            self._balance_sheet_statement.loc[:, "Total Liabilities", :]
            .T.rolling(trailing)
            .mean()
            .T
            if trailing
            else self._balance_sheet_statement.loc[:, "Total Liabilities", :]
        )
        net_income = (
            self._income_statement.loc[:, "Net Income", :].T.rolling(trailing).sum().T
            if trailing
            else self._income_statement.loc[:, "Net Income", :]
        )
        income_tax_expense = (
            self._income_statement.loc[:, "Income Tax Expense", :]
            .T.rolling(trailing)
            .sum()
            .T
            if trailing
            else self._income_statement.loc[:, "Income Tax Expense", :]
        )
        interest_expense = (
            self._income_statement.loc[:, "Interest Expense", :]
            .T.rolling(trailing)
            .sum()
            .T
            if trailing
            else self._income_statement.loc[:, "Interest Expense", :]
        )
        revenue = (
            self._income_statement.loc[:, "Revenue", :].T.rolling(trailing).sum().T
            if trailing
            else self._income_statement.loc[:, "Revenue", :]
        )

        working_capital = liquidity_model.get_working_capital(
            current_assets,
            current_liabilities,
        )

        altman_z_score["Working Capital to Total Assets"] = (
            altman_model.get_working_capital_to_total_assets_ratio(
                working_capital=working_capital,
                total_assets=total_assets,
            )
        )

        altman_z_score["Retained Earnings to Total Assets"] = (
            altman_model.get_retained_earnings_to_total_assets_ratio(
                retained_earnings=retained_earnings,
                total_assets=total_assets,
            )
        )

        altman_z_score["EBIT to Total Assets"] = (
            altman_model.get_earnings_before_interest_and_taxes_to_total_assets_ratio(
                ebit=(net_income + income_tax_expense + interest_expense),
                total_assets=total_assets,
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
            share_price=share_prices,
            total_shares_outstanding=(
                average_shares.T.rolling(trailing).mean().T
                if trailing
                else average_shares
            ),
        )

        altman_z_score["Market Value to Total Liabilities"] = (
            altman_model.get_market_value_of_equity_to_book_value_of_total_liabilities_ratio(
                market_value_of_equity=market_cap,
                total_liabilities=total_liabilities,
            )
        )

        altman_z_score["Sales to Total Assets"] = (
            altman_model.get_sales_to_total_assets_ratio(
                sales=revenue,
                total_assets=total_assets,
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

        altman_results = finalize_dataset(
            dataset=altman_results,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="columns",
        )

        return filter_columns(altman_results, show_columns)

    @handle_errors
    def get_piotroski_score(
        self,
        trailing: int | None = None,
        show_columns: list[str] | None = None,
    ) -> pd.DataFrame:
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
        Information to Separate Winners from Losers." Journal of Accounting Research, Vol. 38,
        Supplement, 2000, pp. 1-41.

        Also known as: Piotroski F-score, financial strength, quality score.

        Args:
            trailing (int | None, optional): The trailing period to use for the calculation. Defaults to None.
            show_columns (list[str] | None, optional): List of columns to show in the results. If None, all columns
                will be shown. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame containing the Piotroski F-Score and its components.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA", "MSFT"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.models.get_piotroski_score().loc["AAPL"]
        ```

        Which returns:

        |                                     |   2022 |   2023 |   2024 |   2025 |   2026 |
        |:------------------------------------|-------:|-------:|-------:|-------:|-------:|
        | Return on Assets Criteria           |      1 |      1 |      1 |      1 |    nan |
        | Operating Cashflow Criteria         |      1 |      1 |      1 |      1 |    nan |
        | Change in Return on Assets Criteria |      1 |      0 |      0 |      1 |    nan |
        | Accruals Criteria                   |      1 |      1 |      1 |      0 |    nan |
        | Change in Leverage Criteria         |      1 |      1 |      1 |      1 |    nan |
        | Change in Current Ratio Criteria    |      0 |      1 |      0 |      1 |    nan |
        | Number of Shares Criteria           |      1 |      1 |      1 |      1 |    nan |
        | Gross Margin Criteria               |      1 |      1 |      1 |      1 |    nan |
        | Asset Turnover Criteria             |      1 |      0 |      1 |      1 |    nan |
        | Piotroski Score                     |      8 |      7 |      7 |      8 |    nan |

        Periods for which the financial statements have not been reported yet are returned as NaN
        rather than being scored zero across the board.
        """
        piotroski_score = {}

        total_assets = (
            self._balance_sheet_statement.loc[:, "Total Assets", :]
            .T.rolling(trailing)
            .mean()
            .T
            if trailing
            else self._balance_sheet_statement.loc[:, "Total Assets", :]
        )
        net_income = (
            self._income_statement.loc[:, "Net Income", :].T.rolling(trailing).sum().T
            if trailing
            else self._income_statement.loc[:, "Net Income", :]
        )
        operating_cashflow = (
            self._cash_flow_statement.loc[:, "Operating Cash Flow", :]
            .T.rolling(trailing)
            .sum()
            .T
            if trailing
            else self._cash_flow_statement.loc[:, "Operating Cash Flow", :]
        )
        revenue = (
            self._income_statement.loc[:, "Revenue", :].T.rolling(trailing).sum().T
            if trailing
            else self._income_statement.loc[:, "Revenue", :]
        )
        cost_of_goods_sold = (
            self._income_statement.loc[:, "Cost of Goods Sold", :]
            .T.rolling(trailing)
            .sum()
            .T
            if trailing
            else self._income_statement.loc[:, "Cost of Goods Sold", :]
        )
        common_stock_issued = (
            self._cash_flow_statement.loc[:, "Common Stock Issued", :]
            .T.rolling(trailing)
            .sum()
            .T
            if trailing
            else self._cash_flow_statement.loc[:, "Common Stock Issued", :]
        )
        long_term_debt = (
            self._balance_sheet_statement.loc[:, "Long Term Debt", :]
            .T.rolling(trailing)
            .mean()
            .T
            if trailing
            else self._balance_sheet_statement.loc[:, "Long Term Debt", :]
        )
        current_assets = (
            self._balance_sheet_statement.loc[:, "Total Current Assets", :]
            .T.rolling(trailing)
            .mean()
            .T
            if trailing
            else self._balance_sheet_statement.loc[:, "Total Current Assets", :]
        )
        current_liabilities = (
            self._balance_sheet_statement.loc[:, "Total Current Liabilities", :]
            .T.rolling(trailing)
            .mean()
            .T
            if trailing
            else self._balance_sheet_statement.loc[:, "Total Current Liabilities", :]
        )

        average_total_assets = total_assets.T.rolling(2).mean().T

        piotroski_score["Return on Assets Criteria"] = (
            piotroski_model.get_return_on_assets_criteria(
                net_income=net_income,
                average_total_assets=average_total_assets,
            )
        )

        piotroski_score["Operating Cashflow Criteria"] = (
            piotroski_model.get_operating_cashflow_criteria(
                operating_cashflow=operating_cashflow,
            )
        )

        piotroski_score["Change in Return on Assets Criteria"] = (
            piotroski_model.get_change_in_return_on_asset_criteria(
                net_income=net_income,
                average_total_assets=average_total_assets,
            )
        )

        # Piotroski scales both the Return on Assets and the Cash Flow from Operations by the
        # same asset base, which is what makes this signal reduce to the sign of accruals
        piotroski_score["Accruals Criteria"] = piotroski_model.get_accruals_criteria(
            net_income=net_income,
            average_total_assets=average_total_assets,
            operating_cashflow=operating_cashflow,
            total_assets=average_total_assets,
        )

        piotroski_score["Change in Leverage Criteria"] = (
            piotroski_model.get_change_in_leverage_criteria(
                long_term_debt=long_term_debt,
                average_total_assets=average_total_assets,
            )
        )

        piotroski_score["Change in Current Ratio Criteria"] = (
            piotroski_model.get_change_in_current_ratio_criteria(
                current_assets=current_assets,
                current_liabilities=current_liabilities,
            )
        )

        piotroski_score["Number of Shares Criteria"] = (
            piotroski_model.get_number_of_shares_criteria(
                common_stock_issued=common_stock_issued,
            )
        )

        piotroski_score["Gross Margin Criteria"] = (
            piotroski_model.get_gross_margin_criteria(
                revenue=revenue,
                cost_of_goods_sold=cost_of_goods_sold,
            )
        )

        piotroski_score["Asset Turnover Criteria"] = (
            piotroski_model.get_asset_turnover_ratio_criteria(
                sales=revenue,
                average_total_assets=average_total_assets,
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
            pd.concat(piotroski_score).swaplevel(0, 1).reindex(self._tickers, level=0)
        )

        # Every criterion is a boolean comparison, so a period for which the statements have
        # not been reported yet compares NaN against a number, evaluates to False and scores
        # a perfect zero — the worst possible F-Score — rather than being reported as missing.
        # Mask those periods out explicitly before dropping them.
        reported = (
            total_assets.notna() & net_income.notna() & operating_cashflow.notna()
        )
        reported = (
            reported.reindex(
                index=piotroski_results.index.get_level_values(0),
                columns=piotroski_results.columns,
            )
            .fillna(False)
            .astype(bool)
            .set_axis(piotroski_results.index)
        )

        piotroski_results = piotroski_results.astype(float).where(reported)

        piotroski_results = piotroski_results.dropna(axis=1, how="all")

        # Two columns are lost rather than one. The averaged total assets already make
        # the first period NaN, so the criteria that compare against the previous period
        # only become meaningful from the third period onwards. Keeping the second
        # column would silently score those criteria 0 even when they improved.
        piotroski_results = piotroski_results[piotroski_results.columns[2:]]

        # The F-Score is a count of satisfied criteria, so report it as such whenever every
        # remaining period is fully reported and the float cast above is no longer needed
        if not piotroski_results.isna().to_numpy().any():
            piotroski_results = piotroski_results.astype(int)

        return filter_columns(
            piotroski_results.loc[:, self._start_date : self._end_date], show_columns
        )

    @handle_errors
    def get_beneish_m_score(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
        show_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        The Beneish M-Score is a probabilistic model, developed by Messod Beneish, that uses eight
        financial ratios derived from a company's financial statements to identify whether a company
        has manipulated its earnings. It is a natural companion to the Altman Z-Score and Piotroski
        F-Score, using the same normalized financial statements as its input.

        The formula is as follows:

            M-Score = -4.84 + 0.92 * DSRI + 0.528 * GMI + 0.404 * AQI + 0.892 * SGI + 0.115 * DEPI
            - 0.172 * SGAI + 4.679 * TATA - 0.327 * LVGI

        The eight variables are:

            - DSRI: Days Sales in Receivables Index
            - GMI: Gross Margin Index
            - AQI: Asset Quality Index
            - SGI: Sales Growth Index
            - DEPI: Depreciation Index
            - SGAI: Selling, General and Administrative Expenses Index
            - TATA: Total Accruals to Total Assets
            - LVGI: Leverage Index

        Also known as: Beneish M-Score, earnings manipulation score.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to None.
            growth (bool, optional): Whether to calculate the growth of the values. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int | None, optional): The trailing period to use for the calculation. Defaults to None.
            show_columns (list[str] | None, optional): List of columns to show in the results. If None, all
                columns will be shown. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame containing the Beneish M-Score and its components.

        Notes:
        - A M-Score greater than -1.78 suggests that the company is likely to be an earnings
        manipulator. A M-Score lower than -1.78 suggests the company is unlikely to be a manipulator.
        - As with the Altman Z-Score and Piotroski F-Score, this is a probabilistic, not a
        definitive, indicator and should be combined with further fundamental analysis.
        - Every component compares the current period to the prior period, so the very first
        period in the results will always be NaN.

        References:
        - Beneish, Messod D. "The Detection of Earnings Manipulation." Financial Analysts Journal,
        Vol. 55, No. 5, 1999, pp. 24-36.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "MSFT"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.models.get_beneish_m_score().loc["AAPL"]
        ```

        Which returns:

        |                                  |    2021 |    2022 |    2023 |
        |:---------------------------------|--------:|--------:|--------:|
        | Days Sales in Receivables Index  |  1.0322 |  1.0975 |  1.0297 |
        | Gross Margin Index               |  0.9151 |  0.9647 |  0.9814 |
        | Asset Quality Index              |  1.1404 |  0.9841 |  0.9387 |
        | Sales Growth Index               |  1.3326 |  1.0779 |  0.972  |
        | Depreciation Index               |  1.0566 |  1.0635 |  0.9982 |
        | SGA Expenses Index               |  0.8279 |  1.0595 |  1.0222 |
        | Leverage Index                   |  1.0608 |  1.0729 |  0.9516 |
        | Total Accruals to Total Assets   | -0.0267 | -0.0634 | -0.0384 |
        | Beneish M-Score                  | -2.2503 | -2.6691 | -2.6802 |
        """
        beneish_m_score = {}

        net_receivables = (
            self._balance_sheet_statement.loc[:, "Net Receivables", :]
            .T.rolling(trailing)
            .mean()
            .T
            if trailing
            else self._balance_sheet_statement.loc[:, "Net Receivables", :]
        )
        revenue = (
            self._income_statement.loc[:, "Revenue", :].T.rolling(trailing).sum().T
            if trailing
            else self._income_statement.loc[:, "Revenue", :]
        )
        cost_of_goods_sold = (
            self._income_statement.loc[:, "Cost of Goods Sold", :]
            .T.rolling(trailing)
            .sum()
            .T
            if trailing
            else self._income_statement.loc[:, "Cost of Goods Sold", :]
        )
        total_current_assets = (
            self._balance_sheet_statement.loc[:, "Total Current Assets", :]
            .T.rolling(trailing)
            .mean()
            .T
            if trailing
            else self._balance_sheet_statement.loc[:, "Total Current Assets", :]
        )
        property_plant_and_equipment = (
            self._balance_sheet_statement.loc[:, "Property, Plant and Equipment", :]
            .T.rolling(trailing)
            .mean()
            .T
            if trailing
            else self._balance_sheet_statement.loc[
                :, "Property, Plant and Equipment", :
            ]
        )
        total_assets = (
            self._balance_sheet_statement.loc[:, "Total Assets", :]
            .T.rolling(trailing)
            .mean()
            .T
            if trailing
            else self._balance_sheet_statement.loc[:, "Total Assets", :]
        )
        depreciation_and_amortization = (
            self._income_statement.loc[:, "Depreciation and Amortization", :]
            .T.rolling(trailing)
            .sum()
            .T
            if trailing
            else self._income_statement.loc[:, "Depreciation and Amortization", :]
        )
        selling_general_and_administrative_expenses = (
            self._income_statement.loc[
                :, "Selling, General and Administrative Expenses", :
            ]
            .T.rolling(trailing)
            .sum()
            .T
            if trailing
            else self._income_statement.loc[
                :, "Selling, General and Administrative Expenses", :
            ]
        )
        total_current_liabilities = (
            self._balance_sheet_statement.loc[:, "Total Current Liabilities", :]
            .T.rolling(trailing)
            .mean()
            .T
            if trailing
            else self._balance_sheet_statement.loc[:, "Total Current Liabilities", :]
        )
        long_term_debt = (
            self._balance_sheet_statement.loc[:, "Long Term Debt", :]
            .T.rolling(trailing)
            .mean()
            .T
            if trailing
            else self._balance_sheet_statement.loc[:, "Long Term Debt", :]
        )
        net_income = (
            self._income_statement.loc[:, "Net Income", :].T.rolling(trailing).sum().T
            if trailing
            else self._income_statement.loc[:, "Net Income", :]
        )
        cash_flow_from_operations = (
            self._cash_flow_statement.loc[:, "Operating Cash Flow", :]
            .T.rolling(trailing)
            .sum()
            .T
            if trailing
            else self._cash_flow_statement.loc[:, "Operating Cash Flow", :]
        )

        beneish_m_score["Days Sales in Receivables Index"] = (
            beneish_model.get_days_sales_in_receivables_index(
                net_receivables=net_receivables,
                revenue=revenue,
            )
        )

        beneish_m_score["Gross Margin Index"] = beneish_model.get_gross_margin_index(
            revenue=revenue,
            cost_of_goods_sold=cost_of_goods_sold,
        )

        beneish_m_score["Asset Quality Index"] = beneish_model.get_asset_quality_index(
            total_current_assets=total_current_assets,
            property_plant_and_equipment=property_plant_and_equipment,
            total_assets=total_assets,
        )

        beneish_m_score["Sales Growth Index"] = beneish_model.get_sales_growth_index(
            revenue=revenue,
        )

        beneish_m_score["Depreciation Index"] = beneish_model.get_depreciation_index(
            depreciation_and_amortization=depreciation_and_amortization,
            property_plant_and_equipment=property_plant_and_equipment,
        )

        beneish_m_score["SGA Expenses Index"] = (
            beneish_model.get_selling_general_and_administrative_expenses_index(
                selling_general_and_administrative_expenses=(
                    selling_general_and_administrative_expenses
                ),
                revenue=revenue,
            )
        )

        beneish_m_score["Leverage Index"] = beneish_model.get_leverage_index(
            total_current_liabilities=total_current_liabilities,
            long_term_debt=long_term_debt,
            total_assets=total_assets,
        )

        beneish_m_score["Total Accruals to Total Assets"] = (
            beneish_model.get_total_accruals_to_total_assets(
                net_income=net_income,
                cash_flow_from_operations=cash_flow_from_operations,
                total_assets=total_assets,
            )
        )

        beneish_m_score["Beneish M-Score"] = beneish_model.get_beneish_m_score(
            days_sales_in_receivables_index=beneish_m_score[
                "Days Sales in Receivables Index"
            ],
            gross_margin_index=beneish_m_score["Gross Margin Index"],
            asset_quality_index=beneish_m_score["Asset Quality Index"],
            sales_growth_index=beneish_m_score["Sales Growth Index"],
            depreciation_index=beneish_m_score["Depreciation Index"],
            selling_general_and_administrative_expenses_index=beneish_m_score[
                "SGA Expenses Index"
            ],
            leverage_index=beneish_m_score["Leverage Index"],
            total_accruals_to_total_assets=beneish_m_score[
                "Total Accruals to Total Assets"
            ],
        )

        beneish_results = (
            pd.concat(beneish_m_score)
            .dropna(axis=1, how="all")
            .swaplevel(0, 1)
            .reindex(self._tickers, level=0)
        )

        beneish_results = finalize_dataset(
            dataset=beneish_results,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="columns",
        )

        return filter_columns(beneish_results, show_columns)

    @handle_errors
    def get_ohlson_o_score(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
        show_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculates the Ohlson O-Score, a financial metric used to predict the likelihood of a company
        going bankrupt. Unlike the Altman Z-Score, which is built with multiple discriminant analysis,
        the O-Score's coefficients come from a fitted logistic regression (logit) model. This is why
        the two models are usually reported side by side rather than one being treated as a replacement
        for the other: the Z-Score is only meaningful compared against Altman's empirically derived
        threshold bands, while the O-Score is directly interpretable as a probability of bankruptcy
        once passed through the logistic transform.

        The formula is as follows:

            - SIZE = ln(Total Assets)
            - TLTA = Total Liabilities / Total Assets
            - WCTA = Working Capital / Total Assets
            - CLCA = Current Liabilities / Current Assets
            - OENEG = 1 if Total Liabilities > Total Assets else 0
            - NITA = Net Income / Total Assets
            - FUTL = Operating Cash Flow / Total Liabilities
            - INTWO = 1 if Net Income was negative for the last two years else 0
            - CHIN = (Net Income (t) - Net Income (t-1)) / (|Net Income (t)| + |Net Income (t-1)|)
            - O-Score = -1.32 - 0.407 * SIZE + 6.03 * TLTA - 1.43 * WCTA + 0.0757 * CLCA
            - 1.72 * OENEG - 2.37 * NITA - 1.83 * FUTL + 0.285 * INTWO - 0.521 * CHIN
            - Bankruptcy Probability = 1 / (1 + e^(-O-Score))

        The Ohlson O-Score can be interpreted as follows:

            - Ohlson's (1980) original cutoff is a bankruptcy probability of 0.038 (3.8%), the
            threshold that minimized the sum of Type I and Type II misclassification errors on his
            sample. It is deliberately far below the naive 0.50 midpoint because bankruptcy is a
            rare event. Equivalently, in raw O-Score terms the cutoff sits at ln(0.038 / 0.962),
            i.e. approximately -3.23.
            - A higher probability indicates a higher likelihood of bankruptcy.

        Also known as: Ohlson O-Score, bankruptcy prediction, financial distress score.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to None.
            growth (bool, optional): Whether to calculate the growth of the values. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int | None, optional): The trailing period to use for the calculation. Defaults to None.
            show_columns (list[str] | None, optional): List of columns to show in the results. If None, all
                columns will be shown. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame containing the Ohlson O-Score, its bankruptcy probability, and its
            components.

        Notes:
            - Ohlson's (1980) original SIZE term deflates Total Assets by the US GNP price-level index
            (rebased to 1968), a US-specific 1970s macro series that isn't available in this toolkit and
            isn't meaningful outside the original US sample. This implementation simplifies/omits that
            deflator and uses ln(Total Assets) in nominal terms instead. This shifts the O-Score (and
            resulting probability) by roughly a constant amount across all observations in a given
            currency/period — it does not change the *ranking* of companies relative to one another, but
            the absolute probability estimate should not be compared directly to studies that apply the
            deflator.
            - Ohlson's original FUTL term uses "Funds from Operations", an accounting-flow concept that
            predates standardized cash flow statements. This implementation approximates Funds from
            Operations with Operating Cash Flow, the standard simplification used in modern
            reproductions of the O-Score.
            - The Beneish M-Score, Altman Z-Score, Piotroski F-Score and Ohlson O-Score are natural
            companions, using the same normalized financial statements as their input.
            - As with the other bankruptcy and distress models, this is a probabilistic, not a
            definitive, indicator and should be combined with further fundamental analysis.

        References:
        - Ohlson, James A. "Financial Ratios and the Probabilistic Prediction of Bankruptcy." Journal of
        Accounting Research, Vol. 18, No. 1, 1980, pp. 109-131.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "MSFT"], api_key="FINANCIAL_MODELING_PREP_KEY")

        ohlson_o_score = toolkit.models.get_ohlson_o_score()

        ohlson_o_score.loc["AAPL"]
        ```

        Which returns:

        |                                             |     2020 |     2021 |     2022 |     2023 |
        |:--------------------------------------------|---------:|---------:|---------:|---------:|
        | Log of Total Assets                         |  26.5037 |  26.5841 |  26.589  |  26.5886 |
        | Total Liabilities to Total Assets            |   0.7983 |   0.8203 |   0.8564 |   0.8237 |
        | Working Capital to Total Assets              |   0.1183 |   0.0267 |  -0.0527 |  -0.0049 |
        | Current Liabilities to Current Assets        |   0.7334 |   0.9306 |   1.1372 |   1.0121 |
        | Negative Equity Indicator                    |   0      |   0      |   0      |   0      |
        | Net Income to Total Assets                   |   0.1773 |   0.2697 |   0.2829 |   0.2751 |
        | Funds from Operations to Total Liabilities   |   0.312  |   0.3614 |   0.4044 |   0.3806 |
        | Negative Income Indicator                    |   0      |   0      |   0      |   0      |
        | Change in Net Income                         | nan      |   0.245  |   0.0263 |  -0.0143 |
        | Ohlson O-Score                               | nan      |  -8.5895 |  -8.2408 |  -8.4318 |
        | Ohlson Bankruptcy Probability                | nan      |   0.0002 |   0.0003 |   0.0002 |

        Note that the first period is NaN because the Change in Net Income and Negative Income
        Indicator components require a prior period to compare against.
        """
        ohlson_o_score = {}

        current_assets = (
            self._balance_sheet_statement.loc[:, "Total Current Assets", :]
            .T.rolling(trailing)
            .mean()
            .T
            if trailing
            else self._balance_sheet_statement.loc[:, "Total Current Assets", :]
        )
        current_liabilities = (
            self._balance_sheet_statement.loc[:, "Total Current Liabilities", :]
            .T.rolling(trailing)
            .mean()
            .T
            if trailing
            else self._balance_sheet_statement.loc[:, "Total Current Liabilities", :]
        )
        total_assets = (
            self._balance_sheet_statement.loc[:, "Total Assets", :]
            .T.rolling(trailing)
            .mean()
            .T
            if trailing
            else self._balance_sheet_statement.loc[:, "Total Assets", :]
        )
        total_liabilities = (
            self._balance_sheet_statement.loc[:, "Total Liabilities", :]
            .T.rolling(trailing)
            .mean()
            .T
            if trailing
            else self._balance_sheet_statement.loc[:, "Total Liabilities", :]
        )
        net_income = (
            self._income_statement.loc[:, "Net Income", :].T.rolling(trailing).sum().T
            if trailing
            else self._income_statement.loc[:, "Net Income", :]
        )
        operating_cash_flow = (
            self._cash_flow_statement.loc[:, "Operating Cash Flow", :]
            .T.rolling(trailing)
            .sum()
            .T
            if trailing
            else self._cash_flow_statement.loc[:, "Operating Cash Flow", :]
        )

        working_capital = liquidity_model.get_working_capital(
            current_assets,
            current_liabilities,
        )

        ohlson_o_score["Log of Total Assets"] = ohlson_model.get_log_of_total_assets(
            total_assets=total_assets,
        )

        ohlson_o_score["Total Liabilities to Total Assets"] = (
            ohlson_model.get_total_liabilities_to_total_assets_ratio(
                total_liabilities=total_liabilities,
                total_assets=total_assets,
            )
        )

        ohlson_o_score["Working Capital to Total Assets"] = (
            ohlson_model.get_working_capital_to_total_assets_ratio(
                working_capital=working_capital,
                total_assets=total_assets,
            )
        )

        ohlson_o_score["Current Liabilities to Current Assets"] = (
            ohlson_model.get_current_liabilities_to_current_assets_ratio(
                current_liabilities=current_liabilities,
                current_assets=current_assets,
            )
        )

        ohlson_o_score["Negative Equity Indicator"] = (
            ohlson_model.get_negative_equity_indicator(
                total_liabilities=total_liabilities,
                total_assets=total_assets,
            )
        )

        ohlson_o_score["Net Income to Total Assets"] = (
            ohlson_model.get_net_income_to_total_assets_ratio(
                net_income=net_income,
                total_assets=total_assets,
            )
        )

        ohlson_o_score["Funds from Operations to Total Liabilities"] = (
            ohlson_model.get_funds_from_operations_to_total_liabilities_ratio(
                funds_from_operations=operating_cash_flow,
                total_liabilities=total_liabilities,
            )
        )

        ohlson_o_score["Negative Income Indicator"] = (
            ohlson_model.get_negative_income_indicator(
                net_income=net_income,
            )
        )

        ohlson_o_score["Change in Net Income"] = (
            ohlson_model.get_change_in_net_income_ratio(
                net_income=net_income,
            )
        )

        ohlson_o_score["Ohlson O-Score"] = ohlson_model.get_ohlson_o_score(
            log_of_total_assets=ohlson_o_score["Log of Total Assets"],
            total_liabilities_to_total_assets_ratio=ohlson_o_score[
                "Total Liabilities to Total Assets"
            ],
            working_capital_to_total_assets_ratio=ohlson_o_score[
                "Working Capital to Total Assets"
            ],
            current_liabilities_to_current_assets_ratio=ohlson_o_score[
                "Current Liabilities to Current Assets"
            ],
            negative_equity_indicator=ohlson_o_score["Negative Equity Indicator"],
            net_income_to_total_assets_ratio=ohlson_o_score[
                "Net Income to Total Assets"
            ],
            funds_from_operations_to_total_liabilities_ratio=ohlson_o_score[
                "Funds from Operations to Total Liabilities"
            ],
            negative_income_indicator=ohlson_o_score["Negative Income Indicator"],
            change_in_net_income_ratio=ohlson_o_score["Change in Net Income"],
        )

        ohlson_o_score["Ohlson Bankruptcy Probability"] = (
            ohlson_model.get_ohlson_bankruptcy_probability(
                ohlson_o_score=ohlson_o_score["Ohlson O-Score"],
            )
        )

        ohlson_results = (
            pd.concat(ohlson_o_score)
            .dropna(axis=1, how="all")
            .swaplevel(0, 1)
            .reindex(self._tickers, level=0)
        )

        ohlson_results = finalize_dataset(
            dataset=ohlson_results,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="columns",
        )

        return filter_columns(ohlson_results, show_columns)

    @handle_errors
    def get_zmijewski_score(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
        show_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculates the Zmijewski Score, a financial metric used to predict the likelihood of a company
        going bankrupt. It is a simpler, three-variable model compared to both the Altman Z-Score and
        the Ohlson O-Score, and is commonly used as a robustness check when those two models disagree,
        since it is estimated on a different sample and with a different statistical technique (a
        probit model rather than discriminant analysis or logistic regression).

        The formula is as follows:

            - Net Income to Total Assets = Net Income / Total Assets
            - Total Liabilities to Total Assets = Total Liabilities / Total Assets
            - Current Assets to Current Liabilities = Current Assets / Current Liabilities
            - X = -4.3 - 4.5 * Net Income to Total Assets + 5.7 * Total Liabilities to Total Assets
            - 0.004 * Current Assets to Current Liabilities
            - Bankruptcy Probability = Phi(X), the standard normal cumulative distribution function

        The Zmijewski Score can be interpreted as follows:

            - A higher X (and therefore a higher bankruptcy probability) indicates a higher likelihood
            of financial distress.

        Also known as: Zmijewski Score, ZFC score, bankruptcy prediction, financial distress score.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to None.
            growth (bool, optional): Whether to calculate the growth of the values. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int | None, optional): The trailing period to use for the calculation. Defaults to None.
            show_columns (list[str] | None, optional): List of columns to show in the results. If None, all
                columns will be shown. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame containing the Zmijewski Score, its bankruptcy probability, and its
            components.

        Notes:
            - Unlike the Ohlson O-Score, which uses a logistic (sigmoid) link to convert its score into
            a probability, the Zmijewski Score uses a probit link (the standard normal CDF). Applying a
            logistic transform to the Zmijewski Score instead would produce an incorrect probability.
            - As with the Altman Z-Score, Piotroski F-Score, Beneish M-Score and Ohlson O-Score, this is
            a probabilistic, not a definitive, indicator and should be combined with further fundamental
            analysis.

        References:
        - Zmijewski, Mark E. "Methodological Issues Related to the Estimation of Financial Distress
        Prediction Models." Journal of Accounting Research, Vol. 22, 1984, pp. 59-82.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "MSFT"], api_key="FINANCIAL_MODELING_PREP_KEY")

        zmijewski_score = toolkit.models.get_zmijewski_score()

        zmijewski_score.loc["AAPL"]
        ```

        Which returns:

        |                                          |    2020 |    2021 |    2022 |    2023 |
        |:-----------------------------------------|--------:|--------:|--------:|--------:|
        | Net Income to Total Assets               |  0.1773 |  0.2697 |  0.2829 |  0.2751 |
        | Total Liabilities to Total Assets         |  0.7983 |  0.8203 |  0.8564 |  0.8237 |
        | Current Assets to Current Liabilities     |  1.3636 |  1.0746 |  0.8794 |  0.988  |
        | Zmijewski Score                           | -0.553  | -0.8427 | -0.6955 | -0.8466 |
        | Zmijewski Bankruptcy Probability          |  0.2901 |  0.1997 |  0.2434 |  0.1986 |
        """
        zmijewski_score = {}

        current_assets = (
            self._balance_sheet_statement.loc[:, "Total Current Assets", :]
            .T.rolling(trailing)
            .mean()
            .T
            if trailing
            else self._balance_sheet_statement.loc[:, "Total Current Assets", :]
        )
        current_liabilities = (
            self._balance_sheet_statement.loc[:, "Total Current Liabilities", :]
            .T.rolling(trailing)
            .mean()
            .T
            if trailing
            else self._balance_sheet_statement.loc[:, "Total Current Liabilities", :]
        )
        total_assets = (
            self._balance_sheet_statement.loc[:, "Total Assets", :]
            .T.rolling(trailing)
            .mean()
            .T
            if trailing
            else self._balance_sheet_statement.loc[:, "Total Assets", :]
        )
        total_liabilities = (
            self._balance_sheet_statement.loc[:, "Total Liabilities", :]
            .T.rolling(trailing)
            .mean()
            .T
            if trailing
            else self._balance_sheet_statement.loc[:, "Total Liabilities", :]
        )
        net_income = (
            self._income_statement.loc[:, "Net Income", :].T.rolling(trailing).sum().T
            if trailing
            else self._income_statement.loc[:, "Net Income", :]
        )

        zmijewski_score["Net Income to Total Assets"] = (
            zmijewski_model.get_net_income_to_total_assets_ratio(
                net_income=net_income,
                total_assets=total_assets,
            )
        )

        zmijewski_score["Total Liabilities to Total Assets"] = (
            zmijewski_model.get_total_liabilities_to_total_assets_ratio(
                total_liabilities=total_liabilities,
                total_assets=total_assets,
            )
        )

        zmijewski_score["Current Assets to Current Liabilities"] = (
            zmijewski_model.get_current_assets_to_current_liabilities_ratio(
                current_assets=current_assets,
                current_liabilities=current_liabilities,
            )
        )

        zmijewski_score["Zmijewski Score"] = zmijewski_model.get_zmijewski_score(
            net_income_to_total_assets_ratio=zmijewski_score[
                "Net Income to Total Assets"
            ],
            total_liabilities_to_total_assets_ratio=zmijewski_score[
                "Total Liabilities to Total Assets"
            ],
            current_assets_to_current_liabilities_ratio=zmijewski_score[
                "Current Assets to Current Liabilities"
            ],
        )

        zmijewski_score["Zmijewski Bankruptcy Probability"] = (
            zmijewski_model.get_zmijewski_bankruptcy_probability(
                zmijewski_score=zmijewski_score["Zmijewski Score"],
            )
        )

        zmijewski_results = (
            pd.concat(zmijewski_score)
            .dropna(axis=1, how="all")
            .swaplevel(0, 1)
            .reindex(self._tickers, level=0)
        )

        zmijewski_results = finalize_dataset(
            dataset=zmijewski_results,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="columns",
        )

        return filter_columns(zmijewski_results, show_columns)

    @handle_errors
    def get_springate_score(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
        show_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculates the Springate Score, a financial metric used to predict the likelihood of
        a company going bankrupt. It follows the same multiple discriminant analysis
        methodology as the Altman Z-Score, but was calibrated on a smaller, Canadian-firm
        sample using four financial ratios instead of five.

        The formula is as follows:

            - Working Capital to Total Assets = Working Capital / Total Assets
            - EBIT to Total Assets = EBIT / Total Assets
            - EBT to Current Liabilities = Earnings Before Taxes / Total Current Liabilities
            - Sales to Total Assets = Sales / Total Assets
            - Springate Score = 1.03 * Working Capital to Total Assets +
            3.07 * EBIT to Total Assets + 0.66 * EBT to Current Liabilities +
            0.4 * Sales to Total Assets

        The Springate Score can be interpreted as follows:

            - A Springate Score of less than 0.862 indicates a high likelihood of bankruptcy.
            - A Springate Score of greater than 0.862 indicates a low likelihood of bankruptcy.

        Also known as: Springate Score, S-Score, bankruptcy prediction, financial distress score.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to None.
            growth (bool, optional): Whether to calculate the growth of the values. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int | None, optional): The trailing period to use for the calculation. Defaults to None.
            show_columns (list[str] | None, optional): List of columns to show in the results. If None, all
                columns will be shown. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame containing the Springate Score and its components.

        Notes:
        - As with the Altman Z-Score, Piotroski F-Score, Beneish M-Score, Ohlson O-Score and
        Zmijewski Score, this is a probabilistic, not a definitive, indicator and should be
        combined with further fundamental analysis.
        - EBIT is approximated as Net Income + Income Tax Expense + Interest Expense,
        consistent with the Altman Z-Score calculation elsewhere in this module.

        References:
        - Springate, Gordon L.V. "Predicting the Possibility of Failure in a Canadian Firm."
        Unpublished M.B.A. Research Project, Simon Fraser University, 1978.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "MSFT"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.models.get_springate_score().loc["AAPL"]
        ```

        Which returns:

        |                                   |    2021 |    2022 |
        |:----------------------------------|--------:|--------:|
        | Working Capital to Total Assets   |  0.0267 | -0.0527 |
        | EBIT to Total Assets              |  0.3187 |  0.3459 |
        | EBT to Current Liabilities        |  0.8703 |  0.7735 |
        | Sales to Total Assets             |  1.0422 |  1.1179 |
        | Springate Score                   |  1.997  |  1.9655 |
        """
        springate_score = {}

        current_assets = (
            self._balance_sheet_statement.loc[:, "Total Current Assets", :]
            .T.rolling(trailing)
            .mean()
            .T
            if trailing
            else self._balance_sheet_statement.loc[:, "Total Current Assets", :]
        )
        current_liabilities = (
            self._balance_sheet_statement.loc[:, "Total Current Liabilities", :]
            .T.rolling(trailing)
            .mean()
            .T
            if trailing
            else self._balance_sheet_statement.loc[:, "Total Current Liabilities", :]
        )
        total_assets = (
            self._balance_sheet_statement.loc[:, "Total Assets", :]
            .T.rolling(trailing)
            .mean()
            .T
            if trailing
            else self._balance_sheet_statement.loc[:, "Total Assets", :]
        )
        net_income = (
            self._income_statement.loc[:, "Net Income", :].T.rolling(trailing).sum().T
            if trailing
            else self._income_statement.loc[:, "Net Income", :]
        )
        income_tax_expense = (
            self._income_statement.loc[:, "Income Tax Expense", :]
            .T.rolling(trailing)
            .sum()
            .T
            if trailing
            else self._income_statement.loc[:, "Income Tax Expense", :]
        )
        interest_expense = (
            self._income_statement.loc[:, "Interest Expense", :]
            .T.rolling(trailing)
            .sum()
            .T
            if trailing
            else self._income_statement.loc[:, "Interest Expense", :]
        )
        income_before_tax = (
            self._income_statement.loc[:, "Income Before Tax", :]
            .T.rolling(trailing)
            .sum()
            .T
            if trailing
            else self._income_statement.loc[:, "Income Before Tax", :]
        )
        revenue = (
            self._income_statement.loc[:, "Revenue", :].T.rolling(trailing).sum().T
            if trailing
            else self._income_statement.loc[:, "Revenue", :]
        )

        working_capital = liquidity_model.get_working_capital(
            current_assets,
            current_liabilities,
        )

        springate_score["Working Capital to Total Assets"] = (
            springate_model.get_working_capital_to_total_assets_ratio(
                working_capital=working_capital,
                total_assets=total_assets,
            )
        )

        springate_score["EBIT to Total Assets"] = (
            springate_model.get_ebit_to_total_assets_ratio(
                ebit=(net_income + income_tax_expense + interest_expense),
                total_assets=total_assets,
            )
        )

        springate_score["EBT to Current Liabilities"] = (
            springate_model.get_ebt_to_current_liabilities_ratio(
                ebt=income_before_tax,
                current_liabilities=current_liabilities,
            )
        )

        springate_score["Sales to Total Assets"] = (
            springate_model.get_sales_to_total_assets_ratio(
                sales=revenue,
                total_assets=total_assets,
            )
        )

        springate_score["Springate Score"] = springate_model.get_springate_score(
            working_capital_to_total_assets_ratio=springate_score[
                "Working Capital to Total Assets"
            ],
            ebit_to_total_assets_ratio=springate_score["EBIT to Total Assets"],
            ebt_to_current_liabilities_ratio=springate_score[
                "EBT to Current Liabilities"
            ],
            sales_to_total_assets_ratio=springate_score["Sales to Total Assets"],
        )

        springate_results = (
            pd.concat(springate_score)
            .dropna(axis=1, how="all")
            .swaplevel(0, 1)
            .reindex(self._tickers, level=0)
        )

        springate_results = finalize_dataset(
            dataset=springate_results,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="columns",
        )

        return filter_columns(springate_results, show_columns)

    @handle_errors
    def get_grover_score(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
        show_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculates the Grover Score, a financial metric used to predict the likelihood of a
        company going bankrupt. It was developed by re-estimating the coefficients of a
        reduced-form Altman Z-Score and adding a Return on Assets term, using a sample that
        paired each of Altman's original bankrupt firms with a matched non-bankrupt firm from
        the same industry and year.

        The formula is as follows:

            - Working Capital to Total Assets = Working Capital / Total Assets
            - EBIT to Total Assets = EBIT / Total Assets
            - Return on Assets = Net Income / Total Assets
            - Grover Score = 1.65 * Working Capital to Total Assets +
            3.404 * EBIT to Total Assets - 0.016 * Return on Assets + 0.057

        The Grover Score can be interpreted as follows:

            - A Grover Score of -0.02 or lower indicates a high likelihood of bankruptcy.
            - A Grover Score of 0.01 or higher indicates a low likelihood of bankruptcy
            (per some secondary sources), leaving a gray area in between the two thresholds.

        Also known as: Grover Score, G-Score, bankruptcy prediction, financial distress score.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to None.
            growth (bool, optional): Whether to calculate the growth of the values. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int | None, optional): The trailing period to use for the calculation. Defaults to None.
            show_columns (list[str] | None, optional): List of columns to show in the results. If None, all
                columns will be shown. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame containing the Grover Score and its components.

        Notes:
        - As with the Altman Z-Score, Springate Score, Ohlson O-Score and Zmijewski Score,
        this is a probabilistic, not a definitive, indicator and should be combined with
        further fundamental analysis.
        - EBIT is approximated as Net Income + Income Tax Expense + Interest Expense,
        consistent with the Altman Z-Score calculation elsewhere in this module. Return on
        Assets uses the point-in-time Total Assets balance (not averaged), matching the
        original Grover (2001) specification.

        References:
        - Grover, Jeffrey S. "Validating the Grover Bankruptcy Model." Doctoral dissertation,
        University of North Texas, 2003.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "MSFT"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.models.get_grover_score().loc["AAPL"]
        ```

        Which returns:

        |                                   |    2021 |    2022 |
        |:----------------------------------|--------:|--------:|
        | Working Capital to Total Assets   |  0.0267 | -0.0527 |
        | EBIT to Total Assets              |  0.3187 |  0.3459 |
        | Return on Assets                  |  0.2697 |  0.2829 |
        | Grover Score                      |  1.1814 |  1.1432 |
        """
        grover_score = {}

        current_assets = (
            self._balance_sheet_statement.loc[:, "Total Current Assets", :]
            .T.rolling(trailing)
            .mean()
            .T
            if trailing
            else self._balance_sheet_statement.loc[:, "Total Current Assets", :]
        )
        current_liabilities = (
            self._balance_sheet_statement.loc[:, "Total Current Liabilities", :]
            .T.rolling(trailing)
            .mean()
            .T
            if trailing
            else self._balance_sheet_statement.loc[:, "Total Current Liabilities", :]
        )
        total_assets = (
            self._balance_sheet_statement.loc[:, "Total Assets", :]
            .T.rolling(trailing)
            .mean()
            .T
            if trailing
            else self._balance_sheet_statement.loc[:, "Total Assets", :]
        )
        net_income = (
            self._income_statement.loc[:, "Net Income", :].T.rolling(trailing).sum().T
            if trailing
            else self._income_statement.loc[:, "Net Income", :]
        )
        income_tax_expense = (
            self._income_statement.loc[:, "Income Tax Expense", :]
            .T.rolling(trailing)
            .sum()
            .T
            if trailing
            else self._income_statement.loc[:, "Income Tax Expense", :]
        )
        interest_expense = (
            self._income_statement.loc[:, "Interest Expense", :]
            .T.rolling(trailing)
            .sum()
            .T
            if trailing
            else self._income_statement.loc[:, "Interest Expense", :]
        )

        working_capital = liquidity_model.get_working_capital(
            current_assets,
            current_liabilities,
        )

        grover_score["Working Capital to Total Assets"] = (
            grover_model.get_working_capital_to_total_assets_ratio(
                working_capital=working_capital,
                total_assets=total_assets,
            )
        )

        grover_score["EBIT to Total Assets"] = (
            grover_model.get_ebit_to_total_assets_ratio(
                ebit=(net_income + income_tax_expense + interest_expense),
                total_assets=total_assets,
            )
        )

        grover_score["Return on Assets"] = grover_model.get_return_on_assets_ratio(
            net_income=net_income,
            total_assets=total_assets,
        )

        grover_score["Grover Score"] = grover_model.get_grover_score(
            working_capital_to_total_assets_ratio=grover_score[
                "Working Capital to Total Assets"
            ],
            ebit_to_total_assets_ratio=grover_score["EBIT to Total Assets"],
            return_on_assets_ratio=grover_score["Return on Assets"],
        )

        grover_results = (
            pd.concat(grover_score)
            .dropna(axis=1, how="all")
            .swaplevel(0, 1)
            .reindex(self._tickers, level=0)
        )

        grover_results = finalize_dataset(
            dataset=grover_results,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="columns",
        )

        return filter_columns(grover_results, show_columns)

    @handle_errors
    def get_fulmer_h_score(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
        show_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculates the Fulmer H-Score, a financial metric used to predict the likelihood of a
        company going bankrupt. Unlike the Altman Z-Score, which was built on relatively
        large, listed manufacturing companies, the Fulmer H-Score was calibrated on a
        matched-pair sample of small, failed and non-failed U.S. companies, using nine
        financial ratios selected via step-wise multiple discriminant analysis.

        The formula is as follows:

            H-Score = 5.528 * V1 + 0.212 * V2 + 0.073 * V3 + 1.270 * V4 - 0.120 * V5
            + 2.335 * V6 + 0.575 * V7 + 1.083 * V8 + 0.894 * V9 - 6.075

        The nine variables are:

            - V1: Retained Earnings to Total Assets
            - V2: Sales to Total Assets
            - V3: EBT to Total Equity
            - V4: Cash Flow (Operating Cash Flow) to Total Liabilities
            - V5: Total Debt to Total Assets
            - V6: Current Liabilities to Total Assets
            - V7: Log of Tangible Total Assets (Total Assets less Goodwill and Intangible Assets)
            - V8: Working Capital to Total Liabilities
            - V9: Log of EBIT to Interest Expense

        The Fulmer H-Score can be interpreted as follows:

            - An H-Score of less than 0 predicts failure.
            - An H-Score of greater than 0 predicts non-failure.

        Also known as: Fulmer H-Score, H factor, bankruptcy prediction, financial distress score.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to None.
            growth (bool, optional): Whether to calculate the growth of the values. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int | None, optional): The trailing period to use for the calculation. Defaults to None.
            show_columns (list[str] | None, optional): List of columns to show in the results. If None, all
                columns will be shown. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame containing the Fulmer H-Score and its components.

        Notes:
        - Because V7 and V9 involve a natural logarithm, periods where Tangible Total Assets
        is zero or negative, where EBIT and Interest Expense do not share the same sign, or
        where the company reports no Interest Expense at all, will produce NaN and therefore
        no H-Score for that period. This is a structural limitation of the log-linear Fulmer
        specification, not a bug — the same kind of limitation documented for the Graham
        Number elsewhere in this toolkit.
        - Because V7 uses an un-normalized dollar figure, the H-Score is sensitive to the
        absolute scale of a company's total assets; treat results for very large companies
        (far outside the small-firm sample the model was calibrated on) with additional
        caution.
        - As with the Altman Z-Score, Springate Score and Grover Score, this is a
        probabilistic, not a definitive, indicator and should be combined with further
        fundamental analysis.

        References:
        - Fulmer, John G., James E. Moon, Thomas A. Gavin, and Michael J. Erwin. "A
        Bankruptcy Classification Model for Small Firms." Journal of Commercial Bank
        Lending, Vol. 66, No. 11, 1984, pp. 25-37.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "MSFT"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.models.get_fulmer_h_score().loc["AAPL"]
        ```

        Which returns:

        |                                       |    2021 |    2022 |
        |:--------------------------------------|--------:|--------:|
        | Retained Earnings to Total Assets     |  0.0158 | -0.0087 |
        | Sales to Total Assets                 |  1.0422 |  1.1179 |
        | EBT to Total Equity                   |  1.731  |  2.3505 |
        | Cash Flow to Total Liabilities        |  0.3614 |  0.4044 |
        | Debt to Total Assets                  |  0.3889 |  0.3756 |
        | Current Liabilities to Total Assets   |  0.3575 |  0.4365 |
        | Log of Tangible Total Assets          | 26.5841 | 26.589  |
        | Working Capital to Total Liabilities  |  0.0325 | -0.0615 |
        | Log of EBIT to Interest Expense       |  3.7445 |  3.729  |
        | Fulmer H-Score                        | 14.2755 | 14.329  |
        """
        fulmer_h_score = {}

        current_assets = (
            self._balance_sheet_statement.loc[:, "Total Current Assets", :]
            .T.rolling(trailing)
            .mean()
            .T
            if trailing
            else self._balance_sheet_statement.loc[:, "Total Current Assets", :]
        )
        current_liabilities = (
            self._balance_sheet_statement.loc[:, "Total Current Liabilities", :]
            .T.rolling(trailing)
            .mean()
            .T
            if trailing
            else self._balance_sheet_statement.loc[:, "Total Current Liabilities", :]
        )
        total_assets = (
            self._balance_sheet_statement.loc[:, "Total Assets", :]
            .T.rolling(trailing)
            .mean()
            .T
            if trailing
            else self._balance_sheet_statement.loc[:, "Total Assets", :]
        )
        total_liabilities = (
            self._balance_sheet_statement.loc[:, "Total Liabilities", :]
            .T.rolling(trailing)
            .mean()
            .T
            if trailing
            else self._balance_sheet_statement.loc[:, "Total Liabilities", :]
        )
        total_debt = (
            self._balance_sheet_statement.loc[:, "Total Debt", :]
            .T.rolling(trailing)
            .mean()
            .T
            if trailing
            else self._balance_sheet_statement.loc[:, "Total Debt", :]
        )
        total_equity = (
            self._balance_sheet_statement.loc[:, "Total Equity", :]
            .T.rolling(trailing)
            .mean()
            .T
            if trailing
            else self._balance_sheet_statement.loc[:, "Total Equity", :]
        )
        retained_earnings = (
            self._balance_sheet_statement.loc[:, "Retained Earnings", :]
            .T.rolling(trailing)
            .mean()
            .T
            if trailing
            else self._balance_sheet_statement.loc[:, "Retained Earnings", :]
        )
        goodwill_and_intangible_assets = (
            self._balance_sheet_statement.loc[:, "Goodwill and Intangible Assets", :]
            .T.rolling(trailing)
            .mean()
            .T
            if trailing
            else self._balance_sheet_statement.loc[
                :, "Goodwill and Intangible Assets", :
            ]
        )
        revenue = (
            self._income_statement.loc[:, "Revenue", :].T.rolling(trailing).sum().T
            if trailing
            else self._income_statement.loc[:, "Revenue", :]
        )
        income_before_tax = (
            self._income_statement.loc[:, "Income Before Tax", :]
            .T.rolling(trailing)
            .sum()
            .T
            if trailing
            else self._income_statement.loc[:, "Income Before Tax", :]
        )
        interest_expense = (
            self._income_statement.loc[:, "Interest Expense", :]
            .T.rolling(trailing)
            .sum()
            .T
            if trailing
            else self._income_statement.loc[:, "Interest Expense", :]
        )
        net_income = (
            self._income_statement.loc[:, "Net Income", :].T.rolling(trailing).sum().T
            if trailing
            else self._income_statement.loc[:, "Net Income", :]
        )
        income_tax_expense = (
            self._income_statement.loc[:, "Income Tax Expense", :]
            .T.rolling(trailing)
            .sum()
            .T
            if trailing
            else self._income_statement.loc[:, "Income Tax Expense", :]
        )
        operating_cash_flow = (
            self._cash_flow_statement.loc[:, "Operating Cash Flow", :]
            .T.rolling(trailing)
            .sum()
            .T
            if trailing
            else self._cash_flow_statement.loc[:, "Operating Cash Flow", :]
        )

        working_capital = liquidity_model.get_working_capital(
            current_assets,
            current_liabilities,
        )
        ebit = net_income + income_tax_expense + interest_expense
        tangible_total_assets = fulmer_model.get_tangible_total_assets(
            total_assets=total_assets,
            goodwill_and_intangible_assets=goodwill_and_intangible_assets,
        )

        fulmer_h_score["Retained Earnings to Total Assets"] = (
            fulmer_model.get_retained_earnings_to_total_assets_ratio(
                retained_earnings=retained_earnings,
                total_assets=total_assets,
            )
        )

        fulmer_h_score["Sales to Total Assets"] = (
            fulmer_model.get_sales_to_total_assets_ratio(
                sales=revenue,
                total_assets=total_assets,
            )
        )

        fulmer_h_score["EBT to Total Equity"] = fulmer_model.get_ebt_to_equity_ratio(
            ebt=income_before_tax,
            total_equity=total_equity,
        )

        fulmer_h_score["Cash Flow to Total Liabilities"] = (
            fulmer_model.get_cash_flow_to_total_liabilities_ratio(
                cash_flow=operating_cash_flow,
                total_liabilities=total_liabilities,
            )
        )

        fulmer_h_score["Debt to Total Assets"] = (
            fulmer_model.get_debt_to_total_assets_ratio(
                total_debt=total_debt,
                total_assets=total_assets,
            )
        )

        fulmer_h_score["Current Liabilities to Total Assets"] = (
            fulmer_model.get_current_liabilities_to_total_assets_ratio(
                current_liabilities=current_liabilities,
                total_assets=total_assets,
            )
        )

        fulmer_h_score["Log of Tangible Total Assets"] = (
            fulmer_model.get_log_of_tangible_total_assets(
                tangible_total_assets=tangible_total_assets,
            )
        )

        fulmer_h_score["Working Capital to Total Liabilities"] = (
            fulmer_model.get_working_capital_to_total_liabilities_ratio(
                working_capital=working_capital,
                total_liabilities=total_liabilities,
            )
        )

        fulmer_h_score["Log of EBIT to Interest Expense"] = (
            fulmer_model.get_log_of_ebit_to_interest_expense_ratio(
                ebit=ebit,
                interest_expense=interest_expense,
            )
        )

        fulmer_h_score["Fulmer H-Score"] = fulmer_model.get_fulmer_h_score(
            retained_earnings_to_total_assets_ratio=fulmer_h_score[
                "Retained Earnings to Total Assets"
            ],
            sales_to_total_assets_ratio=fulmer_h_score["Sales to Total Assets"],
            ebt_to_equity_ratio=fulmer_h_score["EBT to Total Equity"],
            cash_flow_to_total_liabilities_ratio=fulmer_h_score[
                "Cash Flow to Total Liabilities"
            ],
            debt_to_total_assets_ratio=fulmer_h_score["Debt to Total Assets"],
            current_liabilities_to_total_assets_ratio=fulmer_h_score[
                "Current Liabilities to Total Assets"
            ],
            log_of_tangible_total_assets=fulmer_h_score["Log of Tangible Total Assets"],
            working_capital_to_total_liabilities_ratio=fulmer_h_score[
                "Working Capital to Total Liabilities"
            ],
            log_of_ebit_to_interest_expense_ratio=fulmer_h_score[
                "Log of EBIT to Interest Expense"
            ],
        )

        fulmer_results = (
            pd.concat(fulmer_h_score)
            .dropna(axis=1, how="all")
            .swaplevel(0, 1)
            .reindex(self._tickers, level=0)
        )

        fulmer_results = finalize_dataset(
            dataset=fulmer_results,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="columns",
        )

        return filter_columns(fulmer_results, show_columns)

    @handle_errors
    def get_present_value_of_growth_opportunities(
        self,
        calculate_daily: bool = False,
        diluted: bool = True,
        include_dividends: bool = False,
        trailing: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ) -> pd.DataFrame:
        """
        The Present Value of Growth Opportunities (PVGO) is a financial metric that represents the
        present value of a company's future growth opportunities. It is calculated as the difference
        between the company's current stock price and the discounted value of its future cash flows.

        The formula is as follows:

            - PVGO = Stock Price — Earnings Per Share / Weighted Average Cost of Capital

        Also known as: PVGO, growth value.

        Args:
            calculate_daily (bool, optional): Whether to calculate the PVGO using daily historical data.
            Defaults to False.
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            include_dividends (bool, optional): Whether to deduct Preferred Dividends Paid from Net
            Income when calculating the Earnings per Share, so that the earnings figure reflects what is
            attributable to common shareholders only. Defaults to False.
            trailing (int | None, optional): The trailing period to use for the calculation. Defaults to None.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the values. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.DataFrame: DataFrame containing the PVGO values.

        Notes:
        - The textbook PVGO discounts the no-growth value of the company (Earnings per Share / r) at
        the cost of equity, since both the share price and the Earnings per Share are equity-only,
        per-share quantities. This implementation discounts at the Weighted Average Cost of Capital
        instead, which blends in the (typically lower, tax-shielded) cost of debt and therefore
        generally understates PVGO. Prefer comparing PVGO across companies, or over time for the same
        company, over reading absolute levels literally.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.models.get_present_value_of_growth_opportunities()
        ```

        Which returns:

        |      |    AAPL |    TSLA |
        |:-----|--------:|--------:|
        | 2021 | 160.807 | 348.912 |
        | 2022 | 154.58  | 133.203 |
        | 2023 | 168.39  | 239.608 |
        | 2024 | 222.742 | 399.662 |
        | 2025 | 232.279 | 446.273 |
        """
        wacc = self.get_weighted_average_cost_of_capital(
            show_full_results=False, trailing=trailing
        )

        average_shares = (
            self._income_statement.loc[:, "Weighted Average Shares Diluted", :]
            if diluted
            else self._income_statement.loc[:, "Weighted Average Shares", :]
        )

        # Preferred Dividends Paid is reported on the cash flow statement using the
        # cash-flow-impact convention (an outflow is negative), while the Earnings per Share
        # formula subtracts a positive-magnitude figure — without the absolute value the
        # preferred dividends would be added back to Net Income instead of deducted from it
        dividends = (
            self._cash_flow_statement.loc[:, "Preferred Dividends Paid", :].abs()
            if include_dividends
            else 0
        )

        net_income = (
            self._income_statement.loc[:, "Net Income", :].T.rolling(trailing).sum().T
            if trailing
            else self._income_statement.loc[:, "Net Income", :]
        )
        preferred_dividends = (
            dividends.T.rolling(trailing).sum().T
            if trailing and include_dividends
            else dividends
        )
        avg_shares = (
            average_shares.T.rolling(trailing).mean().T if trailing else average_shares
        )

        earnings_per_share = valuation_model.get_earnings_per_share(
            net_income=net_income,
            preferred_dividends=preferred_dividends,
            average_outstanding_shares=avg_shares,
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
            pvgo = calculate_growth(
                dataset=pvgo,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )
            if standardize:
                pvgo = calculate_standardization(
                    dataset=pvgo,
                    rounding=rounding if rounding else self._rounding,
                    axis="rows",
                )
            return pvgo.loc[self._start_date :]

        pvgo = pvgo.round(rounding if rounding else self._rounding)

        if standardize:
            pvgo = calculate_standardization(
                dataset=pvgo,
                rounding=rounding if rounding else self._rounding,
                axis="rows",
            )

        # When there is no data found for any ticker, drop the row
        pvgo = pvgo.dropna(how="all", axis=0)

        return pvgo.loc[self._start_date :]

    @handle_errors
    def get_sustainable_growth_rate(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        The Sustainable Growth Rate (SGR) is the maximum rate at which a company can grow its
        revenue, using internally generated funds only, without having to raise additional equity
        or increase its financial leverage.

        The formula is as follows:

            - Retention Ratio = 1 - Dividend Payout Ratio
            - SGR = Return on Equity * Retention Ratio

        Also known as: SGR, self-sustainable growth rate.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to None.
            growth (bool, optional): Whether to calculate the growth of the values. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int | None, optional): The trailing period to use for the calculation. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame containing the Sustainable Growth Rate.

        Notes:
        - Growing faster than the SGR without external financing typically requires either
        improving profitability, reducing the dividend payout, or increasing leverage.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "MSFT"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.models.get_sustainable_growth_rate()
        ```

        Which returns:

        |      |   2021 |   2022 |   2023 |
        |:-----|-------:|-------:|-------:|
        | AAPL | 1.2491 | 1.4937 | 1.4531 |
        | MSFT | 0.3438 | 0.354  | 0.282  |
        """
        net_income = (
            self._income_statement.loc[:, "Net Income", :].T.rolling(trailing).sum().T
            if trailing
            else self._income_statement.loc[:, "Net Income", :]
        )
        average_total_equity = (
            self._balance_sheet_statement.loc[:, "Total Equity", :]
            .T.rolling(trailing)
            .mean()
            .T
            if trailing
            else self._balance_sheet_statement.loc[:, "Total Equity", :]
            .T.rolling(2)
            .mean()
            .T
        )
        dividends = (
            self._cash_flow_statement.loc[:, "Dividends Paid", :]
            .T.rolling(trailing)
            .sum()
            .T
            if trailing
            else self._cash_flow_statement.loc[:, "Dividends Paid", :]
        )

        return_on_equity = profitability_model.get_return_on_equity(
            net_income=net_income,
            average_total_equity=average_total_equity,
        )
        dividend_payout_ratio = valuation_model.get_dividend_payout_ratio(
            dividends=dividends,
            net_income=net_income,
        )
        retention_ratio = valuation_model.get_reinvestment_ratio(
            dividend_payout_ratio=dividend_payout_ratio,
        )

        sustainable_growth_rate = growth_model.get_sustainable_growth_rate(
            return_on_equity=return_on_equity,
            retention_ratio=retention_ratio,
        )

        return finalize_dataset(
            dataset=sustainable_growth_rate,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="columns",
        )

    @handle_errors
    def get_internal_growth_rate(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        The Internal Growth Rate (IGR) is the maximum rate at which a company can grow its
        revenue using only its retained earnings, without raising any external financing
        (neither debt nor equity).

        The formula is as follows:

            - Retention Ratio = 1 - Dividend Payout Ratio
            - IGR = (Return on Assets * Retention Ratio) / (1 - (Return on Assets * Retention Ratio))

        Also known as: IGR.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to None.
            growth (bool, optional): Whether to calculate the growth of the values. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int | None, optional): The trailing period to use for the calculation. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame containing the Internal Growth Rate.

        Notes:
        - The IGR is more conservative than the Sustainable Growth Rate (SGR) since it assumes
        no additional debt is raised to fund growth, whereas the SGR assumes the company
        maintains its current level of financial leverage.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "MSFT"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.models.get_internal_growth_rate()
        ```

        Which returns:

        |      |   2021 |   2022 |   2023 |
        |:-----|-------:|-------:|-------:|
        | AAPL | 0.3118 | 0.3183 | 0.3028 |
        | MSFT | 0.164  | 0.1853 | 0.1565 |
        """
        net_income = (
            self._income_statement.loc[:, "Net Income", :].T.rolling(trailing).sum().T
            if trailing
            else self._income_statement.loc[:, "Net Income", :]
        )
        average_total_assets = (
            self._balance_sheet_statement.loc[:, "Total Assets", :]
            .T.rolling(trailing)
            .mean()
            .T
            if trailing
            else self._balance_sheet_statement.loc[:, "Total Assets", :]
            .T.rolling(2)
            .mean()
            .T
        )
        dividends = (
            self._cash_flow_statement.loc[:, "Dividends Paid", :]
            .T.rolling(trailing)
            .sum()
            .T
            if trailing
            else self._cash_flow_statement.loc[:, "Dividends Paid", :]
        )

        return_on_assets = profitability_model.get_return_on_assets(
            net_income=net_income,
            average_total_assets=average_total_assets,
        )
        dividend_payout_ratio = valuation_model.get_dividend_payout_ratio(
            dividends=dividends,
            net_income=net_income,
        )
        retention_ratio = valuation_model.get_reinvestment_ratio(
            dividend_payout_ratio=dividend_payout_ratio,
        )

        internal_growth_rate = growth_model.get_internal_growth_rate(
            return_on_assets=return_on_assets,
            retention_ratio=retention_ratio,
        )

        return finalize_dataset(
            dataset=internal_growth_rate,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="columns",
        )

    @handle_errors
    def get_graham_number(
        self,
        diluted: bool = True,
        trailing: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ) -> pd.DataFrame:
        """
        Calculate the Graham Number, a conservative estimate of a stock's fair value
        based on its earnings and book value, as devised by Benjamin Graham.

        The Graham Number is intended as an upper bound on the price a defensive
        investor should pay for a stock. It is most meaningful for stable,
        profitable companies with positive book value — for companies with negative
        earnings or negative book value the result is not meaningful (the square
        root of a negative number is undefined and will show up as NaN).

        The formula is as follows:

        - Graham Number = √(22.5 x Earnings per Share x Book Value per Share)

        Also known as: Graham fair value.

        Args:
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            trailing (int | None, optional): The trailing period to use for the calculation. Defaults to None.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the values. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.DataFrame: DataFrame containing the Graham Number values.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.models.get_graham_number()
        ```

        Which returns:

        |      |    2021 |    2022 |    2023 |    2024 |    2025 |
        |:-----|--------:|--------:|--------:|--------:|--------:|
        | AAPL | 21.7378 | 20.662  | 23.2901 | 22.4927 | 28.7292 |
        | TSLA | 18.1054 | 32.3757 | 41.7451 | 30.9185 | 23.7345 |
        """
        average_shares = (
            self._income_statement.loc[:, "Weighted Average Shares Diluted", :]
            if diluted
            else self._income_statement.loc[:, "Weighted Average Shares", :]
        )

        net_income = (
            self._income_statement.loc[:, "Net Income", :].T.rolling(trailing).sum().T
            if trailing
            else self._income_statement.loc[:, "Net Income", :]
        )
        avg_shares = (
            average_shares.T.rolling(trailing).mean().T if trailing else average_shares
        )

        earnings_per_share = valuation_model.get_earnings_per_share(
            net_income=net_income,
            preferred_dividends=0,
            average_outstanding_shares=avg_shares,
        )

        total_shareholder_equity = (
            self._balance_sheet_statement.loc[:, "Total Shareholder Equity", :]
            .T.rolling(trailing)
            .mean()
            .T
            if trailing
            else self._balance_sheet_statement.loc[:, "Total Shareholder Equity", :]
        )
        preferred_stock = (
            self._balance_sheet_statement.loc[:, "Preferred Stock", :]
            .T.rolling(trailing)
            .mean()
            .T
            if trailing
            else self._balance_sheet_statement.loc[:, "Preferred Stock", :]
        )

        book_value_per_share = valuation_model.get_book_value_per_share(
            total_shareholder_equity=total_shareholder_equity,
            preferred_equity=preferred_stock,
            common_shares_outstanding=avg_shares,
        )

        graham_number = intrinsic_model.get_graham_number(
            earnings_per_share=earnings_per_share,
            book_value_per_share=book_value_per_share,
        )

        return finalize_dataset(
            dataset=graham_number,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="columns",
        )

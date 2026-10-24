"""Models Module"""

__docformat__ = "google"

from collections.abc import Callable

import pandas as pd

from financetoolkit.models import (
    altman_model,
    beneish_model,
    credit_model,
    damodaran_model,
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
    quality_model,
    springate_model,
    wacc_model,
    zmijewski_model,
)
from financetoolkit.performance.performance_model import get_beta
from financetoolkit.ratios import liquidity_model, profitability_model, valuation_model
from financetoolkit.utilities.dataframe_model import concat_frames, filter_columns
from financetoolkit.utilities.error_model import handle_errors
from financetoolkit.utilities.logger_model import get_logger
from financetoolkit.utilities.statistics_model import (
    apply_rounding,
    calculate_growth,
    calculate_standardization,
    finalize_dataset,
)

# pylint: disable=too-many-instance-attributes,too-many-locals,too-many-lines
# ruff: noqa: E501


logger = get_logger()


class Models:
    """
    The Models module is meant to execute well-known models such
    as DUPONT and the Discounted Cash Flow (DCF) model. These models
    are also directly related to the data retrieved from the Toolkit module.

    """

    def __init__(
        self,
        tickers: str | list[str],
        historical_data: dict[str, pd.DataFrame],
        risk_free_rate_data: dict[str, pd.DataFrame],
        balance: pd.DataFrame,
        income: pd.DataFrame,
        cash: pd.DataFrame,
        quarterly: bool = False,
        rounding: int | None = 4,
        start_date: str | None = None,
        end_date: str | None = None,
        peer_function: (
            Callable[[dict[str, list[str]] | None], pd.DataFrame] | None
        ) = None,
    ):
        """
        Initializes the Models Controller Class.

        Args:
            tickers (str | list[str]): The ticker(s) to use for the models.
            historical_data (dict[str, pd.DataFrame]): The historical data per period.
            risk_free_rate_data (dict[str, pd.DataFrame]): The risk free rate data per period.
            balance (pd.DataFrame): The balance sheet data.
            income (pd.DataFrame): The income statement data.
            cash (pd.DataFrame): The cash flow statement data.
            quarterly (bool, optional): Whether to use quarterly or yearly data. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            peer_function (Callable | None, optional): Compares the companies with their peers,
                used by get_peer_comparison. Defaults to None.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["TSLA", "AMZN"], api_key="FINANCIAL_MODELING_PREP_KEY",
                        end_date="2025-12-31",
        quarterly=True, start_date='2022-12-31')

        dupont_analysis = toolkit.models.get_extended_dupont_analysis()

        dupont_analysis.loc['AMZN']
        ```

        Which returns:

        |                         |   2022Q4 |   2023Q1 |   2023Q2 |
        |:------------------------|---------:|---------:|---------:|
        | Interest Burden Ratio   |  -0.3467 |   0.863  |   0.9835 |
        | Tax Burden Ratio        |  -0.2929 |   0.7699 |   0.8936 |
        | Operating Profit Margin |   0.0183 |   0.0375 |   0.0572 |
        | Asset Turnover          |   0.3349 |   0.2748 |   0.2853 |
        | Equity Multiplier       |   3.1426 |   3.0843 |   2.9152 |
        | Return on Equity        |   0.002  |   0.0211 |   0.0418 |
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
        self._peer_function = peer_function

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
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int | None, optional): The trailing period to use for the calculation. Defaults to None.
            show_columns (list[str] | None, optional): Restrict the result to these columns. Defaults to None,
                which shows all columns.

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

        toolkit = Toolkit(
            ["AAPL", "TSLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

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
                rounding=rounding if rounding is not None else self._rounding,
                axis="columns",
            )

        self._dupont_analysis = apply_rounding(
            self._dupont_analysis, rounding if rounding is not None else self._rounding
        )

        if standardize:
            standardize_rounding = rounding if rounding is not None else self._rounding
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
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int | None, optional): The trailing period to use for the calculation. Defaults to None.
            show_columns (list[str] | None, optional): Restrict the result to these columns. Defaults to None,
                which shows all columns.

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

        toolkit = Toolkit(
            ["AAPL", "TSLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        extended_dupont_analysis = toolkit.models.get_extended_dupont_analysis()

        extended_dupont_analysis.loc["AAPL"]
        ```

        Which returns:

        |                         |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:------------------------|-------:|-------:|-------:|-------:|-------:|
        | Interest Burden Ratio   | 1.0024 | 0.9972 | 0.9951 | 1.0022 | 0.9976 |
        | Tax Burden Ratio        | 0.867  | 0.838  | 0.8528 | 0.7591 | 0.8439 |
        | Operating Profit Margin | 0.2978 | 0.3029 | 0.2982 | 0.3151 | 0.3197 |
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
                rounding=rounding if rounding is not None else self._rounding,
                axis="columns",
            )

        self._extended_dupont_analysis = apply_rounding(
            self._extended_dupont_analysis,
            rounding if rounding is not None else self._rounding,
        )

        if standardize:
            standardize_rounding = rounding if rounding is not None else self._rounding
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
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (list[str] | None, optional): Restrict the result to these columns. Defaults to None,
                which shows all columns.

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

        toolkit = Toolkit(
            ["AAPL", "TSLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        enterprise_value_breakdown = toolkit.models.get_enterprise_value_breakdown()

        enterprise_value_breakdown.loc["AAPL"]
        ```

        Which returns:

        |                           |          2021 |          2022 |          2023 |          2024 |          2025 |
        |:--------------------------|--------------:|--------------:|--------------:|--------------:|--------------:|
        | Share Price               | 173.45        | 127.65        | 190.21        | 248.62        | 271.12        |
        | Market Capitalization     |   2.92522e+12 |   2.08399e+12 |   3.0077e+12  |   3.83076e+12 |   4.06807e+12 |
        | Total Debt                |   1.36522e+11 |   1.3248e+11  |   1.2393e+11  |   1.19059e+11 |   1.12377e+11 |
        | Minority Interest         |   0           |   0           |   0           |   0           |   0           |
        | Preferred Equity          |   0           |   0           |   0           |   0           |   0           |
        | Cash and Cash Equivalents |   3.494e+10   |   2.3646e+10  |   2.9965e+10  |   2.9943e+10  |   3.5934e+10  |
        | Enterprise Value          |   3.0268e+12  |   2.19282e+12 |   3.10167e+12 |   3.91988e+12 |   4.14452e+12 |
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
                rounding=rounding if rounding is not None else self._rounding,
            )

        self._enterprise_value_breakdown = apply_rounding(
            self._enterprise_value_breakdown,
            rounding if rounding is not None else self._rounding,
        )

        if standardize:
            standardize_rounding = rounding if rounding is not None else self._rounding
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

        toolkit = Toolkit(
            ["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_tobins_q_ratio().loc["AAPL"]
        ```

        Which returns:

        |                        |        2021 |        2022 |        2023 |         2024 |         2025 |
        |:-----------------------|------------:|------------:|------------:|-------------:|-------------:|
        | Market Value of Equity | 2.92522e+12 | 2.08399e+12 | 3.0077e+12  |  3.83076e+12 |  4.06807e+12 |
        | Total Liabilities      | 2.87912e+11 | 3.02083e+11 | 2.90437e+11 |  3.0803e+11  |  2.85508e+11 |
        | Total Assets           | 3.51002e+11 | 3.52755e+11 | 3.52583e+11 |  3.6498e+11  |  3.59241e+11 |
        | Tobin's Q Ratio        | 9.1542      | 6.7641      | 9.3542      | 11.3398      | 12.1188      |
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
            concat_frames(tobins_q_ratio)
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
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int | None, optional): The trailing period to use for the calculation. Defaults to None.
            show_columns (list[str] | None, optional): Restrict the result to these columns. Defaults to None,
                which shows all columns.

        Returns:
            pd.DataFrame: DataFrame containing the WACC values.

        Notes:
        - The Cost of Equity is approximated with the Capital Asset Pricing Model (CAPM).
        - The Market Value of Debt is approximated as the Total Debt.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "TSLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_weighted_average_cost_of_capital().loc["AAPL"]
        ```

        Which returns:

        |                                  |          2021 |         2022 |       2023 |        2024 |        2025 |
        |:---------------------------------|--------------:|-------------:|-----------:|------------:|------------:|
        | Market Value Equity              |   2.92522e+12 |  2.08399e+12 | 3.0077e+12 | 3.83076e+12 | 4.06807e+12 |
        | Market Value Debt                |   1.36522e+11 |  1.3248e+11  | 1.2393e+11 | 1.19059e+11 | 1.12377e+11 |
        | Cost of Equity                   | nan           | -0.2485      | 0.2874     | 0.2425      | 0.2117      |
        | Cost of Debt                     |   0.0194      |  0.0221      | 0.0317     | 0           | 0           |
        | Corporate Tax Rate               |   0.133       |  0.162       | 0.1472     | 0.2409      | 0.1561      |
        | Weighted Average Cost of Capital | nan           | -0.2326      | 0.2771     | 0.2351      | 0.206       |
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
                rounding=rounding if rounding is not None else self._rounding,
                axis="columns",
            )

        self._weighted_average_cost_of_capital = apply_rounding(
            self._weighted_average_cost_of_capital,
            rounding if rounding is not None else self._rounding,
        )

        if standardize:
            standardize_rounding = rounding if rounding is not None else self._rounding
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

        toolkit = Toolkit(
            ["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_economic_value_added().loc["AAPL"]
        ```

        Which returns:

        |                                  |          2021 |         2022 |        2023 |        2024 |        2025 |
        |:---------------------------------|--------------:|-------------:|------------:|------------:|------------:|
        | Net Operating Profit After Taxes |   9.69732e+10 |  1.02259e+11 | 1.00349e+11 | 9.3736e+10  | 1.1201e+11  |
        | Invested Capital                 |   1.93614e+11 |  1.91382e+11 | 1.84614e+11 | 1.81042e+11 | 1.8106e+11  |
        | Weighted Average Cost of Capital | nan           | -0.2326      | 0.2771      | 0.2351      | 0.206       |
        | Economic Value Added             | nan           |  1.46775e+11 | 4.91926e+10 | 5.11729e+10 | 7.47117e+10 |
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
            concat_frames(eva)
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

        toolkit = Toolkit(
            ["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_market_value_added().loc["AAPL"]
        ```

        Which returns:

        |                        |        2021 |        2022 |        2023 |        2024 |        2025 |
        |:-----------------------|------------:|------------:|------------:|------------:|------------:|
        | Market Value of Equity | 2.92522e+12 | 2.08399e+12 | 3.0077e+12  | 3.83076e+12 | 4.06807e+12 |
        | Market Value of Debt   | 1.36522e+11 | 1.3248e+11  | 1.2393e+11  | 1.19059e+11 | 1.12377e+11 |
        | Invested Capital       | 1.93614e+11 | 1.91382e+11 | 1.84614e+11 | 1.81042e+11 | 1.8106e+11  |
        | Market Value Added     | 2.86813e+12 | 2.02509e+12 | 2.94702e+12 | 3.76878e+12 | 3.99939e+12 |
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

        # A market value is a point-in-time quantity, so the closing balance of Total Debt is used as its proxy — not the averaged balance that Invested Capital is built from, which would otherwise cancel the debt out of the Market Value Added entirely  # noqa: E501
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
            concat_frames(mva)
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

    @handle_errors
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

        toolkit = Toolkit(
            ["AAPL", "TSLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_intrinsic_valuation(0.05, 0.025, 0.094).loc["AAPL"]
        ```

        Which returns:

        |                      |   Periods = 5 |
        |:---------------------|--------------:|
        | Terminal Value       |   1.87255e+12 |
        | Cash Flow Projection |   1.26055e+11 |
        | Enterprise Value     |   1.63229e+12 |
        | Equity Value         |   1.55585e+12 |
        | Intrinsic Value      | 103.691       |
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
            growth_rate_dict = dict.fromkeys(self._tickers, growth_rate)

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
            perpetual_growth_rate_dict = dict.fromkeys(
                self._tickers, perpetual_growth_rate
            )

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
            wacc_dict = dict.fromkeys(self._tickers, weighted_average_cost_of_capital)

        intrinsic_values_dict = {}
        for ticker in self._tickers:
            perpetual_growth_rate_float = perpetual_growth_rate_dict[ticker]
            growth_rate_float = growth_rate_dict[ticker]
            weighted_average_cost_of_capital_float = wacc_dict[ticker]
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
            intrinsic_values_dict[ticker] = intrinsic_values_dict[ticker].pipe(
                apply_rounding, rounding if rounding is not None else self._rounding
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

        toolkit = Toolkit(
            ["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_free_cash_flow_to_firm().loc["AAPL"]
        ```

        Which returns:

        |                                  |        2021 |         2022 |        2023 |        2024 |       2025 |
        |:---------------------------------|------------:|-------------:|------------:|------------:|-----------:|
        | Net Operating Profit After Taxes | 9.69732e+10 |  1.02259e+11 | 1.00349e+11 |  9.3736e+10 | 1.1201e+11 |
        | Depreciation and Amortization    | 1.1284e+10  |  1.1104e+10  | 1.1519e+10  |  1.1445e+10 | 1.1698e+10 |
        | Capital Expenditure              | 1.1085e+10  |  1.0708e+10  | 1.0959e+10  |  9.447e+09  | 1.2715e+10 |
        | Change in Net Working Capital    | 4.911e+09   | -1.2e+09     | 6.577e+09   | -3.651e+09  | 2.5e+10    |
        | Free Cash Flow to Firm           | 9.22612e+10 |  1.03855e+11 | 9.43321e+10 |  9.9385e+10 | 8.5993e+10 |
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
            concat_frames(fcff)
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

        toolkit = Toolkit(
            ["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_free_cash_flow_to_equity().loc["AAPL"]
        ```

        Which returns:

        |                               |        2021 |         2022 |        2023 |        2024 |        2025 |
        |:------------------------------|------------:|-------------:|------------:|------------:|------------:|
        | Net Income                    | 9.468e+10   |  9.9803e+10  |  9.6995e+10 |  9.3736e+10 |  1.1201e+11 |
        | Depreciation and Amortization | 1.1284e+10  |  1.1104e+10  |  1.1519e+10 |  1.1445e+10 |  1.1698e+10 |
        | Capital Expenditure           | 1.1085e+10  |  1.0708e+10  |  1.0959e+10 |  9.447e+09  |  1.2715e+10 |
        | Change in Net Working Capital | 4.911e+09   | -1.2e+09     |  6.577e+09  | -3.651e+09  |  2.5e+10    |
        | Net Borrowing                 | 1.2665e+10  | -1.23e+08    | -9.901e+09  | -5.998e+09  | -8.483e+09  |
        | Free Cash Flow to Equity      | 1.02633e+11 |  1.01276e+11 |  8.1077e+10 |  9.3387e+10 |  7.751e+10  |
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
            concat_frames(fcfe)
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

    @handle_errors
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

        toolkit = Toolkit(
            ["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_gorden_growth_model(0.20, 0.05)
        ```

        Which returns:

        |      |   AAPL |    MSFT |
        |:-----|-------:|--------:|
        | 2021 | 6.055  | 16.1    |
        | 2022 | 6.37   | 17.78   |
        | 2023 | 6.65   | 19.53   |
        | 2024 | 6.93   | 21.56   |
        | 2025 | 7.21   | 23.8    |
        | 2026 | 7.5705 | 24.99   |
        | 2027 | 7.949  | 26.2395 |
        | 2028 | 8.3465 | 27.5515 |
        | 2029 | 8.7638 | 28.929  |
        | 2030 | 9.202  | 30.3755 |
        """
        dividends_per_share = self._historical_data[
            "quarterly" if self._quarterly else "yearly"
        ]["Dividends"]

        gorden_growth_model: dict[str, dict[str, float]] = {}

        periods = pd.period_range(
            start=dividends_per_share.index[0],
            periods=project_periods + len(dividends_per_share.index),
            freq="Q" if self._quarterly else "Y",
        )

        for ticker in self._tickers:
            gorden_growth_model[ticker] = {}
            last_known_period = dividends_per_share.index[0]

            for period in periods:
                if period in dividends_per_share.index:
                    dividends_per_share_value = dividends_per_share.loc[period, ticker]
                    last_known_period = period
                else:
                    distance = periods.get_loc(period) - periods.get_loc(
                        last_known_period
                    )
                    dividends_per_share_value = (
                        dividends_per_share.loc[last_known_period, ticker]
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

        gorden_growth_model_df = apply_rounding(
            gorden_growth_model_df, rounding if rounding is not None else self._rounding
        )

        return gorden_growth_model_df.loc[self._start_date :]

    @handle_errors
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

        toolkit = Toolkit(
            ["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_two_stage_dividend_discount_model(0.10, 0.12, 0.03).loc["AAPL"]
        ```

        Which returns:

        |                                 |   High-Growth Periods = 5 |
        |:--------------------------------|--------------------------:|
        | Final High-Growth Dividend      |                    1.8152 |
        | High-Growth Phase Present Value |                    5.4378 |
        | Terminal Value                  |                   26.7095 |
        | Terminal Value Present Value    |                   16.5845 |
        | Intrinsic Value                 |                   22.0223 |
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
            rate_of_return_dict = dict.fromkeys(self._tickers, rate_of_return)

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
            high_growth_rate_dict = dict.fromkeys(self._tickers, high_growth_rate)

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
            stable_growth_rate_dict = dict.fromkeys(self._tickers, stable_growth_rate)

        two_stage_ddm_dict = {}
        for ticker in self._tickers:
            rate_of_return_float = rate_of_return_dict[ticker]
            high_growth_rate_float = high_growth_rate_dict[ticker]
            stable_growth_rate_float = stable_growth_rate_dict[ticker]
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
            two_stage_ddm_dict[ticker] = two_stage_ddm_dict[ticker].pipe(
                apply_rounding, rounding if rounding is not None else self._rounding
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

        toolkit = Toolkit(
            ["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_residual_income().loc["AAPL"]
        ```

        Which returns:

        |                      |          2021 |        2022 |        2023 |        2024 |        2025 |
        |:---------------------|--------------:|------------:|------------:|------------:|------------:|
        | Net Income           |   9.468e+10   |  9.9803e+10 | 9.6995e+10  | 9.3736e+10  | 1.1201e+11  |
        | Cost of Equity       | nan           | -0.2485     | 0.2874      | 0.2425      | 0.2117      |
        | Book Value of Equity |   6.42145e+10 |  5.6881e+10 | 5.6409e+10  | 5.9548e+10  | 6.53415e+10 |
        | Residual Income      | nan           |  1.1394e+11 | 8.07818e+10 | 7.92984e+10 | 9.81755e+10 |
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
            concat_frames(residual_income)
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
            show_columns (list[str] | None, optional): Restrict the result to these columns. Defaults to None,
                which shows all columns.

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

        toolkit = Toolkit(
            ["AAPL", "TSLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        altman_z_score = toolkit.models.get_altman_z_score()

        altman_z_score.loc["AAPL"]
        ```

        Which returns:

        |                                   |    2021 |    2022 |    2023 |    2024 |    2025 |
        |:----------------------------------|--------:|--------:|--------:|--------:|--------:|
        | Working Capital to Total Assets   |  0.0267 | -0.0527 | -0.0049 | -0.0641 | -0.0492 |
        | Retained Earnings to Total Assets |  0.0158 | -0.0087 | -0.0006 | -0.0525 | -0.0397 |
        | EBIT to Total Assets              |  0.3187 |  0.3459 |  0.3337 |  0.3383 |  0.3695 |
        | Market Value to Total Liabilities | 10.1601 |  6.8987 | 10.3558 | 12.4363 | 14.2485 |
        | Sales to Total Assets             |  1.0422 |  1.1179 |  1.0871 |  1.0714 |  1.1584 |
        | Altman Z-Score                    |  8.244  |  6.3233 |  8.3951 |  9.4993 | 10.8122 |
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
            concat_frames(altman_z_score)
            .dropna(axis=1, how="all")
            .swaplevel(0, 1)
            .reindex(self._tickers, level=0)
        )

        # The model was estimated on manufacturing firms. A company without any cost of
        # goods sold, such as a bank or an insurer, has no working capital or sales in its
        # sense, so its score (e.g. "distress") says nothing; the result says so.
        cost_of_goods_sold = self._income_statement.loc[:, "Cost of Goods Sold", :]

        if not_applicable := [
            ticker
            for ticker in cost_of_goods_sold.index
            # Most periods without, as a stray figure in one year does not make one.
            if (cost_of_goods_sold.loc[ticker].dropna() == 0).mean()
            >= 0.5  # noqa: PLR2004
        ]:
            logger.warning(
                "The Altman Z-Score does not apply to companies that report no cost of "
                "goods sold, as banks and insurers do: the model was estimated on "
                "manufacturing firms, so it is no measure of financial distress for %s.",
                ", ".join(not_applicable),
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

        toolkit = Toolkit(
            ["AAPL", "TSLA", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_piotroski_score().loc["AAPL"]
        ```

        Which returns:

        |                                     |   2022 |   2023 |   2024 |   2025 |
        |:------------------------------------|-------:|-------:|-------:|-------:|
        | Return on Assets Criteria           |      1 |      1 |      1 |      1 |
        | Operating Cashflow Criteria         |      1 |      1 |      1 |      1 |
        | Change in Return on Assets Criteria |      1 |      0 |      0 |      1 |
        | Accruals Criteria                   |      1 |      1 |      1 |      0 |
        | Change in Leverage Criteria         |      1 |      1 |      1 |      1 |
        | Change in Current Ratio Criteria    |      0 |      1 |      0 |      1 |
        | Number of Shares Criteria           |      1 |      1 |      1 |      1 |
        | Gross Margin Criteria               |      1 |      1 |      1 |      1 |
        | Asset Turnover Criteria             |      1 |      0 |      1 |      1 |
        | Piotroski Score                     |      8 |      7 |      7 |      8 |

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

        # Piotroski scales both the Return on Assets and the Cash Flow from Operations by the same asset base, which is what makes this signal reduce to the sign of accruals  # noqa: E501
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
            concat_frames(piotroski_score)
            .swaplevel(0, 1)
            .reindex(self._tickers, level=0)
        )

        # Every criterion is a boolean comparison, so a period for which the statements have not been reported yet compares NaN against a number, evaluates to False and scores a perfect zero — the worst possible F-Score — rather than being reported as missing. Mask those periods out; must cover every field feeding any of the nine criteria, not just ROA/CFO/accruals.  # noqa: E501
        reported = (
            total_assets.notna()
            & net_income.notna()
            & operating_cashflow.notna()
            & long_term_debt.notna()
            & current_assets.notna()
            & current_liabilities.notna()
            & revenue.notna()
            & cost_of_goods_sold.notna()
            & common_stock_issued.notna()
        )

        # Five criteria compare against the previous period; require it reported too, or NaN > x resolves False.
        reported_previous_period = reported.shift(1, axis=1)
        change_based_criteria = {
            "Change in Return on Assets Criteria",
            "Change in Leverage Criteria",
            "Change in Current Ratio Criteria",
            "Gross Margin Criteria",
            "Asset Turnover Criteria",
            "Piotroski Score",
        }

        def _broadcast(mask: pd.DataFrame) -> pd.DataFrame:
            return (
                mask.reindex(
                    index=piotroski_results.index.get_level_values(0),
                    columns=piotroski_results.columns,
                )
                .fillna(False)
                .astype(bool)
                .set_axis(piotroski_results.index)
            )

        reported = _broadcast(reported)
        reported_previous_period = _broadcast(reported_previous_period)

        needs_previous_period = piotroski_results.index.get_level_values(1).isin(
            change_based_criteria
        )
        full_mask = reported.copy()
        full_mask.loc[needs_previous_period] = (
            reported.loc[needs_previous_period]
            & reported_previous_period.loc[needs_previous_period]
        )

        piotroski_results = piotroski_results.astype(float).where(full_mask)

        piotroski_results = piotroski_results.dropna(axis=1, how="all")

        # Two columns are lost rather than one. The averaged total assets already make the first period NaN, so the criteria that compare against the previous period only become meaningful from the third period onwards. Keeping the second column would silently score those criteria 0 even when they improved.  # noqa: E501
        piotroski_results = piotroski_results[piotroski_results.columns[2:]]

        # The F-Score is a count of satisfied criteria, so report it as such whenever every remaining period is fully reported and the float cast above is no longer needed  # noqa: E501
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

        toolkit = Toolkit(
            ["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_beneish_m_score().loc["AAPL"]
        ```

        Which returns:

        |                                 |    2021 |    2022 |    2023 |    2024 |    2025 |
        |:--------------------------------|--------:|--------:|--------:|--------:|--------:|
        | Days Sales in Receivables Index |  1.0322 |  1.0975 |  1.0297 |  1.0647 |  1.0349 |
        | Gross Margin Index              |  0.9151 |  0.9647 |  0.9814 |  0.9551 |  0.9851 |
        | Asset Quality Index             |  1.1404 |  0.9841 |  0.9387 |  1.0389 |  0.9863 |
        | Sales Growth Index              |  1.3326 |  1.0779 |  0.972  |  1.0202 |  1.0643 |
        | Depreciation Index              |  1.0566 |  1.0635 |  0.9982 |  0.8725 |  1.0539 |
        | SGA Expenses Index              |  0.8279 |  1.0595 |  1.0222 |  1.026  |  0.9938 |
        | Leverage Index                  |  1.0608 |  1.0729 |  0.9516 |  1.0526 |  0.9455 |
        | Total Accruals to Total Assets  | -0.0267 | -0.0634 | -0.0384 | -0.0672 |  0.0015 |
        | Beneish M-Score                 | -2.2503 | -2.6691 | -2.6802 | -2.7611 | -2.3721 |
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
            concat_frames(beneish_m_score)
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

        toolkit = Toolkit(
            ["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        ohlson_o_score = toolkit.models.get_ohlson_o_score()

        ohlson_o_score.loc["AAPL"]
        ```

        Which returns:

        |                                            |    2021 |    2022 |    2023 |    2024 |    2025 |
        |:-------------------------------------------|--------:|--------:|--------:|--------:|--------:|
        | Log of Total Assets                        | 26.5841 | 26.589  | 26.5886 | 26.6231 | 26.6073 |
        | Total Liabilities to Total Assets          |  0.8203 |  0.8564 |  0.8237 |  0.844  |  0.7948 |
        | Working Capital to Total Assets            |  0.0267 | -0.0527 | -0.0049 | -0.0641 | -0.0492 |
        | Current Liabilities to Current Assets      |  0.9306 |  1.1372 |  1.0121 |  1.153  |  1.1195 |
        | Negative Equity Indicator                  |  0      |  0      |  0      |  0      |  0      |
        | Net Income to Total Assets                 |  0.2697 |  0.2829 |  0.2751 |  0.2568 |  0.3118 |
        | Funds from Operations to Total Liabilities |  0.3614 |  0.4044 |  0.3806 |  0.3839 |  0.3905 |
        | Negative Income Indicator                  |  0      |  0      |  0      |  0      |  0      |
        | Change in Net Income                       |  0.245  |  0.0263 | -0.0143 | -0.0171 |  0.0888 |
        | Ohlson O-Score                             | -8.5895 | -8.2408 | -8.4318 | -8.1898 | -8.7015 |
        | Ohlson Bankruptcy Probability              |  0.0002 |  0.0003 |  0.0002 |  0.0003 |  0.0002 |

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
            concat_frames(ohlson_o_score)
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

        toolkit = Toolkit(
            ["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        zmijewski_score = toolkit.models.get_zmijewski_score()

        zmijewski_score.loc["AAPL"]
        ```

        Which returns:

        |                                       |    2021 |    2022 |    2023 |    2024 |    2025 |
        |:--------------------------------------|--------:|--------:|--------:|--------:|--------:|
        | Net Income to Total Assets            |  0.2697 |  0.2829 |  0.2751 |  0.2568 |  0.3118 |
        | Total Liabilities to Total Assets     |  0.8203 |  0.8564 |  0.8237 |  0.844  |  0.7948 |
        | Current Assets to Current Liabilities |  1.0746 |  0.8794 |  0.988  |  0.8673 |  0.8933 |
        | Zmijewski Score                       | -0.8427 | -0.6955 | -0.8466 | -0.6486 | -1.1766 |
        | Zmijewski Bankruptcy Probability      |  0.1997 |  0.2434 |  0.1986 |  0.2583 |  0.1197 |
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
            concat_frames(zmijewski_score)
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

        toolkit = Toolkit(
            ["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_springate_score().loc["AAPL"]
        ```

        Which returns:

        |                                 |   2021 |    2022 |    2023 |    2024 |    2025 |
        |:--------------------------------|-------:|--------:|--------:|--------:|--------:|
        | Working Capital to Total Assets | 0.0267 | -0.0527 | -0.0049 | -0.0641 | -0.0492 |
        | EBIT to Total Assets            | 0.3187 |  0.3459 |  0.3337 |  0.3383 |  0.3695 |
        | EBT to Current Liabilities      | 0.8703 |  0.7735 |  0.7827 |  0.7001 |  0.8014 |
        | Sales to Total Assets           | 1.0422 |  1.1179 |  1.0871 |  1.0714 |  1.1584 |
        | Springate Score                 | 1.997  |  1.9655 |  1.9709 |  1.8632 |  2.0759 |
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
            concat_frames(springate_score)
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

        toolkit = Toolkit(
            ["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_grover_score().loc["AAPL"]
        ```

        Which returns:

        |                                 |   2021 |    2022 |    2023 |    2024 |    2025 |
        |:--------------------------------|-------:|--------:|--------:|--------:|--------:|
        | Working Capital to Total Assets | 0.0267 | -0.0527 | -0.0049 | -0.0641 | -0.0492 |
        | EBIT to Total Assets            | 0.3187 |  0.3459 |  0.3337 |  0.3383 |  0.3695 |
        | Return on Assets                | 0.2697 |  0.2829 |  0.2751 |  0.2568 |  0.3118 |
        | Grover Score                    | 1.1814 |  1.1432 |  1.1805 |  1.0988 |  1.2285 |
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
            concat_frames(grover_score)
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

        toolkit = Toolkit(
            ["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_fulmer_h_score().loc["AAPL"]
        ```

        Which returns:

        |                                      |    2021 |    2022 |    2023 |     2024 |     2025 |
        |:-------------------------------------|--------:|--------:|--------:|---------:|---------:|
        | Retained Earnings to Total Assets    |  0.0158 | -0.0087 | -0.0006 |  -0.0525 |  -0.0397 |
        | Sales to Total Assets                |  1.0422 |  1.1179 |  1.0871 |   1.0714 |   1.1584 |
        | EBT to Total Equity                  |  1.731  |  2.3505 |  1.8301 |   2.1683 |   1.8001 |
        | Cash Flow to Total Liabilities       |  0.3614 |  0.4044 |  0.3806 |   0.3839 |   0.3905 |
        | Debt to Total Assets                 |  0.3889 |  0.3756 |  0.3515 |   0.3262 |   0.3128 |
        | Current Liabilities to Total Assets  |  0.3575 |  0.4365 |  0.4121 |   0.4833 |   0.4611 |
        | Log of Tangible Total Assets         | 26.5841 | 26.589  | 26.5886 |  26.6231 |  26.6073 |
        | Working Capital to Total Liabilities |  0.0325 | -0.0615 | -0.006  |  -0.076  |  -0.0619 |
        | Log of EBIT to Interest Expense      |  3.7445 |  3.729  |  3.3985 | nan      | nan      |
        | Fulmer H-Score                       | 14.2755 | 14.329  | 14.0094 | nan      | nan      |
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
            concat_frames(fulmer_h_score)
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

        toolkit = Toolkit(
            ["AAPL", "TSLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_present_value_of_growth_opportunities()
        ```

        Which returns:

        | Date   |    AAPL |    TSLA |
        |:-------|--------:|--------:|
        | 2022   | 153.932 | 133.831 |
        | 2023   | 168.073 | 240.426 |
        | 2024   | 222.744 | 399.952 |
        | 2025   | 234.882 | 446.584 |
        """
        wacc = self.get_weighted_average_cost_of_capital(
            show_full_results=False, trailing=trailing
        )

        average_shares = (
            self._income_statement.loc[:, "Weighted Average Shares Diluted", :]
            if diluted
            else self._income_statement.loc[:, "Weighted Average Shares", :]
        )

        # Preferred Dividends Paid is reported on the cash flow statement using the cash-flow-impact convention (an outflow is negative), while the Earnings per Share formula subtracts a positive-magnitude figure — without the absolute value the preferred dividends would be added back to Net Income instead of deducted from it  # noqa: E501
        net_income = (
            self._income_statement.loc[:, "Net Income", :].T.rolling(trailing).sum().T
            if trailing
            else self._income_statement.loc[:, "Net Income", :]
        )

        if include_dividends:
            dividends = self._cash_flow_statement.loc[
                :, "Preferred Dividends Paid", :
            ].abs()
            preferred_dividends = (
                dividends.T.rolling(trailing).sum().T if trailing else dividends
            )
        else:
            preferred_dividends = 0
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
                rounding=rounding if rounding is not None else self._rounding,
                axis="index",
            )
            if standardize:
                pvgo = calculate_standardization(
                    dataset=pvgo,
                    rounding=rounding if rounding is not None else self._rounding,
                    axis="rows",
                )
            return pvgo.loc[self._start_date :]

        pvgo = apply_rounding(
            pvgo, rounding if rounding is not None else self._rounding
        )

        if standardize:
            pvgo = calculate_standardization(
                dataset=pvgo,
                rounding=rounding if rounding is not None else self._rounding,
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

        toolkit = Toolkit(
            ["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_sustainable_growth_rate()
        ```

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | AAPL | 1.2491 | 1.4937 | 1.4531 | 1.3183 | 1.4782 |
        | MSFT | 0.3438 | 0.354  | 0.282  | 0.2796 | 0.2541 |
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

        toolkit = Toolkit(
            ["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_internal_growth_rate()
        ```

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | AAPL | 0.3118 | 0.3183 | 0.3028 | 0.2801 | 0.3638 |
        | MSFT | 0.164  | 0.1853 | 0.1565 | 0.1677 | 0.1594 |
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

        toolkit = Toolkit(
            ["AAPL", "TSLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

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

    def _rows(self, statement: pd.DataFrame, row: str) -> pd.DataFrame:
        """A statement row per ticker (rows) and period (columns); NaN when not reported."""
        if row not in statement.index.get_level_values(1):
            return pd.DataFrame(
                float("nan"), index=self._tickers, columns=statement.columns
            )
        return (
            statement.loc[:, row, :]
            .reindex(self._tickers)
            .apply(pd.to_numeric, errors="coerce")
        )

    def _period_prices(self) -> pd.DataFrame:
        """
        The share price at the end of every statement period, per ticker, as traded (adjusted
        for splits, not for dividends): a market value is the shares outstanding at the
        price paid then, which the dividend-adjusted price understates for past periods.
        """
        columns = self._balance_sheet_statement.columns
        prices = self._historical_data["quarterly" if self._quarterly else "yearly"][
            "Close"
        ]
        prices = prices.reindex(columns=self._tickers)
        prices.index = prices.index.asfreq(columns.freq)
        return prices.reindex(columns).T

    def _latest_value(self, statement: pd.DataFrame, ticker: str, row: str) -> float:
        """The latest reported value of a statement row for a ticker, or NaN."""
        values = self._rows(statement, row).loc[ticker].dropna()
        return float(values.iloc[-1]) if not values.empty else float("nan")

    def _latest_price(self, ticker: str) -> float:
        """The latest daily share price of a ticker, as traded."""
        prices = self._historical_data["daily"]["Close"][ticker].dropna()
        return float(prices.iloc[-1]) if not prices.empty else float("nan")

    def _per_ticker(
        self, value: float | dict[str, float] | None, name: str
    ) -> dict[str, float | None]:
        """A rate that is one number for every ticker or a number per ticker."""
        if isinstance(value, dict):
            missing = [ticker for ticker in self._tickers if ticker not in value]
            if missing:
                raise ValueError(f"No {name} given for: {', '.join(missing)}.")
            return {ticker: float(value[ticker]) for ticker in self._tickers}
        return dict.fromkeys(self._tickers, None if value is None else float(value))

    def _resolve_wacc(
        self, weighted_average_cost_of_capital: float | dict[str, float] | None
    ) -> dict[str, float]:
        """
        The discount rate per ticker: as given, otherwise the latest calculated WACC. The WACC
        is a yearly rate, so on quarterly data its flows (such as the interest expense) are
        summed over four quarters, whatever the trailing of the valued cash flows.
        """
        rates = self._per_ticker(
            weighted_average_cost_of_capital, "weighted average cost of capital"
        )
        if any(rate is None for rate in rates.values()):
            calculated = self.get_weighted_average_cost_of_capital(
                show_full_results=False, trailing=4 if self._quarterly else None
            )
            for ticker in self._tickers:
                if rates[ticker] is None:
                    history = (
                        calculated.loc[ticker].dropna()
                        if ticker in calculated.index
                        else pd.Series(dtype=float)
                    )
                    rates[ticker] = (
                        float(history.iloc[-1]) if not history.empty else float("nan")
                    )
        return rates  # type: ignore[return-value]

    def _combine(self, results: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Per-period results as ticker and metric (rows) by period (columns)."""
        combined = concat_frames(results).swaplevel(0, 1)
        order = {name: position for position, name in enumerate(results)}
        combined = combined.sort_index(
            level=1, key=lambda names: names.map(order), sort_remaining=False
        )
        return combined.reindex(self._tickers, level=0)

    def _calendar(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        A ticker by period frame on every period from the first to the last, so that a
        period no statement reports is a NaN column and positions and rolling windows count
        calendar periods rather than reported ones.
        """
        columns = frame.columns
        if not isinstance(columns, pd.PeriodIndex) or columns.empty:
            return frame
        return frame.reindex(
            columns=pd.period_range(columns.min(), columns.max(), freq=columns.freq)
        )

    def _calendar_sum(
        self, statement: pd.DataFrame, row: str, trailing: int | None
    ) -> pd.DataFrame:
        """
        A statement row on the calendar periods, summed over `trailing` periods where set.
        Without min_periods every window must be complete, so a missing period leaves the
        sums that include it NaN instead of letting them span more periods.
        """
        frame = self._calendar(self._rows(statement, row))
        return frame.T.rolling(trailing).sum().T if trailing else frame

    def _latest_capital(self, ticker: str) -> dict[str, float]:
        """The latest cash, total debt and diluted shares of a ticker, as a DCF takes them."""
        return {
            "cash_and_cash_equivalents": self._latest_value(
                self._balance_sheet_statement, ticker, "Cash and Cash Equivalents"
            ),
            "total_debt": self._latest_value(
                self._balance_sheet_statement, ticker, "Total Debt"
            ),
            "shares_outstanding": self._latest_value(
                self._income_statement, ticker, "Weighted Average Shares Diluted"
            ),
        }

    def _period_risk_free_rate(self, columns: pd.PeriodIndex) -> pd.Series:
        """The risk free rate of every statement period."""
        risk_free = self._risk_free_rate_data[
            "quarterly" if self._quarterly else "yearly"
        ]["Adj Close"]
        risk_free.index = risk_free.index.asfreq(
            self._balance_sheet_statement.columns.freq
        )
        return risk_free.reindex(columns)

    def _valuation_cash_flows(
        self, cash_flow_type: str, trailing: int | None
    ) -> pd.DataFrame:
        """
        The cash flow a valuation projects, per ticker (rows) and calendar period
        (columns), summed over `trailing` periods where set. A missing period leaves the
        sums that include it NaN instead of letting them span more periods.

        The Free Cash Flow to Firm is built as in `get_free_cash_flow_to_firm` of the
        Intrinsic module, NOPAT + Depreciation and Amortization - Capital Expenditure -
        Change in Net Working Capital, with the Operating Income as EBIT: the cash flow to
        all providers of capital, which the WACC discounts into an enterprise value. The
        reported Free Cash Flow and Operating Cash Flow are after interest.
        """
        valid = ["Free Cash Flow to Firm", "Free Cash Flow", "Operating Cash Flow"]
        if cash_flow_type not in valid:
            raise ValueError(f"The cash flow type must be one of: {', '.join(valid)}.")

        def summed(statement: pd.DataFrame, row: str) -> pd.DataFrame:
            return self._calendar_sum(statement, row, trailing)

        if cash_flow_type != "Free Cash Flow to Firm":
            return summed(self._cash_flow_statement, cash_flow_type)

        effective_tax_rate = profitability_model.get_effective_tax_rate(
            income_tax_expense=summed(self._income_statement, "Income Tax Expense"),
            income_before_tax=summed(self._income_statement, "Income Before Tax"),
        )
        nopat = eva_model.get_net_operating_profit_after_taxes(
            ebit=summed(self._income_statement, "Operating Income"),
            effective_tax_rate=effective_tax_rate,
        )
        # The cash flow statement shows a use of cash as negative; the FCFF formula takes
        # the capital expenditure and the increase in working capital as positive amounts.
        return intrinsic_model.get_free_cash_flow_to_firm(
            net_operating_profit_after_taxes=nopat,
            depreciation_and_amortization=summed(
                self._income_statement, "Depreciation and Amortization"
            ),
            capital_expenditure=-summed(
                self._cash_flow_statement, "Capital Expenditure"
            ),
            change_in_net_working_capital=-summed(
                self._cash_flow_statement, "Change in Working Capital"
            ),
        )  # type: ignore[return-value]

    @handle_errors
    def get_reverse_discounted_cash_flow(
        self,
        perpetual_growth_rate: float | dict[str, float] = 0.025,
        weighted_average_cost_of_capital: float | dict[str, float] | None = None,
        periods: int = 5,
        cash_flow_type: str = "Free Cash Flow to Firm",
        trailing: int | None = None,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the growth rate the share price implies, a reverse Discounted Cash Flow
        (DCF). Instead of estimating growth to arrive at a value, it takes the current share
        price as given and solves for the yearly cash flow growth over the projection
        periods at which the DCF value per share equals that price. The answer is the
        market's expectation, which can then be compared with the company's history: an
        implied growth far above what the company has achieved signals optimism.

        The DCF is the same as in `get_intrinsic_valuation`, by default on the Free Cash
        Flow to the Firm:

            - Cash Flow Projection_t = Cash Flow_t-1 * (1 + Implied Growth Rate)
            - Terminal Value = Last Projection * (1 + Perpetual Growth Rate) / (WACC - Perpetual Growth Rate)
            - Equity Value = Present Value of Projections and Terminal Value + Cash - Total Debt
            - Implied Growth Rate solves Equity Value / Shares Outstanding = Share Price

        For more information, see the following book:

        - Mauboussin, M.J. and Rappaport, A. (2001). "Expectations Investing: Reading Stock
          Prices for Better Returns." Harvard Business School Press.

        Also known as: reverse DCF, market-implied growth, expectations investing.

        Args:
            perpetual_growth_rate (float | dict[str, float], optional): The growth after the
                projection periods, one number or a number per ticker. Defaults to 0.025.
            weighted_average_cost_of_capital (float | dict[str, float] | None, optional):
                The discount rate, one number or a number per ticker. Defaults to None,
                which uses the latest Weighted Average Cost of Capital of each company,
                calculated over a year (the last 4 quarters with quarterly data) whatever
                `trailing` the cash flows use, as the rate is a yearly one.
            periods (int, optional): The number of projection periods. Defaults to 5.
            cash_flow_type (str, optional): The cash flow to project: "Free Cash Flow to
                Firm" (see `get_free_cash_flow_to_firm`), the cash flow to debt and equity
                holders that the WACC discounts into an enterprise value; or "Free Cash Flow"
                or "Operating Cash Flow" as reported, which are after interest (cash flows
                to equity), so that discounting them at the WACC and subtracting the debt
                counts the cost of debt twice. Defaults to "Free Cash Flow to Firm".
            trailing (int | None, optional): Sum the cash flow over this many periods (e.g.
                4 quarters) instead of taking the latest one. Defaults to None, which sums
                the last 4 quarters (a trailing twelve months) for quarterly data and takes
                the latest year otherwise, as the projection periods are years.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.

        Returns:
            pd.DataFrame: Per ticker the Share Price, the Cash Flow used, the Weighted
            Average Cost of Capital, the Perpetual Growth Rate, the Implied Growth Rate and,
            for comparison, the Historical Growth Rate (the compound annual growth of the
            cash flow as used, so summed over `trailing` periods where set, from the first
            to the latest reported period).

        Notes:
        - A negative cash flow, or a discount rate at or below the perpetual growth rate,
          has no implied growth rate and gives NaN.
        - The Historical Growth Rate counts four quarters as a year with quarterly data and
          is NaN when the first or latest cash flow is not positive. The years are those
          between the first and the latest cash flow on the calendar, and a trailing sum
          that includes a missing period is NaN rather than one over more periods.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "NVDA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_reverse_discounted_cash_flow(weighted_average_cost_of_capital=0.09)
        ```

        Which returns:

        |      |   Share Price |   Cash Flow |   Weighted Average Cost of Capital |   Perpetual Growth Rate |   Implied Growth Rate |   Historical Growth Rate |
        |:-----|--------------:|------------:|-----------------------------------:|------------------------:|----------------------:|-------------------------:|
        | AAPL |        271.86 | 8.62639e+10 |                               0.09 |                   0.025 |                0.308  |                   0.0545 |
        | MSFT |        483.62 | 7.0121e+10  |                               0.09 |                   0.025 |                0.333  |                   0.1183 |
        | NVDA |        186.5  | 9.15284e+10 |                               0.09 |                   0.025 |                0.3182 |                   0.8976 |
        """
        perpetual = self._per_ticker(perpetual_growth_rate, "perpetual growth rate")
        # The projection periods and the growth rates are yearly, so a quarterly cash flow is
        # summed into a year rather than projected as if it were one.
        if trailing is None and self._quarterly:
            trailing = 4
        periods_per_year = 4 if self._quarterly else 1
        wacc = self._resolve_wacc(weighted_average_cost_of_capital)
        cash_flows = self._valuation_cash_flows(cash_flow_type, trailing)

        rows = {}
        for ticker in self._tickers:
            # Summed on the calendar periods before the NaNs are dropped, so that a missing
            # period does not stretch a trailing sum over more periods than `trailing`.
            history = cash_flows.loc[ticker].dropna()
            if history.empty:
                continue
            cash_flow = float(history.iloc[-1])
            share_price = self._latest_price(ticker)
            implied = intrinsic_model.get_implied_growth_rate(
                share_price=share_price,
                cash_flow=cash_flow,
                perpetual_growth_rate=perpetual[ticker],  # type: ignore[arg-type]
                weighted_average_cost_of_capital=wacc[ticker],
                **self._latest_capital(ticker),
                periods=periods,
            )
            # The years between the first and the latest cash flow, counted on the calendar so
            # that a missing period in between does not shorten them.
            years = (
                (history.index[-1] - history.index[0]).n / periods_per_year
                if isinstance(history.index, pd.PeriodIndex)
                else (len(history) - 1) / periods_per_year
            )
            first = float(history.iloc[0])
            historical = (
                (cash_flow / first) ** (1 / years) - 1
                if years > 0 and first > 0 and cash_flow > 0
                else float("nan")
            )
            rows[ticker] = {
                "Share Price": share_price,
                "Cash Flow": cash_flow,
                "Weighted Average Cost of Capital": wacc[ticker],
                "Perpetual Growth Rate": perpetual[ticker],
                "Implied Growth Rate": implied,
                "Historical Growth Rate": historical,
            }

        return apply_rounding(
            pd.DataFrame(rows).T, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_earnings_power_value(
        self,
        weighted_average_cost_of_capital: float | dict[str, float] | None = None,
        periods: int = 5,
        trailing: int | None = None,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Earnings Power Value (EPV), the value of a company's current earnings
        if they never grew. Bruce Greenwald uses it to separate what is known (today's
        earnings power) from what is speculative (growth): the operating margin is averaged
        over several years and applied to the latest yearly revenue, taxed, and capitalized
        at the cost of capital.

            - Normalized Operating Income = Average Operating Margin * Latest Revenue
            - NOPAT = Normalized Operating Income * (1 - Average Tax Rate)
            - Earnings Power Value = NOPAT / WACC
            - EPV per Share = (EPV + Cash - Total Debt) / Shares Outstanding

        A share price well above the EPV per share means the market pays for growth, which
        only adds value when the company earns more than its cost of capital on it.

        For more information, see the following book:

        - Greenwald, B.C.N., Kahn, J., Sonkin, P.D. and van Biema, M. (2001). "Value
          Investing: From Graham to Buffett and Beyond." Wiley.

        Also known as: EPV, no-growth value, Greenwald valuation.

        Args:
            weighted_average_cost_of_capital (float | dict[str, float] | None, optional):
                The discount rate, one number or a number per ticker. Defaults to None,
                which uses the latest Weighted Average Cost of Capital of each company,
                calculated over a year (the last 4 quarters with quarterly data) whatever
                `trailing` the cash flows use, as the rate is a yearly one.
            periods (int, optional): The number of periods the operating margin and tax rate
                are averaged over; with `trailing`, the number of trailing sums, `trailing`
                calendar periods apart so that they do not overlap, counted back from the
                latest reported one. A sum that includes a missing period is left out of
                the average. Defaults to 5.
            trailing (int | None, optional): Sum the revenue, operating income, taxes and
                income before tax over this many periods (e.g. 4 quarters), so that the
                margin is applied to a year of revenue. Defaults to None, which uses 4 (a
                trailing twelve months) for quarterly data and the yearly figures otherwise.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.

        Returns:
            pd.DataFrame: Per ticker the Average Operating Margin, Normalized Operating
            Income, Tax Rate, Normalized NOPAT, Weighted Average Cost of Capital, Earnings
            Power Value, Equity Value, Earnings Power Value per Share, Share Price and the
            Price to EPV ratio.

        Notes:
        - Maintenance capital expenditure is assumed to equal depreciation, so operating
          income stands in for the cash the business could pay out without shrinking.
        - With quarterly data and the default trailing twelve months, the margins of the
          last five years are averaged and applied to the revenue of the last four quarters,
          as with yearly data.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "KO"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_earnings_power_value(weighted_average_cost_of_capital=0.08)
        ```

        Which returns:

        |      |   Average Operating Margin |   Normalized Operating Income |   Tax Rate |   Normalized NOPAT |   Weighted Average Cost of Capital |   Earnings Power Value |   Equity Value |   Earnings Power Value per Share |   Share Price |   Price to EPV |
        |:-----|---------------------------:|------------------------------:|-----------:|-------------------:|-----------------------------------:|-----------------------:|---------------:|---------------------------------:|--------------:|---------------:|
        | AAPL |                     0.3067 |                   1.27656e+11 |     0.1679 |        1.06229e+11 |                               0.08 |            1.32786e+12 |    1.25141e+12 |                          83.4015 |        271.86 |         3.2597 |
        | KO   |                     0.2534 |                   1.21476e+10 |     0.1861 |        9.88662e+09 |                               0.08 |            1.23583e+11 |    8.66388e+10 |                          20.0878 |         69.91 |         3.4802 |
        | MSFT |                     0.4314 |                   1.2153e+11  |     0.1636 |        1.01652e+11 |                               0.08 |            1.27066e+12 |    1.18871e+12 |                         159.238  |        483.62 |         3.0371 |
        """
        # The NOPAT is capitalized as a yearly amount, so quarterly figures are summed into a
        # year (a trailing twelve months) before the margin is applied to the revenue.
        if trailing is None and self._quarterly:
            trailing = 4
        wacc = self._resolve_wacc(weighted_average_cost_of_capital)

        # On the calendar periods, so that the steps below count calendar periods.
        revenue, operating_income, taxes, income_before_tax = (
            self._calendar_sum(self._income_statement, row, trailing)
            for row in (
                "Revenue",
                "Operating Income",
                "Income Tax Expense",
                "Income Before Tax",
            )
        )

        rows = {}
        for ticker in self._tickers:
            reported = (
                revenue.loc[ticker].notna() & operating_income.loc[ticker].notna()
            )
            if not reported.any():
                continue
            # Every `trailing`-th calendar period counted back from the latest reported one,
            # so that the averaged margins cover separate years instead of overlapping
            # twelve-month windows, keeping those reported: a missing period drops its year
            # rather than shifting the steps onto overlapping windows.
            latest = int(reported.to_numpy().nonzero()[0][-1])
            positions = list(range(latest, -1, -(trailing or 1)))[:periods][::-1]
            columns = revenue.columns[positions][reported.iloc[positions].to_numpy()]
            if columns.empty:
                continue
            result = intrinsic_model.get_earnings_power_value(
                revenue=revenue.loc[ticker, columns],
                operating_income=operating_income.loc[ticker, columns],
                income_tax_expense=taxes.loc[ticker, columns],
                income_before_tax=income_before_tax.loc[ticker, columns],
                weighted_average_cost_of_capital=wacc[ticker],
                **self._latest_capital(ticker),
            )
            share_price = self._latest_price(ticker)
            result = pd.concat(
                [
                    result.iloc[:4],
                    pd.Series({"Weighted Average Cost of Capital": wacc[ticker]}),
                    result.iloc[4:],
                    pd.Series(
                        {
                            "Share Price": share_price,
                            "Price to EPV": share_price
                            / result["Earnings Power Value per Share"],
                        }
                    ),
                ]
            )
            rows[ticker] = result

        return apply_rounding(
            pd.DataFrame(rows).T, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_monte_carlo_intrinsic_valuation(
        self,
        growth_rate: float | dict[str, float],
        growth_rate_volatility: float = 0.02,
        perpetual_growth_rate: float | dict[str, float] = 0.025,
        perpetual_growth_rate_volatility: float = 0.005,
        weighted_average_cost_of_capital: float | dict[str, float] | None = None,
        weighted_average_cost_of_capital_volatility: float = 0.01,
        periods: int = 5,
        simulations: int = 10000,
        cash_flow_type: str = "Free Cash Flow to Firm",
        trailing: int | None = None,
        seed: int | None = 0,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the distribution of the intrinsic value per share when the inputs of the
        Discounted Cash Flow model are uncertain. A single DCF value hides how sensitive it
        is to its assumptions; here the growth rate, the perpetual growth rate and the
        discount rate are drawn from normal distributions around their expected values in
        every simulation, and the DCF of `get_intrinsic_valuation` is calculated for each
        draw. The spread of the outcomes, and the probability that the value exceeds the
        share price, show how robust the conclusion is.

        Draws in which the discount rate does not exceed the perpetual growth rate by at
        least half a percentage point are redrawn, as the terminal value is undefined there.

        For more information, see the following book:

        - Damodaran, A. (2012). "Investment Valuation: Tools and Techniques for Determining
          the Value of Any Asset." Wiley, 3rd edition, chapter 33 (probabilistic approaches).

        Also known as: Monte Carlo DCF, probabilistic valuation, DCF simulation.

        Args:
            growth_rate (float | dict[str, float]): The expected yearly growth over the
                projection periods, one number or a number per ticker.
            growth_rate_volatility (float, optional): The standard deviation of the growth
                rate. Defaults to 0.02.
            perpetual_growth_rate (float | dict[str, float], optional): The expected growth
                after the projection periods. Defaults to 0.025.
            perpetual_growth_rate_volatility (float, optional): Its standard deviation.
                Defaults to 0.005.
            weighted_average_cost_of_capital (float | dict[str, float] | None, optional):
                The expected discount rate. Defaults to None, which uses the latest Weighted
                Average Cost of Capital of each company, calculated over a year (the last 4
                quarters with quarterly data) whatever `trailing` the cash flows use, as the
                rate is a yearly one.
            weighted_average_cost_of_capital_volatility (float, optional): Its standard
                deviation. Defaults to 0.01.
            periods (int, optional): The number of projection periods. Defaults to 5.
            simulations (int, optional): The number of simulations. Defaults to 10,000.
            cash_flow_type (str, optional): The cash flow to project: "Free Cash Flow to
                Firm" (see `get_free_cash_flow_to_firm`), the cash flow to debt and equity
                holders that the WACC discounts into an enterprise value; or "Free Cash Flow"
                or "Operating Cash Flow" as reported, which are after interest (cash flows
                to equity), so that discounting them at the WACC and subtracting the debt
                counts the cost of debt twice. Defaults to "Free Cash Flow to Firm".
            trailing (int | None, optional): Sum the cash flow over this many periods (e.g.
                4 quarters) instead of taking the latest one. Defaults to None, which sums
                the last 4 quarters (a trailing twelve months) for quarterly data and takes
                the latest year otherwise, as the projection periods are years.
            seed (int | None, optional): The seed of the random numbers, so that results
                can be reproduced. Defaults to 0; None gives different draws every time.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.

        Returns:
            pd.DataFrame: Per ticker the Mean, Standard Deviation and the 5th, 25th, 50th,
            75th and 95th Percentile of the value per share, the Share Price and the
            Probability Undervalued (the share of simulations above the share price).

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_monte_carlo_intrinsic_valuation(
            growth_rate=0.08, weighted_average_cost_of_capital=0.09
        )
        ```

        Which returns:

        |      |    Mean |   Standard Deviation |   5th Percentile |   25th Percentile |   50th Percentile |   75th Percentile |   95th Percentile |   Share Price |   Probability Undervalued |
        |:-----|--------:|---------------------:|-----------------:|------------------:|------------------:|------------------:|------------------:|--------------:|--------------------------:|
        | AAPL | 113.466 |              24.52   |          82.0179 |           96.6542 |           109.564 |           125.871 |           158.283 |        271.86 |                    0.0004 |
        | MSFT | 182.735 |              40.0623 |         131.353  |          155.267  |           176.361 |           203.002 |           255.96  |        483.62 |                    0.0002 |
        """
        growth = self._per_ticker(growth_rate, "growth rate")
        perpetual = self._per_ticker(perpetual_growth_rate, "perpetual growth rate")
        # The growth rates are yearly, so a quarterly cash flow is summed into a year.
        if trailing is None and self._quarterly:
            trailing = 4
        wacc = self._resolve_wacc(weighted_average_cost_of_capital)
        cash_flows = self._valuation_cash_flows(cash_flow_type, trailing)

        rows = {}
        for ticker in self._tickers:
            # Summed on the calendar periods before the NaNs are dropped, so that a missing
            # period does not stretch a trailing sum over more periods than `trailing`.
            history = cash_flows.loc[ticker].dropna()
            if history.empty:
                continue
            values = intrinsic_model.get_monte_carlo_intrinsic_value(
                cash_flow=float(history.iloc[-1]),
                growth_rate=growth[ticker],  # type: ignore[arg-type]
                growth_rate_volatility=growth_rate_volatility,
                perpetual_growth_rate=perpetual[ticker],  # type: ignore[arg-type]
                perpetual_growth_rate_volatility=perpetual_growth_rate_volatility,
                weighted_average_cost_of_capital=wacc[ticker],
                weighted_average_cost_of_capital_volatility=weighted_average_cost_of_capital_volatility,
                **self._latest_capital(ticker),
                periods=periods,
                simulations=simulations,
                seed=seed,
            )
            share_price = self._latest_price(ticker)
            percentiles = pd.Series(values).quantile([0.05, 0.25, 0.5, 0.75, 0.95])
            rows[ticker] = {
                "Mean": float(values.mean()),
                "Standard Deviation": float(values.std()),
                "5th Percentile": percentiles[0.05],
                "25th Percentile": percentiles[0.25],
                "50th Percentile": percentiles[0.5],
                "75th Percentile": percentiles[0.75],
                "95th Percentile": percentiles[0.95],
                "Share Price": share_price,
                "Probability Undervalued": (
                    float((values > share_price).mean())
                    if pd.notna(values).any()
                    else float("nan")
                ),
            }

        return apply_rounding(
            pd.DataFrame(rows).T, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_merton_distance_to_default(
        self,
        time_horizon: float = 1.0,
        rounding: int | None = None,
        show_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the distance to default and probability of default with the Merton
        (1974) structural credit model. Equity is seen as a call option on the company's
        assets with its debt as strike price: shareholders receive what is left after the
        debt is repaid. The market value and volatility of the assets, which cannot be
        observed, are solved from the market value and volatility of the equity:

            - E = V N(d1) - D e^(-rT) N(d2)
            - σ_E E = N(d1) σ_V V
            - Distance to Default = (ln(V/D) + (r - σ_V²/2) T) / (σ_V √T)
            - Probability of Default = N(-Distance to Default)

        where E is the market capitalization, σ_E the volatility of the share price over the
        year before, D the default point (short-term debt plus half of the long-term debt,
        the KMV convention), r the risk free rate and T the time horizon. The distance to
        default is the number of standard deviations the asset value is above the default
        point; it reacts to the share price immediately, long before a rating changes.

        The risk free rate is the Toolkit's (the 10-year Treasury yield unless set otherwise
        with `risk_free_rate`) at every period end, where Vassalou and Xing (2004) and
        Bharath and Shumway (2008) use the one-year rate for a one-year horizon; the
        difference moves the distance to default only slightly.

        For more information, see the following papers:

        - Merton, R.C. (1974). "On the Pricing of Corporate Debt: The Risk Structure of
          Interest Rates." The Journal of Finance, 29(2), 449-470.
        - Crosbie, P. and Bohn, J. (2003). "Modeling Default Risk." Moody's KMV.
        - Vassalou, M. and Xing, Y. (2004). "Default Risk in Equity Returns." The Journal of
          Finance, 59(2), 831-868.
        - Bharath, S.T. and Shumway, T. (2008). "Forecasting Default with the Merton Distance
          to Default Model." The Review of Financial Studies, 21(3), 1339-1369.

        Also known as: KMV model, structural credit model, expected default frequency.

        Args:
            time_horizon (float, optional): The horizon in years over which default is
                measured. Defaults to 1.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            show_columns (list[str] | None, optional): The periods to show. Defaults to None,
                all periods.

        Returns:
            pd.DataFrame: Per ticker and period the Equity Value, Equity Volatility, Default
            Point, Asset Value, Asset Volatility, Distance to Default and Probability of
            Default (a decimal).

        Notes:
        - The probability is the model's risk-neutral probability; it ranks companies by
          credit risk well but tends to understate the actual default rate of safe ones.
        - Financial companies, whose debt is their raw material, are outside the model's
          intended use.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["F", "AAPL"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_merton_distance_to_default().loc["F"]
        ```

        Which returns:

        |                        |        2021 |        2022 |        2023 |        2024 |        2025 |
        |:-----------------------|------------:|------------:|------------:|------------:|------------:|
        | Equity Value           | 8.37862e+10 | 4.66828e+10 | 4.92598e+10 | 3.98079e+10 | 5.22045e+10 |
        | Equity Volatility      | 0.4163      | 0.491       | 0.355       | 0.36        | 0.3319      |
        | Default Point          | 9.3892e+10  | 9.45665e+10 | 9.945e+10   | 1.07294e+11 | 1.1133e+11  |
        | Asset Value            | 1.76267e+11 | 1.3761e+11  | 1.44934e+11 | 1.42307e+11 | 1.58997e+11 |
        | Asset Volatility       | 0.1979      | 0.1675      | 0.1207      | 0.1008      | 0.109       |
        | Distance to Default    | 3.1592      | 2.3878      | 3.3814      | 3.2062      | 3.5969      |
        | Probability of Default | 0.0008      | 0.0085      | 0.0004      | 0.0007      | 0.0002      |
        """
        balance = self._balance_sheet_statement
        shares = self._rows(self._income_statement, "Weighted Average Shares Diluted")
        equity_value = (self._period_prices() * shares).reindex(
            index=self._tickers, columns=balance.columns
        )
        default_point = self._rows(balance, "Short Term Debt").fillna(
            0
        ) + 0.5 * self._rows(balance, "Long Term Debt").fillna(0)

        # The volatility of the daily returns over the 252 trading days (at least 60) up to
        # every period end, annualized, whether the statements are yearly or quarterly.
        daily_returns = self._historical_data["daily"]["Return"].reindex(
            columns=self._tickers
        )
        daily_returns.index = daily_returns.index.to_timestamp()
        volatility = daily_returns.rolling(252, min_periods=60).std() * (252**0.5)
        period_end = balance.columns.to_timestamp(how="end")
        equity_volatility = (
            volatility.reindex(volatility.index.union(period_end))
            .ffill()
            .reindex(period_end)
        )
        equity_volatility.index = balance.columns
        equity_volatility = equity_volatility.T

        risk_free = self._period_risk_free_rate(balance.columns)

        model_metrics = [
            "Default Point",
            "Asset Value",
            "Asset Volatility",
            "Distance to Default",
            "Probability of Default",
        ]
        results = {
            "Equity Value": equity_value,
            "Equity Volatility": equity_volatility,
            **{
                metric: pd.DataFrame(
                    index=self._tickers, columns=balance.columns, dtype=float
                )
                for metric in model_metrics
            },
        }

        for ticker in self._tickers:
            for period in balance.columns:
                model = credit_model.get_merton_model(
                    equity_value=float(equity_value.loc[ticker, period]),
                    equity_volatility=float(equity_volatility.loc[ticker, period]),
                    default_point=float(default_point.loc[ticker, period]),
                    risk_free_rate=float(risk_free.get(period, float("nan"))),
                    time_horizon=time_horizon,
                )
                for metric in model_metrics:
                    results[metric].loc[ticker, period] = model[metric]

        merton = self._combine(results).dropna(axis=1, how="all")
        merton = apply_rounding(
            merton, rounding if rounding is not None else self._rounding
        )

        return filter_columns(
            merton.loc[:, self._start_date : self._end_date], show_columns
        )

    @handle_errors
    def get_synthetic_credit_rating(
        self,
        firm_type: str = "auto",
        rounding: int | None = None,
        show_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate a synthetic credit rating and the cost of debt that comes with it, from the
        interest coverage ratio (EBIT divided by interest expense), following Aswath
        Damodaran. Most companies have no traded bonds or rating of their own; the coverage
        ratio is the measure that explains ratings best, so it is mapped to the rating that
        companies with a similar ratio have, and to that rating's default spread over the
        risk free rate:

            - Interest Coverage Ratio = EBIT / Interest Expense
            - Rating and Default Spread from Damodaran's synthetic rating table
            - Pre-tax Cost of Debt = Risk Free Rate + Default Spread

        The table, with this year's default spreads, is read from Damodaran's website. Large
        and small companies use different coverage ranges, as smaller companies need a
        higher coverage for the same rating.

        For more information, see the following book and data:

        - Damodaran, A. (2012). "Investment Valuation." Wiley, 3rd edition, chapter 8.
        - Damodaran, A. "Ratings, Interest Coverage Ratios and Default Spread."
          https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/ratings.html

        Also known as: implied credit rating, interest coverage rating, cost of debt.

        Args:
            firm_type (str, optional): The table to use: "large" (market capitalization above
                five billion), "small" (smaller and riskier firms), "financial" (banks and
                other financial firms) or "auto", which picks large or small from the market
                capitalization of every period. Defaults to "auto".
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            show_columns (list[str] | None, optional): The periods to show. Defaults to None,
                all periods.

        Returns:
            pd.DataFrame: Per ticker and period the Interest Coverage Ratio, Rating, Default
            Spread, Risk Free Rate and Pre-tax Cost of Debt.

        Notes:
        - A company that reports an interest expense of zero gets the best rating; its
          coverage ratio, which is infinite, shows as NaN. A period without a reported
          interest expense gets no rating.
        - A coverage ratio below the lowest bound of the table gets the worst rating.
        - The market capitalization is compared with five billion in the reporting
          currency; for companies that do not report in US dollars, pass the firm type.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["F", "AAPL", "T"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_synthetic_credit_rating()
        ```

        Which returns:

        |                                     | 2021     | 2022    | 2023    | 2024    | 2025    |
        |:------------------------------------|:---------|:--------|:--------|:--------|:--------|
        | ('AAPL', 'Interest Coverage Ratio') | 41.1905  | 40.7496 | 29.062  | nan     | nan     |
        | ('AAPL', 'Rating')                  | Aaa/AAA  | Aaa/AAA | Aaa/AAA | Aaa/AAA | Aaa/AAA |
        | ('AAPL', 'Default Spread')          | 0.004    | 0.004   | 0.004   | 0.004   | 0.004   |
        | ('AAPL', 'Risk Free Rate')          | 0.0151   | 0.0388  | 0.0387  | 0.0457  | 0.0416  |
        | ('AAPL', 'Pre-tax Cost of Debt')    | 0.0191   | 0.0428  | 0.0427  | 0.0497  | 0.0456  |
        | ('F', 'Interest Coverage Ratio')    | 2.7915   | 5.0905  | 4.1306  | 4.5695  | 2.0191  |
        | ('F', 'Rating')                     | Baa2/BBB | A2/A    | A3/A-   | A2/A    | Ba2/BB  |
        | ('F', 'Default Spread')             | 0.0111   | 0.0078  | 0.0089  | 0.0078  | 0.0184  |
        | ('F', 'Risk Free Rate')             | 0.0151   | 0.0388  | 0.0387  | 0.0457  | 0.0416  |
        | ('F', 'Pre-tax Cost of Debt')       | 0.0262   | 0.0466  | 0.0476  | 0.0535  | 0.06    |
        | ('T', 'Interest Coverage Ratio')    | 3.8675   | -0.7472 | 3.7011  | 3.5969  | 3.6743  |
        | ('T', 'Rating')                     | A3/A-    | D2/D    | A3/A-   | A3/A-   | A3/A-   |
        | ('T', 'Default Spread')             | 0.0089   | 0.19    | 0.0089  | 0.0089  | 0.0089  |
        | ('T', 'Risk Free Rate')             | 0.0151   | 0.0388  | 0.0387  | 0.0457  | 0.0416  |
        | ('T', 'Pre-tax Cost of Debt')       | 0.024    | 0.2288  | 0.0476  | 0.0546  | 0.0505  |
        """
        firm_type = firm_type.lower()
        if firm_type not in ("auto", "large", "small", "financial"):
            raise ValueError(
                "The firm type must be 'auto', 'large', 'small' or 'financial'."
            )

        table = damodaran_model.get_synthetic_rating_table()
        if table.empty:
            raise ValueError(
                "Damodaran's synthetic rating table could not be retrieved."
            )

        ebit = self._rows(self._income_statement, "Operating Income")
        interest = self._rows(self._income_statement, "Interest Expense").abs()
        # Only a reported interest expense of zero means there is nothing to cover (an
        # infinite ratio, the best rating); a missing one leaves the ratio unknown.
        coverage = (
            (ebit / interest.where(interest > 0))
            .mask(interest == 0, float("inf"))
            .where(ebit.notna())
        )

        market_cap = self._period_prices() * self._rows(
            self._income_statement, "Weighted Average Shares Diluted"
        )

        risk_free = self._period_risk_free_rate(coverage.columns)

        rows = {}
        for ticker in self._tickers:
            for period in coverage.columns:
                ratio = coverage.loc[ticker, period]
                if pd.isna(ratio):
                    continue
                kind = firm_type
                if kind == "auto":
                    cap = market_cap.loc[ticker, period]
                    kind = (
                        "large"
                        if pd.notna(cap) and cap > credit_model.LARGE_FIRM_MARKET_CAP
                        else "small"
                    )
                rating, spread = credit_model.get_synthetic_rating(
                    float(ratio), table[table["Firm Type"] == kind.capitalize()]
                )
                rate = float(risk_free.get(period, float("nan")))
                rows[(ticker, period)] = {
                    "Interest Coverage Ratio": (
                        float(ratio) if ratio != float("inf") else float("nan")
                    ),
                    "Rating": rating,
                    "Default Spread": spread,
                    "Risk Free Rate": rate,
                    "Pre-tax Cost of Debt": rate + spread,
                }

        if not rows:
            raise ValueError("No interest coverage ratio could be calculated.")

        long = pd.DataFrame(rows).T
        numeric = [
            "Interest Coverage Ratio",
            "Default Spread",
            "Risk Free Rate",
            "Pre-tax Cost of Debt",
        ]
        long[numeric] = apply_rounding(
            long[numeric].astype(float),
            rounding if rounding is not None else self._rounding,
        )
        result = long.stack().unstack(level=1)
        result.index.names = [None, None]
        order = {name: position for position, name in enumerate(long.columns)}
        result = result.sort_index(
            level=1, key=lambda names: names.map(order), sort_remaining=False
        ).reindex(self._tickers, level=0)

        return filter_columns(
            result.loc[:, self._start_date : self._end_date], show_columns
        )

    @handle_errors
    def get_peer_comparison(
        self,
        peers: dict[str, list[str]] | None = None,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Compare every company with its peers on valuation, profitability, growth and
        leverage. For each metric it shows the company's latest value, the median and mean
        of its peers and the company's percentile among them (the share of peers with a
        lower value), which tells at a glance whether a company is expensive or cheap,
        and better or worse run, than comparable companies.

        The metrics are the Price-to-Earnings, EV-to-EBITDA, Price-to-Book and
        Price-to-Sales ratios, the Gross, Operating and Net Profit Margin, the Return on
        Equity and on Invested Capital, Revenue Growth, the Debt-to-Equity ratio and the
        Dividend Yield, all of the latest fiscal year.

        The Peer Mean of the four valuation multiples is the harmonic mean of the positive
        values, the standard average of multiples: the arithmetic mean is dominated by
        peers with earnings (or book value, or sales) close to zero, and a negative multiple
        says nothing about the price paid. The other metrics use the arithmetic mean.

        For more information, see the following papers:

        - Baker, M. and Ruback, R.S. (1999). "Estimating Industry Multiples." Working paper,
          Harvard University.
        - Liu, J., Nissim, D. and Thomas, J. (2002). "Equity Valuation Using Multiples."
          Journal of Accounting Research, 40(1), 135-172.

        Also known as: comparable companies analysis, comps, relative valuation, peer
        group analysis.

        Args:
            peers (dict[str, list[str]] | None, optional): The peers per ticker, e.g.
                {"AAPL": ["MSFT", "GOOGL"]}. Defaults to None, which uses the peers
                Financial Modeling Prep lists for each company (same sector and exchange,
                similar size).
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.

        Returns:
            pd.DataFrame: Per ticker and metric the Company value, the Peer Median and Peer
            Mean (harmonic for the valuation multiples), the Percentile (0 to 1) and the
            number of Peers with a value.

        Notes:
        - A high percentile is not always good: for valuation ratios it means expensive,
          for margins and returns it means strong.
        - Retrieving the peers' financial statements takes a few seconds per peer.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_peer_comparison()
        ```

        Which returns:

        |                                        |   Company |   Peer Median |   Peer Mean |   Percentile |   Peers |
        |:---------------------------------------|----------:|--------------:|------------:|-------------:|--------:|
        | ('MSFT', 'Price-to-Earnings')          |   29.2276 |       30.5776 |     27.5171 |          0.5 |      10 |
        | ('MSFT', 'EV-to-EBITDA')               |   20.7309 |       19.8087 |     17.9245 |          0.5 |      10 |
        | ('MSFT', 'Price-to-Book')              |    8.8365 |        7.2576 |      6.1536 |          0.6 |      10 |
        | ('MSFT', 'Price-to-Sales')             |   11.7803 |        6.1724 |      4.2807 |          0.9 |      10 |
        | ('MSFT', 'Gross Margin')               |    0.6794 |        0.6242 |      0.6164 |          0.6 |      10 |
        | ('MSFT', 'Operating Margin')           |    0.4678 |        0.1998 |      0.2226 |          1   |      10 |
        | ('MSFT', 'Net Profit Margin')          |    0.4031 |        0.1682 |      0.187  |          1   |      10 |
        | ('MSFT', 'Return on Equity')           |    0.3404 |        0.2392 |      0.287  |          0.6 |      10 |
        | ('MSFT', 'Return on Invested Capital') |    0.209  |        0.1001 |      0.1434 |          0.8 |      10 |
        | ('MSFT', 'Revenue Growth')             |    0.1779 |        0.1146 |      0.128  |          0.8 |      10 |
        | ('MSFT', 'Debt-to-Equity')             |    0.2912 |        0.3313 |      0.7928 |          0.5 |      10 |
        | ('MSFT', 'Dividend Yield')             |    0.0052 |        0.0088 |      0.0096 |          0.4 |      10 |
        """
        if self._peer_function is None:
            raise ValueError("Peer data is not available for this Toolkit.")

        result = self._peer_function(peers)
        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

    def _score_inputs(self, trailing: int | None) -> dict[str, pd.DataFrame]:
        """
        The statement rows the quality and manipulation scores use, per ticker. With
        `trailing`, the flows are summed over that many periods (4 quarters give a trailing
        twelve months) while the balance sheet items stay as reported at the period end, so
        that compared `trailing` periods back they give the change over the year and the
        assets at its start, as in the yearly statements the scores were estimated on.
        """

        def flow(statement: pd.DataFrame, row: str) -> pd.DataFrame:
            frame = self._rows(statement, row)
            return frame.T.rolling(trailing).sum().T if trailing else frame

        balance_rows = [
            "Total Assets",
            "Total Current Assets",
            "Total Current Liabilities",
            "Total Liabilities",
            "Cash and Short Term Investments",
            "Cash and Cash Equivalents",
            "Short Term Investments",
            "Long Term Investments",
            "Short Term Debt",
            "Long Term Debt",
            "Preferred Stock",
            "Accounts Receivable",
            "Inventory",
            "Property, Plant and Equipment",
            "Gross Property, Plant and Equipment",
            "Other Current Assets",
            "Tax Payables",
            "Total Debt",
            "Minority Interest",
        ]
        income_rows = [
            "Revenue",
            "Cost of Goods Sold",
            "Operating Income",
            "Net Income",
            "Research and Development Expenses",
            "Selling and Marketing Expenses",
            "Depreciation and Amortization",
        ]
        cash_rows = [
            "Operating Cash Flow",
            "Capital Expenditure",
            "Cash Flow from Investing",
            "Long Term Debt Issued",
            "Gross Long Term Debt Issued",
            "Common Stock Issued",
        ]
        inputs = {
            row: self._rows(self._balance_sheet_statement, row) for row in balance_rows
        }
        inputs.update({row: flow(self._income_statement, row) for row in income_rows})
        inputs.update({row: flow(self._cash_flow_statement, row) for row in cash_rows})
        return inputs

    def _finish_score(
        self,
        results: dict[str, pd.DataFrame],
        rounding: int | None,
        show_columns: list[str] | None,
        first_columns: int = 1,
        name: str | None = None,
    ) -> pd.DataFrame:
        """
        Combine score components, drop the periods without a comparison and round. When the
        statements hold no more periods than the comparison needs, the result is empty, and
        a warning says how many periods `start_date` must cover.
        """
        combined = self._combine(results)
        if name and len(combined.columns) <= first_columns:
            logger.warning(
                "%s needs at least %s %s of statements and %s were available, so it could "
                "not be calculated. Set an earlier start_date to obtain it.",
                name,
                first_columns + 1,
                "quarters" if self._quarterly else "years",
                len(combined.columns),
            )
        combined = combined[combined.columns[first_columns:]].dropna(axis=1, how="all")
        combined = apply_rounding(
            combined, rounding if rounding is not None else self._rounding
        )
        return filter_columns(
            combined.loc[:, self._start_date : self._end_date], show_columns
        )

    @handle_errors
    def get_magic_formula(
        self,
        diluted: bool = True,
        trailing: int | None = None,
        rounding: int | None = None,
        show_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate Joel Greenblatt's Magic Formula ranking. Companies are ranked on two
        measures and the ranks are added; the lowest total combines a good business with a
        low price:

            - Earnings Yield = EBIT / Enterprise Value (how cheap the company is)
            - Return on Capital = EBIT / (Net Working Capital + Net Fixed Assets) (how good
              the business is)

        Net working capital excludes cash and short-term debt, so the return on capital
        measures the operating business. The ranking is across the Toolkit's tickers, so
        pass the universe to choose from.

        For more information, see the following book:

        - Greenblatt, J. (2006). "The Little Book That Beats the Market." Wiley.

        Also known as: Greenblatt formula, earnings yield and return on capital ranking.

        Args:
            diluted (bool, optional): Whether to use the diluted share count for the market
                capitalization. Defaults to True.
            trailing (int | None, optional): Sum the operating income over this many
                periods (e.g. 4 quarters, a trailing twelve months). The enterprise value and
                the capital are taken at the period end. Defaults to None, which is 4 for
                quarterly data (a year of operating income) and 1 otherwise.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            show_columns (list[str] | None, optional): The periods to show. Defaults to None,
                all periods.

        Returns:
            pd.DataFrame: Per ticker and period the Earnings Yield, Return on Capital, their
            ranks and the Magic Formula Rank (1 is the most attractive).

        Notes:
        - With quarterly data and the default trailing twelve months, the first ranks are in
          the fourth quarter of statements. When the statements cover fewer periods, the
          result is empty and a warning names the number of periods needed; set an earlier
          `start_date` to obtain the ranks.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "GOOGL", "META", "INTC"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_magic_formula().xs("Magic Formula Rank", level=1)
        ```

        Which returns:

        |       |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:------|-------:|-------:|-------:|-------:|-------:|
        | AAPL  |      2 |      1 |      1 |      2 |      2 |
        | GOOGL |      4 |      2 |      2 |      1 |      4 |
        | INTC  |      3 |      5 |      5 |      5 |      5 |
        | META  |      1 |      2 |      2 |      4 |      2 |
        | MSFT  |      5 |      4 |      4 |      2 |      1 |
        """
        # The scores were estimated on yearly statements, so quarterly data is summed into a
        # year (a trailing twelve months) unless a trailing period is given.
        if trailing is None and self._quarterly:
            trailing = 4
        inputs = self._score_inputs(trailing)
        shares = self._rows(
            self._income_statement,
            "Weighted Average Shares Diluted" if diluted else "Weighted Average Shares",
        )
        market_cap = self._period_prices() * shares
        enterprise_value = (
            market_cap
            + inputs["Total Debt"].fillna(0)
            + inputs["Minority Interest"].fillna(0)
            + inputs["Preferred Stock"].fillna(0)
            - inputs["Cash and Cash Equivalents"].fillna(0)
        )
        net_working_capital = (
            inputs["Total Current Assets"] - inputs["Cash and Cash Equivalents"]
        ) - (inputs["Total Current Liabilities"] - inputs["Short Term Debt"].fillna(0))

        results = quality_model.get_magic_formula(
            operating_income=inputs["Operating Income"],
            enterprise_value=enterprise_value,
            net_working_capital=net_working_capital,
            net_fixed_assets=inputs["Property, Plant and Equipment"],
        )
        # The ranks need the operating income of a full trailing sum, so the periods before
        # the first one are left out (and the result is empty, with a warning, without one).
        return self._finish_score(
            results,
            rounding,
            show_columns,
            first_columns=(trailing - 1) if trailing else 0,
            name="The Magic Formula",
        )

    @handle_errors
    def get_mohanram_g_score(
        self,
        variance_periods: int = 5,
        trailing: int | None = None,
        rounding: int | None = None,
        show_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate Partha Mohanram's G-Score, a score from 0 to 8 that separates winners
        from losers among growth (low book-to-market) companies, where value measures such
        as the Piotroski F-Score say little. Each signal is 1 when the company does better
        than the median of its peers, the Toolkit's tickers:

        - Profitability: return on assets (G1) and cash flow return on assets (G2) above
          the median, and cash flow above net income (G3).
        - Naive extrapolation: a lower than median variance of the return on assets (G4)
          and of revenue growth (G5).
        - Accounting conservatism: research and development (G6), capital expenditure (G7)
          and advertising (G8, selling and marketing expenses) relative to assets above the
          median, as these investments depress current earnings.

        As in the paper, the return on assets, the cash flow return on assets and the three
        intensities are scaled by the total assets at the start of the year, the variances
        are taken over the past five years with at least three available, and a signal
        that cannot be calculated for a company (a missing item, or too short a history)
        scores 0.

        For more information, see the following paper:

        - Mohanram, P.S. (2005). "Separating Winners from Losers among Low Book-to-Market
          Stocks using Financial Statement Analysis." Review of Accounting Studies, 10,
          133-170.

        Also known as: G-Score, growth stock score.

        Args:
            variance_periods (int, optional): The number of yearly values the variances of
                G4 (return on assets) and G5 (revenue growth) are calculated over, of which at
                least three (or all, when fewer are asked for) must be available. With
                `trailing` the values are `trailing` periods apart (one a year with 4
                quarters); without it, one per period. Defaults to 5, the five years of the
                paper.
            trailing (int | None, optional): Sum the flows over this many periods (e.g. 4
                quarters, a trailing twelve months) and compare with the values `trailing`
                periods back, so that with quarterly data every change is year over year.
                Balance sheet items are taken at the period end. Defaults to None, which is
                4 for quarterly data (a year) and 1 otherwise; pass 1 for quarterly data to
                compare quarter over quarter.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            show_columns (list[str] | None, optional): The periods to show. Defaults to None,
                all periods.

        Returns:
            pd.DataFrame: Per ticker and period the eight signals and the G-Score.

        Notes:
        - The paper compares with industry medians; here the medians are those of the
          tickers given, so pass a peer group of the same industry.
        - The variances need three yearly values and revenue growth needs the year before
          the first, so the first G-Score is in the fourth year of statements (with
          quarterly data, the sixteenth quarter). A company with a shorter history than its
          peers scores 0 on G4 and G5. When the statements cover fewer periods than that,
          the result is empty and a warning names the number of periods needed; set an
          earlier `start_date` to obtain the score.
        - Net Income stands in for the net income before extraordinary items of the paper, as
          the statements do not report extraordinary items separately.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "ADBE"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_mohanram_g_score().xs("G-Score", level=1)
        ```

        Which returns:

        |       |   2023 |   2024 |   2025 |
        |:------|-------:|-------:|-------:|
        | AAPL  |      4 |      4 |      3 |
        | ADBE  |      5 |      5 |      5 |
        | GOOGL |      4 |      4 |      6 |
        | META  |      6 |      5 |      4 |
        | MSFT  |      4 |      5 |      4 |
        | NVDA  |      3 |      3 |      3 |
        """
        # The scores were estimated on yearly statements, so quarterly data is summed into a
        # year (a trailing twelve months) unless a trailing period is given.
        if trailing is None and self._quarterly:
            trailing = 4
        inputs = self._score_inputs(trailing)
        lag = trailing or 1
        results = quality_model.get_mohanram_g_score(
            net_income=inputs["Net Income"],
            operating_cash_flow=inputs["Operating Cash Flow"],
            total_assets=inputs["Total Assets"],
            revenue=inputs["Revenue"],
            research_and_development=inputs["Research and Development Expenses"],
            capital_expenditure=inputs["Capital Expenditure"],
            selling_and_marketing=inputs["Selling and Marketing Expenses"],
            variance_periods=variance_periods,
            lag=lag,
        )
        # Signals that cannot be calculated score 0, so the periods in which the variance of
        # revenue growth is not available for any company (the first trailing sums, plus the
        # minimum number of growth rates a year apart) are left out instead of shown as 0.
        first_columns = (lag - 1 if trailing else 0) + lag * min(
            quality_model.MOHANRAM_MINIMUM_VARIANCE_PERIODS, variance_periods
        )
        return self._finish_score(
            results,
            rounding,
            show_columns,
            first_columns=first_columns,
            name="The Mohanram G-Score",
        )

    @handle_errors
    def get_dechow_f_score(
        self,
        trailing: int | None = None,
        rounding: int | None = None,
        show_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Dechow F-Score, the probability that a company's financial statements
        are misstated relative to the average company. It was estimated on companies the
        SEC charged with misstating their earnings, and combines:

        - Accrual quality: RSST accruals (the change in non-cash working capital,
          non-current operating assets and financial assets), the changes in receivables and
          inventory, and the share of soft assets (assets other than cash and short-term
          investments and net property, plant and equipment).
        - Performance: the change in cash sales and in return on assets.
        - Incentives: whether the company issued long-term debt or stock.

            Predicted Value = -7.893 + 0.790 RSST Accruals + 2.518 ΔReceivables + 1.191
            ΔInventory + 1.979 Soft Assets + 0.171 ΔCash Sales - 0.932 ΔROA + 1.029 Issuance
            Probability = e^PV / (1 + e^PV), F-Score = Probability / 0.0037

        An F-Score of 1 is average risk; above 1 the risk is elevated, above 1.85
        substantial and above 2.45 high.

        For more information, see the following paper:

        - Dechow, P.M., Ge, W., Larson, C.R. and Sloan, R.G. (2011). "Predicting Material
          Accounting Misstatements." Contemporary Accounting Research, 28(1), 17-82.

        Also known as: F-Score (Dechow), misstatement score, earnings manipulation
        probability.

        Args:
            trailing (int | None, optional): Sum the flows over this many periods (e.g. 4
                quarters, a trailing twelve months) and compare with the values `trailing`
                periods back, so that with quarterly data every change is year over year.
                Balance sheet items are taken at the period end. Defaults to None, which is
                4 for quarterly data (a year) and 1 otherwise; pass 1 for quarterly data to
                compare quarter over quarter.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            show_columns (list[str] | None, optional): The periods to show. Defaults to None,
                all periods.

        Returns:
            pd.DataFrame: Per ticker and period the seven inputs, the Predicted Value, the
            Probability of Misstatement and the F-Score.

        Notes:
        - The paper counts the gross issuance of long-term debt. Where the statements only
          report the net issuance (issuance minus repayments), as Financial Modeling Prep's
          do, a company that issued debt but repaid more counts as not issuing.
        - The model was estimated on yearly statements, so with quarterly data the flows
          are summed over the trailing twelve months by default.
        - The changes compare with the year before and the change in return on assets with
          the return a year back, so the first F-Score is in the third year of statements
          (with quarterly data and the default trailing twelve months, the ninth quarter).
          When the statements cover fewer periods, the result is empty and a warning names
          the number of periods needed; set an earlier `start_date` to obtain the score.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "SMCI"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_dechow_f_score().xs("F-Score", level=1)
        ```

        Which returns:

        |      |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|
        | AAPL | 0.4226 | 0.3916 | 0.4136 | 0.4214 |
        | MSFT | 0.811  | 0.7759 | 1.018  | 0.8628 |
        | SMCI | 2.8035 | 0.6471 | 6.128  | 0.8275 |
        """
        # The scores were estimated on yearly statements, so quarterly data is summed into a
        # year (a trailing twelve months) unless a trailing period is given.
        if trailing is None and self._quarterly:
            trailing = 4
        inputs = self._score_inputs(trailing)
        lag = trailing or 1
        results = quality_model.get_dechow_f_score(
            total_assets=inputs["Total Assets"],
            current_assets=inputs["Total Current Assets"],
            current_liabilities=inputs["Total Current Liabilities"],
            total_liabilities=inputs["Total Liabilities"],
            cash_and_short_term_investments=inputs["Cash and Short Term Investments"],
            short_term_investments=inputs["Short Term Investments"],
            long_term_investments=inputs["Long Term Investments"],
            short_term_debt=inputs["Short Term Debt"],
            long_term_debt=inputs["Long Term Debt"],
            preferred_stock=inputs["Preferred Stock"],
            accounts_receivable=inputs["Accounts Receivable"],
            inventory=inputs["Inventory"],
            property_plant_and_equipment=inputs["Property, Plant and Equipment"],
            revenue=inputs["Revenue"],
            net_income=inputs["Net Income"],
            # The gross issuance where the statements report it (Yahoo Finance's do), the
            # net issuance otherwise.
            debt_issued=inputs["Gross Long Term Debt Issued"].fillna(
                inputs["Long Term Debt Issued"]
            ),
            stock_issued=inputs["Common Stock Issued"],
            lag=lag,
        )
        # The change in return on assets compares with the return a year back, which uses
        # the assets of the year before that, so the first full score is two years in.
        first_columns = 2 * lag
        return self._finish_score(
            results,
            rounding,
            show_columns,
            first_columns=first_columns,
            name="The Dechow F-Score",
        )

    @handle_errors
    def get_montier_c_score(
        self,
        trailing: int | None = None,
        rounding: int | None = None,
        show_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate James Montier's C-Score, a count from 0 to 6 of warning signs that a
        company is "cooking the books". Each sign is 1 when present:

        - A growing gap between net income and cash flow from operations.
        - Rising days sales outstanding: receivables growing faster than sales.
        - Rising days sales of inventory: inventory growing faster than cost of sales.
        - Rising other current assets relative to revenue, a place to park costs.
        - Falling depreciation relative to gross property, plant and equipment, which
          flatters earnings.
        - Total asset growth above 10%, typical of serial acquirers.

        Montier found that companies scoring 5 or 6 underperformed the market markedly.

        For more information, see the following paper:

        - Montier, J. (2008). "Cooking the Books, or, More Sailing Under the Black Flag."
          Société Générale, Mind Matters.

        Also known as: C-Score, cooking the books score.

        Args:
            trailing (int | None, optional): Sum the flows over this many periods (e.g. 4
                quarters, a trailing twelve months) and compare with the values `trailing`
                periods back, so that with quarterly data every change is year over year.
                Balance sheet items are taken at the period end. Defaults to None, which is
                4 for quarterly data (a year) and 1 otherwise; pass 1 for quarterly data to
                compare quarter over quarter.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            show_columns (list[str] | None, optional): The periods to show. Defaults to None,
                all periods.

        Returns:
            pd.DataFrame: Per ticker and period the six signals and the C-Score.

        Notes:
        - Gross property, plant and equipment is reported by Yahoo Finance but not by
          Financial Modeling Prep; for a company without it, depreciation is compared with
          the net amount, whose ratio also rises as the assets age and so can hide a decline.
        - The signs compare with the year before, so the first C-Score is in the second year
          of statements (with quarterly data and the default trailing twelve months, the
          eighth quarter). When the statements cover fewer periods, the result is empty and a
          warning names the number of periods needed; set an earlier `start_date` to obtain
          the score.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "SMCI"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_montier_c_score().xs("C-Score", level=1)
        ```

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | AAPL |      4 |      2 |      3 |      2 |      3 |
        | MSFT |      5 |      2 |      5 |      3 |      2 |
        | SMCI |      3 |      5 |      2 |      5 |      2 |
        """
        # The scores were estimated on yearly statements, so quarterly data is summed into a
        # year (a trailing twelve months) unless a trailing period is given.
        if trailing is None and self._quarterly:
            trailing = 4
        inputs = self._score_inputs(trailing)
        lag = trailing or 1
        # Montier measures depreciation against gross fixed assets. A company whose
        # statements have no gross amount uses the net amount in every period, as switching
        # between the two would show a change that is not there.
        gross = inputs["Gross Property, Plant and Equipment"]
        fixed_assets = gross.where(
            gross.notna().any(axis=1), inputs["Property, Plant and Equipment"], axis=0
        )
        results = quality_model.get_montier_c_score(
            net_income=inputs["Net Income"],
            operating_cash_flow=inputs["Operating Cash Flow"],
            accounts_receivable=inputs["Accounts Receivable"],
            revenue=inputs["Revenue"],
            inventory=inputs["Inventory"],
            cost_of_goods_sold=inputs["Cost of Goods Sold"],
            other_current_assets=inputs["Other Current Assets"],
            depreciation_and_amortization=inputs["Depreciation and Amortization"],
            property_plant_and_equipment=fixed_assets,
            total_assets=inputs["Total Assets"],
            lag=lag,
        )
        # The signs compare a full trailing sum with the one a year back.
        first_columns = (lag - 1 if trailing else 0) + lag
        return self._finish_score(
            results,
            rounding,
            show_columns,
            first_columns=first_columns,
            name="The Montier C-Score",
        )

    @handle_errors
    def get_accruals_ratio(
        self,
        trailing: int | None = None,
        rounding: int | None = None,
        show_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the accruals ratios: the part of earnings that is not backed by cash,
        relative to the average of the total assets at the start and the end of the year.
        Richard Sloan showed that earnings driven by accruals are less persistent than cash
        earnings and that companies with high accruals earn lower future returns (the
        accrual anomaly).

            - Balance Sheet Accruals Ratio = [(ΔCurrent Assets - ΔCash) - (ΔCurrent
              Liabilities - ΔShort-term Debt - ΔTaxes Payable) - Depreciation] / Average
              Total Assets
            - Cash Flow Accruals Ratio = (Net Income - Cash Flow from Operations - Cash Flow
              from Investing) / Average Total Assets

        The balance sheet version is Sloan's (1996). The cash flow version, earnings minus
        free cash flow, takes the accruals from the cash flow statement instead: Hribar and
        Collins (2002) showed that balance sheet changes misstate accruals when a company
        acquires, divests or discontinues operations (their own measure is earnings before
        extraordinary items minus cash flow from operations), and subtracting the investing
        cash flow as well counts the investment in long-term operating assets as an
        accrual, as the broad accruals of Richardson, Sloan, Soliman and Tuna (2005) do.
        Values near zero or negative are a sign of high earnings quality.

        For more information, see the following papers:

        - Sloan, R.G. (1996). "Do Stock Prices Fully Reflect Information in Accruals and
          Cash Flows about Future Earnings?" The Accounting Review, 71(3), 289-315.
        - Hribar, P. and Collins, D.W. (2002). "Errors in Estimating Accruals: Implications
          for Empirical Research." Journal of Accounting Research, 40(1), 105-134.
        - Richardson, S.A., Sloan, R.G., Soliman, M.T. and Tuna, I. (2005). "Accrual
          Reliability, Earnings Persistence and Stock Prices." Journal of Accounting and
          Economics, 39(3), 437-485.

        Also known as: Sloan ratio, accrual ratio, earnings quality ratio.

        Args:
            trailing (int | None, optional): Sum the flows over this many periods (e.g. 4
                quarters, a trailing twelve months) and compare with the values `trailing`
                periods back, so that with quarterly data every change is year over year.
                Balance sheet items are taken at the period end. Defaults to None, which is
                4 for quarterly data (a year) and 1 otherwise; pass 1 for quarterly data to
                compare quarter over quarter.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            show_columns (list[str] | None, optional): The periods to show. Defaults to None,
                all periods.

        Returns:
            pd.DataFrame: Per ticker and period the Balance Sheet Accruals Ratio and the Cash
            Flow Accruals Ratio.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_accruals_ratio()
        ```

        Which returns:

        |                                          |    2021 |    2022 |    2023 |    2024 |    2025 |
        |:-----------------------------------------|--------:|--------:|--------:|--------:|--------:|
        | ('AAPL', 'Balance Sheet Accruals Ratio') | -0.1047 | -0.0446 | -0.0115 | -0.0285 | -0.108  |
        | ('AAPL', 'Cash Flow Accruals Ratio')     |  0.0154 |  0      | -0.0489 | -0.0765 | -0.0405 |
        | ('MSFT', 'Balance Sheet Accruals Ratio') | -0.0687 | -0.1109 | -0.0683 | -0.1017 | -0.0735 |
        | ('MSFT', 'Cash Flow Accruals Ratio')     |  0.0381 |  0.0401 |  0.0192 |  0.144  |  0.0677 |
        """
        # The scores were estimated on yearly statements, so quarterly data is summed into a
        # year (a trailing twelve months) unless a trailing period is given.
        if trailing is None and self._quarterly:
            trailing = 4
        inputs = self._score_inputs(trailing)
        results = quality_model.get_accruals_ratios(
            current_assets=inputs["Total Current Assets"],
            cash_and_cash_equivalents=inputs["Cash and Cash Equivalents"],
            current_liabilities=inputs["Total Current Liabilities"],
            short_term_debt=inputs["Short Term Debt"],
            tax_payables=inputs["Tax Payables"],
            depreciation_and_amortization=inputs["Depreciation and Amortization"],
            net_income=inputs["Net Income"],
            operating_cash_flow=inputs["Operating Cash Flow"],
            investing_cash_flow=inputs["Cash Flow from Investing"],
            total_assets=inputs["Total Assets"],
            lag=trailing or 1,
        )
        # The balance sheet changes look a year back, by when the trailing sums are complete.
        return self._finish_score(
            results, rounding, show_columns, first_columns=trailing or 1
        )

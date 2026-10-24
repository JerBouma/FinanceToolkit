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
        return on equity (ROE). It breaks down the ROE into three key components: Net Profit Margin, Asset
        Turnover, and Equity Multiplier.

        The formula is as follows:

            - Net Profit Margin = Net Income / Revenue
            - Asset Turnover = Revenue / Average Total Assets
            - Equity Multiplier = Average Total Assets / Average Total Equity
            - ROE = Net Profit Margin * Asset Turnover * Equity Multiplier

        The value of the decomposition is in comparing the three factors with peers and over time: a
        retailer earns its return on equity through asset turnover, a software company through its margin
        and a bank through leverage. A rising return on equity driven only by the Equity Multiplier, for
        example through share buybacks that shrink equity, is of lower quality than one driven by margin or
        turnover, and with negative equity the decomposition stops being meaningful.

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
            pd.DataFrame: DataFrame containing Dupont analysis results, including Net Profit Margin, Asset
                          Turnover, Equity Multiplier, and the calculated ROE values.

        Notes:
            - The Net Profit Margin is the ratio of Net Income to Total Revenue, indicating the percentage
            of revenue that translates into profit.
            - Asset Turnover measures the efficiency of a company's use of its assets to generate sales
            revenue.
            - The Equity Multiplier measures financial leverage, the use of debt to finance a company's
            operations, which can amplify returns as well as risks.

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
            - Cash and Cash Equivalents: The cash and the investments that are readily convertible to cash
            within three months, such as money market funds and Treasury bills.
            - Short Term Investments: The marketable securities that are not cash equivalents.

        The Enterprise Value is calculated as the sum of Market Cap, Debt, Preferred Equity,
        Minority Interest, minus Cash and Cash Equivalents and Short Term Investments. Cash and
        marketable securities are deducted together (Damodaran), the same Enterprise Value as
        toolkit.ratios.get_enterprise_value returns.

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
        | Share Price               | 177.57        | 129.93        | 192.53        | 250.42        | 271.86        |
        | Market Capitalization     |   2.9947e+12  |   2.12121e+12 |   3.04439e+12 |   3.8585e+12  |   4.07918e+12 |
        | Total Debt                |   1.36522e+11 |   1.3248e+11  |   1.2393e+11  |   1.19059e+11 |   1.12377e+11 |
        | Minority Interest         |   0           |   0           |   0           |   0           |   0           |
        | Preferred Equity          |   0           |   0           |   0           |   0           |   0           |
        | Cash and Cash Equivalents |   3.494e+10   |   2.3646e+10  |   2.9965e+10  |   2.9943e+10  |   3.5934e+10  |
        | Enterprise Value          |   3.06859e+12 |   2.20539e+12 |   3.10676e+12 |   3.91238e+12 |   4.13686e+12 |
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
            .loc[begin:end, "Close"][self._tickers]
            .T
        )

        self._enterprise_value_breakdown = enterprise_model.get_enterprise_value_breakdown(
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
            # Deducted with the cash, as in the Ratios module's Enterprise Value.
            short_term_investments=(
                self._balance_sheet_statement.loc[:, "Short Term Investments", :]
                if "Short Term Investments"
                in self._balance_sheet_statement.index.get_level_values(1)
                else None
            ),
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

        Because the book value of assets leaves out most intangibles (brands, software, research),
        companies built on them show a Q far above 1 without being overvalued in Tobin's sense, so the
        ratio is most informative within an industry and over time. In macroeconomics the aggregate Q of
        the corporate sector is read as a signal for investment: a Q above 1 makes new capital worth more
        than it costs.

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

        |                        |        2021 |        2022 |        2023 |        2024 |         2025 |
        |:-----------------------|------------:|------------:|------------:|------------:|-------------:|
        | Market Value of Equity | 2.9947e+12  | 2.12121e+12 | 3.04439e+12 |  3.8585e+12 |  4.07918e+12 |
        | Total Liabilities      | 2.87912e+11 | 3.02083e+11 | 2.90437e+11 |  3.0803e+11 |  2.85508e+11 |
        | Total Assets           | 3.51002e+11 | 3.52755e+11 | 3.52583e+11 |  3.6498e+11 |  3.59241e+11 |
        | Tobin's Q Ratio        | 9.3521      | 6.8696      | 9.4583      | 11.4158     | 12.1497      |
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
            .loc[begin:end, "Close"][self._tickers]
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

    def _earnings_before_interest_and_taxes(
        self, trailing: int | None = None
    ) -> pd.DataFrame:
        """
        The Earnings Before Interest and Taxes (EBIT) the models use, taken as the reported
        Operating Income, consistent with the EBIT based ratios of the Ratios module.

        Args:
            trailing (int | None, optional): The number of periods to sum. Defaults to None,
                which is a single period.

        Returns:
            pd.DataFrame: The EBIT per ticker (rows) and period (columns).
        """
        operating_income = self._income_statement.loc[:, "Operating Income", :]

        return (
            operating_income.T.rolling(trailing).sum().T
            if trailing
            else operating_income
        )

    @handle_errors
    def _capital_asset_pricing_inputs(
        self,
        risk_free_rate: pd.Series,
        benchmark_returns: pd.Series,
        market_risk_premium: float | None,
        trailing: int | None = None,
    ) -> tuple[pd.Series, pd.Series]:
        """
        The risk-free rate and the expected market return for the Cost of Equity, over the
        span of the flows it is compared with. The market return is the risk-free rate plus
        the market risk premium, by default the benchmark's average return above the
        risk-free rate over every period of the data, so that the Cost of Equity does not
        follow the market up and down from year to year.

        With quarterly data the annual yield and premium are divided by four, matching the
        quarterly returns and costs of debt. With `trailing` set, the flows (Net Income,
        Interest Expense, NOPAT) are sums over that many periods, so the rates are the
        per-period rates times `trailing`: with quarterly data and a trailing 4 quarters,
        the annual rates that belong to trailing twelve month flows.

        Args:
            risk_free_rate (pd.Series): The annual risk-free yield per period.
            benchmark_returns (pd.Series): The benchmark's return per period.
            market_risk_premium (float | None): An annual premium to use instead.
            trailing (int | None, optional): The number of periods the flows are summed
                over. Defaults to None, which is a single period.

        Returns:
            tuple[pd.Series, pd.Series]: The risk-free rate and the expected market return,
            both over the span of the flows.
        """
        periods_per_year = 4 if self._quarterly else 1
        span = trailing if trailing else 1

        risk_free_rate = risk_free_rate / periods_per_year
        premium = (
            (benchmark_returns - risk_free_rate).mean()
            if market_risk_premium is None
            else market_risk_premium / periods_per_year
        )

        # Scaled after the premium is averaged, which compares per-period benchmark returns with the per-period rate.  # noqa: E501
        risk_free_rate = risk_free_rate * span

        return risk_free_rate, risk_free_rate + premium * span

    def get_weighted_average_cost_of_capital(
        self,
        show_full_results: bool = True,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
        market_risk_premium: float | None = None,
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
            - Cost of Equity = Risk Free Rate + Beta * Market Risk Premium
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

        The WACC is the discount rate for the free cash flow to the firm and the hurdle rate for new
        investments, so analysts usually build it from forward-looking inputs: a long-term government bond
        yield, an equity risk premium of a few percent, a beta measured over several years and the yield on
        the company's debt. Small changes matter: in a DCF with 2.5% perpetual growth, moving the WACC from
        8% to 9% lowers the terminal value by about 15%.

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
            market_risk_premium (float | None, optional): The annual equity risk premium the Cost of Equity
                uses. Defaults to None, which takes the benchmark's average return above the risk-free rate
                over every period of the data.
            show_columns (list[str] | None, optional): Restrict the result to these columns. Defaults to None,
                which shows all columns.

        Returns:
            pd.DataFrame: DataFrame containing the WACC values.

        Notes:
        - The Cost of Equity follows from the Capital Asset Pricing Model (CAPM): the risk-free rate of
        the period plus Beta times the market risk premium, which defaults to the benchmark's average
        return above the risk-free rate over every period of the data (set `market_risk_premium` to use
        another premium, e.g. an implied equity risk premium). Beta is estimated from the period's daily
        returns. With quarterly data the rates are per quarter, the risk-free yield divided by four.
        - With `trailing` set, the Interest Expense is summed over the trailing periods, so the Cost of
        Debt covers that span, and the risk-free rate and premium of the Cost of Equity are the per-period
        rates times `trailing`. With quarterly data and trailing=4, every rate is an annual rate.
        - The Market Value of Debt is approximated as the Total Debt.
        - The Cost of Debt is the Interest Expense divided by the Total Debt at the end of the period
        (the average over the trailing periods with `trailing` set), so debt raised or repaid during the
        period moves it away from the rate on the average debt outstanding.
        - The Corporate Tax Rate is the effective tax rate (Income Tax Expense / Income Before Tax),
        kept within 0 and 1 so that a tax benefit does not turn the tax shield negative, and 0 for a
        period without a pre-tax profit, which leaves no taxable income for the interest to shield.

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

        |                                  |        2021 |        2022 |        2023 |        2024 |        2025 |
        |:---------------------------------|------------:|------------:|------------:|------------:|------------:|
        | Market Value Equity              | 2.9947e+12  | 2.12121e+12 | 3.04439e+12 | 3.8585e+12  | 4.07918e+12 |
        | Market Value Debt                | 1.36522e+11 | 1.3248e+11  | 1.2393e+11  | 1.19059e+11 | 1.12377e+11 |
        | Cost of Equity                   | 0.1305      | 0.1442      | 0.1376      | 0.1273      | 0.1533      |
        | Cost of Debt                     | 0.0194      | 0.0221      | 0.0317      | 0           | 0           |
        | Corporate Tax Rate               | 0.133       | 0.162       | 0.1472      | 0.2409      | 0.1561      |
        | Weighted Average Cost of Capital | 0.1255      | 0.1368      | 0.1333      | 0.1235      | 0.1492      |
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
            .loc[begin:end, "Close"][self._tickers]
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
        # The Cost of Equity covers the span of the flows: a trailing sum of Interest Expense is the cost of debt over that span.  # noqa: E501
        risk_free_rate, benchmark_returns = self._capital_asset_pricing_inputs(
            risk_free_rate, benchmark_returns, market_risk_premium, trailing
        )

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
        market_risk_premium: float | None = None,
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

        EVA is mostly used to compare value creation across business units or companies and over time, and
        as a basis for management compensation. It is equivalent to (Return on Invested Capital - WACC) *
        Invested Capital, so a company with a high return on capital but a small capital base can create
        less EVA than a larger, less profitable one.

        Also known as: EVA, economic profit.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to None.
            growth (bool, optional): Whether to calculate the growth of the values. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int | None, optional): The trailing period to use for the calculation. Defaults to None.
            market_risk_premium (float | None, optional): The annual equity risk premium the Cost of Equity
                uses. Defaults to None, which takes the benchmark's average return above the risk-free rate
                over every period of the data.
            show_columns (list[str] | None, optional): List of columns to show in the results. If None, all
                columns will be shown. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame containing the EVA and its components.

        Notes:
        - A positive EVA indicates that the company is generating returns in excess of its cost of
        capital, i.e. it is creating value for its capital providers. A negative EVA indicates the
        company is destroying value.
        - EBIT is the reported Operating Income, as in the other models of this module and the
        EBIT based ratios of the Ratios module.
        - Invested Capital is the sum of the two-period average of Total Equity and the two-period
        average of Total Debt, i.e. the capital employed over the course of the period rather than
        its closing balance, consistent with the Return on Invested Capital calculation in the
        Ratios module.
        - The Weighted Average Cost of Capital comes from `get_weighted_average_cost_of_capital`,
        with the market risk premium given by `market_risk_premium` (by default the benchmark's
        average return above the risk-free rate over the data) and the same `trailing`, so that it
        covers the span of the NOPAT (an annual rate for trailing=4 with quarterly data).

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

        |                                  |        2021 |        2022 |        2023 |        2024 |        2025 |
        |:---------------------------------|------------:|------------:|------------:|------------:|------------:|
        | Net Operating Profit After Taxes | 9.44563e+10 | 1.00083e+11 | 9.74768e+10 | 9.35318e+10 | 1.12281e+11 |
        | Invested Capital                 | 1.93614e+11 | 1.91382e+11 | 1.84614e+11 | 1.81042e+11 | 1.8106e+11  |
        | Weighted Average Cost of Capital | 0.1255      | 0.1368      | 0.1333      | 0.1235      | 0.1492      |
        | Economic Value Added             | 7.01567e+10 | 7.39066e+10 | 7.28641e+10 | 7.11773e+10 | 8.52711e+10 |
        """
        eva = {}

        income_tax_expense = (
            self._income_statement.loc[:, "Income Tax Expense", :]
            .T.rolling(trailing)
            .sum()
            .T
            if trailing
            else self._income_statement.loc[:, "Income Tax Expense", :]
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

        ebit = self._earnings_before_interest_and_taxes(trailing)
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

        # Full precision, so that the EVA is not charged a WACC rounded to the display rounding.
        eva["Weighted Average Cost of Capital"] = (
            self.get_weighted_average_cost_of_capital(
                show_full_results=False,
                trailing=trailing,
                market_risk_premium=market_risk_premium,
                rounding=15,
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
        | Market Value of Equity | 2.9947e+12  | 2.12121e+12 | 3.04439e+12 | 3.8585e+12  | 4.07918e+12 |
        | Market Value of Debt   | 1.36522e+11 | 1.3248e+11  | 1.2393e+11  | 1.19059e+11 | 1.12377e+11 |
        | Invested Capital       | 1.93614e+11 | 1.91382e+11 | 1.84614e+11 | 1.81042e+11 | 1.8106e+11  |
        | Market Value Added     | 2.93761e+12 | 2.06231e+12 | 2.98371e+12 | 3.79651e+12 | 4.01049e+12 |
        """
        mva = {}

        years = self._cash_flow_statement.columns
        begin, end = str(years[0]), str(years[-1])

        share_prices = (
            self._historical_data["quarterly" if self._quarterly else "yearly"]
            .loc[begin:end, "Close"][self._tickers]
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
        cash_flow_type: str = "Free Cash Flow to Firm",
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
            - Enterprise Value = Sum of Present Value of Cash Flow Projections + Terminal Value /
            (1 + Weighted Average Cost of Capital)^Periods
            - Equity Value = Enterprise Value — Total Debt + Cash and Cash Equivalents
            - Intrinsic Value = Equity Value / Weighted Average Shares Diluted

        The base cash flow is by default the Free Cash Flow to the Firm (see `get_free_cash_flow_to_firm`),
        the cash flow available to both debt and equity holders. That is the cash flow the Weighted
        Average Cost of Capital discounts into an Enterprise Value, from which the debt is then subtracted.

        In a five-year DCF the discounted terminal value is often 60% to 80% of the enterprise value, so the
        result is driven mostly by the gap between the WACC and the perpetual growth rate. Practitioners
        therefore show the intrinsic value as a range over several WACC and growth assumptions rather than a
        single number, and keep the perpetual growth rate at or below long-run nominal GDP growth.

        Also known as: DCF, discounted cash flow, fair value, intrinsic value.

        Args:
            growth_rate (float, list or dict): The growth rate to use for the cash flow projections. Can be one number
            to use for all tickers, or a list or dict that contains a growth rate for each ticker.
            perpetual_growth_rate (float, list or dict): The perpetual growth rate to use for the terminal value.
            Can be one number to use for all tickers, or a list or dict that contains a perpetual growth rate for
            each ticker.
            weighted_average_cost_of_capital (float, list or dict): The weighted average cost of capital used to
            discount the projected cash flows and to calculate the terminal value. Can be one number to use for
            all tickers, or a list or dict that contains a weighted average cost of capital for each ticker.
            periods (int, optional): The number of periods to use for the cash flow projections. Defaults to 5.
            cash_flow_type (str, optional): The type of cash flow to use for the cash flow projections.
            Defaults to "Free Cash Flow to Firm". The other options are "Free Cash Flow" and "Operating
            Cash Flow" as reported on the cash flow statement; both are after interest, so discounting
            them at the WACC and subtracting the debt counts the cost of debt twice.
            trailing (int | None, optional): The number of trailing periods to sum for the base cash flow.
            When set, uses the sum of the last N periods instead of only the most recent period. Defaults to
            None, which uses 4 (the trailing twelve months) with quarterly data and the most recent year with
            yearly data.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.

        Returns:
            pd.DataFrame: DataFrame containing the intrinsic value for each ticker.

        Notes:
        - The results are highly dependent on the input. Therefore, think carefully about each input parameter to
        ensure the results are accurate (given your beliefs)
        - The growth rate, perpetual growth rate and WACC are annual rates, so the base cash flow is an annual
        one: with quarterly data the last four quarters are summed. Setting `trailing` with yearly data sums
        several years into one base cash flow, which logs a warning.
        - The WACC must be greater than the perpetual growth rate; otherwise the terminal value is infinite
        and the valuation is not calculated.
        - The Equity Value is divided by the Weighted Average Shares Diluted of the most recent period.

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
        | Terminal Value       |   1.6355e+12  |
        | Cash Flow Projection |   1.10097e+11 |
        | Enterprise Value     |   1.42566e+12 |
        | Equity Value         |   1.34921e+12 |
        | Intrinsic Value      |  89.9194      |
        """
        if cash_flow_type not in [
            "Free Cash Flow to Firm",
            "Free Cash Flow",
            "Operating Cash Flow",
        ]:
            raise ValueError(
                "Invalid cash flow type. Please choose from the following: 'Free Cash Flow to Firm', "
                "'Free Cash Flow', 'Operating Cash Flow'"
            )

        # The growth and discount rates are annual, so a quarterly base cash flow is the trailing twelve months.
        if trailing is None and self._quarterly:
            trailing = 4
        elif trailing and not self._quarterly:
            logger.warning(
                "The trailing parameter sums %s years of cash flow into the base cash flow of the DCF, "
                "while the growth rates and the WACC are annual rates.",
                trailing,
            )

        cash_flows = (
            self.get_free_cash_flow_to_firm(rounding=15).xs(
                "Free Cash Flow to Firm", level=1
            )
            if cash_flow_type == "Free Cash Flow to Firm"
            else self._cash_flow_statement.loc[:, cash_flow_type, :]
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
            cash_flow_series = cash_flows.loc[ticker].dropna()

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
        - EBIT is the reported Operating Income, as in the Altman Z-Score and Economic Value
        Added calculations elsewhere in this module.
        - The Capital Expenditure and Change in Working Capital line items from the cash flow
        statement are stored using a cash-flow-impact sign convention (a use of cash is
        negative). This method negates them internally so that they represent positive
        magnitudes (amount spent / amount of the increase), matching the standard academic
        FCFF formula. See `get_free_cash_flow_to_firm` in the Intrinsic module for details.
        - FCFF is the default base cash flow of `get_intrinsic_valuation`, which discounts it
        at the Weighted Average Cost of Capital.

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

        |                                  |        2021 |         2022 |        2023 |         2024 |        2025 |
        |:---------------------------------|------------:|-------------:|------------:|-------------:|------------:|
        | Net Operating Profit After Taxes | 9.44563e+10 |  1.00083e+11 | 9.74768e+10 |  9.35318e+10 | 1.12281e+11 |
        | Depreciation and Amortization    | 1.1284e+10  |  1.1104e+10  | 1.1519e+10  |  1.1445e+10  | 1.1698e+10  |
        | Capital Expenditure              | 1.1085e+10  |  1.0708e+10  | 1.0959e+10  |  9.447e+09   | 1.2715e+10  |
        | Change in Net Working Capital    | 4.911e+09   | -1.2e+09     | 6.577e+09   | -3.651e+09   | 2.5e+10     |
        | Free Cash Flow to Firm           | 8.97443e+10 |  1.01679e+11 | 9.14598e+10 |  9.91808e+10 | 8.62639e+10 |
        """
        fcff = {}

        income_tax_expense = (
            self._income_statement.loc[:, "Income Tax Expense", :]
            .T.rolling(trailing)
            .sum()
            .T
            if trailing
            else self._income_statement.loc[:, "Income Tax Expense", :]
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

        ebit = self._earnings_before_interest_and_taxes(trailing)
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

    def _complete_period_dividends(self) -> pd.DataFrame:
        """
        The Dividends per Share of every complete period. The last period of the historical
        data is dropped when the data ends before that period does (the current, unfinished
        year or quarter), since its dividends cover only part of the period and would understate
        the base dividend of the dividend discount models.

        Returns:
            pd.DataFrame: The Dividends per Share per period (rows) and ticker (columns).
        """
        dividends_per_share = self._historical_data[
            "quarterly" if self._quarterly else "yearly"
        ]["Dividends"]

        if dividends_per_share.empty:
            return dividends_per_share

        last_observation = self._historical_data["daily"].index[-1].to_timestamp()
        last_period_end = dividends_per_share.index[-1].end_time.normalize()

        # A period counts as complete when at most one business day of it is missing from the data, which allows for an exchange holiday (such as Good Friday) on the last business day of the period.  # noqa: E501
        missing_business_days = len(
            pd.bdate_range(last_observation + pd.Timedelta(days=1), last_period_end)
        )

        if missing_business_days > 1:
            return dividends_per_share.iloc[:-1]

        return dividends_per_share

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
        periods beyond the last complete one are projected forward at the given growth rate. The current,
        unfinished period is projected as well, since its dividends so far cover only part of it. The first
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
        dividends_per_share = self._complete_period_dividends()

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
        - The base dividend used for the projection is the Dividends Per Share of the last complete
        period for each ticker; the current, unfinished period is left out since its dividends so far
        cover only part of it. For companies that do not pay a dividend, this model is
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
        dividends_per_share = self._complete_period_dividends()

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
        market_risk_premium: float | None = None,
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

            - Residual Income_t = Net Income_t - (Cost of Equity * Book Value of Equity_t-1)

        Also known as: RI, economic profit (equity variant), abnormal earnings.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to None.
            growth (bool, optional): Whether to calculate the growth of the values. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int | None, optional): The trailing period to use for the calculation. Defaults to None.
            market_risk_premium (float | None, optional): The annual equity risk premium the Cost of Equity
                uses. Defaults to None, which takes the benchmark's average return above the risk-free rate
                over every period of the data.
            show_columns (list[str] | None, optional): List of columns to show in the results. If None, all
                columns will be shown. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame containing the Residual Income and its components.

        Notes:
        - A positive Residual Income indicates that the company generated more profit than
        equity holders required given the capital they have invested. A negative Residual
        Income indicates the company failed to earn its equity holders' required return, even
        if it still reported a positive Net Income.
        - The Cost of Equity follows from the Capital Asset Pricing Model (CAPM), as in
        `get_weighted_average_cost_of_capital`: the risk-free rate of the period plus Beta times
        the market risk premium, by default the benchmark's average return above the risk-free
        rate over the data (see `market_risk_premium`). It covers the span of the Net Income: a
        quarter with quarterly data, and the per-period rate times `trailing` when the Net Income
        is a trailing sum (an annual rate for trailing twelve months, trailing=4 with quarterly data).
        - The Book Value of Equity is the Total Shareholder Equity minus Preferred Stock at the
        beginning of the period (the previous period's closing balance, or that of `trailing` periods
        back with `trailing` set), the capital the shareholders had invested while the Net Income was
        earned. Preferred equity is excluded since Residual Income (like Net Income) belongs to common
        equity holders only.

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

        |                      |        2021 |       2022 |        2023 |        2024 |       2025 |
        |:---------------------|------------:|-----------:|------------:|------------:|-----------:|
        | Net Income           | 9.468e+10   | 9.9803e+10 | 9.6995e+10  | 9.3736e+10  | 1.1201e+11 |
        | Cost of Equity       | 0.1305      | 0.1442     | 0.1376      | 0.1273      | 0.1533     |
        | Book Value of Equity | 6.5339e+10  | 6.309e+10  | 5.0672e+10  | 6.2146e+10  | 5.695e+10  |
        | Residual Income      | 8.61558e+10 | 9.0708e+10 | 9.00202e+10 | 8.58257e+10 | 1.0328e+11 |
        """
        residual_income = {}

        net_income = (
            self._income_statement.loc[:, "Net Income", :].T.rolling(trailing).sum().T
            if trailing
            else self._income_statement.loc[:, "Net Income", :]
        )
        # The capital charge applies to the book value the shareholders had invested at the start of the period (Ohlson, 1995): the closing balance of the previous period, or of `trailing` periods back for a trailing sum of Net Income.  # noqa: E501
        beginning_book_value_of_equity = (
            self._balance_sheet_statement.loc[:, "Total Shareholder Equity", :]
            - self._balance_sheet_statement.loc[:, "Preferred Stock", :]
        ).shift(trailing if trailing else 1, axis=1)

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
        # The Cost of Equity covers the span of the trailing Net Income it is charged against.
        risk_free_rate, benchmark_returns = self._capital_asset_pricing_inputs(
            risk_free_rate, benchmark_returns, market_risk_premium, trailing
        )

        cost_of_equity = wacc_model.get_cost_of_equity(
            risk_free_rate=risk_free_rate,
            beta=beta,
            benchmark_returns=benchmark_returns,
        ).T

        residual_income["Net Income"] = net_income
        residual_income["Cost of Equity"] = cost_of_equity
        residual_income["Book Value of Equity"] = beginning_book_value_of_equity

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

        The original Z-Score was estimated on US listed manufacturers, and its thresholds are not meant for
        banks, insurers or most service companies; Altman's later Z'' variant drops the sales term for
        non-manufacturers. Share buybacks charged to retained earnings and accumulated losses both lower the
        retained earnings term, so a company that returns most of its earnings scores lower without being
        closer to distress, which is why the score is best read as a trend and next to the market-based term.

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
            - EBIT is the reported Operating Income, as in the EBIT based ratios of the Ratios module.

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
        | EBIT to Total Assets              |  0.3104 |  0.3386 |  0.3242 |  0.3376 |  0.3704 |
        | Market Value to Total Liabilities | 10.4015 |  7.022  | 10.4821 | 12.5264 | 14.2874 |
        | Sales to Total Assets             |  1.0422 |  1.1179 |  1.0871 |  1.0714 |  1.1584 |
        | Altman Z-Score                    |  8.3615 |  6.373  |  8.4394 |  9.5508 | 10.8385 |
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
                ebit=self._earnings_before_interest_and_taxes(trailing),
                total_assets=total_assets,
            )
        )

        years = self._balance_sheet_statement.columns
        begin, end = str(years[0]), str(years[-1])

        share_prices = (
            self._historical_data["quarterly" if self._quarterly else "yearly"]
            .loc[begin:end, "Close"][self._tickers]
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

        Piotroski designed the F-Score to separate winners from losers among value stocks (high
        book-to-market), where scores of 8 or 9 marked the strongest companies and scores of 0 or 1 the
        weakest. Because most criteria compare this year with last year, the score measures the direction
        of change rather than the level of financial strength, and a single strong year after a weak one can
        score highly.

        Following Piotroski (2000), the Return on Assets, the Operating Cash Flow in the accruals signal and
        the Asset Turnover are scaled by the total assets at the beginning of the period, while the
        leverage signal divides long-term debt by the average of the beginning and ending total assets.

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
        | Change in Return on Assets Criteria |      0 |      0 |      0 |      1 |
        | Accruals Criteria                   |      1 |      1 |      1 |      0 |
        | Change in Leverage Criteria         |      1 |      1 |      1 |      1 |
        | Change in Current Ratio Criteria    |      0 |      1 |      0 |      1 |
        | Number of Shares Criteria           |      1 |      1 |      1 |      1 |
        | Gross Margin Criteria               |      1 |      1 |      1 |      1 |
        | Asset Turnover Criteria             |      0 |      0 |      1 |      1 |
        | Piotroski Score                     |      6 |      7 |      7 |      8 |

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

        # Piotroski (2000) scales ROA, CFO, accruals and asset turnover by beginning-of-year total assets and only the leverage signal by average total assets. With a trailing window the beginning of the window lies `trailing` periods back.  # noqa: E501
        beginning_total_assets = self._balance_sheet_statement.loc[
            :, "Total Assets", :
        ].shift(trailing if trailing else 1, axis=1)
        average_total_assets = total_assets.T.rolling(2).mean().T

        piotroski_score["Return on Assets Criteria"] = (
            piotroski_model.get_return_on_assets_criteria(
                net_income=net_income,
                beginning_total_assets=beginning_total_assets,
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
                beginning_total_assets=beginning_total_assets,
            )
        )

        # Piotroski scales both the Return on Assets and the Cash Flow from Operations by the same asset base, which is what makes this signal reduce to the sign of accruals  # noqa: E501
        piotroski_score["Accruals Criteria"] = piotroski_model.get_accruals_criteria(
            net_income=net_income,
            operating_cashflow=operating_cashflow,
            beginning_total_assets=beginning_total_assets,
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
                beginning_total_assets=beginning_total_assets,
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

        # Two columns are lost rather than one. The beginning and averaged total assets already make the first period NaN, so the criteria that compare against the previous period only become meaningful from the third period onwards. Keeping the second column would silently score those criteria 0 even when they improved.  # noqa: E501
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

        The -1.78 threshold belongs to this eight-variable model; the five-variable version uses -2.22.
        Fast-growing companies are flagged more often because sales growth itself raises the score, and the
        model was estimated without financial companies and is not suited to banks and insurers, so a score
        above the threshold is a prompt to read the accounts, not a finding.

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
        - Total Accruals are Net Income minus Cash Flow from Operations, the cash flow form of
        Beneish, Lee and Nichols (2013), rather than the balance sheet form of Beneish (1999).
        - The Depreciation Index uses Depreciation and Amortization, since statements report the two
        as one line, where Beneish (1999) uses depreciation alone.

        References:
        - Beneish, Messod D. "The Detection of Earnings Manipulation." Financial Analysts Journal,
        Vol. 55, No. 5, 1999, pp. 24-36.
        - Beneish, Messod D., Charles M. C. Lee, and D. Craig Nichols. "Earnings Manipulation and
        Expected Returns." Financial Analysts Journal, Vol. 69, No. 2, 2013, pp. 57-82.

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
            i.e. approximately -3.23. Since this implementation omits Ohlson's price-level
            deflator, its scores sit about 2.6 below Ohlson's scale and this cutoff flags far too few
            companies (see the Notes).
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
            isn't meaningful outside the original US sample. This implementation omits that deflator and
            uses ln(Total Assets) in nominal terms instead. This lowers every O-Score by roughly 0.407 *
            ln(price index); with the US price level of the 2020s at roughly six times its 1968 level (an
            index of about 600 to 650), that is about 2.6. The resulting probabilities are therefore much
            lower than in Ohlson's study and the 0.038 cutoff (-3.23) flags far too few companies; it
            corresponds to roughly -5.8 on this scale. The *ranking* of companies relative to one another
            within a given currency and period is unaffected.
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
            - X = -4.336 - 4.513 * Net Income to Total Assets + 5.679 * Total Liabilities to Total Assets
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
        | Zmijewski Score                       | -0.8994 | -0.7531 | -0.9034 | -0.7056 | -1.2333 |
        | Zmijewski Bankruptcy Probability      |  0.1842 |  0.2257 |  0.1831 |  0.2402 |  0.1087 |
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
        - EBIT is the reported Operating Income, as in the Altman Z-Score calculation elsewhere
        in this module.

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
        | EBIT to Total Assets            | 0.3104 |  0.3386 |  0.3242 |  0.3376 |  0.3704 |
        | EBT to Current Liabilities      | 0.8703 |  0.7735 |  0.7827 |  0.7001 |  0.8014 |
        | Sales to Total Assets           | 1.0422 |  1.1179 |  1.0871 |  1.0714 |  1.1584 |
        | Springate Score                 | 1.9716 |  1.9429 |  1.9416 |  1.861  |  2.0786 |
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
                ebit=self._earnings_before_interest_and_taxes(trailing),
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
        - EBIT is the reported Operating Income, as in the Altman Z-Score calculation elsewhere
        in this module. Return on
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
        | EBIT to Total Assets            | 0.3104 |  0.3386 |  0.3242 |  0.3376 |  0.3704 |
        | Return on Assets                | 0.2697 |  0.2829 |  0.2751 |  0.2568 |  0.3118 |
        | Grover Score                    | 1.1532 |  1.1181 |  1.148  |  1.0963 |  1.2316 |
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
                ebit=self._earnings_before_interest_and_taxes(trailing),
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
        - EBIT is the reported Operating Income, as in the Altman Z-Score calculation elsewhere
        in this module.
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
        | Log of EBIT to Interest Expense      |  3.7182 |  3.7074 |  3.3694 | nan      | nan      |
        | Fulmer H-Score                       | 14.252  | 14.3098 | 13.9834 | nan      | nan      |
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
        ebit = self._earnings_before_interest_and_taxes(trailing)
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
        market_risk_premium: float | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ) -> pd.DataFrame:
        """
        The Present Value of Growth Opportunities (PVGO) is a financial metric that represents the
        present value of a company's future growth opportunities. It is calculated as the difference
        between the share price and the value of the company's current earnings as a perpetuity without
        growth, i.e. Earnings per Share divided by the Cost of Equity.

        The formula is as follows:

            - PVGO = Stock Price — Earnings Per Share / Cost of Equity

        Also known as: PVGO, growth value.

        Args:
            calculate_daily (bool, optional): Whether to calculate the PVGO using daily historical data.
            Defaults to False.
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            include_dividends (bool, optional): Whether to deduct Preferred Dividends Paid from Net
            Income when calculating the Earnings per Share, so that the earnings figure reflects what is
            attributable to common shareholders only. Defaults to False.
            trailing (int | None, optional): The trailing period to use for the calculation. Defaults to None.
            market_risk_premium (float | None, optional): The annual equity risk premium the Cost of Equity
                uses. Defaults to None, which takes the benchmark's average return above the risk-free rate
                over every period of the data.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the values. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.DataFrame: DataFrame containing the PVGO values.

        Notes:
        - The no-growth value (Earnings per Share / r) is discounted at the Cost of Equity, since both
        the share price and the Earnings per Share are equity-only, per-share quantities (Brealey, Myers
        and Allen, "Principles of Corporate Finance"; Bodie, Kane and Marcus, "Investments"). The Cost of
        Equity is the CAPM Cost of Equity of `get_weighted_average_cost_of_capital`.
        - The Cost of Equity uses the same `trailing` as the Earnings per Share, so both cover the same
        span (with quarterly data, a quarter without `trailing` and a year with trailing=4).
        - When the Earnings per Share is negative, the no-growth value is negative and the PVGO
        exceeds the share price, which has no meaningful interpretation.

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

        | Date   |     AAPL |    TSLA |
        |:-------|---------:|--------:|
        | 2022   |  87.5241 | 103.349 |
        | 2023   | 147.966  | 230.336 |
        | 2024   | 202.626  | 395.734 |
        | 2025   | 223.16   | 445.228 |
        """
        # Full precision, so that the rounding of the display does not move the no-growth value.
        wacc_components = self.get_weighted_average_cost_of_capital(
            show_full_results=True,
            trailing=trailing,
            market_risk_premium=market_risk_premium,
            rounding=15,
        )
        # With a single ticker the full results come without the ticker level.
        cost_of_equity = (
            wacc_components.loc[["Cost of Equity"]].set_axis(self._tickers)
            if len(self._tickers) == 1
            else wacc_components.xs("Cost of Equity", level=1)
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
            self._historical_data["daily"]["Close"]
            if calculate_daily
            else self._historical_data["quarterly" if self._quarterly else "yearly"][
                "Close"
            ]
        )

        pvgo = growth_model.get_present_value_of_growth_opportunities(
            cost_of_equity=cost_of_equity,
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

            - Return on Equity = Net Income / Total Equity at the beginning of the period
            - Retention Ratio = 1 - Dividend Payout Ratio
            - SGR = Return on Equity * Retention Ratio

        The SGR assumes the company keeps its leverage and pays out only through dividends. Companies that
        return cash mainly through share buybacks have a low payout ratio on paper and a return on equity
        inflated by a shrinking equity base, so their SGR overstates what they can fund internally (Apple's
        exceeds 100% in every year of the example below).

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
        - Following Higgins ("Analysis for Financial Management"), the Return on Equity is measured
        on the equity at the beginning of the period (the previous period's closing balance, or that of
        `trailing` periods back with `trailing` set): the retained earnings then grow that equity by
        exactly the SGR. This equals ROE * b / (1 - ROE * b) with the Return on Equity measured on the
        ending equity.

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
        | AAPL | 1.2276 | 1.3467 | 1.6177 | 1.2632 | 1.696  |
        | MSFT | 0.3783 | 0.3846 | 0.3156 | 0.3218 | 0.2896 |
        """
        net_income = (
            self._income_statement.loc[:, "Net Income", :].T.rolling(trailing).sum().T
            if trailing
            else self._income_statement.loc[:, "Net Income", :]
        )
        # Higgins measures the Return on Equity on the equity at the beginning of the period.
        beginning_total_equity = self._balance_sheet_statement.loc[
            :, "Total Equity", :
        ].shift(trailing if trailing else 1, axis=1)
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
            average_total_equity=beginning_total_equity,
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

            - Return on Assets = Net Income / Total Assets at the end of the period
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
        - The Return on Assets is measured on the Total Assets at the end of the period, as in
        Ross, Westerfield and Jaffe ("Corporate Finance"). The formula then equals the Return on
        Assets on the beginning Total Assets times the Retention Ratio, the growth the retained
        earnings add to the beginning assets, consistent with the beginning-equity definition of
        the Sustainable Growth Rate.

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
        | AAPL | 0.2962 | 0.3173 | 0.3029 | 0.274  | 0.3677 |
        | MSFT | 0.1548 | 0.176  | 0.1462 | 0.1489 | 0.1436 |
        """
        net_income = (
            self._income_statement.loc[:, "Net Income", :].T.rolling(trailing).sum().T
            if trailing
            else self._income_statement.loc[:, "Net Income", :]
        )
        # The ROA * b / (1 - ROA * b) form of the Internal Growth Rate measures ROA on the ending assets.
        ending_total_assets = self._balance_sheet_statement.loc[:, "Total Assets", :]
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
            average_total_assets=ending_total_assets,
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

        The constant 22.5 is Graham's limit of a price-to-earnings ratio of 15 times a
        price-to-book ratio of 1.5, so the Graham Number is the price at which the product of
        the two ratios reaches that limit. For companies whose value lies mostly in
        intangibles, or that buy back many shares, book value is small and the Graham Number
        sits far below the market price.

        Also known as: Graham fair value.

        Args:
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            trailing (int | None, optional): The trailing period to use for the calculation. Defaults to None,
                which uses 4 (the trailing twelve months of earnings) with quarterly data and a single period
                with yearly data.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the values. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.DataFrame: DataFrame containing the Graham Number values.

        Notes:
        - Graham's limits of 15 times earnings and 1.5 times book value refer to annual earnings, so
        with quarterly data the Earnings per Share are the sum of the last four quarters by default.
        - Earnings and Book Value per Share use the Weighted Average Shares Diluted (or the Weighted
        Average Shares with diluted=False).

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
        # Graham's multiples apply to annual earnings, so a quarterly Earnings per Share is the trailing twelve months.  # noqa: E501
        if trailing is None and self._quarterly:
            trailing = 4

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

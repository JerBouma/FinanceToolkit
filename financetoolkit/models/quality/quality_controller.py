"""Quality Models Module"""

__docformat__ = "google"

from typing import Literal

import numpy as np
import pandas as pd

from financetoolkit.helpers import check_trailing, trailing_sum
from financetoolkit.models.quality import beneish_model, piotroski_model, quality_model
from financetoolkit.performance.performance_model import get_beta
from financetoolkit.utilities.dataframe_model import concat_frames, filter_shown_columns
from financetoolkit.utilities.error_model import (
    check_option,
    handle_errors,
    validate_window,
)
from financetoolkit.utilities.logger_model import get_logger
from financetoolkit.utilities.sources_model import track_sources

# ruff: noqa: E501

logger = get_logger()


@check_trailing
@filter_shown_columns
@track_sources
class QualityModels:
    """
    Quality and earnings manipulation scores.

    Piotroski, Beneish, Mohanram, Dechow, Montier, the magic formula, the quality score, the
    accruals ratio and the modified Jones discretionary accruals.
    """

    @handle_errors
    def get_piotroski_score(
        self,
        trailing: int | None = None,
        show_columns: str | list[str] | None = None,
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

        The formula is as follows, where (t-1) is the same period a year earlier (the previous column on
        yearly data, four quarters back on quarterly data, so quarterly comparisons are year over year) and
        Beginning Total Assets are the total assets at the end of the previous period (`trailing` periods back
        when set, one year earlier with four trailing quarters):

        - Return on Assets (ROA) = Net Income / Beginning Total Assets
        - Return on Assets Criteria = 1 if ROA > 0, else 0
        - Operating Cashflow Criteria = 1 if Operating Cash Flow > 0, else 0
        - Change in Return on Assets Criteria = 1 if ROA (t) > ROA (t-1), else 0
        - Accruals Criteria = 1 if Operating Cash Flow / Beginning Total Assets > ROA, else 0
        - Average Total Assets = (Beginning Total Assets + Total Assets) / 2
        - Change in Leverage Criteria = 1 if Long Term Debt / Average Total Assets is lower than in (t-1), else 0
        - Change in Current Ratio Criteria = 1 if Total Current Assets / Total Current Liabilities is higher than
          in (t-1), else 0
        - Number of Shares Criteria = 1 if Common Stock Issued = 0, else 0
        - Gross Margin Criteria = 1 if (Revenue - Cost of Goods Sold) / Revenue is higher than in (t-1), else 0
        - Asset Turnover Criteria = 1 if Revenue / Beginning Total Assets is higher than in (t-1), else 0
        - Piotroski Score = SUM of the nine criteria (0 to 9)

        With `trailing`, the flows (net income, operating cash flow, revenue, cost of goods sold and common stock
        issued) are summed over the trailing periods, while the balance sheet items (total assets, long term debt,
        current assets and current liabilities) are taken at the end of each period.

        Please see Piotroski, Joseph D. "Value Investing: The Use of Historical Financial Statement
        Information to Separate Winners from Losers." Journal of Accounting Research, Vol. 38,
        Supplement, 2000, pp. 1-41.

        Also known as: Piotroski F-score, financial strength, quality score.

        Args:
            trailing (int | None, optional): The trailing period to use for the calculation. The window
                counts calendar periods, so a period without a report makes it NaN. Defaults to None, which
                is 4 on quarterly data (a trailing twelve months), as the score was estimated on yearly
                statements.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: DataFrame containing the Piotroski F-Score and its components.

        Periods for which the financial statements have not been reported yet are returned as NaN
        rather than being scored zero across the board.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["F", "GM", "TSLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_piotroski_score().xs("Piotroski Score", level=1)
        ```

        Which returns:

        |      |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|
        | F    |      5 |      6 |      6 |      3 |
        | GM   |      6 |      6 |      7 |      5 |
        | TSLA |      8 |      3 |      4 |      5 |
        """
        # The score was estimated on yearly statements, so quarterly data is summed into a
        # year (a trailing twelve months) unless a trailing period is given.
        if trailing is None and self._quarterly:
            trailing = 4
        piotroski_score = {}

        # Every change-based criterion compares with the same period a year earlier: the previous column on yearly data and four quarters back on quarterly data, so seasonality does not drive the signal.  # noqa: E501
        comparison_lag = 4 if self._quarterly else 1

        net_income = trailing_sum(
            self._income_statement.loc[:, "Net Income", :], trailing
        )
        operating_cashflow = trailing_sum(
            self._cash_flow_statement.loc[:, "Operating Cash Flow", :], trailing
        )
        revenue = trailing_sum(self._income_statement.loc[:, "Revenue", :], trailing)
        cost_of_goods_sold = trailing_sum(
            self._income_statement.loc[:, "Cost of Goods Sold", :], trailing
        )
        common_stock_issued = trailing_sum(
            self._cash_flow_statement.loc[:, "Common Stock Issued", :], trailing
        )
        # Piotroski (2000) takes the balance sheet items at the fiscal period end, so they stay point in time while the flows above cover the trailing window.  # noqa: E501
        total_assets = self._balance_sheet_statement.loc[:, "Total Assets", :]
        long_term_debt = self._balance_sheet_statement.loc[:, "Long Term Debt", :]
        current_assets = self._balance_sheet_statement.loc[:, "Total Current Assets", :]
        current_liabilities = self._balance_sheet_statement.loc[
            :, "Total Current Liabilities", :
        ]

        # Piotroski (2000) scales ROA, CFO, accruals and asset turnover by the total assets at the beginning of the window the flows cover (the previous period's closing balance, or that of `trailing` periods back, one year earlier with trailing=4 on quarterly data) and the leverage signal by the average of that beginning and the ending balance.  # noqa: E501
        beginning_total_assets = total_assets.shift(trailing or 1, axis=1)
        average_total_assets = (beginning_total_assets + total_assets) / 2

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
                lag=comparison_lag,
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
                lag=comparison_lag,
            )
        )

        piotroski_score["Change in Current Ratio Criteria"] = (
            piotroski_model.get_change_in_current_ratio_criteria(
                current_assets=current_assets,
                current_liabilities=current_liabilities,
                lag=comparison_lag,
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
                lag=comparison_lag,
            )
        )

        piotroski_score["Asset Turnover Criteria"] = (
            piotroski_model.get_asset_turnover_ratio_criteria(
                sales=revenue,
                beginning_total_assets=beginning_total_assets,
                lag=comparison_lag,
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
            & beginning_total_assets.notna()
            & net_income.notna()
            & operating_cashflow.notna()
            & long_term_debt.notna()
            & current_assets.notna()
            & current_liabilities.notna()
            & revenue.notna()
            & cost_of_goods_sold.notna()
            & common_stock_issued.notna()
        )

        piotroski_results = piotroski_results.astype(float).where(
            piotroski_model.get_reported_mask(
                reported, comparison_lag, piotroski_results
            )
        )

        # Keep only the periods with a score for at least one ticker. The beginning total assets make the first period (or `trailing` periods) incomplete and the change-based criteria need the period a year earlier complete as well, so a period with only the level criteria is dropped rather than reported without a score.  # noqa: E501
        piotroski_results = piotroski_results.loc[
            :, piotroski_results.xs("Piotroski Score", level=1).notna().any()
        ]

        # The F-Score is a count of satisfied criteria, so report it as such whenever every remaining period is fully reported and the float cast above is no longer needed  # noqa: E501
        if not piotroski_results.isna().to_numpy().any():
            piotroski_results = piotroski_results.astype(int)

        return piotroski_results.loc[:, self._start_date : self._end_date]

    @handle_errors
    def get_beneish_m_score(
        self,
        method: str = "eight",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        The Beneish M-Score shows how likely it is that a company has manipulated its earnings.

        It is a probabilistic model, developed by Messod Beneish, that uses eight financial
        ratios derived from a company's financial statements to identify whether a company has
        manipulated its earnings. It is a natural companion to the Altman Z-Score and Piotroski
        F-Score, using the same normalized financial statements as its input.

        The formula is as follows, where (t-1) is the same period a year earlier (the previous column on yearly
        data, four quarters back on quarterly data, so quarterly comparisons are year over year):

        - DSRI (Days Sales in Receivables Index) = (Net Receivables / Revenue) (t) / (Net Receivables / Revenue) (t-1)
        - GMI (Gross Margin Index) = Gross Margin (t-1) / Gross Margin (t), with Gross Margin = (Revenue - Cost of
          Goods Sold) / Revenue
        - AQI (Asset Quality Index) = Asset Quality (t) / Asset Quality (t-1), with Asset Quality = 1 - (Total
          Current Assets + Property, Plant and Equipment) / Total Assets
        - SGI (Sales Growth Index) = Revenue (t) / Revenue (t-1)
        - DEPI (Depreciation Index) = Depreciation Rate (t-1) / Depreciation Rate (t), with Depreciation Rate =
          Depreciation and Amortization / (Depreciation and Amortization + Property, Plant and Equipment)
        - SGAI (SG&A Expenses Index) = (SG&A Expenses / Revenue) (t) / (SG&A Expenses / Revenue) (t-1)
        - LVGI (Leverage Index) = Leverage (t) / Leverage (t-1), with Leverage = (Total Current Liabilities + Long
          Term Debt) / Total Assets
        - TATA (Total Accruals to Total Assets) = (Net Income - Operating Cash Flow) / Total Assets
        - M-Score = -4.84 + 0.92 * DSRI + 0.528 * GMI + 0.404 * AQI + 0.892 * SGI + 0.115 * DEPI - 0.172 * SGAI +
          4.679 * TATA - 0.327 * LVGI

        With `trailing`, the flows are summed over the trailing periods, while the balance sheet items are taken
        at the end of each period.

        The five-variable model of Beneish (1997) leaves out SGAI, LVGI and TATA:

        - M-Score (five variables) = -6.065 + 0.823 * DSRI + 0.906 * GMI + 0.593 * AQI + 0.717 * SGI + 0.107 * DEPI

        The -1.78 threshold belongs to the eight-variable model; the five-variable version uses -2.22.
        Fast-growing companies are flagged more often because sales growth itself raises the score, and the
        model was estimated without financial companies and is not suited to banks and insurers, so a score
        above the threshold is a prompt to read the accounts, not a finding.

        Also known as: Beneish M-Score, earnings manipulation score.

        Args:
            method (str, optional): The variant of the M-Score. Options are: "eight" (the eight-variable
                model of Beneish, 1999) or "five" (the five-variable model of Beneish, 1997, which returns its
                five indices and score). Defaults to "eight".
            rounding (int, optional): The number of decimals to round the results to. Defaults to None.
            growth (bool, optional): Whether to calculate the growth of the values. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int | None, optional): The trailing period to use for the calculation. The window
                counts calendar periods, so a period without a report makes it NaN. Defaults to None, which
                is 4 on quarterly data (a trailing twelve months), as the score was estimated on yearly
                statements.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: DataFrame containing the Beneish M-Score and its components.

        Notes:
        - A M-Score greater than -1.78 suggests that the company is likely to be an earnings
        manipulator. A M-Score lower than -1.78 suggests the company is unlikely to be a manipulator.
        - As with the Altman Z-Score and Piotroski F-Score, this is a probabilistic, not a
        definitive, indicator and should be combined with further fundamental analysis.
        - Every index compares the current period with the same period a year earlier, so the first
        year of the results (one period on yearly data, four on quarterly data) has no M-Score.
        - Total Accruals are Net Income minus Cash Flow from Operations, the cash flow form of
        Beneish, Lee and Nichols (2013), rather than the balance sheet form of Beneish (1999).
        - The Depreciation Index uses Depreciation and Amortization, since statements report the two
        as one line, where Beneish (1999) uses depreciation alone.

        References:
        - Beneish, Messod D. "Detecting GAAP Violation: Implications for Assessing Earnings
        Management among Firms with Extreme Financial Performance." Journal of Accounting and
        Public Policy, Vol. 16, No. 3, 1997, pp. 271-309.
        - Beneish, Messod D. "The Detection of Earnings Manipulation." Financial Analysts Journal,
        Vol. 55, No. 5, 1999, pp. 24-36.
        - Beneish, Messod D., Charles M. C. Lee, and D. Craig Nichols. "Earnings Manipulation and
        Expected Returns." Financial Analysts Journal, Vol. 69, No. 2, 2013, pp. 57-82.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["NFLX", "DIS", "CMCSA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_beneish_m_score().xs("Beneish M-Score", level=1)
        ```

        Which returns:

        |       |    2021 |    2022 |    2023 |    2024 |    2025 |
        |:------|--------:|--------:|--------:|--------:|--------:|
        | CMCSA | -2.6609 | -2.8625 | -2.661  | -2.6852 | -2.734  |
        | DIS   | -2.5143 | -2.5357 | -2.6392 | -2.6963 | -2.595  |
        | NFLX  | -1.6697 | -1.3345 | -2.5628 | -2.3319 | -2.1611 |
        """
        method = method.lower()
        check_option("method", method, ("eight", "five"))
        # The score was estimated on yearly statements, so quarterly data is summed into a
        # year (a trailing twelve months) unless a trailing period is given.
        if trailing is None and self._quarterly:
            trailing = 4
        beneish_m_score = {}

        # Every index compares with the same period a year earlier: the previous column on yearly data and four quarters back on quarterly data, so seasonality does not drive the score.  # noqa: E501
        comparison_lag = 4 if self._quarterly else 1

        # Beneish (1999) takes the balance sheet items at the fiscal period end, so they stay point in time while the flows cover the trailing window.  # noqa: E501
        net_receivables = self._balance_sheet_statement.loc[:, "Net Receivables", :]
        revenue = trailing_sum(self._income_statement.loc[:, "Revenue", :], trailing)
        cost_of_goods_sold = trailing_sum(
            self._income_statement.loc[:, "Cost of Goods Sold", :], trailing
        )
        total_current_assets = self._balance_sheet_statement.loc[
            :, "Total Current Assets", :
        ]
        property_plant_and_equipment = self._balance_sheet_statement.loc[
            :, "Property, Plant and Equipment", :
        ]
        total_assets = self._balance_sheet_statement.loc[:, "Total Assets", :]
        depreciation_and_amortization = trailing_sum(
            self._income_statement.loc[:, "Depreciation and Amortization", :], trailing
        )
        selling_general_and_administrative_expenses = trailing_sum(
            self._income_statement.loc[
                :, "Selling, General and Administrative Expenses", :
            ],
            trailing,
        )
        total_current_liabilities = self._balance_sheet_statement.loc[
            :, "Total Current Liabilities", :
        ]
        long_term_debt = self._balance_sheet_statement.loc[:, "Long Term Debt", :]
        net_income = trailing_sum(
            self._income_statement.loc[:, "Net Income", :], trailing
        )
        cash_flow_from_operations = trailing_sum(
            self._cash_flow_statement.loc[:, "Operating Cash Flow", :], trailing
        )

        beneish_m_score["Days Sales in Receivables Index"] = (
            beneish_model.get_days_sales_in_receivables_index(
                net_receivables=net_receivables,
                revenue=revenue,
                lag=comparison_lag,
            )
        )

        beneish_m_score["Gross Margin Index"] = beneish_model.get_gross_margin_index(
            revenue=revenue,
            cost_of_goods_sold=cost_of_goods_sold,
            lag=comparison_lag,
        )

        beneish_m_score["Asset Quality Index"] = beneish_model.get_asset_quality_index(
            total_current_assets=total_current_assets,
            property_plant_and_equipment=property_plant_and_equipment,
            total_assets=total_assets,
            lag=comparison_lag,
        )

        beneish_m_score["Sales Growth Index"] = beneish_model.get_sales_growth_index(
            revenue=revenue,
            lag=comparison_lag,
        )

        beneish_m_score["Depreciation Index"] = beneish_model.get_depreciation_index(
            depreciation_and_amortization=depreciation_and_amortization,
            property_plant_and_equipment=property_plant_and_equipment,
            lag=comparison_lag,
        )

        beneish_m_score["SGA Expenses Index"] = (
            beneish_model.get_selling_general_and_administrative_expenses_index(
                selling_general_and_administrative_expenses=(
                    selling_general_and_administrative_expenses
                ),
                revenue=revenue,
                lag=comparison_lag,
            )
        )

        beneish_m_score["Leverage Index"] = beneish_model.get_leverage_index(
            total_current_liabilities=total_current_liabilities,
            long_term_debt=long_term_debt,
            total_assets=total_assets,
            lag=comparison_lag,
        )

        beneish_m_score["Total Accruals to Total Assets"] = (
            beneish_model.get_total_accruals_to_total_assets(
                net_income=net_income,
                cash_flow_from_operations=cash_flow_from_operations,
                total_assets=total_assets,
            )
        )

        if method == "five":
            five_variables = {
                name: beneish_m_score[name]
                for name in [
                    "Days Sales in Receivables Index",
                    "Gross Margin Index",
                    "Asset Quality Index",
                    "Sales Growth Index",
                    "Depreciation Index",
                ]
            }
            five_variables["Beneish M-Score"] = (
                beneish_model.get_beneish_five_variable_m_score(
                    days_sales_in_receivables_index=five_variables[
                        "Days Sales in Receivables Index"
                    ],
                    gross_margin_index=five_variables["Gross Margin Index"],
                    asset_quality_index=five_variables["Asset Quality Index"],
                    sales_growth_index=five_variables["Sales Growth Index"],
                    depreciation_index=five_variables["Depreciation Index"],
                )
            )
            return self._finalize_components(
                five_variables, rounding, growth, lag, standardize
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

        return self._finalize_components(
            beneish_m_score, rounding, growth, lag, standardize
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
            return trailing_sum(self._rows(statement, row), trailing)

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

    @handle_errors
    def get_magic_formula(
        self,
        diluted: bool = True,
        trailing: int | None = None,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate Joel Greenblatt's Magic Formula ranking, which shows the good businesses at a
        low price.

        Companies are ranked on two measures and the ranks are added; the lowest total combines
        a good business with a low price:

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
            trailing (int | None, optional): Sum the operating income over this many periods
                (e.g. 4 quarters, a trailing twelve months). The enterprise value and the
                capital are taken at the period end. The window counts calendar periods, so a
                period without a report makes it NaN. Defaults to None, which is 4 for
                quarterly data (a year of operating income) and 1 otherwise.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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
        | AAPL  |      1 |      1 |      1 |      1 |      1 |
        | GOOGL |      4 |      2 |      2 |      1 |      4 |
        | INTC  |      3 |      5 |      5 |      5 |      5 |
        | META  |      1 |      2 |      2 |      3 |      2 |
        | MSFT  |      4 |      4 |      4 |      3 |      2 |
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
            operating_income=inputs["Operating Income"].reindex(self._peer_tickers),
            enterprise_value=enterprise_value.reindex(self._peer_tickers),
            net_working_capital=net_working_capital.reindex(self._peer_tickers),
            net_fixed_assets=inputs["Property, Plant and Equipment"].reindex(
                self._peer_tickers
            ),
        )
        # The ranks need the operating income of a full trailing sum, so the periods before
        # the first one are left out (and the result is empty, with a warning, without one).
        return self._finish_score(
            results,
            rounding,
            first_columns=(trailing - 1) if trailing else 0,
            name="The Magic Formula",
        )

    @handle_errors
    def get_mohanram_g_score(
        self,
        variance_periods: int = 5,
        trailing: int | None = None,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate Partha Mohanram's G-Score, which separates winners from losers among growth
        companies.

        It is a score from 0 to 8 for companies with a low book-to-market ratio, where value
        measures such as the Piotroski F-Score say little.

        Each signal is 1 when the company does better than the median of its peers, the
        Toolkit's tickers:

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
            variance_periods (int, optional): The number of yearly values (at least 2) the
                variances of G4 (return on assets) and G5 (revenue growth) are calculated over,
                of which at least three (or all, when fewer are asked for) must be available. With
                `trailing` the values are `trailing` periods apart (one a year with 4
                quarters); without it, one per period. Defaults to 5, the five years of the
                paper.
            trailing (int | None, optional): Sum the flows over this many periods (e.g. 4
                quarters, a trailing twelve months) and compare with the values `trailing`
                periods back, so that with quarterly data every change is year over year.
                Balance sheet items are taken at the period end. The window counts calendar
                periods, so a period without a report makes it NaN. Defaults to None, which
                is 4 for quarterly data (a year) and 1 otherwise; pass 1 for quarterly data
                to compare quarter over quarter.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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
        - The variances are of yearly values, as in the paper (Tables 2 to 4: the variance of
          the return on assets and of the annual sales growth over the past three to five
          years), also with quarterly data, where the values are trailing years a year apart.
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
        | AAPL  |      4 |      4 |      2 |
        | ADBE  |      5 |      5 |      5 |
        | GOOGL |      4 |      4 |      6 |
        | META  |      6 |      5 |      4 |
        | MSFT  |      4 |      5 |      3 |
        | NVDA  |      3 |      3 |    nan |
        """
        # A variance needs two values at least.
        validate_window("variance_periods", variance_periods, 2)

        # The scores were estimated on yearly statements, so quarterly data is summed into a
        # year (a trailing twelve months) unless a trailing period is given.
        if trailing is None and self._quarterly:
            trailing = 4
        inputs = {
            name: frame.reindex(self._peer_tickers)
            for name, frame in self._score_inputs(trailing).items()
        }
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
            first_columns=first_columns,
            name="The Mohanram G-Score",
        )

    @handle_errors
    def get_dechow_f_score(
        self,
        trailing: int | None = None,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Dechow F-Score, the probability that a company's financial statements are
        misstated relative to the average company.

        It was estimated on companies the SEC charged with misstating their earnings, and
        combines:

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
                Balance sheet items are taken at the period end. The window counts calendar
                periods, so a period without a report makes it NaN. Defaults to None, which
                is 4 for quarterly data (a year) and 1 otherwise; pass 1 for quarterly data
                to compare quarter over quarter.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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
        | AAPL | 0.4226 | 0.3916 | 0.3913 | 0.3962 |
        | MSFT | 0.811  | 0.7759 | 1.018  | 0.8628 |
        | SMCI | 2.797  | 0.6471 | 6.128  | 0.8275 |
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
            first_columns=first_columns,
            name="The Dechow F-Score",
        )

    @handle_errors
    def get_montier_c_score(
        self,
        trailing: int | None = None,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate James Montier's C-Score, a count from 0 to 6 of warning signs that a company
        is "cooking the books".

        Each sign is 1 when present:

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
                Balance sheet items are taken at the period end. The window counts calendar
                periods, so a period without a report makes it NaN. Defaults to None, which
                is 4 for quarterly data (a year) and 1 otherwise; pass 1 for quarterly data
                to compare quarter over quarter.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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
        | AAPL |      4 |      2 |      3 |      3 |      3 |
        | MSFT |      5 |      2 |      5 |      3 |      2 |
        | SMCI |      5 |      5 |      2 |      5 |      2 |
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
            first_columns=first_columns,
            name="The Montier C-Score",
        )

    @handle_errors
    def get_quality_score(
        self,
        growth_years: int = 5,
        trailing: int | None = None,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        The Quality Score follows the Quality Minus Junk (QMJ) score of Asness, Frazzini and
        Pedersen (2019).

        A company is of high quality when it is profitable, growing, safe and pays out to its
        shareholders. Every measure is ranked across the tickers given in each period and turned
        into a z-score, each of the four components is the z-score of the average of its
        measures, and the Quality Score is the z-score of their sum. A positive score is a
        higher quality than the average of the tickers given.

        The measures are as follows (a higher value is a higher quality, so the measures marked
        "low" enter with a minus sign):

            - Profitability: Gross Profits to Assets ((Revenue - Cost of Goods Sold) / Total
            Assets, Novy-Marx 2013), Return on Equity, Return on Assets, Cash Flow to Assets
            ((Operating Cash Flow - Capital Expenditure) / Total Assets), Gross Margin and low
            accruals ((Operating Cash Flow - Net Income) / Total Assets)
            - Growth: the change in Gross Profits to Assets, Return on Equity, Return on Assets,
            Cash Flow to Assets and Gross Margin over `growth_years`
            - Safety: low Beta (of the period's returns against the benchmark), low leverage
            (Total Debt / Total Assets), low Ohlson O-Score, high Altman Z-Score and low
            earnings volatility (the standard deviation of the Return on Equity over
            `growth_years`)
            - Payout: low equity issuance (-ln of the change in the diluted shares over a year),
            low debt issuance (-ln of the change in Total Debt over a year) and the net payout
            ratio ((Dividends + Share Repurchases - Shares Issued) / Net Income)

        Also known as: QMJ, Quality Minus Junk, quality factor, quality composite.

        Args:
            growth_years (int, optional): The number of years over which the growth and the
                earnings volatility are measured. Defaults to 5, as in the paper.
            trailing (int | None, optional): Sum the flows over this many periods. The window
                counts calendar periods, so a period without a report makes it NaN. Defaults to
                None, which is 4 for quarterly data (a trailing twelve months) and 1 otherwise.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the rounding of the Toolkit (4 unless set
                otherwise).
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per ticker and period the z-scores of Profitability, Growth, Safety
            and Payout, and the Quality Score.

        Notes:
        - The scores are relative to the tickers given: AQR ranks every stock in the market,
        while this ranks the tickers passed to the Toolkit, so pass a broad or comparable set
        (at least three tickers, preferably many more). With fewer than three, it is empty.
        - The growth measures need `growth_years` of earlier statements, so the first Quality
        Score is `growth_years` after the first period (with quarterly data, plus the three
        quarters of the first trailing year). Set an earlier start_date to obtain it.
        - The paper measures growth as the change in profits per share scaled by the assets
        of five years earlier; this takes the change in the profitability ratios themselves,
        and it leaves out the idiosyncratic volatility of the safety component.
        - A ticker without one of the components has no Quality Score.
        - Asness, C. S., A. Frazzini and L. H. Pedersen, "Quality Minus Junk," Review of
        Accounting Studies 24, 2019, pp. 34-112. Novy-Marx, R., "The Other Side of Value: The
        Gross Profitability Premium," Journal of Financial Economics 108(1), 2013, pp. 1-28.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "INTC", "IBM", "ORCL"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2016-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_quality_score().xs("Quality Score", level=1)
        ```

        Which returns:

        |       |    2019 |    2020 |    2021 |    2022 |    2023 |    2024 |     2025 |
        |:------|--------:|--------:|--------:|--------:|--------:|--------:|---------:|
        | AAPL  |  0.6124 |  0.6124 |  1.0206 |  1.4289 |  0.6124 |  0.6124 |   1.3363 |
        | GOOGL | -1.0206 | -1.0206 |  0.2041 |  0.6124 |  1.0206 |  1.0206 |   0.8018 |
        | IBM   | -1.4289 | -1.4289 | -0.6124 | -0.6124 | -0.6124 | -0.6124 |  -0.2673 |
        | INTC  | -0.6124 | -0.6124 | -1.0206 | -1.4289 | -1.4289 | -1.4289 |  -1.3363 |
        | META  |  1.0206 |  1.4289 |  1.4289 |  0.2041 | -0.2041 |  0.2041 |   0.2673 |
        | MSFT  |  0.2041 |  1.0206 |  0.6124 |  1.0206 |  0.2041 | -0.2041 |  -0.8018 |
        | NVDA  |  1.4289 |  0.2041 | -0.2041 | -0.2041 |  1.4289 |  1.4289 | nan      |
        | ORCL  | -0.2041 | -0.2041 | -1.4289 | -1.0206 | -1.0206 | -1.0206 | nan      |
        """
        if len(self._peer_tickers) < quality_model.QUALITY_MINIMUM_TICKERS:
            logger.warning(
                "The Quality Score ranks the tickers against each other and needs at least "
                "three tickers, while %s were given.",
                len(self._peer_tickers),
            )
            return pd.DataFrame()

        if trailing is None and self._quarterly:
            trailing = 4
        year = 4 if self._quarterly else 1
        growth_lag = growth_years * year
        inputs = self._score_inputs(trailing)

        total_assets = inputs["Total Assets"]
        net_income = inputs["Net Income"]
        operating_cash_flow = inputs["Operating Cash Flow"]
        gross_profit = inputs["Revenue"] - inputs["Cost of Goods Sold"]
        total_debt = inputs["Total Debt"]
        total_equity = self._rows(self._balance_sheet_statement, "Total Equity")
        shares = self._rows(self._income_statement, "Weighted Average Shares Diluted")

        profitability = {
            "Gross Profits to Assets": gross_profit / total_assets,
            "Return on Equity": net_income / total_equity,
            "Return on Assets": net_income / total_assets,
            "Cash Flow to Assets": (
                operating_cash_flow - inputs["Capital Expenditure"].abs()
            )
            / total_assets,
            "Gross Margin": gross_profit / inputs["Revenue"],
            "Low Accruals": (operating_cash_flow - net_income) / total_assets,
        }
        growth = {
            name: profitability[name] - profitability[name].shift(growth_lag, axis=1)
            for name in [
                "Gross Profits to Assets",
                "Return on Equity",
                "Return on Assets",
                "Cash Flow to Assets",
                "Gross Margin",
            ]
        }

        period = "quarterly" if self._quarterly else "yearly"
        returns = self._within_historical_data[period]["Return"]
        # Periods by tickers, as the tickers are at least three.
        beta = pd.DataFrame(
            get_beta(returns[self._tickers], returns[self._benchmark_name])
        ).T
        beta.columns = beta.columns.asfreq(total_assets.columns.freq)

        safety = {
            "Low Beta": -beta.reindex(
                index=self._tickers, columns=total_assets.columns
            ),
            "Low Leverage": -(total_debt / total_assets),
            "Low Ohlson O-Score": -self.get_ohlson_o_score(trailing=trailing)
            .xs("Ohlson O-Score", level=1)
            .reindex(index=self._tickers, columns=total_assets.columns),
            "Altman Z-Score": self.get_altman_z_score(trailing=trailing)
            .xs("Altman Z-Score", level=1)
            .reindex(index=self._tickers, columns=total_assets.columns),
            "Low Earnings Volatility": -profitability["Return on Equity"]
            .T.rolling(growth_lag, min_periods=growth_lag)
            .std()
            .T,
        }

        # Logarithms of the change only exist between two positive values.
        def log_change(frame: pd.DataFrame) -> pd.DataFrame:
            previous = frame.shift(year, axis=1)
            return np.log(frame.where(frame > 0) / previous.where(previous > 0))

        cash_flow = self._cash_flow_statement
        dividends = trailing_sum(self._rows(cash_flow, "Dividends Paid"), trailing)
        buybacks = trailing_sum(
            self._rows(cash_flow, "Common Stock Purchased"), trailing
        )
        net_payout = (
            dividends.abs().fillna(0)
            + buybacks.abs().fillna(0)
            - inputs["Common Stock Issued"].abs().fillna(0)
        )
        payout = {
            "Low Equity Issuance": -log_change(shares),
            "Low Debt Issuance": -log_change(total_debt),
            "Net Payout Ratio": net_payout / net_income.where(net_income > 0),
        }

        results = quality_model.get_quality_minus_junk(
            {
                component: {
                    name: frame.reindex(self._peer_tickers)
                    for name, frame in measures.items()
                }
                for component, measures in {
                    "Profitability": profitability,
                    "Growth": growth,
                    "Safety": safety,
                    "Payout": payout,
                }.items()
            }
        )

        return self._finish_score(
            results,
            rounding,
            first_columns=growth_lag + ((trailing or 1) - 1),
            name="The Quality Score",
        )

    @handle_errors
    def get_accrual_quality(
        self,
        method: Literal["dechow_dichev", "mcnichols"] = "dechow_dichev",
        window: int = 8,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the accrual quality of Dechow and Dichev (2002).

        How well a company's working capital accruals map into its operating cash flows of the
        year before, the year itself and the year after. Accruals anticipate cash flows (a sale
        on credit is collected next year), so accruals that the cash flows do not explain are
        estimation errors; their standard deviation measures the quality of the accruals and of
        the earnings they are part of. A higher value means lower quality. Francis et al. (2005)
        found that companies with poor accrual quality have a higher cost of capital.

        The regression is estimated per company over the `window` years before the period:

            - ΔWC (t) = b0 + b1 CFO (t - 1) + b2 CFO (t) + b3 CFO (t + 1) + e (t)
            - McNichols (2002): ΔWC (t) = b0 + b1 CFO (t - 1) + b2 CFO (t) +
              b3 CFO (t + 1) + b4 ΔRevenue (t) + b5 PP&E (t) + e (t)
            - Accrual Quality = σ(e), the standard deviation of the residuals (ddof=1)

        with ΔWC the increase in working capital (minus the Change in Working Capital of the cash
        flow statement: the change in receivables and inventory less the change in payables,
        taxes payable and other working capital) and every variable divided by the average total
        assets of its year.

        For more information, see the following sources:

        - Dechow, P. M. and Dichev, I. D. (2002). "The Quality of Accruals and Earnings: The Role
          of Accrual Estimation Errors." The Accounting Review, 77(Supplement), 35-59.
        - McNichols, M. F. (2002). "Discussion of The Quality of Accruals and Earnings: The Role of
          Accrual Estimation Errors." The Accounting Review, 77(Supplement), 61-69.
        - Francis, J., LaFond, R., Olsson, P. and Schipper, K. (2005). "The Market Pricing of
          Accruals Quality." Journal of Accounting and Economics, 39(2), 295-327.

        Also known as: Dechow-Dichev model, accruals quality, AQ, DD model, McNichols model.

        Args:
            method (str, optional): Options are: "dechow_dichev" (Dechow and Dichev, 2002), "mcnichols" (McNichols,
                2002, which adds the change in revenue and the property, plant and equipment). Defaults to
                "dechow_dichev".
            window (int, optional): The number of years every regression is estimated on, at
                least one more than the coefficients it estimates: 5 for the Dechow and Dichev
                model and 7 for the McNichols model. Dechow and Dichev required at least 8 years
                per company. Defaults to 8.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per ticker and period the Working Capital Accruals and the Cash Flow from
            Operations (both over the average total assets) and the Accrual Quality.

        Notes:
        - No look-ahead: the regression needs the cash flow of the year after each accrual, so the
          Accrual Quality of period T uses the accruals of the `window` years up to T - 1, whose
          next-year cash flow is known at T at the latest. It therefore needs `window` + 2 years of
          statements: set a start_date at least that many years back.
        - Dechow and Dichev estimate one regression per company over its whole history and
          Francis et al. (2005) estimate them per industry and year; this rolls a regression per
          company over the window, so the measure is known at each period.
        - On quarterly data the flows are summed over the trailing twelve months and the years of
          the regression are four quarters apart.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2010-01-01",
            end_date="2025-12-31",
        )

        # The last six years, each estimated on the eight years before it
        toolkit.models.get_accrual_quality().xs("Accrual Quality", level=1).iloc[:, -6:]
        ```

        Which returns:

        |      |   2020 |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|-------:|
        | AAPL | 0.0295 | 0.0282 | 0.0291 | 0.0199 | 0.0216 | 0.0216 |
        | MSFT | 0.0546 | 0.054  | 0.0346 | 0.0352 | 0.0225 | 0.0197 |
        """
        method = method.lower()
        check_option("method", method, ("dechow_dichev", "mcnichols"))
        # One year more than the coefficients of the regression: the intercept and the three
        # cash flows, plus the change in revenue and the PP&E of the McNichols model.
        validate_window("window", window, 5 if method == "dechow_dichev" else 7)
        year = 4 if self._quarterly else 1
        trailing = year if self._quarterly else None

        total_assets = self._rows(self._balance_sheet_statement, "Total Assets")
        average_assets = (total_assets + total_assets.shift(year, axis=1)) / 2
        average_assets = average_assets.where(average_assets > 0)

        def scaled_flow(statement: pd.DataFrame, row: str) -> pd.DataFrame:
            return trailing_sum(self._rows(statement, row), trailing) / average_assets

        working_capital_accruals = -scaled_flow(
            self._cash_flow_statement, "Change in Working Capital"
        )
        operating_cash_flow = scaled_flow(
            self._cash_flow_statement, "Operating Cash Flow"
        )
        extra = {}
        if method == "mcnichols":
            revenue = trailing_sum(
                self._rows(self._income_statement, "Revenue"), trailing
            )
            extra = {
                "change_in_revenue": (revenue - revenue.shift(year, axis=1))
                / average_assets,
                "property_plant_and_equipment": self._rows(
                    self._balance_sheet_statement, "Property, Plant and Equipment"
                )
                / average_assets,
            }

        results = {
            "Working Capital Accruals": working_capital_accruals,
            "Cash Flow from Operations": operating_cash_flow,
            "Accrual Quality": quality_model.get_accrual_quality(
                working_capital_accruals=working_capital_accruals,
                operating_cash_flow=operating_cash_flow,
                window=window,
                lag=year,
                **extra,
            ),
        }

        # The inputs are shown for every year, so a window longer than the history leaves
        # only the Accrual Quality empty, which the result alone would not explain.
        available_years = working_capital_accruals.shape[1] // year
        if (
            results["Accrual Quality"].isna().all(axis=None)
            and available_years < window + 2
        ):
            logger.warning(
                "The Accrual Quality needs %s years of statements (the window of %s plus the "
                "year before and after it) and %s were available, so it could not be "
                "calculated. Set an earlier start_date or a smaller window to obtain it.",
                window + 2,
                window,
                available_years,
            )

        return self._finish_score(
            results, rounding, first_columns=0, name="The Accrual Quality"
        )

    @handle_errors
    def get_earnings_attributes(
        self,
        window: int = 10,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the accounting-based earnings attributes of Francis, LaFond, Olsson and
        Schipper (2004).

        These are the persistence, predictability and smoothness of a company's earnings over
        the past years. Persistent and predictable earnings are more useful to forecast with,
        and the paper finds that companies with less favourable attributes have a higher cost of
        equity. Together with the accrual quality (`get_accrual_quality`) they are the paper's
        accounting attributes.

        The formula is as follows, each over the `window` years up to the period:

            - Persistence = φ1 of the regression EPS (t) = φ0 + φ1 EPS (t - 1) + ν (t)
            - Predictability = sqrt(σ²(ν)), the square root of the error variance of that
              regression, sqrt(Σ ν² / (N - 2))
            - Smoothness = σ(Net Income (t) / Total Assets (t - 1)) / σ(Cash Flow from Operations
              (t) / Total Assets (t - 1)), the standard deviations with ddof=1

        with EPS the net income from continuing operations (the earnings before extraordinary
        items of the paper) over the weighted average shares. A persistence near 1 means earnings
        shocks last; larger predictability and smoothness values mean less predictable and less
        smooth earnings (as in the paper, larger values are less desirable).

        For more information, see the following sources:

        - Francis, J., LaFond, R., Olsson, P. M. and Schipper, K. (2004). "Costs of Equity and
          Earnings Attributes." The Accounting Review, 79(4), 967-1010.

        Also known as: earnings persistence, earnings predictability, earnings smoothness, earnings
        quality attributes, Francis LaFond Olsson Schipper.

        Args:
            window (int, optional): The number of years of every window, at least 3 for the
                regression; the paper uses 10. Defaults to 10.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per ticker and period the Persistence, Predictability and Smoothness.

        Notes:
        - The windows only use the years up to the period (no look-ahead) and need every one of
          them, plus the year before for the lagged EPS and the beginning total assets: set a
          start_date at least `window` + 1 years back.
        - The net income from continuing operations falls back on the net income where the
          statements do not report it.
        - On quarterly data the flows are summed over the trailing twelve months and the years of
          the window are four quarters apart.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2010-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_earnings_attributes()
        ```

        Which returns:

        |                            |    2018 |    2019 |   2020 |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:---------------------------|--------:|--------:|-------:|-------:|-------:|-------:|-------:|-------:|
        | ('AAPL', 'Persistence')    |  0.9719 |  0.8779 | 0.8644 | 1.3562 | 1.1528 | 0.9926 | 0.9229 | 1.015  |
        | ('AAPL', 'Predictability') |  0.3459 |  0.3408 | 0.3384 | 0.7336 | 0.7608 | 0.7721 | 0.7744 | 0.8457 |
        | ('AAPL', 'Smoothness')     |  0.7622 |  0.7474 | 0.7391 | 0.8799 | 0.8643 | 0.9365 | 0.9284 | 1.1048 |
        | ('MSFT', 'Persistence')    | -0.0581 | -0.3582 | 0.6982 | 1.1725 | 1.1663 | 1.0312 | 1.0803 | 1.0464 |
        | ('MSFT', 'Predictability') |  0.5798 |  1.0244 | 1.2588 | 1.4298 | 1.3595 | 1.4282 | 1.4569 | 1.2686 |
        | ('MSFT', 'Smoothness')     |  1.2413 |  1.2008 | 1.2202 | 1.1846 | 1.5726 | 1.616  | 1.4189 | 1.3102 |
        """
        validate_window("window", window, 3)
        year = 4 if self._quarterly else 1
        trailing = year if self._quarterly else None

        def flow(statement: pd.DataFrame, row: str) -> pd.DataFrame:
            return trailing_sum(self._rows(statement, row), trailing)

        earnings = flow(
            self._income_statement, "Net Income from Continuing Operations"
        ).combine_first(flow(self._income_statement, "Net Income"))
        shares = self._rows(self._income_statement, "Weighted Average Shares")
        beginning_assets = self._rows(
            self._balance_sheet_statement, "Total Assets"
        ).shift(year, axis=1)
        beginning_assets = beginning_assets.where(beginning_assets > 0)

        results = quality_model.get_earnings_attributes(
            earnings_per_share=earnings / shares.where(shares > 0),
            earnings_to_assets=earnings / beginning_assets,
            cash_flow_to_assets=flow(self._cash_flow_statement, "Operating Cash Flow")
            / beginning_assets,
            window=window,
            lag=year,
        )

        return self._finish_score(
            results, rounding, first_columns=0, name="The Earnings Attributes"
        )

    @handle_errors
    def get_graham_criteria(
        self,
        minimum_revenue: float = 800_000_000,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the seven criteria Benjamin Graham sets for the stocks of the defensive
        investor in The Intelligent Investor (chapter 14), with the number met.

        They are an adequate size, a strong financial condition, earnings stability, a dividend
        record, earnings growth, a moderate price to earnings and a moderate price to assets.
        Graham meant them as a minimum of quality and a maximum of price for an investor who
        does not want to analyse companies in depth.

        The formula is as follows:

            - Adequate Size: Revenue >= minimum_revenue
            - Strong Financial Condition: Current Assets >= 2 * Current Liabilities and Long Term
              Debt <= Current Assets - Current Liabilities (net current assets)
            - Earnings Stability: a positive Net Income in each of the past 10 years
            - Dividend Record: dividends paid in each of the past 20 years
            - Earnings Growth: the average EPS of the last 3 years at least a third above the
              average EPS of the 3 years ending ten years earlier
            - Moderate Price to Earnings: Price <= 15 * the average EPS of the last 3 years
            - Moderate Price to Assets: Price / Earnings * Price / Book Value <= 22.5, with the
              earnings the average EPS of the last 3 years (Graham's 15 times earnings and 1.5
              times book value, a lower multiple of the one allowing a higher multiple of the
              other)
            - Criteria Met = the number of criteria met; Criteria Evaluated = the number that
              could be evaluated

        Every criterion is 1 when met, 0 when not and NaN when the statements do not go back far
        enough (the dividend record needs 20 years of statements, the earnings growth 13).

        For more information, see the following sources:

        - Graham, B. (1973). "The Intelligent Investor." Harper & Row, 4th revised edition,
          chapter 14 (Stock Selection for the Defensive Investor).

        Also known as: Graham defensive investor criteria, Graham screen, Graham's seven criteria,
        defensive stock selection.

        Args:
            minimum_revenue (float, optional): The yearly revenue of the size criterion, in the
                reporting currency. Graham asked for at least $100 million of sales for an
                industrial company in 1971, about $800 million in 2025 prices (the US consumer price
                index rose about eightfold). Defaults to 800,000,000.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per ticker and period the seven criteria (1, 0 or NaN), Criteria Met and
            Criteria Evaluated.

        Notes:
        - Graham's criteria for public utilities differ (debt below twice the equity at book
          value instead of the current ratio); these are the criteria for industrial companies.
        - The price is the close at the end of the fiscal period (traded, adjusted for splits),
          the EPS the diluted EPS and the book value the Total Shareholder Equity over the diluted
          weighted average shares. The dividend record counts a year with dividends paid on the
          cash flow statement.
        - On quarterly data the flows are summed over the trailing twelve months and the years
          are four quarters apart.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["KO", "JNJ", "PG"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2000-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_graham_criteria().loc["KO"].iloc[:, -3:]
        ```

        Which returns:

        |                            |   2023 |   2024 |   2025 |
        |:---------------------------|-------:|-------:|-------:|
        | Adequate Size              |      1 |      1 |      1 |
        | Strong Financial Condition |      0 |      0 |      0 |
        | Earnings Stability         |      1 |      1 |      1 |
        | Dividend Record            |      1 |      1 |      1 |
        | Earnings Growth            |      0 |      0 |      1 |
        | Moderate Price to Earnings |      0 |      0 |      0 |
        | Moderate Price to Assets   |      0 |      0 |      0 |
        | Criteria Met               |      3 |      3 |      4 |
        | Criteria Evaluated         |      7 |      7 |      7 |
        """
        year = 4 if self._quarterly else 1
        trailing = year if self._quarterly else None
        balance = self._balance_sheet_statement

        def flow(statement: pd.DataFrame, row: str) -> pd.DataFrame:
            return trailing_sum(self._rows(statement, row), trailing)

        shares = self._rows(self._income_statement, "Weighted Average Shares Diluted")
        criteria = quality_model.get_graham_criteria(
            revenue=flow(self._income_statement, "Revenue"),
            total_current_assets=self._rows(balance, "Total Current Assets"),
            total_current_liabilities=self._rows(balance, "Total Current Liabilities"),
            long_term_debt=self._rows(balance, "Long Term Debt"),
            net_income=flow(self._income_statement, "Net Income"),
            dividends_paid=flow(self._cash_flow_statement, "Dividends Paid"),
            earnings_per_share=flow(self._income_statement, "EPS Diluted"),
            share_price=self._period_prices(),
            book_value_per_share=self._rows(balance, "Total Shareholder Equity")
            / shares.where(shares > 0),
            minimum_revenue=minimum_revenue,
            lag=year,
        )
        frames = list(criteria.values())
        criteria["Criteria Met"] = sum(frame.fillna(0) for frame in frames).where(
            sum(frame.notna() for frame in frames) > 0
        )
        criteria["Criteria Evaluated"] = sum(
            frame.notna().astype(float) for frame in frames
        ).where(sum(frame.notna() for frame in frames) > 0)

        return self._finish_score(criteria, rounding, first_columns=0)

    @handle_errors
    def get_accruals_ratio(
        self,
        trailing: int | None = None,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the accruals ratios: the part of earnings that is not backed by cash, relative
        to the average of the total assets at the start and the end of the year.

        Richard Sloan showed that earnings driven by accruals are less persistent than cash
        earnings and that companies with high accruals earn lower future returns (the accrual
        anomaly).

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
                Balance sheet items are taken at the period end. The window counts calendar
                periods, so a period without a report makes it NaN. Defaults to None, which
                is 4 for quarterly data (a year) and 1 otherwise; pass 1 for quarterly data
                to compare quarter over quarter.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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
        return self._finish_score(results, rounding, first_columns=trailing or 1)

    @handle_errors
    def get_discretionary_accruals(
        self,
        method: str = "time_series",
        min_observations: int = 10,
        intercept: bool = False,
        trailing: int | None = None,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the discretionary accruals of the modified Jones model.

        They are the part of a company's accruals that its change in revenues and its fixed
        assets do not explain and that is therefore attributed to management's discretion. Jones
        (1991) modelled the accruals a company needs as a function of its change in revenues
        (working capital moves with sales) and its property, plant and equipment (depreciation).
        Dechow, Sloan and Sweeney (1995) subtracted the change in receivables from the change in
        revenues, because revenue booked on credit can itself be managed, and found this
        modified model the most powerful at detecting earnings management. Large positive
        discretionary accruals point at earnings that were managed up.

        The formula is as follows:

            - Total Accruals = (Net Income - Cash Flow from Operations) / Total Assets(t-1)
            - Estimation: Total Accruals = a1 × 1 / A(t-1) + a2 × ΔRevenue / A(t-1) +
              a3 × PPE / A(t-1) + e
            - Non-Discretionary Accruals = a1 × 1 / A(t-1) +
              a2 × (ΔRevenue - ΔReceivables) / A(t-1) + a3 × PPE / A(t-1)
            - Discretionary Accruals = Total Accruals - Non-Discretionary Accruals

        Here A(t-1) is the total assets at the start of the year, ΔRevenue and
        ΔReceivables the changes over the year and PPE the property, plant and equipment
        at the end of it. The coefficients a1, a2 and a3 are estimated with ordinary least
        squares without an intercept, as in Jones (1991); with an intercept a constant a0
        is added to the estimation and the Non-Discretionary Accruals (Kothari, Leone and
        Wasley, 2005).

        For more information, see the following papers:

        - Jones, J.J. (1991). "Earnings Management During Import Relief Investigations."
          Journal of Accounting Research, 29(2), 193-228.
        - Dechow, P.M., Sloan, R.G. and Sweeney, A.P. (1995). "Detecting Earnings
          Management." The Accounting Review, 70(2), 193-225.
        - Kothari, S.P., Leone, A.J. and Wasley, C.E. (2005). "Performance Matched
          Discretionary Accrual Measures." Journal of Accounting and Economics, 39(1),
          163-197.

        Also known as: modified Jones model, abnormal accruals, Jones model, earnings
        management.

        Referenced against:
        - statsmodels, OLS (time series): https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html
        - statsmodels, OLS (with an intercept): https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html

        Args:
            method (str, optional): How the coefficients are estimated. Options are: "time_series" (for every year
                on the company's own earlier years, as Jones (1991) and Dechow, Sloan and Sweeney (1995) did, with
                the change in revenues in the estimation and the change in receivables subtracted only for the year
                itself), "cross_section" (per year across the tickers of the Toolkit, with the change in revenues
                minus the change in receivables in the estimation as well, which is meaningful with many companies
                of one industry). Defaults to "time_series".
            min_observations (int, optional): The fewest observations a regression is
                estimated on: earlier years of the company with "time_series", tickers with
                "cross_section". At least the number of coefficients: 3, or 4 with an
                intercept. Defaults to 10, the minimum estimation period of Dechow, Sloan and
                Sweeney (1995).
            intercept (bool, optional): Whether to add a constant to the regression, as
                Kothari, Leone and Wasley (2005) do, in both methods. It removes the average
                level of the accruals from the other coefficients at the cost of a degree of
                freedom (one observation more is needed). Defaults to False, the formulation
                of Jones (1991) and Dechow, Sloan and Sweeney (1995).
            trailing (int | None, optional): Sum the flows over this many periods (e.g. 4
                quarters, a trailing twelve months) and compare with the values `trailing`
                periods back, so that with quarterly data every change is year over year.
                Balance sheet items are taken at the period end. The window counts calendar
                periods, so a period without a report makes it NaN. Defaults to None, which
                is 4 for quarterly data (a year) and 1 otherwise; pass 1 for quarterly data
                to compare quarter over quarter.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per ticker and period the Total Accruals, Non-Discretionary
            Accruals and Discretionary Accruals, as a share of the total assets at the start
            of the period. The last two are NaN where fewer than `min_observations`
            observations are available.

        Notes:
        - Total accruals are Net Income minus Cash Flow from Operations, the cash flow form
          of Hribar and Collins (2002) that the Beneish M-Score's Total Accruals to Total
          Assets also uses; Jones (1991) and Dechow, Sloan and Sweeney (1995) derived them
          from balance sheet changes, which misstate them around acquisitions and
          divestitures.
        - Jones (1991) uses gross property, plant and equipment. Where a company's
          statements report no gross amount (Financial Modeling Prep's do not), the net
          amount is used in every year, as switching between the two would move the ratio.
        - The time series design estimates every year on the years before it only (an
          expanding window ending a year earlier), so there is no look-ahead. With the
          default of 10 observations, the first discretionary accruals are in the twelfth
          year of statements (a year for the beginning assets and ten to estimate on), so set
          a `start_date` well over a decade back; the Toolkit's data plan may cover fewer
          years, in which case only the Total Accruals are returned. Lowering
          `min_observations` gives earlier but noisier estimates.
        - The cross-section design uses the companies of the same year, so it needs at
          least `min_observations` tickers that report every input. Many later papers
          estimate it per industry and year and add an intercept (Kothari, Leone and
          Wasley, 2005), which the intercept option does.
        - Without an intercept the PPE term carries the average level of the accruals, so a
          company whose PPE intensity or size trends away from its estimation years gets
          discretionary accruals of one sign. In the example below AAPL's and MSFT's PPE
          averaged about 0.11 of their beginning assets in the estimation years and is 0.15
          to 0.17 (AAPL) and 0.24 to 0.45 (MSFT) in the years shown, where their
          discretionary accruals are positive in every year. An intercept removes this
          level bias within the estimation years at the cost of a degree of freedom, and
          gives both companies discretionary accruals of either sign in these years.
        - The expanding window of the time series design starts at the first year loaded,
          so the estimates, and with them the discretionary accruals, depend on the
          `start_date` of the Toolkit.
        - On quarterly data the flows are summed over the trailing twelve months and the
          time series estimation uses the same quarter of earlier years, so that the
          observations do not overlap.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2005-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_discretionary_accruals().loc[:, "2021":]
        ```

        Which returns:

        |                                        |    2021 |    2022 |    2023 |    2024 |    2025 |
        |:---------------------------------------|--------:|--------:|--------:|--------:|--------:|
        | ('AAPL', 'Total Accruals')             | -0.0289 | -0.0637 | -0.0384 | -0.0695 |  0.0014 |
        | ('AAPL', 'Non-Discretionary Accruals') | -0.1208 | -0.084  | -0.0696 | -0.0707 | -0.0794 |
        | ('AAPL', 'Discretionary Accruals')     |  0.0919 |  0.0203 |  0.0312 |  0.0012 |  0.0808 |
        | ('MSFT', 'Total Accruals')             | -0.0513 | -0.0488 | -0.0417 | -0.0738 | -0.067  |
        | ('MSFT', 'Non-Discretionary Accruals') | -0.1076 | -0.0973 | -0.1047 | -0.0961 | -0.1061 |
        | ('MSFT', 'Discretionary Accruals')     |  0.0563 |  0.0485 |  0.063  |  0.0223 |  0.039  |
        """
        validate_window("min_observations", min_observations, 4 if intercept else 3)

        # The model was estimated on yearly statements, so quarterly data is summed into a
        # year (a trailing twelve months) unless a trailing period is given.
        if trailing is None and self._quarterly:
            trailing = 4
        inputs = self._score_inputs(trailing)
        lag = trailing or 1
        # Jones (1991) uses gross fixed assets. A company whose statements have no gross
        # amount uses the net amount in every period, as switching between the two would
        # show a change that is not there.
        gross = inputs["Gross Property, Plant and Equipment"]
        fixed_assets = gross.where(
            gross.notna().any(axis=1), inputs["Property, Plant and Equipment"], axis=0
        )
        results = quality_model.get_modified_jones_accruals(
            net_income=inputs["Net Income"],
            operating_cash_flow=inputs["Operating Cash Flow"],
            total_assets=inputs["Total Assets"],
            revenue=inputs["Revenue"],
            accounts_receivable=inputs["Accounts Receivable"],
            property_plant_and_equipment=fixed_assets,
            lag=lag,
            method=method,
            min_observations=min_observations,
            intercept=intercept,
        )
        # The accruals are scaled by the assets a year back.
        return self._finish_score(
            results, rounding, first_columns=lag, name="The modified Jones model"
        )

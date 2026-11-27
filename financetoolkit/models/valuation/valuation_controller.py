"""Valuation Models Module"""

__docformat__ = "google"

from functools import cache
from typing import Literal

import numpy as np
import pandas as pd

from financetoolkit.helpers import (
    calendar_periods,
    check_trailing,
    require_input,
    trailing_mean,
    trailing_sum,
)
from financetoolkit.models import helpers
from financetoolkit.models.profitability import growth_model
from financetoolkit.models.valuation import (
    cfroi_model,
    easton_model,
    enterprise_model,
    eva_model,
    intrinsic_model,
    wacc_model,
)
from financetoolkit.performance.performance_model import get_beta
from financetoolkit.ratios import valuation_model
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import (
    InvalidArgumentError,
    check_option,
    handle_errors,
    validate_window,
)
from financetoolkit.utilities.logger_model import get_logger
from financetoolkit.utilities.sources_model import track_sources
from financetoolkit.utilities.statistics_model import (
    apply_rounding,
    calculate_growth,
    calculate_standardization,
    finalize_dataset,
)

# ruff: noqa: E501

logger = get_logger()

# The growth rate a company keeps forever when none is given: about the long-run expected
# inflation in the United States, and below the risk-free rate, which Damodaran (Investment
# Valuation, chapter 12) sets as the upper limit of a stable growth rate. The reverse DCF and
# the Monte Carlo valuation use it as their default perpetual growth rate as well.
LONG_RUN_GROWTH_RATE = 0.025

# The highest growth rate in Graham's table of the multiples his growth formula gives (The
# Intelligent Investor, chapter 11), the limit of the growth rate the formula takes by default.
GRAHAM_MAXIMUM_GROWTH_RATE = 0.20


@check_trailing
@filter_shown_columns
@track_sources
class ValuationModels:
    """
    Valuation models: intrinsic value, discounted cash flows, dividend discount models, the cost of capital, economic value added and peer comparison.
    """

    @handle_errors
    def get_enterprise_value_breakdown(
        self,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
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

        Cash and marketable securities are deducted together (Damodaran), the same Enterprise Value
        as toolkit.ratios.get_enterprise_value returns.

        The formula is as follows:

        - Market Capitalization = Share Price * Weighted Average Shares Diluted (Weighted
          Average Shares without dilution), with the share price the traded close at the end of
          each quarter or year
        - Enterprise Value = Market Capitalization + Total Debt + Preferred Equity + Minority Interest -
          Cash and Cash Equivalents - Short Term Investments

        This breakdown is displayed in a DataFrame for each company and includes the option to show growth values as well.

        Also known as: EV breakdown, enterprise value components, EV bridge.

        Args:
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the values. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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
            ["F"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_enterprise_value_breakdown().loc[
            ["Market Capitalization", "Total Debt", "Cash and Cash Equivalents", "Enterprise Value"]
        ]
        ```

        Which returns:

        |                           |        2021 |        2022 |        2023 |        2024 |        2025 |
        |:--------------------------|------------:|------------:|------------:|------------:|------------:|
        | Market Capitalization     | 8.37862e+10 | 4.66828e+10 | 4.92598e+10 | 3.98079e+10 | 5.22045e+10 |
        | Total Debt                | 1.39485e+11 | 1.40474e+11 | 1.51107e+11 | 1.60862e+11 | 1.67573e+11 |
        | Cash and Cash Equivalents | 2.054e+10   | 2.5134e+10  | 2.4862e+10  | 2.2935e+10  | 2.3356e+10  |
        | Enterprise Value          | 1.73781e+11 | 1.43012e+11 | 1.60221e+11 | 1.62345e+11 | 1.81318e+11 |
        """
        average_shares = self._income_statement.loc[
            :,
            "Weighted Average Shares Diluted" if diluted else "Weighted Average Shares",
            :,
        ]

        share_prices = self._period_prices()

        enterprise_value_breakdown = enterprise_model.get_enterprise_value_breakdown(
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
            short_term_investments=(
                self._balance_sheet_statement.loc[:, "Short Term Investments", :]
                if "Short Term Investments"
                in self._balance_sheet_statement.index.get_level_values(1)
                else None
            ),
        )

        if growth:
            enterprise_value_breakdown = calculate_growth(
                dataset=enterprise_value_breakdown,
                lag=lag,
                rounding=rounding if rounding is not None else self._rounding,
            )
        else:
            enterprise_value_breakdown = apply_rounding(
                enterprise_value_breakdown,
                rounding if rounding is not None else self._rounding,
            )

        if standardize:
            enterprise_value_breakdown = calculate_standardization(
                dataset=enterprise_value_breakdown.loc[
                    :, self._start_date : self._end_date
                ],
                rounding=rounding if rounding is not None else self._rounding,
                axis="columns",
            )

        if len(self._tickers) == 1:
            enterprise_value_breakdown = enterprise_value_breakdown.droplevel(level=0)

        return enterprise_value_breakdown.loc[:, self._start_date : self._end_date]

    @handle_errors
    def get_tobins_q_ratio(
        self,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
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
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: DataFrame containing the Market Value of Equity, Total Liabilities,
            Total Assets and Tobin's Q Ratio.

        Notes:
        - This implementation approximates the market value of debt with the book value of
        Total Liabilities, and the replacement cost of assets with the book value of Total
        Assets, consistent with the simplifications used in the Weighted Average Cost of
        Capital calculation elsewhere in this module.
        - The market value of equity (the share price at the end of the fiscal period times the
        period's weighted average shares) and the balances are taken at the end of the period,
        the same date, on yearly and quarterly data alike.

        References:
        - Tobin, James. "A General Equilibrium Approach to Monetary Theory." Journal of Money,
        Credit and Banking, Vol. 1, No. 1, 1969, pp. 15-29.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["INTC", "QCOM", "TXN"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_tobins_q_ratio().xs("Tobin's Q Ratio", level=1)
        ```

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | INTC | 1.6797 | 1.0312 | 1.5308 | 0.9076 | 1.178  |
        | QCOM | 4.3523 | 3.2533 | 3.0273 | 4.0076 | 4.3057 |
        | TXN  | 7.6086 | 6.0875 | 5.3046 | 5.3769 | 5.1094 |
        """
        tobins_q_ratio = {}

        average_shares = self._income_statement.loc[
            :,
            "Weighted Average Shares Diluted" if diluted else "Weighted Average Shares",
            :,
        ]

        share_prices = self._period_prices()

        # The market value and the balances are taken at the end of the period, the same date.
        total_liabilities = self._balance_sheet_statement.loc[:, "Total Liabilities", :]
        total_assets = self._balance_sheet_statement.loc[:, "Total Assets", :]

        market_cap = valuation_model.get_market_cap(
            share_price=share_prices,
            total_shares_outstanding=average_shares,
        )

        tobins_q_ratio["Market Value of Equity"] = market_cap
        tobins_q_ratio["Total Liabilities"] = total_liabilities
        tobins_q_ratio["Total Assets"] = total_assets
        tobins_q_ratio["Tobin's Q Ratio"] = enterprise_model.get_tobins_q_ratio(
            market_value_of_equity=market_cap,
            total_liabilities=total_liabilities,
            total_assets=total_assets,
        )

        return self._finalize_components(
            tobins_q_ratio, rounding, growth, lag, standardize
        )

    @handle_errors
    def _capital_asset_pricing_inputs(
        self,
        risk_free_rate: pd.Series,
        benchmark_returns: pd.Series,
        market_risk_premium: float | None,
        trailing: int | None = None,
    ) -> tuple[pd.Series, pd.Series | pd.DataFrame]:
        """
        The risk-free rate and the expected market return for the Cost of Equity, over the
        span of the flows it is compared with. The market return is the risk-free rate plus
        the market risk premium:

        - market_risk_premium when it is given, for every company.
        - Otherwise the equity risk premium of each company's country (Damodaran's Total
          Equity Risk Premium, from FinancialModelingPrep), a forward-looking premium of a
          few percent. The source keeps no history, so every period uses today's premium,
          also periods long past: a backtest sees the country risk known today.
        - For a company whose country premium is not available (no API key, or a country
          the table does not cover), the benchmark's average return above the risk-free
          rate over every period of the data. A short or strong sample makes this premium
          large: over 2022-2026 the S&P 500's was about 15% a year.

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
                over. The window counts calendar periods, so a period without a report makes
                it NaN. Defaults to None, which is a single period.

        Returns:
            tuple[pd.Series, pd.Series | pd.DataFrame]: The risk-free rate and the expected
            market return, both over the span of the flows. The market return has a column
            per ticker when the premium differs per company.
        """
        periods_per_year = 4 if self._quarterly else 1
        span = trailing or 1

        risk_free_rate = risk_free_rate / periods_per_year
        historical_premium = (benchmark_returns - risk_free_rate).mean()

        # Scaled after the premium is averaged, which compares per-period benchmark returns with the per-period rate.  # noqa: E501
        risk_free_span = risk_free_rate * span

        if market_risk_premium is not None:
            return (
                risk_free_span,
                risk_free_span + market_risk_premium / periods_per_year * span,
            )

        country_premiums = self._country_risk_premiums

        if country_premiums.isna().all():
            return risk_free_span, risk_free_span + historical_premium * span

        if without_premium := country_premiums[country_premiums.isna()].index.tolist():
            logger.info(
                "No country equity risk premium is available for %s, so their Cost of Equity "
                "uses the benchmark's average return above the risk-free rate (%.4f a period).",
                ", ".join(without_premium),
                historical_premium,
            )

        premiums = (country_premiums / periods_per_year).fillna(historical_premium)

        return risk_free_span, pd.DataFrame(
            {
                ticker: risk_free_span + premium * span
                for ticker, premium in premiums.items()
            }
        )

    def _weighted_average_cost_of_capital(
        self,
        diluted: bool = True,
        trailing: int | None = None,
        market_risk_premium: float | None = None,
    ) -> pd.DataFrame:
        """
        The full results of get_weighted_average_cost_of_capital per ticker and line item,
        unrounded and for every period of the statements. The valuation models discount with
        it directly, so that a missing input (such as the benchmark) raises to their own error
        handling rather than handing them an empty result.

        Args:
            diluted (bool, optional): Whether to use diluted shares. Defaults to True.
            trailing (int | None, optional): The trailing period. The window counts calendar
                periods, so a period without a report makes it NaN. Defaults to None.
            market_risk_premium (float | None, optional): The annual equity risk premium.
                Defaults to None, see get_weighted_average_cost_of_capital.

        Returns:
            pd.DataFrame: The WACC and its components per ticker (rows) and period (columns).
        """
        average_shares = self._income_statement.loc[
            :,
            "Weighted Average Shares Diluted" if diluted else "Weighted Average Shares",
            :,
        ]

        years = self._cash_flow_statement.columns
        begin, end = str(years[0]), str(years[-1])

        share_prices = self._period_prices()

        risk_free_rate = self._risk_free_rate_data[
            "quarterly" if self._quarterly else "yearly"
        ].loc[begin:end, "Adj Close"]

        beta = self._beta_per_period(begin, end, trailing)

        benchmark_returns = self._historical_data[
            "quarterly" if self._quarterly else "yearly"
        ].loc[begin:end, "Return"][self._benchmark_name]
        # The Cost of Equity covers the span of the flows: a trailing sum of Interest Expense is the cost of debt over that span.  # noqa: E501
        risk_free_rate, benchmark_returns = self._capital_asset_pricing_inputs(
            risk_free_rate, benchmark_returns, market_risk_premium, trailing
        )

        return wacc_model.get_weighted_average_cost_of_capital(
            share_price=share_prices,
            total_shares_outstanding=average_shares,
            interest_expense=trailing_sum(
                self._income_statement.loc[:, "Interest Expense", :], trailing
            ),
            total_debt=self._balance_sheet_statement.loc[:, "Total Debt", :],
            risk_free_rate=risk_free_rate,
            beta=beta,
            benchmark_returns=benchmark_returns,
            income_tax_expense=trailing_sum(
                self._income_statement.loc[:, "Income Tax Expense", :], trailing
            ),
            income_before_tax=trailing_sum(
                self._income_statement.loc[:, "Income Before Tax", :], trailing
            ),
        )

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
        market_risk_premium: float | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        The Weighted Average Cost of Capital (WACC) is a financial metric used to estimate the
        cost of capital for a company.

        It represents the average rate of return a company must pay to its investors for using
        their capital. WACC takes into account the cost of both equity and debt, weighted by
        their respective proportions in the company's capital structure.

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
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the values. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int | None, optional): The trailing period to use for the calculation. The window counts
                calendar periods, so a period without a report makes it NaN. Defaults to None.
            market_risk_premium (float | None, optional): The annual equity risk premium the Cost of Equity
                uses. Defaults to None, which takes the equity risk premium of each company's country
                (Damodaran's Total Equity Risk Premium, see get_market_risk_premium) and, where that is
                not available, the benchmark's average return above the risk-free rate over the data.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: DataFrame containing the WACC values.

        Notes:
        - The Cost of Equity follows from the Capital Asset Pricing Model (CAPM): the risk-free rate of
        the period plus Beta times the market risk premium. That premium defaults to the equity risk
        premium of the company's country (Damodaran's Total Equity Risk Premium, about 4.5% for the
        United States), today's premium for every period, as the source keeps no history. Without it (no API key, or a country the
        table does not cover) it is the benchmark's average return above the risk-free rate over the
        data, which a short or strong sample makes large: about 15% a year for the S&P 500 over
        2022-2026. Set `market_risk_premium` to use one premium for every company. Beta is estimated
        from the period's daily returns. With quarterly data the rates are per quarter, the risk-free yield divided by four.
        - With `trailing` set, the Interest Expense is summed over the trailing periods, so the
          Cost of Debt covers that span, the risk-free rate and premium of the Cost of Equity
          are the per-period rates times `trailing` and the Beta is estimated from the daily
          returns of the trailing periods. With quarterly data and four trailing quarters, every
          rate is an annual rate on a year of daily returns.
        - The Market Value of Debt is approximated as the Total Debt.
        - The capital weights are taken at the end of the fiscal period, also with `trailing` set: the
        Market Value of Equity is the closing share price on that day times the period's weighted
        average shares, and the debt is the Total Debt on the balance sheet of that day.
        - The Cost of Debt is the Interest Expense divided by the Total Debt at the end of the period,
        so debt raised or repaid during the period moves it away from the rate on the average debt
        outstanding. A period with debt but an Interest Expense of 0 (financial statements that
        include the interest in another line item or leave it out) has an unknown Cost of Debt, so
        its Cost of Debt and Weighted Average Cost of Capital are NaN rather than a WACC that treats
        the debt as free. Without any debt the WACC is the Cost of Equity.
        - The Corporate Tax Rate is the effective tax rate (Income Tax Expense / Income Before Tax),
        kept within 0 and 1 so that a tax benefit does not turn the tax shield negative, and 0 for a
        period without a pre-tax profit, which leaves no taxable income for the interest to shield.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["VZ", "T", "TMUS"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2022-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_weighted_average_cost_of_capital().xs(
            "Weighted Average Cost of Capital", level=1
        )
        ```

        Which returns:

        |      |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|
        | T    | 0.0453 | 0.0457 | 0.0358 | 0.0411 |
        | TMUS | 0.0469 | 0.0437 | 0.0447 | 0.0425 |
        | VZ   | 0.0304 | 0.0407 | 0.036  | 0.0377 |
        """
        weighted_average_cost_of_capital = self._weighted_average_cost_of_capital(
            diluted=diluted,
            trailing=trailing,
            market_risk_premium=market_risk_premium,
        )
        if growth:
            weighted_average_cost_of_capital = calculate_growth(
                dataset=weighted_average_cost_of_capital,
                lag=lag,
                rounding=rounding if rounding is not None else self._rounding,
                axis="columns",
            )
        else:
            weighted_average_cost_of_capital = apply_rounding(
                weighted_average_cost_of_capital,
                rounding if rounding is not None else self._rounding,
            )

        if standardize:
            weighted_average_cost_of_capital = calculate_standardization(
                dataset=weighted_average_cost_of_capital.loc[
                    :, self._start_date : self._end_date
                ],
                rounding=rounding if rounding is not None else self._rounding,
                axis="columns",
            )

        if not show_full_results:
            weighted_average_cost_of_capital = weighted_average_cost_of_capital.loc[
                :, "Weighted Average Cost of Capital", :
            ]
        elif len(self._tickers) == 1:
            weighted_average_cost_of_capital = (
                weighted_average_cost_of_capital.droplevel(level=0)
            )

        return weighted_average_cost_of_capital.loc[
            :, self._start_date : self._end_date
        ]

    @handle_errors
    def get_implied_cost_of_capital(
        self,
        modified: bool = True,
        min_analysts: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate Easton's (2004) implied cost of equity capital: the return on equity the share
        price implies.

        It is the expected rate of return on equity at which the share price equals the value of
        the next two years of earnings expected by analysts. Where the Cost of Equity of
        `get_weighted_average_cost_of_capital` is the return the Capital Asset Pricing Model
        (CAPM) requires from a beta and a market risk premium, this is the return the market
        price itself implies (ex-ante).

        The formula is as follows:

            - PEG: r = √((EPS₂ − EPS₁) / P₀)
            - MPEG: r = A + √(A² + (EPS₂ − EPS₁) / P₀)
            - A = DPS₁ / (2 × P₀)

        Where EPS₁ and EPS₂ are the analyst consensus (average) EPS estimates of the two fiscal
        years after the period, P₀ the share price at the end of the period and DPS₁ the
        dividends per share of the next year. The MPEG rate is the positive root of Easton's
        valuation equation P₀ = EPS₁ / r + (EPS₂ + r × DPS₁ − (1 + r) × EPS₁) / r², which assumes the
        abnormal growth in earnings stays constant after the second year; the PEG rate leaves
        the dividends out (DPS₁ = 0). The rate is a yearly rate, annually compounded.

        See: Easton, P. D. (2004). PE Ratios, PEG Ratios, and Estimating the Implied Expected
        Rate of Return on Equity Capital. The Accounting Review, 79(1), 73-95.

        The estimates are the analysts' consensus from FinancialModelingPrep, the same estimates
        as `get_forward_price_earnings_ratio` of the Ratios module uses, which requires an
        `api_key` and a Premium FMP subscription. They are fetched once per Toolkit instance.
        These are the current analyst estimates, also for past periods, so the result of every
        period, the example below included, changes as analysts revise their estimates.

        Also known as: implied cost of capital, ICC, PEG / MPEG cost of equity, Easton model.

        Args:
            modified (bool, optional): Whether to return the MPEG rate, which includes the
                dividends and is the version Easton prefers, instead of the PEG rate. Defaults to
                True.
            min_analysts (int | None, optional): The minimum number of analysts behind the EPS
                consensus of an estimate period; a period that relies on one covered by fewer is
                NaN. The number per period is the "Number of Analysts EPS" of
                `Toolkit.get_analyst_estimates`. Defaults to None, which keeps every period.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the values. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The implied cost of equity capital per ticker (rows) and fiscal period
                (columns) as a decimal.

        Notes:
        - EPS₁ and EPS₂ of a fiscal year are the consensus estimates of the next two fiscal years.
        On quarterly data each is a year of estimates, as in the forward P/E: EPS₁ the sum of the
        estimates of the four quarters after the period and EPS₂ that of the four quarters after
        those.
        - P₀ is the traded close (adjusted for splits, not for dividends) on the last trading day up
        to the end of the fiscal period, as the other market values of the Models module.
        - DPS₁ is the dividends per share paid over the year up to the end of the fiscal period.
        FinancialModelingPrep has no dividend forecasts, and Easton allows the current dividend in
        their place; a growth assumption on top would add an input the data does not give. The
        dividends move the rate little: A is half the dividend yield.
        - The rate is NaN where the estimated earnings do not grow (EPS₂ ≤ EPS₁), the price is not
        positive or an estimate is missing: the model has no real, positive rate there, so it is
        never shown as 0.
        - FinancialModelingPrep keeps the latest consensus for every fiscal year, not the consensus as
        it stood at the end of each period, so for past periods the estimates may include later
        revisions, the same as the forward P/E. Only the latest periods are ex-ante estimates;
        earlier ones show the rate the price implied for the earnings now expected for those years.
        - On quarterly data FinancialModelingPrep gives estimates for the coming quarters only, and a
        rate needs eight quarters of estimates after its period, so only the latest periods have one.
        - Without daily data from a year before the start date, the dividends of the first periods
        cover less than a year, which is logged.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["KO", "PEP", "MDLZ"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_implied_cost_of_capital()
        ```

        Which returns:

        |      |   2021 |   2022 |     2023 |   2024 |   2025 |
        |:-----|-------:|-------:|---------:|-------:|-------:|
        | KO   | 0.0736 | 0.0665 |   0.0665 | 0.0927 | 0.0679 |
        | MDLZ | 0.0793 | 0.0653 | nan      | 0.0695 | 0.0944 |
        | PEP  | 0.0803 | 0.071  | nan      | 0.0641 | 0.0589 |
        """
        analyst_estimates = self._analyst_estimates

        if analyst_estimates.empty:
            logger.error(
                "No analyst estimates available. This requires an api_key to be set and a "
                "Premium FMP subscription."
            )
            return pd.DataFrame()

        estimated_eps = analyst_estimates.loc[:, "Estimated EPS Average", :]
        if min_analysts is not None:
            # A period without a reported count is left out too: its depth is unknown.
            analyst_count = (
                analyst_estimates.loc[:, "Number of Analysts EPS", :]
                if "Number of Analysts EPS"
                in analyst_estimates.index.get_level_values(1)
                else pd.DataFrame(
                    float("nan"),
                    index=estimated_eps.index,
                    columns=estimated_eps.columns,
                )
            )
            estimated_eps = estimated_eps.where(
                analyst_count.reindex_like(estimated_eps) >= min_analysts
            )

        # Every period from the first statement period to the last estimate, so that a shift by one column is a shift by one period and a missing estimate leaves a gap.  # noqa: E501
        columns = self._balance_sheet_statement.columns
        estimated_eps.columns = pd.PeriodIndex(estimated_eps.columns).asfreq(
            columns.freq
        )
        periods = pd.period_range(
            min(columns.min(), estimated_eps.columns.min()),
            max(columns.max(), estimated_eps.columns.max()),
            freq=columns.freq,
        )
        estimated_eps = estimated_eps.reindex(
            index=self._tickers, columns=periods
        ).astype(float)

        # A year of estimates: on quarterly data each quarter with the three after it (a rolling sum over the reversed columns).  # noqa: E501
        periods_per_year = 4 if self._quarterly else 1
        if self._quarterly:
            estimated_eps = (
                estimated_eps.iloc[:, ::-1]
                .T.rolling(4, min_periods=4)
                .sum()
                .T.iloc[:, ::-1]
            )

        earnings_per_share_one = estimated_eps.shift(-1, axis=1).reindex(
            columns=columns
        )
        earnings_per_share_two = estimated_eps.shift(
            -1 - periods_per_year, axis=1
        ).reindex(columns=columns)

        # The dividends paid over the year up to each fiscal period end, from the daily data of a year before the start date where available so the first periods have a full year.  # noqa: E501
        daily = self._historical_data.get("lookback")
        if daily is None or daily.empty:
            daily = self._historical_data["daily"]
            logger.warning(
                "No daily data from a year before the start date, so the dividends per share of "
                "the first periods cover less than a year in the implied cost of capital."
            )
        dividends = daily["Dividends"].reindex(columns=self._tickers).fillna(0)
        dividends.index = dividends.index.to_timestamp()
        dividends_per_share = self._values_at_market_dates(
            dividends.rolling("365D").sum()
        )

        implied_cost_of_capital = easton_model.get_implied_cost_of_capital(
            earnings_per_share_one=earnings_per_share_one,
            earnings_per_share_two=earnings_per_share_two,
            share_price=self._period_prices(),
            dividends_per_share=dividends_per_share,
            modified=modified,
        )

        return finalize_dataset(
            dataset=implied_cost_of_capital,
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
    def get_economic_value_added(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
        market_risk_premium: float | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Economic Value Added (EVA) shows the value a company creates above the return its
        capital providers require.

        It captures whether a company is generating returns above its true cost of capital,
        which distinguishes it from purely accounting-based measures such as Net Income.

        The formula is as follows:

            - NOPAT = EBIT * (1 - Corporate Tax Rate)
            - Invested Capital = Total Equity + Total Debt
            - Return on Invested Capital = NOPAT / Invested Capital
            - EVA Spread = Return on Invested Capital - Weighted Average Cost of Capital
            - EVA = NOPAT - (Weighted Average Cost of Capital * Invested Capital) = EVA Spread * Invested Capital

        EVA is mostly used to compare value creation across business units or companies and over time, and
        as a basis for management compensation. It is equivalent to (Return on Invested Capital - WACC) *
        Invested Capital, so a company with a high return on capital but a small capital base can create
        less EVA than a larger, less profitable one.

        The EVA Spread, the return on capital above its cost (Stewart, 1991), compares value creation
        between companies of different sizes.

        For more information, see the following sources:

        - Stewart, G. B. (1991). "The Quest for Value." HarperBusiness.
        - Stern, J. M., Stewart, G. B. and Chew, D. H. (1995). "The EVA Financial Management
          System." Journal of Applied Corporate Finance, 8(2), 32-46.

        Also known as: EVA, economic profit, EVA spread, economic spread, ROIC minus WACC.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to None.
            growth (bool, optional): Whether to calculate the growth of the values. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int | None, optional): The number of periods to sum the flows (EBIT, taxes)
                over, with the Total Equity and Total Debt averaged over the trailing + 1 period ends
                that open and close those periods, so trailing=1 equals the default average of the
                opening and closing balance. E.g. 4 with quarterly data gives the trailing twelve
                months (TTM). The window counts calendar periods, so a period without a report makes
                it NaN. Defaults to None.
            market_risk_premium (float | None, optional): The annual equity risk premium the Cost of Equity
                uses. Defaults to None, which takes the equity risk premium of each company's country
                (Damodaran's Total Equity Risk Premium, see get_market_risk_premium) and, where that is
                not available, the benchmark's average return above the risk-free rate over the data.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: DataFrame containing the EVA and its components: NOPAT, Invested Capital, the
            WACC, the Return on Invested Capital, the EVA Spread and the EVA.

        Notes:
        - A positive EVA indicates that the company is generating returns in excess of its cost of
        capital, i.e. it is creating value for its capital providers. A negative EVA indicates the
        company is destroying value.
        - EBIT is the reported Operating Income, as in the other models of this module and the
        EBIT based ratios of the Ratios module.
        - The Corporate Tax Rate is the effective tax rate (Income Tax Expense / Income Before Tax)
        kept within 0 and 1, and 0 for a period without a pre-tax profit, the same rate the
        Weighted Average Cost of Capital uses (see `get_weighted_average_cost_of_capital`).
        - Invested Capital is the sum of the average Total Equity and the average Total Debt over
        the period ends that open and close the period (the trailing + 1 period ends with
        `trailing` set), i.e. the capital employed over the course of the period rather than its
        closing balance, consistent with the Return on Invested Capital calculation in the Ratios
        module.
        - The Weighted Average Cost of Capital comes from
          `get_weighted_average_cost_of_capital`, with the market risk premium given by
          `market_risk_premium` (by default the equity risk premium of the company's country)
          and the same `trailing`, so that it covers the span of the NOPAT (an annual rate for
          four trailing quarters).

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["V", "MA", "HD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2022-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_economic_value_added().xs("Economic Value Added", level=1)
        ```

        Which returns:

        |    |        2022 |        2023 |        2024 |         2025 |
        |:---|------------:|------------:|------------:|-------------:|
        | HD | 1.51589e+10 | 1.23635e+10 | 1.2118e+10  | nan          |
        | MA | 8.7863e+09  | 9.8349e+09  | 1.15046e+10 |   1.3298e+10 |
        | V  | 1.1435e+10  | 1.30387e+10 | 1.5465e+10  |   1.5322e+10 |
        """
        eva = {}

        income_tax_expense = trailing_sum(
            self._income_statement.loc[:, "Income Tax Expense", :], trailing
        )
        income_before_tax = trailing_sum(
            self._income_statement.loc[:, "Income Before Tax", :], trailing
        )
        # The invested capital of the return on invested capital of the Ratios module: averaged
        # over the period ends that open and close the (trailing) periods the NOPAT covers.
        period_ends = (trailing or 1) + 1
        total_equity = trailing_mean(
            self._balance_sheet_statement.loc[:, "Total Equity", :], period_ends
        )
        total_debt = trailing_mean(
            self._balance_sheet_statement.loc[:, "Total Debt", :], period_ends
        )

        ebit = self._earnings_before_interest_and_taxes(trailing)
        # The tax rate the WACC uses: kept within 0 and 1 and 0 without a pre-tax profit, so that a tax benefit does not lift NOPAT above EBIT and a loss year is not taxed at the ratio of two negatives.  # noqa: E501
        effective_tax_rate = wacc_model.get_corporate_tax_rate(
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
            self._weighted_average_cost_of_capital(
                trailing=trailing, market_risk_premium=market_risk_premium
            ).loc[:, "Weighted Average Cost of Capital", :]
        )

        invested_capital = eva["Invested Capital"]
        eva["Return on Invested Capital"] = eva[
            "Net Operating Profit After Taxes"
        ] / invested_capital.where(invested_capital > 0)
        eva["Economic Value Added Spread"] = eva_model.get_economic_value_added_spread(
            return_on_invested_capital=eva["Return on Invested Capital"],
            weighted_average_cost_of_capital=eva["Weighted Average Cost of Capital"],
        )

        eva["Economic Value Added"] = eva_model.get_economic_value_added(
            net_operating_profit_after_taxes=eva["Net Operating Profit After Taxes"],
            weighted_average_cost_of_capital=eva["Weighted Average Cost of Capital"],
            invested_capital=eva["Invested Capital"],
        )

        return self._finalize_components(eva, rounding, growth, lag, standardize)

    @handle_errors
    def get_market_value_added(
        self,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Market Value Added (MVA) shows the difference between a company's market value and the
        capital invested in it.

        The market value is that of both its equity and its debt, and the capital is all that
        has historically been invested. It is the market-priced counterpart to Economic Value
        Added (EVA): where EVA measures a single period's excess return over the cost of
        capital, MVA reflects the market's cumulative, forward-looking verdict on all of a
        company's expected future EVA.

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
            trailing (int | None, optional): The number of periods the Invested Capital is averaged
                over. The window counts calendar periods, so a period without a report makes it
                NaN. Defaults to None, which uses the closing balances.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: DataFrame containing the MVA and its components.

        Notes:
        - A positive MVA indicates that the market believes management has created value in
        excess of the capital invested in the company. A negative MVA indicates the market
        values the company below the capital that has historically been invested in it.
        - The Market Value of Debt is approximated as the closing book value of Total Debt,
        the same simplification used in the Weighted Average Cost of Capital calculation
        elsewhere in this module.
        - The Market Value of Equity is the closing share price at the end of the fiscal period
        times the period's weighted average shares, and the Invested Capital is the Total
        Equity plus the Total Debt on the balance sheet of that day, so both sides of the MVA
        have the same date. With the debt at its book value on both sides, it cancels out:
        the MVA is the market value of equity minus the book Total Equity.
        - With `trailing` set, the Invested Capital is the average of the closing balances of
        the trailing periods, while the market values stay those at the end of the period.

        References:
        - Stern, Joel M., G. Bennett Stewart, and Donald H. Chew. "The EVA Financial
        Management System." Journal of Applied Corporate Finance, Vol. 8, No. 2, 1995, pp. 32-46.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["LLY", "JNJ", "MRK"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_market_value_added().xs("Market Value Added", level=1)
        ```

        Which returns:

        |     |        2021 |        2022 |        2023 |        2024 |        2025 |
        |:----|------------:|------------:|------------:|------------:|------------:|
        | JNJ | 3.83418e+11 | 3.93774e+11 | 3.32543e+11 | 2.80894e+11 | 4.22872e+11 |
        | LLY | 2.4267e+11  | 3.2017e+11  | 5.15679e+11 | 6.83662e+11 | 9.3853e+11  |
        | MRK | 1.56255e+11 | 2.35977e+11 | 2.40039e+11 | 2.06407e+11 | 2.11225e+11 |
        """
        mva = {}

        share_prices = self._period_prices()

        average_shares = self._income_statement.loc[
            :,
            "Weighted Average Shares Diluted" if diluted else "Weighted Average Shares",
            :,
        ]

        # The invested capital on the balance sheet of the day the market value is taken on (an
        # average over the trailing periods only with `trailing` set), so both sides of the MVA
        # have the same date.
        total_equity = trailing_mean(
            self._balance_sheet_statement.loc[:, "Total Equity", :], trailing
        )
        total_debt = trailing_mean(
            self._balance_sheet_statement.loc[:, "Total Debt", :], trailing
        )

        market_cap = valuation_model.get_market_cap(
            share_price=share_prices,
            total_shares_outstanding=average_shares,
        )

        # A market value is a point-in-time quantity, so the closing balance of Total Debt is its proxy. With the invested capital at the same date, the debt cancels out and the MVA is the market value of equity minus the book Total Equity.  # noqa: E501
        market_value_of_debt = self._balance_sheet_statement.loc[:, "Total Debt", :]

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

        return self._finalize_components(mva, rounding, growth, lag, standardize)

    @handle_errors
    def get_return_on_incremental_invested_capital(
        self,
        years: int = 1,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Return on Incremental Invested Capital (ROIIC): the growth in after-tax
        operating profit over the capital invested to produce it.

        Where the Return on Invested Capital measures the return on all the capital in place,
        ROIIC measures the return on the newest investments, which shows whether growth creates
        value: growth adds value only while the ROIIC exceeds the cost of capital.

        The formula is as follows, with n the number of years:

            - NOPAT = EBIT * (1 - Tax Rate)
            - Invested Capital = Total Equity + Total Debt
            - ROIIC (t) = (NOPAT (t) - NOPAT (t - n)) / (Invested Capital (t - 1) - Invested Capital (t - 1 - n))

        Following Mauboussin and Callahan, the invested capital is lagged by a year: the capital
        invested in a year earns its profit in the years after. A longer span (`years`, often 3 or
        5) smooths the lumpy year-to-year changes.

        For more information, see the following sources:

        - Mauboussin, M. J. and Callahan, D. (2014). "Calculating Return on Invested Capital."
          Credit Suisse Global Financial Strategies.
        - Koller, T., Goedhart, M. and Wessels, D. (2020). "Valuation: Measuring and Managing the
          Value of Companies." Wiley, 7th edition.

        Also known as: ROIIC, incremental ROIC, return on new invested capital, RONIC.

        Args:
            years (int, optional): The number of years n the changes span, at least 1. Defaults to 1.
            rounding (int, optional): The number of decimals to round the results to. Defaults to None.
            growth (bool, optional): Whether to calculate the growth of the values. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int | None, optional): The number of periods to sum the NOPAT over. The window
                counts calendar periods, so a period without a report makes it NaN. Defaults to None,
                which is 4 on quarterly data (a trailing twelve months, with every lag counted in
                years of four quarters) and single years on yearly data.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per ticker the Net Operating Profit After Taxes, the Change in NOPAT, the
            Invested Capital (a year earlier), the Change in Invested Capital and the Return on
            Incremental Invested Capital (a decimal).

        Notes:
        - NOPAT and the Invested Capital are those of `get_economic_value_added` (the effective tax
        rate kept within 0 and 1, EBIT the reported Operating Income), with the Invested Capital at
        the period end rather than averaged, as its change is what was invested.
        - The ROIIC is NaN where the invested capital did not grow: a ratio on shrinking capital
        has no meaning as a return (a fall in NOPAT on less capital would show as a positive return).

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["COST", "WMT", "TGT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_return_on_incremental_invested_capital().xs(
            "Return on Incremental Invested Capital", level=1
        )
        ```

        Which returns:

        |      |   2021 |   2022 |     2023 |    2024 |   2025 |
        |:-----|-------:|-------:|---------:|--------:|-------:|
        | COST |    nan |    nan |   0.0657 |  0.2493 |    nan |
        | TGT  |    nan |    nan |   1.3141 | -0.0469 |    nan |
        | WMT  |    nan |    nan | nan      |  0.264  |    nan |
        """
        validate_window("years", years, 1)
        trailing, periods_per_year = self._annual_trailing(trailing)
        span = years * periods_per_year

        nopat = self._net_operating_profit_after_taxes(trailing).reindex(self._tickers)
        invested_capital = (
            self._rows(self._balance_sheet_statement, "Total Equity")
            + self._rows(self._balance_sheet_statement, "Total Debt")
        ).shift(periods_per_year, axis=1)

        components = {}
        components["Net Operating Profit After Taxes"] = nopat
        components["Change in NOPAT"] = nopat - nopat.shift(span, axis=1)
        components["Invested Capital"] = invested_capital
        components["Change in Invested Capital"] = invested_capital - (
            invested_capital.shift(span, axis=1)
        )
        components["Return on Incremental Invested Capital"] = (
            eva_model.get_return_on_incremental_invested_capital(
                change_in_net_operating_profit_after_taxes=components[
                    "Change in NOPAT"
                ],
                change_in_invested_capital=components["Change in Invested Capital"],
            )
        )

        return self._finalize_components(components, rounding, growth, lag, standardize)

    @handle_errors
    def get_owner_earnings(
        self,
        method: Literal["total", "greenwald"] = "total",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate Warren Buffett's owner earnings: the cash a business could pay its owners
        while keeping its competitive position and unit volume.

        Buffett defines them in the 1986 Berkshire Hathaway letter as the reported earnings plus
        depreciation, depletion and amortization and other non-cash charges, less the average
        yearly capital expenditure the business needs to keep its position (and any additional
        working capital that requires).

        The formula is as follows:

            - Owner Earnings = Net Income + Depreciation and Amortization - Maintenance Capital
              Expenditure - Change in Working Capital
            - Owner Earnings per Share = Owner Earnings / Weighted Average Shares Diluted

        The statements do not split the capital expenditure into maintenance and growth, which
        Buffett himself calls a guess. There are two estimates of the maintenance part:

            - Total: all the capital expenditure, the conservative estimate (owner earnings then
              differ from the free cash flow only by the non-cash items other than depreciation).
            - Greenwald: the capital expenditure minus the growth capital expenditure, the
              investment the growth in revenue required at the company's ratio of property, plant
              and equipment to revenue a year earlier (Greenwald et al., 2001):
              Maintenance Capital Expenditure = Capital Expenditure - PP&E / Revenue (t - 1) *
              max(Change in Revenue, 0), at least 0.

        For more information, see the following sources:

        - Buffett, W. E. (1986). "Chairman's Letter." Berkshire Hathaway Inc. Annual Report 1986,
          appendix on purchase-price accounting adjustments and owner earnings.
        - Greenwald, B. C. N., Kahn, J., Sonkin, P. D. and van Biema, M. (2001). "Value Investing:
          From Graham to Buffett and Beyond." Wiley, chapter 6.

        Also known as: owner earnings, Buffett owner earnings, maintenance free cash flow.

        Args:
            method (str, optional): The estimate of the maintenance capital expenditure. Options are: "total" (all
                the capital expenditure), "greenwald" (the capital expenditure minus the growth capital expenditure,
                Greenwald et al., 2001). Defaults to "total".
            rounding (int, optional): The number of decimals to round the results to. Defaults to None.
            growth (bool, optional): Whether to calculate the growth of the values. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int | None, optional): The number of periods to sum the flows over. The window
                counts calendar periods, so a period without a report makes it NaN. Defaults to None,
                which uses single periods (per quarter on quarterly data).
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per ticker the Net Income, Depreciation and Amortization, Capital
            Expenditure, Maintenance Capital Expenditure, Change in Working Capital, Owner Earnings
            and Owner Earnings per Share.

        Notes:
        - The Change in Working Capital is that of the cash flow statement with its sign turned, so
        an increase in working capital (a use of cash) is positive and lowers the owner earnings.
        - Depreciation and Amortization is that of the cash flow statement; other non-cash charges
        (such as share-based compensation) are not added back, as Buffett adds back only the
        charges that do not reflect an economic cost.
        - With Greenwald's estimate on quarterly data, the change in revenue is measured against
          the same period a year earlier and divided over the four quarters of that year when
          `trailing` is not set.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["KO", "PEP", "MDLZ"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_owner_earnings().xs("Owner Earnings", level=1)
        ```

        Which returns:

        |      |       2021 |      2022 |      2023 |      2024 |      2025 |
        |:-----|-----------:|----------:|----------:|----------:|----------:|
        | KO   | 1.1181e+10 | 8.713e+09 | 9.144e+09 | 3.408e+09 | 4.837e+09 |
        | MDLZ | 3.799e+09  | 2.405e+09 | 4.553e+09 | 3.557e+09 | 1.556e+09 |
        | PEP  | 6.926e+09  | 6.095e+09 | 6.737e+09 | 6.379e+09 | 6.358e+09 |
        """
        method = method.lower()
        check_option("method", method, ("total", "greenwald"))
        periods_per_year = 4 if self._quarterly else 1

        def flow(statement: pd.DataFrame, row: str) -> pd.DataFrame:
            return trailing_sum(self._rows(statement, row), trailing)

        capital_expenditure = -flow(self._cash_flow_statement, "Capital Expenditure")

        if method == "total":
            maintenance_capital_expenditure = capital_expenditure
        else:
            revenue = self._rows(self._income_statement, "Revenue")
            # The change in revenue over the window the capital expenditure covers: compared with a year earlier, a year's change divided over its quarters for a single quarter.  # noqa: E501
            window = trailing or 1
            yearly_revenue = trailing_sum(revenue, periods_per_year)
            change_in_revenue = (
                (yearly_revenue - yearly_revenue.shift(periods_per_year, axis=1))
                * window
                / periods_per_year
            )
            ppe_to_revenue = (
                self._rows(
                    self._balance_sheet_statement, "Property, Plant and Equipment"
                )
                / yearly_revenue.where(yearly_revenue > 0)
            ).shift(periods_per_year, axis=1)
            maintenance_capital_expenditure = (
                intrinsic_model.get_greenwald_maintenance_capital_expenditure(
                    capital_expenditure=capital_expenditure,
                    property_plant_and_equipment_to_revenue=ppe_to_revenue,
                    change_in_revenue=change_in_revenue,
                )
            )

        components = {}
        components["Net Income"] = flow(self._income_statement, "Net Income")
        components["Depreciation and Amortization"] = flow(
            self._cash_flow_statement, "Depreciation and Amortization"
        )
        components["Capital Expenditure"] = capital_expenditure
        components["Maintenance Capital Expenditure"] = maintenance_capital_expenditure
        components["Change in Working Capital"] = -flow(
            self._cash_flow_statement, "Change in Working Capital"
        )
        components["Owner Earnings"] = intrinsic_model.get_owner_earnings(
            net_income=components["Net Income"],
            depreciation_and_amortization=components["Depreciation and Amortization"],
            maintenance_capital_expenditure=maintenance_capital_expenditure,
            change_in_working_capital=components["Change in Working Capital"],
        )
        components["Owner Earnings per Share"] = components["Owner Earnings"] / (
            self._rows(self._income_statement, "Weighted Average Shares Diluted")
        )

        return self._finalize_components(components, rounding, growth, lag, standardize)

    @handle_errors
    def get_unlevered_beta(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the unlevered (asset) beta of Hamada (1972): the beta each company's shares
        would have without debt, its business risk apart from its financial leverage.

        Unlevered betas can be compared across companies with different amounts of debt,
        averaged over an industry and relevered at a company's own leverage (Damodaran's
        bottom-up beta, see `wacc_model.get_relevered_beta`).

        The formula is as follows:

            - Levered Beta = the regression beta of the shares against the benchmark over the period
            - Debt to Equity = Total Debt / Market Value of Equity
            - Unlevered Beta = Levered Beta / (1 + (1 - Tax Rate) * Debt to Equity)

        For more information, see the following sources:

        - Hamada, R. S. (1972). "The Effect of the Firm's Capital Structure on the Systematic Risk
          of Common Stocks." The Journal of Finance, 27(2), 435-452.
        - Damodaran, A. (2012). "Investment Valuation." Wiley, 3rd edition, chapter 8.

        Also known as: asset beta, unlevered beta, Hamada equation, business risk beta.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to None.
            growth (bool, optional): Whether to calculate the growth of the values. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int | None, optional): The number of periods the beta is measured over (and the
                taxes summed over). The window counts calendar periods, so a period without a report
                makes it NaN. Defaults to None, which is a single period.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per ticker the Levered Beta, Debt to Equity, Tax Rate and Unlevered Beta.

        Notes:
        - The Levered Beta, the market value of equity and the tax rate are those of
        `get_weighted_average_cost_of_capital`: the beta of the daily returns within the period
        against the benchmark, the closing price at the fiscal period end times the diluted
        weighted average shares, and the effective tax rate kept within 0 and 1 (0 without a
        pre-tax profit). The debt is the book value of Total Debt, the usual stand-in for its
        market value.
        - The debt is assumed to carry no market risk (a debt beta of zero), as Hamada and
        Damodaran's bottom-up betas do. Damodaran also corrects the unlevered beta for the cash a
        company holds (Unlevered Beta / (1 - Cash / Firm Value)), which this leaves out.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["F", "GM", "TSLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_unlevered_beta().xs("Unlevered Beta", level=1)
        ```

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | F    | 0.4549 | 0.3568 | 0.3472 | 0.2686 | 0.1996 |
        | GM   | 0.6333 | 0.4549 | 0.4088 | 0.3555 | 0.3454 |
        | TSLA | 1.9416 | 1.7191 | 2.223  | 2.3614 | 2.2258 |
        """
        years = self._cash_flow_statement.columns
        beta = self._beta_per_period(str(years[0]), str(years[-1]), trailing).T
        wacc = self._weighted_average_cost_of_capital(trailing=trailing)

        market_value_equity = wacc.xs("Market Value Equity", level=1).reindex(
            self._tickers
        )
        debt_to_equity = self._rows(
            self._balance_sheet_statement, "Total Debt"
        ) / market_value_equity.where(market_value_equity > 0)
        tax_rate = wacc.xs("Corporate Tax Rate", level=1).reindex(self._tickers)

        components = {}
        components["Levered Beta"] = beta.reindex(
            index=self._tickers, columns=debt_to_equity.columns
        )
        components["Debt to Equity"] = debt_to_equity
        components["Tax Rate"] = tax_rate
        components["Unlevered Beta"] = wacc_model.get_unlevered_beta(
            levered_beta=components["Levered Beta"],
            debt_to_equity=debt_to_equity,
            tax_rate=tax_rate,
        )

        return self._finalize_components(components, rounding, growth, lag, standardize)

    @handle_errors
    def get_capitalized_research_and_development(
        self,
        amortization_years: int = 5,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Capitalize the research and development expenses as Aswath Damodaran does: accounting
        treats R&D as an expense, while it is an investment whose benefit lasts several years.

        Every year's R&D becomes an asset amortized straight-line over its amortizable life,
        which changes the operating income, the net income, the book equity, the invested
        capital and with them the return on capital of research-intensive companies.

        The formula is as follows, with n the amortizable life in years:

            - Research Asset (t) = Σ (k = 0 .. n - 1) R&D (t - k) * (n - k) / n
            - Amortization (t) = Σ (k = 1 .. n) R&D (t - k) / n
            - Adjusted Operating Income = Operating Income + R&D - Amortization
            - Adjusted Net Income = Net Income + R&D - Amortization
            - Adjusted Total Equity = Total Equity + Research Asset
            - Adjusted Invested Capital = Total Equity + Total Debt + Research Asset
            - Adjusted Return on Invested Capital = Adjusted Operating Income * (1 - Tax Rate) /
              Adjusted Invested Capital at the beginning of the period

        The expense of the current year is not amortized yet, and that of n years ago is written
        off in full this year, as in Damodaran's R&D converter.

        For more information, see the following sources:

        - Damodaran, A. (2012). "Investment Valuation." Wiley, 3rd edition, chapter 9.
        - Damodaran, A. "R&D Converter" (R&D.xls),
          https://pages.stern.nyu.edu/~adamodar/New_Home_Page/spreadsh.htm
        - Lev, B. and Sougiannis, T. (1996). "The Capitalization, Amortization, and Value-Relevance
          of R&D." Journal of Accounting and Economics, 21(1), 107-138.

        Also known as: R&D capitalization, research asset, capitalized R&D, adjusted operating
        income for R&D.

        Args:
            amortization_years (int, optional): The amortizable life n of the research asset in
                years, at least 1. Damodaran uses 2 years for software, 3 to 5 for most industries and up
                to 10 for pharmaceuticals (the life it takes to turn research into products). Defaults to 5.
            rounding (int, optional): The number of decimals to round the results to. Defaults to None.
            growth (bool, optional): Whether to calculate the growth of the values. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per ticker the Research and Development Expenses, Research Asset,
            Amortization of Research Asset, Adjusted Operating Income, Adjusted Net Income,
            Adjusted Total Equity, Adjusted Invested Capital and Adjusted Return on Invested Capital.

        Notes:
        - The asset needs the expenses of the n years before: with a start date less than n years
        back the results are NaN (set an earlier start_date). A year without a reported R&D expense
        leaves the years that include it NaN rather than counting it as zero.
        - On quarterly data every quarter holds the expenses of the trailing year, the asset and
        amortization are those of the years ending with the quarter, and the adjusted incomes are
        trailing twelve months figures.
        - The tax rate is the effective tax rate kept within 0 and 1 (0 without a pre-tax profit),
        as in `get_economic_value_added`. Damodaran notes that the tax benefit of R&D stays that of
        the expense, which these adjustments keep by not changing the taxes.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["MRK", "PFE", "LLY"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2015-01-01",
            end_date="2025-12-31",
        )

        # The last four years, the first with ten years of spending behind them
        toolkit.models.get_capitalized_research_and_development(
            amortization_years=10
        ).loc["MRK"].iloc[:, -4:]
        ```

        Which returns:

        |                                     |         2022 |          2023 |        2024 |        2025 |
        |:------------------------------------|-------------:|--------------:|------------:|------------:|
        | Research and Development Expenses   |   1.3548e+10 |   3.0531e+10  | 1.7938e+10  | 1.2514e+10  |
        | Research Asset                      | nan          |   8.18812e+10 | 8.74511e+10 | 8.65212e+10 |
        | Amortization of Research Asset      | nan          |   1.00653e+10 | 1.23681e+10 | 1.34439e+10 |
        | Adjusted Operating Income           | nan          |   2.34197e+10 | 2.57909e+10 | 2.25561e+10 |
        | Adjusted Net Income                 | nan          |   2.08307e+10 | 2.26869e+10 | 1.73241e+10 |
        | Adjusted Total Equity               | nan          |   1.19516e+11 | 1.33823e+11 | 1.39183e+11 |
        | Adjusted Invested Capital           | nan          |   1.55784e+11 | 1.72093e+11 | 1.89717e+11 |
        | Adjusted Return on Invested Capital | nan          | nan           | 0.1423      | 0.1136      |
        """
        validate_window("amortization_years", amortization_years, 1)
        periods_per_year = 4 if self._quarterly else 1
        trailing = periods_per_year if self._quarterly else None

        def flow(statement: pd.DataFrame, row: str) -> pd.DataFrame:
            return trailing_sum(self._rows(statement, row), trailing)

        research_and_development = flow(
            self._income_statement, "Research and Development Expenses"
        )

        # Companies that report no research and development have nothing to capitalize,
        # which would otherwise only show as empty rows.
        without_research = research_and_development.index[
            research_and_development.isna().all(axis=1)
        ]
        if len(without_research):
            logger.warning(
                "No Research and Development Expenses are reported for %s, so there is no "
                "research asset to capitalize and their adjusted figures are not calculated.",
                ", ".join(map(str, without_research)),
            )
        research_asset, amortization = eva_model.get_research_and_development_asset(
            research_and_development=research_and_development,
            amortization_years=amortization_years,
            periods_per_year=periods_per_year,
        )
        adjustment = research_and_development - amortization
        balance = self._balance_sheet_statement

        components = {}
        components["Research and Development Expenses"] = research_and_development
        components["Research Asset"] = research_asset
        components["Amortization of Research Asset"] = amortization
        components["Adjusted Operating Income"] = (
            flow(self._income_statement, "Operating Income") + adjustment
        )
        components["Adjusted Net Income"] = (
            flow(self._income_statement, "Net Income") + adjustment
        )
        components["Adjusted Total Equity"] = (
            self._rows(balance, "Total Equity") + research_asset
        )
        components["Adjusted Invested Capital"] = (
            self._rows(balance, "Total Equity")
            + self._rows(balance, "Total Debt")
            + research_asset
        )
        tax_rate = wacc_model.get_corporate_tax_rate(
            income_tax_expense=flow(self._income_statement, "Income Tax Expense"),
            income_before_tax=flow(self._income_statement, "Income Before Tax"),
        )
        beginning_capital = components["Adjusted Invested Capital"].shift(
            periods_per_year, axis=1
        )
        components["Adjusted Return on Invested Capital"] = (
            components["Adjusted Operating Income"]
            * (1 - tax_rate)
            / beginning_capital.where(beginning_capital > 0)
        )

        return self._finalize_components(components, rounding, growth, lag, standardize)

    @handle_errors
    def get_cash_flow_return_on_investment(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Cash Flow Return on Investment (CFROI) of Bartley Madden (the HOLT model):
        the internal rate of return of a company's assets seen as one project.

        The gross investment buys a gross cash flow for every year of the life of the
        depreciating assets, and the non-depreciating assets (working capital) come back at the
        end. Where the return on invested capital divides an accounting profit by depreciated
        book values, the CFROI compares cash flows with the original cost of the assets, which
        makes companies with old and new assets comparable.

        The formula is as follows, solved for the CFROI:

            - Gross Cash Flow = Net Income + Depreciation and Amortization + Interest Expense
            - Depreciating Assets = Gross Property, Plant and Equipment (Net Property, Plant and
              Equipment where the gross amount is not reported)
            - Non-Depreciating Assets = Total Current Assets - (Total Current Liabilities - Short
              Term Debt)
            - Gross Investment = Depreciating Assets + Non-Depreciating Assets
            - Asset Life = Depreciating Assets / Depreciation and Amortization
            - Gross Investment = Σ (t = 1 .. Asset Life) Gross Cash Flow / (1 + CFROI)^t +
              Non-Depreciating Assets / (1 + CFROI)^Asset Life

        This is the simplified, nominal form of Madden's model that the statements allow. HOLT's
        CFROI also restates the gross investment and the cash flows for inflation (a real
        return), capitalizes operating leases and R&D and adds the rental expense to the cash flow;
        practitioner versions differ in these choices, so the levels are best compared between
        companies and over time rather than with a published CFROI.

        For more information, see the following sources:

        - Madden, B. J. (1999). "CFROI Valuation: A Total System Approach to Valuing the Firm."
          Butterworth-Heinemann.
        - Madden, B. J. (2010). "Wealth Creation: A Systems Mindset for Building and Investing in
          Businesses for the Long Term." Wiley.

        Also known as: CFROI, HOLT CFROI, cash flow return on investment, Madden CFROI.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to None.
            growth (bool, optional): Whether to calculate the growth of the values. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per ticker the Gross Cash Flow, Depreciating Assets, Non-Depreciating
            Assets, Gross Investment, Asset Life (years) and the Cash Flow Return on Investment (a
            decimal per year, annually compounded).

        Notes:
        - The balance sheet items are averaged over the period ends that open and close the year,
        as the invested capital of `get_economic_value_added`. On quarterly data the flows are
        those of the trailing twelve months, so the CFROI stays a yearly rate.
        - FinancialModelingPrep reports the property, plant and equipment net of depreciation. The
        net amount understates the original cost and its life (Net PP&E / Depreciation) is the
        remaining life of the assets, which gives a remaining-life CFROI. Where the statements
        report the gross amount (Yahoo Finance), the gross amount and the full life are used.
        - The CFROI is NaN where the life or the gross investment is not positive or no rate
        between -99% and 1,000% solves the equation (such as a gross cash flow that never
        returns the investment).

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["HD", "LOW"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_cash_flow_return_on_investment().loc["HD"]
        ```

        Which returns:

        |                                |         2021 |         2022 |        2023 |       2024 |
        |:-------------------------------|-------------:|-------------:|------------:|-----------:|
        | Gross Cash Flow                |  2.0642e+10  |  2.1697e+10  |  2.0333e+10 | 2.0888e+10 |
        | Depreciating Assets            |  3.0917e+10  |  3.18695e+10 |  3.3305e+10 | 3.4666e+10 |
        | Non-Depreciating Assets        |  5.2855e+09  |  7.218e+09   |  9.86e+09   | 8.524e+09  |
        | Gross Investment               |  3.62025e+10 |  3.90875e+10 |  4.3165e+10 | 4.319e+10  |
        | Asset Life                     | 10.8026      | 10.7124      | 10.2572     | 9.2172     |
        | Cash Flow Return on Investment |  0.5664      |  0.551       |  0.4637     | 0.4726     |
        """
        periods_per_year = 4 if self._quarterly else 1
        trailing = periods_per_year if self._quarterly else None
        balance = self._balance_sheet_statement

        def flow(statement: pd.DataFrame, row: str) -> pd.DataFrame:
            return trailing_sum(self._rows(statement, row), trailing)

        def average(row: str) -> pd.DataFrame:
            # The opening and closing balance of the year.
            values = self._rows(balance, row)
            return (values + values.shift(periods_per_year, axis=1)) / 2

        depreciation = flow(self._income_statement, "Depreciation and Amortization")
        gross_property = average("Gross Property, Plant and Equipment")
        depreciating_assets = gross_property.combine_first(
            average("Property, Plant and Equipment")
        )
        non_depreciating_assets = average("Total Current Assets") - (
            average("Total Current Liabilities") - average("Short Term Debt").fillna(0)
        )

        components = {}
        components["Gross Cash Flow"] = (
            flow(self._income_statement, "Net Income")
            + depreciation
            + flow(self._income_statement, "Interest Expense").abs().fillna(0)
        )
        components["Depreciating Assets"] = depreciating_assets
        components["Non-Depreciating Assets"] = non_depreciating_assets
        components["Gross Investment"] = depreciating_assets + non_depreciating_assets
        components["Asset Life"] = depreciating_assets / depreciation.where(
            depreciation > 0
        )
        components["Cash Flow Return on Investment"] = (
            cfroi_model.get_cash_flow_return_on_investment(
                gross_investment=components["Gross Investment"],
                gross_cash_flow=components["Gross Cash Flow"],
                non_depreciating_assets=non_depreciating_assets,
                asset_life=components["Asset Life"],
            )
        )

        return self._finalize_components(components, rounding, growth, lag, standardize)

    @handle_errors
    def get_intrinsic_valuation(
        self,
        growth_rate: float | list | dict[str, float] | None = None,
        perpetual_growth_rate: float | list | dict[str, float] | None = None,
        weighted_average_cost_of_capital: float | list | dict[str, float] | None = None,
        periods: int = 5,
        cash_flow_type: str = "Free Cash Flow to Firm",
        trailing: int | None = None,
        historical: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Intrinsic value shows what a share is worth by its discounted future cash flows,
        independent of its market price.

        It is a fundamental concept in finance and investing that represents the true worth of
        an asset, security, or investment, independent of prevailing market sentiment. It is a
        concept often associated with the value investing philosophy, made famous by legendary
        investors like Benjamin Graham and Warren Buffett. Understanding intrinsic value is
        crucial for investors looking to make informed decisions about where to allocate their
        capital.

        This functionality uses DCF, or Discounted Cash Flow which is a widely used financial
        valuation method that allows investors and analysts to estimate the intrinsic value of
        an investment or business based on its expected future cash flows. It is a fundamental
        tool in finance and investment analysis, providing a systematic way to assess the
        present value of future cash flows while considering the time value of money.

        The formula is as follows:

            - Cash Flow Projection_t = Cash Flow_t-1 * (1 + Growth Rate)
            - Terminal Value = Last Cash Flow Projection * (1 + Perpetual Growth Rate) /
            (Weighted Average Cost of Capital — Perpetual Growth Rate)
            - Present Value of Cash Flows = Sum of Cash Flow Projection_t / (1 + Weighted Average
            Cost of Capital)^t for t = 1 to Periods
            - Present Value of Terminal Value = Terminal Value / (1 + Weighted Average Cost of
            Capital)^Periods
            - Enterprise Value = Present Value of Cash Flows + Present Value of Terminal Value
            - Equity Value = Enterprise Value + Cash and Cash Equivalents + Short Term Investments —
            Total Debt — Minority Interest — Preferred Stock
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
            growth_rate (float, list or dict, optional): The growth rate to use for the cash flow projections. Can be
            one number to use for all tickers, or a list or dict that contains a growth rate for each ticker.
            Defaults to None, which uses the growth the company's reinvestment supports (Damodaran): the
            Reinvestment Rate times the Return on Invested Capital, its reinvestment (NOPAT minus the Free Cash
            Flow to Firm) per year as a share of its invested capital (equity plus debt) over the data up to
            the period valued: the latest period, or each period with `historical=True`.
            perpetual_growth_rate (float, list or dict, optional): The perpetual growth rate to use for the
            terminal value. Can be one number to use for all tickers, or a list or dict that contains a perpetual
            growth rate for each ticker. Defaults to None, which uses the company's growth (as for growth_rate)
            kept within 0 and the long-run growth rate of 2.5%, and 2.5% where it has none.
            weighted_average_cost_of_capital (float, list or dict, optional): The weighted average cost of capital
            used to discount the projected cash flows and to calculate the terminal value. Can be one number to
            use for all tickers, or a list or dict that contains a weighted average cost of capital for each
            ticker. Defaults to None, which uses the Weighted Average Cost of Capital of each company of the
            period valued, calculated over a year (the last 4 quarters with quarterly data, with a beta of
            their daily returns).
            periods (int, optional): The number of periods to use for the cash flow projections, at least 1.
                Defaults to 5.
            cash_flow_type (str, optional): The type of cash flow to use for the cash flow projections. Both
                reported cash flows are after interest, so discounting them at the WACC and subtracting the debt
                counts the cost of debt twice. Options are: "Free Cash Flow to Firm", "Free Cash Flow" (as
                reported on the cash flow statement), "Operating Cash Flow" (as reported on the cash flow
                statement). Defaults to "Free Cash Flow to Firm".
            trailing (int | None, optional): The number of trailing periods to sum for the base cash flow.
                The window counts calendar periods, so a period without a report makes it NaN.
            When set, uses the sum of the latest N consecutive calendar periods that are all reported instead
            of only the most recent period. Defaults to None, which uses 4 (the trailing twelve months) with
            quarterly data and the most recent year with yearly data.
            historical (bool, optional): Whether to value every period instead of only the most recent one,
            which gives the intrinsic value per share over time. Each period is valued from the base cash
            flow ending in it and the cash, debt and shares reported at its end. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: DataFrame containing per ticker the Terminal Value, the Cash Flow
            Projection (the last projected cash flow), the Present Value of Cash Flows, the
            Present Value of Terminal Value, the Enterprise Value, the Equity Value and the
            Intrinsic Value, or for the historical valuation the intrinsic value per share for
            each ticker (rows) and period (columns).

        Notes:
        - The results are highly dependent on the input. Therefore, think carefully about each input parameter to
        ensure the results are accurate (given your beliefs)
        - The growth rate, perpetual growth rate and WACC are annual rates, so the base cash flow is an annual
        one: with quarterly data the four quarters of the latest year with all four reported are summed. A
        company reporting half-yearly has no such year and is not valued with quarterly data (use yearly
        data). Setting `trailing` with yearly data sums several years into one base cash flow, which logs a
        warning.
        - The WACC must be greater than the perpetual growth rate; otherwise the terminal value is infinite
        and the valuation is not calculated. Within 2 percentage points of it the valuation is calculated
        with a warning, as a tenth of a percentage point on either rate then moves it by about 5%.
        - The cash flows are discounted at the end of each period and the Terminal Value at the end
        of the last projected period, so the Present Value of Cash Flows and the Present Value of
        Terminal Value add up to the Enterprise Value and show how much of it the terminal value
        carries.
        - The Total Debt, Cash and Cash Equivalents and Short Term Investments are those of the most
        recent period. Short Term Investments are added back with the cash, the same cash and marketable
        securities `get_enterprise_value_breakdown` deducts, and count as 0 when the balance sheet has no
        such row.
        - The Equity Value is divided by the Weighted Average Shares Diluted of the most recent period.
        - In the historical valuation no period uses figures reported after it, so the series
          shows what this DCF gave at each period end, with the growth and WACC known at that
          period end when they are not given. A period whose cash flow, cash, debt or shares are
          missing is NaN, while the single valuation takes the most recent value of each.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["ADBE", "CRM", "INTU"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_intrinsic_valuation().xs("Intrinsic Value", level=1)
        ```

        Which returns:

        |      |   Periods = 5 |
        |:-----|--------------:|
        | ADBE |      213.31   |
        | CRM  |       67.9174 |
        | INTU |      172.144  |

        And the same valuation for every period:

        ```python
        toolkit.models.get_intrinsic_valuation(historical=True)
        ```

        Which returns:

        |      |     2021 |     2022 |     2023 |     2024 |    2025 |
        |:-----|---------:|---------:|---------:|---------:|--------:|
        | ADBE | 132.425  | 112.433  | 102.593  | 130.005  | 212.536 |
        | CRM  |  18.2502 |  11.8861 |  51.1607 |  67.9382 | nan     |
        | INTU | 102.101  |  60.4477 |  99.9192 |  97.2924 | 173.485 |
        """
        check_option(
            "cash_flow_type",
            cash_flow_type,
            ["Free Cash Flow to Firm", "Free Cash Flow", "Operating Cash Flow"],
        )
        validate_window("periods", periods, 1)

        # The growth and discount rates are annual, so a quarterly base cash flow is the trailing twelve months.
        if trailing is None and self._quarterly:
            trailing = 4
        elif trailing and not self._quarterly:
            logger.warning(
                "The trailing parameter sums %s years of cash flow into the base cash flow of the DCF, "
                "while the growth rates and the WACC are annual rates.",
                trailing,
            )

        if cash_flow_type == "Free Cash Flow to Firm":
            free_cash_flow_to_firm = self.get_free_cash_flow_to_firm(rounding=15)
            require_input(free_cash_flow_to_firm, "free cash flow to firm")
            cash_flows = free_cash_flow_to_firm.xs("Free Cash Flow to Firm", level=1)
        else:
            cash_flows = self._cash_flow_statement.loc[:, cash_flow_type, :]

        operating_growth = cache(self._operating_growth)

        if historical:
            # Every period is valued with the rates and growth known at its end.
            periods_index = cash_flows.columns
            return self._historical_intrinsic_values(
                cash_flows=cash_flows,
                growth_rates=helpers.rates_per_period(
                    growth_rate,
                    self._tickers,
                    periods_index,
                    "growth rate",
                    operating_growth,
                ),
                perpetual_growth_rates=helpers.rates_per_period(
                    perpetual_growth_rate,
                    self._tickers,
                    periods_index,
                    "perpetual growth rate",
                    lambda: self._stable_growth(operating_growth()),
                ),
                waccs=helpers.rates_per_period(
                    weighted_average_cost_of_capital,
                    self._tickers,
                    periods_index,
                    "weighted average cost of capital",
                    self._weighted_average_cost_of_capital_rates,
                ),
                periods=periods,
                trailing=trailing,
                rounding=rounding,
            )

        growth_rate_dict = helpers.rates_per_ticker(
            growth_rate,
            self._tickers,
            "growth rate",
            lambda: self._latest(operating_growth()),
        )
        perpetual_growth_rate_dict = helpers.rates_per_ticker(
            perpetual_growth_rate,
            self._tickers,
            "perpetual growth rate",
            lambda: self._stable_growth(self._latest(operating_growth())),
        )
        wacc_dict = helpers.rates_per_ticker(
            weighted_average_cost_of_capital,
            self._tickers,
            "weighted average cost of capital",
            self._latest_weighted_average_cost_of_capital,
        )

        intrinsic_values_dict = {}
        for ticker in self._tickers:
            perpetual_growth_rate_float = perpetual_growth_rate_dict[ticker]
            growth_rate_float = growth_rate_dict[ticker]
            weighted_average_cost_of_capital_float = wacc_dict[ticker]
            if not helpers.exceeds_growth(
                ticker,
                weighted_average_cost_of_capital_float,
                perpetual_growth_rate_float,
                "weighted average cost of capital",
            ):
                continue
            # A value per share needs the shares, which the Total of a basket has none of.
            shares_outstanding = self._income_statement.loc[
                ticker, "Weighted Average Shares Diluted"
            ].dropna()
            if shares_outstanding.empty:
                continue
            # Summed on the calendar periods before the NaNs are dropped, so that the base is the
            # latest complete trailing window: a company reporting half-yearly has no complete
            # four-quarter year and is not valued, rather than summing two years of cash flow.
            cash_flow_series = (
                trailing_sum(cash_flows.loc[[ticker]], trailing).loc[ticker].dropna()
            )
            if cash_flow_series.empty:
                logger.warning(
                    "%s has no %s periods in a row with a %s, so its intrinsic value is not calculated.",
                    ticker,
                    trailing,
                    cash_flow_type,
                )
                continue

            intrinsic_values_dict[ticker] = intrinsic_model.get_intrinsic_value(
                cash_flow=cash_flow_series.iloc[-1],
                growth_rate=growth_rate_float,
                perpetual_growth_rate=perpetual_growth_rate_float,
                weighted_average_cost_of_capital=weighted_average_cost_of_capital_float,
                **{
                    name: value
                    for name, value in self._latest_capital(ticker).items()
                    if name != "shares_outstanding"
                },
                shares_outstanding=shares_outstanding.iloc[-1],
                periods=periods,
            )
            intrinsic_values_dict[ticker] = intrinsic_values_dict[ticker].pipe(
                apply_rounding, rounding if rounding is not None else self._rounding
            )

        if not intrinsic_values_dict:
            return pd.DataFrame()

        return pd.concat(intrinsic_values_dict).loc[:, self._start_date :]

    def _historical_intrinsic_values(
        self,
        cash_flows: pd.DataFrame,
        growth_rates: pd.DataFrame,
        perpetual_growth_rates: pd.DataFrame,
        waccs: pd.DataFrame,
        periods: int,
        trailing: int | None,
        rounding: int | None,
    ) -> pd.DataFrame:
        """
        The intrinsic value per share of every period, each from the base cash flow ending
        in that period, the cash, debt and shares reported at its end and the rates of that
        period (ticker by period frames), so that no period uses figures reported after it.
        """
        cash_flows = cash_flows.reindex(self._tickers)
        base_cash_flows = trailing_sum(cash_flows, trailing)
        cash = self._rows(self._balance_sheet_statement, "Cash and Cash Equivalents")
        debt = self._rows(self._balance_sheet_statement, "Total Debt")
        # Added back with the cash, and the minority interest and preferred stock deducted
        # with the debt, as get_enterprise_value_breakdown does the reverse; not reported
        # counts as zero.
        investments = self._rows(
            self._balance_sheet_statement, "Short Term Investments"
        ).fillna(0)
        minority_interest = self._rows(
            self._balance_sheet_statement, "Minority Interest"
        ).fillna(0)
        preferred_stock = self._rows(
            self._balance_sheet_statement, "Preferred Stock"
        ).fillna(0)
        shares = self._rows(self._income_statement, "Weighted Average Shares Diluted")

        intrinsic_values = pd.DataFrame(
            float("nan"), index=self._tickers, columns=base_cash_flows.columns
        )
        waccs = waccs.reindex(columns=base_cash_flows.columns)
        perpetual_growth_rates = perpetual_growth_rates.reindex(
            columns=base_cash_flows.columns
        )
        growth_rates = growth_rates.reindex(columns=base_cash_flows.columns)
        exceeds = helpers.exceeds_growth_per_period(
            waccs, perpetual_growth_rates, "weighted average cost of capital"
        )
        for ticker in self._tickers:
            for period in base_cash_flows.columns:
                if not exceeds.loc[ticker, period]:
                    continue
                inputs = [
                    base_cash_flows.loc[ticker, period],
                    cash.loc[ticker, period],
                    debt.loc[ticker, period],
                    shares.loc[ticker, period],
                    growth_rates.loc[ticker, period],
                    waccs.loc[ticker, period],
                ]
                if (
                    any(pd.isna(value) for value in inputs)
                    or not shares.loc[ticker, period]
                ):
                    continue
                intrinsic_values.loc[ticker, period] = (
                    intrinsic_model.get_intrinsic_value(
                        cash_flow=float(base_cash_flows.loc[ticker, period]),
                        growth_rate=float(growth_rates.loc[ticker, period]),
                        perpetual_growth_rate=float(
                            perpetual_growth_rates.loc[ticker, period]
                        ),
                        weighted_average_cost_of_capital=float(
                            waccs.loc[ticker, period]
                        ),
                        cash_and_cash_equivalents=float(cash.loc[ticker, period]),
                        total_debt=float(debt.loc[ticker, period]),
                        short_term_investments=float(investments.loc[ticker, period]),
                        minority_interest=float(minority_interest.loc[ticker, period]),
                        preferred_stock=float(preferred_stock.loc[ticker, period]),
                        shares_outstanding=float(shares.loc[ticker, period]),
                        periods=periods,
                    )
                    .loc["Intrinsic Value"]
                    .iloc[0]
                )

        intrinsic_values = apply_rounding(
            intrinsic_values.dropna(axis=1, how="all"),
            rounding if rounding is not None else self._rounding,
        )
        return intrinsic_values.loc[:, self._start_date : self._end_date]

    @handle_errors
    def get_free_cash_flow_to_firm(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Free Cash Flow to the Firm (FCFF) is the cash flow available to all providers of
        capital, both debt and equity holders.

        It is what is left after the company has paid all of its operating expenses and invested
        in the assets needed to sustain its operations. Because it is measured before any
        financing cash flows, FCFF is capital-structure neutral, making it the cash flow base
        typically discounted at the Weighted Average Cost of Capital (WACC) when valuing the
        Enterprise Value of a company directly.

        The formula is as follows:

            - NOPAT = EBIT * (1 - Corporate Tax Rate)
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
            trailing (int | None, optional): The trailing period to use for the calculation. The window counts
                calendar periods, so a period without a report makes it NaN. Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: DataFrame containing the FCFF and its components.

        Notes:
        - EBIT is the reported Operating Income, as in the Altman Z-Score and Economic Value
        Added calculations elsewhere in this module.
        - The Corporate Tax Rate is the effective tax rate (Income Tax Expense / Income Before Tax)
        kept within 0 and 1, and 0 for a period without a pre-tax profit, the same rate the
        Weighted Average Cost of Capital uses (see `get_weighted_average_cost_of_capital`).
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
            ["XOM", "CVX"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_free_cash_flow_to_firm().xs("Free Cash Flow to Firm", level=1)
        ```

        Which returns:

        |     |        2021 |        2022 |        2023 |        2024 |        2025 |
        |:----|------------:|------------:|------------:|------------:|------------:|
        | CVX | 1.98409e+10 | 3.51075e+10 | 2.2617e+10  | 1.97109e+10 | 1.16812e+10 |
        | XOM | 3.08399e+10 | 5.28525e+10 | 2.78286e+10 | 2.68446e+10 | 1.83253e+10 |
        """
        fcff = {}

        income_tax_expense = trailing_sum(
            self._income_statement.loc[:, "Income Tax Expense", :], trailing
        )
        income_before_tax = trailing_sum(
            self._income_statement.loc[:, "Income Before Tax", :], trailing
        )
        depreciation_and_amortization = trailing_sum(
            self._income_statement.loc[:, "Depreciation and Amortization", :], trailing
        )
        capital_expenditure = -1 * (
            trailing_sum(
                self._cash_flow_statement.loc[:, "Capital Expenditure", :], trailing
            )
        )
        change_in_net_working_capital = -1 * (
            trailing_sum(
                self._cash_flow_statement.loc[:, "Change in Working Capital", :],
                trailing,
            )
        )

        ebit = self._earnings_before_interest_and_taxes(trailing)
        # The tax rate the WACC uses: kept within 0 and 1 and 0 without a pre-tax profit, so that a tax benefit does not lift NOPAT above EBIT and a loss year is not taxed at the ratio of two negatives.  # noqa: E501
        effective_tax_rate = wacc_model.get_corporate_tax_rate(
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

        return self._finalize_components(fcff, rounding, growth, lag, standardize)

    @handle_errors
    def get_free_cash_flow_to_equity(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Free Cash Flow to Equity (FCFE) is the cash flow available to a company's common equity
        holders.

        It is what is left after all operating expenses, reinvestment needs, and net payments to
        (or from) debt holders have been accounted for. Unlike FCFF, FCFE is a levered cash flow
        measure and should be discounted at the Cost of Equity, not the Weighted Average Cost of
        Capital, when used to value equity directly.

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
            trailing (int | None, optional): The trailing period to use for the calculation. The window counts
                calendar periods, so a period without a report makes it NaN. Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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
            ["CMCSA", "CHTR", "VZ"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_free_cash_flow_to_equity().xs("Free Cash Flow to Equity", level=1)
        ```

        Which returns:

        |       |       2021 |       2022 |       2023 |        2024 |       2025 |
        |:------|-----------:|-----------:|-----------:|------------:|-----------:|
        | CHTR  | 1.5532e+10 | 1.1e+10    | 1.955e+09  | -1.31e+09   | 2.43e+09   |
        | CMCSA | 6.168e+09  | 3.618e+09  | 1.1394e+10 |  1.6448e+10 | 2.1732e+10 |
        | VZ    | 3.9649e+10 | 1.5507e+10 | 8.042e+09  |  1.0354e+10 | 2.3559e+10 |
        """
        fcfe = {}

        net_income = trailing_sum(
            self._income_statement.loc[:, "Net Income", :], trailing
        )
        depreciation_and_amortization = trailing_sum(
            self._income_statement.loc[:, "Depreciation and Amortization", :], trailing
        )
        capital_expenditure = -1 * (
            trailing_sum(
                self._cash_flow_statement.loc[:, "Capital Expenditure", :], trailing
            )
        )
        change_in_net_working_capital = -1 * (
            trailing_sum(
                self._cash_flow_statement.loc[:, "Change in Working Capital", :],
                trailing,
            )
        )
        net_borrowing = trailing_sum(
            self._cash_flow_statement.loc[:, "Net Debt Issued", :], trailing
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

        return self._finalize_components(fcfe, rounding, growth, lag, standardize)

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

    def _annual_dividends_per_share(self) -> pd.DataFrame:
        """
        Returns the annual Dividends per Share the dividend discount models start from: the
        dividends of each complete year, or with quarterly data the sum of the trailing four
        complete quarters. The rates of return and growth rates of these models are annual, so a
        single quarter's dividend would understate the base dividend by about a factor four.

        Returns:
            pd.DataFrame: The annual Dividends per Share per period (rows) and ticker (columns),
            without the first three quarters of quarterly data, which lack a full trailing year.
        """
        dividends_per_share = self._complete_period_dividends()

        if self._quarterly:
            # Only a full trailing year of quarters gives an annual dividend; fewer quarters would understate it.  # noqa: E501
            dividends_per_share = (
                dividends_per_share.rolling(4, min_periods=4).sum().iloc[3:]
            )

        return dividends_per_share

    @handle_errors
    def get_gordon_growth_model(
        self,
        rate_of_return: float | list | dict[str, float] | None = None,
        growth_rate: float | list | dict[str, float] | None = None,
        project_periods: int = 5,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        The Gordon Growth Model shows the intrinsic value of a stock from its expected future
        dividends.

        It is also known as the Dividend Discount Model (DDM) with Constant Growth. The model
        assumes that dividends will grow at a constant rate indefinitely.

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

        The Rate of Return must exceed the growth rate; periods where it doesn't are left out with a
        warning. Within 2 percentage points of it the value is calculated with a warning, as a tenth of a
        percentage point on either rate then moves it by about 5%.

        Also known as: Gordon Growth Model, DDM, dividend discount model, GGM.

        Args:
            rate_of_return (float, list or dict, optional): The required rate of return, one
                number, or a list or dict with a rate per ticker. Defaults to None, which uses
                each company's Cost of Equity of the period valued (the last one for the
                projections): the CAPM with the equity risk premium of its country and a beta
                of a year of daily returns, as in `get_weighted_average_cost_of_capital`.
            growth_rate (float, list or dict, optional): The growth rate of the dividends,
                which holds forever. Defaults to None, which uses the compound annual growth of
                the company's Dividends per Share up to the period valued, kept within 0 and the
                long-run growth rate of 2.5% (2.5% where it has none).
            project_periods (int, optional): The number of years to project the intrinsic value for, beyond the
                last complete period. Defaults to 5.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: DataFrame containing the intrinsic value for each ticker over time.

        Notes:
        - The results are highly dependent on the input. Therefore, think carefully about each input parameter to
        ensure the results are accurate (given your beliefs)
        - The rate of return and the growth rate are annual rates, so the model values an annual dividend:
        the Dividends per Share of each year, or with quarterly data the sum of the trailing four complete
        quarters. The first three quarters of quarterly data lack a full trailing year and are left out.
        - Each historical period is valued off that period's *actual* annual Dividends per Share, and only
        the years beyond the last complete period are projected forward at the given growth rate, one year
        per projection period (four quarters apart with quarterly data). With yearly data the current,
        unfinished year is projected as well, since its dividends so far cover only part of it. The first
        year of the historical window will often be understated, since it only covers the part of the year
        that falls inside the requested date range and so captures only part of the year's dividends. Use a
        start_date at least one full year before the first period you intend to read.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["KO"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_gordon_growth_model()
        ```

        Which returns:

        |      |       KO |
        |:-----|---------:|
        | 2021 | 133.744  |
        | 2022 |  68.4186 |
        | 2023 |  59.3591 |
        | 2024 | 102.976  |
        | 2025 | 104.728  |
        | 2026 | 107.347  |
        | 2027 | 110.03   |
        | 2028 | 112.781  |
        | 2029 | 115.601  |
        | 2030 | 118.491  |
        """
        dividends_per_share = self._annual_dividends_per_share()

        # The rate of return and growth rate are annual, so each projection step is one year: four quarters ahead with quarterly data.  # noqa: E501
        step = 4 if self._quarterly else 1
        last_known_period = dividends_per_share.index[-1]
        projected_periods = [
            last_known_period + step * distance
            for distance in range(1, project_periods + 1)
        ]

        # Every period is valued with the rate and growth known at its end; the projections
        # with those of the last period.
        rates = helpers.rates_per_period(
            rate_of_return,
            self._tickers,
            dividends_per_share.index,
            "rate of return",
            self._cost_of_equity,
        )
        growth_rates = helpers.rates_per_period(
            growth_rate,
            self._tickers,
            dividends_per_share.index,
            "growth rate",
            lambda: self._stable_growth(self._dividend_growth()),
        )
        exceeds = helpers.exceeds_growth_per_period(
            rates, growth_rates, "rate of return"
        )

        gordon_growth_model: dict[str, dict[str, float]] = {}

        for ticker in self._tickers:
            gordon_growth_model[ticker] = {}

            for period in dividends_per_share.index:
                if not exceeds.loc[ticker, period]:
                    continue
                gordon_growth_model[ticker][period] = (
                    intrinsic_model.get_gordon_growth_model(
                        dividends_per_share=dividends_per_share.loc[period, ticker],
                        rate_of_return=rates.loc[ticker, period],
                        growth_rate=growth_rates.loc[ticker, period],
                    )
                )

            rate = rates.loc[ticker, last_known_period]
            growth = growth_rates.loc[ticker, last_known_period]
            if not exceeds.loc[ticker, last_known_period]:
                continue

            for distance, period in enumerate(projected_periods, start=1):
                gordon_growth_model[ticker][period] = (
                    intrinsic_model.get_gordon_growth_model(
                        dividends_per_share=dividends_per_share.loc[
                            last_known_period, ticker
                        ]
                        * (1 + growth) ** distance,
                        rate_of_return=rate,
                        growth_rate=growth,
                    )
                )

        gordon_growth_model_df = pd.DataFrame(gordon_growth_model).dropna(how="all")

        gordon_growth_model_df = apply_rounding(
            gordon_growth_model_df, rounding if rounding is not None else self._rounding
        )

        return gordon_growth_model_df.loc[self._start_date :]

    @handle_errors
    def get_two_stage_dividend_discount_model(
        self,
        rate_of_return: float | list | dict[str, float] | None = None,
        high_growth_rate: float | list | dict[str, float] | None = None,
        stable_growth_rate: float | list | dict[str, float] | None = None,
        high_growth_periods: int = 5,
        transition_periods: int = 0,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        The Two-Stage Dividend Discount Model extends the (single-stage) Gordon Growth Model to
        companies that are not expected to grow at a constant rate forever.

        It explicitly projects and discounts dividends over an initial high-growth phase, and
        then values everything from the end of that phase onward as a perpetuity growing at a
        lower, more sustainable stable rate (a Gordon Growth terminal value).

        The formula is as follows:

            - Dividend Projection_t = Dividends Per Share * (1 + High Growth Rate)^t
            - High-Growth Phase Present Value = Sum of Dividend Projection_t / (1 + Rate of
            Return)^t, for t = 1, ..., High Growth Periods
            - Terminal Value = Last Dividend Projection * (1 + Stable Growth Rate) /
            (Rate of Return - Stable Growth Rate)
            - Intrinsic Value = High-Growth Phase Present Value +
            (Terminal Value / (1 + Rate of Return)^High Growth Periods)

        With transition periods it becomes the Three-Stage Dividend Discount Model: between
        the high-growth and the stable phase, the growth rate declines in equal steps from the
        high to the stable rate, and those dividends are discounted explicitly as well. In
        transition period k (of n), dividends grow at High Growth Rate - (High Growth Rate -
        Stable Growth Rate) * k / (n + 1), and the Terminal Value is taken at the end of the
        transition phase.

        Also known as: two-stage DDM, two-stage dividend discount model, three-stage DDM,
        three-stage dividend discount model.

        Args:
            rate_of_return (float, list or dict, optional): The required rate of return
            (discount rate) used to discount both phases. Can be one number to use for all
            tickers, or a list or dict that contains a rate of return for each ticker. Defaults
            to None, which uses the latest Cost of Equity of each company: the CAPM with the
            equity risk premium of its country, as in `get_weighted_average_cost_of_capital`.
            high_growth_rate (float, list or dict, optional): The constant growth rate applied
            to dividends during the explicit high-growth phase. Can be one number to use for all
            tickers, or a list or dict that contains a high growth rate for each ticker.
            Defaults to None, which uses the compound annual growth of the company's Dividends
            per Share up to the latest period.
            stable_growth_rate (float, list or dict, optional): The perpetual (terminal) growth
            rate applied to dividends from the end of the high-growth phase onward. Can be one
            number to use for all tickers, or a list or dict that contains a stable growth rate
            for each ticker. Defaults to None, which uses the company's dividend growth kept
            within 0 and the long-run growth rate of 2.5% (2.5% where it has none).
            high_growth_periods (int, optional): The number of years in the explicit
            high-growth phase. Defaults to 5.
            transition_periods (int, optional): The number of years in which the growth rate
            declines linearly from the high to the stable growth rate (the Three-Stage Dividend
            Discount Model). Defaults to 0, no transition phase.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: DataFrame containing the intrinsic value and its components for
            each ticker, including the present value of the transition phase when there is
            one.

        Notes:
        - The results are highly dependent on the input. Therefore, think carefully about
        each input parameter to ensure the results are accurate (given your beliefs).
        - The Rate of Return must be greater than the Stable Growth Rate, otherwise the
        Terminal Value formula divides by a non-positive number. Within 2 percentage points of
        it the value is calculated with a warning, as a tenth of a percentage point on either
        rate then moves it by about 5%.
        - The base dividend used for the projection is the annual Dividends Per Share of the last complete
        period for each ticker: the last complete year, or with quarterly data the sum of the last four
        complete quarters, since the rates are annual and each projection period is a year. The current,
        unfinished period is left out since its dividends so far cover only part of it. For companies that
        do not pay a dividend, this model is not meaningful (the projections and resulting intrinsic value
        will be zero).

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["JNJ", "PG", "PEP"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_two_stage_dividend_discount_model().xs(
            "Intrinsic Value", level=1
        )
        ```

        Which returns:

        |     |   High-Growth Periods = 5 |
        |:----|--------------------------:|
        | JNJ |                   296.375 |
        | PEP |                   294.698 |
        | PG  |                   201.233 |
        """
        dividends_per_share = self._annual_dividends_per_share()

        dividend_growth = cache(lambda: self._latest(self._dividend_growth()))
        rate_of_return_dict = helpers.rates_per_ticker(
            rate_of_return, self._tickers, "rate of return", self._latest_cost_of_equity
        )
        high_growth_rate_dict = helpers.rates_per_ticker(
            high_growth_rate, self._tickers, "high growth rate", dividend_growth
        )
        stable_growth_rate_dict = helpers.rates_per_ticker(
            stable_growth_rate,
            self._tickers,
            "stable growth rate",
            lambda: self._stable_growth(dividend_growth()),
        )

        two_stage_ddm_dict = {}
        for ticker in self._tickers:
            rate_of_return_float = rate_of_return_dict[ticker]
            high_growth_rate_float = high_growth_rate_dict[ticker]
            stable_growth_rate_float = stable_growth_rate_dict[ticker]
            if not helpers.exceeds_growth(
                ticker,
                rate_of_return_float,
                stable_growth_rate_float,
                "rate of return",
            ):
                continue
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
                    transition_periods=transition_periods,
                )
            )
            two_stage_ddm_dict[ticker] = two_stage_ddm_dict[ticker].pipe(
                apply_rounding, rounding if rounding is not None else self._rounding
            )

        two_stage_ddm_results = pd.concat(two_stage_ddm_dict)

        return two_stage_ddm_results.loc[:, self._start_date :]

    @handle_errors
    def get_h_model(
        self,
        rate_of_return: float | list | dict[str, float] | None = None,
        high_growth_rate: float | list | dict[str, float] | None = None,
        stable_growth_rate: float | list | dict[str, float] | None = None,
        high_growth_periods: int = 10,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        The H-Model of Fuller and Hsia (1984) shows the value of a stock whose dividend growth
        declines linearly to a stable rate.

        The growth starts at a high rate and declines over a number of years, after which it
        grows at the stable rate forever. It is a closed-form approximation of the three-stage
        dividend discount model (see `get_two_stage_dividend_discount_model` with
        `transition_periods`): the Gordon Growth Model at the stable rate plus the value of the
        extra growth during the decline, which is about half the high-growth period (H) times
        the extra growth.

        The formula is as follows:

            - H = High Growth Periods / 2
            - Intrinsic Value = Dividends Per Share * (1 + Stable Growth Rate) / (Rate of
            Return - Stable Growth Rate) + Dividends Per Share * H * (High Growth Rate -
            Stable Growth Rate) / (Rate of Return - Stable Growth Rate)

        Like the Gordon Growth Model, each period is valued off that period's actual annual
        Dividends per Share, which gives the intrinsic value over time.

        Also known as: H-Model, Fuller-Hsia model, H model dividend discount model.

        Args:
            rate_of_return (float, list or dict, optional): The required rate of return, which
                must be greater than the stable growth rate; one number, or a list or dict with a
                rate per ticker. Defaults to None, which uses each company's Cost of Equity of
                the period valued: the CAPM with the equity risk premium of its country, as in
                `get_weighted_average_cost_of_capital`.
            high_growth_rate (float, list or dict, optional): The growth rate of the dividends
                at the start of the decline. Defaults to None, which uses the compound annual
                growth of the company's Dividends per Share up to the period valued.
            stable_growth_rate (float, list or dict, optional): The growth rate the decline ends
                at, which holds forever after it. Defaults to None, which uses the company's
                dividend growth up to the period valued, kept within 0 and the long-run growth
                rate of 2.5% (2.5% where it has none).
            high_growth_periods (int, optional): The number of years over which the growth
                rate declines from the high to the stable rate. Defaults to 10.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the rounding of the Toolkit (4 unless set
                otherwise).
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The intrinsic value per share for each period (rows) and ticker
            (columns).

        Notes:
        - The results depend heavily on the inputs, in particular the high growth rate and
        how long it lasts. A rate of return within 2 percentage points of the stable growth
        rate is valued with a warning, as a tenth of a percentage point on either rate then
        moves the value by about 5%.
        - The H-Model is close to the exact value of a linearly declining growth rate when
        the high-growth period is short or the gap between the growth rates is small, and
        overstates it as either grows.
        - The rates are annual, so the model values an annual dividend: the Dividends per
        Share of each year, or with quarterly data the sum of the trailing four complete
        quarters. The first year of the window often covers only part of the year's
        dividends, so start the window a year before the first period you intend to read.
        - Fuller, R. J. and C.-C. Hsia, "A Simplified Common Stock Valuation Model,"
        Financial Analysts Journal 40(5), 1984, pp. 49-56.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["KO", "PEP"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_h_model()
        ```

        Which returns:

        |      |       KO |     PEP |
        |:-----|---------:|--------:|
        | 2022 |  75.9677 | 211.132 |
        | 2023 |  65.5951 | 203.877 |
        | 2024 | 115.099  | 382.255 |
        | 2025 | 117.365  | 292.194 |
        """
        dividends_per_share = self._annual_dividends_per_share()

        # Every period is valued with the rates and growth known at its end.
        periods = dividends_per_share.index
        dividend_growth = cache(self._dividend_growth)
        rates = helpers.rates_per_period(
            rate_of_return,
            self._tickers,
            periods,
            "rate of return",
            self._cost_of_equity,
        )
        high_growth_rates = helpers.rates_per_period(
            high_growth_rate,
            self._tickers,
            periods,
            "high growth rate",
            dividend_growth,
        )
        stable_growth_rates = helpers.rates_per_period(
            stable_growth_rate,
            self._tickers,
            periods,
            "stable growth rate",
            lambda: self._stable_growth(dividend_growth()),
        )
        exceeds = helpers.exceeds_growth_per_period(
            rates, stable_growth_rates, "rate of return"
        )

        dividends_per_share = dividends_per_share.reindex(columns=self._tickers)
        h_model = pd.DataFrame(
            {
                ticker: {
                    period: (
                        intrinsic_model.get_h_model(
                            dividends_per_share=dividends_per_share.loc[period, ticker],
                            rate_of_return=rates.loc[ticker, period],
                            high_growth_rate=high_growth_rates.loc[ticker, period],
                            stable_growth_rate=stable_growth_rates.loc[ticker, period],
                            high_growth_periods=high_growth_periods,
                        )
                        if exceeds.loc[ticker, period]
                        else float("nan")
                    )
                    for period in periods
                }
                for ticker in self._tickers
            },
            index=periods,
        ).dropna(how="all")
        h_model.index.name = None

        h_model = apply_rounding(
            h_model, rounding if rounding is not None else self._rounding
        )

        return h_model.loc[self._start_date :]

    @handle_errors
    def get_justified_multiples(
        self,
        rate_of_return: float | list | dict[str, float] | None = None,
        growth_rate: float | None = None,
        diluted: bool = True,
        trailing: int | None = None,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        The justified multiples are the Price-to-Earnings and Price-to-Book ratios that a
        company's fundamentals justify.

        They are the multiples the dividend discount and residual income models imply for a
        constant payout ratio, Return on Equity and growth rate. Comparing them with the actual
        multiples shows whether the market prices a company above or below what its
        profitability, payout and growth justify at a given required rate of return.

        The formula is as follows:

        - Growth Rate = Return on Equity * (1 - Payout Ratio)
        - Justified Trailing P/E = Payout Ratio * (1 + Growth Rate) / (Rate of Return - Growth Rate)
        - Justified Leading P/E = Payout Ratio / (Rate of Return - Growth Rate)
        - Justified P/B = (Return on Equity - Growth Rate) / (Rate of Return - Growth Rate)
        - Price-to-Earnings Ratio = Share Price / (Net Income / Weighted Average Shares)
        - Price-to-Book Ratio = Share Price / (Total Shareholder Equity / Weighted Average Shares)

        The Growth Rate is the sustainable growth rate above unless a growth rate is given.

        Also known as: justified P/E, justified P/B, fundamental P/E, warranted multiples.

        Args:
            rate_of_return (float, list or dict, optional): The required rate of return on
                equity, one number, or a list or dict with a rate per ticker. Defaults to None,
                which uses each company's Cost of Equity of each period: the CAPM with the equity
                risk premium of its country, as in `get_weighted_average_cost_of_capital`.
            growth_rate (float | None, optional): The constant growth rate of the earnings,
                dividends and book value. Defaults to None, which uses the sustainable growth
                rate of each period kept within 0 and the long-run growth rate of 2.5%, as a
                growth rate that holds forever can't exceed the growth of the economy.
            diluted (bool, optional): Whether to use the diluted share count for the actual
                multiples. Defaults to True.
            trailing (int | None, optional): Sum the flows over this many periods and measure
                the Return on Equity on the equity that many periods back. The window counts
                calendar periods, so a period without a report makes it NaN. Defaults to
                None, which is 4 for quarterly data (a trailing twelve months) and 1
                otherwise.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the rounding of the Toolkit (4 unless set
                otherwise).
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per ticker and period the Payout Ratio, Return on Equity, Growth
            Rate, Justified Trailing P/E, Justified Leading P/E, Justified P/B and the actual
            Price-to-Earnings and Price-to-Book Ratios.

        Notes:
        - The justified multiples are NaN where the growth rate is at or above the rate of
        return, since the models then have no finite value, and they mean little for a
        company that pays no dividend or makes a loss.
        - The Return on Equity is measured on the equity at the beginning of the period, as
        in the sustainable growth rate (Higgins), so the first period has none.
        - The actual multiples use the share price at the period end and the reported Net
        Income and equity per share, without the adjustment for preferred dividends that
        `toolkit.ratios.get_price_to_earnings_ratio` makes.
        - CFA Institute, "Equity Valuation: Applications and Processes" and "Market-Based
        Valuation: Price and Enterprise Value Multiples," CFA Program Curriculum Level II.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["KO", "PEP", "PG"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_justified_multiples().loc["KO"]
        ```

        Which returns:

        |                         |    2021 |    2022 |    2023 |    2024 |    2025 |
        |:------------------------|--------:|--------:|--------:|--------:|--------:|
        | Payout Ratio            |  0.7422 |  0.7982 |  0.7422 |  0.7863 |  0.6698 |
        | Return on Equity        |  0.5063 |  0.4149 |  0.4445 |  0.4098 |  0.5273 |
        | Growth Rate             |  0.025  |  0.025  |  0.025  |  0.025  |  0.025  |
        | Justified Trailing P/E  | 58.5587 | 31.0455 | 23.9391 | 41.6972 | 34.3992 |
        | Justified Leading P/E   | 57.1304 | 30.2883 | 23.3552 | 40.6802 | 33.5602 |
        | Justified P/B           | 37.0476 | 14.7954 | 13.1996 | 19.9092 | 25.1687 |
        | Price-to-Earnings Ratio | 26.2994 | 28.9985 | 23.8657 | 25.2999 | 23.0046 |
        | Price-to-Book Ratio     | 11.1732 | 11.4791 |  9.8569 | 10.8209 |  9.3731 |
        """
        if trailing is None and self._quarterly:
            trailing = 4

        return_on_equity, payout_ratio = self._return_on_equity_and_payout(trailing)
        return_on_equity = return_on_equity.reindex(self._tickers)
        payout_ratio = payout_ratio.reindex(self._tickers)
        # Every period uses the rate of return of that period.
        rate_of_return = helpers.rates_per_period(
            rate_of_return,
            self._tickers,
            payout_ratio.columns,
            "rate of return",
            self._cost_of_equity,
        )

        if growth_rate is None:
            # A growth rate kept forever: the sustainable growth rate of the period, kept
            # within 0 and the long-run growth rate like every stable growth rate of the
            # valuations.
            growth = growth_model.get_sustainable_growth_rate(
                return_on_equity=return_on_equity,
                retention_ratio=valuation_model.get_retention_ratio(
                    dividend_payout_ratio=payout_ratio
                ),
            ).clip(lower=0, upper=LONG_RUN_GROWTH_RATE)
        else:
            growth = pd.DataFrame(
                growth_rate,
                index=return_on_equity.index,
                columns=return_on_equity.columns,
            )

        net_income = self._rows(self._income_statement, "Net Income")
        net_income = trailing_sum(net_income, trailing)
        shares = self._rows(
            self._income_statement,
            "Weighted Average Shares Diluted" if diluted else "Weighted Average Shares",
        )
        equity = self._rows(self._balance_sheet_statement, "Total Shareholder Equity")
        prices = self._period_prices().reindex(self._tickers)

        results = {
            "Payout Ratio": payout_ratio,
            "Return on Equity": return_on_equity,
            "Growth Rate": growth,
            "Justified Trailing P/E": intrinsic_model.get_justified_price_to_earnings(
                payout_ratio, rate_of_return, growth
            ),
            "Justified Leading P/E": intrinsic_model.get_justified_price_to_earnings(
                payout_ratio, rate_of_return, growth, leading=True
            ),
            "Justified P/B": intrinsic_model.get_justified_price_to_book(
                return_on_equity, rate_of_return, growth
            ),
            "Price-to-Earnings Ratio": prices / (net_income / shares),
            "Price-to-Book Ratio": prices / (equity / shares),
        }

        return self._finish_score(results, rounding, first_columns=trailing or 1)

    @handle_errors
    def get_residual_income(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
        market_risk_premium: float | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Residual Income is a measure of the profit a company generates in excess of the return
        required by its equity holders.

        It is the equity-side counterpart to Economic Value Added (EVA), and underpins the
        Residual Income Model, an alternative equity valuation lens to a traditional Discounted
        Cash Flow (DCF) that is particularly useful when a company's free cash flows are
        negative or unpredictable but its accounting earnings are more stable.

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
            trailing (int | None, optional): The trailing period to use for the calculation. The window counts
                calendar periods, so a period without a report makes it NaN. Defaults to None.
            market_risk_premium (float | None, optional): The annual equity risk premium the Cost of Equity
                uses. Defaults to None, which takes the equity risk premium of each company's country
                (Damodaran's Total Equity Risk Premium, see get_market_risk_premium) and, where that is
                not available, the benchmark's average return above the risk-free rate over the data.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: DataFrame containing the Residual Income and its components.

        Notes:
        - A positive Residual Income indicates that the company generated more profit than
        equity holders required given the capital they have invested. A negative Residual
        Income indicates the company failed to earn its equity holders' required return, even
        if it still reported a positive Net Income.
        - The Cost of Equity follows from the Capital Asset Pricing Model (CAPM), as in
          `get_weighted_average_cost_of_capital`: the risk-free rate of the period plus Beta
          times the market risk premium, by default the equity risk premium of the company's
          country (see `market_risk_premium`). It covers the span of the Net Income: a quarter
          with quarterly data, and the per-period rate times `trailing` when the Net Income is a
          trailing sum (an annual rate for the trailing twelve months of quarterly data), with
          the Beta estimated from the daily returns of the same trailing periods.
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
            ["V", "MA", "AXP"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2022-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_residual_income().xs("Residual Income", level=1)
        ```

        Which returns:

        |     |        2022 |        2023 |        2024 |        2025 |
        |:----|------------:|------------:|------------:|------------:|
        | AXP | 5.62881e+09 | 5.94452e+09 | 7.56109e+09 | 7.62666e+09 |
        | MA  | 9.35653e+09 | 1.06972e+10 | 1.23826e+10 | 1.44721e+10 |
        | V   | 1.24393e+10 | 1.48221e+10 | 1.71906e+10 | 1.69441e+10 |
        """
        residual_income = {}

        net_income = trailing_sum(
            self._income_statement.loc[:, "Net Income", :], trailing
        )
        # The capital charge applies to the book value the shareholders had invested at the start of the period (Ohlson, 1995): the closing balance of the previous period, or of `trailing` periods back for a trailing sum of Net Income.  # noqa: E501
        beginning_book_value_of_equity = (
            self._balance_sheet_statement.loc[:, "Total Shareholder Equity", :]
            - self._balance_sheet_statement.loc[:, "Preferred Stock", :]
        ).shift(trailing or 1, axis=1)

        years = self._balance_sheet_statement.columns
        begin, end = str(years[0]), str(years[-1])

        risk_free_rate = self._risk_free_rate_data[
            "quarterly" if self._quarterly else "yearly"
        ].loc[begin:end, "Adj Close"]

        beta = self._beta_per_period(begin, end, trailing)

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

        return self._finalize_components(
            residual_income, rounding, growth, lag, standardize
        )

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
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        The Present Value of Growth Opportunities (PVGO) is a financial metric that represents
        the present value of a company's future growth opportunities.

        It is calculated as the difference between the share price and the value of the
        company's current earnings as a perpetuity without growth, i.e. Earnings per Share
        divided by the Cost of Equity.

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
            trailing (int | None, optional): The trailing period to use for the calculation. The window counts
                calendar periods, so a period without a report makes it NaN. Defaults to None.
            market_risk_premium (float | None, optional): The annual equity risk premium the Cost of Equity
                uses. Defaults to None, which takes the equity risk premium of each company's country
                (Damodaran's Total Equity Risk Premium, see get_market_risk_premium) and, where that is
                not available, the benchmark's average return above the risk-free rate over the data.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the values. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: DataFrame containing the PVGO values.

        Notes:
        - The no-growth value (Earnings per Share / r) is discounted at the Cost of Equity, since both
        the share price and the Earnings per Share are equity-only, per-share quantities (Brealey, Myers
        and Allen, "Principles of Corporate Finance"; Bodie, Kane and Marcus, "Investments"). The Cost of
        Equity is the CAPM Cost of Equity of `get_weighted_average_cost_of_capital`.
        - The Cost of Equity uses the same `trailing` as the Earnings per Share, so both cover
          the same span (with quarterly data, a quarter without `trailing` and a year with four
          trailing quarters).
        - When the Earnings per Share is negative, the no-growth value is negative and the PVGO
        exceeds the share price, which has no meaningful interpretation.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["NVDA", "TXN", "ADI"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2022-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_present_value_of_growth_opportunities()
        ```

        Which returns:

        | Date   |      ADI |     NVDA |      TXN |
        |:-------|---------:|---------:|---------:|
        | 2022   |  84.5011 |  18.9955 |  44.4789 |
        | 2023   |  91.829  |  51.8345 |  96.3404 |
        | 2024   | 196.881  | 124.329  | 136.656  |
        | 2025   | 193.895  | nan      | 119.909  |
        """
        # Full precision, so that the rounding of the display does not move the no-growth value.
        cost_of_equity = self._weighted_average_cost_of_capital(
            trailing=trailing, market_risk_premium=market_risk_premium
        ).xs("Cost of Equity", level=1)

        average_shares = self._income_statement.loc[
            :,
            "Weighted Average Shares Diluted" if diluted else "Weighted Average Shares",
            :,
        ]

        # Preferred Dividends Paid is reported on the cash flow statement using the cash-flow-impact convention (an outflow is negative), while the Earnings per Share formula subtracts a positive-magnitude figure — without the absolute value the preferred dividends would be added back to Net Income instead of deducted from it  # noqa: E501
        net_income = trailing_sum(
            self._income_statement.loc[:, "Net Income", :], trailing
        )

        if include_dividends:
            dividends = self._cash_flow_statement.loc[
                :, "Preferred Dividends Paid", :
            ].abs()
            preferred_dividends = trailing_sum(dividends, trailing)
        else:
            preferred_dividends = 0
        # Without shares (0 reported) there are no earnings per share, rather than infinite
        # ones that would make the growth opportunities infinite.
        avg_shares = trailing_mean(average_shares, trailing)
        avg_shares = avg_shares.where(avg_shares > 0)

        earnings_per_share = valuation_model.get_earnings_per_share(
            net_income=net_income,
            preferred_dividends=preferred_dividends,
            average_outstanding_shares=avg_shares,
        )

        # The period values compare the earnings with the close at the end of the fiscal
        # period, and the daily values apply each period's earnings from that day on.
        pvgo = growth_model.get_present_value_of_growth_opportunities(
            cost_of_equity=cost_of_equity,
            earnings_per_share=earnings_per_share,
            close_prices=(
                self._historical_data["daily"]["Close"][self._tickers]
                if calculate_daily
                else self._period_prices().T.rename_axis("Date")
            ),
            calculate_daily=calculate_daily,
            period_end_dates=self._market_dates(),
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
                    dataset=pvgo.loc[self._start_date :],
                    rounding=rounding if rounding is not None else self._rounding,
                    axis="rows",
                )
            return pvgo.loc[self._start_date :]

        pvgo = apply_rounding(
            pvgo, rounding if rounding is not None else self._rounding
        )

        if standardize:
            pvgo = calculate_standardization(
                dataset=pvgo.loc[self._start_date :],
                rounding=rounding if rounding is not None else self._rounding,
                axis="rows",
            )

        # When there is no data found for any ticker, drop the row
        pvgo = pvgo.dropna(how="all", axis=0)

        return pvgo.loc[self._start_date :]

    @handle_errors
    def get_graham_number(
        self,
        diluted: bool = True,
        trailing: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
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
            trailing (int | None, optional): The trailing period to use for the calculation. The window counts
                calendar periods, so a period without a report makes it NaN. Defaults to None, which uses 4
                (the trailing twelve months of earnings) with quarterly data and a single period with yearly
                data.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the values. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: DataFrame containing the Graham Number values.

        Notes:
        - Graham's limits of 15 times earnings and 1.5 times book value refer to annual earnings, so
        with quarterly data the Earnings per Share are the sum of the last four quarters by default.
        - Earnings and Book Value per Share use the Weighted Average Shares Diluted (or the
          Weighted Average Shares without dilution).
        - Only the Earnings per Share are trailing: the earnings are a flow over the trailing periods,
        divided by the average shares over those periods. The Book Value per Share is a stock, the Total
        Shareholder Equity minus the Preferred Stock at the end of the period divided by the shares of
        that period.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["BAC", "C", "WFC"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_graham_number()
        ```

        Which returns:

        |     |     2021 |     2022 |    2023 |     2024 |     2025 |
        |:----|---------:|---------:|--------:|---------:|---------:|
        | BAC |  47.9904 |  46.3468 | 47.4941 |  50.4045 |  56.4118 |
        | C   | 142.751  | 120.826  | 93.472  | 117.004  | 127.785  |
        | WFC |  68.3912 |  55.5576 | 69.7188 |  74.7554 |  85.6834 |
        """
        # Graham's multiples apply to annual earnings, so a quarterly Earnings per Share is the trailing twelve months.  # noqa: E501
        if trailing is None and self._quarterly:
            trailing = 4

        average_shares = self._income_statement.loc[
            :,
            "Weighted Average Shares Diluted" if diluted else "Weighted Average Shares",
            :,
        ]
        earnings_per_share = self._trailing_earnings_per_share(diluted, trailing)

        # Book value is a stock at the balance sheet date, so equity and preferred stock stay point in time and are divided by the latest shares; only the earnings, a flow, cover the trailing year.  # noqa: E501
        book_value_per_share = valuation_model.get_book_value_per_share(
            total_shareholder_equity=self._balance_sheet_statement.loc[
                :, "Total Shareholder Equity", :
            ],
            preferred_equity=self._balance_sheet_statement.loc[:, "Preferred Stock", :],
            common_shares_outstanding=average_shares,
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

    @handle_errors
    def get_graham_intrinsic_value(
        self,
        growth_rate: float | list | dict[str, float] | None = None,
        bond_yield: float | list | dict[str, float] | None = None,
        diluted: bool = True,
        trailing: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the intrinsic value of a share with Benjamin Graham's growth formula.

        It is the value of a growth stock from its earnings, their expected growth and the yield
        of high grade corporate bonds. Graham gave the formula in The Intelligent Investor
        (fourth revised edition, 1973, chapter 11) as Earnings per Share x (8.5 + 2g), with 8.5
        the price-to-earnings ratio of a company without growth and g the growth rate of its
        earnings expected over the next 7 to 10 years, in percent. In 1974 he scaled it by 4.4 /
        Y, with 4.4 the average yield of AAA corporate bonds in 1962 and Y their current yield,
        so that the value falls when interest rates rise.

        The formula is as follows:

        - Intrinsic Value = Earnings per Share x (8.5 + 2 x g) x 4.4 / Y

        Graham wrote g and Y in percent (5 for 5%). The growth rate and the bond yield of this
        method are decimals like every rate in the Finance Toolkit (0.05 for 5%), and the
        formula multiplies them by 100 before it applies Graham's constants, so the value is
        Graham's. The bond yield is Moody's seasoned Aaa corporate bond yield, the yield to
        maturity of bonds with 20 years or more to maturity that Moody's publishes on their
        semi-annual coupon basis, used as published, as Graham's 4.4 was.

        Graham meant the formula as a way to compare growth stocks and warned that growth
        projections are unreliable: a higher g raises the value without limit, so the value
        of a fast grower rests mostly on the growth assumed. A company with a loss has no
        value by the formula, so it is NaN.

        Also known as: Graham formula, Graham growth formula, Benjamin Graham intrinsic value,
        revised Graham formula.

        Args:
            growth_rate (float, list or dict, optional): The expected annual growth rate of the
                earnings per share over the next 7 to 10 years, as a decimal: one number, or a
                list or dict with a rate per ticker. Defaults to None, which uses the compound
                annual growth of the company's Earnings per Share over the 10 years up to each
                period (or all the years before it when there are fewer, at least three, NaN
                with less history), kept within 0 and 0.20, the range of Graham's own table of
                growth rates, and NaN where the earnings at either end are not positive.
            bond_yield (float, list or dict, optional): The yield of AAA corporate bonds, as a
                decimal: one number, or a list or dict with a yield per ticker. Defaults to None,
                which uses Moody's seasoned Aaa corporate bond yield (FRED series DAAA) on the
                last day up to the end of each fiscal period, which needs a FRED API key.
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            trailing (int | None, optional): The trailing period to use for the calculation. The window counts
                calendar periods, so a period without a report makes it NaN. Defaults to None, which uses 4
                (the trailing twelve months of earnings) with quarterly data and a single period with yearly
                data.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the values. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: DataFrame containing the intrinsic value per share.

        Notes:
        - Graham's formula takes the earnings of the last twelve months, so with quarterly data
          the Earnings per Share are the sum of the last four quarters by default; it is the Net
          Income divided by the Weighted Average Shares Diluted (or the Weighted Average Shares
          without dilution), as in `get_graham_number`.
        - The default growth rate looks back, while Graham's g is the growth expected ahead: pass a
        growth_rate (e.g. from analyst estimates, see `toolkit.get_analyst_estimates`) to value on a
        forecast. Each period only uses the earnings up to it, and the bond yield of the end of that
        period, so no period is valued with figures that were only known later. The growth reaches back
        as far as the statements the Toolkit holds (a year before the start date) and needs three
        years of them, so the first periods are NaN and an earlier start_date gives a longer growth
        window.
        - Compare the value with the share price (`toolkit.get_historical_data`) for Graham's margin of
        safety.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["KO", "PEP", "JNJ"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            fred_api_key="FRED_API_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_graham_intrinsic_value()
        ```

        Which returns:

        |     |   2021 |   2022 |    2023 |     2024 |     2025 |
        |:----|-------:|-------:|--------:|---------:|---------:|
        | JNJ |    nan |    nan | 630.081 |  51.9832 | 349.49   |
        | KO  |    nan |    nan |  72.556 |  50.1366 |  76.9353 |
        | PEP |    nan |    nan | 160.224 | 138.396  |  74.4379 |
        """
        # Graham's formula values the earnings of the last twelve months, so a quarterly Earnings per Share is the trailing twelve months.  # noqa: E501
        if trailing is None and self._quarterly:
            trailing = 4

        earnings_per_share = self._trailing_earnings_per_share(diluted, trailing)

        growth_rates = helpers.rates_per_period(
            growth_rate,
            self._tickers,
            earnings_per_share.columns,
            "growth rate",
            lambda: self._earnings_per_share_growth(earnings_per_share),
        )
        bond_yields = helpers.rates_per_period(
            bond_yield,
            self._tickers,
            earnings_per_share.columns,
            "bond yield",
            self._aaa_corporate_bond_yields,
        )

        graham_intrinsic_value = intrinsic_model.get_graham_intrinsic_value(
            earnings_per_share=earnings_per_share.reindex(self._tickers),
            growth_rate=growth_rates,
            bond_yield=bond_yields,
        )

        return finalize_dataset(
            dataset=graham_intrinsic_value,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="columns",
        )

    def _trailing_earnings_per_share(
        self, diluted: bool, trailing: int | None
    ) -> pd.DataFrame:
        """
        The Earnings per Share of Graham's formulas per ticker (rows) and period (columns): the
        Net Income summed over `trailing` periods divided by the average (diluted) shares over
        those periods.
        """
        average_shares = self._income_statement.loc[
            :,
            "Weighted Average Shares Diluted" if diluted else "Weighted Average Shares",
            :,
        ]

        return valuation_model.get_earnings_per_share(
            net_income=trailing_sum(
                self._income_statement.loc[:, "Net Income", :], trailing
            ),
            preferred_dividends=0,
            average_outstanding_shares=trailing_mean(average_shares, trailing),
        )

    def _earnings_per_share_growth(
        self, earnings_per_share: pd.DataFrame
    ) -> pd.DataFrame:
        """
        The growth rate of Graham's intrinsic value when none is given: the compound annual
        growth of the Earnings per Share over the 10 years up to each period (columns), or
        from the first period with earnings when that is later, from at least three years
        back, as a shorter span says little about the 7 to 10 years Graham's g looks ahead
        (NaN with less history). Each period only uses the earnings up to it. A compound rate needs positive earnings
        at both ends, so it is NaN otherwise, and it is kept within 0 and 0.20, the growth
        rates Graham tabulated in The Intelligent Investor (chapter 11).
        """
        periods_per_year = 4 if self._quarterly else 1
        window = 10 * periods_per_year
        growth = pd.DataFrame(
            float("nan"),
            index=earnings_per_share.index,
            columns=earnings_per_share.columns,
        )

        for ticker in earnings_per_share.index:
            history = earnings_per_share.loc[ticker]
            first = history.first_valid_index()
            if first is None:
                continue
            first_position = history.index.get_loc(first)

            for position in range(first_position + 3 * periods_per_year, len(history)):
                base_position = max(first_position, position - window)
                base, latest = history.iloc[base_position], history.iloc[position]
                if base > 0 and latest > 0:
                    years = (position - base_position) / periods_per_year
                    growth.iloc[growth.index.get_loc(ticker), position] = (
                        latest / base
                    ) ** (1 / years) - 1

        return growth.clip(lower=0, upper=GRAHAM_MAXIMUM_GROWTH_RATE)

    def _aaa_corporate_bond_yields(self) -> pd.DataFrame:
        """
        Moody's seasoned Aaa corporate bond yield per ticker (rows) and period (columns), on
        the last day with a yield up to each ticker's fiscal period end (see _market_dates),
        so that no period is valued with a yield that was only known later.
        """
        if self._corporate_bond_yield_function is None:
            raise ValueError(
                "No source for the AAA corporate bond yield was given, so pass a bond_yield."
            )

        market_dates = self._market_dates()
        # The yield up to 10 days before a date counts (a holiday or weekend before it).
        start = (market_dates.min().min() - pd.Timedelta(days=15)).strftime("%Y-%m-%d")
        end = market_dates.max().max().strftime("%Y-%m-%d")

        return self._values_at_market_dates(
            self._corporate_bond_yield_function(start, end)
        )

    def _latest_value(self, statement: pd.DataFrame, ticker: str, row: str) -> float:
        """The latest reported value of a statement row for a ticker, or NaN."""
        values = self._rows(statement, row).loc[ticker].dropna()
        return float(values.iloc[-1]) if not values.empty else float("nan")

    def _latest_price(self, ticker: str) -> float:
        """The latest daily share price of a ticker, as traded."""
        prices = self._historical_data["daily"]["Close"][ticker].dropna()
        return float(prices.iloc[-1]) if not prices.empty else float("nan")

    def _latest(self, frame: pd.DataFrame) -> pd.Series:
        """The latest value of each ticker (rows) of a ticker by period frame."""
        return frame.T.ffill().iloc[-1] if not frame.empty else pd.Series(dtype=float)

    def _beta_per_period(
        self, begin: str, end: str, trailing: int | None
    ) -> pd.DataFrame:
        """
        The Beta of each period (rows) and ticker (columns) from the daily returns within it,
        for the Cost of Equity. With `trailing` set, a period's Beta covers the daily returns
        of the trailing periods, the span of the rest of the Cost of Equity: a quarter's 63
        days give an unstable Beta, a year a usable one.
        """
        returns_within = self._within_historical_data[
            "quarterly" if self._quarterly else "yearly"
        ]["Return"][list(dict.fromkeys([*self._tickers, self._benchmark_name]))]
        if trailing:
            returns_within = helpers.trailing_within_periods(returns_within, trailing)
        returns_within = returns_within.loc[begin:end]

        return get_beta(
            returns_within[self._tickers], returns_within[self._benchmark_name]
        )

    def _latest_weighted_average_cost_of_capital(self) -> pd.Series:
        """The latest WACC per ticker, for the valuations of today."""
        return self._latest(self._weighted_average_cost_of_capital_rates())

    def _latest_cost_of_equity(self) -> pd.Series:
        """The latest Cost of Equity per ticker, for the valuations of today."""
        return self._latest(self._cost_of_equity())

    def _weighted_average_cost_of_capital_rates(self) -> pd.DataFrame:
        """
        The WACC per ticker (rows) and period (columns), the discount rate of the valuations
        when none is given. The WACC is a yearly rate, so on quarterly data it covers four
        quarters (interest expense and a beta of a year of daily returns), whatever the
        trailing of the valued cash flows.
        """
        return (
            self._weighted_average_cost_of_capital(
                trailing=4 if self._quarterly else None
            )
            .loc[:, "Weighted Average Cost of Capital", :]
            .reindex(self._tickers)
        )

    def _cost_of_equity(self) -> pd.DataFrame:
        """
        The Cost of Equity per ticker (rows) and period (columns), the required return of the
        dividend discount models and justified multiples when none is given: the CAPM with
        each company's country equity risk premium and a beta of a year of daily returns, as
        in `get_weighted_average_cost_of_capital`.
        """
        return (
            self._weighted_average_cost_of_capital(
                trailing=4 if self._quarterly else None
            )
            .xs("Cost of Equity", level=1)
            .reindex(self._tickers)
        )

    def _dividend_growth(self) -> pd.DataFrame:
        """
        The compound annual growth of each company's Dividends per Share from its first year
        with a dividend up to each period (columns), the high growth rate of the dividend
        discount models when none is given. Each period only uses the dividends up to it. NaN
        until a company has paid dividends for more than a year.
        """
        dividends = self._annual_dividends_per_share().where(lambda frame: frame > 0)
        periods_per_year = 4 if self._quarterly else 1
        growth = {}
        for ticker in self._tickers:
            history = (
                dividends[ticker].dropna()
                if ticker in dividends
                else pd.Series(dtype=float)
            )
            if history.empty:
                continue
            years = pd.Series(
                [
                    (period.ordinal - history.index[0].ordinal) / periods_per_year
                    for period in history.index
                ],
                index=history.index,
            )
            growth[ticker] = ((history / history.iloc[0]) ** (1 / years) - 1).where(
                years >= 1
            )
        return pd.DataFrame(growth, index=dividends.index).T.reindex(self._tickers)

    def _operating_growth(self) -> pd.DataFrame:
        """
        The growth of each company's operating income that its reinvestment supports, the
        growth rate of the discounted cash flow valuations when none is given (Damodaran,
        Investment Valuation, chapter 11): Reinvestment Rate x Return on Invested Capital,
        which equals the reinvestment per year as a share of the invested capital.

        - Reinvestment = NOPAT - Free Cash Flow to Firm (the capital expenditure above the
          depreciation and amortization, plus the increase in working capital)
        - Growth = Sum of Reinvestment / Number of Years / Average Invested Capital, with the
          Invested Capital the Total Shareholder Equity plus the Total Debt, as in the Return
          on Invested Capital

        Reinvestment is lumpy, so each period (column) takes it over all the periods up to
        and including it, from at least a year of data; it never uses later periods. A company
        that reinvests less than its depreciation has a negative growth.
        """
        components = self.get_free_cash_flow_to_firm(rounding=15)
        # Without the cash flow items (an empty result) no company has a growth of its own.
        if components.empty:
            return pd.DataFrame(index=self._tickers, dtype=float)
        nopat = components.xs("Net Operating Profit After Taxes", level=1)
        free_cash_flow = components.xs("Free Cash Flow to Firm", level=1)
        reinvestment = (nopat - free_cash_flow).reindex(self._tickers).T
        invested_capital = (
            (
                self._rows(self._balance_sheet_statement, "Total Shareholder Equity")
                + self._rows(self._balance_sheet_statement, "Total Debt")
            )
            .reindex(self._tickers)
            .T.reindex(reinvestment.index)
        )

        periods_per_year = 4 if self._quarterly else 1
        years = reinvestment.notna().cumsum() / periods_per_year
        total_reinvestment = reinvestment.fillna(0).cumsum()
        return (
            (total_reinvestment / years / invested_capital.expanding().mean())
            .where(years >= 1)
            .T
        )

    def _stable_growth(
        self, company_growth: pd.Series | pd.DataFrame
    ) -> pd.Series | pd.DataFrame:
        """
        The growth rate a company keeps forever when none is given: its own growth, kept
        within 0 and the long-run growth rate, and the long-run rate where it has none. A
        stable growth rate can't exceed the growth of the economy, for which the risk-free
        rate is the upper limit (Damodaran, Investment Valuation, chapter 12); the long-run
        rate stays well below it. A company that shrinks today is not assumed to shrink
        forever, so the stable rate is at least 0.
        """
        return (
            company_growth.reindex(self._tickers)
            .clip(lower=0, upper=LONG_RUN_GROWTH_RATE)
            .fillna(LONG_RUN_GROWTH_RATE)
        )

    def _calendar_sum(
        self, statement: pd.DataFrame, row: str, trailing: int | None
    ) -> pd.DataFrame:
        """
        A statement row on the calendar periods, summed over `trailing` periods where set.
        Without min_periods every window must be complete, so a missing period leaves the
        sums that include it NaN instead of letting them span more periods.
        """
        return trailing_sum(calendar_periods(self._rows(statement, row)), trailing)

    def _latest_capital(self, ticker: str) -> dict[str, float]:
        """
        The latest cash, short term investments, total debt, minority interest, preferred
        stock and diluted shares of a ticker, as a DCF takes them: the claims and assets
        get_intrinsic_valuation adds to and deducts from the Enterprise Value, so that every
        valuation on it arrives at the same Equity Value. A short term investment, minority
        interest or preferred stock that is not reported counts as zero.
        """
        balance = self._balance_sheet_statement

        def optional(row: str) -> float:
            value = self._latest_value(balance, ticker, row)
            return 0.0 if pd.isna(value) else value

        return {
            "cash_and_cash_equivalents": self._latest_value(
                balance, ticker, "Cash and Cash Equivalents"
            ),
            "short_term_investments": optional("Short Term Investments"),
            "total_debt": self._latest_value(balance, ticker, "Total Debt"),
            "minority_interest": optional("Minority Interest"),
            "preferred_stock": optional("Preferred Stock"),
            "shares_outstanding": self._latest_value(
                self._income_statement, ticker, "Weighted Average Shares Diluted"
            ),
        }

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
        check_option("cash_flow_type", cash_flow_type, valid)

        def summed(statement: pd.DataFrame, row: str) -> pd.DataFrame:
            return self._calendar_sum(statement, row, trailing)

        if cash_flow_type != "Free Cash Flow to Firm":
            return summed(self._cash_flow_statement, cash_flow_type)

        # The tax rate of the WACC and the EVA (the effective rate within 0 and 1, and 0 without
        # a pre-tax profit), so that a tax benefit or a loss does not raise the NOPAT above EBIT.
        effective_tax_rate = wacc_model.get_corporate_tax_rate(
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
        perpetual_growth_rate: float | dict[str, float] = LONG_RUN_GROWTH_RATE,
        weighted_average_cost_of_capital: float | dict[str, float] | None = None,
        periods: int = 5,
        cash_flow_type: str = "Free Cash Flow to Firm",
        trailing: int | None = None,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the growth rate the share price implies, a reverse Discounted Cash Flow (DCF).

        Instead of estimating growth to arrive at a value, it takes the current share price as
        given and solves for the yearly cash flow growth over the projection periods at which
        the DCF value per share equals that price. The answer is the market's expectation, which
        can then be compared with the company's history: an implied growth far above what the
        company has achieved signals optimism.

        The DCF is the same as in `get_intrinsic_valuation`, by default on the Free Cash
        Flow to the Firm:

            - Cash Flow Projection_t = Cash Flow_t-1 * (1 + Implied Growth Rate)
            - Terminal Value = Last Projection * (1 + Perpetual Growth Rate) / (WACC - Perpetual Growth Rate)
            - Equity Value = Present Value of Projections and Terminal Value + Cash + Short Term
              Investments - Total Debt - Minority Interest - Preferred Stock
            - Implied Growth Rate solves Equity Value / Shares Outstanding = Share Price

        For more information, see the following book:

        - Rappaport, A. and Mauboussin, M.J. (2001). "Expectations Investing: Reading Stock
          Prices for Better Returns." Harvard Business School Press.

        Also known as: reverse DCF, market-implied growth, expectations investing.

        Args:
            perpetual_growth_rate (float | dict[str, float], optional): The growth after the
                projection periods, one number or a number per ticker. Defaults to the
                long-run growth rate of 2.5%.
            weighted_average_cost_of_capital (float | dict[str, float] | None, optional):
                The discount rate, one number or a number per ticker. Defaults to None,
                which uses the latest Weighted Average Cost of Capital of each company,
                calculated over a year (the last 4 quarters with quarterly data) whatever
                `trailing` the cash flows use, as the rate is a yearly one.
            periods (int, optional): The number of projection periods, at least 1. Defaults to 5.
            cash_flow_type (str, optional): The cash flow to project. The reported cash flows are after interest
                (cash flows to equity), so that discounting them at the WACC and subtracting the debt counts the
                cost of debt twice. Options are: "Free Cash Flow to Firm" (see get_free_cash_flow_to_firm, the cash
                flow to debt and equity holders that the WACC discounts into an enterprise value), "Free Cash Flow"
                (as reported), "Operating Cash Flow" (as reported). Defaults to "Free Cash Flow to Firm".
            trailing (int | None, optional): Sum the cash flow over this many periods (e.g.
                4 quarters) instead of taking the latest one. The window counts calendar
                periods, so a period without a report makes it NaN. Defaults to None, which
                sums the last 4 quarters (a trailing twelve months) for quarterly data and
                takes the latest year otherwise, as the projection periods are years.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per ticker the Share Price, the Cash Flow used, the Weighted
            Average Cost of Capital, the Perpetual Growth Rate, the Implied Growth Rate and,
            for comparison, the Historical Growth Rate (the compound annual growth of the
            cash flow as used, so summed over `trailing` periods where set, from the first
            to the latest reported period) with the Historical Growth Period it covers, the
            first and latest period such as "2021-2025" (or "2024Q1-2025Q3" with quarterly
            data).

        Notes:
        - A negative cash flow, or a discount rate at or below the perpetual growth rate,
          has no implied growth rate and gives NaN.
        - The Historical Growth Rate counts four quarters as a year with quarterly data and
          is NaN when the first or latest cash flow is not positive. The years are those
          between the first and the latest cash flow on the calendar, and a trailing sum
          that includes a missing period is NaN rather than one over more periods. The
          Historical Growth Period is empty (NaN) where the Historical Growth Rate is.

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

        |      |   Share Price |   Cash Flow |   Weighted Average Cost of Capital |   Perpetual Growth Rate |   Implied Growth Rate |   Historical Growth Rate | Historical Growth Period   |
        |:-----|--------------:|------------:|-----------------------------------:|------------------------:|----------------------:|-------------------------:|:---------------------------|
        | AAPL |        271.86 | 8.62639e+10 |                               0.09 |                   0.025 |                0.3068 |                   0.0545 | 2020-2025                  |
        | MSFT |        483.62 | 7.0121e+10  |                               0.09 |                   0.025 |                0.328  |                   0.1183 | 2020-2025                  |
        | NVDA |        186.5  | 5.98933e+10 |                               0.09 |                   0.025 |                0.4431 |                   1.0031 | 2020-2024                  |
        """
        validate_window("periods", periods, 1)

        perpetual = helpers.rates_per_ticker(
            perpetual_growth_rate, self._tickers, "perpetual growth rate"
        )
        # The projection periods and the growth rates are yearly, so a quarterly cash flow is
        # summed into a year rather than projected as if it were one.
        if trailing is None and self._quarterly:
            trailing = 4
        periods_per_year = 4 if self._quarterly else 1
        wacc = helpers.rates_per_ticker(
            weighted_average_cost_of_capital,
            self._tickers,
            "weighted average cost of capital",
            self._latest_weighted_average_cost_of_capital,
        )
        cash_flows = self._valuation_cash_flows(cash_flow_type, trailing)

        rows = {}
        without_price = []
        for ticker in self._tickers:
            # Summed on the calendar periods before the NaNs are dropped, so that a missing
            # period does not stretch a trailing sum over more periods than `trailing`.
            history = cash_flows.loc[ticker].dropna()
            if history.empty:
                continue
            cash_flow = float(history.iloc[-1])
            share_price = self._latest_price(ticker)
            if np.isnan(share_price):
                without_price.append(ticker)
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
                # The first and latest period the Historical Growth Rate is measured over.
                "Historical Growth Period": (
                    f"{history.index[0]}-{history.index[-1]}"
                    if not np.isnan(historical)
                    else np.nan
                ),
            }

        # A date range after the prices (or without them) still has the statements of the
        # year before it, so the result would otherwise show cash flows without a reason for
        # the missing implied growth rate.
        if without_price:
            logger.warning(
                "%s has no share price between %s and %s, so the implied growth rate is NaN and the "
                "cash flow is the latest one before the end date. Please choose a date range with prices.",
                ", ".join(without_price),
                self._start_date,
                self._end_date,
            )

        # Built per row so that the numbers keep a float dtype next to the period labels.
        return apply_rounding(
            pd.DataFrame.from_dict(rows, orient="index"),
            rounding if rounding is not None else self._rounding,
        )

    @handle_errors
    def get_earnings_power_value(
        self,
        weighted_average_cost_of_capital: float | dict[str, float] | None = None,
        periods: int = 5,
        trailing: int | None = None,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Earnings Power Value (EPV), the value of a company's current earnings if
        they never grew.

        Bruce Greenwald uses it to separate what is known (today's earnings power) from what is
        speculative (growth): the operating margin is averaged over several years and applied to
        the latest yearly revenue, taxed, and capitalized at the cost of capital.

            - Normalized Operating Income = Average Operating Margin * Latest Revenue
            - NOPAT = Normalized Operating Income * (1 - Average Tax Rate)
            - Earnings Power Value = NOPAT / WACC
            - EPV per Share = (EPV + Cash + Short Term Investments - Total Debt - Minority Interest -
              Preferred Stock) / Shares Outstanding

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
                the average. At least 1. Defaults to 5.
            trailing (int | None, optional): Sum the revenue, operating income, taxes and
                income before tax over this many periods (e.g. 4 quarters), so that the
                margin is applied to a year of revenue. The window counts calendar periods,
                so a period without a report makes it NaN. Defaults to None, which uses 4 (a
                trailing twelve months) for quarterly data and the yearly figures otherwise.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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
        | AAPL |                     0.3067 |                   1.27656e+11 |     0.1679 |        1.06229e+11 |                               0.08 |            1.32786e+12 |    1.27018e+12 |                          84.652  |        271.86 |         3.2115 |
        | KO   |                     0.2534 |                   1.21476e+10 |     0.1861 |        9.88662e+09 |                               0.08 |            1.23583e+11 |    8.81348e+10 |                          20.4347 |         69.91 |         3.4211 |
        | MSFT |                     0.4314 |                   1.2153e+11  |     0.1636 |        1.01652e+11 |                               0.08 |            1.27066e+12 |    1.25303e+12 |                         167.853  |        483.62 |         2.8812 |
        """
        validate_window("periods", periods, 1)

        # The NOPAT is capitalized as a yearly amount, so quarterly figures are summed into a
        # year (a trailing twelve months) before the margin is applied to the revenue.
        if trailing is None and self._quarterly:
            trailing = 4
        wacc = helpers.rates_per_ticker(
            weighted_average_cost_of_capital,
            self._tickers,
            "weighted average cost of capital",
            self._latest_weighted_average_cost_of_capital,
        )

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
                            # No ratio to an Earnings Power Value of 0 (or none per share).
                            "Price to EPV": (
                                share_price / result["Earnings Power Value per Share"]
                                if result["Earnings Power Value per Share"] != 0
                                else np.nan
                            ),
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
        growth_rate: float | dict[str, float] | None = None,
        growth_rate_volatility: float = 0.02,
        perpetual_growth_rate: float | dict[str, float] | None = None,
        perpetual_growth_rate_volatility: float = 0.005,
        weighted_average_cost_of_capital: float | dict[str, float] | None = None,
        weighted_average_cost_of_capital_volatility: float = 0.01,
        periods: int = 5,
        simulations: int = 10000,
        cash_flow_type: str = "Free Cash Flow to Firm",
        trailing: int | None = None,
        seed: int | None = 0,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the distribution of the intrinsic value per share when the inputs of the
        Discounted Cash Flow model are uncertain.

        A single DCF value hides how sensitive it is to its assumptions; here the growth rate,
        the perpetual growth rate and the discount rate are drawn from normal distributions
        around their expected values in every simulation, and the DCF of
        `get_intrinsic_valuation` is calculated for each draw. The spread of the outcomes, and
        the probability that the value exceeds the share price, show how robust the conclusion
        is.

        Draws in which the discount rate does not exceed the perpetual growth rate by at
        least half a percentage point are redrawn, as the terminal value is undefined there.

        For more information, see the following book:

        - Damodaran, A. (2012). "Investment Valuation: Tools and Techniques for Determining
          the Value of Any Asset." Wiley, 3rd edition, chapter 33 (probabilistic approaches).

        Also known as: Monte Carlo DCF, probabilistic valuation, DCF simulation.

        Args:
            growth_rate (float | dict[str, float] | None, optional): The expected yearly
                growth over the projection periods, one number or a number per ticker. Defaults
                to None, which uses the growth the company's reinvestment supports, as in
                `get_intrinsic_valuation`.
            growth_rate_volatility (float, optional): The standard deviation of the growth
                rate, 0 or more. Defaults to 0.02.
            perpetual_growth_rate (float | dict[str, float] | None, optional): The expected
                growth after the projection periods. Defaults to None, which uses the company's
                growth kept within 0 and the long-run growth rate of 2.5% (2.5% where it has none).
            perpetual_growth_rate_volatility (float, optional): Its standard deviation, 0 or
                more. Defaults to 0.005.
            weighted_average_cost_of_capital (float | dict[str, float] | None, optional):
                The expected discount rate. Defaults to None, which uses the latest Weighted
                Average Cost of Capital of each company, calculated over a year (the last 4
                quarters with quarterly data) whatever `trailing` the cash flows use, as the
                rate is a yearly one.
            weighted_average_cost_of_capital_volatility (float, optional): Its standard
                deviation, 0 or more. Defaults to 0.01.
            periods (int, optional): The number of projection periods, at least 1. Defaults to 5.
            simulations (int, optional): The number of simulations. Defaults to 10,000.
            cash_flow_type (str, optional): The cash flow to project. The reported cash flows are after interest
                (cash flows to equity), so that discounting them at the WACC and subtracting the debt counts the
                cost of debt twice. Options are: "Free Cash Flow to Firm" (see get_free_cash_flow_to_firm, the cash
                flow to debt and equity holders that the WACC discounts into an enterprise value), "Free Cash Flow"
                (as reported), "Operating Cash Flow" (as reported). Defaults to "Free Cash Flow to Firm".
            trailing (int | None, optional): Sum the cash flow over this many periods (e.g.
                4 quarters) instead of taking the latest one. The window counts calendar
                periods, so a period without a report makes it NaN. Defaults to None, which
                sums the last 4 quarters (a trailing twelve months) for quarterly data and
                takes the latest year otherwise, as the projection periods are years.
            seed (int | None, optional): The seed of the random numbers, so that results
                can be reproduced. Defaults to 0; None gives different draws every time.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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

        toolkit.models.get_monte_carlo_intrinsic_valuation()
        ```

        Which returns:

        |      |     Mean |   Standard Deviation |   5th Percentile |   25th Percentile |   50th Percentile |   75th Percentile |   95th Percentile |   Share Price |   Probability Undervalued |
        |:-----|---------:|---------------------:|-----------------:|------------------:|------------------:|------------------:|------------------:|--------------:|--------------------------:|
        | AAPL |  83.9988 |              16.3235 |          62.3819 |           72.7029 |           81.6781 |           92.6049 |           113.888 |        271.86 |                    0.0001 |
        | MSFT | 209.732  |              54.4736 |         146.029  |          174.188  |          199.786  |          233.56   |           305.64  |        483.62 |                    0.0026 |
        """
        validate_window("periods", periods, 1)
        for name, volatility in (
            ("growth_rate_volatility", growth_rate_volatility),
            ("perpetual_growth_rate_volatility", perpetual_growth_rate_volatility),
            (
                "weighted_average_cost_of_capital_volatility",
                weighted_average_cost_of_capital_volatility,
            ),
        ):
            if not volatility >= 0:
                raise InvalidArgumentError(
                    f"{name} is a standard deviation, which is 0 or more, not {volatility!r}. "
                    f"Please pass 0 or a positive number as {name}."
                )

        operating_growth = cache(lambda: self._latest(self._operating_growth()))
        growth = helpers.rates_per_ticker(
            growth_rate, self._tickers, "growth rate", operating_growth
        )
        perpetual = helpers.rates_per_ticker(
            perpetual_growth_rate,
            self._tickers,
            "perpetual growth rate",
            lambda: self._stable_growth(operating_growth()),
        )
        # The growth rates are yearly, so a quarterly cash flow is summed into a year.
        if trailing is None and self._quarterly:
            trailing = 4
        wacc = helpers.rates_per_ticker(
            weighted_average_cost_of_capital,
            self._tickers,
            "weighted average cost of capital",
            self._latest_weighted_average_cost_of_capital,
        )
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
                    # Without a share price or a simulated value there is nothing to compare.
                    if pd.notna(share_price) and pd.notna(values).any()
                    else float("nan")
                ),
            }

        return apply_rounding(
            pd.DataFrame(rows).T, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_peer_comparison(
        self,
        peers: dict[str, list[str]] | None = None,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Compare every company with its peers on valuation, profitability, growth and leverage.

        For each metric it shows the company's latest value, the median and mean of its peers
        and the company's percentile among them (the share of peers with a lower value), which
        tells at a glance whether a company is expensive or cheap, and better or worse run, than
        comparable companies.

        The metrics are the Price-to-Earnings, EV-to-EBITDA, Price-to-Book and
        Price-to-Sales ratios, the Gross, Operating and Net Profit Margin, the Return on
        Equity and on Invested Capital, Revenue Growth, the Debt-to-Equity ratio and the
        Dividend Yield, all of the latest period up to the Toolkit's end date: the fiscal year,
        or with quarterly data the quarter, with the ratios over the trailing year and the
        Revenue Growth over the same quarter a year earlier. The peers are retrieved with the
        Toolkit's settings (quarterly, point_in_time and convert_currency), so the Company
        column equals the Toolkit's own ratios of that period.

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
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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
        | ('MSFT', 'Price-to-Earnings')          |   36.4636 |       30.6062 |     29.3028 |          0.7 |      10 |
        | ('MSFT', 'EV-to-EBITDA')               |   22.9332 |       20.9841 |     19.0334 |          0.6 |      10 |
        | ('MSFT', 'Price-to-Book')              |   10.8105 |        7.3435 |      6.5553 |          0.7 |      10 |
        | ('MSFT', 'Price-to-Sales')             |   13.1802 |        6.2949 |      4.5172 |          1   |      10 |
        | ('MSFT', 'Gross Margin')               |    0.6882 |        0.6508 |      0.6211 |          0.5 |      10 |
        | ('MSFT', 'Operating Margin')           |    0.4562 |        0.1875 |      0.2187 |          1   |      10 |
        | ('MSFT', 'Net Profit Margin')          |    0.3615 |        0.1602 |      0.1816 |          1   |      10 |
        | ('MSFT', 'Return on Equity')           |    0.3328 |        0.2406 |      0.3172 |          0.6 |      10 |
        | ('MSFT', 'Return on Invested Capital') |    0.2576 |        0.1339 |      0.1691 |          0.8 |      10 |
        | ('MSFT', 'Revenue Growth')             |    0.1493 |        0.0962 |      0.1108 |          0.8 |      10 |
        | ('MSFT', 'Debt-to-Equity')             |    0.3266 |        0.22   |      0.9011 |          0.6 |      10 |
        | ('MSFT', 'Dividend Yield')             |    0.0065 |        0.008  |      0.009  |          0.5 |      10 |
        """
        if self._peer_function is None:
            raise InvalidArgumentError(
                "The peer data comes from the Toolkit. Please use this method from a Toolkit "
                "(toolkit.models), or pass peer_function when creating the Models class."
            )

        result = self._peer_function(peers)
        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

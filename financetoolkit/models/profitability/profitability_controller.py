"""Profitability Models Module"""

__docformat__ = "google"

from typing import Literal

import pandas as pd

from financetoolkit.helpers import check_trailing, trailing_mean, trailing_sum
from financetoolkit.models.profitability import dupont_model, growth_model
from financetoolkit.ratios import profitability_model, valuation_model
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import check_option, handle_errors
from financetoolkit.utilities.sources_model import track_sources
from financetoolkit.utilities.statistics_model import (
    apply_rounding,
    calculate_growth,
    calculate_standardization,
    drop_non_finite,
    finalize_dataset,
)

# ruff: noqa: E501


@check_trailing
@filter_shown_columns
@track_sources
class ProfitabilityModels:
    """
    Return decompositions and growth: the DuPont analyses, the Nissim-Penman decomposition, the return decomposition and the sustainable and internal growth rates.
    """

    @handle_errors
    def get_dupont_analysis(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
        method: Literal["standard", "extended"] = "standard",
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Perform a Dupont analysis to breakdown the return on equity (ROE) into its components.

        The Dupont analysis is a method used to dissect and understand the factors that drive a company's
        return on equity (ROE). It breaks down the ROE into three key components: Net Profit Margin, Asset
        Turnover, and Equity Multiplier. The extended (five-factor) Dupont analysis
        splits the Net Profit Margin further into three separate drivers — the Tax Burden, the Interest
        Burden and the Operating Profit Margin — so that the effect of taxation, of financing costs and of
        operating performance on the return on equity can be read separately.

        The formula is as follows:

        For the standard (three-factor) analysis:

            - Net Profit Margin = Net Income / Revenue
            - Asset Turnover = Revenue / Average Total Assets
            - Equity Multiplier = Average Total Assets / Average Total Shareholder Equity
            - ROE = Net Profit Margin * Asset Turnover * Equity Multiplier

        For the extended (five-factor) analysis:

            - Interest Burden Ratio = Income Before Tax / Operating Income
            - Tax Burden Ratio = Net Income / Income Before Tax
            - Operating Profit Margin = Operating Income / Revenue
            - Asset Turnover = Revenue / Average Total Assets
            - Equity Multiplier = Average Total Assets / Average Total Shareholder Equity
            - ROE = Interest Burden Ratio * Tax Burden Ratio * Operating Profit Margin *
            Asset Turnover * Equity Multiplier

        The value of the decomposition is in comparing the factors with peers and over time: a
        retailer earns its return on equity through asset turnover, a software company through its margin
        and a bank through leverage. A rising return on equity driven only by the Equity Multiplier, for
        example through share buybacks that shrink equity, is of lower quality than one driven by margin or
        turnover, and with negative equity the decomposition stops being meaningful.

        Also known as: DuPont, ROE decomposition, three-factor DuPont, extended DuPont, five-factor DuPont,
        ROE breakdown.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the values. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int | None, optional): The number of periods to sum flows (income statement items)
                over, with the Total Assets and Total Shareholder Equity averaged over the trailing + 1
                period ends that open and close those periods, so trailing=1 equals the default average
                of the opening and closing balance. E.g. 4 with quarterly data gives the trailing twelve
                months (TTM). The window counts calendar periods, so a period without a report makes it
                NaN. Defaults to None.
            method (str, optional): Options are: "standard" (the three-factor decomposition), "extended" (the
                five-factor decomposition). Defaults to "standard".
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: DataFrame containing Dupont analysis results, including Net Profit Margin, Asset
                          Turnover, Equity Multiplier, and the calculated ROE values. In
                          the extended analysis the Interest Burden Ratio, Tax Burden Ratio and Operating
                          Profit Margin take the place of the Net Profit Margin.

        Notes:
            - The Net Profit Margin is the ratio of Net Income to Total Revenue, indicating the percentage
            of revenue that translates into profit.
            - The Interest Burden Ratio (Income Before Tax / Operating Income) measures how much of
            operating profit survives the cost of debt financing. It equals 1 for a company with no
            net interest expense and falls as interest costs rise.
            - The Tax Burden Ratio (Net Income / Income Before Tax) measures how much of pre-tax
            profit survives taxation, i.e. it equals (1 - Effective Tax Rate).
            - The Operating Profit Margin measures operating performance before financing and tax
            effects.
            - Asset Turnover measures the efficiency of a company's use of its assets to generate sales
            revenue.
            - The Equity Multiplier measures financial leverage, the use of debt to finance a company's
            operations, which can amplify returns as well as risks.
            - The equity is the Total Shareholder Equity, which excludes minority interests, as the
            Net Income is the income attributable to the shareholders; this is the Return on Equity
            of the Ratios module.
            - Multiplying the Interest Burden Ratio, Tax Burden Ratio and Operating Profit Margin back
            together reproduces the Net Profit Margin, so both decompositions resolve to the same ROE.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["GOOGL"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_dupont_analysis()
        ```

        Which returns:

        |                   |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:------------------|-------:|-------:|-------:|-------:|-------:|
        | Net Profit Margin | 0.2951 | 0.212  | 0.2401 | 0.286  | 0.3281 |
        | Asset Turnover    | 0.759  | 0.7807 | 0.8009 | 0.821  | 0.7706 |
        | Equity Multiplier | 1.4317 | 1.4269 | 1.4228 | 1.4013 | 1.4122 |
        | Return on Equity  | 0.3207 | 0.2362 | 0.2736 | 0.3291 | 0.357  |
        """
        check_option("method", method, ("standard", "extended"))

        # The balances are averaged over the period ends that open and close the (trailing) periods,
        # as in the Return on Equity, Asset Turnover and Return on Assets of the Ratios module, so
        # that the components multiply to the Return on Equity that module reports.
        period_ends = (trailing or 1) + 1
        income = self._income_statement
        balance = self._balance_sheet_statement
        net_income = trailing_sum(income.loc[:, "Net Income", :], trailing)
        revenue = trailing_sum(income.loc[:, "Revenue", :], trailing)
        average_total_assets = trailing_mean(
            balance.loc[:, "Total Assets", :], period_ends
        )
        average_total_equity = trailing_mean(
            balance.loc[:, "Total Shareholder Equity", :], period_ends
        )

        if method == "extended":
            dupont_analysis = dupont_model.get_extended_dupont_analysis(
                operating_income=trailing_sum(
                    income.loc[:, "Operating Income", :], trailing
                ),
                income_before_tax=trailing_sum(
                    income.loc[:, "Income Before Tax", :], trailing
                ),
                net_income=net_income,
                total_revenue=revenue,
                average_total_assets=average_total_assets,
                average_total_equity=average_total_equity,
            )
        else:
            dupont_analysis = dupont_model.get_dupont_analysis(
                net_income, revenue, average_total_assets, average_total_equity
            )

        # A period without revenue (or equity) has no margin (or multiplier): NaN like any
        # other undefined ratio rather than infinite.
        dupont_analysis = drop_non_finite(dupont_analysis)

        if growth:
            dupont_analysis = calculate_growth(
                dataset=dupont_analysis,
                lag=lag,
                rounding=rounding if rounding is not None else self._rounding,
                axis="columns",
            )
        else:
            dupont_analysis = apply_rounding(
                dupont_analysis, rounding if rounding is not None else self._rounding
            )

        if standardize:
            dupont_analysis = calculate_standardization(
                dataset=dupont_analysis.loc[:, self._start_date : self._end_date],
                rounding=rounding if rounding is not None else self._rounding,
                axis="columns",
            )

        if len(self._tickers) == 1:
            dupont_analysis = dupont_analysis.droplevel(level=0)

        return dupont_analysis.loc[:, self._start_date : self._end_date]

    @handle_errors
    def get_nissim_penman_decomposition(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Perform the Nissim and Penman (2001) decomposition of the return on common equity (ROCE)
        into the return on the operations and the effect of how they are financed.

        Where the DuPont analyses split the return on equity into margins, turnover and an equity
        multiplier that mixes operating and financing liabilities, Nissim and Penman separate the
        operating activities from the financing activities. The return on common equity becomes
        the return on net operating assets (RNOA) plus a leverage effect: the financial leverage
        (FLEV) times the spread between the RNOA and the net borrowing cost (NBC). Borrowing
        raises the return on equity only while the operations earn more than the debt costs, and
        a company with net financial assets (negative FLEV) has a return on equity below its
        RNOA when its financial assets earn less than its operations.

        The formula is as follows:

            - Net Financial Expense = (Interest Expense - Interest Income) x (1 - Effective Tax Rate)
            - Operating Income = Net Income + Net Financial Expense
            - Net Financial Obligations = Total Debt + Preferred Stock - Cash and Short Term Investments -
              Long Term Investments
            - Common Equity = Total Shareholder Equity - Preferred Stock
            - Net Operating Assets = Net Financial Obligations + Common Equity
            - Return on Net Operating Assets (RNOA) = Operating Income / Average Net Operating Assets
            - Net Borrowing Cost (NBC) = Net Financial Expense / Average Net Financial Obligations
            - Spread = RNOA - NBC
            - Financial Leverage (FLEV) = Average Net Financial Obligations / Average Common Equity
            - Leverage Effect = FLEV x Spread
            - Return on Common Equity (ROCE) = RNOA + Leverage Effect

        The balance sheet is split into operating and financial items as follows:

            - Financial assets: the cash and cash equivalents, the short term investments and the
            long term investments. Nissim and Penman treat the marketable securities as financial;
            the long term investments of the statements are mostly those securities (Apple's long
            term marketable securities), but also hold equity method stakes (Coca-Cola's bottlers),
            which Nissim and Penman count as operating and which are financial here.
            - Financial obligations: the Total Debt, which holds the short and long term debt and
            the lease obligations (capital leases are debt in Nissim and Penman), and the preferred
            stock, which Nissim and Penman treat as a financial claim.
            - Common equity: the Total Shareholder Equity minus the preferred stock.
            - Operating: everything else, so the net operating assets are the net financial
            obligations plus the common equity (the operating assets minus the operating
            liabilities and the minority interest).

        Also known as: Nissim-Penman decomposition, ROCE decomposition, advanced DuPont analysis.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the values. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int | None, optional): The number of periods to sum the income statement items
                over, with the balance sheet items averaged over the trailing + 1 period ends that
                open and close those periods, as in the Ratios module. E.g. 4 with quarterly data
                gives the trailing twelve months (TTM). The window counts calendar periods, so a
                period without a report makes it NaN. Defaults to None, which uses single-period
                flows over the average of the opening and closing balance.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: DataFrame containing the Return on Net Operating Assets, the Net Borrowing
                Cost, the Spread, the Financial Leverage, the Leverage Effect and the Return on Common
                Equity per ticker and period.

        Notes:
            - Nissim, D. and S. H. Penman, "Ratio Analysis and Equity Valuation: From Research to
            Practice," Review of Accounting Studies 6, 2001, pp. 109-154.
            - The Return on Common Equity equals Net Income / Average Common Equity exactly, as the
            net operating assets are the net financial obligations plus the common equity and the
            operating income the net income plus the net financial expense. It differs from the
            Return on Equity of the Ratios module only by the preferred stock, which that ratio
            keeps in the equity.
            - Nissim and Penman add the minority interest to the net operating assets and its share
            of the income to the operating income, and multiply the decomposition by a minority
            sharing ratio. The statements give no minority share of the income, so the minority
            interest is deducted from the net operating assets instead: the operating income and
            the net operating assets are both the share of the parent's shareholders, and the
            decomposition adds up without a sharing ratio.
            - Nissim and Penman use the comprehensive income to common shareholders; the net
            income here leaves out the other comprehensive income (such as currency translation
            differences). The net financial expense is taxed at the effective tax rate (bounded to
            [0, 1], 0 for a pre-tax loss) instead of the marginal statutory rate, and holds no
            preferred dividends, which the statements do not report.
            - Missing data is treated in two ways. A short or long term investment, preferred
            stock or interest income that is not reported is absent and counts as zero (a company
            with debt but no interest income is normal). A period with neither an interest
            expense nor an interest income (both missing or zero, as in Apple's statements from
            fiscal 2024) has an unknown net financial expense rather than a zero one, so the
            Return on Net Operating Assets, the Net Borrowing Cost, the Spread, the Leverage
            Effect and the Return on Common Equity are NaN for that period (and every trailing
            window that holds it); the Financial Leverage remains.
            - The Return on Net Operating Assets, the Net Borrowing Cost and the Financial Leverage
            are also available as `Toolkit.ratios.get_return_on_net_operating_assets`,
            `get_net_borrowing_cost` and `get_financial_leverage`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["KO"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_nissim_penman_decomposition()
        ```

        Which returns:

        |                                |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-------------------------------|-------:|-------:|-------:|-------:|-------:|
        | Return on Net Operating Assets | 0.311  | 0.2795 | 0.3197 | 0.3076 | 0.337  |
        | Net Borrowing Cost             | 0.0765 | 0.0299 | 0.0508 | 0.0497 | 0.057  |
        | Spread                         | 0.2345 | 0.2496 | 0.2689 | 0.2579 | 0.28   |
        | Financial Leverage             | 0.644  | 0.5033 | 0.4032 | 0.4304 | 0.4383 |
        | Leverage Effect                | 0.151  | 0.1256 | 0.1085 | 0.111  | 0.1227 |
        | Return on Common Equity        | 0.462  | 0.4051 | 0.4282 | 0.4186 | 0.4597 |
        """
        income = self._income_statement
        balance = self._balance_sheet_statement
        # The short and long term investments, the preferred stock and the interest income count
        # as zero where they are not reported (they are absent); a period without any data stays
        # NaN through the Total Debt, the equity and the interest expense.
        preferred_stock = self._rows(balance, "Preferred Stock").fillna(0)
        net_financial_obligations = profitability_model.get_net_financial_obligations(
            self._rows(balance, "Total Debt"),
            preferred_stock,
            self._rows(balance, "Cash and Cash Equivalents").add(
                self._rows(balance, "Short Term Investments"), fill_value=0
            ),
            self._rows(balance, "Long Term Investments").fillna(0),
        )
        common_equity = (
            self._rows(balance, "Total Shareholder Equity") - preferred_stock
        )

        interest_income = self._rows(income, "Interest Income").fillna(0)
        interest_expense = self._rows(income, "Interest Expense")
        # Interest income alone may be absent, but without either interest row the net financial
        # expense is unknown, not zero: those periods are NaN before the trailing sums.
        interest_expense = interest_expense.where(
            (interest_expense.fillna(0) != 0) | (interest_income != 0)
        )

        net_financial_expense = profitability_model.get_net_financial_expense(
            trailing_sum(interest_expense, trailing),
            trailing_sum(interest_income, trailing),
            trailing_sum(self._rows(income, "Income Tax Expense"), trailing),
            trailing_sum(self._rows(income, "Income Before Tax"), trailing),
        )

        # The balances are averaged over the period ends that open and close the (trailing)
        # periods, as the Ratios module does, so the components equal those ratios.
        balance_periods = trailing + 1 if trailing else 2
        nissim_penman_decomposition = drop_non_finite(
            dupont_model.get_nissim_penman_decomposition(
                net_income=trailing_sum(self._rows(income, "Net Income"), trailing),
                net_financial_expense=net_financial_expense,
                average_net_financial_obligations=trailing_mean(
                    net_financial_obligations, balance_periods
                ),
                average_common_equity=trailing_mean(common_equity, balance_periods),
            )
        )

        if growth:
            nissim_penman_decomposition = calculate_growth(
                dataset=nissim_penman_decomposition,
                lag=lag,
                rounding=rounding if rounding is not None else self._rounding,
                axis="columns",
            )
        else:
            nissim_penman_decomposition = apply_rounding(
                nissim_penman_decomposition,
                rounding if rounding is not None else self._rounding,
            )

        if standardize:
            nissim_penman_decomposition = calculate_standardization(
                dataset=nissim_penman_decomposition.loc[
                    :, self._start_date : self._end_date
                ],
                rounding=rounding if rounding is not None else self._rounding,
                axis="columns",
            )

        if len(self._tickers) == 1:
            nissim_penman_decomposition = nissim_penman_decomposition.droplevel(level=0)

        return nissim_penman_decomposition.loc[:, self._start_date : self._end_date]

    @handle_errors
    def get_return_decomposition(
        self,
        lag: int | None = None,
        diluted: bool = True,
        trailing: int | None = None,
        logarithmic: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        The Return Decomposition splits the share price return over a window into the growth of
        the Earnings per Share and the change of the Price-to-Earnings ratio.

        The ratio is the multiple, and it adds the dividends paid over the window for the total
        return. It shows whether a share rose because the company earned more or because
        investors paid more for the same earnings, e.g. a share whose earnings beat expectations
        but whose price fell because its multiple contracted.

        Since the share price equals the Earnings per Share times the P/E ratio, the price
        relative over a window is the product of the earnings relative and the multiple
        relative, which is the earnings and valuation split of equity returns used in return
        attribution (e.g. Bogle, "Common Sense on Mutual Funds", and the building blocks of
        Grinold and Kroner, "The Equity Risk Premium", 2002).

        The formula is as follows:

            - 1 + Price Return = (1 + EPS Growth) * (1 + P/E Change)
            - EPS Growth = EPS_t / EPS_t-lag - 1
            - P/E Change = (Price_t / EPS_t) / (Price_t-lag / EPS_t-lag) - 1
            - Dividend Return = Dividends per Share paid over the window / Price_t-lag
            - Total Return = Price Return + Dividend Return

        In logarithmic form the parts are log returns that add up:
        ln(1 + Total Return) = ln(1 + EPS Growth) + ln(1 + P/E Change) + ln(1 + Dividends / Price_t).

        Also known as: earnings and multiple decomposition, P/E expansion and contraction,
        multiple expansion, return attribution by earnings growth and re-rating.

        Args:
            lag (int | None, optional): The number of periods the window spans. Defaults to None,
                which compares each period with the same period a year earlier: 1 with annual
                data and 4 with quarterly data.
            diluted (bool, optional): Whether to use the diluted Earnings per Share. Defaults to
                True.
            trailing (int | None, optional): The number of periods to sum the Net Income over
                (and average the shares over). The window counts calendar periods, so a period
                without a report makes it NaN. Defaults to None, which uses the trailing twelve
                months (4 quarters) with quarterly data and single years with annual data, as
                the P/E ratio does.
            logarithmic (bool, optional): Whether to return log returns, which add up, instead of
                simple returns, which multiply. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the rounding of the Toolkit (4 unless set otherwise).
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The Price Return, EPS Growth, P/E Change, Dividend Return and Total
            Return per ticker (rows) and period (columns), as decimals.

        Notes:
        - The share price is the traded Close (adjusted for splits, not for dividends) at the end
        of each fiscal period, and the Earnings per Share is the Net Income over the weighted
        average (diluted) shares, as in `ratios.get_earnings_per_share`.
        - The P/E ratio has no meaning when the earnings are zero or negative, so the EPS Growth
          and P/E Change are NaN when the Earnings per Share at either end of the window is not
          positive. The Price Return, Dividend Return and Total Return remain; the growth of the
          earnings themselves is the growth of `ratios.get_earnings_per_share`.
        - Dividends are added as paid, without reinvestment, so the Total Return differs slightly
        from a return on the dividend-adjusted price.
        - The price is the close on the last trading day of each fiscal period (the filing day
        with point_in_time), the date the earnings reported for it run up to, and the dividends
        are those paid between two such days. The calendar period's close and dividends are
        used where the daily prices do not reach those days.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AMD", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_return_decomposition()
        ```

        Which returns:

        |                             |    2022 |    2023 |    2024 |    2025 |
        |:----------------------------|--------:|--------:|--------:|--------:|
        | ('AMD', 'Price Return')     | -0.5568 |  1.2759 | -0.1507 |  0.7173 |
        | ('AMD', 'EPS Growth')       | -0.6734 | -0.3745 |  0.9075 |  1.6433 |
        | ('AMD', 'P/E Change')       |  0.3571 |  2.6387 | -0.5548 | -0.3503 |
        | ('AMD', 'Dividend Return')  |  0      |  0      |  0      |  0      |
        | ('AMD', 'Total Return')     | -0.5568 |  1.2759 | -0.1507 |  0.7173 |
        | ('MSFT', 'Price Return')    | -0.0519 |  0.3259 |  0.3125 |  0.1129 |
        | ('MSFT', 'EPS Growth')      |  0.1979 |  0.0039 |  0.2185 |  0.156  |
        | ('MSFT', 'P/E Change')      | -0.2085 |  0.3208 |  0.0771 | -0.0373 |
        | ('MSFT', 'Dividend Return') |  0.0089 |  0.0104 |  0.0086 |  0.0072 |
        | ('MSFT', 'Total Return')    | -0.043  |  0.3363 |  0.3211 |  0.1201 |
        """
        if trailing is None and self._quarterly:
            trailing = 4
        lag = lag if lag is not None else (4 if self._quarterly else 1)

        if not isinstance(lag, int) or lag < 1:
            raise ValueError(
                "The lag is the number of periods of the window: 1 or more."
            )

        shares = trailing_mean(
            self._rows(
                self._income_statement,
                (
                    "Weighted Average Shares Diluted"
                    if diluted
                    else "Weighted Average Shares"
                ),
            ),
            trailing,
        )
        net_income = trailing_sum(
            self._rows(self._income_statement, "Net Income"), trailing
        )

        earnings_per_share = valuation_model.get_earnings_per_share(
            net_income=net_income,
            preferred_dividends=0,
            average_outstanding_shares=shares,
        )

        columns = self._income_statement.columns
        dividends = self._historical_data["quarterly" if self._quarterly else "yearly"][
            "Dividends"
        ].reindex(columns=self._tickers)
        dividends.index = dividends.index.asfreq(columns.freq)
        dividends = dividends.reindex(columns).T
        # The dividends paid within each fiscal period, after the previous period's price
        # date up to and including its own, so the window between the prices holds exactly
        # the dividends paid in it; the calendar period's dividends where the daily data do
        # not cover both dates.
        paid_to_date = self._values_at_market_dates(
            self._historical_data["daily"]["Dividends"].fillna(0).cumsum()
        )
        dividends = (
            paid_to_date.diff(axis=1)
            .reindex(index=self._tickers, columns=columns)
            .combine_first(dividends)
        )

        decomposition = growth_model.get_return_decomposition(
            close_prices=self._period_prices(),
            earnings_per_share=earnings_per_share,
            dividends_per_share=dividends,
            lag=lag,
            logarithmic=logarithmic,
        )

        decomposition = apply_rounding(
            decomposition.loc[:, self._start_date : self._end_date],
            rounding if rounding is not None else self._rounding,
        ).dropna(how="all", axis=1)

        if len(self._tickers) == 1:
            return decomposition.droplevel(level=0)

        return decomposition

    @handle_errors
    def get_sustainable_growth_rate(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        The Sustainable Growth Rate (SGR) is the maximum rate at which a company can grow its
        revenue without new equity or more leverage.

        It uses internally generated funds only.

        The formula is as follows:

            - Return on Equity = Net Income / Total Shareholder Equity at the beginning of the period
            - Retention Ratio = 1 - Dividend Payout Ratio
            - SGR = Return on Equity * Retention Ratio

        The SGR assumes the company keeps its leverage and pays out only through dividends. Companies that
        return cash mainly through share buybacks have a low payout ratio on paper and a return on equity
        inflated by a shrinking equity base, so their SGR overstates what they can fund internally.

        Also known as: SGR, self-sustainable growth rate.

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
            pd.DataFrame: DataFrame containing the Sustainable Growth Rate.

        Notes:
        - Growing faster than the SGR without external financing typically requires either
        improving profitability, reducing the dividend payout, or increasing leverage.
        - Following Higgins ("Analysis for Financial Management"), the Return on Equity is measured
        on the equity at the beginning of the period (the previous period's closing balance, or that of
        `trailing` periods back with `trailing` set): the retained earnings then grow that equity by
        exactly the SGR. The two forms are equal under the model's assumption that equity grows only
        through retained earnings: with that assumption the SGR equals ROE * b / (1 - ROE * b) with the
        Return on Equity measured on the ending equity. Share issuance, buybacks and other changes in
        equity break the equality.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["NKE", "LULU", "DECK"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_sustainable_growth_rate()
        ```

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |
        |:-----|-------:|-------:|-------:|-------:|
        | DECK | 0.3129 | 0.3359 | 0.4302 | 0.4584 |
        | LULU | 0.3812 | 0.312  | 0.4923 | 0.4288 |
        | NKE  | 0.3297 | 0.2001 | 0.2521 | 0.0637 |
        """
        return_on_equity, dividend_payout_ratio = self._return_on_equity_and_payout(
            trailing
        )
        retention_ratio = valuation_model.get_retention_ratio(
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
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        The Internal Growth Rate (IGR) is the maximum rate at which a company can grow its
        revenue using only its retained earnings.

        It raises no external financing, neither debt nor equity.

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
            trailing (int | None, optional): The trailing period to use for the calculation. The window counts
                calendar periods, so a period without a report makes it NaN. Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: DataFrame containing the Internal Growth Rate.

        Notes:
        - The IGR is more conservative than the Sustainable Growth Rate (SGR) since it assumes
        no additional debt is raised to fund growth, whereas the SGR assumes the company
        maintains its current level of financial leverage.
        - The Return on Assets is measured on the Total Assets at the end of the period, as in
        Ross, Westerfield and Jaffe ("Corporate Finance"). Under the model's assumption that assets
        grow only through retained earnings, the formula equals the Return on Assets on the beginning
        Total Assets times the Retention Ratio, the growth the retained earnings add to the beginning
        assets, consistent with the beginning-equity definition of the Sustainable Growth Rate. Assets
        funded otherwise (new debt, equity or payables) break the equality.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["ADBE", "INTU", "ADSK"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_internal_growth_rate()
        ```

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |     2025 |
        |:-----|-------:|-------:|-------:|-------:|---------:|
        | ADBE | 0.2151 | 0.2122 | 0.2229 | 0.2254 |   0.3188 |
        | ADSK | 0.0613 | 0.0955 | 0.1006 | 0.1144 | nan      |
        | INTU | 0.1004 | 0.0489 | 0.0569 | 0.0639 |   0.0782 |
        """
        net_income = trailing_sum(
            self._income_statement.loc[:, "Net Income", :], trailing
        )
        # The ROA * b / (1 - ROA * b) form of the Internal Growth Rate measures ROA on the ending assets.
        ending_total_assets = self._balance_sheet_statement.loc[:, "Total Assets", :]
        dividends = trailing_sum(
            self._cash_flow_statement.loc[:, "Dividends Paid", :], trailing
        )

        return_on_assets = profitability_model.get_return_on_assets(
            net_income=net_income,
            average_total_assets=ending_total_assets,
        )
        dividend_payout_ratio = valuation_model.get_dividend_payout_ratio(
            dividends=dividends,
            net_income=net_income,
        )
        retention_ratio = valuation_model.get_retention_ratio(
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
    def get_fundamental_growth_rate(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate Damodaran's fundamental growth rate of operating income.

        It is the growth that a company's reinvestment produces at its return on capital, with
        the reinvestment rate and the sales-to-capital ratio it is built from. Where the
        sustainable growth rate grows the equity with the retained net income, this grows the
        operating income with the capital reinvested in the operations, the growth a discounted
        cash flow valuation of the firm assumes.

        The formula is as follows:

            - NOPAT = EBIT * (1 - Tax Rate)
            - Net Capital Expenditure = Capital Expenditure - Depreciation and Amortization
            - Reinvestment = Net Capital Expenditure + Change in Working Capital
            - Reinvestment Rate = Reinvestment / NOPAT
            - Invested Capital = Total Equity + Total Debt - Cash and Short Term Investments
            - Return on Invested Capital = NOPAT / Invested Capital at the beginning of the period
            - Fundamental Growth Rate = Reinvestment Rate * Return on Invested Capital
            - Sales to Capital = Revenue / Invested Capital at the beginning of the period

        The reinvestment is NOPAT minus the Free Cash Flow to Firm of `get_free_cash_flow_to_firm`,
        so a company that reinvests less than its depreciation has a negative reinvestment rate and
        a negative fundamental growth. The sales-to-capital ratio links revenue growth to the
        reinvestment it needs: Reinvestment = Change in Revenue / Sales to Capital.

        For more information, see the following sources:

        - Damodaran, A. (2012). "Investment Valuation." Wiley, 3rd edition, chapter 11 (Estimating
          Growth) and chapter 12 (Closure in Valuation).

        Also known as: Damodaran reinvestment rate, fundamental growth, expected growth in EBIT,
        reinvestment rate times return on capital, sales to capital ratio, sales to invested capital.

        Args:
            rounding (int, optional): The number of decimals to round the results to. Defaults to None.
            growth (bool, optional): Whether to calculate the growth of the values. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int | None, optional): The number of periods to sum the flows over, with the
                invested capital taken `trailing` periods back. The window counts calendar periods, so
                a period without a report makes it NaN. Defaults to None, which is 4 on quarterly data
                (a trailing twelve months against the capital a year earlier) and single years on
                yearly data.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per ticker the NOPAT, Net Capital Expenditure, Change in Working Capital,
            Reinvestment, Reinvestment Rate, Invested Capital (at the beginning of the period), Return
            on Invested Capital, Fundamental Growth Rate and Sales to Capital, rates as decimals.

        Notes:
        - The tax rate is the effective tax rate kept within 0 and 1 (0 without a pre-tax profit), as
        in `get_economic_value_added`, and EBIT the reported Operating Income.
        - The Capital Expenditure is the purchase of property, plant and equipment of the cash flow
        statement; Damodaran also counts acquisitions, which are lumpy and left out here. The Change
        in Working Capital is that of the cash flow statement (an increase is positive here), the
        non-cash working capital Damodaran uses.
        - The Invested Capital is Damodaran's: the book value of equity and debt net of cash, at the
        beginning of the period, as the return on capital measures the income on the capital already
        in place. A non-positive invested capital leaves the return and the ratio NaN, and a
        non-positive NOPAT the reinvestment rate.
        - This is a different measure from the retention ratio of the Ratios module
        (`get_retention_ratio`, 1 minus the dividend payout), the equity counterpart that the
        sustainable growth rate multiplies with the return on equity.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["MSFT", "GOOGL", "META"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_fundamental_growth_rate().loc["MSFT"]
        ```

        Which returns:

        |                                  |       2021 |         2022 |        2023 |         2024 |        2025 |
        |:---------------------------------|-----------:|-------------:|------------:|-------------:|------------:|
        | Net Operating Profit After Taxes | 6.0249e+10 |  7.24487e+10 | 7.17226e+10 |  8.94819e+10 | 1.05869e+11 |
        | Net Capital Expenditure          | 8.936e+09  |  9.426e+09   | 1.4246e+10  |  2.219e+10   | 3.0398e+10  |
        | Change in Working Capital        | 9.36e+08   | -4.46e+08    | 2.388e+09   | -1.824e+09   | 5.35e+09    |
        | Reinvestment                     | 9.872e+09  |  8.98e+09    | 1.6634e+10  |  2.0366e+10  | 3.5748e+10  |
        | Reinvestment Rate                | 0.1639     |  0.1239      | 0.2319      |  0.2276      | 0.3377      |
        | Invested Capital                 | 6.3887e+10 |  9.401e+10   | 1.40193e+11 |  1.74408e+11 | 2.90798e+11 |
        | Return on Invested Capital       | 0.9431     |  0.7706      | 0.5116      |  0.5131      | 0.3641      |
        | Fundamental Growth Rate          | 0.1545     |  0.0955      | 0.1187      |  0.1168      | 0.1229      |
        | Sales to Capital                 | 2.631      |  2.109       | 1.5116      |  1.4055      | 0.9688      |
        """
        trailing, periods_per_year = self._annual_trailing(trailing)
        span = trailing or periods_per_year

        def flow(statement: pd.DataFrame, row: str) -> pd.DataFrame:
            return trailing_sum(self._rows(statement, row), trailing)

        balance = self._balance_sheet_statement
        components = {}
        components["Net Operating Profit After Taxes"] = (
            self._net_operating_profit_after_taxes(trailing).reindex(self._tickers)
        )
        components["Net Capital Expenditure"] = -flow(
            self._cash_flow_statement, "Capital Expenditure"
        ) - flow(self._income_statement, "Depreciation and Amortization")
        components["Change in Working Capital"] = -flow(
            self._cash_flow_statement, "Change in Working Capital"
        )
        components["Reinvestment"] = (
            components["Net Capital Expenditure"]
            + components["Change in Working Capital"]
        )
        components["Reinvestment Rate"] = growth_model.get_reinvestment_rate(
            net_capital_expenditure=components["Net Capital Expenditure"],
            change_in_working_capital=components["Change in Working Capital"],
            net_operating_profit_after_taxes=components[
                "Net Operating Profit After Taxes"
            ],
        )
        # Damodaran's invested capital (book equity and debt net of cash) at the beginning of the period.  # noqa: E501
        invested_capital = (
            self._rows(balance, "Total Equity")
            + self._rows(balance, "Total Debt")
            - self._rows(balance, "Cash and Short Term Investments")
        ).shift(span, axis=1)
        components["Invested Capital"] = invested_capital
        components["Return on Invested Capital"] = components[
            "Net Operating Profit After Taxes"
        ] / invested_capital.where(invested_capital > 0)
        components["Fundamental Growth Rate"] = (
            growth_model.get_fundamental_growth_rate(
                reinvestment_rate=components["Reinvestment Rate"],
                return_on_invested_capital=components["Return on Invested Capital"],
            )
        )
        components["Sales to Capital"] = growth_model.get_sales_to_capital_ratio(
            revenue=flow(self._income_statement, "Revenue"),
            invested_capital=invested_capital,
        )

        return self._finalize_components(components, rounding, growth, lag, standardize)

    @handle_errors
    def get_rule_of_40(
        self,
        margin: str = "free_cash_flow",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Rule of 40 score: the revenue growth plus the profit margin, the measure
        of software and other subscription companies that trade growth against profitability.

        A score of 40% (0.40) or more is seen as healthy: a company growing 30% a year should
        keep a margin of at least 10%, one that grows 10% a margin of 30%.

        The formula is as follows:

            - Revenue Growth = Revenue / Revenue a year earlier - 1
            - Free Cash Flow Margin = Free Cash Flow / Revenue (or EBITDA Margin = EBITDA / Revenue)
            - Rule of 40 Score = Revenue Growth + Free Cash Flow Margin (or EBITDA Margin)

        The rule comes from venture capital practice for software companies (attributed to Brad
        Feld, 2015) and has no academic source; investors use the free cash flow margin and the
        EBITDA margin, which the `margin` parameter chooses between.

        For more information, see the following sources:

        - Feld, B. (2015). "The Rule of 40% For a Healthy SaaS Company." Feld Thoughts
          (https://feld.com/archives/2015/02/rule-40-healthy-saas-company/).

        Also known as: Rule of 40, 40% rule, growth plus margin, SaaS rule of 40.

        Args:
            margin (str, optional): The margin to add. Options are: "free_cash_flow" (Free Cash Flow / Revenue),
                "ebitda" (EBITDA / Revenue). Defaults to "free_cash_flow".
            rounding (int, optional): The number of decimals to round the results to. Defaults to None.
            growth (bool, optional): Whether to calculate the growth of the values. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int | None, optional): The number of periods to sum the revenue and cash flows
                over. The window counts calendar periods, so a period without a report makes it NaN.
                Defaults to None, which is 4 on quarterly data (a trailing twelve months, its growth
                against the trailing twelve months a year earlier) and single years on yearly data.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per ticker the Revenue Growth, the margin (Free Cash Flow Margin or EBITDA
            Margin) and the Rule of 40 Score, as decimals.

        Notes:
        - The revenue growth compares with the same period a year earlier (four quarters back on
        quarterly data), so the first year has no score. Growth from a negative or zero revenue is
        NaN.
        - The Free Cash Flow is that of the cash flow statement (Operating Cash Flow minus Capital
        Expenditure).

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["CRM", "NOW", "ADBE"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_rule_of_40().xs("Rule of 40 Score", level=1)
        ```

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |     2025 |
        |:-----|-------:|-------:|-------:|-------:|---------:|
        | ADBE | 0.6627 | 0.5354 | 0.4601 | 0.4741 |   0.5198 |
        | CRM  | 0.446  | 0.3848 | 0.3843 | 0.4153 | nan      |
        | NOW  | 0.6097 | 0.5287 | 0.5396 | 0.5353 |   0.5535 |
        """
        margin = margin.lower()
        check_option("margin", margin, ("free_cash_flow", "ebitda"))
        trailing, periods_per_year = self._annual_trailing(trailing)

        revenue = trailing_sum(self._rows(self._income_statement, "Revenue"), trailing)
        previous_revenue = revenue.shift(periods_per_year, axis=1)
        if margin == "free_cash_flow":
            name = "Free Cash Flow Margin"
            profit = trailing_sum(
                self._rows(self._cash_flow_statement, "Free Cash Flow"), trailing
            )
        else:
            name = "EBITDA Margin"
            profit = trailing_sum(
                self._rows(self._income_statement, "EBITDA"), trailing
            )

        components = {}
        components["Revenue Growth"] = (
            revenue / previous_revenue.where(previous_revenue > 0) - 1
        )
        components[name] = profit / revenue.where(revenue > 0)
        components["Rule of 40 Score"] = growth_model.get_rule_of_40(
            revenue_growth=components["Revenue Growth"],
            profit_margin=components[name],
        )

        return self._finalize_components(components, rounding, growth, lag, standardize)

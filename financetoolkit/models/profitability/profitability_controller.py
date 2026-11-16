"""Profitability Models Module"""

__docformat__ = "google"

import pandas as pd

from financetoolkit.models.base.base_controller import ModelsBase
from financetoolkit.models.profitability import dupont_model, growth_model
from financetoolkit.ratios import profitability_model, valuation_model
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import handle_errors
from financetoolkit.utilities.sources_model import track_sources
from financetoolkit.utilities.statistics_model import (
    apply_rounding,
    calculate_growth,
    calculate_standardization,
    finalize_dataset,
)

# ruff: noqa: E501


@filter_shown_columns
@track_sources
class ProfitabilityModels(ModelsBase):
    """
    Return decompositions and growth: the DuPont analyses, the return decomposition and the sustainable and internal growth rates.
    """

    @handle_errors
    def get_dupont_analysis(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
        show_columns: str | list[str] | None = None,
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
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the values. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int | None, optional): The trailing period to use for the calculation. Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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
        if trailing:
            dupont_analysis = dupont_model.get_dupont_analysis(
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
            dupont_analysis = dupont_model.get_dupont_analysis(
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
    def get_extended_dupont_analysis(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        trailing: int | None = None,
        show_columns: str | list[str] | None = None,
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
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the values. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int | None, optional): The trailing period to use for the calculation. Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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
            ["MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.models.get_extended_dupont_analysis()
        ```

        Which returns:

        |                         |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:------------------------|-------:|-------:|-------:|-------:|-------:|
        | Interest Burden Ratio   | 1.017  | 1.004  | 1.0089 | 0.985  | 0.9619 |
        | Tax Burden Ratio        | 0.8617 | 0.8689 | 0.8102 | 0.8177 | 0.8237 |
        | Operating Profit Margin | 0.4159 | 0.4206 | 0.4177 | 0.4464 | 0.4562 |
        | Asset Turnover          | 0.5293 | 0.5676 | 0.5456 | 0.5305 | 0.4981 |
        | Equity Multiplier       | 2.4399 | 2.2643 | 2.0839 | 1.9468 | 1.8484 |
        | Return on Equity        | 0.4708 | 0.4715 | 0.3882 | 0.3713 | 0.3328 |
        """
        if trailing:
            extended_dupont_analysis = dupont_model.get_extended_dupont_analysis(
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
            extended_dupont_analysis = dupont_model.get_extended_dupont_analysis(
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
            extended_dupont_analysis = calculate_growth(
                dataset=extended_dupont_analysis,
                lag=lag,
                rounding=rounding if rounding is not None else self._rounding,
                axis="columns",
            )
        else:
            extended_dupont_analysis = apply_rounding(
                extended_dupont_analysis,
                rounding if rounding is not None else self._rounding,
            )

        if standardize:
            extended_dupont_analysis = calculate_standardization(
                dataset=extended_dupont_analysis.loc[
                    :, self._start_date : self._end_date
                ],
                rounding=rounding if rounding is not None else self._rounding,
                axis="columns",
            )

        if len(self._tickers) == 1:
            extended_dupont_analysis = extended_dupont_analysis.droplevel(level=0)

        return extended_dupont_analysis.loc[:, self._start_date : self._end_date]

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
        the Earnings per Share and the change of the Price-to-Earnings ratio (the multiple), and
        adds the dividends paid over the window for the total return. It shows whether a share
        rose because the company earned more or because investors paid more for the same
        earnings, e.g. a share whose earnings beat expectations but whose price fell because its
        multiple contracted.

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

        With logarithmic=True the parts are log returns that add up:
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
                (and average the shares over). Defaults to None, which uses the trailing twelve
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
        of each period, the same price the P/E ratio uses, and the Earnings per Share is the Net
        Income over the weighted average (diluted) shares, as in `ratios.get_earnings_per_share`.
        - The P/E ratio has no meaning when the earnings are zero or negative, so the EPS Growth
        and P/E Change are NaN when the Earnings per Share at either end of the window is not
        positive. The Price Return, Dividend Return and Total Return remain; the growth of the
        earnings themselves is `ratios.get_earnings_per_share(growth=True)`.
        - Dividends are added as paid, without reinvestment, so the Total Return differs slightly
        from a return on the dividend-adjusted price.
        - Each period is a calendar period that holds the fiscal period reported in it (see the
        Toolkit), so the price at the end of the calendar period is compared with the earnings
        reported for it, as in the P/E ratio.

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
        | ('AMD', 'Price Return')     | -0.5499 |  1.2759 | -0.1806 |  0.773  |
        | ('AMD', 'EPS Growth')       | -0.6734 | -0.3745 |  0.9075 |  1.6433 |
        | ('AMD', 'P/E Change')       |  0.3782 |  2.6387 | -0.5704 | -0.3292 |
        | ('AMD', 'Dividend Return')  |  0      |  0      |  0      |  0      |
        | ('AMD', 'Total Return')     | -0.5499 |  1.2759 | -0.1806 |  0.773  |
        | ('MSFT', 'Price Return')    | -0.2869 |  0.568  |  0.1209 |  0.1474 |
        | ('MSFT', 'EPS Growth')      |  0.1979 |  0.0039 |  0.2185 |  0.156  |
        | ('MSFT', 'P/E Change')      | -0.4047 |  0.562  | -0.0801 | -0.0075 |
        | ('MSFT', 'Dividend Return') |  0.0076 |  0.0116 |  0.0082 |  0.0081 |
        | ('MSFT', 'Total Return')    | -0.2794 |  0.5796 |  0.1291 |  0.1554 |
        """
        trailing = (
            trailing if trailing is not None else (4 if self._quarterly else None)
        )
        lag = lag if lag is not None else (4 if self._quarterly else 1)

        if not isinstance(lag, int) or lag < 1:
            raise ValueError(
                "The lag is the number of periods of the window: 1 or more."
            )

        shares = self._rows(
            self._income_statement,
            "Weighted Average Shares Diluted" if diluted else "Weighted Average Shares",
        )
        net_income = self._rows(self._income_statement, "Net Income")

        if trailing:
            net_income = net_income.T.rolling(trailing).sum().T
            shares = shares.T.rolling(trailing).mean().T

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
        revenue, using internally generated funds only, without having to raise additional equity
        or increase its financial leverage.

        The formula is as follows:

            - Return on Equity = Net Income / Total Equity at the beginning of the period
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
            trailing (int | None, optional): The trailing period to use for the calculation. Defaults to None.
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
        show_columns: str | list[str] | None = None,
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

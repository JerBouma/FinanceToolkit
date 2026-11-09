"""Ratios Module"""

__docformat__ = "google"


from collections.abc import Callable
from functools import cached_property

import numpy as np
import pandas as pd

from financetoolkit.cache import policy_model, ticker_model
from financetoolkit.cache.cache_controller import get_active_cache
from financetoolkit.fmp_model import get_analyst_estimates as _get_analyst_estimates
from financetoolkit.helpers import get_goodwill_and_intangible_assets, handle_portfolio
from financetoolkit.ratios import (
    efficiency_model,
    liquidity_model,
    profitability_model,
    solvency_model,
    valuation_model,
)
from financetoolkit.ratios.helpers import map_period_data_to_daily_data
from financetoolkit.utilities import factor_model, logger_model
from financetoolkit.utilities.error_model import handle_errors
from financetoolkit.utilities.statistics_model import (
    apply_rounding,
    calculate_growth,
    calculate_standardization,
    drop_non_finite,
    finalize_dataset,
)

logger = logger_model.get_logger()

# The ratios per collection that get_factor_analysis compares across tickers. Per-share figures
# and currency amounts (earnings per share, enterprise value, working capital) are left out:
# their level reflects the share count or the size of the company, not its valuation or
# quality, so they only rank tickers meaningfully as a size factor, which the market cap is.
FACTOR_COLLECTIONS = {
    "valuation": [
        "get_price_to_earnings_ratio",
        "get_price_to_earnings_growth_ratio",
        "get_price_to_book_ratio",
        "get_price_to_sales_ratio",
        "get_price_to_cash_flow_ratio",
        "get_price_to_free_cash_flow_ratio",
        "get_earnings_yield",
        "get_dividend_yield",
        "get_dividend_payout_ratio",
        "get_buyback_yield",
        "get_shareholder_yield",
        "get_net_share_issuance",
        "get_composite_equity_issuance",
        "get_net_external_financing",
        "get_ev_to_sales_ratio",
        "get_ev_to_ebit",
        "get_ev_to_ebitda_ratio",
        "get_ev_to_operating_cashflow_ratio",
        "get_ev_to_free_cash_flow_ratio",
        "get_market_cap",
        "get_free_cash_flow_payout_ratio",
        "get_ev_to_invested_capital_ratio",
        "get_price_to_tangible_book_ratio",
    ],
    "profitability": [
        "get_gross_margin",
        "get_operating_margin",
        "get_net_profit_margin",
        "get_ebitda_margin",
        "get_free_cash_flow_margin",
        "get_interest_coverage_ratio",
        "get_income_before_tax_profit_margin",
        "get_effective_tax_rate",
        "get_return_on_assets",
        "get_cash_return_on_assets",
        "get_return_on_equity",
        "get_return_on_invested_capital",
        "get_return_on_capital_employed",
        "get_return_on_tangible_assets",
        "get_income_quality_ratio",
        "get_net_income_per_ebt",
        "get_free_cash_flow_operating_cash_flow_ratio",
        "get_EBT_to_EBIT",
        "get_EBIT_to_revenue",
        "get_cash_tax_rate",
        "get_tax_rate_divergence",
        "get_gross_profitability",
        "get_operating_profitability",
        "get_cash_based_operating_profitability",
        "get_cash_return_on_invested_capital",
        "get_free_cash_flow_conversion",
        "get_free_cash_flow_to_ebitda_ratio",
        "get_percent_accruals",
    ],
    "solvency": [
        "get_debt_to_assets_ratio",
        "get_asset_coverage_ratio",
        "get_debt_to_equity_ratio",
        "get_debt_service_coverage_ratio",
        "get_equity_multiplier",
        "get_free_cash_flow_yield",
        "get_net_debt_to_ebitda_ratio",
        "get_gross_debt_to_ebitda_ratio",
        "get_cash_flow_coverage_ratio",
        "get_capex_coverage_ratio",
        "get_capex_dividend_coverage_ratio",
        "get_debt_to_capital_ratio",
        "get_preferred_dividend_coverage_ratio",
        "get_interest_paid_to_expense_ratio",
        "get_ebitda_interest_coverage_ratio",
        "get_ebitda_minus_capex_interest_coverage_ratio",
        "get_cash_interest_coverage_ratio",
        "get_funds_from_operations_to_debt_ratio",
        "get_discretionary_cash_flow_to_debt_ratio",
        "get_degree_of_operating_leverage",
        "get_degree_of_financial_leverage",
        "get_degree_of_total_leverage",
    ],
    "liquidity": [
        "get_current_ratio",
        "get_quick_ratio",
        "get_cash_ratio",
        "get_operating_cash_flow_ratio",
        "get_operating_cash_flow_sales_ratio",
        "get_short_term_coverage_ratio",
        "get_defensive_interval_ratio",
    ],
    "efficiency": [
        "get_days_of_inventory_outstanding",
        "get_days_of_sales_outstanding",
        "get_operating_cycle",
        "get_days_of_accounts_payable_outstanding",
        "get_cash_conversion_cycle",
        "get_cash_conversion_efficiency",
        "get_receivables_turnover",
        "get_inventory_turnover_ratio",
        "get_accounts_payables_turnover_ratio",
        "get_sga_to_revenue_ratio",
        "get_fixed_asset_turnover",
        "get_asset_turnover_ratio",
        "get_operating_ratio",
        "get_research_and_development_ratio",
        "get_selling_and_marketing_ratio",
        "get_general_and_administrative_ratio",
        "get_stock_based_compensation_ratio",
        "get_deferred_revenue_ratio",
        "get_working_capital_turnover_ratio",
        "get_capex_intensity",
        "get_capex_to_depreciation_ratio",
        "get_asset_growth",
        "get_net_operating_assets",
    ],
}
FACTOR_COLLECTIONS["all"] = [
    metric for metrics in FACTOR_COLLECTIONS.values() for metric in metrics
]

# The examples' output tables are wider than the line length.
# ruff: noqa: E501


class Ratios:
    """
    The Ratios Module contains over 50+ ratios that can be used to analyse companies.
    These ratios are divided into 5 categories which are efficiency, liquidity,
    profitability, solvency and valuation. Each ratio is calculated using
    the data from the Toolkit module.
    """

    def __init__(
        self,
        tickers: list[str] | Callable[[], list[str]],
        historical: dict[str, pd.DataFrame],
        balance: pd.DataFrame | Callable[[], pd.DataFrame],
        income: pd.DataFrame | Callable[[], pd.DataFrame],
        cash: pd.DataFrame | Callable[[], pd.DataFrame],
        quarterly: bool = False,
        rounding: int | None = 4,
        start_date: str | None = None,
        end_date: str | None = None,
        api_key: str = "",
        sleep_timer: bool = False,
        user_subscription: str = "Free",
        analyst_estimates_cache: dict | None = None,
        period_end_dates: pd.DataFrame | Callable[[], pd.DataFrame] | None = None,
    ):
        """
        Initializes the Ratios Controller Class.

        Each dataset is either the data itself or a function without arguments that retrieves
        it, which is called when a ratio first needs that dataset. The Toolkit passes functions,
        so a ratio retrieves only what it uses: the gross margin only the income statement, a
        valuation ratio also the prices.

        Args:
            tickers (list[str] | Callable[[], list[str]]): The tickers to use for the calculations.
            historical (dict[str, pd.DataFrame]): The historical data to use for the calculations,
                with a "period" and a "daily" key. A LazyDict (financetoolkit.helpers) retrieves
                each when it is first read.
            balance (pd.DataFrame | Callable[[], pd.DataFrame]): The balance sheet data to use for
                the calculations.
            income (pd.DataFrame | Callable[[], pd.DataFrame]): The income statement data to use
                for the calculations.
            cash (pd.DataFrame | Callable[[], pd.DataFrame]): The cash flow statement data to use
                for the calculations.
            quarterly (bool, optional): Whether to use quarterly data. Defaults to False.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            api_key (str, optional): FinancialModelingPrep API key, used only by
                `get_forward_price_earnings_ratio` and `get_forward_price_earnings_growth_ratio` to
                lazily fetch (and cache) analyst estimates the first time either method is called —
                not on initialization or on any other ratio. Defaults to "", which means those two
                methods will report that the data is unavailable (requires a Premium FMP
                subscription).
            sleep_timer (bool, optional): Whether to set a sleep timer when the rate limit is
                reached while fetching analyst estimates. Only works with a Premium subscription.
                Defaults to False.
            user_subscription (str, optional): The FMP subscription plan, used only for the analyst
                estimates fetch. Defaults to "Free".
            analyst_estimates_cache (dict, optional): A mutable dict used to cache the fetched
                analyst estimates. The Toolkit passes the same dict instance in every time this
                class is (re)constructed (each access to `toolkit.ratios` builds a fresh Ratios
                instance), so this is what makes the fetch-once behaviour survive across separate
                `toolkit.ratios.<method>()` calls rather than resetting every time. Defaults to
                None, which uses a private dict scoped to this instance only.
            period_end_dates (pd.DataFrame | Callable[[], pd.DataFrame], optional): The last day
                of the fiscal period behind each statement column, with the tickers as index and
                the periods as columns, as the Toolkit determines it (the filing days with
                point_in_time). Daily ratios (show_daily=True) apply each period's figures from that
                day on. Defaults to None, which applies them from the end of the calendar period the
                column is labelled with.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "TSLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        profitability_ratios = toolkit.ratios.collect_profitability_ratios()

        profitability_ratios.loc['AAPL']
        ```

        Which returns:

        |                                             |    2021 |    2022 |    2023 |     2024 |     2025 |
        |:--------------------------------------------|--------:|--------:|--------:|---------:|---------:|
        | Gross Margin                                |  0.4178 |  0.4331 |  0.4413 |   0.4621 |   0.4691 |
        | Operating Margin                            |  0.2978 |  0.3029 |  0.2982 |   0.3151 |   0.3197 |
        | Net Profit Margin                           |  0.2588 |  0.2531 |  0.2531 |   0.2397 |   0.2692 |
        | EBITDA Margin                               |  0.3287 |  0.331  |  0.3283 |   0.3444 |   0.3478 |
        | Free Cash Flow Margin                       |  0.2541 |  0.2826 |  0.2598 |   0.2783 |   0.2373 |
        | Interest Coverage Ratio                     | 41.1905 | 40.7496 | 29.062  | nan      | nan      |
        | Income Before Tax Profit Margin             |  0.2985 |  0.302  |  0.2967 |   0.3158 |   0.3189 |
        | Effective Tax Rate                          |  0.133  |  0.162  |  0.1472 |   0.2409 |   0.1561 |
        | Return on Assets                            |  0.2806 |  0.2836 |  0.275  |   0.2613 |   0.3093 |
        | Cash Return on Assets                       |  0.3083 |  0.3471 |  0.3134 |   0.3296 |   0.3079 |
        | Return on Equity                            |  1.4744 |  1.7546 |  1.7195 |   1.5741 |   1.7142 |
        | Return on Invested Capital                  |  0.4879 |  0.5229 |  0.528  |   0.5166 |   0.6201 |
        | Return on Capital Employed                  |  0.4831 |  0.6009 |  0.5514 |   0.6534 |   0.6872 |
        | Return on Tangible Assets                   |  0.2806 |  0.2836 |  0.275  |   0.2613 |   0.3093 |
        | Income Quality Ratio                        |  1.0988 |  1.2239 |  1.1397 |   1.2616 |   0.9953 |
        | Net Income per EBT                          |  0.867  |  0.838  |  0.8528 |   0.7591 |   0.8439 |
        | Free Cash Flow to Operating Cash Flow Ratio |  0.8935 |  0.9123 |  0.9009 |   0.9201 |   0.8859 |
        | EBT to EBIT Ratio                           |  0.9764 |  0.976  |  0.9666 |   1      |   1      |
        | EBIT to Revenue                             |  0.3058 |  0.3095 |  0.307  |   0.3158 |   0.3189 |
        | Cash Tax Rate                               |  0.2324 |  0.1643 |  0.1642 |   0.2114 |   0.3267 |
        | Tax Rate Divergence                         |  0.0994 |  0.0023 |  0.017  |  -0.0295 |   0.1706 |
        """
        self._tickers_source = tickers
        self._balance_source = balance
        self._income_source = income
        self._cash_source = cash
        self._period_end_dates_source = period_end_dates
        self._historical = historical
        self._api_key: str = api_key
        self._sleep_timer: bool = sleep_timer
        self._user_subscription: str = user_subscription
        self._analyst_estimates_cache: dict = (
            analyst_estimates_cache if analyst_estimates_cache is not None else {}
        )
        self._available_custom_ratios_options: list[str] = []
        self._custom_ratios: pd.DataFrame = pd.DataFrame()
        self._custom_ratios_growth: pd.DataFrame = pd.DataFrame()
        self._rounding: int | None = rounding
        self._quarterly: bool = quarterly
        self._portfolio_weights: dict | None = None
        self._start_date: str | None = start_date
        self._end_date: str | None = end_date

        # Initialization of Fundamentals Variables
        self._all_ratios: pd.DataFrame = pd.DataFrame()
        self._all_ratios_settings: tuple | None = None
        self._all_ratios_growth: pd.DataFrame = pd.DataFrame()
        self._efficiency_ratios: pd.DataFrame = pd.DataFrame()
        self._efficiency_ratios_growth: pd.DataFrame = pd.DataFrame()
        self._liquidity_ratios: pd.DataFrame = pd.DataFrame()
        self._liquidity_ratios_growth: pd.DataFrame = pd.DataFrame()
        self._profitability_ratios: pd.DataFrame = pd.DataFrame()
        self._profitability_ratios_growth: pd.DataFrame = pd.DataFrame()
        self._solvency_ratios: pd.DataFrame = pd.DataFrame()
        self._solvency_ratios_growth: pd.DataFrame = pd.DataFrame()
        self._valuation_ratios: pd.DataFrame = pd.DataFrame()
        self._valuation_ratios_growth: pd.DataFrame = pd.DataFrame()

    @cached_property
    def _tickers(self) -> list[str]:
        """The tickers, retrieved when first needed."""
        return (
            self._tickers_source()
            if callable(self._tickers_source)
            else self._tickers_source
        )

    @cached_property
    def _tickers_without_portfolio(self) -> list[str]:
        """The tickers other than the Portfolio."""
        return [ticker for ticker in self._tickers if ticker != "Portfolio"]

    @cached_property
    def _balance_sheet_statement(self) -> pd.DataFrame:
        """The balance sheet statement, retrieved when a ratio first needs it."""
        return (
            self._balance_source()
            if callable(self._balance_source)
            else self._balance_source
        )

    @cached_property
    def _income_statement(self) -> pd.DataFrame:
        """The income statement, retrieved when a ratio first needs it."""
        return (
            self._income_source()
            if callable(self._income_source)
            else self._income_source
        )

    @cached_property
    def _cash_flow_statement(self) -> pd.DataFrame:
        """The cash flow statement, retrieved when a ratio first needs it."""
        return self._cash_source() if callable(self._cash_source) else self._cash_source

    @cached_property
    def _period_end_dates(self) -> pd.DataFrame | None:
        """The fiscal period end dates, retrieved when a daily ratio first needs them."""
        if callable(self._period_end_dates_source):
            return self._period_end_dates_source()
        return self._period_end_dates_source

    @property
    def _historical_data(self) -> pd.DataFrame:
        """The historical data of the period (yearly or quarterly)."""
        return self._historical["period"]

    @property
    def _daily_historical_data(self) -> pd.DataFrame:
        """The daily historical data."""
        return self._historical["daily"]

    def _load_statements(self) -> None:
        """
        Retrieves the three statements before a collection, so that it fails once, naming
        the reason, when none of them can be retrieved rather than once for every ratio.
        """
        _ = (
            self._balance_sheet_statement,
            self._income_statement,
            self._cash_flow_statement,
        )

    def _get_days(self, days: int | float | None, trailing: int | None) -> float:
        """
        Returns the number of days the flow figures of a ratio cover.

        Without trailing, a flow (revenue, cost of goods sold, expenses) covers one period: 365
        days for annual data and 365 / 4 for quarterly data. With trailing, the flows are summed
        over `trailing` periods, so the window is `trailing` times as long. An explicit `days`
        is taken as the length of that whole window and returned unchanged.

        Args:
            days (int | float | None): The number of days given by the caller, or None.
            trailing (int | None): The number of periods the flows are summed over, or None.

        Returns:
            float: The number of days of the window the flows cover.
        """
        if days:
            return days

        return (365 / 4 if self._quarterly else 365) * (trailing or 1)

    def _get_valuation_trailing(self, trailing: int | None) -> int | None:
        """
        Returns the trailing window of the ratios that are annual by nature.

        Prices are compared with a year of earnings, sales or cash flows, and the gross,
        operating and cash-based operating profitability are annual measures in their sources,
        so on quarterly data the flows default to the trailing twelve months (4 quarters), as
        for the price-to-earnings ratio. An explicit `trailing` is returned unchanged.

        Args:
            trailing (int | None): The trailing window given by the caller, or None.

        Returns:
            int | None: The trailing window to use, None for annual data without trailing.
        """
        if trailing is not None:
            return trailing

        return 4 if self._quarterly else None

    def _get_share_prices(
        self, begin: str, end: str, show_daily: bool = False
    ) -> pd.DataFrame:
        """
        Returns the traded close of the tickers, as used by every price-based ratio.

        Args:
            begin (str): The first period (or date) to return.
            end (str): The last period to return; daily prices run up to the latest date.
            show_daily (bool, optional): Whether to return daily prices (dates as index) instead
                of period-end prices (tickers as index, periods as columns). Defaults to False.

        Returns:
            pd.DataFrame: The daily or period-end closing prices.
        """
        # A ticker without prices (a failed download or custom historical data that leaves
        # it out) becomes NaN, so it doesn't empty the ratios of the other tickers.
        if show_daily:
            return self._daily_historical_data.loc[begin:, "Close"].reindex(
                columns=self._tickers_without_portfolio
            )

        return (
            self._historical_data.loc[begin:end, "Close"]
            .reindex(columns=self._tickers_without_portfolio)
            .T
        )

    def _get_cash_and_short_term_investments(self) -> pd.DataFrame:
        """
        Returns the cash and short-term investments (marketable securities) of every period,
        as the enterprise value and the invested capital deduct them.

        They are built from the two separate rows, so a missing combined row never leaves
        the cash out.

        Returns:
            pd.DataFrame: The cash and short-term investments with the tickers as index and
            the periods as columns.
        """
        cash_and_short_term_investments = self._balance_sheet_statement.loc[
            :, "Cash and Cash Equivalents", :
        ]

        if (
            "Short Term Investments"
            in self._balance_sheet_statement.index.get_level_values(1)
        ):
            cash_and_short_term_investments = cash_and_short_term_investments.add(
                self._balance_sheet_statement.loc[:, "Short Term Investments", :],
                fill_value=0,
            )

        return cash_and_short_term_investments

    @staticmethod
    def _sum_trailing(data: pd.DataFrame, trailing: int | None) -> pd.DataFrame:
        """
        Sums a flow (income or cash flow statement item) over the trailing periods.

        Args:
            data (pd.DataFrame): The period data, with the periods as columns.
            trailing (int | None): The number of periods to sum over, or None for no sum.

        Returns:
            pd.DataFrame: The trailing sums, or the data itself without trailing.
        """
        return data.T.rolling(trailing).sum().T if trailing else data

    @staticmethod
    def _mean_trailing(data: pd.DataFrame, trailing: int | None) -> pd.DataFrame:
        """
        Averages a balance sheet item over the trailing period ends.

        Args:
            data (pd.DataFrame): The period data, with the periods as columns.
            trailing (int | None): The number of period ends to average, or None for none.

        Returns:
            pd.DataFrame: The trailing averages, or the data itself without trailing.
        """
        return data.T.rolling(trailing).mean().T if trailing else data

    @handle_errors
    def collect_all_ratios(
        self,
        include_dividends: bool = False,
        diluted: bool = True,
        days: int | float | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculates and collects all ratios based on the provided data.

        Args:
            include_dividends (bool, optional): Whether to deduct preferred dividends from net income in the
                earnings-based ratios. Defaults to False.
            diluted (bool, optional): Whether to use diluted shares for the calculation.
                Defaults to True.
            days (int | float, optional): The number of days in the window the flows cover. Defaults to
                None, which uses 365 for annual data and 365 / 4 for quarterly data, multiplied by
                `trailing` when set (e.g. 365 for quarterly data with trailing=4).
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.Series or pd.DataFrame: Ratios calculated based on the specified parameters.

        Notes:
        - The method calculates various ratios for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "TSLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.collect_all_ratios().loc['AAPL']
        ```

        Which returns:

        |                             |         2021 |         2022 |         2023 |         2024 |         2025 |
        |:----------------------------|-------------:|-------------:|-------------:|-------------:|-------------:|
        | EV-to-Sales                 |  8.3883      |  5.5928      |  8.1056      | 10.0052      |  9.9405      |
        | EV-to-EBIT                  | 28.1653      | 18.4649      | 27.1806      | 31.7522      | 31.0925      |
        | EV-to-EBITDA                | 25.522       | 16.8942      | 24.6921      | 29.0536      | 28.5797      |
        | EV-to-Operating-Cash-Flow   | 29.4949      | 18.0546      | 28.1046      | 33.0846      | 37.1078      |
        | Tangible Asset Value        |  6.309e+10   |  5.0672e+10  |  6.2146e+10  |  5.695e+10   |  7.3733e+10  |
        | Net Current Asset Value     | -1.53076e+11 | -1.66678e+11 | -1.46871e+11 | -1.55043e+11 | -1.37551e+11 |
        | EV-to-Free-Cash-Flow        | 33.0122      | 19.7894      | 31.1974      | 35.9571      | 41.885       |
        | Buyback Yield               |  0.0283      |  0.0421      |  0.0255      |  0.0246      |  0.0222      |
        | Shareholder Yield           |  0.0332      |  0.0491      |  0.0304      |  0.0286      |  0.026       |
        | SBC-Adjusted Free Cash Flow |  8.5047e+10  |  1.02405e+11 |  8.8751e+10  |  9.7119e+10  |  8.5904e+10  |
        """
        self._load_statements()

        # Every category is collected with this call's settings: ratios stored by an earlier
        # call can have been calculated with another trailing, days, diluted or dividend choice.
        self.collect_efficiency_ratios(days=days, trailing=trailing)
        self.collect_liquidity_ratios(trailing=trailing)
        self.collect_profitability_ratios(trailing=trailing)
        self.collect_solvency_ratios(diluted=diluted, trailing=trailing)
        self.collect_valuation_ratios(
            include_dividends=include_dividends, diluted=diluted, trailing=trailing
        )

        # The settings these ratios were calculated with, so that the custom ratios, which
        # build on the default ratios, can tell whether they need to be recalculated.
        self._all_ratios_settings = (include_dividends, diluted, days, trailing)

        self._all_ratios = pd.concat(
            [
                self._efficiency_ratios,
                self._liquidity_ratios,
                self._profitability_ratios,
                self._solvency_ratios,
                self._valuation_ratios,
            ]
        )

        # Each category lists every ticker, so the rows are grouped per ticker again (in the
        # order of the tickers, keeping the categories in order within each) for one block
        # per ticker that can be sliced.
        tickers = self._all_ratios.index.get_level_values(0)
        self._all_ratios = self._all_ratios.iloc[
            np.argsort(
                pd.Categorical(tickers, categories=tickers.unique()).codes,
                kind="stable",
            )
        ]

        self._all_ratios = apply_rounding(
            self._all_ratios, rounding if rounding is not None else self._rounding
        )

        # Sorted again so the index follows the financial statements' order.
        available_columns = [
            column
            for column in self._income_statement.columns
            if column in self._all_ratios
        ]
        self._all_ratios = self._all_ratios.reindex(available_columns, axis=1)

        if growth:
            self._all_ratios_growth = calculate_growth(
                dataset=self._all_ratios,
                lag=lag,
                compound=compound,
                rounding=rounding if rounding is not None else self._rounding,
                axis="columns",
            )

        all_ratios = self._all_ratios
        all_ratios_growth = self._all_ratios_growth

        if standardize:
            standardize_rounding = rounding if rounding is not None else self._rounding
            if growth:
                all_ratios_growth = calculate_standardization(
                    dataset=all_ratios_growth,
                    rounding=standardize_rounding,
                    axis="columns",
                )
            else:
                all_ratios = calculate_standardization(
                    dataset=all_ratios, rounding=standardize_rounding, axis="columns"
                )

        if len(self._tickers) == 1:
            return (
                all_ratios_growth.loc[self._tickers[0]]
                if growth
                else all_ratios.loc[self._tickers[0]]
            ).loc[:, self._start_date : self._end_date]

        return (all_ratios_growth if growth else all_ratios).loc[
            :, self._start_date : self._end_date
        ]

    @handle_errors
    def collect_custom_ratios(
        self,
        custom_ratios_dict: dict | None = None,
        options: bool = False,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
    ):
        """
        Calculates all Custom Ratios based on the data provided.

        Note that any of the following characters are considered as operators:
            +, -, *, /, **, %, //, <, >, ==, !=, >=, <=, (, )
        using any of the above characters as part of the column naming will result into an error.

        Args:
            custom_ratios_dict (dict | None, optional): A dictionary containing the custom ratios to
                calculate. Defaults to None.
            options (bool): Whether to return the available names to use in the custom ratios. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.DataFrame: Custom ratios calculated based on the specified parameters.

        Notes:
        - The method calculates various custom ratios for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        custom_ratios = {
            'WC / Net Income as %': '(Working Capital / Net Income) * 100',
            'Large Revenues': 'Revenue > 1000000000',
            'Quick Assets': 'Cash and Short Term Investments + Accounts Receivable',
            'Cash Op Expenses':'Cost of Goods Sold + Selling, General and Administrative Expenses '
            '- Depreciation and Amortization',
            'Daily Cash Op Expenses': 'Cash Op Expenses / 365',
            'Defensive Interval':'Quick Assets / Daily Cash Op Expenses'
        }

        companies = Toolkit(
            tickers=["AAPL", "MSFT", "GOOGL", "AMZN"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2022-10-01",
            quarterly=True,
            end_date="2025-12-31",
        )

        custom_ratios = companies.ratios.collect_custom_ratios(
            custom_ratios_dict=custom_ratios
        )

        custom_ratios.loc['AMZN']
        ```

        Which returns:

        |                        |          2022Q4 |         2023Q1 |         2023Q2 |        2023Q3 |        2023Q4 |        2024Q1 |        2024Q2 |        2024Q3 |        2024Q4 |        2025Q1 |        2025Q2 |        2025Q3 |        2025Q4 |
        |:-----------------------|----------------:|---------------:|---------------:|--------------:|--------------:|--------------:|--------------:|--------------:|--------------:|--------------:|--------------:|--------------:|--------------:|
        | WC / Net Income as %   | -3094.24        | -357.787       | -114.904       | -22.4618      |  69.9736      | 105.685       | 112.236       |  93.3912      |  57.1686      |  49.4774      |  24.7688      |   7.8822      |  52.2744      |
        | Large Revenues         |     1           |    1           |    1           |   1           |   1           |   1           |   1           |   1           |   1           |   1           |   1           |   1           |   1           |
        | Quick Assets           |     1.12386e+11 |    1.02051e+11 |    1.03895e+11 |   1.07589e+11 |   1.39033e+11 |   1.32842e+11 |   1.39198e+11 |   1.39689e+11 |   1.56653e+11 |   1.48781e+11 |   1.50595e+11 |   1.55372e+11 |   1.90758e+11 |
        | Cash Op Expenses       |     8.8646e+10  |    6.9883e+10  |    7.1731e+10  |   7.6003e+10  |   9.4645e+10  |   7.3353e+10  |   7.53e+10    |   8.0857e+10  |   9.9249e+10  |   7.5105e+10  |   7.9963e+10  |   8.6435e+10  |   1.07456e+11 |
        | Daily Cash Op Expenses |     2.42866e+08 |    1.9146e+08  |    1.96523e+08 |   2.08227e+08 |   2.59301e+08 |   2.00967e+08 |   2.06301e+08 |   2.21526e+08 |   2.71915e+08 |   2.05767e+08 |   2.19077e+08 |   2.36808e+08 |   2.944e+08   |
        | Defensive Interval     |   462.75        |  533.014       |  528.665       | 516.69        | 536.183       | 661.014       | 674.731       | 630.576       | 576.11        | 723.055       | 687.408       | 656.109       | 647.955       |
        """
        self._load_statements()
        # Custom ratios build on the ratios with their default settings.
        if self._all_ratios.empty or self._all_ratios_settings != (
            False,
            True,
            None,
            None,
        ):
            self.collect_all_ratios()

        if not custom_ratios_dict and not options:
            logger.error(
                "Please define custom ratios dictionary to the custom_ratios_dict parameter. See "
                "https://www.jeroenbouma.com/projects/financetoolkit/custom-ratios how to do this."
            )
            return None

        if options:
            logger.info(
                "The following names are available to be used in the Custom Ratios calculations."
            )

            self._available_custom_ratios_options = list(
                set(
                    list(self._balance_sheet_statement.index.get_level_values(level=1))
                    + list(self._income_statement.index.get_level_values(level=1))
                    + list(self._cash_flow_statement.index.get_level_values(level=1))
                    + list(self._all_ratios.index.get_level_values(level=1))
                )
            )

            self._available_custom_ratios_options.sort()

            return self._available_custom_ratios_options

        # The guard at the top already returned when no dictionary was given, so this
        # only spells that out for the type checker.
        custom_ratios_dict = custom_ratios_dict or {}

        custom_ratios = pd.DataFrame(
            0,
            index=pd.MultiIndex.from_product(
                [self._tickers, custom_ratios_dict.keys()]
            ),
            columns=self._balance_sheet_statement.columns,
        )

        total_financials = pd.concat(
            [
                self._balance_sheet_statement,
                self._income_statement,
                self._cash_flow_statement,
                self._all_ratios,
                custom_ratios,
            ],
            axis=0,
        )

        total_financials = total_financials[
            ~total_financials.index.duplicated(keep="first")
        ]

        formula_dict = {}
        for name, formula in custom_ratios_dict.items():
            # Reordered so a formula depending on another is evaluated after it.
            for sub_name, sub_formula in custom_ratios_dict.items():
                if sub_name in formula:
                    formula_dict[sub_name] = sub_formula

            if name not in formula_dict:
                formula_dict[name] = formula

        for name, formula in formula_dict.items():
            formula_names = formula

            for operator in [
                "+",
                "-",
                "*",
                "/",
                "**",
                "%",
                "//",
                "<",
                ">",
                "==",
                "!=",
                ">=",
                "<=",
                "(",
                ")",
            ]:
                formula_names = formula_names.replace(operator, "SPLIT")

            formula_names = formula_names.split("SPLIT")

            formula_names = [
                clean_name
                for clean_name in formula_names
                if clean_name not in ["", " "]
            ]

            formula_adjusted = formula

            for formula_section in formula_names:
                formula_section_stripped = formula_section.strip()
                if formula_section_stripped in total_financials.index.get_level_values(
                    1
                ):
                    formula_adjusted = formula_adjusted.replace(
                        formula_section_stripped,
                        f"total_financials.loc[:, '{formula_section_stripped}', :]",
                    )
                else:
                    try:
                        float(formula_section_stripped)
                    except ValueError:
                        formula_adjusted = None
                        logger.error(
                            "Column %s not found in total_financials and is not a number. "
                            "Therefore the formula %s is invalid. Use collect_custom_ratios(options=True) "
                            "to see the available columns.",
                            formula_section_stripped,
                            formula,
                        )
                        break

            if formula_adjusted:
                calculation = eval(formula_adjusted)  # noqa

                # A zero denominator is undefined (NaN), as in every other ratio.
                total_financials.loc[:, name, :] = drop_non_finite(
                    calculation.astype(np.float64)
                ).to_numpy()

                self._custom_ratios = total_financials.loc[
                    :, list(custom_ratios_dict.keys()), :
                ]
                self._custom_ratios = self._custom_ratios.sort_index(
                    axis=0, level=0, sort_remaining=False
                )

                self._custom_ratios = self._custom_ratios.pipe(
                    apply_rounding, rounding if rounding is not None else self._rounding
                )

        if growth:
            self._custom_ratios_growth = calculate_growth(
                dataset=self._custom_ratios,
                lag=lag,
                compound=compound,
                rounding=rounding if rounding is not None else self._rounding,
                axis="columns",
            )

        custom_ratios = self._custom_ratios
        custom_ratios_growth = self._custom_ratios_growth

        if standardize:
            standardize_rounding = rounding if rounding is not None else self._rounding
            if growth:
                custom_ratios_growth = calculate_standardization(
                    dataset=custom_ratios_growth,
                    rounding=standardize_rounding,
                    axis="columns",
                )
            else:
                custom_ratios = calculate_standardization(
                    dataset=custom_ratios, rounding=standardize_rounding, axis="columns"
                )

        if len(self._tickers) == 1:
            return (
                custom_ratios_growth.loc[self._tickers[0]]
                if growth
                else custom_ratios.loc[self._tickers[0]]
            ).loc[:, self._start_date : self._end_date]

        return (custom_ratios_growth if growth else custom_ratios).loc[
            :, self._start_date : self._end_date
        ]

    @handle_errors
    def collect_efficiency_ratios(
        self,
        days: int | float | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculates and collects all Efficiency Ratios based on the provided data.

        Args:
            days (int | float, optional): The number of days in the window the flows cover. Defaults to
                None, which uses 365 for annual data and 365 / 4 for quarterly data, multiplied by
                `trailing` when set (e.g. 365 for quarterly data with trailing=4).
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.Series or pd.DataFrame: Efficiency ratios calculated based on the specified parameters.

        Notes:
        - The method calculates various efficiency ratios for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "TSLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.collect_efficiency_ratios().loc['AAPL']
        ```

        Which returns:

        |                                      |     2021 |     2022 |     2023 |     2024 |     2025 |
        |:-------------------------------------|---------:|---------:|---------:|---------:|---------:|
        | Days of Inventory Outstanding        |   9.1181 |   9.4097 |   9.6109 |  11.814  |  10.7405 |
        | Days of Sales Outstanding            |  21.1517 |  25.2057 |  27.4699 |  29.3645 |  32.0949 |
        | Operating Cycle                      |  30.2698 |  34.6154 |  37.0808 |  41.1785 |  42.8354 |
        | Days of Accounts Payable Outstanding |  83.1683 |  97.0504 | 108.003  | 114.15   | 114.657  |
        | Cash Conversion Cycle                | -52.8985 | -62.435  | -70.9225 | -72.9716 | -71.8218 |
        | Cash Conversion Efficiency           |   0.2844 |   0.3098 |   0.2884 |   0.3024 |   0.2679 |
        | Receivables Turnover                 |  17.2563 |  14.4808 |  13.2873 |  12.43   |  11.3725 |
        | Inventory Turnover Ratio             |  40.0303 |  38.7899 |  37.9777 |  30.8955 |  33.9834 |
        | Accounts Payable Turnover Ratio      |   4.3887 |   3.7609 |   3.3795 |   3.1975 |   3.1834 |
        | SGA-to-Revenue Ratio                 |   0.0601 |   0.0636 |   0.065  |   0.0667 |   0.0663 |
        | Fixed Asset Turnover                 |   7.7125 |   7.7273 |   7.1702 |   7.091  |   7.1167 |
        | Asset Turnover Ratio                 |   1.0841 |   1.1206 |   1.0868 |   1.0899 |   1.1493 |
        | Operating Ratio                      |   0.7022 |   0.6971 |   0.7018 |   0.6849 |   0.6803 |
        | R&D Intensity Ratio                  |   0.0599 |   0.0666 |   0.078  |   0.0802 |   0.083  |
        | S&M to Revenue Ratio                 |   0      |   0      |   0      |   0.0477 |   0.0469 |
        | G&A to Revenue Ratio                 |   0      |   0      |   0      |   0.0191 |   0.0194 |
        | SBC to Revenue Ratio                 |   0.0216 |   0.0229 |   0.0283 |   0.0299 |   0.0309 |
        | Deferred Revenue Ratio               |   0.0208 |   0.0201 |   0.021  |   0.0211 |   0.0218 |
        | Working Capital Turnover Ratio       |  15.346  | -85.519  | -37.7268 | -31.0999 | -20.2615 |
        | Capex Intensity                      |   0.0303 |   0.0272 |   0.0286 |   0.0242 |   0.0306 |
        | Capex to Depreciation                |   0.9824 |   0.9643 |   0.9514 |   0.8254 |   1.0869 |
        """
        self._load_statements()

        efficiency_ratios_dict: dict = {}

        efficiency_ratios_dict["Days of Inventory Outstanding"] = (
            self.get_days_of_inventory_outstanding(days=days, trailing=trailing)
        )
        efficiency_ratios_dict["Days of Sales Outstanding"] = (
            self.get_days_of_sales_outstanding(days=days, trailing=trailing)
        )
        efficiency_ratios_dict["Operating Cycle"] = self.get_operating_cycle(
            days=days, trailing=trailing
        )
        efficiency_ratios_dict["Days of Accounts Payable Outstanding"] = (
            self.get_days_of_accounts_payable_outstanding(days=days, trailing=trailing)
        )
        efficiency_ratios_dict["Cash Conversion Cycle"] = (
            self.get_cash_conversion_cycle(days=days, trailing=trailing)
        )
        efficiency_ratios_dict["Cash Conversion Efficiency"] = (
            self.get_cash_conversion_efficiency(trailing=trailing)
        )
        efficiency_ratios_dict["Receivables Turnover"] = self.get_receivables_turnover(
            trailing=trailing
        )
        efficiency_ratios_dict["Inventory Turnover Ratio"] = (
            self.get_inventory_turnover_ratio(trailing=trailing)
        )
        efficiency_ratios_dict["Accounts Payable Turnover Ratio"] = (
            self.get_accounts_payables_turnover_ratio(trailing=trailing)
        )
        efficiency_ratios_dict["SGA-to-Revenue Ratio"] = self.get_sga_to_revenue_ratio(
            trailing=trailing
        )
        efficiency_ratios_dict["Fixed Asset Turnover"] = self.get_fixed_asset_turnover(
            trailing=trailing
        )
        efficiency_ratios_dict["Asset Turnover Ratio"] = self.get_asset_turnover_ratio(
            trailing=trailing
        )
        efficiency_ratios_dict["Operating Ratio"] = self.get_operating_ratio(
            trailing=trailing
        )
        efficiency_ratios_dict["R&D Intensity Ratio"] = (
            self.get_research_and_development_ratio(trailing=trailing)
        )
        efficiency_ratios_dict["S&M to Revenue Ratio"] = (
            self.get_selling_and_marketing_ratio(trailing=trailing)
        )
        efficiency_ratios_dict["G&A to Revenue Ratio"] = (
            self.get_general_and_administrative_ratio(trailing=trailing)
        )
        efficiency_ratios_dict["SBC to Revenue Ratio"] = (
            self.get_stock_based_compensation_ratio(trailing=trailing)
        )
        efficiency_ratios_dict["Deferred Revenue Ratio"] = (
            self.get_deferred_revenue_ratio(trailing=trailing)
        )
        efficiency_ratios_dict["Working Capital Turnover Ratio"] = (
            self.get_working_capital_turnover_ratio(trailing=trailing)
        )
        efficiency_ratios_dict["Capex Intensity"] = self.get_capex_intensity(
            trailing=trailing
        )
        efficiency_ratios_dict["Capex to Depreciation"] = (
            self.get_capex_to_depreciation_ratio(trailing=trailing)
        )
        efficiency_ratios_dict["Asset Growth"] = self.get_asset_growth(
            trailing=trailing
        )
        efficiency_ratios_dict["Net Operating Assets"] = self.get_net_operating_assets()

        self._efficiency_ratios = (
            pd.concat(efficiency_ratios_dict)
            .swaplevel(0, 1)
            .sort_index(level=0, sort_remaining=False)
            .dropna(axis="columns", how="all")
        )

        # Ensure the ticker order remains the same as in self._tickers
        self._efficiency_ratios = self._efficiency_ratios.loc[self._tickers]

        self._efficiency_ratios = self._efficiency_ratios.pipe(
            apply_rounding, rounding if rounding is not None else self._rounding
        )

        # Sorted again so the index follows the financial statements' order.
        available_columns = [
            column
            for column in self._income_statement.columns
            if column in self._efficiency_ratios
        ]
        self._efficiency_ratios = self._efficiency_ratios.reindex(
            available_columns, axis=1
        )

        if growth:
            self._efficiency_ratios_growth = calculate_growth(
                dataset=self._efficiency_ratios,
                lag=lag,
                compound=compound,
                rounding=rounding if rounding is not None else self._rounding,
                axis="columns",
            )

        efficiency_ratios = self._efficiency_ratios
        efficiency_ratios_growth = self._efficiency_ratios_growth

        if standardize:
            standardize_rounding = rounding if rounding is not None else self._rounding
            if growth:
                efficiency_ratios_growth = calculate_standardization(
                    dataset=efficiency_ratios_growth,
                    rounding=standardize_rounding,
                    axis="columns",
                )
            else:
                efficiency_ratios = calculate_standardization(
                    dataset=efficiency_ratios,
                    rounding=standardize_rounding,
                    axis="columns",
                )

        if len(self._tickers) == 1:
            return (
                efficiency_ratios_growth.loc[self._tickers[0]]
                if growth
                else efficiency_ratios.loc[self._tickers[0]]
            ).loc[:, self._start_date : self._end_date]

        return (efficiency_ratios_growth if growth else efficiency_ratios).loc[
            :, self._start_date : self._end_date
        ]

    @handle_portfolio
    @handle_errors
    def get_asset_turnover_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the asset turnover ratio, an efficiency ratio that measures how
        efficiently a company uses its assets to generate sales.

        The asset turnover ratio is calculated by dividing the company's net sales
        (revenue) by its average total assets. It measures how well a company utilizes
        its assets to generate revenue. A higher asset turnover ratio indicates that the
        company is generating more revenue per unit of assets, which is generally seen
        as a positive sign of operational efficiency.

        The formula is as follows:

        - Asset Turnover Ratio = Net Sales / Average Total Assets

        Also known as: asset efficiency, revenue per asset.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.Series: Asset turnover ratio values.

        Notes:
        - The method retrieves historical data and calculates the asset turnover ratio
          for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["TGT", "BBY", "DG"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_asset_turnover_ratio()
        ```

        Which returns:

        |     |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:----|-------:|-------:|-------:|-------:|-------:|
        | BBY | 2.8307 | 2.7801 | 2.8243 | 2.7919 | 2.8311 |
        | DG  | 1.3114 | 1.366  | 1.2923 | 1.3116 | 1.3761 |
        | TGT | 2.018  | 2.0368 | 1.9765 | 1.884  | 1.7872 |
        """
        if trailing:
            asset_turnover_ratio = efficiency_model.get_asset_turnover_ratio(
                self._sum_trailing(
                    self._income_statement.loc[:, "Revenue", :], trailing
                ),
                self._mean_trailing(
                    self._balance_sheet_statement.loc[:, "Total Assets", :], trailing
                ),
            )
        else:
            asset_turnover_ratio = efficiency_model.get_asset_turnover_ratio(
                self._income_statement.loc[:, "Revenue", :],
                self._balance_sheet_statement.loc[:, "Total Assets", :]
                .T.rolling(2)
                .mean()
                .T,
            )

        return finalize_dataset(
            dataset=asset_turnover_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_inventory_turnover_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the inventory turnover ratio, an efficiency ratio that measures
        how quickly a company sells its inventory.

        The inventory turnover ratio is calculated by dividing the cost of goods sold
        (COGS) by the average inventory value. It indicates how many times a company's
        inventory is sold and replaced over a period. A higher inventory turnover ratio
        suggests that a company is effectively managing its inventory by quickly
        converting it into sales.

        The formula is as follows:

        - Inventory Turnover Ratio = Cost of Goods Sold / Average Inventory

        Inventory turnover is only comparable within an industry: a grocer turns its
        stock many times a year, a maker of aircraft or luxury goods a few times at
        most. Analysts watch the trend and the gap with peers, since a falling ratio
        often comes before markdowns or write-downs. Because the ratio divides cost of
        goods sold by inventory, the inventory valuation method (LIFO or FIFO) and
        year-end stock building for a peak season both move it; quarterly data with
        trailing=4 smooths the seasonal effect.

        Also known as: stock turnover ratio.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.Series: Inventory turnover ratio values.

        Notes:
        - The method retrieves historical data and calculates the inventory turnover ratio
          for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["HD", "LOW", "TSCO"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_inventory_turnover_ratio()
        ```

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | HD   | 5.1854 | 4.4565 | 4.4354 | 4.7811 | 4.458  |
        | LOW  | 3.7987 | 3.5865 | 3.2481 | 3.2532 | 3.3076 |
        | TSCO | 4.1535 | 3.7678 | 3.4834 | 3.4585 | 3.4987 |
        """
        if trailing:
            inventory_turnover_ratio = efficiency_model.get_inventory_turnover_ratio(
                self._sum_trailing(
                    self._income_statement.loc[:, "Cost of Goods Sold", :], trailing
                ),
                self._mean_trailing(
                    self._balance_sheet_statement.loc[:, "Inventory", :], trailing
                ),
            )
        else:
            inventory_turnover_ratio = efficiency_model.get_inventory_turnover_ratio(
                self._income_statement.loc[:, "Cost of Goods Sold", :],
                self._balance_sheet_statement.loc[:, "Inventory", :]
                .T.rolling(2)
                .mean()
                .T,
            )

        return finalize_dataset(
            dataset=inventory_turnover_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_days_of_inventory_outstanding(
        self,
        days: int | float | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the days sales in inventory ratio, an efficiency ratio that measures
        how long it takes a company to sell its inventory.

        The days sales in inventory ratio (DSI) is calculated by dividing the average
        inventory by the cost of goods sold (COGS) and then multiplying by the number
        of days in the period. It represents the average number of days it takes for
        a company to sell its inventory. A lower DSI indicates that the company is
        selling its inventory more quickly.

        The formula is as follows:

        - Days Sales in Inventory Ratio = (Average Inventory / Cost of Goods Sold) * Days

        Also known as: DIO, days inventory outstanding.

        Args:
            days (int | float, optional): The number of days in the window the flows cover. Defaults to
                None, which uses 365 for annual data and 365 / 4 for quarterly data, multiplied by
                `trailing` when set (e.g. 365 for quarterly data with trailing=4).
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.DataFrame: Days sales in inventory ratio values.

        Notes:
        - The method retrieves historical data and calculates the DSI ratio for each
          asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["NKE", "LULU", "DECK"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_days_of_inventory_outstanding()
        ```

        Which returns:

        |      |    2021 |    2022 |    2023 |     2024 |     2025 |
        |:-----|--------:|--------:|--------:|---------:|---------:|
        | DECK |  92.864 | 105.297 |  96.625 |  84.2594 |  77.4483 |
        | LULU | 111.215 | 121.754 | 126.114 | 116.91   | 119.035  |
        | NKE  | 110.479 | 106.465 | 102.373 | 103.283  | 103.284  |
        """
        days = self._get_days(days, trailing)

        if trailing:
            days_of_inventory_outstanding = (
                efficiency_model.get_days_of_inventory_outstanding(
                    self._mean_trailing(
                        self._balance_sheet_statement.loc[:, "Inventory", :], trailing
                    ),
                    self._sum_trailing(
                        self._income_statement.loc[:, "Cost of Goods Sold", :], trailing
                    ),
                    days,
                )
            )
        else:
            days_of_inventory_outstanding = (
                efficiency_model.get_days_of_inventory_outstanding(
                    self._balance_sheet_statement.loc[:, "Inventory", :]
                    .T.rolling(2)
                    .mean()
                    .T,
                    self._income_statement.loc[:, "Cost of Goods Sold", :],
                    days,
                )
            )

        return finalize_dataset(
            dataset=days_of_inventory_outstanding,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_days_of_sales_outstanding(
        self,
        days: int | float | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the days of sales outstanding ratio, an efficiency ratio that measures
        the average number of days it takes a company to collect payment on its
        credit sales.

        The days of sales outstanding (DSO) ratio is calculated by dividing the average
        accounts receivable by revenue and then multiplying by the number of days in the
        period. It represents the average number of days it takes for a company to
        collect payment on its sales. A lower DSO indicates that the company is
        collecting payments more quickly. Revenue stands in for credit sales, which
        companies rarely disclose, so for a business that also sells for cash the DSO
        understates how long its credit customers take to pay.

        The formula is as follows:

        - Days of Sales Outstanding Ratio = (Average Accounts Receivable / Revenue) * Days

        DSO is read against the payment terms a company offers and against its own
        history: a rising DSO, with receivables growing faster than revenue, often
        points to looser credit terms, slower-paying customers or revenue booked ahead
        of collection. Compare it within an industry, since terms differ widely between
        retailers, who collect within days, and industrial or software suppliers.

        Also known as: DSO, days sales outstanding, receivable days.

        Args:
            days (int | float, optional): The number of days in the window the flows cover. Defaults to
                None, which uses 365 for annual data and 365 / 4 for quarterly data, multiplied by
                `trailing` when set (e.g. 365 for quarterly data with trailing=4).
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.DataFrame: Days of sales outstanding ratio values.

        Notes:
        - The method retrieves historical data and calculates the DSO ratio for each
          asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["HON", "EMR", "ETN"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_days_of_sales_outstanding()
        ```

        Which returns:

        |     |    2021 |    2022 |    2023 |    2024 |    2025 |
        |:----|--------:|--------:|--------:|--------:|--------:|
        | EMR | 81.4702 | 69.1713 | 57.5119 | 56.8095 | 61.0629 |
        | ETN | 57.6565 | 64.8406 | 67.277  | 66.7118 | 66.5293 |
        | HON | 72.4704 | 73.4302 | 82.7661 | 80.6865 | 75.2577 |
        """
        days = self._get_days(days, trailing)

        if trailing:
            days_of_sales_outstanding = efficiency_model.get_days_of_sales_outstanding(
                self._mean_trailing(
                    self._balance_sheet_statement.loc[:, "Accounts Receivable", :],
                    trailing,
                ),
                self._sum_trailing(
                    self._income_statement.loc[:, "Revenue", :], trailing
                ),
                days,
            )
        else:
            days_of_sales_outstanding = efficiency_model.get_days_of_sales_outstanding(
                self._balance_sheet_statement.loc[:, "Accounts Receivable", :]
                .T.rolling(2)
                .mean()
                .T,
                self._income_statement.loc[:, "Revenue", :],
                days,
            )

        return finalize_dataset(
            dataset=days_of_sales_outstanding,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_operating_cycle(
        self,
        days: int | float | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the operating cycle ratio, an efficiency ratio that measures the average
        number of days it takes a company to turn its inventory into cash.

        The operating cycle represents the total time required to purchase inventory,
        convert it into finished goods, sell the goods to customers, and collect the
        accounts receivable. It is calculated by adding the days sales in inventory (DSI)
        and the days of sales outstanding (DSO).

        The formula is as follows:

        - Operating Cycle Ratio = Days of Sales in Inventory + Days of Sales Outstanding

        Also known as: inventory-to-cash cycle.

        Args:
            days (int | float, optional): The number of days in the window the flows cover. Defaults to
                None, which uses 365 for annual data and 365 / 4 for quarterly data, multiplied by
                `trailing` when set (e.g. 365 for quarterly data with trailing=4).
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.DataFrame: Operating cycle ratio values.

        Notes:
        - The method retrieves historical data and calculates the operating cycle ratio for each
          asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AMAT", "LRCX", "KLAC"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_operating_cycle()
        ```

        Which returns:

        |      |    2021 |    2022 |    2023 |    2024 |    2025 |
        |:-----|--------:|--------:|--------:|--------:|--------:|
        | AMAT | 187.185 | 213.516 | 227.837 | 212.291 | 209.118 |
        | KLAC | 253.665 | 250.855 | 279.334 | 341.358 | 301.427 |
        | LRCX | 171.022 | 207.597 | 240.81  | 275.374 | 222.92  |
        """
        days = self._get_days(days, trailing)

        if trailing:
            days_of_inventory = efficiency_model.get_days_of_inventory_outstanding(
                self._mean_trailing(
                    self._balance_sheet_statement.loc[:, "Inventory", :], trailing
                ),
                self._sum_trailing(
                    self._income_statement.loc[:, "Cost of Goods Sold", :], trailing
                ),
                days,
            )

            days_of_sales = efficiency_model.get_days_of_sales_outstanding(
                self._mean_trailing(
                    self._balance_sheet_statement.loc[:, "Accounts Receivable", :],
                    trailing,
                ),
                self._sum_trailing(
                    self._income_statement.loc[:, "Revenue", :], trailing
                ),
                days,
            )
        else:
            days_of_inventory = efficiency_model.get_days_of_inventory_outstanding(
                self._balance_sheet_statement.loc[:, "Inventory", :]
                .T.rolling(2)
                .mean()
                .T,
                self._income_statement.loc[:, "Cost of Goods Sold", :],
                days,
            )
            days_of_sales = efficiency_model.get_days_of_sales_outstanding(
                self._balance_sheet_statement.loc[:, "Accounts Receivable", :]
                .T.rolling(2)
                .mean()
                .T,
                self._income_statement.loc[:, "Revenue", :],
                days,
            )

        operating_cycle = efficiency_model.get_operating_cycle(
            days_of_inventory, days_of_sales
        )

        return finalize_dataset(
            dataset=operating_cycle,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_accounts_payables_turnover_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the accounts payable turnover ratio, an efficiency ratio that measures how
        quickly a company pays its suppliers.

        The accounts payable turnover ratio indicates how many times, on average, a company
        pays off its accounts payable during a specific period. A higher turnover ratio
        means the company pays its suppliers more quickly, which can reflect short payment
        terms or ample cash but ties up working capital. A lower ratio means it takes
        longer to pay and finances part of its working capital with supplier credit; a
        ratio that keeps falling can also signal payment difficulties.

        The formula is as follows:

        - Accounts Payable Turnover Ratio = Cost of Goods Sold / Average Accounts Payable

        Also known as: payables efficiency, AP turnover.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.DataFrame: Accounts payable turnover ratio values.

        Notes:
        - The method retrieves historical data and calculates the accounts payable turnover ratio for each
          asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["PG", "CL", "KMB"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_accounts_payables_turnover_ratio()
        ```

        Which returns:

        |     |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:----|-------:|-------:|-------:|-------:|-------:|
        | CL  | 4.9638 | 5.1597 | 5.0514 | 4.5647 | 4.1762 |
        | KMB | 3.7492 | 3.6472 | 3.5893 | 3.4957 | 3.1215 |
        | PG  | 2.8776 | 2.9478 | 2.9009 | 2.7267 | 2.6912 |
        """
        if trailing:
            accounts_payables_turnover_ratio = (
                efficiency_model.get_accounts_payables_turnover_ratio(
                    self._sum_trailing(
                        self._income_statement.loc[:, "Cost of Goods Sold", :], trailing
                    ),
                    self._mean_trailing(
                        self._balance_sheet_statement.loc[:, "Accounts Payable", :],
                        trailing,
                    ),
                )
            )
        else:
            accounts_payables_turnover_ratio = (
                efficiency_model.get_accounts_payables_turnover_ratio(
                    self._income_statement.loc[:, "Cost of Goods Sold", :],
                    self._balance_sheet_statement.loc[:, "Accounts Payable", :]
                    .T.rolling(2)
                    .mean()
                    .T,
                )
            )

        return finalize_dataset(
            dataset=accounts_payables_turnover_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_days_of_accounts_payable_outstanding(
        self,
        days: int | float | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the days payables outstanding, an efficiency ratio that measures the
        number of days it takes a company to pay its suppliers.

        The days payables outstanding (DPO) ratio is used to assess how efficiently a company
        manages its accounts payable. It calculates the average number of days it takes for
        a company to pay its suppliers after receiving an invoice. A higher DPO ratio indicates
        that the company is taking longer to pay its suppliers, which may have implications for
        its relationships with suppliers.

        The formula is as follows:

        - Days Payables Outstanding = (Average Accounts Payable / Cost of Goods Sold) * Days

        Also known as: DPO, days payable outstanding.

        Args:
            days (int | float, optional): The number of days in the window the flows cover. Defaults to
                None, which uses 365 for annual data and 365 / 4 for quarterly data, multiplied by
                `trailing` when set (e.g. 365 for quarterly data with trailing=4).
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.DataFrame: Days payables outstanding (DPO) ratio values.

        Notes:
        - The method retrieves historical data and calculates the DPO ratio for each
          asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["GM", "F", "TSLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_days_of_accounts_payable_outstanding()
        ```

        Which returns:

        |      |    2021 |    2022 |    2023 |    2024 |    2025 |
        |:-----|--------:|--------:|--------:|--------:|--------:|
        | F    | 72.7319 | 65.8213 | 61.9989 | 56.6067 | 55.4167 |
        | GM   | 72.691  | 69.371  | 71.0554 | 63.4048 | 54.8851 |
        | TSLA | 72.951  | 76.1207 | 68.4805 | 61.1935 | 60.6784 |
        """
        days = self._get_days(days, trailing)

        if trailing:
            days_of_accounts_payable_outstanding = (
                efficiency_model.get_days_of_accounts_payable_outstanding(
                    self._sum_trailing(
                        self._income_statement.loc[:, "Cost of Goods Sold", :], trailing
                    ),
                    self._mean_trailing(
                        self._balance_sheet_statement.loc[:, "Accounts Payable", :],
                        trailing,
                    ),
                    days,
                )
            )
        else:
            days_of_accounts_payable_outstanding = (
                efficiency_model.get_days_of_accounts_payable_outstanding(
                    self._income_statement.loc[:, "Cost of Goods Sold", :],
                    self._balance_sheet_statement.loc[:, "Accounts Payable", :]
                    .T.rolling(2)
                    .mean()
                    .T,
                    days,
                )
            )

        return finalize_dataset(
            dataset=days_of_accounts_payable_outstanding,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_cash_conversion_cycle(
        self,
        days: int | float | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Cash Conversion Cycle, which measures the amount of time it takes for a company to convert
        its investments in inventory and accounts receivable into cash, while considering the time it takes to pay
        its accounts payable. This ratio is also known as Cash-to-Cash Cycle (C2C) or Net Operating Cycle.

        The Cash Conversion Cycle (CCC) is an important measure of a company's liquidity management and efficiency
        in managing its working capital. It takes into account the time it takes to sell inventory, collect payments
        from customers, and pay suppliers. A shorter CCC indicates that a company is able to quickly convert its
        investments into cash, which can be a positive sign of efficient operations.

        The formula is as follows:

        - Cash Conversion Cycle = Days of Sales in Inventory + Days of Sales Outstanding — Days of Accounts Payable
            Outstanding

        A negative cash conversion cycle means suppliers are paid after customers have paid, so growth releases cash
        instead of absorbing it; large retailers and platforms with strong buying power are the usual examples. Read
        the cycle through its three parts (`get_days_of_inventory_outstanding`, `get_days_of_sales_outstanding` and
        `get_days_of_accounts_payable_outstanding`), because a shorter cycle bought by stretching suppliers is less
        durable than one from faster-moving inventory.

        Also known as: CCC.

        Args:
            days (int | float, optional): The number of days in the window the flows cover. Defaults to
                None, which uses 365 for annual data and 365 / 4 for quarterly data, multiplied by
                `trailing` when set (e.g. 365 for quarterly data with trailing=4).
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.DataFrame: Cash Conversion Cycle (CCC) values.

        Notes:
        - The method retrieves historical data and calculates the CCC for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the CCC values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AMZN", "COST", "WMT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_cash_conversion_cycle()
        ```

        Which returns:

        |      |     2021 |     2022 |     2023 |     2024 |     2025 |
        |:-----|---------:|---------:|---------:|---------:|---------:|
        | AMZN | -41.1953 | -30.918  | -27.9651 | -31.7282 | -42.2411 |
        | COST |  -1.1462 |   1.4176 |   2.7455 |   2.2723 |   2.071  |
        | WMT  |   3.463  |   6.4477 |   5.0509 |   3.5502 |   3.216  |
        """
        days = self._get_days(days, trailing)

        if trailing:
            days_of_inventory = efficiency_model.get_days_of_inventory_outstanding(
                self._mean_trailing(
                    self._balance_sheet_statement.loc[:, "Inventory", :], trailing
                ),
                self._sum_trailing(
                    self._income_statement.loc[:, "Cost of Goods Sold", :], trailing
                ),
                days,
            )

            days_of_sales = efficiency_model.get_days_of_sales_outstanding(
                self._mean_trailing(
                    self._balance_sheet_statement.loc[:, "Accounts Receivable", :],
                    trailing,
                ),
                self._sum_trailing(
                    self._income_statement.loc[:, "Revenue", :], trailing
                ),
                days,
            )

            days_of_payables = (
                efficiency_model.get_days_of_accounts_payable_outstanding(
                    self._sum_trailing(
                        self._income_statement.loc[:, "Cost of Goods Sold", :], trailing
                    ),
                    self._mean_trailing(
                        self._balance_sheet_statement.loc[:, "Accounts Payable", :],
                        trailing,
                    ),
                    days,
                )
            )
        else:
            days_of_inventory = efficiency_model.get_days_of_inventory_outstanding(
                self._balance_sheet_statement.loc[:, "Inventory", :]
                .T.rolling(2)
                .mean()
                .T,
                self._income_statement.loc[:, "Cost of Goods Sold", :],
                days,
            )
            days_of_sales = efficiency_model.get_days_of_sales_outstanding(
                self._balance_sheet_statement.loc[:, "Accounts Receivable", :]
                .T.rolling(2)
                .mean()
                .T,
                self._income_statement.loc[:, "Revenue", :],
                days,
            )

            days_of_payables = (
                efficiency_model.get_days_of_accounts_payable_outstanding(
                    self._income_statement.loc[:, "Cost of Goods Sold", :],
                    self._balance_sheet_statement.loc[:, "Accounts Payable", :]
                    .T.rolling(2)
                    .mean()
                    .T,
                    days,
                )
            )

        cash_conversion_cycle = efficiency_model.get_cash_conversion_cycle(
            days_of_inventory, days_of_sales, days_of_payables
        )

        return finalize_dataset(
            dataset=cash_conversion_cycle,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_cash_conversion_efficiency(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the cash conversion efficiency, an efficiency ratio that measures how
        efficiently a company converts its sales into cash. It is also known as the cash
        conversion ratio.

        The cash conversion efficiency ratio is calculated by dividing the operating cash
        flow by the revenue. It indicates how much of a company's sales are converted into
        cash. A higher cash conversion efficiency ratio is generally favorable, as it
        suggests that the company is able to convert its sales into cash more efficiently.

        The formula is as follows:

        - Cash Conversion Efficiency Ratio = Operating Cash Flow / Revenue

        Also known as: CCE, cash efficiency ratio.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.DataFrame: Cash conversion efficiency ratio values.

        Notes:
        - The method retrieves historical data and calculates the cash conversion
        efficiency for each asset in the Toolkit instance. It is the same calculation as
        `get_operating_cash_flow_sales_ratio`.
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["ADBE", "CRM", "NOW"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_cash_conversion_efficiency()
        ```

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | ADBE | 0.458  | 0.4452 | 0.3762 | 0.3746 | 0.422  |
        | CRM  | 0.2265 | 0.2268 | 0.2936 | 0.3455 | 0.3611 |
        | NOW  | 0.3716 | 0.3758 | 0.3788 | 0.3885 | 0.41   |
        """
        if trailing:
            cash_conversion_efficiency = (
                efficiency_model.get_cash_conversion_efficiency(
                    self._sum_trailing(
                        self._cash_flow_statement.loc[
                            :, "Cash Flow from Operations", :
                        ],
                        trailing,
                    ),
                    self._sum_trailing(
                        self._income_statement.loc[:, "Revenue", :], trailing
                    ),
                )
            )
        else:
            cash_conversion_efficiency = (
                efficiency_model.get_cash_conversion_efficiency(
                    self._cash_flow_statement.loc[:, "Cash Flow from Operations", :],
                    self._income_statement.loc[:, "Revenue", :],
                )
            )

        return finalize_dataset(
            dataset=cash_conversion_efficiency,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_receivables_turnover(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the receivables turnover, a ratio that measures how efficiently a
        company uses its assets by comparing the amount of credit extended to customers to
        the amount of sales generated.

        The receivables turnover ratio is an important measure of how well a company manages its
        accounts receivable. It indicates how quickly a company collects payments from its customers.
        A higher turnover ratio is generally favorable as it suggests that the company is collecting
        payments more quickly, which improves its cash flow and working capital management.
        Revenue is used in place of net credit sales, which companies rarely disclose.

        The formula is as follows:

        - Receivables Turnover Ratio = Revenue / Average Accounts Receivable

        Also known as: receivables efficiency, accounts receivable turnover.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).
        Returns:
            pd.DataFrame: Receivables turnover ratio values.

        Notes:
        - The method retrieves historical data and calculates the receivables turnover ratio for each
          asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["MDT", "ABT", "SYK"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_receivables_turnover()
        ```

        Which returns:

        |     |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:----|-------:|-------:|-------:|-------:|-------:|
        | ABT | 6.6778 | 6.8718 | 6.2754 | 6.2194 | 5.9685 |
        | MDT | 5.7543 | 5.4077 | 5.338  | 5.3052 | 5.5273 |
        | SYK | 5.9787 | 5.6016 | 5.5929 | 5.8295 | 6.2587 |
        """
        if trailing:
            receivables_turnover = efficiency_model.get_receivables_turnover(
                self._mean_trailing(
                    self._balance_sheet_statement.loc[:, "Accounts Receivable", :],
                    trailing,
                ),
                self._sum_trailing(
                    self._income_statement.loc[:, "Revenue", :], trailing
                ),
            )
        else:
            receivables_turnover = efficiency_model.get_receivables_turnover(
                self._balance_sheet_statement.loc[:, "Accounts Receivable", :]
                .T.rolling(2)
                .mean()
                .T,
                self._income_statement.loc[:, "Revenue", :],
            )

        return finalize_dataset(
            dataset=receivables_turnover,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_sga_to_revenue_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the sales, general, and administrative (SG&A) expenses to revenue ratio,
        which measures the SG&A expenses relative to the revenue of the company.

        The SG&A to revenue ratio is calculated by dividing the total SG&A expenses by the
        company's revenue, as a fraction (0.06 is 6%). It provides insight into the
        efficiency of a company's cost management and its ability to control its overhead
        costs.

        The formula is as follows:

        - SG&A to Revenue Ratio = SG&A Expenses / Revenue

        Also known as: SG&A ratio, selling general administrative ratio.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.DataFrame: SG&A to revenue ratio values.

        Notes:
        - The method retrieves historical data and calculates the SG&A to revenue ratio for
        each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["KO", "PEP", "MDLZ"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_sga_to_revenue_ratio()
        ```

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | KO   | 0.3069 | 0.2945 | 0.3057 | 0.3069 | 0.3029 |
        | MDLZ | 0.2181 | 0.2344 | 0.2222 | 0.2041 | 0.1861 |
        | PEP  | 0.393  | 0.3989 | 0.401  | 0.4049 | 0.3978 |
        """
        if trailing:
            sga_to_revenue_ratio = efficiency_model.get_sga_to_revenue_ratio(
                self._sum_trailing(
                    self._income_statement.loc[
                        :, "Selling, General and Administrative Expenses", :
                    ],
                    trailing,
                ),
                self._sum_trailing(
                    self._income_statement.loc[:, "Revenue", :], trailing
                ),
            )
        else:
            sga_to_revenue_ratio = efficiency_model.get_sga_to_revenue_ratio(
                self._income_statement.loc[
                    :, "Selling, General and Administrative Expenses", :
                ],
                self._income_statement.loc[:, "Revenue", :],
            )

        return finalize_dataset(
            dataset=sga_to_revenue_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_fixed_asset_turnover(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Fixed Asset Turnover ratio, an efficiency ratio that
        measures how efficiently a company uses its fixed assets to generate sales.

        The Fixed Asset Turnover ratio is calculated by dividing the company's revenue by
        its average net property, plant and equipment (PP&E after accumulated depreciation),
        as defined by the CFA Institute. It indicates how well a company is utilizing its
        physical capital to generate revenue. A higher ratio suggests more efficient
        utilization of these assets. Goodwill, intangible assets and long-term investments
        are left out, so acquisitions do not dilute the ratio. Because net PP&E shrinks as
        assets depreciate, a company with an old asset base shows a higher ratio than one
        that has just invested, and leased assets only count when they are capitalised.

        The formula is as follows:

        - Fixed Asset Turnover Ratio = Revenue / Average Net Property, Plant and Equipment

        Also known as: PP&E turnover, fixed asset utilization.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.DataFrame: Fixed Asset Turnover ratio values.

        Notes:
        - The method retrieves historical data and calculates the Fixed Asset Turnover ratio
        for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["DAL", "UAL", "LUV"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_fixed_asset_turnover()
        ```

        Which returns:

        |     |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:----|-------:|-------:|-------:|-------:|-------:|
        | DAL | 0.877  | 1.3299 | 1.4049 | 1.4215 | 1.4046 |
        | LUV | 0.9246 | 1.3543 | 1.3266 | 1.3399 | 1.344  |
        | UAL | 0.6775 | 1.1979 | 1.3091 | 1.2617 | 1.2002 |
        """
        if trailing:
            fixed_asset_turnover = efficiency_model.get_fixed_asset_turnover(
                self._sum_trailing(
                    self._income_statement.loc[:, "Revenue", :], trailing
                ),
                self._mean_trailing(
                    self._balance_sheet_statement.loc[
                        :, "Property, Plant and Equipment", :
                    ],
                    trailing,
                ),
            )
        else:
            fixed_asset_turnover = efficiency_model.get_fixed_asset_turnover(
                self._income_statement.loc[:, "Revenue", :],
                self._balance_sheet_statement.loc[:, "Property, Plant and Equipment", :]
                .T.rolling(2)
                .mean()
                .T,
            )

        return finalize_dataset(
            dataset=fixed_asset_turnover,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_working_capital_turnover_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the working capital turnover ratio, an efficiency ratio that
        measures how effectively a company uses its working capital to generate
        revenue.

        A high working capital turnover ratio indicates that a company is generating a
        large amount of revenue relative to the working capital it employs, which can
        signal an efficient (or, if extreme, undercapitalized) operation. A low ratio
        can indicate excess inventory, slow receivables collection, or otherwise
        underutilized working capital.

        The formula is as follows:

        - Working Capital Turnover Ratio = Revenue / Average Working Capital

        The ratio only reads well when working capital is clearly positive. When current
        assets and current liabilities are close, a small change in either swings the
        ratio to very large or negative values, which says little about efficiency, so
        check `get_working_capital` alongside it. Companies that run on negative working
        capital, such as retailers paid by their customers before they pay their
        suppliers, show a negative ratio that reflects their business model rather than
        weak efficiency.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.DataFrame: Working capital turnover ratio values.

        Notes:
        - The method retrieves historical data and calculates the working capital
        turnover ratio for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio
        values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["FAST", "GWW", "MSM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_working_capital_turnover_ratio()
        ```

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | FAST | 2.9601 | 3.096  | 3.1299 | 3.0898 | 3.1055 |
        | GWW  | 5.2497 | 5.5883 | 5.1413 | 4.9943 | 5.1439 |
        | MSM  | 4.1018 | 4.7031 | 5.397  | 6.1099 | 6.9814 |
        """
        working_capital = liquidity_model.get_working_capital(
            self._balance_sheet_statement.loc[:, "Total Current Assets", :],
            self._balance_sheet_statement.loc[:, "Total Current Liabilities", :],
        )

        if trailing:
            working_capital_turnover_ratio = (
                efficiency_model.get_working_capital_turnover_ratio(
                    self._sum_trailing(
                        self._income_statement.loc[:, "Revenue", :], trailing
                    ),
                    self._mean_trailing(working_capital, trailing),
                )
            )
        else:
            working_capital_turnover_ratio = (
                efficiency_model.get_working_capital_turnover_ratio(
                    self._income_statement.loc[:, "Revenue", :],
                    working_capital.T.rolling(2).mean().T,
                )
            )

        return finalize_dataset(
            dataset=working_capital_turnover_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_operating_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the operating ratio, a financial metric that measures the efficiency
        of a company's operations by comparing its operating expenses to its revenue.

        The operating ratio is calculated by dividing the sum of the cost of goods sold
        and the operating expenses by revenue. It shows how much of each dollar of
        revenue is absorbed by the costs of running the business; one minus the operating
        ratio is roughly the operating margin. It provides insight into how efficiently a
        company is managing its operations.

        The formula is as follows:

        - Operating Ratio = (Operating Expenses + Cost of Goods Sold) / Revenue

        Railroads and trucking companies report the operating ratio as their headline
        efficiency measure, and lower is better. Because cost of goods sold and operating
        expenses are added together, the ratio does not depend on how a company splits
        its costs between those two lines, which makes it easier to compare than the
        expense lines on their own. It leaves out interest and taxes, so it says nothing
        about leverage.

        Also known as: operating efficiency ratio, operating expense ratio.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.DataFrame: Operating ratio values.

        Notes:
        - The method retrieves historical data and calculates the operating ratio for each
        asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["UNP", "CSX", "NSC"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_operating_ratio()
        ```

        Which returns:

        |     |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:----|-------:|-------:|-------:|-------:|-------:|
        | CSX | 0.588  | 0.5991 | 0.627  | 0.6309 | 0.6792 |
        | NSC | 0.6082 | 0.6286 | 0.769  | 0.6895 | 0.6709 |
        | UNP | 0.5717 | 0.6013 | 0.6235 | 0.5995 | 0.5986 |
        """
        if trailing:
            operating_ratio = efficiency_model.get_operating_ratio(
                self._sum_trailing(
                    self._income_statement.loc[:, "Operating Expenses", :], trailing
                ),
                self._sum_trailing(
                    self._income_statement.loc[:, "Cost of Goods Sold", :], trailing
                ),
                self._sum_trailing(
                    self._income_statement.loc[:, "Revenue", :], trailing
                ),
            )
        else:
            operating_ratio = efficiency_model.get_operating_ratio(
                self._income_statement.loc[:, "Operating Expenses", :],
                self._income_statement.loc[:, "Cost of Goods Sold", :],
                self._income_statement.loc[:, "Revenue", :],
            )

        return finalize_dataset(
            dataset=operating_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_research_and_development_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the research and development (R&D) intensity ratio, an efficiency
        ratio that measures how much a company reinvests in research and development
        relative to its revenue.

        This ratio is particularly relevant for comparing companies in technology,
        pharmaceutical and other innovation-driven industries, where R&D spending is
        a key driver of future growth.

        The formula is as follows:

        - R&D Intensity Ratio = Research and Development Expenses / Revenue

        R&D intensity is compared within an industry and over time: a falling ratio
        can mean revenue is growing faster than the research budget, not that
        research is being cut. Accounting matters across borders, since IFRS requires
        development costs to be capitalized once a project meets certain criteria
        while US GAAP expenses nearly all of them, so a company reporting under IFRS
        can show a lower ratio for the same spending.

        Also known as: R&D intensity, R&D to sales ratio.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.DataFrame: R&D intensity ratio values.

        Notes:
        - The method retrieves historical data and calculates the R&D intensity ratio for
        each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["NVDA", "AMD", "INTC"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_research_and_development_ratio()
        ```

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | AMD  | 0.1731 | 0.2121 | 0.2589 | 0.2504 | 0.2336 |
        | INTC | 0.1922 | 0.278  | 0.2959 | 0.3116 | 0.2606 |
        | NVDA | 0.1957 | 0.2721 | 0.1424 | 0.099  | 0.0857 |
        """
        if trailing:
            rd_ratio = efficiency_model.get_research_and_development_ratio(
                self._sum_trailing(
                    self._income_statement.loc[
                        :, "Research and Development Expenses", :
                    ],
                    trailing,
                ),
                self._sum_trailing(
                    self._income_statement.loc[:, "Revenue", :], trailing
                ),
            )
        else:
            rd_ratio = efficiency_model.get_research_and_development_ratio(
                self._income_statement.loc[:, "Research and Development Expenses", :],
                self._income_statement.loc[:, "Revenue", :],
            )

        return finalize_dataset(
            dataset=rd_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_selling_and_marketing_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the selling and marketing (S&M) expenses to revenue ratio, an
        efficiency ratio that measures the proportion of revenue spent on selling and
        marketing activities.

        This ratio isolates the selling and marketing component of the combined SG&A
        expense line (see `get_sga_to_revenue_ratio`), which is useful for comparing
        customer-acquisition efficiency independently of administrative overhead.

        The formula is as follows:

        - S&M to Revenue Ratio = Selling and Marketing Expenses / Revenue

        Also known as: S&M ratio, sales and marketing intensity.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.DataFrame: S&M to revenue ratio values.

        Notes:
        - The method retrieves historical data and calculates the S&M to revenue ratio for
        each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.
        - Not every company reports Selling and Marketing Expenses separately from
        General and Administrative Expenses. For those companies the ratio shows 0,
        which means the split is not reported, not that nothing was spent; use
        `get_sga_to_revenue_ratio` instead.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["CRM", "HUBS", "DDOG"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_selling_and_marketing_ratio()
        ```

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | CRM  | 0.4475 | 0.4314 | 0.3694 | 0.3498 | 0.3455 |
        | DDOG | 0.2911 | 0.2957 | 0.2863 | 0.2819 | 0.2791 |
        | HUBS | 0.4995 | 0.5119 | 0.4924 | 0.4639 | 0.4405 |
        """
        if trailing:
            sm_ratio = efficiency_model.get_selling_and_marketing_ratio(
                self._sum_trailing(
                    self._income_statement.loc[:, "Selling and Marketing Expenses", :],
                    trailing,
                ),
                self._sum_trailing(
                    self._income_statement.loc[:, "Revenue", :], trailing
                ),
            )
        else:
            sm_ratio = efficiency_model.get_selling_and_marketing_ratio(
                self._income_statement.loc[:, "Selling and Marketing Expenses", :],
                self._income_statement.loc[:, "Revenue", :],
            )

        return finalize_dataset(
            dataset=sm_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_general_and_administrative_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the general and administrative (G&A) expenses to revenue ratio, an
        efficiency ratio that measures the proportion of revenue spent on general and
        administrative overhead.

        This ratio isolates the administrative component of the combined SG&A expense
        line (see `get_sga_to_revenue_ratio`), which is useful for assessing overhead
        efficiency independently of selling and marketing spend.

        The formula is as follows:

        - G&A to Revenue Ratio = General and Administrative Expenses / Revenue

        Also known as: G&A ratio, overhead ratio.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.DataFrame: G&A to revenue ratio values.

        Notes:
        - The method retrieves historical data and calculates the G&A to revenue ratio for
        each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.
        - Not every company reports General and Administrative Expenses separately
        from Selling and Marketing Expenses. For those companies the ratio shows 0,
        or, when the whole SG&A line is reported as general and administrative, the
        full SG&A ratio; compare with `get_sga_to_revenue_ratio` before reading it as
        overhead alone.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["WDAY", "SNOW", "INTU"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_general_and_administrative_ratio()
        ```

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | INTU | 0.1019 | 0.1147 | 0.0905 | 0.0871 | 0.085  |
        | SNOW | 0.2174 | 0.1432 | 0.1151 | 0.1137 | 0.1174 |
        | WDAY | 0.0946 | 0.0964 | 0.0967 | 0.0971 | 0.1401 |
        """
        if trailing:
            ga_ratio = efficiency_model.get_general_and_administrative_ratio(
                self._sum_trailing(
                    self._income_statement.loc[
                        :, "General and Administrative Expenses", :
                    ],
                    trailing,
                ),
                self._sum_trailing(
                    self._income_statement.loc[:, "Revenue", :], trailing
                ),
            )
        else:
            ga_ratio = efficiency_model.get_general_and_administrative_ratio(
                self._income_statement.loc[:, "General and Administrative Expenses", :],
                self._income_statement.loc[:, "Revenue", :],
            )

        return finalize_dataset(
            dataset=ga_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_stock_based_compensation_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the stock-based compensation (SBC) to revenue ratio, an efficiency
        ratio that measures how much of a company's revenue is being used to
        compensate employees through non-cash equity awards.

        A high or rising SBC-to-revenue ratio is a common quality-of-earnings flag,
        particularly for technology companies, since SBC is added back in cash flow
        from operations but represents real economic dilution for shareholders.

        The formula is as follows:

        - SBC to Revenue Ratio = Stock Based Compensation / Revenue

        Also known as: SBC intensity, equity compensation ratio.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.DataFrame: SBC to revenue ratio values.

        Notes:
        - The method retrieves historical data and calculates the SBC to revenue ratio for
        each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["UBER", "LYFT", "DASH"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_stock_based_compensation_ratio()
        ```

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | DASH | 0.0994 | 0.135  | 0.126  | 0.1025 | 0.0766 |
        | LYFT | 0.2258 | 0.1833 | 0.11   | 0.0572 | 0.051  |
        | UBER | 0.0669 | 0.0562 | 0.0519 | 0.0408 | 0.0351 |
        """
        if trailing:
            sbc_ratio = efficiency_model.get_stock_based_compensation_ratio(
                self._sum_trailing(
                    self._cash_flow_statement.loc[:, "Stock Based Compensation", :],
                    trailing,
                ),
                self._sum_trailing(
                    self._income_statement.loc[:, "Revenue", :], trailing
                ),
            )
        else:
            sbc_ratio = efficiency_model.get_stock_based_compensation_ratio(
                self._cash_flow_statement.loc[:, "Stock Based Compensation", :],
                self._income_statement.loc[:, "Revenue", :],
            )

        return finalize_dataset(
            dataset=sbc_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_deferred_revenue_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the deferred revenue ratio, an efficiency ratio that measures the
        size of a company's deferred revenue (payments collected for goods or
        services not yet delivered) relative to its revenue.

        This ratio is a common leading indicator for subscription and SaaS
        businesses, where a growing deferred revenue balance relative to revenue can
        signal accelerating future revenue recognition.

        The formula is as follows:

        - Deferred Revenue Ratio = Deferred Revenue / Revenue

        Without trailing, the deferred revenue balance at the end of the period is used, so
        the ratio shows the backlog of prepaid revenue at that date. With trailing, the
        balance is averaged over the `trailing` period ends to match the summed revenue.

        Also known as: deferred revenue intensity, unearned revenue ratio.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.DataFrame: Deferred revenue ratio values.

        Notes:
        - The method retrieves historical data and calculates the deferred revenue ratio for
        each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.
        - Not every company reports Deferred Revenue on its Balance Sheet, in which case this
        ratio will be unavailable.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["ORCL", "ADBE", "WDAY"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_deferred_revenue_ratio()
        ```

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | ADBE | 0.2998 | 0.3009 | 0.3007 | 0.2851 | 0.2905 |
        | ORCL | 0.1969 | 0.1796 | 0.1758 | 0.1635 | 0.1472 |
        | WDAY | 0.6054 | 0.5726 | 0.5589 | 0.5289 | 0.5245 |
        """
        if trailing:
            deferred_revenue_ratio = efficiency_model.get_deferred_revenue_ratio(
                self._mean_trailing(
                    self._balance_sheet_statement.loc[:, "Deferred Revenue", :],
                    trailing,
                ),
                self._sum_trailing(
                    self._income_statement.loc[:, "Revenue", :], trailing
                ),
            )
        else:
            deferred_revenue_ratio = efficiency_model.get_deferred_revenue_ratio(
                self._balance_sheet_statement.loc[:, "Deferred Revenue", :],
                self._income_statement.loc[:, "Revenue", :],
            )

        return finalize_dataset(
            dataset=deferred_revenue_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_capex_intensity(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the capital expenditure intensity, the share of revenue a company
        reinvests in property, plant and equipment.

        Capex intensity shows how capital-heavy a business is: utilities, telecoms,
        semiconductor manufacturers and energy producers spend a large part of their
        revenue on capital expenditure, software and services companies very little. Over
        time a rising intensity points to an investment cycle (capacity being built ahead
        of revenue), a falling one to a business that is harvesting its existing assets.
        Damodaran uses it, together with the capex to depreciation ratio, to judge how much
        a company has to reinvest to grow.

        The formula is as follows:

        - Capex Intensity = |Capital Expenditure| / Revenue

        Capital expenditure is reported as a negative cash outflow, so its magnitude is used.
        A revenue of zero or less gives NaN.

        Also known as: capex to revenue, capex to sales, capex ratio, capital spending intensity.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over. E.g. 4 with quarterly data gives the trailing twelve
                months (TTM) and 3 with annual data a three-year cumulative figure. Defaults to
                None, which uses single-period flows.

        Returns:
            pd.DataFrame: Capex intensity values.

        Notes:
        - Damodaran, A., "Investment Valuation," 3rd edition, Wiley, 2012 (reinvestment and
        net capital expenditure).
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["TSM", "INTC", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_capex_intensity()
        ```

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | INTC | 0.2573 | 0.3973 | 0.4748 | 0.4509 | 0.2771 |
        | MSFT | 0.1227 | 0.1205 | 0.1326 | 0.1814 | 0.2291 |
        | TSM  | 0.5351 | 0.4813 | 0.442  | 0.3303 | 0.3367 |
        """
        capex_intensity = efficiency_model.get_capex_intensity(
            self._sum_trailing(
                self._cash_flow_statement.loc[:, "Capital Expenditure", :], trailing
            ),
            self._sum_trailing(self._income_statement.loc[:, "Revenue", :], trailing),
        )

        return finalize_dataset(
            dataset=capex_intensity,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_capex_to_depreciation_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the capital expenditure to depreciation ratio, which compares what a
        company invests in its assets with what wears off them.

        Depreciation and amortization approximate the investment needed to keep the asset
        base as it is, so a ratio above 1 means the company is growing its assets, a ratio
        around 1 that it is maintaining them, and a ratio persistently below 1 that it is
        underinvesting, which flatters the free cash flow for a while but erodes the
        business. Damodaran's net capital expenditure (capital expenditure minus
        depreciation) is the same comparison as a difference. Acquisitions are not capital
        expenditure, so a company that grows by buying others can show a low ratio while
        its assets grow.

        The formula is as follows:

        - Capex to Depreciation = |Capital Expenditure| / Depreciation and Amortization

        Capital expenditure is reported as a negative cash outflow, so its magnitude is used.
        Depreciation and amortization of zero or less gives NaN.

        Also known as: capex to D&A, capex to depreciation, reinvestment ratio, maintenance capex ratio.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (cash flow statement
                items) over. E.g. 4 with quarterly data gives the trailing twelve months (TTM)
                and 3 with annual data a three-year cumulative figure. Defaults to None, which
                uses single-period flows.

        Returns:
            pd.DataFrame: Capex to depreciation values.

        Notes:
        - Damodaran, A., "Investment Valuation," 3rd edition, Wiley, 2012 (net capital
        expenditure and reinvestment).
        - Depreciation and amortization include the amortization of acquired intangibles,
        which is no wear of the assets that capital expenditure replaces, so the ratio is low
        for companies with large acquired intangibles.
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["TSM", "INTC", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_capex_to_depreciation_ratio()
        ```

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | INTC | 1.724  | 1.9217 | 2.6817 | 2.1042 | 1.2512 |
        | MSFT | 1.7647 | 1.6519 | 2.0278 | 1.9956 | 1.8901 |
        | TSM  | 2.011  | 2.492  | 1.7952 | 1.4424 | 1.8639 |
        """
        capex_to_depreciation = efficiency_model.get_capex_to_depreciation_ratio(
            self._sum_trailing(
                self._cash_flow_statement.loc[:, "Capital Expenditure", :], trailing
            ),
            self._sum_trailing(
                self._cash_flow_statement.loc[:, "Depreciation and Amortization", :],
                trailing,
            ),
        )

        return finalize_dataset(
            dataset=capex_to_depreciation,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_asset_growth(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the asset growth of Cooper, Gulen and Schill (2008): the change in total
        assets over a year relative to the total assets at the start of it.

        Cooper, Gulen and Schill find that the growth of a company's total assets predicts
        its returns strongly and negatively: the companies that grow their assets the most
        earn far lower returns over the following years than those that grow them the
        least, more so than the book-to-market ratio, size or past returns predict. Asset
        growth captures every form of investment (capital expenditure, acquisitions,
        working capital) and of the financing behind it, and is the investment measure of
        the CMA (conservative minus aggressive) factor of the Fama-French five-factor model.

        The formula is as follows:

        - Asset Growth = (Total Assets - Beginning Total Assets) / Beginning Total Assets

        The total assets are those at the end of the period and the beginning total assets
        those a year earlier: the end of the year before with annual data and four quarters
        back with quarterly data, so quarterly values are year over year. Beginning total
        assets of zero or less give NaN.

        Also known as: total asset growth, investment, CMA investment, ASSETG.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods the total assets are compared
                over. E.g. 3 with annual data gives the growth over three years. Defaults to
                None, which uses a year: 4 quarters with quarterly data and 1 year with annual
                data.

        Returns:
            pd.DataFrame: Asset growth values.

        Notes:
        - Cooper, M. J., H. Gulen and M. J. Schill, "Asset Growth and the Cross-Section of
        Stock Returns," Journal of Finance 63(4), 2008, pp. 1609-1651. Fama, E. F. and K. R.
        French, "A Five-Factor Asset Pricing Model," Journal of Financial Economics 116(1),
        2015, pp. 1-22.
        - The first period has no total assets a year earlier and is NaN, so set an earlier
        start_date to obtain it.
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "NVDA", "XOM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_asset_growth()
        ```

        Which returns:

        |      |   2021 |    2022 |    2023 |   2024 |     2025 |
        |:-----|-------:|--------:|--------:|-------:|---------:|
        | AAPL | 0.0837 |  0.005  | -0.0005 | 0.0352 |  -0.0157 |
        | MSFT | 0.1078 |  0.0931 |  0.1292 | 0.2432 |   0.2086 |
        | NVDA | 0.5348 | -0.068  |  0.596  | 0.6979 | nan      |
        | XOM  | 0.0186 |  0.0889 |  0.0196 | 0.205  |  -0.0099 |
        """
        # A year back by default: four quarters on quarterly data, so the growth is year over year.
        window = self._get_valuation_trailing(trailing) or 1

        total_assets = self._balance_sheet_statement.loc[:, "Total Assets", :]

        asset_growth = efficiency_model.get_asset_growth(
            total_assets, total_assets.shift(window, axis=1)
        )

        return finalize_dataset(
            dataset=asset_growth,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_net_operating_assets(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
    ) -> pd.DataFrame:
        """
        Calculate the net operating assets of Hirshleifer, Hou, Teoh and Zhang (2004): the
        operating assets minus the operating liabilities, scaled by the total assets at the
        start of the period.

        Net operating assets are the cumulative difference between the operating income
        and the free cash flow a company has reported over the years, so a high value means
        its earnings have run ahead of its cash for a long time (a bloated balance sheet).
        Hirshleifer et al. show that investors overlook this, so that companies with high
        net operating assets earn lower earnings growth and lower returns over the
        following years than those with low net operating assets.

        The formula is as follows:

        - Operating Assets = Total Assets - Cash and Short Term Investments
        - Operating Liabilities = Total Assets - Short Term Debt - Long Term Debt - Total Equity
        - Net Operating Assets = (Operating Assets - Operating Liabilities) / Beginning Total Assets

        The paper deducts the debt in current liabilities, the long term debt, the minority
        interest, the preferred stock and the common equity from the total assets for the
        operating liabilities; the total equity here holds the last three. The debt is the
        short and long term debt rows of the statements, without separately reported lease
        liabilities. The cash is the cash and cash equivalents plus the short term
        investments. The balances are those at the end of
        the period and the beginning total assets those at the end of the period before
        (the year before with annual data, the quarter before with quarterly data, as in
        the quarterly version of Hou, Xue and Zhang). Beginning total assets of zero or less
        give NaN.

        Also known as: NOA, scaled net operating assets, balance sheet bloat.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.DataFrame: Net operating assets values.

        Notes:
        - Hirshleifer, D., K. Hou, S. H. Teoh and Y. Zhang, "Do Investors Overvalue Firms with
        Bloated Balance Sheets?," Journal of Accounting and Economics 38, 2004, pp. 297-331.
        Hou, K., C. Xue and L. Zhang, "Replicating Anomalies," Review of Financial Studies
        33(5), 2020, pp. 2019-2133.
        - The first period has no total assets of the period before and is NaN, so set an
        earlier start_date to obtain it.
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "KO", "XOM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_net_operating_assets()
        ```

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | AAPL | 0.3865 | 0.3488 | 0.3166 | 0.2791 | 0.3225 |
        | KO   | 0.6335 | 0.5654 | 0.6024 | 0.5765 | 0.6554 |
        | MSFT | 0.2319 | 0.3343 | 0.3898 | 0.5937 | 0.5703 |
        | XOM  | 0.6453 | 0.6218 | 0.5927 | 0.7582 | 0.6604 |
        """

        def balance(row: str) -> pd.DataFrame:
            return self._balance_sheet_statement.loc[:, row, :]

        net_operating_assets = efficiency_model.get_net_operating_assets(
            balance("Total Assets"),
            self._get_cash_and_short_term_investments(),
            balance("Short Term Debt"),
            balance("Long Term Debt"),
            balance("Total Equity"),
            balance("Total Assets").shift(1, axis=1),
        )

        return finalize_dataset(
            dataset=net_operating_assets,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_errors
    def collect_liquidity_ratios(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculates and collects all Liquidity Ratios based on the provided data.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.DataFrame: Liquidity ratios calculated based on the specified parameters.

        Notes:
        - The method calculates various liquidity ratios for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "TSLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        liquidity_ratios = toolkit.ratios.collect_liquidity_ratios()

        liquidity_ratios.loc['AAPL']
        ```

        Which returns:

        |                                    |        2021 |         2022 |        2023 |         2024 |         2025 |
        |:-----------------------------------|------------:|-------------:|------------:|-------------:|-------------:|
        | Current Ratio                      |   1.0746    |   0.8794     |   0.988     |   0.8673     |   0.8933     |
        | Quick Ratio                        |   0.7086    |   0.4967     |   0.6267    |   0.5589     |   0.5704     |
        | Cash Ratio                         |   0.4992    |   0.3137     |   0.4236    |   0.3695     |   0.3302     |
        | Working Capital                    |   9.355e+09 |  -1.8577e+10 |  -1.742e+09 |  -2.3405e+10 |  -1.7674e+10 |
        | Operating Cash Flow Ratio          |   0.8291    |   0.7933     |   0.7607    |   0.6704     |   0.6731     |
        | Operating Cash Flow to Sales Ratio |   0.2844    |   0.3098     |   0.2884    |   0.3024     |   0.2679     |
        | Short Term Coverage Ratio          |   6.6635    |   5.7864     |   6.9933    |   5.6638     |   5.4839     |
        | Defensive Interval Ratio           | 132.153     | 105.836      | 129.097     | 140.35       | 127.05       |
        """
        self._load_statements()
        liquidity_ratios_dict: dict = {}

        liquidity_ratios_dict["Current Ratio"] = self.get_current_ratio(
            trailing=trailing
        )
        liquidity_ratios_dict["Quick Ratio"] = self.get_quick_ratio(trailing=trailing)
        liquidity_ratios_dict["Cash Ratio"] = self.get_cash_ratio(trailing=trailing)
        liquidity_ratios_dict["Working Capital"] = self.get_working_capital(
            trailing=trailing
        )
        liquidity_ratios_dict["Operating Cash Flow Ratio"] = (
            self.get_operating_cash_flow_ratio(trailing=trailing)
        )
        liquidity_ratios_dict["Operating Cash Flow to Sales Ratio"] = (
            self.get_operating_cash_flow_sales_ratio(trailing=trailing)
        )
        liquidity_ratios_dict["Short Term Coverage Ratio"] = (
            self.get_short_term_coverage_ratio(trailing=trailing)
        )
        liquidity_ratios_dict["Defensive Interval Ratio"] = (
            self.get_defensive_interval_ratio(trailing=trailing)
        )

        self._liquidity_ratios = (
            pd.concat(liquidity_ratios_dict)
            .swaplevel(0, 1)
            .sort_index(level=0, sort_remaining=False)
            .dropna(axis="columns", how="all")
        )

        self._liquidity_ratios = self._liquidity_ratios.loc[self._tickers]

        self._liquidity_ratios = self._liquidity_ratios.pipe(
            apply_rounding, rounding if rounding is not None else self._rounding
        )

        # Sorted again so the index follows the financial statements' order.
        available_columns = [
            column
            for column in self._income_statement.columns
            if column in self._liquidity_ratios
        ]
        self._liquidity_ratios = self._liquidity_ratios.reindex(
            available_columns, axis=1
        )

        if growth:
            self._liquidity_ratios_growth = calculate_growth(
                dataset=self._liquidity_ratios,
                lag=lag,
                compound=compound,
                rounding=rounding if rounding is not None else self._rounding,
                axis="columns",
            )

        liquidity_ratios = self._liquidity_ratios
        liquidity_ratios_growth = self._liquidity_ratios_growth

        if standardize:
            standardize_rounding = rounding if rounding is not None else self._rounding
            if growth:
                liquidity_ratios_growth = calculate_standardization(
                    dataset=liquidity_ratios_growth,
                    rounding=standardize_rounding,
                    axis="columns",
                )
            else:
                liquidity_ratios = calculate_standardization(
                    dataset=liquidity_ratios,
                    rounding=standardize_rounding,
                    axis="columns",
                )

        if len(self._tickers) == 1:
            return (
                liquidity_ratios_growth.loc[self._tickers[0]]
                if growth
                else liquidity_ratios.loc[self._tickers[0]]
            ).loc[:, self._start_date : self._end_date]

        return (liquidity_ratios_growth if growth else liquidity_ratios).loc[
            :, self._start_date : self._end_date
        ]

    @handle_portfolio
    @handle_errors
    def get_current_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the current ratio, a liquidity ratio that measures a company's ability
        to pay off its short-term liabilities with its current assets.

        The current ratio is calculated by dividing a company's current assets by its
        current liabilities. It indicates whether a company can meet its short-term
        obligations using its short-term assets.

        The formula is as follows:

        - Current Ratio = Current Assets / Current Liabilities

        A current ratio above 1 is the textbook comfort level, but the right level
        depends on the business: companies that collect from customers before paying
        suppliers, such as large retailers and restaurant chains, run below 1 without
        strain, while manufacturers with long production cycles need more. Lenders read
        it together with the quick ratio (`get_quick_ratio`), since inventory can be hard
        to sell at book value, and with the trend, since the year-end balance sheet can
        be managed to look better than the average day.

        Also known as: short-term liquidity, working capital ratio.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.DataFrame: Current ratio values.

        Notes:
        - The method retrieves historical data and calculates the current ratio for each
        asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["PFE", "MRK", "BMY"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_current_ratio()
        ```

        Which returns:

        |     |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:----|-------:|-------:|-------:|-------:|-------:|
        | BMY | 1.521  | 1.2459 | 1.4271 | 1.2526 | 1.2551 |
        | MRK | 1.2678 | 1.4737 | 1.252  | 1.3646 | 1.5362 |
        | PFE | 1.3989 | 1.2165 | 0.9067 | 1.1713 | 1.1599 |
        """
        if trailing:
            current_ratio = liquidity_model.get_current_ratio(
                self._mean_trailing(
                    self._balance_sheet_statement.loc[:, "Total Current Assets", :],
                    trailing,
                ),
                self._mean_trailing(
                    self._balance_sheet_statement.loc[
                        :, "Total Current Liabilities", :
                    ],
                    trailing,
                ),
            )
        else:
            current_ratio = liquidity_model.get_current_ratio(
                self._balance_sheet_statement.loc[:, "Total Current Assets", :],
                self._balance_sheet_statement.loc[:, "Total Current Liabilities", :],
            )

        return finalize_dataset(
            dataset=current_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_quick_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the quick ratio (also known as the acid-test ratio), a more stringent
        measure of liquidity that excludes inventory from current assets.

        This ratio is also referred to as the Acid Test Ratio.

        The quick ratio is calculated by adding cash and cash equivalents, short-term
        investments and accounts receivable, and dividing the sum by current liabilities.
        It leaves out inventory, prepaid expenses and other current assets, which cannot
        quickly be turned into cash. It provides insight into a company's ability to
        cover its short-term liabilities using its most liquid assets without relying on
        inventory.

        The formula is as follows:

        - Quick Ratio = (Cash and Cash Equivalents + Short Term Investments + Accounts Receivable) / Current Liabilities

        Also known as: acid-test ratio, liquid ratio.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.DataFrame: Quick ratio values.

        Notes:
        - The method retrieves historical data and calculates the quick ratio for each
        asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["BA", "LMT", "RTX"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_quick_ratio()
        ```

        Which returns:

        |     |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:----|-------:|-------:|-------:|-------:|-------:|
        | BA  | 0.2303 | 0.2191 | 0.194  | 0.2978 | 0.2969 |
        | LMT | 0.3977 | 0.318  | 0.211  | 0.2489 | 0.3438 |
        | RTX | 0.4935 | 0.3919 | 0.3726 | 0.3214 | 0.3766 |
        """
        if trailing:
            quick_ratio = liquidity_model.get_quick_ratio(
                self._mean_trailing(
                    self._balance_sheet_statement.loc[
                        :, "Cash and Cash Equivalents", :
                    ],
                    trailing,
                ),
                self._mean_trailing(
                    self._balance_sheet_statement.loc[:, "Short Term Investments", :],
                    trailing,
                ),
                self._mean_trailing(
                    self._balance_sheet_statement.loc[:, "Accounts Receivable", :],
                    trailing,
                ),
                self._mean_trailing(
                    self._balance_sheet_statement.loc[
                        :, "Total Current Liabilities", :
                    ],
                    trailing,
                ),
            )
        else:
            quick_ratio = liquidity_model.get_quick_ratio(
                self._balance_sheet_statement.loc[:, "Cash and Cash Equivalents", :],
                self._balance_sheet_statement.loc[:, "Short Term Investments", :],
                self._balance_sheet_statement.loc[:, "Accounts Receivable", :],
                self._balance_sheet_statement.loc[:, "Total Current Liabilities", :],
            )

        return finalize_dataset(
            dataset=quick_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_cash_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the cash ratio, a liquidity ratio that measures a company's ability
        to pay off its short-term liabilities with its cash and cash equivalents.

        The cash ratio is calculated by dividing the sum of cash and cash equivalents and
        short-term investments by current liabilities. It provides insight into a
        company's immediate ability to cover its short-term obligations using its most
        liquid assets.

        The formula is as follows:

        - Cash Ratio = (Cash and Cash Equivalents + Short Term Investments) / Current Liabilities

        Also known as: most conservative liquidity ratio.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.DataFrame: Cash ratio values.

        Notes:
        - The method retrieves historical data and calculates the cash ratio for each
        asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["GOOGL", "META", "NFLX"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_cash_ratio()
        ```

        Which returns:

        |       |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:------|-------:|-------:|-------:|-------:|-------:|
        | GOOGL | 2.1734 | 1.6416 | 1.3557 | 1.0733 | 1.2345 |
        | META  | 2.271  | 1.5074 | 2.0464 | 2.3162 | 1.9503 |
        | NFLX  | 0.7101 | 0.7639 | 0.8056 | 0.8911 | 0.8253 |
        """
        if trailing:
            cash_ratio = liquidity_model.get_cash_ratio(
                self._mean_trailing(
                    self._balance_sheet_statement.loc[
                        :, "Cash and Cash Equivalents", :
                    ],
                    trailing,
                ),
                self._mean_trailing(
                    self._balance_sheet_statement.loc[:, "Short Term Investments", :],
                    trailing,
                ),
                self._mean_trailing(
                    self._balance_sheet_statement.loc[
                        :, "Total Current Liabilities", :
                    ],
                    trailing,
                ),
            )
        else:
            cash_ratio = liquidity_model.get_cash_ratio(
                self._balance_sheet_statement.loc[:, "Cash and Cash Equivalents", :],
                self._balance_sheet_statement.loc[:, "Short Term Investments", :],
                self._balance_sheet_statement.loc[:, "Total Current Liabilities", :],
            )

        return finalize_dataset(
            dataset=cash_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_working_capital(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the working capital, which is the difference between a company's current assets
        and current liabilities.

        The working capital is calculated by subtracting total current liabilities from total
        current assets. It represents the company's short-term financial health and its ability
        to cover its current obligations using its liquid assets.

        The formula is as follows:

        - Working Capital = Current Assets — Current Liabilities

        Also known as: net current assets, operating liquidity.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.DataFrame: Working capital values.

        Notes:
        - The method retrieves historical data and calculates the working capital for each
        asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the working capital
        values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["TXN", "QCOM", "ADI"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_working_capital()
        ```

        Which returns:

        |      |       2021 |        2022 |        2023 |        2024 |        2025 |
        |:-----|-----------:|------------:|------------:|------------:|------------:|
        | ADI  | 2.608e+09  | 2.49534e+09 | 1.18305e+09 | 2.49637e+09 | 3.86226e+09 |
        | QCOM | 8.124e+09  | 8.858e+09   | 1.2836e+10  | 1.4727e+10  | 1.661e+10   |
        | TXN  | 1.1116e+10 | 1.1036e+10  | 1.1802e+10  | 1.1383e+10  | 1.0591e+10  |
        """
        if trailing:
            working_capital = liquidity_model.get_working_capital(
                self._mean_trailing(
                    self._balance_sheet_statement.loc[:, "Total Current Assets", :],
                    trailing,
                ),
                self._mean_trailing(
                    self._balance_sheet_statement.loc[
                        :, "Total Current Liabilities", :
                    ],
                    trailing,
                ),
            )
        else:
            working_capital = liquidity_model.get_working_capital(
                self._balance_sheet_statement.loc[:, "Total Current Assets", :],
                self._balance_sheet_statement.loc[:, "Total Current Liabilities", :],
            )

        return finalize_dataset(
            dataset=working_capital,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_operating_cash_flow_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the operating cash flow ratio, a liquidity ratio that measures a company's
        ability to pay off its current liabilities with its operating cash flow.

        The operating cash flow ratio is calculated by dividing operating cash flow by
        current liabilities. It indicates whether a company's operating cash flow is
        sufficient to cover its short-term obligations.

        The formula is as follows:

        - Operating Cash Flow Ratio = Cash Flow from Operations / Current Liabilities

        Also known as: OCF ratio, cash liquidity.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.DataFrame: Operating cash flow ratio values.

        Notes:
        - The method retrieves historical data and calculates the operating cash flow ratio for each
        asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["UNH", "CVS", "CI"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_operating_cash_flow_ratio()
        ```

        Which returns:

        |     |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:----|-------:|-------:|-------:|-------:|-------:|
        | CI  | 0.165  | 0.21   | 0.2425 | 0.1787 | 0.1704 |
        | CVS | 0.2694 | 0.233  | 0.1695 | 0.1076 | 0.12   |
        | UNH | 0.2854 | 0.2937 | 0.2935 | 0.2332 | 0.1714 |
        """
        if trailing:
            operating_cash_flow_ratio = liquidity_model.get_operating_cash_flow_ratio(
                self._sum_trailing(
                    self._cash_flow_statement.loc[:, "Cash Flow from Operations", :],
                    trailing,
                ),
                self._mean_trailing(
                    self._balance_sheet_statement.loc[
                        :, "Total Current Liabilities", :
                    ],
                    trailing,
                ),
            )
        else:
            operating_cash_flow_ratio = liquidity_model.get_operating_cash_flow_ratio(
                self._cash_flow_statement.loc[:, "Cash Flow from Operations", :],
                self._balance_sheet_statement.loc[:, "Total Current Liabilities", :],
            )

        return finalize_dataset(
            dataset=operating_cash_flow_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_operating_cash_flow_sales_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the operating cash flow to sales ratio, a liquidity ratio that
        measures the ability of a company to generate cash from its sales.

        The operating cash flow to sales ratio is calculated by dividing operating cash flow by
        sales revenue. It indicates the proportion of sales revenue that is converted into cash
        from operating activities.

        The formula is as follows:

        - Operating Cash Flow to Sales Ratio = Cash Flow from Operations / Revenue

        Also known as: cash flow to sales ratio.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.DataFrame: Operating cash flow to sales ratio values.

        Notes:
        - The method retrieves historical data and calculates the operating cash flow to sales ratio for each
        asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["V", "MA", "PYPL"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_operating_cash_flow_sales_ratio()
        ```

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | MA   | 0.5011 | 0.5034 | 0.4773 | 0.5247 | 0.5382 |
        | PYPL | 0.2285 | 0.2112 | 0.1627 | 0.2343 | 0.1934 |
        | V    | 0.6317 | 0.6431 | 0.6356 | 0.5553 | 0.5765 |
        """
        if trailing:
            operating_cash_flow_sales_ratio = (
                liquidity_model.get_operating_cash_flow_sales_ratio(
                    self._sum_trailing(
                        self._cash_flow_statement.loc[
                            :, "Cash Flow from Operations", :
                        ],
                        trailing,
                    ),
                    self._sum_trailing(
                        self._income_statement.loc[:, "Revenue", :], trailing
                    ),
                )
            )
        else:
            operating_cash_flow_sales_ratio = (
                liquidity_model.get_operating_cash_flow_sales_ratio(
                    self._cash_flow_statement.loc[:, "Cash Flow from Operations", :],
                    self._income_statement.loc[:, "Revenue", :],
                )
            )

        return finalize_dataset(
            dataset=operating_cash_flow_sales_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_short_term_coverage_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ):
        """
        Calculate the short-term coverage ratio, a liquidity ratio that measures a company's
        ability to pay off its short-term obligations with its operating cash flow.

        The short-term coverage ratio is calculated by dividing operating cash flow by short-term debt.
        It assesses the company's ability to meet its short-term obligations using its operating cash flow.

        The formula is as follows:

        - Short Term Coverage Ratio = Cash Flow from Operations / Short Term Debt

        Also known as: short-term debt coverage.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["EXC", "AEP", "D"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_short_term_coverage_ratio()
        ```

        Which returns:

        |     |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:----|-------:|-------:|-------:|-------:|-------:|
        | AEP | 0.8054 | 0.8014 | 0.942  | 1.1614 | 1.4332 |
        | D   | 1.2796 | 0.5473 | 0.5977 | 1.1877 | 1.1017 |
        | EXC | 0.8856 | 1.1098 | 1.1979 | 1.6815 | 2.2742 |
        """
        if trailing:
            short_term_coverage_ratio = liquidity_model.get_short_term_coverage_ratio(
                self._sum_trailing(
                    self._cash_flow_statement.loc[:, "Cash Flow from Operations", :],
                    trailing,
                ),
                self._mean_trailing(
                    self._balance_sheet_statement.loc[:, "Short Term Debt", :], trailing
                ),
            )
        else:
            short_term_coverage_ratio = liquidity_model.get_short_term_coverage_ratio(
                self._cash_flow_statement.loc[:, "Cash Flow from Operations", :],
                self._balance_sheet_statement.loc[:, "Short Term Debt", :],
            )

        return finalize_dataset(
            dataset=short_term_coverage_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_defensive_interval_ratio(
        self,
        days: int | float | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ):
        """
        Calculate the defensive interval ratio (DIR), a liquidity ratio that measures
        how many days a company could continue to cover its operating expenses using
        only its existing defensive (most liquid) assets, without relying on
        additional revenue.

        Unlike the current, quick, and cash ratios, which express liquidity relative
        to current liabilities, the defensive interval ratio expresses liquidity
        relative to the company's actual daily cash burn rate, making it a more direct
        measure of how long a company could survive a sudden stop in incoming cash
        flow.

        The formula is as follows:

        - Defensive Interval Ratio = (Cash and Cash Equivalents + Short Term Investments +
          Accounts Receivable) / Daily Operating Expenses

        Where Daily Operating Expenses = (Cost of Goods Sold + Operating Expenses -
        Depreciation and Amortization) / Days, the daily cash expenditures (AccountingTools,
        Wall Street Prep). The cost of goods sold is included because the depreciation and
        amortization from the cash flow statement also covers the part charged to it, and
        because paying suppliers and production staff is the largest cash cost for most
        companies. Days is the length of the window the expenses cover: 365 for annual
        data, 365 / 4 for quarterly data, times `trailing` when the expenses are summed
        over several periods, so the ratio is in calendar days for any frequency.

        Also known as: defensive interval period, basic defense interval.

        Args:
            days (int | float, optional): The number of days in the window the expenses cover, used
                to turn them into daily expenditures. Defaults to None, which uses 365 for annual data
                and 365 / 4 for quarterly data, multiplied by `trailing` when set.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.DataFrame: Defensive interval ratio values, expressed in days.

        Notes:
        - The method retrieves historical data and calculates the defensive interval
        ratio for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio
        values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AMGN", "GILD", "BIIB"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_defensive_interval_ratio()
        ```

        Which returns:

        |      |    2021 |    2022 |    2023 |    2024 |    2025 |
        |:-----|--------:|--------:|--------:|--------:|--------:|
        | AMGN | 315.9   | 406.808 | 410.687 | 332.729 | 326.593 |
        | BIIB | 273.146 | 379.204 | 158.724 | 237.852 | 289.966 |
        | GILD | 262.095 | 228.268 | 258.786 | 239.902 | 353.914 |
        """
        days = self._get_days(days, trailing)

        # Daily expenditures count all cash costs: the cost of goods sold plus the operating
        # expenses, minus the non-cash depreciation and amortization charged to either of them.
        if trailing:
            operating_expenses = self._sum_trailing(
                self._income_statement.loc[:, "Operating Expenses", :], trailing
            )
            cost_of_goods_sold = self._sum_trailing(
                self._income_statement.loc[:, "Cost of Goods Sold", :], trailing
            )
            depreciation_and_amortization = self._sum_trailing(
                self._cash_flow_statement.loc[:, "Depreciation and Amortization", :],
                trailing,
            )
            daily_operating_expenses = (
                cost_of_goods_sold + operating_expenses - depreciation_and_amortization
            ) / days

            defensive_interval_ratio = liquidity_model.get_defensive_interval_ratio(
                self._mean_trailing(
                    self._balance_sheet_statement.loc[
                        :, "Cash and Cash Equivalents", :
                    ],
                    trailing,
                ),
                self._mean_trailing(
                    self._balance_sheet_statement.loc[:, "Short Term Investments", :],
                    trailing,
                ),
                self._mean_trailing(
                    self._balance_sheet_statement.loc[:, "Accounts Receivable", :],
                    trailing,
                ),
                daily_operating_expenses,
            )
        else:
            operating_expenses = self._income_statement.loc[:, "Operating Expenses", :]
            cost_of_goods_sold = self._income_statement.loc[:, "Cost of Goods Sold", :]
            depreciation_and_amortization = self._cash_flow_statement.loc[
                :, "Depreciation and Amortization", :
            ]
            daily_operating_expenses = (
                cost_of_goods_sold + operating_expenses - depreciation_and_amortization
            ) / days

            defensive_interval_ratio = liquidity_model.get_defensive_interval_ratio(
                self._balance_sheet_statement.loc[:, "Cash and Cash Equivalents", :],
                self._balance_sheet_statement.loc[:, "Short Term Investments", :],
                self._balance_sheet_statement.loc[:, "Accounts Receivable", :],
                daily_operating_expenses,
            )

        return finalize_dataset(
            dataset=defensive_interval_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_errors
    def collect_profitability_ratios(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculates and collects all Profitability Ratios based on the provided data.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.DataFrame: Profitability ratios calculated based on the specified parameters.

        Notes:
        - The method calculates various profitability ratios for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "TSLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        profitability_ratios = toolkit.ratios.collect_profitability_ratios()

        profitability_ratios.loc['AAPL']
        ```

        Which returns:

        |                                             |    2021 |    2022 |    2023 |     2024 |     2025 |
        |:--------------------------------------------|--------:|--------:|--------:|---------:|---------:|
        | Gross Margin                                |  0.4178 |  0.4331 |  0.4413 |   0.4621 |   0.4691 |
        | Operating Margin                            |  0.2978 |  0.3029 |  0.2982 |   0.3151 |   0.3197 |
        | Net Profit Margin                           |  0.2588 |  0.2531 |  0.2531 |   0.2397 |   0.2692 |
        | EBITDA Margin                               |  0.3287 |  0.331  |  0.3283 |   0.3444 |   0.3478 |
        | Free Cash Flow Margin                       |  0.2541 |  0.2826 |  0.2598 |   0.2783 |   0.2373 |
        | Interest Coverage Ratio                     | 41.1905 | 40.7496 | 29.062  | nan      | nan      |
        | Income Before Tax Profit Margin             |  0.2985 |  0.302  |  0.2967 |   0.3158 |   0.3189 |
        | Effective Tax Rate                          |  0.133  |  0.162  |  0.1472 |   0.2409 |   0.1561 |
        | Return on Assets                            |  0.2806 |  0.2836 |  0.275  |   0.2613 |   0.3093 |
        | Cash Return on Assets                       |  0.3083 |  0.3471 |  0.3134 |   0.3296 |   0.3079 |
        | Return on Equity                            |  1.4744 |  1.7546 |  1.7195 |   1.5741 |   1.7142 |
        | Return on Invested Capital                  |  0.4879 |  0.5229 |  0.528  |   0.5166 |   0.6201 |
        | Return on Capital Employed                  |  0.4831 |  0.6009 |  0.5514 |   0.6534 |   0.6872 |
        | Return on Tangible Assets                   |  0.2806 |  0.2836 |  0.275  |   0.2613 |   0.3093 |
        | Income Quality Ratio                        |  1.0988 |  1.2239 |  1.1397 |   1.2616 |   0.9953 |
        | Net Income per EBT                          |  0.867  |  0.838  |  0.8528 |   0.7591 |   0.8439 |
        | Free Cash Flow to Operating Cash Flow Ratio |  0.8935 |  0.9123 |  0.9009 |   0.9201 |   0.8859 |
        | EBT to EBIT Ratio                           |  0.9764 |  0.976  |  0.9666 |   1      |   1      |
        | EBIT to Revenue                             |  0.3058 |  0.3095 |  0.307  |   0.3158 |   0.3189 |
        | Cash Tax Rate                               |  0.2324 |  0.1643 |  0.1642 |   0.2114 |   0.3267 |
        | Tax Rate Divergence                         |  0.0994 |  0.0023 |  0.017  |  -0.0295 |   0.1706 |
        | Gross Profitability                         |  0.4354 |  0.4841 |  0.4797 |   0.495  |   0.5434 |
        | Operating Profitability                     |  1.685  |  2.2992 |  1.7759 |   2.1636 |   1.8045 |
        | Cash-Based Operating Profitability          |  0.3943 |  0.4204 |  0.4009 |   0.4394 |   0.4742 |
        | Cash Return on Invested Capital             |  0.4801 |  0.5823 |  0.5394 |   0.601  |   0.5455 |
        | Free Cash Flow Conversion                   |  0.9818 |  1.1166 |  1.0267 |   1.1608 |   0.8818 |
        | Free Cash Flow to EBITDA                    |  0.7731 |  0.8537 |  0.7915 |   0.808  |   0.6823 |
        """
        self._load_statements()
        profitability_ratios_dict: dict = {}

        profitability_ratios_dict["Gross Margin"] = self.get_gross_margin(
            trailing=trailing
        )
        profitability_ratios_dict["Operating Margin"] = self.get_operating_margin(
            trailing=trailing
        )
        profitability_ratios_dict["Net Profit Margin"] = self.get_net_profit_margin(
            trailing=trailing
        )
        profitability_ratios_dict["EBITDA Margin"] = self.get_ebitda_margin(
            trailing=trailing
        )
        profitability_ratios_dict["Free Cash Flow Margin"] = (
            self.get_free_cash_flow_margin(trailing=trailing)
        )
        profitability_ratios_dict["Interest Coverage Ratio"] = (
            self.get_interest_coverage_ratio(trailing=trailing)
        )
        profitability_ratios_dict["Income Before Tax Profit Margin"] = (
            self.get_income_before_tax_profit_margin(trailing=trailing)
        )
        profitability_ratios_dict["Effective Tax Rate"] = self.get_effective_tax_rate(
            trailing=trailing
        )
        profitability_ratios_dict["Return on Assets"] = self.get_return_on_assets(
            trailing=trailing
        )
        profitability_ratios_dict["Cash Return on Assets"] = (
            self.get_cash_return_on_assets(trailing=trailing)
        )
        profitability_ratios_dict["Return on Equity"] = self.get_return_on_equity(
            trailing=trailing
        )
        profitability_ratios_dict["Return on Invested Capital"] = (
            self.get_return_on_invested_capital(trailing=trailing)
        )
        profitability_ratios_dict["Return on Capital Employed"] = (
            self.get_return_on_capital_employed(trailing=trailing)
        )
        profitability_ratios_dict["Return on Tangible Assets"] = (
            self.get_return_on_tangible_assets(trailing=trailing)
        )
        profitability_ratios_dict["Income Quality Ratio"] = (
            self.get_income_quality_ratio(trailing=trailing)
        )
        profitability_ratios_dict["Net Income per EBT"] = self.get_net_income_per_ebt(
            trailing=trailing
        )
        profitability_ratios_dict["Free Cash Flow to Operating Cash Flow Ratio"] = (
            self.get_free_cash_flow_operating_cash_flow_ratio(trailing=trailing)
        )
        profitability_ratios_dict["EBT to EBIT Ratio"] = self.get_EBT_to_EBIT(
            trailing=trailing
        )
        profitability_ratios_dict["EBIT to Revenue"] = self.get_EBIT_to_revenue(
            trailing=trailing
        )
        profitability_ratios_dict["Cash Tax Rate"] = self.get_cash_tax_rate(
            trailing=trailing
        )
        profitability_ratios_dict["Tax Rate Divergence"] = self.get_tax_rate_divergence(
            trailing=trailing
        )
        profitability_ratios_dict["Gross Profitability"] = self.get_gross_profitability(
            trailing=trailing
        )
        profitability_ratios_dict["Operating Profitability"] = (
            self.get_operating_profitability(trailing=trailing)
        )
        profitability_ratios_dict["Cash-Based Operating Profitability"] = (
            self.get_cash_based_operating_profitability(trailing=trailing)
        )
        profitability_ratios_dict["Cash Return on Invested Capital"] = (
            self.get_cash_return_on_invested_capital(trailing=trailing)
        )
        profitability_ratios_dict["Free Cash Flow Conversion"] = (
            self.get_free_cash_flow_conversion(trailing=trailing)
        )
        profitability_ratios_dict["Free Cash Flow to EBITDA"] = (
            self.get_free_cash_flow_to_ebitda_ratio(trailing=trailing)
        )
        profitability_ratios_dict["Percent Accruals"] = self.get_percent_accruals(
            trailing=trailing
        )

        self._profitability_ratios = (
            pd.concat(profitability_ratios_dict)
            .swaplevel(0, 1)
            .sort_index(level=0, sort_remaining=False)
            .dropna(axis="columns", how="all")
        )

        self._profitability_ratios = self._profitability_ratios.loc[self._tickers]

        self._profitability_ratios = self._profitability_ratios.pipe(
            apply_rounding, rounding if rounding is not None else self._rounding
        )

        # Sorted again so the index follows the financial statements' order.
        available_columns = [
            column
            for column in self._income_statement.columns
            if column in self._profitability_ratios
        ]
        self._profitability_ratios = self._profitability_ratios.reindex(
            available_columns, axis=1
        )

        if growth:
            self._profitability_ratios_growth = calculate_growth(
                dataset=self._profitability_ratios,
                lag=lag,
                compound=compound,
                rounding=rounding if rounding is not None else self._rounding,
                axis="columns",
            )

        profitability_ratios = self._profitability_ratios
        profitability_ratios_growth = self._profitability_ratios_growth

        if standardize:
            standardize_rounding = rounding if rounding is not None else self._rounding
            if growth:
                profitability_ratios_growth = calculate_standardization(
                    dataset=profitability_ratios_growth,
                    rounding=standardize_rounding,
                    axis="columns",
                )
            else:
                profitability_ratios = calculate_standardization(
                    dataset=profitability_ratios,
                    rounding=standardize_rounding,
                    axis="columns",
                )

        if len(self._tickers) == 1:
            return (
                profitability_ratios_growth.loc[self._tickers[0]]
                if growth
                else profitability_ratios.loc[self._tickers[0]]
            ).loc[:, self._start_date : self._end_date]

        return (profitability_ratios_growth if growth else profitability_ratios).loc[
            :, self._start_date : self._end_date
        ]

    @handle_portfolio
    @handle_errors
    def get_gross_margin(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the gross margin, a profitability ratio that measures the percentage of
        revenue that exceeds the cost of goods sold.

        The gross margin ratio is calculated by subtracting the cost of goods sold (COGS) from
        the total revenue and then dividing the result by the total revenue. It represents the
        portion of revenue that contributes to covering other expenses and generating profit.

        The formula is as follows:

        - Gross Margin Ratio = (Revenue — Cost of Goods Sold) / Revenue

        Gross margin shows pricing power and production cost before overheads, and is best
        compared within an industry and over time. What sits in cost of goods sold differs
        between companies (some include depreciation, freight or warehousing, others report
        them lower down), so small gaps between peers can be accounting rather than
        economics. The trend matters most: a steady decline points to rising input costs
        that cannot be passed on, or to a shift in the sales mix.

        Also known as: gross profit margin, gross profit percentage.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.DataFrame: Gross margin ratio values.

        Notes:
        - The method retrieves historical data and calculates the gross margin ratio for each
        asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["MC.PA", "RMS.PA", "CFR.SW"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_gross_margin()
        ```

        Which returns:

        |        |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-------|-------:|-------:|-------:|-------:|-------:|
        | CFR.SW | 0.627  | 0.6874 | 0.6808 | 0.6691 | 0.644  |
        | MC.PA  | 0.6829 | 0.6844 | 0.688  | 0.6703 | 0.6624 |
        | RMS.PA | 0.6558 | 0.6603 | 0.7229 | 0.7026 | 0.7111 |
        """
        cost_of_goods_sold = self._income_statement.loc[:, "Cost of Goods Sold", :]

        if trailing:
            cost_of_goods_sold = self._sum_trailing(cost_of_goods_sold, trailing)
            gross_margin = profitability_model.get_gross_margin(
                self._sum_trailing(
                    self._income_statement.loc[:, "Revenue", :], trailing
                ),
                cost_of_goods_sold,
            )
        else:
            gross_margin = profitability_model.get_gross_margin(
                self._income_statement.loc[:, "Revenue", :],
                cost_of_goods_sold,
            )

        # Banks and insurers sell no goods, so they report no cost of goods sold and their
        # gross margin would be a meaningless 1.0; it is left out instead.
        without_cost_of_goods = cost_of_goods_sold == 0
        gross_margin = gross_margin.mask(without_cost_of_goods)

        if without_cost_of_goods.to_numpy().any():
            logger.warning(
                "The gross margin does not apply to companies that report no cost of goods "
                "sold, as banks and insurers do, so it is left out for %s in those periods.",
                ", ".join(
                    without_cost_of_goods.index[without_cost_of_goods.any(axis=1)]
                ),
            )

        return finalize_dataset(
            dataset=gross_margin,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_operating_margin(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the operating margin, a profitability ratio that measures the percentage of
        revenue that remains after deducting operating expenses.

        The operating margin ratio is calculated by dividing operating income (revenue minus the
        cost of goods sold and the operating expenses) by revenue. It indicates how efficiently
        a company is managing its operating expenses in relation to its revenue.

        The formula is as follows:

        - Operating Margin Ratio = Operating Income / Revenue

        Operating margin is the usual yardstick for the profitability of the core business,
        before financing and taxes. Companies with high fixed costs show operating leverage:
        their margin rises quickly when revenue grows and falls quickly when it shrinks.
        Restructuring charges and impairments are included, so a single weak year should be
        checked for one-off items before comparing with peers.

        Also known as: EBIT margin, operating income margin.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.DataFrame: Operating margin ratio values.

        Notes:
        - The method retrieves historical data and calculates the operating margin ratio for each
        asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["BKNG", "EXPE", "ABNB"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_operating_margin()
        ```

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | ABNB | 0.0716 | 0.2145 | 0.1531 | 0.23   | 0.212  |
        | BKNG | 0.2278 | 0.2985 | 0.2731 | 0.3183 | 0.3448 |
        | EXPE | 0.0216 | 0.093  | 0.0805 | 0.0963 | 0.127  |
        """
        if trailing:
            operating_margin = profitability_model.get_operating_margin(
                self._sum_trailing(
                    self._income_statement.loc[:, "Operating Income", :], trailing
                ),
                self._sum_trailing(
                    self._income_statement.loc[:, "Revenue", :], trailing
                ),
            )
        else:
            operating_margin = profitability_model.get_operating_margin(
                self._income_statement.loc[:, "Operating Income", :],
                self._income_statement.loc[:, "Revenue", :],
            )

        return finalize_dataset(
            dataset=operating_margin,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_net_profit_margin(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the net profit margin, a profitability ratio that measures the percentage
        of profit a company earns per dollar of revenue.

        The net profit margin ratio is calculated by dividing the net income by the total revenue.
        It indicates the portion of each dollar of revenue that represents profit after all expenses
        have been deducted. A higher net profit margin is generally considered favorable.

        The formula is as follows:

        - Net Profit Margin Ratio = Net Income / Revenue

        Also known as: net margin, bottom line margin, net income margin.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.DataFrame: Net profit margin ratio values.

        Notes:
        - The method retrieves historical data and calculates the net profit margin ratio for each
        asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["LLY", "NVO", "AZN"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_net_profit_margin()
        ```

        Which returns:

        |     |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:----|-------:|-------:|-------:|-------:|-------:|
        | AZN | 0.003  | 0.0741 | 0.13   | 0.1301 | 0.1746 |
        | LLY | 0.1971 | 0.2188 | 0.1536 | 0.2351 | 0.3167 |
        | NVO | 0.3392 | 0.3138 | 0.3603 | 0.3478 | 0.3314 |
        """
        if trailing:
            net_profit_margin = profitability_model.get_net_profit_margin(
                self._sum_trailing(
                    self._income_statement.loc[:, "Net Income", :], trailing
                ),
                self._sum_trailing(
                    self._income_statement.loc[:, "Revenue", :], trailing
                ),
            )
        else:
            net_profit_margin = profitability_model.get_net_profit_margin(
                self._income_statement.loc[:, "Net Income", :],
                self._income_statement.loc[:, "Revenue", :],
            )

        return finalize_dataset(
            dataset=net_profit_margin,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_ebitda_margin(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the EBITDA margin, a profitability ratio that measures the
        percentage of revenue that remains as earnings before interest, taxes,
        depreciation and amortization (EBITDA).

        EBITDA margin approximates a company's core operating profitability before the
        effects of financing decisions (interest), tax jurisdictions, and non-cash
        accounting choices around fixed and intangible assets (depreciation and
        amortization). This makes it a commonly used metric to compare operating
        performance across companies with different capital structures, tax regimes,
        and depreciation policies.

        The formula is as follows:

        - EBITDA Margin = (Operating Income + Depreciation and Amortization) / Revenue

        Also known as: EBITDA-to-revenue ratio.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.DataFrame: EBITDA margin values.

        Notes:
        - The method retrieves historical data and calculates the EBITDA margin for
        each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio
        values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["LIN", "AI.PA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_ebitda_margin()
        ```

        Which returns:

        |       |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:------|-------:|-------:|-------:|-------:|-------:|
        | AI.PA | 0.2649 | 0.2257 | 0.2555 | 0.2754 | 0.3023 |
        | LIN   | 0.3221 | 0.3196 | 0.3628 | 0.3749 | 0.383  |
        """
        if trailing:
            ebitda_margin = profitability_model.get_ebitda_margin(
                self._sum_trailing(
                    self._income_statement.loc[:, "Operating Income", :], trailing
                ),
                self._sum_trailing(
                    self._cash_flow_statement.loc[
                        :, "Depreciation and Amortization", :
                    ],
                    trailing,
                ),
                self._sum_trailing(
                    self._income_statement.loc[:, "Revenue", :], trailing
                ),
            )
        else:
            ebitda_margin = profitability_model.get_ebitda_margin(
                self._income_statement.loc[:, "Operating Income", :],
                self._cash_flow_statement.loc[:, "Depreciation and Amortization", :],
                self._income_statement.loc[:, "Revenue", :],
            )

        return finalize_dataset(
            dataset=ebitda_margin,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_interest_burden_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Compute the Interest Burden Ratio, the component of the extended (five-step)
        DuPont decomposition that isolates the drag interest expense places on a
        company's operating profit.

        The Interest Burden Ratio is calculated by dividing earnings before tax (EBT)
        by earnings before interest and taxes (EBIT, proxied here by Operating Income).
        It expresses the share of operating profit that survives interest expense, so
        it sits between 0 and 1 for a company with debt: a value close to 1 means
        interest barely dents operating profit, while a low value signals a heavy
        interest load. Values slightly above 1 occur when non-operating income (e.g.
        interest income) exceeds interest expense.

        Note that this is related to, but not the same as, the Interest Coverage Ratio
        (`get_interest_coverage_ratio`), which divides EBIT by interest expense and is
        unbounded above. When income before tax is roughly operating income minus
        interest expense, the interest burden is about 1 minus the reciprocal of the
        interest coverage ratio.

        The formula is as follows:

        - Interest Burden Ratio = Income Before Tax / Operating Income

        Also known as: EBT to EBIT ratio, interest burden.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.DataFrame: Interest Burden Ratio values.

        Notes:
        - The method retrieves historical data and calculates the Interest Burden Ratio for each
        asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["CHTR", "TMUS", "DIS"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_interest_burden_ratio()
        ```

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | CHTR | 0.5885 | 0.6095 | 0.5481 | 0.5664 | 0.5589 |
        | DIS  | 0.6999 | 0.7806 | 0.5304 | 0.6353 | 0.8678 |
        | TMUS | 0.4862 | 0.4808 | 0.771  | 0.8169 | 0.7619 |
        """
        if trailing:
            interest_burden_ratio = profitability_model.get_interest_burden_ratio(
                self._sum_trailing(
                    self._income_statement.loc[:, "Income Before Tax", :], trailing
                ),
                self._sum_trailing(
                    self._income_statement.loc[:, "Operating Income", :], trailing
                ),
            )
        else:
            interest_burden_ratio = profitability_model.get_interest_burden_ratio(
                self._income_statement.loc[:, "Income Before Tax", :],
                self._income_statement.loc[:, "Operating Income", :],
            )

        return finalize_dataset(
            dataset=interest_burden_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_income_before_tax_profit_margin(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Pretax Profit Margin, which is the ratio of a company's pre-tax profit to
        its revenue, indicating how much profit a company makes before paying taxes on its earnings.

        The Pretax Profit Margin is calculated by dividing the pre-tax profit by the revenue.
        It provides insight into how efficiently a company is able to generate profits from its revenue.

        The formula is as follows:

        - Pretax Profit Margin = Income Before Tax / Revenue

        Also known as: pretax margin, EBT margin.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.DataFrame: Pretax Profit Margin values.

        Notes:
        - The method retrieves historical data and calculates the Pretax Profit Margin for each
        asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["HSY", "GIS", "HRL"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_income_before_tax_profit_margin()
        ```

        Which returns:

        |     |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:----|-------:|-------:|-------:|-------:|-------:|
        | GIS | 0.1749 | 0.1603 | 0.1568 | 0.1484 | 0.0179 |
        | HRL | 0.0989 | 0.1026 | 0.0837 | 0.0869 | 0.0548 |
        | HSY | 0.2003 | 0.184  | 0.1945 | 0.2208 | 0.1038 |
        """
        if trailing:
            income_before_tax_profit_margin = (
                profitability_model.get_income_before_tax_profit_margin(
                    self._sum_trailing(
                        self._income_statement.loc[:, "Income Before Tax", :], trailing
                    ),
                    self._sum_trailing(
                        self._income_statement.loc[:, "Revenue", :], trailing
                    ),
                )
            )
        else:
            income_before_tax_profit_margin = (
                profitability_model.get_income_before_tax_profit_margin(
                    self._income_statement.loc[:, "Income Before Tax", :],
                    self._income_statement.loc[:, "Revenue", :],
                )
            )

        return finalize_dataset(
            dataset=income_before_tax_profit_margin,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_effective_tax_rate(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the effective tax rate, a financial ratio that measures the
        percentage of pretax income that is paid as taxes.

        The effective tax rate is calculated by dividing the income tax expense by the
        pre-tax income.

        The formula is as follows:

        - Effective Tax Rate = Income Tax Expense / Income Before Tax

        Analysts compare the effective tax rate with the statutory rates of the
        countries a company earns its profits in; a rate well below them usually comes
        from profits booked in lower-tax countries, research credits or one-off
        benefits. Single years can be far from the trend because of settlements, law
        changes or the release of a valuation allowance, which can even make the rate
        negative, so forecasts usually start from a normalized rate. `get_cash_tax_rate`
        shows what was actually paid. A period with zero or negative income before tax
        has no meaningful tax rate and is NaN.

        Also known as: ETR, actual tax rate.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.DataFrame: Effective tax rate values.

        Notes:
        - The method retrieves historical data and calculates the effective tax rate for each
        asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["GOOGL", "META", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_effective_tax_rate()
        ```

        Which returns:

        |       |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:------|-------:|-------:|-------:|-------:|-------:|
        | GOOGL | 0.162  | 0.1592 | 0.1391 | 0.1644 | 0.1678 |
        | META  | 0.1674 | 0.195  | 0.1756 | 0.1175 | 0.2964 |
        | MSFT  | 0.1383 | 0.1311 | 0.1898 | 0.1823 | 0.1763 |
        """
        # A tax rate is only meaningful on a pre-tax profit: with a loss the ratio flips sign, so
        # periods with zero or negative income before tax are NaN.
        income_before_tax = self._sum_trailing(
            self._income_statement.loc[:, "Income Before Tax", :], trailing
        )

        effective_tax_rate = profitability_model.get_effective_tax_rate(
            self._sum_trailing(
                self._income_statement.loc[:, "Income Tax Expense", :], trailing
            ),
            income_before_tax.where(income_before_tax > 0),
        )

        return finalize_dataset(
            dataset=effective_tax_rate,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_return_on_assets(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the return on assets (ROA), a profitability ratio that measures how
        efficiently a company uses its assets to generate profits.

        The return on assets is calculated by dividing the net income by the average total assets. Note
        that it is false to take the total assets at the end of the period given that income statements
        report over the period whereas a balance sheet reports on the period.

        The formula is as follows:

        - Return on Assets = Net Income / Average Total Assets

        Return on assets is the natural measure for banks and insurers, whose assets are mostly loans
        and investments, and where a return of about 1% is the usual benchmark for a bank. Outside
        finance it is best compared within an industry, since asset-light businesses such as software
        and services earn far higher returns on assets than utilities or manufacturers. Because net
        income is after interest, two companies with the same operations but different leverage show
        different returns on assets.

        Also known as: ROA, asset profitability.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.DataFrame: Return on assets (ROA) values.

        Notes:
        - The method retrieves historical data and calculates the ROA for each asset in the
        Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["BAC", "WFC", "C", "USB"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_return_on_assets()
        ```

        Which returns:

        |     |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:----|-------:|-------:|-------:|-------:|-------:|
        | BAC | 0.0102 | 0.0084 | 0.008  | 0.0079 | 0.0087 |
        | C   | 0.0091 | 0.0058 | 0.0033 | 0.0048 | 0.0052 |
        | USB | 0.0135 | 0.0088 | 0.0075 | 0.0088 | 0.0105 |
        | WFC | 0.0107 | 0.0066 | 0.0094 | 0.0096 | 0.0101 |
        """
        if trailing:
            return_on_assets = profitability_model.get_return_on_assets(
                self._sum_trailing(
                    self._income_statement.loc[:, "Net Income", :], trailing
                ),
                self._mean_trailing(
                    self._balance_sheet_statement.loc[:, "Total Assets", :], trailing
                ),
            )
        else:
            return_on_assets = profitability_model.get_return_on_assets(
                self._income_statement.loc[:, "Net Income", :],
                self._balance_sheet_statement.loc[:, "Total Assets", :]
                .T.rolling(2)
                .mean()
                .T,
            )

        return finalize_dataset(
            dataset=return_on_assets,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_cash_return_on_assets(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the cash return on assets (Cash ROA), a profitability ratio that
        measures how efficiently a company uses its assets to generate operating cash
        flow.

        Unlike the return on assets, which uses accrual-based net income, the cash
        return on assets uses operating cash flow, making it less sensitive to
        non-cash accounting choices (e.g. depreciation method, revenue recognition
        timing, working capital accruals). Comparing cash ROA to ROA is a useful
        earnings-quality cross-check: a cash ROA that persistently trails ROA can
        indicate that reported profits are not being converted into cash.

        The formula is as follows:

        - Cash Return on Assets = Cash Flow from Operations / Average Total Assets

        Also known as: cash ROA.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.DataFrame: Cash return on assets (Cash ROA) values.

        Notes:
        - The method retrieves historical data and calculates the cash ROA for each
        asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio
        values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["UPS", "FDX"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_cash_return_on_assets()
        ```

        Which returns:

        |     |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:----|-------:|-------:|-------:|-------:|-------:|
        | FDX | 0.1165 | 0.1018 | 0.0955 | 0.0806 | 0.0957 |
        | UPS | 0.2277 | 0.2007 | 0.1442 | 0.1436 | 0.118  |
        """
        if trailing:
            cash_return_on_assets = profitability_model.get_cash_return_on_assets(
                self._sum_trailing(
                    self._cash_flow_statement.loc[:, "Cash Flow from Operations", :],
                    trailing,
                ),
                self._mean_trailing(
                    self._balance_sheet_statement.loc[:, "Total Assets", :], trailing
                ),
            )
        else:
            cash_return_on_assets = profitability_model.get_cash_return_on_assets(
                self._cash_flow_statement.loc[:, "Cash Flow from Operations", :],
                self._balance_sheet_statement.loc[:, "Total Assets", :]
                .T.rolling(2)
                .mean()
                .T,
            )

        return finalize_dataset(
            dataset=cash_return_on_assets,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_return_on_equity(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the return on equity (ROE), a profitability ratio that measures how
        efficiently a company generates profits using its shareholders' equity.

        The return on equity is calculated by dividing the net income by the average shareholders' equity.
        Shareholders' equity represents the residual interest in the assets of a company after deducting liabilities.
        Note that it is false to take the total equity at the end of the period, given that an income statement
        reports over the period whereas a balance sheet reports at a point in time; the average equity over the
        period is used instead.

        ROE provides insight into the company's ability to generate profits from the investments made by
        its shareholders. A higher ROE indicates that the company is using its equity effectively to generate
        higher returns for its shareholders.

        The formula is as follows:

        - Return on Equity = Net Income / Average Shareholders' Equity (or Total Equity)

        ROE is usually read through the DuPont analysis (`Toolkit.models.get_dupont_analysis`), which splits it into
        net margin, asset turnover and the equity multiplier, because the same ROE can come from high margins or from
        a thin equity base. Large buybacks and accumulated losses shrink equity and can push ROE to extreme or negative
        values (well above 1 for heavy buyers of their own shares), in which case ROIC or ROA says more. For banks and
        insurers ROE is set against the cost of equity, and together with the price-to-book ratio it explains much of
        their valuation.

        Also known as: ROE, shareholders equity return, equity profitability.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.DataFrame: Return on equity (ROE) values.

        Notes:
        - The method retrieves historical data and calculates the ROE for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["JPM", "GS", "MS"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_return_on_equity()
        ```

        Which returns:

        |     |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:----|-------:|-------:|-------:|-------:|-------:|
        | GS  | 0.2055 | 0.0948 | 0.0676 | 0.1132 | 0.132  |
        | JPM | 0.1622 | 0.1224 | 0.154  | 0.1691 | 0.1575 |
        | MS  | 0.1389 | 0.1014 | 0.0848 | 0.1246 | 0.149  |
        """
        if trailing:
            return_on_equity = profitability_model.get_return_on_equity(
                self._sum_trailing(
                    self._income_statement.loc[:, "Net Income", :], trailing
                ),
                self._mean_trailing(
                    self._balance_sheet_statement.loc[:, "Total Equity", :], trailing
                ),
            )

        else:
            return_on_equity = profitability_model.get_return_on_equity(
                self._income_statement.loc[:, "Net Income", :],
                self._balance_sheet_statement.loc[:, "Total Equity", :]
                .T.rolling(window=2)
                .mean()
                .T,
            )

        return finalize_dataset(
            dataset=return_on_equity,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_return_on_invested_capital(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ):
        """
        Calculate the return on invested capital (ROIC), a financial ratio that measures
        the company's return on the capital invested in it, including both equity and debt.

        The return on invested capital is calculated by dividing the net operating profit after taxes (NOPAT)
        by the average invested capital: total equity plus total debt (Damodaran). NOPAT is operating income
        (EBIT) taxed at the effective tax rate (Income Tax Expense / Income Before Tax, bounded to [0, 1]; set to
        0 for a period with a pre-tax loss). Because EBIT is before interest, the return does not depend on how
        the capital is financed, which makes the ratio a measure of how efficiently a company generates returns
        for all of its investors. Cash is not deducted from the invested capital: for companies that hold large
        cash balances, deducting it can shrink the base to near zero or below.

        The formula is as follows:

        - Return on Invested Capital = Operating Income x (1 — Effective Tax Rate) / (Average Total Equity +
          Average Total Debt)

        ROIC is compared with the weighted average cost of capital
        (`Toolkit.models.get_weighted_average_cost_of_capital`): a company that earns more than its cost of capital
        creates value as it invests and grows, one that earns less destroys it. It is less distorted by buybacks and
        leverage than ROE.

        Also known as: ROIC, return on capital (ROC).

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.DataFrame: Return on invested capital (ROIC) values.

        Notes:
        - The method retrieves historical data and calculates the ROIC for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["CAT", "DE", "CMI"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_return_on_invested_capital()
        ```

        Which returns:

        |     |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:----|-------:|-------:|-------:|-------:|-------:|
        | CAT | 0.1015 | 0.1127 | 0.1853 | 0.1823 | 0.1392 |
        | CMI | 0.1541 | 0.1392 | 0.051  | 0.1728 | 0.1486 |
        | DE  | 0.0949 | 0.1008 | 0.1439 | 0.1014 | 0.0754 |
        """
        if trailing:
            return_on_invested_capital = (
                profitability_model.get_return_on_invested_capital(
                    self._sum_trailing(
                        self._income_statement.loc[:, "Operating Income", :], trailing
                    ),
                    self._sum_trailing(
                        self._income_statement.loc[:, "Income Tax Expense", :], trailing
                    ),
                    self._sum_trailing(
                        self._income_statement.loc[:, "Income Before Tax", :], trailing
                    ),
                    self._mean_trailing(
                        self._balance_sheet_statement.loc[:, "Total Equity", :],
                        trailing,
                    ),
                    self._mean_trailing(
                        self._balance_sheet_statement.loc[:, "Total Debt", :], trailing
                    ),
                )
            )
        else:
            return_on_invested_capital = (
                profitability_model.get_return_on_invested_capital(
                    self._income_statement.loc[:, "Operating Income", :],
                    self._income_statement.loc[:, "Income Tax Expense", :],
                    self._income_statement.loc[:, "Income Before Tax", :],
                    self._balance_sheet_statement.loc[:, "Total Equity", :]
                    .T.rolling(2)
                    .mean()
                    .T,
                    self._balance_sheet_statement.loc[:, "Total Debt", :]
                    .T.rolling(2)
                    .mean()
                    .T,
                )
            )

        return finalize_dataset(
            dataset=return_on_invested_capital,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_income_quality_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ):
        """
        Calculate the income quality ratio, a financial metric that measures the cash flow from
        operating activities relative to the net income of the company.

        The income quality ratio provides insights into the quality of a company's reported earnings.
        By comparing the cash flow from operating activities to the net income, this ratio helps assess
        whether a company's reported profits are backed by actual cash flow. A higher income quality
        ratio suggests higher earnings quality and a better ability to convert profits into cash flow.

        The formula is as follows:

        - Income Quality Ratio = Cash Flow from Operations / Net Income

        A ratio above 1 means reported earnings are backed by at least as much operating cash flow,
        which is typical for companies with large depreciation charges. A ratio that stays below 1 for
        several years, for example because receivables or capitalized costs keep growing, is a common
        warning sign in earnings-quality screens. The ratio loses meaning, and can flip sign, when net
        income is close to zero or negative.

        Also known as: earnings quality ratio, cash flow to net income ratio.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.DataFrame: Income quality ratio values.

        Notes:
        - The method retrieves historical data and calculates the income quality ratio for each asset
        in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["NFLX", "DIS"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_income_quality_ratio()
        ```

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | DIS  | 2.7905 | 1.911  | 4.1912 | 2.8099 | 1.4593 |
        | NFLX | 0.0767 | 0.4511 | 1.3451 | 0.845  | 0.9242 |
        """
        if trailing:
            income_quality_ratio = profitability_model.get_income_quality_ratio(
                self._sum_trailing(
                    self._cash_flow_statement.loc[:, "Cash Flow from Operations", :],
                    trailing,
                ),
                self._sum_trailing(
                    self._income_statement.loc[:, "Net Income", :], trailing
                ),
            )
        else:
            income_quality_ratio = profitability_model.get_income_quality_ratio(
                self._cash_flow_statement.loc[:, "Cash Flow from Operations", :],
                self._income_statement.loc[:, "Net Income", :],
            )

        return finalize_dataset(
            dataset=income_quality_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_return_on_tangible_assets(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ):
        """
        Calculate the return on tangible assets, a financial ratio that measures the amount of profit
        generated by a company's tangible assets.

        The return on tangible assets (ROTA) provides insights into the efficiency with which a company
        utilizes its tangible assets to generate profits. It is calculated by dividing net income by the
        average tangible assets: total assets minus goodwill and other intangible assets. Liabilities are
        not deducted, so the ratio is the return on the whole tangible asset base however it is financed,
        like the return on assets but without the acquired goodwill and intangibles that inflate the asset
        base of acquisitive companies. A ROTA well above the ROA therefore points to a large share of
        goodwill and intangible assets on the balance sheet.

        The formula is as follows:

        - Return on Tangible Assets = Net Income / (Average Total Assets — Average Goodwill and
          Intangible Assets)

        Also known as: ROTA, tangible asset return.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.DataFrame: Return on tangible assets (ROTA) values.

        Notes:
        - The method retrieves historical data and calculates the ROTA for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["NUE", "STLD", "RS"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_return_on_tangible_assets()
        ```

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | NUE  | 0.3474 | 0.3215 | 0.1685 | 0.074  | 0.0641 |
        | RS   | 0.2441 | 0.2715 | 0.1831 | 0.1229 | 0.1049 |
        | STLD | 0.3172 | 0.3069 | 0.1778 | 0.1082 | 0.0795 |
        """
        if trailing:
            return_on_tangible_assets = (
                profitability_model.get_return_on_tangible_assets(
                    self._sum_trailing(
                        self._income_statement.loc[:, "Net Income", :], trailing
                    ),
                    self._mean_trailing(
                        self._balance_sheet_statement.loc[:, "Total Assets", :],
                        trailing,
                    ),
                    self._mean_trailing(
                        get_goodwill_and_intangible_assets(
                            self._balance_sheet_statement
                        ),
                        trailing,
                    ),
                )
            )
        else:
            return_on_tangible_assets = (
                profitability_model.get_return_on_tangible_assets(
                    self._income_statement.loc[:, "Net Income", :],
                    self._balance_sheet_statement.loc[:, "Total Assets", :]
                    .T.rolling(2)
                    .mean()
                    .T,
                    get_goodwill_and_intangible_assets(self._balance_sheet_statement)
                    .T.rolling(2)
                    .mean()
                    .T,
                )
            )

        return finalize_dataset(
            dataset=return_on_tangible_assets,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_return_on_capital_employed(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ):
        """
        Calculate the return on capital employed (ROCE), a profitability ratio that measures the amount of return
        a company generates from the capital it has invested in the business.

        Return on capital employed (ROCE) is a crucial financial metric that evaluates the efficiency and profitability
        of a company's utilization of both equity and debt capital to generate profits. It assesses how well the company
        generates earnings relative to the total capital invested in the business.

        The formula is as follows:

        - Return on Capital Employed = EBIT / (Total Assets — Current Liabilities)

        EBIT is taken as operating income, so non-operating gains and losses such as interest
        income or investment results do not enter the return.

        ROCE is widely used for capital-intensive industries such as oil and gas, mining, utilities and telecoms,
        because it measures the operating return on all long-term funding, equity and debt, regardless of the
        capital structure. It is compared with the cost of capital and with peers over a full cycle, since commodity
        producers show very high ROCE at the top of a price cycle. The capital employed is taken at the end of the
        period (averaged over the window when trailing is set), so a large acquisition late in the year lowers the
        ratio before the acquired earnings come in.

        Also known as: ROCE.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.DataFrame: Return on capital employed (ROCE) values.

        Notes:
        - The method retrieves historical data and calculates the ROCE for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["XOM", "CVX", "SHEL", "BP"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_return_on_capital_employed()
        ```

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | BP   | 0.0516 | 0.2192 | 0.1599 | 0.0642 | 0.0781 |
        | CVX  | 0.0757 | 0.1787 | 0.1473 | 0.1332 | 0.0574 |
        | SHEL | 0.0722 | 0.1962 | 0.0989 | 0.1025 | 0.0958 |
        | XOM  | 0.0851 | 0.2134 | 0.143  | 0.1035 | 0.0901 |
        """
        if trailing:
            return_on_capital_employed = (
                profitability_model.get_return_on_capital_employed(
                    self._sum_trailing(
                        self._income_statement.loc[:, "Operating Income", :], trailing
                    ),
                    self._mean_trailing(
                        self._balance_sheet_statement.loc[:, "Total Assets", :],
                        trailing,
                    ),
                    self._mean_trailing(
                        self._balance_sheet_statement.loc[
                            :, "Total Current Liabilities", :
                        ],
                        trailing,
                    ),
                )
            )
        else:
            return_on_capital_employed = (
                profitability_model.get_return_on_capital_employed(
                    self._income_statement.loc[:, "Operating Income", :],
                    self._balance_sheet_statement.loc[:, "Total Assets", :],
                    self._balance_sheet_statement.loc[
                        :, "Total Current Liabilities", :
                    ],
                )
            )

        return finalize_dataset(
            dataset=return_on_capital_employed,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_net_income_per_ebt(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ):
        """
        Calculate the net income per earnings before taxes (EBT), a profitability ratio that measures
        the net income generated for each dollar of EBT.

        The net income per earnings before taxes (EBT) ratio helps evaluate the extent to which a company's net
        income is generated from its operating activities before considering the impact of income taxes. It gives
        insights into how effectively a company generates profit relative to its taxable income.

        The formula is as follows:

        - Net Income per EBT = Net Income / (Net Income + Income Tax Expense)

        Earnings before tax is reconstructed from the income statement as Net Income
        plus Income Tax Expense rather than read from the reported Income Before Tax
        line, so this can differ slightly from `get_tax_burden_ratio` when a company
        reports minority interests or discontinued operations below the tax line.

        Also known as: net income to pre-tax income.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.DataFrame: Net income per earnings before taxes (EBT) values.

        Notes:
        - The method retrieves historical data and calculates the net income per EBT for each asset in
        the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values using
        the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["SAP.DE", "SIE.DE", "ASML.AS"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_net_income_per_ebt()
        ```

        Which returns:

        |         |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:--------|-------:|-------:|-------:|-------:|-------:|
        | ASML.AS | 0.8521 | 0.8529 | 0.8452 | 0.8184 | 0.8268 |
        | SAP.DE  | 0.7576 | 0.6123 | 0.7791 | 0.6593 | 0.7087 |
        | SIE.DE  | 0.7387 | 0.576  | 0.7535 | 0.7816 | 0.7935 |
        """
        if trailing:
            net_income_per_ebt = profitability_model.get_net_income_per_ebt(
                self._sum_trailing(
                    self._income_statement.loc[:, "Net Income", :], trailing
                ),
                self._sum_trailing(
                    self._income_statement.loc[:, "Income Tax Expense", :], trailing
                ),
            )
        else:
            net_income_per_ebt = profitability_model.get_net_income_per_ebt(
                self._income_statement.loc[:, "Net Income", :],
                self._income_statement.loc[:, "Income Tax Expense", :],
            )

        return finalize_dataset(
            dataset=net_income_per_ebt,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_free_cash_flow_operating_cash_flow_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ):
        """
        Calculate the free cash flow to operating cash flow ratio, a profitability ratio that measures
        the amount of free cash flow a company generates for every dollar of operating cash flow.

        The free cash flow to operating cash flow ratio helps assess how well a company's operating activities
        translate into free cash flow, which is the cash available after all expenses and investments. A higher
        ratio indicates that the company is generating strong free cash flow relative to its operating cash flow,
        which could signify efficient capital management.

        The formula is as follows:

        - Free Cash Flow to Operating Cash Flow Ratio = Free Cash Flow / Cash Flow from Operations

        Also known as: FCF to OCF ratio.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.DataFrame: Free cash flow to operating cash flow ratio values.

        Notes:
        - The method retrieves historical data and calculates the ratio for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AMZN", "META", "GOOGL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_free_cash_flow_operating_cash_flow_ratio()
        ```

        Which returns:

        |       |    2021 |    2022 |   2023 |   2024 |   2025 |
        |:------|--------:|--------:|-------:|-------:|-------:|
        | AMZN  | -0.3179 | -0.3613 | 0.3793 | 0.2837 | 0.0552 |
        | GOOGL |  0.7312 |  0.6559 | 0.683  | 0.5807 | 0.4448 |
        | META  |  0.6781 |  0.3773 | 0.6166 | 0.5921 | 0.3982 |
        | MSFT  |  0.7313 |  0.7317 | 0.6791 | 0.6248 | 0.5259 |
        """
        if trailing:
            free_cash_flow_operating_cash_flow_ratio = (
                profitability_model.get_free_cash_flow_operating_cash_flow_ratio(
                    self._sum_trailing(
                        self._cash_flow_statement.loc[:, "Free Cash Flow", :], trailing
                    ),
                    self._sum_trailing(
                        self._cash_flow_statement.loc[
                            :, "Cash Flow from Operations", :
                        ],
                        trailing,
                    ),
                )
            )
        else:
            free_cash_flow_operating_cash_flow_ratio = (
                profitability_model.get_free_cash_flow_operating_cash_flow_ratio(
                    self._cash_flow_statement.loc[:, "Free Cash Flow", :],
                    self._cash_flow_statement.loc[:, "Cash Flow from Operations", :],
                )
            )

        return finalize_dataset(
            dataset=free_cash_flow_operating_cash_flow_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_free_cash_flow_margin(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the free cash flow margin, a profitability ratio that measures the
        percentage of revenue that is converted into free cash flow.

        Unlike the net profit margin, which can be distorted by non-cash accounting
        items (e.g. depreciation, stock-based compensation, deferred taxes) and by
        working capital timing, the free cash flow margin reflects the cash a company
        actually generates, after capital expenditures, for every dollar of revenue. A
        persistently low or declining free cash flow margin relative to the net profit
        margin can be a quality-of-earnings warning sign.

        The formula is as follows:

        - Free Cash Flow Margin = Free Cash Flow / Revenue

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.DataFrame: Free cash flow margin values.

        Notes:
        - The method retrieves historical data and calculates the free cash flow
        margin for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio
        values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["MCO", "SPGI", "MSCI"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_free_cash_flow_margin()
        ```

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | MCO  | 0.3001 | 0.2178 | 0.3178 | 0.3557 | 0.3336 |
        | MSCI | 0.4322 | 0.4547 | 0.4528 | 0.5139 | 0.4942 |
        | SPGI | 0.4294 | 0.2248 | 0.2854 | 0.3917 | 0.3558 |
        """
        if trailing:
            free_cash_flow_margin = profitability_model.get_free_cash_flow_margin(
                self._sum_trailing(
                    self._cash_flow_statement.loc[:, "Free Cash Flow", :], trailing
                ),
                self._sum_trailing(
                    self._income_statement.loc[:, "Revenue", :], trailing
                ),
            )
        else:
            free_cash_flow_margin = profitability_model.get_free_cash_flow_margin(
                self._cash_flow_statement.loc[:, "Free Cash Flow", :],
                self._income_statement.loc[:, "Revenue", :],
            )

        return finalize_dataset(
            dataset=free_cash_flow_margin,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_tax_burden_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ):
        """
        Calculate the tax burden ratio, which is the ratio of a company's
        net income to its income before tax, indicating how much of a
        company's income is retained after taxes.

        The tax burden ratio measures the portion of a company's pretax earnings
        that it keeps after taxes; one minus the ratio is roughly the effective
        tax rate. A higher ratio means a lower tax charge. It is one of the
        components of the extended (five-step) DuPont analysis. This ratio
        provides insights into the tax efficiency of the company and its ability
        to manage its tax liabilities.

        The formula is as follows:

        - Tax Burden Ratio = Net Income / Income Before Tax

        Also known as: tax efficiency ratio.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.DataFrame: Tax burden ratio values.

        Notes:
        - The method retrieves historical data and calculates the ratio for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["ACN", "ORCL", "CSCO"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_tax_burden_ratio()
        ```

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | ACN  | 0.7619 | 0.7486 | 0.7527 | 0.7497 | 0.7483 |
        | CSCO | 0.7986 | 0.8159 | 0.8234 | 0.8436 | 0.9171 |
        | ORCL | 0.8782 | 0.9317 | 0.8915 | 0.8787 | 0.8662 |
        """
        if trailing:
            tax_burden_ratio = profitability_model.get_tax_burden_ratio(
                self._sum_trailing(
                    self._income_statement.loc[:, "Net Income", :], trailing
                ),
                self._sum_trailing(
                    self._income_statement.loc[:, "Income Before Tax", :], trailing
                ),
            )
        else:
            tax_burden_ratio = profitability_model.get_tax_burden_ratio(
                self._income_statement.loc[:, "Net Income", :],
                self._income_statement.loc[:, "Income Before Tax", :],
            )

        return finalize_dataset(
            dataset=tax_burden_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_EBT_to_EBIT(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ):
        """
        Calculate the EBT to EBIT ratio, which shows how much of a company's earnings before
        interest and taxes (EBIT) is left after interest expense.

        A ratio close to 1 means interest takes little of the operating profit; a lower ratio
        means a heavier interest load. Taxes play no part in the ratio. Unlike
        `get_interest_burden_ratio`, which uses the reported income before tax and operating
        income, both figures here are rebuilt from net income, income tax expense and interest
        expense.

        The formula is as follows:

        - EBT to EBIT = (Net Income + Income Tax Expense) / (Net Income + Income Tax Expense + Interest Expense)

        Also known as: EBT to EBIT, interest burden.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.DataFrame: EBT to EBIT ratio values.

        Notes:
        - The method retrieves historical data and calculates the ratio for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["HLT", "MAR", "CHH"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_EBT_to_EBIT()
        ```

        Which returns:

        |     |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:----|-------:|-------:|-------:|-------:|-------:|
        | CHH | 0.8894 | 0.9085 | 0.8403 | 0.819  | 0.8331 |
        | HLT | 0.5865 | 0.8067 | 0.7838 | 0.7577 | 0.7697 |
        | MAR | 0.7375 | 0.8854 | 0.8567 | 0.8193 | 0.8075 |
        """
        if trailing:
            EBT_to_EBIT = profitability_model.get_EBT_to_EBIT(
                self._sum_trailing(
                    self._income_statement.loc[:, "Net Income", :], trailing
                )
                + self._sum_trailing(
                    self._income_statement.loc[:, "Income Tax Expense", :], trailing
                ),
                self._sum_trailing(
                    self._income_statement.loc[:, "Net Income", :], trailing
                )
                + self._sum_trailing(
                    self._income_statement.loc[:, "Income Tax Expense", :], trailing
                )
                + self._sum_trailing(
                    self._income_statement.loc[:, "Interest Expense", :], trailing
                ),
            )
        else:
            EBT_to_EBIT = profitability_model.get_EBT_to_EBIT(
                self._income_statement.loc[:, "Net Income", :]
                + self._income_statement.loc[:, "Income Tax Expense", :],
                self._income_statement.loc[:, "Net Income", :]
                + self._income_statement.loc[:, "Income Tax Expense", :]
                + self._income_statement.loc[:, "Interest Expense", :],
            )

        return finalize_dataset(
            dataset=EBT_to_EBIT,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_EBIT_to_revenue(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ):
        """
        Calculate the EBIT per Revenue, which is the ratio of a company's earnings
        before interest and taxes to its revenue, indicating how much profit a
        company generates from its operations before paying interest on debt
        and taxes on its earnings.

        The EBIT to Revenue ratio measures the company's ability to generate profit
        before financing costs and taxes relative to its revenue. EBIT is rebuilt from
        the bottom of the income statement (net income plus income tax expense plus
        interest expense), the same EBIT as in `get_EBT_to_EBIT`. It therefore includes
        non-operating income and expenses such as interest income, investment gains
        and other income, which is why it differs from the operating margin
        (`get_operating_margin`), which divides the reported operating income by
        revenue. A higher ratio indicates that a larger portion of the company's
        revenue is converted into profit before interest and taxes.

        The formula is as follows:

        - EBIT to Revenue = (Net Income + Income Tax Expense + Interest Expense) / Revenue

        Also known as: EBIT margin.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.DataFrame: EBIT to Revenue ratio values.

        Notes:
        - The method retrieves historical data and calculates the ratio for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["LMT", "GD", "NOC"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_EBIT_to_revenue()
        ```

        Which returns:

        |     |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:----|-------:|-------:|-------:|-------:|-------:|
        | GD  | 0.1119 | 0.1123 | 0.1037 | 0.1034 | 0.1031 |
        | LMT | 0.1211 | 0.1107 | 0.1334 | 0.1021 | 0.0938 |
        | NOC | 0.2662 | 0.1733 | 0.0736 | 0.1374 | 0.1366 |
        """
        if trailing:
            EBIT_to_revenue = profitability_model.get_EBIT_to_revenue(
                self._sum_trailing(
                    self._income_statement.loc[:, "Net Income", :], trailing
                )
                + self._sum_trailing(
                    self._income_statement.loc[:, "Income Tax Expense", :], trailing
                )
                + self._sum_trailing(
                    self._income_statement.loc[:, "Interest Expense", :], trailing
                ),
                self._sum_trailing(
                    self._income_statement.loc[:, "Revenue", :], trailing
                ),
            )
        else:
            EBIT_to_revenue = profitability_model.get_EBIT_to_revenue(
                self._income_statement.loc[:, "Net Income", :]
                + self._income_statement.loc[:, "Income Tax Expense", :]
                + self._income_statement.loc[:, "Interest Expense", :],
                self._income_statement.loc[:, "Revenue", :],
            )

        return finalize_dataset(
            dataset=EBIT_to_revenue,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_cash_tax_rate(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the cash tax rate, which measures the percentage of pretax income
        that is actually paid out in cash taxes, as opposed to the accrual-based
        effective tax rate.

        The formula is as follows:

        - Cash Tax Rate = Income Taxes Paid / Income Before Tax

        The cash tax rate shows the taxes a company actually paid, which can differ
        for years from the tax expense in the income statement because of deferred
        taxes, prepayments and settlements of earlier years. Analysts compare it
        with the effective tax rate to see whether reported earnings rely on taxes
        that are deferred rather than saved. When the data has no income taxes paid
        for a year, the rate shows as 0, which should be read as missing, not as a
        company that paid no tax. A period with zero or negative income before tax has
        no meaningful tax rate and is NaN.

        Also known as: cash effective tax rate.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.DataFrame: Cash tax rate values.

        Notes:
        - The method retrieves historical data and calculates the cash tax rate for each
        asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["PG", "PEP", "COST"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_cash_tax_rate()
        ```

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | COST | 0.2286 | 0.2474 | 0.2632 | 0.2381 | 0.2696 |
        | PEP  | 0.1968 | 0.2584 | 0.2218 | 0.2565 | 0.3012 |
        | PG   | 0.217  | 0.2122 | 0.2331 | 0.2326 | 0.2258 |
        """
        # A tax rate is only meaningful on a pre-tax profit: with a loss the ratio flips sign, so
        # periods with zero or negative income before tax are NaN.
        income_before_tax = self._sum_trailing(
            self._income_statement.loc[:, "Income Before Tax", :], trailing
        )

        cash_tax_rate = profitability_model.get_cash_tax_rate(
            self._sum_trailing(
                self._cash_flow_statement.loc[:, "Income Taxes Paid", :], trailing
            ),
            income_before_tax.where(income_before_tax > 0),
        )

        return finalize_dataset(
            dataset=cash_tax_rate,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_tax_rate_divergence(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the tax rate divergence, which measures the difference between the
        cash tax rate and the accrual-based effective tax rate.

        A persistently positive divergence indicates the company is paying more in
        cash taxes than it is recognizing as tax expense (e.g. due to the reversal of
        deferred tax liabilities), while a persistently negative divergence indicates
        the opposite and can be a quality-of-earnings red flag if it stems from
        aggressive tax deferral rather than timing differences.

        The formula is as follows:

        - Tax Rate Divergence = Cash Tax Rate - Effective Tax Rate

        Because the divergence is the cash tax rate minus the effective tax rate, a
        year without reported income taxes paid shows a divergence equal to minus the
        effective tax rate (-0.27 against an effective rate of 0.27), so check
        `get_cash_tax_rate` for zeros before reading a large negative value as aggressive
        tax deferral. The measure is unstable when income before tax is small and NaN when
        it is zero or negative.

        Also known as: cash-accrual tax gap.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.DataFrame: Tax rate divergence values.

        Notes:
        - The method retrieves historical data and calculates the tax rate divergence for
        each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "HD", "V"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_tax_rate_divergence()
        ```

        Which returns:

        |      |    2021 |   2022 |    2023 |    2024 |   2025 |
        |:-----|--------:|-------:|--------:|--------:|-------:|
        | AAPL |  0.0994 | 0.0023 |  0.017  | -0.0295 | 0.1706 |
        | HD   |  0.0092 | 0.0028 |  0.0121 | -0.0488 | 0.0216 |
        | V    | -0.0461 | 0.031  | -0.0157 |  0.067  | 0.0167 |
        """
        tax_rate_divergence = profitability_model.get_tax_rate_divergence(
            self.get_cash_tax_rate(trailing=trailing),
            self.get_effective_tax_rate(trailing=trailing),
        )

        return finalize_dataset(
            dataset=tax_rate_divergence,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_gross_profitability(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the gross profitability of Novy-Marx (2013), the gross profit a company
        earns on its assets.

        Novy-Marx shows that gross profit is the cleanest accounting measure of economic
        profitability: the further down the income statement, the more the profit is
        polluted by expenses that are really investments (selling, research and
        administrative expenses) or by financing and taxes. Gross profitability predicts
        stock returns about as well as book-to-market does, and profitable companies
        outperform unprofitable ones even though they trade at higher valuations, which
        makes it the profitability factor that complements value.

        The formula is as follows:

        - Gross Profitability = (Revenue - Cost of Goods Sold) / Total Assets

        The total assets are those at the end of the fiscal year, the same date as the gross
        profit, as in the paper (a point in time, not averaged). Gross profitability is an
        annual measure, so with quarterly data the gross profit is that of the trailing
        twelve months by default.

        Also known as: GP/A, gross profits to assets, gross profitability premium, Novy-Marx profitability.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum the revenue and cost of goods
                sold over; the total assets stay those at the end of the last period. Defaults to
                None, which uses the trailing twelve months (4 quarters) with quarterly data and
                single years with annual data.

        Returns:
            pd.DataFrame: Gross profitability values.

        Notes:
        - Novy-Marx, R., "The Other Side of Value: The Gross Profitability Premium," Journal
        of Financial Economics 108(1), 2013, pp. 1-28.
        - The Quality Score (`Toolkit.models.get_quality_score`) uses the same measure as one
        of its profitability components.
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "KO", "XOM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_gross_profitability()
        ```

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | AAPL | 0.4354 | 0.4841 | 0.4797 | 0.495  | 0.5434 |
        | KO   | 0.2469 | 0.2695 | 0.2787 | 0.2858 | 0.2819 |
        | MSFT | 0.3471 | 0.3717 | 0.3545 | 0.3339 | 0.3132 |
        | XOM  | 0.1914 | 0.2793 | 0.2236 | 0.1692 | 0.1564 |
        """
        trailing = self._get_valuation_trailing(trailing)

        gross_profitability = profitability_model.get_gross_profitability(
            self._sum_trailing(self._income_statement.loc[:, "Revenue", :], trailing),
            self._sum_trailing(
                self._income_statement.loc[:, "Cost of Goods Sold", :], trailing
            ),
            self._balance_sheet_statement.loc[:, "Total Assets", :],
        )

        return finalize_dataset(
            dataset=gross_profitability,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_operating_profitability(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the operating profitability of Fama and French (2015), the measure behind
        the profitability factor RMW (robust minus weak) of their five-factor model: the
        operating profit after interest that a company earns on its book equity.

        Fama and French find that companies with a robust operating profitability earn
        higher returns than those with a weak one, also after controlling for value, size
        and investment. Book equity follows their definition: the stockholders' equity
        (without minority interest) plus the balance sheet deferred taxes, minus the
        preferred stock.

        The formula is as follows:

        - Operating Profitability = (Revenue - Cost of Goods Sold - Selling, General and
          Administrative Expenses - Research and Development Expenses - Interest Expense) /
          Book Equity
        - Book Equity = Total Shareholder Equity + Deferred Tax Liabilities - Preferred Stock

        The selling, general and administrative expenses of Compustat (XSGA), which Fama and
        French use, include research and development, while the statements here report it
        separately, so both are deducted. The book equity is that at the end of the fiscal
        year, the same date as the profit (a point in time, not averaged), and a book equity
        of zero or less has no value (NaN), as Fama and French leave those companies out.
        Operating profitability is an annual measure, so with quarterly data the profit is
        that of the trailing twelve months by default.

        Also known as: OP, RMW, robust minus weak, Fama-French profitability.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum the income statement items
                over; the book equity stays that at the end of the last period. Defaults to None,
                which uses the trailing twelve months (4 quarters) with quarterly data and single
                years with annual data.

        Returns:
            pd.DataFrame: Operating profitability values.

        Notes:
        - Fama, E. F. and K. R. French, "A Five-Factor Asset Pricing Model," Journal of
        Financial Economics 116(1), 2015, pp. 1-22.
        - Large buybacks shrink the book equity, which can make the ratio very large; the
        gross profitability and cash-based operating profitability scale by total assets
        instead.
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["JNJ", "PFE", "MRK", "ABBV"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_operating_profitability()
        ```

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | ABBV | 0.6533 | 0.6413 | 0.5861 | 0.8289 | 3.614  |
        | JNJ  | 0.2547 | 0.2567 | 0.3145 | 0.2894 | 0.2788 |
        | MRK  | 0.3033 | 0.3703 | 0.0631 | 0.4123 | 0.4095 |
        | PFE  | 0.292  | 0.4188 | 0.0774 | 0.1897 | 0.2206 |
        """
        trailing = self._get_valuation_trailing(trailing)

        def flow(row: str) -> pd.DataFrame:
            return self._sum_trailing(self._income_statement.loc[:, row, :], trailing)

        operating_profitability = profitability_model.get_operating_profitability(
            flow("Revenue"),
            flow("Cost of Goods Sold"),
            flow("Selling, General and Administrative Expenses"),
            flow("Research and Development Expenses"),
            flow("Interest Expense"),
            self._balance_sheet_statement.loc[:, "Total Shareholder Equity", :],
            self._balance_sheet_statement.loc[:, "Deferred Tax Liabilities", :],
            self._balance_sheet_statement.loc[:, "Preferred Stock", :],
        )

        return finalize_dataset(
            dataset=operating_profitability,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_cash_based_operating_profitability(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the cash-based operating profitability of Ball, Gerakos, Linnainmaa and
        Nikolaev (2016): operating profit with the working capital accruals taken out, so
        that it measures the operating profit received in cash, scaled by the total assets
        at the start of the year.

        Ball et al. show that cash-based operating profitability predicts stock returns
        better than the profitability measures that include accruals, and that it subsumes
        the accrual anomaly: once the cash-based profitability is known, accruals no longer
        predict returns. The operating profit is that of Ball et al. (2015), with research
        and development added back as an investment rather than an expense of the period.

        The formula is as follows:

        - Cash-Based Operating Profitability = (Revenue - Cost of Goods Sold - Selling,
          General and Administrative Expenses - Δ Receivables - Δ Inventory - Δ Prepaids
          + Δ Deferred Revenue + Δ Accounts Payable + Δ Accrued Expenses) / Beginning Total
          Assets

        The selling, general and administrative expenses are without research and
        development (Compustat XSGA - XRD in the paper), the receivables are the total
        (trade and other) receivables and the deferred revenue the current and non-current
        deferred revenue. Each change is that of the balance over the window of the flows
        (a year with annual data, the year before the quarter with quarterly data), and the
        total assets are those at the start of the window: the end of the year before with
        annual data, the quarter four quarters back with quarterly data. Ball et al. (2016)
        scale by these lagged total assets, unlike the end-of-year assets of Ball et al.
        (2015) and Novy-Marx (2013); Chen and Zimmermann note this in their replication and
        Wang (2020) defines the measure on lagged total assets as well.
        It is an annual measure, so with quarterly data the flows are those of the trailing
        twelve months by default.

        Also known as: CbOP, cash operating profitability, cash-based profitability, accrual-free profitability.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum the income statement items
                over, which is also the number of periods the balance changes span and the
                number of periods the total assets are taken before. Defaults to None, which uses the
                trailing twelve months (4 quarters) with quarterly data and single years with
                annual data.

        Returns:
            pd.DataFrame: Cash-based operating profitability values.

        Notes:
        - Ball, R., J. Gerakos, J. T. Linnainmaa and V. Nikolaev, "Accruals, Cash Flows, and
        Operating Profitability in the Cross Section of Stock Returns," Journal of Financial
        Economics 121(1), 2016, pp. 28-45. Ball, R., J. Gerakos, J. T. Linnainmaa and V.
        Nikolaev, "Deflating Profitability," Journal of Financial Economics 117(2), 2015,
        pp. 225-248.
        - Chen, A. Y. and T. Zimmermann, "Open Source Cross-Sectional Asset Pricing,"
        Critical Finance Review 11(2), 2022, pp. 207-264 (the CBOperProf signal and its
        lagged-assets variant). Wang (2020), SSRN working paper 3507251, Table A1.
        - The first period has no earlier balances to take the changes and the total assets
        from and is NaN, so set an earlier start_date to obtain it.
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "KO", "XOM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_cash_based_operating_profitability()
        ```

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | AAPL | 0.3943 | 0.4204 | 0.4009 | 0.4394 | 0.4742 |
        | KO   | 0.1425 | 0.2178 | 0.1388 | 0.2021 | 0.0879 |
        | MSFT | 0.3035 | 0.3256 | 0.3208 | 0.3452 | 0.3146 |
        | XOM  | 0.1542 | 0.2493 | 0.2015 | 0.1796 | 0.1769 |
        """
        trailing = self._get_valuation_trailing(trailing)

        def flow(row: str) -> pd.DataFrame:
            return self._sum_trailing(self._income_statement.loc[:, row, :], trailing)

        # The balances change over the same window as the flows: a year on annual data and,
        # with the default trailing twelve months, the four quarters before on quarterly data.
        # The total assets are those at the start of that window, as in Ball et al. (2016).
        def change(balance: pd.DataFrame) -> pd.DataFrame:
            return balance.diff(trailing or 1, axis=1)

        def balance(row: str) -> pd.DataFrame:
            return self._balance_sheet_statement.loc[:, row, :]

        cash_based_operating_profitability = (
            profitability_model.get_cash_based_operating_profitability(
                flow("Revenue"),
                flow("Cost of Goods Sold"),
                flow("Selling, General and Administrative Expenses"),
                change(balance("Net Receivables")),
                change(balance("Inventory")),
                change(balance("Prepaids")),
                change(
                    balance("Deferred Revenue")
                    + balance("Deferred Revenue Non Current")
                ),
                change(balance("Accounts Payable")),
                change(balance("Accrued Expenses")),
                balance("Total Assets").shift(trailing or 1, axis=1),
            )
        )

        return finalize_dataset(
            dataset=cash_based_operating_profitability,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_cash_return_on_invested_capital(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the cash return on invested capital (CROIC), the free cash flow a company
        generates on the capital that shareholders and lenders have invested in it.

        Where the return on invested capital measures the after-tax operating profit, CROIC
        measures the cash left after the investments needed to keep and grow the business,
        so it shows how much of the return can actually be paid out, used for buybacks or
        reinvested. The invested capital is the same as in `get_return_on_invested_capital`
        (total equity plus total debt, averaged over the period), so the two compare
        directly: a CROIC persistently below the ROIC means profits are tied up in working
        capital or capital expenditure.

        The formula is as follows:

        - Cash Return on Invested Capital = Free Cash Flow / (Average Total Equity +
          Average Total Debt)

        Also known as: CROIC, cash ROIC, FCF ROIC, free cash flow return on invested capital.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data) against
                the average of the opening and closing invested capital, as for the return on
                invested capital.

        Returns:
            pd.DataFrame: Cash return on invested capital values.

        Notes:
        - CROIC is a practitioner measure (used by, among others, Morningstar); definitions
        of the invested capital differ, and this one matches the return on invested capital
        of the Finance Toolkit.
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["CAT", "DE", "CMI"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_cash_return_on_invested_capital()
        ```

        Which returns:

        |     |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:----|-------:|-------:|-------:|-------:|-------:|
        | CAT | 0.0885 | 0.0964 | 0.1776 | 0.153  | 0.1216 |
        | CMI | 0.1064 | 0.0642 | 0.1543 | 0.0155 | 0.118  |
        | DE  | 0.0814 | 0.013  | 0.0521 | 0.0509 | 0.0362 |
        """
        # The same invested capital as the return on invested capital: averaged over the
        # trailing period ends, or over the opening and closing balance of a single period.
        window = trailing or 2

        cash_return_on_invested_capital = (
            profitability_model.get_cash_return_on_invested_capital(
                self._sum_trailing(
                    self._cash_flow_statement.loc[:, "Free Cash Flow", :], trailing
                ),
                self._mean_trailing(
                    self._balance_sheet_statement.loc[:, "Total Equity", :], window
                ),
                self._mean_trailing(
                    self._balance_sheet_statement.loc[:, "Total Debt", :], window
                ),
            )
        )

        return finalize_dataset(
            dataset=cash_return_on_invested_capital,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_free_cash_flow_conversion(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the free cash flow conversion, the share of net income that a company
        turns into free cash flow.

        A conversion around or above 1 means the reported earnings are backed by cash, as
        depreciation and other non-cash charges outweigh the capital expenditure and the
        working capital tied up. A conversion persistently well below 1 is an earnings
        quality warning: profits are reported but do not arrive as cash, because of heavy
        investment, growing receivables or inventory, or aggressive accruals. Credit
        analysts and equity analysts use it alongside the income quality ratio
        (`get_income_quality_ratio`, operating cash flow over net income), which is the same
        comparison before capital expenditure.

        The formula is as follows:

        - Free Cash Flow Conversion = Free Cash Flow / Net Income

        With a loss the ratio flips sign and has no meaning, so a net income of zero or less
        gives NaN.

        Also known as: FCF conversion, cash conversion ratio, FCF to net income, free cash flow to earnings.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over. E.g. 4 with quarterly data gives the trailing twelve
                months (TTM) and 3 with annual data a three-year cumulative figure. Defaults to
                None, which uses single-period flows.

        Returns:
            pd.DataFrame: Free cash flow conversion values.

        Notes:
        - The ratio is used in the cash flow analysis of the CFA curriculum (Financial
        Reporting and Analysis) and by the rating agencies (S&P, Moody's).
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "AMZN", "NFLX"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_free_cash_flow_conversion()
        ```

        Which returns:

        |      |    2021 |     2022 |   2023 |   2024 |   2025 |
        |:-----|--------:|---------:|-------:|-------:|-------:|
        | AAPL |  0.9818 |   1.1166 | 1.0267 | 1.1608 | 0.8818 |
        | AMZN | -0.4414 | nan      | 1.0589 | 0.5549 | 0.0991 |
        | NFLX | -0.0258 |   0.3603 | 1.2807 | 0.7945 | 0.8616 |
        """
        free_cash_flow_conversion = profitability_model.get_free_cash_flow_conversion(
            self._sum_trailing(
                self._cash_flow_statement.loc[:, "Free Cash Flow", :], trailing
            ),
            self._sum_trailing(
                self._income_statement.loc[:, "Net Income", :], trailing
            ),
        )

        return finalize_dataset(
            dataset=free_cash_flow_conversion,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_free_cash_flow_to_ebitda_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the free cash flow to EBITDA ratio, the share of EBITDA that remains as
        free cash flow after interest, taxes, working capital and capital expenditure.

        EBITDA is often used as a proxy for cash flow (in leverage ratios such as net debt to
        EBITDA), and this ratio shows how good a proxy it is for a company. Credit analysts
        use it to see how much of the EBITDA is available to repay debt: capital-heavy
        businesses (utilities, telecoms, manufacturers) convert little of their EBITDA,
        asset-light businesses (software, services) most of it. EBITDA is operating income
        plus depreciation and amortization, as in the EBITDA margin.

        The formula is as follows:

        - Free Cash Flow to EBITDA = Free Cash Flow / (Operating Income + Depreciation and
          Amortization)

        An EBITDA of zero or less gives NaN, as the ratio flips sign.

        Also known as: FCF/EBITDA, EBITDA cash conversion, EBITDA to FCF conversion, cash conversion of EBITDA.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over. E.g. 4 with quarterly data gives the trailing twelve
                months (TTM) and 3 with annual data a three-year cumulative figure. Defaults to
                None, which uses single-period flows.

        Returns:
            pd.DataFrame: Free cash flow to EBITDA values.

        Notes:
        - The ratio is part of the cash flow analysis of the rating agencies (S&P Global
        Ratings' corporate methodology, Moody's) and of the CFA curriculum.
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["T", "VZ", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_free_cash_flow_to_ebitda_ratio()
        ```

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | MSFT | 0.6877 | 0.6659 | 0.5809 | 0.5623 | 0.4402 |
        | T    | 0.5813 | 0.9228 | 0.4699 | 0.4127 | 0.4237 |
        | VZ   | 0.3997 | 0.2187 | 0.4027 | 0.3902 | 0.4133 |
        """
        free_cash_flow_to_ebitda = (
            profitability_model.get_free_cash_flow_to_ebitda_ratio(
                self._sum_trailing(
                    self._cash_flow_statement.loc[:, "Free Cash Flow", :], trailing
                ),
                self._sum_trailing(
                    self._income_statement.loc[:, "Operating Income", :], trailing
                ),
                self._sum_trailing(
                    self._cash_flow_statement.loc[
                        :, "Depreciation and Amortization", :
                    ],
                    trailing,
                ),
            )
        )

        return finalize_dataset(
            dataset=free_cash_flow_to_ebitda,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_percent_accruals(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the percent accruals of Hafzalla, Lundholm and Van Winkle (2011): the
        accruals, the part of the net income that is not received in cash, as a share of the
        net income itself.

        Sloan (1996) scaled the accruals by total assets, which mostly sorts companies by the
        size of their earnings changes. Hafzalla, Lundholm and Van Winkle scale them by the
        absolute net income instead, which measures how much of the earnings consists of
        accruals: the companies whose earnings are mostly accruals earn markedly lower
        returns afterwards, and the companies with the most negative percent accruals (cash
        flows well above earnings) higher returns, a larger spread than that of the accruals
        scaled by total assets. With a loss, a positive value still means the accruals make
        the loss smaller than the cash flow would.

        The formula is as follows:

        - Percent Accruals = (Net Income - Cash Flow from Operations) / |Net Income|

        The paper takes the income before extraordinary items, which is the net income here.
        A net income of zero gives NaN. It is an annual measure, so with quarterly data the
        flows are those of the trailing twelve months by default.

        Also known as: percent operating accruals, PctAcc, accruals to earnings.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum the net income and cash
                flow from operations over. Defaults to None, which uses the trailing twelve
                months (4 quarters) with quarterly data and single years with annual data.

        Returns:
            pd.DataFrame: Percent accruals values.

        Notes:
        - Hafzalla, N., R. Lundholm and E. M. Van Winkle, "Percent Accruals," The Accounting
        Review 86(1), 2011, pp. 209-236. Sloan, R. G., "Do Stock Prices Fully Reflect
        Information in Accruals and Cash Flows about Future Earnings?," The Accounting Review
        71(3), 1996, pp. 289-315 (the accruals scaled by total assets, see
        `toolkit.models.get_accruals_ratio`).
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "KO", "XOM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_percent_accruals()
        ```

        Which returns:

        |      |    2021 |    2022 |    2023 |    2024 |    2025 |
        |:-----|--------:|--------:|--------:|--------:|--------:|
        | AAPL | -0.0988 | -0.2239 | -0.1397 | -0.2616 |  0.0047 |
        | KO   | -0.2921 | -0.1547 | -0.0826 |  0.3599 |  0.4348 |
        | MSFT | -0.2525 | -0.2241 | -0.2103 | -0.3451 | -0.3371 |
        | XOM  | -1.0889 | -0.3778 | -0.5376 | -0.6337 | -0.8018 |
        """
        trailing = self._get_valuation_trailing(trailing)

        percent_accruals = profitability_model.get_percent_accruals(
            self._sum_trailing(
                self._income_statement.loc[:, "Net Income", :], trailing
            ),
            self._sum_trailing(
                self._cash_flow_statement.loc[:, "Operating Cash Flow", :], trailing
            ),
        )

        return finalize_dataset(
            dataset=percent_accruals,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_errors
    def collect_solvency_ratios(
        self,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculates and collects all Solvency Ratios based on the provided data.

        Args:
            diluted (bool, optional): Whether to use diluted shares for the calculation. Defaults to True.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.DataFrame: Solvency ratios calculated based on the specified parameters.

        Notes:
        - The method calculates various solvency ratios for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "TSLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        solvency_ratios = toolkit.ratios.collect_solvency_ratios()

        solvency_ratios.loc['AAPL']
        ```

        Which returns:

        |                                    |     2021 |     2022 |     2023 |     2024 |     2025 |
        |:-----------------------------------|---------:|---------:|---------:|---------:|---------:|
        | Debt-to-Assets Ratio               |   0.3889 |   0.3756 |   0.3515 |   0.3262 |   0.3128 |
        | Asset Coverage Ratio               |   1.7663 |   1.6597 |   1.8001 |   1.7594 |   1.9038 |
        | Debt-to-Equity Ratio               |   2.1639 |   2.6145 |   1.9942 |   2.0906 |   1.5241 |
        | Debt Service Coverage Ratio        |   6.5852 |   5.4299 |   6.3739 |   6.4496 |   7.1203 |
        | Equity Multiplier                  |   5.255  |   6.1862 |   6.252  |   6.0251 |   5.5418 |
        | Free Cash Flow Yield               |   0.031  |   0.0525 |   0.0327 |   0.0282 |   0.0242 |
        | Net-Debt to EBITDA Ratio           |   0.8449 |   0.8337 |   0.7468 |   0.6618 |   0.5281 |
        | Gross Debt to EBITDA Ratio         |   1.1355 |   1.0149 |   0.985  |   0.8841 |   0.7764 |
        | Cash Flow Coverage Ratio           |   0.7621 |   0.922  |   0.892  |   0.9932 |   0.992  |
        | CAPEX Coverage Ratio               |   9.3855 |  11.4075 |  10.087  |  12.5176 |   8.7678 |
        | Dividend CAPEX Coverage Ratio      |   4.0716 |   4.781  |   4.2543 |   4.7913 |   3.9623 |
        | Debt-to-Capital Ratio              |   0.6839 |   0.7233 |   0.666  |   0.6764 |   0.6038 |
        | Preferred Dividend Coverage Ratio  | nan      | nan      | nan      | nan      | nan      |
        | Interest Paid to Expense Ratio     |   1.0159 |   0.9775 |   0.9669 | nan      | nan      |
        | EBITDA Interest Coverage           |  45.4567 |  44.538  |  31.9908 | nan      | nan      |
        | (EBITDA - CapEx) Interest Coverage |  41.2658 |  40.8847 |  29.2044 | nan      | nan      |
        | Cash Interest Coverage             |  49.1664 |  50.4674 |  34.979  | nan      | nan      |
        | FFO to Debt                        |   0.733  |   0.8842 |   0.9213 | nan      | nan      |
        | DCF to Debt                        |  -0.0548 |   0.0543 |   0.0566 |  -0.0116 |  -0.0655 |
        | Degree of Operating Leverage       |   1.935  |   1.2352 |   1.5355 |   3.8574 |   1.2421 |
        | Degree of Financial Leverage       |   1.0087 |   0.5621 |   0.6543 |  -0.4308 |   2.4427 |
        | Degree of Total Leverage           |   1.9518 |   0.6943 |   1.0047 |  -1.6617 |   3.034  |
        """
        self._load_statements()
        solvency_ratios_dict: dict = {}

        solvency_ratios_dict["Debt-to-Assets Ratio"] = self.get_debt_to_assets_ratio(
            trailing=trailing
        )
        solvency_ratios_dict["Asset Coverage Ratio"] = self.get_asset_coverage_ratio(
            trailing=trailing
        )
        solvency_ratios_dict["Debt-to-Equity Ratio"] = self.get_debt_to_equity_ratio(
            trailing=trailing
        )
        solvency_ratios_dict["Debt Service Coverage Ratio"] = (
            self.get_debt_service_coverage_ratio(trailing=trailing)
        )
        solvency_ratios_dict["Equity Multiplier"] = self.get_equity_multiplier(
            trailing=trailing
        )
        solvency_ratios_dict["Free Cash Flow Yield"] = self.get_free_cash_flow_yield(
            diluted=diluted, trailing=trailing
        )
        solvency_ratios_dict["Net-Debt to EBITDA Ratio"] = (
            self.get_net_debt_to_ebitda_ratio(trailing=trailing)
        )
        solvency_ratios_dict["Gross Debt to EBITDA Ratio"] = (
            self.get_gross_debt_to_ebitda_ratio(trailing=trailing)
        )
        solvency_ratios_dict["Cash Flow Coverage Ratio"] = (
            self.get_cash_flow_coverage_ratio(trailing=trailing)
        )
        solvency_ratios_dict["CAPEX Coverage Ratio"] = self.get_capex_coverage_ratio(
            trailing=trailing
        )
        solvency_ratios_dict["Dividend CAPEX Coverage Ratio"] = (
            self.get_capex_dividend_coverage_ratio(trailing=trailing)
        )
        solvency_ratios_dict["Debt-to-Capital Ratio"] = self.get_debt_to_capital_ratio(
            trailing=trailing
        )
        solvency_ratios_dict["Preferred Dividend Coverage Ratio"] = (
            self.get_preferred_dividend_coverage_ratio(trailing=trailing)
        )
        solvency_ratios_dict["Interest Paid to Expense Ratio"] = (
            self.get_interest_paid_to_expense_ratio(trailing=trailing)
        )
        solvency_ratios_dict["EBITDA Interest Coverage"] = (
            self.get_ebitda_interest_coverage_ratio(trailing=trailing)
        )
        solvency_ratios_dict["(EBITDA - CapEx) Interest Coverage"] = (
            self.get_ebitda_minus_capex_interest_coverage_ratio(trailing=trailing)
        )
        solvency_ratios_dict["Cash Interest Coverage"] = (
            self.get_cash_interest_coverage_ratio(trailing=trailing)
        )
        solvency_ratios_dict["FFO to Debt"] = (
            self.get_funds_from_operations_to_debt_ratio(trailing=trailing)
        )
        solvency_ratios_dict["DCF to Debt"] = (
            self.get_discretionary_cash_flow_to_debt_ratio(trailing=trailing)
        )
        solvency_ratios_dict["Degree of Operating Leverage"] = (
            self.get_degree_of_operating_leverage(trailing=trailing)
        )
        solvency_ratios_dict["Degree of Financial Leverage"] = (
            self.get_degree_of_financial_leverage(trailing=trailing)
        )
        solvency_ratios_dict["Degree of Total Leverage"] = (
            self.get_degree_of_total_leverage(trailing=trailing)
        )

        self._solvency_ratios = (
            pd.concat(solvency_ratios_dict)
            .swaplevel(0, 1)
            .sort_index(level=0, sort_remaining=False)
            .dropna(axis="columns", how="all")
        )

        self._solvency_ratios = self._solvency_ratios.loc[self._tickers]

        self._solvency_ratios = self._solvency_ratios.pipe(
            apply_rounding, rounding if rounding is not None else self._rounding
        )

        # Sorted again so the index follows the financial statements' order.
        available_columns = [
            column
            for column in self._income_statement.columns
            if column in self._solvency_ratios
        ]
        self._solvency_ratios = self._solvency_ratios.reindex(available_columns, axis=1)

        if growth:
            self._solvency_ratios_growth = calculate_growth(
                dataset=self._solvency_ratios,
                lag=lag,
                compound=compound,
                rounding=rounding if rounding is not None else self._rounding,
                axis="columns",
            )

        solvency_ratios = self._solvency_ratios
        solvency_ratios_growth = self._solvency_ratios_growth

        if standardize:
            standardize_rounding = rounding if rounding is not None else self._rounding
            if growth:
                solvency_ratios_growth = calculate_standardization(
                    dataset=solvency_ratios_growth,
                    rounding=standardize_rounding,
                    axis="columns",
                )
            else:
                solvency_ratios = calculate_standardization(
                    dataset=solvency_ratios,
                    rounding=standardize_rounding,
                    axis="columns",
                )

        if len(self._tickers) == 1:
            return (
                solvency_ratios_growth.loc[self._tickers[0]]
                if growth
                else solvency_ratios.loc[self._tickers[0]]
            ).loc[:, self._start_date : self._end_date]

        return (solvency_ratios_growth if growth else solvency_ratios).loc[
            :, self._start_date : self._end_date
        ]

    @handle_portfolio
    @handle_errors
    def get_debt_to_assets_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ):
        """
        Calculate the debt to assets ratio, a solvency ratio that measures the proportion
        of a company's assets that are financed by debt.

        This ratio, also known as the Debt Ratio, indicates the percentage of a company's
        total assets that are funded by debt. It is a measure of a company's financial
        leverage and indicates the extent to which a company relies on borrowed funds to
        finance its operations. A higher ratio implies a higher level of debt in the company's
        capital structure, which could increase financial risk.

        The formula is as follows:

        - Debt to Assets Ratio = Total Debt / Total Assets

        Also known as: asset leverage, debt ratio.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.DataFrame: Debt to assets ratio values.

        Notes:
        - The method retrieves historical data and calculates the ratio for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["NEE", "DUK", "SO"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_debt_to_assets_ratio()
        ```

        Which returns:

        |     |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:----|-------:|-------:|-------:|-------:|-------:|
        | DUK | 0.4025 | 0.4206 | 0.4548 | 0.4574 | 0.4642 |
        | NEE | 0.3891 | 0.4088 | 0.4125 | 0.433  | 0.4495 |
        | SO  | 0.4349 | 0.4384 | 0.4557 | 0.4565 | 0.4757 |
        """
        if trailing:
            debt_to_assets_ratio = solvency_model.get_debt_to_assets_ratio(
                self._mean_trailing(
                    self._balance_sheet_statement.loc[:, "Total Debt", :], trailing
                ),
                self._mean_trailing(
                    self._balance_sheet_statement.loc[:, "Total Assets", :], trailing
                ),
            )
        else:
            debt_to_assets_ratio = solvency_model.get_debt_to_assets_ratio(
                self._balance_sheet_statement.loc[:, "Total Debt", :],
                self._balance_sheet_statement.loc[:, "Total Assets", :],
            )

        return finalize_dataset(
            dataset=debt_to_assets_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_asset_coverage_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ):
        """
        Calculate the asset coverage ratio, a solvency ratio that measures how well a
        company's tangible assets, after settling non-debt current liabilities, can
        cover its total debt.

        This ratio is commonly used by lenders and bondholders to assess the extent to
        which a company's hard (tangible) assets would be available to repay debt
        obligations in a liquidation scenario, since intangible assets (e.g. goodwill)
        typically have little to no recovery value and non-debt current liabilities are
        assumed to be settled first out of current assets. Short-term debt is netted
        out of current liabilities before subtracting, since it is already captured in
        total debt and would otherwise be double-counted.

        The formula is as follows:

        - Asset Coverage Ratio = [(Total Assets - Goodwill - Intangible Assets) -
          (Total Current Liabilities - Short Term Debt)] / Total Debt

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.DataFrame: Asset coverage ratio values.

        Notes:
        - The method retrieves historical data and calculates the ratio for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["KMI", "WMB", "OKE"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_asset_coverage_ratio()
        ```

        Which returns:

        |     |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:----|-------:|-------:|-------:|-------:|-------:|
        | KMI | 1.4053 | 1.3851 | 1.4027 | 1.4371 | 1.5344 |
        | OKE | 1.4978 | 1.5692 | 1.6095 | 1.526  | 1.5656 |
        | WMB | 1.5739 | 1.6223 | 1.5972 | 1.6317 | 1.6245 |
        """
        intangible_assets = get_goodwill_and_intangible_assets(
            self._balance_sheet_statement
        )

        if trailing:
            asset_coverage_ratio = solvency_model.get_asset_coverage_ratio(
                self._mean_trailing(
                    self._balance_sheet_statement.loc[:, "Total Assets", :], trailing
                ),
                self._mean_trailing(intangible_assets, trailing),
                self._mean_trailing(
                    self._balance_sheet_statement.loc[
                        :, "Total Current Liabilities", :
                    ],
                    trailing,
                ),
                self._mean_trailing(
                    self._balance_sheet_statement.loc[:, "Short Term Debt", :], trailing
                ),
                self._mean_trailing(
                    self._balance_sheet_statement.loc[:, "Total Debt", :], trailing
                ),
            )
        else:
            asset_coverage_ratio = solvency_model.get_asset_coverage_ratio(
                self._balance_sheet_statement.loc[:, "Total Assets", :],
                intangible_assets,
                self._balance_sheet_statement.loc[:, "Total Current Liabilities", :],
                self._balance_sheet_statement.loc[:, "Short Term Debt", :],
                self._balance_sheet_statement.loc[:, "Total Debt", :],
            )

        return finalize_dataset(
            dataset=asset_coverage_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_debt_to_equity_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ):
        """
        Calculate the debt to equity ratio, a solvency ratio that measures the
        proportion of a company's equity that is financed by debt. This ratio is also known
        as the Gearing Ratio.

        The debt to equity ratio, for short the D/E ratio, indicates the relative
        contribution of debt and equity to a company's capital structure. It helps assess
        the level of financial risk a company carries due to its debt obligations. A higher
        ratio implies a higher reliance on debt to finance the business, which could increase
        risk but also potentially lead to higher returns for shareholders.

        The formula is as follows:

        - Debt to Equity Ratio = Total Debt / Total Equity

        The debt to equity ratio is compared within an industry, since utilities, real
        estate and companies with a financing arm (car makers, machinery makers) carry far
        more debt than software companies. Equity is a book value, so large buybacks or
        accumulated losses can make it small or negative, and the ratio then becomes
        extreme or negative without saying anything about the debt load; the debt to
        capital ratio (`get_debt_to_capital_ratio`) and net debt to EBITDA
        (`get_net_debt_to_ebitda_ratio`) are more robust in those cases.

        Also known as: D/E ratio, leverage ratio, gearing ratio.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.DataFrame: Debt to equity ratio values.

        Notes:
        - The method retrieves historical data and calculates the ratio for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["CAT", "DE", "CNH"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_debt_to_equity_ratio()
        ```

        Which returns:

        |     |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:----|-------:|-------:|-------:|-------:|-------:|
        | CAT | 2.288  | 2.3279 | 1.9422 | 1.9703 | 2.0326 |
        | CNH | 4.5462 | 3.3243 | 3.3897 | 3.4969 | 3.4548 |
        | DE  | 2.6435 | 2.5643 | 2.9102 | 2.8555 | 2.4584 |
        """
        if trailing:
            debt_to_equity_ratio = solvency_model.get_debt_to_equity_ratio(
                self._mean_trailing(
                    self._balance_sheet_statement.loc[:, "Total Debt", :], trailing
                ),
                self._mean_trailing(
                    self._balance_sheet_statement.loc[:, "Total Equity", :], trailing
                ),
            )
        else:
            debt_to_equity_ratio = solvency_model.get_debt_to_equity_ratio(
                self._balance_sheet_statement.loc[:, "Total Debt", :],
                self._balance_sheet_statement.loc[:, "Total Equity", :],
            )

        return finalize_dataset(
            dataset=debt_to_equity_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_interest_coverage_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ):
        """
        Calculate the interest coverage ratio, a solvency ratio that measures a company's
        ability to pay its interest expenses on outstanding debt.

        The interest coverage ratio evaluates a company's ability to meet its interest
        obligations from its earnings before interest and taxes (EBIT), taken as operating
        income, following the CFA Institute definition. A higher ratio indicates a company's
        stronger ability to cover its interest payments using its earnings, implying lower
        financial risk. Conversely, a lower ratio suggests a company may have difficulty
        meeting its interest obligations and could be at higher risk of default.

        The formula is as follows:

        - Interest Coverage Ratio = EBIT / Interest Expense

        Lenders and rating agencies use interest coverage to judge how much room a company has
        before its earnings no longer cover its interest bill, and loan covenants often set a
        minimum. Some covenants use EBITDA instead of EBIT, which gives a higher ratio, most of
        all for capital-intensive companies. The ratio is NaN when no interest expense is
        reported, which is often a gap in the reported data rather than an absence of debt.

        Also known as: times interest earned (TIE), EBIT interest coverage.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.DataFrame: Interest coverage ratio values.

        Notes:
        - The method retrieves historical data and calculates the ratio for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["MKC", "CPB", "HSY"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_interest_coverage_ratio()
        ```

        Which returns:

        |     |    2021 |    2022 |    2023 |    2024 |   2025 |
        |:----|--------:|--------:|--------:|--------:|-------:|
        | CPB |  6.2714 |  6.2381 |  6.9787 |  4.743  | 3.4116 |
        | HSY | 15.7396 | 16.1375 | 15.8557 | 16.6262 | 6.8308 |
        | MKC |  7.4312 |  5.7921 |  4.6254 |  5.0635 | 5.4577 |
        """
        if trailing:
            interest_coverage_ratio = solvency_model.get_interest_coverage_ratio(
                self._sum_trailing(
                    self._income_statement.loc[:, "Operating Income", :], trailing
                ),
                self._sum_trailing(
                    self._income_statement.loc[:, "Interest Expense", :], trailing
                ),
            )
        else:
            interest_coverage_ratio = solvency_model.get_interest_coverage_ratio(
                self._income_statement.loc[:, "Operating Income", :],
                self._income_statement.loc[:, "Interest Expense", :],
            )

        return finalize_dataset(
            dataset=interest_coverage_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_equity_multiplier(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ):
        """
        Calculate the equity multiplier, a solvency ratio that measures the degree to which
        a company uses borrowed money (debt) to finance its operations and growth.

        The equity multiplier helps assess the financial leverage of a company by indicating
        how much of its assets are financed by equity versus debt. A higher equity multiplier
        suggests that the company relies more on debt financing, which can amplify returns but
        also increases financial risk. Conversely, a lower equity multiplier indicates a
        larger portion of assets is financed by equity, potentially lowering financial risk.

        The formula is as follows:

        - Equity Multiplier = Average Total Assets / Average Total Equity

        The equity multiplier is the leverage component of the DuPont analysis: return on
        equity equals return on assets times the equity multiplier. Banks run multipliers of
        around 10 or more, which is why their small returns on assets translate into returns
        on equity in the teens. For other companies a rising multiplier can come from new
        debt but also from buybacks that shrink equity.

        Also known as: financial leverage multiplier.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.DataFrame: Equity multiplier values.

        Notes:
        - The method retrieves historical data and calculates the ratio for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["HSBA.L", "BNP.PA", "SAN.MC"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_equity_multiplier()
        ```

        Which returns:

        |        |    2021 |    2022 |    2023 |    2024 |    2025 |
        |:-------|--------:|--------:|--------:|--------:|--------:|
        | BNP.PA | 23.2121 | 23.2436 | 21.5821 | 20.1378 | 20.6444 |
        | HSBA.L | 14.43   | 15.1159 | 15.8512 | 15.7336 | 16.5358 |
        | SAN.MC | 16.4782 | 17.1112 | 17.4988 | 17.1772 | 16.8333 |
        """
        if trailing:
            equity_multiplier = solvency_model.get_equity_multiplier(
                self._mean_trailing(
                    self._balance_sheet_statement.loc[:, "Total Assets", :], trailing
                ),
                self._mean_trailing(
                    self._balance_sheet_statement.loc[:, "Total Equity", :], trailing
                ),
            )
        else:
            equity_multiplier = solvency_model.get_equity_multiplier(
                self._balance_sheet_statement.loc[:, "Total Assets", :]
                .T.rolling(2)
                .mean()
                .T,
                self._balance_sheet_statement.loc[:, "Total Equity", :]
                .T.rolling(2)
                .mean()
                .T,
            )

        return finalize_dataset(
            dataset=equity_multiplier,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_debt_service_coverage_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ):
        """
        Calculate the debt service coverage ratio, a solvency ratio that measures how well a
        company's operating earnings cover its debt service: the interest of the period and the
        principal due within a year.

        The debt service coverage ratio compares EBITDA (operating income plus depreciation and
        amortization) with the total debt service, the interest expense plus the short-term debt
        on the balance sheet, which includes the current portion of long-term debt. A ratio
        below 1 means the operating earnings do not cover the debt service, so the company
        depends on refinancing or cash reserves; lenders often require 1.25 or more. A company
        without short-term debt and interest has no debt service, which gives NaN.

        The formula is as follows:

        - Debt Service Coverage Ratio = (Operating Income + Depreciation and Amortization) /
          (Interest Expense + Short Term Debt x Window Years)

        The short-term debt is the principal due over the next year, so it is scaled to the
        length of the window the earnings and interest cover: a quarter (0.25) with quarterly
        data, a year with yearly data and the trailing multiple of either with trailing (e.g.
        trailing=4 on quarterly data compares a year of earnings with a year of principal).

        Also known as: DSCR, debt service ratio.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.DataFrame: Debt service coverage ratio values.

        Notes:
        - The method retrieves historical data and calculates the ratio for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["WM", "RSG", "WCN"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_debt_service_coverage_ratio()
        ```

        Which returns:

        |     |    2021 |    2022 |   2023 |   2024 |   2025 |
        |:----|--------:|--------:|-------:|-------:|-------:|
        | RSG | 10.7809 |  4.5724 | 3.0393 | 3.5467 | 4.4872 |
        | WCN |  9.8794 | 10.422  | 7.8257 | 7.451  | 7.4022 |
        | WM  |  4.7698 |  7.0556 | 7.223  | 3.3464 | 3.2914 |
        """
        # The years covered by the summed earnings and interest, to which the principal due
        # within a year is scaled.
        window_years = (trailing or 1) * (0.25 if self._quarterly else 1.0)

        if trailing:
            debt_service_coverage_ratio = (
                solvency_model.get_debt_service_coverage_ratio(
                    self._sum_trailing(
                        self._income_statement.loc[:, "Operating Income", :], trailing
                    ),
                    self._sum_trailing(
                        self._cash_flow_statement.loc[
                            :, "Depreciation and Amortization", :
                        ],
                        trailing,
                    ),
                    self._sum_trailing(
                        self._income_statement.loc[:, "Interest Expense", :], trailing
                    ),
                    self._mean_trailing(
                        self._balance_sheet_statement.loc[:, "Short Term Debt", :],
                        trailing,
                    ),
                    window_years=window_years,
                )
            )
        else:
            debt_service_coverage_ratio = (
                solvency_model.get_debt_service_coverage_ratio(
                    self._income_statement.loc[:, "Operating Income", :],
                    self._cash_flow_statement.loc[
                        :, "Depreciation and Amortization", :
                    ],
                    self._income_statement.loc[:, "Interest Expense", :],
                    self._balance_sheet_statement.loc[:, "Short Term Debt", :],
                    window_years=window_years,
                )
            )

        return finalize_dataset(
            dataset=debt_service_coverage_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_free_cash_flow_yield(
        self,
        show_daily: bool = False,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ):
        """
        Calculates the free cash flow yield ratio, which measures the free cash flow
        relative to the market capitalization of the company.

        The free cash flow yield ratio is a measure of how efficiently a company generates
        free cash flow relative to its market value. It provides insights into whether the
        company's valuation is reasonable compared to the amount of cash it generates.

        The formula is as follows:

        - Free Cash Flow Yield Ratio = Free Cash Flow / Market Capitalization

        The free cash flow yield is the cash a company generates for its shareholders as a
        share of what the market pays for them, and is often set against bond yields or
        the dividend yield to judge how much room there is for dividends and buybacks.
        Free cash flow swings with capital expenditure and working capital, so a single
        year can mislead, and a high yield can also mean the market expects free cash flow
        to fall.

        Also known as: FCF yield.

        Args:
            show_daily (bool, optional): Whether to show daily values: the close of each trading day
                against the statement figures of the latest period that has ended on or before
                that day (also when it ended before the start date), so no figures are used before
                their period closes. Defaults to False.
            diluted (bool, optional): Whether to use diluted shares for market capitalization. Defaults to True.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses the
                trailing twelve months (4 quarters) with quarterly data and single years with
                annual data.

        Returns:
            pd.DataFrame: Free cash flow yield ratio values.

        Notes:
        - The method retrieves historical data and calculates the ratio for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["MO", "PM", "BTI"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_free_cash_flow_yield()
        ```

        Which returns:

        |     |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:----|-------:|-------:|-------:|-------:|-------:|
        | BTI | 0.1492 | 0.1348 | 0.1953 | 0.1505 | 0.0614 |
        | MO  | 0.0942 | 0.0976 | 0.1268 | 0.0959 | 0.0938 |
        | PM  | 0.0758 | 0.0619 | 0.054  | 0.0575 | 0.0427 |
        """
        # A yield compares the price with a year of free cash flow, so quarterly data defaults to
        # the trailing twelve months.
        trailing = self._get_valuation_trailing(trailing)

        average_shares = (
            self._income_statement.loc[:, "Weighted Average Shares Diluted", :]
            if diluted
            else self._income_statement.loc[:, "Weighted Average Shares", :]
        )

        # The free cash flow is summed over the trailing periods before it is spread over the days.
        free_cash_flow = self._sum_trailing(
            self._cash_flow_statement.loc[:, "Free Cash Flow", :], trailing
        )

        years = self._balance_sheet_statement.columns
        begin, end = str(years[0]), str(years[-1])

        share_prices = self._get_share_prices(begin, end, show_daily)

        if show_daily:
            average_shares = map_period_data_to_daily_data(
                period_data=average_shares,
                daily_dates=share_prices.index,
                quarterly=self._quarterly,
                period_end_dates=self._period_end_dates,
            )

            free_cash_flow = map_period_data_to_daily_data(
                period_data=free_cash_flow,
                daily_dates=share_prices.index,
                quarterly=self._quarterly,
                period_end_dates=self._period_end_dates,
            )

        # The market cap uses the close at the end of the period (or on the day), not an average.
        market_cap = valuation_model.get_market_cap(share_prices, average_shares)

        free_cash_flow_yield = solvency_model.get_free_cash_flow_yield(
            free_cash_flow, market_cap
        )

        return finalize_dataset(
            dataset=free_cash_flow_yield,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
            axis="rows" if show_daily else "columns",
            row_slice=show_daily,
        )

    @handle_portfolio
    @handle_errors
    def get_net_debt_to_ebitda_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ):
        """
        Calculates the net debt to EBITDA ratio, which measures the net debt of the company
        relative to its EBITDA (Earnings Before Interest, Taxes, Depreciation, and Amortization).

        The net debt to EBITDA ratio is a measure of a company's ability to manage its debt
        obligations in relation to its earnings and cash flow. A lower ratio indicates better
        financial health and a stronger ability to manage debt.

        The formula is as follows:

        - Net Debt to EBITDA Ratio = Net Debt / EBITDA

        Net debt to EBITDA is the leverage measure lenders, rating agencies and loan
        covenants use most: it reads as the number of years of EBITDA needed to repay the
        debt net of cash. What counts as high depends on how stable the earnings are, so
        utilities and telecoms carry more than cyclical companies. A negative value means
        the company holds more cash than debt, and the ratio becomes meaningless when
        EBITDA is close to zero or negative.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.DataFrame: Net debt to EBITDA ratio values.

        Notes:
        - The method retrieves historical data and calculates the ratio for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["VZ", "TMUS", "CHTR"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_net_debt_to_ebitda_ratio()
        ```

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | CHTR | 4.5163 | 4.5991 | 4.5983 | 4.3482 | 4.3822 |
        | TMUS | 4.3904 | 5.3127 | 4.0131 | 3.524  | 3.6175 |
        | VZ   | 3.6331 | 3.6523 | 3.7214 | 3.3851 | 3.8134 |
        """
        if trailing:
            net_debt_to_ebitda_ratio = solvency_model.get_net_debt_to_ebitda_ratio(
                self._sum_trailing(
                    self._income_statement.loc[:, "Operating Income", :], trailing
                ),
                self._sum_trailing(
                    self._cash_flow_statement.loc[
                        :, "Depreciation and Amortization", :
                    ],
                    trailing,
                ),
                self._mean_trailing(
                    self._balance_sheet_statement.loc[:, "Net Debt", :], trailing
                ),
            )
        else:
            net_debt_to_ebitda_ratio = solvency_model.get_net_debt_to_ebitda_ratio(
                self._income_statement.loc[:, "Operating Income", :],
                self._cash_flow_statement.loc[:, "Depreciation and Amortization", :],
                self._balance_sheet_statement.loc[:, "Net Debt", :],
            )

        return finalize_dataset(
            dataset=net_debt_to_ebitda_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_gross_debt_to_ebitda_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ):
        """
        Calculates the gross debt to EBITDA ratio, which measures the total (gross)
        debt of the company relative to its EBITDA (Earnings Before Interest, Taxes,
        Depreciation, and Amortization).

        This differs from the Net-Debt to EBITDA Ratio in that it uses total (gross)
        debt rather than net debt (total debt minus cash and cash equivalents). Gross
        debt to EBITDA is a more conservative leverage measure since it does not
        assume that a company's cash balance would actually be used to pay down debt,
        which matters when comparing companies with restricted cash, cash earmarked
        for other purposes, or when assessing gross refinancing risk rather than net
        economic leverage.

        The formula is as follows:

        - Gross Debt to EBITDA Ratio = Total Debt / EBITDA

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.DataFrame: Gross debt to EBITDA ratio values.

        Notes:
        - The method retrieves historical data and calculates the ratio for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["ORCL", "IBM", "AVGO"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_gross_debt_to_ebitda_ratio()
        ```

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | AVGO | 2.766  | 2.0812 | 1.9782 | 2.8785 | 1.9013 |
        | IBM  | 4.1515 | 4.1625 | 4.2157 | 3.9615 | 3.8345 |
        | ORCL | 5.4    | 4.7123 | 4.3955 | 4.3646 | 5.1938 |
        """
        if trailing:
            gross_debt_to_ebitda_ratio = solvency_model.get_gross_debt_to_ebitda_ratio(
                self._mean_trailing(
                    self._balance_sheet_statement.loc[:, "Total Debt", :], trailing
                ),
                self._sum_trailing(
                    self._income_statement.loc[:, "Operating Income", :], trailing
                ),
                self._sum_trailing(
                    self._cash_flow_statement.loc[
                        :, "Depreciation and Amortization", :
                    ],
                    trailing,
                ),
            )
        else:
            gross_debt_to_ebitda_ratio = solvency_model.get_gross_debt_to_ebitda_ratio(
                self._balance_sheet_statement.loc[:, "Total Debt", :],
                self._income_statement.loc[:, "Operating Income", :],
                self._cash_flow_statement.loc[:, "Depreciation and Amortization", :],
            )

        return finalize_dataset(
            dataset=gross_debt_to_ebitda_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_cash_flow_coverage_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ):
        """
        Calculate the cash flow coverage ratio, a solvency ratio that measures a company's
        ability to pay off its debt with its operating cash flow.

        The cash flow coverage ratio assesses a company's ability to meet its debt obligations
        by comparing its operating cash flow to its total debt. A higher ratio indicates a
        stronger ability to cover its debt with cash generated from operations.

        The formula is as follows:

        - Cash Flow Coverage Ratio = Cash Flow from Operations / Total Debt

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.DataFrame: Cash flow coverage ratio values.

        Notes:
        - The method retrieves historical data and calculates the ratio for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["MCD", "SBUX", "YUM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_cash_flow_coverage_ratio()
        ```

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | MCD  | 0.1852 | 0.1517 | 0.181  | 0.1819 | 0.1925 |
        | SBUX | 0.2537 | 0.1847 | 0.2443 | 0.2362 | 0.1784 |
        | YUM  | 0.1407 | 0.1127 | 0.1332 | 0.1375 | 0.1688 |
        """
        if trailing:
            cash_flow_coverage_ratio = solvency_model.get_cash_flow_coverage_ratio(
                self._sum_trailing(
                    self._cash_flow_statement.loc[:, "Cash Flow from Operations", :],
                    trailing,
                ),
                self._mean_trailing(
                    self._balance_sheet_statement.loc[:, "Total Debt", :], trailing
                ),
            )
        else:
            cash_flow_coverage_ratio = solvency_model.get_cash_flow_coverage_ratio(
                self._cash_flow_statement.loc[:, "Cash Flow from Operations", :],
                self._balance_sheet_statement.loc[:, "Total Debt", :],
            )

        return finalize_dataset(
            dataset=cash_flow_coverage_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_capex_coverage_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ):
        """
        Calculate the capital expenditure coverage ratio, a solvency ratio that
        measures a company's ability to cover its capital expenditures with its
        cash flow from operations.

        The capex coverage ratio evaluates a company's ability to fund its capital
        expenditures, which are essential for maintaining and growing its business,
        using the cash generated from its operations. A higher ratio indicates a
        stronger ability to fund capital investments from operating cash flow.

        The formula is as follows:

        - Capital Expenditure Coverage Ratio = Cash Flow from Operations / Capital Expenditure

        Also known as: capital expenditure coverage.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.DataFrame: Capital expenditure coverage ratio values.

        Notes:
        - The method retrieves historical data and calculates the ratio for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["TSM", "INTC", "MU"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_capex_coverage_ratio()
        ```

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | INTC | 1.449  | 0.6161 | 0.4455 | 0.3461 | 0.6621 |
        | MU   | 1.2431 | 1.2581 | 0.2031 | 1.0144 | 1.1052 |
        | TSM  | 1.3093 | 1.4781 | 1.2999 | 1.9102 | 1.7738 |
        """
        if trailing:
            capex_coverage_ratio = solvency_model.get_capex_coverage_ratio(
                self._sum_trailing(
                    self._cash_flow_statement.loc[:, "Cash Flow from Operations", :],
                    trailing,
                ),
                self._sum_trailing(
                    self._cash_flow_statement.loc[:, "Capital Expenditure", :], trailing
                ),
            )
        else:
            capex_coverage_ratio = solvency_model.get_capex_coverage_ratio(
                self._cash_flow_statement.loc[:, "Cash Flow from Operations", :],
                self._cash_flow_statement.loc[:, "Capital Expenditure", :],
            )

        return finalize_dataset(
            dataset=capex_coverage_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_capex_dividend_coverage_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ):
        """
        Calculate the dividend paid and capital expenditure coverage ratio, a solvency ratio
        that measures a company's ability to cover both its capital expenditures and dividend
        payments with its cash flow from operations.

        The capex dividend coverage ratio assesses whether a company's cash flow from
        operations is sufficient to cover both its capital expenditures (which are essential
        for maintaining and growing its business) and its dividend payments to shareholders.
        A higher ratio indicates a stronger ability to fund both capex and dividends from
        operating cash flow.

        The formula is as follows:

        - Dividend Paid and Capital Expenditure Coverage Ratio = Cash Flow from Operations /
            (Capital Expenditure + Dividends Paid)

        Also known as: capex dividend coverage.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.DataFrame: Dividend paid and capex coverage ratio values.

        Notes:
        - The method retrieves historical data and calculates the ratio for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["XOM", "CVX", "COP"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_capex_dividend_coverage_ratio()
        ```

        Which returns:

        |     |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:----|-------:|-------:|-------:|-------:|-------:|
        | COP | 2.2122 | 1.7824 | 1.1862 | 1.2766 | 1.1963 |
        | CVX | 1.5968 | 2.1621 | 1.3108 | 1.1148 | 1.1276 |
        | XOM | 1.7826 | 2.303  | 1.5021 | 1.3417 | 1.14   |
        """
        if trailing:
            dividend_capex_coverage_ratio = (
                solvency_model.get_dividend_capex_coverage_ratio(
                    self._sum_trailing(
                        self._cash_flow_statement.loc[
                            :, "Cash Flow from Operations", :
                        ],
                        trailing,
                    ),
                    self._sum_trailing(
                        self._cash_flow_statement.loc[:, "Capital Expenditure", :],
                        trailing,
                    ),
                    self._sum_trailing(
                        self._cash_flow_statement.loc[:, "Dividends Paid", :], trailing
                    ),
                )
            )
        else:
            dividend_capex_coverage_ratio = (
                solvency_model.get_dividend_capex_coverage_ratio(
                    self._cash_flow_statement.loc[:, "Cash Flow from Operations", :],
                    self._cash_flow_statement.loc[:, "Capital Expenditure", :],
                    self._cash_flow_statement.loc[:, "Dividends Paid", :],
                )
            )

        return finalize_dataset(
            dataset=dividend_capex_coverage_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_debt_to_capital_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ):
        """
        Calculate the debt to capital ratio, a solvency ratio that measures the
        proportion of a company's total capital (debt plus equity) that is financed
        by debt.

        Unlike the debt to equity ratio, which can theoretically exceed one or become
        negative with low or negative equity, the debt to capital ratio is bounded
        between 0 and 1 under normal circumstances, making it easier to compare
        across companies with very different capital structures.

        The formula is as follows:

        - Debt to Capital Ratio = Total Debt / (Total Debt + Total Equity)

        Also known as: capitalization ratio.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.DataFrame: Debt to capital ratio values.

        Notes:
        - The method retrieves historical data and calculates the ratio for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["BMY", "AMGN", "GILD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_debt_to_capital_ratio()
        ```

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | AMGN | 0.8325 | 0.9141 | 0.912  | 0.9109 | 0.8631 |
        | BMY  | 0.5588 | 0.5668 | 0.5844 | 0.7575 | 0.7181 |
        | GILD | 0.559  | 0.5433 | 0.5234 | 0.5812 | 0.5244 |
        """
        if trailing:
            debt_to_capital_ratio = solvency_model.get_debt_to_capital_ratio(
                self._mean_trailing(
                    self._balance_sheet_statement.loc[:, "Total Debt", :], trailing
                ),
                self._mean_trailing(
                    self._balance_sheet_statement.loc[:, "Total Equity", :], trailing
                ),
            )
        else:
            debt_to_capital_ratio = solvency_model.get_debt_to_capital_ratio(
                self._balance_sheet_statement.loc[:, "Total Debt", :],
                self._balance_sheet_statement.loc[:, "Total Equity", :],
            )

        return finalize_dataset(
            dataset=debt_to_capital_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_preferred_dividend_coverage_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ):
        """
        Calculate the preferred dividend coverage ratio, a solvency ratio that
        measures a company's ability to pay dividends owed to preferred shareholders
        out of its net income.

        The formula is as follows:

        - Preferred Dividend Coverage Ratio = Net Income / |Preferred Dividends Paid|

        Also known as: preferred dividend cover.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.DataFrame: Preferred dividend coverage ratio values.

        Notes:
        - The method retrieves historical data and calculates the ratio for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.
        - This ratio is only meaningful for companies that have preferred stock
        outstanding. It is NaN when no preferred dividends are reported separately in
        the cash flow statement, which also happens for some companies that do pay them.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["PNC", "USB", "COF"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_preferred_dividend_coverage_ratio()
        ```

        Which returns:

        |     |    2021 |    2022 |    2023 |    2024 |    2025 |
        |:----|--------:|--------:|--------:|--------:|--------:|
        | COF | 43.6679 | 30.8947 | 20.0965 | 19.4956 |  8.6548 |
        | PNC | 23.2146 | 18.9635 | 12.2926 | 15.6136 | 21.3506 |
        | USB | 24.6916 | 18.398  | 14.8123 | 16.5983 | 21.5389 |
        """
        if trailing:
            preferred_dividend_coverage_ratio = (
                solvency_model.get_preferred_dividend_coverage_ratio(
                    self._sum_trailing(
                        self._income_statement.loc[:, "Net Income", :], trailing
                    ),
                    self._sum_trailing(
                        self._cash_flow_statement.loc[:, "Preferred Dividends Paid", :],
                        trailing,
                    ),
                )
            )
        else:
            preferred_dividend_coverage_ratio = (
                solvency_model.get_preferred_dividend_coverage_ratio(
                    self._income_statement.loc[:, "Net Income", :],
                    self._cash_flow_statement.loc[:, "Preferred Dividends Paid", :],
                )
            )

        return finalize_dataset(
            dataset=preferred_dividend_coverage_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_interest_paid_to_expense_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ):
        """
        Calculate the interest paid to interest expense ratio, which measures how
        much of the accrual-based interest expense reported on the income statement
        was actually paid out in cash during the period.

        A ratio consistently below one can indicate that interest is being accrued
        (e.g. on payment-in-kind debt) rather than paid, while a ratio well above one
        can indicate the payment of previously accrued interest or a mismatch between
        the cash and accrual reporting periods, both of which are relevant
        quality-of-earnings signals.

        The formula is as follows:

        - Interest Paid to Expense Ratio = Interest Paid / Interest Expense

        A common reason for interest paid to exceed the interest expense is
        capitalized interest: companies building plants, fabs or networks add part of
        their interest to the cost of the asset, so the income statement shows less
        interest than was paid. When the data has no interest paid for a year, the
        ratio shows as 0, which means the figure is missing, not that no interest was paid.

        Also known as: cash interest coverage, interest cash conversion.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.DataFrame: Interest paid to interest expense ratio values.

        Notes:
        - The method retrieves historical data and calculates the ratio for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["INTC", "MU", "HD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_interest_paid_to_expense_ratio()
        ```

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | HD   | 0.9421 | 0.8961 | 0.931  | 0.9474 | 0.9971 |
        | INTC | 0.9129 | 0.9254 | 0.6982 | 1.1978 | 1.0137 |
        | MU   | 0.9344 | 0.8148 | 0.8325 | 0.895  | 0.8763 |
        """
        if trailing:
            interest_paid_to_expense_ratio = (
                solvency_model.get_interest_paid_to_expense_ratio(
                    self._sum_trailing(
                        self._cash_flow_statement.loc[:, "Interest Paid", :], trailing
                    ),
                    self._sum_trailing(
                        self._income_statement.loc[:, "Interest Expense", :], trailing
                    ),
                )
            )
        else:
            interest_paid_to_expense_ratio = (
                solvency_model.get_interest_paid_to_expense_ratio(
                    self._cash_flow_statement.loc[:, "Interest Paid", :],
                    self._income_statement.loc[:, "Interest Expense", :],
                )
            )

        return finalize_dataset(
            dataset=interest_paid_to_expense_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_ebitda_interest_coverage_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the EBITDA interest coverage ratio, the number of times the earnings before
        interest, taxes, depreciation and amortization cover the interest expense.

        EBITDA to interest is one of the supplementary ratios of S&P Global Ratings and a
        coverage measure in many of Moody's rating methodologies. It is higher than the EBIT
        interest coverage (`get_interest_coverage_ratio`) by the depreciation and amortization,
        which makes it the usual measure in loan covenants, but it ignores that the assets
        have to be replaced, so for capital-intensive companies the (EBITDA - CapEx) coverage
        (`get_ebitda_minus_capex_interest_coverage_ratio`) is the stricter test.

        The formula is as follows:

        - EBITDA Interest Coverage = (Operating Income + Depreciation and Amortization) / Interest Expense

        The interest expense is the gross expense, as S&P Global Ratings does not net the
        interest income. A zero interest expense gives NaN, which is often a gap in the
        reported data rather than an absence of debt.

        Also known as: EBITDA to interest, EBITDA coverage, EBITDA/interest, EBITDA interest cover.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over. E.g. 4 with quarterly data gives the trailing twelve
                months (TTM) and 3 with annual data a three-year cumulative figure. Defaults to
                None, which uses single-period flows.

        Returns:
            pd.DataFrame: EBITDA interest coverage values.

        Notes:
        - S&P Global Ratings, "Corporate Methodology: Ratios And Adjustments," April 1, 2019
        (key ratios: EBITDA to interest).
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["T", "UNP", "HSY"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_ebitda_interest_coverage_ratio()
        ```

        Which returns:

        |     |    2021 |    2022 |    2023 |    2024 |   2025 |
        |:----|--------:|--------:|--------:|--------:|-------:|
        | HSY | 18.1655 | 18.8425 | 18.455  | 19.2379 | 9.2591 |
        | T   |  6.5336 |  2.1883 |  6.507  |  6.648  | 6.744  |
        | UNP |  9.9793 |  9.5696 |  8.5075 |  9.5437 | 9.3995 |
        """
        ebitda_interest_coverage = solvency_model.get_ebitda_interest_coverage_ratio(
            self._sum_trailing(
                self._income_statement.loc[:, "Operating Income", :], trailing
            ),
            self._sum_trailing(
                self._cash_flow_statement.loc[:, "Depreciation and Amortization", :],
                trailing,
            ),
            self._sum_trailing(
                self._income_statement.loc[:, "Interest Expense", :], trailing
            ),
        )

        return finalize_dataset(
            dataset=ebitda_interest_coverage,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_ebitda_minus_capex_interest_coverage_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the EBITDA minus capital expenditure interest coverage ratio, the number of
        times the EBITDA that is left after capital expenditure covers the interest expense.

        Moody's uses this coverage in its rating methodologies for capital-intensive sectors
        (such as automobile manufacturers), because EBITDA alone counts the cash needed to
        replace the assets as available for the lenders. A company whose (EBITDA - CapEx)
        coverage is far below its EBITDA coverage depends on its capital spending staying
        low, or on new borrowing, to keep paying its interest.

        The formula is as follows:

        - (EBITDA - CapEx) Interest Coverage = (Operating Income + Depreciation and Amortization - |Capital Expenditure|) / Interest Expense

        Capital expenditure is reported as a negative cash outflow, so its magnitude is
        deducted. A zero interest expense gives NaN.

        Also known as: (EBITDA - CapEx) / interest expense, EBITDA less capex coverage, EBITDA minus capex to interest.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over. E.g. 4 with quarterly data gives the trailing twelve
                months (TTM) and 3 with annual data a three-year cumulative figure. Defaults to
                None, which uses single-period flows.

        Returns:
            pd.DataFrame: (EBITDA - CapEx) interest coverage values.

        Notes:
        - Moody's Investors Service, "Automobile Manufacturer Industry," rating methodology,
        June 2017 ((EBITDA - CapEx) / Interest Expense).
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["T", "UNP", "HSY"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_ebitda_minus_capex_interest_coverage_ratio()
        ```

        Which returns:

        |     |    2021 |    2022 |    2023 |    2024 |   2025 |
        |:----|--------:|--------:|--------:|--------:|-------:|
        | HSY | 14.3466 | 15.1345 | 13.6806 | 15.7618 | 6.7126 |
        | T   |  4.0654 | -1.0086 |  3.8392 |  3.6439 | 3.6808 |
        | UNP |  7.4417 |  6.7215 |  5.8164 |  6.8235 | 6.5034 |
        """
        ebitda_minus_capex_interest_coverage = (
            solvency_model.get_ebitda_minus_capex_interest_coverage_ratio(
                self._sum_trailing(
                    self._income_statement.loc[:, "Operating Income", :], trailing
                ),
                self._sum_trailing(
                    self._cash_flow_statement.loc[
                        :, "Depreciation and Amortization", :
                    ],
                    trailing,
                ),
                self._sum_trailing(
                    self._cash_flow_statement.loc[:, "Capital Expenditure", :],
                    trailing,
                ),
                self._sum_trailing(
                    self._income_statement.loc[:, "Interest Expense", :], trailing
                ),
            )
        )

        return finalize_dataset(
            dataset=ebitda_minus_capex_interest_coverage,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_cash_interest_coverage_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the cash interest coverage ratio, the number of times the operating cash
        flow before interest and taxes covers the interest paid in cash.

        Where the interest coverage compares accrual figures (earnings with the interest
        expense), this ratio compares cash with cash, so it is not flattered by earnings
        that are not received in cash or by interest that is accrued (payment-in-kind debt)
        rather than paid.

        The formula is as follows:

        - Cash Interest Coverage = (Cash Flow from Operations + Interest Paid + Income Taxes Paid) / Interest Paid

        Under US GAAP the interest and taxes paid are part of the cash flow from operations,
        so they are added back. IFRS lets companies report the interest paid under financing
        activities instead, in which case adding it back overstates the coverage by one. A
        zero or unreported interest paid gives NaN.

        Also known as: cash flow interest coverage, CFO interest coverage, cash coverage ratio.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over. E.g. 4 with quarterly data gives the trailing twelve
                months (TTM) and 3 with annual data a three-year cumulative figure. Defaults to
                None, which uses single-period flows.

        Returns:
            pd.DataFrame: Cash interest coverage values.

        Notes:
        - CFA Institute, "Financial Reporting and Analysis," CFA Program Curriculum
        (cash flow statement analysis: interest coverage).
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["T", "UNP", "HSY"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_cash_interest_coverage_ratio()
        ```

        Which returns:

        |     |    2021 |    2022 |    2023 |    2024 |     2025 |
        |:----|--------:|--------:|--------:|--------:|---------:|
        | HSY | 19.4618 | 20.3474 | 17.3451 | 16.2044 |  13.3412 |
        | T   |  6.6391 |  5.1965 |  6.4156 |  6.7806 | nan      |
        | UNP | 10.8344 | 10.8806 |  8.78   |  9.481  |   8.5004 |
        """
        cash_interest_coverage = solvency_model.get_cash_interest_coverage_ratio(
            self._sum_trailing(
                self._cash_flow_statement.loc[:, "Cash Flow from Operations", :],
                trailing,
            ),
            self._sum_trailing(
                self._cash_flow_statement.loc[:, "Interest Paid", :], trailing
            ),
            self._sum_trailing(
                self._cash_flow_statement.loc[:, "Income Taxes Paid", :], trailing
            ),
        )

        return finalize_dataset(
            dataset=cash_interest_coverage,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_funds_from_operations_to_debt_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the funds from operations (FFO) to debt ratio, one of the two core ratios
        with which S&P Global Ratings assesses a company's financial risk.

        FFO is the cash a company generates from its operations before working capital and
        investment: EBITDA minus the interest and taxes paid in cash. As a share of the
        debt it shows how quickly the company could repay its debt from its recurring cash
        flow. S&P maps the ratio to its financial risk profiles: above 60% is minimal risk,
        45% to 60% modest, 30% to 45% intermediate, 20% to 30% significant, 12% to 20%
        aggressive and below 12% highly leveraged (standard volatility table; companies with
        more volatile cash flows are held to stricter thresholds).

        The formula is as follows:

        - FFO = Operating Income + Depreciation and Amortization + Stock-Based Compensation - Interest Paid - Income Taxes Paid
        - FFO to Debt = FFO / Total Debt

        S&P's EBITDA excludes share-based compensation payable in shares, so it is added back,
        and impairments, which the operating income here includes.
        The debt is the reported total debt at the end of the period; S&P's adjusted debt
        also adds leases, pension deficits and other debt-like items and deducts surplus
        cash, which the statements do not allow to repeat. FFO to debt is an annual measure,
        so with quarterly data the flows are those of the trailing twelve months by default.
        A total debt of zero or less gives NaN. So does an interest paid of zero: a company
        with debt pays interest, so zero means the interest paid is not reported, and
        counting it as zero would overstate the FFO by the whole interest bill.

        Also known as: FFO/debt, FFO to total debt, funds from operations to debt.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum the flows over; the total debt
                stays that at the end of the last period. Defaults to None, which uses the
                trailing twelve months (4 quarters) with quarterly data and single years with
                annual data.

        Returns:
            pd.DataFrame: FFO to debt values.

        Notes:
        - S&P Global Ratings, "Corporate Methodology: Ratios And Adjustments," April 1, 2019
        (core ratios: FFO to debt; FFO is EBITDA minus cash interest paid minus cash tax paid).
        - S&P Global Ratings, "Corporate Methodology," 2013 (financial risk profile table).
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["T", "UNP", "HSY"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_funds_from_operations_to_debt_ratio()
        ```

        Which returns:

        |     |   2021 |   2022 |   2023 |   2024 |     2025 |
        |:----|-------:|-------:|-------:|-------:|---------:|
        | HSY | 0.3762 | 0.4597 | 0.5067 | 0.5537 |   0.3053 |
        | T   | 0.1839 | 0.0361 | 0.2232 | 0.2502 | nan      |
        | UNP | 0.2823 | 0.2588 | 0.253  | 0.293  |   0.3285 |
        """
        # FFO to debt compares a year of cash flow with the debt, so quarterly data defaults to
        # the trailing twelve months.
        trailing = self._get_valuation_trailing(trailing)

        funds_from_operations_to_debt = (
            solvency_model.get_funds_from_operations_to_debt_ratio(
                self._sum_trailing(
                    self._income_statement.loc[:, "Operating Income", :], trailing
                ),
                self._sum_trailing(
                    self._cash_flow_statement.loc[
                        :, "Depreciation and Amortization", :
                    ],
                    trailing,
                ),
                self._sum_trailing(
                    self._cash_flow_statement.loc[:, "Stock Based Compensation", :],
                    trailing,
                ),
                self._sum_trailing(
                    self._cash_flow_statement.loc[:, "Interest Paid", :], trailing
                ),
                self._sum_trailing(
                    self._cash_flow_statement.loc[:, "Income Taxes Paid", :],
                    trailing,
                ),
                self._balance_sheet_statement.loc[:, "Total Debt", :],
            )
        )

        return finalize_dataset(
            dataset=funds_from_operations_to_debt,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_discretionary_cash_flow_to_debt_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the discretionary cash flow (DCF) to debt ratio, a supplementary ratio of
        S&P Global Ratings: the cash flow left after capital expenditure, dividends and share
        buybacks as a share of the debt.

        Where FFO to debt (`get_funds_from_operations_to_debt_ratio`) measures the cash flow
        before the company's spending choices, DCF to debt measures what is left after them,
        and so what could actually go to repaying debt. A negative value means the company
        spends more on investment and shareholders than its operations bring in, and funds
        the difference with cash or new debt.

        The formula is as follows:

        - DCF = Cash Flow from Operations - |Capital Expenditure| - |Dividends Paid| - |Common Stock Purchased|
        - DCF to Debt = DCF / Total Debt

        Capital expenditure, dividends and buybacks are reported as negative cash outflows, so
        their magnitudes are deducted. The dividends are those on common and preferred stock;
        S&P also deducts dividends to minority shareholders, which the statements do not
        separate. The debt is the reported total debt at the end of the period. DCF to debt
        is an annual measure, so with quarterly data the flows are those of the trailing
        twelve months by default. A total debt of zero or less gives NaN.

        Also known as: DCF/debt, discretionary cash flow to debt.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum the flows over; the total debt
                stays that at the end of the last period. Defaults to None, which uses the
                trailing twelve months (4 quarters) with quarterly data and single years with
                annual data.

        Returns:
            pd.DataFrame: DCF to debt values.

        Notes:
        - S&P Global Ratings, "Corporate Methodology: Ratios And Adjustments," April 1, 2019
        (supplementary ratios: DCF to debt; DCF is free operating cash flow minus dividends
        minus share buybacks).
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["T", "UNP", "HSY"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_discretionary_cash_flow_to_debt_ratio()
        ```

        Which returns:

        |     |    2021 |    2022 |   2023 |   2024 |    2025 |
        |:----|--------:|--------:|-------:|-------:|--------:|
        | HSY |  0.0824 |  0.1259 | 0.0777 | 0.0636 |  0.1229 |
        | T   |  0.0519 |  0.0107 | 0.0783 | 0.0716 |  0.0269 |
        | UNP | -0.1269 | -0.1058 | 0.0262 | 0.0362 | -0.0131 |
        """
        # DCF to debt compares a year of cash flow with the debt, so quarterly data defaults to
        # the trailing twelve months.
        trailing = self._get_valuation_trailing(trailing)

        discretionary_cash_flow_to_debt = (
            solvency_model.get_discretionary_cash_flow_to_debt_ratio(
                self._sum_trailing(
                    self._cash_flow_statement.loc[:, "Cash Flow from Operations", :],
                    trailing,
                ),
                self._sum_trailing(
                    self._cash_flow_statement.loc[:, "Capital Expenditure", :],
                    trailing,
                ),
                self._sum_trailing(
                    self._cash_flow_statement.loc[:, "Dividends Paid", :], trailing
                ),
                self._sum_trailing(
                    self._cash_flow_statement.loc[:, "Common Stock Purchased", :],
                    trailing,
                ),
                self._balance_sheet_statement.loc[:, "Total Debt", :],
            )
        )

        return finalize_dataset(
            dataset=discretionary_cash_flow_to_debt,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_degree_of_operating_leverage(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the degree of operating leverage (DOL), the percentage change in operating
        income for each percentage change in revenue.

        Operating leverage comes from fixed operating costs: when revenue rises, the fixed
        costs stay the same, so operating income rises faster, and the reverse when revenue
        falls. A DOL of 3 means that a 1% change in revenue went with a 3% change in operating
        income. Companies with high fixed costs (airlines, manufacturers, software with high
        development costs) have a high DOL, which makes their earnings more sensitive to the
        business cycle (business risk).

        The formula is as follows:

        - DOL = %Δ Operating Income / %Δ Revenue

        The changes are period over period with annual data and year over year with quarterly
        data (a quarter against the same quarter a year earlier), so seasonality does not
        enter. Each change is (Value - Previous Value) / |Previous Value|, so a change from a
        negative base keeps its sign. The ratio is unstable when the denominator barely
        changes: a small change in it gives a very large degree, and no change gives NaN.

        Also known as: DOL, operating leverage, operating gearing.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum the flows over before their
                changes are compared, e.g. 4 with quarterly data compares trailing twelve months
                with those a year earlier. Defaults to None, which uses single-period flows.

        Returns:
            pd.DataFrame: Degree of operating leverage values.

        Notes:
        - CFA Institute, "Corporate Issuers," CFA Program Curriculum (measures of leverage:
        degree of operating, financial and total leverage).
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "F", "DAL"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_degree_of_operating_leverage()
        ```

        Which returns:

        |      |    2021 |   2022 |    2023 |    2024 |     2025 |
        |:-----|--------:|-------:|--------:|--------:|---------:|
        | AAPL |  1.935  | 1.2352 |  1.5355 |  3.8574 |   1.2421 |
        | DAL  |  1.5371 | 1.3605 |  3.4421 |  1.3863 |  -1.0336 |
        | F    | 29.0263 | 1.7165 | -1.4021 | -1.0535 | -40.905  |
        """
        degree_of_operating_leverage = solvency_model.get_degree_of_leverage(
            self._sum_trailing(
                self._income_statement.loc[:, "Operating Income", :], trailing
            ),
            self._sum_trailing(self._income_statement.loc[:, "Revenue", :], trailing),
            lag=4 if self._quarterly else 1,
        )

        return finalize_dataset(
            dataset=degree_of_operating_leverage,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_degree_of_financial_leverage(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the degree of financial leverage (DFL), the percentage change in net income
        for each percentage change in operating income.

        Financial leverage comes from fixed financing costs, mostly interest: when operating
        income rises, the interest stays the same, so net income rises faster, and the
        reverse when it falls. A company without debt has a DFL of about 1; the more of its
        operating income goes to interest, the higher the DFL and the financial risk.

        The formula is as follows:

        - DFL = %Δ Net Income / %Δ Operating Income

        The changes are period over period with annual data and year over year with quarterly
        data (a quarter against the same quarter a year earlier), so seasonality does not
        enter. Each change is (Value - Previous Value) / |Previous Value|, so a change from a
        negative base keeps its sign. The ratio is unstable when the denominator barely
        changes: a small change in it gives a very large degree, and no change gives NaN.

        Net income also moves with one-off items (non-operating gains, tax changes,
        discontinued operations), which enter the DFL as well.

        Also known as: DFL, financial leverage, financial gearing.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum the flows over before their
                changes are compared, e.g. 4 with quarterly data compares trailing twelve months
                with those a year earlier. Defaults to None, which uses single-period flows.

        Returns:
            pd.DataFrame: Degree of financial leverage values.

        Notes:
        - CFA Institute, "Corporate Issuers," CFA Program Curriculum (measures of leverage:
        degree of operating, financial and total leverage).
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "F", "DAL"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_degree_of_financial_leverage()
        ```

        Which returns:

        |      |   2021 |    2022 |     2023 |    2024 |     2025 |
        |:-----|-------:|--------:|---------:|--------:|---------:|
        | AAPL | 1.0087 |  0.5621 |   0.6543 | -0.4308 |   2.4427 |
        | DAL  | 0.8883 |  3.939  |   4.9147 | -2.9113 | -15.5172 |
        | F    | 7.1557 | -4.0617 | -19.857  | -6.6974 |   4.7545 |
        """
        degree_of_financial_leverage = solvency_model.get_degree_of_leverage(
            self._sum_trailing(
                self._income_statement.loc[:, "Net Income", :], trailing
            ),
            self._sum_trailing(
                self._income_statement.loc[:, "Operating Income", :], trailing
            ),
            lag=4 if self._quarterly else 1,
        )

        return finalize_dataset(
            dataset=degree_of_financial_leverage,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_degree_of_total_leverage(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the degree of total leverage (DTL), the percentage change in net income for
        each percentage change in revenue: the operating and financial leverage combined.

        The DTL is the product of the degree of operating leverage
        (`get_degree_of_operating_leverage`) and the degree of financial leverage
        (`get_degree_of_financial_leverage`) over the same periods, and measures how strongly
        a change in revenue carries through to the earnings of the shareholders.

        The formula is as follows:

        - DTL = %Δ Net Income / %Δ Revenue = DOL x DFL

        The changes are period over period with annual data and year over year with quarterly
        data (a quarter against the same quarter a year earlier), so seasonality does not
        enter. Each change is (Value - Previous Value) / |Previous Value|, so a change from a
        negative base keeps its sign. The ratio is unstable when the denominator barely
        changes: a small change in it gives a very large degree, and no change gives NaN.

        Also known as: DTL, degree of combined leverage, DCL, combined leverage, total leverage.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum the flows over before their
                changes are compared, e.g. 4 with quarterly data compares trailing twelve months
                with those a year earlier. Defaults to None, which uses single-period flows.

        Returns:
            pd.DataFrame: Degree of total leverage values.

        Notes:
        - CFA Institute, "Corporate Issuers," CFA Program Curriculum (measures of leverage:
        degree of operating, financial and total leverage).
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "F", "DAL"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_degree_of_total_leverage()
        ```

        Which returns:

        |      |     2021 |    2022 |    2023 |    2024 |      2025 |
        |:-----|---------:|--------:|--------:|--------:|----------:|
        | AAPL |   1.9518 |  0.6943 |  1.0047 | -1.6617 |    3.034  |
        | DAL  |   1.3653 |  5.359  | 16.9169 | -4.0358 |   16.0389 |
        | F    | 207.703  | -6.9718 | 27.8421 |  7.0554 | -194.484  |
        """
        degree_of_total_leverage = solvency_model.get_degree_of_leverage(
            self._sum_trailing(
                self._income_statement.loc[:, "Net Income", :], trailing
            ),
            self._sum_trailing(self._income_statement.loc[:, "Revenue", :], trailing),
            lag=4 if self._quarterly else 1,
        )

        return finalize_dataset(
            dataset=degree_of_total_leverage,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_errors
    def collect_valuation_ratios(
        self,
        include_dividends: bool = False,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculates and collects all Valuation Ratios based on the provided data.

        Args:
            include_dividends (bool, optional): Whether to deduct preferred dividends from net income in the
                earnings-based ratios. Defaults to False.
            diluted (bool, optional): Whether to use diluted shares for the calculation. Defaults to True.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.DataFrame: Valuation ratios calculated based on the specified parameters.

        Notes:
        - The method calculates various valuation ratios for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
          using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "TSLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        valuation_ratios = toolkit.ratios.collect_valuation_ratios()

        valuation_ratios.loc['AAPL']
        ```

        Which returns:

        |                             |         2021 |         2022 |         2023 |          2024 |         2025 |
        |:----------------------------|-------------:|-------------:|-------------:|--------------:|-------------:|
        | Earnings per Share          |  5.614       |  6.1132      |  6.1341      |   6.0836      |  7.465       |
        | Revenue per Share           | 21.691       | 24.1536      | 24.2393      |  25.3785      | 27.7354      |
        | Price-to-Earnings           | 31.6297      | 21.254       | 31.3871      |  41.1634      | 36.418       |
        | Price-to-Earnings-Growth    |  0.443       |  2.3908      | 92.315       | -50.1993      |  1.6036      |
        | Book Value per Share        |  3.7409      |  3.1038      |  3.9302      |   3.6961      |  4.914       |
        | Price-to-Book               | 47.4672      | 41.8617      | 48.9877      |  67.7523      | 55.3236      |
        | Interest Debt per Share     |  8.2519      |  8.2943      |  8.0862      |   7.727       |  7.4895      |
        | CAPEX per Share             | -0.6573      | -0.6559      | -0.6931      |  -0.6131      | -0.8474      |
        | Earnings Yield              |  0.0316      |  0.047       |  0.0319      |   0.0243      |  0.0275      |
        | Dividend Payout Ratio       |  0.1528      |  0.1487      |  0.1549      |   0.1625      |  0.1377      |
        | Dividend Yield              |  0.0049      |  0.007       |  0.0049      |   0.004       |  0.0038      |
        | Weighted Dividend Yield     |  0.0048      |  0.007       |  0.0049      |   0.0039      |  0.0038      |
        | Price-to-Cash-Flow          | 28.7847      | 17.3655      | 27.5403      |  32.6289      | 36.5905      |
        | Price-to-Free-Cash-Flow     | 32.2174      | 19.0341      | 30.5711      |  35.4618      | 41.301       |
        | Price-to-Sales              |  8.1863      |  5.3793      |  7.9429      |   9.8674      |  9.8019      |
        | Market Cap                  |  2.9947e+12  |  2.12121e+12 |  3.04439e+12 |   3.8585e+12  |  4.07918e+12 |
        | Enterprise Value            |  3.06859e+12 |  2.20539e+12 |  3.10676e+12 |   3.91238e+12 |  4.13686e+12 |
        | EV-to-Sales                 |  8.3883      |  5.5928      |  8.1056      |  10.0052      |  9.9405      |
        | EV-to-EBIT                  | 28.1653      | 18.4649      | 27.1806      |  31.7522      | 31.0925      |
        | EV-to-EBITDA                | 25.522       | 16.8942      | 24.6921      |  29.0536      | 28.5797      |
        | EV-to-Operating-Cash-Flow   | 29.4949      | 18.0546      | 28.1046      |  33.0846      | 37.1078      |
        | Tangible Asset Value        |  6.309e+10   |  5.0672e+10  |  6.2146e+10  |   5.695e+10   |  7.3733e+10  |
        | Net Current Asset Value     | -1.53076e+11 | -1.66678e+11 | -1.46871e+11 |  -1.55043e+11 | -1.37551e+11 |
        | EV-to-Free-Cash-Flow        | 33.0122      | 19.7894      | 31.1974      |  35.9571      | 41.885       |
        | Buyback Yield               |  0.0283      |  0.0421      |  0.0255      |   0.0246      |  0.0222      |
        | Shareholder Yield           |  0.0332      |  0.0491      |  0.0304      |   0.0286      |  0.026       |
        | SBC-Adjusted Free Cash Flow |  8.5047e+10  |  1.02405e+11 |  8.8751e+10  |   9.7119e+10  |  8.5904e+10  |
        | Free Cash Flow Payout Ratio |  0.1556      |  0.1332      |  0.1509      |   0.14        |  0.1561      |
        | EV-to-Invested-Capital      | 22.4029      | 16.3546      | 24.9497      |  35.2982      | 31.4798      |
        | Price-to-Tangible-Book      | 47.4672      | 41.8617      | 48.9877      |  67.7523      | 55.3236      |
        """
        self._load_statements()
        valuation_ratios_dict: dict = {}

        valuation_ratios_dict["Earnings per Share"] = self.get_earnings_per_share(
            include_dividends=include_dividends, diluted=diluted, trailing=trailing
        )
        valuation_ratios_dict["Revenue per Share"] = self.get_revenue_per_share(
            diluted=diluted, trailing=trailing
        )
        valuation_ratios_dict["Price-to-Earnings"] = self.get_price_to_earnings_ratio(
            include_dividends=include_dividends, diluted=diluted, trailing=trailing
        )
        valuation_ratios_dict["Price-to-Earnings-Growth"] = (
            self.get_price_to_earnings_growth_ratio(
                include_dividends=include_dividends, diluted=diluted
            )
        )
        valuation_ratios_dict["Book Value per Share"] = self.get_book_value_per_share(
            diluted=diluted, trailing=trailing
        )
        valuation_ratios_dict["Price-to-Book"] = self.get_price_to_book_ratio(
            diluted=diluted, trailing=trailing
        )
        valuation_ratios_dict["Interest Debt per Share"] = (
            self.get_interest_debt_per_share(diluted=diluted, trailing=trailing)
        )
        valuation_ratios_dict["CAPEX per Share"] = self.get_capex_per_share(
            diluted=diluted, trailing=trailing
        )
        valuation_ratios_dict["Earnings Yield"] = self.get_earnings_yield(
            include_dividends=include_dividends, diluted=diluted, trailing=trailing
        )
        valuation_ratios_dict["Dividend Payout Ratio"] = self.get_dividend_payout_ratio(
            trailing=trailing
        )

        dividend_yield = self.get_dividend_yield(trailing=trailing)
        dividend_yield_columns = [
            column
            for column in dividend_yield.columns
            if column in valuation_ratios_dict["Dividend Payout Ratio"].columns
        ]
        valuation_ratios_dict["Dividend Yield"] = dividend_yield.loc[
            :, dividend_yield_columns
        ]

        valuation_ratios_dict["Weighted Dividend Yield"] = (
            self.get_weighted_dividend_yield(diluted=diluted, trailing=trailing)
        )
        valuation_ratios_dict["Price-to-Cash-Flow"] = self.get_price_to_cash_flow_ratio(
            diluted=diluted, trailing=trailing
        )
        valuation_ratios_dict["Price-to-Free-Cash-Flow"] = (
            self.get_price_to_free_cash_flow_ratio(diluted=diluted, trailing=trailing)
        )
        valuation_ratios_dict["Price-to-Sales"] = self.get_price_to_sales_ratio(
            diluted=diluted, trailing=trailing
        )
        valuation_ratios_dict["Market Cap"] = self.get_market_cap(diluted=diluted)
        valuation_ratios_dict["Enterprise Value"] = self.get_enterprise_value(
            diluted=diluted
        )
        valuation_ratios_dict["EV-to-Sales"] = self.get_ev_to_sales_ratio(
            diluted=diluted, trailing=trailing
        )
        valuation_ratios_dict["EV-to-EBIT"] = self.get_ev_to_ebit(
            diluted=diluted, trailing=trailing
        )
        valuation_ratios_dict["EV-to-EBITDA"] = self.get_ev_to_ebitda_ratio(
            diluted=diluted, trailing=trailing
        )
        valuation_ratios_dict["EV-to-Operating-Cash-Flow"] = (
            self.get_ev_to_operating_cashflow_ratio(diluted=diluted, trailing=trailing)
        )
        valuation_ratios_dict["Tangible Asset Value"] = self.get_tangible_asset_value(
            trailing=trailing
        )
        valuation_ratios_dict["Net Current Asset Value"] = (
            self.get_net_current_asset_value(trailing=trailing)
        )
        valuation_ratios_dict["EV-to-Free-Cash-Flow"] = (
            self.get_ev_to_free_cash_flow_ratio(diluted=diluted, trailing=trailing)
        )
        valuation_ratios_dict["Buyback Yield"] = self.get_buyback_yield(
            diluted=diluted, trailing=trailing
        )
        valuation_ratios_dict["Shareholder Yield"] = self.get_shareholder_yield(
            diluted=diluted, trailing=trailing
        )
        valuation_ratios_dict["SBC-Adjusted Free Cash Flow"] = (
            self.get_sbc_adjusted_free_cash_flow(trailing=trailing)
        )
        valuation_ratios_dict["Free Cash Flow Payout Ratio"] = (
            self.get_free_cash_flow_payout_ratio(trailing=trailing)
        )
        valuation_ratios_dict["EV-to-Invested-Capital"] = (
            self.get_ev_to_invested_capital_ratio(diluted=diluted, trailing=trailing)
        )
        valuation_ratios_dict["Price-to-Tangible-Book"] = (
            self.get_price_to_tangible_book_ratio(diluted=diluted, trailing=trailing)
        )

        # The issuance measures use the basic shares, as the paper's shares outstanding.
        valuation_ratios_dict["Net Share Issuance"] = self.get_net_share_issuance(
            trailing=trailing
        )
        valuation_ratios_dict["Composite Equity Issuance"] = (
            self.get_composite_equity_issuance(trailing=trailing)
        )
        valuation_ratios_dict["Net External Financing"] = (
            self.get_net_external_financing(trailing=trailing)
        )
        self._valuation_ratios = (
            pd.concat(valuation_ratios_dict)
            .swaplevel(0, 1)
            .sort_index(level=0, sort_remaining=False)
            .dropna(axis="columns", how="all")
        )

        self._valuation_ratios = self._valuation_ratios.loc[self._tickers]

        self._valuation_ratios = self._valuation_ratios.pipe(
            apply_rounding, rounding if rounding is not None else self._rounding
        )

        # Sorted again so the index follows the financial statements' order.
        available_columns = [
            column
            for column in self._income_statement.columns
            if column in self._valuation_ratios
        ]
        self._valuation_ratios = self._valuation_ratios.reindex(
            available_columns, axis=1
        )

        if growth:
            self._valuation_ratios_growth = calculate_growth(
                dataset=self._valuation_ratios,
                lag=lag,
                compound=compound,
                rounding=rounding if rounding is not None else self._rounding,
                axis="columns",
            )

        valuation_ratios = self._valuation_ratios
        valuation_ratios_growth = self._valuation_ratios_growth

        if standardize:
            standardize_rounding = rounding if rounding is not None else self._rounding
            if growth:
                valuation_ratios_growth = calculate_standardization(
                    dataset=valuation_ratios_growth,
                    rounding=standardize_rounding,
                    axis="columns",
                )
            else:
                valuation_ratios = calculate_standardization(
                    dataset=valuation_ratios,
                    rounding=standardize_rounding,
                    axis="columns",
                )

        if len(self._tickers) == 1:
            return (
                valuation_ratios_growth.loc[self._tickers[0]]
                if growth
                else valuation_ratios.loc[self._tickers[0]]
            ).loc[:, self._start_date : self._end_date]

        return (valuation_ratios_growth if growth else valuation_ratios).loc[
            :, self._start_date : self._end_date
        ]

    def _get_earnings_per_share(
        self, include_dividends: bool, diluted: bool, trailing: int | None
    ) -> pd.DataFrame:
        """
        Returns the earnings per share of every period, including those before the start date
        that the daily ratios carry over to the first days.

        Args:
            include_dividends (bool): Whether to deduct preferred dividends from net income.
            diluted (bool): Whether to use the diluted weighted average shares.
            trailing (int | None): The number of periods to sum the earnings over, or None.

        Returns:
            pd.DataFrame: The earnings per share with the tickers as index and the periods as
            columns.
        """
        average_shares = (
            self._income_statement.loc[:, "Weighted Average Shares Diluted", :]
            if diluted
            else self._income_statement.loc[:, "Weighted Average Shares", :]
        )

        if trailing:
            # Preferred Dividends Paid is reported as a negative cash outflow, so the magnitude is taken before it is subtracted from Net Income.  # noqa: E501
            dividends = (
                self._sum_trailing(
                    self._cash_flow_statement.loc[:, "Preferred Dividends Paid", :],
                    trailing,
                ).abs()
                if include_dividends
                else 0
            )

            earnings_per_share = valuation_model.get_earnings_per_share(
                self._sum_trailing(
                    self._income_statement.loc[:, "Net Income", :], trailing
                ),
                dividends,
                average_shares,
            )
        else:
            # Preferred Dividends Paid is reported as a negative cash outflow, so the magnitude is taken before it is subtracted from Net Income.  # noqa: E501
            dividends = (
                self._cash_flow_statement.loc[:, "Preferred Dividends Paid", :].abs()
                if include_dividends
                else 0
            )

            earnings_per_share = valuation_model.get_earnings_per_share(
                self._income_statement.loc[:, "Net Income", :],
                dividends,
                average_shares,
            )

        return earnings_per_share

    @handle_portfolio
    @handle_errors
    def get_earnings_per_share(
        self,
        include_dividends: bool = False,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ):
        """
        Calculate the earnings per share (EPS), a valuation ratio that measures the amount
        of net income earned per share of outstanding common stock.

        The earnings per share (EPS) is a widely used financial metric that helps investors
        understand the profitability of a company on a per-share basis. It provides insight
        into the portion of a company's earnings that is allocated to each outstanding share
        of its common stock. EPS is an important measure for investors and analysts when
        assessing a company's financial performance and comparing it to other companies.

        The formula is as follows:

        - Earnings per Share (EPS) = (Net Income — |Preferred Dividends Paid|) / Weighted Average (Diluted) Shares

        The preferred dividends are only deducted when include_dividends is True; with the
        default, EPS is net income divided by the weighted average (diluted) shares.

        Also known as: EPS, net income per share.

        Args:
            include_dividends (bool, optional): Whether to deduct the preferred dividends paid from net income.
                Defaults to False.
            diluted (bool, optional): Whether to use diluted earnings per share. Defaults to True.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.DataFrame: Earnings per share (EPS) values.

        Notes:
        - The method retrieves historical data and calculates the EPS for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the EPS values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["V", "MA", "AXP"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_earnings_per_share()
        ```

        Which returns:

        |     |    2021 |    2022 |    2023 |    2024 |    2025 |
        |:----|--------:|--------:|--------:|--------:|--------:|
        | AXP | 10.0215 |  9.8404 | 11.212  | 14.0182 | 15.375  |
        | MA  |  8.7571 | 10.2266 | 11.834  | 13.8878 | 16.6682 |
        | V   |  5.6266 |  7.0023 |  8.2844 |  9.7304 | 10.0982 |
        """
        earnings_per_share = self._get_earnings_per_share(
            include_dividends, diluted, trailing
        )

        return finalize_dataset(
            dataset=earnings_per_share,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_revenue_per_share(
        self,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ):
        """
        Calculate the revenue per share, a valuation ratio that measures the amount
        of revenue generated per outstanding share of a company's stock.

        The revenue per share is an important metric that provides insight into a
        company's ability to generate revenue on a per-share basis. It can help investors
        understand the company's revenue-generation efficiency and its overall financial health.

        The formula is as follows:

        - Revenue per Share = Revenue / Weighted Average (Diluted) Shares

        Also known as: sales per share.

        Args:
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.DataFrame: Revenue per share values.

        Notes:
        - The method retrieves historical data and calculates the revenue per share for each asset in
        the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the revenue per share values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["MCK", "COR", "CAH"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_revenue_per_share()
        ```

        Which returns:

        |     |     2021 |     2022 |     2023 |     2024 |     2025 |
        |:----|---------:|---------:|---------:|---------:|---------:|
        | CAH |  552.609 |  649.914 |  782.363 |  918.328 |  919.744 |
        | COR | 1026.5   | 1129.62  | 1281.45  | 1467.71  | 1646.05  |
        | MCK | 1712.95  | 1945.93  | 2303.89  | 2802.9   | 3250.85  |
        """
        average_shares = (
            self._income_statement.loc[:, "Weighted Average Shares Diluted", :]
            if diluted
            else self._income_statement.loc[:, "Weighted Average Shares", :]
        )

        if trailing:
            revenue_per_share = valuation_model.get_revenue_per_share(
                self._sum_trailing(
                    self._income_statement.loc[:, "Revenue", :], trailing
                ),
                average_shares,
            )
        else:
            revenue_per_share = valuation_model.get_revenue_per_share(
                self._income_statement.loc[:, "Revenue", :], average_shares
            )

        return finalize_dataset(
            dataset=revenue_per_share,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_price_to_earnings_ratio(
        self,
        show_daily: bool = False,
        include_dividends: bool = False,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ):
        """
        Calculate the price earnings ratio (P/E), a valuation ratio that compares a
        company's stock price to its earnings per share.

        The price earnings ratio is a widely used valuation metric that helps investors
        assess the relative value of a company's stock. A higher P/E ratio may indicate
        that the market has high expectations for the company's future growth, while a
        lower P/E ratio may suggest that the company is undervalued.

        The formula is as follows:

        - Price to Earnings Ratio (P/E) = Share Price / Earnings per Share (EPS)

        The P/E is compared with the company's own history, its peers and the market,
        and is read together with growth: a high P/E is the market paying for expected
        growth. For cyclical companies it is misleading at the extremes, lowest at peak
        earnings and highest in a downturn, so analysts often use normalized or forward
        earnings (`get_forward_price_earnings_ratio`). A negative or tiny EPS gives a
        negative or extreme P/E that should be left out of averages.

        Also known as: PE ratio, P/E ratio, price-earnings ratio, earnings multiple.

        Args:
            show_daily (bool, optional): Whether to show daily values: the close of each trading day
                against the statement figures of the latest period that has ended on or before
                that day (also when it ended before the start date), so no figures are used before
                their period closes. Defaults to False.
            include_dividends (bool, optional): Whether to deduct preferred dividends from net income.
                Defaults to False.
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses the
                trailing twelve months (4 quarters) with quarterly data and single years with
                annual data.

        Returns:
            pd.DataFrame: Price earnings ratio (P/E) values.

        Notes:
        - The method retrieves historical data and calculates the P/E ratio for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the P/E ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["COST", "WMT", "TGT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_price_to_earnings_ratio()
        ```

        Which returns:

        |      |    2021 |    2022 |    2023 |    2024 |     2025 |
        |:-----|--------:|--------:|--------:|--------:|---------:|
        | COST | 50.3805 | 34.7419 | 46.6265 | 55.3169 |  47.3603 |
        | TGT  | 16.4167 | 24.9133 | 15.9285 | 15.2594 | nan      |
        | WMT  | 29.683  | 33.1872 | 27.4726 | 37.5653 | nan      |
        """
        # Prices are compared with a year of earnings, so quarterly data defaults to the trailing
        # twelve months. The earnings are summed over the periods before they are spread over days.
        eps = self._get_earnings_per_share(
            include_dividends, diluted, self._get_valuation_trailing(trailing)
        )

        years = eps.columns
        begin, end = str(years[0]), str(years[-1])

        share_prices = self._get_share_prices(begin, end, show_daily)

        if show_daily:
            eps = map_period_data_to_daily_data(
                period_data=eps,
                daily_dates=share_prices.index,
                quarterly=self._quarterly,
                period_end_dates=self._period_end_dates,
            )

        price_to_earnings_ratio = valuation_model.get_price_to_earnings_ratio(
            share_prices, eps
        )

        return finalize_dataset(
            dataset=price_to_earnings_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
            axis="rows" if show_daily else "columns",
            row_slice=show_daily,
        )

    @handle_portfolio
    @handle_errors
    def get_price_to_earnings_growth_ratio(
        self,
        use_ebitda_growth_rate: bool = False,
        include_dividends: bool = False,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
    ):
        """
        Calculate the price earnings to growth (PEG) ratio, a valuation metric that
        measures the ratio of the price-to-earnings ratio to earnings growth rate.

        The price-to-earnings growth (PEG) ratio provides a more comprehensive valuation
        measure compared to the P/E ratio alone. It takes into account a company's earnings
        growth rate, allowing investors to assess whether a stock is overvalued or undervalued
        relative to its growth prospects.

        The formula is as follows:

        - Price Earnings to Growth Ratio (PEG) = Price Earnings Ratio (P/E) / (EPS Growth Rate * 100)

        The growth rate is the growth of the (diluted) earnings per share on the previous
        year, or of EBITDA when use_ebitda_growth_rate is True, multiplied by 100 so that it
        enters in percent. With quarterly data both the P/E and the growth use trailing twelve
        month figures, and the growth compares them with those of four quarters earlier.

        A PEG around 1 is the traditional rule of thumb for a growth stock priced in line with
        its growth. This historical PEG uses a single period of earnings growth, so it swings
        widely, explodes when earnings barely grow and turns negative when they fall, which is
        why practitioners prefer forward or multi-year growth
        (`get_forward_price_earnings_growth_ratio`) and leave negative values out of
        comparisons.

        Also known as: PEG ratio, price earnings growth.

        Args:
            use_ebitda_growth_rate (bool, optional): Whether to use EBITDA growth rate for the calculation.
                Defaults to False.
            include_dividends (bool, optional): Whether to deduct preferred dividends from net income, in both
                the P/E and the EPS growth (not in the EBITDA growth). Defaults to False.
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.DataFrame: Price earnings to growth (PEG) ratio values.

        Notes:
        - The method retrieves historical data and calculates the PEG ratio for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the PEG ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["MA", "ADP", "COST"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_price_to_earnings_growth_ratio()
        ```

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | ADP  | 6.2692 | 2.2192 | 1.6493 | 2.9504 | 2.6649 |
        | COST | 2.016  | 2.0916 | 6.0241 | 3.2539 | 4.7694 |
        | MA   | 1.0968 | 2.0264 | 2.2927 | 2.1854 | 1.7108 |
        """
        # On quarterly data both the earnings and the growth cover a year: trailing twelve
        # month figures compared with those of four quarters earlier.
        trailing_metric = 4 if self._quarterly else 1

        if use_ebitda_growth_rate:
            growth_rate = (
                self._income_statement.loc[:, "EBITDA", :]
                .T.rolling(trailing_metric)
                .sum()
                .T
            )

            growth_rate = calculate_growth(dataset=growth_rate, lag=trailing_metric)
        else:
            growth_rate = self.get_earnings_per_share(
                include_dividends,
                diluted=diluted,
                growth=True,
                lag=trailing_metric,
                trailing=trailing_metric,
            )

        price_earnings = self.get_price_to_earnings_ratio(
            include_dividends=include_dividends, diluted=diluted
        )

        price_to_earnings_growth_ratio = (
            valuation_model.get_price_to_earnings_growth_ratio(
                price_earnings, growth_rate * 100
            )
        )

        return finalize_dataset(
            dataset=price_to_earnings_growth_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    def _get_or_fetch_analyst_estimates(self) -> pd.DataFrame:
        """
        Return the cached analyst estimates, fetching them on first use only.

        Used exclusively by `get_forward_price_earnings_ratio` and
        `get_forward_price_earnings_growth_ratio` — no other ratio touches analyst estimates,
        so initializing or using the rest of the Ratios module never triggers this fetch.

        The cache lives in `self._analyst_estimates_cache`, a dict the Toolkit passes in by
        reference (see `analyst_estimates_cache` on `__init__`). Every access to `toolkit.ratios`
        constructs a brand new Ratios instance, so caching the DataFrame directly on `self` would
        not survive across separate `toolkit.ratios.<method>()` calls — the shared dict is what
        makes the fetch-once behaviour actually hold.

        Returns:
            pd.DataFrame: The analyst estimates, or an empty DataFrame if no `api_key` was
            provided or the fetch failed (e.g. no Premium FMP subscription).
        """
        if "data" not in self._analyst_estimates_cache and self._api_key:
            # Same source, dataset and parameters as Toolkit.get_analyst_estimates.
            fetched_analyst_estimates, _ = ticker_model.collect_per_ticker(
                cache=get_active_cache(),
                source=policy_model.FINANCIAL_MODELING_PREP,
                dataset="analyst_estimates",
                tickers=self._tickers_without_portfolio,
                ticker_axis=ticker_model.TICKER_ON_INDEX,
                parameters={
                    "quarter": self._quarterly,
                    "start_date": self._start_date,
                },
                collector=lambda tickers: _get_analyst_estimates(
                    tickers=tickers,
                    api_key=self._api_key,
                    quarter=self._quarterly,
                    start_date=self._start_date,
                    rounding=self._rounding,
                    sleep_timer=self._sleep_timer,
                    user_subscription=self._user_subscription,
                ),
            )
            self._analyst_estimates_cache["data"] = fetched_analyst_estimates

        return self._analyst_estimates_cache.get("data", pd.DataFrame())

    def _get_consensus_eps(
        self, analyst_estimates: pd.DataFrame, min_analysts: int | None
    ) -> pd.DataFrame:
        """
        Return the consensus (average) EPS estimate per ticker and period, leaving out the
        periods whose EPS consensus rests on fewer than `min_analysts` analysts.

        Far-out years are often covered by a handful of analysts (or one), which makes their
        consensus, and every forward ratio built on it, a single opinion rather than a
        consensus. On quarterly data a thin quarter also leaves out the year-long windows it is
        part of, since the forward ratios sum four quarters.

        Args:
            analyst_estimates (pd.DataFrame): The analyst estimates, with the "Estimated EPS
                Average" and "Number of Analysts EPS" rows per ticker.
            min_analysts (int | None): The minimum number of analysts behind the EPS
                consensus, or None to keep every period.

        Returns:
            pd.DataFrame: The consensus EPS estimate per ticker and period, NaN where too few
                analysts cover it.
        """
        estimated_eps = analyst_estimates.loc[:, "Estimated EPS Average", :]

        if min_analysts is None:
            return estimated_eps

        # A period without a reported count is left out too: its depth is unknown.
        analyst_count = (
            analyst_estimates.loc[:, "Number of Analysts EPS", :]
            if "Number of Analysts EPS" in analyst_estimates.index.get_level_values(1)
            else pd.DataFrame(
                np.nan, index=estimated_eps.index, columns=estimated_eps.columns
            )
        )

        return estimated_eps.where(
            analyst_count.reindex_like(estimated_eps) >= min_analysts
        )

    def _get_next_twelve_months_eps(self, estimated_eps: pd.DataFrame) -> pd.DataFrame:
        """
        Return the estimated EPS of a year from each estimate period onwards.

        A P/E compares the price with a year of earnings. Yearly estimates already cover a
        year; quarterly estimates cover a quarter, so on quarterly data each period is given
        the sum of its own estimate and those of the next three quarters (the next twelve
        months from that quarter). Periods without four consecutive quarterly estimates are
        NaN.

        Args:
            estimated_eps (pd.DataFrame): The consensus EPS estimate per ticker and period.

        Returns:
            pd.DataFrame: The estimated EPS over a year starting at each period.
        """
        if not self._quarterly or estimated_eps.empty:
            return estimated_eps

        # Every quarter in the range, so a missing estimate leaves a gap rather than
        # pulling a later quarter into the window.
        quarters = pd.period_range(
            estimated_eps.columns.min(), estimated_eps.columns.max(), freq="Q"
        )
        full = estimated_eps.reindex(columns=quarters).astype(float)

        # A rolling sum over the reversed columns sums each quarter with the three after it.
        next_four_quarters = (
            full.iloc[:, ::-1].T.rolling(4, min_periods=4).sum().T.iloc[:, ::-1]
        )

        return next_four_quarters.reindex(columns=estimated_eps.columns)

    @handle_portfolio
    @handle_errors
    def get_forward_price_earnings_ratio(
        self,
        min_analysts: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
    ):
        """
        Calculate the forward price earnings ratio (forward P/E), a valuation ratio that compares
        the current stock price to the analyst consensus (average) EPS estimate for a future period,
        instead of the trailing EPS used by the regular P/E ratio.

        The formula is as follows:

        - Forward P/E = Latest Stock Price / Estimated EPS Average

        Where the latest stock price is the most recent traded close, used for every estimate period. On
        quarterly data the estimated EPS of a period is the sum of the estimates of that quarter and the
        next three, so the ratio compares the price with a year of estimated earnings, as it does with
        yearly data.

        The forward P/E is the standard valuation multiple in equity research because prices
        look ahead, but it rests on consensus estimates, which tend to be revised as results
        come in. Consensus EPS is usually on the adjusted basis companies guide on, while the
        trailing P/E (`get_price_to_earnings_ratio`) uses reported diluted EPS, so for
        companies with large amortization or restructuring charges the forward P/E looks much
        lower than the trailing one without any change in earnings power.

        Note that this requires an `api_key` to be set on the Toolkit and a Premium FMP
        subscription. Analyst estimates are fetched once per Toolkit instance and cached
        across calls to this (or the other forward-looking ratio) method — never on
        initialization or on any other ratio.

        Also known as: forward P/E, projected P/E.

        Args:
            min_analysts (int | None, optional): The minimum number of analysts behind the EPS
                consensus of a period; periods covered by fewer are NaN. The number per period is
                the "Number of Analysts EPS" of `Toolkit.get_analyst_estimates`. Defaults to None,
                which keeps every period.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.DataFrame: The current share price divided by the consensus EPS estimate of each
                period in the analyst estimates. Only the periods after the latest reported fiscal
                year are forward P/E ratios; earlier columns divide today's price by the estimates
                made for years that have already been reported.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["NVDA", "AMD", "AVGO"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_forward_price_earnings_ratio(min_analysts=5).loc[:, "2026":"2030"]
        ```

        Which returns:

        |      |    2026 |    2027 |    2028 |     2029 |     2030 |
        |:-----|--------:|--------:|--------:|---------:|---------:|
        | AMD  | 26.8499 | 12.2758 |  8.3568 |   6.2147 |   4.824  |
        | AVGO | 29.6209 | 17.9157 | 11.2984 | nan      | nan      |
        | NVDA | 20.0741 | 11.798  |  8.7249 |   7.2852 |   5.6192 |
        """
        analyst_estimates = self._get_or_fetch_analyst_estimates()

        if analyst_estimates.empty:
            logger.error(
                "No analyst estimates available. This requires an api_key to be set and a "
                "Premium FMP subscription."
            )
            return None

        # A year of estimated earnings from each period onwards (four quarters on quarterly data).
        estimated_eps = self._get_next_twelve_months_eps(
            self._get_consensus_eps(analyst_estimates, min_analysts)
        )

        stock_price = (
            self._daily_historical_data["Close"]
            .ffill()
            .iloc[-1][self._tickers_without_portfolio]
            .reindex(estimated_eps.index)
        )
        stock_price_broadcast = pd.DataFrame(
            {period: stock_price for period in estimated_eps.columns},
            index=estimated_eps.index,
        )

        forward_price_earnings_ratio = valuation_model.get_price_to_earnings_ratio(
            stock_price=stock_price_broadcast, earnings_per_share=estimated_eps
        )

        return finalize_dataset(
            dataset=forward_price_earnings_ratio,
            start_date=self._start_date,
            end_date=None,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_forward_price_earnings_growth_ratio(
        self,
        min_analysts: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
    ):
        """
        Calculate the forward price earnings to growth (forward PEG) ratio, a valuation metric
        that measures the forward P/E ratio relative to the growth implied by the analyst
        consensus (average) EPS estimate versus the company's most recently reported actual EPS.

        The formula is as follows:

            - Estimated EPS Growth Rate = (Estimated EPS / Trailing EPS) ^ (1 / Years) - 1
            - Forward PEG = Forward P/E / (Estimated EPS Growth Rate * 100)

        The growth rate is the annualized growth from the most recently reported EPS to the
        estimate of each period, with Years the number of years between the two, so the PEG
        of every estimate period is based on a yearly growth rate. Periods up to the latest
        reported period, and periods where either EPS is zero or negative, have no compound
        growth rate and are NaN. The trailing EPS is the reported diluted EPS, while
        consensus estimates are usually on an adjusted basis, which inflates the implied
        growth for companies with large non-cash charges.

        On quarterly data both EPS figures cover a year: the estimate is the sum of the
        estimates of the period and the next three quarters, the trailing EPS the sum of the
        last four reported quarters, and Years runs from the end of the latest reported
        quarter to the end of the estimated window.

        Note that this requires an `api_key` to be set on the Toolkit and a Premium FMP
        subscription. Analyst estimates are fetched once per Toolkit instance and cached
        across calls to this (or the other forward-looking ratio) method — never on
        initialization or on any other ratio.

        Also known as: forward PEG, PEG on forward growth.

        Args:
            min_analysts (int | None, optional): The minimum number of analysts behind the EPS
                consensus of a period; periods covered by fewer are NaN. The number per period is
                the "Number of Analysts EPS" of `Toolkit.get_analyst_estimates`. Defaults to None,
                which keeps every period.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.DataFrame: Forward PEG ratio for each period covered by the analyst estimates.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["MA", "AXP", "JPM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_forward_price_earnings_growth_ratio().loc[:, "2026":"2030"]
        ```

        Which returns:

        |     |   2026 |   2027 |   2028 |     2029 |     2030 |
        |:----|-------:|-------:|-------:|---------:|---------:|
        | AXP | 1.2602 | 1.2788 | 1.1079 |   0.8708 |   0.97   |
        | JPM | 0.5538 | 0.9696 | 1.0544 | nan      | nan      |
        | MA  | 1.3735 | 1.3529 | 1.2061 |   1.0848 |   1.0166 |
        """
        analyst_estimates = self._get_or_fetch_analyst_estimates()

        if analyst_estimates.empty:
            logger.error(
                "No analyst estimates available. This requires an api_key to be set and a "
                "Premium FMP subscription."
            )
            return None

        raw_estimated_eps = self._get_consensus_eps(analyst_estimates, min_analysts)

        # Both EPS figures cover a year: on quarterly data the estimate of the period and
        # the next three quarters against the last four reported quarters.
        estimated_eps = self._get_next_twelve_months_eps(raw_estimated_eps)

        reported_eps = self._income_statement.loc[:, "EPS Diluted", :]

        trailing_reported_eps = (
            reported_eps.T.rolling(4, min_periods=4).sum().T.where(reported_eps.notna())
            if self._quarterly
            else reported_eps
        )

        trailing_eps = (
            trailing_reported_eps.ffill(axis=1).iloc[:, -1].reindex(estimated_eps.index)
        )
        trailing_eps_broadcast = pd.DataFrame(
            {period: trailing_eps for period in estimated_eps.columns},
            index=estimated_eps.index,
        )

        # The growth is annualized over the years between the latest reported period of each
        # ticker and each estimate period, so every estimate gives a yearly growth rate.
        periods_per_year = 4 if self._quarterly else 1
        latest_reported_period = reported_eps.apply(
            lambda row: row.last_valid_index(), axis=1
        ).reindex(estimated_eps.index)
        # On quarterly data the estimated window ends three quarters after its first period,
        # while the trailing window ends at the latest reported quarter. Estimate periods up to
        # the latest reported one are not forward-looking and get no growth rate.
        window_extension = 3 if self._quarterly else 0
        years_ahead = pd.DataFrame(
            {
                period: [
                    (
                        (
                            (
                                (period.year - latest.year) * periods_per_year
                                + (
                                    period.quarter - latest.quarter
                                    if self._quarterly
                                    else 0
                                )
                            )
                            + window_extension
                        )
                        / periods_per_year
                        if isinstance(latest, pd.Period) and period > latest
                        else np.nan
                    )
                    for latest in latest_reported_period
                ]
                for period in estimated_eps.columns
            },
            index=estimated_eps.index,
        )

        estimated_eps_growth_rate = valuation_model.get_estimated_eps_growth_rate(
            estimated_eps=estimated_eps,
            trailing_eps=trailing_eps_broadcast,
            years=years_ahead,
        )

        stock_price = (
            self._daily_historical_data["Close"]
            .ffill()
            .iloc[-1][self._tickers_without_portfolio]
            .reindex(estimated_eps.index)
        )
        stock_price_broadcast = pd.DataFrame(
            {period: stock_price for period in estimated_eps.columns},
            index=estimated_eps.index,
        )
        forward_price_earnings_ratio = valuation_model.get_price_to_earnings_ratio(
            stock_price=stock_price_broadcast, earnings_per_share=estimated_eps
        )

        forward_price_earnings_growth_ratio = (
            valuation_model.get_price_to_earnings_growth_ratio(
                forward_price_earnings_ratio, estimated_eps_growth_rate * 100
            )
        )

        return finalize_dataset(
            dataset=forward_price_earnings_growth_ratio,
            start_date=self._start_date,
            end_date=None,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    def _get_book_value_per_share(
        self, diluted: bool, trailing: int | None
    ) -> pd.DataFrame:
        """
        Returns the book value per share of every period, including those before the start
        date that the daily ratios carry over to the first days.

        Args:
            diluted (bool): Whether to use the diluted weighted average shares.
            trailing (int | None): The number of period ends to average over, or None.

        Returns:
            pd.DataFrame: The book value per share with the tickers as index and the periods as
            columns.
        """
        average_shares = (
            self._income_statement.loc[:, "Weighted Average Shares Diluted", :]
            if diluted
            else self._income_statement.loc[:, "Weighted Average Shares", :]
        )

        if trailing:
            book_value_per_share = valuation_model.get_book_value_per_share(
                self._mean_trailing(
                    self._balance_sheet_statement.loc[:, "Total Shareholder Equity", :],
                    trailing,
                ),
                self._mean_trailing(
                    self._balance_sheet_statement.loc[:, "Preferred Stock", :], trailing
                ),
                self._mean_trailing(average_shares, trailing),
            )
        else:
            book_value_per_share = valuation_model.get_book_value_per_share(
                self._balance_sheet_statement.loc[:, "Total Shareholder Equity", :],
                self._balance_sheet_statement.loc[:, "Preferred Stock", :],
                average_shares,
            )

        return book_value_per_share

    @handle_portfolio
    @handle_errors
    def get_book_value_per_share(
        self,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ):
        """
        Calculate the book value per share, a valuation ratio that measures the
        amount of common equity value per share outstanding.

        The book value per share is a fundamental valuation metric that reflects
        the net worth of a company attributed to each outstanding share of common stock.

        The formula is as follows:

        - Book Value per Share = (Total Shareholder Equity — Preferred Stock) / Weighted Average (Diluted) Shares

        Also known as: BVPS, net asset value per share.

        Args:
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.DataFrame: Book value per share values.

        Notes:
        - The method retrieves historical data and calculates the book value per share for each asset
        in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the book value per share values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["CB", "TRV", "ALL"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_book_value_per_share()
        ```

        Which returns:

        |     |     2021 |     2022 |     2023 |     2024 |    2025 |
        |:----|---------:|---------:|---------:|---------:|--------:|
        | ALL |  77.5961 |  57.2198 |  60.0724 |  72.5952 | 108.081 |
        | CB  | 134.734  | 119.282  | 143.666  | 155.426  | 186.02  |
        | TRV | 115.179  |  89.9458 | 107.326  | 120.571  | 144.525 |
        """
        book_value_per_share = self._get_book_value_per_share(diluted, trailing)

        return finalize_dataset(
            dataset=book_value_per_share,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_price_to_book_ratio(
        self,
        show_daily: bool = False,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ):
        """
        Calculate the price to book ratio, a valuation ratio that compares a
        company's market price to its book value per share.

        The price to book ratio is a key valuation metric that helps investors
        assess whether a company's stock is overvalued or undervalued relative to its
        underlying net asset value.

        The formula is as follows:

        - Price to Book Ratio = Share Price / Book Value per Share

        The price to book ratio matters most for banks and insurers, whose balance
        sheets consist mostly of financial assets and liabilities, so that book value
        is a fair measure of what shareholders own. It is read together with return on
        equity: a bank that earns more than its cost of equity can justify trading
        above book value, one that earns less trades below it. For companies whose
        value lies in brands, software or people, book value says little, and buybacks
        can shrink it until the ratio becomes extreme.

        Also known as: P/B ratio.

        Args:
            show_daily (bool, optional): Whether to show daily values: the close of each trading day
                against the statement figures of the latest period that has ended on or before
                that day (also when it ended before the start date), so no figures are used before
                their period closes. Defaults to False.
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.DataFrame: Price to book ratio values.

        Notes:
        - The method retrieves historical data and calculates the price to book ratio for each asset
        in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the price to book ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["JPM", "BAC", "C", "WFC"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_price_to_book_ratio()
        ```

        Which returns:

        |     |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:----|-------:|-------:|-------:|-------:|-------:|
        | BAC | 1.5519 | 1.105  | 1.0335 | 1.2655 | 1.4971 |
        | C   | 0.6764 | 0.4876 | 0.5356 | 0.7017 | 1.1306 |
        | JPM | 1.8484 | 1.5033 | 1.6661 | 2.1254 | 2.6176 |
        | WFC | 1.173  | 0.9854 | 1.1012 | 1.5174 | 1.8228 |
        """
        book_value_per_share = self._get_book_value_per_share(
            diluted, trailing if trailing else None
        )

        years = book_value_per_share.columns
        begin, end = str(years[0]), str(years[-1])

        if show_daily:
            share_prices = self._daily_historical_data.loc[begin:, "Close"][
                self._tickers_without_portfolio
            ]

            book_value_per_share = map_period_data_to_daily_data(
                period_data=book_value_per_share,
                daily_dates=share_prices.index,
                quarterly=self._quarterly,
                period_end_dates=self._period_end_dates,
            )
        else:
            share_prices = self._historical_data.loc[begin:end, "Close"][
                self._tickers_without_portfolio
            ].T

        price_to_book_ratio = valuation_model.get_price_to_book_ratio(
            share_prices, book_value_per_share
        )

        return finalize_dataset(
            dataset=price_to_book_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
            axis="rows" if show_daily else "columns",
            row_slice=show_daily,
        )

    @handle_portfolio
    @handle_errors
    def get_interest_debt_per_share(
        self,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ):
        """
        Calculate the interest debt per share, a valuation ratio that measures a company's
        total debt plus one year of interest expense per outstanding share.

        The interest debt per share ratio shows how much debt, with its interest, ranks
        ahead of each share, and is often compared with the share price. Because the total
        debt is added to the interest expense, the value is mostly debt per share.

        The formula is as follows:

        - Interest Debt per Share = (Interest Expense + Total Debt) / Weighted Average (Diluted) Shares

        Args:
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.DataFrame: Interest debt per share values.

        Notes:
        - The method retrieves historical data and calculates the interest debt per share ratio for each asset
        in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the interest debt per share values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["T", "VZ", "CMCSA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_interest_debt_per_share()
        ```

        Which returns:

        |       |    2021 |    2022 |    2023 |    2024 |    2025 |
        |:------|--------:|--------:|--------:|--------:|--------:|
        | CMCSA | 22.411  | 23.449  | 27.3869 | 26.4143 | 31.5875 |
        | T     | 26.9929 | 21.1965 | 22.2638 | 20.4981 | 25.1832 |
        | VZ    | 43.7176 | 42.7957 | 42.8154 | 41.4912 | 48.9927 |
        """
        average_shares = (
            self._income_statement.loc[:, "Weighted Average Shares Diluted", :]
            if diluted
            else self._income_statement.loc[:, "Weighted Average Shares", :]
        )

        if trailing:
            interest_debt_per_share = valuation_model.get_interest_debt_per_share(
                self._sum_trailing(
                    self._income_statement.loc[:, "Interest Expense", :], trailing
                ),
                self._mean_trailing(
                    self._balance_sheet_statement.loc[:, "Total Debt", :], trailing
                ),
                self._mean_trailing(average_shares, trailing),
            )
        else:
            interest_debt_per_share = valuation_model.get_interest_debt_per_share(
                self._income_statement.loc[:, "Interest Expense", :],
                self._balance_sheet_statement.loc[:, "Total Debt", :],
                average_shares,
            )

        return finalize_dataset(
            dataset=interest_debt_per_share,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_capex_per_share(
        self,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ):
        """
        Calculate the capex per share, a valuation ratio that measures the amount of
        capital expenditures made per outstanding share of a company's stock.

        The capex per share ratio provides insight into how much capital a company invests
        in its operations and growth initiatives relative to its shareholder base. It can
        help investors assess the level of reinvestment into the business.

        The formula is as follows:

        - CAPEX per Share = Capital Expenditure / Weighted Average (Diluted) Shares

        The values are negative because capital expenditure is reported as a cash outflow
        in the cash flow statement; take the magnitude when comparing with sources that
        quote capex per share as a positive amount. Capex per share is usually compared
        with operating cash flow or depreciation per share: capex well above depreciation
        for years points to expansion, capex below it to an ageing asset base.

        Also known as: capital expenditure per share.

        Args:
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.DataFrame: Capex per share values.

        Notes:
        - The method retrieves historical data and calculates the capex per share ratio for each asset
        in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the capex per share values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["UNP", "CSX", "NSC"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_capex_per_share()
        ```

        Which returns:

        |     |    2021 |    2022 |     2023 |     2024 |    2025 |
        |:----|--------:|--------:|---------:|---------:|--------:|
        | CSX | -0.7942 | -0.9963 |  -1.1212 |  -1.3016 | -1.5569 |
        | NSC | -5.925  | -8.2683 | -10.3298 | -10.5168 | -9.8043 |
        | UNP | -4.4797 | -5.8013 |  -5.9095 |  -5.672  | -6.3875 |
        """
        average_shares = (
            self._income_statement.loc[:, "Weighted Average Shares Diluted", :]
            if diluted
            else self._income_statement.loc[:, "Weighted Average Shares", :]
        )

        if trailing:
            capex_per_share = valuation_model.get_capex_per_share(
                self._sum_trailing(
                    self._cash_flow_statement.loc[:, "Capital Expenditure", :], trailing
                ),
                average_shares,
            )
        else:
            capex_per_share = valuation_model.get_capex_per_share(
                self._cash_flow_statement.loc[:, "Capital Expenditure", :],
                average_shares,
            )

        return finalize_dataset(
            dataset=capex_per_share,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_dividend_yield(
        self,
        show_daily: bool = False,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ):
        """
        Calculate the dividend yield ratio, a valuation ratio that measures the
        amount of dividends distributed per share of stock relative to the stock's price.

        The dividend yield ratio is used by investors to assess the income potential
        of an investment in a company's stock based on the dividends it pays out. A higher
        dividend yield can be attractive to income-seeking investors.

        The formula is as follows:

        - Dividend Yield = Dividends per Share / Share Price

        The dividends per share are those paid over a year: the trailing four quarters with
        quarterly data, each year with annual data and, with show_daily, the 365 calendar
        days up to each day. The share price is the close at the end of the period (or day).

        Also known as: annual dividend yield, income yield.

        Args:
            show_daily (bool, optional): Whether to show daily values: the close of each trading day
                against the dividends paid in the 365 calendar days up to that day (or the
                `trailing` periods in days). The prices start a year before the start date, so
                a window of up to a year is complete on the start date; days less than one window
                after a ticker's first price are NaN. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses the
                trailing twelve months (4 quarters) with quarterly data and single years with
                annual data.

        Returns:
            pd.DataFrame: Dividend yield values.

        Notes:
        - The method retrieves historical data and calculates the dividend yield ratio for each asset
        in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the dividend yield values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["KO", "PEP", "JNJ"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_dividend_yield()
        ```

        Which returns:

        |     |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:----|-------:|-------:|-------:|-------:|-------:|
        | JNJ | 0.0245 | 0.0252 | 0.03   | 0.034  | 0.0248 |
        | KO  | 0.0284 | 0.0277 | 0.0312 | 0.0312 | 0.0292 |
        | PEP | 0.0245 | 0.025  | 0.0291 | 0.0351 | 0.0392 |
        """
        # A dividend yield compares the price with a year of dividends, so quarterly data defaults
        # to the trailing twelve months.
        trailing = self._get_valuation_trailing(trailing)

        if show_daily:
            share_prices = self._daily_historical_data.loc[:, "Close"][
                self._tickers_without_portfolio
            ]
            daily_dividends = self._daily_historical_data.loc[:, "Dividends"][
                self._tickers_without_portfolio
            ]

            # Dividends are only paid on a few days a year, so each day sums the dividends of the
            # calendar window the yield covers (365 days, or the trailing periods in days). Days
            # less than one window after the first price have an incomplete window and stay NaN.
            window_days = round(self._get_days(None, trailing))

            dividends = daily_dividends.copy()
            dividends.index = (
                dividends.index.to_timestamp()
                if isinstance(dividends.index, pd.PeriodIndex)
                else pd.DatetimeIndex(dividends.index)
            )
            dividends = dividends.rolling(f"{window_days}D").sum()
            dividends.loc[
                dividends.index
                < dividends.index[0] + pd.Timedelta(days=window_days - 1)
            ] = np.nan
            dividends.index = daily_dividends.index
        else:
            share_prices = self._historical_data.loc[:, "Close"][
                self._tickers_without_portfolio
            ].T
            dividends = self._sum_trailing(
                self._historical_data.loc[:, "Dividends"][
                    self._tickers_without_portfolio
                ].T,
                trailing,
            )

        dividend_yield = valuation_model.get_dividend_yield(dividends, share_prices)

        return finalize_dataset(
            dataset=dividend_yield,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
            axis="rows" if show_daily else "columns",
            row_slice=show_daily,
        )

    @handle_portfolio
    @handle_errors
    def get_weighted_dividend_yield(
        self,
        show_daily: bool = False,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ):
        """
        Calculate the weighted dividend yield ratio, a valuation ratio that measures the
        amount of dividends distributed per share of stock relative to the stock's price.

        This dividend yield ratio takes into account the (diluted) weighted average shares and actual
        dividends paid as found in the cash flow statement. It provides a more accurate reflection
        of the dividends paid out per share, considering any changes in the number of shares.

        The formula is as follows:

        - Weighted Dividend Yield = (|Dividends Paid| / Weighted Average (Diluted) Shares) / Share Price

        Also known as: blended dividend yield.

        Args:
            show_daily (bool, optional): Whether to show daily values: the close of each trading day
                against the statement figures of the latest period that has ended on or before
                that day (also when it ended before the start date), so no figures are used before
                their period closes. Defaults to False.
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses the
                trailing twelve months (4 quarters) with quarterly data and single years with
                annual data.

        Returns:
            pd.DataFrame: Weighted dividend yield values.

        Notes:
        - The method retrieves historical data and calculates the weighted dividend yield ratio for each asset
        in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the weighted dividend yield values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["TXN", "QCOM", "ADI"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_weighted_dividend_yield()
        ```

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | ADI  | 0.0157 | 0.018  | 0.0167 | 0.0169 | 0.0143 |
        | QCOM | 0.0143 | 0.0257 | 0.0213 | 0.0212 | 0.0201 |
        | TXN  | 0.022  | 0.0281 | 0.0292 | 0.0278 | 0.0316 |
        """
        # A yield compares the price with a year of dividends, so quarterly data defaults to the
        # trailing twelve months.
        trailing = self._get_valuation_trailing(trailing)

        average_shares = (
            self._income_statement.loc[:, "Weighted Average Shares Diluted", :]
            if diluted
            else self._income_statement.loc[:, "Weighted Average Shares", :]
        )

        # Dividends Paid is reported as a negative cash outflow, so the magnitude is used. It is
        # summed over the trailing periods before it is spread over the days.
        dividends_paid = self._sum_trailing(
            abs(self._cash_flow_statement.loc[:, "Dividends Paid", :]), trailing
        )

        years = self._cash_flow_statement.columns
        begin, end = str(years[0]), str(years[-1])

        share_prices = self._get_share_prices(begin, end, show_daily)

        if show_daily:
            average_shares = map_period_data_to_daily_data(
                period_data=average_shares,
                daily_dates=share_prices.index,
                quarterly=self._quarterly,
                period_end_dates=self._period_end_dates,
            )

            dividends_paid = map_period_data_to_daily_data(
                period_data=dividends_paid,
                daily_dates=share_prices.index,
                quarterly=self._quarterly,
                period_end_dates=self._period_end_dates,
            )

        weighted_dividend_yield = valuation_model.get_weighted_dividend_yield(
            dividends_paid,
            average_shares,
            share_prices,
        )

        return finalize_dataset(
            dataset=weighted_dividend_yield,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
            axis="rows" if show_daily else "columns",
            row_slice=show_daily,
        )

    @handle_portfolio
    @handle_errors
    def get_price_to_cash_flow_ratio(
        self,
        show_daily: bool = False,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ):
        """
        Calculate the price to cash flow ratio, a valuation ratio that compares a
        company's market price to its operating cash flow per share.

        The price to cash flow ratio is a key valuation metric that helps investors
        assess the relative value of a company's stock. It is similar to the price to
        earnings ratio, but uses cash flow instead of earnings in the denominator.

        The formula is as follows:

        - Price to Cash Flow Ratio = Share Price / Cash Flow from Operations per Share

        Also known as: P/CF ratio.

        Args:
            show_daily (bool, optional): Whether to show daily values: the close of each trading day
                against the statement figures of the latest period that has ended on or before
                that day (also when it ended before the start date), so no figures are used before
                their period closes. Defaults to False.
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses the
                trailing twelve months (4 quarters) with quarterly data and single years with
                annual data.

        Returns:
            pd.DataFrame: Price to cash flow ratio values.

        Notes:
        - The method retrieves historical data and calculates the price to cash flow ratio for each asset
        in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the price to cash flow ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["SHEL", "BP", "TTE"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_price_to_cash_flow_ratio()
        ```

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | BP   | 3.8083 | 2.7006 | 3.2687 | 3.0351 | 3.7607 |
        | SHEL | 3.7559 | 3.0844 | 4.1282 | 3.6452 | 5.0988 |
        | TTE  | 4.3059 | 3.3711 | 4.0313 | 4.1606 | 5.0908 |
        """
        # Price-based ratios compare the price with a year of flows, so quarterly data
        # defaults to the trailing twelve months.
        trailing = self._get_valuation_trailing(trailing)

        market_cap = self.get_market_cap(diluted=diluted, show_daily=show_daily)

        # The flows are summed over the trailing periods first and only then spread over the
        # days, so a trailing window always covers periods and never days.
        cash_flow_from_operations = self._sum_trailing(
            self._cash_flow_statement.loc[:, "Cash Flow from Operations", :], trailing
        )

        if show_daily:
            cash_flow_from_operations = map_period_data_to_daily_data(
                period_data=cash_flow_from_operations,
                daily_dates=market_cap.index,
                quarterly=self._quarterly,
                period_end_dates=self._period_end_dates,
            )

        price_to_cash_flow_ratio = valuation_model.get_price_to_cash_flow_ratio(
            market_cap, cash_flow_from_operations
        )

        return finalize_dataset(
            dataset=price_to_cash_flow_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
            axis="rows" if show_daily else "columns",
            row_slice=show_daily,
        )

    @handle_portfolio
    @handle_errors
    def get_price_to_free_cash_flow_ratio(
        self,
        show_daily: bool = False,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ):
        """
        Calculate the price to free cash flow ratio, a valuation ratio that compares a
        company's market price to its free cash flow per share.

        This ratio provides insight into how the market values a company's ability to generate free cash flow.

        The formula is as follows:

        - Price to Free Cash Flow Ratio = Market Cap / Free Cash Flow

        Also known as: P/FCF ratio.

        Args:
            show_daily (bool, optional): Whether to show daily values: the close of each trading day
                against the statement figures of the latest period that has ended on or before
                that day (also when it ended before the start date), so no figures are used before
                their period closes. Defaults to False.
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses the
                trailing twelve months (4 quarters) with quarterly data and single years with
                annual data.

        Returns:
            pd.DataFrame: Price to free cash flow ratio values.

        Notes:
        - The method retrieves historical data and calculates the price to free cash flow ratio for each asset
        in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the price to free cash flow ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["ADBE", "CRM", "INTU"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_price_to_free_cash_flow_ratio()
        ```

        Which returns:

        |      |    2021 |    2022 |    2023 |    2024 |    2025 |
        |:-----|--------:|--------:|--------:|--------:|--------:|
        | ADBE | 39.57   | 21.4313 | 39.4468 | 25.4167 | 15.1691 |
        | CRM  | 46.8527 | 20.9397 | 27.2615 | 26.1893 | 17.5846 |
        | INTU | 56.1917 | 30.2018 | 36.9585 | 38.5183 | 30.8178 |
        """
        # Price-based ratios compare the price with a year of flows, so quarterly data
        # defaults to the trailing twelve months.
        trailing = self._get_valuation_trailing(trailing)

        market_cap = self.get_market_cap(diluted=diluted, show_daily=show_daily)

        # The flows are summed over the trailing periods first and only then spread over the
        # days, so a trailing window always covers periods and never days.
        free_cash_flow = self._sum_trailing(
            self._cash_flow_statement.loc[:, "Free Cash Flow", :], trailing
        )

        if show_daily:
            free_cash_flow = map_period_data_to_daily_data(
                period_data=free_cash_flow,
                daily_dates=market_cap.index,
                quarterly=self._quarterly,
                period_end_dates=self._period_end_dates,
            )

        price_to_free_cash_flow_ratio = (
            valuation_model.get_price_to_free_cash_flow_ratio(
                market_cap, free_cash_flow
            )
        )

        return finalize_dataset(
            dataset=price_to_free_cash_flow_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
            axis="rows" if show_daily else "columns",
            row_slice=show_daily,
        )

    @handle_portfolio
    @handle_errors
    def get_price_to_sales_ratio(
        self,
        show_daily: bool = False,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ):
        """
        Calculate the price to sales ratio (P/S), a valuation ratio that compares a
        company's market capitalization to its total revenue.

        The price to sales ratio is particularly useful for valuing companies that are
        not yet profitable (and therefore have no meaningful P/E ratio), since revenue
        is typically positive even when earnings are not, and is less susceptible to
        accounting distortions than earnings-based multiples. It is, however, less
        informative than earnings- or cash-flow-based multiples for mature, profitable
        companies since it ignores profitability and cost structure entirely.

        The formula is as follows:

        - Price to Sales Ratio = Market Cap / Revenue

        Also known as: P/S ratio, sales multiple.

        Args:
            show_daily (bool, optional): Whether to show daily values: the close of each trading day
                against the statement figures of the latest period that has ended on or before
                that day (also when it ended before the start date), so no figures are used before
                their period closes. Defaults to False.
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses the
                trailing twelve months (4 quarters) with quarterly data and single years with
                annual data.

        Returns:
            pd.DataFrame: Price to sales ratio values.

        Notes:
        - The method retrieves historical data and calculates the price to sales ratio for each asset
        in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the price to sales ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["NVDA", "AMD", "INTC"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_price_to_sales_ratio()
        ```

        Which returns:

        |      |    2021 |    2022 |    2023 |    2024 |    2025 |
        |:-----|--------:|--------:|--------:|--------:|--------:|
        | AMD  | 10.7614 |  4.3114 | 10.5618 |  7.6685 | 10.1148 |
        | INTC |  2.6655 |  1.7282 |  3.903  |  1.6161 |  3.3903 |
        | NVDA | 27.7019 | 13.5824 | 20.2731 | 25.5249 | 21.1721 |
        """
        # Price-based ratios compare the price with a year of flows, so quarterly data
        # defaults to the trailing twelve months.
        trailing = self._get_valuation_trailing(trailing)

        market_cap = self.get_market_cap(diluted=diluted, show_daily=show_daily)

        # The flows are summed over the trailing periods first and only then spread over the
        # days, so a trailing window always covers periods and never days.
        revenue = self._sum_trailing(
            self._income_statement.loc[:, "Revenue", :], trailing
        )

        if show_daily:
            revenue = map_period_data_to_daily_data(
                period_data=revenue,
                daily_dates=market_cap.index,
                quarterly=self._quarterly,
                period_end_dates=self._period_end_dates,
            )

        price_to_sales_ratio = valuation_model.get_price_to_sales_ratio(
            market_cap, revenue
        )

        return finalize_dataset(
            dataset=price_to_sales_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
            axis="rows" if show_daily else "columns",
            row_slice=show_daily,
        )

    @handle_portfolio
    @handle_errors
    def get_market_cap(
        self,
        show_daily: bool = False,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
    ):
        """
        Calculates the market capitalization of the company.

        Market capitalization, often referred to as "market cap," is the total value of a company's
        outstanding shares of stock in the stock market. It is calculated by multiplying the share
        price at the end of each period by the weighted average (diluted) number of shares
        outstanding in that period. The share price is the close as traded, adjusted for splits
        but not for dividends, as are the prices of every price-based ratio (P/E, P/B, the
        enterprise value multiples and the dividend, earnings and free cash flow yields).

        The formula is as follows:

        - Market Capitalization = Share Price * Weighted Average (Diluted) Shares

        Market cap is a point-in-time value, so the price is the close on the last trading
        day of each calendar period (or of each day with show_daily) and never an average.
        The periods follow the calendar, so for a company whose fiscal year ends in, for
        example, September, the price of the year is the December close while the shares
        come from the fiscal year. With show_daily, each day uses the shares of the latest
        period that has ended.

        Also known as: market capitalization.

        Args:
            show_daily (bool, optional): Whether to show daily values: the close of each trading day
                against the statement figures of the latest period that has ended on or before
                that day (also when it ended before the start date), so no figures are used before
                their period closes. Defaults to False.
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.DataFrame: Market capitalization values.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["NVDA", "AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_market_cap()
        ```

        Which returns:

        |      |        2021 |        2022 |        2023 |        2024 |        2025 |
        |:-----|------------:|------------:|------------:|------------:|------------:|
        | AAPL | 2.9947e+12  | 2.12121e+12 | 3.04439e+12 | 3.8585e+12  | 4.07918e+12 |
        | MSFT | 2.55872e+12 | 1.80824e+12 | 2.80977e+12 | 3.14818e+12 | 3.61022e+12 |
        | NVDA | 7.45569e+11 | 3.66373e+11 | 1.23508e+12 | 3.33093e+12 | 4.57186e+12 |
        """
        average_shares = (
            self._income_statement.loc[:, "Weighted Average Shares Diluted", :]
            if diluted
            else self._income_statement.loc[:, "Weighted Average Shares", :]
        )

        years = self._cash_flow_statement.columns
        begin, end = str(years[0]), str(years[-1])

        # Market cap is a point-in-time value: the traded close at the end of each period (or on
        # each day) times the shares, never a price averaged over a trailing window.
        share_prices = self._get_share_prices(begin, end, show_daily)

        if show_daily:
            average_shares = map_period_data_to_daily_data(
                period_data=average_shares,
                daily_dates=share_prices.index,
                quarterly=self._quarterly,
                period_end_dates=self._period_end_dates,
            )

        market_cap = valuation_model.get_market_cap(share_prices, average_shares)

        return finalize_dataset(
            dataset=market_cap,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
            axis="rows" if show_daily else "columns",
            row_slice=show_daily,
        )

    @handle_portfolio
    @handle_errors
    def get_enterprise_value(
        self,
        show_daily: bool = False,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
    ):
        """
        Calculates the Enterprise Value (EV) of a company. The Enterprise Value (EV)
        is a measure of a company's total value, often used as a more comprehensive
        alternative to market capitalization. It is calculated as the sum of a company's
        market capitalization, outstanding debt, minority interest, and
        preferred equity, minus the cash and cash equivalents.

        The formula is as follows:

        - Enterprise Value = Market Capitalization + Total Debt + Minority Interest + Preferred Equity
            — (Cash and Cash Equivalents + Short Term Investments)

        Enterprise value is the price of the whole business, equity and debt together,
        which makes it the right numerator for multiples of operating results such as EV
        to EBITDA or EV to sales, and comparable across companies with different capital
        structures. Cash and marketable securities (short-term investments) are both
        deducted (Damodaran), since both are available to pay off the debt. Enterprise value is a
        point-in-time value: the market cap and the balance sheet items are taken at the end
        of each period, or on each day with show_daily, where each day uses the balance
        sheet of the latest period that has ended.

        Also known as: EV, total enterprise value.

        Args:
            show_daily (bool, optional): Whether to show daily values: the close of each trading day
                against the statement figures of the latest period that has ended on or before
                that day (also when it ended before the start date), so no figures are used before
                their period closes. Defaults to False.
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.DataFrame: Enterprise Value values.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["NFLX", "DIS", "WBD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_enterprise_value()
        ```

        Which returns:

        |      |        2021 |        2022 |        2023 |        2024 |        2025 |
        |:-----|------------:|------------:|------------:|------------:|------------:|
        | DIS  | 3.39164e+11 | 2.12744e+11 | 2.15456e+11 | 2.52223e+11 | 2.50508e+11 |
        | NFLX | 2.86423e+11 | 1.41371e+11 | 2.28687e+11 | 3.99933e+11 | 4.10176e+11 |
        | WBD  | 2.83757e+10 | 6.52312e+10 | 6.88567e+10 | 6.09905e+10 | 1.02163e+11 |
        """
        # Enterprise value is a point-in-time value, so the balance sheet items are taken at the
        # end of each period, like the share price in the market cap. Cash and marketable
        # securities (short-term investments) are deducted together (Damodaran).
        balance_items = {
            "total_debt": self._balance_sheet_statement.loc[:, "Total Debt", :],
            "minority_interest": self._balance_sheet_statement.loc[
                :, "Minority Interest", :
            ],
            "preferred_equity": self._balance_sheet_statement.loc[
                :, "Preferred Stock", :
            ],
            "cash_and_cash_equivalents": self._get_cash_and_short_term_investments(),
        }

        market_cap = self.get_market_cap(diluted=diluted, show_daily=show_daily)

        if show_daily:
            balance_items = {
                item: map_period_data_to_daily_data(
                    period_data=values,
                    daily_dates=market_cap.index,
                    quarterly=self._quarterly,
                    period_end_dates=self._period_end_dates,
                )
                for item, values in balance_items.items()
            }

        enterprise_value = valuation_model.get_enterprise_value(
            market_cap=market_cap, **balance_items
        )

        return finalize_dataset(
            dataset=enterprise_value,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
            axis="rows" if show_daily else "columns",
            row_slice=show_daily,
        )

    @handle_portfolio
    @handle_errors
    def get_ev_to_sales_ratio(
        self,
        show_daily: bool = False,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ):
        """
        Calculate the EV to sales ratio, a valuation ratio that compares a company's
        enterprise value (EV) to its total revenue.

        This ratio compares the enterprise value (EV) to the total revenue generated
        by the company. It can provide insights into how efficiently a company is using
        its revenue to generate value for its investors.

        The formula is as follows:

        - Enterprise Value to Sales Ratio = Enterprise Value / Total Revenue

        Also known as: EV/Sales.

        Args:
            show_daily (bool, optional): Whether to show daily values: the close of each trading day
                against the statement figures of the latest period that has ended on or before
                that day (also when it ended before the start date), so no figures are used before
                their period closes. Defaults to False.
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses the
                trailing twelve months (4 quarters) with quarterly data and single years with
                annual data.

        Returns:
            pd.DataFrame: EV to Sales Ratio values.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["NOW", "WDAY", "CRM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_ev_to_sales_ratio()
        ```

        Which returns:

        |      |    2021 |    2022 |    2023 |    2024 |    2025 |
        |:-----|--------:|--------:|--------:|--------:|--------:|
        | CRM  |  9.4734 |  4.2668 |  7.3823 |  8.5235 |  6.2821 |
        | NOW  | 22.1821 | 10.6245 | 15.902  | 19.7984 | 11.7871 |
        | WDAY | 13.2039 |  6.3974 |  9.4665 |  7.6732 |  5.7531 |
        """
        # Price-based ratios compare the price with a year of flows, so quarterly data
        # defaults to the trailing twelve months.
        trailing = self._get_valuation_trailing(trailing)

        enterprise_value = self.get_enterprise_value(
            diluted=diluted, show_daily=show_daily
        )

        # The flows are summed over the trailing periods first and only then spread over the
        # days, so a trailing window always covers periods and never days.
        revenue = self._sum_trailing(
            self._income_statement.loc[:, "Revenue", :], trailing
        )

        if show_daily:
            revenue = map_period_data_to_daily_data(
                period_data=revenue,
                daily_dates=enterprise_value.index,
                quarterly=self._quarterly,
                period_end_dates=self._period_end_dates,
            )

        ev_to_sales_ratio = valuation_model.get_ev_to_sales_ratio(
            enterprise_value, revenue
        )

        return finalize_dataset(
            dataset=ev_to_sales_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
            axis="rows" if show_daily else "columns",
            row_slice=show_daily,
        )

    @handle_portfolio
    @handle_errors
    def get_ev_to_ebitda_ratio(
        self,
        show_daily: bool = False,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ):
        """
        Calculate the enterprise value over EBITDA ratio, a valuation ratio that
        measures a company's total value (including debt and equity) relative to its EBITDA.

        This ratio helps investors understand how many times the enterprise value exceeds the
        company's EBITDA, providing insights into the company's debt load and operating performance.

        The formula is as follows:

        - Enterprise Value to EBITDA Ratio = Enterprise Value / EBITDA

        EV to EBITDA is the most common multiple in sector comparisons and takeover pricing
        because it is not affected by capital structure, tax rates or depreciation policy.
        That is also its limit: it ignores the capital expenditure needed to sustain EBITDA,
        so capital-intensive companies look cheaper than they are, and EV to EBIT
        (`get_ev_to_ebit`) or EV to free cash flow (`get_ev_to_free_cash_flow_ratio`) is a
        useful cross-check.

        Also known as: EV/EBITDA.

        Args:
            show_daily (bool, optional): Whether to show daily values: the close of each trading day
                against the statement figures of the latest period that has ended on or before
                that day (also when it ended before the start date), so no figures are used before
                their period closes. Defaults to False.
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses the
                trailing twelve months (4 quarters) with quarterly data and single years with
                annual data.

        Returns:
            pd.DataFrame: EV to EBITDA Ratio values.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["TMO", "DHR", "A"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_ev_to_ebitda_ratio()
        ```

        Which returns:

        |     |    2021 |    2022 |    2023 |    2024 |    2025 |
        |:----|--------:|--------:|--------:|--------:|--------:|
        | A   | 30.0759 | 24.0884 | 26.0937 | 23.6894 | 22.7554 |
        | DHR | 23.3741 | 19.6229 | 24.9274 | 24.4558 | 23.2822 |
        | TMO | 23.0008 | 20.5519 | 21.5992 | 21.0334 | 22.7626 |
        """
        # Price-based ratios compare the price with a year of flows, so quarterly data
        # defaults to the trailing twelve months.
        trailing = self._get_valuation_trailing(trailing)

        enterprise_value = self.get_enterprise_value(
            diluted=diluted, show_daily=show_daily
        )

        # The flows are summed over the trailing periods first and only then spread over the
        # days, so a trailing window always covers periods and never days.
        operating_income = self._sum_trailing(
            self._income_statement.loc[:, "Operating Income", :], trailing
        )
        depreciation_and_amortization = self._sum_trailing(
            self._cash_flow_statement.loc[:, "Depreciation and Amortization", :],
            trailing,
        )

        if show_daily:
            operating_income = map_period_data_to_daily_data(
                period_data=operating_income,
                daily_dates=enterprise_value.index,
                quarterly=self._quarterly,
                period_end_dates=self._period_end_dates,
            )
            depreciation_and_amortization = map_period_data_to_daily_data(
                period_data=depreciation_and_amortization,
                daily_dates=enterprise_value.index,
                quarterly=self._quarterly,
                period_end_dates=self._period_end_dates,
            )

        ev_to_ebitda_ratio = valuation_model.get_ev_to_ebitda_ratio(
            enterprise_value, operating_income, depreciation_and_amortization
        )

        return finalize_dataset(
            dataset=ev_to_ebitda_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
            axis="rows" if show_daily else "columns",
            row_slice=show_daily,
        )

    @handle_portfolio
    @handle_errors
    def get_ev_to_operating_cashflow_ratio(
        self,
        show_daily: bool = False,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ):
        """
        Calculate the enterprise value over operating cash flow ratio, a valuation ratio that
        measures a company's total value (including debt and equity) relative to its operating cash flow.

        The ratio is a valuation metric that helps investors assess the company's valuation
        relative to its operating cash flow. This ratio provides insights into how many times
        the enterprise value exceeds the company's operating cash flow, indicating the company's
        ability to generate cash from its operations.

        The formula is as follows:

        - Enterprise Value to Operating Cash Flow Ratio = Enterprise Value / Operating Cash Flow

        Args:
            show_daily (bool, optional): Whether to show daily values: the close of each trading day
                against the statement figures of the latest period that has ended on or before
                that day (also when it ended before the start date), so no figures are used before
                their period closes. Defaults to False.
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses the
                trailing twelve months (4 quarters) with quarterly data and single years with
                annual data.

        Returns:
            pd.DataFrame: EV to Operating Cash Flow Ratio values.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["UNH", "ELV", "CI"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_ev_to_operating_cashflow_ratio()
        ```

        Which returns:

        |     |    2021 |    2022 |    2023 |    2024 |    2025 |
        |:----|--------:|--------:|--------:|--------:|--------:|
        | CI  | 14.8285 | 14.8491 |  9.4144 |  9.8596 | 10.1182 |
        | ELV | 16.0786 | 17.0313 | 16.3969 | 19.0153 | 23.6923 |
        | UNH | 22.6854 | 20.6807 | 18.639  | 21.8009 | 18.1884 |
        """
        # Price-based ratios compare the price with a year of flows, so quarterly data
        # defaults to the trailing twelve months.
        trailing = self._get_valuation_trailing(trailing)

        enterprise_value = self.get_enterprise_value(
            diluted=diluted, show_daily=show_daily
        )

        # The flows are summed over the trailing periods first and only then spread over the
        # days, so a trailing window always covers periods and never days.
        cash_flow_from_operations = self._sum_trailing(
            self._cash_flow_statement.loc[:, "Cash Flow from Operations", :], trailing
        )

        if show_daily:
            cash_flow_from_operations = map_period_data_to_daily_data(
                period_data=cash_flow_from_operations,
                daily_dates=enterprise_value.index,
                quarterly=self._quarterly,
                period_end_dates=self._period_end_dates,
            )

        ev_to_operating_cashflow_ratio = (
            valuation_model.get_ev_to_operating_cashflow_ratio(
                enterprise_value, cash_flow_from_operations
            )
        )

        return finalize_dataset(
            dataset=ev_to_operating_cashflow_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
            axis="rows" if show_daily else "columns",
            row_slice=show_daily,
        )

    @handle_portfolio
    @handle_errors
    def get_earnings_yield(
        self,
        show_daily: bool = False,
        include_dividends: bool = False,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ):
        """
        Calculate the earnings yield ratio, a valuation ratio that measures the earnings per share
        relative to the market price per share.

        The earnings yield ratio is a valuation metric that provides insights into how much a
        company's earnings contribute to its stock price. It compares the earnings per share
        to the market price per share, helping investors understand the earnings potential of
        the company relative to its current market value

        The formula is as follows:

        - Earnings Yield Ratio = Earnings per Share / Share Price

        Also known as: E/P ratio, inverse PE ratio.

        Args:
            show_daily (bool, optional): Whether to show daily values: the close of each trading day
                against the statement figures of the latest period that has ended on or before
                that day (also when it ended before the start date), so no figures are used before
                their period closes. Defaults to False.
            include_dividends (bool, optional): Whether to deduct preferred dividends from net income.
                Defaults to False.
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses the
                trailing twelve months (4 quarters) with quarterly data and single years with
                annual data.

        Returns:
            pd.DataFrame: Earnings Yield Ratio values.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["F", "GM", "TM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_earnings_yield()
        ```

        Which returns:

        |    |   2021 |    2022 |   2023 |   2024 |     2025 |
        |:---|-------:|--------:|-------:|-------:|---------:|
        | F  | 0.2141 | -0.0424 | 0.0882 | 0.1477 |  -0.1567 |
        | GM | 0.1143 |  0.1823 | 0.2038 | 0.1195 |   0.0402 |
        | TM | 0.0986 |  0.0971 | 0.1382 | 0.1225 | nan      |
        """
        # The earnings yield is the inverse of the P/E, so it uses the same earnings: the trailing
        # twelve months by default on quarterly data.
        eps = self._get_earnings_per_share(
            include_dividends, diluted, self._get_valuation_trailing(trailing)
        )

        years = eps.columns
        begin, end = str(years[0]), str(years[-1])

        share_prices = self._get_share_prices(begin, end, show_daily)

        if show_daily:
            eps = map_period_data_to_daily_data(
                period_data=eps,
                daily_dates=share_prices.index,
                quarterly=self._quarterly,
                period_end_dates=self._period_end_dates,
            )

        earnings_yield = valuation_model.get_earnings_yield(eps, share_prices)

        return finalize_dataset(
            dataset=earnings_yield,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
            axis="rows" if show_daily else "columns",
            row_slice=show_daily,
        )

    @handle_portfolio
    @handle_errors
    def get_dividend_payout_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ):
        """
        Calculate the Dividend payout ratio, a financial metric that measures the proportion
        of earnings paid out as dividends to shareholders.

        The payout ratio is a financial metric that helps investors assess the
        portion of a company's earnings that is being distributed to shareholders
        in the form of dividends. It's a valuable indicator for dividend investors as
        it indicates the sustainability of dividend payments and the company's
        approach to distributing profits.

        The formula is as follows:

        - Dividend Payout Ratio = Dividends Paid / Net Income

        The payout ratio shows how much room a company has to keep or raise its dividend. A
        ratio above 1 means the dividend is paid partly from reserves or borrowing, which
        cannot last unless profits were depressed by one-off charges. Because it uses net
        income, it is often checked against free cash flow, and for real estate investment
        trusts, whose net income is reduced by depreciation, it is not a useful measure.

        Also known as: payout ratio.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.DataFrame: Payout Ratio values.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["JNJ", "MCD", "TXN"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_dividend_payout_ratio()
        ```

        Which returns:

        |     |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:----|-------:|-------:|-------:|-------:|-------:|
        | JNJ | 0.5284 | 0.6511 | 0.3348 | 0.8405 | 0.4619 |
        | MCD | 0.5194 | 0.6747 | 0.5352 | 0.5922 | 0.5973 |
        | TXN | 0.5023 | 0.4933 | 0.7036 | 1.0042 | 1.0052 |
        """
        if trailing:
            payout_ratio = valuation_model.get_dividend_payout_ratio(
                self._sum_trailing(
                    self._cash_flow_statement.loc[:, "Dividends Paid", :], trailing
                ),
                self._sum_trailing(
                    self._income_statement.loc[:, "Net Income", :], trailing
                ),
            )
        else:
            payout_ratio = valuation_model.get_dividend_payout_ratio(
                self._cash_flow_statement.loc[:, "Dividends Paid", :],
                self._income_statement.loc[:, "Net Income", :],
            )

        return finalize_dataset(
            dataset=payout_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_reinvestment_rate(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ):
        """
        Calculate the Reinvestment rate, a financial metric that measures the proportion
        of earnings retained by the company.

        The reinvestment rate is a financial metric that helps investors assess the
        portion of a company's earnings that is being retained by the company
        for future growth. It's a valuable indicator for dividend investors as
        it indicates the sustainability of dividend payments and the company's
        approach to distributing profits.

        The formula is as follows:

        - Reinvestment Rate = 1 — Dividend Payout Ratio

        Also known as: plowback ratio, retention rate.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.DataFrame: Reinvestment Rate values.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["COST", "V", "NKE"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_reinvestment_rate()
        ```

        Which returns:

        |      |    2021 |   2022 |   2023 |    2024 |   2025 |
        |:-----|--------:|-------:|-------:|--------:|-------:|
        | COST | -0.148  | 0.7437 | 0.8012 | -0.2272 | 0.7305 |
        | NKE  |  0.6962 | 0.6032 | 0.6195 |  0.2855 | 0.2255 |
        | V    |  0.7727 | 0.7859 | 0.7828 |  0.7864 | 0.7666 |
        """
        if trailing:
            dividend_payout_ratio = self.get_dividend_payout_ratio(trailing=trailing)
            reinvestment_ratio = valuation_model.get_reinvestment_ratio(
                dividend_payout_ratio=dividend_payout_ratio,
            )
        else:
            dividend_payout_ratio = self.get_dividend_payout_ratio()
            reinvestment_ratio = valuation_model.get_reinvestment_ratio(
                dividend_payout_ratio=dividend_payout_ratio,
            )

        return finalize_dataset(
            dataset=reinvestment_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_tangible_asset_value(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ):
        """
        Calculate the tangible asset value, a financial metric that represents the book value
        of a company's equity after deducting goodwill and other intangible assets such as
        patents, brands and customer relationships: what remains of its tangible assets once
        all liabilities are paid.

        The formula is as follows:

        - Tangible Asset Value = Total Assets — Total Liabilities — Goodwill and Intangible Assets

        Also known as: net tangible assets.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.DataFrame: Tangible Asset Value values.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["PFE", "ABBV", "MRK"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_tangible_asset_value()
        ```

        Which returns:

        |      |        2021 |        2022 |        2023 |        2024 |        2025 |
        |:-----|------------:|------------:|------------:|------------:|------------:|
        | ABBV | -9.2894e+10 | -8.2308e+10 | -7.7506e+10 | -9.166e+10  | -9.1509e+10 |
        | MRK  | -5.94e+09   |  4.585e+09  | -1.573e+09  |  8.334e+09  |  4.402e+09  |
        | PFE  |  3.109e+09  |  1.172e+09  | -4.3395e+10 | -3.5442e+10 | -3.822e+10  |
        """
        # Goodwill and the other intangible assets are both deducted, so only the assets with a
        # physical or financial substance remain.
        tangible_asset_value = valuation_model.get_tangible_asset_value(
            self._mean_trailing(
                self._balance_sheet_statement.loc[:, "Total Assets", :], trailing
            ),
            self._mean_trailing(
                self._balance_sheet_statement.loc[:, "Total Liabilities", :], trailing
            ),
            self._mean_trailing(
                get_goodwill_and_intangible_assets(self._balance_sheet_statement),
                trailing,
            ),
        )

        return finalize_dataset(
            dataset=tangible_asset_value,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_net_current_asset_value(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ):
        """
        Calculate the net current asset value, a conservative liquidation-value metric introduced
        by Benjamin Graham that represents the total value of a company's current assets minus
        *all* of its liabilities (not just its current liabilities). It approximates what would
        be left for shareholders if the company were liquidated, paying off every liability using
        only the current (most liquid) assets and ignoring any value from fixed/non-current assets.

        The formula is as follows:

        - Net Current Asset Value = Total Current Assets — Total Liabilities

        Also known as: NCAV. Note that NCAV is related to, but distinct from, the Graham Number
        (`sqrt(22.5 * Earnings per Share * Book Value per Share)`, see `Toolkit.models.get_graham_number`)
        — both are Benjamin Graham value-investing metrics, but NCAV is a liquidation-value estimate
        while the Graham Number is a fair-value price estimate based on earnings and book value.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.DataFrame: Net Current Asset Value values.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["NVDA", "GOOGL", "META"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_net_current_asset_value()
        ```

        Which returns:

        |       |       2021 |        2022 |       2023 |       2024 |        2025 |
        |:------|-----------:|------------:|-----------:|-----------:|------------:|
        | GOOGL | 8.051e+10  |  5.5675e+10 | 5.2517e+10 | 3.8539e+10 |  2.6022e+10 |
        | META  | 2.5558e+10 | -4.65e+08   | 8.91e+09   | 6.628e+09  | -4.0056e+10 |
        | NVDA  | 1.1254e+10 |  3.992e+09  | 2.1595e+10 | 4.7852e+10 |  7.6095e+10 |
        """
        if trailing:
            net_current_asset_value = valuation_model.get_net_current_asset_value(
                self._mean_trailing(
                    self._balance_sheet_statement.loc[:, "Total Current Assets", :],
                    trailing,
                ),
                self._mean_trailing(
                    self._balance_sheet_statement.loc[:, "Total Liabilities", :],
                    trailing,
                ),
            )
        else:
            net_current_asset_value = valuation_model.get_net_current_asset_value(
                self._balance_sheet_statement.loc[:, "Total Current Assets", :],
                self._balance_sheet_statement.loc[:, "Total Liabilities", :],
            )

        return finalize_dataset(
            dataset=net_current_asset_value,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_ev_to_ebit(
        self,
        show_daily: bool = False,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ):
        """
        Calculate the enterprise value over earnings before interest and taxes (EBIT) ratio,
        which is a valuation metric that compares a company's total value (including debt and equity)
        relative to its earnings before interest and taxes.

        The formula is as follows:

        - Enterprise Value to EBIT Ratio = Enterprise Value / EBIT

        EBIT is the reported operating income, the same operating profit that EV to EBITDA
        (`get_ev_to_ebitda_ratio`) adds depreciation and amortization to (Damodaran), so the
        two multiples differ only by the depreciation and amortization. The enterprise value
        is taken at the end of each period (or on each day with show_daily), while EBIT is
        summed over the trailing periods.

        Args:
            show_daily (bool, optional): Whether to show daily values: the close of each trading day
                against the statement figures of the latest period that has ended on or before
                that day (also when it ended before the start date), so no figures are used before
                their period closes. Defaults to False.
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses the
                trailing twelve months (4 quarters) with quarterly data and single years with
                annual data.

        Returns:
            pd.DataFrame: Enterprise Value over EBIT values.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["SHW", "PPG", "ECL"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_ev_to_ebit()
        ```

        Which returns:

        |     |    2021 |    2022 |    2023 |    2024 |    2025 |
        |:----|--------:|--------:|--------:|--------:|--------:|
        | ECL | 45.5001 | 31.1699 | 29.9295 | 28.4083 | 28.7329 |
        | PPG | 28.3518 | 21.8704 | 19.9977 | 14.5487 | 13.4171 |
        | SHW | 39.551  | 24.8091 | 25.537  | 26.0717 | 24.5359 |
        """
        # Price-based ratios compare the price with a year of flows, so quarterly data
        # defaults to the trailing twelve months.
        trailing = self._get_valuation_trailing(trailing)

        enterprise_value = self.get_enterprise_value(
            diluted=diluted, show_daily=show_daily
        )

        # The flows are summed over the trailing periods first and only then spread over the
        # days, so a trailing window always covers periods and never days.
        ebit = self._sum_trailing(
            self._income_statement.loc[:, "Operating Income", :], trailing
        )

        if show_daily:
            ebit = map_period_data_to_daily_data(
                period_data=ebit,
                daily_dates=enterprise_value.index,
                quarterly=self._quarterly,
                period_end_dates=self._period_end_dates,
            )

        ev_to_ebit = valuation_model.get_ev_to_ebit(enterprise_value, ebit)

        return finalize_dataset(
            dataset=ev_to_ebit,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
            axis="rows" if show_daily else "columns",
            row_slice=show_daily,
        )

    @handle_portfolio
    @handle_errors
    def get_ev_to_free_cash_flow_ratio(
        self,
        show_daily: bool = False,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ):
        """
        Calculate the EV to free cash flow ratio, a valuation ratio that compares a
        company's enterprise value (EV) to its free cash flow.

        Unlike EV to Operating Cash Flow, this ratio nets out capital expenditures,
        giving a valuation multiple based on the cash actually available to all
        capital providers after reinvestment in the business.

        The formula is as follows:

        - EV to Free Cash Flow Ratio = Enterprise Value / Free Cash Flow

        Also known as: EV/FCF.

        Args:
            show_daily (bool, optional): Whether to show daily values: the close of each trading day
                against the statement figures of the latest period that has ended on or before
                that day (also when it ended before the start date), so no figures are used before
                their period closes. Defaults to False.
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses the
                trailing twelve months (4 quarters) with quarterly data and single years with
                annual data.

        Returns:
            pd.DataFrame: EV to free cash flow ratio values.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["V", "MA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_ev_to_free_cash_flow_ratio()
        ```

        Which returns:

        |    |    2021 |    2022 |    2023 |    2024 |    2025 |
        |:---|--------:|--------:|--------:|--------:|--------:|
        | MA | 41.9184 | 32.0227 | 35.3215 | 34.7836 | 31.69   |
        | V  | 33.0331 | 25.1707 | 27.6636 | 34.662  | 32.1372 |
        """
        # Price-based ratios compare the price with a year of flows, so quarterly data
        # defaults to the trailing twelve months.
        trailing = self._get_valuation_trailing(trailing)

        enterprise_value = self.get_enterprise_value(
            diluted=diluted, show_daily=show_daily
        )

        # The flows are summed over the trailing periods first and only then spread over the
        # days, so a trailing window always covers periods and never days.
        free_cash_flow = self._sum_trailing(
            self._cash_flow_statement.loc[:, "Free Cash Flow", :], trailing
        )

        if show_daily:
            free_cash_flow = map_period_data_to_daily_data(
                period_data=free_cash_flow,
                daily_dates=enterprise_value.index,
                quarterly=self._quarterly,
                period_end_dates=self._period_end_dates,
            )

        ev_to_free_cash_flow_ratio = valuation_model.get_ev_to_free_cash_flow_ratio(
            enterprise_value, free_cash_flow
        )

        return finalize_dataset(
            dataset=ev_to_free_cash_flow_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
            axis="rows" if show_daily else "columns",
            row_slice=show_daily,
        )

    @handle_portfolio
    @handle_errors
    def get_buyback_yield(
        self,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ):
        """
        Calculate the buyback yield, a valuation ratio that measures the net amount
        of common stock repurchased (net of new shares issued) relative to the
        company's market capitalization.

        A positive buyback yield means the company is a net repurchaser of its own
        stock (shareholder-friendly), while a negative buyback yield means the
        company is a net issuer of new shares (dilutive).

        The formula is as follows:

        - Buyback Yield = -(Common Stock Purchased + Common Stock Issued) / Market Capitalization

        The buyback yield is net of shares issued, which matters for companies that
        pay much of their staff in shares: their gross buybacks can be large while
        the net yield is small, because the buybacks mainly offset the dilution from
        stock-based compensation. Buybacks are more flexible than dividends and vary
        with the share price and the cash available, so the yield is best read over
        several years and added to the dividend yield (`get_shareholder_yield`).

        Also known as: net repurchase yield.

        Args:
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses the
                trailing twelve months (4 quarters) with quarterly data and single years with
                annual data.

        Returns:
            pd.DataFrame: Buyback yield values.

        Notes:
        - The method retrieves historical data and calculates the buyback yield for each
        asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AZO", "ORLY", "AAPL"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_buyback_yield()
        ```

        Which returns:

        |      |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----|-------:|-------:|-------:|-------:|-------:|
        | AAPL | 0.0283 | 0.0421 | 0.0255 | 0.0246 | 0.0222 |
        | AZO  | 0.0668 | 0.083  | 0.0712 | 0.052  | 0.024  |
        | ORLY | 0.0511 | 0.0613 | 0.0554 | 0.028  | 0.0258 |
        """
        # A yield compares the market cap with a year of buybacks, so quarterly data defaults to
        # the trailing twelve months.
        trailing = self._get_valuation_trailing(trailing)

        market_cap = self.get_market_cap(diluted=diluted)

        buyback_yield = valuation_model.get_buyback_yield(
            self._sum_trailing(
                self._cash_flow_statement.loc[:, "Common Stock Purchased", :], trailing
            ),
            self._sum_trailing(
                self._cash_flow_statement.loc[:, "Common Stock Issued", :], trailing
            ),
            market_cap,
        )

        return finalize_dataset(
            dataset=buyback_yield,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_shareholder_yield(
        self,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ):
        """
        Calculate the total shareholder yield, a valuation ratio that combines the
        dividend yield and the buyback yield to measure the total cash returned to
        shareholders relative to the company's market capitalization.

        The formula is as follows:

        - Shareholder Yield = Dividend Yield + Buyback Yield

        The buyback yield is net of the shares issued, so the shareholder yield is the net
        payout yield of Boudoukh, Michaely, Richardson and Roberts (2007): dividends plus
        repurchases minus issuance, relative to the market capitalization, which predicts
        returns better than the dividend yield alone.

        Also known as: total shareholder yield, total return of capital, net payout yield.

        Args:
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses the
                trailing twelve months (4 quarters) with quarterly data and single years with
                annual data.

        Returns:
            pd.DataFrame: Shareholder yield values.

        Notes:
        - Boudoukh, J., Michaely, R., Richardson, M. and Roberts, M. R., "On the Importance
        of Measuring Payout Yield: Implications for Empirical Asset Pricing," Journal of
        Finance 62(2), 2007, pp. 877-915.
        - The method retrieves historical data and calculates the shareholder yield for each
        asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["HD", "LOW"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_shareholder_yield()
        ```

        Which returns:

        |     |   2021 |   2022 |   2023 |   2024 |
        |:----|-------:|-------:|-------:|-------:|
        | HD  | 0.0489 | 0.0439 | 0.0461 | 0.0238 |
        | LOW | 0.0829 | 0.1309 | 0.0665 | 0.0472 |
        """
        # Both yields default to the trailing twelve months on quarterly data.
        trailing = self._get_valuation_trailing(trailing)

        # Unrounded, so that the sum is not off by the rounding of both parts.
        dividend_yield = self.get_dividend_yield(trailing=trailing, rounding=15)
        buyback_yield = self.get_buyback_yield(
            diluted=diluted, trailing=trailing, rounding=15
        )

        dividend_yield_columns = [
            column
            for column in dividend_yield.columns
            if column in buyback_yield.columns
        ]

        shareholder_yield = valuation_model.get_shareholder_yield(
            dividend_yield.loc[:, dividend_yield_columns],
            buyback_yield.loc[:, dividend_yield_columns],
        )

        return finalize_dataset(
            dataset=shareholder_yield,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_sbc_adjusted_free_cash_flow(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ):
        """
        Calculate the stock-based compensation (SBC) adjusted free cash flow, which
        deducts non-cash SBC expenses from free cash flow to give a more conservative
        view of the cash actually available to shareholders.

        Free cash flow already excludes SBC as a cash expense (it is added back in
        the cash flow from operations), which can overstate the cash available to
        shareholders. Subtracting SBC treats it as if it were a real cash cost, which
        is a common quality-of-earnings adjustment, especially for companies that
        rely heavily on equity compensation.

        The formula is as follows:

        - SBC-Adjusted Free Cash Flow = Free Cash Flow - Stock Based Compensation

        Also known as: SBC-adjusted FCF.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (income and cash flow
                statement items) over, with balance sheet items averaged over the same number of
                period ends. E.g. 4 with quarterly data gives the trailing twelve months (TTM) and 3
                with annual data a three-year cumulative figure. Defaults to None, which uses
                single-period flows (per quarter and not annualized with quarterly data).

        Returns:
            pd.DataFrame: SBC-adjusted free cash flow values.

        Notes:
        - The method retrieves historical data and calculates the SBC-adjusted free cash flow
        for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["CRM", "ADBE", "NOW"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_sbc_adjusted_free_cash_flow()
        ```

        Which returns:

        |      |      2021 |      2022 |      2023 |      2024 |       2025 |
        |:-----|----------:|----------:|----------:|----------:|-----------:|
        | ADBE | 5.824e+09 | 5.956e+09 | 5.224e+09 | 6.04e+09  | 7.91e+09   |
        | CRM  | 2.504e+09 | 3.034e+09 | 6.711e+09 | 9.251e+09 | 1.0893e+10 |
        | NOW  | 6.61e+08  | 7.72e+08  | 1.1e+09   | 1.669e+09 | 2.621e+09  |
        """
        if trailing:
            sbc_adjusted_free_cash_flow = (
                valuation_model.get_sbc_adjusted_free_cash_flow(
                    self._sum_trailing(
                        self._cash_flow_statement.loc[:, "Free Cash Flow", :], trailing
                    ),
                    self._sum_trailing(
                        self._cash_flow_statement.loc[:, "Stock Based Compensation", :],
                        trailing,
                    ),
                )
            )
        else:
            sbc_adjusted_free_cash_flow = (
                valuation_model.get_sbc_adjusted_free_cash_flow(
                    self._cash_flow_statement.loc[:, "Free Cash Flow", :],
                    self._cash_flow_statement.loc[:, "Stock Based Compensation", :],
                )
            )

        return finalize_dataset(
            dataset=sbc_adjusted_free_cash_flow,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_free_cash_flow_payout_ratio(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the free cash flow payout ratio, the share of free cash flow paid out as
        dividends.

        The dividend payout ratio (`get_dividend_payout_ratio`) compares the dividends with
        the net income, but dividends are paid from cash. A free cash flow payout ratio well
        below 1 leaves room for buybacks, debt repayment or a higher dividend; a ratio
        persistently above 1 means the dividend is funded with cash reserves or new debt and
        is at risk of a cut.

        The formula is as follows:

        - Free Cash Flow Payout Ratio = |Dividends Paid| / Free Cash Flow

        Dividends are reported as a negative cash outflow, so their magnitude is used. A free
        cash flow of zero or less gives NaN.

        Also known as: FCF payout ratio, dividends to free cash flow, cash dividend payout ratio, FCF dividend payout.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum flows (cash flow statement
                items) over. E.g. 4 with quarterly data gives the trailing twelve months (TTM)
                and 3 with annual data a three-year cumulative figure. Defaults to None, which
                uses single-period flows.

        Returns:
            pd.DataFrame: Free cash flow payout ratio values.

        Notes:
        - Damodaran, A., "Investment Valuation," 3rd edition, Wiley, 2012 (dividends and
        free cash flow to equity).
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["KO", "PFE", "VZ"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_free_cash_flow_payout_ratio()
        ```

        Which returns:

        |     |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:----|-------:|-------:|-------:|-------:|-------:|
        | KO  | 0.6442 | 0.7988 | 0.8158 | 1.7631 | 1.6577 |
        | PFE | 0.2922 | 0.3451 | 1.9293 | 0.9672 | 1.0766 |
        | VZ  | 0.5425 | 1.0388 | 0.5893 | 0.5945 | 0.5835 |
        """
        free_cash_flow_payout_ratio = valuation_model.get_free_cash_flow_payout_ratio(
            self._sum_trailing(
                self._cash_flow_statement.loc[:, "Dividends Paid", :], trailing
            ),
            self._sum_trailing(
                self._cash_flow_statement.loc[:, "Free Cash Flow", :], trailing
            ),
        )

        return finalize_dataset(
            dataset=free_cash_flow_payout_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_ev_to_invested_capital_ratio(
        self,
        show_daily: bool = False,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the enterprise value to invested capital ratio, the market value of a
        company's operations against the capital invested in them.

        Koller, Goedhart and Wessels show that a company creates value when its return on
        invested capital exceeds its cost of capital, and EV to invested capital is how the
        market prices that: a ratio above 1 means the market expects the company to earn
        more than its cost of capital on the capital invested, a ratio below 1 that it
        expects less. Read together with the return on invested capital
        (`get_return_on_invested_capital`), a high ROIC with a low EV/IC points to value.

        The formula is as follows:

        - Invested Capital = Total Equity + Total Debt - Cash and Short Term Investments
        - EV to Invested Capital = Enterprise Value / Invested Capital

        The invested capital is measured from the financing side, with the same items as the
        enterprise value (`get_enterprise_value`): the equity includes minority interest and
        preferred stock, and the cash and short-term investments are deducted from both. It
        is a point in time at the end of each period, as the enterprise value is. An invested
        capital of zero or less gives NaN.

        Also known as: EV/IC, enterprise value to invested capital, firm market-to-book.

        Args:
            show_daily (bool, optional): Whether to show daily values: the close of each trading day
                against the balance sheet of the latest period that has ended on or before that
                day (also when it ended before the start date), so no figures are used before
                their period closes. Defaults to False.
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of period ends to average the balance sheet
                items over, as for the price to book ratio. Defaults to None, which uses the
                balance sheet at the end of each period.

        Returns:
            pd.DataFrame: EV to invested capital values.

        Notes:
        - Koller, T., Goedhart, M. and Wessels, D., "Valuation: Measuring and Managing the
        Value of Companies," 7th edition, Wiley, 2020 (invested capital and the
        EV-to-invested-capital multiple).
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "KO", "F"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_ev_to_invested_capital_ratio()
        ```

        Which returns:

        |      |    2021 |    2022 |    2023 |    2024 |    2025 |
        |:-----|--------:|--------:|--------:|--------:|--------:|
        | AAPL | 22.4029 | 16.3546 | 24.9497 | 35.2982 | 31.4798 |
        | F    |  1.2546 |  1.0247 |  1.0422 |  0.97   |  1.0985 |
        | KO   |  5.1209 |  5.6096 |  5.0137 |  5.2427 |  4.9835 |
        """
        enterprise_value = self.get_enterprise_value(
            diluted=diluted, show_daily=show_daily
        )

        balance_items = {
            "total_equity": self._mean_trailing(
                self._balance_sheet_statement.loc[:, "Total Equity", :], trailing
            ),
            "total_debt": self._mean_trailing(
                self._balance_sheet_statement.loc[:, "Total Debt", :], trailing
            ),
            "cash_and_short_term_investments": self._mean_trailing(
                self._get_cash_and_short_term_investments(), trailing
            ),
        }

        if show_daily:
            balance_items = {
                item: map_period_data_to_daily_data(
                    period_data=values,
                    daily_dates=enterprise_value.index,
                    quarterly=self._quarterly,
                    period_end_dates=self._period_end_dates,
                )
                for item, values in balance_items.items()
            }

        ev_to_invested_capital = valuation_model.get_ev_to_invested_capital_ratio(
            enterprise_value, **balance_items
        )

        return finalize_dataset(
            dataset=ev_to_invested_capital,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
            axis="rows" if show_daily else "columns",
            row_slice=show_daily,
        )

    @handle_portfolio
    @handle_errors
    def get_price_to_tangible_book_ratio(
        self,
        show_daily: bool = False,
        diluted: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the price to tangible book ratio, the market value of the common equity
        against its book value without goodwill and other intangible assets.

        Goodwill and acquired intangibles cannot be sold separately and are written off
        first when a company gets into trouble, so the tangible book value is the more
        conservative measure of what the shareholders own. The ratio is the usual valuation
        measure for banks and insurers, whose assets are mostly financial, and shows for
        acquisitive companies how much of the price is paid for past acquisitions.

        The formula is as follows:

        - Tangible Book Value = Total Shareholder Equity - Preferred Stock - Goodwill and Intangible Assets
        - Price to Tangible Book = Market Capitalization / Tangible Book Value

        The market capitalization uses the same close and shares as the price to book ratio
        (`get_price_to_book_ratio`), so both differ only by the goodwill and intangibles. A
        tangible book value of zero or less (common after large acquisitions or buybacks)
        gives NaN.

        Also known as: P/TBV, price to tangible book value, price to tangible equity, P/TB.

        Args:
            show_daily (bool, optional): Whether to show daily values: the close of each trading day
                against the balance sheet of the latest period that has ended on or before that
                day (also when it ended before the start date), so no figures are used before
                their period closes. Defaults to False.
            diluted (bool, optional): Whether to use diluted shares in the calculation. Defaults to True.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of period ends to average the balance sheet
                items over, as for the price to book ratio. Defaults to None, which uses the
                balance sheet at the end of each period.

        Returns:
            pd.DataFrame: Price to tangible book values.

        Notes:
        - CFA Institute, "Equity Valuation: Applications and Processes," CFA Program
        Curriculum (price to book value: adjusting book value for intangible assets).
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["JPM", "C", "WFC"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_price_to_tangible_book_ratio()
        ```

        Which returns:

        |     |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:----|-------:|-------:|-------:|-------:|-------:|
        | C   | 0.7874 | 0.562  | 0.616  | 0.8017 | 1.2871 |
        | JPM | 2.3656 | 1.9517 | 2.1204 | 2.6528 | 3.2247 |
        | WFC | 1.467  | 1.2662 | 1.3809 | 1.9093 | 2.251  |
        """
        market_cap = self.get_market_cap(diluted=diluted, show_daily=show_daily)

        balance_items = {
            "total_shareholder_equity": self._mean_trailing(
                self._balance_sheet_statement.loc[:, "Total Shareholder Equity", :],
                trailing,
            ),
            "preferred_stock": self._mean_trailing(
                self._balance_sheet_statement.loc[:, "Preferred Stock", :], trailing
            ),
            "goodwill_and_intangible_assets": self._mean_trailing(
                get_goodwill_and_intangible_assets(self._balance_sheet_statement),
                trailing,
            ),
        }

        if show_daily:
            balance_items = {
                item: map_period_data_to_daily_data(
                    period_data=values,
                    daily_dates=market_cap.index,
                    quarterly=self._quarterly,
                    period_end_dates=self._period_end_dates,
                )
                for item, values in balance_items.items()
            }

        price_to_tangible_book = valuation_model.get_price_to_tangible_book_ratio(
            market_cap, **balance_items
        )

        return finalize_dataset(
            dataset=price_to_tangible_book,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
            axis="rows" if show_daily else "columns",
            row_slice=show_daily,
        )

    @handle_portfolio
    @handle_errors
    def get_net_share_issuance(
        self,
        diluted: bool = False,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the net share issuance of Pontiff and Woodgate (2008): the log change in
        the split-adjusted number of shares over a year.

        Pontiff and Woodgate show that companies that issue shares earn lower returns over
        the following year and companies that buy back shares higher returns, and that the
        share issuance predicts returns more strongly than size, book-to-market or momentum
        in the years after 1970. The measure is positive for a net issuer of shares and
        negative for a company that buys back more shares than it issues.

        The formula is as follows:

        - Net Share Issuance = ln(Shares / Shares a Year Earlier)

        The shares are the weighted average shares of the period, which the statements
        report adjusted for later stock splits, so only issuance and repurchases change them.
        The paper takes the shares outstanding at a date; the weighted average moves with
        them but spreads a change over the period it occurs in. The shares a year earlier
        are those of the year before with annual data and four quarters back with quarterly
        data. Shares of zero or less give NaN.

        Also known as: share issuance, ISSUE, net stock issues, change in shares outstanding.

        Args:
            diluted (bool, optional): Whether to use the diluted weighted average shares instead of
                the basic ones. Defaults to False, as the dilution from options and convertibles
                moves with the share price rather than with issuance.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods the shares are compared over.
                E.g. 5 with annual data gives the five-year issuance. Defaults to None, which
                uses a year: 4 quarters with quarterly data and 1 year with annual data.

        Returns:
            pd.DataFrame: Net share issuance values.

        Notes:
        - Pontiff, J. and A. Woodgate, "Share Issuance and Cross-Sectional Returns," Journal
        of Finance 63(2), 2008, pp. 921-945.
        - The first period has no shares a year earlier and is NaN, so set an earlier
        start_date to obtain it.
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "TSLA", "META", "AMZN"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_net_share_issuance()
        ```

        Which returns:

        |      |    2021 |    2022 |    2023 |    2024 |    2025 |
        |:-----|--------:|--------:|--------:|--------:|--------:|
        | AAPL | -0.0382 | -0.0295 | -0.0295 | -0.0258 | -0.0261 |
        | AMZN |  0.0119 |  0.0068 |  0.0112 |  0.0163 |  0.0173 |
        | META | -0.0127 | -0.0465 | -0.043  | -0.0157 | -0.0051 |
        | TSLA |  0.0556 |  0.0562 |  0.0138 |  0.0074 |  0.0087 |
        """
        # A year back by default: four quarters on quarterly data, so the change is year over year.
        window = self._get_valuation_trailing(trailing) or 1

        shares = (
            self._income_statement.loc[:, "Weighted Average Shares Diluted", :]
            if diluted
            else self._income_statement.loc[:, "Weighted Average Shares", :]
        )

        net_share_issuance = valuation_model.get_net_share_issuance(
            shares, shares.shift(window, axis=1)
        )

        return finalize_dataset(
            dataset=net_share_issuance,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_composite_equity_issuance(
        self,
        diluted: bool = False,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the composite equity issuance of Daniel and Titman (2006): the growth in
        market capitalization over five years that the stock return does not explain.

        A company's market capitalization grows with its stock return and with the equity
        it issues, so what remains after the return is the net equity issued: seasoned
        offerings, shares issued to employees and in stock-financed acquisitions, minus
        buybacks and dividends. Daniel and Titman show that this composite issuance
        predicts returns negatively and, unlike past returns, independently of the
        book-to-market ratio: companies that issue equity tend to do so when their shares
        are expensive, and those that return cash when they are cheap.

        The formula is as follows:

        - Composite Equity Issuance = ln(Market Cap / Market Cap Five Years Earlier)
          - ln(Total Return over the Five Years)

        The total return is the log change in the price adjusted for dividends and splits,
        so a company that pays dividends without issuing shares has a negative value. The
        market cap is the close (adjusted for splits only) at the end of each period times
        the weighted average shares, as in `get_market_cap`, and the periods follow the
        calendar. The window is five years, as in the paper: 5 years with annual data and
        20 quarters with quarterly data.

        Also known as: CEI, iota, composite issuance, composite share issuance.

        Args:
            diluted (bool, optional): Whether to use the diluted weighted average shares instead of
                the basic ones. Defaults to False, as the dilution from options and convertibles
                moves with the share price rather than with issuance.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods of the window. E.g. 1 with annual
                data gives the issuance over a year. Defaults to None, which uses five years:
                20 quarters with quarterly data and 5 years with annual data.

        Returns:
            pd.DataFrame: Composite equity issuance values.

        Notes:
        - Daniel, K. and S. Titman, "Market Reactions to Tangible and Intangible
        Information," Journal of Finance 61(4), 2006, pp. 1605-1643.
        - The prices and statements reach about a year before the start date, so with the
        five-year window the first years are NaN; set an earlier start_date to obtain them.
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "TSLA", "META", "AMZN"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2016-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_composite_equity_issuance()
        ```

        Which returns:

        |      |   2016 |   2017 |   2018 |    2019 |    2020 |    2021 |    2022 |    2023 |    2024 |    2025 |
        |:-----|-------:|-------:|-------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|
        | AAPL |    nan |    nan |    nan | -0.361  | -0.3589 | -0.331  | -0.3028 | -0.2715 | -0.2165 | -0.176  |
        | AMZN |    nan |    nan |    nan |  0.067  |  0.0683 |  0.0653 |  0.0595 |  0.0563 |  0.0583 |  0.0635 |
        | META |    nan |    nan |    nan |  0.074  |  0.0045 | -0.0355 | -0.0945 | -0.1158 | -0.1227 | -0.13   |
        | TSLA |    nan |    nan |    nan |  0.3515 |  0.3754 |  0.3133 |  0.2302 |  0.2152 |  0.1858 |  0.1417 |
        """
        window = trailing or (20 if self._quarterly else 5)

        shares = (
            self._income_statement.loc[:, "Weighted Average Shares Diluted", :]
            if diluted
            else self._income_statement.loc[:, "Weighted Average Shares", :]
        )

        # The period prices from a year before the start date onwards, as the statements are,
        # so the first window can start before the start date.
        prices = self._historical["quarterly" if self._quarterly else "yearly"]

        def period_prices(column: str) -> pd.DataFrame:
            return (
                prices[column]
                .reindex(columns=self._tickers_without_portfolio)
                .T.reindex(columns=shares.columns)
            )

        market_cap = valuation_model.get_market_cap(period_prices("Close"), shares)
        total_return_price = period_prices("Adj Close")

        composite_equity_issuance = valuation_model.get_composite_equity_issuance(
            market_cap,
            market_cap.shift(window, axis=1),
            total_return_price,
            total_return_price.shift(window, axis=1),
        )

        return finalize_dataset(
            dataset=composite_equity_issuance,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_portfolio
    @handle_errors
    def get_net_external_financing(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        compound: bool = False,
        standardize: bool = False,
        trailing: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the net external financing of Bradshaw, Richardson and Sloan (2006): the
        cash a company raises from shareholders and lenders, net of what it returns to them,
        scaled by its average total assets.

        Bradshaw, Richardson and Sloan show that the net external financing predicts returns
        negatively: companies that raise the most cash from equity and debt investors earn
        the lowest returns afterwards, and the relation holds for equity and for debt
        financing separately. Analysts are also most optimistic about the companies that
        raise the most financing, which fits the view that companies raise capital when
        investors overvalue them.

        The formula is as follows:

        - Net External Financing = (Net Equity Financing + Net Debt Financing) / Average
          Total Assets
        - Net Equity Financing = Common and Preferred Stock Issued - Stock Repurchased -
          Dividends Paid
        - Net Debt Financing = Long Term Debt Issued - Long Term Debt Repaid + Change in
          Short Term Debt

        The flows are those of the cash flow statement and the average total assets the
        average of the total assets at the start and the end of the window. It is an annual
        measure, so with quarterly data the flows are those of the trailing twelve months
        and the total assets are averaged over the start and end of that year by default.
        Average total assets of zero or less give NaN.

        Also known as: XFIN, ΔXFIN, net financing, external financing.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            compound (bool, optional): Whether the growth is the compound annual growth rate over
                the lag (e.g. lag=5 for the five-year CAGR) instead of the total change. Defaults to False.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            trailing (int, optional): The number of periods to sum the cash flow statement
                items over, with the total assets averaged over the start and the end of that
                window. Defaults to None, which uses the trailing twelve months (4 quarters)
                with quarterly data and single years with annual data.

        Returns:
            pd.DataFrame: Net external financing values.

        Notes:
        - Bradshaw, M. T., S. A. Richardson and R. G. Sloan, "The Relation Between Corporate
        Financing Activities, Analysts' Forecasts and Stock Returns," Journal of Accounting
        and Economics 42(1-2), 2006, pp. 53-85.
        - The first period has no total assets at the start of the window and is NaN, so set
        an earlier start_date to obtain it.
        - If `growth` is set to True, the method calculates the growth of the ratio values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "KO", "XOM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_net_external_financing()
        ```

        Which returns:

        |      |    2021 |    2022 |    2023 |    2024 |    2025 |
        |:-----|--------:|--------:|--------:|--------:|--------:|
        | AAPL | -0.2568 | -0.2966 | -0.2906 | -0.3238 | -0.3165 |
        | KO   | -0.0708 | -0.0979 | -0.0824 | -0.0694 | -0.0896 |
        | MSFT | -0.1447 | -0.1661 | -0.1105 | -0.0789 | -0.0874 |
        | XOM  | -0.1034 | -0.1054 | -0.0884 | -0.0997 | -0.083  |
        """
        trailing = self._get_valuation_trailing(trailing)

        def flow(row: str) -> pd.DataFrame:
            return self._sum_trailing(
                self._cash_flow_statement.loc[:, row, :], trailing
            )

        # The issuance and repurchases of common and preferred stock and of long and short
        # term debt are taken line by line: the aggregate net issuance lines of the data
        # provider do not always add them up.
        total_assets = self._balance_sheet_statement.loc[:, "Total Assets", :]

        net_external_financing = valuation_model.get_net_external_financing(
            flow("Common Stock Issued")
            + flow("Common Stock Purchased")
            + flow("Net Preferred Stock Issued"),
            flow("Dividends Paid"),
            flow("Long Term Debt Issued") + flow("Short Term Debt Issued"),
            (total_assets + total_assets.shift(trailing or 1, axis=1)) / 2,
        )

        return finalize_dataset(
            dataset=net_external_financing,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            compound=compound,
            rounding=rounding,
            standardize=standardize,
        )

    @handle_errors
    def get_factor_analysis(
        self,
        metrics: str | list[str] = "valuation",
        period: str = "monthly",
        horizons: list[int] | int | None = None,
        quantiles: int = 5,
        periods_apart: int = 1,
        detail: str | None = None,
        column: str | None = None,
        rounding: int | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Factor analysis (predictive power) of the ratios: whether the tickers that score high
        on a ratio go on to return more than those that score low. Per period, the information
        coefficient (IC) is the Spearman rank correlation between the ratio of the tickers at
        the end of the period and their return over the following horizon: 1 means the ratio
        ranks the later returns perfectly, -1 in reverse and 0 that it says nothing about them.
        The IC measures skill in Grinold and Kahn's fundamental law of active management,
        IR ≈ IC × √breadth: a small but consistent IC across many tickers and periods gives a
        high information ratio.

        The summary has a row per ratio and horizon with the mean IC, its standard deviation,
        the IC information ratio (mean divided by standard deviation), the t-statistic of the
        mean with its p-value, the share of periods with a positive IC, the number of periods,
        the top minus bottom spread (the mean forward return of the highest quantile minus
        that of the lowest), the mean turnover of the top and bottom quantiles and the mean
        rank autocorrelation of the ratio from one period to the next. With detail, it returns
        one of those parts per period (or per quantile) for a single ratio.

        The ratios are calculated daily from the closing price where they can be (show_daily)
        and otherwise per statement period. Statement figures apply from the end of each fiscal
        period, or with point_in_time from the day they were filed, never earlier, so no day
        uses figures that were not known yet. The returns use the dividend and split adjusted
        close and start at the close of the period the ratio belongs to. Each ticker keeps its
        own history: a ticker without a ratio or a return in a period is left out of that
        period only.

        Args:
            metrics (str | list[str], optional): A ratio method name, e.g.
                "get_price_to_earnings_ratio", a list of them, or a collection: "valuation",
                "profitability", "solvency", "liquidity", "efficiency" or "all". Defaults to
                "valuation", the price and enterprise value multiples, yields and the market
                cap. Per-share figures and currency amounts are not in the collections, as
                their level depends on the share count or company size.
            period (str, optional): The frequency at which the ratios are evaluated: daily,
                weekly, monthly, quarterly or yearly. Defaults to "monthly".
            horizons (list[int] | int | None, optional): The number of periods the forward
                returns cover. Defaults to None, which uses about a period, a quarter and a
                year ahead (1, 3 and 12 months for monthly).
            quantiles (int, optional): The number of quantiles the tickers are sorted into per
                period. Defaults to 5.
            periods_apart (int, optional): The number of periods between the dates the
                quantile turnover and the rank autocorrelation compare. Defaults to 1.
            detail (str | None, optional): Returns one part of the analysis of a single ratio
                instead of the summary: "information_coefficient" (per period and horizon),
                "quantile_returns" (the mean forward return per quantile and horizon),
                "quantile_turnover" (per period and quantile) or "rank_autocorrelation" (per
                period). Defaults to None.
            column (str | None, optional): Not used by the ratios, which have one value per
                ticker; it exists for the modules whose metrics have several outputs. Defaults
                to None.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            **kwargs: Parameters passed to the ratios that take them, e.g. trailing=4 or
                diluted=False.

        Returns:
            pd.DataFrame: The summary per ratio and horizon, or with detail that part of the
            analysis of the ratio.

        Notes:
            - The IC is reported as measured: a low P/E that goes on to outperform gives a
              negative IC for the P/E. Read the sign as the direction of the relationship.
            - The cross-section is the tickers of the Toolkit, so use at least ten and
              preferably twenty or more for a meaningful rank correlation and quantiles.
            - A horizon longer than one period makes consecutive returns overlap, which
              inflates the t-statistic; judge those horizons by their mean IC and IC
              information ratio.
            - A ratio that returns no data (e.g. one missing a statement line item), or that
              fewer than three tickers have in every period (e.g. the preferred dividend
              coverage ratio), is left out of a collection with a warning.
            - Based on alphalens (Quantopian) and on Grinold and Kahn (2000), Active Portfolio
              Management.

        Also known as: predictive power, information coefficient, factor testing, alpha signal

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "JNJ",
             "XOM", "PG", "KO", "WMT", "V", "UNH", "HD", "CVX"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2016-01-01",
            end_date="2025-12-31",
        )

        toolkit.ratios.get_factor_analysis(
            ["get_price_to_earnings_ratio", "get_price_to_book_ratio", "get_return_on_equity"],
            horizons=12,
        )
        ```

        Which returns:
        |                                              |   Mean IC |   IC Standard Deviation |   IC Information Ratio |   t-Statistic |   p-Value |   Positive Share |   Observations |   Top Minus Bottom |   Quantile Turnover |   Rank Autocorrelation |
        |:---------------------------------------------|----------:|------------------------:|-----------------------:|--------------:|----------:|-----------------:|---------------:|-------------------:|--------------------:|-----------------------:|
        | ('get_price_to_earnings_ratio', '12 Months') |    0.0505 |                  0.2917 |                 0.173  |        1.7981 |    0.075  |           0.5741 |            108 |             0.0854 |              0.0872 |                 0.9499 |
        | ('get_price_to_book_ratio', '12 Months')     |    0.1163 |                  0.2959 |                 0.393  |        4.0845 |    0.0001 |           0.6759 |            108 |             0.1699 |              0.0476 |                 0.978  |
        | ('get_return_on_equity', '12 Months')        |    0.0423 |                  0.3551 |                 0.1191 |        1.1915 |    0.2363 |           0.5    |            100 |             0.065  |              0.0517 |                 0.9671 |
        """
        factors = factor_model.get_daily_factors(
            self,
            factor_model.select_metrics(metrics, FACTOR_COLLECTIONS),
            kwargs,
            {"show_daily": True},
            self._tickers,
            column,
            self._period_end_dates,
        )
        result = factor_model.get_factor_analysis(
            factors,
            self._daily_historical_data["Adj Close"],
            period,
            horizons,
            quantiles,
            periods_apart,
            detail,
            self._start_date,
            self._end_date,
        )

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

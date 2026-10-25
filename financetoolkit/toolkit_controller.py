"""Toolkit Module"""

__docformat__ = "google"


import os
import warnings
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

import pandas as pd

from financetoolkit import currencies_model
from financetoolkit.cache import cache_controller, policy_model, ticker_model
from financetoolkit.discovery.discovery_model import (
    search_crypto_news as _search_crypto_news,
    search_forex_news as _search_forex_news,
    search_press_releases as _search_press_releases,
    search_stock_news as _search_stock_news,
)
from financetoolkit.fmp_model import (
    determine_subscription_plan as _determine_subscription_plan,
    get_analyst_estimates as _get_analyst_estimates,
    get_commitment_of_traders as _get_commitment_of_traders,
    get_company_notes as _get_company_notes,
    get_dividend_calendar as _get_dividend_calendar,
    get_earnings_calendar as _get_earnings_calendar,
    get_earnings_call_transcripts as _get_earnings_call_transcripts,
    get_employee_count as _get_employee_count,
    get_esg_scores as _get_esg_scores,
    get_etf_country_weightings as _get_etf_country_weightings,
    get_etf_holdings as _get_etf_holdings,
    get_etf_information as _get_etf_information,
    get_etf_sector_weightings as _get_etf_sector_weightings,
    get_executive_compensation as _get_executive_compensation,
    get_executives as _get_executives,
    get_insider_trade_statistics as _get_insider_trade_statistics,
    get_market_risk_premium as _get_market_risk_premium,
    get_mergers_acquisitions as _get_mergers_acquisitions,
    get_profile as _get_profile,
    get_quote as _get_quote,
    get_rating as _get_rating,
    get_ratios_ttm as _get_ratios_ttm,
    get_revenue_segmentation as _get_revenue_segmentation,
    get_shares_float as _get_shares_float,
    get_stock_grades as _get_stock_grades,
    get_stock_splits as _get_stock_splits,
)
from financetoolkit.fundamentals_model import collect_financial_statements
from financetoolkit.helpers import LazyDict, determine_period_end_dates
from financetoolkit.historical_model import (
    calculate_period_average as _calculate_period_average,
    convert_daily_to_other_period as _convert_daily_to_other_period,
    get_historical_data as _get_historical_data,
    get_historical_statistics as _get_historical_statistics,
)
from financetoolkit.normalization_model import (
    get_normalization_files as _get_normalization_files,
    initialize_statements_and_normalization as _initialize_statements_and_normalization,
)
from financetoolkit.utilities import logger_model, validation_model
from financetoolkit.utilities.dataframe_model import filter_columns
from financetoolkit.utilities.statistics_model import apply_rounding, calculate_growth

if TYPE_CHECKING:
    # TYPE_CHECKING only: the module controllers are imported when first used, so that
    # importing the Finance Toolkit does not load every module and its dependencies.
    from financetoolkit.econometrics.econometrics_controller import Econometrics
    from financetoolkit.economics.economics_controller import Economics
    from financetoolkit.etf.etf_controller import ETF
    from financetoolkit.fixedincome.fixedincome_controller import FixedIncome
    from financetoolkit.models.models_controller import Models
    from financetoolkit.options.options_controller import Options
    from financetoolkit.performance.performance_controller import Performance
    from financetoolkit.portfolio.construction_controller import (
        PortfolioConstruction,
    )
    from financetoolkit.ratios.ratios_controller import Ratios
    from financetoolkit.risk.risk_controller import Risk
    from financetoolkit.technicals.technicals_controller import Technicals

# Displays messages, warnings and errors when the Finance Toolkit hits issues.
logger_model.setup_logger()
logger = logger_model.get_logger()

# Division by zero is normal in these calculations, not a bug.
warnings.filterwarnings("ignore", category=RuntimeWarning)

# pylint: disable=too-many-instance-attributes,too-many-lines,line-too-long,too-many-locals
# pylint: disable=too-many-function-args,too-many-public-methods
# ruff: noqa: E501

TICKER_LIMIT = 20

# Used as the Toolkit's default API key when set as an environment variable.
API_KEY: str = os.environ.get("FINANCIAL_MODELING_PREP_API_KEY", "")
FRED_API_KEY: str = os.environ.get("FRED_API_KEY", "")


class Toolkit:
    """
    The Finance Toolkit is an open-source toolkit in which
    all 500+ financial methods
    are written down in the most simplistic way allowing for complete transparency
    of the calculation method. This allows you to not have to rely on metrics
    from other providers and, given a financial statement, allow for efficient manual
    calculations. This leads to one uniform method of calculation being applied that
    is available and understood by everyone.
    """

    def __init__(
        self,
        tickers: list | str | None = None,
        api_key: str = API_KEY,
        start_date: str | None = None,
        end_date: str | None = None,
        quarterly: bool = False,
        use_cached_data: bool | str | cache_controller.Cache | None = None,
        risk_free_rate: str = "10y",
        benchmark_ticker: str | None = "SPY",
        enforce_source: str | None = None,
        historical: pd.DataFrame = pd.DataFrame(),
        balance: pd.DataFrame = pd.DataFrame(),
        income: pd.DataFrame = pd.DataFrame(),
        cash: pd.DataFrame = pd.DataFrame(),
        format_location: str = "",
        convert_currency: bool | None = None,
        reverse_dates: bool = True,
        intraday_period: str | None = None,
        rounding: int | None = 4,
        remove_invalid_tickers: bool = False,
        sleep_timer: bool | None = None,
        progress_bar: bool = True,
        fred_api_key: str = FRED_API_KEY,
        allow_stale_oecd_cache: bool = True,
    ):
        """
        Initializes a Toolkit object with a ticker or a list of tickers. The way the Toolkit is initialized
        will define how the data is collected. For example, if you enable the quarterly flag, you will
        be able to collect quarterly data. Next to that, you can define the start and end date to specify
        a specific range. Data retrieved from an external source (statements, prices, economic indicators and so on)
        is cached by default, so it is only retrieved again once it may have changed; what the Finance Toolkit
        calculates itself is never cached. Set use_cached_data to False to switch this off, or to a string, e.g.
        "datasets", to store the cache in a specific location.

        The cache keeps track of what it already holds per ticker and per date range, so changing a parameter
        does not throw the rest away. Widening the period only retrieves the years that were missing, adding a
        ticker only retrieves that ticker, and repeating a request retrieves nothing at all.

        It is good to note that the Finance Toolkit will always attempt to acquire data from Financial Modeling Prep
        if an API key is set. If this isn't the case, the data comes from Yahoo Finance. In case you have an API key
        set and the current plan doesn't allow for the data to be collected, the Toolkit will automatically switch to
        Yahoo Finance. You can disable this behaviour by setting the enforce_source variable to "FinancialModelingPrep".

        For more information on the capabilities of the Finance Toolkit see here: https://www.jeroenbouma.com/projects/financetoolkit

        Args:
            tickers (list | str | None): A string or a list of strings containing the company ticker(s). E.g. 'TSLA' or 'MSFT'.
            Find tickers on various websites or via the FinanceDatabase: https://github.com/JerBouma/financedatabase. Defaults to None.
            api_key (str): An API key from FinancialModelingPrep. Obtain one at https://www.jeroenbouma.com/fmp or leave it
            empty to use Yahoo Finance where possible. Defaults to the value of the FINANCIAL_MODELING_PREP_API_KEY
            environment variable if set, otherwise an empty string.
            start_date (str | None): A string containing the start date of the data. Needs to be formatted as YYYY-MM-DD.
            Defaults to 5 years/quarters back from today depending on the 'quarterly' flag.
            end_date (str | None): A string containing the end date of the data. Needs to be formatted as YYYY-MM-DD.
            Defaults to today.
            quarterly (bool): A boolean indicating whether to collect quarterly data. Defaults to False (yearly).
            Note that historical data can still be collected for any period and interval.
            use_cached_data (bool | str | None): Whether to cache the data retrieved from external sources. None or
            True uses the shared cache database in the user configuration directory, which is also the one the MCP
            server reads and writes, False retrieves everything every time and a string is the path to a dedicated
            cache folder or database file. Defaults to None, which caches unless the FINANCE_TOOLKIT_CACHE_ENABLED environment
            variable is set to 0.
            risk_free_rate (str): The risk-free rate identifier ('13w', '5y', '10y', '30y'). Based on US Treasury Yields.
            Used for calculations like Excess Returns. Defaults to "10y".
            benchmark_ticker (str | None): The benchmark ticker (e.g., 'SPY' for S&P 500). Used for comparative analysis
            (CAPM, Alpha, Beta). Defaults to "SPY". Set to None to disable benchmark comparison.
            enforce_source (str | None): Enforce data source ('FinancialModelingPrep' or 'YahooFinance').
            Defaults to None (uses FMP if api_key provided, otherwise YahooFinance, with fallback).
            historical (pd.DataFrame): Custom historical price data. See
            https://www.jeroenbouma.com/projects/financetoolkit/docs/external-datasets for how to supply your own data.
            Defaults to an empty DataFrame.
            balance (pd.DataFrame): Custom balance sheet data. See the external datasets guide linked above. Defaults to an empty DataFrame.
            income (pd.DataFrame): Custom income statement data. See the external datasets guide linked above. Defaults to an empty DataFrame.
            cash (pd.DataFrame): Custom cash flow statement data. See the external datasets guide linked above. Defaults to an empty DataFrame.
            format_location (str): Path to custom normalization files. Defaults to "".
            convert_currency (bool | None): Convert financial statements currency to match historical data currency.
            Important for cross-ticker comparison and calculations involving both data types.
            Defaults to None (True if FMP plan is Premium, False if Free). Can be overridden.
            reverse_dates (bool): Reverse the order of dates in financial statements (oldest first). Defaults to True.
            intraday_period (str | None): Intraday data interval ('1min', '5min', '15min', '30min', '1hour').
            Enables short-term analysis using Risk, Performance, and Technicals modules. Requires FMP Premium.
            Defaults to None (no intraday data).
            rounding (int | None): Number of decimal places for results. Defaults to 4.
            remove_invalid_tickers (bool): Remove tickers that fail data retrieval. Defaults to False.
            sleep_timer (bool | None): Enable sleep timer on FMP rate limit (requires Premium).
            Defaults to None (determined by FMP plan: True for Premium, False for Free).
            progress_bar (bool): Show progress bar for operations involving multiple tickers. Defaults to True.
            fred_api_key (str): A FRED API key used to retrieve ICE BofA bond index data via the fixedincome module
            (option-adjusted spread, effective yield, total return, yield to worst). Obtain a free key at
            https://fred.stlouisfed.org/docs/api/api_key.html. Can also be set via the FRED_API_KEY environment
            variable. Defaults to the value of FRED_API_KEY if set, otherwise an empty string.
            allow_stale_oecd_cache (bool): the OECD API (used by the economics module) enforces a hard rate
            limit (60 downloads/hour). When True, a rate-limited call falls back to the most recently cached
            successful response for that query instead of empty data -- opt-in, since served data may be
            stale. Independent of use_cached_data (which governs this Toolkit's own config/ticker cache, a
            different mechanism). Defaults to True.

        Fiscal periods and calendar periods:

            Financial statements are labelled with the calendar period in which the majority of the
            fiscal period falls, not with the company's own fiscal label and not with the calendar
            period its final day happens to sit in. A fiscal period is a span of time and the date a
            provider reports is only its last day, so labelling by that day alone would place NVIDIA's
            fiscal year February 2023 to January 2024 in calendar 2024 even though eleven of its twelve
            months are 2023.

            Concretely, a fiscal year ending in January through May is labelled with the preceding
            calendar year, and a fiscal quarter is labelled with the calendar quarter containing the
            month before its end. Both are the same majority rule at two frequencies, so a company's
            yearly and quarterly statements stay consistent with each other: NVIDIA's fiscal 2024 is
            labelled 2023 and its four quarters are labelled 2023Q1 through 2023Q4. Companies reporting
            on calendar quarter ends (March, June, September and December) are never relabelled, and
            neither are fiscal years ending in June through December. Which tickers were relabelled is
            reported in the log and available on the Toolkit as _fiscal_year_adjustments.

            The last day of each fiscal period is kept as well: daily ratios apply a period's figures
            from that day on (not from the end of its calendar label), and currency translation uses
            the closing rate on that day and the average rate over the fiscal period. The dates come
            from the Period End Date in the statistics statement; when a source does not report it,
            the latest day the period can end on given its label is used, so that figures are never
            applied before the period ended.

            This convention is what makes financial statements line up with price history, since prices
            are always in calendar time. It also means the labels are not the company's own fiscal year
            numbering: what NVIDIA calls fiscal 2024 appears here as 2023.

        As an example:

        ```python
        from financetoolkit import Toolkit

        # Simple example
        toolkit = Toolkit(
            tickers=["TSLA", "ASML"],
            api_key="FINANCIAL_MODELING_PREP_KEY")

        # Obtaining quarterly data
        toolkit = Toolkit(
            tickers=["AAPL", "GOOGL"],
            quarterly=True,
            api_key="FINANCIAL_MODELING_PREP_KEY")

        # Enforce a specific source
        toolkit = Toolkit(
            tickers=["ASML", "BABA"],
            quarterly=True,
            enforce_source="YahooFinance")

        # Including a start and end date
        toolkit = Toolkit(
            tickers=["MSFT", "MU"],
            start_date="2020-01-01",
            end_date="2023-01-01",
            quarterly=True,
            api_key="FINANCIAL_MODELING_PREP_KEY")

        # Storing the cache in a specific folder, or use_cached_data=False to not cache
        toolkit = Toolkit(
            tickers=["WMT", "AAPL"],
            quarterly=True,
            api_key="FINANCIAL_MODELING_PREP_KEY",
            use_cached_data="datasets")

        # Changing the benchmark and risk free rate
        toolkit = Toolkit(
            tickers="AMZN",
            benchmark_ticker="^DJI",
            risk_free_rate="30y",
            api_key="FINANCIAL_MODELING_PREP_KEY")
        ```
        """
        # A copied documentation example passes the placeholder key, treated as no key at all.
        api_key = validation_model.resolve_api_key(api_key)

        self._api_key = api_key
        self._fred_api_key = validation_model.resolve_fred_api_key(fred_api_key)
        self._risk_free_rate = risk_free_rate
        self._rounding = rounding
        self._remove_invalid_tickers = remove_invalid_tickers
        self._invalid_tickers: list = []

        self._cache = cache_controller.resolve_cache(use_cached_data)
        self._use_cached_data = self._cache.enabled
        self._cache_location = self._cache.location

        # Published so the OECD/FRED/ECB/Fed free functions pick up this cache too.
        cache_controller.set_active_cache(self._cache)
        self._allow_stale_oecd_cache = allow_stale_oecd_cache
        self._benchmark_ticker = benchmark_ticker

        # Everything the user can get wrong is checked in one place; this raises on
        # invalid input and hands back the normalized ticker list (upper-cased, ISIN
        # codes converted, duplicates and the benchmark ticker removed).
        validated_tickers = validation_model.validate_toolkit_parameters(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            risk_free_rate=risk_free_rate,
            enforce_source=enforce_source,
            api_key=api_key,
            intraday_period=intraday_period,
            benchmark_ticker=benchmark_ticker,
        )

        self._start_date = (
            start_date
            if start_date
            else (
                datetime.now() - timedelta(days=90 * 5 if quarterly else 365 * 5)
            ).strftime("%Y-%m-%d")
        )
        self._end_date = end_date if end_date else datetime.now().strftime("%Y-%m-%d")
        self._quarterly = quarterly

        # One extra period so two-period metrics have a prior value for the first year.
        if quarterly:
            _lookback_dt = datetime.strptime(self._start_date, "%Y-%m-%d") - timedelta(
                days=92
            )
        else:
            _lookback_dt = datetime.strptime(self._start_date, "%Y-%m-%d") - timedelta(
                days=366
            )
        self._lookback_start_date = _lookback_dt.strftime("%Y-%m-%d")

        # The cache tracks each ticker and range, so these arguments are always used.

        self._tickers: list[str] = validated_tickers
        self._enforce_source: str | None = enforce_source

        if sleep_timer is None:
            # Determines the plan, which drives the sleep timer and other components.
            self._fmp_plan, invalid_api_key = _determine_subscription_plan(
                api_key=api_key
            )

            if invalid_api_key and api_key:
                self._enforce_source = "YahooFinance"
                logger.error(
                    "You have entered an invalid API key from Financial Modeling Prep. Obtain your API key for free "
                    "and get 15% off the Premium plans by using the following affiliate link.\nThis also supports "
                    "the project: https://www.jeroenbouma.com/fmp\nUsing Yahoo Finance as data source instead."
                )
        else:
            self._fmp_plan = "Premium"

        self._sleep_timer = (
            sleep_timer if sleep_timer is not None else self._fmp_plan != "Free"
        )

        self._progress_bar = progress_bar

        if self._api_key or self._use_cached_data:
            # Initialize attributes to empty DataFrames
            self._profile: pd.DataFrame = pd.DataFrame()
            self._quote: pd.DataFrame = pd.DataFrame()
            self._rating: pd.DataFrame = pd.DataFrame()
            self._analyst_estimates: pd.DataFrame = pd.DataFrame()
            self._analyst_estimates_growth: pd.DataFrame = pd.DataFrame()
            self._dividend_calendar: pd.DataFrame = pd.DataFrame()
            self._earnings_calendar: pd.DataFrame = pd.DataFrame()
            self._esg_scores: pd.DataFrame = pd.DataFrame()
            self._revenue_geographic_segmentation: pd.DataFrame = pd.DataFrame()
            self._revenue_product_segmentation: pd.DataFrame = pd.DataFrame()
            self._revenue_geographic_segmentation_growth: pd.DataFrame = pd.DataFrame()
            self._revenue_product_segmentation_growth: pd.DataFrame = pd.DataFrame()
            self._market_risk_premium: pd.DataFrame = pd.DataFrame()
            self._commitment_of_traders: pd.DataFrame = pd.DataFrame()
            self._executives: pd.DataFrame = pd.DataFrame()
            self._executive_compensation: pd.DataFrame = pd.DataFrame()
            self._company_notes: pd.DataFrame = pd.DataFrame()
            self._employee_count: pd.DataFrame = pd.DataFrame()
            self._shares_float: pd.DataFrame = pd.DataFrame()
            self._mergers_acquisitions: pd.DataFrame = pd.DataFrame()
            self._stock_splits: pd.DataFrame = pd.DataFrame()
            self._insider_trade_statistics: pd.DataFrame = pd.DataFrame()
            self._stock_grades: pd.DataFrame = pd.DataFrame()
            self._etf_holdings: pd.DataFrame = pd.DataFrame()
            self._etf_information: pd.DataFrame = pd.DataFrame()
            self._etf_country_weightings: pd.DataFrame = pd.DataFrame()
            self._etf_sector_weightings: pd.DataFrame = pd.DataFrame()
            self._earnings_call_transcripts: pd.DataFrame = pd.DataFrame()
            # Which selection the stored transcripts hold, so switching between the latest
            # transcript and the full range does not serve the other one.
            self._earnings_call_transcripts_selection: (
                tuple[bool, str | None] | None
            ) = None

            # Resolved per ticker on request, so a different list reuses what it shares.

        self._intraday_period = intraday_period

        # Resolved per ticker and range on request, so an overlap is reused.
        self._intraday_historical_data: pd.DataFrame = pd.DataFrame()

        # Use provided historical data if available, otherwise start empty.
        self._historical = historical

        self._daily_historical_data: pd.DataFrame = (
            historical if not historical.empty else pd.DataFrame()
        )
        # None means "not fetched by us yet", so pre-supplied `historical` is never auto-invalidated below.
        self._daily_historical_data_params: tuple | None = None
        # Per period, the daily data and the settings its conversion was made with.
        self._period_historical_data_sources: dict[str, tuple] = {}

        # Initialize other periods as empty DataFrames. They will be populated on demand.
        self._weekly_historical_data: pd.DataFrame = pd.DataFrame()
        self._monthly_historical_data: pd.DataFrame = pd.DataFrame()
        self._quarterly_historical_data: pd.DataFrame = pd.DataFrame()
        self._yearly_historical_data: pd.DataFrame = pd.DataFrame()
        self._historical_statistics: pd.DataFrame = pd.DataFrame()

        # Initialization of the Financial Statements and Normalization
        self._reverse_dates = reverse_dates

        (
            self._balance_sheet_statement,
            self._income_statement,
            self._cash_flow_statement,
            self._statistics_statement,
            self._fmp_balance_sheet_statement_generic,
            self._yf_balance_sheet_statement_generic,
            self._fmp_income_statement_generic,
            self._yf_income_statement_generic,
            self._fmp_cash_flow_statement_generic,
            self._yf_cash_flow_statement_generic,
            self._fmp_statistics_statement_generic,
            self._yf_statistics_statement_generic,
        ) = _initialize_statements_and_normalization(
            balance=balance,
            income=income,
            cash=cash,
            format_location=format_location,
            reverse_dates=self._reverse_dates,
            start_date=self._start_date,
            end_date=self._end_date,
            quarterly=self._quarterly,
        )

        self._balance_sheet_statement_growth: pd.DataFrame = pd.DataFrame()
        self._income_statement_growth: pd.DataFrame = pd.DataFrame()
        self._cash_flow_statement_growth: pd.DataFrame = pd.DataFrame()
        self._currencies: list = []
        self._statement_currencies: pd.Series = pd.Series()
        self._fiscal_year_adjustments: dict[str, list[dict]] = {}
        self._convert_currency = (
            convert_currency
            if convert_currency is not None
            else self._fmp_plan != "Free"
        )

        # Initialization of Risk Free Rate
        self._daily_risk_free_rate: pd.DataFrame = pd.DataFrame()
        self._weekly_risk_free_rate: pd.DataFrame = pd.DataFrame()
        self._monthly_risk_free_rate: pd.DataFrame = pd.DataFrame()
        self._quarterly_risk_free_rate: pd.DataFrame = pd.DataFrame()
        self._yearly_risk_free_rate: pd.DataFrame = pd.DataFrame()

        # Initialization of Treasury Variables
        self._daily_treasury_data: pd.DataFrame = pd.DataFrame()
        self._weekly_treasury_data: pd.DataFrame = pd.DataFrame()
        self._monthly_treasury_data: pd.DataFrame = pd.DataFrame()
        self._quarterly_treasury_data: pd.DataFrame = pd.DataFrame()
        self._yearly_treasury_data: pd.DataFrame = pd.DataFrame()

        # Initialization of the Exchange Rate Variables
        self._daily_exchange_rate_data: pd.DataFrame = pd.DataFrame()
        self._weekly_exchange_rate_data: pd.DataFrame = pd.DataFrame()
        self._monthly_exchange_rate_data: pd.DataFrame = pd.DataFrame()
        self._quarterly_exchange_rate_data: pd.DataFrame = pd.DataFrame()
        self._yearly_exchange_rate_data: pd.DataFrame = pd.DataFrame()

        # Initialization of the Portfolio Variables
        self._portfolio_weights: dict | None = None

        # Shared across every `toolkit.ratios` access so estimates are fetched once.
        self._analyst_estimates_cache: dict = {}

        pd.set_option("display.float_format", str)

    @property
    def ratios(self) -> "Ratios":
        """
        The Ratios Module contains over 50+ ratios that can be used to analyse companies. These ratios
        are divided into 5 categories which are efficiency, liquidity, profitability, solvency and
        valuation. Each ratio is calculated using the data from the Toolkit module.

        Some examples of ratios are the Current Ratio, Debt to Equity Ratio, Return on Assets (ROA),
        Return on Equity (ROE), Return on Invested Capital (ROIC), Return on Capital Employed (ROCE),
        Price to Earnings Ratio (P/E), Price to Book Ratio (P/B), Price to Sales Ratio (P/S), Price
        to Cash Flow Ratio (P/CF), Price to Free Cash Flow Ratio (P/FCF), Dividend Yield and
        Dividend Payout Ratio.

        Next to that, it is also possible to define custom ratios.

        See the following link for more information: https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios

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
        # Imported when first used, so the Finance Toolkit loads only what is needed.
        from financetoolkit.ratios.ratios_controller import Ratios  # noqa: PLC0415

        # Every dataset is retrieved when a ratio first needs it, so a ratio downloads only
        # what it uses (the gross margin only the income statement).
        period = "quarterly" if self._quarterly else "yearly"

        ratios = Ratios(
            tickers=lambda: self._get_statement_tickers("Ratios")
            + (["Portfolio"] if "Portfolio" in self._tickers else []),
            historical=LazyDict(
                lambda key: self._get_module_historical_data(
                    period if key == "period" else key
                )
            ),
            balance=lambda: self._get_module_statement(
                "balance_sheet_statement", "Ratios"
            ),
            income=lambda: self._get_module_statement("income_statement", "Ratios"),
            cash=lambda: self._get_module_statement("cash_flow_statement", "Ratios"),
            quarterly=self._quarterly,
            rounding=self._rounding,
            start_date=self._start_date,
            end_date=self._end_date,
            api_key=self._api_key,
            sleep_timer=self._sleep_timer,
            user_subscription=self._fmp_plan,
            analyst_estimates_cache=self._analyst_estimates_cache,
            period_end_dates=lambda: self._get_period_end_dates(
                self._get_module_statement("balance_sheet_statement", "Ratios")
            ),
        )

        if self._portfolio_weights:
            ratios._portfolio_weights = self._portfolio_weights

        return ratios

    @property
    def models(self) -> "Models":
        """
        Gives access to the Models module. The Models module is meant to execute well-known models
        such as DUPONT and the Discounted Cash Flow (DCF) model. These models are also directly
        related to the data retrieved from the Toolkit module.

        See the following link for more information: https://www.jeroenbouma.com/projects/financetoolkit/docs/models

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["TSLA", "AMZN"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            quarterly=True,
            start_date='2022-12-31',
            end_date="2025-12-31",
        )

        dupont_analysis = toolkit.models.get_extended_dupont_analysis()

        dupont_analysis.loc['AMZN']
        ```

        Which returns:

        |                         |   2022Q4 |   2023Q1 |   2023Q2 |   2023Q3 |   2023Q4 |   2024Q1 |   2024Q2 |   2024Q3 |   2024Q4 |   2025Q1 |   2025Q2 |   2025Q3 |   2025Q4 |
        |:------------------------|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|
        | Interest Burden Ratio   |  -0.3467 |   0.863  |   0.9835 |   1.0891 |   1.0361 |   0.8426 |   1.0395 |   1.0358 |   1.0531 |   1.1779 |   1.0872 |   1.6127 |   1.0465 |
        | Tax Burden Ratio        |  -0.2929 |   0.7699 |   0.8936 |   0.8108 |   0.7763 |   0.8087 |   0.8841 |   0.85   |   0.8959 |   0.79   |   0.8715 |   0.7541 |   0.8108 |
        | Operating Profit Margin |   0.0183 |   0.0375 |   0.0572 |   0.0782 |   0.0777 |   0.1068 |   0.0992 |   0.1096 |   0.1129 |   0.1182 |   0.1143 |   0.0967 |   0.1171 |
        | Asset Turnover          | nan      |   0.2748 |   0.2853 |   0.2967 |   0.335  |   0.2707 |   0.2726 |   0.2789 |   0.3105 |   0.2455 |   0.2531 |   0.2555 |   0.2761 |
        | Equity Multiplier       | nan      |   3.0843 |   2.9152 |   2.7433 |   2.6367 |   2.5298 |   2.3963 |   2.2991 |   2.2188 |   2.1427 |   2.0721 |   2.0047 |   1.9802 |
        | Return on Equity        | nan      |   0.0211 |   0.0418 |   0.0562 |   0.0552 |   0.0498 |   0.0595 |   0.0619 |   0.0734 |   0.0579 |   0.0568 |   0.0602 |   0.0543 |
        """
        # Imported when first used, so the Finance Toolkit loads only what is needed.
        from financetoolkit.models.models_controller import Models  # noqa: PLC0415

        # Every dataset is retrieved when a model first needs it.

        def compare_with_peers(peers: dict[str, list[str]] | None) -> pd.DataFrame:
            # Imported when first used, as only the peer comparison needs them.
            from financetoolkit.fmp_model import get_stock_peers  # noqa: PLC0415
            from financetoolkit.models import peers_model  # noqa: PLC0415

            tickers = self._get_statement_tickers("Models")

            peer_map = {
                ticker: list((peers or {}).get(ticker) or [])
                or get_stock_peers(ticker, self._api_key, self._fmp_plan)
                for ticker in tickers
            }
            universe = sorted(
                set(tickers) | {peer for group in peer_map.values() for peer in group}
            )
            # The latest fiscal year is what is compared, so three years of data suffice.
            others = Toolkit(
                tickers=universe,
                api_key=self._api_key,
                start_date=(datetime.today() - timedelta(days=365 * 3)).strftime(
                    "%Y-%m-%d"
                ),
                use_cached_data=self._cache,
                enforce_source=self._enforce_source,
                benchmark_ticker=None,
                progress_bar=False,
            )
            ratios = others.ratios
            income = others.get_income_statement(growth=True)
            metrics = pd.DataFrame(
                {
                    "Price-to-Earnings": peers_model.get_latest_values(
                        ratios.get_price_to_earnings_ratio()
                    ),
                    "EV-to-EBITDA": peers_model.get_latest_values(
                        ratios.get_ev_to_ebitda_ratio()
                    ),
                    "Price-to-Book": peers_model.get_latest_values(
                        ratios.get_price_to_book_ratio()
                    ),
                    "Price-to-Sales": peers_model.get_latest_values(
                        ratios.get_price_to_sales_ratio()
                    ),
                    "Gross Margin": peers_model.get_latest_values(
                        ratios.get_gross_margin()
                    ),
                    "Operating Margin": peers_model.get_latest_values(
                        ratios.get_operating_margin()
                    ),
                    "Net Profit Margin": peers_model.get_latest_values(
                        ratios.get_net_profit_margin()
                    ),
                    "Return on Equity": peers_model.get_latest_values(
                        ratios.get_return_on_equity()
                    ),
                    "Return on Invested Capital": peers_model.get_latest_values(
                        ratios.get_return_on_invested_capital()
                    ),
                    "Revenue Growth": peers_model.get_latest_values(
                        income.loc[:, "Revenue", :]
                    ),
                    "Debt-to-Equity": peers_model.get_latest_values(
                        ratios.get_debt_to_equity_ratio()
                    ),
                    "Dividend Yield": peers_model.get_latest_values(
                        ratios.get_dividend_yield()
                    ),
                }
            )
            return pd.concat(
                {
                    ticker: peers_model.get_peer_comparison(
                        metrics, ticker, peer_map[ticker]
                    )
                    for ticker in tickers
                }
            )

        return Models(
            tickers=lambda: self._get_statement_tickers("Models"),
            historical_data=LazyDict(self._get_module_historical_data),
            risk_free_rate_data=LazyDict(self._get_module_risk_free_rate),
            balance=lambda: self._get_module_statement(
                "balance_sheet_statement", "Models"
            ),
            income=lambda: self._get_module_statement("income_statement", "Models"),
            cash=lambda: self._get_module_statement("cash_flow_statement", "Models"),
            quarterly=self._quarterly,
            rounding=self._rounding,
            start_date=self._start_date,
            end_date=self._end_date,
            peer_function=compare_with_peers,
        )

    @property
    def options(self) -> "Options":
        """
        This gives access to the Options module. The Options Module is meant to provide Options valuations
        based on real market data. This includes the Black-Scholes model and in the future the Binomial model
        and the Monte Carlo model. It also includes all available first-order, second-order and third-order
        Greeks such as Delta, Gamma, Theta, Vega, Rho, Charm, Vanna, Vomma, Veta, Speed and Zomma.

        It gives insights in the sensitivity of an option to changes in the underlying asset price, volatility,
        years to maturity, dividend yilds and interest rates and several derivatives of these sensitivities.

        See the following link for more information: https://www.jeroenbouma.com/projects/financetoolkit/docs/options

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["TSLA", "MU"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        all_greeks = toolkit.options.collect_all_greeks(start_date='2024-01-03')

        all_greeks.loc['TSLA', '2024-01-04']
        ```

        Which returns:

        |   Strike Price |   Delta |   Dual Delta |   Vega |   Theta |    Rho |   Epsilon |   Lambda |   Gamma |   Dual Gamma |   Vanna |    Charm |   Vomma |    Vera |    Veta |     PD |   Speed |   Zomma |   Color |   Ultima |
        |---------------:|--------:|-------------:|-------:|--------:|-------:|----------:|---------:|--------:|-------------:|--------:|---------:|--------:|--------:|--------:|-------:|--------:|--------:|--------:|---------:|
        |            180 |  1      |      -0.9999 | 0      | -0.0191 | 0.4931 |   -0.6533 |   4.0782 |  0      |       0      | -0      |   0      |  0      | -0      | -0      | 0      | -0      |  0      | -0      |   0      |
        |            185 |  1      |      -0.9999 | 0      | -0.0196 | 0.5068 |   -0.6533 |   4.4595 |  0      |       0      | -0      |   0      |  0      | -0      | -0      | 0      | -0      |  0      | -0      |   0      |
        |            190 |  1      |      -0.9999 | 0      | -0.0202 | 0.5205 |   -0.6533 |   4.9195 |  0      |       0      | -0      |   0      |  0      | -0      | -0      | 0      | -0      |  0      | -0      |   0      |
        |            195 |  1      |      -0.9999 | 0      | -0.0207 | 0.5342 |   -0.6533 |   5.4853 |  0      |       0      | -0      |   0      |  0      | -0      | -0      | 0      | -0      |  0      | -0      |   0      |
        |            200 |  1      |      -0.9999 | 0      | -0.0212 | 0.5479 |   -0.6533 |   6.1981 |  0      |       0      | -0      |   0      |  0      | -0      | -0      | 0      | -0      |  0      | -0      |   0.0001 |
        |            205 |  1      |      -0.9999 | 0      | -0.0217 | 0.5616 |   -0.6533 |   7.1239 |  0      |       0      | -0      |   0.0002 |  0.0001 | -0      | -0      | 0      | -0      |  0      | -0.0002 |   0.0071 |
        |            210 |  1      |      -0.9999 | 0      | -0.0223 | 0.5753 |   -0.6533 |   8.3748 |  0      |       0      | -0.0001 |   0.0124 |  0.0071 | -0.0001 | -0      | 0      | -0      |  0.0001 | -0.0079 |   0.2261 |
        |            215 |  0.9999 |      -0.9998 | 0.0001 | -0.0243 | 0.5889 |   -0.6532 |  10.1573 |  0.0001 |       0.0001 | -0.0031 |   0.3048 |  0.1423 | -0.002  | -0.0004 | 0.0001 | -0      |  0.0016 | -0.1539 |   2.7483 |
        |            220 |  0.9979 |      -0.9976 | 0.0008 | -0.0454 | 0.6013 |   -0.6519 |  12.8781 |  0.001  |       0.0011 | -0.0345 |   3.3891 |  1.2332 | -0.0227 | -0.0037 | 0.0011 | -0.0004 |  0.0129 | -1.2633 |  11.7212 |
        |            225 |  0.9808 |      -0.9794 | 0.0058 | -0.1808 | 0.6037 |   -0.6408 |  17.297  |  0.0069 |       0.0078 | -0.1766 |  17.3444 |  4.5655 | -0.117  | -0.0152 | 0.0078 | -0.0022 |  0.0414 | -4.0652 |  10.4124 |
        |            230 |  0.9022 |      -0.8971 | 0.0216 | -0.604  | 0.5653 |   -0.5894 |  24.4765 |  0.0256 |       0.0275 | -0.4047 |  39.6576 |  6.5361 | -0.2703 | -0.0283 | 0.0275 | -0.005  |  0.0303 | -2.9369 | -16.4877 |
        |            235 |  0.7031 |      -0.6932 | 0.0432 | -1.1838 | 0.4463 |   -0.4593 |  35.2268 |  0.0513 |       0.0528 | -0.3236 |  31.4245 |  2.1541 | -0.2232 | -0.0273 | 0.0528 | -0.0043 | -0.0694 |  6.8815 | -10.901  |
        |            240 |  0.4163 |      -0.4053 | 0.0487 | -1.3254 | 0.2665 |   -0.272  |  49.5371 |  0.0579 |       0.0571 |  0.173  | -17.5904 |  0.4562 |  0.0997 | -0.0256 | 0.0571 |  0.0016 | -0.1017 | 10.0098 |  -2.5045 |
        |            245 |  0.1735 |      -0.1663 | 0.032  | -0.8684 | 0.1116 |   -0.1133 |  66.7011 |  0.038  |       0.036  |  0.4598 | -45.6744 |  5.3973 |  0.2916 | -0.0307 | 0.036  |  0.0051 | -0.0062 |  0.5668 | -20.8797 |
        |            250 |  0.049  |      -0.0461 | 0.0127 | -0.343  | 0.0316 |   -0.032  |  85.8277 |  0.015  |       0.0137 |  0.316  | -31.292  |  6.5283 |  0.203  | -0.024  | 0.0137 |  0.0036 |  0.0497 | -4.9373 |  -2.5872 |
        |            255 |  0.0092 |      -0.0086 | 0.0031 | -0.0841 | 0.006  |   -0.006  | 106.169  |  0.0037 |       0.0032 |  0.1099 | -10.8635 |  3.2297 |  0.0709 | -0.0103 | 0.0032 |  0.0013 |  0.0315 | -3.1218 |  15.635  |
        |            260 |  0.0012 |      -0.0011 | 0.0005 | -0.0132 | 0.0008 |   -0.0008 | 127.197  |  0.0006 |       0.0005 |  0.0222 |  -2.1908 |  0.8418 |  0.0143 | -0.0025 | 0.0005 |  0.0003 |  0.0089 | -0.8831 |   9.8831 |
        |            265 |  0.0001 |      -0.0001 | 0      | -0.0014 | 0.0001 |   -0.0001 | 148.561  |  0.0001 |       0      |  0.0028 |  -0.2741 |  0.1287 |  0.0018 | -0.0004 | 0      |  0      |  0.0014 | -0.1402 |   2.6    |
        |            270 |  0      |      -0      | 0      | -0.0001 | 0      |   -0      | 170.037  |  0      |       0      |  0.0002 |  -0.0222 |  0.0123 |  0.0001 | -0      | 0      |  0      |  0.0001 | -0.0137 |   0.3703 |
        |            275 |  0      |      -0      | 0      | -0      | 0      |   -0      | 191.483  |  0      |       0      |  0      |  -0.0012 |  0.0008 |  0      | -0      | 0      |  0      |  0      | -0.0009 |   0.0317 |
        |            280 |  0      |      -0      | 0      | -0      | 0      |   -0      | 212.807  |  0      |       0      |  0      |  -0      |  0      |  0      | -0      | 0      |  0      |  0      | -0      |   0.0017 |
        |            285 |  0      |      -0      | 0      | -0      | 0      |   -0      | 233.949  |  0      |       0      |  0      |  -0      |  0      |  0      | -0      | 0      |  0      |  0      | -0      |   0.0001 |
        |            290 |  0      |      -0      | 0      | -0      | 0      |   -0      | 254.872  |  0      |       0      |  0      |  -0      |  0      |  0      | -0      | 0      |  0      |  0      | -0      |   0      |
        |            295 |  0      |      -0      | 0      | -0      | 0      |   -0      | 275.552  |  0      |       0      |  0      |  -0      |  0      |  0      | -0      | 0      |  0      |  0      | -0      |   0      |
        """
        # Imported when first used, so the Finance Toolkit loads only what is needed.
        from financetoolkit.options.options_controller import Options  # noqa: PLC0415

        # The prices and the risk free rate are retrieved when a calculation first needs them.

        def next_earnings_dates() -> dict[str, str]:
            # The first announcement from today on, per ticker, for the implied earnings
            # move. The earnings calendar stops at the end date, so the dates are read
            # from the announcements Financial Modeling Prep lists, estimates included.
            from financetoolkit.fmp_model import (  # noqa: PLC0415
                get_upcoming_earnings_date,
            )

            if not self._api_key:
                return {}
            dates = {}
            for ticker in self._tickers:
                upcoming = get_upcoming_earnings_date(
                    ticker, self._api_key, self._fmp_plan
                )
                if upcoming:
                    dates[ticker] = upcoming
            return dates

        return Options(
            tickers=self._tickers,
            daily_historical=lambda: self._get_module_historical_data("daily"),
            annual_historical=lambda: self._get_module_historical_data("yearly"),
            risk_free_rate=lambda: self._get_module_risk_free_rate("daily"),
            quarterly=self._quarterly,
            rounding=self._rounding,
            start_date=self._start_date,
            end_date=self._end_date,
            earnings_function=next_earnings_dates,
        )

    @property
    def technicals(self) -> "Technicals":
        """
        This gives access to the Technicals module. The Technicals Module contains
        nearly 50 Technical Indicators that can be used to analyse companies. These indicators are
        divided into 3 categories: breadth, overlap and volatility. Each indicator is calculated using
        the data from the Toolkit module.

        Some examples of technical indicators are the Average Directional Index (ADX), the
        Accumulation/Distribution Line (ADL), the Average True Range (ATR), the Bollinger Bands (BBANDS),
        the Commodity Channel Index (CCI), the Chaikin Oscillator (CHO), the Chaikin Money Flow (CMF),
        the Double Exponential Moving Average (DEMA), the Exponential Moving Average (EMA) and
        the Moving Average Convergence Divergence (MACD).

        See the following link for more information: https://www.jeroenbouma.com/projects/financetoolkit/docs/technicals

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "TSLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        average_directional_index = toolkit.technicals.get_average_directional_index()
        ```

        Which returns:

        | Date       |    AAPL |    TSLA |   Benchmark |
        |:-----------|--------:|--------:|------------:|
        | 2025-12-24 | 22.018  | 24.0777 |     14.0995 |
        | 2025-12-26 | 20.9256 | 24.3229 |     14.8314 |
        | 2025-12-29 | 20.0592 | 23.2159 |     14.7624 |
        | 2025-12-30 | 19.276  | 21.7824 |     14.6983 |
        | 2025-12-31 | 18.7175 | 20.3475 |     13.7116 |

        """
        # Imported when first used, so the Finance Toolkit loads only what is needed.
        from financetoolkit.technicals.technicals_controller import (  # noqa: PLC0415
            Technicals,
        )

        # The prices of a period are retrieved when an indicator first asks for that period.
        technicals = Technicals(
            tickers=lambda: self._get_historical_tickers(include_benchmark=True),
            historical_data=LazyDict(self._get_module_historical_data),
            rounding=self._rounding,
            start_date=self._start_date,
            end_date=self._end_date,
        )

        if self._portfolio_weights:
            technicals._portfolio_weights = self._portfolio_weights

        return technicals

    @property
    def performance(self) -> "Performance":
        """
        This gives access to the Performance module. The Performance Module is meant to calculate metrics related
        to the risk-return relationship. These are things such as Beta, Sharpe Ratio, Sortino Ratio, CAPM,
        Alpha and the Treynor Ratio.

        It gives insights in the performance a stock has to e.g. a benchmark that is not easily identified by
        looking at the raw data. This class is closely related to the Risk class which highlights things
        such as Value at Risk (VaR) and Maximum Drawdown.

        See the following link for more information: https://www.jeroenbouma.com/projects/financetoolkit/docs/performance

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "TSLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.performance.get_capital_asset_pricing_model(period='quarterly')
        ```

        Which returns:

        | Date   |    AAPL |    TSLA |
        |:-------|--------:|--------:|
        | 2021Q2 |  0.1154 |  0.1364 |
        | 2021Q3 |  0.0066 |  0.0052 |
        | 2021Q4 |  0.1214 |  0.1869 |
        | 2022Q1 | -0.0575 | -0.1007 |
        | 2022Q2 | -0.2134 | -0.332  |
        | 2022Q3 | -0.0594 | -0.0818 |
        | 2022Q4 |  0.106  |  0.0998 |
        | 2023Q1 |  0.083  |  0.1515 |
        | 2023Q2 |  0.1032 |  0.1762 |
        | 2023Q3 | -0.0414 | -0.1013 |
        | 2023Q4 |  0.1021 |  0.2457 |
        | 2024Q1 |  0.1038 |  0.1407 |
        | 2024Q2 |  0.0524 |  0.0588 |
        | 2024Q3 |  0.0571 |  0.1518 |
        | 2024Q4 |  0.0215 |  0.0531 |
        | 2025Q1 | -0.0386 | -0.1508 |
        | 2025Q2 |  0.1473 |  0.2035 |
        | 2025Q3 |  0.1029 |  0.1764 |
        | 2025Q4 |  0.0236 |  0.0498 |
        """
        # Imported when first used, so the Finance Toolkit loads only what is needed.
        from financetoolkit.performance.performance_controller import (  # noqa: PLC0415
            Performance,
        )

        # The prices and risk free rate of a period are retrieved when a metric first asks
        # for that period.
        performance = Performance(
            tickers=lambda: self._get_historical_tickers(include_benchmark=False),
            historical_data=LazyDict(self._get_module_historical_data),
            risk_free_rate_data=LazyDict(
                lambda period: self._get_module_risk_free_rate(period)["Adj Close"]
            ),
            quarterly=self._quarterly,
            rounding=self._rounding,
            start_date=self._start_date,
            end_date=self._end_date,
            intraday_period=self._intraday_period,
            progress_bar=self._progress_bar,
        )

        if self._portfolio_weights:
            performance._portfolio_weights = self._portfolio_weights

        return performance

    @property
    def risk(self) -> "Risk":
        """
        This gives access to the Risk module. The Risk Module is meant to calculate metrics related to risk such
        as Value at Risk (VaR), Conditional Value at Risk (cVaR), EMWA/GARCH models and similar models. It also
        houses cross-asset systemic risk and liquidity measures (CoVaR, Tail Dependence, Amihud Illiquidity,
        Roll Spread).

        Note that the time-series diagnostic and econometric tests (unit root tests, cointegration, Granger
        causality, ARCH-LM, Jarque-Bera and similar tests) live in the separate Econometrics module instead.

        It gives insights in the risk a stock composes that is not perceived as easily by looking at the data.
        This class is closely related to the Performance class which highlights things such as Sharpe Ratio and
        Sortino Ratio.

        See the following link for more information: https://www.jeroenbouma.com/projects/financetoolkit/docs/risk

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "TSLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_value_at_risk(period='yearly')
        ```

        Which returns:

        |           |   Value |
        |:----------|--------:|
        | AAPL      | -0.2108 |
        | TSLA      | -0.5357 |
        | Benchmark | -0.1279 |
        """
        # Imported when first used, so the Finance Toolkit loads only what is needed.
        from financetoolkit.risk.risk_controller import Risk  # noqa: PLC0415

        # The prices and risk free rate of a period are retrieved when a metric first asks
        # for that period.
        risk = Risk(
            tickers=lambda: self._get_historical_tickers(include_benchmark=True),
            historical_data=LazyDict(self._get_module_historical_data),
            risk_free_rate_data=LazyDict(
                lambda period: self._get_module_risk_free_rate(period)["Adj Close"]
            ),
            intraday_period=self._intraday_period,
            quarterly=self._quarterly,
            rounding=self._rounding,
            start_date=self._start_date,
            end_date=self._end_date,
        )

        if self._portfolio_weights:
            risk._portfolio_weights = self._portfolio_weights

        return risk

    @property
    def econometrics(self) -> "Econometrics":
        """
        This gives access to the Econometrics module, a thin wrapper that funnels this Toolkit's
        price/return data through `statsmodels` and `linearmodels` -- regression (OLS/WLS/GLS/
        Logit/Probit/Quantile), panel data (Fixed/Random Effects, Hausman), causal inference
        (IV-2SLS, Difference-in-Differences, Regression Discontinuity, Propensity Score Matching),
        specification/hypothesis tests (Breusch-Pagan, White, Durbin-Watson, VIF, RESET, Chow,
        t/F/LR/Wald tests), stationarity (Augmented Dickey-Fuller, KPSS, Phillips-Perron,
        Zivot-Andrews unit root tests), long-run equilibrium relationships (Engle-Granger and
        Johansen cointegration), predictive lead-lag relationships (Granger causality), model/
        residual diagnostics (ARCH-LM, Jarque-Bera, Ljung-Box, Variance Ratio, CUSUM), forecast
        comparison (Diebold-Mariano), and time series forecasting (ARIMA, VAR, VECM).

        This class is closely related to the Risk class, which houses the risk measures (VaR,
        CVaR, GARCH) that these tests often inform the choice of.

        Requires the optional `financetoolkit[econometrics]` extra (`statsmodels` and
        `linearmodels`) -- install with `pip install financetoolkit[econometrics]`.

        See the following link for more information: https://www.jeroenbouma.com/projects/financetoolkit/docs/econometrics

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.econometrics.get_augmented_dickey_fuller(period='yearly')
        ```
        """
        try:
            from financetoolkit.econometrics.econometrics_controller import (  # noqa: PLC0415
                Econometrics,
            )
        except ImportError as error:
            raise ImportError(
                "The Econometrics module requires the optional 'econometrics' extra "
                "(statsmodels and linearmodels). Install it with: "
                "pip install financetoolkit[econometrics]"
            ) from error

        # The prices of a period are retrieved when a method first asks for that period.
        econometrics = Econometrics(
            tickers=lambda: self._get_historical_tickers(include_benchmark=True),
            historical_data=LazyDict(self._get_module_historical_data),
            intraday_period=self._intraday_period,
            quarterly=self._quarterly,
            rounding=self._rounding,
            start_date=self._start_date,
            end_date=self._end_date,
        )

        if self._portfolio_weights:
            econometrics._portfolio_weights = self._portfolio_weights

        return econometrics

    @property
    def etf(self) -> "ETF":
        """
        The ETF module analyzes the Toolkit's exchange traded funds (and other funds): their
        holdings, sector and country weightings, profile and premium or discount to NAV, how closely
        they track the Benchmark, how concentrated and how much alike they are, what their
        holdings are valued at and how liquid they are.

        See the following link for more information: https://www.jeroenbouma.com/projects/financetoolkit/docs/etf

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["VOO", "QQQ", "AGG"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.etf.get_profile()
        ```
        """
        # Imported when first used, so the Finance Toolkit loads only what is needed.
        from financetoolkit.etf.etf_controller import ETF  # noqa: PLC0415

        if not self._start_date:
            self._start_date = (datetime.today() - timedelta(days=365 * 10)).strftime(
                "%Y-%m-%d"
            )
        if not self._end_date:
            self._end_date = datetime.today().strftime("%Y-%m-%d")

        def historical() -> pd.DataFrame:
            # Only the analyses of prices need them, not the holdings and weightings.
            self.get_historical_data(period="daily")
            return self._daily_historical_data

        def ratios(tickers: list[str]) -> pd.DataFrame:
            ratios_ttm, _ = _get_ratios_ttm(
                tickers, api_key=self._api_key, user_subscription=self._fmp_plan
            )
            return ratios_ttm

        def currencies() -> pd.Series:
            # The currency each ticker is listed in, to compare its close with the NAV.
            statistics = self.get_historical_statistics()
            if isinstance(statistics, pd.Series):
                return pd.Series({self._tickers[0]: statistics.get("Currency")})
            if statistics.empty or "Currency" not in statistics.index:
                return pd.Series(dtype=object)
            return statistics.loc["Currency"]

        return ETF(
            tickers=self._tickers,
            historical_function=historical,
            fund_data_function=self._collect_fund_data,
            ratios_function=ratios,
            start_date=self._start_date,
            end_date=self._end_date,
            rounding=self._rounding,
            currency_function=currencies,
        )

    @property
    def portfolio(self) -> "PortfolioConstruction":
        """
        This gives access to the Portfolio Construction module, which builds and examines
        portfolios from the assets of this Toolkit: optimal weights (minimum variance,
        maximum Sharpe ratio, mean-variance, risk parity, Hierarchical Risk Parity,
        inverse volatility, minimum CVaR and Black-Litterman), the efficient frontier, risk
        contributions, historical stress tests, look-through exposure of funds and a
        Brinson-Fachler attribution against the benchmark.

        When the Toolkit comes from a Portfolio (`Portfolio.toolkit`), the methods that take
        weights use the Portfolio's latest weights by default.

        See the following link for more information: https://www.jeroenbouma.com/projects/financetoolkit/docs/portfolio-optimization

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "JNJ", "XOM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.portfolio.get_risk_parity_weights()
        ```
        """
        # Imported when first used, so the Finance Toolkit loads only what is needed.
        from financetoolkit.portfolio.construction_controller import (  # noqa: PLC0415
            PortfolioConstruction,
        )

        # The prices and the risk free rate of a period are retrieved when a calculation
        # first needs them.
        historical_data = LazyDict(self._get_module_historical_data)
        risk_free_rate_data = LazyDict(
            lambda period: self._get_module_risk_free_rate(period)["Adj Close"]
        )

        def other_toolkit(tickers: list[str], start: str, end: str) -> "Toolkit":
            # Other tickers (sector funds, the benchmark, earlier crises) get a Toolkit of
            # their own that shares this one's key, cache and source.
            return Toolkit(
                tickers=tickers,
                api_key=self._api_key,
                start_date=start,
                end_date=end,
                use_cached_data=self._cache,
                enforce_source=self._enforce_source,
                benchmark_ticker=None,
                progress_bar=False,
            )

        def prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
            historical = other_toolkit(tickers, start, end).get_historical_data(
                period="daily"
            )
            if historical.empty:
                return pd.DataFrame(columns=tickers)
            adjusted = historical["Adj Close"]
            if isinstance(adjusted, pd.Series):
                adjusted = adjusted.to_frame(tickers[0])
            return adjusted

        def funds(kind: str, tickers: list[str]) -> pd.DataFrame:
            own = set(tickers) <= set(self._tickers)
            source = (
                self
                if own
                else other_toolkit(tickers, self._start_date, self._end_date)
            )
            if kind == "sector":
                result = source._collect_fund_data("sector_weightings")
            elif kind == "country":
                result = source._collect_fund_data("country_weightings")
            else:
                holdings = source._collect_fund_data("holdings")
                if holdings is None or holdings.empty:
                    return pd.DataFrame()
                # One fund comes back indexed by holding, several by fund and holding.
                if not isinstance(holdings.index, pd.MultiIndex):
                    holdings = pd.concat({tickers[0]: holdings})
                weights = pd.to_numeric(holdings["Weight"], errors="coerce")
                result = pd.DataFrame(
                    {
                        fund: weights.loc[fund].groupby(level=0).sum()
                        for fund in holdings.index.get_level_values(0).unique()
                    }
                )
            if result is None:
                return pd.DataFrame()
            if isinstance(result, pd.Series):
                result = result.to_frame(tickers[0])
            return result

        portfolio = PortfolioConstruction(
            tickers=self._tickers,
            historical_data=historical_data,
            risk_free_rate_data=risk_free_rate_data,
            benchmark_name=self._benchmark_ticker,
            portfolio_weights=self._portfolio_weights,
            price_function=prices,
            profile_function=self.get_profile,
            fund_function=funds,
            rounding=self._rounding,
            start_date=self._start_date,
            end_date=self._end_date,
        )

        return portfolio

    @property
    def fixedincome(self) -> "FixedIncome":
        """
        This gives access to the Fixed Income module. This module contains a wide variety of fixed income
        related calculations such as the Effective Yield, the Macaulay Duration, the Modified Duration,
        the Convexity, the Yield to Maturity and models such as Black and Bachelier to valuate derivative
        instruments such as Swaptions.

        Next to that, it is also possible to acquire Central Bank Rates and ICE BofA Indices such as the
        ICE BofA US High Yield Index, the ICE BofA US Corporate Index and the ICE BofA US Treasury Index.

        Note that this class can also be directly accessed by importing the FixedIncome class directly via
        from financetoolkit import FixedIncome. This is useful if you only want to use the FixedIncome class
        and not the other classes within the Toolkit module.

        See the following link for more information: https://www.jeroenbouma.com/projects/financetoolkit/docs/fixedincome

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(
            start_date='2024-01-01',
            end_date='2024-01-15',
            fred_api_key='FRED_API_KEY',
        )

        fixedincome.get_ice_bofa_effective_yield(maturity=False)
        ```

        Which returns:

        | Date       |      AAA |       AA |        A |      BBB |       BB |        B |      CCC |
        |:-----------|---------:|---------:|---------:|---------:|---------:|---------:|---------:|
        | 2024-01-01 | nan      | nan      | nan      | nan      | nan      | nan      | nan      |
        | 2024-01-02 |   0.0459 |   0.0473 |   0.0509 |   0.0543 |   0.0622 |   0.0763 |   0.1333 |
        | 2024-01-03 |   0.0459 |   0.0474 |   0.051  |   0.0544 |   0.0634 |   0.0779 |   0.1358 |
        | 2024-01-04 |   0.0466 |   0.0481 |   0.0518 |   0.0551 |   0.0639 |   0.0784 |   0.1367 |
        | 2024-01-05 |   0.047  |   0.0485 |   0.0521 |   0.0554 |   0.0641 |   0.0787 |   0.137  |
        | 2024-01-08 |   0.0465 |   0.0481 |   0.0517 |   0.055  |   0.0633 |   0.0776 |   0.1365 |
        | 2024-01-09 |   0.0464 |   0.048  |   0.0516 |   0.0548 |   0.0629 |   0.0771 |   0.1359 |
        | 2024-01-10 |   0.0464 |   0.048  |   0.0515 |   0.0547 |   0.0622 |   0.0762 |   0.1351 |
        | 2024-01-11 |   0.0456 |   0.0472 |   0.0507 |   0.054  |   0.0619 |   0.076  |   0.1344 |
        | 2024-01-12 |   0.0451 |   0.0467 |   0.0502 |   0.0534 |   0.0613 |   0.0753 |   0.1338 |
        | 2024-01-15 |   0.0451 |   0.0467 |   0.0501 |   0.0533 |   0.0611 |   0.0751 |   0.1328 |
        """
        # Imported when first used, so the Finance Toolkit loads only what is needed.
        from financetoolkit.fixedincome.fixedincome_controller import (  # noqa: PLC0415
            FixedIncome,
        )

        return FixedIncome(
            start_date=self._start_date,
            end_date=self._end_date,
            quarterly=self._quarterly,
            rounding=self._rounding,
            fred_api_key=self._fred_api_key,
            api_key=self._api_key,
            cache=self._cache,
        )

    @property
    def economics(self) -> "Economics":
        """
        This gives access to the Economics module. This module contains a wide variety of economic data
        obtained from OECD. These include things such as the Consumer Price Index (CPI), the Producer
        Price Index (PPI), the Unemployment Rate, the GDP Growth Rate, the Long and Short Term Interest
        Rate and the Consumer Confidence Index.

        Note that this class can also be directly accessed by importing the Economics class directly via
        from financetoolkit import Economics. This is useful if you only want to use the Economics class
        and not the other classes within the Toolkit module.

        See the following link for more information: https://www.jeroenbouma.com/projects/financetoolkit/docs/economics

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "ASML"], start_date="2021-01-01", end_date="2025-12-31")

        cpi = toolkit.economics.get_consumer_price_index(period='yearly')

        cpi.loc['2015':, ['United States', 'Netherlands', 'Japan']]
        ```

        Which returns:

        |      |   United States |   Netherlands |   Japan |
        |:-----|----------------:|--------------:|--------:|
        | 2021 |         114.325 |       110.389 | 101.567 |
        | 2022 |         123.474 |       121.426 | 104.112 |
        | 2023 |         128.557 |       126.092 | 107.503 |
        | 2024 |         132.349 |       130.311 | 110.457 |
        | 2025 |         135.831 |       134.486 | 113.98  |
        """
        # Imported when first used, so the Finance Toolkit loads only what is needed.
        from financetoolkit.economics.economics_controller import (  # noqa: PLC0415
            Economics,
        )

        return Economics(
            start_date=self._start_date,
            end_date=self._end_date,
            quarterly=self._quarterly,
            rounding=self._rounding,
            fred_api_key=self._fred_api_key,
            allow_stale_oecd_cache=self._allow_stale_oecd_cache,
            cache=self._cache,
            api_key=self._api_key,
        )

    def get_profile(self):
        """
        Obtain the profile of the specified tickers. These include important metrics
        such as the beta, market capitalization, currency, isin, industry, and ipo date
        that give an overall understanding about the company.

        Also known as: company description, sector, industry, CEO, employee count.

        Args:

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["ASML.AS", "SAP.DE", "MC.PA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.get_profile().loc[["Industry", "Exchange", "Currency", "ISIN", "Beta"]]
        ```

        Which returns:

        |          | ASML.AS        | SAP.DE                 | MC.PA        |
        |:---------|:---------------|:-----------------------|:-------------|
        | Industry | Semiconductors | Software - Application | Luxury Goods |
        | Exchange | AMS            | XETRA                  | PAR          |
        | Currency | EUR            | EUR                    | EUR          |
        | ISIN     | NL0010273215   | DE0007164600           | FR0000121014 |
        | Beta     | 1.302          | 0.775                  | 0.879        |
        """
        if not self._api_key:
            logger.error(
                "The requested data requires the api_key parameter to be set, consider "
                "obtaining a key with the following link: "
                "https://www.jeroenbouma.com/fmp"
                "\nThe free plan allows for 250 requests per day, a limit of 5 years and has no "
                "quarterly data. Consider upgrading your plan. You can get 15% off by using the "
                "above affiliate link which also supports the project."
            )
            return None

        if self._profile.empty:
            self._profile, self._invalid_tickers = self._collect_per_ticker(
                dataset="profile",
                tickers=self._tickers,
                ticker_axis=ticker_model.TICKER_ON_COLUMNS,
                collector=lambda tickers: _get_profile(
                    tickers=tickers,
                    api_key=self._api_key,
                    user_subscription=self._fmp_plan,
                ),
            )

        if self._remove_invalid_tickers:
            self._tickers = [
                ticker
                for ticker in self._tickers
                if ticker not in self._invalid_tickers
            ]

        return self._profile

    def get_quote(self):
        """
        Get the quote of the specified tickers: the price and its change, the day's
        open, low and high, the previous close, the year low and high, the 50 and 200
        day average prices, the volume, the market capitalization, the exchange and the
        time of the quote.

        Also known as: real-time price, current stock price, live quote.

        Args:

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["V", "MA", "AXP"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.get_quote().loc[["Price", "Year High", "Year Low"]]
        ```

        Which returns:

        |           |      V |     MA |    AXP |
        |:----------|-------:|-------:|-------:|
        | Price     | 407.92 | 623.6  | 313.03 |
        | Year High | 414.6  | 633.29 | 387.49 |
        | Year Low  | 293.89 | 464.52 | 290.97 |
        """
        if not self._api_key:
            logger.error(
                "The requested data requires the api_key parameter to be set, consider "
                "obtaining a key with the following link: "
                "https://www.jeroenbouma.com/fmp"
                "\nThe free plan allows for 250 requests per day, a limit of 5 years and has no "
                "quarterly data. Consider upgrading your plan. You can get 15% off by using the "
                "above affiliate link which also supports the project."
            )
            return None

        if self._quote.empty:
            self._quote, self._invalid_tickers = self._collect_per_ticker(
                dataset="quote",
                tickers=self._tickers,
                ticker_axis=ticker_model.TICKER_ON_COLUMNS,
                collector=lambda tickers: _get_quote(
                    tickers=tickers,
                    api_key=self._api_key,
                    user_subscription=self._fmp_plan,
                ),
            )

        if self._remove_invalid_tickers:
            self._tickers = [
                ticker
                for ticker in self._tickers
                if ticker not in self._invalid_tickers
            ]

        return self._quote

    def get_rating(self):
        """
        Get the rating of the specified tickers: a letter rating and an overall score based on
        six scores from 1 (weak) to 5 (strong). These scores are categorized as follows:

        - An overall rating (the letter rating and the Rating Score)
        - Discounted Cash Flow (DCF)
        - Return on Equity (ROE)
        - Return on Assets (ROA)
        - Debt to Equity (DE)
        - Price Earnings (PE)
        - Price to Book (PB)

        Also known as: stock rating, fundamental score, quantitative rating.

        Args:

        Returns:
            pd.DataFrame: The stock rating information for the specified tickers.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["V"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        rating = toolkit.get_rating()

        rating[["DCF Score", "ROE Score", "ROA Score", "DE Score", "PE Score", "PB Score"]].tail(1)
        ```

        Which returns:

        | date                |   DCF Score |   ROE Score |   ROA Score |   DE Score |   PE Score |   PB Score |
        |:--------------------|------------:|------------:|------------:|-----------:|-----------:|-----------:|
        | 2025-12-31 00:00:00 |           4 |           5 |           5 |          2 |          1 |          1 |
        """
        if not self._api_key:
            logger.error(
                "The requested data requires the api_key parameter to be set, consider "
                "obtaining a key with the following link: "
                "https://www.jeroenbouma.com/fmp"
                "\nThe free plan allows for 250 requests per day, a limit of 5 years and has no "
                "quarterly data. Consider upgrading your plan. You can get 15% off by using the "
                "above affiliate link which also supports the project."
            )
            return None

        if self._rating.empty:
            self._rating, self._invalid_tickers = self._collect_per_ticker(
                dataset="rating",
                tickers=self._tickers,
                ticker_axis=ticker_model.TICKER_ON_INDEX,
                parameters={"user_subscription": self._fmp_plan},
                collector=lambda tickers: _get_rating(
                    tickers=tickers,
                    api_key=self._api_key,
                    user_subscription=self._fmp_plan,
                ),
            )

        if self._remove_invalid_tickers:
            self._tickers = [
                ticker
                for ticker in self._tickers
                if ticker not in self._invalid_tickers
            ]

        # The full history is kept, and the ratings between the start and end date returned.
        rating = self._rating
        if isinstance(rating, pd.DataFrame) and not rating.empty:
            dates = pd.Series(rating.index.get_level_values(-1).astype(str).str[:10])
            keep = dates.ge(self._start_date or "") & dates.le(self._end_date or "9999")
            rating = rating[keep.to_numpy()]

            if len(self._tickers) == 1 and self._tickers[0] in rating.index:
                rating = rating.loc[self._tickers[0]]

        return rating

    def get_analyst_estimates(
        self,
        overwrite: bool = False,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        show_columns: list[str] | None = None,
    ):
        """
        Obtain analyst estimates regarding revenues, EBITDA, EBIT, Net Income
        SGA Expenses and EPS. The number of analysts are also reported.

        Note that this information requires a Premium FMP subscription.

        Also known as: earnings estimates, revenue estimates, analyst consensus.

        Args:
            overwrite (bool, optional): Defines whether to overwrite the existing data. Defaults to False.
            rounding (int | None, optional): Defines the number of decimal places to round the data to. Defaults to None.
            growth (bool, optional): Defines whether to return the growth of the data. Defaults to False.
            lag (int | list[int], optional): Defines the number of periods to lag the growth data by. Defaults to 1.
            show_columns (list[str] | None): A list of column names to keep in the result. Invalid
            names are reported and ignored. Defaults to None, which keeps every column.

        Returns:
            pandas.DataFrame: The analyst estimates for the specified tickers.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["NVDA", "AMD", "AVGO"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2026-01-01",
            end_date="2028-12-31",
        )

        toolkit.get_analyst_estimates(show_columns=["Estimated Revenue Average"])
        ```

        Which returns:

        |      |        2026 |        2027 |        2028 |        2029 |        2030 |          2031 |
        |:-----|------------:|------------:|------------:|------------:|------------:|--------------:|
        | NVDA | 2.13656e+11 | 4.09929e+11 | 6.95178e+11 | 9.25214e+11 | 1.10649e+12 |   1.21149e+12 |
        | AMD  | 5.13395e+10 | 9.04101e+10 | 1.25666e+11 | 1.6376e+11  | 2.32364e+11 | nan           |
        | AVGO | 1.06009e+11 | 1.74279e+11 | 2.81861e+11 | 4.16476e+11 | 5.52387e+11 | nan           |
        """
        if not self._api_key:
            logger.error(
                "The requested data requires the api_key parameter to be set, consider obtaining "
                "a key with the following link: https://www.jeroenbouma.com/fmp"
                "\nThis functionality also requires a Premium subscription. You can get 15% off by "
                "using the above affiliate link which also supports the project."
            )
            return None

        if self._analyst_estimates.empty or overwrite:
            (
                self._analyst_estimates,
                self._invalid_tickers,
            ) = self._collect_per_ticker(
                dataset="analyst_estimates",
                tickers=self._tickers,
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
                    rounding=rounding if rounding is not None else self._rounding,
                    sleep_timer=self._sleep_timer,
                    user_subscription=self._fmp_plan,
                ),
            )

        if self._remove_invalid_tickers:
            self._tickers = [
                ticker
                for ticker in self._tickers
                if ticker not in self._invalid_tickers
            ]

        if growth:
            self._analyst_estimates_growth = calculate_growth(
                self._analyst_estimates,
                lag=lag,
                rounding=rounding if rounding is not None else self._rounding,
            )

        if len(self._tickers) == 1 and not self._analyst_estimates.empty:
            result = (
                self._analyst_estimates_growth.loc[self._tickers[0]]
                if growth
                else self._analyst_estimates.loc[self._tickers[0]]
            )
            return filter_columns(result, show_columns)

        result = self._analyst_estimates_growth if growth else self._analyst_estimates
        return filter_columns(result, show_columns)

    def get_earnings_calendar(
        self,
        actual_dates: bool = True,
        overwrite: bool = False,
        rounding: int | None = None,
        show_columns: list[str] | None = None,
    ):
        """
        Obtain Earnings Calendars for any range of companies. You have the option to
        obtain the actual dates or to convert to the corresponding quarters.

        Note that this information requires a Premium FMP subscription.

        The surprise is the reported EPS against the estimate, usually as a percentage of the
        estimate. Most companies report above the consensus, so a small beat is the norm, and
        the share price tends to react to the size of the surprise, to revenue against its
        estimate and to the guidance given on the call rather than to a beat as such.

        Also known as: earnings dates, earnings schedule, reporting date.

        Args:
            actual_dates (bool): Defines whether to return the actual dates or the corresponding quarters.
            overwrite (bool): Defines whether to overwrite the existing data.
            show_columns (list[str] | None): A list of column names to keep in the result. Invalid
            names are reported and ignored. Defaults to None, which keeps every column.
            rounding (int | None): The number of decimals to round the results to. Defaults to None,
            which uses the rounding set on the Toolkit.

        Returns:
            pd.DataFrame: The earnings calendar for the specified tickers.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["COST"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2024-01-01",
            end_date="2025-12-31",
        )

        toolkit.get_earnings_calendar(show_columns=["EPS", "Estimated EPS"])
        ```

        Which returns:

        | date                |   EPS |   Estimated EPS |
        |:--------------------|------:|----------------:|
        | 2024-03-07 00:00:00 |  3.92 |            3.62 |
        | 2024-05-30 00:00:00 |  3.78 |            3.7  |
        | 2024-09-26 00:00:00 |  5.29 |            5.08 |
        | 2024-12-12 00:00:00 |  4.04 |            3.79 |
        | 2025-03-06 00:00:00 |  4.02 |            4.09 |
        | 2025-05-29 00:00:00 |  4.28 |            4.24 |
        | 2025-09-25 00:00:00 |  5.87 |            5.8  |
        | 2025-12-11 00:00:00 |  4.5  |            4.27 |
        """
        if not self._api_key:
            logger.error(
                "The requested data requires the api_key parameter to be set, consider obtaining a key with the "
                "following link: https://www.jeroenbouma.com/fmp"
                "\nThis functionality also requires a Premium subscription. You can get 15% off by using "
                "the above affiliate link which also supports the project."
            )
            return None

        if self._earnings_calendar.empty or overwrite:
            (
                self._earnings_calendar,
                self._invalid_tickers,
            ) = self._collect_per_ticker(
                dataset="earnings_calendar",
                tickers=self._tickers,
                ticker_axis=ticker_model.TICKER_ON_INDEX,
                parameters={
                    "start_date": self._start_date,
                    "end_date": self._end_date,
                    "actual_dates": actual_dates,
                    "user_subscription": self._fmp_plan,
                },
                collector=lambda tickers: _get_earnings_calendar(
                    tickers=tickers,
                    api_key=self._api_key,
                    start_date=self._start_date,
                    end_date=self._end_date,
                    actual_dates=actual_dates,
                    sleep_timer=self._sleep_timer,
                    user_subscription=self._fmp_plan,
                ),
            )

        earnings_calendar = apply_rounding(
            self._earnings_calendar,
            rounding if rounding is not None else self._rounding,
        )

        if self._remove_invalid_tickers:
            self._tickers = [
                ticker
                for ticker in self._tickers
                if ticker not in self._invalid_tickers
            ]

        if len(self._tickers) == 1 and not self._earnings_calendar.empty:
            return filter_columns(earnings_calendar.loc[self._tickers[0]], show_columns)

        return filter_columns(earnings_calendar, show_columns)

    def get_stock_news(
        self,
        pages: int = 1,
        limit: int = 100,
        show_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Obtain the latest news articles for the tickers of this Toolkit instance, whether
        they are stocks, cryptocurrencies or currency pairs. Qualitative companion to the
        toolkit's quantitative data. Automatically filtered to this Toolkit instance's
        start_date and end_date.

        Each ticker is matched with the right news feed. A currency pair such as EURUSD or
        EURUSD=X is searched in the forex news. Any other ticker is searched in both the
        stock and the crypto news, since a ticker such as BTCUSD cannot be told apart from
        a stock by its format; the feed it does not belong to simply returns nothing.

        Also known as: ticker news, company news feed, crypto news, forex news.

        Args:
            pages (int, optional): The number of pages to collect per news feed, each page is a
                separate API call, e.g. pages=5 makes 5 calls. Defaults to 1.
            limit (int, optional): The number of articles to return per page. Defaults to 100.
            show_columns (list[str] | None): A list of column names to keep in the result. Invalid
            names are reported and ignored. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The latest news articles for the specified tickers, newest first.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        stock_news = toolkit.get_stock_news(limit=5)

        stock_news[["Symbol", "Publisher", "Title"]]
        ```

        Which returns:

        | Published Date      | Symbol   | Publisher       | Title                                                               |
        |:--------------------|:---------|:----------------|:--------------------------------------------------------------------|
        | 2025-12-31 17:34:00 | AAPL     | GuruFocus       | Market Today: Buffett era ends; Tesla warns; stocks cap strong 2025 |
        | 2025-12-31 16:11:00 | AAPL     | The Motley Fool | Here Are My Top 2 Stocks to Buy for 2026 and Beyond                 |
        | 2025-12-31 14:35:53 | AAPL     | Schwab Network  | Ca$htag$: Apple (AAPL) Strong Holiday Season                        |
        | 2025-12-31 14:15:00 | MSFT     | The Motley Fool | The Best Tech Stocks to Buy in January for 2026 Gains               |
        | 2025-12-31 14:14:48 | AAPL     | CNBC Television | Apple's AI challenges in 2026                                       |
        """
        currency_pairs = [
            ticker
            for ticker in self._tickers
            if currencies_model.is_currency_pair(ticker)
        ]
        other_tickers = [
            ticker for ticker in self._tickers if ticker not in currency_pairs
        ]
        search = {
            "api_key": self._api_key,
            "limit": limit,
            "pages": pages,
            "start_date": self._start_date,
            "end_date": self._end_date,
            "user_subscription": self._fmp_plan,
        }

        news_frames = []

        if other_tickers:
            # The crypto feed is asked first because it only returns crypto tickers, which
            # are then left out of the stock feed request. The stock feed carries crypto
            # articles as well, so asking it for both would let them use up its limit.
            crypto_news = _search_crypto_news(symbols=other_tickers, **search)
            crypto_tickers = (
                set(crypto_news["Symbol"]) if not crypto_news.empty else set()
            )
            stock_tickers = [
                ticker for ticker in other_tickers if ticker not in crypto_tickers
            ]

            news_frames.append(crypto_news)

            if stock_tickers:
                news_frames.append(_search_stock_news(symbols=stock_tickers, **search))

        if currency_pairs:
            # The forex feed knows the pairs without the "=X" suffix Yahoo Finance uses.
            news_frames.append(
                _search_forex_news(
                    symbols=[
                        pair.upper().removesuffix("=X") for pair in currency_pairs
                    ],
                    **search,
                )
            )

        news_frames = [frame for frame in news_frames if not frame.empty]

        if not news_frames:
            return pd.DataFrame()

        # A crypto ticker without recent crypto news is still asked of the stock feed, so
        # an article can come back from both feeds; it is kept once.
        stock_news = pd.concat(news_frames).sort_index(ascending=False)
        stock_news = stock_news[~stock_news.duplicated(subset=["Symbol", "URL"])]

        return filter_columns(stock_news, show_columns)

    def get_press_releases(
        self,
        pages: int = 1,
        limit: int = 100,
        show_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Obtain the latest official company press releases for the tickers of this
        Toolkit instance, such as earnings announcements and corporate communications.
        Automatically filtered to this Toolkit instance's start_date and end_date.

        Also known as: corporate announcements, company press release feed.

        Args:
            pages (int, optional): The number of pages to collect, each page is a
                separate API call, e.g. pages=5 makes 5 calls. Defaults to 1.
            limit (int, optional): The number of articles to return per page. Defaults to 100.
            show_columns (list[str] | None): A list of column names to keep in the result. Invalid
            names are reported and ignored. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The latest press releases for the specified tickers.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        press_releases = toolkit.get_press_releases(limit=5)

        press_releases[["Symbol", "Publisher", "Title"]]
        ```

        Which returns:

        | Published Date      | Symbol   | Publisher     | Title                                                                                                 |
        |:--------------------|:---------|:--------------|:------------------------------------------------------------------------------------------------------|
        | 2025-12-18 09:00:00 | MSFT     | PRNewsWire    | Cognizant and Microsoft Expand Partnership to Advance AI Transformation and Frontier Firm Experiences |
        | 2025-12-18 04:00:00 | MSFT     | Business Wire | Reply Recognized as a Microsoft Azure Expert Managed Services Provider for the Sixth Consecutive Year |
        | 2025-12-17 20:00:00 | AAPL     | Business Wire | Apple announces changes to iOS in Japan                                                               |
        | 2025-12-17 16:30:00 | MSFT     | Accesswire    | ProsperOps Achieves Microsoft Azure IP Co-Sell Status                                                 |
        | 2025-12-17 08:00:00 | MSFT     | Business Wire | EcoVadis Wins Microsoft Local Partner Award FY25 in AI Transformation - Scale Category                |
        """
        press_releases = _search_press_releases(
            api_key=self._api_key,
            symbols=self._tickers,
            limit=limit,
            pages=pages,
            start_date=self._start_date,
            end_date=self._end_date,
            user_subscription=self._fmp_plan,
        )

        return filter_columns(press_releases, show_columns)

    def get_revenue_geographic_segmentation(
        self,
        overwrite: bool = False,
        show_columns: list[str] | None = None,
    ):
        """
        Obtain revenue by geographic segmentation (e.g. United States, Europe, Asia).

        Note that this information requires a Premium FMP subscription.

        Also known as: revenue by region, geographic revenue breakdown.

        Args:
            overwrite (bool): Defines whether to overwrite the existing data.
            show_columns (list[str] | None): A list of column names to keep in the result. Invalid
            names are reported and ignored. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The revenue by geographic segmentation for the specified tickers.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.get_revenue_geographic_segmentation()
        ```

        Which returns:

        |              |        2021 |        2022 |       2023 |        2024 |        2025 |
        |:-------------|------------:|------------:|-----------:|------------:|------------:|
        | Americas     | 1.53306e+11 | 1.69658e+11 | 1.6256e+11 | 1.67045e+11 | 1.78353e+11 |
        | Asia Pacific | 2.6356e+10  | 2.9375e+10  | 2.9615e+10 | 3.0658e+10  | 3.3696e+10  |
        | China        | 6.8366e+10  | 7.42e+10    | 7.2559e+10 | 6.6952e+10  | 6.4377e+10  |
        | Europe       | 8.9307e+10  | 9.5118e+10  | 9.4294e+10 | 1.01328e+11 | 1.11032e+11 |
        | Japan        | 2.8482e+10  | 2.5977e+10  | 2.4257e+10 | 2.5052e+10  | 2.8703e+10  |
        """
        if not self._api_key:
            logger.error(
                "The requested data requires the api_key parameter to be set, consider obtaining a key with the "
                "following link: https://www.jeroenbouma.com/fmp"
                "\nThis functionality also requires a Professional or Enterprise subscription. "
                "You can get 15% off by using the above affiliate link which also supports the project."
            )
            return None

        if self._revenue_geographic_segmentation.empty or overwrite:
            (
                self._revenue_geographic_segmentation,
                self._invalid_tickers,
            ) = self._collect_per_ticker(
                dataset="revenue_geographic_segmentation",
                tickers=self._tickers,
                ticker_axis=ticker_model.TICKER_ON_INDEX,
                parameters={
                    "quarter": (
                        self._quarterly if self._fmp_plan == "Premium" else False
                    ),
                    "start_date": self._start_date,
                    "end_date": self._end_date,
                },
                collector=lambda tickers: _get_revenue_segmentation(
                    tickers=tickers,
                    method="geographic",
                    api_key=self._api_key,
                    quarter=self._quarterly if self._fmp_plan == "Premium" else False,
                    start_date=self._start_date,
                    end_date=self._end_date,
                    sleep_timer=self._sleep_timer,
                    user_subscription=self._fmp_plan,
                ),
            )

        if self._remove_invalid_tickers:
            self._tickers = [
                ticker
                for ticker in self._tickers
                if ticker not in self._invalid_tickers
            ]

        if len(self._tickers) == 1 and not self._revenue_geographic_segmentation.empty:
            return filter_columns(
                self._revenue_geographic_segmentation.loc[self._tickers[0]],
                show_columns,
            )

        return filter_columns(self._revenue_geographic_segmentation, show_columns)

    def get_revenue_product_segmentation(
        self,
        overwrite: bool = False,
        show_columns: list[str] | None = None,
    ):
        """
        Obtain revenue by product segmentation (e.g. iPad, Advertisement, Windows).

        Note that this information requires a Premium FMP subscription.

        Also known as: revenue by product, product segment revenue.

        Args:
            overwrite (bool): Defines whether to overwrite the existing data.
            show_columns (list[str] | None): A list of column names to keep in the result. Invalid
            names are reported and ignored. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The revenue by product segmentation for the specified tickers.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["NVDA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.get_revenue_product_segmentation()
        ```

        Which returns:

        |                            |      2021 |       2022 |       2023 |       2024 |        2025 |
        |:---------------------------|----------:|-----------:|-----------:|-----------:|------------:|
        | Automotive                 | 5.36e+08  | 5.66e+08   | 9.03e+08   | 1.091e+09  | 1.694e+09   |
        | Data Center                | 6.696e+09 | 1.0613e+10 | 1.5005e+10 | 4.7525e+10 | 1.15186e+11 |
        | Gaming                     | 7.759e+09 | 1.2462e+10 | 9.067e+09  | 1.0447e+10 | 1.135e+10   |
        | Oem And Other              | 6.31e+08  | 1.162e+09  | 4.55e+08   | 3.06e+08   | 3.89e+08    |
        | Professional Visualization | 1.053e+09 | 2.111e+09  | 1.544e+09  | 1.553e+09  | 1.878e+09   |
        """
        if not self._api_key:
            logger.error(
                "The requested data requires the api_key parameter to be set, consider obtaining a key with the "
                "following link: https://www.jeroenbouma.com/fmp"
                "\nThis functionality also requires a Professional or Enterprise subscription. You can get 15% off by using "
                "the above affiliate link which also supports the project."
            )
            return None

        if self._revenue_product_segmentation.empty or overwrite:
            (
                self._revenue_product_segmentation,
                self._invalid_tickers,
            ) = self._collect_per_ticker(
                dataset="revenue_product_segmentation",
                tickers=self._tickers,
                ticker_axis=ticker_model.TICKER_ON_INDEX,
                parameters={
                    "quarter": (
                        self._quarterly if self._fmp_plan == "Premium" else False
                    ),
                    "start_date": self._start_date,
                    "end_date": self._end_date,
                },
                collector=lambda tickers: _get_revenue_segmentation(
                    tickers=tickers,
                    method="product",
                    api_key=self._api_key,
                    quarter=self._quarterly if self._fmp_plan == "Premium" else False,
                    start_date=self._start_date,
                    end_date=self._end_date,
                    sleep_timer=self._sleep_timer,
                    user_subscription=self._fmp_plan,
                ),
            )

        if self._remove_invalid_tickers:
            self._tickers = [
                ticker
                for ticker in self._tickers
                if ticker not in self._invalid_tickers
            ]

        if len(self._tickers) == 1 and not self._revenue_product_segmentation.empty:
            return filter_columns(
                self._revenue_product_segmentation.loc[self._tickers[0]], show_columns
            )

        return filter_columns(self._revenue_product_segmentation, show_columns)

    def get_historical_data(
        self,
        enforce_source: str | None = None,
        period: str = "daily",
        return_column: str = "Adj Close",
        include_dividends: bool = True,
        fill_nan: bool = True,
        overwrite: bool = False,
        rounding: int | None = None,
        show_ticker_seperation: bool = True,
        show_columns: list[str] | None = None,
    ):
        """
        Returns historical data for the specified tickers. This contains the following columns:
            - Open: The opening price for the period.
            - High: The highest price for the period.
            - Low: The lowest price for the period.
            - Close: The closing price for the period.
            - Adj Close: The adjusted closing price for the period.
            - Volume: The volume for the period.
            - Dividends: The dividends for the period.
            - Return: The return for the period.
            - Cumulative Return: The cumulative return for the period.

        Volatility, Excess Return and Excess Volatility are not included here. These are available
        as dedicated calculations in the Risk module (e.g. toolkit.risk.get_volatility,
        toolkit.risk.get_excess_volatility) and the Performance module (e.g.
        toolkit.performance.get_excess_return) instead.

        If a benchmark ticker is selected, it also calculates the benchmark ticker together with the results.
        By default this is set to "SPY" (S&P 500 Index) but can be any ticker. This is relevant for calculations
        for models such as CAPM, Alpha and Beta.

        Important to note is that when an api_key is included in the Toolkit initialization that the data
        collection defaults to FinancialModelingPrep which is a more stable source and utilises your subscription.
        However, if this is undesired, it can be disabled by setting enforce_source to "YahooFinance". If
        data collection fails from FinancialModelingPrep it automatically reverts back to YahooFinance.

        Also known as: OHLCV, price history, open high low close volume.

        Args:
            enforce_source (str, optional): A string containing the historical source you wish to enforce.
            This can be either FinancialModelingPrep or YahooFinance. Defaults to no enforcement.
            period (str): The interval at which the historical data should be
            returned - daily, weekly, monthly, quarterly, or yearly.
            Defaults to "daily".
            return_column (str): The column to use for the return calculation. Defaults to "Adj Close".
            include_dividends (bool): Defines whether to include dividends in the return calculation.
            Defaults to True.
            fill_nan (bool): Defines whether to forward fill NaN values. This defaults
            to True to prevent holes in the dataset. This is especially relevant for
            technical indicators.
            overwrite (bool): Defines whether to overwrite the existing data. If this is not enabled, the function
            will return the earlier retrieved data. This is done to prevent too many API calls. Defaults to False.
            rounding (int): Defines the number of decimal places to round the data to.
            show_ticker_seperation (bool, optional): A boolean representing whether to show which tickers
            acquired data from FinancialModelingPrep and which tickers acquired data from YahooFinance.
            show_columns (list[str], optional): A list of columns to include in the output. Valid columns
            are Open, High, Low, Close, Adj Close, Volume, Dividends, Return and Cumulative Return.
            Invalid column names are logged as warnings. If all
            provided columns are invalid the full dataset is returned. Defaults to None (all columns).

        Raises:
            ValueError: If an invalid value is specified for period.

        Returns:
            pandas.DataFrame: The historical data for the specified tickers.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["CAT", "DE", "URI"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        toolkit.get_historical_data(show_columns=["Close"]).tail()
        ```

        Which returns:

        | Date       |    CAT |     DE |    URI |   Benchmark |
        |:-----------|-------:|-------:|-------:|------------:|
        | 2025-12-24 | 583.76 | 467.44 | 824.77 |      690.38 |
        | 2025-12-26 | 583    | 468.93 | 827.94 |      690.31 |
        | 2025-12-29 | 578.71 | 470.24 | 819.51 |      687.85 |
        | 2025-12-30 | 577.39 | 469.94 | 817.25 |      687.01 |
        | 2025-12-31 | 572.87 | 465.57 | 809.32 |      681.92 |
        """
        if enforce_source is not None and enforce_source not in [
            "FinancialModelingPrep",
            "YahooFinance",
        ]:
            raise ValueError(
                "The enforce_source parameter must be either 'FinancialModelingPrep' or 'YahooFinance'."
            )

        if self._daily_risk_free_rate.empty or overwrite:
            self.get_treasury_data(
                risk_free_rate=self._risk_free_rate,
                show_errors=False,
                fill_nan=fill_nan,
            )

        resolved_enforce_source = (
            enforce_source if enforce_source is not None else self._enforce_source
        )
        resolved_rounding = rounding if rounding is not None else self._rounding
        daily_historical_params = (
            resolved_enforce_source,
            return_column,
            include_dividends,
            fill_nan,
            resolved_rounding,
        )
        # Auto-invalidate only data we fetched ourselves before, on a param change; never touches pre-supplied `historical`.
        params_changed = (
            self._daily_historical_data_params is not None
            and self._daily_historical_data_params != daily_historical_params
        )

        if self._daily_historical_data.empty or overwrite or params_changed:
            self._daily_historical_data, self._invalid_tickers = _get_historical_data(
                tickers=(
                    self._tickers + [self._benchmark_ticker]
                    if self._benchmark_ticker
                    else self._tickers
                ),
                api_key=self._api_key,
                enforce_source=resolved_enforce_source,
                start=self._start_date,
                end=self._end_date,
                interval="1d",
                return_column=return_column,
                include_dividends=include_dividends,
                fill_nan=fill_nan,
                rounding=resolved_rounding,
                sleep_timer=self._sleep_timer,
                show_ticker_seperation=show_ticker_seperation,
                show_errors=True,
                user_subscription=self._fmp_plan,
                cache=self._cache,
            )
            self._daily_historical_data_params = daily_historical_params

            # Change the benchmark ticker name to Benchmark
            if not self._daily_historical_data.empty:
                self._daily_historical_data = self._daily_historical_data.rename(
                    columns={self._benchmark_ticker: "Benchmark"}, level=1
                )

        if self._remove_invalid_tickers:
            self._tickers = [
                ticker
                for ticker in self._tickers
                if ticker not in self._invalid_tickers
            ]

        if self._daily_historical_data.empty:
            return pd.DataFrame()

        # Named as the other periods are, whichever is requested first.
        if self._daily_historical_data.index.name != "Date":
            self._daily_historical_data = self._daily_historical_data.rename_axis(
                "Date"
            )

        if period == "daily":
            historical_data = self._daily_historical_data.loc[
                self._start_date : self._end_date, :
            ]
            # The first row of the window has no preceding observation, so its Return stays NaN; Cumulative Return is already anchored at 1 there by its own calculation, so nothing depends on fabricating a zero.

        elif period in ("weekly", "monthly", "quarterly", "yearly"):
            if getattr(self, f"_{period}_risk_free_rate").empty or overwrite:
                self.get_treasury_data(
                    period=period, risk_free_rate=self._risk_free_rate
                )

            # Every module (risk, performance, models, ...) asks for a period again each
            # time it is accessed, so the conversion is kept until the daily data it was made
            # from or one of its settings changes. The daily data is compared by identity, as
            # it is replaced rather than changed when it is retrieved again.
            settings = (
                self._start_date,
                self._end_date,
                rounding if rounding is not None else self._rounding,
                return_column,
            )
            converted_from = self._period_historical_data_sources.get(period)

            if (
                overwrite
                or getattr(self, f"_{period}_historical_data").empty
                or converted_from is None
                or converted_from[0] is not self._daily_historical_data
                or converted_from[1] != settings
            ):
                setattr(
                    self,
                    f"_{period}_historical_data",
                    _convert_daily_to_other_period(
                        period=period,
                        daily_historical_data=self._daily_historical_data,
                        start=self._start_date,
                        end=self._end_date,
                        rounding=settings[2],
                        return_column=return_column,
                    ),
                )
                self._period_historical_data_sources[period] = (
                    self._daily_historical_data,
                    settings,
                )

            historical_data = getattr(self, f"_{period}_historical_data").loc[
                self._start_date : self._end_date, :
            ]
            # The first row of the window has no preceding observation, so its Return stays NaN; Cumulative Return is already anchored at 1 there by its own calculation, so nothing depends on fabricating a zero.

        else:
            raise ValueError(
                "Please choose from daily, weekly, monthly, quarterly or yearly as period."
            )

        requested: list[str] = []

        if show_columns is not None:
            valid_columns = (
                historical_data.columns.get_level_values(0).unique().tolist()
            )
            invalid = [c for c in show_columns if c not in valid_columns]
            for col in invalid:
                logger.warning(
                    f"Column '{col}' is not a valid column for get_historical_data. "
                    f"Valid columns are: {valid_columns}"
                )
            requested = [c for c in show_columns if c in valid_columns]
            if requested:
                mask = historical_data.columns.get_level_values(0).isin(requested)
                historical_data = historical_data.loc[:, mask]

        if len(self._tickers) == 1 and not self._benchmark_ticker:
            result = historical_data.xs(self._tickers[0], level=1, axis="columns")
            if len(requested) == 1:
                return result.iloc[:, 0]
            return result

        if len(requested) == 1:
            return historical_data.droplevel(0, axis="columns")

        return historical_data

    def get_intraday_data(
        self,
        period: str = "1hour",
        return_column: str = "Close",
        fill_nan: bool = True,
        rounding: int | None = None,
        show_columns: list[str] | None = None,
    ):
        """
        Returns intraday historical data for the specified tickers. This contains the following columns:
            - Open: The opening price for the period.
            - High: The highest price for the period.
            - Low: The lowest price for the period.
            - Close: The closing price for the period.
            - Volume: The volume for the period.
            - Return: The return for the period.
            - Cumulative Return: The cumulative return for the period.

        Volatility is not included here. This is available as a dedicated calculation in the
        Risk module instead (e.g. toolkit.risk.get_volatility).

        Keep in mind that this data is available for a shorter period. This means that the start date is
        ignored if the difference between the start and end date is bigger than the maximum period.

        If a benchmark ticker is selected, it also calculates the benchmark ticker together with the results.
        By default this is set to "SPY" (S&P 500 Index) but can be any ticker. This is relevant for calculations
        for models such as CAPM, Alpha and Beta.

        Please note that this functionality is only available through Financial Modeling Prep. Therefore, an
        api_key is required to use this functionality.

        Also known as: tick data, minute data, intraday price history.

        Args:
            period (str, optional): The intraday interval to fetch (e.g. "1min", "5min", "1hour").
                Defaults to "1hour".
            return_column (str, optional): The column to use for the return calculation. Defaults to "Close".
            fill_nan (bool, optional): Defines whether to forward fill NaN values. Defaults to True.
            rounding (int | None, optional): Defines the number of decimal places to round the data to. Defaults to None.
            show_columns (list[str] | None): A list of column names to keep in the result. Invalid
            names are reported and ignored. Defaults to None, which keeps every column.

        Returns:
            pandas.DataFrame: The intraday data for the specified tickers.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["NVDA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-12-29",
            end_date="2025-12-31",
        )

        toolkit.get_intraday_data(period="1hour", show_columns=["Close"]).tail()
        ```

        Which returns:

        | date             |   ('Close', 'NVDA') |   ('Close', 'Benchmark') |
        |:-----------------|--------------------:|-------------------------:|
        | 2025-12-31 11:00 |              188.92 |                   685    |
        | 2025-12-31 12:00 |              188.97 |                   685.35 |
        | 2025-12-31 13:00 |              188.05 |                   685.04 |
        | 2025-12-31 14:00 |              187.13 |                   682.69 |
        | 2025-12-31 15:00 |              186.53 |                   681.87 |
        """
        if not self._api_key:
            logger.error(
                "The requested data requires the api_key parameter to be set, consider obtaining a key with the "
                "following link: https://www.jeroenbouma.com/fmp"
                "\nThis functionality also requires a Professional or Enterprise subscription. You can get 15% off by using "
                "the above affiliate link which also supports the project."
            )
            return None

        if period not in ["1min", "5min", "15min", "30min", "1hour"]:
            raise ValueError(
                "Please choose from 1min, 5min, 15min, 30min or 1hour as period."
            )

        if self._intraday_period != period or self._intraday_historical_data.empty:
            (
                self._intraday_historical_data,
                self._invalid_tickers,
            ) = _get_historical_data(
                tickers=(
                    self._tickers + [self._benchmark_ticker]
                    if self._benchmark_ticker
                    else self._tickers
                ),
                api_key=self._api_key,
                enforce_source=None,
                start=self._start_date,
                end=self._end_date,
                interval=period,
                return_column=return_column,
                include_dividends=False,
                fill_nan=fill_nan,
                rounding=rounding if rounding is not None else self._rounding,
                sleep_timer=self._sleep_timer,
                show_errors=True,
                log_message="Obtaining intraday data",
                user_subscription=self._fmp_plan,
                cache=self._cache,
            )

        # Save the period to prevent having to reacquire the data
        self._intraday_period = period

        if self._intraday_historical_data.empty:
            return pd.DataFrame()

        # Change the benchmark ticker name to Benchmark
        self._intraday_historical_data = self._intraday_historical_data.rename(
            columns={self._benchmark_ticker: "Benchmark"}, level=1
        )

        if self._remove_invalid_tickers:
            self._tickers = [
                ticker
                for ticker in self._tickers
                if ticker not in self._invalid_tickers
            ]

        historical_data = self._intraday_historical_data.loc[
            self._start_date : self._end_date, :
        ]

        # The first row of the window has no preceding observation, so its Return stays NaN rather than being reported as an unchanged period.

        if show_columns is not None:
            historical_data = filter_columns(historical_data, show_columns)

        if len(self._tickers) == 1 and not self._benchmark_ticker:
            return historical_data.xs(self._tickers[0], level=1, axis="columns")

        return historical_data

    def get_dividend_calendar(
        self,
        overwrite: bool = False,
        rounding: int | None = None,
        show_columns: list[str] | None = None,
    ):
        """
        Obtain Dividend Calendars for any range of companies. It includes the following columns:
            - Date: The date of the dividend.
            - Adj Dividend: The adjusted dividend amount.
            - Dividend: The dividend amount.
            - Yield: The annualized dividend yield at the time of the dividend, as a decimal
            (0.0054 is 0.54%).
            - Record Date: The record date of the dividend.
            - Payment Date: The payment date of the dividend.
            - Declaration Date: The declaration date of the dividend.

        If a company does not pay any dividend, the function will mention that it was not able
        to find any dividend data for that company.

        Also known as: dividend dates, ex-dividend date, dividend history.

        Args:
            overwrite (bool): Defines whether to overwrite the existing data.
            rounding (int): Defines the number of decimal places to round the data to.
            show_columns (list[str] | None): A list of column names to keep in the result. Invalid
            names are reported and ignored. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The dividend calendar for the specified tickers.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["JNJ"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.get_dividend_calendar(show_columns=["Dividend"])
        ```

        Which returns:

        | date       |   Dividend |
        |:-----------|-----------:|
        | 2021-02-22 |       1.01 |
        | 2021-05-24 |       1.06 |
        | 2021-08-23 |       1.06 |
        | 2021-11-22 |       1.06 |
        | 2022-02-18 |       1.06 |
        | 2022-05-23 |       1.13 |
        | 2022-08-22 |       1.13 |
        | 2022-11-21 |       1.13 |
        | 2023-02-17 |       1.13 |
        | 2023-05-22 |       1.19 |
        | 2023-08-25 |       1.19 |
        | 2023-11-20 |       1.19 |
        | 2024-02-16 |       1.19 |
        | 2024-05-20 |       1.24 |
        | 2024-08-27 |       1.24 |
        | 2024-11-26 |       1.24 |
        | 2025-02-18 |       1.24 |
        | 2025-05-27 |       1.3  |
        | 2025-08-26 |       1.3  |
        | 2025-11-25 |       1.3  |
        """
        if not self._api_key:
            logger.error(
                "The requested data requires the api_key parameter to be set, consider obtaining a key with the "
                "following link: https://www.jeroenbouma.com/fmp"
                "\nThis functionality also requires a Premium subscription. You can get 15% off by using "
                "the above affiliate link which also supports the project."
            )
            return None

        if self._dividend_calendar.empty or overwrite:
            (
                self._dividend_calendar,
                self._invalid_tickers,
            ) = self._collect_per_ticker(
                dataset="dividend_calendar",
                tickers=self._tickers,
                ticker_axis=ticker_model.TICKER_ON_INDEX,
                parameters={
                    "start_date": self._start_date,
                    "end_date": self._end_date,
                    "user_subscription": self._fmp_plan,
                },
                collector=lambda tickers: _get_dividend_calendar(
                    tickers=tickers,
                    api_key=self._api_key,
                    start_date=self._start_date,
                    end_date=self._end_date,
                    sleep_timer=self._sleep_timer,
                    user_subscription=self._fmp_plan,
                ),
            )

        dividend_calendar = apply_rounding(
            self._dividend_calendar,
            rounding if rounding is not None else self._rounding,
        )

        if self._remove_invalid_tickers:
            self._tickers = [
                ticker
                for ticker in self._tickers
                if ticker not in self._invalid_tickers
            ]

        if len(self._tickers) == 1 and not self._dividend_calendar.empty:
            return filter_columns(dividend_calendar.loc[self._tickers[0]], show_columns)

        if dividend_calendar.empty and self._fmp_plan == "Free":
            logger.warning(
                "Dividend data is only available for Premium subscriptions and higher. Get 15% off by using the following link: "
                "https://www.jeroenbouma.com/fmp"
            )

        return filter_columns(dividend_calendar, show_columns)

    def get_esg_scores(
        self,
        overwrite: bool = False,
        rounding: int | None = None,
        show_columns: list[str] | None = None,
    ):
        """
        ESG scores, which stands for Environmental, Social, and Governance scores, are a crucial
        metric used by investors and organizations to assess a company's sustainability and
        ethical practices. These scores provide valuable insights into a company's performance
        in three key areas:

        - Environmental (E): The environmental component evaluates a company's
        impact on the planet and its efforts to mitigate environmental risks. It includes
        factors like carbon emissions, energy efficiency, water management, and waste
        reduction. A high environmental score indicates a company's commitment to eco-friendly
        practices and reducing its ecological footprint.

        - Social (S): The social component focuses on how a company interacts with its employees,
        customers, suppliers, and the communities in which it operates. Key factors in the
        social score include labor practices, diversity and inclusion, human rights,
        product safety, and community engagement. A strong social score reflects a company's
        dedication to fostering positive relationships and contributing positively to society.

        - Governance (G): Governance examines a company's internal structures, policies, and
        leadership. It assesses aspects such as board independence, executive compensation,
        transparency, and the presence of anti-corruption measures. A high governance score
        signifies strong leadership and a commitment to maintaining high ethical standards
        and accountability

        ESG scores provide investors with a holistic view of a company's sustainability and
        ethical practices, allowing them to make more informed investment decisions. These scores
        are increasingly used to identify socially responsible investments and guide capital towards
        companies that prioritize long-term sustainability and responsible business practices. As
        the importance of ESG considerations continues to grow, companies are motivated to improve
        their ESG scores, not only for ethical reasons but also to attract investors who value
        sustainable and responsible business practices.

        Also known as: environmental social governance, sustainability, ESG rating.

        Args:
            overwrite (bool): Defines whether to overwrite the existing data.
            rounding (int): Defines the number of decimal places to round the data to.
            show_columns (list[str] | None): A list of column names to keep in the result. Invalid
            names are reported and ignored. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The ESG scores for the specified tickers.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["XOM", "SHEL", "BP"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.get_esg_scores(show_columns=["ESG Score"])
        ```

        Which returns:

        | date   |   ('ESG Score', 'XOM') |   ('ESG Score', 'SHEL') |   ('ESG Score', 'BP') |
        |:-------|-----------------------:|------------------------:|----------------------:|
        | 2021   |                  66.43 |                  nan    |                 66.14 |
        | 2022   |                  68.75 |                   65.39 |                 67.54 |
        | 2023   |                  69.98 |                   60.49 |                 70.49 |
        | 2024   |                  68    |                   64.17 |                 62.26 |
        | 2025   |                  68.75 |                   61.74 |                 64.47 |
        """
        if not self._api_key:
            logger.error(
                "The requested data requires the api_key parameter to be set, consider obtaining a key with the "
                "following link: https://www.jeroenbouma.com/fmp"
                "\nThis functionality also requires a Premium subscription. You can get 15% off by using "
                "the above affiliate link which also supports the project."
            )
            return None

        if self._esg_scores.empty or overwrite:
            (
                self._esg_scores,
                self._invalid_tickers,
            ) = self._collect_per_ticker(
                dataset="esg_scores",
                tickers=self._tickers,
                ticker_axis=ticker_model.TICKER_ON_COLUMNS,
                parameters={
                    "quarter": self._quarterly,
                    "start_date": self._start_date,
                    "end_date": self._end_date,
                },
                collector=lambda tickers: _get_esg_scores(
                    tickers=tickers,
                    api_key=self._api_key,
                    quarter=self._quarterly,
                    start_date=self._start_date,
                    end_date=self._end_date,
                    sleep_timer=self._sleep_timer,
                    user_subscription=self._fmp_plan,
                ),
            )

        esg_scores = apply_rounding(
            self._esg_scores, rounding if rounding is not None else self._rounding
        )

        if self._remove_invalid_tickers:
            self._tickers = [
                ticker
                for ticker in self._tickers
                if ticker not in self._invalid_tickers
            ]

        if show_columns is not None:
            esg_scores = filter_columns(esg_scores, show_columns)

        if len(self._tickers) == 1 and not self._esg_scores.empty:
            return esg_scores.xs(self._tickers[0], axis=1, level=1)

        return esg_scores

    def get_market_risk_premium(self, overwrite: bool = False):
        """
        Obtains the equity risk premium by country -- the equity risk premium of a mature market
        plus a country risk premium, the country's default spread scaled by the volatility of its
        equities relative to its bonds, following the approach popularized by Aswath Damodaran --
        which is widely used to calibrate country-specific costs of equity and discount rates in a
        multi-country setting.

        In a cost of equity the premium is applied in one of two common ways: beta times the Total
        Equity Risk Premium, or beta times the mature market premium with the Country Risk Premium
        added on top. For a company that earns its revenue in several countries the premium is
        often weighted by where the revenue comes from rather than by where the company is listed.

        Also known as: country risk premium, Damodaran equity risk premium.

        Args:
            overwrite (bool): Defines whether to overwrite the existing data.

        Returns:
            pd.DataFrame: The market risk premium by country, including the continent, Country
            Risk Premium and Total Equity Risk Premium (both as decimals, 0.0446 for 4.46%).

        The premiums are decimal fractions (0.0446 for 4.46%).

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AMZN", "TSLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        market_risk_premium = toolkit.get_market_risk_premium()

        market_risk_premium.loc[
            ["United States", "Germany", "Japan", "Brazil", "India", "South Africa"],
            ["Country Risk Premium", "Total Equity Risk Premium"],
        ]
        ```

        Which returns:

        | Country       |   Country Risk Premium |   Total Equity Risk Premium |
        |:--------------|-----------------------:|----------------------------:|
        | United States |                 0.0023 |                      0.0446 |
        | Germany       |                 0      |                      0.0423 |
        | Japan         |                 0.0091 |                      0.0514 |
        | Brazil        |                 0.0324 |                      0.0747 |
        | India         |                 0.0285 |                      0.0708 |
        | South Africa  |                 0.039  |                      0.0813 |
        """
        if not self._api_key:
            logger.error(
                "The requested data requires the api_key parameter to be set, consider obtaining a key with the "
                "following link: https://www.jeroenbouma.com/fmp"
            )
            return None

        if self._market_risk_premium.empty or overwrite:
            # Published per country rather than per ticker, so it is one cache entry.
            cached_premium = (
                None
                if overwrite
                else self._cache.get(
                    source=policy_model.FINANCIAL_MODELING_PREP,
                    dataset="market_risk_premium",
                    entity=policy_model.MARKET_RISK_PREMIUM_ENTITY,
                )
            )

            if cached_premium is not None:
                self._market_risk_premium = cached_premium
            else:
                self._market_risk_premium = _get_market_risk_premium(
                    api_key=self._api_key,
                    user_subscription=self._fmp_plan,
                )

                if not self._market_risk_premium.empty:
                    self._cache.set(
                        source=policy_model.FINANCIAL_MODELING_PREP,
                        dataset="market_risk_premium",
                        entity=policy_model.MARKET_RISK_PREMIUM_ENTITY,
                        data=self._market_risk_premium,
                    )

        return self._market_risk_premium

    def get_commitment_of_traders(self, overwrite: bool = False):
        """
        Obtains the CFTC Commitment of Traders (COT) report for the tickers the Toolkit was
        initialized with. Published weekly by the U.S. Commodity Futures Trading Commission, it
        breaks down open interest in futures markets by trader type -- Non-Commercial (large
        speculators), Commercial (hedgers) and Non-Reportable (small traders) -- and is widely
        used to gauge positioning and sentiment in commodity, currency, interest rate and stock
        index futures markets.

        Note that this data is only available for CFTC-tracked futures markets. Tickers without a
        corresponding futures contract (e.g. most individual equities) return no data.

        Positioning is usually read from the net position of the non-commercial traders (long
        minus short), often as a share of open interest and against its own history: an extreme
        reading means a crowded trade. Commercial traders are mostly hedgers whose positions mirror
        their physical business, so a large commercial net short in a commodity is normal rather
        than a bearish view. The report shows the positions of Tuesday and is published on Friday.

        Also known as: COT report, CFTC positioning data, speculator/hedger positioning.

        Args:
            overwrite (bool): Defines whether to overwrite the existing data.

        Returns:
            pd.DataFrame: The Commitment of Traders report for the specified tickers.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["GC"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        toolkit.get_commitment_of_traders()[["Non-Commercial Long", "Non-Commercial Short"]].tail()
        ```

        Which returns:

        | date                |   Non-Commercial Long |   Non-Commercial Short |
        |:--------------------|----------------------:|-----------------------:|
        | 2025-12-02 00:00:00 |                261331 |                  43771 |
        | 2025-12-09 00:00:00 |                268485 |                  44599 |
        | 2025-12-16 00:00:00 |                280920 |                  46942 |
        | 2025-12-23 00:00:00 |                290161 |                  49461 |
        | 2025-12-30 00:00:00 |                275592 |                  44419 |
        """
        if not self._api_key:
            logger.error(
                "The requested data requires the api_key parameter to be set, consider obtaining a key with the "
                "following link: https://www.jeroenbouma.com/fmp"
            )
            return None

        if self._commitment_of_traders.empty or overwrite:
            (
                self._commitment_of_traders,
                self._invalid_tickers,
            ) = self._collect_per_ticker(
                dataset="commitment_of_traders",
                tickers=self._tickers,
                ticker_axis=ticker_model.TICKER_ON_COLUMNS,
                parameters={
                    "start_date": self._start_date,
                    "end_date": self._end_date,
                },
                collector=lambda tickers: _get_commitment_of_traders(
                    tickers=tickers,
                    api_key=self._api_key,
                    start_date=self._start_date,
                    end_date=self._end_date,
                    user_subscription=self._fmp_plan,
                ),
            )

        if self._remove_invalid_tickers:
            self._tickers = [
                ticker
                for ticker in self._tickers
                if ticker not in self._invalid_tickers
            ]

        if len(self._tickers) == 1 and not self._commitment_of_traders.empty:
            return self._commitment_of_traders.xs(self._tickers[0], axis=1, level=1)

        return self._commitment_of_traders

    def _missing_api_key_message(self) -> None:
        """Logs the standard message for the datasets that require a FinancialModelingPrep key."""
        logger.error(
            "The requested data requires the api_key parameter to be set, consider obtaining a key with the "
            "following link: https://www.jeroenbouma.com/fmp"
            "\nThis functionality also requires a Premium subscription. You can get 15% off by using "
            "the above affiliate link which also supports the project."
        )

    def _remove_invalid(self) -> None:
        """Drops the tickers without data when remove_invalid_tickers is set, as the other getters do."""
        if self._remove_invalid_tickers:
            self._tickers = [
                ticker
                for ticker in self._tickers
                if ticker not in self._invalid_tickers
            ]

    def get_executives(
        self, overwrite: bool = False, show_columns: list[str] | None = None
    ):
        """
        Obtain the key executives of each company: their title, pay, gender, year of birth
        and whether they are still active. This shows who leads the company and how the
        leadership team is composed, which is useful when assessing management quality or
        when following a change at the top.

        Also known as: management team, company officers, leadership.

        Args:
            overwrite (bool): Defines whether to overwrite the existing data.
            show_columns (list[str] | None): A list of column names to keep in the result. Invalid
            names are reported and ignored. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The executives per ticker, indexed by their name. Pay is only
            disclosed for the executives named in the proxy statement, so the best paid
            come first and the executives without disclosed pay follow.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["JPM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.get_executives(show_columns=["Title", "Year Born"])
        ```

        Which returns:

        | Name                 | Title                                                                             |   Year Born |
        |:---------------------|:----------------------------------------------------------------------------------|------------:|
        | Mary Callahan Erdoes | Chief Executive Officer of Asset & Wealth Management and Executive Vice President |        1967 |
        | Douglas Petno        | CEO of the Commercial & Investment Bank and Co-President                          |        1966 |
        | Troy Larry Rohrbaugh | CEO of Consumer & Community Banking and Co-President                              |        1970 |
        | Jeremy Barnum        | Executive Vice President & Chief Financial Officer                                |        1973 |
        | James Dimon          | Chairman & Chief Executive Officer                                                |        1956 |
        | Jennifer A. Piepszak | Chief Operating Officer                                                           |        1971 |
        | Daniel Eduardo Pinto | Vice Chairman                                                                     |        1963 |
        | Todd Anthony Combs   | Head Strategic Investment Group of Security & Resiliency Initiative               |        1971 |
        | Lori Ann Beer        | Global Chief Information Officer                                                  |        1968 |
        | Elena A. Korablina   | MD, Firmwide Controller & Principal Accounting Officer                            |        1974 |
        """
        if not self._api_key:
            self._missing_api_key_message()
            return None

        if self._executives.empty or overwrite:
            self._executives, self._invalid_tickers = self._collect_per_ticker(
                dataset="executives",
                tickers=self._tickers,
                ticker_axis=ticker_model.TICKER_ON_INDEX,
                collector=lambda tickers: _get_executives(
                    tickers=tickers,
                    api_key=self._api_key,
                    user_subscription=self._fmp_plan,
                ),
            )

        self._remove_invalid()

        if len(self._tickers) == 1 and not self._executives.empty:
            return filter_columns(self._executives.loc[self._tickers[0]], show_columns)

        return filter_columns(self._executives, show_columns)

    def get_executive_compensation(
        self,
        overwrite: bool = False,
        show_columns: list[str] | None = None,
    ):
        """
        Obtain the compensation of each company's executives per year as reported in the
        proxy statement (DEF 14A): salary, bonus, stock and option awards, incentive plan
        compensation, other compensation and the total. Comparing pay with the company's
        performance shows how well management incentives are aligned with shareholders.

        Automatically filtered to the years of this Toolkit instance's start_date and end_date.

        Also known as: executive pay, management compensation, proxy statement compensation.

        Args:
            overwrite (bool): Defines whether to overwrite the existing data.
            show_columns (list[str] | None): A list of column names to keep in the result. Invalid
            names are reported and ignored. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The compensation per ticker, indexed by year and executive.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["JPM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        toolkit.get_executive_compensation(show_columns=["Salary", "Incentive Plan Compensation"])
        ```

        Which returns:

        |                                                               |   Salary |   Incentive Plan Compensation |
        |:--------------------------------------------------------------|---------:|------------------------------:|
        | (2025, 'Douglas Petno Co-CEO, Commercial & Investment Bank')  |  1000000 |                      26500000 |
        | (2025, 'James Dimon Chairman and CEO')                        |  1500000 |                      41500000 |
        | (2025, 'Jeremy Barnum CFO')                                   |  1000000 |                      18500000 |
        | (2025, 'Mary Callahan Erdoes CEO, Asset & Wealth Management') |  1000000 |                      30000000 |
        | (2025, 'Troy Rohrbaugh Co-CEO, Commercial & Investment Bank') |  1000000 |                      26500000 |
        """
        if not self._api_key:
            self._missing_api_key_message()
            return None

        if self._executive_compensation.empty or overwrite:
            self._executive_compensation, self._invalid_tickers = (
                self._collect_per_ticker(
                    dataset="executive_compensation",
                    tickers=self._tickers,
                    ticker_axis=ticker_model.TICKER_ON_INDEX,
                    parameters={
                        "start_date": self._start_date,
                        "end_date": self._end_date,
                    },
                    collector=lambda tickers: _get_executive_compensation(
                        tickers=tickers,
                        api_key=self._api_key,
                        start_date=self._start_date,
                        end_date=self._end_date,
                        user_subscription=self._fmp_plan,
                    ),
                )
            )

        self._remove_invalid()

        if len(self._tickers) == 1 and not self._executive_compensation.empty:
            return filter_columns(
                self._executive_compensation.loc[self._tickers[0]], show_columns
            )

        return filter_columns(self._executive_compensation, show_columns)

    def get_company_notes(
        self, overwrite: bool = False, show_columns: list[str] | None = None
    ):
        """
        Obtain the securities other than common stock that each company has registered
        for listing, mostly notes with their coupon and maturity in the title (e.g.
        "1.625% Notes due 2026") and the exchange they are listed on. The list can
        include notes that have since matured or been redeemed and other securities such
        as depositary shares of preferred stock, so it shows which debt a company has
        listed rather than what is outstanding today.

        Also known as: listed debt, bonds issued, debt securities.

        Args:
            overwrite (bool): Defines whether to overwrite the existing data.
            show_columns (list[str] | None): A list of column names to keep in the result. Invalid
            names are reported and ignored. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The notes per ticker, indexed by their title.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["KO"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.get_company_notes(show_columns=["Exchange"])
        ```

        Which returns:

        | Title                        | Exchange   |
        |:-----------------------------|:-----------|
        | Floating Rate Notes Due 2019 | NYSE       |
        | 0.000% Notes Due 2021        | NYSE       |
        | 1.125% Notes Due 2022        | NYSE       |
        | 0.75% Notes Due 2023         | NYSE       |
        | 0.500% Notes Due 2024        | NYSE       |
        | 1.875% Notes Due 2026        | NYSE       |
        | 1.125% Notes Due 2027        | NYSE       |
        | 1.625% Notes Due 2035        | NYSE       |
        | 1.100% Notes Due 2036        | NYSE       |
        """
        if not self._api_key:
            self._missing_api_key_message()
            return None

        if self._company_notes.empty or overwrite:
            self._company_notes, self._invalid_tickers = self._collect_per_ticker(
                dataset="company_notes",
                tickers=self._tickers,
                ticker_axis=ticker_model.TICKER_ON_INDEX,
                collector=lambda tickers: _get_company_notes(
                    tickers=tickers,
                    api_key=self._api_key,
                    user_subscription=self._fmp_plan,
                ),
            )

        self._remove_invalid()

        if len(self._tickers) == 1 and not self._company_notes.empty:
            return filter_columns(
                self._company_notes.loc[self._tickers[0]], show_columns
            )

        return filter_columns(self._company_notes, show_columns)

    def get_employee_count(self, overwrite: bool = False):
        """
        Obtain the number of employees each company reported in its annual filings over
        time. The most recent row is the current employee count and the rows before it
        show how the workforce developed, which can be set against revenue or profit to
        see how productive the workforce is.

        Automatically filtered to this Toolkit instance's start_date and end_date.

        Also known as: headcount, number of employees, workforce size.

        Args:
            overwrite (bool): Defines whether to overwrite the existing data.

        Returns:
            pd.DataFrame: The employee count per reporting year (rows) and ticker (columns).

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["WMT", "AMZN"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.get_employee_count()
        ```

        Which returns:

        | Period   |     WMT |    AMZN |
        |:---------|--------:|--------:|
        | 2021     | 2300000 | 1608000 |
        | 2022     | 2300000 | 1541000 |
        | 2023     | 2100000 | 1525000 |
        | 2024     | 2100000 | 1556000 |
        | 2025     | 2100000 | 1576000 |
        """
        if not self._api_key:
            self._missing_api_key_message()
            return None

        if self._employee_count.empty or overwrite:
            self._employee_count, self._invalid_tickers = self._collect_per_ticker(
                dataset="employee_count",
                tickers=self._tickers,
                ticker_axis=ticker_model.TICKER_ON_COLUMNS,
                parameters={
                    "start_date": self._start_date,
                    "end_date": self._end_date,
                    "user_subscription": self._fmp_plan,
                },
                collector=lambda tickers: _get_employee_count(
                    tickers=tickers,
                    api_key=self._api_key,
                    start_date=self._start_date,
                    end_date=self._end_date,
                    sleep_timer=self._sleep_timer,
                    user_subscription=self._fmp_plan,
                ),
            )

        self._remove_invalid()

        return self._employee_count

    def get_shares_float(self, overwrite: bool = False):
        """
        Obtain the free float of each company: the number of shares available for public
        trading, the number of shares outstanding and the free float as the share of the
        outstanding shares that can be traded (as a decimal). A low free float means few
        shares change hands, which tends to make a stock less liquid and more volatile.

        Also known as: free float, float shares, public float, share liquidity.

        Args:
            overwrite (bool): Defines whether to overwrite the existing data.

        Returns:
            pd.DataFrame: The share float figures (rows) per ticker (columns).

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["META", "GOOGL", "BRK-B"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.get_shares_float().loc[["Free Float"]]
        ```

        Which returns:

        |            |     META |    GOOGL |    BRK-B |
        |:-----------|---------:|---------:|---------:|
        | Free Float | 0.863739 | 0.893911 | 0.656007 |
        """
        if not self._api_key:
            self._missing_api_key_message()
            return None

        if self._shares_float.empty or overwrite:
            self._shares_float, self._invalid_tickers = self._collect_per_ticker(
                dataset="shares_float",
                tickers=self._tickers,
                ticker_axis=ticker_model.TICKER_ON_COLUMNS,
                collector=lambda tickers: _get_shares_float(
                    tickers=tickers,
                    api_key=self._api_key,
                    user_subscription=self._fmp_plan,
                ),
            )

        self._remove_invalid()

        return self._shares_float

    def get_mergers_acquisitions(
        self, overwrite: bool = False, show_columns: list[str] | None = None
    ):
        """
        Obtain the mergers and acquisitions each company took part in, as the acquirer or
        as the target, based on the merger filings (S-4) with the SEC. Every deal comes
        with the other party, the filing date and a link to the filing.

        The search uses the company name from the profile, so the profile is retrieved
        first when it is not available yet. Deals of companies with a similar name are
        left out: only the deals in which the ticker itself is involved are kept.

        Automatically filtered to this Toolkit instance's start_date and end_date.

        Also known as: M&A, acquisitions, mergers, takeovers.

        Args:
            overwrite (bool): Defines whether to overwrite the existing data.
            show_columns (list[str] | None): A list of column names to keep in the result. Invalid
            names are reported and ignored. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The deals per ticker, indexed by their transaction date.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["CVX"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.get_mergers_acquisitions(show_columns=["Role", "Acquirer Name", "Target Name"])
        ```

        Which returns:

        | Transaction Date    | Role     | Acquirer Name   | Target Name      |
        |:--------------------|:---------|:----------------|:-----------------|
        | 2021-03-22 00:00:00 | Acquirer | CHEVRON CORP    | NBLX             |
        | 2023-06-20 00:00:00 | Acquirer | CHEVRON CORP    | PDC Energy, Inc. |
        | 2023-08-03 00:00:00 | Acquirer | CHEVRON CORP    | PDC              |
        | 2024-02-26 00:00:00 | Acquirer | CHEVRON CORP    | Hess             |
        """
        if not self._api_key:
            self._missing_api_key_message()
            return None

        if self._mergers_acquisitions.empty or overwrite:
            profile = self.get_profile()
            company_names = (
                profile.loc["Company Name"].to_dict()
                if profile is not None and "Company Name" in profile.index
                else {}
            )

            self._mergers_acquisitions, self._invalid_tickers = (
                self._collect_per_ticker(
                    dataset="mergers_acquisitions",
                    tickers=self._tickers,
                    ticker_axis=ticker_model.TICKER_ON_INDEX,
                    parameters={
                        "start_date": self._start_date,
                        "end_date": self._end_date,
                    },
                    collector=lambda tickers: _get_mergers_acquisitions(
                        tickers=tickers,
                        company_names=company_names,
                        api_key=self._api_key,
                        start_date=self._start_date,
                        end_date=self._end_date,
                        user_subscription=self._fmp_plan,
                    ),
                )
            )

        # A company without deals is not an invalid ticker, so nothing is removed here.

        if len(self._tickers) == 1 and not self._mergers_acquisitions.empty:
            return filter_columns(
                self._mergers_acquisitions.loc[self._tickers[0]], show_columns
            )

        return filter_columns(self._mergers_acquisitions, show_columns)

    def get_stock_splits(self, overwrite: bool = False):
        """
        Obtain the stock splits of each company, with the split ratio as a numerator and
        denominator: a 4-for-1 split has numerator 4 and denominator 1, while a reverse
        split has a numerator smaller than its denominator. Splits change the number of
        shares but not the value of the company, which is why historical prices are
        adjusted for them.

        Automatically filtered to this Toolkit instance's start_date and end_date.

        Also known as: share splits, reverse splits, split history.

        Args:
            overwrite (bool): Defines whether to overwrite the existing data.

        Returns:
            pd.DataFrame: The splits per ticker, indexed by their date.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["NVDA", "AAPL", "TSLA", "GOOGL", "AMZN"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2020-01-01",
            end_date="2025-12-31",
        )

        toolkit.get_stock_splits()[["Numerator", "Denominator"]]
        ```

        Which returns:

        |                         |   Numerator |   Denominator |
        |:------------------------|------------:|--------------:|
        | ('NVDA', '2021-07-20')  |           4 |             1 |
        | ('NVDA', '2024-06-10')  |          10 |             1 |
        | ('AAPL', '2020-08-31')  |           4 |             1 |
        | ('TSLA', '2020-08-31')  |           5 |             1 |
        | ('TSLA', '2022-08-25')  |           3 |             1 |
        | ('GOOGL', '2022-07-18') |          20 |             1 |
        | ('AMZN', '2022-06-06')  |          20 |             1 |
        """
        if not self._api_key:
            self._missing_api_key_message()
            return None

        if self._stock_splits.empty or overwrite:
            self._stock_splits, self._invalid_tickers = self._collect_per_ticker(
                dataset="stock_splits",
                tickers=self._tickers,
                ticker_axis=ticker_model.TICKER_ON_INDEX,
                parameters={
                    "start_date": self._start_date,
                    "end_date": self._end_date,
                    "user_subscription": self._fmp_plan,
                },
                collector=lambda tickers: _get_stock_splits(
                    tickers=tickers,
                    api_key=self._api_key,
                    start_date=self._start_date,
                    end_date=self._end_date,
                    user_subscription=self._fmp_plan,
                ),
            )

        # A company that never split is not an invalid ticker, so nothing is removed here.

        if len(self._tickers) == 1 and not self._stock_splits.empty:
            return self._stock_splits.loc[self._tickers[0]]

        return self._stock_splits

    def get_insider_trade_statistics(
        self,
        overwrite: bool = False,
        rounding: int | None = None,
        show_columns: list[str] | None = None,
    ):
        """
        Obtain quarterly statistics on the trades of each company's insiders (officers,
        directors and large shareholders), based on their Form 4 filings: the number of
        acquisitions and disposals, their ratio, the number of shares acquired and
        disposed of, and the number of open market purchases and sales.

        Insiders know their company best, so heavy buying can signal confidence while
        persistent selling can be worth a closer look. Note that many disposals are
        routine, such as sales to cover taxes on vested stock awards.

        Automatically filtered to this Toolkit instance's start_date and end_date.

        Also known as: insider trading, insider transactions, Form 4 statistics.

        Args:
            overwrite (bool): Defines whether to overwrite the existing data.
            rounding (int): Defines the number of decimal places to round the data to.
            show_columns (list[str] | None): A list of column names to keep in the result. Invalid
            names are reported and ignored. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The statistics per ticker, indexed by quarter.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["META"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2024-01-01",
            end_date="2025-12-31",
        )

        toolkit.get_insider_trade_statistics(show_columns=["Acquired Transactions", "Disposed Transactions"])
        ```

        Which returns:

        | Period   |   Acquired Transactions |   Disposed Transactions |
        |:---------|------------------------:|------------------------:|
        | 2024Q1   |                     206 |                    1965 |
        | 2024Q2   |                      59 |                     322 |
        | 2024Q3   |                      38 |                     664 |
        | 2024Q4   |                      56 |                     487 |
        | 2025Q1   |                      77 |                     992 |
        | 2025Q2   |                      67 |                     194 |
        | 2025Q3   |                      50 |                     271 |
        | 2025Q4   |                      41 |                      82 |
        """
        if not self._api_key:
            self._missing_api_key_message()
            return None

        if self._insider_trade_statistics.empty or overwrite:
            self._insider_trade_statistics, self._invalid_tickers = (
                self._collect_per_ticker(
                    dataset="insider_trade_statistics",
                    tickers=self._tickers,
                    ticker_axis=ticker_model.TICKER_ON_INDEX,
                    parameters={
                        "start_date": self._start_date,
                        "end_date": self._end_date,
                    },
                    collector=lambda tickers: _get_insider_trade_statistics(
                        tickers=tickers,
                        api_key=self._api_key,
                        start_date=self._start_date,
                        end_date=self._end_date,
                        user_subscription=self._fmp_plan,
                    ),
                )
            )

        insider_trade_statistics = apply_rounding(
            self._insider_trade_statistics,
            rounding if rounding is not None else self._rounding,
        )

        self._remove_invalid()

        if len(self._tickers) == 1 and not self._insider_trade_statistics.empty:
            return filter_columns(
                insider_trade_statistics.loc[self._tickers[0]], show_columns
            )

        return filter_columns(insider_trade_statistics, show_columns)

    def get_stock_grades(
        self, overwrite: bool = False, show_columns: list[str] | None = None
    ):
        """
        Obtain the grades analysts gave each company: per date and grading company the
        previous and the new grade, and whether the grade was upgraded, downgraded or
        maintained. Following how the grades change over time shows how sentiment among
        analysts develops, with upgrades and downgrades by well-followed firms often
        moving the share price.

        Automatically filtered to this Toolkit instance's start_date and end_date.

        Also known as: analyst ratings, upgrades and downgrades, analyst grades.

        Args:
            overwrite (bool): Defines whether to overwrite the existing data.
            show_columns (list[str] | None): A list of column names to keep in the result. Invalid
            names are reported and ignored. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The grades per ticker, indexed by date and grading company.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["NKE"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-10-01",
            end_date="2025-12-31",
        )

        toolkit.get_stock_grades()
        ```

        Which returns:

        |                                         | Previous Grade   | New Grade      | Action   |
        |:----------------------------------------|:-----------------|:---------------|:---------|
        | ('2025-10-01', 'Barclays')              | Equal Weight     | Equal Weight   | Maintain |
        | ('2025-10-01', 'JP Morgan')             | Overweight       | Overweight     | Maintain |
        | ('2025-10-01', 'Morgan Stanley')        | Equal Weight     | Equal Weight   | Maintain |
        | ('2025-10-01', 'Needham')               | Buy              | Buy            | Maintain |
        | ('2025-10-01', 'Piper Sandler')         | Overweight       | Overweight     | Maintain |
        | ('2025-10-01', 'Stifel')                | Hold             | Hold           | Maintain |
        | ('2025-10-01', 'Telsey Advisory Group') | Market Perform   | Market Perform | Maintain |
        | ('2025-10-01', 'Truist Securities')     | Buy              | Buy            | Maintain |
        | ('2025-10-02', 'Keybanc')               | Sector Weight    | Overweight     | Upgrade  |
        | ('2025-11-13', 'Wells Fargo')           | Equal Weight     | Overweight     | Upgrade  |
        | ('2025-12-09', 'Citigroup')             | Neutral          | Neutral        | Maintain |
        | ('2025-12-10', 'Telsey Advisory Group') | Market Perform   | Market Perform | Maintain |
        | ('2025-12-12', 'BTIG')                  | Buy              | Buy            | Maintain |
        | ('2025-12-19', 'B of A Securities')     | Buy              | Buy            | Maintain |
        | ('2025-12-19', 'Barclays')              | Equal Weight     | Equal Weight   | Maintain |
        | ('2025-12-19', 'Bernstein')             | Outperform       | Outperform     | Maintain |
        | ('2025-12-19', 'Citigroup')             | Neutral          | Neutral        | Maintain |
        | ('2025-12-19', 'Goldman Sachs')         | Buy              | Buy            | Maintain |
        | ('2025-12-19', 'Guggenheim')            | Buy              | Buy            | Maintain |
        | ('2025-12-19', 'Needham')               | Buy              | Buy            | Maintain |
        | ('2025-12-19', 'Piper Sandler')         | Overweight       | Overweight     | Maintain |
        | ('2025-12-19', 'Stifel')                | Hold             | Hold           | Maintain |
        | ('2025-12-19', 'Telsey Advisory Group') | Market Perform   | Market Perform | Maintain |
        | ('2025-12-19', 'Truist Securities')     | Buy              | Buy            | Maintain |
        | ('2025-12-19', 'UBS')                   | Neutral          | Neutral        | Maintain |
        | ('2025-12-19', 'Wells Fargo')           | Overweight       | Overweight     | Maintain |
        """
        if not self._api_key:
            self._missing_api_key_message()
            return None

        if self._stock_grades.empty or overwrite:
            self._stock_grades, self._invalid_tickers = self._collect_per_ticker(
                dataset="stock_grades",
                tickers=self._tickers,
                ticker_axis=ticker_model.TICKER_ON_INDEX,
                parameters={"start_date": self._start_date, "end_date": self._end_date},
                collector=lambda tickers: _get_stock_grades(
                    tickers=tickers,
                    api_key=self._api_key,
                    start_date=self._start_date,
                    end_date=self._end_date,
                    user_subscription=self._fmp_plan,
                ),
            )

        self._remove_invalid()

        if len(self._tickers) == 1 and not self._stock_grades.empty:
            return filter_columns(
                self._stock_grades.loc[self._tickers[0]], show_columns
            )

        return filter_columns(self._stock_grades, show_columns)

    def _collect_fund_data(
        self, dataset: str, overwrite: bool = False
    ) -> pd.DataFrame | None:
        """
        Collect a fund dataset of the tickers from Financial Modeling Prep, once per instance.
        `toolkit.etf` presents them (get_holdings, get_information, get_country_weightings and
        get_sector_weightings) and builds its analyses on them.

        Args:
            dataset (str): The dataset: holdings, information, country_weightings or
                sector_weightings.
            overwrite (bool, optional): Whether to collect the dataset again. Defaults to False.

        Returns:
            pd.DataFrame | None: The holdings per fund and asset, or the other datasets with a
            column per fund. None without an API key.
        """
        if not self._api_key:
            self._missing_api_key_message()
            return None

        # The holdings have a row per asset of every fund, the other datasets a column per fund.
        collector, ticker_axis = {
            "holdings": (_get_etf_holdings, ticker_model.TICKER_ON_INDEX),
            "information": (_get_etf_information, ticker_model.TICKER_ON_COLUMNS),
            "country_weightings": (
                _get_etf_country_weightings,
                ticker_model.TICKER_ON_COLUMNS,
            ),
            "sector_weightings": (
                _get_etf_sector_weightings,
                ticker_model.TICKER_ON_COLUMNS,
            ),
        }[dataset]
        attribute = f"_etf_{dataset}"

        if getattr(self, attribute).empty or overwrite:
            # Not every ticker is a fund, so a ticker without fund data is not invalid.
            data, _ = self._collect_per_ticker(
                dataset=f"etf_{dataset}",
                tickers=self._tickers,
                ticker_axis=ticker_axis,
                collector=lambda tickers: collector(
                    tickers=tickers,
                    api_key=self._api_key,
                    user_subscription=self._fmp_plan,
                ),
            )
            setattr(self, attribute, data)

        return getattr(self, attribute)

    def get_earnings_call_transcripts(
        self,
        latest: bool = True,
        fiscal_period: str | None = None,
        overwrite: bool = False,
    ):
        """
        Obtain the earnings call transcripts of each company: the full text of the call,
        with management's prepared remarks followed by the questions of analysts and the
        answers. Transcripts explain the numbers in the financial statements in
        management's own words, and with that are a strong input for text analysis and AI
        models, e.g. to track sentiment, guidance or recurring themes over time.

        A transcript is long (often 40,000 to 60,000 characters) and every quarter is a
        separate request. By default only the most recent transcript is retrieved. With
        fiscal_period (e.g. "2025Q4") the call of that fiscal quarter is retrieved, and with
        latest=False every transcript between this Toolkit instance's start_date and
        end_date. A fiscal period without a transcript is logged together with the most
        recent periods that do have one. Each transcript is cached on its own, since a
        published transcript does not change, so a quarter is only retrieved once.

        Also known as: earnings call, conference call transcript, earnings transcript.

        Args:
            latest (bool): Whether to only retrieve the most recent transcript. When False,
                every transcript between the start_date and end_date is retrieved. Defaults to True.
            fiscal_period (str | None): The fiscal quarter of the call to retrieve, written as
                "2025Q4" (or "2025-Q4", "Q4 2025"). It is the company's own fiscal year, which
                differs from the calendar year for e.g. Apple and Microsoft. Takes precedence
                over latest. Defaults to None.
            overwrite (bool): Defines whether to overwrite the existing data.

        Returns:
            pd.DataFrame: The date and transcript per ticker, indexed by fiscal period.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["NVDA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        transcripts = toolkit.get_earnings_call_transcripts(fiscal_period="2025Q4")

        transcripts["Transcript"].str[:80]
        ```

        Which returns:

        | Fiscal Period   | Transcript                                                                       |
        |:----------------|:---------------------------------------------------------------------------------|
        | 2025Q4          | Christa: Good afternoon. My name is Christa, and I will be your conference opera |
        """
        if not self._api_key:
            self._missing_api_key_message()
            return None

        if (
            self._earnings_call_transcripts.empty
            or overwrite
            or self._earnings_call_transcripts_selection != (latest, fiscal_period)
        ):
            self._earnings_call_transcripts, self._invalid_tickers = (
                self._collect_per_ticker(
                    dataset="earnings_call_transcripts_selection",
                    tickers=self._tickers,
                    ticker_axis=ticker_model.TICKER_ON_INDEX,
                    parameters={
                        "latest": latest,
                        "fiscal_period": fiscal_period,
                        "start_date": self._start_date,
                        "end_date": self._end_date,
                    },
                    collector=lambda tickers: _get_earnings_call_transcripts(
                        tickers=tickers,
                        api_key=self._api_key,
                        start_date=self._start_date,
                        end_date=self._end_date,
                        latest=latest,
                        user_subscription=self._fmp_plan,
                        fiscal_period=fiscal_period,
                    ),
                )
            )
            self._earnings_call_transcripts_selection = (latest, fiscal_period)

        self._remove_invalid()

        if len(self._tickers) == 1 and not self._earnings_call_transcripts.empty:
            return self._earnings_call_transcripts.loc[self._tickers[0]]

        return self._earnings_call_transcripts

    def get_historical_statistics(self):
        """
        Retrieve statistics about each ticker's historical data. This is especially useful to understand why certain
        tickers might fluctuate more than others as it could be due to local regulations or the currency the instrument
        is denoted in. It returns:

            - Currency: The currency the instrument is denoted in.
            - Symbol: The symbol of the instrument.
            - Exchange Name: The name of the exchange the instrument is listed on.
            - Instrument Type: The type of instrument.
            - First Trade Date: The date the instrument was first traded.
            - Regular Market Time: The time the instrument is traded.
            - GMT Offset: The GMT offset.
            - Timezone: The timezone the instrument is traded in.
            - Exchange Timezone Name: The name of the timezone the instrument is traded in.

        Also known as: listing details, instrument information, exchange and currency.

        Returns:
            pd.DataFrame: A DataFrame containing the statistics for each ticker.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["ASML.AS", "ASML"],
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.get_historical_statistics().loc[
            ["Currency", "Exchange Name", "Instrument Type", "First Trade Date", "Exchange Timezone Name"]
        ]
        ```

        Which returns:

        |                        | ASML.AS          | ASML             |
        |:-----------------------|:-----------------|:-----------------|
        | Currency               | EUR              | USD              |
        | Exchange Name          | AMS              | NMS              |
        | Instrument Type        | EQUITY           | EQUITY           |
        | First Trade Date       | 1998-07-20       | 1995-03-15       |
        | Exchange Timezone Name | Europe/Amsterdam | America/New_York |
        """

        if self._historical_statistics.empty:
            self._historical_statistics, _ = _get_historical_statistics(
                tickers=self._tickers,
                api_key=self._api_key if self._api_key is not None else None,
            )

        if len(self._tickers) == 1 and not self._historical_statistics.empty:
            return self._historical_statistics[self._tickers[0]]

        return self._historical_statistics

    def get_treasury_data(
        self,
        enforce_source: str | None = None,
        period: str = "daily",
        risk_free_rate: str | None = None,
        fill_nan: bool = True,
        divide_ohlc_by: int | float | None = 100,
        rounding: int | None = None,
        show_errors: bool = False,
    ):
        """
        Retrieve daily, weekly, monthly, quarterly or yearly US Treasury yields for four maturities, 13 weeks,
        5 years, 10 years and 30 years, from FinancialModelingPrep or, as a free alternative, YahooFinance. With
        the default divide_ohlc_by of 100 the yields are returned as decimals (0.042 is 4.2%).

        A week, month, quarter or year holds the first (Open), highest, lowest and last (Close) yield of
        the period. The risk free rate the Performance, Risk and Models modules use for such a period is
        instead the average of the daily yields within it, which approximates what a holder earned over
        the period rather than where the yield happened to close at its end.

        Also known as: US Treasury yields, yield curve, treasury rates.

        Args:
            period (str): The interval at which the treasury data should be returned - daily, weekly, monthly, quarterly, or yearly.
            fill_nan (bool): Defines whether to forward fill NaN values. This defaults
            to True to prevent holes in the dataset. This is especially relevant for
            technical indicators.
            risk_free_rate (str | None, optional): One maturity to return ('13w', '5y', '10y' or
                '30y'). Defaults to None, which returns all four. The Toolkit's own risk free rate
                stays the maturity set on the Toolkit either way.
            divide_ohlc_by (int | float | None, optional): A value to divide the yields by. Treasury
                yields are published in percent, so this defaults to 100 to return decimals.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the rounding set on the Toolkit.
            show_errors (bool, optional): Whether to report retrieval errors. Defaults to False.
            enforce_source (str | None, optional): Forces this specific call to use a given
                source, either "FinancialModelingPrep" or "YahooFinance". This takes precedence
                over the source set on the Toolkit itself. Defaults to None, which falls back to
                the Toolkit's own enforce_source.

        Returns:
            pd.DataFrame: A DataFrame containing the treasury data.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        toolkit.get_treasury_data(period="monthly")["Close"]
        ```

        Which returns:

        | Date    |   13 Week |   5 Year |   10 Year |   30 Year |
        |:--------|----------:|---------:|----------:|----------:|
        | 2025-01 |    0.0419 |   0.0436 |    0.0457 |    0.0481 |
        | 2025-02 |    0.0419 |   0.0403 |    0.0423 |    0.0452 |
        | 2025-03 |    0.0419 |   0.0398 |    0.0425 |    0.0461 |
        | 2025-04 |    0.0419 |   0.0375 |    0.0418 |    0.0468 |
        | 2025-05 |    0.0423 |   0.0398 |    0.0442 |    0.0493 |
        | 2025-06 |    0.0419 |   0.038  |    0.0423 |    0.0478 |
        | 2025-07 |    0.0424 |   0.0396 |    0.0436 |    0.0489 |
        | 2025-08 |    0.0404 |   0.037  |    0.0423 |    0.0492 |
        | 2025-09 |    0.0385 |   0.0373 |    0.0415 |    0.0473 |
        | 2025-10 |    0.0372 |   0.0372 |    0.041  |    0.0467 |
        | 2025-11 |    0.037  |   0.036  |    0.0402 |    0.0466 |
        | 2025-12 |    0.0355 |   0.0372 |    0.0416 |    0.0484 |
        """
        risk_free_names = {
            "13w": "13 Week",
            "5y": "5 Year",
            "10y": "10 Year",
            "30y": "30 Year",
        }
        treasury_names = {
            "^IRX": "13 Week",
            "^FVX": "5 Year",
            "^TNX": "10 Year",
            "^TYX": "30 Year",
        }

        if risk_free_rate and risk_free_rate not in risk_free_names:
            raise ValueError(
                "Please choose from 13w, 5y, 10y or 30y as risk_free_rate."
            )

        # Every maturity is retrieved and kept, so the Toolkit's own risk free rate stays
        # available; risk_free_rate only narrows what is returned.
        requested_maturity = risk_free_names[risk_free_rate] if risk_free_rate else None
        risk_free_rate_tickers = list(treasury_names)
        risk_free_rate = risk_free_names[self._risk_free_rate]

        def select(treasury_data: pd.DataFrame) -> pd.DataFrame:
            treasury_data = treasury_data.loc[self._start_date : self._end_date, :]
            if requested_maturity is None:
                return treasury_data
            return treasury_data.loc[
                :,
                treasury_data.columns.get_level_values(1) == requested_maturity,
            ]

        specific_rates = (
            [
                treasury_names[ticker]
                in self._daily_treasury_data.columns.get_level_values(1)
                for ticker in risk_free_rate_tickers
            ]
            if not self._daily_treasury_data.empty
            else []
        )

        if enforce_source is not None and enforce_source not in [
            "FinancialModelingPrep",
            "YahooFinance",
        ]:
            raise ValueError(
                "The enforce_source parameter must be either 'FinancialModelingPrep' or 'YahooFinance'."
            )

        if self._daily_treasury_data.empty or False in specific_rates:
            # Collects when treasury data is empty or holds only the historical subselection.
            (
                self._daily_treasury_data,
                _,
            ) = _get_historical_data(
                tickers=risk_free_rate_tickers,
                api_key=self._api_key,
                enforce_source=(
                    enforce_source
                    if enforce_source is not None
                    else self._enforce_source
                ),
                start=self._start_date,
                end=self._end_date,
                divide_ohlc_by=divide_ohlc_by,
                rounding=rounding if rounding is not None else self._rounding,
                show_errors=show_errors,
                fill_nan=fill_nan,
                sleep_timer=self._sleep_timer,
                log_message="Obtaining treasury data",
                user_subscription=self._fmp_plan,
                cache=self._cache,
            )

            if not self._daily_treasury_data.empty:
                self._daily_treasury_data = self._daily_treasury_data.rename(
                    columns=treasury_names, level=1
                )
                self._daily_treasury_data = self._align_treasury_data_to_trading_days(
                    self._daily_treasury_data
                )
                self._daily_risk_free_rate = self._daily_treasury_data.xs(
                    risk_free_rate, level=1, axis=1
                )

            if self._daily_treasury_data.empty:
                logger.debug(
                    "No treasury data could be retrieved. This is usually due to an invalid API key, "
                    "reaching the API limit, or unavailability within Yahoo Finance. Consider "
                    "obtaining a key with the following link or upgrading you plan: https://www.jeroenbouma.com/fmp"
                    "\nYou can get 15% off by using the above affiliate link which also supports the project."
                )
                return pd.DataFrame()

        if period == "daily":
            return select(self._daily_treasury_data)
        if period == "weekly":
            self._weekly_treasury_data = _convert_daily_to_other_period(
                period=period,
                daily_historical_data=self._daily_treasury_data,
                start=self._start_date,
                end=self._end_date,
                rounding=rounding if rounding is not None else self._rounding,
            )

            self._weekly_risk_free_rate = self._average_risk_free_rate(
                period=period,
                treasury_data=self._weekly_treasury_data,
                risk_free_rate=risk_free_rate,
                rounding=rounding,
            )

            return select(self._weekly_treasury_data)
        if period == "monthly":
            self._monthly_treasury_data = _convert_daily_to_other_period(
                period=period,
                daily_historical_data=self._daily_treasury_data,
                start=self._start_date,
                end=self._end_date,
                rounding=rounding if rounding is not None else self._rounding,
            )

            self._monthly_risk_free_rate = self._average_risk_free_rate(
                period=period,
                treasury_data=self._monthly_treasury_data,
                risk_free_rate=risk_free_rate,
                rounding=rounding,
            )

            return select(self._monthly_treasury_data)
        if period == "quarterly":
            self._quarterly_treasury_data = _convert_daily_to_other_period(
                period=period,
                daily_historical_data=self._daily_treasury_data,
                start=self._start_date,
                end=self._end_date,
                rounding=rounding if rounding is not None else self._rounding,
            )

            self._quarterly_risk_free_rate = self._average_risk_free_rate(
                period=period,
                treasury_data=self._quarterly_treasury_data,
                risk_free_rate=risk_free_rate,
                rounding=rounding,
            )

            return select(self._quarterly_treasury_data)
        if period == "yearly":
            self._yearly_treasury_data = _convert_daily_to_other_period(
                period=period,
                daily_historical_data=self._daily_treasury_data,
                start=self._start_date,
                end=self._end_date,
                rounding=rounding if rounding is not None else self._rounding,
            )

            self._yearly_risk_free_rate = self._average_risk_free_rate(
                period=period,
                treasury_data=self._yearly_treasury_data,
                risk_free_rate=risk_free_rate,
                rounding=rounding,
            )

            return select(self._yearly_treasury_data)

        raise ValueError(
            "Please choose from daily, weekly, monthly, quarterly or yearly as period."
        )

    def get_exchange_rates(
        self,
        period: str = "daily",
        return_column: str = "Adj Close",
        fill_nan: bool = True,
        overwrite: bool = False,
        rounding: int | None = None,
        show_ticker_seperation: bool = True,
    ):
        """
        This functionality looks at the exchange rates between the currency of the historical data and the currency
        of the financial statements. Given that these can deviate from each other, e.g. the historical data is in USD
        but the financial statements are in EUR, it is important to adjust for this. This is especially relevant for
        models that use the historical data and the financial statements.

        This function therefore shows the exchange rates that are used to convert the financial statements to the
        currency of the historical data. The historical market data is quote currency and the financial statements
        are base currency.

        Note that you can get currency data from any currency as well by supplying the currency as a ticker. For example,
        if you want to get the exchange rates between USD and EUR you can use USDEUR=X as a ticker.

        Important to note is that when an api_key is included in the Toolkit initialization that the data
        collection defaults to FinancialModelingPrep which is a more stable source and utilises your subscription.
        However, if this is undesired, it can be disabled by setting enforce_source to "YahooFinance". If
        data collection fails from FinancialModelingPrep it automatically reverts back to YahooFinance.

        Also known as: currency exchange, FX rates, foreign exchange rates.

        Args:
            period (str): The interval at which the historical data should be
            returned - daily, weekly, monthly, quarterly, or yearly.
            Defaults to "daily".
            return_column (str): The column to use for the return calculation. Defaults to "Adj Close".
            fill_nan (bool): Defines whether to forward fill NaN values. This defaults
            to True to prevent holes in the dataset. This is especially relevant for
            technical indicators.
            overwrite (bool): Defines whether to overwrite the existing data.
            rounding (int): Defines the number of decimal places to round the returned data to. Defaults
            to None, which uses the rounding of the Toolkit. The rates the financial statements are
            translated with are never rounded.
            show_ticker_seperation (bool, optional): A boolean representing whether to show which tickers
            acquired data from FinancialModelingPrep and which tickers acquired data from YahooFinance.

        Raises:
            ValueError: If an invalid value is specified for period.

        Returns:
            pandas.DataFrame: The historical exchange rate data.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            "ASML",
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.get_exchange_rates(period="monthly")
        ```

        Which returns:

        | Date    |   Open |   High |    Low |   Close |   Adj Close |      Volume |   Return |   Cumulative Return |
        |:--------|-------:|-------:|-------:|--------:|------------:|------------:|---------:|--------------------:|
        | 2025-03 | 1.0402 | 1.0954 | 1.0388 |  1.0815 |        1.08 | 1.1265e+07  |   0.0385 |              0.8926 |
        | 2025-04 | 1.0815 | 1.1573 | 1.0778 |  1.1328 |        1.13 | 1.3243e+07  |   0.0463 |              0.9339 |
        | 2025-05 | 1.1328 | 1.1422 | 1.1065 |  1.1346 |        1.13 | 9.02118e+06 |   0      |              0.9339 |
        | 2025-06 | 1.1346 | 1.1788 | 1.1345 |  1.1787 |        1.18 | 8.07454e+06 |   0.0442 |              0.9752 |
        | 2025-07 | 1.1787 | 1.183  | 1.1401 |  1.1415 |        1.14 | 7.03696e+06 |  -0.0339 |              0.9421 |
        | 2025-08 | 1.1417 | 1.1743 | 1.1392 |  1.1707 |        1.17 | 3.44184e+06 |   0.0263 |              0.9669 |
        | 2025-09 | 1.1696 | 1.1919 | 1.1608 |  1.1734 |        1.17 | 3.49176e+06 |   0      |              0.9669 |
        | 2025-10 | 1.1736 | 1.1779 | 1.1522 |  1.1536 |        1.15 | 4.08101e+06 |  -0.0171 |              0.9504 |
        | 2025-11 | 1.153  | 1.1656 | 1.1469 |  1.1601 |        1.16 | 3.2838e+06  |   0.0087 |              0.9587 |
        | 2025-12 | 1.1594 | 1.1808 | 1.1589 |  1.1746 |        1.17 | 2.67505e+06 |   0.0086 |              0.9669 |
        """
        if not self._currencies or overwrite:
            if self._historical_statistics.empty:
                self.get_historical_statistics()
            if self._statistics_statement.empty:
                self.get_statistics_statement()

            if not self._statistics_statement.empty:
                (
                    self._statement_currencies,
                    self._currencies,
                ) = currencies_model.determine_currencies(
                    statement_currencies=self._statistics_statement.xs(
                        "Reported Currency", axis=0, level=1
                    ),
                    historical_currencies=self._historical_statistics.loc["Currency"],
                )

        # Separate same-currency comparisons from actual exchange rates; GBP and GBp are the same currency in different units, so there is no rate to retrieve for that pair either, and the unit difference is applied as a factor during conversion.
        currencies_to_collect_data_for = [
            currency
            for currency in self._currencies
            if currencies_model.get_major_currency(currency[:3])
            != currencies_model.get_major_currency(currency[3:6])
        ]
        currencies_to_fill_to_one = [
            currency
            for currency in self._currencies
            if currencies_model.get_major_currency(currency[:3])
            == currencies_model.get_major_currency(currency[3:6])
        ]

        if self._daily_exchange_rate_data.empty or overwrite:
            if currencies_to_collect_data_for:
                # A handful of pairs need a different ticker than the generic BASE+QUOTE+"=X" (see currencies_model.NATIVE_MINOR_UNIT_TICKERS).
                fx_ticker_map = {
                    currency: currencies_model.get_fx_ticker(
                        currency[:3], currency[3:6]
                    )
                    for currency in currencies_to_collect_data_for
                }

                self._daily_exchange_rate_data, _ = _get_historical_data(
                    tickers=list(fx_ticker_map.values()),
                    api_key=self._api_key,
                    enforce_source=self._enforce_source,
                    start=self._lookback_start_date,
                    end=self._end_date,
                    interval="1d",
                    return_column=return_column,
                    include_dividends=False,
                    fill_nan=fill_nan,
                    # Kept unrounded: the statements are translated from these rates and a rate
                    # such as KRWUSD or VNDUSD loses most (or all) of its value at four decimals.
                    # Only what this method returns is rounded.
                    rounding=None,
                    sleep_timer=self._sleep_timer,
                    show_ticker_seperation=show_ticker_seperation,
                    log_message="Obtaining currency exchange data",
                    user_subscription=self._fmp_plan,
                    cache=self._cache,
                )

                if self._daily_exchange_rate_data.empty:
                    # None of the requested pairs resolved on either provider; a NaN-filled placeholder keeps a valid date index so conversion can warn per ticker instead of failing outright.
                    self._daily_exchange_rate_data = pd.DataFrame(
                        data=float("nan"),
                        index=pd.PeriodIndex(
                            pd.date_range(
                                start=self._lookback_start_date,
                                end=self._end_date,
                                freq="D",
                            )
                        ),
                        columns=pd.MultiIndex.from_product(
                            [
                                [
                                    "Open",
                                    "High",
                                    "Low",
                                    "Close",
                                    "Adj Close",
                                    "Volume",
                                    "Return",
                                    "Cumulative Return",
                                ],
                                currencies_to_collect_data_for,
                            ]
                        ),
                    )
                else:
                    # Map the requested ticker symbols back onto the canonical currency identifiers used everywhere else.
                    self._daily_exchange_rate_data = (
                        self._daily_exchange_rate_data.rename(
                            columns={v: k for k, v in fx_ticker_map.items()}, level=1
                        )
                    )
            else:
                # A placeholder DataFrame for when no conversion is needed.
                self._daily_exchange_rate_data = pd.DataFrame(
                    data=1,
                    index=pd.PeriodIndex(
                        pd.date_range(
                            start=self._lookback_start_date,
                            end=self._end_date,
                            freq="D",
                        )
                    ),
                    columns=pd.MultiIndex.from_tuples(
                        [
                            ("Open", "USDUSD=X"),
                            ("High", "USDUSD=X"),
                            ("Low", "USDUSD=X"),
                            ("Close", "USDUSD=X"),
                            ("Adj Close", "USDUSD=X"),
                            ("Volume", "USDUSD=X"),
                            ("Return", "USDUSD=X"),
                            ("Cumulative Return", "USDUSD=X"),
                        ]
                    ),
                )

            # A ticker such as USDUSD=X should always be 1, added here.
            if currencies_to_fill_to_one:
                upper_columns = self._daily_exchange_rate_data.columns.get_level_values(
                    level=0
                ).unique()
                for currency in currencies_to_fill_to_one:
                    for column in upper_columns:
                        self._daily_exchange_rate_data[column, currency] = 1

                self._daily_exchange_rate_data = self._daily_exchange_rate_data.reindex(
                    upper_columns, axis=1, level=0
                )

        if period == "daily":
            historical_data = apply_rounding(
                self._daily_exchange_rate_data.loc[
                    self._start_date : self._end_date, :
                ],
                rounding if rounding is not None else self._rounding,
            )
            # The first row of the window has no preceding observation, so its Return stays NaN; Cumulative Return is already anchored at 1 there by its own calculation, so nothing depends on fabricating a zero.

            if len(self._currencies) == 1:
                return historical_data.xs(self._currencies[0], level=1, axis="columns")

            return historical_data

        if period == "weekly":
            self._weekly_exchange_rate_data = _convert_daily_to_other_period(
                period="weekly",
                daily_historical_data=self._daily_exchange_rate_data,
                start=self._start_date,
                end=self._end_date,
                rounding=rounding if rounding is not None else self._rounding,
            )

            historical_data = self._weekly_exchange_rate_data.loc[
                self._start_date : self._end_date, :
            ]
            # The first row of the window has no preceding observation, so its Return stays NaN; Cumulative Return is already anchored at 1 there by its own calculation, so nothing depends on fabricating a zero.

            if len(self._currencies) == 1:
                return historical_data.xs(self._currencies[0], level=1, axis="columns")

            return historical_data

        if period == "monthly":
            self._monthly_exchange_rate_data = _convert_daily_to_other_period(
                period="monthly",
                daily_historical_data=self._daily_exchange_rate_data,
                start=self._start_date,
                end=self._end_date,
                rounding=rounding if rounding is not None else self._rounding,
            )

            historical_data = self._monthly_exchange_rate_data.loc[
                self._start_date : self._end_date, :
            ]
            # The first row of the window has no preceding observation, so its Return stays NaN; Cumulative Return is already anchored at 1 there by its own calculation, so nothing depends on fabricating a zero.

            if len(self._currencies) == 1:
                return historical_data.xs(self._currencies[0], level=1, axis="columns")

            return historical_data

        if period == "quarterly":
            self._quarterly_exchange_rate_data = _convert_daily_to_other_period(
                period="quarterly",
                daily_historical_data=self._daily_exchange_rate_data,
                start=self._start_date,
                end=self._end_date,
                rounding=rounding if rounding is not None else self._rounding,
            )

            historical_data = self._quarterly_exchange_rate_data.loc[
                self._start_date : self._end_date, :
            ]
            # The first row of the window has no preceding observation, so its Return stays NaN; Cumulative Return is already anchored at 1 there by its own calculation, so nothing depends on fabricating a zero.

            if len(self._currencies) == 1:
                return historical_data.xs(self._currencies[0], level=1, axis="columns")

            return historical_data

        if period == "yearly":
            self._yearly_exchange_rate_data = _convert_daily_to_other_period(
                period="yearly",
                daily_historical_data=self._daily_exchange_rate_data,
                start=self._start_date,
                end=self._end_date,
                rounding=rounding if rounding is not None else self._rounding,
            )

            historical_data = self._yearly_exchange_rate_data.loc[
                self._start_date : self._end_date, :
            ]
            # The first row of the window has no preceding observation, so its Return stays NaN; Cumulative Return is already anchored at 1 there by its own calculation, so nothing depends on fabricating a zero.

            return historical_data

        raise ValueError(
            "Please choose from daily, weekly, monthly, quarterly or yearly as period."
        )

    def get_balance_sheet_statement(
        self,
        enforce_source: str | None = None,
        overwrite: bool = False,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        show_columns: list[str] | None = None,
    ):
        """
        Retrieves the balance sheet statement data for the specified tickers. The balance sheet statement
        is a financial statement that provides a snapshot of a company's financial position at a specific
        point in time. It shows the company's assets, liabilities, and shareholders' equity. The balance sheet
        statement is divided into three main sections:

        - Assets: Assets are resources owned by the company that have economic value and can be used to
        generate revenue. Assets are typically divided into current assets and non-current assets.
        - Liabilities: Liabilities are obligations that the company owes to external parties. Liabilities
        are also divided into current liabilities and non-current liabilities.
        - Shareholders' Equity: Shareholders' equity represents the company's net worth or book value. It
        is calculated as the difference between the company's assets and liabilities.

        Note that the balance sheet statement is a financial statement that provides a snapshot of a
        company's financial position at a specific point in time. Therefore, trailing results are not
        available for this statement.

        With convert_currency, the statement is converted to the currency of the historical data at the
        closing rate, the exchange rate at the end of each period (IAS 21.39).

        Also known as: assets, liabilities, shareholders equity, financial position.

        Args:
            enforce_source (str | None, optional): Forces this specific call to use a given
                source, either "FinancialModelingPrep" or "YahooFinance". This takes precedence
                over the source set on the Toolkit itself, so one instance can pull historical
                data from the free Yahoo Finance source while still using a FinancialModelingPrep
                key for the financial statements (or the other way around). Defaults to None,
                which falls back to the Toolkit's own enforce_source.
            overwrite (bool): Defines whether to overwrite the existing data.
            rounding (int): Defines the number of decimal places to round the data to.
            growth (bool): Defines whether to return the growth of the data.
            lag (int | list[int]): Defines the number of periods to lag the growth data by.
            E.g. 4 with quarterly data gives the growth against the same quarter a year earlier.
            show_columns (list[str] | None): A list of column names to keep in the result. Invalid
            names are reported and ignored. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: A pandas DataFrame with the retrieved balance sheet statement data.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["NEE", "DUK", "SO"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.get_balance_sheet_statement(show_columns=["Total Debt"])
        ```

        Which returns:

        |     |       2021 |       2022 |       2023 |       2024 |       2025 |
        |:----|-----------:|-----------:|-----------:|-----------:|-----------:|
        | DUK | 6.8263e+10 | 7.4911e+10 | 8.0457e+10 | 8.523e+10  | 9.0869e+10 |
        | NEE | 5.4827e+10 | 6.4966e+10 | 7.3211e+10 | 8.2333e+10 | 9.5619e+10 |
        | SO  | 5.547e+10  | 5.9135e+10 | 6.349e+10  | 6.6277e+10 | 7.4075e+10 |
        """
        convert_currency = bool(
            self._convert_currency
            and (self._balance_sheet_statement.empty or overwrite)
        )

        if enforce_source is not None and enforce_source not in [
            "FinancialModelingPrep",
            "YahooFinance",
        ]:
            raise ValueError(
                "The enforce_source parameter must be either 'FinancialModelingPrep' or 'YahooFinance'."
            )

        # A per-call enforce_source overrides the one the Toolkit was initialised with.
        source = enforce_source if enforce_source is not None else self._enforce_source

        if (
            not self._api_key
            and self._balance_sheet_statement.empty
            and source == "FinancialModelingPrep"
        ):
            logger.error(
                "The requested data requires the api_key parameter to be set or the enforce_source "
                "parameter set to 'YahooFinance', consider obtaining a key with the following link: "
                "https://www.jeroenbouma.com/fmp"
                "\nThe free plan allows for 250 requests per day, a limit of 5 years and has no "
                "quarterly data. Consider upgrading your plan. You can get 15% off by using the "
                "above affiliate link which also supports the project."
            )
            return None

        # Correct for the case where a Portfolio ticker exists
        ticker_list = [ticker for ticker in self._tickers if ticker != "Portfolio"]

        if self._balance_sheet_statement.empty or overwrite:
            (
                self._balance_sheet_statement,
                self._statistics_statement,
                self._invalid_tickers,
                _fy_adj,
            ) = collect_financial_statements(
                tickers=ticker_list,
                statement="balance",
                api_key=self._api_key,
                quarter=self._quarterly,
                start_date=self._lookback_start_date,
                end_date=self._end_date,
                rounding=rounding if rounding is not None else self._rounding,
                fmp_statement_format=self._fmp_balance_sheet_statement_generic,
                fmp_statistics_format=self._fmp_statistics_statement_generic,
                yf_statistics_format=self._yf_statistics_statement_generic,
                yf_statement_format=self._yf_balance_sheet_statement_generic,
                sleep_timer=self._sleep_timer,
                user_subscription=self._fmp_plan,
                enforce_source=source,
                cache=self._cache,
            )
            self._fiscal_year_adjustments.update(_fy_adj)

            if convert_currency:
                self.get_exchange_rates(
                    period="quarterly" if self._quarterly else "yearly",
                )

                if not self._statement_currencies.empty:
                    # A balance sheet is a position at the end of the period, so it is converted at
                    # the closing rate (IAS 21.39): the rate on the last day of the fiscal period.
                    self._balance_sheet_statement = currencies_model.convert_currencies(
                        financial_statement_data=self._balance_sheet_statement,
                        financial_statement_currencies=self._statement_currencies,
                        exchange_rate_data=self._get_statement_exchange_rates(
                            self._balance_sheet_statement, rate="closing"
                        ),
                        financial_statement_name="balance sheet statement",
                    )

        if self._remove_invalid_tickers:
            self._tickers = [
                ticker
                for ticker in self._tickers
                if ticker not in self._invalid_tickers
            ]

        balance_sheet_statement = self._balance_sheet_statement

        if growth:
            self._balance_sheet_statement_growth = calculate_growth(
                balance_sheet_statement,
                lag=lag,
                rounding=rounding if rounding is not None else self._rounding,
                axis="columns",
            ).truncate(before=self._start_date, axis=1)

        balance_sheet_statement = apply_rounding(
            balance_sheet_statement,
            rounding if rounding is not None else self._rounding,
        )

        balance_sheet_statement = balance_sheet_statement.truncate(
            before=self._start_date, axis=1
        )

        if len(self._tickers) == 1 and not self._balance_sheet_statement.empty:
            result = (
                self._balance_sheet_statement_growth.loc[self._tickers[0]]
                if growth
                else balance_sheet_statement.loc[self._tickers[0]]
            )
            return filter_columns(result, show_columns)

        result = (
            self._balance_sheet_statement_growth if growth else balance_sheet_statement
        )
        return filter_columns(result, show_columns)

    def get_income_statement(
        self,
        enforce_source: str | None = None,
        overwrite: bool = False,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
        show_columns: list[str] | None = None,
    ):
        """
        Retrieves the income statement data for the specified tickers. The income statement is a financial
        statement that shows a company's revenues and expenses over a specific period. It is used to calculate
        a company's net income.

        The income statement is a financial statement that shows a company's revenues and expenses over a specific
        period. Therefore, trailing results are available for this statement. A trailing result sums each
        item over the window, except the Weighted Average Shares, Weighted Average Shares Diluted and Tax
        Rate for Calcs rows, which keep the value of the latest period.

        With convert_currency, the statement is converted to the currency of the historical data at the
        average exchange rate of each period, as IAS 21.39-40 does for income and expenses. EPS and EPS
        Diluted are converted as well; share counts and ratios are not.

        Also known as: profit and loss, revenue, net income, earnings, P&L statement.

        Args:
            enforce_source (str | None, optional): Forces this specific call to use a given
                source, either "FinancialModelingPrep" or "YahooFinance". This takes precedence
                over the source set on the Toolkit itself, so one instance can pull historical
                data from the free Yahoo Finance source while still using a FinancialModelingPrep
                key for the financial statements (or the other way around). Defaults to None,
                which falls back to the Toolkit's own enforce_source.
            overwrite (bool): Defines whether to overwrite the existing data.
            rounding (int): Defines the number of decimal places to round the data to.
            growth (bool): Defines whether to return the growth of the data.
            lag (int | list[int]): Defines the number of periods to lag the growth data by.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.
            show_columns (list[str] | None): A list of column names to keep in the result. Invalid
            names are reported and ignored. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: A pandas DataFrame with the retrieved income statement data.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["NKE"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.get_income_statement(show_columns=["Revenue", "Net Income"])
        ```

        Which returns:

        |            |      2021 |       2022 |       2023 |       2024 |       2025 |
        |:-----------|----------:|-----------:|-----------:|-----------:|-----------:|
        | Revenue    | 4.671e+10 | 5.1217e+10 | 5.1362e+10 | 4.6309e+10 | 4.6398e+10 |
        | Net Income | 6.046e+09 | 5.07e+09   | 5.7e+09    | 3.219e+09  | 3.108e+09  |
        """
        convert_currency = bool(
            self._convert_currency and (self._income_statement.empty or overwrite)
        )

        if enforce_source is not None and enforce_source not in [
            "FinancialModelingPrep",
            "YahooFinance",
        ]:
            raise ValueError(
                "The enforce_source parameter must be either 'FinancialModelingPrep' or 'YahooFinance'."
            )

        # A per-call enforce_source overrides the one the Toolkit was initialised with.
        source = enforce_source if enforce_source is not None else self._enforce_source

        if (
            not self._api_key
            and self._income_statement.empty
            and source == "FinancialModelingPrep"
        ):
            logger.error(
                "The requested data requires the api_key parameter to be set or the enforce_source "
                "parameter set to 'YahooFinance', consider obtaining a key with the following link: "
                "https://www.jeroenbouma.com/fmp"
                "\nThe free plan allows for 250 requests per day, a limit of 5 years and has no "
                "quarterly data. Consider upgrading your plan. You can get 15% off by using the "
                "above affiliate link which also supports the project."
            )
            return None

        # Correct for the case where a Portfolio ticker exists
        ticker_list = [ticker for ticker in self._tickers if ticker != "Portfolio"]

        if self._income_statement.empty or overwrite:
            (
                self._income_statement,
                self._statistics_statement,
                self._invalid_tickers,
                _fy_adj,
            ) = collect_financial_statements(
                tickers=ticker_list,
                statement="income",
                api_key=self._api_key,
                quarter=self._quarterly,
                start_date=self._lookback_start_date,
                end_date=self._end_date,
                rounding=rounding if rounding is not None else self._rounding,
                fmp_statement_format=self._fmp_income_statement_generic,
                fmp_statistics_format=self._fmp_statistics_statement_generic,
                yf_statistics_format=self._yf_statistics_statement_generic,
                yf_statement_format=self._yf_income_statement_generic,
                sleep_timer=self._sleep_timer,
                user_subscription=self._fmp_plan,
                enforce_source=source,
                cache=self._cache,
            )
            self._fiscal_year_adjustments.update(_fy_adj)

            if convert_currency:
                self.get_exchange_rates(
                    period="quarterly" if self._quarterly else "yearly",
                )
                if not self._statement_currencies.empty:
                    self._income_statement = currencies_model.convert_currencies(
                        financial_statement_data=self._income_statement,
                        financial_statement_currencies=self._statement_currencies,
                        # Income and expenses at the average rate of the fiscal period (IAS 21.39-40), EPS included.  # noqa: E501
                        exchange_rate_data=self._get_statement_exchange_rates(
                            self._income_statement, rate="average"
                        ),
                        items_not_to_adjust=[
                            "Gross Profit Ratio",
                            "EBITDA Ratio",
                            "Operating Income Ratio",
                            "Income Before Tax Ratio",
                            "Net Income Ratio",
                            "Tax Rate for Calcs",
                            "Weighted Average Shares",
                            "Weighted Average Shares Diluted",
                        ],
                        financial_statement_name="income statement",
                    )

        if self._remove_invalid_tickers:
            self._tickers = [
                ticker
                for ticker in self._tickers
                if ticker not in self._invalid_tickers
            ]

        income_statement = self._income_statement

        if trailing:
            # Share counts and the tax rate are no amounts earned over the period, so a sum over
            # the window means nothing for them: they are kept at the value of the latest period.
            not_summed = income_statement.loc[
                income_statement.index.get_level_values(1).isin(
                    [
                        "Weighted Average Shares",
                        "Weighted Average Shares Diluted",
                        "Tax Rate for Calcs",
                    ]
                )
            ]

            # The rolling window is calculated for the rest of the income statement.
            income_statement = self._income_statement.T.rolling(trailing).sum().T

            income_statement.loc[not_summed.index] = not_summed

        if growth:
            self._income_statement_growth = calculate_growth(
                income_statement,
                lag=lag,
                rounding=rounding if rounding is not None else self._rounding,
                axis="columns",
            ).truncate(before=self._start_date, axis=1)

        income_statement = apply_rounding(
            income_statement, rounding if rounding is not None else self._rounding
        )

        income_statement = income_statement.truncate(before=self._start_date, axis=1)

        if len(self._tickers) == 1 and not self._income_statement.empty:
            result = (
                self._income_statement_growth.loc[self._tickers[0]]
                if growth
                else income_statement.loc[self._tickers[0]]
            )
            return filter_columns(result, show_columns)

        result = self._income_statement_growth if growth else income_statement
        return filter_columns(result, show_columns)

    def get_cash_flow_statement(
        self,
        enforce_source: str | None = None,
        overwrite: bool = False,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
        show_columns: list[str] | None = None,
    ):
        """
        Retrieves the cash flow statement data for the specified tickers. The cash flow statement is a financial
        statement that shows how changes in balance sheet accounts and income affect cash and cash equivalents.
        It breaks the analysis down to operating, investing and financing activities.

        The cash flow statement is a financial statement that shows how changes in balance sheet accounts and income
        affect cash and cash equivalents. Therefore, trailing results are available for this statement. A
        trailing result sums each cash flow over the window, while Cash Beginning of Period is the cash at
        the beginning of the first period of the window and Cash End of Period the cash at the end of the
        last one.

        With convert_currency, the cash flows are converted to the currency of the historical data at the
        average exchange rate of each period (IAS 7.25), Cash End of Period at the closing rate of the
        period and Cash Beginning of Period at the closing rate of the period before, so it equals the
        Cash End of Period before it. Net Change in Cash is the change in the converted cash and Forex
        Changes on Cash holds the effect of the exchange rate changes on the cash held (IAS 7.28).

        Also known as: operating cash flow, investing activities, financing activities.

        Args:
            enforce_source (str | None, optional): Forces this specific call to use a given
                source, either "FinancialModelingPrep" or "YahooFinance". This takes precedence
                over the source set on the Toolkit itself, so one instance can pull historical
                data from the free Yahoo Finance source while still using a FinancialModelingPrep
                key for the financial statements (or the other way around). Defaults to None,
                which falls back to the Toolkit's own enforce_source.
            overwrite (bool): Defines whether to overwrite the existing data.
            rounding (int): Defines the number of decimal places to round the data to.
            growth (bool): Defines whether to return the growth of the data.
            lag (int | list[int]): Defines the number of periods to lag the growth data by.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.
            show_columns (list[str] | None): A list of column names to keep in the result. Invalid
            names are reported and ignored. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: A pandas DataFrame with the retrieved cash flow statement data.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["CVX"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.get_cash_flow_statement(
            show_columns=["Operating Cash Flow", "Capital Expenditure", "Free Cash Flow", "Dividends Paid"]
        )
        ```

        Which returns:

        |                     |        2021 |        2022 |        2023 |        2024 |        2025 |
        |:--------------------|------------:|------------:|------------:|------------:|------------:|
        | Operating Cash Flow |  2.9187e+10 |  4.9602e+10 |  3.5609e+10 |  3.1492e+10 |  3.3939e+10 |
        | Capital Expenditure | -8.1e+09    | -1.1974e+10 | -1.5829e+10 | -1.6448e+10 | -1.7347e+10 |
        | Free Cash Flow      |  2.1087e+10 |  3.7628e+10 |  1.978e+10  |  1.5044e+10 |  1.6592e+10 |
        | Dividends Paid      | -1.0179e+10 | -1.0968e+10 | -1.1336e+10 | -1.18e+10   | -1.2751e+10 |
        """
        convert_currency = bool(
            self._convert_currency and (self._cash_flow_statement.empty or overwrite)
        )

        if enforce_source is not None and enforce_source not in [
            "FinancialModelingPrep",
            "YahooFinance",
        ]:
            raise ValueError(
                "The enforce_source parameter must be either 'FinancialModelingPrep' or 'YahooFinance'."
            )

        # A per-call enforce_source overrides the one the Toolkit was initialised with.
        source = enforce_source if enforce_source is not None else self._enforce_source

        if (
            not self._api_key
            and self._cash_flow_statement.empty
            and source == "FinancialModelingPrep"
        ):
            logger.error(
                "The requested data requires the api_key parameter to be set or the enforce_source "
                "parameter set to 'YahooFinance', consider obtaining a key with the following link: "
                "https://www.jeroenbouma.com/fmp"
                "\nThe free plan allows for 250 requests per day, a limit of 5 years and has no "
                "quarterly data. Consider upgrading your plan. You can get 15% off by using the "
                "above affiliate link which also supports the project."
            )
            return None

        # Correct for the case where a Portfolio ticker exists
        ticker_list = [ticker for ticker in self._tickers if ticker != "Portfolio"]

        if self._cash_flow_statement.empty or overwrite:
            (
                self._cash_flow_statement,
                self._statistics_statement,
                self._invalid_tickers,
                _fy_adj,
            ) = collect_financial_statements(
                tickers=ticker_list,
                statement="cashflow",
                api_key=self._api_key,
                quarter=self._quarterly,
                start_date=self._lookback_start_date,
                end_date=self._end_date,
                rounding=rounding if rounding is not None else self._rounding,
                fmp_statement_format=self._fmp_cash_flow_statement_generic,
                fmp_statistics_format=self._fmp_statistics_statement_generic,
                yf_statistics_format=self._yf_statistics_statement_generic,
                yf_statement_format=self._yf_cash_flow_statement_generic,
                sleep_timer=self._sleep_timer,
                user_subscription=self._fmp_plan,
                enforce_source=source,
                cache=self._cache,
            )
            self._fiscal_year_adjustments.update(_fy_adj)

            if convert_currency:
                self.get_exchange_rates(
                    period="quarterly" if self._quarterly else "yearly",
                )

                if not self._statement_currencies.empty:
                    # Cash flows at the average rate of the fiscal period (IAS 7.25), the cash held
                    # at the closing rates: the cash at the end at the rate on the period end, the
                    # cash at the beginning at the rate on the previous period end, so Cash
                    # Beginning of Period equals the Cash End of Period of the period before it.
                    # The difference is reported in Forex Changes on Cash (IAS 7.28). The first
                    # period's opening rate lies before the exchange rate data when the data starts
                    # at that period, which leaves its Cash Beginning of Period NaN.
                    self._cash_flow_statement = currencies_model.convert_currencies(
                        financial_statement_data=self._cash_flow_statement,
                        financial_statement_currencies=self._statement_currencies,
                        exchange_rate_data=self._get_statement_exchange_rates(
                            self._cash_flow_statement, rate="average"
                        ),
                        financial_statement_name="cash flow statement",
                        item_exchange_rate_data={
                            "Cash End of Period": self._get_statement_exchange_rates(
                                self._cash_flow_statement, rate="closing"
                            ),
                            "Cash Beginning of Period": self._get_statement_exchange_rates(
                                self._cash_flow_statement, rate="opening"
                            ),
                        },
                    )

        if self._remove_invalid_tickers:
            self._tickers = [
                ticker
                for ticker in self._tickers
                if ticker not in self._invalid_tickers
            ]

        cash_flow_statement = self._cash_flow_statement

        if trailing:
            cash_flow_statement = self._cash_flow_statement.T.rolling(trailing).sum().T

            # Cash held is a balance, not a flow: the window starts with the cash at the
            # beginning of its first period and ends with the cash at the end of its last one.
            items = cash_flow_statement.index.get_level_values(1)
            cash_end = items == "Cash End of Period"
            cash_beginning = items == "Cash Beginning of Period"
            cash_flow_statement.loc[cash_end] = self._cash_flow_statement.loc[
                cash_end
            ].where(cash_flow_statement.loc[cash_end].notna())
            cash_flow_statement.loc[cash_beginning] = (
                self._cash_flow_statement.loc[cash_beginning]
                .shift(trailing - 1, axis=1)
                .where(cash_flow_statement.loc[cash_beginning].notna())
            )

        if growth:
            self._cash_flow_statement_growth = calculate_growth(
                cash_flow_statement,
                lag=lag,
                rounding=rounding if rounding is not None else self._rounding,
                axis="columns",
            ).truncate(before=self._start_date, axis=1)

        cash_flow_statement = apply_rounding(
            cash_flow_statement, rounding if rounding is not None else self._rounding
        )

        cash_flow_statement = cash_flow_statement.truncate(
            before=self._start_date, axis=1
        )

        if len(self._tickers) == 1 and not self._cash_flow_statement.empty:
            result = (
                self._cash_flow_statement_growth.loc[self._tickers[0]]
                if growth
                else cash_flow_statement.loc[self._tickers[0]]
            )
            return filter_columns(result, show_columns)

        result = self._cash_flow_statement_growth if growth else cash_flow_statement
        return filter_columns(result, show_columns)

    def get_statistics_statement(
        self,
        enforce_source: str | None = None,
        overwrite: bool = False,
        rounding: int | None = None,
        show_columns: list[str] | None = None,
    ):
        """
        Retrieves the filing details of each financial statement period: the reported currency, the CIK,
        the filing and acceptance dates, the fiscal period and the links to the filing. With convert_currency,
        the Toolkit uses the reported currency to convert the financial statements to the currency of the
        historical data.

        Note that this also obtains the balance sheet statement at the same time given that it's the same
        API call. This is done to reduce the number of API calls to FinancialModelingPrep.

        Also known as: filing details, reporting metadata, reported currency.

        Args:
            enforce_source (str | None, optional): Forces this specific call to use a given
                source, either "FinancialModelingPrep" or "YahooFinance". This takes precedence
                over the source set on the Toolkit itself, so one instance can pull historical
                data from the free Yahoo Finance source while still using a FinancialModelingPrep
                key for the financial statements (or the other way around). Defaults to None,
                which falls back to the Toolkit's own enforce_source.
            overwrite (bool): Defines whether to overwrite the existing data.
            rounding (int): Defines the number of decimal places to round the data to.
            show_columns (list[str] | None): A list of column names to keep in the result. Invalid
            names are reported and ignored. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: A pandas DataFrame with the retrieved statistics statement data.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["KO", "PEP"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.get_statistics_statement(show_columns=["Accepted Date"]).T
        ```

        Which returns:

        | date   | KO                  | PEP                 |
        |:-------|:--------------------|:--------------------|
        | 2020   | 2021-02-25 13:47:17 | 2021-02-10 17:41:49 |
        | 2021   | 2022-02-22 15:45:48 | 2022-02-09 18:03:39 |
        | 2022   | 2023-02-21 16:03:20 | 2023-02-08 18:06:37 |
        | 2023   | 2024-02-20 13:21:24 | 2024-02-08 17:58:51 |
        | 2024   | 2025-02-20 15:17:12 | 2025-02-03 18:22:47 |
        | 2025   | 2026-02-20 09:46:32 | 2026-02-02 18:07:04 |
        """
        if not self._api_key and self._statistics_statement.empty:
            logger.error(
                "The requested data requires the api_key parameter to be set, consider "
                "obtaining a key with the following link: https://www.jeroenbouma.com/fmp"
                "\nThe free plan allows for 250 requests per day, a limit of 5 years and has no "
                "quarterly data. Consider upgrading your plan. You can get 15% off by using the "
                "above affiliate link which also supports the project."
            )
            return None

        if enforce_source is not None and enforce_source not in [
            "FinancialModelingPrep",
            "YahooFinance",
        ]:
            raise ValueError(
                "The enforce_source parameter must be either 'FinancialModelingPrep' or 'YahooFinance'."
            )

        # Correct for the case where a Portfolio ticker exists
        ticker_list = [ticker for ticker in self._tickers if ticker != "Portfolio"]

        if self._statistics_statement.empty or overwrite:
            (
                self._balance_sheet_statement,
                self._statistics_statement,
                self._invalid_tickers,
                _fy_adj,
            ) = collect_financial_statements(
                tickers=ticker_list,
                statement="balance",
                api_key=self._api_key,
                quarter=self._quarterly,
                start_date=self._lookback_start_date,
                end_date=self._end_date,
                rounding=rounding if rounding is not None else self._rounding,
                fmp_statement_format=self._fmp_balance_sheet_statement_generic,
                fmp_statistics_format=self._fmp_statistics_statement_generic,
                yf_statistics_format=self._yf_statistics_statement_generic,
                yf_statement_format=self._yf_balance_sheet_statement_generic,
                sleep_timer=self._sleep_timer,
                user_subscription=self._fmp_plan,
                enforce_source=(
                    enforce_source
                    if enforce_source is not None
                    else self._enforce_source
                ),
                cache=self._cache,
            )
            self._fiscal_year_adjustments.update(_fy_adj)

        if self._remove_invalid_tickers:
            self._tickers = [
                ticker
                for ticker in self._tickers
                if ticker not in self._invalid_tickers
            ]

        if len(self._tickers) == 1 and not self._statistics_statement.empty:
            return filter_columns(
                self._statistics_statement.loc[self._tickers[0]], show_columns
            )

        return filter_columns(self._statistics_statement, show_columns)

    def get_normalization_files(self, path: str = ""):
        """
        Copies the normalization files to a folder based on path. By default, this is the path
        of the 'Downloads' folder.

        This function is relevant if you want to supply your own datasets. See the external
        datasets guide for how: https://www.jeroenbouma.com/projects/financetoolkit/docs/external-datasets

        The same function is available without a Toolkit instance as
        `from financetoolkit import get_normalization_files`.

        Args:
            path (str, optional): The folder to save the files to. Defaults to "", which
                saves them to the 'Downloads' folder of your home directory.

        Returns:
            None: The four csv files (balance, income, cash and statistics) are saved to
            the chosen folder.
        """
        _get_normalization_files(path)

    def _get_module_statement(self, statement: str, module: str) -> pd.DataFrame:
        """
        Returns a financial statement for the Ratios or Models module, retrieving it when a
        calculation first needs it. When the statement cannot be retrieved, the other
        statements are retrieved as well to tell whether any data is available at all.

        Args:
            statement (str): "balance_sheet_statement", "income_statement" or
                "cash_flow_statement".
            module (str): "Ratios" or "Models", the module the statement is for.

        Returns:
            pd.DataFrame: The statement, which is empty when only it could not be retrieved.

        Raises:
            ValueError: When an API key is required but missing, or when none of the
                statements can be retrieved.
        """
        if getattr(self, f"_{statement}").empty:
            if not self._api_key and module == "Models":
                raise ValueError(
                    "The models class requires an API key from FinancialModelPrep. "
                    "Get an API key here: https://www.jeroenbouma.com/fmp"
                )
            if not self._api_key and self._enforce_source == "FinancialModelingPrep":
                raise ValueError(
                    "The ratios class requires an API key from FinancialModelPrep if you wish to enforce the usage. "
                    "of Financial Modeling Prep. Get an API key here: https://www.jeroenbouma.com/fmp"
                )

            logger.info("Obtaining financial statements")
            getattr(self, f"get_{statement}")()

        if getattr(self, f"_{statement}").empty:
            statements = [
                "balance_sheet_statement",
                "income_statement",
                "cash_flow_statement",
            ]
            for other in statements:
                if other != statement and getattr(self, f"_{other}").empty:
                    getattr(self, f"get_{other}")()

            if all(getattr(self, f"_{other}").empty for other in statements):
                raise ValueError(
                    f"The datasets could not be populated and therefore the {module} class cannot "
                    "calculate this. This is usually because no tickers are equities, you have "
                    "reached the API limit or entered an invalid API key."
                )

        return getattr(self, f"_{statement}")

    def _get_statement_tickers(self, module: str) -> list[str]:
        """
        Returns the tickers with financial statements, which leaves out those without (such as
        ETFs). The statements hold the same tickers, so the first one already retrieved names
        them; otherwise the balance sheet statement is retrieved.

        Args:
            module (str): "Ratios" or "Models", the module the tickers are for.

        Returns:
            list[str]: The tickers of the financial statements.
        """
        retrieved = [
            statement
            for statement in (
                self._balance_sheet_statement,
                self._income_statement,
                self._cash_flow_statement,
            )
            if not statement.empty
        ]
        statement = (
            retrieved[0]
            if retrieved
            else self._get_module_statement("balance_sheet_statement", module)
        )

        return statement.index.get_level_values(0).unique().tolist()

    def _get_module_historical_data(self, period: str) -> pd.DataFrame:
        """
        Returns the historical data of a period for the modules, retrieved when a calculation
        first asks for that period. The daily data is downloaded once and the weekly, monthly,
        quarterly and yearly data is converted from it; the intraday data is only downloaded
        when the Toolkit has an intraday_period.

        Args:
            period (str): "intraday", "daily", "weekly", "monthly", "quarterly" or "yearly".

        Returns:
            pd.DataFrame: The historical data of the period, empty for intraday data without an
            intraday_period.

        Raises:
            KeyError: For any other period, as a dictionary of the periods would.
        """
        if period == "intraday":
            if self._intraday_period:
                self.get_intraday_data(period=self._intraday_period)
            return self._intraday_historical_data

        if period not in ("daily", "weekly", "monthly", "quarterly", "yearly"):
            raise KeyError(period)

        self.get_historical_data(period=period)

        return getattr(self, f"_{period}_historical_data")

    def _get_module_risk_free_rate(self, period: str) -> pd.DataFrame:
        """
        Returns the risk free rate of a period for the modules, retrieved with the historical
        data of that period when a calculation first asks for it.

        Args:
            period (str): "daily", "weekly", "monthly", "quarterly" or "yearly".

        Returns:
            pd.DataFrame: The risk free rate of the period.

        Raises:
            KeyError: For any other period, as a dictionary of the periods would.
        """
        if period not in ("daily", "weekly", "monthly", "quarterly", "yearly"):
            raise KeyError(period)

        self.get_historical_data(period=period)

        return getattr(self, f"_{period}_risk_free_rate")

    def _get_historical_tickers(self, include_benchmark: bool) -> list[str]:
        """
        Returns the tickers of the daily historical data, retrieving it when needed, followed
        by the Portfolio when the Toolkit holds one.

        Args:
            include_benchmark (bool): Whether to keep the Benchmark.

        Returns:
            list[str]: The tickers of the historical data.
        """
        tickers = (
            self._get_module_historical_data("daily")
            .columns.get_level_values(1)
            .unique()
            .tolist()
        )

        if not include_benchmark and "Benchmark" in tickers:
            tickers.remove("Benchmark")

        return tickers + ["Portfolio"] if "Portfolio" in self._tickers else tickers

    def _get_period_end_dates(
        self, statement: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """
        Returns the last day of the fiscal period behind every statement column per ticker.

        Statement columns are calendar periods (see "Fiscal periods and calendar periods" on
        the Toolkit), while the fiscal period they hold ends on its own date, e.g. 30 June for
        Microsoft's yearly statements and the end of January for NVIDIA's. These dates are
        what the daily ratios apply the figures from and what the closing and average exchange
        rates of the statements are taken over. They come from the Period End Date statistic
        (FinancialModelingPrep) and otherwise from the latest date the period can end on given
        its label and _fiscal_year_adjustments (see helpers.determine_period_end_dates).

        Args:
            statement (pd.DataFrame | None, optional): The statement whose periods and tickers
                to use. Defaults to None, which uses the balance sheet statement.

        Returns:
            pd.DataFrame: The period end dates with the tickers as index and the periods as
            columns.
        """
        statement = self._balance_sheet_statement if statement is None else statement

        if statement.empty:
            return pd.DataFrame()

        return determine_period_end_dates(
            periods=pd.PeriodIndex(statement.columns),
            tickers=statement.index.get_level_values(0).unique().tolist(),
            quarter=self._quarterly,
            statistics_statement=self._statistics_statement,
            fiscal_year_adjustments=self._fiscal_year_adjustments,
        )

    def _get_statement_exchange_rates(
        self, statement: pd.DataFrame, rate: str
    ) -> pd.DataFrame:
        """
        Returns the exchange rate of each statement period per ticker, taken from the
        unrounded daily exchange rates over the ticker's fiscal periods.

        IAS 21.39-40 translates the balance sheet at the closing rate, the rate on the last
        day of the fiscal period, and income and expenses at the rates on the dates of the
        transactions, for which the average rate of the fiscal period is the usual
        approximation; IAS 7.25 does the same for cash flows. The opening rate, the closing
        rate of the previous period, translates the cash at the beginning of the period.

        Args:
            statement (pd.DataFrame): The statement to translate, which sets the periods and
                tickers.
            rate (str): "closing", "average" or "opening".

        Returns:
            pd.DataFrame: The rate per period (index) and ticker (columns).
        """
        # Not rounded: a rate such as JPYUSD needs more decimals than the Toolkit's rounding.
        return currencies_model.get_statement_exchange_rates(
            daily_exchange_rates=self._daily_exchange_rate_data["Adj Close"],
            period_end_dates=self._get_period_end_dates(statement),
            statement_currencies=self._statement_currencies,
            quarter=self._quarterly,
            rate=rate,
        )

    def _average_risk_free_rate(
        self,
        period: str,
        treasury_data: pd.DataFrame,
        risk_free_rate: str,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Builds the risk free rate of each week, month, quarter or year from the average of
        the daily Treasury yields within it.

        The yield is kept annualized (the Performance, Risk and Models modules convert it to
        the period frequency), but the yield of a period is the average of its daily yields
        rather than the yield on its last day. The converted rate then approximates the
        return a holder of the Treasury earned over the period, and does not depend on where
        the yield happened to close at the end of the period, which a return measured over
        that period could not have known at its start either. The Return, Cumulative Return
        and Volume columns are kept from the period treasury data.

        Args:
            period (str): The period (weekly, monthly, quarterly or yearly).
            treasury_data (pd.DataFrame): The treasury data converted to the period, with the
                metric as the first column level and the maturity as the second.
            risk_free_rate (str): The maturity used as the risk free rate, e.g. "10 Year".
            rounding (int | None, optional): The number of decimals to round to, or None to
                use the Toolkit's rounding. Defaults to None.

        Returns:
            pd.DataFrame: The risk free rate per period, with the metric as the columns.
        """
        period_rate = treasury_data.xs(risk_free_rate, level=1, axis=1).copy()
        yield_columns = [
            column
            for column in ["Open", "High", "Low", "Close", "Adj Close"]
            if column in period_rate.columns
        ]

        average_yield = _calculate_period_average(
            period=period,
            daily_data=self._daily_treasury_data.xs(risk_free_rate, level=1, axis=1),
            columns=yield_columns,
            rounding=rounding if rounding is not None else self._rounding,
        )

        period_rate[yield_columns] = average_yield.reindex(period_rate.index)[
            yield_columns
        ]

        return period_rate

    def _align_treasury_data_to_trading_days(
        self, treasury_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Fills the treasury series on equity trading days the bond market was closed for.

        The bond market observes holidays the equity market does not, Columbus Day and
        Veterans Day among them, so a Treasury yield series has gaps on days that do
        have a stock return. Every one of those gaps produces a NaN excess return that
        is then dropped from the within-period Sharpe, Sortino, Kappa and Appraisal
        ratios and from the market timing regressions, silently shortening the sample.

        The last published yield is carried forward across such a gap, which is the
        correct reading of an instrument that simply did not trade that day. Only the
        yields are carried forward: the Return and Cumulative Return columns stay NaN on
        an inserted day, since no return was actually realized on it. Gaps before the
        first or after the last published yield are left alone.

        Args:
            treasury_data (pd.DataFrame): the daily treasury data, with the metric as
                the first column level and the maturity as the second.

        Returns:
            pd.DataFrame: the treasury data reindexed onto the trading days of the
            historical data and forward filled across bond market holidays.
        """
        if treasury_data.empty or self._daily_historical_data.empty:
            return treasury_data

        trading_days = self._daily_historical_data.index
        trading_days = trading_days[
            (trading_days >= treasury_data.index.min())
            & (trading_days <= treasury_data.index.max())
        ]

        treasury_data = treasury_data.reindex(treasury_data.index.union(trading_days))

        carried_forward = [
            column
            for column in treasury_data.columns
            if column[0] not in ("Return", "Cumulative Return")
        ]
        treasury_data[carried_forward] = treasury_data[carried_forward].ffill()

        return treasury_data

    def _collect_per_ticker(
        self,
        dataset: str,
        tickers: list[str],
        ticker_axis: str,
        collector: Callable[[list[str]], Any],
        parameters: dict | None = None,
    ) -> tuple[pd.DataFrame, list[str]]:
        """
        Retrieve a per-ticker dataset, requesting only the tickers not already cached.

        The company endpoints return one frame covering every requested ticker, which
        historically meant that adding a single ticker re-requested all of them. The
        frame is therefore split per ticker on the way into the cache and reassembled
        on the way out, so a later call only pays for what it does not already have.

        Args:
            dataset (str): The dataset name to cache under, e.g. "profile".
            tickers (list[str]): The tickers being requested.
            ticker_axis (str): Whether the ticker sits on the column or the index axis
                of the returned frame, see the constants in ticker_model.
            collector (Callable[[list[str]], Any]): Called with the tickers that are not
                cached. May return a frame, or a (frame, invalid_tickers) tuple.
            parameters (dict | None): Parameters that change the returned data, such as
                the period or date range the endpoint was queried for.

        Returns:
            tuple[pd.DataFrame, list[str]]: The combined frame and the tickers the
                source reported as invalid during this call.
        """
        return ticker_model.collect_per_ticker(
            cache=self._cache,
            source=policy_model.FINANCIAL_MODELING_PREP,
            dataset=dataset,
            tickers=tickers,
            ticker_axis=ticker_axis,
            collector=collector,
            parameters=parameters,
        )

    def get_cache_contents(self) -> pd.DataFrame:
        """
        Show what the cache currently holds, grouped by source and dataset.

        The cache stores data per source, per dataset and per entity (a ticker, a
        country, a series identifier), which makes it possible to remove part of it
        rather than all of it. This method is the counterpart to clear_cache: it
        shows what is there so that removing something is an informed decision.

        The cache is inspected even when this Toolkit was created with
        use_cached_data=False, so a cache filled by an earlier session can always
        be reviewed.

        Returns:
            pd.DataFrame: One row per source and dataset combination, with the number
                of entities, the number of stored entries and when they were written.
                An empty DataFrame when the cache holds nothing.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.get_historical_data()

        toolkit.get_cache_contents()
        ```

        Which returns:

        |    | source                | dataset                             |   entities |   entries | oldest_write        | newest_write        |
        |---:|:----------------------|:------------------------------------|-----------:|----------:|:--------------------|:--------------------|
        | 25 | FinancialModelingPrep | earnings_calendar                   |          4 |         4 | 2026-10-24 18:40:06 | 2026-10-24 18:40:06 |
        | 26 | FinancialModelingPrep | earnings_call_transcripts           |          2 |         2 | 2026-10-24 18:41:46 | 2026-10-24 18:41:46 |
        | 27 | FinancialModelingPrep | earnings_call_transcripts_selection |          2 |         2 | 2026-10-24 18:41:46 | 2026-10-24 18:41:46 |
        | 28 | FinancialModelingPrep | employee_count                      |          2 |         2 | 2026-10-24 18:41:06 | 2026-10-24 18:41:06 |
        | 29 | FinancialModelingPrep | esg_scores                          |          3 |         3 | 2026-10-24 18:40:40 | 2026-10-24 18:40:40 |
        """
        contents = cache_controller.get_cache(
            location=self._cache_location, enabled=True
        ).get_contents()

        if not contents:
            return pd.DataFrame()

        return pd.DataFrame(
            [
                {
                    "source": entry["source"],
                    "dataset": entry["dataset"],
                    "entities": len(entry["entities"]),
                    "entries": entry["entries"],
                    "oldest_write": cache_controller.format_timestamp(
                        entry["oldest_write"]
                    ),
                    "newest_write": cache_controller.format_timestamp(
                        entry["newest_write"]
                    ),
                }
                for entry in contents
            ]
        )

    def clear_cache(
        self,
        source: str | None = None,
        dataset: str | None = None,
        ticker: str | None = None,
        confirm: bool = False,
    ) -> int:
        """
        Remove cached data, either all of it or only the part you specify.

        The Finance Toolkit never clears the cache on its own. A cache can represent
        a large amount of downloaded data and a meaningful part of an API quota, so
        discarding it is always an explicit action. Even a change in the cache's own
        internal structure only produces a warning pointing at this method rather
        than removing anything.

        Because the cache is stored per source, per dataset and per entity, removal
        can be narrowed instead of wholesale. Clearing a single stale ticker, or
        everything retrieved from one provider, leaves the rest of the cache intact.

        Args:
            source (str | None): Only remove data from this source, for example
                "FinancialModelingPrep", "YahooFinance", "OECD", "FRED" or
                "GlobalMacroDatabase". These match the names used by enforce_source.
                Defaults to None, which matches every source.
            dataset (str | None): Only remove this dataset within the source, for
                example "historical", "intraday" or "statements". Defaults to None,
                which matches every dataset.
            ticker (str | None): Only remove this entity, for example "AAPL" or a
                country code for macroeconomic data. Defaults to None, which matches
                every entity.
            confirm (bool): Required to be True when no source, dataset or ticker is
                given, since that removes the entire cache. Defaults to False.

        Raises:
            ValueError: If the whole cache would be removed without confirm being set.

        Returns:
            int: The number of stored entries that were removed.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "MSFT"], api_key="FINANCIAL_MODELING_PREP_KEY")

        # Remove only the price history of a single ticker
        toolkit.clear_cache(source="YahooFinance", ticker="AAPL")

        # Remove everything retrieved from the OECD
        toolkit.clear_cache(source="OECD")

        # Remove the entire cache
        toolkit.clear_cache(confirm=True)
        ```
        """
        is_full_clear = source is None and dataset is None and ticker is None

        if is_full_clear and not confirm:
            raise ValueError(
                "This would remove the entire cache. Narrow it down with the source, "
                "dataset or ticker parameter, or pass confirm=True to remove "
                "everything. Use get_cache_contents() to see what is currently stored."
            )

        cache = cache_controller.get_cache(location=self._cache_location, enabled=True)
        removed = cache.remove(source=source, dataset=dataset, entity=ticker)

        logger.info(
            "Removed %d cached entries from %s.",
            removed,
            cache.location,
        )

        return removed

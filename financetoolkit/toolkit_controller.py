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
    get_revenue_segmentation as _get_revenue_segmentation,
    get_shares_float as _get_shares_float,
    get_stock_grades as _get_stock_grades,
    get_stock_splits as _get_stock_splits,
)
from financetoolkit.fundamentals_model import collect_financial_statements
from financetoolkit.historical_model import (
    convert_daily_to_other_period as _convert_daily_to_other_period,
    get_historical_data as _get_historical_data,
    get_historical_statistics as _get_historical_statistics,
)
from financetoolkit.normalization_model import (
    copy_normalization_files as _copy_normalization_files,
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
    from financetoolkit.fixedincome.fixedincome_controller import FixedIncome
    from financetoolkit.models.models_controller import Models
    from financetoolkit.options.options_controller import Options
    from financetoolkit.performance.performance_controller import Performance
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
        use_cached_data: bool | str | None = None,
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
            https://www.jeroenbouma.com/projects/financetoolkit/external-datasets for how to supply your own data.
            Defaults to an empty DataFrame.
            balance (pd.DataFrame): Custom balance sheet data. See notebook link above. Defaults to an empty DataFrame.
            income (pd.DataFrame): Custom income statement data. See notebook link above. Defaults to an empty DataFrame.
            cash (pd.DataFrame): Custom cash flow statement data. See notebook link above. Defaults to an empty DataFrame.
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
        self._fred_api_key = fred_api_key
        self._risk_free_rate = risk_free_rate
        self._rounding = rounding
        self._remove_invalid_tickers = remove_invalid_tickers
        self._invalid_tickers: list = []

        (
            self._use_cached_data,
            self._cache_location,
        ) = cache_controller.parse_use_cached_data(use_cached_data)
        self._cache = cache_controller.get_cache(
            location=self._cache_location, enabled=self._use_cached_data
        )

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
            self._earnings_call_transcripts_latest: bool | None = None

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
        | Interest Coverage Ratio                     | 45.4567 | 44.538  | 31.9908 | inf      | inf      |
        | Income Before Tax Profit Margin             |  0.2985 |  0.302  |  0.2967 |   0.3158 |   0.3189 |
        | Effective Tax Rate                          |  0.133  |  0.162  |  0.1472 |   0.2409 |   0.1561 |
        | Return on Assets                            |  0.2806 |  0.2836 |  0.275  |   0.2613 |   0.3093 |
        | Cash Return on Assets                       |  0.3083 |  0.3471 |  0.3134 |   0.3296 |   0.3079 |
        | Return on Equity                            |  1.4744 |  1.7546 |  1.7195 |   1.5741 |   1.7142 |
        | Return on Invested Capital                  |  0.4143 |  0.4439 |  0.444  |   0.4336 |   0.5335 |
        | Return on Capital Employed                  |  0.496  |  0.6139 |  0.5677 |   0.6548 |   0.6855 |
        | Return on Tangible Assets                   |  1.4744 |  1.7546 |  1.7195 |   1.5741 |   1.7142 |
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

        empty_data: list = []

        if (
            not self._api_key
            and (
                self._balance_sheet_statement.empty
                or self._income_statement.empty
                or self._cash_flow_statement.empty
            )
            and self._enforce_source == "FinancialModelingPrep"
        ):
            raise ValueError(
                "The ratios class requires an API key from FinancialModelPrep if you wish to enforce the usage. "
                "of Financial Modeling Prep. Get an API key here: https://www.jeroenbouma.com/fmp"
            )

        if self._balance_sheet_statement.empty:
            empty_data.append("Balance Sheet Statement")
        if self._income_statement.empty:
            empty_data.append("Income Statement")
        if self._cash_flow_statement.empty:
            empty_data.append("Cash Flow Statement")

        if empty_data:
            logger.info("Obtaining financial statements")
            for statement in empty_data:
                if statement == "Balance Sheet Statement":
                    self.get_balance_sheet_statement()
                if statement == "Income Statement":
                    self.get_income_statement()
                if statement == "Cash Flow Statement":
                    self.get_cash_flow_statement()

        if (
            self._balance_sheet_statement.empty
            and self._income_statement.empty
            and self._cash_flow_statement.empty
        ):
            raise ValueError(
                "The datasets could not be populated and therefore the Ratios class cannot be initialized. "
                "This is usually because no tickers are equities, you have reached the API limit or "
                "entered an invalid API key."
            )

        if not self._start_date:
            self._start_date = (
                f"{self._balance_sheet_statement.columns[0].year - 5}-01-01"
            )
        if not self._end_date:
            self._end_date = (
                f"{self._balance_sheet_statement.columns[-1].year + 5}-01-01"
            )

        if self._quarterly:
            self.get_historical_data(period="quarterly")
        else:
            self.get_historical_data(period="yearly")

        historical = {
            "period": (
                self._quarterly_historical_data
                if self._quarterly
                else self._yearly_historical_data
            ),
            "daily": self._daily_historical_data,
        }

        tickers = (
            self._balance_sheet_statement.index.get_level_values(0).unique().tolist()
        )

        ratios = Ratios(
            tickers=(
                tickers + ["Portfolio"] if "Portfolio" in self._tickers else tickers
            ),
            historical=historical,
            balance=self._balance_sheet_statement,
            income=self._income_statement,
            cash=self._cash_flow_statement,
            quarterly=self._quarterly,
            rounding=self._rounding,
            start_date=self._start_date,
            end_date=self._end_date,
            api_key=self._api_key,
            sleep_timer=self._sleep_timer,
            user_subscription=self._fmp_plan,
            analyst_estimates_cache=self._analyst_estimates_cache,
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

        |                         |   2022Q4 |   2023Q1 |   2023Q2 |
        |:------------------------|---------:|---------:|---------:|
        | Interest Burden Ratio   |  -0.3467 |   0.863  |   0.9835 |
        | Tax Burden Ratio        |  -0.2929 |   0.7699 |   0.8936 |
        | Operating Profit Margin |   0.0183 |   0.0375 |   0.0572 |
        | Asset Turnover          |   0.3349 |   0.2748 |   0.2853 |
        | Equity Multiplier       |   3.1426 |   3.0843 |   2.9152 |
        | Return on Equity        |   0.002  |   0.0211 |   0.0418 |
        """
        # Imported when first used, so the Finance Toolkit loads only what is needed.
        from financetoolkit.models.models_controller import Models  # noqa: PLC0415

        empty_data: list = []

        if not self._api_key and (
            self._balance_sheet_statement.empty
            or self._income_statement.empty
            or self._cash_flow_statement.empty
        ):
            raise ValueError(
                "The models class requires an API key from FinancialModelPrep. "
                "Get an API key here: https://www.jeroenbouma.com/fmp"
            )

        if self._balance_sheet_statement.empty:
            empty_data.append("Balance Sheet Statement")
        if self._income_statement.empty:
            empty_data.append("Income Statement")
        if self._cash_flow_statement.empty:
            empty_data.append("Cash Flow Statement")

        if empty_data:
            logger.info("Obtaining financial statements")
            for statement in empty_data:
                if statement == "Balance Sheet Statement":
                    self.get_balance_sheet_statement()
                if statement == "Income Statement":
                    self.get_income_statement()
                if statement == "Cash Flow Statement":
                    self.get_cash_flow_statement()

        if (
            self._balance_sheet_statement.empty
            and self._income_statement.empty
            and self._cash_flow_statement.empty
        ):
            raise ValueError(
                "The datasets could not be populated and therefore the Ratios class cannot be initialized. "
                "This is usually because no tickers are equities, you have reached the API limit or "
                "entered an invalid API key."
            )

        if not self._start_date:
            self._start_date = (
                f"{self._balance_sheet_statement.columns[0].year - 5}-01-01"
            )
        if not self._end_date:
            self._end_date = (
                f"{self._balance_sheet_statement.columns[-1].year + 5}-01-01"
            )

        for period in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            self.get_historical_data(period=period)

        historical_data = {
            "daily": self._daily_historical_data,
            "weekly": self._weekly_historical_data,
            "monthly": self._monthly_historical_data,
            "quarterly": self._quarterly_historical_data,
            "yearly": self._yearly_historical_data,
        }

        risk_free_rate_data = {
            "daily": self._daily_risk_free_rate,
            "weekly": self._weekly_risk_free_rate,
            "monthly": self._monthly_risk_free_rate,
            "quarterly": self._quarterly_risk_free_rate,
            "yearly": self._yearly_risk_free_rate,
        }

        tickers = (
            self._balance_sheet_statement.index.get_level_values(0).unique().tolist()
        )

        return Models(
            tickers=tickers,
            historical_data=historical_data,
            risk_free_rate_data=risk_free_rate_data,
            balance=self._balance_sheet_statement,
            income=self._income_statement,
            cash=self._cash_flow_statement,
            quarterly=self._quarterly,
            rounding=self._rounding,
            start_date=self._start_date,
            end_date=self._end_date,
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
        |            180 |  1      |      -0.9999 | 0      | -0.0193 | 0.4931 |   -0.6533 |   4.0782 |  0      |       0      | -0      |   0      |  0      | -0      | -0      | 0      | -0      |  0      | -0      |   0      |
        |            185 |  1      |      -0.9999 | 0      | -0.0198 | 0.5068 |   -0.6533 |   4.4595 |  0      |       0      | -0      |   0      |  0      | -0      | -0      | 0      | -0      |  0      | -0      |   0      |
        |            190 |  1      |      -0.9999 | 0      | -0.0204 | 0.5205 |   -0.6533 |   4.9195 |  0      |       0      | -0      |   0      |  0      | -0      | -0      | 0      | -0      |  0      | -0      |   0      |
        |            195 |  1      |      -0.9999 | 0      | -0.0209 | 0.5342 |   -0.6533 |   5.4853 |  0      |       0      | -0      |   0      |  0      | -0      | -0      | 0      | -0      |  0      | -0      |   0.0002 |
        |            200 |  1      |      -0.9999 | 0      | -0.0214 | 0.5479 |   -0.6533 |   6.1981 |  0      |       0      | -0      |   0.0003 |  0.0002 | -0      | -0      | 0      | -0      |  0      | -0.0002 |   0.0065 |
        |            205 |  1      |      -0.9999 | 0      | -0.022  | 0.5616 |   -0.6533 |   7.1239 |  0      |       0      | -0.0001 |   0.0097 |  0.0048 | -0.0001 | -0      | 0      | -0      |  0      | -0.0053 |   0.1335 |
        |            210 |  0.9999 |      -0.9998 | 0      | -0.0235 | 0.5752 |   -0.6532 |   8.3742 |  0      |       0      | -0.0015 |   0.1722 |  0.0714 | -0.001  | -0.0002 | 0      | -0      |  0.0007 | -0.0778 |   1.3082 |
        |            215 |  0.9991 |      -0.9989 | 0.0004 | -0.0346 | 0.5884 |   -0.6527 |  10.1489 |  0.0004 |       0.0005 | -0.0143 |   1.6563 |  0.5604 | -0.0095 | -0.002  | 0.0005 | -0.0001 |  0.0051 | -0.5876 |   5.9321 |
        |            220 |  0.9927 |      -0.9919 | 0.0025 | -0.1034 | 0.5979 |   -0.6485 |  12.8004 |  0.0025 |       0.003  | -0.0766 |   8.8524 |  2.3355 | -0.0507 | -0.0087 | 0.003  | -0.0008 |  0.0196 | -2.264  |  10.617  |
        |            225 |  0.9614 |      -0.9584 | 0.0105 | -0.355  | 0.5908 |   -0.628  |  16.8575 |  0.0106 |       0.0119 | -0.2287 |  26.4045 |  5.0433 | -0.1523 | -0.0212 | 0.0119 | -0.0024 |  0.0343 | -3.9573 |   0.4951 |
        |            230 |  0.8655 |      -0.8581 | 0.027  | -0.8792 | 0.5407 |   -0.5654 |  22.8791 |  0.0273 |       0.0294 | -0.3657 |  42.119  |  5.0451 | -0.2463 | -0.0294 | 0.0294 | -0.0039 |  0.008  | -0.8883 | -14.4266 |
        |            235 |  0.6767 |      -0.6646 | 0.0448 | -1.4399 | 0.4279 |   -0.442  |  31.1644 |  0.0453 |       0.0467 | -0.2405 |  27.4427 |  1.3756 | -0.1694 | -0.0267 | 0.0467 | -0.0028 | -0.0575 |  6.6838 |  -6.0896 |
        |            240 |  0.4305 |      -0.4174 | 0.049  | -1.5675 | 0.2745 |   -0.2812 |  41.601  |  0.0496 |       0.0489 |  0.1289 | -15.4004 |  0.2817 |  0.0708 | -0.0254 | 0.0489 |  0.0009 | -0.0752 |  8.707  |  -1.3284 |
        |            245 |  0.2132 |      -0.2036 | 0.0363 | -1.1574 | 0.1367 |   -0.1393 |  53.7837 |  0.0367 |       0.0348 |  0.3795 | -44.314  |  3.7676 |  0.238  | -0.0302 | 0.0348 |  0.0035 | -0.0197 |  2.2468 | -13.8988 |
        |            250 |  0.0803 |      -0.0754 | 0.0186 | -0.5925 | 0.0516 |   -0.0524 |  67.2285 |  0.0188 |       0.0171 |  0.3372 | -39.2457 |  5.9056 |  0.2152 | -0.0281 | 0.0171 |  0.0033 |  0.0301 | -3.518  |  -9.1559 |
        |            255 |  0.0228 |      -0.0211 | 0.0067 | -0.2149 | 0.0147 |   -0.0149 |  81.5108 |  0.0068 |       0.006  |  0.1731 | -20.1217 |  4.3191 |  0.1112 | -0.0171 | 0.006  |  0.0017 |  0.0329 | -3.8307 |   7.2307 |
        |            260 |  0.0049 |      -0.0044 | 0.0018 | -0.0563 | 0.0032 |   -0.0032 |  96.308  |  0.0018 |       0.0015 |  0.0584 |  -6.7872 |  1.8839 |  0.0377 | -0.0069 | 0.0015 |  0.0006 |  0.0162 | -1.8861 |  11.1559 |
        |            265 |  0.0008 |      -0.0007 | 0.0003 | -0.0109 | 0.0005 |   -0.0005 | 111.391  |  0.0003 |       0.0003 |  0.0137 |  -1.5964 |  0.5417 |  0.0089 | -0.0019 | 0.0003 |  0.0001 |  0.0049 | -0.5728 |   6.0301 |
        |            270 |  0.0001 |      -0.0001 | 0      | -0.0016 | 0.0001 |   -0.0001 | 126.604  |  0      |       0      |  0.0023 |  -0.2715 |  0.1086 |  0.0015 | -0.0004 | 0      |  0      |  0.001  | -0.1183 |   1.8733 |
        |            275 |  0      |      -0      | 0      | -0.0002 | 0      |   -0      | 141.839  |  0      |       0      |  0.0003 |  -0.0343 |  0.0158 |  0.0002 | -0.0001 | 0      |  0      |  0.0002 | -0.0175 |   0.3819 |
        |            280 |  0      |      -0      | 0      | -0      | 0      |   -0      | 157.025  |  0      |       0      |  0      |  -0.0033 |  0.0017 |  0      | -0      | 0      |  0      |  0      | -0.0019 |   0.0546 |
        |            285 |  0      |      -0      | 0      | -0      | 0      |   -0      | 172.114  |  0      |       0      |  0      |  -0.0002 |  0.0001 |  0      | -0      | 0      |  0      |  0      | -0.0002 |   0.0057 |
        |            290 |  0      |      -0      | 0      | -0      | 0      |   -0      | 187.072  |  0      |       0      |  0      |  -0      |  0      |  0      | -0      | 0      |  0      |  0      | -0      |   0.0004 |
        |            295 |  0      |      -0      | 0      | -0      | 0      |   -0      | 201.877  |  0      |       0      |  0      |  -0      |  0      |  0      | -0      | 0      |  0      |  0      | -0      |   0      |
        """
        # Imported when first used, so the Finance Toolkit loads only what is needed.
        from financetoolkit.options.options_controller import Options  # noqa: PLC0415

        if not self._start_date:
            self._start_date = (datetime.today() - timedelta(days=365 * 10)).strftime(
                "%Y-%m-%d"
            )
        if not self._end_date:
            self._end_date = datetime.today().strftime("%Y-%m-%d")

        self.get_historical_data(period="daily")
        self.get_historical_data(period="yearly")

        return Options(
            tickers=self._tickers,
            daily_historical=self._daily_historical_data,
            annual_historical=self._yearly_historical_data,
            risk_free_rate=self._daily_risk_free_rate,
            quarterly=self._quarterly,
            rounding=self._rounding,
            start_date=self._start_date,
            end_date=self._end_date,
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
        | 2025-12-24 | 21.9724 | 24.0777 |     13.6364 |
        | 2025-12-26 | 20.8908 | 24.3229 |     14.1712 |
        | 2025-12-29 | 20.0343 | 23.2159 |     13.912  |
        | 2025-12-30 | 19.2603 | 21.7824 |     13.6713 |
        | 2025-12-31 | 18.7103 | 20.3475 |     12.9966 |

        """
        # Imported when first used, so the Finance Toolkit loads only what is needed.
        from financetoolkit.technicals.technicals_controller import (  # noqa: PLC0415
            Technicals,
        )

        if not self._start_date:
            self._start_date = (datetime.today() - timedelta(days=365 * 10)).strftime(
                "%Y-%m-%d"
            )
        if not self._end_date:
            self._end_date = datetime.today().strftime("%Y-%m-%d")

        for period in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            self.get_historical_data(period=period)

        if self._intraday_period:
            if self._intraday_period in ["1min", "5min", "15min", "30min", "1hour"]:
                self.get_intraday_data(period=self._intraday_period)
            else:
                raise ValueError(
                    "The intraday period must be one of '1min', '5min', '15min', '30min' or '1hour'."
                )

        tickers = (
            self._daily_historical_data.columns.get_level_values(1).unique().tolist()
        )

        historical_data = {
            "intraday": self._intraday_historical_data,
            "daily": self._daily_historical_data,
            "weekly": self._weekly_historical_data,
            "monthly": self._monthly_historical_data,
            "quarterly": self._quarterly_historical_data,
            "yearly": self._yearly_historical_data,
        }

        technicals = Technicals(
            tickers=(
                tickers + ["Portfolio"] if "Portfolio" in self._tickers else tickers
            ),
            historical_data=historical_data,
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
        | 2021Q2 |  0.1157 |  0.1366 |
        | 2021Q3 |  0.0064 |  0.0053 |
        | 2021Q4 |  0.1214 |  0.1869 |
        | 2022Q1 | -0.0577 | -0.1017 |
        | 2022Q2 | -0.2135 | -0.3321 |
        | 2022Q3 | -0.0597 | -0.0828 |
        | 2022Q4 |  0.1059 |  0.0998 |
        | 2023Q1 |  0.0831 |  0.152  |
        | 2023Q2 |  0.1032 |  0.1756 |
        | 2023Q3 | -0.0416 | -0.1029 |
        | 2023Q4 |  0.102  |  0.2474 |
        | 2024Q1 |  0.1038 |  0.1406 |
        | 2024Q2 |  0.0525 |  0.0589 |
        | 2024Q3 |  0.0571 |  0.1525 |
        | 2024Q4 |  0.0217 |  0.0516 |
        | 2025Q1 | -0.0386 | -0.1499 |
        | 2025Q2 |  0.1473 |  0.2038 |
        | 2025Q3 |  0.1031 |  0.1768 |
        | 2025Q4 |  0.0236 |  0.0495 |
        """
        # Imported when first used, so the Finance Toolkit loads only what is needed.
        from financetoolkit.performance.performance_controller import (  # noqa: PLC0415
            Performance,
        )

        if not self._start_date:
            self._start_date = (datetime.today() - timedelta(days=365 * 10)).strftime(
                "%Y-%m-%d"
            )
        if not self._end_date:
            self._end_date = datetime.today().strftime("%Y-%m-%d")

        for period in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            self.get_historical_data(period=period)

        if self._intraday_period:
            if self._intraday_period in ["1min", "5min", "15min", "30min", "1hour"]:
                self.get_intraday_data(period=self._intraday_period)
            else:
                raise ValueError(
                    "The intraday period must be one of '1min', '5min', '15min', '30min' or '1hour'."
                )

        historical_data = {
            "intraday": self._intraday_historical_data,
            "daily": self._daily_historical_data,
            "weekly": self._weekly_historical_data,
            "monthly": self._monthly_historical_data,
            "quarterly": self._quarterly_historical_data,
            "yearly": self._yearly_historical_data,
        }

        risk_free_rate_data = {
            "daily": self._daily_risk_free_rate["Adj Close"],
            "weekly": self._weekly_risk_free_rate["Adj Close"],
            "monthly": self._monthly_risk_free_rate["Adj Close"],
            "quarterly": self._quarterly_risk_free_rate["Adj Close"],
            "yearly": self._yearly_risk_free_rate["Adj Close"],
        }

        tickers = (
            self._daily_historical_data.columns.get_level_values(1).unique().tolist()
        )

        if "Benchmark" in tickers:
            tickers.remove("Benchmark")

        performance = Performance(
            tickers=(
                tickers + ["Portfolio"] if "Portfolio" in self._tickers else tickers
            ),
            historical_data=historical_data,
            risk_free_rate_data=risk_free_rate_data,
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

        |           |       0 |
        |:----------|--------:|
        | AAPL      | -0.2109 |
        | TSLA      | -0.5357 |
        | Benchmark | -0.1279 |
        """
        # Imported when first used, so the Finance Toolkit loads only what is needed.
        from financetoolkit.risk.risk_controller import Risk  # noqa: PLC0415

        if not self._start_date:
            self._start_date = (datetime.today() - timedelta(days=365 * 10)).strftime(
                "%Y-%m-%d"
            )
        if not self._end_date:
            self._end_date = datetime.today().strftime("%Y-%m-%d")

        for period in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            self.get_historical_data(period=period)

        if self._intraday_period:
            if self._intraday_period in ["1min", "5min", "15min", "30min", "1hour"]:
                self.get_intraday_data(period=self._intraday_period)
            else:
                raise ValueError(
                    "The intraday period must be one of '1min', '5min', '15min', '30min' or '1hour'."
                )

        tickers = (
            self._daily_historical_data.columns.get_level_values(1).unique().tolist()
        )

        historical_data = {
            "intraday": self._intraday_historical_data,
            "daily": self._daily_historical_data,
            "weekly": self._weekly_historical_data,
            "monthly": self._monthly_historical_data,
            "quarterly": self._quarterly_historical_data,
            "yearly": self._yearly_historical_data,
        }

        risk_free_rate_data = {
            "daily": self._daily_risk_free_rate["Adj Close"],
            "weekly": self._weekly_risk_free_rate["Adj Close"],
            "monthly": self._monthly_risk_free_rate["Adj Close"],
            "quarterly": self._quarterly_risk_free_rate["Adj Close"],
            "yearly": self._yearly_risk_free_rate["Adj Close"],
        }

        risk = Risk(
            tickers=(
                tickers + ["Portfolio"] if "Portfolio" in self._tickers else tickers
            ),
            historical_data=historical_data,
            risk_free_rate_data=risk_free_rate_data,
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

        if not self._start_date:
            self._start_date = (datetime.today() - timedelta(days=365 * 10)).strftime(
                "%Y-%m-%d"
            )
        if not self._end_date:
            self._end_date = datetime.today().strftime("%Y-%m-%d")

        for period in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            self.get_historical_data(period=period)

        if self._intraday_period:
            if self._intraday_period in ["1min", "5min", "15min", "30min", "1hour"]:
                self.get_intraday_data(period=self._intraday_period)
            else:
                raise ValueError(
                    "The intraday period must be one of '1min', '5min', '15min', '30min' or '1hour'."
                )

        tickers = (
            self._daily_historical_data.columns.get_level_values(1).unique().tolist()
        )

        historical_data = {
            "intraday": self._intraday_historical_data,
            "daily": self._daily_historical_data,
            "weekly": self._weekly_historical_data,
            "monthly": self._monthly_historical_data,
            "quarterly": self._quarterly_historical_data,
            "yearly": self._yearly_historical_data,
        }

        econometrics = Econometrics(
            tickers=(
                tickers + ["Portfolio"] if "Portfolio" in self._tickers else tickers
            ),
            historical_data=historical_data,
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
            ["MSFT", "AAPL"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.get_profile()
        ```

        Which returns:

        |                       | MSFT                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | AAPL                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
        |:----------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
        | Symbol                | MSFT                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | AAPL                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
        | Price                 | 529.76                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | 336.67                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
        | Market Capitalization | 3933759368000                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | 4944792144520                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
        | Beta                  | 1.099                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | 1.069                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
        | Last Dividend         | 3.64                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | 1.06                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
        | Range                 | 349.2-553.72                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | 243.42-345.34                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
        | Change                | 0.46                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | 3.04                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
        | Change %              | 0.08690724                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | 0.91119                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
        | Volume                | 16040951                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | 33380854                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
        | Average Volume        | 33816200                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | 51183728                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
        | Company Name          | Microsoft Corporation                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | Apple Inc.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
        | Currency              | USD                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | USD                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
        | CIK                   | 789019                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | 320193                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
        | ISIN                  | US5949181045                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | US0378331005                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
        | CUSIP                 | 594918104                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | 37833100                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
        | Exchange Full Name    | NASDAQ Global Select                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | NASDAQ Global Select                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
        | Exchange              | NASDAQ                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | NASDAQ                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
        | Industry              | Software - Infrastructure                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | Consumer Electronics                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
        | Website               | https://www.microsoft.com                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | https://www.apple.com                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
        | Description           | Microsoft Corporation is a prominent global technology firm that invents, markets, and provides ongoing assistance for a diverse range of software, digital services, computing devices, and comprehensive solutions. Its operations are organized into three primary divisions: Productivity and Business Processes, Intelligent Cloud, and More Personal Computing. The Productivity and Business Processes segment delivers crucial tools for both enterprises and individual users. This includes the extensive Office suite (comprising Exchange, SharePoint, Microsoft Teams, Office 365 Security and Compliance, Microsoft Viva, and Skype for Business), along with popular consumer offerings like Skype, Outlook.com, OneDrive, and LinkedIn. It also features Dynamics 365, a suite of integrated cloud and on-premises business applications tailored for organizations. The Intelligent Cloud division focuses on sophisticated infrastructure and platform services. Here, Microsoft licenses key products such as SQL Server, Windows Servers, Visual Studio, System Center, and associated Client Access Licenses. It also includes GitHub, a leading platform for developer collaboration and code hosting; Nuance, offering advanced AI solutions for healthcare and businesses; and Azure, its expansive cloud computing platform. This segment further encompasses enterprise support, Microsoft consulting services, and Nuance professional services, assisting clients with the development, deployment, and management of Microsoft's server and desktop technologies, alongside offering product training and certification. Finally, the More Personal Computing segment covers a broad spectrum of consumer and commercial computing experiences. It generates revenue through Windows operating system licensing, including agreements with original equipment manufacturers (OEMs), non-volume licensing, and various Windows Commercial offerings (such as volume licensing and cloud services), as well as patent licensing and Windows Internet of Things (IoT). This division also supplies its own hardware, including Surface devices, PC accessories, and gaming/entertainment consoles. Its Gaming portfolio features Xbox hardware, content, and subscription services, in addition to video games and royalties from third-party titles. Furthermore, it manages search services like Bing and Microsoft's advertising platforms. Microsoft distributes its extensive product line via numerous channels, including original equipment manufacturers, wholesale distributors, and various resellers, complementing direct sales through digital marketplaces, its own online storefronts, and physical retail outlets. The company, established in 1975, maintains its headquarters in Redmond, Washington. | Apple Inc. is a global technology corporation that specializes in the conceptualization, production, and sale of a diverse suite of electronic devices. Its comprehensive hardware lineup features the well-known iPhone smartphones, Mac personal computers, and versatile iPad tablets. The company also supplies a range of wearables, smart home products, and accessories, including AirPods, Apple TV, Apple Watch, items from the Beats brand, and HomePod speakers. Beyond its device offerings, Apple delivers essential support services like AppleCare and robust cloud solutions. It oversees key digital platforms, prominently the App Store, which acts as a central hub for customers to discover and download countless applications and digital content, from e-books and music to videos, games, and podcasts. The company also generates revenue via advertising, leveraging both its proprietary ad platforms and third-party licensing deals. Apple's ecosystem is further bolstered by a wide array of subscription-based services: Apple Arcade for gaming, Apple Fitness+ for personalized wellness, Apple Music for curated audio experiences and on-demand radio, Apple News+ for access to news and magazines, and Apple TV+ for exclusive original video programming. Its financial services portfolio includes the co-branded Apple Card and the mobile payment system, Apple Pay. Additionally, Apple strategically licenses its intellectual property. The company serves a broad clientele that spans individual consumers, small and medium-sized enterprises, as well as institutional clients in the education, corporate, and governmental sectors. Products are distributed through a multi-channel strategy, utilizing Apple's own physical retail locations and online storefronts, a dedicated direct sales team, and collaborations with external partners such as mobile network providers, wholesalers, general retailers, and authorized resellers. The App Store additionally functions as the primary conduit for third-party applications designed for its devices. Founded in 1976, Apple Inc. is headquartered in Cupertino, California. |
        | CEO                   | Satya Nadella                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | John Ternus                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
        | Sector                | Technology                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | Technology                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
        | Country               | US                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | US                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
        | Full Time Employees   | 223000                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | 166000                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
        | Phone                 | 425 882 8080                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | (408) 996-1010                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
        | Address               | One Microsoft Way                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | One Apple Park Way                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
        | City                  | Redmond                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | Cupertino                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
        | State                 | WA                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | CA                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
        | ZIP Code              | 98052-6399                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | 95014                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
        | IPO Date              | 1986-03-13                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | 1980-12-12                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
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
        Get the quote of the specified tickers. These include important metrics
        such as the price, changes, day low, day high, year low, year high, market
        capitalization, volume, average volume, open, previous close, earnings per
        share (EPS), price to earnings ratio (PE), earnings announcement, shares
        outstanding and timestamp that give an overall understanding about the
        company.

        Also known as: real-time price, current stock price, live quote.

        Args:

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["TSLA", "AAPL"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.get_quote()
        ```

        Which returns:

        |                        | TSLA                | AAPL                |
        |:-----------------------|:--------------------|:--------------------|
        | Symbol                 | TSLA                | AAPL                |
        | Name                   | Tesla, Inc.         | Apple Inc.          |
        | Price                  | 377.81              | 336.67              |
        | Change %               | -0.7539100000000001 | 0.91119             |
        | Change                 | -2.87               | 3.04                |
        | Volume                 | 25496215            | 33380854            |
        | Day Low                | 374.43              | 332.79              |
        | Day High               | 382.3499            | 338.67              |
        | Year High              | 498.83              | 345.34              |
        | Year Low               | 297.38              | 243.42              |
        | Market Capitalization  | 1492178500927       | 4944792144520       |
        | Price Average 50 Days  | 350.327             | 322.348             |
        | Price Average 200 Days | 392.1935            | 289.75894           |
        | Exchange               | NASDAQ              | NASDAQ              |
        | Open                   | 378.35              | 337.015             |
        | Previous Close         | 380.68              | 333.63              |
        | Timestamp              | 2026-10-07 20:00:00 | 2026-10-07 20:00:01 |
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
        Get the rating of the specified tickers. These scores and recommendations are categorized
        as follows:

        - An overall rating
        - Discounted Cash Flow (DCF)
        - Return on Equity (ROE)
        - Return on Assets (ROA)
        - Debt to Equity (DE)
        - Price Earnings (PE)
        - Price to Book (PB)

        Also known as: analyst consensus, buy sell hold recommendation.

        Args:

        Raises:
            ValueError: If an API key is not defined for FinancialModelingPrep.

        Returns:
            pd.DataFrame: The stock rating information for the specified tickers.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AMZN", "TSLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        rating = toolkit.get_rating()

        rating.loc['AMZN'].tail()
        ```

        Which returns:

        | date                | Rating   |   Rating Score |   DCF Score |   ROE Score |   ROA Score |   DE Score |   PE Score |   PB Score |
        |:--------------------|:---------|---------------:|------------:|------------:|------------:|-----------:|-----------:|-----------:|
        | 2026-10-01 00:00:00 | B+       |              3 |           2 |           5 |           5 |          2 |          3 |          2 |
        | 2026-10-02 00:00:00 | B+       |              3 |           2 |           5 |           5 |          2 |          3 |          2 |
        | 2026-10-05 00:00:00 | B+       |              3 |           2 |           5 |           5 |          2 |          3 |          2 |
        | 2026-10-06 00:00:00 | B+       |              3 |           2 |           5 |           5 |          2 |          3 |          2 |
        | 2026-10-07 00:00:00 | B+       |              3 |           2 |           5 |           5 |          2 |          3 |          2 |
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

        return self._rating

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
            ["AAPL", "MSFT", "GOOGL", "AMZN"], api_key="FINANCIAL_MODELING_PREP_KEY", start_date="2024-05-01", quarterly=False,
                                                                                                               end_date="2025-12-31",
        )

        analyst_estimates = toolkit.get_analyst_estimates()

        analyst_estimates.loc['AAPL'].iloc[:, :5]
        ```

        Which returns:

        |                               |         2024 |         2025 |         2026 |         2027 |         2028 |
        |:------------------------------|-------------:|-------------:|-------------:|-------------:|-------------:|
        | Estimated Revenue Low         |  3.98249e+11 |  4.08477e+11 |  4.7267e+11  |  5.02764e+11 |  5.27129e+11 |
        | Estimated Revenue High        |  4.02483e+11 |  4.17924e+11 |  4.83194e+11 |  5.57752e+11 |  6.30138e+11 |
        | Estimated Revenue Average     |  4.00366e+11 |  4.15407e+11 |  4.77612e+11 |  5.24595e+11 |  5.64184e+11 |
        | Estimated EBITDA Low          |  1.2031e+11  |  1.41913e+11 |  1.72316e+11 |  1.77525e+11 |  1.85526e+11 |
        | Estimated EBITDA High         |  1.40742e+11 |  1.45705e+11 |  1.73035e+11 |  2.03857e+11 |  2.25144e+11 |
        | Estimated EBITDA Average      |  1.28364e+11 |  1.43809e+11 |  1.72675e+11 |  1.79218e+11 |  1.98406e+11 |
        | Estimated EBIT Low            |  1.08865e+11 |  1.30215e+11 |  1.60268e+11 |  1.64933e+11 |  1.72276e+11 |
        | Estimated EBIT High           |  1.29297e+11 |  1.34007e+11 |  1.60987e+11 |  1.91265e+11 |  2.11894e+11 |
        | Estimated EBIT Average        |  1.16919e+11 |  1.32111e+11 |  1.60628e+11 |  1.66626e+11 |  1.85156e+11 |
        | Estimated Net Income Low      |  9.05919e+10 |  1.08357e+11 |  1.32276e+11 |  1.36158e+11 |  1.42267e+11 |
        | Estimated Net Income High     |  1.07594e+11 |  1.11514e+11 |  1.32874e+11 |  1.5807e+11  |  1.75236e+11 |
        | Estimated Net Income Average  |  9.7294e+10  |  1.09936e+11 |  1.32575e+11 |  1.37566e+11 |  1.52986e+11 |
        | Estimated SGA Expense Low     |  2.56323e+10 |  2.62906e+10 |  3.04222e+10 |  3.23591e+10 |  3.39273e+10 |
        | Estimated SGA Expense High    |  2.59048e+10 |  2.68986e+10 |  3.10995e+10 |  3.58983e+10 |  4.05572e+10 |
        | Estimated SGA Expense Average |  2.57685e+10 |  2.67366e+10 |  3.07403e+10 |  3.37642e+10 |  3.63123e+10 |
        | Estimated EPS Average         |  6.4289      |  7.3818      |  8.8356      |  9.6081      | 10.6865      |
        | Estimated EPS High            |  7.1707      |  7.4319      |  8.8555      | 10.5347      | 11.6787      |
        | Estimated EPS Low             |  6.0376      |  7.2216      |  8.8156      |  9.0743      |  9.4815      |
        | Number of Analysts            | 18           | 26           | 29           | 30           | 22           |
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
            ["AAPL", "MSFT", "GOOGL", "AMZN"], api_key="FINANCIAL_MODELING_PREP_KEY", start_date="2022-08-01", quarterly=False,
                                                                                                               end_date="2025-12-31",
        )

        earning_calendar = toolkit.get_earnings_calendar()

        earning_calendar.loc['AMZN']
        ```

        Which returns:

        | date                |   EPS |   Estimated EPS |     Revenue |   Estimated Revenue | Last Updated   |
        |:--------------------|------:|----------------:|------------:|--------------------:|:---------------|
        | 2022-10-27 00:00:00 |  0.28 |            0.22 | 1.27101e+11 |         1.27308e+11 | 2026-08-17     |
        | 2023-02-02 00:00:00 |  0.25 |            0.18 | 1.49204e+11 |         1.45713e+11 | 2026-08-17     |
        | 2023-04-27 00:00:00 |  0.31 |            0.21 | 1.27358e+11 |         1.24551e+11 | 2025-04-25     |
        | 2023-08-03 00:00:00 |  0.65 |            0.35 | 1.34383e+11 |         1.19573e+11 | 2025-04-25     |
        | 2023-10-26 00:00:00 |  0.94 |            0.58 | 1.43083e+11 |         1.33393e+11 | 2025-04-25     |
        | 2024-02-01 00:00:00 |  1    |            0.8  | 1.69961e+11 |         1.66172e+11 | 2025-04-25     |
        | 2024-04-30 00:00:00 |  0.98 |            0.83 | 1.43313e+11 |         1.42654e+11 | 2025-04-25     |
        | 2024-08-01 00:00:00 |  1.26 |            1.03 | 1.47977e+11 |         1.48665e+11 | 2025-04-25     |
        | 2024-10-31 00:00:00 |  1.43 |            1.14 | 1.58877e+11 |         1.57275e+11 | 2025-04-25     |
        | 2025-02-06 00:00:00 |  1.86 |            1.49 | 1.87792e+11 |         1.87337e+11 | 2025-05-06     |
        | 2025-05-01 00:00:00 |  1.59 |            1.37 | 1.55667e+11 |         1.55148e+11 | 2025-08-01     |
        | 2025-07-31 00:00:00 |  1.68 |            1.31 | 1.67702e+11 |         1.61776e+11 | 2025-10-31     |
        | 2025-10-30 00:00:00 |  1.95 |            1.57 | 1.80169e+11 |         1.77913e+11 | 2026-01-25     |
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
            ["AAPL", "MSFT", "GOOGL", "AMZN"], api_key="FINANCIAL_MODELING_PREP_KEY", start_date="2021-05-01", quarterly=False,
                                                                                                               end_date="2025-12-31",
        )

        geographic_segmentation = toolkit.get_revenue_geographic_segmentation()

        geographic_segmentation.loc['AAPL']
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
            ["AAPL", "MSFT", "GOOGL", "AMZN"], api_key="FINANCIAL_MODELING_PREP_KEY", start_date="2021-05-01", quarterly=False,
                                                                                                               end_date="2025-12-31",
        )

        product_segmentation = toolkit.get_revenue_product_segmentation()

        product_segmentation.loc['MSFT']
        ```

        Which returns:

        |                                                                 |       2021 |       2022 |       2023 |       2024 |       2025 |
        |:----------------------------------------------------------------|-----------:|-----------:|-----------:|-----------:|-----------:|
        | Devices                                                         | 6.791e+09  | 6.991e+09  | 5.521e+09  | 4.706e+09  | 0          |
        | Dynamics                                                        | 0          | 0          | 5.437e+09  | 0          | 0          |
        | Dynamics Products And Cloud Services                            | 0          | 0          | 0          | 6.481e+09  | 7.827e+09  |
        | Enterprise Services                                             | 6.943e+09  | 7.407e+09  | 7.722e+09  | 7.594e+09  | 7.76e+09   |
        | Gaming                                                          | 1.537e+10  | 1.623e+10  | 1.5466e+10 | 2.1503e+10 | 2.3455e+10 |
        | Linked In Corporation                                           | 1.0289e+10 | 1.3816e+10 | 1.5145e+10 | 1.6372e+10 | 1.7812e+10 |
        | Microsoft Three Six Five Commercial Products And Cloud Services | 0          | 0          | 0          | 0          | 8.7767e+10 |
        | Microsoft Three Six Five Consumer Products And Cloud Services   | 0          | 0          | 0          | 0          | 7.404e+09  |
        | Office Products And Cloud Services                              | 3.9872e+10 | 4.4862e+10 | 4.8728e+10 | 5.4875e+10 | 0          |
        | Other Products And Services                                     | 4.479e+09  | 5.291e+09  | 2.11e+08   | 4.5e+07    | 7.2e+07    |
        | Search Advertising                                              | 8.528e+09  | 1.1591e+10 | 1.2208e+10 | 1.2576e+10 | 1.3878e+10 |
        | Server Products And Cloud Services                              | 5.2589e+10 | 6.7321e+10 | 7.997e+10  | 9.7726e+10 | 9.8435e+10 |
        | Windows                                                         | 2.3227e+10 | 2.4761e+10 | 2.1507e+10 | 2.3244e+10 | 1.7314e+10 |

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
            "AAPL",
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.get_historical_data(period="yearly")
        ```

        Which returns:

        | Date   |   ('Open', 'AAPL') |   ('Open', 'Benchmark') |   ('High', 'AAPL') |   ('High', 'Benchmark') |   ('Low', 'AAPL') |   ('Low', 'Benchmark') |
        |:-------|-------------------:|------------------------:|-------------------:|------------------------:|------------------:|-----------------------:|
        | 2021   |             133.52 |                  375.31 |             182.13 |                  479    |            116.21 |                 364.82 |
        | 2022   |             177.83 |                  476.3  |             182.94 |                  479.98 |            125.87 |                 348.11 |
        | 2023   |             130.28 |                  384.37 |             199.62 |                  477.55 |            124.17 |                 377.83 |
        | 2024   |             187.15 |                  472.16 |             260.1  |                  609.07 |            164.08 |                 466.43 |
        | 2025   |             248.93 |                  589.39 |             288.62 |                  691.66 |            169.21 |                 481.8  |
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

            # Every module (risk, performance, models, ...) asks for every period when it
            # is created, so the conversion is kept until the daily data it was made from
            # or one of its settings changes. The daily data is compared by identity, as
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
            "MSFT",
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.get_intraday_data(period="1min")
        ```

        Which returns:

        | date             |   ('Open', 'MSFT') |   ('Open', 'Benchmark') |   ('High', 'MSFT') |   ('High', 'Benchmark') |   ('Low', 'MSFT') |   ('Low', 'Benchmark') |
        |:-----------------|-------------------:|------------------------:|-------------------:|------------------------:|------------------:|-----------------------:|
        | 2025-12-31 15:50 |             483.72 |                  682.77 |             483.83 |                  682.98 |            483.51 |                 682.63 |
        | 2025-12-31 15:51 |             483.59 |                  682.71 |             483.63 |                  682.76 |            483.43 |                 682.52 |
        | 2025-12-31 15:52 |             483.51 |                  682.6  |             483.86 |                  682.73 |            483.5  |                 682.57 |
        | 2025-12-31 15:53 |             483.82 |                  682.59 |             484.19 |                  682.81 |            483.78 |                 682.52 |
        | 2025-12-31 15:54 |             484.06 |                  682.68 |             484.3  |                  682.79 |            483.72 |                 682.49 |
        | 2025-12-31 15:55 |             483.61 |                  682.5  |             484.44 |                  683.02 |            483.32 |                 682.34 |
        | 2025-12-31 15:56 |             484.4  |                  682.97 |             484.4  |                  682.98 |            483.9  |                 682.38 |
        | 2025-12-31 15:57 |             484.11 |                  682.59 |             484.18 |                  682.68 |            483.82 |                 682.36 |
        | 2025-12-31 15:58 |             483.93 |                  682.41 |             483.95 |                  682.43 |            483.72 |                 682.18 |
        | 2025-12-31 15:59 |             483.8  |                  682.19 |             483.85 |                  682.19 |            483.4  |                 681.75 |
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
            pd.DataFrame: The earnings calendar for the specified tickers.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "GOOGL", "AMZN"], api_key="FINANCIAL_MODELING_PREP_KEY", start_date="2022-08-01", quarterly=False,
                                                                                                               end_date="2025-12-31",
        )

        dividend_calendar = toolkit.get_dividend_calendar()

        dividend_calendar.loc['AAPL']
        ```

        Which returns:

        | date       |   Adj Dividend |   Dividend |   Yield | Record Date   | Payment Date   | Declaration Date   |
        |:-----------|---------------:|-----------:|--------:|:--------------|:---------------|:-------------------|
        | 2022-08-05 |           0.23 |       0.23 |  0.5443 | 2022-08-08    | 2022-08-11     | 2022-07-28         |
        | 2022-11-04 |           0.23 |       0.23 |  0.6576 | 2022-11-07    | 2022-11-10     | 2022-10-27         |
        | 2023-02-10 |           0.23 |       0.23 |  0.6092 | 2023-02-13    | 2023-02-16     | 2023-02-02         |
        | 2023-05-12 |           0.24 |       0.24 |  0.5389 | 2023-05-15    | 2023-05-18     | 2023-05-04         |
        | 2023-08-11 |           0.24 |       0.24 |  0.5287 | 2023-08-14    | 2023-08-17     | 2023-08-03         |
        | 2023-11-10 |           0.24 |       0.24 |  0.5097 | 2023-11-13    | 2023-11-16     | 2023-11-02         |
        | 2024-02-09 |           0.24 |       0.24 |  0.5083 | 2024-02-12    | 2024-02-15     | 2024-02-01         |
        | 2024-05-10 |           0.25 |       0.25 |  0.5299 | 2024-05-13    | 2024-05-16     | 2024-05-02         |
        | 2024-08-12 |           0.25 |       0.25 |  0.4505 | 2024-08-12    | 2024-08-15     | 2024-08-01         |
        | 2024-11-08 |           0.25 |       0.25 |  0.4362 | 2024-11-11    | 2024-11-14     | 2024-10-31         |
        | 2025-02-10 |           0.25 |       0.25 |  0.4393 | 2025-02-10    | 2025-02-13     | 2025-01-30         |
        | 2025-05-12 |           0.26 |       0.26 |  0.4791 | 2025-05-12    | 2025-05-15     | 2025-05-01         |
        | 2025-08-11 |           0.26 |       0.26 |  0.449  | 2025-08-11    | 2025-08-14     | 2025-07-31         |
        | 2025-11-10 |           0.26 |       0.26 |  0.3823 | 2025-11-10    | 2025-11-13     | 2025-10-30         |
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
            ["MSFT", "TSLA", "AMZN"], api_key="FINANCIAL_MODELING_PREP_KEY", start_date="2022-08-01", quarterly=False,
                                                                                                      end_date="2025-12-31",
        )

        esg_scores = toolkit.get_esg_scores()

        esg_scores.xs("MSFT", level=1, axis=1)
        ```

        Which returns:

        | date   |   Environmental Score |   Social Score |   Governance Score |   ESG Score |
        |:-------|----------------------:|---------------:|-------------------:|------------:|
        | 2022   |                 72.22 |          58.05 |              61.27 |       63.85 |
        | 2023   |                 72.89 |          58.16 |              60.65 |       63.9  |
        | 2024   |                 71.75 |          58.7  |              60.1  |       63.52 |
        | 2025   |                 72.04 |          57.36 |              60.64 |       63.35 |
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
        Obtains the equity market risk premium by country -- the country default spread plus the
        equity risk premium, following the approach popularized by Aswath Damodaran -- which is
        widely used to calibrate country-specific costs of equity and discount rates in a
        multi-country setting.

        Also known as: country risk premium, Damodaran equity risk premium.

        Args:
            overwrite (bool): Defines whether to overwrite the existing data.

        Raises:
            ValueError: If an API key is not defined for FinancialModelingPrep.

        Returns:
            pd.DataFrame: The market risk premium by country, including the continent, Country
            Risk Premium and Total Equity Risk Premium (both as decimals, 0.0446 for 4.46%).

        Changed in v2.2.2: this used to be returned in percentage points (4.46 for 4.46%).

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

        market_risk_premium.loc[['United States', 'Germany', 'Brazil']]
        ```

        Which returns:

        | Country       | Continent     |   Country Risk Premium |   Total Equity Risk Premium |
        |:--------------|:--------------|-----------------------:|----------------------------:|
        | United States | North America |                 0.0023 |                      0.0446 |
        | Germany       | Europe        |                 0      |                      0.0423 |
        | Brazil        | South America |                 0.0324 |                      0.0747 |
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

        Also known as: COT report, CFTC positioning data, speculator/hedger positioning.

        Args:
            overwrite (bool): Defines whether to overwrite the existing data.

        Raises:
            ValueError: If an API key is not defined for FinancialModelingPrep.

        Returns:
            pd.DataFrame: The Commitment of Traders report for the specified tickers.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["NG", "GC"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        commitment_of_traders = toolkit.get_commitment_of_traders()

        commitment_of_traders.xs("NG", level=1, axis=1)[
            ["Open Interest", "Non-Commercial Long", "Non-Commercial Short", "Commercial Long", "Commercial Short"]
        ].tail()
        ```

        Which returns:

        | date                |   Open Interest |   Non-Commercial Long |   Non-Commercial Short |   Commercial Long |   Commercial Short |
        |:--------------------|----------------:|----------------------:|-----------------------:|------------------:|-------------------:|
        | 2024-01-30 00:00:00 |         1471807 |                279539 |                 382722 |            526952 |             450698 |
        | 2024-02-06 00:00:00 |         1533041 |                301020 |                 415251 |            539246 |             456560 |
        | 2024-02-13 00:00:00 |         1554063 |                334504 |                 471061 |            552780 |             453300 |
        | 2024-02-20 00:00:00 |         1592460 |                356334 |                 510206 |            567791 |             452247 |
        | 2024-02-27 00:00:00 |         1500882 |                326328 |                 467881 |            545380 |             433185 |
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
            pd.DataFrame: The executives per ticker, indexed by their name.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.get_executives().loc["AAPL"].head()
        ```

        Which returns:

        | Name                 | Title                                             |           Pay | Currency   | Gender   |   Year Born |   Title Since | Active   |
        |:---------------------|:--------------------------------------------------|--------------:|:-----------|:---------|------------:|--------------:|:---------|
        | Jennifer G. Newstead | Senior VP of Government Affairs & General Counsel | nan           | USD        | female   |        1970 |           nan | True     |
        | Adrian Perica        | Vice President of Corporate Development           | nan           | USD        | male     |        1974 |           nan | True     |
        | Craig Federighi      | Senior Vice President of Software Engineering     | nan           | USD        | male     |        1969 |           nan | True     |
        | Eduardo H. Cue       | Senior Vice President of Services and Health      |   2.80746e+06 | USD        | male     |        1964 |           nan | True     |
        | Greg Joswiak         | Senior Vice President of Worldwide Marketing      | nan           | USD        | male     |         nan |           nan | True     |
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
            ["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        toolkit.get_executive_compensation().loc["AAPL"].head()
        ```

        Which returns:

        |                                                                              | Filing Date   | Accepted Date       |   Salary |   Bonus |   Stock Award |   Option Award |   Incentive Plan Compensation |   All Other Compensation |    Total | Link                                                                                             |
        |:-----------------------------------------------------------------------------|:--------------|:--------------------|---------:|--------:|--------------:|---------------:|------------------------------:|-------------------------:|---------:|:-------------------------------------------------------------------------------------------------|
        | (2025, 'Deirdre O’Brien Senior Vice President, Retail + People')             | 2026-01-08    | 2026-01-08 16:31:36 |  1000000 |       0 |      22009766 |              0 |                       4000000 |                    37867 | 27047633 | https://www.sec.gov/Archives/edgar/data/320193/000130817926000008/0001308179-26-000008-index.htm |
        | (2025, 'Kate Adams Senior Vice President, General Counsel and Secretary')    | 2026-01-08    | 2026-01-08 16:31:36 |  1000000 |       0 |      22009766 |              0 |                       4000000 |                    22482 | 27032248 | https://www.sec.gov/Archives/edgar/data/320193/000130817926000008/0001308179-26-000008-index.htm |
        | (2025, 'Kevan Parekh Senior Vice President, Chief Financial Officer')        | 2026-01-08    | 2026-01-08 16:31:36 |   891519 |       0 |      18433135 |              0 |                       3120317 |                    22338 | 22467309 | https://www.sec.gov/Archives/edgar/data/320193/000130817926000008/0001308179-26-000008-index.htm |
        | (2025, 'Luca Maestri Former Senior Vice President, Chief Financial Officer') | 2026-01-08    | 2026-01-08 16:31:36 |   819231 |       0 |      13003031 |              0 |                       1638462 |                    22204 | 15482928 | https://www.sec.gov/Archives/edgar/data/320193/000130817926000008/0001308179-26-000008-index.htm |
        | (2025, 'Sabih Khan Senior Vice President, Chief Operating Officer')          | 2026-01-08    | 2026-01-08 16:31:36 |  1000000 |       0 |      22009766 |              0 |                       4000000 |                    21905 | 27031671 | https://www.sec.gov/Archives/edgar/data/320193/000130817926000008/0001308179-26-000008-index.htm |
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
        Obtain the notes each company has listed: the debt securities it issued, with
        their coupon and maturity in the title (e.g. "1.625% Notes due 2026") and the
        exchange they are listed on. This gives a quick view of a company's listed debt
        and when it matures.

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
            ["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.get_company_notes().loc["AAPL"]
        ```

        Which returns:

        | Title                 | Exchange   |    CIK |
        |:----------------------|:-----------|-------:|
        | 0.000% Notes due 2025 | NASDAQ     | 320193 |
        | 1.625% Notes due 2026 | NASDAQ     | 320193 |
        | 2.000% Notes due 2027 | NASDAQ     | 320193 |
        | 1.375% Notes due 2029 | NASDAQ     | 320193 |
        | 3.050% Notes due 2029 | NASDAQ     | 320193 |
        | 0.500% Notes due 2031 | NASDAQ     | 320193 |
        | 3.600% Notes due 2042 | NASDAQ     | 320193 |
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
            ["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2018-01-01",
            end_date="2025-12-31",
        )

        toolkit.get_employee_count()
        ```

        Which returns:

        | Period   |   AAPL |   MSFT |
        |:---------|-------:|-------:|
        | 2018     | 132000 | 131000 |
        | 2019     | 137000 | 144000 |
        | 2020     | 147000 | 163000 |
        | 2021     | 154000 | 181000 |
        | 2022     | 164000 | 221000 |
        | 2023     | 161000 | 221000 |
        | 2024     | 164000 | 228000 |
        | 2025     | 166000 | 228000 |
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
            ["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.get_shares_float()
        ```

        Which returns:

        |                    | AAPL                                                                                | MSFT                                                                                |
        |:-------------------|:------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|
        | Date               | 2026-10-07 22:03:55                                                                 | 2026-10-08 00:12:55                                                                 |
        | Free Float         | 0.9987879921341868                                                                  | 0.9985093936476086                                                                  |
        | Float Shares       | 14576491739                                                                         | 7414481428                                                                          |
        | Outstanding Shares | 14594180000                                                                         | 7425550000                                                                          |
        | Source             | https://www.sec.gov/Archives/edgar/data/320193/000032019326000020/aapl-20260627.htm | https://www.sec.gov/Archives/edgar/data/789019/000119312526323660/msft-20260630.htm |
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
            ["MSFT", "GOOGL"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="1990-01-01",
            end_date="2025-12-31",
        )

        toolkit.get_mergers_acquisitions().loc["MSFT"].head()
        ```

        Which returns:

        | Transaction Date    | Role     | Acquirer Symbol   | Acquirer Name   |   Target Symbol | Target Name   | Accepted Date       | Link                                                                                            |
        |:--------------------|:---------|:------------------|:----------------|----------------:|:--------------|:--------------------|:------------------------------------------------------------------------------------------------|
        | 1995-02-09 00:00:00 | Acquirer | MSFT              | MICROSOFT CORP  |             nan | ChipSoft      | 1995-02-09 00:00:00 | https://www.sec.gov/Archives/edgar/data/789019/0000891020-95-000018.txt                         |
        | 1999-11-02 00:00:00 | Acquirer | MSFT              | MICROSOFT CORP  |             nan | Visio's       | 1999-11-02 00:00:00 | https://www.sec.gov/Archives/edgar/data/789019/000103221099001490/0001032210-99-001490.txt      |
        | 2001-02-01 00:00:00 | Acquirer | MSFT              | MICROSOFT CORP  |             nan | GENTLEMEN     | 2001-02-01 00:00:00 | https://www.sec.gov/Archives/edgar/data/789019/000103221001000126/0001032210-01-000126-0001.txt |
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
            ["AAPL", "NVDA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2000-01-01",
            end_date="2025-12-31",
        )

        toolkit.get_stock_splits()
        ```

        Which returns:

        |                                            |   Numerator |   Denominator | Split Type   |
        |:-------------------------------------------|------------:|--------------:|:-------------|
        | ('AAPL', Timestamp('2000-06-21 00:00:00')) |           2 |             1 | stock-split  |
        | ('AAPL', Timestamp('2005-02-28 00:00:00')) |           2 |             1 | stock-split  |
        | ('AAPL', Timestamp('2014-06-09 00:00:00')) |           7 |             1 | stock-split  |
        | ('AAPL', Timestamp('2020-08-31 00:00:00')) |           4 |             1 | stock-split  |
        | ('NVDA', Timestamp('2000-06-27 00:00:00')) |           2 |             1 | stock-split  |
        | ('NVDA', Timestamp('2001-09-12 00:00:00')) |           2 |             1 | stock-split  |
        | ('NVDA', Timestamp('2006-04-07 00:00:00')) |           2 |             1 | stock-split  |
        | ('NVDA', Timestamp('2007-09-11 00:00:00')) |           3 |             2 | stock-split  |
        | ('NVDA', Timestamp('2021-07-20 00:00:00')) |           4 |             1 | stock-split  |
        | ('NVDA', Timestamp('2024-06-10 00:00:00')) |          10 |             1 | stock-split  |
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
            ["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        toolkit.get_insider_trade_statistics().loc["AAPL"]
        ```

        Which returns:

        | Period   |   Acquired Transactions |   Disposed Transactions |   Acquired/Disposed Ratio |   Total Acquired |   Total Disposed |   Average Acquired |   Average Disposed |   Total Purchases |   Total Sales |
        |:---------|------------------------:|------------------------:|--------------------------:|-----------------:|-----------------:|-------------------:|-------------------:|------------------:|--------------:|
        | 2025Q1   |                      14 |                       8 |                    1.75   |            19255 |  12128           |            1375.36 |             1516   |                 0 |             1 |
        | 2025Q2   |                       6 |                      38 |                    0.1579 |           466004 | 892618           |           77667.3  |            23489.9 |                 0 |            13 |
        | 2025Q3   |                       6 |                       3 |                    2      |           391455 | 125256           |           65242.5  |            41752   |                 0 |             2 |
        | 2025Q4   |                       6 |                      33 |                    0.1818 |           578243 |      1.11303e+06 |           96373.8  |            33728.1 |                 0 |            15 |
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

        toolkit = Toolkit(["AAPL", "MSFT"], api_key="FINANCIAL_MODELING_PREP_KEY", start_date="2026-09-01")

        toolkit.get_stock_grades().loc["AAPL"].tail()
        ```

        Which returns:

        |                                                          | Previous Grade   | New Grade   | Action   |
        |:---------------------------------------------------------|:-----------------|:------------|:---------|
        | (Timestamp('2026-09-18 00:00:00'), 'Evercore ISI Group') | Outperform       | Outperform  | Maintain |
        | (Timestamp('2026-09-23 00:00:00'), 'B of A Securities')  | Buy              | Buy         | Maintain |
        | (Timestamp('2026-09-29 00:00:00'), 'Morgan Stanley')     | Overweight       | Overweight  | Maintain |
        | (Timestamp('2026-10-01 00:00:00'), 'Morgan Stanley')     | Overweight       | Overweight  | Maintain |
        | (Timestamp('2026-10-01 00:00:00'), 'Needham')            | Hold             | Hold        | Maintain |
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

    def get_etf_holdings(
        self, overwrite: bool = False, show_columns: list[str] | None = None
    ):
        """
        Obtain the holdings of each ETF or fund: every asset it holds with the number of
        shares, the market value and the weight in the fund (as a decimal). This shows
        what the fund is actually exposed to and how concentrated it is.

        Tickers that are not an ETF or fund have no holdings and return no data. They are
        not removed from the Toolkit instance.

        Also known as: fund holdings, ETF constituents, fund portfolio.

        Args:
            overwrite (bool): Defines whether to overwrite the existing data.
            show_columns (list[str] | None): A list of column names to keep in the result. Invalid
            names are reported and ignored. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The holdings per ticker, indexed by asset.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["QQQ", "VTI"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.get_etf_holdings().loc["QQQ"].head()
        ```

        Which returns:

        | Asset   | Name                       | ISIN         | CUSIP     |      Shares |    Weight |   Market Value | Updated At          |
        |:--------|:---------------------------|:-------------|:----------|------------:|----------:|---------------:|:--------------------|
        | NVDA    | NVIDIA Corp                | US67066G1040 | 67066G104 | 1.80964e+08 | 0.0850518 |    4.32937e+10 | 2026-10-07 14:14:58 |
        | AAPL    | Apple Inc                  | US0378331005 | 037833100 | 1.09586e+08 | 0.0718253 |    3.65611e+10 | 2026-10-07 14:14:58 |
        | MSFT    | Microsoft Corp             | US5949181045 | 594918104 | 5.57574e+07 | 0.057978  |    2.95124e+10 | 2026-10-07 14:14:58 |
        | MU      | Micron Technology Inc      | US5951121038 | 595112103 | 2.34119e+07 | 0.0480888 |    2.44785e+10 | 2026-10-07 14:14:58 |
        | AMD     | Advanced Micro Devices Inc | US0079031078 | 007903107 | 3.38406e+07 | 0.043174  |    2.19767e+10 | 2026-10-07 14:14:58 |
        """
        if not self._api_key:
            self._missing_api_key_message()
            return None

        if self._etf_holdings.empty or overwrite:
            # Not every ticker is a fund, so a ticker without holdings is not invalid.
            self._etf_holdings, _ = self._collect_per_ticker(
                dataset="etf_holdings",
                tickers=self._tickers,
                ticker_axis=ticker_model.TICKER_ON_INDEX,
                collector=lambda tickers: _get_etf_holdings(
                    tickers=tickers,
                    api_key=self._api_key,
                    user_subscription=self._fmp_plan,
                ),
            )

        if len(self._tickers) == 1 and not self._etf_holdings.empty:
            return filter_columns(
                self._etf_holdings.loc[self._tickers[0]], show_columns
            )

        return filter_columns(self._etf_holdings, show_columns)

    def get_etf_information(self, overwrite: bool = False):
        """
        Obtain the profile of each ETF or fund: its issuer, asset class, domicile,
        inception date, expense ratio (as a decimal), assets under management, net asset
        value and number of holdings. The expense ratio in particular determines how much
        of the return is lost to costs every year.

        Tickers that are not an ETF or fund return no data. They are not removed from the
        Toolkit instance.

        Also known as: fund profile, ETF profile, fund information.

        Args:
            overwrite (bool): Defines whether to overwrite the existing data.

        Returns:
            pd.DataFrame: The information (rows) per ticker (columns).

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["QQQ", "VTI"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.get_etf_information().drop(["Description", "Website"])
        ```

        Which returns:

        |                         | QQQ                         | VTI                                         |
        |:------------------------|:----------------------------|:--------------------------------------------|
        | Name                    | Invesco QQQ Trust, Series 1 | Vanguard Morningstar Total Stock Market ETF |
        | ISIN                    | US46090E1038                | US9229087690                                |
        | CUSIP                   | 46090E103                   | 922908769                                   |
        | Asset Class             | Equity                      | Large Cap Equity                            |
        | Domicile                | US                          | US                                          |
        | ETF Company             | Invesco                     | Vanguard                                    |
        | Inception Date          | 1999-03-10                  | 2001-05-24                                  |
        | Expense Ratio           | 0.0018                      | 0.0003                                      |
        | Assets Under Management | 508484484641                | 2300000000000                               |
        | Average Volume          | 39278732                    | 3259438                                     |
        | NAV                     | 759.5                       | 381.13                                      |
        | NAV Currency            | USD                         | USD                                         |
        | Holdings Count          | 102                         | 3598                                        |
        | Actively Trading        | True                        | True                                        |
        | Updated At              | 2026-10-08T01:48:10.006Z    | 2026-10-08T02:22:30.046Z                    |
        """
        if not self._api_key:
            self._missing_api_key_message()
            return None

        if self._etf_information.empty or overwrite:
            # Not every ticker is a fund, so a ticker without information is not invalid.
            self._etf_information, _ = self._collect_per_ticker(
                dataset="etf_information",
                tickers=self._tickers,
                ticker_axis=ticker_model.TICKER_ON_COLUMNS,
                collector=lambda tickers: _get_etf_information(
                    tickers=tickers,
                    api_key=self._api_key,
                    user_subscription=self._fmp_plan,
                ),
            )

        return self._etf_information

    def get_etf_country_weightings(self, overwrite: bool = False):
        """
        Obtain how each ETF or fund is allocated across countries, as decimals. Two funds
        tracking similar markets can differ considerably here, which matters for the
        currency and political risk the fund carries.

        Tickers that are not an ETF or fund return no data. They are not removed from the
        Toolkit instance.

        Also known as: country allocation, geographic exposure, country exposure.

        Args:
            overwrite (bool): Defines whether to overwrite the existing data.

        Returns:
            pd.DataFrame: The weight per country (rows) and ticker (columns).

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["QQQ", "VTI"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.get_etf_country_weightings().head()
        ```

        Which returns:

        | Country        |    QQQ |    VTI |
        |:---------------|-------:|-------:|
        | United States  | 0.9445 | 0.9733 |
        | United Kingdom | 0.0164 | 0.0048 |
        | Canada         | 0.0096 | 0.0007 |
        | Singapore      | 0.0074 | 0.0025 |
        | Netherlands    | 0.0135 | 0.0003 |
        """
        if not self._api_key:
            self._missing_api_key_message()
            return None

        if self._etf_country_weightings.empty or overwrite:
            # Not every ticker is a fund, so a ticker without weightings is not invalid.
            self._etf_country_weightings, _ = self._collect_per_ticker(
                dataset="etf_country_weightings",
                tickers=self._tickers,
                ticker_axis=ticker_model.TICKER_ON_COLUMNS,
                collector=lambda tickers: _get_etf_country_weightings(
                    tickers=tickers,
                    api_key=self._api_key,
                    user_subscription=self._fmp_plan,
                ),
            )

        return self._etf_country_weightings

    def get_etf_sector_weightings(self, overwrite: bool = False):
        """
        Obtain how each ETF or fund is allocated across sectors, as decimals. This shows
        whether a fund that looks broad is in fact concentrated in a few sectors, such as
        technology in many large-cap indices.

        Tickers that are not an ETF or fund return no data. They are not removed from the
        Toolkit instance.

        Also known as: sector allocation, sector exposure, sector breakdown.

        Args:
            overwrite (bool): Defines whether to overwrite the existing data.

        Returns:
            pd.DataFrame: The weight per sector (rows) and ticker (columns).

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["QQQ", "VTI"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.get_etf_sector_weightings()
        ```

        Which returns:

        | Sector                 |          QQQ |       VTI |
        |:-----------------------|-------------:|----------:|
        | Basic Materials        |   0.00912283 | 0.0222419 |
        | Cash & Others          |   0.00248307 | 0.001929  |
        | Communication Services |   0.118947   | 0.086689  |
        | Consumer Cyclical      |   0.0972185  | 0.0898289 |
        | Consumer Defensive     |   0.0549628  | 0.0425139 |
        | Energy                 |   0.00442936 | 0.037377  |
        | Financial Services     |   0.0018637  | 0.123878  |
        | Healthcare             |   0.0358905  | 0.0998086 |
        | Industrials            |   0.054947   | 0.0881844 |
        | Technology             |   0.610092   | 0.365041  |
        | Utilities              |   0.0100427  | 0.0197311 |
        | Real Estate            | nan          | 0.0227773 |
        """
        if not self._api_key:
            self._missing_api_key_message()
            return None

        if self._etf_sector_weightings.empty or overwrite:
            # Not every ticker is a fund, so a ticker without weightings is not invalid.
            self._etf_sector_weightings, _ = self._collect_per_ticker(
                dataset="etf_sector_weightings",
                tickers=self._tickers,
                ticker_axis=ticker_model.TICKER_ON_COLUMNS,
                collector=lambda tickers: _get_etf_sector_weightings(
                    tickers=tickers,
                    api_key=self._api_key,
                    user_subscription=self._fmp_plan,
                ),
            )

        return self._etf_sector_weightings

    def get_earnings_call_transcripts(
        self, latest: bool = True, overwrite: bool = False
    ):
        """
        Obtain the earnings call transcripts of each company: the full text of the call,
        with management's prepared remarks followed by the questions of analysts and the
        answers. Transcripts explain the numbers in the financial statements in
        management's own words, and with that are a strong input for text analysis and AI
        models, e.g. to track sentiment, guidance or recurring themes over time.

        A transcript is long (often 40,000 to 60,000 characters) and every quarter is a
        separate request. By default only the most recent transcript is retrieved. With
        latest=False every transcript between this Toolkit instance's start_date and
        end_date is retrieved instead. Each transcript is cached on its own, since a
        published transcript does not change, so a quarter is only retrieved once.

        Also known as: earnings call, conference call transcript, earnings transcript.

        Args:
            latest (bool): Whether to only retrieve the most recent transcript. When False,
                every transcript between the start_date and end_date is retrieved. Defaults to True.
            overwrite (bool): Defines whether to overwrite the existing data.

        Returns:
            pd.DataFrame: The date and transcript per ticker, indexed by fiscal period.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        transcripts = toolkit.get_earnings_call_transcripts()

        transcripts["Transcript"].str[:100]
        ```

        Which returns:

        |                    | Transcript                                                                                           |
        |:-------------------|:-----------------------------------------------------------------------------------------------------|
        | ('AAPL', '2026Q3') | Suhasini Chandramouli: Good afternoon, welcome to the Apple Q3 fiscal year 2026 earnings conference  |
        | ('MSFT', '2026Q4') | Operator: Greetings, and welcome to the Microsoft Fiscal Year 2026 Fourth Quarter Earnings Conferenc |
        """
        if not self._api_key:
            self._missing_api_key_message()
            return None

        if (
            self._earnings_call_transcripts.empty
            or overwrite
            or self._earnings_call_transcripts_latest != latest
        ):
            self._earnings_call_transcripts, self._invalid_tickers = (
                self._collect_per_ticker(
                    dataset="earnings_call_transcripts_selection",
                    tickers=self._tickers,
                    ticker_axis=ticker_model.TICKER_ON_INDEX,
                    parameters={
                        "latest": latest,
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
                    ),
                )
            )
            self._earnings_call_transcripts_latest = latest

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

        Also known as: key statistics over time, historical key metrics.

        Returns:
            pd.DataFrame: A DataFrame containing the statistics for each ticker.

        As an example:

        ```python
        from financetoolkit import Toolkit

        companies = Toolkit(
            ["AMZN", "^HSI", "IWDA.AS", "0P0000Z8RO.T"],
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        companies.get_historical_statistics()
        ```

        Which returns:

        |                        | AMZN             | ^HSI           | IWDA.AS          | 0P0000Z8RO.T   |
        |:-----------------------|:-----------------|:---------------|:-----------------|:---------------|
        | Currency               | USD              | HKD            | EUR              | JPY            |
        | Symbol                 | AMZN             | ^HSI           | IWDA.AS          | 0P0000Z8RO.T   |
        | Exchange Name          | NMS              | HKG            | AMS              | JPX            |
        | Instrument Type        | EQUITY           | INDEX          | ETF              | MUTUALFUND     |
        | First Trade Date       | 1997-05-15       | 1986-12-31     | 2009-09-25       | 2018-01-04     |
        | Regular Market Time    | 2026-10-07       | 2026-10-08     | 2026-10-08       | 2026-10-07     |
        | GMT Offset             | -14400           | 28800          | 7200             | 32400          |
        | Timezone               | EDT              | HKT            | CEST             | JST            |
        | Exchange Timezone Name | America/New_York | Asia/Hong_Kong | Europe/Amsterdam | Asia/Tokyo     |
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
        Retrieve daily, weekly, monthly, quarterly or yearly treasury data. This can be from FinancialModelingPrep
        or from YahooFinance. FinancialModelingPrep is by far a more extensive dataset containing daily data from
        1 month to 30 years. YahooFinance only contains daily data for 5, 10 and 30 years but is a free alternative.

        Also known as: US Treasury yields, yield curve, treasury rates.

        Args:
            period (str): The interval at which the treasury data should be returned - daily, weekly, monthly, quarterly, or yearly.
            fill_nan (bool): Defines whether to forward fill NaN values. This defaults
            to True to prevent holes in the dataset. This is especially relevant for
            technical indicators.
            risk_free_rate (str | None, optional): The maturity to return as the risk free rate
                ('13w', '5y', '10y' or '30y'). Defaults to None, which uses the maturity set on
                the Toolkit.
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

        companies = Toolkit(
            ["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2023-08-10",
            end_date="2025-12-31",
        )

        companies.get_treasury_data()
        ```

        Which returns:

        | date       |   ('Open', '13 Week') |   ('Open', '5 Year') |   ('Open', '10 Year') |   ('Open', '30 Year') |   ('High', '13 Week') |   ('High', '5 Year') |
        |:-----------|----------------------:|---------------------:|----------------------:|----------------------:|----------------------:|---------------------:|
        | 2025-12-24 |                0.0355 |               0.0374 |                0.0417 |                0.0483 |                0.0356 |               0.0374 |
        | 2025-12-26 |                0.0355 |               0.0369 |                0.0412 |                0.0479 |                0.0355 |               0.0371 |
        | 2025-12-29 |                0.0354 |               0.0368 |                0.0412 |                0.048  |                0.0354 |               0.0369 |
        | 2025-12-30 |                0.0356 |               0.037  |                0.0414 |                0.0483 |                0.0356 |               0.037  |
        | 2025-12-31 |                0.0354 |               0.0368 |                0.0413 |                0.0481 |                0.0355 |               0.0373 |

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

        if risk_free_rate:
            if risk_free_rate not in ["13w", "5y", "10y", "30y"]:
                raise ValueError(
                    "Please choose from 13w, 5y, 10y or 30y as risk_free_rate."
                )

            risk_free_rate = (
                "^IRX"
                if risk_free_rate == "13w"
                else (
                    "^FVX"
                    if risk_free_rate == "5y"
                    else ("^TNX" if risk_free_rate == "10y" else "^TYX")
                )
            )

        risk_free_rate_tickers = (
            ["^IRX", "^FVX", "^TNX", "^TYX"] if not risk_free_rate else [risk_free_rate]
        )

        risk_free_rate = risk_free_names[self._risk_free_rate]

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
            return self._daily_treasury_data.loc[self._start_date : self._end_date, :]
        if period == "weekly":
            self._weekly_treasury_data = _convert_daily_to_other_period(
                period=period,
                daily_historical_data=self._daily_treasury_data,
                start=self._start_date,
                end=self._end_date,
                rounding=rounding if rounding is not None else self._rounding,
            )

            self._weekly_risk_free_rate = self._weekly_treasury_data.xs(
                risk_free_rate, level=1, axis=1
            )

            return self._weekly_treasury_data.loc[self._start_date : self._end_date, :]
        if period == "monthly":
            self._monthly_treasury_data = _convert_daily_to_other_period(
                period=period,
                daily_historical_data=self._daily_treasury_data,
                start=self._start_date,
                end=self._end_date,
                rounding=rounding if rounding is not None else self._rounding,
            )

            self._monthly_risk_free_rate = self._monthly_treasury_data.xs(
                risk_free_rate, level=1, axis=1
            )

            return self._monthly_treasury_data.loc[self._start_date : self._end_date, :]
        if period == "quarterly":
            self._quarterly_treasury_data = _convert_daily_to_other_period(
                period=period,
                daily_historical_data=self._daily_treasury_data,
                start=self._start_date,
                end=self._end_date,
                rounding=rounding if rounding is not None else self._rounding,
            )

            self._quarterly_risk_free_rate = self._quarterly_treasury_data.xs(
                risk_free_rate, level=1, axis=1
            )

            return self._quarterly_treasury_data.loc[
                self._start_date : self._end_date, :
            ]
        if period == "yearly":
            self._yearly_treasury_data = _convert_daily_to_other_period(
                period=period,
                daily_historical_data=self._daily_treasury_data,
                start=self._start_date,
                end=self._end_date,
                rounding=rounding if rounding is not None else self._rounding,
            )

            self._yearly_risk_free_rate = self._yearly_treasury_data.xs(
                risk_free_rate, level=1, axis=1
            )

            return self._yearly_treasury_data.loc[self._start_date : self._end_date, :]

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
            rounding (int): Defines the number of decimal places to round the data to.
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

        | Date    |   Open |   High |    Low |   Close |   Adj Close |   Volume |   Dividends |   Return |   Cumulative Return |
        |:--------|-------:|-------:|-------:|--------:|------------:|---------:|------------:|---------:|--------------------:|
        | 2025-03 | 1.0414 | 1.0954 | 1.039  |  1.0824 |      1.0824 |        0 |           0 |   0.0413 |              0.8931 |
        | 2025-04 | 1.0819 | 1.1547 | 1.078  |  1.1389 |      1.1389 |        0 |           0 |   0.0522 |              0.9397 |
        | 2025-05 | 1.1325 | 1.1419 | 1.1075 |  1.1378 |      1.1378 |        0 |           0 |  -0.001  |              0.9388 |
        | 2025-06 | 1.1353 | 1.1772 | 1.1356 |  1.1727 |      1.1727 |        0 |           0 |   0.0307 |              0.9676 |
        | 2025-07 | 1.1787 | 1.183  | 1.1407 |  1.1429 |      1.1429 |        0 |           0 |  -0.0254 |              0.943  |
        | 2025-08 | 1.1424 | 1.1731 | 1.1395 |  1.1682 |      1.1682 |        0 |           0 |   0.0221 |              0.9639 |
        | 2025-09 | 1.1692 | 1.1873 | 1.161  |  1.1731 |      1.1731 |        0 |           0 |   0.0042 |              0.9679 |
        | 2025-10 | 1.1736 | 1.1779 | 1.1524 |  1.1572 |      1.1572 |        0 |           0 |  -0.0136 |              0.9548 |
        | 2025-11 | 1.1528 | 1.1654 | 1.147  |  1.16   |      1.16   |        0 |           0 |   0.0024 |              0.9571 |
        | 2025-12 | 1.1602 | 1.1809 | 1.159  |  1.1747 |      1.1747 |        0 |           0 |   0.0127 |              0.9692 |
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
                    rounding=rounding if rounding is not None else self._rounding,
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
            historical_data = self._daily_exchange_rate_data.loc[
                self._start_date : self._end_date, :
            ]
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
            E.g. when selecting 4 with quarterly data, the TTM is calculated.
            show_columns (list[str] | None): A list of column names to keep in the result. Invalid
            names are reported and ignored. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: A pandas DataFrame with the retrieved balance sheet statement data.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["MSFT", "MU"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            quarterly=True,
            start_date='2022-05-01',
            end_date="2025-12-31",
        )

        balance_sheet_statements = toolkit.get_balance_sheet_statement()

        balance_sheet_statements.loc['MU']
        ```

        Which returns:

        |                                           |      2022Q2 |      2022Q3 |      2022Q4 |      2023Q1 |      2023Q2 |
        |:------------------------------------------|------------:|------------:|------------:|------------:|------------:|
        | Cash and Cash Equivalents                 |  9.157e+09  |  8.262e+09  |  9.574e+09  |  9.798e+09  |  9.298e+09  |
        | Short Term Investments                    |  1.07e+09   |  1.069e+09  |  1.007e+09  |  1.02e+09   |  1.054e+09  |
        | Cash and Short Term Investments           |  1.0227e+10 |  9.331e+09  |  1.0581e+10 |  1.0818e+10 |  1.0352e+10 |
        | Accounts Receivable                       |  5.896e+09  |  4.765e+09  |  2.875e+09  |  1.891e+09  |  2.042e+09  |
        | Other Receivables                         |  3.33e+08   |  3.65e+08   |  4.43e+08   |  3.87e+08   |  3.87e+08   |
        | Net Receivables                           |  6.229e+09  |  5.13e+09   |  3.318e+09  |  2.278e+09  |  2.429e+09  |
        | Inventory                                 |  5.629e+09  |  6.663e+09  |  8.359e+09  |  8.129e+09  |  8.238e+09  |
        | Prepaids                                  |  0          |  0          |  0          |  0          |  0          |
        | Other Current Assets                      |  6.23e+08   |  6.57e+08   |  6.63e+08   |  6.73e+08   |  7.15e+08   |
        | Total Current Assets                      |  2.2708e+10 |  2.1781e+10 |  2.2921e+10 |  2.1898e+10 |  2.1734e+10 |
        | Property, Plant and Equipment             |  3.7355e+10 |  3.9227e+10 |  4.0028e+10 |  3.9758e+10 |  3.9382e+10 |
        | Goodwill                                  |  1.228e+09  |  1.228e+09  |  1.228e+09  |  1.228e+09  |  1.252e+09  |
        | Intangible Assets                         |  4.15e+08   |  4.21e+08   |  4.28e+08   |  4.1e+08    |  4.1e+08    |
        | Goodwill and Intangible Assets            |  1.643e+09  |  1.649e+09  |  1.656e+09  |  1.638e+09  |  1.662e+09  |
        | Long Term Investments                     |  1.75e+09   |  1.647e+09  |  1.426e+09  |  1.212e+09  |  9.73e+08   |
        | Tax Assets                                |  6.82e+08   |  7.02e+08   |  6.72e+08   |  6.97e+08   |  7.08e+08   |
        | Other Fixed Assets                        |  1.158e+09  |  1.277e+09  |  1.171e+09  |  1.317e+09  |  1.221e+09  |
        | Fixed Assets                              |  4.2588e+10 |  4.4502e+10 |  4.4953e+10 |  4.4622e+10 |  4.3946e+10 |
        | Other Assets                              |  0          |  0          |  0          |  0          |  0          |
        | Total Assets                              |  6.5296e+10 |  6.6283e+10 |  6.7874e+10 |  6.652e+10  |  6.568e+10  |
        | Accounts Payable                          |  2.019e+09  |  2.142e+09  |  1.789e+09  |  1.689e+09  |  1.64e+09   |
        | Other Payables                            |  3.82e+08   |  2.59e+09   |  2.713e+09  |  1.953e+09  |  1.671e+09  |
        | Total Payables                            |  2.401e+09  |  4.732e+09  |  4.502e+09  |  3.642e+09  |  3.311e+09  |
        | Accrued Expenses                          |  8.75e+08   |  1.358e+09  |  9.36e+08   |  6.68e+08   |  8.66e+08   |
        | Short Term Debt                           |  1.65e+08   |  0          |  6.2e+07    |  1.06e+08   |  1.06e+08   |
        | Current Capital Lease Obligations Current |  0          |  1.03e+08   |  1.09e+08   |  1.31e+08   |  1.53e+08   |
        | Tax Payables                              |  3.82e+08   |  4.2e+08    |  4.19e+08   |  2.41e+08   |  1.48e+08   |
        | Deferred Revenue                          |  5.7e+07    |  0          |  0          |  0          |  0          |
        | Other Current Liabilities                 |  3.511e+09  |  1.346e+09  |  9.16e+08   |  7.08e+08   |  6.68e+08   |
        | Total Current Liabilities                 |  7.009e+09  |  7.539e+09  |  6.525e+09  |  5.255e+09  |  5.104e+09  |
        | Capital Lease Obligations Non Current     |  1.451e+09  |  1.393e+09  |  1.43e+09   |  1.55e+09   |  1.621e+09  |
        | Long Term Debt                            |  6.034e+09  |  6.02e+09   |  9.289e+09  |  1.1097e+10 |  1.1968e+10 |
        | Deferred Revenue Non Current              |  6.63e+08   |  5.89e+08   |  5.16e+08   |  5.29e+08   |  6.32e+08   |
        | Deferred Tax Liabilities                  |  0          |  0          |  0          |  0          |  0          |
        | Other Non Current Liabilities             |  8.58e+08   |  8.35e+08   |  8.08e+08   |  8.32e+08   |  9.5e+08    |
        | Total Non Current Liabilities             |  9.006e+09  |  8.837e+09  |  1.2043e+10 |  1.4008e+10 |  1.5171e+10 |
        | Other Liabilities                         |  0          |  0          |  0          |  0          |  0          |
        | Capital Lease Obligations                 |  1.451e+09  |  1.496e+09  |  1.539e+09  |  1.681e+09  |  1.774e+09  |
        | Total Debt                                |  7.65e+09   |  7.516e+09  |  1.089e+10  |  1.2884e+10 |  1.3848e+10 |
        | Net Debt                                  | -1.507e+09  | -7.46e+08   |  1.316e+09  |  3.086e+09  |  4.55e+09   |
        | Total Investments                         |  2.82e+09   |  2.716e+09  |  2.433e+09  |  2.232e+09  |  2.027e+09  |
        | Total Liabilities                         |  1.6015e+10 |  1.6376e+10 |  1.8568e+10 |  1.9263e+10 |  2.0275e+10 |
        | Treasury Stock                            | -6.343e+09  | -7.127e+09  | -7.552e+09  | -7.552e+09  | -7.552e+09  |
        | Preferred Stock                           |  0          |  0          |  0          |  0          |  0          |
        | Common Stock                              |  1.22e+08   |  1.23e+08   |  1.23e+08   |  1.23e+08   |  1.24e+08   |
        | Retained Earnings                         |  4.5916e+10 |  4.7274e+10 |  4.6873e+10 |  4.4426e+10 |  4.2391e+10 |
        | Additional Paid In Capital                |  9.95e+09   |  1.0197e+10 |  1.0335e+10 |  1.0633e+10 |  1.0782e+10 |
        | Accumulated Other Comprehensive Income    | -3.64e+08   | -5.6e+08    | -4.73e+08   | -3.73e+08   | -3.4e+08    |
        | Other Total Shareholder Equity            |  0          |  0          |  0          |  0          |  0          |
        | Total Shareholder Equity                  |  4.9281e+10 |  4.9907e+10 |  4.9306e+10 |  4.7257e+10 |  4.5405e+10 |
        | Total Equity                              |  4.9281e+10 |  4.9907e+10 |  4.9306e+10 |  4.7257e+10 |  4.5405e+10 |
        | Minority Interest                         |  0          |  0          |  0          |  0          |  0          |
        | Total Liabilities and Equity              |  6.5296e+10 |  6.6283e+10 |  6.7874e+10 |  6.652e+10  |  6.568e+10  |
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
                    self._balance_sheet_statement = currencies_model.convert_currencies(
                        financial_statement_data=self._balance_sheet_statement,
                        financial_statement_currencies=self._statement_currencies,
                        exchange_rate_data=(
                            self._quarterly_exchange_rate_data["Adj Close"]
                            if self._quarterly
                            else self._yearly_exchange_rate_data["Adj Close"]
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
        period. Therefore, trailing results are available for this statement.

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
            ["TSLA", "MU"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            quarterly=True,
            start_date='2022-05-01',
            end_date="2025-12-31",
        )

        income_sheet_statements = toolkit.get_income_statement()

        income_sheet_statements.loc['TSLA']
        ```

        Which returns:

        |                                              |      2022Q2 |      2022Q3 |      2022Q4 |      2023Q1 |      2023Q2 |
        |:---------------------------------------------|------------:|------------:|------------:|------------:|------------:|
        | Revenue                                      |  1.6934e+10 |  2.1454e+10 |  2.4318e+10 |  2.3329e+10 |  2.4927e+10 |
        | Cost of Goods Sold                           |  1.27e+10   |  1.6072e+10 |  1.8541e+10 |  1.8818e+10 |  2.0394e+10 |
        | Gross Profit                                 |  4.234e+09  |  5.382e+09  |  5.777e+09  |  4.511e+09  |  4.533e+09  |
        | Research and Development Expenses            |  6.67e+08   |  7.33e+08   |  8.1e+08    |  7.71e+08   |  9.43e+08   |
        | General and Administrative Expenses          |  9.61e+08   |  9.61e+08   |  1.032e+09  |  1.076e+09  |  1.191e+09  |
        | Selling and Marketing Expenses               |  0          |  0          |  0          |  0          |  0          |
        | Selling, General and Administrative Expenses |  9.61e+08   |  9.61e+08   |  1.032e+09  |  1.076e+09  |  1.191e+09  |
        | Other Expenses                               |  1.42e+08   |  0          |  3.4e+07    |  0          |  0          |
        | Operating Expenses                           |  1.77e+09   |  1.694e+09  |  1.876e+09  |  1.847e+09  |  2.134e+09  |
        | Cost and Expenses                            |  1.447e+10  |  1.7766e+10 |  2.0417e+10 |  2.0665e+10 |  2.2528e+10 |
        | Interest Income                              |  2.6e+07    |  8.6e+07    |  1.57e+08   |  2.13e+08   |  2.38e+08   |
        | Interest Expense                             |  4.4e+07    |  5.3e+07    |  3.3e+07    |  2.9e+07    |  2.8e+07    |
        | Net Interest Income                          | -1.8e+07    |  3.3e+07    |  1.24e+08   |  1.84e+08   |  2.1e+08    |
        | Depreciation and Amortization                |  9.22e+08   |  9.56e+08   |  9.89e+08   |  1.046e+09  |  1.154e+09  |
        | EBITDA                                       |  3.44e+09   |  4.645e+09  |  5.005e+09  |  3.875e+09  |  4.119e+09  |
        | EBIT                                         |  2.518e+09  |  3.689e+09  |  4.016e+09  |  2.829e+09  |  2.965e+09  |
        | Non Operating Income Excluding Interest      | -5.4e+07    | -1e+06      | -1.15e+08   | -1.65e+08   | -5.66e+08   |
        | Operating Income                             |  2.464e+09  |  3.688e+09  |  3.901e+09  |  2.664e+09  |  2.399e+09  |
        | Total Other Income Expenses                  |  1e+07      | -5.2e+07    |  8.2e+07    |  1.36e+08   |  5.38e+08   |
        | Income Before Tax                            |  2.474e+09  |  3.636e+09  |  3.983e+09  |  2.8e+09    |  2.937e+09  |
        | Income Tax Expense                           |  2.05e+08   |  3.05e+08   |  2.76e+08   |  2.61e+08   |  3.23e+08   |
        | Net Income from Continuing Operations        |  2.269e+09  |  3.331e+09  |  3.707e+09  |  2.539e+09  |  2.614e+09  |
        | Net Income from Discontinued Operations      |  0          |  0          |  0          |  0          |  0          |
        | Other Adjustments to Net Income              |  0          |  0          |  0          |  0          |  0          |
        | Net Income before Deductions                 |  2.259e+09  |  3.292e+09  |  3.714e+09  |  2.513e+09  |  2.703e+09  |
        | Net Income Deductions                        |  0          |  0          | -8e+06      |  0          |  0          |
        | Net Income                                   |  2.256e+09  |  3.292e+09  |  3.722e+09  |  2.518e+09  |  2.703e+09  |
        | EPS                                          |  0.73       |  1.05       |  1.18       |  0.8        |  0.85       |
        | EPS Diluted                                  |  0.65       |  0.95       |  1.07       |  0.73       |  0.78       |
        | Weighted Average Shares                      |  3.111e+09  |  3.146e+09  |  3.16e+09   |  3.166e+09  |  3.171e+09  |
        | Weighted Average Shares Diluted              |  3.464e+09  |  3.468e+09  |  3.475e+09  |  3.468e+09  |  3.478e+09  |
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
                        exchange_rate_data=(
                            self._quarterly_exchange_rate_data["Adj Close"]
                            if self._quarterly
                            else self._yearly_exchange_rate_data["Adj Close"]
                        ),
                        items_not_to_adjust=[
                            "Gross Profit Ratio",
                            "EBITDA Ratio",
                            "Operating Income Ratio",
                            "Income Before Tax Ratio",
                            "Net Income Ratio",
                            "EPS",
                            "EPS Diluted",
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
            # The trailing period does not apply to the Weighted Average Shares rows.
            weighted_average_shares = income_statement.loc[
                :, ["Weighted Average Shares", "Weighted Average Shares Diluted"], :
            ]

            # The rolling window is calculated for the rest of the income statement.
            income_statement = self._income_statement.T.rolling(trailing).sum().T

            # Weighted Average Shares are kept at the current value rather than summed.
            income_statement.loc[weighted_average_shares.index] = (
                weighted_average_shares
            )

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
        affect cash and cash equivalents. Therefore, trailing results are available for this statement.

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
            ["MU", "AMZN"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            quarterly=True,
            start_date='2022-09-01',
            end_date="2025-12-31",
        )

        cash_flow_statements = toolkit.get_cash_flow_statement()

        cash_flow_statements.loc['AMZN']
        ```

        Which returns:

        |                                 |      2022Q3 |      2022Q4 |      2023Q1 |      2023Q2 |
        |:--------------------------------|------------:|------------:|------------:|------------:|
        | Net Income                      |  2.872e+09  |  2.78e+08   |  3.172e+09  |  6.75e+09   |
        | Depreciation and Amortization   |  1.0327e+10 |  1.3145e+10 |  1.1123e+10 |  1.1589e+10 |
        | Deferred Income Tax             | -8.25e+08   | -3.367e+09  | -4.72e+08   | -2.744e+09  |
        | Stock Based Compensation        |  5.556e+09  |  5.606e+09  |  4.748e+09  |  7.127e+09  |
        | Change in Working Capital       | -5.254e+09  |  1.0526e+10 | -1.4317e+10 | -6.293e+09  |
        | Change in Accounts Receivables  | -4.794e+09  | -8.788e+09  |  4.724e+09  | -2.041e+09  |
        | Change in Inventory             |  7.32e+08   |  3.18e+09   |  3.71e+08   | -2.373e+09  |
        | Change in Accounts Payables     | -1.226e+09  |  9.852e+09  | -1.1264e+10 |  3.029e+09  |
        | Change in Other Working Capital |  3.4e+07    |  6.282e+09  | -8.148e+09  | -4.908e+09  |
        | Other Non Cash Items            | -1.272e+09  |  2.985e+09  |  5.34e+08   |  4.7e+07    |
        | Cash Flow from Operations       |  1.1404e+10 |  2.9173e+10 |  4.788e+09  |  1.6476e+10 |
        | Property, Plant and Equipment   | -1.6378e+10 | -1.6592e+10 | -1.4207e+10 | -1.1455e+10 |
        | Acquisitions                    | -8.85e+08   | -8.31e+08   | -3.513e+09  | -3.16e+08   |
        | Purchases of Investments        | -2.39e+08   | -2.33e+08   | -3.38e+08   | -4.96e+08   |
        | Sales of Investments            |  5.57e+08   |  5.683e+09  |  1.115e+09  |  1.551e+09  |
        | Other Investing Activities      |  1.337e+09  |  1.152e+09  |  1.137e+09  |  1.043e+09  |
        | Cash Flow from Investing        | -1.5608e+10 | -1.0821e+10 | -1.5806e+10 | -9.673e+09  |
        | Net Debt Issued                 |  3.016e+09  |  8.6e+07    |  6.354e+09  | -6.539e+09  |
        | Long Term Debt Issued           | -1.406e+09  |  5.276e+09  | -2.823e+09  | -3.297e+09  |
        | Short Term Debt Issued          |  4.422e+09  | -5.19e+09   |  9.177e+09  | -3.242e+09  |
        | Net Stock Issued                |  0          |  0          |  0          |  0          |
        | Net Common Stock Issued         |  0          |  0          |  0          |  0          |
        | Common Stock Issued             |  0          |  0          |  0          |  0          |
        | Common Stock Purchased          |  0          |  0          |  0          |  0          |
        | Net Preferred Stock Issued      |  0          |  0          |  0          |  0          |
        | Common Dividends Paid           |  0          |  0          |  0          |  0          |
        | Preferred Dividends Paid        |  0          |  0          |  0          |  0          |
        | Dividends Paid                  |  0          |  0          |  0          |  0          |
        | Other Financing Activities      |  0          |  0          |  0          |  0          |
        | Cash Flow from Financing        |  3.016e+09  |  8.6e+07    |  6.354e+09  | -6.539e+09  |
        | Forex Changes on Cash           | -1.334e+09  |  6.37e+08   |  1.45e+08   |  6.9e+07    |
        | Net Change in Cash              | -2.522e+09  |  1.9075e+10 | -4.519e+09  |  3.33e+08   |
        | Cash End of Period              |  3.5178e+10 |  5.4253e+10 |  4.9734e+10 |  5.0067e+10 |
        | Cash Beginning of Period        |  3.77e+10   |  3.5178e+10 |  5.4253e+10 |  4.9734e+10 |
        | Operating Cash Flow             |  1.1404e+10 |  2.9173e+10 |  4.788e+09  |  1.6476e+10 |
        | Capital Expenditure             | -1.6378e+10 | -1.6592e+10 | -1.4207e+10 | -1.1455e+10 |
        | Free Cash Flow                  | -4.974e+09  |  1.2581e+10 | -9.419e+09  |  5.021e+09  |
        | Income Taxes Paid               |  7.42e+08   |  1.695e+09  |  6.19e+08   |  3.735e+09  |
        | Interest Paid                   |  4.31e+08   |  7.68e+08   |  5.42e+08   |  1.072e+09  |
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
                    self._cash_flow_statement = currencies_model.convert_currencies(
                        financial_statement_data=self._cash_flow_statement,
                        financial_statement_currencies=self._statement_currencies,
                        exchange_rate_data=(
                            self._quarterly_exchange_rate_data["Adj Close"]
                            if self._quarterly
                            else self._yearly_exchange_rate_data["Adj Close"]
                        ),
                        financial_statement_name="cash flow statement",
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
        Retrieves the balance, cash and income statistics for the company(s) from the specified source.

        Note that this also obtains the balance sheet statement at the same time given that it's the same
        API call. This is done to reduce the number of API calls to FinancialModelingPrep.

        Also known as: key stats, shares outstanding, float data.

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
            "TSLA",
            api_key="FINANCIAL_MODELING_PREP_KEY",
            quarterly=True,
            start_date='2023-05-01',
            end_date="2025-12-31",
        )

        toolkit.get_statistics_statement()
        ```

        Which returns:

        |                   | 2023Q2              |
        |:------------------|:--------------------|
        | Reported Currency | USD                 |
        | CIK ID            | 1318605             |
        | Filling Date      | nan                 |
        | Accepted Date     | 2023-07-21 18:08:29 |
        | Calendar Year     | nan                 |
        | Period            | Q2                  |
        | SEC Link          | nan                 |
        | Document Link     | nan                 |

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

        This function is relevant if you want to supply your own datasets. See for a proper
        guide the following
        notebook: https://www.jeroenbouma.com/projects/financetoolkit/external-datasets

        Args:
            path (str, optional): The path where to save the files to.

        Returns:
            Three csv files saved to the desired location.
        """
        if path:
            _copy_normalization_files(path)
        else:
            _copy_normalization_files()

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

        |    | source       | dataset                   |   entities |   entries | oldest_write        | newest_write        |
        |---:|:-------------|:--------------------------|-----------:|----------:|:--------------------|:--------------------|
        | 25 | USTreasury   | par_yield_curve_real      |          1 |         1 | 2026-10-06 19:31:44 | 2026-10-06 19:31:44 |
        | 26 | USTreasury   | par_yield_curve_real_year |          2 |         2 | 2026-10-06 19:31:44 | 2026-10-06 19:31:44 |
        | 27 | USTreasury   | par_yield_curve_year      |          2 |         2 | 2026-10-06 19:31:44 | 2026-10-06 19:31:44 |
        | 28 | YahooFinance | historical                |          1 |         1 | 2026-10-06 22:09:08 | 2026-10-06 22:09:08 |
        | 29 | YahooFinance | historical_statistics     |          3 |         3 | 2026-09-29 20:46:10 | 2026-09-29 20:46:10 |
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
        toolkit.clear_cache(source=policy_model.OECD)

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

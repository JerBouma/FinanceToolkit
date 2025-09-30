"""Toolkit Module"""

__docformat__ = "google"


import re
import warnings
from collections import Counter
from datetime import datetime, timedelta

import pandas as pd

from financetoolkit import currencies_model, helpers
from financetoolkit.economics.economics_controller import Economics
from financetoolkit.fixedincome.fixedincome_controller import FixedIncome
from financetoolkit.fmp_model import (
    get_analyst_estimates as _get_analyst_estimates,
    get_dividend_calendar as _get_dividend_calendar,
    get_earnings_calendar as _get_earnings_calendar,
    get_esg_scores as _get_esg_scores,
    get_financial_data as _get_financial_data,
    get_profile as _get_profile,
    get_quote as _get_quote,
    get_rating as _get_rating,
    get_revenue_segmentation as _get_revenue_segmentation,
)
from financetoolkit.fundamentals_model import collect_financial_statements
from financetoolkit.historical_model import (
    convert_daily_to_other_period as _convert_daily_to_other_period,
    get_historical_data as _get_historical_data,
    get_historical_statistics as _get_historical_statistics,
)
from financetoolkit.models.models_controller import Models
from financetoolkit.normalization_model import (
    copy_normalization_files as _copy_normalization_files,
    initialize_statements_and_normalization as _initialize_statements_and_normalization,
)
from financetoolkit.options.options_controller import Options
from financetoolkit.performance.performance_controller import Performance
from financetoolkit.ratios.ratios_controller import Ratios
from financetoolkit.risk.risk_controller import Risk
from financetoolkit.technicals.technicals_controller import Technicals
from financetoolkit.utilities import cache_model, logger_model

# Set up logger, this is meant to display useful messages, warnings or errors when
# the Finance Toolkit runs into issues or does something that might not be entirely
# logical at first
logger_model.setup_logger()
logger = logger_model.get_logger()

# Runtime errors are ignored on purpose given the nature of the calculations
# sometimes leading to division by zero or other mathematical errors. This is however
# for financial analysis purposes not an issue and should not be considered as a bug.
warnings.filterwarnings("ignore", category=RuntimeWarning)

# pylint: disable=too-many-instance-attributes,too-many-lines,line-too-long,too-many-locals
# pylint: disable=too-many-function-args,too-many-public-methods
# ruff: noqa: E501

TICKER_LIMIT = 20

try:
    from tqdm import tqdm

    ENABLE_TQDM = True
except ImportError:
    ENABLE_TQDM = False


class Toolkit:
    """
    The Finance Toolkit is an open-source toolkit in which
    all 150+ financial ratios, indicators and performance measurements
    are written down in the most simplistic way allowing for complete transparency
    of the calculation method. This allows you to not have to rely on metrics
    from other providers and, given a financial statement, allow for efficient manual
    calculations. This leads to one uniform method of calculation being applied that
    is available and understood by everyone.
    """

    def __init__(
        self,
        tickers: list | str | None = None,
        api_key: str = "",
        start_date: str | None = None,
        end_date: str | None = None,
        quarterly: bool = False,
        use_cached_data: bool | str = False,
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
    ):
        """
        Initializes a Toolkit object with a ticker or a list of tickers. The way the Toolkit is initialized
        will define how the data is collected. For example, if you enable the quarterly flag, you will
        be able to collect quarterly data. Next to that, you can define the start and end date to specify
        a specific range. Another option is to work with cached data. This is useful when you have collected
        data before and want to use this data again. This can be done by setting the use_cached_data variable
        to True. If you want to use a specific location to store the cached data, you can define this as a string,
        e.g. "datasets".

        It is good to note that the Finance Toolkit will always attempt to acquire data from Financial Modeling Prep
        if an API key is set. If this isn't the case, the data comes from Yahoo Finance. In case you have an API key
        set and the current plan doesn't allow for the data to be collected, the Toolkit will automatically switch to
        Yahoo Finance. You can disable this behaviour by setting the enforce_source variable to "FinancialModelingPrep".

        For more information on the capabilities of the Finance Toolkit see here: https://www.jeroenbouma.com/projects/financetoolkit

        Args:
            tickers (list | str | None): A string or a list of strings containing the company ticker(s). E.g. 'TSLA' or 'MSFT'.
            Find tickers on various websites or via the FinanceDatabase: https://github.com/JerBouma/financedatabase. Defaults to None.
            api_key (str): An API key from FinancialModelingPrep. Obtain one here: https://www.jeroenbouma.com/fmp. Defaults to "".
            start_date (str | None): A string containing the start date of the data. Needs to be formatted as YYYY-MM-DD.
            Defaults to 5 years/quarters back from today depending on the 'quarterly' flag.
            end_date (str | None): A string containing the end date of the data. Needs to be formatted as YYYY-MM-DD.
            Defaults to today.
            quarterly (bool): A boolean indicating whether to collect quarterly data. Defaults to False (yearly).
            Note that historical data can still be collected for any period and interval.
            use_cached_data (bool | str): A boolean indicating whether to use cached data. If True, uses a 'cached' folder.
            If a string is provided, uses that string as the path to the cache folder. Defaults to False.
            risk_free_rate (str): The risk-free rate identifier ('13w', '5y', '10y', '30y'). Based on US Treasury Yields.
            Used for calculations like Excess Returns. Defaults to "10y".
            benchmark_ticker (str | None): The benchmark ticker (e.g., 'SPY' for S&P 500). Used for comparative analysis
            (CAPM, Alpha, Beta). Defaults to "SPY". Set to None to disable benchmark comparison.
            enforce_source (str | None): Enforce data source ('FinancialModelingPrep' or 'YahooFinance').
            Defaults to None (uses FMP if api_key provided, otherwise YahooFinance, with fallback).
            historical (pd.DataFrame): Custom historical price data. See notebook:
            https://www.jeroenbouma.com/projects/financetoolkit/external-datasets. Defaults to an empty DataFrame.
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

        # Working with cached data
        toolkit = Toolkit(
            tickers=["WMT", "AAPL"],
            quarterly=True,
            api_key="FINANCIAL_MODELING_PREP_KEY",
            use_cached_data=True)

        # Changing the benchmark and risk free rate
        toolkit = Toolkit(
            tickers="AMZN",
            benchmark_ticker="^DJI",
            risk_free_rate="30y",
            api_key="FINANCIAL_MODELING_PREP_KEY")
        ```
        """
        self._api_key = api_key
        self._risk_free_rate = risk_free_rate
        self._rounding = rounding
        self._remove_invalid_tickers = remove_invalid_tickers
        self._invalid_tickers: list = []

        self._use_cached_data = (
            use_cached_data if isinstance(use_cached_data, bool) else True
        )
        self._cached_data_location = (
            "cached" if isinstance(use_cached_data, bool) else use_cached_data
        )
        self._benchmark_ticker = benchmark_ticker

        if start_date and re.match(r"^\d{4}-\d{2}-\d{2}$", start_date) is None:
            raise ValueError(
                "Please input a valid start date (%Y-%m-%d) like '2010-01-01'"
            )
        if end_date and re.match(r"^\d{4}-\d{2}-\d{2}$", end_date) is None:
            raise ValueError(
                "Please input a valid end date (%Y-%m-%d) like '2020-01-01'"
            )
        if start_date and end_date and start_date > end_date:
            raise ValueError(
                f"Please ensure the start date {start_date} is before the end date {end_date}"
            )

        if risk_free_rate not in [
            "13w",
            "5y",
            "10y",
            "30y",
        ]:
            raise ValueError(
                "Please select a valid risk free rate (13w, 5y, 10y or 30y)"
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

        if use_cached_data:
            cached_configurations = cache_model.load_cached_data(
                cached_data_location=self._cached_data_location,
                file_name="configurations.pickle",
                method="pickle",
                return_empty_type={},
            )

            if cached_configurations:  # Check if dictionary is not empty
                cached_overwrites = []
                # Map cache keys to tuples of (initial_value, attribute_name)
                config_mapping = {
                    "start_date": (self._start_date, "_start_date"),
                    "end_date": (self._end_date, "_end_date"),
                    "quarterly": (self._quarterly, "_quarterly"),
                    "benchmark_ticker": (self._benchmark_ticker, "_benchmark_ticker"),
                    "risk_free_rate": (self._risk_free_rate, "_risk_free_rate"),
                }

                # Compare initial values with cached values, update instance if different
                for key, (initial_value, attr_name) in config_mapping.items():
                    cached_value = cached_configurations.get(key)
                    # Check if cached value exists and is different from the initial value
                    if cached_value is not None and initial_value != cached_value:
                        setattr(
                            self, attr_name, cached_value
                        )  # Update instance attribute
                        cached_overwrites.append(f"{key} ({cached_value})")

                # Handle tickers separately: compare input tickers with cached tickers
                cached_tickers = cached_configurations.get("tickers")
                # Check if cached tickers exist and are different from the input tickers
                if cached_tickers is not None and tickers != cached_tickers and tickers:
                    # Only log the change if the user actually provided tickers initially
                    cached_overwrites.append("tickers")
                    tickers = cached_tickers

                if cached_overwrites:
                    folder = (
                        "cached"
                        if isinstance(use_cached_data, bool)
                        else use_cached_data
                    )
                    logger.info(
                        "The following variables are overwritten by the cached "
                        "configurations: %s\n"
                        "If this is undesirable, please set the use_cached_data variable "
                        "to False, delete the directory %s or select a new "
                        "location for the cached data by changing the use_cached_data "
                        "variable to a string.",
                        ", ".join(cached_overwrites),
                        folder,
                    )
            else:
                # Save the current configuration if no cache exists
                # Use the values as they are before potential overwrites from cache
                cache_model.save_cached_data(
                    cached_data={
                        "tickers": tickers,  # Use the initial tickers list/str
                        "start_date": self._start_date,
                        "end_date": self._end_date,
                        "quarterly": self._quarterly,
                        "benchmark_ticker": self._benchmark_ticker,
                        "risk_free_rate": self._risk_free_rate,  # Use the initial risk_free_rate
                    },
                    cached_data_location=self._cached_data_location,
                    file_name="configurations.pickle",
                    method="pickle",
                    include_message=False,
                )

        if isinstance(tickers, str):
            tickers = [tickers.upper()]
        elif isinstance(tickers, list):
            tickers = [
                ticker.upper() if ticker != "Portfolio" else ticker
                for ticker in tickers
            ]
        elif tickers is None:
            raise ValueError("Please input a ticker or a list of tickers.")
        else:
            raise TypeError("Tickers must be a string or a list of strings.")

        self._tickers: list[str] = []

        for ticker in tickers:
            # Check whether the ticker is in ISIN format and if say so convert it to a ticker
            self._tickers.append(helpers.convert_isin_to_ticker(ticker))

        # Take out duplicate tickers if applicable
        deduplicated_tickers = list(set(self._tickers))

        if len(deduplicated_tickers) != len(self._tickers):
            duplicate_tickers = [
                ticker for ticker, count in Counter(self._tickers).items() if count > 1
            ]
            logger.warning(
                "Found duplicate tickers, duplicate entries of the following tickers are removed: %s",
                ", ".join(duplicate_tickers),
            )
            self._tickers = deduplicated_tickers

        if self._benchmark_ticker in self._tickers:
            logger.warning(
                "Please note that the benchmark ticker (%s) is also "
                "included in the tickers. Therefore, this ticker will be removed from the "
                "tickers list. If this is not desired, please set the benchmark_ticker to None.",
                self._benchmark_ticker,
            )
            self._tickers.remove(self._benchmark_ticker)

        self._enforce_source: str | None = enforce_source

        if self._enforce_source not in [None, "FinancialModelingPrep", "YahooFinance"]:
            raise ValueError(
                "Please select either FinancialModelingPrep or YahooFinance as the "
                "enforced source."
            )
        if self._enforce_source == "FinancialModelingPrep" and not self._api_key:
            raise ValueError(
                "Please input an API key from FinancialModelingPrep if you wish to use "
                "historical data from FinancialModelingPrep."
            )

        if sleep_timer is None:
            # This tests the API key to determine the subscription plan. This is relevant for the sleep timer
            # but also for other components of the Toolkit. This prevents wait timers from occurring while
            # it wouldn't result to any other answer than a rate limit error.
            determine_plan = _get_financial_data(
                url=f"https://financialmodelingprep.com/stable/income-statement?symbol=AAPL&apikey={api_key}&limit=10",
                sleep_timer=False,
                user_subscription="Free",
            )

            self._fmp_plan = "Premium"

            for option in [
                "PREMIUM QUERY PARAMETER",
                "EXCLUSIVE ENDPOINT",
                "NO DATA",
                "BANDWIDTH LIMIT REACH",
                "INVALID API KEY",
                "LIMIT REACH",
            ]:
                if option in determine_plan:
                    if option == "INVALID API KEY" and api_key:
                        self._enforce_source = "YahooFinance"
                        logger.error(
                            "You have entered an invalid API key from Financial Modeling Prep. Obtain your API key for free "
                            "and get 15%% off the Premium plans by using the following affiliate link.\nThis also supports "
                            "the project: https://www.jeroenbouma.com/fmp. Using Yahoo Finance as data source instead."
                        )
                    self._fmp_plan = "Free"
                    break
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

            # Define attributes and their corresponding cache file names
            cached_attributes = {
                "_profile": "profile.pickle",
                "_quote": "quote.pickle",
                "_rating": "rating.pickle",
                "_analyst_estimates": "analyst_estimates.pickle",
                "_analyst_estimates_growth": "analyst_estimates_growth.pickle",
                "_dividend_calendar": "dividend_calendar.pickle",
                "_earnings_calendar": "earnings_calendar.pickle",
                "_esg_scores": "esg_scores.pickle",
                "_revenue_geographic_segmentation": "revenue_geographic_segmentation.pickle",
                "_revenue_product_segmentation": "revenue_product_segmentation.pickle",
            }

            # Initialize FinancialModelingPrep Variables
            for attr_name, file_name in cached_attributes.items():
                data = (
                    cache_model.load_cached_data(
                        cached_data_location=self._cached_data_location,
                        file_name=file_name,
                    )
                    if self._use_cached_data
                    else pd.DataFrame()
                )
                setattr(self, attr_name, data)

        if intraday_period and intraday_period not in [
            "1min",
            "5min",
            "15min",
            "30min",
            "1hour",
        ]:
            raise ValueError(
                "Please select a valid intraday period (1min, 5min, 15min, 30min or 1hour)"
            )

        self._intraday_period = intraday_period

        # Load intraday data from cache if specified, otherwise initialize empty DataFrame
        self._intraday_historical_data: pd.DataFrame = (
            cache_model.load_cached_data(
                cached_data_location=self._cached_data_location,
                file_name="intraday_historical_data.pickle",
            )
            if self._use_cached_data
            else pd.DataFrame()
        )

        # Use provided historical data if available, otherwise load daily data from cache or initialize empty DataFrame
        self._historical = historical

        self._daily_historical_data: pd.DataFrame = (
            historical
            if not historical.empty
            else (
                cache_model.load_cached_data(
                    cached_data_location=self._cached_data_location,
                    file_name="daily_historical_data.pickle",
                )
                if self._use_cached_data
                else pd.DataFrame()
            )
        )

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
        ) = _initialize_statements_and_normalization(
            balance=balance,
            income=income,
            cash=cash,
            format_location=format_location,
            reverse_dates=self._reverse_dates,
            use_cached_data=use_cached_data,
            cached_data_location=self._cached_data_location,
            start_date=self._start_date,
            end_date=self._end_date,
            quarterly=self._quarterly,
        )

        self._balance_sheet_statement_growth: pd.DataFrame = pd.DataFrame()
        self._income_statement_growth: pd.DataFrame = pd.DataFrame()
        self._cash_flow_statement_growth: pd.DataFrame = pd.DataFrame()
        self._currencies: list = []
        self._statement_currencies: pd.Series = pd.Series()
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

        pd.set_option("display.float_format", str)

    @property
    def ratios(self) -> Ratios:
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

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        profitability_ratios = toolkit.ratios.collect_profitability_ratios()

        profitability_ratios.loc['AAPL']
        ```

        Which returns:

        |                                             |     2018 |     2019 |     2020 |     2021 |     2022 |
        |:--------------------------------------------|---------:|---------:|---------:|---------:|---------:|
        | Gross Margin                                | 0.383437 | 0.378178 | 0.382332 | 0.417794 | 0.433096 |
        | Operating Margin                            | 0.26694  | 0.24572  | 0.241473 | 0.297824 | 0.302887 |
        | Net Profit Margin                           | 0.224142 | 0.212381 | 0.209136 | 0.258818 | 0.253096 |
        | Interest Burden Ratio                       | 1.02828  | 1.02827  | 1.01211  | 1.00237  | 0.997204 |
        | Income Before Tax Profit Margin             | 0.274489 | 0.252666 | 0.244398 | 0.298529 | 0.30204  |
        | Effective Tax Rate                          | 0.183422 | 0.159438 | 0.144282 | 0.133023 | 0.162045 |
        | Return on Assets (ROA)                      | 0.162775 | 0.16323  | 0.177256 | 0.269742 | 0.282924 |
        | Return on Equity (ROE)                      | 0.555601 | 0.610645 | 0.878664 | 1.50071  | 1.96959  |
        | Return on Invested Capital (ROIC)           | 0.269858 | 0.293721 | 0.344126 | 0.503852 | 0.562645 |
        | Return on Capital Employed (ROCE)           | 0.305968 | 0.297739 | 0.320207 | 0.495972 | 0.613937 |
        | Return on Tangible Assets                   | 0.555601 | 0.610645 | 0.878664 | 1.50071  | 1.96959  |
        | Income Quality Ratio                        | 1.30073  | 1.25581  | 1.4052   | 1.09884  | 1.22392  |
        | Net Income per EBT                          | 0.816578 | 0.840562 | 0.855718 | 0.866977 | 0.837955 |
        | Free Cash Flow to Operating Cash Flow Ratio | 0.828073 | 0.848756 | 0.909401 | 0.893452 | 0.912338 |
        | EBT to EBIT Ratio                           | 0.957448 | 0.948408 | 0.958936 | 0.976353 | 0.975982 |
        | EBIT to Revenue                             | 0.286688 | 0.26641  | 0.254864 | 0.305759 | 0.309473 |

        """
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
            empty_data_iterator = (
                tqdm(empty_data, desc="Obtaining financial statements")
                if ENABLE_TQDM & self._progress_bar
                else empty_data
            )

            for statement in empty_data_iterator:
                if statement == "Balance Sheet Statement":
                    self.get_balance_sheet_statement(progress_bar=False)
                if statement == "Income Statement":
                    self.get_income_statement(progress_bar=False)
                if statement == "Cash Flow Statement":
                    self.get_cash_flow_statement(progress_bar=False)

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
        )

        if self._portfolio_weights:
            ratios._portfolio_weights = self._portfolio_weights

        return ratios

    @property
    def models(self) -> Models:
        """
        Gives access to the Models module. The Models module is meant to execute well-known models
        such as DUPONT and the Discounted Cash Flow (DCF) model. These models are also directly
        related to the data retrieved from the Toolkit module.

        See the following link for more information: https://www.jeroenbouma.com/projects/financetoolkit/docs/models

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["TSLA", "AMZN"], api_key="FINANCIAL_MODELING_PREP_KEY", quarterly=True, start_date='2022-12-31')

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
            empty_data_iterator = (
                tqdm(empty_data, desc="Obtaining financial statements")
                if ENABLE_TQDM & self._progress_bar
                else empty_data
            )

            for statement in empty_data_iterator:
                if statement == "Balance Sheet Statement":
                    self.get_balance_sheet_statement(progress_bar=False)
                if statement == "Income Statement":
                    self.get_income_statement(progress_bar=False)
                if statement == "Cash Flow Statement":
                    self.get_cash_flow_statement(progress_bar=False)

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
        )

    @property
    def options(self) -> Options:
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

        toolkit = Toolkit(["TSLA", "MU"], api_key="FINANCIAL_MODELING_PREP_KEY")

        all_greeks = toolkit.options.collect_all_greeks(start_date='2024-01-03')

        all_greeks.loc['TSLA', '2024-01-04']
        ```

        Which returns:

        |   Strike Price |   Delta |   Dual Delta |   Vega |   Theta |    Rho |   Epsilon |   Lambda |   Gamma |   Dual Gamma |   Vanna |    Charm |   Vomma |    Vera |      Veta |     PD |   Speed |   Zomma |   Color |   Ultima |
        |---------------:|--------:|-------------:|-------:|--------:|-------:|----------:|---------:|--------:|-------------:|--------:|---------:|--------:|--------:|----------:|-------:|--------:|--------:|--------:|---------:|
        |            180 |  1      |      -0.9999 | 0      | -0.0193 | 0.0049 |   -0.6533 |   0.0408 |  0      |       0      | -0      |   0      |  0      | -0      |    0      | 0      | -0      |  0      |  0      |   0      |
        |            185 |  1      |      -0.9999 | 0      | -0.0198 | 0.0051 |   -0.6533 |   0.0446 |  0      |       0      | -0      |   0      |  0      | -0      |    0      | 0      | -0      |  0      |  0      |   0      |
        |            190 |  1      |      -0.9999 | 0      | -0.0204 | 0.0052 |   -0.6533 |   0.0492 |  0      |       0      | -0      |   0      |  0      | -0      |    0      | 0      | -0      |  0      |  0      |   0      |
        |            195 |  1      |      -0.9999 | 0      | -0.0209 | 0.0053 |   -0.6533 |   0.0549 |  0      |       0      | -0      |   0      |  0      | -0      |    0      | 0      | -0      |  0      |  0      |   0      |
        |            200 |  1      |      -0.9999 | 0      | -0.0214 | 0.0055 |   -0.6533 |   0.062  |  0      |       0      | -0      |   0      |  0      | -0      |    0.0014 | 0      | -0      |  0      |  0      |   0      |
        |            205 |  1      |      -0.9999 | 0      | -0.022  | 0.0056 |   -0.6533 |   0.0712 |  0      |       0      | -0      |   0.0005 |  0.0003 | -0      |    0.1236 | 0      | -0      |  0      |  0.0004 |   0.0001 |
        |            210 |  1      |      -0.9999 | 0      | -0.0226 | 0.0058 |   -0.6533 |   0.0837 |  0      |       0      | -0.0002 |   0.0221 |  0.0119 | -0.0001 |    4.6313 | 0      | -0      |  0.0001 |  0.0132 |   0.0034 |
        |            215 |  0.9998 |      -0.9997 | 0.0001 | -0.0254 | 0.0059 |   -0.6532 |   0.1016 |  0.0001 |       0.0001 | -0.0044 |   0.4426 |  0.1942 | -0.0029 |   77.6496 | 0.0001 | -0.0001 |  0.0021 |  0.209  |   0.0336 |
        |            220 |  0.9973 |      -0.9969 | 0.001  | -0.0526 | 0.006  |   -0.6515 |   0.1287 |  0.0012 |       0.0014 | -0.0414 |   4.1955 |  1.4351 | -0.0273 |  600.92   | 0.0014 | -0.0005 |  0.0144 |  1.4569 |   0.1196 |
        |            225 |  0.9777 |      -0.976  | 0.0066 | -0.2079 | 0.006  |   -0.6387 |   0.1723 |  0.0076 |       0.0086 | -0.1884 |  19.0888 |  4.7244 | -0.1249 | 2187.89   | 0.0086 | -0.0022 |  0.0407 |  4.1228 |   0.0829 |
        |            230 |  0.8953 |      -0.8898 | 0.0226 | -0.6528 | 0.0056 |   -0.5849 |   0.2419 |  0.0261 |       0.028  | -0.3993 |  40.3564 |  6.2557 | -0.267  | 3816.31   | 0.028  | -0.0048 |  0.0253 |  2.5239 |  -0.1641 |
        |            235 |  0.6978 |      -0.6874 | 0.0435 | -1.2304 | 0.0044 |   -0.4558 |   0.3442 |  0.0502 |       0.0516 | -0.306  |  30.653  |  1.9785 | -0.2119 | 3623.7    | 0.0516 | -0.0039 | -0.0672 | -6.8719 |  -0.0977 |
        |            240 |  0.4192 |      -0.4078 | 0.0488 | -1.3691 | 0.0027 |   -0.2739 |   0.4789 |  0.0562 |       0.0555 |  0.1634 | -17.1438 |  0.4159 |  0.0934 | 3407.79   | 0.0555 |  0.0014 | -0.096  | -9.7512 |  -0.0222 |
        |            245 |  0.1812 |      -0.1736 | 0.0329 | -0.9207 | 0.0012 |   -0.1184 |   0.6396 |  0.0379 |       0.0359 |  0.4445 | -45.5549 |  5.0536 |  0.2814 | 4080.87   | 0.0359 |  0.0048 | -0.0098 | -0.9474 |  -0.1945 |
        |            250 |  0.0544 |      -0.0513 | 0.0138 | -0.3848 | 0.0004 |   -0.0355 |   0.8183 |  0.0159 |       0.0144 |  0.3232 | -33.01   |  6.468  |  0.2073 | 3328.37   | 0.0144 |  0.0036 |  0.0461 |  4.7176 |  -0.0443 |
        |            255 |  0.0112 |      -0.0104 | 0.0037 | -0.1028 | 0.0001 |   -0.0073 |   1.0084 |  0.0042 |       0.0037 |  0.1223 | -12.477  |  3.4845 |  0.0789 | 1542.52   | 0.0037 |  0.0014 |  0.0325 |  3.3216 |   0.1424 |
        |            260 |  0.0016 |      -0.0015 | 0.0006 | -0.018  | 0      |   -0.001  |   1.205  |  0.0007 |       0.0006 |  0.0276 |  -2.8148 |  1.0161 |  0.0179 |  421.028  | 0.0006 |  0.0003 |  0.0104 |  1.0578 |   0.1054 |
        |            265 |  0.0002 |      -0.0001 | 0.0001 | -0.0021 | 0      |   -0.0001 |   1.4049 |  0.0001 |       0.0001 |  0.004  |  -0.4041 |  0.1783 |  0.0026 |   71.3544 | 0.0001 |  0      |  0.0019 |  0.1933 |   0.0322 |
        |            270 |  0      |      -0      | 0      | -0.0002 | 0      |   -0      |   1.6059 |  0      |       0      |  0.0004 |  -0.0385 |  0.02   |  0.0002 |    7.8471 | 0      |  0      |  0.0002 |  0.0222 |   0.0054 |
        |            275 |  0      |      -0      | 0      | -0      | 0      |   -0      |   1.8068 |  0      |       0      |  0      |  -0.0025 |  0.0015 |  0      |    0.5804 | 0      |  0      |  0      |  0.0017 |   0.0006 |
        |            280 |  0      |      -0      | 0      | -0      | 0      |   -0      |   2.0066 |  0      |       0      |  0      |  -0.0001 |  0.0001 |  0      |    0.0297 | 0      |  0      |  0      |  0.0001 |   0      |
        |            285 |  0      |      -0      | 0      | -0      | 0      |   -0      |   2.2048 |  0      |       0      |  0      |  -0      |  0      |  0      |    0.0011 | 0      |  0      |  0      |  0      |   0      |
        |            290 |  0      |      -0      | 0      | -0      | 0      |   -0      |   2.401  |  0      |       0      |  0      |  -0      |  0      |  0      |    0      | 0      |  0      |  0      |  0      |   0      |
        |            295 |  0      |      -0      | 0      | -0      | 0      |   -0      |   2.595  |  0      |       0      |  0      |  -0      |  0      |  0      |    0      | 0      |  0      |  0      |  0      |   0      |
        """
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
        )

    @property
    def technicals(self) -> Technicals:
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

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        average_directional_index = toolkit.technicals.get_average_directional_index()
        ```

        Which returns:

        | Date       |    AAPL |    MSFT |
        |:-----------|--------:|--------:|
        | 2023-08-21 | 62.8842 | 36.7468 |
        | 2023-08-22 | 65.7063 | 36.5525 |
        | 2023-08-23 | 67.3596 | 35.5149 |
        | 2023-08-24 | 66.4527 | 35.4399 |
        | 2023-08-25 | 63.4837 | 32.3323 |

        """
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
    def performance(self) -> Performance:
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

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.performance.get_capital_asset_pricing_model(period='quarterly')
        ```

        Which returns:

        | Date   |    AAPL |    TSLA |
        |:-------|--------:|--------:|
        | 2022Q3 | -0.0684 | -0.1047 |
        | 2022Q4 |  0.0857 |  0.0828 |
        | 2023Q1 |  0.075  |  0.1121 |
        | 2023Q2 |  0.0922 |  0.1342 |
        | 2023Q3 |  0.0052 | -0.0482 |
        """
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
    def risk(self) -> Risk:
        """
        This gives access to the Risk module. The Risk Module is meant to calculate metrics related to risk such
        as Value at Risk (VaR), Conditional Value at Risk (cVaR), EMWA/GARCH models and similar models.

        It gives insights in the risk a stock composes that is not perceived as easily by looking at the data.
        This class is closely related to the Performance class which highlights things such as Sharpe Ratio and
        Sortino Ratio.

        See the following link for more information: https://www.jeroenbouma.com/projects/financetoolkit/docs/risk

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.risk.get_value_at_risk(period='yearly')
        ```

        Which returns:

        | Date   |    AAPL |    TSLA |
        |:-------|--------:|--------:|
        | 2012   |  0      |  0      |
        | 2013   |  0.1754 |  4.96   |
        | 2014   |  1.7515 |  0.9481 |
        | 2015   | -0.1958 |  0.1454 |
        | 2016   |  0.4177 | -0.3437 |
        | 2017   |  2.6368 |  1.2225 |
        | 2018   | -0.2786 |  0.0718 |
        | 2019   |  3.2243 |  0.4707 |
        | 2020   |  1.729  |  8.3319 |
        | 2021   |  1.3179 |  0.8797 |
        | 2022   | -0.8026 | -1.0046 |
        | 2023   |  1.8549 |  1.8238 |
        """
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

        risk = Risk(
            tickers=(
                tickers + ["Portfolio"] if "Portfolio" in self._tickers else tickers
            ),
            historical_data=historical_data,
            intraday_period=self._intraday_period,
            quarterly=self._quarterly,
            rounding=self._rounding,
        )

        if self._portfolio_weights:
            risk._portfolio_weights = self._portfolio_weights

        return risk

    @property
    def fixedincome(self) -> FixedIncome:
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
        )

        fixedincome.get_effective_yield(maturity=False)
        ```

        Which returns:

        | Date       |    AAA |     AA |      A |    BBB |     BB |      B |    CCC |
        |:-----------|-------:|-------:|-------:|-------:|-------:|-------:|-------:|
        | 2024-01-01 | 0.0456 | 0.047  | 0.0505 | 0.054  | 0.0613 | 0.0752 | 0.1319 |
        | 2024-01-02 | 0.0459 | 0.0473 | 0.0509 | 0.0543 | 0.0622 | 0.0763 | 0.1333 |
        | 2024-01-03 | 0.0459 | 0.0474 | 0.051  | 0.0544 | 0.0634 | 0.0779 | 0.1358 |
        | 2024-01-04 | 0.0466 | 0.0481 | 0.0518 | 0.0551 | 0.0639 | 0.0784 | 0.1367 |
        | 2024-01-05 | 0.047  | 0.0485 | 0.0521 | 0.0554 | 0.0641 | 0.0787 | 0.137  |
        | 2024-01-08 | 0.0465 | 0.0481 | 0.0517 | 0.055  | 0.0633 | 0.0776 | 0.1365 |
        | 2024-01-09 | 0.0464 | 0.048  | 0.0516 | 0.0548 | 0.0629 | 0.0771 | 0.1359 |
        | 2024-01-10 | 0.0464 | 0.048  | 0.0515 | 0.0547 | 0.0622 | 0.0762 | 0.1351 |
        | 2024-01-11 | 0.0456 | 0.0472 | 0.0507 | 0.054  | 0.0619 | 0.076  | 0.1344 |
        | 2024-01-12 | 0.0451 | 0.0467 | 0.0502 | 0.0534 | 0.0613 | 0.0753 | 0.1338 |
        | 2024-01-15 | 0.0451 | 0.0467 | 0.0501 | 0.0533 | 0.0611 | 0.0751 | 0.1328 |
        """
        return FixedIncome(
            start_date=self._start_date,
            end_date=self._end_date,
            quarterly=self._quarterly,
            rounding=self._rounding,
        )

    @property
    def economics(self) -> Economics:
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

        toolkit = Toolkit(["AMZN", "ASML"])

        cpi = toolkit.economics.get_consumer_price_index(period='yearly')

        cpi.loc['2015':, ['United States', 'Netherlands', 'Japan']]
        ```

        Which returns:

        |      |   United States |   Netherlands |    Japan |
        |:-----|----------------:|--------------:|---------:|
        | 2015 |         100     |       100     | 100      |
        | 2016 |         101.262 |       100.317 |  99.8727 |
        | 2017 |         103.419 |       101.703 | 100.356  |
        | 2018 |         105.945 |       103.435 | 101.349  |
        | 2019 |         107.865 |       106.159 | 101.824  |
        | 2020 |         109.195 |       107.51  | 101.799  |
        | 2021 |         114.325 |       110.387 | 101.561  |
        | 2022 |         123.474 |       121.427 | 104.098  |
        """
        return Economics(
            start_date=self._start_date,
            end_date=self._end_date,
            quarterly=self._quarterly,
            rounding=self._rounding,
        )

    def get_profile(self, progress_bar: bool | None = None):
        """
        Obtain the profile of the specified tickers. These include important metrics
        such as the beta, market capitalization, currency, isin, industry, and ipo date
        that give an overall understanding about the company.

        Args:
            progress_bar (bool, optional): Whether to show a progress bar. Defaults to None.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["MSFT", "AAPL"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.get_profile()
        ```

        Which returns:

        |                       | MSFT                      | AAPL                  |
        |:----------------------|:--------------------------|:----------------------|
        | Symbol                | MSFT                      | AAPL                  |
        | Price                 | 316.48                    | 174.49                |
        | Beta                  | 0.903706                  | 1.286802              |
        | Average Volume        | 28153120                  | 57348456              |
        | Market Capitalization | 2353183809372             | 2744500935588         |
        | Last Dividend         | 2.7199999999999998        | 0.96                  |
        | Range                 | 213.43-366.78             | 124.17-198.23         |
        | Changes               | -0.4                      | 0.49                  |
        | Company Name          | Microsoft Corporation     | Apple Inc.            |
        | Currency              | USD                       | USD                   |
        | CIK                   | 789019                    | 320193                |
        | ISIN                  | US5949181045              | US0378331005          |
        | CUSIP                 | 594918104                 | 37833100              |
        | Exchange              | NASDAQ Global Select      | NASDAQ Global Select  |
        | Exchange Short Name   | NASDAQ                    | NASDAQ                |
        | Industry              | Software—Infrastructure   | Consumer Electronics  |
        | Website               | https://www.microsoft.com | https://www.apple.com |
        | CEO                   | Mr. Satya  Nadella        | Mr. Timothy D. Cook   |
        | Sector                | Technology                | Technology            |
        | Country               | US                        | US                    |
        | Full Time Employees   | 221000                    | 164000                |
        | Phone                 | 425 882 8080              | 408 996 1010          |
        | Address               | One Microsoft Way         | One Apple Park Way    |
        | City                  | Redmond                   | Cupertino             |
        | State                 | WA                        | CA                    |
        | ZIP Code              | 98052-6399                | 95014                 |
        | DCF Difference        | 4.56584                   | 4.15176               |
        | DCF                   | 243.594                   | 150.082               |
        | IPO Date              | 1986-03-13                | 1980-12-12            |
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
            self._profile, self._invalid_tickers = _get_profile(
                tickers=self._tickers,
                api_key=self._api_key,
                progress_bar=(
                    progress_bar if progress_bar is not None else self._progress_bar
                ),
                user_subscription=self._fmp_plan,
            )

            if self._use_cached_data:
                cache_model.save_cached_data(
                    cached_data=self._profile,
                    cached_data_location=self._cached_data_location,
                    file_name="profile.pickle",
                )

        if self._remove_invalid_tickers:
            self._tickers = [
                ticker
                for ticker in self._tickers
                if ticker not in self._invalid_tickers
            ]

        return self._profile

    def get_quote(self, progress_bar: bool | None = None):
        """
        Get the quote of the specified tickers. These include important metrics
        such as the price, changes, day low, day high, year low, year high, market
        capitalization, volume, average volume, open, previous close, earnings per
        share (EPS), price to earnings ratio (PE), earnings announcement, shares
        outstanding and timestamp that give an overall understanding about the
        company.

        Args:
            progress_bar (bool, optional): Whether to show a progress bar. Defaults to None.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["TSLA", "AAPL"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.get_quote()
        ```

        Which returns:

        |                        | TSLA                | AAPL                 |
        |:-----------------------|:--------------------|:---------------------|
        | Symbol                 | TSLA                | AAPL                 |
        | Name                   | Tesla, Inc.         | Apple Inc.           |
        | Price                  | 443.21              | 254.43               |
        | Change %               | 0.6380600000000001  | -0.40319000000000005 |
        | Change                 | 2.81                | -1.03                |
        | Volume                 | 78840639            | 39443231             |
        | Day Low                | 439.5               | 253.01               |
        | Day High               | 450.98              | 254.87               |
        | Year High              | 488.54              | 260.1                |
        | Year Low               | 212.11              | 169.21               |
        | Market Capitalization  | 1429511519286       | 3775840427700        |
        | Price Average 50 Days  | 354.2336            | 229.3156             |
        | Price Average 200 Days | 334.7398            | 222.012              |
        | Exchange               | NASDAQ              | NASDAQ               |
        | Open                   | 444.355             | 254.64               |
        | Previous Close         | 440.4               | 255.46               |
        | Timestamp              | 2025-09-29 20:00:00 | 2025-09-29 20:00:01  |
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
            self._quote, self._invalid_tickers = _get_quote(
                tickers=self._tickers,
                api_key=self._api_key,
                progress_bar=(
                    progress_bar if progress_bar is not None else self._progress_bar
                ),
                user_subscription=self._fmp_plan,
            )

            if self._use_cached_data:
                cache_model.save_cached_data(
                    cached_data=self._quote,
                    cached_data_location=self._cached_data_location,
                    file_name="quote.pickle",
                )

        if self._remove_invalid_tickers:
            self._tickers = [
                ticker
                for ticker in self._tickers
                if ticker not in self._invalid_tickers
            ]

        return self._quote

    def get_rating(self, progress_bar: bool | None = None):
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

        Args:
            progress_bar (bool, optional): Whether to show a progress bar. Defaults to None.

        Raises:
            ValueError: If an API key is not defined for FinancialModelingPrep.

        Returns:
            pd.DataFrame: The stock rating information for the specified tickers.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        rating = toolkit.get_rating()

        rating.loc['AMZN'].tail()
        ```

        Which returns:

        | date                | Rating   |   Rating Score |   DCF Score |   ROE Score |   ROA Score |   DE Score |   PE Score |   PB Score |
        |:--------------------|:---------|---------------:|------------:|------------:|------------:|-----------:|-----------:|-----------:|
        | 2025-09-23 00:00:00 | B+       |              3 |           2 |           5 |           5 |          3 |          2 |          1 |
        | 2025-09-24 00:00:00 | B+       |              3 |           2 |           5 |           5 |          3 |          2 |          1 |
        | 2025-09-25 00:00:00 | B+       |              3 |           2 |           5 |           5 |          3 |          2 |          1 |
        | 2025-09-26 00:00:00 | B+       |              3 |           2 |           5 |           5 |          3 |          2 |          1 |
        | 2025-09-29 00:00:00 | B+       |              3 |           2 |           5 |           5 |          3 |          2 |          1 |
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
            self._rating, self._invalid_tickers = _get_rating(
                tickers=self._tickers,
                api_key=self._api_key,
                progress_bar=(
                    progress_bar if progress_bar is not None else self._progress_bar
                ),
                user_subscription=self._fmp_plan,
            )

            if self._use_cached_data:
                cache_model.save_cached_data(
                    cached_data=self._rating,
                    cached_data_location=self._cached_data_location,
                    file_name="rating.pickle",
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
        progress_bar: bool | None = None,
    ):
        """
        Obtain analyst estimates regarding revenues, EBITDA, EBIT, Net Income
        SGA Expenses and EPS. The number of analysts are also reported.

        Note that this information requires a Premium FMP subscription.

        Args:
            limit (int): Defines the maximum years or quarters to obtain.
            overwrite (bool): Defines whether to overwrite the existing data.
            rounding (int): Defines the number of decimal places to round the data to.
            growth (bool): Defines whether to return the growth of the data.
            lag (int | str): Defines the number of periods to lag the growth data by.
            trailing (int): Defines whether to select a trailing period. E.g. when selecting 4 with quarterly data, the TTM is calculated.

        Returns:
            pandas.DataFrame: The analyst estimates for the specified tickers.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "GOOGL", "AMZN"], api_key="FINANCIAL_MODELING_PREP_KEY", start_date="2024-05-01", quarterly=False
        )

        analyst_estimates = toolkit.get_analyst_estimates()

        analyst_estimates.loc['AAPL'].iloc[:, :5]
        ```

        Which returns:

        |                               |         2024 |         2025 |         2026 |         2027 |         2028 |
        |:------------------------------|-------------:|-------------:|-------------:|-------------:|-------------:|
        | Estimated Revenue Low         |  3.89951e+11 |  4.06152e+11 |  4.29886e+11 |  4.61474e+11 |  4.55389e+11 |
        | Estimated Revenue High        |  3.9221e+11  |  4.19587e+11 |  4.5212e+11  |  4.63255e+11 |  5.05369e+11 |
        | Estimated Revenue Average     |  3.90481e+11 |  4.15035e+11 |  4.38728e+11 |  4.62364e+11 |  4.76309e+11 |
        | Estimated EBITDA Low          |  1.28061e+11 |  1.33382e+11 |  1.41176e+11 |  1.51549e+11 |  1.49551e+11 |
        | Estimated EBITDA High         |  1.28803e+11 |  1.37794e+11 |  1.48477e+11 |  1.52134e+11 |  1.65965e+11 |
        | Estimated EBITDA Average      |  1.28235e+11 |  1.36299e+11 |  1.4408e+11  |  1.51842e+11 |  1.56421e+11 |
        | Estimated EBIT Low            |  1.15692e+11 |  1.20498e+11 |  1.2754e+11  |  1.36911e+11 |  1.35106e+11 |
        | Estimated EBIT High           |  1.16362e+11 |  1.24484e+11 |  1.34136e+11 |  1.3744e+11  |  1.49934e+11 |
        | Estimated EBIT Average        |  1.15849e+11 |  1.23134e+11 |  1.30163e+11 |  1.37175e+11 |  1.41313e+11 |
        | Estimated Net Income Low      |  9.45908e+10 |  1.11489e+11 |  1.18544e+11 |  1.22161e+11 |  1.38181e+11 |
        | Estimated Net Income High     |  1.05866e+11 |  1.14578e+11 |  1.28409e+11 |  1.46962e+11 |  1.58172e+11 |
        | Estimated Net Income Average  |  9.7294e+10  |  1.13033e+11 |  1.23476e+11 |  1.25929e+11 |  1.46549e+11 |
        | Estimated SGA Expense Low     |  2.55838e+10 |  2.66468e+10 |  2.82039e+10 |  3.02763e+10 |  2.98771e+10 |
        | Estimated SGA Expense High    |  2.5732e+10  |  2.75282e+10 |  2.96626e+10 |  3.03931e+10 |  3.31562e+10 |
        | Estimated SGA Expense Average |  2.56186e+10 |  2.72295e+10 |  2.8784e+10  |  3.03347e+10 |  3.12496e+10 |
        | Estimated EPS Average         |  6.7082      |  7.3761      |  8.0086      |  8.7606      |  9.5111      |
        | Estimated EPS High            |  6.8708      |  7.4362      |  8.3339      |  9.538       | 10.2655      |
        | Estimated EPS Low             |  6.139       |  7.2358      |  7.6936      |  7.9284      |  8.9681      |
        | Number of Analysts            | 25           | 29           | 30           | 19           |  9           |
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
            ) = _get_analyst_estimates(
                tickers=self._tickers,
                api_key=self._api_key,
                quarter=self._quarterly,
                start_date=self._start_date,
                rounding=rounding if rounding else self._rounding,
                sleep_timer=self._sleep_timer,
                progress_bar=(
                    progress_bar if progress_bar is not None else self._progress_bar
                ),
                user_subscription=self._fmp_plan,
            )

            if self._use_cached_data:
                cache_model.save_cached_data(
                    cached_data=self._analyst_estimates,
                    cached_data_location=self._cached_data_location,
                    file_name="analyst_estimates.pickle",
                )

        if self._remove_invalid_tickers:
            self._tickers = [
                ticker
                for ticker in self._tickers
                if ticker not in self._invalid_tickers
            ]

        if growth:
            self._analyst_estimates_growth = helpers.calculate_growth(
                self._analyst_estimates,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
            )

        if len(self._tickers) == 1 and not self._analyst_estimates.empty:
            return (
                self._analyst_estimates_growth.loc[self._tickers[0]]
                if growth
                else self._analyst_estimates.loc[self._tickers[0]]
            )

        return self._analyst_estimates_growth if growth else self._analyst_estimates

    def get_earnings_calendar(
        self,
        actual_dates: bool = True,
        overwrite: bool = False,
        rounding: int | None = None,
        progress_bar: bool | None = None,
    ):
        """
        Obtain Earnings Calendars for any range of companies. You have the option to
        obtain the actual dates or to convert to the corresponding quarters.

        Note that this information requires a Premium FMP subscription.

        Args:
            actual_dates (bool): Defines whether to return the actual dates or the corresponding quarters.
            overwrite (bool): Defines whether to overwrite the existing data.

        Returns:
            pd.DataFrame: The earnings calendar for the specified tickers.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "GOOGL", "AMZN"], api_key="FINANCIAL_MODELING_PREP_KEY", start_date="2022-08-01", quarterly=False
        )

        earning_calendar = toolkit.get_earnings_calendar()

        earning_calendar.loc['AMZN']
        ```

        Which returns:

        | date                |   EPS |   Estimated EPS |     Revenue |   Estimated Revenue | Last Updated   |
        |:--------------------|------:|----------------:|------------:|--------------------:|:---------------|
        | 2022-10-27 00:00:00 |  0.2  |            0.22 | 1.27101e+11 |         1.27308e+11 | 2025-04-25     |
        | 2023-02-02 00:00:00 |  0.21 |            0.15 | 1.49204e+11 |         1.45713e+11 | 2025-04-25     |
        | 2023-04-27 00:00:00 |  0.31 |            0.21 | 1.27358e+11 |         1.24551e+11 | 2025-04-25     |
        | 2023-08-03 00:00:00 |  0.65 |            0.35 | 1.34383e+11 |         1.19573e+11 | 2025-04-25     |
        | 2023-10-26 00:00:00 |  0.94 |            0.58 | 1.43083e+11 |         1.33393e+11 | 2025-04-25     |
        | 2024-02-01 00:00:00 |  1    |            0.8  | 1.69961e+11 |         1.66172e+11 | 2025-04-25     |
        | 2024-04-30 00:00:00 |  0.98 |            0.83 | 1.43313e+11 |         1.42654e+11 | 2025-04-25     |
        | 2024-08-01 00:00:00 |  1.26 |            1.03 | 1.47977e+11 |         1.48665e+11 | 2025-04-25     |
        | 2024-10-31 00:00:00 |  1.43 |            1.14 | 1.58877e+11 |         1.57275e+11 | 2025-04-25     |
        | 2025-02-06 00:00:00 |  1.86 |            1.49 | 1.87792e+11 |         1.87337e+11 | 2025-05-06     |
        | 2025-05-01 00:00:00 |  1.59 |            1.37 | 1.55667e+11 |         1.55148e+11 | 2025-08-01     |
        | 2025-07-31 00:00:00 |  1.68 |            1.31 | 1.67702e+11 |         1.61776e+11 | 2025-09-30     |
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
            ) = _get_earnings_calendar(
                tickers=self._tickers,
                api_key=self._api_key,
                start_date=self._start_date,
                end_date=self._end_date,
                actual_dates=actual_dates,
                sleep_timer=self._sleep_timer,
                progress_bar=(
                    progress_bar if progress_bar is not None else self._progress_bar
                ),
                user_subscription=self._fmp_plan,
            )

            if self._use_cached_data:
                cache_model.save_cached_data(
                    cached_data=self._earnings_calendar,
                    cached_data_location=self._cached_data_location,
                    file_name="earnings_calendar.pickle",
                )

        earnings_calendar = self._earnings_calendar.round(
            rounding if rounding else self._rounding
        )

        if self._remove_invalid_tickers:
            self._tickers = [
                ticker
                for ticker in self._tickers
                if ticker not in self._invalid_tickers
            ]

        if len(self._tickers) == 1 and not self._earnings_calendar.empty:
            return earnings_calendar.loc[self._tickers[0]]

        return earnings_calendar

    def get_revenue_geographic_segmentation(
        self, overwrite: bool = False, progress_bar: bool | None = None
    ):
        """
        Obtain revenue by geographic segmentation (e.g. United States, Europe, Asia).

        Note that this information requires a Premium FMP subscription.

        Args:
            overwrite (bool): Defines whether to overwrite the existing data.
            progress_bar (bool, optional): Whether to show a progress bar. Defaults to None.

        Returns:
            pd.DataFrame: The revenue by geographic segmentation for the specified tickers.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "GOOGL", "AMZN"], api_key="FINANCIAL_MODELING_PREP_KEY", start_date="2021-05-01", quarterly=False
        )

        geographic_segmentation = toolkit.get_revenue_geographic_segmentation()

        geographic_segmentation.loc['AAPL']
        ```

        Which returns:

        |              |       2020 |       2021 |       2022 |       2023 |
        |:-------------|-----------:|-----------:|-----------:|-----------:|
        | Americas     | 4.631e+10  | 5.1496e+10 | 4.9278e+10 | 3.5383e+10 |
        | Asia Pacific | 8.225e+09  | 9.81e+09   | 9.535e+09  | 5.63e+09   |
        | China        | 2.1313e+10 | 2.5783e+10 | 2.3905e+10 | 1.5758e+10 |
        | Europe       | 2.7306e+10 | 2.9749e+10 | 2.7681e+10 | 2.0205e+10 |
        | Japan        | 8.285e+09  | 7.107e+09  | 6.755e+09  | 4.821e+09  |

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
            ) = _get_revenue_segmentation(
                tickers=self._tickers,
                method="geographic",
                api_key=self._api_key,
                quarter=self._quarterly if self._fmp_plan == "Premium" else False,
                start_date=self._start_date,
                end_date=self._end_date,
                sleep_timer=self._sleep_timer,
                progress_bar=(
                    progress_bar if progress_bar is not None else self._progress_bar
                ),
                user_subscription=self._fmp_plan,
            )

            if self._use_cached_data:
                cache_model.save_cached_data(
                    cached_data=self._revenue_geographic_segmentation,
                    cached_data_location=self._cached_data_location,
                    file_name="revenue_geographic_segmentation.pickle",
                )

        if self._remove_invalid_tickers:
            self._tickers = [
                ticker
                for ticker in self._tickers
                if ticker not in self._invalid_tickers
            ]

        if len(self._tickers) == 1 and not self._revenue_geographic_segmentation.empty:
            return self._revenue_geographic_segmentation.loc[self._tickers[0]]

        return self._revenue_geographic_segmentation

    def get_revenue_product_segmentation(
        self, overwrite: bool = False, progress_bar: bool | None = None
    ):
        """
        Obtain revenue by product segmentation (e.g. iPad, Advertisement, Windows).

        Note that this information requires a Premium FMP subscription.

        Args:
            overwrite (bool): Defines whether to overwrite the existing data.
            progress_bar (bool, optional): Whether to show a progress bar. Defaults to None.

        Returns:
            pd.DataFrame: The revenue by product segmentation for the specified tickers.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "GOOGL", "AMZN"], api_key="FINANCIAL_MODELING_PREP_KEY", start_date="2021-05-01", quarterly=False
        )

        product_segmentation = toolkit.get_revenue_product_segmentation()

        product_segmentation.loc['MSFT']
        ```

        Which returns:

        |                                    |     2022Q2 |     2022Q3 |     2022Q4 |     2023Q1 |      2023Q2 |
        |:-----------------------------------|-----------:|-----------:|-----------:|-----------:|------------:|
        | Devices                            | 1.581e+09  | 1.448e+09  | 1.43e+09   | 1.282e+09  |  1.361e+09  |
        | Enterprise Services                | 1.902e+09  | 1.876e+09  | 1.862e+09  | 2.007e+09  |  1.977e+09  |
        | Gaming                             | 3.455e+09  | 3.61e+09   | 4.758e+09  | 3.607e+09  |  3.491e+09  |
        | Linked In Corporation              | 3.712e+09  | 3.663e+09  | 3.876e+09  | 3.697e+09  |  3.909e+09  |
        | Office Products And Cloud Services | 1.1639e+10 | 1.1548e+10 | 1.1837e+10 | 1.2438e+10 |  1.2905e+10 |
        | Other Products And Services        | 1.403e+09  | 1.348e+09  | 1.359e+09  | 1.428e+09  | -3.924e+09  |
        | Search And News Advertising        | 2.926e+09  | 2.928e+09  | 3.223e+09  | 3.045e+09  |  3.012e+09  |
        | Server Products And Cloud Services | 1.8839e+10 | 1.8388e+10 | 1.9594e+10 | 2.0025e+10 |  2.1963e+10 |
        | Windows                            | 6.408e+09  | 5.313e+09  | 4.808e+09  | 5.328e+09  |  6.058e+09  |

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
            ) = _get_revenue_segmentation(
                tickers=self._tickers,
                method="product",
                api_key=self._api_key,
                quarter=self._quarterly if self._fmp_plan == "Premium" else False,
                start_date=self._start_date,
                end_date=self._end_date,
                sleep_timer=self._sleep_timer,
                progress_bar=(
                    progress_bar if progress_bar is not None else self._progress_bar
                ),
                user_subscription=self._fmp_plan,
            )

            if self._use_cached_data:
                cache_model.save_cached_data(
                    cached_data=self._revenue_product_segmentation,
                    cached_data_location=self._cached_data_location,
                    file_name="revenue_product_segmentation.pickle",
                )

        if self._remove_invalid_tickers:
            self._tickers = [
                ticker
                for ticker in self._tickers
                if ticker not in self._invalid_tickers
            ]

        if len(self._tickers) == 1 and not self._revenue_product_segmentation.empty:
            return self._revenue_product_segmentation.loc[self._tickers[0]]

        return self._revenue_product_segmentation

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
        progress_bar: bool | None = None,
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
            - Volatility: The volatility for the period.
            - Excess Return: The excess return for the period. This is defined as the return minus
            the a predefined risk free rate. Only calculated when excess_return is True.
            - Excess Volatility: The excess volatility for the period. This is defined as the volatility
            of the excess return. Only calculated when excess_return is True.
            - Cumulative Return: The cumulative return for the period.

        If a benchmark ticker is selected, it also calculates the benchmark ticker together with the results.
        By default this is set to "SPY" (S&P 500 Index) but can be any ticker. This is relevant for calculations
        for models such as CAPM, Alpha and Beta.

        Important to note is that when an api_key is included in the Toolkit initialization that the data
        collection defaults to FinancialModelingPrep which is a more stable source and utilises your subscription.
        However, if this is undesired, it can be disabled by setting enforce_source to "YahooFinance". If
        data collection fails from FinancialModelingPrep it automatically reverts back to YahooFinance.

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
            progress_bar (bool, optional): Whether to show a progress bar. Defaults to None.

        Raises:
            ValueError: If an invalid value is specified for period.

        Returns:
            pandas.DataFrame: The historical data for the specified tickers.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit("AAPL", api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.get_historical_data(period="yearly")
        ```

        Which returns:

        | Date   |     Open |     High |      Low |    Close |   Adj Close |      Volume |   Dividends |     Return |   Volatility |   Excess Return |   Excess Volatility |   Cumulative Return |
        |:-------|---------:|---------:|---------:|---------:|------------:|------------:|------------:|-----------:|-------------:|----------------:|--------------------:|--------------------:|
        | 2013   |  19.7918 |  20.0457 |  19.7857 |  20.0364 |     17.5889 | 2.23084e+08 |    0.108929 |  0         |     0.240641 |       0         |            0.244248 |             1       |
        | 2014   |  28.205  |  28.2825 |  27.5525 |  27.595  |     24.734  | 1.65614e+08 |    0.461429 |  0.406225  |     0.216574 |       0.384525  |            0.219536 |             1.40623 |
        | 2015   |  26.7525 |  26.7575 |  26.205  |  26.315  |     23.9886 | 1.63649e+08 |    0.5075   | -0.0301373 |     0.267373 |      -0.0528273 |            0.269845 |             1.36385 |
        | 2016   |  29.1625 |  29.3    |  28.8575 |  28.955  |     26.9824 | 1.22345e+08 |    0.5575   |  0.124804  |     0.233383 |       0.100344  |            0.240215 |             1.53406 |
        | 2017   |  42.63   |  42.6475 |  42.305  |  42.3075 |     40.0593 | 1.04e+08    |    0.615    |  0.484644  |     0.176058 |       0.460594  |            0.17468  |             2.27753 |
        | 2018   |  39.6325 |  39.84   |  39.12   |  39.435  |     37.9    | 1.40014e+08 |    0.705    | -0.0539019 |     0.287421 |      -0.0807619 |            0.289905 |             2.15477 |
        | 2019   |  72.4825 |  73.42   |  72.38   |  73.4125 |     71.615  | 1.00806e+08 |    0.76     |  0.889578  |     0.261384 |       0.870388  |            0.269945 |             4.0716  |
        | 2020   | 134.08   | 134.74   | 131.72   | 132.69   |    130.559  | 9.91166e+07 |    0.8075   |  0.823067  |     0.466497 |       0.813897  |            0.470743 |             7.4228  |
        | 2021   | 178.09   | 179.23   | 177.26   | 177.57   |    175.795  | 6.40623e+07 |    0.865    |  0.346482  |     0.251019 |       0.331362  |            0.251429 |             9.99467 |
        | 2022   | 128.41   | 129.95   | 127.43   | 129.93   |    129.378  | 7.70342e+07 |    0.91     | -0.264042  |     0.356964 |      -0.302832  |            0.377293 |             7.35566 |
        | 2023   | 187.84   | 188.51   | 187.68   | 188.108  |    188.108  | 4.72009e+06 |    0.71     |  0.453941  |     0.213359 |       0.412901  |            0.22327  |            10.6947  |
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

        if self._daily_historical_data.empty or overwrite:
            self._daily_historical_data, self._invalid_tickers = _get_historical_data(
                tickers=(
                    self._tickers + [self._benchmark_ticker]
                    if self._benchmark_ticker
                    else self._tickers
                ),
                api_key=self._api_key,
                enforce_source=(
                    enforce_source
                    if enforce_source is not None
                    else self._enforce_source
                ),
                start=self._start_date,
                end=self._end_date,
                interval="1d",
                return_column=return_column,
                risk_free_rate=self._daily_risk_free_rate,
                include_dividends=include_dividends,
                progress_bar=(
                    progress_bar if progress_bar is not None else self._progress_bar
                ),
                fill_nan=fill_nan,
                rounding=rounding if rounding else self._rounding,
                sleep_timer=self._sleep_timer,
                show_ticker_seperation=show_ticker_seperation,
                show_errors=True,
            )

            # Change the benchmark ticker name to Benchmark
            if not self._daily_historical_data.empty:
                self._daily_historical_data = self._daily_historical_data.rename(
                    columns={self._benchmark_ticker: "Benchmark"}, level=1
                )

            if self._use_cached_data:
                cache_model.save_cached_data(
                    cached_data=self._daily_historical_data,
                    cached_data_location=self._cached_data_location,
                    file_name="daily_historical_data.pickle",
                )

        if self._remove_invalid_tickers:
            self._tickers = [
                ticker
                for ticker in self._tickers
                if ticker not in self._invalid_tickers
            ]

        if self._daily_historical_data.empty:
            return pd.DataFrame()

        if period == "daily":
            historical_data = self._daily_historical_data.loc[
                self._start_date : self._end_date, :
            ].copy()
            historical_data.loc[historical_data.index[0], "Return"] = 0

            if len(self._tickers) == 1 and not self._benchmark_ticker:
                return historical_data.xs(self._tickers[0], level=1, axis="columns")

            return historical_data

        if period == "weekly":
            if self._weekly_risk_free_rate.empty or overwrite:
                self.get_treasury_data(
                    period="weekly", risk_free_rate=self._risk_free_rate
                )

            self._weekly_historical_data = _convert_daily_to_other_period(
                period="weekly",
                daily_historical_data=self._daily_historical_data,
                start=self._start_date,
                end=self._end_date,
                risk_free_rate=self._weekly_risk_free_rate,
                rounding=rounding if rounding else self._rounding,
            )

            historical_data = self._weekly_historical_data.loc[
                self._start_date : self._end_date, :
            ].copy()
            historical_data.loc[historical_data.index[0], "Return"] = 0

            if len(self._tickers) == 1 and not self._benchmark_ticker:
                return historical_data.xs(self._tickers[0], level=1, axis="columns")

            return historical_data

        if period == "monthly":
            if self._monthly_risk_free_rate.empty or overwrite:
                self.get_treasury_data(
                    period="monthly", risk_free_rate=self._risk_free_rate
                )

            self._monthly_historical_data = _convert_daily_to_other_period(
                period="monthly",
                daily_historical_data=self._daily_historical_data,
                start=self._start_date,
                end=self._end_date,
                risk_free_rate=self._monthly_risk_free_rate,
                rounding=rounding if rounding else self._rounding,
            )

            historical_data = self._monthly_historical_data.loc[
                self._start_date : self._end_date, :
            ].copy()
            historical_data.loc[historical_data.index[0], "Return"] = 0

            if len(self._tickers) == 1 and not self._benchmark_ticker:
                return historical_data.xs(self._tickers[0], level=1, axis="columns")

            return historical_data

        if period == "quarterly":
            if self._quarterly_risk_free_rate.empty or overwrite:
                self.get_treasury_data(
                    period="quarterly", risk_free_rate=self._risk_free_rate
                )

            self._quarterly_historical_data = _convert_daily_to_other_period(
                period="quarterly",
                daily_historical_data=self._daily_historical_data,
                start=self._start_date,
                end=self._end_date,
                risk_free_rate=self._quarterly_risk_free_rate,
                rounding=rounding if rounding else self._rounding,
            )

            historical_data = self._quarterly_historical_data.loc[
                self._start_date : self._end_date, :
            ].copy()
            historical_data.loc[historical_data.index[0], "Return"] = 0

            if len(self._tickers) == 1 and not self._benchmark_ticker:
                return historical_data.xs(self._tickers[0], level=1, axis="columns")

            return historical_data

        if period == "yearly":
            if self._yearly_risk_free_rate.empty or overwrite:
                self.get_treasury_data(
                    period="yearly", risk_free_rate=self._risk_free_rate
                )

            self._yearly_historical_data = _convert_daily_to_other_period(
                period="yearly",
                daily_historical_data=self._daily_historical_data,
                start=self._start_date,
                end=self._end_date,
                risk_free_rate=self._yearly_risk_free_rate,
                rounding=rounding if rounding else self._rounding,
            )

            historical_data = self._yearly_historical_data.loc[
                self._start_date : self._end_date, :
            ].copy()
            historical_data.loc[historical_data.index[0], "Return"] = 0

            if len(self._tickers) == 1 and not self._benchmark_ticker:
                return historical_data.xs(self._tickers[0], level=1, axis="columns")

            return historical_data

        raise ValueError(
            "Please choose from daily, weekly, monthly, quarterly or yearly as period."
        )

    def get_intraday_data(
        self,
        period: str = "1hour",
        return_column: str = "Close",
        fill_nan: bool = True,
        rounding: int | None = None,
        progress_bar: bool | None = None,
    ):
        """
        Returns intraday historical data for the specified tickers. This contains the following columns:
            - Open: The opening price for the period.
            - High: The highest price for the period.
            - Low: The lowest price for the period.
            - Close: The closing price for the period.
            - Volume: The volume for the period.
            - Return: The return for the period.
            - Volatility: The volatility for the period.
            - Cumulative Return: The cumulative return for the period.

        Keep in mind that this data is available for a shorter period. This means that the start date is
        ignored if the difference between the start and end date is bigger than the maximum period.

        If a benchmark ticker is selected, it also calculates the benchmark ticker together with the results.
        By default this is set to "SPY" (S&P 500 Index) but can be any ticker. This is relevant for calculations
        for models such as CAPM, Alpha and Beta.

        Please note that this functionality is only available through Financial Modeling Prep. Therefore, an
        api_key is required to use this functionality.

        Args:
            start (str): The start date for the historical data. Defaults to None.
            end (str): The end date for the historical data. Defaults to None.
            period (str): The interval at which the historical data should be
            returned - daily, weekly, monthly, quarterly, or yearly.
            Defaults to "daily".
            return_column (str): The column to use for the return calculation. Defaults to "Close".
            fill_nan (bool): Defines whether to forward fill NaN values. This defaults
            to True to prevent holes in the dataset. This is especially relevant for
            technical indicators.
            rounding (int): Defines the number of decimal places to round the data to.
            progress_bar (bool, optional): Whether to show a progress bar. Defaults to None.

        Returns:
            pandas.DataFrame: The intraday data for the specified tickers.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit("MSFT", api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.get_intraday_data(period="1min")
        ```

        Which returns:

        | date             |   Open |   High |     Low |   Close |   Volume |   Return |   Volatility |   Cumulative Return |
        |:-----------------|-------:|-------:|--------:|--------:|---------:|---------:|-------------:|--------------------:|
        | 2024-01-19 15:45 | 397.64 | 397.88 | 397.63  | 397.88  |    49202 |   0.0006 |       0.0005 |              1.0266 |
        | 2024-01-19 15:46 | 397.86 | 397.93 | 397.788 | 397.82  |    68913 |  -0.0002 |       0.0005 |              1.0264 |
        | 2024-01-19 15:47 | 397.81 | 397.97 | 397.76  | 397.78  |    62605 |  -0.0001 |       0.0005 |              1.0263 |
        | 2024-01-19 15:48 | 397.78 | 397.85 | 397.675 | 397.845 |    62146 |   0.0002 |       0.0005 |              1.0265 |
        | 2024-01-19 15:49 | 397.85 | 397.97 | 397.8   | 397.94  |    72700 |   0.0002 |       0.0005 |              1.0267 |
        | 2024-01-19 15:50 | 397.92 | 398.27 | 397.9   | 398.04  |   140754 |   0.0003 |       0.0005 |              1.027  |
        | 2024-01-19 15:51 | 398.04 | 398.15 | 397.96  | 398     |   122208 |  -0.0001 |       0.0005 |              1.0269 |
        | 2024-01-19 15:52 | 397.99 | 398.26 | 397.98  | 398.05  |    83546 |   0.0001 |       0.0005 |              1.027  |
        | 2024-01-19 15:53 | 398.04 | 398.12 | 397.98  | 398.09  |    85098 |   0.0001 |       0.0005 |              1.0271 |
        | 2024-01-19 15:54 | 398.1  | 398.52 | 398.03  | 398.45  |   187358 |   0.0009 |       0.0005 |              1.028  |
        | 2024-01-19 15:55 | 398.45 | 398.62 | 398.25  | 398.335 |   237902 |  -0.0003 |       0.0005 |              1.0278 |
        | 2024-01-19 15:56 | 398.33 | 398.44 | 398.3   | 398.415 |   149157 |   0.0002 |       0.0005 |              1.028  |
        | 2024-01-19 15:57 | 398.42 | 398.5  | 398.29  | 398.43  |   181074 |   0      |       0.0005 |              1.028  |
        | 2024-01-19 15:58 | 398.46 | 398.47 | 398.29  | 398.35  |   278802 |  -0.0002 |       0.0005 |              1.0278 |
        | 2024-01-19 15:59 | 398.35 | 398.66 | 398.22  | 398.66  |   586344 |   0.0008 |       0.0005 |              1.0286 |
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
                risk_free_rate=pd.DataFrame(),
                include_dividends=False,
                progress_bar=(
                    progress_bar if progress_bar is not None else self._progress_bar
                ),
                fill_nan=fill_nan,
                rounding=rounding if rounding else self._rounding,
                sleep_timer=self._sleep_timer,
                show_errors=True,
                tqdm_message="Obtaining intraday data",
            )

            if self._use_cached_data:
                cache_model.save_cached_data(
                    cached_data=self._intraday_historical_data,
                    cached_data_location=self._cached_data_location,
                    file_name="intraday_historical_data.pickle",
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
        ].copy()

        historical_data.loc[historical_data.index[0], "Return"] = 0

        if len(self._tickers) == 1 and not self._benchmark_ticker:
            return historical_data.xs(self._tickers[0], level=1, axis="columns")

        return historical_data

    def get_dividend_calendar(
        self,
        overwrite: bool = False,
        rounding: int | None = None,
        progress_bar: bool | None = None,
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

        Args:
            overwrite (bool): Defines whether to overwrite the existing data.
            rounding (int): Defines the number of decimal places to round the data to.
            progress_bar (bool, optional): Whether to show a progress bar. Defaults to None.

        Returns:
            pd.DataFrame: The earnings calendar for the specified tickers.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "GOOGL", "AMZN"], api_key="FINANCIAL_MODELING_PREP_KEY", start_date="2022-08-01", quarterly=False
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
            ) = _get_dividend_calendar(
                tickers=self._tickers,
                api_key=self._api_key,
                start_date=self._start_date,
                end_date=self._end_date,
                sleep_timer=self._sleep_timer,
                progress_bar=(
                    progress_bar if progress_bar is not None else self._progress_bar
                ),
                user_subscription=self._fmp_plan,
            )

            if self._use_cached_data:
                cache_model.save_cached_data(
                    cached_data=self._dividend_calendar,
                    cached_data_location=self._cached_data_location,
                    file_name="dividend_calendar.pickle",
                )

        dividend_calendar = self._dividend_calendar.round(
            rounding if rounding else self._rounding
        )

        if self._remove_invalid_tickers:
            self._tickers = [
                ticker
                for ticker in self._tickers
                if ticker not in self._invalid_tickers
            ]

        if len(self._tickers) == 1 and not self._dividend_calendar.empty:
            return dividend_calendar.loc[self._tickers[0]]

        if dividend_calendar.empty and self._fmp_plan == "Free":
            logger.warning(
                "Dividend data is only available for Premium subscriptions and higher. Get 15% off by using the following link: "
                "https://www.jeroenbouma.com/fmp"
            )

        return dividend_calendar

    def get_esg_scores(
        self,
        overwrite: bool = False,
        rounding: int | None = None,
        progress_bar: bool | None = None,
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

        Args:
            overwrite (bool): Defines whether to overwrite the existing data.
            rounding (int): Defines the number of decimal places to round the data to.

        Returns:
            pd.DataFrame: The ESG scores for the specified tickers.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["MSFT", "TSLA", "AMZN"], api_key="FINANCIAL_MODELING_PREP_KEY", start_date="2022-08-01", quarterly=False
        )

        esg_scores = toolkit.get_esg_scores()

        esg_scores.xs("MSFT", level=1, axis=1)
        ```

        Which returns:

        | date   |   Environmental Score |   Social Score |   Governance Score |   ESG Score |
        |:-------|----------------------:|---------------:|-------------------:|------------:|
        | 2022   |                 73.03 |          58.66 |              61.4  |       64.37 |
        | 2023   |                 72.89 |          58.16 |              60.65 |       63.9  |
        | 2024   |                 72.53 |          58.08 |              60.7  |       63.77 |
        | 2025   |                 71.85 |          57.64 |              59.62 |       63.04 |
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
            ) = _get_esg_scores(
                tickers=self._tickers,
                api_key=self._api_key,
                quarter=self._quarterly,
                start_date=self._start_date,
                end_date=self._end_date,
                sleep_timer=self._sleep_timer,
                progress_bar=(
                    progress_bar if progress_bar is not None else self._progress_bar
                ),
                user_subscription=self._fmp_plan,
            )

        esg_scores = self._esg_scores.round(rounding if rounding else self._rounding)

        if self._remove_invalid_tickers:
            self._tickers = [
                ticker
                for ticker in self._tickers
                if ticker not in self._invalid_tickers
            ]

        if len(self._tickers) == 1 and not self._esg_scores.empty:
            return esg_scores.xs(self._tickers[0], axis=1, level=1)

        return esg_scores

    def get_historical_statistics(self, progress_bar: bool | None = None):
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

        Args:
            progress_bar (bool): Defines whether to show a progress bar. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the statistics for each ticker.

        As an example:

        ```python
        from financetoolkit import Toolkit

        companies = Toolkit(["AMZN", "^HSI", "IWDA.AS", "0P0000Z8RO.T"])

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
        | Regular Market Time    | 2023-09-22       | 2023-09-22     | 2023-09-22       | 2023-09-21     |
        | GMT Offset             | -14400           | 28800          | 7200             | 32400          |
        | Timezone               | EDT              | HKT            | CEST             | JST            |
        | Exchange Timezone Name | America/New_York | Asia/Hong_Kong | Europe/Amsterdam | Asia/Tokyo     |
        """

        if self._historical_statistics.empty:
            self._historical_statistics, _ = _get_historical_statistics(
                tickers=self._tickers,
                api_key=self._api_key if self._api_key is not None else None,
                progress_bar=(
                    progress_bar if progress_bar is not None else self._progress_bar
                ),
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

        Args:
            period (str): The interval at which the treasury data should be returned - daily, weekly, monthly, quarterly, or yearly.
            fill_nan (bool): Defines whether to forward fill NaN values. This defaults
            to True to prevent holes in the dataset. This is especially relevant for
            technical indicators.

        Returns:
            pd.DataFrame: A DataFrame containing the treasury data.

        As an example:

        ```python
        from financetoolkit import Toolkit

        companies = Toolkit(["AAPL", "MSFT"], api_key="FINANCIAL_MODELING_PREP_KEY", start_date="2023-08-10")

        companies.get_treasury_data()
        ```

        Which returns:

        | date       |   13 Week |   5 Year |   10 Year |   30 Year |
        |:-----------|----------:|---------:|----------:|----------:|
        | 2023-10-16 |    0.0533 |   0.0472 |    0.0471 |    0.0487 |
        | 2023-10-17 |    0.0534 |   0.0487 |    0.0485 |    0.0495 |
        | 2023-10-18 |    0.0533 |   0.0492 |    0.049  |    0.05   |
        | 2023-10-19 |    0.0531 |   0.0496 |    0.0499 |    0.051  |
        | 2023-10-20 |    0.053  |   0.0491 |    0.0496 |    0.0512 |

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
            # It collects data in the scenarios where the treasury data is empty or only contains one column which generally
            # means the data was collected for the historical data functionality which only requires a subselection
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
                progress_bar=False,
                divide_ohlc_by=divide_ohlc_by,
                rounding=rounding if rounding else self._rounding,
                show_errors=show_errors,
                fill_nan=fill_nan,
                sleep_timer=self._sleep_timer,
                tqdm_message="Obtaining treasury data",
            )

            if not self._daily_treasury_data.empty:
                self._daily_treasury_data = self._daily_treasury_data.rename(
                    columns=treasury_names, level=1
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
                rounding=rounding if rounding else self._rounding,
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
                rounding=rounding if rounding else self._rounding,
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
                rounding=rounding if rounding else self._rounding,
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
                rounding=rounding if rounding else self._rounding,
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
        progress_bar: bool | None = None,
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

        Args:
            start (str): The start date for the exchange data. Defaults to None.
            end (str): The end date for the exchange data. Defaults to None.
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

        toolkit = Toolkit("ASML", api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.get_exchange_rates(period="monthly")
        ```

        Which returns:

        | Date    |   Open |   High |    Low |   Close |   Adj Close |   Volume |   Return |   Volatility |   Cumulative Return |
        |:--------|-------:|-------:|-------:|--------:|------------:|---------:|---------:|-------------:|--------------------:|
        | 2023-03 | 1.0905 | 1.0926 | 1.0861 |  1.0905 |      1.0905 |        0 |   0.0277 |       0.0298 |              0.7896 |
        | 2023-04 | 1.1011 | 1.1037 | 1.0963 |  1.0969 |      1.0969 |   131812 |   0.0059 |       0.0278 |              0.7943 |
        | 2023-05 | 1.0693 | 1.0771 | 1.066  |  1.076  |      1.0733 |   162069 |  -0.0215 |       0.0143 |              0.7772 |
        | 2023-06 | 1.09   | 1.09   | 1.08   |  1.09   |      1.0868 |        0 |   0.0126 |       0.0179 |              0.787  |
        | 2023-07 | 1.0996 | 1.102  | 1.0952 |  1.1007 |      1.1024 |   183278 |   0.0144 |       0.0225 |              0.7983 |
        | 2023-08 | 1.0842 | 1.0882 | 1.077  |  1.0796 |      1.09   |   171695 |  -0.0112 |       0.0219 |              0.7893 |
        | 2023-09 | 1.06   | 1.06   | 1.06   |  1.06   |      1.06   |        0 |  -0.0275 |       0.0264 |              0.7676 |
        | 2023-10 | 1.0614 | 1.0674 | 1.0556 |  1.0578 |      1.0615 |   184667 |   0.0014 |       0.0232 |              0.7686 |
        | 2023-11 | 1.0973 | 1.0984 | 1.0878 |  1.0892 |      1.0974 |   173646 |   0.0338 |       0.0202 |              0.7946 |
        | 2023-12 | 1.088  | 1.0898 | 1.0848 |  1.0871 |      1.0871 |    90494 |  -0.0094 |       0.0173 |              0.7872 |
        """
        if not self._currencies or overwrite:
            if self._historical_statistics.empty:
                self.get_historical_statistics(
                    progress_bar=(
                        progress_bar if progress_bar is not None else self._progress_bar
                    )
                )
            if self._statistics_statement.empty:
                self.get_statistics_statement(
                    progress_bar=(
                        progress_bar if progress_bar is not None else self._progress_bar
                    )
                )

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

        # Separate currencies that are merely a comparison between the same currency
        # and currencies that are actual exchange rates.
        currencies_to_collect_data_for = [
            currency for currency in self._currencies if currency[:3] != currency[3:6]
        ]
        currencies_to_fill_to_one = [
            currency for currency in self._currencies if currency[:3] == currency[3:6]
        ]

        if self._daily_exchange_rate_data.empty or overwrite:
            if currencies_to_collect_data_for:
                self._daily_exchange_rate_data, _ = _get_historical_data(
                    tickers=currencies_to_collect_data_for,
                    api_key=self._api_key,
                    enforce_source=self._enforce_source,
                    start=self._start_date,
                    end=self._end_date,
                    interval="1d",
                    return_column=return_column,
                    risk_free_rate=pd.DataFrame(),
                    include_dividends=False,
                    progress_bar=(
                        progress_bar if progress_bar is not None else self._progress_bar
                    ),
                    fill_nan=fill_nan,
                    rounding=rounding if rounding else self._rounding,
                    sleep_timer=self._sleep_timer,
                    show_ticker_seperation=show_ticker_seperation,
                    tqdm_message="Obtaining exchange data",
                )
            else:
                # In case there is no conversion needed, it should create a placeholder
                # DataFrame that works with the rest of the Toolkit.
                self._daily_exchange_rate_data = pd.DataFrame(
                    data=1,
                    index=pd.PeriodIndex(
                        pd.date_range(
                            start=self._start_date, end=self._end_date, freq="D"
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
                            ("Volatility", "USDUSD=X"),
                            ("Cumulative Return", "USDUSD=X"),
                        ]
                    ),
                )

            # For exchange data, it is possible that a ticker such as USDUSD=X
            # exists which should always be 1. This data is added here.
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
            ].copy()
            historical_data.loc[historical_data.index[0], "Return"] = 0

            if len(self._currencies) == 1:
                return historical_data.xs(self._currencies[0], level=1, axis="columns")

            return historical_data

        if period == "weekly":
            self._weekly_exchange_rate_data = _convert_daily_to_other_period(
                period="weekly",
                daily_historical_data=self._daily_exchange_rate_data,
                start=self._start_date,
                end=self._end_date,
                risk_free_rate=pd.DataFrame(),
                rounding=rounding if rounding else self._rounding,
            )

            historical_data = self._weekly_exchange_rate_data.loc[
                self._start_date : self._end_date, :
            ].copy()
            historical_data.loc[historical_data.index[0], "Return"] = 0

            if len(self._currencies) == 1:
                return historical_data.xs(self._currencies[0], level=1, axis="columns")

            return historical_data

        if period == "monthly":
            self._monthly_exchange_rate_data = _convert_daily_to_other_period(
                period="monthly",
                daily_historical_data=self._daily_exchange_rate_data,
                start=self._start_date,
                end=self._end_date,
                risk_free_rate=pd.DataFrame(),
                rounding=rounding if rounding else self._rounding,
            )

            historical_data = self._monthly_exchange_rate_data.loc[
                self._start_date : self._end_date, :
            ].copy()
            historical_data.loc[historical_data.index[0], "Return"] = 0

            if len(self._currencies) == 1:
                return historical_data.xs(self._currencies[0], level=1, axis="columns")

            return historical_data

        if period == "quarterly":
            self._quarterly_exchange_rate_data = _convert_daily_to_other_period(
                period="quarterly",
                daily_historical_data=self._daily_exchange_rate_data,
                start=self._start_date,
                end=self._end_date,
                risk_free_rate=pd.DataFrame(),
                rounding=rounding if rounding else self._rounding,
            )

            historical_data = self._quarterly_exchange_rate_data.loc[
                self._start_date : self._end_date, :
            ].copy()
            historical_data.loc[historical_data.index[0], "Return"] = 0

            if len(self._currencies) == 1:
                return historical_data.xs(self._currencies[0], level=1, axis="columns")

            return historical_data

        if period == "yearly":
            self._yearly_exchange_rate_data = _convert_daily_to_other_period(
                period="yearly",
                daily_historical_data=self._daily_exchange_rate_data,
                start=self._start_date,
                end=self._end_date,
                risk_free_rate=pd.DataFrame(),
                rounding=rounding if rounding else self._rounding,
            )

            historical_data = self._yearly_exchange_rate_data.loc[
                self._start_date : self._end_date, :
            ].copy()
            historical_data.loc[historical_data.index[0], "Return"] = 0

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
        progress_bar: bool | None = None,
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

        Args:
            enforce_source (bool): Defines whether to enforce the source of the data. This can be
                either "FinancialModelingPrep" or "YahooFinance". Defaults to None.
            overwrite (bool): Defines whether to overwrite the existing data.
            rounding (int): Defines the number of decimal places to round the data to.
            growth (bool): Defines whether to return the growth of the data.
            lag (int | str): Defines the number of periods to lag the growth data by.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.
            progress_bar (bool): Defines whether to show a progress bar.

        Returns:
            pd.DataFrame: A pandas DataFrame with the retrieved balance sheet statement data.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["MSFT", "MU"], api_key="FINANCIAL_MODELING_PREP_KEY", quarterly=True, start_date='2022-05-01')

        balance_sheet_statements = toolkit.get_balance_sheet_statement()

        balance_sheet_statements.loc['MU']
        ```

        Which returns:

        |                                          |      2022Q2 |      2022Q3 |      2022Q4 |      2023Q1 |      2023Q2 |
        |:-----------------------------------------|------------:|------------:|------------:|------------:|------------:|
        | Cash and Cash Equivalents                |  9.157e+09  |  8.262e+09  |  9.574e+09  |  9.798e+09  |  9.298e+09  |
        | Short Term Investments                   |  1.07e+09   |  1.069e+09  |  1.007e+09  |  1.02e+09   |  1.054e+09  |
        | Cash and Short Term Investments          |  1.0227e+10 |  9.331e+09  |  1.0581e+10 |  1.0818e+10 |  1.0352e+10 |
        | Accounts Receivable                      |  6.229e+09  |  5.13e+09   |  3.318e+09  |  2.278e+09  |  2.429e+09  |
        | Inventory                                |  5.629e+09  |  6.663e+09  |  8.359e+09  |  8.129e+09  |  8.238e+09  |
        | Other Current Assets                     |  6.08e+08   |  6.44e+08   |  6.63e+08   |  6.73e+08   |  7.15e+08   |
        | Total Current Assets                     |  2.2708e+10 |  2.1781e+10 |  2.2921e+10 |  2.1898e+10 |  2.1734e+10 |
        | Property, Plant and Equipment            |  3.7355e+10 |  3.9227e+10 |  4.0028e+10 |  3.9758e+10 |  3.9382e+10 |
        | Goodwill                                 |  1.228e+09  |  1.228e+09  |  1.228e+09  |  1.228e+09  |  1.252e+09  |
        | Intangible Assets                        |  4.15e+08   |  4.21e+08   |  4.28e+08   |  4.1e+08    |  4.1e+08    |
        | Long Term Investments                    |  1.646e+09  |  1.647e+09  |  1.426e+09  |  1.212e+09  |  9.73e+08   |
        | Tax Assets                               |  6.82e+08   |  7.02e+08   |  6.72e+08   |  6.97e+08   |  7.08e+08   |
        | Other Fixed Assets                       |  1.262e+09  |  1.277e+09  |  1.171e+09  |  1.317e+09  |  1.221e+09  |
        | Fixed Assets                             |  4.2588e+10 |  4.4502e+10 |  4.4953e+10 |  4.4622e+10 |  4.3946e+10 |
        | Other Assets                             |  0          |  0          |  0          |  0          |  0          |
        | Total Assets                             |  6.5296e+10 |  6.6283e+10 |  6.7874e+10 |  6.652e+10  |  6.568e+10  |
        | Accounts Payable                         |  2.019e+09  |  2.142e+09  |  1.789e+09  |  1.689e+09  |  1.64e+09   |
        | Short Term Debt                          |  1.07e+08   |  1.03e+08   |  1.71e+08   |  2.37e+08   |  2.59e+08   |
        | Tax Payables                             |  3.82e+08   |  4.2e+08    |  4.19e+08   |  2.41e+08   |  1.48e+08   |
        | Deferred Revenue                         |  0          |  0          |  0          |  0          | -1.64e+09   |
        | Other Current Liabilities                |  4.883e+09  |  5.294e+09  |  4.565e+09  |  3.329e+09  |  4.845e+09  |
        | Total Current Liabilities                |  7.009e+09  |  7.539e+09  |  6.525e+09  |  5.255e+09  |  5.104e+09  |
        | Long Term Debt                           |  7.485e+09  |  7.413e+09  |  1.0719e+10 |  1.2647e+10 |  1.3589e+10 |
        | Deferred Revenue Non Current             |  6.63e+08   |  5.89e+08   |  5.16e+08   |  5.29e+08   |  6.32e+08   |
        | Deferred Tax Liabilities                 |  0          |  0          |  0          |  0          |  0          |
        | Other Non Current Liabilities            |  8.58e+08   |  8.35e+08   |  8.08e+08   |  8.32e+08   |  9.5e+08    |
        | Total Non Current Liabilities            |  9.006e+09  |  8.837e+09  |  1.2043e+10 |  1.4008e+10 |  1.5171e+10 |
        | Other Liabilities                        |  0          |  0          |  0          |  0          |  0          |
        | Capital Lease Obligations                |  6.29e+08   |  6.1e+08    |  6.25e+08   |  6.1e+08    |  6.03e+08   |
        | Total Liabilities                        |  1.6015e+10 |  1.6376e+10 |  1.8568e+10 |  1.9263e+10 |  2.0275e+10 |
        | Preferred Stock                          |  0          |  0          |  0          |  0          |  0          |
        | Common Stock                             |  1.22e+08   |  1.23e+08   |  1.23e+08   |  1.23e+08   |  1.24e+08   |
        | Retained Earnings                        |  4.5916e+10 |  4.7274e+10 |  4.6873e+10 |  4.4426e+10 |  4.2391e+10 |
        | Accumulated Other Comprehensive Income   | -3.64e+08   | -5.6e+08    | -4.73e+08   | -3.73e+08   | -3.4e+08    |
        | Other Total Shareholder Equity           |  3.607e+09  |  3.07e+09   |  2.783e+09  |  3.081e+09  |  3.23e+09   |
        | Total Shareholder Equity                 |  4.9281e+10 |  4.9907e+10 |  4.9306e+10 |  4.7257e+10 |  4.5405e+10 |
        | Total Equity                             |  4.9281e+10 |  4.9907e+10 |  4.9306e+10 |  4.7257e+10 |  4.5405e+10 |
        | Total Liabilities and Shareholder Equity |  6.5296e+10 |  6.6283e+10 |  6.7874e+10 |  6.652e+10  |  6.568e+10  |
        | Minority Interest                        |  0          |  0          |  0          |  0          |  0          |
        | Total Liabilities and Equity             |  6.5296e+10 |  6.6283e+10 |  6.7874e+10 |  6.652e+10  |  6.568e+10  |
        | Total Investments                        |  2.716e+09  |  2.716e+09  |  2.433e+09  |  2.232e+09  |  2.027e+09  |
        | Total Debt                               |  7.592e+09  |  7.516e+09  |  1.089e+10  |  1.2884e+10 |  1.3848e+10 |
        | Net Debt                                 | -1.565e+09  | -7.46e+08   |  1.316e+09  |  3.086e+09  |  4.55e+09   |
        """
        convert_currency = bool(
            self._convert_currency
            and (self._balance_sheet_statement.empty or overwrite)
        )

        if (
            not self._api_key
            and self._balance_sheet_statement.empty
            and self._enforce_source == "FinancialModelingPrep"
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

        if enforce_source is not None and enforce_source not in [
            "FinancialModelingPrep",
            "YahooFinance",
        ]:
            raise ValueError(
                "The enforce_source parameter must be either 'FinancialModelingPrep' or 'YahooFinance'."
            )

        # Correct for the case where a Portfolio ticker exists
        ticker_list = [ticker for ticker in self._tickers if ticker != "Portfolio"]

        if self._balance_sheet_statement.empty or overwrite:
            (
                self._balance_sheet_statement,
                self._statistics_statement,
                self._invalid_tickers,
            ) = collect_financial_statements(
                tickers=ticker_list,
                statement="balance",
                api_key=self._api_key,
                quarter=self._quarterly,
                start_date=self._start_date,
                end_date=self._end_date,
                rounding=rounding if rounding else self._rounding,
                fmp_statement_format=self._fmp_balance_sheet_statement_generic,
                fmp_statistics_format=self._fmp_statistics_statement_generic,
                yf_statement_format=self._yf_balance_sheet_statement_generic,
                sleep_timer=self._sleep_timer,
                progress_bar=(
                    progress_bar if progress_bar is not None else self._progress_bar
                ),
                user_subscription=self._fmp_plan,
                enforce_source=(
                    enforce_source
                    if enforce_source is not None
                    else self._enforce_source
                ),
            )

            if convert_currency:
                self.get_exchange_rates(
                    period="quarterly" if self._quarterly else "yearly",
                    progress_bar=(
                        progress_bar if progress_bar is not None else self._progress_bar
                    ),
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

            if self._use_cached_data:
                cache_model.save_cached_data(
                    cached_data=self._balance_sheet_statement,
                    cached_data_location=self._cached_data_location,
                    file_name="balance_sheet_statement.pickle",
                )

        if self._remove_invalid_tickers:
            self._tickers = [
                ticker
                for ticker in self._tickers
                if ticker not in self._invalid_tickers
            ]

        balance_sheet_statement = self._balance_sheet_statement

        if growth:
            self._balance_sheet_statement_growth = helpers.calculate_growth(
                balance_sheet_statement,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="columns",
            )

        balance_sheet_statement = balance_sheet_statement.round(
            rounding if rounding else self._rounding
        )

        if len(self._tickers) == 1 and not self._balance_sheet_statement.empty:
            return (
                self._balance_sheet_statement_growth.loc[self._tickers[0]]
                if growth
                else balance_sheet_statement.loc[self._tickers[0]]
            )

        return (
            self._balance_sheet_statement_growth if growth else balance_sheet_statement
        )

    def get_income_statement(
        self,
        enforce_source: str | None = None,
        overwrite: bool = False,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
        progress_bar: bool | None = None,
    ):
        """
        Retrieves the income statement data for the specified tickers. The income statement is a financial
        statement that shows a company's revenues and expenses over a specific period. It is used to calculate
        a company's net income.

        The income statement is a financial statement that shows a company's revenues and expenses over a specific
        period. Therefore, trailing results are available for this statement.

        Args:
            enforce_source (bool): Defines whether to enforce the source of the data. This can be
                either "FinancialModelingPrep" or "YahooFinance". Defaults to None.
            overwrite (bool): Defines whether to overwrite the existing data.
            rounding (int): Defines the number of decimal places to round the data to.
            growth (bool): Defines whether to return the growth of the data.
            lag (int | str): Defines the number of periods to lag the growth data by.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.
            progress_bar (bool): Defines whether to show a progress bar.

        Returns:
            pd.DataFrame: A pandas DataFrame with the retrieved income statement data.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["TSLA", "MU"], api_key="FINANCIAL_MODELING_PREP_KEY", quarterly=True, start_date='2022-05-01')

        income_sheet_statements = toolkit.get_income_statement()

        income_sheet_statements.loc['TSLA']
        ```

        Which returns:

        |                                              |     2022Q2 |      2022Q3 |      2022Q4 |      2023Q1 |     2023Q2 |
        |:---------------------------------------------|-----------:|------------:|------------:|------------:|-----------:|
        | Revenue                                      | 1.6934e+10 |  2.1454e+10 |  2.4318e+10 |  2.3329e+10 | 2.4927e+10 |
        | Cost of Goods Sold                           | 1.27e+10   |  1.6072e+10 |  1.8541e+10 |  1.8818e+10 | 2.0394e+10 |
        | Gross Profit                                 | 4.234e+09  |  5.382e+09  |  5.777e+09  |  4.511e+09  | 4.533e+09  |
        | Gross Profit Ratio                           | 0.25003    |  0.250862   |  0.237561   |  0.193364   | 0.181851   |
        | Research and Development Expenses            | 6.67e+08   |  7.33e+08   |  8.1e+08    |  7.71e+08   | 9.43e+08   |
        | General and Administrative Expenses          | 0          |  0          |  0          |  0          | 0          |
        | Selling and Marketing Expenses               | 0          |  0          |  0          |  0          | 0          |
        | Selling, General and Administrative Expenses | 9.61e+08   |  9.61e+08   |  1.032e+09  |  1.076e+09  | 1.191e+09  |
        | Other Expenses                               | 2.8e+07    | -8.5e+07    | -4.2e+07    | -4.8e+07    | 3.28e+08   |
        | Operating Expenses                           | 1.628e+09  |  1.694e+09  |  1.842e+09  |  1.847e+09  | 2.134e+09  |
        | Cost and Expenses                            | 1.4328e+10 |  1.7766e+10 |  2.0383e+10 |  2.0665e+10 | 2.2528e+10 |
        | Interest Income                              | 2.6e+07    |  8.6e+07    |  1.57e+08   |  2.13e+08   | 2.38e+08   |
        | Interest Expense                             | 4.4e+07    |  5.3e+07    |  3.3e+07    |  2.9e+07    | 2.8e+07    |
        | Depreciation and Amortization                | 1.118e+09  |  9.57e+08   |  1.138e+09  |  1.211e+09  | 1.72e+09   |
        | EBITDA                                       | 3.582e+09  |  4.645e+09  |  5.039e+09  |  3.875e+09  | 4.119e+09  |
        | EBITDA Ratio                                 | 0.211527   |  0.21651    |  0.207213   |  0.166102   | 0.165243   |
        | Operating Income                             | 2.464e+09  |  3.688e+09  |  3.901e+09  |  2.664e+09  | 2.399e+09  |
        | Operating Income Ratio                       | 0.145506   |  0.171903   |  0.160416   |  0.114193   | 0.096241   |
        | Total Other Income                           | 1e+07      | -5.2e+07    |  8.2e+07    |  1.36e+08   | 5.38e+08   |
        | Income Before Tax                            | 2.474e+09  |  3.636e+09  |  3.983e+09  |  2.8e+09    | 2.937e+09  |
        | Income Before Tax Ratio                      | 0.146097   |  0.169479   |  0.163788   |  0.120022   | 0.117824   |
        | Income Tax Expense                           | 2.05e+08   |  3.05e+08   |  2.76e+08   |  2.61e+08   | 3.23e+08   |
        | Net Income                                   | 2.259e+09  |  3.292e+09  |  3.687e+09  |  2.513e+09  | 2.703e+09  |
        | Net Income Ratio                             | 0.1334     |  0.153445   |  0.151616   |  0.10772    | 0.108437   |
        | EPS                                          | 0.73       |  1.05       |  1.18       |  0.8        | 0.85       |
        | EPS Diluted                                  | 0.65       |  0.95       |  1.07       |  0.73       | 0.78       |
        | Weighted Average Shares                      | 3.111e+09  |  3.146e+09  |  3.16e+09   |  3.166e+09  | 3.171e+09  |
        | Weighted Average Shares Diluted              | 3.465e+09  |  3.468e+09  |  3.471e+09  |  3.468e+09  | 3.478e+09  |
        """
        convert_currency = bool(
            self._convert_currency and (self._income_statement.empty or overwrite)
        )

        if (
            not self._api_key
            and self._income_statement.empty
            and self._enforce_source == "FinancialModelingPrep"
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

        if enforce_source is not None and enforce_source not in [
            "FinancialModelingPrep",
            "YahooFinance",
        ]:
            raise ValueError(
                "The enforce_source parameter must be either 'FinancialModelingPrep' or 'YahooFinance'."
            )

        # Correct for the case where a Portfolio ticker exists
        ticker_list = [ticker for ticker in self._tickers if ticker != "Portfolio"]

        if self._income_statement.empty or overwrite:
            (
                self._income_statement,
                self._statistics_statement,
                self._invalid_tickers,
            ) = collect_financial_statements(
                tickers=ticker_list,
                statement="income",
                api_key=self._api_key,
                quarter=self._quarterly,
                start_date=self._start_date,
                end_date=self._end_date,
                rounding=rounding if rounding else self._rounding,
                fmp_statement_format=self._fmp_income_statement_generic,
                fmp_statistics_format=self._fmp_statistics_statement_generic,
                yf_statement_format=self._yf_income_statement_generic,
                sleep_timer=self._sleep_timer,
                progress_bar=(
                    progress_bar if progress_bar is not None else self._progress_bar
                ),
                user_subscription=self._fmp_plan,
                enforce_source=(
                    enforce_source
                    if enforce_source is not None
                    else self._enforce_source
                ),
            )

            if convert_currency:
                self.get_exchange_rates(
                    period="quarterly" if self._quarterly else "yearly",
                    progress_bar=(
                        progress_bar if progress_bar is not None else self._progress_bar
                    ),
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

            if self._use_cached_data:
                cache_model.save_cached_data(
                    cached_data=self._income_statement,
                    cached_data_location=self._cached_data_location,
                    file_name="income_statement.pickle",
                )

        if self._remove_invalid_tickers:
            self._tickers = [
                ticker
                for ticker in self._tickers
                if ticker not in self._invalid_tickers
            ]

        income_statement = self._income_statement

        if trailing:
            # This is a special case where the trailing period does not make sense
            # for the Weighted Average Shares and Weighted Average Shares Diluted.
            weighted_average_shares = income_statement.loc[
                :, ["Weighted Average Shares", "Weighted Average Shares Diluted"], :
            ]

            # The rolling window is calculated for the rest of the income statement.
            income_statement = self._income_statement.T.rolling(trailing).sum().T

            # The Weighted Average Shares and Weighted Average Shares Diluted should
            # not be summed up but rather kept equal to the current value.
            income_statement.loc[weighted_average_shares.index] = (
                weighted_average_shares
            )

        if growth:
            self._income_statement_growth = helpers.calculate_growth(
                income_statement,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="columns",
            )

        income_statement = income_statement.round(
            rounding if rounding else self._rounding
        )

        if len(self._tickers) == 1 and not self._income_statement.empty:
            return (
                self._income_statement_growth.loc[self._tickers[0]]
                if growth
                else income_statement.loc[self._tickers[0]]
            )

        return self._income_statement_growth if growth else income_statement

    def get_cash_flow_statement(
        self,
        enforce_source: str | None = None,
        overwrite: bool = False,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        trailing: int | None = None,
        progress_bar: bool | None = None,
    ):
        """
        Retrieves the cash flow statement data for the specified tickers. The cash flow statement is a financial
        statement that shows how changes in balance sheet accounts and income affect cash and cash equivalents.
        It breaks the analysis down to operating, investing and financing activities.

        The cash flow statement is a financial statement that shows how changes in balance sheet accounts and income
        affect cash and cash equivalents. Therefore, trailing results are available for this statement.

        Args:
            enforce_source (bool): Defines whether to enforce the source of the data. This can be
            overwrite (bool): Defines whether to overwrite the existing data.
            rounding (int): Defines the number of decimal places to round the data to.
            growth (bool): Defines whether to return the growth of the data.
            lag (int | str): Defines the number of periods to lag the growth data by.
            trailing (int): Defines whether to select a trailing period.
            E.g. when selecting 4 with quarterly data, the TTM is calculated.
            progress_bar (bool): Defines whether to show a progress bar.

        Returns:
            pd.DataFrame: A pandas DataFrame with the retrieved cash flow statement data.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["MU", "AMZN"], api_key="FINANCIAL_MODELING_PREP_KEY", quarterly=True, start_date='2022-09-01')

        cash_flow_statements = toolkit.get_cash_flow_statement()

        cash_flow_statements.loc['AMZN']
        ```

        Which returns:

        |                               |      2022Q3 |      2022Q4 |      2023Q1 |      2023Q2 |
        |:------------------------------|------------:|------------:|------------:|------------:|
        | Net Income                    |  2.872e+09  |  2.78e+08   |  3.172e+09  |  6.75e+09   |
        | Depreciation and Amortization |  1.0204e+10 |  1.2685e+10 |  1.1123e+10 |  1.1589e+10 |
        | Deferred Income Tax           | -8.25e+08   | -3.367e+09  | -4.72e+08   | -2.744e+09  |
        | Stock Based Compensation      |  5.556e+09  |  5.606e+09  |  4.748e+09  |  7.127e+09  |
        | Change in Working Capital     | -5.254e+09  |  1.0526e+10 | -1.4317e+10 | -6.293e+09  |
        | Accounts Receivables          | -4.794e+09  | -8.788e+09  |  1.521e+09  | -5.167e+09  |
        | Inventory                     |  7.32e+08   |  3.18e+09   |  3.71e+08   | -2.373e+09  |
        | Accounts Payables             | -1.226e+09  |  9.852e+09  | -1.1264e+10 |  3.029e+09  |
        | Other Working Capital         |  3.4e+07    |  6.282e+09  | -4.945e+09  | -1.782e+09  |
        | Other Non Cash Items          | -1.149e+09  |  3.445e+09  |  5.34e+08   |  4.7e+07    |
        | Cash Flow from Operations     |  1.1404e+10 |  2.9173e+10 |  4.788e+09  |  1.6476e+10 |
        | Property, Plant and Equipment | -1.6378e+10 | -1.6592e+10 | -1.4207e+10 | -1.1455e+10 |
        | Acquisitions                  | -8.85e+08   | -8.31e+08   | -3.513e+09  | -3.16e+08   |
        | Purchases of Investments      | -2.39e+08   | -2.33e+08   | -3.38e+08   | -4.96e+08   |
        | Sales of Investments          |  5.57e+08   |  5.683e+09  |  1.115e+09  |  1.551e+09  |
        | Other Investing Activities    |  1.337e+09  |  1.152e+09  |  1.137e+09  |  1.043e+09  |
        | Cash Flow from Investing      | -1.5608e+10 | -1.0821e+10 | -1.5806e+10 | -9.673e+09  |
        | Debt Repayment                | -9.429e+09  | -1.8756e+10 | -6.369e+09  | -1.0861e+10 |
        | Common Stock Issued           |  0          |  0          |  0          |  0          |
        | Common Stock Purchased        |  0          |  6e+09      |  0          |  0          |
        | Dividends Paid                |  0          |  0          |  0          |  0          |
        | Other Financing Activities    |  1.2445e+10 |  1.2842e+10 |  1.2723e+10 |  4.322e+09  |
        | Cash Flow from Financing      |  3.016e+09  |  8.6e+07    |  6.354e+09  | -6.539e+09  |
        | Forex Changes on Cash         | -1.334e+09  |  6.37e+08   |  1.45e+08   |  6.9e+07    |
        | Net Change in Cash            | -2.522e+09  |  1.9075e+10 | -4.519e+09  |  3.33e+08   |
        | Cash End of Period            |  3.5178e+10 |  5.4253e+10 |  4.9734e+10 |  5.0067e+10 |
        | Cash Beginning of Period      |  3.77e+10   |  3.5178e+10 |  5.4253e+10 |  4.9734e+10 |
        | Operating Cash Flow           |  1.1404e+10 |  2.9173e+10 |  4.788e+09  |  1.6476e+10 |
        | Capital Expenditure           | -1.6378e+10 | -1.6592e+10 | -1.4207e+10 | -1.1455e+10 |
        | Free Cash Flow                | -4.974e+09  |  1.2581e+10 | -9.419e+09  |  5.021e+09  |
        """
        convert_currency = bool(
            self._convert_currency and (self._cash_flow_statement.empty or overwrite)
        )

        if (
            not self._api_key
            and self._cash_flow_statement.empty
            and self._enforce_source == "FinancialModelingPrep"
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

        if enforce_source is not None and enforce_source not in [
            "FinancialModelingPrep",
            "YahooFinance",
        ]:
            raise ValueError(
                "The enforce_source parameter must be either 'FinancialModelingPrep' or 'YahooFinance'."
            )

        # Correct for the case where a Portfolio ticker exists
        ticker_list = [ticker for ticker in self._tickers if ticker != "Portfolio"]

        if self._cash_flow_statement.empty or overwrite:
            (
                self._cash_flow_statement,
                self._statistics_statement,
                self._invalid_tickers,
            ) = collect_financial_statements(
                tickers=ticker_list,
                statement="cashflow",
                api_key=self._api_key,
                quarter=self._quarterly,
                start_date=self._start_date,
                end_date=self._end_date,
                rounding=rounding if rounding else self._rounding,
                fmp_statement_format=self._fmp_cash_flow_statement_generic,
                fmp_statistics_format=self._fmp_statistics_statement_generic,
                yf_statement_format=self._yf_cash_flow_statement_generic,
                sleep_timer=self._sleep_timer,
                progress_bar=(
                    progress_bar if progress_bar is not None else self._progress_bar
                ),
                user_subscription=self._fmp_plan,
                enforce_source=(
                    enforce_source
                    if enforce_source is not None
                    else self._enforce_source
                ),
            )

            if convert_currency:
                self.get_exchange_rates(
                    period="quarterly" if self._quarterly else "yearly",
                    progress_bar=(
                        progress_bar if progress_bar is not None else self._progress_bar
                    ),
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

            if self._use_cached_data:
                cache_model.save_cached_data(
                    cached_data=self._cash_flow_statement,
                    cached_data_location=self._cached_data_location,
                    file_name="cash_flow_statement.pickle",
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
            self._cash_flow_statement_growth = helpers.calculate_growth(
                cash_flow_statement,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="columns",
            )

        cash_flow_statement = cash_flow_statement.round(
            rounding if rounding else self._rounding
        )

        if len(self._tickers) == 1 and not self._cash_flow_statement.empty:
            return (
                self._cash_flow_statement_growth.loc[self._tickers[0]]
                if growth
                else cash_flow_statement.loc[self._tickers[0]]
            )
        return self._cash_flow_statement_growth if growth else cash_flow_statement

    def get_statistics_statement(
        self,
        enforce_source: str | None = None,
        overwrite: bool = False,
        progress_bar: bool | None = None,
        rounding: int | None = None,
    ):
        """
        Retrieves the balance, cash and income statistics for the company(s) from the specified source.

        Note that this also obtains the balance sheet statement at the same time given that it's the same
        API call. This is done to reduce the number of API calls to FinancialModelingPrep.

        Args:
            enforce_source (bool): Defines whether to enforce the source of the data. This can be
                either "FinancialModelingPrep" or "YahooFinance". Defaults to None.
            overwrite (bool): Defines whether to overwrite the existing data.
            progress_bar (bool): Defines whether to show a progress bar.
            rounding (int): Defines the number of decimal places to round the data to.

        Returns:
            pd.DataFrame: A pandas DataFrame with the retrieved statistics statement data.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit("TSLA", api_key="FINANCIAL_MODELING_PREP_KEY", quarterly=True, start_date='2023-05-01')

        toolkit.get_statistics_statement()
        ```

        Which returns:

        |                   | 2023Q2                                                                                            |
        |:------------------|:--------------------------------------------------------------------------------------------------|
        | Reported Currency | USD                                                                                               |
        | CIK ID            | 1318605                                                                                           |
        | Filling Date      | 2023-07-24                                                                                        |
        | Accepted Date     | 2023-07-21 18:08:29                                                                               |
        | Calendar Year     | 2023                                                                                              |
        | Period            | Q2                                                                                                |
        | SEC Link          | https://www.sec.gov/Archives/edgar/data/1318605/000095017023033872/0000950170-23-033872-index.htm |
        | Document Link     | https://www.sec.gov/Archives/edgar/data/1318605/000095017023033872/tsla-20230630.htm              |

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
            ) = collect_financial_statements(
                tickers=ticker_list,
                statement="balance",
                api_key=self._api_key,
                quarter=self._quarterly,
                start_date=self._start_date,
                end_date=self._end_date,
                rounding=rounding if rounding else self._rounding,
                fmp_statement_format=self._fmp_balance_sheet_statement_generic,
                fmp_statistics_format=self._fmp_statistics_statement_generic,
                yf_statement_format=self._yf_balance_sheet_statement_generic,
                sleep_timer=self._sleep_timer,
                progress_bar=(
                    progress_bar if progress_bar is not None else self._progress_bar
                ),
                user_subscription=self._fmp_plan,
                enforce_source=(
                    self._enforce_source
                    if enforce_source is not None
                    else self._enforce_source
                ),
            )

            if self._use_cached_data:
                cache_model.save_cached_data(
                    cached_data=self._statistics_statement,
                    cached_data_location=self._cached_data_location,
                    file_name="statistics_statement.pickle",
                )

        if self._remove_invalid_tickers:
            self._tickers = [
                ticker
                for ticker in self._tickers
                if ticker not in self._invalid_tickers
            ]

        if len(self._tickers) == 1 and not self._statistics_statement.empty:
            return self._statistics_statement.loc[self._tickers[0]]

        return self._statistics_statement

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

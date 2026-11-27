"""Performance Module"""

__docformat__ = "google"

import warnings
from collections.abc import Callable
from functools import cached_property
from typing import Literal

import numpy as np
import pandas as pd

from financetoolkit.helpers import LazyDict, handle_portfolio
from financetoolkit.performance import performance_model
from financetoolkit.performance.helpers import determine_within_dataset
from financetoolkit.risk.helpers import (
    check_rolling,
    determine_within_historical_data,
    get_days_per_period,
    get_observations_per_year,
    get_risk_free_rate_per_date,
    get_risk_free_rate_per_ticker,
)
from financetoolkit.risk.risk_model import (
    get_conditional_drawdown_at_risk,
    get_downside_deviation,
    get_kurtosis,
    get_max_drawdown,
    get_pain_index,
    get_skewness,
    get_ui,
    get_volatility,
)
from financetoolkit.utilities import factor_model
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import (
    InvalidArgumentError,
    check_option,
    check_period,
    handle_errors,
    validate_alpha,
)
from financetoolkit.utilities.logger_model import get_logger, log_each_message_once
from financetoolkit.utilities.sources_model import track_sources
from financetoolkit.utilities.statistics_model import (
    PERIOD_TRANSLATION,
    PERIODS_PER_YEAR,
    apply_rounding,
    convert_annualized_rate_to_period,
    finalize_dataset,
    to_multi_index,
    to_period_index,
    validate_period,
)

# The examples' output tables are wider than the line length.
# ruff: noqa: E501

logger = get_logger()

# The fewest days a bull or bear regression is fitted on, about a trading month: the Alpha is
# compounded over a year, so that of a handful of days (a week) is amplified to extremes.
MINIMUM_BULL_BEAR_OBSERVATIONS = 20

# The fewest returns a Covariance or Correlation Matrix is calculated from without a warning.
MINIMUM_MATRIX_OBSERVATIONS = 3


# The performance metrics that get_factor_analysis compares across tickers with "all", each
# measured within every evaluation period: the return itself (momentum and reversal), the
# market sensitivity (the low-beta anomaly) and the risk-adjusted returns.
FACTOR_COLLECTIONS = {
    "all": [
        "get_returns",
        "get_excess_return",
        "get_beta",
        "get_sharpe_ratio",
        "get_sortino_ratio",
        "get_treynor_ratio",
        "get_information_ratio",
        "get_tracking_error",
        ("get_alpha", {"method": "jensen"}),
        "get_calmar_ratio",
        "get_omega_ratio",
        "get_win_rate",
    ],
}


# The style indices get_style_analysis uses by default, Sharpe's (1992) asset classes as
# funds: US large cap value and growth (Russell 1000), small cap value and growth (Russell
# 2000), investment grade bonds (Bloomberg US Aggregate) and cash (1-3 month Treasury bills).
STYLE_TICKERS = ["IWD", "IWF", "IWN", "IWO", "AGG", "BIL"]


@filter_shown_columns
@track_sources
class Performance:
    """
    The Performance module is meant to calculate important performance metrics such
    as Sharpe Ratio, Sortino Ratio, Treynor Ratio, Information Ratio, Jensen’s
    Alpha, Beta, Capital Asset Pricing Model, R-Squared and more.
    """

    def __init__(
        self,
        tickers: list[str] | Callable[[], list[str]],
        historical_data: dict[str, pd.DataFrame],
        risk_free_rate_data: dict[str, pd.Series],
        quarterly: bool | None = None,
        rounding: int | None = 4,
        start_date: str | None = None,
        end_date: str | None = None,
        intraday_period: str | None = None,
        progress_bar: bool = True,
        returns_function: Callable[[list[str], str], pd.DataFrame] | None = None,
        market_cap_function: Callable[[], pd.DataFrame] | None = None,
        profile: pd.DataFrame | Callable[[], pd.DataFrame] | None = None,
    ):
        """
        Initializes the Performance Controller Class.

        The Toolkit passes the tickers as a function without arguments and the period
        dictionaries as a LazyDict (financetoolkit.helpers), so a metric retrieves the data of
        the period it is calculated over when it is first needed, and nothing else.

        Args:
            tickers (list[str] | Callable[[], list[str]]): The tickers to use for the calculations.
            historical_data (dict[str, pd.DataFrame]): The historical data to use for the calculations.
            risk_free_rate_data (dict[str, pd.Series]): The annualized risk free rate per period frequency. The
                daily rate is compounded into the risk free return of every period; the other periods are
                only used without daily historical data.
            quarterly (bool | None, optional): Whether to use quarterly data. Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            start_date (str | None, optional): The start date to use for the calculations. Defaults to None.
            end_date (str | None, optional): The end date to use for the calculations. Defaults to None.
            intraday_period (str | None, optional): The intraday period used for within-period calculations.
                Defaults to None.
            progress_bar (bool, optional): Whether to show a progress bar. Defaults to True.
            returns_function (Callable[[list[str], str], pd.DataFrame] | None, optional): Returns the returns of
                tickers outside the Toolkit for a period, such as the style indices of
                `get_style_analysis`. Defaults to None, which only allows the Toolkit's tickers.
            market_cap_function (Callable[[], pd.DataFrame] | None, optional): Returns the daily market
                capitalization of the tickers, as `get_beta_implied_return` values the Residual with.
                Defaults to None, which leaves the Residual Value empty.
            profile (pd.DataFrame | Callable | None, optional): The company profiles, with
                the tickers as columns and the profile items (such as "Sector" and
                "Industry") as rows, or a function that retrieves them when first needed,
                for the groups of get_factor_analysis. Defaults to None, which leaves
                group="sector" and group="industry" unavailable.

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
        | 2021Q2 |  0.1155 |  0.1366 |
        | 2021Q3 |  0.0067 |  0.0052 |
        | 2021Q4 |  0.1214 |  0.1869 |
        | 2022Q1 | -0.0576 | -0.1007 |
        | 2022Q2 | -0.2134 | -0.332  |
        | 2022Q3 | -0.0594 | -0.0819 |
        | 2022Q4 |  0.106  |  0.0999 |
        | 2023Q1 |  0.0829 |  0.1514 |
        | 2023Q2 |  0.1032 |  0.1765 |
        | 2023Q3 | -0.0414 | -0.101  |
        | 2023Q4 |  0.1021 |  0.2458 |
        | 2024Q1 |  0.1037 |  0.1407 |
        | 2024Q2 |  0.0524 |  0.0589 |
        | 2024Q3 |  0.0571 |  0.1512 |
        | 2024Q4 |  0.0216 |  0.0529 |
        | 2025Q1 | -0.0387 | -0.1505 |
        | 2025Q2 |  0.1472 |  0.2034 |
        | 2025Q3 |  0.103  |  0.1764 |
        | 2025Q4 |  0.0235 |  0.0496 |
        """
        self._tickers_source = tickers
        self._profile_source = profile
        self._benchmark_name = "Benchmark"
        self._quarterly: bool | None = quarterly
        self._rounding: int | None = rounding
        self._start_date: str | None = start_date
        self._end_date: str | None = end_date
        self._progress_bar: bool = progress_bar
        self._intraday_period: str | None = intraday_period
        self._portfolio_weights: dict | None = None
        self._returns_function = returns_function
        self._market_cap_function = market_cap_function

        # Historical Data
        self._historical_data = historical_data
        # The risk free rate is quoted as an annualized yield, so it is converted to the risk free return of each ticker per period: compounded over the ticker's own dates in each period that hold a return, as the period's return compounds them, so a partial period (such as the current year) earns the rate of the dates it covers and a stock next to a cryptocurrency earns the same rate as on its own. Without this, a daily return would have a full year of risk free rate subtracted from it.  # noqa: E501
        self._annualized_risk_free_rate_data = risk_free_rate_data
        self._risk_free_rate_data = LazyDict(self._get_risk_free_rate)

        # Fama and French
        # The Fama and French factors per region, retrieved once each.
        self._fama_and_french_datasets: dict[str, pd.DataFrame] = {}
        self._fama_and_french_model: pd.DataFrame = pd.DataFrame()
        self._fama_and_french_residuals: pd.DataFrame = pd.DataFrame()
        self._carhart_four_factor_model: pd.DataFrame = pd.DataFrame()
        self._factor_asset_correlations: pd.DataFrame = pd.DataFrame()
        self._factor_correlations: pd.DataFrame = pd.DataFrame()

        # The daily and intraday data with their excess returns, and the within period data
        # built from them, each made when a metric first asks for it.
        self._excess_historical_data = LazyDict(self._add_excess_return)
        self._within_historical_data = LazyDict(
            lambda period: determine_within_historical_data(
                period, self._excess_historical_data, self._intraday_period
            )
        )

        # Memo for the column slices handed to the calculations, keyed by (period,
        # column, within_period, benchmark): nearly every metric starts from the same
        # few slices and the MCP server collects many metrics per session. Neither
        # frame is mutated after this point and callers treat the slices as read-only.
        self._data_cache: dict[
            tuple[str, str, bool, bool], pd.DataFrame | pd.Series
        ] = {}

    @cached_property
    def _tickers(self) -> list[str]:
        """The tickers, retrieved when first needed."""
        return (
            self._tickers_source()
            if callable(self._tickers_source)
            else self._tickers_source
        )

    @cached_property
    def _tickers_with_returns(self) -> list[str]:
        """
        The tickers the metrics are calculated for, each from its own returns. The Toolkit of a
        Portfolio holds the Portfolio's own return series as the "Portfolio" ticker, so its
        ratios are those of that series, which include how the holdings move together; a
        weighted average of the holdings' ratios is not the ratio of the Portfolio. A Portfolio
        without returns in the historical data is left out, and `handle_portfolio` gives it the
        weighted average of the tickers' results instead.
        """
        historical_tickers = self._historical_data["daily"].columns.get_level_values(-1)

        return list(
            dict.fromkeys(
                ticker
                for ticker in self._tickers
                if ticker != "Portfolio" or ticker in historical_tickers
            )
        )

    @cached_property
    def _daily_returns(self) -> pd.DataFrame:
        """The daily returns of every column, including the Benchmark."""
        daily_historical_data = self._historical_data["daily"]

        if daily_historical_data.empty or "Return" not in daily_historical_data:
            return pd.DataFrame()

        return daily_historical_data["Return"]

    def _get_risk_free_rate(self, period: str) -> pd.DataFrame:
        """
        The risk free return per period of each ticker (and the Benchmark) on its own
        calendar, see `financetoolkit.risk.helpers.get_risk_free_rate_per_ticker`.
        """
        return get_risk_free_rate_per_ticker(
            self._annualized_risk_free_rate_data, self._historical_data, period
        )

    def _add_excess_return(self, period: str) -> pd.DataFrame:
        """
        The daily or intraday historical data with an "Excess Return" block, the data the
        within-period calculations nest.

        Args:
            period (str): "daily" or "intraday".

        Returns:
            pd.DataFrame: The historical data with its "Excess Return" columns; intraday data
            stays empty without intraday bars.
        """
        if period == "intraday":
            intraday_historical_data = self._historical_data["intraday"]
            if intraday_historical_data.empty:
                return intraday_historical_data
            return self._add_intraday_excess_return(intraday_historical_data.copy())

        # Historical data supplied with an "Excess Return" block of its own gets the one of this
        # risk free rate instead, as the intraday data does: two blocks would duplicate every ticker.
        daily_historical_data = self._historical_data["daily"].drop(
            columns="Excess Return", level=0, errors="ignore"
        )
        # Each ticker earns the risk free rate per observation of its own calendar: a
        # trading day of it per return, or the calendar days since its previous return
        # for a ticker that trades on weekends.
        excess_return = (
            daily_historical_data["Return"] - self._risk_free_rate_data["daily"]
        )
        excess_return.columns = pd.MultiIndex.from_product(
            [["Excess Return"], excess_return.columns]
        )

        return pd.concat([daily_historical_data, excess_return], axis=1)

    def _add_intraday_excess_return(self, intraday_data: pd.DataFrame) -> pd.DataFrame:
        """
        Add an "Excess Return" block to the intraday data, the return of each bar minus
        the risk free rate earned over one bar, so the within-period ratios (Sharpe,
        Sortino, ...) also work on the intraday bars within a day.

        The per-bar rate spreads the daily risk free rate of the bar's day over the
        median number of bars in a trading day:

            - Bar Risk Free Rate = (1 + Daily Risk Free Rate)^(1 / Bars per Day) - 1

        Days without a daily rate (e.g. a bar on a day the treasury data has no
        quote) take the most recent earlier rate.

        Args:
            intraday_data (pd.DataFrame): The intraday historical data, PeriodIndexed.

        Returns:
            pd.DataFrame: The intraday data with its "Excess Return" columns.
        """
        intraday_data = intraday_data.drop(
            columns="Excess Return", level=0, errors="ignore"
        )
        bars = to_period_index(intraday_data.index)
        bar_days = bars.asfreq("D")
        bars_per_day = float(pd.Series(1, index=bars).groupby(bar_days).size().median())

        # The bars of a day share the risk free rate of the dates of the daily returns.
        daily_rate = (
            get_risk_free_rate_per_date(
                self._annualized_risk_free_rate_data["daily"], self._daily_returns
            )
            if not self._daily_returns.empty
            else convert_annualized_rate_to_period(
                self._annualized_risk_free_rate_data["daily"], "daily"
            )
        ).sort_index()
        daily_rate = daily_rate[~daily_rate.index.duplicated(keep="last")]
        bar_rate = (1 + daily_rate.reindex(bar_days, method="ffill")) ** (
            1 / bars_per_day
        ) - 1

        excess_return = intraday_data["Return"].sub(bar_rate.to_numpy(), axis=0)
        excess_return.columns = pd.MultiIndex.from_product(
            [["Excess Return"], excess_return.columns]
        )

        return pd.concat([intraday_data, excess_return], axis=1)

    def _get_observations_per_year(self, period: str) -> pd.Series:
        """
        The number of observations in a year of the returns nested within `period`, per
        ticker, used to annualize a ratio computed from them. Those are daily returns: 252 a
        year for a ticker that trades on trading days, and the observations per year of its
        365 for a ticker that also trades on weekends, such as a cryptocurrency
        (see `financetoolkit.risk.helpers.get_observations_per_year`). For
        period="daily" and "intraday", which nest the intraday bars, a year holds those days
        times the median number of bars per day.

        Args:
            period (str): The period whose within-period observations are annualized.

        Returns:
            pd.Series: The number of within-period observations per year, indexed by ticker.
        """
        observations_per_year = (
            get_observations_per_year(self._daily_returns)
            if not self._daily_returns.empty
            else pd.Series(
                float(PERIODS_PER_YEAR["daily"]), index=self._tickers_with_returns
            )
        )

        if (
            period not in ("daily", "intraday")
            or self._historical_data["intraday"].empty
        ):
            return observations_per_year

        bars = self._get_within_historical_data(period).index.get_level_values(1)
        bars_per_day = (
            pd.Series(1, index=bars).groupby(bars.asfreq("D")).size().median()
        )

        return observations_per_year * float(bars_per_day)

    def _get_ratio_observations_per_year(
        self, period: str, rolling: int | None, columns: pd.Index
    ) -> pd.Series:
        """
        The observations in a year of the returns a Sharpe or Sortino ratio is computed from,
        per ticker, to annualize it with: the periods in a year for a rolling ratio (the daily
        observations of each ticker for period="daily", as a ticker trading on weekends has
        more), the daily (or intraday) returns within each period otherwise.

        Args:
            period (str): The period of the ratio.
            rolling (int | None): The rolling window, None for a ratio per period.
            columns (pd.Index): The tickers of the ratio.

        Returns:
            pd.Series: The observations per year, indexed by ticker.
        """
        if rolling:
            observations_per_year = (
                get_observations_per_year(self._daily_returns)
                if period == "daily"
                else pd.Series(float(PERIODS_PER_YEAR[period]), index=columns)
            )
        else:
            observations_per_year = self._get_observations_per_year(period)

        return observations_per_year.reindex(columns).fillna(
            float(PERIODS_PER_YEAR["daily"])
        )

    def _get_whole_sample_returns(
        self, benchmark: bool = False
    ) -> tuple[pd.DataFrame | pd.Series, pd.Series | float]:
        """
        The daily returns over the date range and their annualized return over that whole
        sample (see `performance_model.get_annualized_return`), the inputs of the reference
        libraries' Calmar, Sterling and Active Premium. Each ticker is annualized with the
        observations in a year of its own calendar, 252 or 365 for one that trades on
        weekends.

        Args:
            benchmark (bool, optional): Whether to return the benchmark instead of the
                tickers. Defaults to False.

        Returns:
            tuple: The daily returns and their annualized return.
        """
        daily_returns = self._get_column(
            "daily", "Return", within_period=False, benchmark=benchmark
        ).loc[self._start_date : self._end_date]

        return daily_returns, performance_model.get_annualized_return(
            daily_returns, get_observations_per_year(daily_returns)
        )

    def _get_benchmark_volatility(self, period: str, rolling: int | None) -> pd.Series:
        """
        The volatility of the benchmark per period, the systematic risk of the modified Treynor
        ratio and the alternative Jensen's alpha divided by Beta: the standard deviation of its
        daily returns within each period scaled to the period (see
        `financetoolkit.risk.risk_model.get_volatility`), or of its period returns over the
        rolling window.

        Args:
            period (str): The period of the ratio.
            rolling (int | None): The rolling window, None for a ratio per period.

        Returns:
            pd.Series: The benchmark volatility, indexed by period.
        """
        if rolling:
            return (
                self._get_column(period, "Return", within_period=False, benchmark=True)
                .rolling(rolling)
                .std()
            )

        return get_volatility(
            self._get_column("daily", "Return", within_period=False, benchmark=True),
            period,
        )

    def _get_downside_deviation(
        self,
        period: str,
        column: str,
        minimum_acceptable_return: float = 0.0,
        benchmark: bool = False,
    ) -> pd.DataFrame | pd.Series:
        """
        The downside deviation of the daily returns within each period below a minimum
        acceptable return, scaled to the period like a volatility: times the square root of
        the daily returns in a period (252 a year, 365 for a ticker that trades on weekends,
        see `financetoolkit.risk.helpers.get_days_per_period`).

        Args:
            period (str): The period to measure the downside deviation over.
            column (str): "Return" or "Excess Return", the latter for a downside deviation
                below the risk-free rate.
            minimum_acceptable_return (float, optional): The minimum acceptable return per
                day. Defaults to 0.0.
            benchmark (bool, optional): Whether to return the benchmark instead of the
                tickers. Defaults to False.

        Returns:
            pd.DataFrame | pd.Series: The downside deviation per period (and ticker).
        """
        within_returns = self._get_column(
            period, column, within_period=True, benchmark=benchmark
        )
        downside_deviation = get_downside_deviation(
            within_returns.to_frame() if benchmark else within_returns,
            minimum_acceptable_return,
        )
        # The calendar of each ticker, which the daily returns hold.
        days_per_period = get_days_per_period(
            self._get_column(
                "daily", "Return", within_period=False, benchmark=benchmark
            ),
            period,
        )
        downside_deviation = downside_deviation * np.sqrt(days_per_period)

        return downside_deviation.iloc[:, 0] if benchmark else downside_deviation

    def _get_within_historical_data(self, period: str) -> pd.DataFrame:
        """
        The within-period historical data for `period`, with the one impossible
        combination turned into a clear error: a single day only nests observations
        when intraday data was fetched, so without it the daily entry does not exist
        and a plain lookup would surface as an unhelpful KeyError. Guarding at the
        access point (rather than per method) keeps the rolling code paths, which
        read the regular historical data, free to use period="daily".
        """
        if period == "daily" and self._historical_data["intraday"].empty:
            raise InvalidArgumentError(
                "Intraday data is required for within-period daily calculations. "
                "Initialise the Toolkit with an intraday_period to use period='daily' here."
            )

        return self._within_historical_data[period]

    def _get_matrix_returns(self, period: str) -> pd.DataFrame:
        """
        The returns of the Covariance and Correlation Matrix, with a warning when the date
        range holds fewer than three of them: a yearly default over a few years leaves a
        handful of points per pair, or none (a matrix of NaN).
        """
        returns = self._historical_data[period]["Return"].loc[
            self._start_date : self._end_date
        ]
        observations = len(returns.dropna(how="all"))

        if observations < MINIMUM_MATRIX_OBSERVATIONS:
            logger.warning(
                "There are %s %s returns between %s and %s, too few for a covariance or correlation. "  # noqa: E501
                "Use a higher frequency period, such as period='monthly'.",
                observations,
                period,
                self._start_date,
                self._end_date,
            )

        return returns

    def _get_column(
        self, period: str, column: str, within_period: bool, benchmark: bool = False
    ) -> pd.DataFrame | pd.Series:
        """
        The `column` slice for the tickers (a DataFrame) or the benchmark (a Series),
        memoized per (period, column, within_period, benchmark) pair -- see
        `_data_cache` in `__init__`. Within-period lookups go through
        `_get_within_historical_data` so the daily-without-intraday error still
        surfaces from the call site that asked for it.
        """
        key = (period, column, within_period, benchmark)

        if key not in self._data_cache:
            frame = (
                self._get_within_historical_data(period)
                if within_period
                else self._historical_data[period]
            )
            self._data_cache[key] = frame.loc[:, column][
                self._benchmark_name if benchmark else self._tickers_with_returns
            ]

        return self._data_cache[key]

    @handle_errors
    @log_each_message_once()
    def collect_all_metrics(
        self,
        period: str | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate all performance metrics in one table: the Alpha, Beta, CAPM and the risk-adjusted
        return ratios of every ticker, to compare the tickers on each of them side by side.

        Args:
            period (str, optional): The period to use for the calculation. Options are: "daily", "weekly",
                "monthly", "quarterly", "yearly". Defaults to "quarterly" if the Toolkit is initialised with
                quarterly=True, otherwise "yearly".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Performance metrics calculated based on the specified parameters, with a
            Multi Index of (metric, ticker) as the columns. For a single-ticker Toolkit the ticker
            level is dropped. `Returns` and `Excess Return` are only included when `period` is not
            "daily".

        Notes:
        - The method calculates various performance metrics for each asset in the Toolkit instance.
        - With growth, the method calculates the growth of the ratio values
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

        toolkit.performance.collect_all_metrics().xs("AAPL", level=1, axis=1)
        ```

        Which returns:

        |      |    Alpha |   Beta |     CAPM |   Jensen's Alpha |   Treynor Ratio |   Sharpe Ratio |   Safety-First Ratio |   Sortino Ratio |   Upside Potential Ratio |   Ulcer Performance Index |   Calmar Ratio |   Sterling Ratio |   Burke Ratio |   Pain Ratio |   Omega Ratio |   Kappa Ratio |   Gain to Pain Ratio |   d Ratio |   Prospect Ratio |   Win Rate |   Upside Capture Ratio |   Downside Capture Ratio |   M2 Ratio |   Omega Excess Return |   Tracking Error |   Information Ratio |   Appraisal Ratio |   STARR Ratio |   Rachev Ratio |   Returns |   Excess Return |
        |:-----|---------:|-------:|---------:|-----------------:|----------------:|---------------:|---------------------:|----------------:|-------------------------:|--------------------------:|---------------:|-----------------:|--------------:|-------------:|--------------:|--------------:|---------------------:|----------:|-----------------:|-----------:|-----------------------:|-------------------------:|-----------:|----------------------:|-----------------:|--------------------:|------------------:|--------------:|---------------:|----------:|----------------:|
        | 2022 |  -0.0823 | 1.3027 |  -0.2457 |          -0.0184 |         -0.2253 |        -0.0483 |              -0.0431 |         -0.0676 |                   0.5071 |                   -1.7999 |        -0.87   |          -1.067  |       -0.1136 |      -1.9984 |        0.8941 |       -0.0521 |              -0.1059 |    1.2511 |          -0.7774 |     0.4781 |                 1.3037 |                   1.3243 |    -0.1699 |               -0.3974 |           0.0115 |             -0.025  |           -0.1106 |       -0.4074 |         1.0649 |   -0.264  |         -0.2935 |
        | 2023 |   0.2283 | 1.1139 |   0.2871 |           0.203  |          0.4045 |         0.1191 |               0.1312 |          0.1843 |                   0.6867 |                    8.175  |         3.282  |           3.3708 |        0.517  |      12.346  |        1.4108 |        0.1286 |               0.4108 |    0.5519 |          -0.4187 |     0.588  |                 1.1755 |                   0.9847 |     0.3297 |                0.4572 |           0.009  |              0.0797 |            1.4414 |        1.0738 |         1.0942 |    0.4901 |          0.4505 |
        | 2024 |   0.0582 | 0.9687 |   0.2424 |           0.0647 |          0.2732 |         0.0701 |               0.0815 |          0.1072 |                   0.6117 |                    4.1633 |         1.9997 |           1.971  |        0.2623 |       5.4831 |        1.251  |        0.0767 |               0.251  |    0.6136 |          -0.5007 |     0.504  |                 1.1048 |                   1.0687 |     0.1891 |                0.2688 |           0.0121 |              0.0208 |            0.3379 |        0.5637 |         1.1444 |    0.3071 |          0.2646 |
        | 2025 |  -0.0867 | 1.2551 |   0.2114 |          -0.1209 |          0.0379 |         0.0189 |               0.027  |          0.0283 |                   0.4827 |                    0.3943 |         0.2913 |           0.4451 |        0.0249 |       0.4983 |        1.0903 |        0.0187 |               0.0903 |    0.8331 |          -0.5231 |     0.472  |                 1.015  |                   1.1072 |     0.0715 |                0.0073 |           0.0139 |             -0.0126 |           -0.5667 |        0.0659 |         1.0731 |    0.0905 |          0.0476 |
        | 2021 | nan      | 1.3195 | nan      |         nan      |        nan      |         0.0858 |               0.0894 |          0.1289 |                   0.642  |                  nan      |       nan      |         nan      |      nan      |     nan      |        1.2629 |        0.0942 |               0.2629 |    0.6913 |          -0.5054 |     0.494  |                 1.3706 |                   1.4316 |   nan      |              nan      |           0.0118 |              0.0267 |          nan      |      nan      |         0.9961 |    0.3806 |          0.3663 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"
        rounding = rounding if rounding is not None else self._rounding

        performance_metrics = {
            "Alpha": self.get_alpha(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
            "Beta": self.get_beta(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
            "CAPM": self.get_capital_asset_pricing_model(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
            "Jensen's Alpha": self.get_alpha(
                period=period,
                method="jensen",
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
            "Treynor Ratio": self.get_treynor_ratio(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
            "Sharpe Ratio": self.get_sharpe_ratio(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
            "Safety-First Ratio": self.get_safety_first_ratio(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
            "Sortino Ratio": self.get_sortino_ratio(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
            "Upside Potential Ratio": self.get_upside_potential_ratio(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
            "Ulcer Performance Index": self.get_ulcer_performance_index(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
            "Calmar Ratio": self.get_calmar_ratio(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
            "Sterling Ratio": self.get_sterling_ratio(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
            "Burke Ratio": self.get_burke_ratio(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
            "Pain Ratio": self.get_pain_ratio(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
            "Omega Ratio": self.get_omega_ratio(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
            "Kappa Ratio": self.get_kappa_ratio(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
            "Gain to Pain Ratio": self.get_gain_to_pain_ratio(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
            "d Ratio": self.get_d_ratio(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
            "Prospect Ratio": self.get_prospect_ratio(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
            "Win Rate": self.get_win_rate(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
            "Upside Capture Ratio": self.get_upside_capture_ratio(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
            "Downside Capture Ratio": self.get_downside_capture_ratio(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
            "M2 Ratio": self.get_m2_ratio(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
            "Omega Excess Return": self.get_omega_excess_return(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
            "Tracking Error": self.get_tracking_error(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
            "Information Ratio": self.get_information_ratio(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
            "Appraisal Ratio": self.get_appraisal_ratio(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
            "STARR Ratio": self.get_starr_ratio(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
            "Rachev Ratio": self.get_rachev_ratio(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
        }

        if period != "daily":
            performance_metrics["Returns"] = self.get_returns(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            )
            performance_metrics["Excess Return"] = self.get_excess_return(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            )

        performance_metrics = pd.concat(performance_metrics, axis=1)

        if len(self._tickers) == 1:
            performance_metrics = performance_metrics.xs(
                self._tickers[0], level=1, axis=1
            )

        return performance_metrics

    @handle_portfolio
    @handle_errors
    def get_beta(
        self,
        period: str | None = None,
        rolling: int | None = None,
        method: Literal[
            "ols",
            "blume",
            "vasicek",
            "dimson",
            "scholes_williams",
            "upside",
            "downside",
        ] = "ols",
        lags: int = 1,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Beta: how strongly an asset's returns move with those of the benchmark, the
        systematic risk that diversification cannot remove.

        Beta is a financial metric used to assess the systematic risk of a stock or investment in
        relation to the overall market. It provides valuable insights into how a particular asset's
        returns tend to move in response to fluctuations in the broader market. A stock's Beta is
        calculated by analyzing its historical price movements and their correlation with the
        movements of a market index, typically the benchmark index like the S&P 500.

        The formula is as follows:

            - Beta = Covariance of Asset Returns and Benchmark Returns / Variance of Benchmark Returns

        For a given period, for example monthly, this translates into the following:

            - Beta = Monthly Covariance of Asset Returns and Benchmark Returns
            / Monthly Variance of Benchmark Returns

        Beta is read against 1, the benchmark's own beta: above 1 the asset tends to amplify market moves, below 1 to
        dampen them. Each period's Beta comes from the daily returns within that period, so a yearly figure rests on
        about 250 observations and can swing with a single turbulent month; a rolling window (for example monthly
        returns over a rolling window of 36) gives a steadier estimate for cost of capital work. A Beta close to zero
        can come from a low correlation with the market rather than from low volatility, so it is worth checking both
        before calling a stock defensive.

        See definition: https://en.wikipedia.org/wiki/Beta_(finance)

        Also known as: market sensitivity, systematic risk.

        Shares similar logic:
        - empyrical, roll_alpha_beta: https://github.com/quantopian/empyrical/blob/0.5.5/empyrical/stats.py#L1058
        - PerformanceAnalytics (R), CAPM.beta: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/CAPM.beta.R#L89
        - empyrical, beta: https://github.com/quantopian/empyrical/blob/0.5.5/empyrical/stats.py#L1314

        Args:
            period (str, optional): The period to use for the calculation. Options are: "daily", "weekly",
                "monthly", "quarterly", "yearly". Defaults to "quarterly" if the Toolkit is initialised with
                quarterly=True, otherwise "yearly".
            rolling (int, optional): The rolling window size to use for the calculation. If you select
                period = 'monthly' and set rolling to 12 you obtain the rolling 12-month Beta. At least 2. Defaults to None.
            method (str, optional): The estimator of the Beta. Options are:
                - "ols": the covariance over the benchmark variance (the formula above).
                - "blume": the adjusted Beta of Bloomberg, 0.67 x OLS Beta + 0.33 (Blume, 1971, 1975).
                - "vasicek": the OLS Beta shrunk towards the mean Beta of the tickers by the precision of
                  each estimate (Vasicek, 1973).
                - "dimson": the sum of the slopes on the lagged, same and leading benchmark returns
                  (Dimson, 1979).
                - "scholes_williams": the lagged, same and leading slopes over the benchmark's
                  autocorrelation (Scholes and Williams, 1977).
                - "upside" or "downside": the OLS Beta on the days the benchmark rose or fell.
                Defaults to "ols".
            lags (int, optional): The number of leads and lags of the Dimson Beta. Defaults to 1.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Beta values.

        Notes:
        - Daily Beta is not an option as the standard deviation for 1 day is close to zero. Therefore, it does
        not give any useful insights.
        - The method retrieves historical data and calculates the Beta for each asset in the Toolkit instance.
        - With growth, the method calculates the growth of the ratio values using the specified `lag`.
        - The OLS Beta of a stock that trades thinly or reacts to the market a day late understates its market
        risk, because part of the reaction falls on the next day. Dimson and Scholes-Williams add the slopes on
        the neighbouring benchmark returns to correct for that; their leads and lags come from the full history,
        so the first day of a period uses the benchmark return of the day before.
        - Blume found that Betas move towards 1 between estimation windows by about a third. Vasicek shrinks each Beta
        in proportion to its uncertainty: the prior is the mean and sample variance (with n - 1 in the denominator) of
        the OLS Betas of the tickers in the same period, so it depends on which tickers are passed. A single ticker
        gets a prior mean of 1 (the market's Beta) with a variance of 0.25.
        - The upside and downside Betas show asymmetric market exposure: a stock with a downside Beta above
        its upside Beta falls harder in down markets than it rises in up markets. The benchmark's direction
        picks the observations: upside uses the periods the benchmark rose, downside those it fell.
        - Every estimator needs at least three observations in a period (Dimson: more than the number of
        slopes plus one), otherwise the period is NaN.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["NVDA", "TSLA", "JPM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.performance.get_beta()
        ```

        Which returns:

        | Date   |   NVDA |   TSLA |    JPM |
        |:-------|-------:|-------:|-------:|
        | 2021   | 2.0431 | 1.9542 | 0.8376 |
        | 2022   | 2.2001 | 1.7404 | 0.8815 |
        | 2023   | 2.0217 | 2.247  | 0.8723 |
        | 2024   | 2.6583 | 2.38   | 0.8006 |
        | 2025   | 1.8727 | 2.2342 | 0.9481 |

        The downside Beta shows how the same stocks moved on the days the S&P 500 fell:

        ```python
        toolkit.performance.get_beta(method="downside")
        ```

        Which returns:

        | Date   |   NVDA |   TSLA |    JPM |
        |:-------|-------:|-------:|-------:|
        | 2021   | 2.0938 | 1.6149 | 0.7877 |
        | 2022   | 1.9875 | 1.5848 | 0.7921 |
        | 2023   | 2.1135 | 2.5952 | 1.048  |
        | 2024   | 2.0508 | 2.6545 | 0.9306 |
        | 2025   | 1.8311 | 1.9833 | 0.9638 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"
        check_rolling(rolling, self._historical_data, period)

        returns = self._get_column(period, "Return", within_period=not rolling)
        benchmark_returns = self._get_column(
            period, "Return", within_period=not rolling, benchmark=True
        )

        if method != "ols":
            beta = performance_model.get_beta_estimate(
                returns,
                benchmark_returns,
                method=method,
                lags=lags,
                window_size=rolling,
            )
        elif rolling:
            beta = performance_model.get_beta(
                returns, benchmark_returns, window=rolling
            )
        else:
            beta = performance_model.get_beta(returns, benchmark_returns)

        return finalize_dataset(
            dataset=beta,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_errors
    def get_beta_implied_return(
        self,
        period: str = "daily",
        beta_window: int = 252,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Beta-Implied Return, the part of an asset's return that its Beta to the
        benchmark explains, and the Residual, the part that is the asset's own.

        When the market falls 4% and a stock with a Beta of 1.8 falls 9%, the Beta accounts for
        about 7.2% of the fall (1.8 × 4%) and the remaining 1.8% is specific to the stock: news,
        earnings or sector moves that the market as a whole did not share. This is the market
        model of Sharpe (1963) without its intercept, the split behind event studies and return
        attribution, which reads a move against what the market alone would have produced.

        The formula is as follows:

            - Implied Return = Beta × Benchmark Return
            - Residual = Actual Return - Implied Return

        The Beta of each period is estimated from the daily returns over the `beta_window` trading
        days that end on the day before the period starts, so the move being explained never helps
        estimate the Beta that explains it, and the figure is what an investor could have known at
        the start of the period. With the default daily period every row explains one day's move;
        a monthly or yearly period explains that month or year with the Beta known when it began.

        Three measures put the Residual in context. The Residual Value is the Residual in money: the
        market value the asset's own move added or took away, the Residual times the market
        capitalization at the close before the period. The Residual Percentile and the Residual
        Largest Since measure the Residual against the earlier residuals, as `get_move_significance`
        does for the return itself: the Percentile is the share of the earlier residuals that were
        smaller in size and Largest Since the date of the last residual at least as large in the same
        direction ("the largest stock-specific fall since"), so the whole history counts without
        returning it.

            - Residual Value = Residual × Market Cap at the previous close
            - Residual Percentile(t) = (Count(|Residual(s)| < |Residual(t)|, s < t) + 0.5 · Count(|Residual(s)| =
              |Residual(t)|, s < t)) / Count(s < t)
            - Residual Largest Since(t) = max(s < t : Residual(s) <= Residual(t)) if Residual(t) < 0,
              and max(s < t : Residual(s) >= Residual(t)) if Residual(t) > 0

        See definition: https://en.wikipedia.org/wiki/Single-index_model

        Also known as: beta-adjusted return, market-implied move, idiosyncratic return, abnormal return.

        Args:
            period (str, optional): The period whose returns are explained. Options are: "daily", "weekly",
                "monthly", "quarterly", "yearly". Defaults to "daily".
            beta_window (int, optional): The number of daily returns the Beta is estimated on. Defaults
                to 252, about one year of trading days, a common length for a market model's
                estimation window that is long enough for a stable Beta and short enough to follow
                changes in the business.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The periods as rows and a MultiIndex of (measure, ticker) as columns, with the
            measures Beta, Benchmark Return, Implied Return, Actual Return, Residual, Residual Value,
            Residual Percentile and Residual Largest Since (the date as text). All returns are decimals.

        Notes:
        - Returns are total returns (Adj Close), the asset's and the benchmark's over the same period.
        - The Implied Return leaves out the risk free rate: the CAPM's expected return, Risk Free Rate +
          Beta × (Benchmark Return — Risk Free Rate), differs from it by (1 — Beta) × the risk free rate,
          a few basis points per day, and is available through `get_capital_asset_pricing_model`.
        - A period whose window has fewer than `beta_window` daily returns before it (the start of the
          data or a recent listing) has no Beta, so its Implied Return and Residual are empty. Start the
          Toolkit at least `beta_window` trading days before the first period of interest.
        - The Beta is the ordinary least squares Beta of `get_beta`, on the daily returns of the window.
        - The market cap is that of the daily `get_market_cap`: the traded split-adjusted Close
          times the weighted average diluted shares of the latest fiscal period that ended on or before
          that day, so no shares are used before their period closes. It is in the currency the ticker
          trades in, and so is the Residual Value. Without shares, as for an ETF or an index, the Residual
          Value is empty.
        - The Percentile needs a year of earlier residuals (252 days, 52 weeks or 20 periods otherwise),
          and Largest Since is empty when no earlier residual was as large, which makes the Residual the
          largest since the first residual. The residuals start `beta_window` trading days after the
          Toolkit's start_date, so choose an early start_date for a long history.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["NVDA", "KO", "AMD", "TSLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2022-01-01",
            end_date="2026-11-05",
        )

        toolkit.performance.get_beta_implied_return().loc["2026-11-05"].unstack(sort=False)
        ```

        Which returns:

        |                        | NVDA              | KO              | AMD              | TSLA              |
        |:-----------------------|:------------------|:----------------|:-----------------|:------------------|
        | Beta                   | 1.8764            | -0.2779         | 3.2862           | 2.1033            |
        | Benchmark Return       | -0.0441           | -0.0441         | -0.0441          | -0.0441           |
        | Implied Return         | -0.0827           | 0.0123          | -0.1449          | -0.0927           |
        | Actual Return          | -0.0926           | -0.0007         | -0.1726          | -0.1077           |
        | Residual               | -0.0099           | -0.013          | -0.0277          | -0.015            |
        | Residual Value         | -48347086729.0694 | -5363943888.836 | -22181169537.895 | -20387104458.3291 |
        | Residual Percentile    | 0.4231            | 0.8482          | 0.7588           | 0.4896            |
        | Residual Largest Since | 2026-11-04        | 2026-10-27      | 2026-11-04       | 2026-11-04        |
        """
        returns = self._get_column("daily", "Return", within_period=False)

        implied_returns = performance_model.get_beta_implied_return(
            returns,
            self._get_column("daily", "Return", within_period=False, benchmark=True),
            period,
            beta_window,
        )

        # The market cap is empty (an empty Series) when no ticker has statements.
        market_cap = self._market_cap_function() if self._market_cap_function else None
        residual_value = performance_model.get_residual_value(
            implied_returns["Residual"],
            (
                market_cap.reindex(columns=returns.columns).astype(float)
                if isinstance(market_cap, pd.DataFrame) and not market_cap.empty
                else pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
            ),
            period,
        )

        measures = list(implied_returns.columns.get_level_values(0).unique())
        measures.insert(measures.index("Residual") + 1, "Residual Value")

        implied_returns = pd.concat(
            [implied_returns, pd.concat({"Residual Value": residual_value}, axis=1)],
            axis=1,
        )[pd.MultiIndex.from_product([measures, returns.columns])]

        implied_returns = implied_returns.loc[self._start_date : self._end_date]

        return apply_rounding(
            implied_returns.dropna(how="all", axis=0),
            rounding if rounding is not None else self._rounding,
        )

    @handle_portfolio
    @handle_errors
    def get_capital_asset_pricing_model(
        self,
        period: str | None = None,
        rolling: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Capital Asset Pricing Model (CAPM): the return an asset should have earned given
        its Beta and what the market returned, the yardstick its actual return is judged against. The
        CAPM estimates the expected return of an investment, such as a stock or portfolio of stocks,
        from its risk and return trade-off with the overall market, and is based on the following key
        components:

            - Risk-Free Rate (Rf): This is the theoretical return an investor could earn from an investment
            with no risk of financial loss. It is typically based on the yield of a government bond.
            - Market Risk Premium (Rm - Rf): This represents the additional return that investors expect
            to earn for taking on the risk of investing in the overall market as opposed to a risk-free asset.
            It is calculated as the difference between the expected return of the market (Rm) and the risk-free
            rate (Rf).
            - Beta (β): Beta is a measure of an asset's or portfolio's sensitivity to market movements. It
            quantifies how much an asset's returns are expected to move in relation to changes in the
            overall market. A beta of 1 indicates that the asset moves in line with the market, while a
            beta greater than 1 suggests higher volatility, and a beta less than 1 indicates lower volatility.

        The Capital Asset Pricing Model (CAPM) is a widely used financial model that helps in
        determining the expected return of an asset or portfolio based on its systematic risk and
        the prevailing risk-free rate in the market. CAPM provides insights into how an asset or
        investment should be priced in order to offer an appropriate rate of return, given its
        level of risk compared to the overall market.

        The formula is as follows:

        - Capital Asset Pricing Model = Risk Free Rate + Beta * (Benchmark Returns — Risk Free Rate)

        The CAPM here uses the benchmark's realised return of each period, so the result is the return the asset
        should have earned given what the market actually did. That makes it a yardstick for Jensen's Alpha rather
        than a forward-looking expected return. For a cost of equity, practitioners use an expected market risk
        premium instead of the realised one (see the Toolkit's `get_market_risk_premium`) together with a longer-run
        Beta. In periods in which the benchmark returns less than the risk-free rate, a high-Beta asset gets a lower
        required return than a low-Beta one.

        See definition: https://en.wikipedia.org/wiki/Capital_asset_pricing_model

        Also known as: CAPM, expected return model, required return for Jensen's alpha.

        Shares similar logic:
        - empyrical, roll_alpha_beta: https://github.com/quantopian/empyrical/blob/0.5.5/empyrical/stats.py#L1058
        - PerformanceAnalytics (R), CAPM.beta: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/CAPM.beta.R#L89
        - empyrical, beta: https://github.com/quantopian/empyrical/blob/0.5.5/empyrical/stats.py#L1314

        Args:
            period (str, optional): The period to use for the calculation. Options are: "daily", "weekly",
                "monthly", "quarterly", "yearly". Defaults to "quarterly" if the Toolkit is initialised with
                quarterly=True, otherwise "yearly".
            rolling (int, optional): The rolling window size to use for the Beta component of the
            calculation. If set, Beta is estimated over a rolling window of this many periods across
            the full return history instead of per `period`. At least 2. Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: CAPM values.

        Notes:
        - Daily CAPM is not an option as the standard deviation for 1 day is close to zero. Therefore, it does
        not give any useful insights.
        - The method retrieves historical data and calculates the CAPM for each asset in the Toolkit instance.
        - The risk-free rate is the Toolkit's risk free rate, the 10-year US Treasury yield by default. Many
        textbook definitions (Sharpe, 1994; the Ken French factors) use a short-term Treasury bill instead, which
        the Toolkit uses when initialised with the 13-week Treasury bill as its risk free rate.
        - With growth, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["JPM", "GS", "MS"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2020-01-01",
            end_date="2025-12-31",
        )

        toolkit.performance.get_capital_asset_pricing_model()
        ```

        Which returns:

        | Date   |     JPM |      GS |      MS |
        |:-------|--------:|--------:|--------:|
        | 2021   |  0.2425 |  0.2807 |  0.3175 |
        | 2022   | -0.1567 | -0.173  | -0.1912 |
        | 2023   |  0.2333 |  0.2759 |  0.3056 |
        | 2024   |  0.2077 |  0.28   |  0.2632 |
        | 2025   |  0.1702 |  0.221  |  0.2173 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"
        check_rolling(rolling, self._historical_data, period)

        if rolling:
            returns = self._get_column(period, "Return", within_period=False)
            benchmark_returns = self._get_column(
                period, "Return", within_period=False, benchmark=True
            )

            beta = performance_model.get_beta(
                returns, benchmark_returns, window=rolling
            )
        else:
            returns = self._get_column(period, "Return", within_period=True)
            benchmark_returns = self._get_column(
                period, "Return", within_period=True, benchmark=True
            )

            beta = performance_model.get_beta(returns, benchmark_returns)

        risk_free_rate = self._risk_free_rate_data[period]
        benchmark_returns = self._get_column(
            period, "Return", within_period=False, benchmark=True
        )

        capm = performance_model.get_capital_asset_pricing_model(
            risk_free_rate, beta, benchmark_returns
        )

        return finalize_dataset(
            dataset=capm,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    def _get_fama_and_french_dataset(self, region: str) -> pd.DataFrame:
        """
        The Fama and French 5 factors of a region, retrieved once per region.

        Args:
            region (str): The region, see performance_model.FACTOR_REGIONS.

        Returns:
            pd.DataFrame: The daily factors as decimals.
        """
        if region not in self._fama_and_french_datasets:
            self._fama_and_french_datasets[region] = (
                performance_model.obtain_fama_and_french_dataset(region=region)
            )
        return self._fama_and_french_datasets[region]

    def _get_rolling_factor_model(
        self,
        factor_dataset: pd.DataFrame,
        factors: list[str],
        rolling: int,
        rounding: int | None,
        growth: bool,
        lag: int | list[int],
        standardize: bool,
    ) -> pd.DataFrame:
        """
        The rolling factor regression of every ticker's daily excess returns (its return minus
        the factor dataset's RF) on the factors, see `performance_model.get_factor_regression`.

        Args:
            factor_dataset (pd.DataFrame): The daily factors and RF.
            factors (list[str]): The factors to regress on.
            rolling (int): The number of daily observations in each window.
            rounding (int | None): The number of decimals to round the results to.
            growth (bool): Whether to calculate the growth of the values.
            lag (int | list[int]): The lag of the growth calculation.
            standardize (bool): Whether to standardize the result.

        Returns:
            pd.DataFrame: The regression of each window with the dates as the index and (ticker,
            parameter) as the columns.
        """
        returns = self._get_column("daily", "Return", within_period=False)
        factor_dataset = factor_dataset.reindex(returns.index)

        rolling_model = pd.concat(
            {
                ticker: performance_model.get_factor_regression(
                    returns[ticker] - factor_dataset["RF"],
                    factor_dataset[factors],
                    window=rolling,
                )
                for ticker in self._tickers_with_returns
            },
            axis=1,
        )

        return finalize_dataset(
            dataset=rolling_model,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_factor_asset_correlations(
        self,
        period: str | None = None,
        factors_to_calculate: list[str] | None = None,
        region: str = "United States",
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the factor asset correlations: how closely each asset's excess returns move with each
        of the Fama and French five factors within each period.

        The major difference between the Fama and French Model here is that the correlation
        is taken as opposed to a Linear Regression in which the R-squared or Slope can be used to
        understand the exposure to each factor.

        For assessing the exposure or influence of a stock to external factors, it's often preferable
        to use R-squared (R²) or Beta because it explicitly measures how well the factors explain the stock's
        returns. A higher R² indicates that the stock's returns are more closely related to the factors,
        and thus, the factors have a greater influence on the stock's performance.

        However, since the results are closely related and tend to point into the same direction it could
        be fine to use correlations as well depending on the level of accuracy required.

        The formula is as follows:

        - Excess Return = Daily Asset Return - Daily Risk Free Rate (RF)
        - Factor Asset Correlation = Pearson Correlation (Daily Excess Return, Daily Factor Return), using the
          daily Fama and French 5 Factor returns within each period (e.g. each year)

        Also known as: factor exposure, asset correlations.

        Args:
            period (str, optional): The period to use for the calculation. Options are: "daily", "weekly",
                "monthly", "quarterly", "yearly". Defaults to "quarterly" if the Toolkit is initialised with
                quarterly=True, otherwise "yearly".
            factors_to_calculate (list of str, optional): List of factors to calculate scores and residuals for.
                Defaults to ["Mkt-RF", "SMB", "HML", "RMW", "CMA"].
            region (str, optional): The region whose factors to use (the factors of Ken French's library
                for that region's stocks). Options are: "United States", "Developed", "Developed ex US",
                "Europe", "Japan", "Asia Pacific ex Japan", "North America". Every region's factors are
                U.S. dollar returns with the U.S. one-month Treasury bill as the risk-free rate (RF),
                so a ticker quoted in another currency is best analysed with its U.S. dollar returns;
                otherwise the exchange rate's moves end up in the factor loadings, mostly the market
                beta. Defaults to "United States".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Factor Asset Correlations, with a Multi Index of (ticker, factor)
            as the columns and one row per period.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["CAT", "PFE", "ORCL"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        toolkit.performance.get_factor_asset_correlations()["CAT"]
        ```

        Which returns:

        |      |   Mkt-RF |    SMB |     HML |     RMW |     CMA |
        |:-----|---------:|-------:|--------:|--------:|--------:|
        | 2025 |    0.698 | 0.2883 | -0.2279 | -0.2868 | -0.0246 |
        """

        factors_to_calculate = (
            factors_to_calculate
            if factors_to_calculate
            else ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
        )

        for factor in factors_to_calculate:
            if factor not in ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]:
                raise InvalidArgumentError(
                    f"Factor {factor} is not a valid factor. Please select from: Mkt-RF, SMB, HML, RMW, CMA."
                )

        period = period if period else "quarterly" if self._quarterly else "yearly"
        returns = self._get_column(period, "Return", within_period=True)

        fama_and_french_dataset = self._get_fama_and_french_dataset(region)

        fama_and_french_period = determine_within_dataset(
            dataset=fama_and_french_dataset, period=period, correlation=False
        )

        merged_df = fama_and_french_period.merge(
            returns, left_index=True, right_index=True
        )

        factor_correlations: dict = {}

        # The first index level repeats once per row, so it has to be de-duplicated before iterating or every period is recomputed once per day it contains.  # noqa: E501
        dataset_periods = merged_df.index.get_level_values(0).unique()

        logger.info("Calculating Factor Asset Correlations")
        for ticker in self._tickers_with_returns:
            factor_correlations[ticker] = {}
            for dataset_period in dataset_periods:
                factor_data = merged_df.loc[dataset_period][factors_to_calculate]
                excess_returns = (
                    merged_df.loc[dataset_period][ticker]
                    - merged_df.loc[dataset_period]["RF"]
                )

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    factor_correlations[ticker][dataset_period] = (
                        performance_model.get_factor_asset_correlations(
                            factors=factor_data, excess_return=excess_returns
                        )
                    )
        factor_asset_correlations = pd.DataFrame.from_dict(
            {
                (ticker, dataset_period): value
                for ticker, factor_scores_ticker in factor_correlations.items()
                for dataset_period, value in factor_scores_ticker.items()
            },
        )

        factor_order = factor_asset_correlations.index

        factor_asset_correlations = (
            factor_asset_correlations.stack(level=1, future_stack=True)
            .unstack(level=0)
            .reindex(factor_order, level=1, axis=1)
            .reindex(self._tickers_with_returns, level=0, axis=1)
        )

        self._factor_asset_correlations = apply_rounding(
            factor_asset_correlations,
            rounding if rounding is not None else self._rounding,
        ).loc[self._start_date : self._end_date]

        return self._factor_asset_correlations

    @handle_errors
    def get_factor_correlations(
        self,
        period: str | None = None,
        factors_to_calculate: list[str] | None = None,
        exclude_risk_free: bool = True,
        region: str = "United States",
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the factor correlations: how closely the Fama and French five factors move with each
        other within each period, which shows how much of what one factor captures another one captures
        as well. The factors are:

            - Market Risk Premium (Mkt-RF): Represents the additional return that investors expect to earn for taking
            on the risk of investing in the overall market as opposed to a risk-free asset.
            - Size Premium (SMB): Reflects the historical excess return of small-cap stocks over large-cap stocks.
            - Value Premium (HML): Captures the historical excess return of value stocks over growth stocks.
            - Profitability (RMW): Measures the historical excess return of high profitability stocks over
            low profitability stocks.
            - Investment (CMA): Quantifies the historical excess return of low investment stocks over
            high investment stocks.

        Optionally, it is also possible to see the correlation between the risk-free rate and each factor.

        The formula is as follows:

        - Factor Correlation = Pearson Correlation (Daily Factor A Return, Daily Factor B Return), using the
          daily Fama and French 5 Factor returns within each period (e.g. each year)

        Also known as: factor model correlations.

        Args:
            period (str, optional): The period to use for the calculation. Options are: "daily", "weekly",
                "monthly", "quarterly", "yearly". Defaults to "quarterly" if the Toolkit is initialised with
                quarterly=True, otherwise "yearly".
            factors_to_calculate (list of str, optional): List of factors to calculate scores and residuals for.
                Defaults to ["Mkt-RF", "SMB", "HML", "RMW", "CMA"].
            exclude_risk_free (bool, optional): Whether to exclude the risk-free rate from the results. Defaults to True.
            region (str, optional): The region whose factors to use (the factors of Ken French's library
                for that region's stocks). Options are: "United States", "Developed", "Developed ex US",
                "Europe", "Japan", "Asia Pacific ex Japan", "North America". Every region's factors are
                U.S. dollar returns with the U.S. one-month Treasury bill as the risk-free rate (RF).
                Defaults to "United States".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Factor Correlations. One correlation matrix per period, stacked
            into a Multi Index of (period, factor) rows with the factors as the columns and
            restricted to the Toolkit's date range.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["SPY"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
            benchmark_ticker=None,
        )

        toolkit.performance.get_factor_correlations()
        ```

        Which returns:

        |                    |   Mkt-RF |     SMB |     HML |     RMW |    CMA |
        |:-------------------|---------:|--------:|--------:|--------:|-------:|
        | ('2025', 'Mkt-RF') |   1      |  0.1744 | -0.3814 | -0.4273 | 0.0586 |
        | ('2025', 'SMB')    |   0.1744 |  1      |  0.2143 | -0.2924 | 0.2143 |
        | ('2025', 'HML')    |  -0.3814 |  0.2143 |  1      |  0.3573 | 0.2218 |
        | ('2025', 'RMW')    |  -0.4273 | -0.2924 |  0.3573 |  1      | 0.0657 |
        | ('2025', 'CMA')    |   0.0586 |  0.2143 |  0.2218 |  0.0657 | 1      |
        """
        factors_to_calculate = (
            factors_to_calculate
            if factors_to_calculate
            else ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
        )

        for factor in factors_to_calculate:
            if factor not in ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]:
                raise InvalidArgumentError(
                    f"Factor {factor} is not a valid factor. Please select from: Mkt-RF, SMB, HML, RMW, CMA."
                )

        period = period if period else "quarterly" if self._quarterly else "yearly"

        factors_to_calculate = (
            factors_to_calculate + ["RF"]
            if not exclude_risk_free
            else factors_to_calculate
        )

        fama_and_french_dataset = self._get_fama_and_french_dataset(region)

        # The correlations are calculated for the whole periods within the date range
        # only, rather than for every period since 1963 and then sliced.
        period_symbol = PERIOD_TRANSLATION[period]
        first_period_start = (
            pd.Period(self._start_date, freq=period_symbol).start_time
            if self._start_date
            else None
        )
        last_period_end = (
            pd.Period(self._end_date, freq=period_symbol).end_time
            if self._end_date
            else None
        )
        factor_dataset = fama_and_french_dataset[factors_to_calculate]
        factor_dates = factor_dataset.index.to_timestamp()
        within_range = np.ones(len(factor_dataset), dtype=bool)

        if first_period_start is not None:
            within_range &= factor_dates >= first_period_start
        if last_period_end is not None:
            within_range &= factor_dates <= last_period_end

        fama_and_french_period = determine_within_dataset(
            factor_dataset.loc[within_range],
            period,
            correlation=True,
        )

        # The Ken French dataset starts in 1963, so without this slice the result covers six decades regardless of the date range the Toolkit was initialised with.  # noqa: E501
        self._factor_correlations = apply_rounding(
            fama_and_french_period, rounding if rounding is not None else self._rounding
        ).loc[self._start_date : self._end_date]

        return self._factor_correlations

    @handle_errors
    def get_fama_and_french_model(
        self,
        period: str | None = None,
        method: str = "multi",
        factors_to_calculate: list[str] | None = None,
        include_daily_residuals: bool = False,
        region: str = "United States",
        rolling: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Fama and French five factor model: how much of each asset's excess return the
        market, size, value, profitability and investment factors explain, and its exposure to each of
        them.

        The Fama and French 5 Factor model is a widely used financial model that helps estimate the expected return
        of financial assets, such as stocks or portfolios, based on five key factors:

            - Market Risk Premium (Mkt-RF): Represents the additional return that investors expect to earn for taking
            on the risk of investing in the overall market as opposed to a risk-free asset.
            - Size Premium (SMB): Reflects the historical excess return of small-cap stocks over large-cap stocks.
            - Value Premium (HML): Captures the historical excess return of value stocks over growth stocks.
            - Profitability (RMW): Measures the historical excess return of high profitability stocks over
            low profitability stocks.
            - Investment (CMA): Quantifies the historical excess return of low investment stocks over
            high investment stocks.

        The model can perform both a Simple Linear Regression on each factor as well as a Multi Linear Regression
        which includes all factors. Generally, a multi linear regression is applied but if you wish to see individual
        R-squared values for each factor you can select the simple linear regression method.

        The model performs a Linear Regression on each factor and defines the regression parameters and residuals
        for each asset over time based on its exposure to these factors.

        The regression formula is as follows for the Multi Linear Regression:

            - Excess Return = Intercept + Beta1 * Mkt-RF + Beta2 * SMB + Beta3 * HML + Beta4 * RMW +
              Beta5 * CMA + Residuals

        And the following for the Simple Linear Regression:

            - Excess Return = Intercept + Slope * Factor Value + Residuals

        So for a given factor, it should hold that the Excess Return equals the entire regression. Note that in this
        calculation the Excess Return refers to the Asset Return minus the Risk Free Rate as reported in the Fama and
        French dataset and will not be the same as the defined Excess Return in the historical data given that this is
        based on the Risk Free Rate defined in the initialization.

        The regression is estimated on the daily observations falling inside each period, so its Intercept, Slope and
        Residuals are all on a daily scale. The `Factor Value` and `Residuals` columns reported by the Simple Linear
        Regression therefore describe the **last daily observation** within the period, which is the only reading for
        which the identity above holds — they are not period-aggregated quantities.

        What is relevant to look at is the influence these factors have on each stock and how much each factor explains
        the stock return. E.g. you will generally see a pretty high influence (Beta or Slope) for the Market Risk Premium
        (Mkt-RF) factor as this is the main factor that explains the stock return (as also prevalent in the CAPM).
        The other factors can fluctuate greatly between stocks depending on which stocks you look at.

        The slopes are read as tilts: a positive HML slope is a value tilt and a negative one a growth tilt, a
        negative SMB slope marks a large-cap profile, and RMW and CMA show exposure to profitable and conservatively
        investing firms. With about 250 daily observations per year the slopes of the smaller factors are noisy, so
        only loadings that are large and persist across periods say much about a stock or fund. The R Squared shows
        how much of the daily excess return the factors explain; a low value leaves most of the return to
        stock-specific news.

        Also known as: Fama-French model, three-factor model, five-factor model, FF3, FF5,
        factor model alpha, multi-factor Jensen's alpha.

        Referenced against:
        - pyfolio, rolling_regression: https://github.com/stefan-jansen/pyfolio-reloaded/blob/0.10.2/src/pyfolio/timeseries.py#L550

        Shares similar logic:
        - statsmodels, RollingOLS: https://www.statsmodels.org/stable/generated/statsmodels.regression.rolling.RollingOLS.html

        Args:
            period (str, optional): The period for the calculation. Options are: "weekly", "monthly",
                "quarterly", "yearly". Defaults to "quarterly" if the Toolkit is initialised with
                quarterly=True, otherwise "yearly".
            method (str, optional): The regression. Options are: "multi" (one regression on all factors
                together), "simple" (a separate regression per factor). Defaults to "multi".
            factors_to_calculate (list of str, optional): List of factors to calculate scores and residuals for.
                Defaults to ["Mkt-RF", "SMB", "HML", "RMW", "CMA"].
            include_daily_residuals (bool, optional): Whether to also return the pointwise (daily)
                regression residuals as a second DataFrame. Defaults to False.
            region (str, optional): The region whose factors to use (the factors of Ken French's library
                for that region's stocks). Options are: "United States", "Developed", "Developed ex US",
                "Europe", "Japan", "Asia Pacific ex Japan", "North America". Every region's factors are
                U.S. dollar returns with the U.S. one-month Treasury bill as the risk-free rate (RF),
                so a ticker quoted in another currency is best analysed with its U.S. dollar returns;
                otherwise the exchange rate's moves end up in the factor loadings, mostly the market
                beta. Defaults to "United States".
            rolling (int | None, optional): The number of trading days of a rolling regression: the multi
                linear regression is fitted on the last `rolling` daily observations at every date instead of
                once per period, so the factor loadings show how the exposures move over time (e.g. 126 for
                the six months pyfolio uses). Only with the multi method. At least 2. Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratio values. Defaults to False.
            lag (int or list of int, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Fama and French 5 Factor model scores for the specified assets, with a
            Multi Index of (ticker, parameter) as the columns and one row per period. When
            `include_daily_residuals` is True a tuple of (scores, residuals) is returned instead.

        Notes:
        - The dataset from Fama and French is not always fully up to date. Therefore, some periods could be excluded.
        - Daily Fama and French results is not an option as it would attempt to do a linear regression on a
        single data point which will not give any meaningful results.
        - The method retrieves historical data and calculates regression parameters and residuals for each asset.
        - With `rolling` the result has a row per date, the last date of each window, and leaves out the residuals.
        It agrees with pyfolio's timeseries.rolling_regression of the excess returns on the factors with a
        rolling_window one shorter (pyfolio's windows hold rolling_window + 1 days).
        - The risk-free rate is typically represented by the return of a risk-free investment, such as a Treasury bond.
        In this case, the Risk Free Rate from the Fama and French dataset is used.
        - With growth, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["IWM", "VTV", "MTUM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        # Calculate Fama and French 5 Factor model scores
        toolkit.performance.get_fama_and_french_model()["IWM"].drop(columns="Mean Squared Error (MSE)")
        ```

        Which returns:

        |      |   Intercept |   Mkt-RF Slope |   SMB Slope |   HML Slope |   RMW Slope |   CMA Slope |   R Squared |
        |:-----|------------:|---------------:|------------:|------------:|------------:|------------:|------------:|
        | 2021 |     -0.0003 |         1.0529 |      0.8267 |      0.0728 |     -0.1204 |     -0.0334 |      0.9868 |
        | 2022 |     -0.0001 |         1.0343 |      0.893  |      0.0645 |     -0.0553 |      0.0654 |      0.9917 |
        | 2023 |     -0.0001 |         1.0292 |      0.8647 |      0.141  |     -0.0731 |      0.0794 |      0.9871 |
        | 2024 |      0      |         1.0396 |      0.8723 |      0.1568 |     -0.0685 |      0.0143 |      0.9838 |
        | 2025 |      0      |         0.9702 |      0.8649 |      0.1317 |     -0.1871 |     -0.0235 |      0.9826 |
        """
        check_rolling(rolling, self._historical_data, "daily")
        check_option("method", method, ["simple", "multi"])

        factors_to_calculate = (
            factors_to_calculate
            if factors_to_calculate
            else ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
        )

        for factor in factors_to_calculate:
            if factor not in ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]:
                raise InvalidArgumentError(
                    f"Factor {factor} is not a valid factor. Please select from: Mkt-RF, SMB, HML, RMW, CMA."
                )

        if rolling:
            if method != "multi":
                raise InvalidArgumentError(
                    "A rolling regression requires method='multi'. Please pass method='multi' or leave out rolling."
                )

            return self._get_rolling_factor_model(
                self._get_fama_and_french_dataset(region),
                factors_to_calculate,
                rolling,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            )

        period = period if period else "quarterly" if self._quarterly else "yearly"
        returns = self._get_column(period, "Return", within_period=True)

        fama_and_french_dataset = self._get_fama_and_french_dataset(region)

        fama_and_french_period = determine_within_dataset(
            fama_and_french_dataset, period, correlation=False
        )

        merged_df = fama_and_french_period.merge(
            returns, left_index=True, right_index=True
        )
        factor_scores: dict = {}
        daily_residuals: dict = {}

        # The first index level repeats once per row, so it has to be de-duplicated before iterating or every period is regressed once per day it contains.  # noqa: E501
        dataset_periods = merged_df.index.get_level_values(0).unique()

        logger.info(
            "Calculating %s Factor Exposures",
            "Multi" if method == "multi" else "Individual",
        )
        for ticker in self._tickers_with_returns:
            factor_scores[ticker] = {}
            daily_residuals[ticker] = {}

            if method == "multi":
                for dataset_period in dataset_periods:
                    factor_data = merged_df.loc[dataset_period][factors_to_calculate]
                    excess_returns = (
                        merged_df.loc[dataset_period][ticker]
                        - merged_df.loc[dataset_period]["RF"]
                    )

                    (
                        factor_scores[ticker][dataset_period],
                        daily_residuals[ticker][dataset_period],
                        error_message,
                    ) = performance_model.get_fama_and_french_model_multi(
                        excess_returns=excess_returns, factor_dataset=factor_data
                    )

                    if error_message:
                        logger.warning(
                            "%s for %s in %s.", error_message, ticker, dataset_period
                        )

            elif method == "simple":
                for factor in factors_to_calculate:
                    factor_scores[ticker][factor] = {}
                    daily_residuals[ticker][factor] = {}
                    for dataset_period in dataset_periods:
                        factor_data = merged_df.loc[dataset_period][factor]
                        excess_returns = (
                            merged_df.loc[dataset_period][ticker]
                            - merged_df.loc[dataset_period]["RF"]
                        )

                        (
                            factor_scores[ticker][factor][dataset_period],
                            daily_residuals[ticker][factor][dataset_period],
                        ) = performance_model.get_fama_and_french_model_single(
                            excess_returns=excess_returns, factor=factor_data
                        )

                        # The regression is fitted on the observations within the period, so the reported Factor Value and Residual describe the last observation of that period, on the same scale.  # noqa: E501
                        factor_scores[ticker][factor][dataset_period][
                            "Factor Value"
                        ] = factor_data.iloc[-1]

                        factor_scores[ticker][factor][dataset_period][
                            "Residuals"
                        ] = excess_returns.iloc[-1] - (
                            factor_scores[ticker][factor][dataset_period]["Slope"]
                            * factor_data.iloc[-1]
                            + factor_scores[ticker][factor][dataset_period]["Intercept"]
                        )

        if method == "multi":
            fama_and_french_model = pd.DataFrame.from_dict(
                {
                    (ticker_name, dataset_period): value
                    for ticker_name, factor_scores_ticker in factor_scores.items()
                    for dataset_period, value in factor_scores_ticker.items()
                },
                orient="index",
            )

            fama_and_french_model = fama_and_french_model.unstack(level=0, sort=False)
            fama_and_french_model.columns = to_multi_index(
                fama_and_french_model.columns
            ).swaplevel(0, 1)

            # Sort the DataFrame with respect to the original column order
            tickers_column_order = fama_and_french_model.columns.get_level_values(
                0
            ).unique()
            parameters_column_order = fama_and_french_model.columns.get_level_values(
                1
            ).unique()

            fama_and_french_model = (
                fama_and_french_model.sort_index(axis=1)
                .reindex(tickers_column_order, level=0, axis=1)
                .reindex(parameters_column_order, level=1, axis=1)
            )
        else:
            fama_and_french_model = pd.DataFrame.from_dict(
                {
                    (dataset_period, factor, ticker_name): value
                    for ticker_name, factor_scores_ticker in factor_scores.items()
                    for factor, factor_scores_factor in factor_scores_ticker.items()
                    for dataset_period, value in factor_scores_factor.items()
                },
                orient="index",
            )

            # Sort the DataFrame with respect to the original column order
            parameters_column_order = fama_and_french_model.columns.unique()
            factor_column_order = fama_and_french_model.index.get_level_values(
                1
            ).unique()
            ticker_column_order = fama_and_french_model.index.get_level_values(
                2
            ).unique()

            fama_and_french_model = fama_and_french_model.stack().unstack(
                level=[2, 1, 3]
            )

            fama_and_french_model = (
                fama_and_french_model.sort_index(axis=1)
                .reindex(parameters_column_order, level=2, axis=1)
                .reindex(factor_column_order, level=1, axis=1)
                .reindex(ticker_column_order, level=0, axis=1)
            )

        self._fama_and_french_model = apply_rounding(
            fama_and_french_model, rounding if rounding is not None else self._rounding
        ).loc[self._start_date : self._end_date]

        if include_daily_residuals:
            if method == "multi":
                daily_residuals_df = pd.DataFrame.from_dict(
                    {
                        (ticker_name, dataset_period): value
                        for ticker_name, residuals_ticker in daily_residuals.items()
                        for dataset_period, value in residuals_ticker.items()
                    },
                    orient="index",
                )

                daily_residuals_df = (
                    daily_residuals_df.stack()
                    .unstack(level=0)
                    .sort_index(axis=1, sort_remaining=False)
                )
            else:
                daily_residuals_df = pd.DataFrame.from_dict(
                    {
                        (dataset_period, factor, ticker_name): value
                        for ticker_name, residuals_ticker in daily_residuals.items()
                        for factor, residuals_factor in residuals_ticker.items()
                        for dataset_period, value in residuals_factor.items()
                    },
                    orient="index",
                )

                daily_residuals_df = (
                    daily_residuals_df.unstack(level=[1, 2])
                    .stack(level=0)
                    .sort_index(axis=1, sort_remaining=False)
                )

            # Reshaping pairs every period with every date in the whole sample, so all but the dates actually inside a period come back as fully empty rows. Dropping them (and sorting what is left) is what makes the date slice below possible at all: it otherwise raises on the non-monotonic index of repeated dates.  # noqa: E501
            daily_residuals_df = (
                daily_residuals_df.dropna(how="all")
                .sort_index()
                .reset_index(level=0, drop=True)
            )

            self._fama_and_french_residuals = apply_rounding(
                daily_residuals_df, rounding if rounding is not None else self._rounding
            ).loc[self._start_date : self._end_date]

            return (
                self._fama_and_french_model,
                self._fama_and_french_residuals,
            )

        return finalize_dataset(
            dataset=self._fama_and_french_model,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_portfolio
    @handle_errors
    def get_carhart_four_factor_model(
        self,
        period: str | None = None,
        region: str = "United States",
        rolling: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Carhart four factor model: how much of each asset's excess return the market,
        size, value and momentum factors explain, and its exposure to each of them.

        The Carhart Four Factor model extends the Fama and French Three Factor model with a
        momentum factor, based on the observation that stocks with high prior returns (winners)
        tend to keep outperforming stocks with low prior returns (losers) over the medium term:

            - Market Risk Premium (Mkt-RF): The excess return of the market over the risk-free rate.
            - Size Premium (SMB): The historical excess return of small-cap stocks over large-cap stocks.
            - Value Premium (HML): The historical excess return of value stocks over growth stocks.
            - Momentum (MOM): The historical excess return of prior winner stocks over prior loser stocks.

        The model performs a Multi Linear Regression (ordinary least squares) on all four factors and defines
        the regression parameters for each asset over time based on its exposure to these factors.

        The formula is as follows:

        - Excess Return = Daily Asset Return - Daily Risk Free Rate (RF, from the Fama and French dataset)
        - Excess Return = Intercept + Beta1 * Mkt-RF + Beta2 * SMB + Beta3 * HML + Beta4 * MOM + Residuals,
          fitted on the daily returns within each period (e.g. each quarter)

        For more information about the method, see the following paper:

        - Carhart, M.M. (1997). "On Persistence in Mutual Fund Performance." The Journal of
        Finance, 52(1), 57-82.

        Also known as: Carhart model, four-factor model, momentum-augmented Fama-French model.

        Referenced against:
        - pyfolio, rolling_regression: https://github.com/stefan-jansen/pyfolio-reloaded/blob/0.10.2/src/pyfolio/timeseries.py#L550

        Shares similar logic:
        - statsmodels, RollingOLS: https://www.statsmodels.org/stable/generated/statsmodels.regression.rolling.RollingOLS.html

        Args:
            period (str, optional): The period for the calculation. Options are: "weekly", "monthly",
                "quarterly", "yearly". Defaults to "quarterly" if the Toolkit is initialised with
                quarterly=True, otherwise "yearly".
            region (str, optional): The region whose factors to use (the factors of Ken French's library
                for that region's stocks). Options are: "United States", "Developed", "Developed ex US",
                "Europe", "Japan", "Asia Pacific ex Japan", "North America". Every region's factors are
                U.S. dollar returns with the U.S. one-month Treasury bill as the risk-free rate (RF),
                so a ticker quoted in another currency is best analysed with its U.S. dollar returns;
                otherwise the exchange rate's moves end up in the factor loadings, mostly the market
                beta. Defaults to "United States".
            rolling (int | None, optional): The number of trading days of a rolling regression: the regression
                is fitted on the last `rolling` daily observations at every date instead of once per period, so
                the factor loadings show how the exposures move over time (e.g. 126 for the six months pyfolio
                uses). At least 2. Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratio values. Defaults to False.
            lag (int or list of int, optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Carhart Four Factor model scores for the specified assets, with a Multi
            Index of (ticker, parameter) as the columns and one row per period.

        Notes:
        - The dataset from Ken French is not always fully up to date. Therefore, some periods could be excluded.
        - Daily Carhart results is not an option as it would attempt to do a linear regression on a single data
        point which will not give any meaningful results.
        - The factors come from the Fama and French **three** factor file rather than the five factor file
        used by `get_fama_and_french_model`. The two files agree on Mkt-RF, HML and RF but not on SMB: the
        five factor SMB averages three separate size legs, while Carhart (1997) extends the three factor
        model and therefore requires the three factor SMB.
        - The risk-free rate is the Risk Free Rate reported in the Fama and French dataset (used here, rather
        than the Toolkit's own risk-free rate, to stay consistent with the momentum factor's construction).
        - With `rolling` the result has a row per date, the last date of each window. It agrees with pyfolio's
        timeseries.rolling_regression of the excess returns on the factors with a rolling_window one shorter
        (pyfolio's windows hold rolling_window + 1 days).
        - With growth, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["XOM", "HD", "ADBE"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        toolkit.performance.get_carhart_four_factor_model(period="quarterly")["XOM"].drop(
            columns="Mean Squared Error (MSE)"
        )
        ```

        Which returns:

        |        |   Intercept |   Mkt-RF Slope |   SMB Slope |   HML Slope |   MOM Slope |   R Squared |
        |:-------|------------:|---------------:|------------:|------------:|------------:|------------:|
        | 2025Q1 |      0.0021 |         0.7335 |      0.2739 |      0.4818 |     -0.7225 |      0.317  |
        | 2025Q2 |     -0.0002 |         0.8881 |     -0.7307 |      1.6062 |     -0.7248 |      0.6805 |
        | 2025Q3 |      0.0003 |         0.3232 |     -0.1805 |      0.6496 |     -0.2369 |      0.2179 |
        | 2025Q4 |      0.0005 |         0.4852 |     -0.2193 |      0.6778 |     -0.0933 |      0.1938 |
        """
        # Carhart (1997) extends the *three* factor model. The three and five factor files share Mkt-RF, HML and RF but their SMB series differ, so the three factor file is the correct source here.  # noqa: E501
        check_rolling(rolling, self._historical_data, "daily")
        three_factor_dataset = (
            performance_model.obtain_fama_and_french_three_factor_dataset(region=region)
        )
        momentum_dataset = performance_model.obtain_carhart_momentum_dataset(
            region=region
        )

        carhart_dataset = three_factor_dataset[["Mkt-RF", "SMB", "HML", "RF"]].merge(
            momentum_dataset, left_index=True, right_index=True
        )
        carhart_dataset = carhart_dataset.rename(columns={"Mom": "MOM"})

        factors_to_calculate = ["Mkt-RF", "SMB", "HML", "MOM"]

        if rolling:
            return self._get_rolling_factor_model(
                carhart_dataset,
                factors_to_calculate,
                rolling,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            )

        period = period if period else "quarterly" if self._quarterly else "yearly"
        returns = self._get_column(period, "Return", within_period=True)

        carhart_period = determine_within_dataset(
            carhart_dataset, period, correlation=False
        )

        merged_df = carhart_period.merge(returns, left_index=True, right_index=True)
        factor_scores: dict = {}

        # The first index level repeats once per row, so it has to be de-duplicated before iterating or every period is regressed once per day it contains.  # noqa: E501
        dataset_periods = merged_df.index.get_level_values(0).unique()

        logger.info("Calculating Carhart Four Factor Exposures")
        for ticker in self._tickers_with_returns:
            factor_scores[ticker] = {}

            for dataset_period in dataset_periods:
                factor_data = merged_df.loc[dataset_period][factors_to_calculate]
                excess_returns = (
                    merged_df.loc[dataset_period][ticker]
                    - merged_df.loc[dataset_period]["RF"]
                )

                (
                    factor_scores[ticker][dataset_period],
                    _,
                    error_message,
                ) = performance_model.get_fama_and_french_model_multi(
                    excess_returns=excess_returns, factor_dataset=factor_data
                )

                if error_message:
                    logger.warning(
                        "%s for %s in %s.", error_message, ticker, dataset_period
                    )

        carhart_model = pd.DataFrame.from_dict(
            {
                (ticker_name, dataset_period): value
                for ticker_name, factor_scores_ticker in factor_scores.items()
                for dataset_period, value in factor_scores_ticker.items()
            },
            orient="index",
        )

        carhart_model = carhart_model.unstack(level=0, sort=False)
        carhart_model.columns = to_multi_index(carhart_model.columns).swaplevel(0, 1)

        tickers_column_order = carhart_model.columns.get_level_values(0).unique()
        parameters_column_order = carhart_model.columns.get_level_values(1).unique()

        carhart_model = (
            carhart_model.sort_index(axis=1)
            .reindex(tickers_column_order, level=0, axis=1)
            .reindex(parameters_column_order, level=1, axis=1)
        )

        self._carhart_four_factor_model = apply_rounding(
            carhart_model, rounding if rounding is not None else self._rounding
        ).loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=self._carhart_four_factor_model,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_style_analysis(
        self,
        style_tickers: str | list[str] | None = None,
        period: str = "monthly",
        rolling: int | None = None,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the returns-based style analysis (Sharpe, 1992) of every ticker: the mix of
        style indices, such as growth and value, large and small caps or bonds, whose returns
        track the ticker's returns most closely. The weights are found by minimizing the
        variance of the difference between the ticker's return and the weighted style returns,
        with every weight between 0 and 1 and the weights adding up to 1, so they read as the
        effective asset mix of a fund without knowing its holdings.

        The R Squared is the share of the ticker's variance the styles explain; what they do
        not explain is the Selection Return (the return of the manager's own choices, annualized)
        with its volatility, the Tracking Error. These three are in-sample: they are measured
        over the same returns the weights are fitted to, which flatters the fit, as the weights
        are chosen to explain exactly those returns.

        A rolling window estimates the weights over a set number of periods instead, which shows how
        a fund's style drifts over time, and adds Sharpe's out-of-sample measure of selection:
        the Out-of-Sample Selection Return of a date is the ticker's return of that period minus
        the return of that period of the style mix estimated over the window that ends one
        period earlier (Sharpe uses the previous 60 months). It is a return per period, not
        annualized, and NaN for the first window, which has no earlier estimate. Its average
        shows whether the manager added value beyond a passive mix of the styles the fund was
        known to follow.

        By default the styles are Sharpe's asset classes as ETFs: IWD and IWF for US large cap
        value and growth, IWN and IWO for small cap value and growth, AGG for bonds and BIL for
        cash. Style tickers that are not in the Toolkit are retrieved for the same dates, and
        every Toolkit ticker that is not a style is analyzed against them. "Benchmark" can be
        one of them.

        For more information, see the following paper:

        - Sharpe, W.F. (1992). "Asset Allocation: Management Style and Performance
          Measurement." The Journal of Portfolio Management, 18(2), 7-19.

        Also known as: returns-based style analysis, RBSA, style regression, effective asset mix,
        selection return.

        Args:
            style_tickers (str | list[str] | None, optional): The tickers of the style indices, as a
                list or a comma-separated string. Defaults to None, which uses IWD, IWF, IWN, IWO,
                AGG and BIL.
            period (str, optional): The return frequency. Options are: "daily", "weekly", "monthly",
                "quarterly", "yearly". Defaults to "monthly".
            rolling (int | None, optional): The number of periods of a rolling window to estimate
                the weights over, which adds the Out-of-Sample Selection Return. At least 2. Defaults to None,
                which uses every period.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the rounding of the Toolkit.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per ticker the weight of every style, the R Squared, the Selection Return
            and the Tracking Error, or with a rolling window the same and the Out-of-Sample Selection
            Return per date (the last date of every window) and ticker. A ticker with no more
            returns than there are styles (in a window) is NaN.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["ARKK", "BRK-B"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2019-01-01",
            end_date="2025-12-31",
        )

        toolkit.performance.get_style_analysis()
        ```

        Which returns:

        |       |    IWD |    IWF |   IWN |    IWO |   AGG |    BIL |   R Squared |   Selection Return |   Tracking Error |
        |:------|-------:|-------:|------:|-------:|------:|-------:|------------:|-------------------:|-----------------:|
        | ARKK  | 0      | 0.0091 |     0 | 0.9909 |     0 | 0      |      0.6123 |             0.0638 |           0.2632 |
        | BRK-B | 0.8793 | 0      |     0 | 0      |     0 | 0.1207 |      0.5662 |             0.0425 |           0.1304 |
        """
        check_rolling(rolling, self._historical_data, period)
        check_period(period, PERIODS_PER_YEAR)
        # A comma-separated string (as the MCP server passes a list) is split.
        style_tickers = (
            STYLE_TICKERS
            if style_tickers is None
            else (
                [
                    ticker.strip()
                    for ticker in style_tickers.split(",")
                    if ticker.strip()
                ]
                if isinstance(style_tickers, str)
                else list(style_tickers)
            )
        )
        returns = self._historical_data[period].loc[:, "Return"]
        if missing := [ticker for ticker in style_tickers if ticker not in returns]:
            if self._returns_function is None:
                raise ValueError(
                    f"The style tickers {', '.join(missing)} are not part of the Toolkit; add "
                    "them to its tickers."
                )
            # Sorted, as the dates only the style tickers have would otherwise follow all the others (pandas keeps the order of a PeriodIndex union) and break the date range and the rolling windows.  # noqa: E501
            returns = pd.concat(
                [
                    returns,
                    self._returns_function(missing, period).reindex(columns=missing),
                ],
                axis=1,
            ).sort_index()
        funds = [
            ticker
            for ticker in self._tickers_with_returns
            if ticker not in style_tickers
        ]
        if not funds:
            raise ValueError(
                "Every ticker is a style ticker, so there is nothing to analyze."
            )

        returns = returns.loc[
            self._start_date : self._end_date, [*funds, *style_tickers]
        ]
        # When the prices start within the range, their first period has no prior close and so no return for any ticker; leaving it in would give the first window one observation fewer than `rolling`.  # noqa: E501
        if (first_return := returns.first_valid_index()) is not None:
            returns = returns.loc[first_return:]
        periods_per_year = PERIODS_PER_YEAR[period]

        # A fund with too few returns (a recent listing, say) comes back as NaN in the model, so it does not empty the result of every other fund.  # noqa: E501
        if rolling is None:
            result = pd.DataFrame(
                {
                    fund: performance_model.get_style_analysis(
                        returns[fund], returns[style_tickers], periods_per_year
                    )
                    for fund in funds
                }
            ).T
        else:
            result = pd.concat(
                {
                    fund: performance_model.get_style_analysis(
                        returns[fund],
                        returns[style_tickers],
                        periods_per_year,
                        window=rolling,
                    )
                    for fund in funds
                },
                axis=1,
            )

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

    @handle_portfolio
    @handle_errors
    def get_alpha(
        self,
        period: str | None = None,
        show_full_results: bool = False,
        rolling: int | None = None,
        method: Literal["standard", "annualized", "jensen"] = "standard",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Alpha: how much an asset returned above (or below) its benchmark, as the plain
        difference of the returns, the annualized Active Premium or Jensen's Alpha above the return its
        Beta explains. A positive Alpha means the investment outperformed the benchmark, a negative one
        that it underperformed.

        The formula is as follows:

            - Alpha = Asset's Actual Return — Benchmark's Actual Return

        With the annualized method it is the Active Premium (or annualized active return) of Bacon
        (2008) instead, one value per ticker over the whole date range:

            - Active Premium = Annualized Return of the Asset − Annualized Return of the Benchmark
            - Annualized Return = PRODUCT(1 + Daily Return) ^ (Observations per Year / N) − 1

        Where N is the number of daily returns and the observations per year are 252 (365 for a
        ticker that trades on weekends, such as a cryptocurrency). The benchmark is annualized over
        the dates from the first to the last return of each ticker, so a ticker listed later is
        compared over its own history.

        With the Jensen method it is Jensen's Alpha (Jensen, 1968) instead, the return of each period
        above the return the Capital Asset Pricing Model (CAPM) expects given the asset's Beta, which
        assesses whether an investment outperformed or underperformed for its systematic risk:

            - Jensen's Alpha = Asset's Actual Return — [Risk-Free Rate + Beta * (Benchmark Return — Risk-Free Rate)]

        Where Beta is estimated from the daily returns within each period (see `get_beta`).

        See definition: https://en.wikipedia.org/wiki/Alpha_(finance) and
        https://en.wikipedia.org/wiki/Jensen%27s_alpha

        For more information, see the following sources:

        - Bacon, C.R. (2008). "Practical Portfolio Performance Measurement and Attribution."
        2nd edition, John Wiley & Sons.
        - Jensen, M.C. (1968). "The Performance of Mutual Funds in the Period 1945-1964." The Journal
        of Finance, 23(2), 389-416.

        Also known as: excess return, outperformance, active return, active premium, Jensen's alpha,
        Jensen alpha, jensens alpha, risk-adjusted excess return.

        Referenced against:
        - PerformanceAnalytics (R), Return.annualized: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/Return.annualized.R#L39

        Shares similar logic:
        - PerformanceAnalytics (R), CAPM.jensenAlpha: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/CAPM.jensenAlpha.R#L34
        - empyrical, alpha: https://github.com/quantopian/empyrical/blob/0.5.5/empyrical/stats.py#L1154
        - empyrical, annual_return: https://github.com/quantopian/empyrical/blob/0.5.5/empyrical/stats.py#L408
        - empyrical, roll_alpha_beta: https://github.com/quantopian/empyrical/blob/0.5.5/empyrical/stats.py#L1058
        - PerformanceAnalytics (R), CAPM.beta: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/CAPM.beta.R#L89
        - empyrical, beta: https://github.com/quantopian/empyrical/blob/0.5.5/empyrical/stats.py#L1314
        - PerformanceAnalytics (R), ActivePremium: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/ActivePremium.R#L35

        Args:
            period (str, optional): The period to use for the calculation. Options are: "daily", "weekly",
                "monthly", "quarterly", "yearly". Defaults to "quarterly" if the Toolkit is initialised with
                quarterly=True, otherwise "yearly".
            show_full_results (bool, optional): Whether to show the actual and benchmark returns next to the
                Alpha. Only used when method="standard" or "annualized". Defaults to False.
            rolling (int, optional): The rolling window size to use for the calculation, in periods, across
            the full return history instead of per `period`. With the standard method Alpha is the rolling
            mean excess return over this many periods; with the Jensen method it is the window of the Beta,
            estimated from the period returns of the last `rolling` periods instead of from the daily
            returns within each period. Not used with the annualized method. At least 2. Defaults to None.
            method (str, optional): The Alpha. Options are: "standard" (the difference of the returns of
                each period), "annualized" (the Active Premium, the difference of the annualized returns
                over the whole date range, one value per ticker; `period` and `rolling` are then not used),
                "jensen" (Jensen's Alpha, the return of each period above the CAPM expected return).
                Defaults to "standard".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Alpha values. With the annualized method a Series with the Active Premium per
            ticker, or with show_full_results=True a DataFrame with the annualized return of each
            ticker and of the benchmark next to it. With the Jensen method Jensen's Alpha values, with
            the periods without a value left out.

        Notes:
        - The method retrieves historical data and calculates the Alpha for each asset in the Toolkit instance.
        - The annualized method agrees with PerformanceAnalytics' ActivePremium on the daily returns (252 observations
        a year). PerformanceAnalytics compares on the dates both have a return; the Toolkit annualizes each on
        its own calendar within the ticker's history, so a cryptocurrency keeps its weekend returns.
        - With the Jensen method, a daily period has no Beta within it, as the standard deviation of a single day is
        close to zero, so it gives no useful insights. The risk-free rate is the Toolkit's risk free rate, the 10-year
        US Treasury yield by default. Many textbook definitions (Sharpe, 1994; the Ken French factors) use a
        short-term Treasury bill instead, which the Toolkit uses when initialised with the 13-week Treasury bill as
        its risk free rate.
        - With growth, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["COST", "WMT", "TGT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2020-01-01",
            end_date="2025-12-31",
        )

        toolkit.performance.get_alpha()
        ```

        Which returns:

        | Date   |    COST |     WMT |     TGT |
        |:-------|--------:|--------:|--------:|
        | 2021   |  0.2309 | -0.2676 |  0.0418 |
        | 2022   | -0.0087 |  0.1772 | -0.1606 |
        | 2023   |  0.2282 | -0.133  | -0.2753 |
        | 2024   |  0.1474 |  0.4911 | -0.2715 |
        | 2025   | -0.2311 |  0.0677 | -0.4221 |
        """
        check_rolling(
            rolling,
            self._historical_data,
            period or ("quarterly" if self._quarterly else "yearly"),
        )
        check_option("method", method, ("standard", "annualized", "jensen"))

        if method == "jensen":
            period = period if period else "quarterly" if self._quarterly else "yearly"

            # With a rolling window the Beta comes from the period returns of the window,
            # otherwise from the daily returns within each period.
            within_period = not rolling
            returns = self._get_column(period, "Return", within_period=within_period)
            benchmark_returns = self._get_column(
                period, "Return", within_period=within_period, benchmark=True
            )
            beta = (
                performance_model.get_beta(returns, benchmark_returns, window=rolling)
                if rolling
                else performance_model.get_beta(returns, benchmark_returns)
            )

            jensens_alpha = performance_model.get_jensens_alpha(
                self._get_column(period, "Return", within_period=False),
                self._risk_free_rate_data[period],
                beta,
                self._get_column(period, "Return", within_period=False, benchmark=True),
            )

            return finalize_dataset(
                dataset=jensens_alpha,
                start_date=self._start_date,
                end_date=self._end_date,
                default_rounding=self._rounding,
                growth=growth,
                lag=lag,
                rounding=rounding,
                standardize=standardize,
                axis="rows",
                row_slice=True,
                dropna=True,
            )

        if method == "annualized":
            daily_returns, annualized_return = self._get_whole_sample_returns()
            benchmark_daily_returns, _ = self._get_whole_sample_returns(benchmark=True)
            benchmark_observations_per_year = get_observations_per_year(
                benchmark_daily_returns
            )

            # The benchmark over the history of each ticker.
            benchmark_annualized_return = pd.Series(
                {
                    ticker: performance_model.get_annualized_return(
                        benchmark_daily_returns.loc[
                            daily_returns[ticker]
                            .first_valid_index() : daily_returns[ticker]
                            .last_valid_index()
                        ],
                        benchmark_observations_per_year,
                    )
                    for ticker in daily_returns.columns
                },
                dtype=float,
            )
            active_premium = annualized_return - benchmark_annualized_return

            if show_full_results:
                active_premium = pd.concat(
                    {
                        "Annualized Return": annualized_return,
                        "Benchmark Annualized Return": benchmark_annualized_return,
                        "Active Premium": active_premium,
                    },
                    axis=1,
                )

            return finalize_dataset(
                dataset=active_premium,
                start_date=self._start_date,
                end_date=self._end_date,
                default_rounding=self._rounding,
                rounding=rounding,
                apply_slice=False,
            )

        period = period if period else "quarterly" if self._quarterly else "yearly"
        returns = self._get_column(period, "Return", within_period=False)
        benchmark_returns = self._get_column(
            period, "Return", within_period=False, benchmark=True
        )

        if rolling:
            alpha = performance_model.get_alpha(
                returns, benchmark_returns, window=rolling
            )
        else:
            alpha = performance_model.get_alpha(returns, benchmark_returns)

        alpha = apply_rounding(
            alpha, rounding if rounding is not None else self._rounding
        ).loc[self._start_date : self._end_date]

        alpha = alpha.dropna(how="all", axis=0)

        if show_full_results:
            full_results = pd.concat([returns, benchmark_returns, alpha], axis=1).loc[
                self._start_date : self._end_date
            ]
            full_results.columns = (
                [f"Actual Return {ticker}" for ticker in self._tickers_with_returns]
                + ["Benchmark Returns"]
                + [f"Alpha {ticker}" for ticker in self._tickers_with_returns]
            )

            return full_results

        return finalize_dataset(
            dataset=alpha,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_portfolio
    @handle_errors
    def get_treynor_ratio(
        self,
        period: str | None = None,
        rolling: int | None = None,
        modified: bool = False,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Treynor Ratio: the excess return an asset earns per unit of market risk (Beta),
        which judges its return against the risk that diversification cannot remove. Also known as
        Treynor's Measure or the Reward-to-Volatility Ratio, it evaluates the performance of investments
        in relation to their market risk exposure.

        The formula is as follows:

        - Treynor Ratio = (Portfolio's Return — Risk-Free Rate) / Portfolio Beta

        The modified Treynor ratio of Bacon (2008) divides by the systematic risk instead, Beta times
        the volatility of the benchmark, which expresses the excess return per unit of market-driven
        volatility like a Sharpe ratio:

        - Modified Treynor Ratio = (Portfolio's Return — Risk-Free Rate) / (Portfolio Beta × Benchmark Volatility)

        Where the benchmark volatility is the standard deviation of its daily returns within the
        period scaled to the period (SQRT(252) for a year), or the standard deviation of its returns
        over the rolling window.

        See definition: https://en.wikipedia.org/wiki/Treynor_ratio

        For more information, see the following sources:

        - Treynor, J.L. (1965). "How to Rate Management of Investment Funds." Harvard Business
        Review, 43(1), 63-75.
        - Bacon, C.R. (2008). "Practical Portfolio Performance Measurement and Attribution."
        2nd edition, John Wiley & Sons, p. 77.

        Also known as: reward-to-volatility ratio.

        Referenced against:
        - PerformanceAnalytics (R), TreynorRatio (modified): https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/TreynorRatio.R#L47

        Shares similar logic:
        - quantstats, treynor_ratio: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L1469
        - empyrical, roll_alpha_beta: https://github.com/quantopian/empyrical/blob/0.5.5/empyrical/stats.py#L1058
        - PerformanceAnalytics (R), CAPM.beta: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/CAPM.beta.R#L89
        - empyrical, beta: https://github.com/quantopian/empyrical/blob/0.5.5/empyrical/stats.py#L1314

        Args:
            period (str, optional): The period to use for the calculation. Options are: "daily", "weekly",
                "monthly", "quarterly", "yearly". Defaults to "quarterly" if the Toolkit is initialised with
                quarterly=True, otherwise "yearly".
            rolling (int, optional): The rolling window size to use for the Beta component of the
            calculation. If set, Beta is estimated over a rolling window of this many periods across
            the full return history instead of per `period`. At least 2. Defaults to None.
            modified (bool, optional): Whether to return the modified Treynor ratio, the excess return
            divided by Beta times the benchmark volatility. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Treynor Ratio values.

        Notes:
        - Daily Treynor Ratio is not an option as the standard deviation for 1 day is close to zero. Therefore,
        it does not give any useful insights.
        - The method retrieves historical data and calculates the TR for each asset in the Toolkit instance.
        - PerformanceAnalytics' TreynorRatio annualizes the excess return over the whole sample and estimates
        Beta on excess returns; the Toolkit takes the return and Beta of each period (Beta on the daily returns
        within it). The modified ratio follows the same formula.
        - The risk-free rate is the Toolkit's risk free rate, the 10-year US Treasury yield by default. Many
        textbook definitions (Sharpe, 1994; the Ken French factors) use a short-term Treasury bill instead, which
        the Toolkit uses when initialised with the 13-week Treasury bill as its risk free rate.
        - With growth, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["UNP", "UPS", "FDX"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2020-01-01",
            end_date="2025-12-31",
        )

        toolkit.performance.get_treynor_ratio()
        ```

        Which returns:

        | Date   |     UNP |     UPS |     FDX |
        |:-------|--------:|--------:|--------:|
        | 2021   |  0.2343 |  0.3679 | -0.0076 |
        | 2022   | -0.2787 | -0.2133 | -0.3425 |
        | 2023   |  0.2005 | -0.0898 |  0.4281 |
        | 2024   | -0.1635 | -0.339  |  0.1565 |
        | 2025   | -0.0065 | -0.265  |  0.008  |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"
        check_rolling(rolling, self._historical_data, period)

        if rolling:
            returns = self._get_column(period, "Return", within_period=False)
            benchmark_returns = self._get_column(
                period, "Return", within_period=False, benchmark=True
            )

            beta = performance_model.get_beta(
                returns, benchmark_returns, window=rolling
            )
        else:
            returns = self._get_column(period, "Return", within_period=True)
            benchmark_returns = self._get_column(
                period, "Return", within_period=True, benchmark=True
            )

            beta = performance_model.get_beta(returns, benchmark_returns)

        period_returns = self._get_column(period, "Return", within_period=False)
        risk_free_rate = self._risk_free_rate_data[period]

        treynor_ratio = performance_model.get_treynor_ratio(
            period_returns,
            risk_free_rate,
            beta,
            self._get_benchmark_volatility(period, rolling) if modified else None,
        )

        return finalize_dataset(
            dataset=treynor_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_portfolio
    @handle_errors
    def get_sharpe_ratio(
        self,
        period: str | None = None,
        rolling: int | None = None,
        method: Literal[
            "standard",
            "smart",
            "lo",
            "adjusted",
            "probabilistic",
            "deflated",
        ] = "standard",
        benchmark_sharpe_ratio: float = 0.0,
        trials_window: int | None = None,
        n_trials: int | None = None,
        lags: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        annualize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Sharpe ratio: the excess return an asset earns per unit of volatility, the
        risk-adjusted return that makes assets with different risks comparable.

        The Sharpe ratio is calculated as the difference between the expected return of the asset or portfolio
        and the risk-free rate of return, divided by the standard deviation of the asset or portfolio's excess return.
        It quantifies the amount of return generated for each unit of risk assumed, providing insights into the
        investment's performance relative to the risk taken.

        The formula is as follows:

            - Sharpe Ratio = Excess Return / Excess Standard Deviation

        By default one Sharpe ratio is reported per `period`, computed from the **daily**
        excess returns falling inside that period. For a given period, for example monthly,
        this translates into the following:

            - Sharpe Ratio = Average Daily Excess Return within the Month
              / Standard Deviation of the Daily Excess Returns within the Month

        For a rolling period, `period` instead sets the frequency of the returns themselves
        and the ratio is computed over a rolling window of `rolling` such returns:

            - Sharpe Ratio = Average Rolling Excess Return / Standard Deviation of Rolling Excess Returns

        Note that the Risk Free Rate is already subtracted from these returns. The Toolkit's
        risk free rate is the 10-year US Treasury yield by default, while Sharpe (1994) and the
        Ken French factor files use a short-term Treasury bill. Initialise the Toolkit with
        the 13-week Treasury bill as its risk free rate (the 5, 10 and 30 year yields are the other
        choices) to use a short rate instead.

        By default the result is **not annualized**: it is a per-observation Sharpe ratio, so a
        value computed from daily returns is roughly SQRT(252) smaller than the annualized figure
        usually quoted in the literature (SQRT(52), SQRT(12) and SQRT(4) for weekly, monthly
        and quarterly returns respectively). Annualizing multiplies by that factor, to
        compare against published annualized Sharpe ratios.

        The plain Sharpe ratio only looks at the mean and standard deviation of returns, implicitly
        assuming Gaussian, i.i.d. returns and ignoring how much uncertainty surrounds the estimate
        itself. The `method` parameter selects one of five corrections for that, each keeping the
        same excess returns and therefore the same underlying Sharpe ratio as its starting point:

        - The Smart Sharpe ratio of quantstats divides the Sharpe ratio by an
        autocorrelation penalty so that serially correlated returns (whose standard deviation
        understates the risk over several periods) no longer flatter it:

            - Smart Sharpe Ratio = SR / SQRT( 1 + 2 · Σ_{k=1}^{n−1} ((n − k) / n) · |ρ|^k )

        where n is the number of excess returns and ρ their first order autocorrelation (see
        `performance_model.get_autocorrelation_penalty`). It is a simplification of the Sharpe ratio of Lo (2002),
        which uses the autocorrelation at every lag rather than powers of the first one, and unlike
        that one it is only annualized on request.

        - The annualized Sharpe ratio of Lo (2002) corrects the annualization for
        serially correlated returns. SQRT(q) * SR assumes independent returns; with
        autocorrelated returns (smoothed prices of illiquid assets, hedge funds, momentum or
        mean reversion) the variance of a yearly return is no longer q times the variance of
        one return, and the yearly Sharpe ratio becomes:

            - SR(q) = η(q) * SR, η(q) = q / SQRT( q + 2 · Σ_{k=1}^{q−1} (q − k) · ρ_k )

        where q is the number of observations in a year and ρ_k the sample autocorrelation of
        the excess returns at lag k. Positive autocorrelation lowers the annualized ratio below
        SQRT(q) * SR, negative autocorrelation raises it. Sample autocorrelations at lags close
        to the sample size rest on a handful of pairs, so the sum runs over `lags` lags and takes
        the autocorrelations beyond them as 0. Each ρ_k is corrected for its small-sample bias
        −(n − k) / (n · (n − 1)) (Moran, 1948), which the (q − k) weights would otherwise turn
        into an overstated ratio on short samples, and fewer than 30 observations give NaN: a
        week or a month of daily returns, or a rolling window below 30, so use quarterly or
        yearly periods or a longer window. The result is always annualized.

        - The Adjusted Sharpe Ratio (ASR, Pezier & White, 2006) penalizes (or
        rewards) the Sharpe ratio for negative skewness and excess kurtosis using a
        Cornish-Fisher-style expansion, so that two strategies with the same Sharpe ratio but
        different tail shapes are no longer scored identically:

            - ASR = SR * [1 + (S / 6) * SR − ((K − 3) / 24) * SR^2]

        - The Probabilistic Sharpe Ratio (PSR) is the probability that the true
        (population) Sharpe ratio exceeds `benchmark_sharpe_ratio`, folding the skewness and
        (non-excess) kurtosis of the underlying returns into the standard error of the Sharpe ratio
        so that a short, lumpy sample no longer looks more convincing than it is:

            - PSR(SR*) = Φ( (SR̂ − SR*) · sqrt(n − 1) / sqrt(1 − γ₃·SR̂ + ((γ₄ − 1) / 4)·SR̂²) )

        - The Deflated Sharpe Ratio (DSR) is the Probabilistic Sharpe Ratio corrected
        for the fact that a reported Sharpe ratio is often the best of many strategy variations,
        parameter combinations, or lookback windows tried during a backtest (multiple testing /
        selection bias / "backtest overfitting"). It estimates the Sharpe ratio one would expect to
        observe purely by chance as the maximum of `n_trials` independent trials under the null
        hypothesis of no skill, and uses that expected maximum as the benchmark SR* in the
        Probabilistic Sharpe Ratio formula instead of a naive benchmark such as 0:

            - SR* = sqrt(Var[SR_trials]) · [ (1 − γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e)) ]

        Where SR̂ is the observed Sharpe ratio, S (γ₃) is the skewness and K (γ₄) the non-excess
        (raw) kurtosis of the same returns, n is the number of return observations, N is `n_trials`,
        Var[SR_trials] is the variance of the Sharpe ratios observed across those N trials,
        γ ≈ 0.5772 is the Euler-Mascheroni constant and Φ is the standard normal CDF. Since
        DSR = PSR(SR*), it is always less than or equal to the Probabilistic Sharpe Ratio computed
        against a benchmark of 0.

        This codebase does not track "N literal strategy trials" — there is no record of how many
        parameter combinations were tried before arriving at the current Toolkit configuration. As a
        documented approximation, `Var[SR_trials]` is estimated from the variance of an auxiliary
        *rolling* Sharpe ratio series (see `get_sharpe_ratio` with a window) computed over a
        `trials_window`-sized window across the full return history, and `n_trials` defaults to the
        number of non-overlapping `trials_window`-sized windows in that return history (at least 1):
        overlapping windows share most of their returns, so counting each of them as a trial would
        overstate N (Bailey and Lopez de Prado, 2014). That series is built from the
        same returns as the Sharpe ratio being deflated, so both are on the same scale: the daily
        excess returns (the observations within each period) for the Sharpe ratio per period, and
        the excess returns of `period` itself for a rolling Sharpe ratio. This treats each window
        as if it were one "trial" — a reasonable proxy for how dispersed the Sharpe ratio
        could plausibly have been under different choices, but not a substitute for passing the
        actual number of variations tried (via `n_trials`) when that is known, since the quality of
        the correction depends directly on it.

        See definition: https://en.wikipedia.org/wiki/Sharpe_ratio

        Also known as: risk-adjusted return, reward-to-variability ratio. The variants are also
        known as the Pezier and White Adjusted Sharpe Ratio (ASR), the Sharpe ratio significance
        probability (PSR) and the backtest overfitting or selection-bias-adjusted Sharpe ratio (DSR).

        Referenced against:
        - quantstats, smart_sharpe (with get_autocorrelation_penalty): https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L1000
        - quantstats, autocorr_penalty: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L875
        - quantstats, probabilistic_sortino_ratio: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L1404
        - quantstats, probabilistic_adjusted_sortino_ratio: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L1434

        Shares similar logic:
        - PerformanceAnalytics (R), Return.excess: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/Return.excess.R#L42
        - quantstats, rolling_sharpe: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L1030
        - PerformanceAnalytics (R), SharpeRatio: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/SharpeRatio.R#L80
        - empyrical, sharpe_ratio: https://github.com/quantopian/empyrical/blob/0.5.5/empyrical/stats.py#L652
        - quantstats, sharpe: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L938
        - PerformanceAnalytics (R), AdjustedSharpeRatio: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/AdjustedSharpeRatio.R#L35
        - quantstats, probabilistic_ratio: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L1297
        - quantstats, probabilistic_sharpe_ratio: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L1369
        - PerformanceAnalytics (R), ProbSharpeRatio: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/ProbSharpeRatio.R#L47
        - RiskLabAI, probabilistic_sharpe_ratio: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/backtest/probabilistic_sharpe_ratio.py#L16
        - ffn, calc_deflated_sharpe_ratio: https://github.com/pmorissette/ffn/blob/v1.4.1/ffn/core.py#L3077
        - ffn, calc_expected_max_sharpe: https://github.com/pmorissette/ffn/blob/v1.4.1/ffn/core.py#L3048

        Args:
            period (str, optional): The period to use for the calculation. Options are: "daily", "weekly",
                "monthly", "quarterly", "yearly". Defaults to "quarterly" if the Toolkit is initialised with
                quarterly=True, otherwise "yearly".
            rolling (int, optional): The rolling period to use for the calculation. If you select
            period = 'monthly' and set rolling to 12 you obtain the rolling 12-month Sharpe Ratio.
            At least 2. Defaults to None, which calculates the Sharpe ratio within each period.
            method (str, optional): Which Sharpe ratio to calculate, as described above. Options are:
                "standard" (the Sharpe ratio), "smart" (the Smart Sharpe ratio), "lo" (the annualized Sharpe
                ratio of Lo), "adjusted" (the Adjusted Sharpe Ratio), "probabilistic" (the Probabilistic
                Sharpe Ratio), "deflated" (the Deflated Sharpe Ratio). Defaults to "standard".
            benchmark_sharpe_ratio (float, optional): The hypothesized or benchmark Sharpe ratio
            (SR*) to test the observed Sharpe ratio against. Only used when method="probabilistic".
            It is a per-observation Sharpe ratio, or an annualized one when annualize=True.
            Defaults to 0.0, i.e. testing whether the strategy has any skill at all above doing nothing.
            trials_window (int, optional): The window size of the auxiliary rolling Sharpe ratio
            series that approximates `Var[SR_trials]` and the default `n_trials`, see above. Only
            used when method="deflated". It counts observations of the returns the Sharpe ratio is
            computed from: daily (or intraday) observations for the Sharpe ratio per period, and
            observations of `period` when `rolling` is set. Defaults to None, which uses, per ticker,
            the median number of its observations in a period (so each trial spans about one period)
            for the Sharpe ratio per period, and half of its return history for a rolling Sharpe
            ratio so that enough overlapping windows exist regardless of `period` or date range.
            n_trials (int, optional): The number of independent (or effectively independent)
            strategy variations, parameter combinations, or lookback windows tried before
            arriving at the reported Sharpe ratio. Only used when method="deflated". Defaults to
            None, which falls back to the number of non-overlapping windows of `trials_window`
            observations the returns hold (at least 1), since the overlapping windows of the
            auxiliary rolling Sharpe ratio series are not independent trials. Pass this explicitly
            whenever the actual number of trials is known.
            lags (int, optional): The number of autocorrelation lags in Lo's η(q). Only used when
            method="lo". Defaults to None, which uses the Newey and West (1994) rule
            floor(4 * (n / 100)^(2/9)) on the n observations (4 for a year of daily returns),
            capped at q − 1.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            annualize (bool, optional): Whether to annualize the Sharpe ratio, multiplying it by
                the square root of the number of observations in a year: 252 for the daily
                returns within each period (365 for a ticker that trades on weekends, such as a
                cryptocurrency), those
                days times the median number of bars per day for the intraday returns within a
                day (period="daily" with an intraday_period), or the periods in a year for a
                rolling Sharpe ratio (e.g. 12 for monthly, and the daily observations per year
                of each ticker for daily). This applies to method="standard", "smart" and
                "adjusted"; method="lo" is annualized either way. The probabilistic and deflated
                methods return probabilities, which do not depend on the time scale as long as the
                observed and the benchmark Sharpe ratio share it (Bailey and López de Prado, 2012),
                so annualize=True returns the same probability; for method="probabilistic" it reads
                `benchmark_sharpe_ratio` as an annualized Sharpe ratio. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Sharpe ratio values. For the Probabilistic and Deflated Sharpe Ratios
            these are probabilities between 0 and 1 rather than ratios.

        Notes:
        - The daily period needs intraday data (initialise the Toolkit with an intraday period): the ratio per day
        is then computed from the intraday bars within each day, with the daily risk free rate spread over the
        bars of a day. Without intraday data a single day holds one return, so the result is empty and logs why.
        - The method retrieves historical data and calculates the Sharpe ratio for each asset in the Toolkit instance.
        - The risk-free rate is the Toolkit's risk free rate, the 10-year US Treasury yield by default; initialise the
        Toolkit with the 13-week Treasury bill as its risk free rate for the short-term rate Sharpe (1994) uses.
        - The Adjusted, Probabilistic and Deflated Sharpe Ratios use the **non-excess (raw)** kurtosis
        convention, i.e. a Normal distribution has a kurtosis of 3, not 0, the non-excess kurtosis of
        `risk_model.get_kurtosis`.
        - With growth, the method calculates the growth of the ratio values using the specified `lag`.
        - The Probabilistic Sharpe Ratio is the probability that each Sharpe ratio is genuine,
        i.e. above `benchmark_sharpe_ratio` once the sample length, skewness and kurtosis are accounted for.
        - The Smart Sharpe ratio agrees with quantstats' smart_sharpe on the same excess returns; quantstats returns a
        penalty of 1 where the autocorrelation is undefined (fewer than two returns, or returns without variation),
        the Toolkit NaN.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["SPY", "TLT", "GLD", "BTCUSD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
            benchmark_ticker=None,
        )

        toolkit.performance.get_sharpe_ratio(annualize=True)
        ```

        Which returns:

        | Date   |     SPY |     TLT |     GLD |   BTCUSD |
        |:-------|--------:|--------:|--------:|---------:|
        | 2021   |  2.022  | -0.3656 | -0.5163 |   0.9441 |
        | 2022   | -0.8297 | -1.8879 | -0.1655 |  -1.3294 |
        | 2023   |  1.5607 |  0.03   |  0.6757 |   2.2623 |
        | 2024   |  1.5037 | -0.811  |  1.3751 |   1.666  |
        | 2025   |  0.7247 |  0.0591 |  2.3919 |  -0.0457 |
        """
        check_option(
            "method",
            method,
            (
                "standard",
                "smart",
                "lo",
                "adjusted",
                "probabilistic",
                "deflated",
            ),
        )

        period = period if period else "quarterly" if self._quarterly else "yearly"
        check_rolling(rolling, self._historical_data, period)

        if rolling:
            period_returns = self._get_column(period, "Return", within_period=False)
            full_excess_return = performance_model.get_excess_return(
                period_returns,
                self._risk_free_rate_data[period][period_returns.columns],
            )

        if rolling:
            excess_return = full_excess_return
            sharpe_ratio = performance_model.get_sharpe_ratio(
                excess_return, window=rolling
            )
        else:
            excess_return = self._get_column(
                period, "Excess Return", within_period=True
            )
            sharpe_ratio = performance_model.get_sharpe_ratio(excess_return)

        observations_per_year = self._get_ratio_observations_per_year(
            period, rolling, excess_return.columns
        )

        if method == "standard":
            result = sharpe_ratio
        elif method == "smart":
            result = sharpe_ratio / (
                performance_model.get_autocorrelation_penalty(
                    excess_return, window=rolling
                )
                if rolling
                else performance_model.get_autocorrelation_penalty(excess_return)
            )
        elif method == "lo":
            # Lo's ratio is annualized by construction, with q the observations in a year
            # of each ticker.
            result = pd.concat(
                [
                    (
                        performance_model.get_lo_sharpe_ratio(
                            excess_return[[column]],
                            observations_per_year[column],
                            lags,
                            window=rolling,
                        )
                        if rolling
                        else performance_model.get_lo_sharpe_ratio(
                            excess_return[[column]],
                            observations_per_year[column],
                            lags,
                        )
                    )
                    for column in excess_return.columns
                ],
                axis=1,
            ).set_axis(excess_return.columns, axis=1)
        else:
            if rolling:
                skewness = get_skewness(excess_return, window=rolling)
                kurtosis = get_kurtosis(excess_return, fisher=False, window=rolling)
                n_observations = rolling
            else:
                skewness = get_skewness(excess_return)
                kurtosis = get_kurtosis(excess_return, fisher=False)
                n_observations = excess_return.groupby(level=0).count()

            if method == "adjusted":
                result = performance_model.get_adjusted_sharpe_ratio(
                    sharpe_ratio=sharpe_ratio,
                    skewness=skewness,
                    kurtosis=kurtosis,
                )
            elif method == "probabilistic":
                # The probability is the same on any time scale as long as both Sharpe
                # ratios share it, so an annualized benchmark is taken to the
                # per-observation scale of the Sharpe ratio of each ticker.
                result = performance_model.get_probabilistic_sharpe_ratio(
                    sharpe_ratio=sharpe_ratio,
                    benchmark_sharpe_ratio=(
                        benchmark_sharpe_ratio / np.sqrt(observations_per_year)
                        if annualize
                        else benchmark_sharpe_ratio
                    ),
                    skewness=skewness,
                    kurtosis=kurtosis,
                    n_observations=n_observations,
                )
            else:
                # The trials are built from the same returns as the Sharpe ratio being
                # deflated: comparing a daily-scale Sharpe ratio with the dispersion of
                # Sharpe ratios of period returns (a much larger scale) would push the
                # expected maximum far above it and the deflated ratio to 0.
                # The default window counts the returns of each ticker rather than the
                # rows, which also hold the dates only other tickers traded on (the
                # weekends of a cryptocurrency next to a stock): half of its returns for a
                # rolling ratio, the median number of its returns in a period otherwise.
                if rolling:
                    trials_returns = full_excess_return
                    default_window = trials_returns.count() // 2
                else:
                    trials_returns = excess_return.droplevel(0)
                    period_counts = excess_return.groupby(level=0).count()
                    default_window = period_counts.where(period_counts > 0).median()

                window = (
                    pd.Series(trials_window, index=trials_returns.columns)
                    if trials_window is not None
                    else default_window.fillna(2).astype(int).clip(lower=2)
                )
                trials_sharpe_ratio_variance = pd.Series(
                    {
                        column: performance_model.get_sharpe_ratio(
                            trials_returns[column], window=int(window[column])
                        ).var()
                        for column in trials_returns.columns
                    },
                    dtype=float,
                )
                # Overlapping windows share most of their returns, so they are not
                # independent trials (Bailey and Lopez de Prado, 2014): the default counts
                # the non-overlapping windows the returns hold.
                trials = (
                    n_trials
                    if n_trials is not None
                    else (trials_returns.count() // window).clip(lower=1)
                )

                result = performance_model.get_deflated_sharpe_ratio(
                    sharpe_ratio=sharpe_ratio,
                    sharpe_ratio_variance=trials_sharpe_ratio_variance,
                    n_trials=trials,
                    n_observations=n_observations,
                    skewness=skewness,
                    kurtosis=kurtosis,
                )

        if annualize and method in ("standard", "smart", "adjusted"):
            # The probabilistic and deflated ratios are probabilities, which annualizing
            # does not change, and Lo's ratio is annualized by construction.
            # The rolling ratio uses the returns of each period, the one per period the
            # observations within it: daily returns, or intraday bars within a day.
            result = result * np.sqrt(
                observations_per_year.reindex(result.columns).to_numpy()
            )

        return finalize_dataset(
            dataset=result,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_portfolio
    @handle_errors
    def get_sortino_ratio(
        self,
        period: str | None = None,
        rolling: int | None = None,
        method: Literal[
            "standard",
            "smart",
            "adjusted",
            "probabilistic",
            "probabilistic_adjusted",
        ] = "standard",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        annualize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Sortino Ratio: the excess return an asset earns per unit of downside risk, so that
        only the volatility of losses counts against it. It measures the excess return per unit of the
        downside deviation, the standard deviation of the returns below the target, which suits
        investors who are primarily concerned with the downside of their investments.

        The formula is as follows:

            - Sortino Ratio = Excess Return / Downside Deviation
            - Downside Deviation = SQRT( (1 / N) * SUM( MIN(Excess Return, 0)^2 ) )

        Where N is the *total* number of observations, not just the negative ones, following
        Sortino & Price (1994). By default one Sortino ratio is reported per `period`,
        computed from the **daily** excess returns falling inside that period. For a given
        period, for example monthly, this translates into the following:

            - Sortino Ratio = Average Daily Excess Return within the Month
              / Downside Deviation of the Daily Excess Returns within the Month

        For a rolling period, `period` instead sets the frequency of the returns themselves
        and the ratio is computed over a rolling window of `rolling` such returns:

            - Sortino Ratio = Average Rolling Excess Return / Rolling Downside Deviation

        Note that the Risk Free Rate is already subtracted from these returns. It is the Toolkit's
        risk free rate, the 10-year US Treasury yield by default; initialise the Toolkit with
        the 13-week Treasury bill as its risk free rate for a short-term rate instead.

        A period needs at least two observations for a Sortino ratio, with a single one the ratio
        would only show its sign, so such periods are left out.

        As with the Sharpe Ratio, the result is **not annualized** by default: it is a
        per-observation ratio, roughly SQRT(252), SQRT(52), SQRT(12) or SQRT(4) smaller than the
        annualized figure for daily, weekly, monthly or quarterly returns respectively. Annualizing
        multiplies by that factor.

        The `method` parameter selects a variant of the same ratio:

        - The Smart Sortino ratio of quantstats is the Sortino ratio divided by the
        autocorrelation penalty that also gives the Smart Sharpe ratio, so that serially
        correlated returns no longer flatter it (see `performance_model.get_autocorrelation_penalty`):

            - Smart Sortino Ratio = Sortino Ratio / SQRT( 1 + 2 · Σ_{k=1}^{n−1} ((n − k) / n) · |ρ|^k )

        - The adjusted Sortino ratio is the Sortino ratio divided by SQRT(2), as Jack Schwager proposed to make it
        comparable with the Sharpe ratio: for symmetric returns the downside deviation is about
        1 / SQRT(2) times the standard deviation, so the plain Sortino ratio is about SQRT(2)
        times the Sharpe ratio.

        - The probabilistic Sortino ratio and the probabilistic adjusted Sortino ratio are the probability that the
        true Sortino (or adjusted Sortino) ratio is above 0, with the standard error of the Probabilistic Sharpe Ratio
        of Bailey and López de Prado (2012) applied to it, as quantstats does:

            - PSR = Φ( SoR · sqrt(n − 1) / sqrt(1 − γ₃·SoR + ((γ₄ − 1) / 4)·SoR²) )

        where SoR is the per-observation (adjusted) Sortino ratio, γ₃ the skewness and γ₄ the
        non-excess kurtosis of the excess returns and n their number. That standard error was
        derived for the Sharpe ratio; for the Sortino ratio it has no formal justification and
        is an approximation, so read the probability as indicative.

        See definition: https://en.wikipedia.org/wiki/Sortino_ratio

        For more information, see the following sources:

        - Sortino, F.A., & Price, L.N. (1994). "Performance Measurement in a Downside Risk
        Framework." Journal of Investing, 3(3), 59-64.
        - Bailey, D.H., & López de Prado, M. (2012). "The Sharpe Ratio Efficient Frontier."
        Journal of Risk, 15(2), 3-44.

        Referenced against:
        - quantstats, smart_sortino (with get_autocorrelation_penalty): https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L1158
        - quantstats, adjusted_sortino: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L1258
        - quantstats, probabilistic_sortino_ratio: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L1404
        - quantstats, probabilistic_adjusted_sortino_ratio: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L1434
        - quantstats, autocorr_penalty: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L875

        Shares similar logic:
        - PerformanceAnalytics (R), Return.excess: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/Return.excess.R#L42
        - quantstats, rolling_sortino: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L1188
        - PerformanceAnalytics (R), SortinoRatio: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/SortinoRatio.R#L48
        - empyrical, sortino_ratio: https://github.com/quantopian/empyrical/blob/0.5.5/empyrical/stats.py#L727
        - quantstats, sortino: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L1085
        - quantstats, probabilistic_ratio: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L1297
        - quantstats, probabilistic_sharpe_ratio: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L1369
        - PerformanceAnalytics (R), ProbSharpeRatio: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/ProbSharpeRatio.R#L47
        - RiskLabAI, probabilistic_sharpe_ratio: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/backtest/probabilistic_sharpe_ratio.py#L16

        Args:
            period (str, optional): The period to use for the calculation. Options are: "daily", "weekly",
                "monthly", "quarterly", "yearly". Defaults to "quarterly" if the Toolkit is initialised with
                quarterly=True, otherwise "yearly".
            rolling (int, optional): The rolling window size to use for the calculation. If set,
            the Sortino ratio is calculated over a rolling window of this many periods across the
            full return history instead of per `period`. At least 2. Defaults to None.
            method (str, optional): Which Sortino ratio to calculate, as described above. Options are:
                "standard" (the Sortino ratio), "smart" (the Smart Sortino ratio), "adjusted" (divided by
                SQRT(2)), "probabilistic" (the probabilistic Sortino ratio), "probabilistic_adjusted" (the
                probabilistic adjusted Sortino ratio). Defaults to "standard".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            annualize (bool, optional): Whether to annualize the Sortino ratio, multiplying it by
                the square root of the number of observations in a year, as `get_sharpe_ratio`
                does (252 for the daily returns within each period, 365 for a ticker that trades
                on weekends, or the periods in a year for a rolling ratio). The probabilistic
                methods return probabilities, which annualizing does not change. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Sortino ratio values. For the probabilistic and
            probabilistic adjusted methods these are probabilities between 0 and 1.

        Notes:
        - The daily period needs intraday data (initialise the Toolkit with an intraday period): the ratio per day
        is then computed from the intraday bars within each day, with the daily risk free rate spread over the
        bars of a day. Without intraday data a single day holds one return, so the result is empty and logs why.
        - The method retrieves historical data and calculates the Sortino ratio for each asset in the Toolkit instance.
        - The risk-free rate is the Toolkit's risk free rate, the 10-year US Treasury yield by default. Many
        textbook definitions (Sharpe, 1994; the Ken French factors) use a short-term Treasury bill instead, which
        the Toolkit uses when initialised with the 13-week Treasury bill as its risk free rate.
        - With growth, the method calculates the growth of the ratio values using the specified `lag`.
        - The variants agree with quantstats' sortino, smart_sortino, adjusted_sortino and the probabilistic Sortino
        ratios (not annualized) on the same excess returns. quantstats returns a penalty of 1 where the
        autocorrelation is undefined (fewer than two returns, or returns without variation), the Toolkit NaN.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["V", "MA", "AXP"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.performance.get_sortino_ratio()
        ```

        Which returns:

        | Date   |       V |      MA |     AXP |
        |:-------|--------:|--------:|--------:|
        | 2021   |  0.0057 |  0.0164 |  0.1224 |
        | 2022   | -0.0051 | -0.0011 | -0.0138 |
        | 2023   |  0.1287 |  0.1004 |  0.0924 |
        | 2024   |  0.0911 |  0.1066 |  0.1909 |
        | 2025   |  0.0374 |  0.0273 |  0.0726 |
        """
        check_option(
            "method",
            method,
            (
                "standard",
                "smart",
                "adjusted",
                "probabilistic",
                "probabilistic_adjusted",
            ),
        )

        period = period if period else "quarterly" if self._quarterly else "yearly"
        check_rolling(rolling, self._historical_data, period)

        if rolling:
            period_returns = self._get_column(period, "Return", within_period=False)
            excess_return = performance_model.get_excess_return(
                period_returns,
                self._risk_free_rate_data[period][period_returns.columns],
            )
            sortino_ratio = performance_model.get_sortino_ratio(
                excess_return, window=rolling
            )
        else:
            excess_return = self._get_column(
                period, "Excess Return", within_period=True
            )

            sortino_ratio = performance_model.get_sortino_ratio(excess_return)

        if method == "smart":
            sortino_ratio = sortino_ratio / (
                performance_model.get_autocorrelation_penalty(
                    excess_return, window=rolling
                )
                if rolling
                else performance_model.get_autocorrelation_penalty(excess_return)
            )
        elif method in ("adjusted", "probabilistic_adjusted"):
            sortino_ratio = sortino_ratio / np.sqrt(2)

        if method in ("probabilistic", "probabilistic_adjusted"):
            if rolling:
                skewness = get_skewness(excess_return, window=rolling)
                kurtosis = get_kurtosis(excess_return, fisher=False, window=rolling)
                n_observations = rolling
            else:
                skewness = get_skewness(excess_return)
                kurtosis = get_kurtosis(excess_return, fisher=False)
                n_observations = excess_return.groupby(level=0).count()

            sortino_ratio = performance_model.get_probabilistic_sharpe_ratio(
                sharpe_ratio=sortino_ratio,
                benchmark_sharpe_ratio=0.0,
                skewness=skewness,
                kurtosis=kurtosis,
                n_observations=n_observations,
            )
        elif annualize:
            sortino_ratio = sortino_ratio * np.sqrt(
                self._get_ratio_observations_per_year(
                    period, rolling, sortino_ratio.columns
                ).to_numpy()
            )

        return finalize_dataset(
            dataset=sortino_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_portfolio
    @handle_errors
    def get_safety_first_ratio(
        self,
        period: str | None = None,
        minimum_acceptable_return: float = 0.0,
        rolling: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate Roy's Safety-First Ratio: how many standard deviations the mean return lies above a
        minimum acceptable return; the higher the ratio, the lower the probability of falling below that
        threshold.

        Roy (1952) argues that an investor first of all wants to avoid a disaster, a return
        below a threshold, and so prefers the portfolio with the lowest probability of falling
        below it. For normally distributed returns that probability is N(-Safety-First Ratio),
        so the best portfolio is the one with the highest ratio: the number of standard
        deviations the expected return lies above the threshold.

        The formula is as follows:

        - Safety-First Ratio = (Mean(Return) - MAR per Observation) / Standard Deviation(Return)
        - MAR per Observation = (1 + MAR)^(1 / Observations per Year) - 1

        Where MAR is the minimum acceptable return, the threshold. It is an annual return, compounded down to the returns the ratio is computed
        from: by default the **daily** returns falling inside each `period`, 252 a year for a
        ticker that trades on trading days and 365 for one that also trades on weekends (such
        as a cryptocurrency). With `rolling`, `period` instead sets the frequency of the returns
        and the ratio is computed over a rolling window of `rolling` such returns, with the
        threshold compounded to that frequency. The standard deviation is the sample one
        (with n - 1 in the denominator). With the risk-free rate as the threshold the ratio is the Sharpe Ratio
        (`get_sharpe_ratio`); the default threshold of 0 is Roy's disaster of losing money.

        As with the Sharpe Ratio, the result is **not annualized**: it is a per-observation
        ratio, so multiply by SQRT(252), SQRT(52), SQRT(12) or SQRT(4) for daily, weekly,
        monthly or quarterly returns respectively before comparing against published
        annualized figures.

        For more information, see the following paper:

        - Roy, A.D. (1952). "Safety First and the Holding of Assets." Econometrica, 20(3),
        431-449.

        Also known as: SFRatio, Roy's safety-first criterion.

        Args:
            period (str, optional): The period to use for the calculation. Options are: "daily", "weekly",
                "monthly", "quarterly", "yearly". Defaults to "quarterly" if the Toolkit is initialised with
                quarterly=True, otherwise "yearly".
            minimum_acceptable_return (float, optional): The minimum acceptable (shortfall) return as an annual
                decimal return, e.g. 0.02 for 2% a year. Defaults to 0.0.
            rolling (int | None, optional): The rolling window size to use for the calculation. If set, the
                ratio is calculated over a rolling window of this many periods across the full return history
                instead of per `period`. At least 2. Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Safety-First Ratio values.

        Notes:
        - The daily period needs intraday data (initialise the Toolkit with an intraday period): the ratio per day
        is then computed from the intraday bars within each day, with the threshold compounded over the bars
        of a year. Without intraday data a single day holds one return, so the result is empty and logs why.
        - The method retrieves historical data and calculates the ratio for each asset in the Toolkit instance.
        - With growth, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["KO", "PEP", "MDLZ"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.performance.get_safety_first_ratio(minimum_acceptable_return=0.02)
        ```

        Which returns:

        | Date   |      KO |     PEP |    MDLZ |
        |:-------|--------:|--------:|--------:|
        | 2021   |  0.0589 |  0.0893 |  0.0602 |
        | 2022   |  0.0323 |  0.0211 |  0.0093 |
        | 2023   | -0.0264 | -0.0177 |  0.0366 |
        | 2024   |  0.0361 | -0.0329 | -0.0637 |
        | 2025   |  0.0509 | -0.0036 | -0.0195 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"
        check_rolling(rolling, self._historical_data, period)

        returns = self._get_column(period, "Return", within_period=not rolling)

        # The observations in a year of the returns the ratio is computed from, per ticker,
        # as in get_sharpe_ratio: the periods in a year for a rolling ratio (the daily
        # observations of each ticker for period="daily"), the daily (or intraday)
        # returns within each period otherwise.
        if rolling:
            observations_per_year = (
                get_observations_per_year(self._daily_returns)
                if period == "daily"
                else pd.Series(float(PERIODS_PER_YEAR[period]), index=returns.columns)
            )
        else:
            observations_per_year = self._get_observations_per_year(period)
        observations_per_year = observations_per_year.reindex(returns.columns).fillna(
            float(PERIODS_PER_YEAR["daily"])
        )
        threshold_per_observation = (1 + minimum_acceptable_return) ** (
            1 / observations_per_year
        ) - 1

        safety_first_ratio = (
            performance_model.get_safety_first_ratio(
                returns, threshold_per_observation, window=rolling
            )
            if rolling
            else performance_model.get_safety_first_ratio(
                returns, threshold_per_observation
            )
        )

        return finalize_dataset(
            dataset=safety_first_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_portfolio
    @handle_errors
    def get_upside_potential_ratio(
        self,
        period: str | None = None,
        rolling: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Upside Potential Ratio: the average gain above a minimum acceptable return per
        unit of downside deviation below it, how much upside an asset offers for the downside it
        carries.

        The Upside Potential Ratio of Sortino, van der Meer & Plantinga (1999) divides the
        expected return above a minimum acceptable return (MAR) by the downside deviation below
        it. Where the Sortino Ratio rewards the mean excess return, which the losses lower, this
        ratio rewards only the potential above the MAR and penalizes only the risk below it,
        each counted once.

        The formula is as follows:

        - Upside Potential Ratio = Upside Potential / Downside Deviation
        - Upside Potential = (1 / N) * SUM( MAX(Excess Return, 0) )
        - Downside Deviation = SQRT( (1 / N) * SUM( MIN(Excess Return, 0)^2 ) )

        Where N is the *total* number of observations in both, as the paper weighs every outcome
        by its probability, and the Excess Return is the return minus the risk-free rate, the
        MAR of `get_sortino_ratio` (whose Downside Deviation this is). By default one ratio is
        reported per `period`, computed from the **daily** excess returns falling inside that
        period. For a rolling period, `period` instead sets the frequency of the returns
        themselves and the ratio is computed over a rolling window of `rolling` such returns.

        A period needs at least two observations, and a period without a return below the MAR
        has no downside to divide by; both are left out.

        As with the Sortino Ratio, the result is a ratio of the observations within the period
        and is not annualized.

        For more information, see the following paper:

        - Sortino, F.A., van der Meer, R., & Plantinga, A. (1999). "The Dutch Triangle."
        Journal of Portfolio Management, 26(1), 50-57.

        Also known as: UPR.

        Shares similar logic:
        - PerformanceAnalytics (R), Return.excess: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/Return.excess.R#L42
        - PerformanceAnalytics (R), UpsidePotentialRatio: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/UpsidePotentialRatio.R#L72

        Args:
            period (str, optional): The period to use for the calculation. Options are: "daily", "weekly",
                "monthly", "quarterly", "yearly". Defaults to "quarterly" if the Toolkit is initialised with
                quarterly=True, otherwise "yearly".
            rolling (int | None, optional): The rolling window size to use for the calculation. If set, the
                ratio is calculated over a rolling window of this many periods across the full return history
                instead of per `period`. At least 2. Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Upside Potential Ratio values.

        Notes:
        - The daily period needs intraday data (initialise the Toolkit with an intraday period): the ratio per day
        is then computed from the intraday bars within each day, with the daily risk free rate spread over the
        bars of a day. Without intraday data a single day holds one return, so the result is empty and logs why.
        - The method retrieves historical data and calculates the ratio for each asset in the Toolkit instance.
        - The risk-free rate is the Toolkit's risk free rate, the 10-year US Treasury yield by default. Many
        textbook definitions (Sharpe, 1994; the Ken French factors) use a short-term Treasury bill instead, which
        the Toolkit uses when initialised with the 13-week Treasury bill as its risk free rate.
        - With growth, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["NKE", "LULU", "ADDYY"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.performance.get_upside_potential_ratio()
        ```

        Which returns:

        | Date   |    NKE |   LULU |   ADDYY |
        |:-------|-------:|-------:|--------:|
        | 2021   | 0.6044 | 0.5902 |  0.493  |
        | 2022   | 0.4878 | 0.513  |  0.4846 |
        | 2023   | 0.4598 | 0.6448 |  0.6168 |
        | 2024   | 0.3159 | 0.4143 |  0.5526 |
        | 2025   | 0.464  | 0.3496 |  0.4248 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"
        check_rolling(rolling, self._historical_data, period)

        if rolling:
            period_returns = self._get_column(period, "Return", within_period=False)
            excess_return = performance_model.get_excess_return(
                period_returns,
                self._risk_free_rate_data[period][period_returns.columns],
            )
            upside_potential_ratio = performance_model.get_upside_potential_ratio(
                excess_return, window=rolling
            )
        else:
            excess_return = self._get_column(
                period, "Excess Return", within_period=True
            )
            upside_potential_ratio = performance_model.get_upside_potential_ratio(
                excess_return
            )

        return finalize_dataset(
            dataset=upside_potential_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_portfolio
    @handle_errors
    def get_ulcer_performance_index(
        self,
        period: str | None = None,
        rolling: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Ulcer Performance Index (UPI), also called the Martin ratio: the excess return an
        asset earns per unit of the Ulcer Index, the depth and length of its drawdowns.

        The formula is as follows:

        - Ulcer Performance Index = (Period Return - Risk-Free Rate) / Ulcer Index
        - Ulcer Index = SQRT(MEAN(Drawdown^2)), with each day's drawdown measured from the highest value since
        the start of the period (Martin's definition), or of the trailing `rolling` days within the period

        It measures the excess return earned per unit of drawdown pain: the depth and the duration of the declines
        both weigh in, where the Sharpe Ratio treats upward and downward swings alike. Similar to the Sharpe Ratio,
        a higher UPI is better than a lower one (since investors prefer more return for less risk).

        Also known as: UPI, Martin ratio.

        Shares similar logic:
        - PerformanceAnalytics (R), MartinRatio: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/MartinRatio.R#L33
        - quantstats, ulcer_performance_index: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L1855
        - ffn, to_ulcer_performance_index: https://github.com/pmorissette/ffn/blob/v1.4.1/ffn/core.py#L2989
        - PerformanceAnalytics (R), Return.excess: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/Return.excess.R#L42

        Args:
            period (str, optional): The period to use for the calculation. Options are: "daily", "weekly",
                "monthly", "quarterly", "yearly". Defaults to "quarterly" if the Toolkit is initialised with
                quarterly=True, otherwise "yearly".
            rolling (int | None, optional): The number of trailing days whose highest value each day's drawdown
                is measured from. At least 2. Defaults to None, which measures it from the highest value since the start of
                the period, as Martin (1987) defines the Ulcer Index.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Ulcer Performance Index values.

        Notes:
        - The method retrieves historical data and calculates the UPI for each asset in the Toolkit instance.
        - The risk-free rate is the Toolkit's risk free rate, the 10-year US Treasury yield by default. Many
        textbook definitions (Sharpe, 1994; the Ken French factors) use a short-term Treasury bill instead, which
        the Toolkit uses when initialised with the 13-week Treasury bill as its risk free rate.
        - With growth, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["VZ", "T", "TMUS"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2020-01-01",
            end_date="2025-12-31",
        )

        toolkit.performance.get_ulcer_performance_index()
        ```

        Which returns:

        | Date   |      VZ |       T |    TMUS |
        |:-------|--------:|--------:|--------:|
        | 2021   | -1.3281 | -0.5668 | -1.2681 |
        | 2022   | -1.25   |  0.2849 |  3.6935 |
        | 2023   | -0.0995 | -0.3592 |  1.6567 |
        | 2024   |  1.8693 | 11.6197 | 11.534  |
        | 2025   |  0.7602 |  1.2863 | -0.7896 |
        """

        check_rolling(rolling, self._historical_data, "daily")
        period = period if period else "quarterly" if self._quarterly else "yearly"
        returns = self._get_column(period, "Return", within_period=True)

        period_returns = self._get_column(period, "Return", within_period=False)
        excess_return = (
            period_returns
            if period == "intraday"
            else performance_model.get_excess_return(
                period_returns,
                self._risk_free_rate_data[period][period_returns.columns],
            )
        )

        ulcer_index = get_ui(returns, rolling)

        ulcer_performance_index = performance_model.get_ulcer_performance_index(
            excess_return, ulcer_index
        )

        return finalize_dataset(
            dataset=ulcer_performance_index,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_portfolio
    @handle_errors
    def get_calmar_ratio(
        self,
        period: str | None = None,
        within_period: bool = True,
        method: Literal["standard", "annualized"] = "standard",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Calmar Ratio: the return an asset earns per unit of its Maximum Drawdown, how well
        the return pays for the worst fall.

        The Calmar Ratio is a risk-adjusted return metric that divides the return of an investment
        portfolio or asset over each period (a calendar-year return for the yearly period, a quarterly
        return for the quarterly one) by its Maximum Drawdown, providing insight into the return
        achieved per unit of the worst historical loss of value.

        The formula is as follows:

        - Calmar Ratio = Return / |Maximum Drawdown|

        With the annualized method it is the definition of the reference libraries instead, one value
        per ticker over the whole date range:

        - Calmar Ratio = Annualized Return / |Maximum Drawdown|
        - Annualized Return = PRODUCT(1 + Daily Return) ^ (Observations per Year / N) − 1

        Where both are measured over the daily returns of the whole date range, N is their number and
        the observations per year are 252 (365 for a ticker that trades on weekends, such as a
        cryptocurrency). Young (1991) measured it over the last 36 months.

        See definition: https://en.wikipedia.org/wiki/Calmar_ratio

        For more information, see the following source:

        - Young, T.W. (1991). "Calmar Ratio: A Smoother Tool." Futures, 20(1), 40.

        Also known as: Drawdown ratio.

        Referenced against:
        - PerformanceAnalytics (R), Return.annualized: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/Return.annualized.R#L39

        Shares similar logic:
        - PerformanceAnalytics (R), CalmarRatio: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/CalmarRatio.R#L50
        - empyrical, calmar_ratio: https://github.com/quantopian/empyrical/blob/0.5.5/empyrical/stats.py#L548
        - quantstats, calmar: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L1788
        - ffn, calc_calmar_ratio: https://github.com/pmorissette/ffn/blob/v1.4.1/ffn/core.py#L2945
        - empyrical, annual_return: https://github.com/quantopian/empyrical/blob/0.5.5/empyrical/stats.py#L408

        Args:
            period (str, optional): The period to use for the calculation. Options are: "daily", "weekly",
                "monthly", "quarterly", "yearly". Defaults to "quarterly" if the Toolkit is initialised with
                quarterly=True, otherwise "yearly".
            within_period (bool, optional): Whether to calculate the Maximum Drawdown within the
            specified period or for the entire period. Thus whether to look at the Maximum Drawdown
            within a specific year (if period = 'yearly') or look at the entirety of all years.
            Defaults to True.
            method (str, optional): The ratio. Options are: "standard" (the ratio per period),
                "annualized" (the annualized return over the whole date range divided by the Maximum
                Drawdown over it, one value per ticker; `period` and `within_period` are then not used).
                Defaults to "standard".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Calmar Ratio values, or with the annualized method a Series with one value
            per ticker.

        Notes:
        - The method retrieves historical data and calculates the Calmar Ratio for each asset in
        the Toolkit instance.
        - The annualized method agrees with PerformanceAnalytics' CalmarRatio, empyrical's calmar_ratio and quantstats'
        calmar on the daily returns (252 observations a year). ffn (calc_calmar_ratio) counts the years of the
        annualized return in calendar days, which differs slightly.
        - With growth, the method calculates the growth of the ratio values using
        the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["VTI", "VEA", "VWO"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2020-01-01",
            end_date="2025-12-31",
        )

        toolkit.performance.get_calmar_ratio()
        ```

        Which returns:

        | Date   |     VTI |     VEA |     VWO |
        |:-------|--------:|--------:|--------:|
        | 2021   |  4.9482 |  1.643  |  0.0948 |
        | 2022   | -0.7709 | -0.5338 | -0.6106 |
        | 2023   |  2.4416 |  1.483  |  0.7946 |
        | 2024   |  2.7771 |  0.3321 |  1.1734 |
        | 2025   |  0.8851 |  2.6195 |  1.7646 |
        """
        check_option("method", method, ("standard", "annualized"))

        if method == "annualized":
            daily_returns, annualized_return = self._get_whole_sample_returns()

            return finalize_dataset(
                dataset=performance_model.get_calmar_ratio(
                    annualized_return, get_max_drawdown(daily_returns)
                ),
                start_date=self._start_date,
                end_date=self._end_date,
                default_rounding=self._rounding,
                rounding=rounding,
                apply_slice=False,
            )

        period = period if period else "quarterly" if self._quarterly else "yearly"

        returns = self._get_column(period, "Return", within_period=within_period)

        maximum_drawdown = get_max_drawdown(returns)

        period_returns = self._get_column(period, "Return", within_period=False)

        calmar_ratio = performance_model.get_calmar_ratio(
            period_returns, maximum_drawdown
        )

        return finalize_dataset(
            dataset=calmar_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_portfolio
    @handle_errors
    def get_sterling_ratio(
        self,
        period: str | None = None,
        within_period: bool = True,
        adjustment: float = 0.1,
        largest_drawdowns: int = 3,
        method: Literal["standard", "annualized"] = "standard",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Sterling Ratio: the return an asset earns per unit of the average depth of its
        deepest drawdowns, a Calmar ratio that rests on more than one drawdown.

        The Sterling Ratio is a risk-adjusted return metric that divides the return of an investment
        portfolio or asset over each period (a calendar-year return for the yearly period, a quarterly
        return for the quarterly one) by the average depth of its deepest drawdowns plus a fixed
        adjustment (conventionally 10%), so that it reflects the typical size of its large drawdowns
        rather than only the single worst one (as the Calmar Ratio does).

        The formula is as follows (Bacon, 2008):

        - Sterling Ratio = Return / (|D_Lar| + Adjustment)
        - D_Lar = Σ Trough_j / d over the d = `largest_drawdowns` deepest drawdowns

        Where each drawdown runs from a peak to the trough before wealth is back at that peak (as in
        `toolkit.risk.get_drawdown_table`), measured on the daily returns within the period by default
        (a drawdown that has not recovered by the end of the period included), and d is 3 unless set
        otherwise, or the number of drawdowns when there are fewer. A period without a drawdown has a
        D_Lar of 0.

        With the annualized method the ratio is taken once over the whole date range instead, one value per
        ticker:

        - Sterling Ratio = Annualized Return / (|D_Lar| + Adjustment)
        - Annualized Return = PRODUCT(1 + Daily Return) ^ (Observations per Year / N) − 1

        Where both are measured over the daily returns of the whole date range, N is their number and
        the observations per year are 252 (365 for a ticker that trades on weekends). With
        one largest drawdown, D_Lar is the Maximum Drawdown, which gives the SterlingRatio of
        PerformanceAnalytics.

        The original form of Deane Sterling Jones, popularised by Kestner (1996), divides the compound annual
        return of the last three years by the average of the maximum drawdown of each of those three years
        plus 10%: an average of annual maxima, one per year, where D_Lar averages the deepest drawdowns
        wherever they fall (two of them can lie in the same year).

        For more information, see the following sources:

        - Bacon, C.R. (2008). "Practical Portfolio Performance Measurement and Attribution."
        2nd edition, John Wiley & Sons.
        - Kestner, L.N. (1996). "Getting a Handle on True Performance." Futures, 25(1), 44-46.

        Also known as: Sterling-Calmar ratio.

        Referenced against:
        - PerformanceAnalytics (R), Return.annualized: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/Return.annualized.R#L39

        Shares similar logic:
        - PerformanceAnalytics (R), SterlingRatio: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/CalmarRatio.R#L85
        - empyrical, annual_return: https://github.com/quantopian/empyrical/blob/0.5.5/empyrical/stats.py#L408

        Args:
            period (str, optional): The period to use for the calculation. Options are: "daily", "weekly",
                "monthly", "quarterly", "yearly". Defaults to "quarterly" if the Toolkit is initialised with
                quarterly=True, otherwise "yearly".
            within_period (bool, optional): Whether to measure the drawdowns on the daily returns within
            each period, or on the period returns of the whole history, one D_Lar for every period. Thus
            whether to look at the drawdowns within a specific year (if period = 'yearly') or at the
            entirety of all years. Defaults to True.
            adjustment (float, optional): The fixed adjustment added to the Average Largest Drawdown,
            conventionally 0.1 (10%). Defaults to 0.1.
            largest_drawdowns (int, optional): The number d of deepest drawdowns averaged into D_Lar;
            1 gives the Maximum Drawdown. Defaults to 3.
            method (str, optional): The ratio. Options are: "standard" (the ratio per period),
                "annualized" (the annualized return over the whole date range divided by the Average
                Largest Drawdown over it plus the adjustment, one value per ticker; `period` and
                `within_period` are then not used). Defaults to "standard".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Sterling Ratio values, or with the annualized method a Series with one value
            per ticker.

        Notes:
        - The method retrieves historical data and calculates the Sterling Ratio for each asset in
        the Toolkit instance.
        - The annualized method with one largest drawdown agrees with PerformanceAnalytics' SterlingRatio on the
        daily returns (252 observations a year, excess = `adjustment`).
        - With growth, the method calculates the growth of the ratio values using
        the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["NEE", "DUK", "SO"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2020-01-01",
            end_date="2025-12-31",
        )

        toolkit.performance.get_sterling_ratio()
        ```

        Which returns:

        | Date   |     NEE |     DUK |     SO |
        |:-------|--------:|--------:|-------:|
        | 2021   |  1.1204 |  1.0689 | 0.9206 |
        | 2022   | -0.236  |  0.0903 | 0.3255 |
        | 2023   | -1.0256 | -0.0856 | 0.1095 |
        | 2024   |  0.8937 |  0.8331 | 1.1689 |
        | 2025   |  0.7371 |  0.7028 | 0.4876 |
        """
        check_option("method", method, ("standard", "annualized"))

        if method == "annualized":
            daily_returns, annualized_return = self._get_whole_sample_returns()

            return finalize_dataset(
                dataset=performance_model.get_sterling_ratio(
                    annualized_return,
                    performance_model.get_average_largest_drawdown(
                        daily_returns, largest_drawdowns
                    ),
                    adjustment,
                ),
                start_date=self._start_date,
                end_date=self._end_date,
                default_rounding=self._rounding,
                rounding=rounding,
                apply_slice=False,
            )

        period = period if period else "quarterly" if self._quarterly else "yearly"

        returns = self._get_column(period, "Return", within_period=within_period)

        average_largest_drawdown = performance_model.get_average_largest_drawdown(
            returns, largest_drawdowns
        )

        period_returns = self._get_column(period, "Return", within_period=False)

        sterling_ratio = performance_model.get_sterling_ratio(
            period_returns, average_largest_drawdown, adjustment
        )

        return finalize_dataset(
            dataset=sterling_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_portfolio
    @handle_errors
    def get_burke_ratio(
        self,
        period: str | None = None,
        within_period: bool = True,
        modified: bool = False,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Burke Ratio: the excess return an asset earns per unit of the square root of its
        summed squared drawdowns, which weighs deep drawdowns more than shallow ones.

        The Burke Ratio is a risk-adjusted return metric that divides the excess return (return
        minus the risk-free rate) of an investment portfolio or asset by the square root of the
        sum of its squared drawdowns, penalizing both the frequency and depth of drawdowns more
        heavily than the Calmar or Sterling Ratios.

        The formula is as follows:

        - Burke Ratio = (Return — Risk-Free Rate) / SQRT(SUM(Drawdown_d^2))

        Where Drawdown_d is the drawdown from the running peak at every observation d on which the
        asset is below that peak, which by default is every such day within the period. This weighs
        the time spent under water as well as the depth, so the ratio differs from the version that
        sums one drawdown per episode.

        The modified Burke ratio of Bacon (2008) divides by the root mean square drawdown instead,
        so that it does not shrink as the number of observations N grows:

        - Modified Burke Ratio = (Return — Risk-Free Rate) / SQRT(SUM(Drawdown_d^2) / N) = Burke Ratio * SQRT(N)

        For more information, see the following sources:

        - Burke, G. (1994). "A Sharper Sharpe Ratio." Futures, 23(3), 56.
        - Bacon, C.R. (2008). "Practical Portfolio Performance Measurement and Attribution."
        2nd edition, John Wiley & Sons, pp. 90-91.

        Shares similar logic:
        - PerformanceAnalytics (R), Return.excess: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/Return.excess.R#L42
        - PerformanceAnalytics (R), BurkeRatio (modified): https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/BurkeRatio.R#L37

        Args:
            period (str, optional): The period to use for the calculation. Options are: "daily", "weekly",
                "monthly", "quarterly", "yearly". Defaults to "quarterly" if the Toolkit is initialised with
                quarterly=True, otherwise "yearly".
            within_period (bool, optional): Whether to calculate the drawdowns within the specified
            period or for the entire period. Thus whether to look at the drawdowns within a specific
            year (if period = 'yearly') or look at the entirety of all years. Defaults to True.
            modified (bool, optional): Whether to return the modified Burke ratio, the Burke ratio
            times the square root of the number of observations the drawdowns are measured over.
            Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Burke Ratio values.

        Notes:
        - The method retrieves historical data and calculates the Burke Ratio for each asset in
        the Toolkit instance.
        - PerformanceAnalytics' BurkeRatio sums one drawdown per episode (consecutive negative returns) and
        annualizes the return over the whole sample, so its values differ; the modified ratio relates to the
        plain one by the same factor SQRT(N) in both.
        - The risk-free rate is the Toolkit's risk free rate, the 10-year US Treasury yield by default. Many
        textbook definitions (Sharpe, 1994; the Ken French factors) use a short-term Treasury bill instead, which
        the Toolkit uses when initialised with the 13-week Treasury bill as its risk free rate.
        - With growth, the method calculates the growth of the ratio values using
        the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["BA", "LMT", "RTX"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2020-01-01",
            end_date="2025-12-31",
        )

        toolkit.performance.get_burke_ratio()
        ```

        Which returns:

        | Date   |      BA |     LMT |     RTX |
        |:-------|--------:|--------:|--------:|
        | 2021   | -0.0305 |  0.0141 |  0.3337 |
        | 2022   | -0.0181 |  0.3253 |  0.1089 |
        | 2023   |  0.2139 | -0.0588 | -0.0744 |
        | 2024   | -0.0697 |  0.0506 |  0.662  |
        | 2025   |  0.1202 | -0.0134 |  1.0514 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        returns = self._get_column(period, "Return", within_period=within_period)

        burke_drawdown_measure = performance_model.get_burke_drawdown_measure(returns)

        period_returns = self._get_column(period, "Return", within_period=False)
        excess_return = performance_model.get_excess_return(
            period_returns, self._risk_free_rate_data[period][period_returns.columns]
        )

        burke_ratio = performance_model.get_burke_ratio(
            excess_return,
            burke_drawdown_measure,
            n_observations=(
                (returns.groupby(level=0).count() if within_period else returns.count())
                if modified
                else None
            ),
        )

        return finalize_dataset(
            dataset=burke_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_portfolio
    @handle_errors
    def get_pain_ratio(
        self,
        period: str | None = None,
        within_period: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Pain Ratio: the excess return an asset earns per unit of the Pain Index, the
        average depth of its drawdowns over every day.

        The Pain Ratio divides the excess return (return minus the risk-free rate) of an
        investment portfolio or asset by its Pain Index, the mean depth of its drawdowns. Where
        the Ulcer Performance Index squares the drawdowns, which makes the deep ones dominate,
        the Pain Index weighs every observation under water by its depth alone.

        The formula is as follows:

        - Pain Ratio = (Period Return - Risk-Free Rate) / Pain Index
        - Pain Index = SUM(|Drawdown_t|) / N
        - Drawdown_t = Wealth_t / Peak_t - 1, with Peak_t the highest wealth since the first return

        Within periods (the default) the observations are the daily returns within each
        period: N is the number of days in the period, and the wealth and its peak restart from 1
        before the first return of every period, so a loss on its first day is a drawdown as
        well. Over the whole history the observations are the `period` returns (such as the
        yearly returns) of the full history: N is the number of those periods, and every period's
        excess return is divided by that one Pain Index, as the Calmar, Sterling and Burke ratios
        do. Observations at a new high count as a drawdown of 0. The risk-free rate is compounded over the days of the
        period, so a partial period (such as the current year) earns the rate of the days it
        covers. Bacon (2013) divides the annualized return minus the risk-free rate; for
        the yearly period that is the same for every full year, and for other periods the ratio
        uses the period's own (not annualized) excess return, as the Ulcer Performance Index
        and the Burke Ratio do.

        For more information, see the following source:

        - Bacon, C.R. (2013). "Practical Risk-Adjusted Performance Measurement." John Wiley &
        Sons, chapter 5 (the Pain Index and Pain Ratio of Zephyr Associates).

        Shares similar logic:
        - PerformanceAnalytics (R), Return.excess: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/Return.excess.R#L42
        - PerformanceAnalytics (R), PainRatio: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/PainRatio.R#L33

        Args:
            period (str, optional): The period to use for the calculation. Options are: "daily", "weekly",
                "monthly", "quarterly", "yearly". Defaults to "quarterly" if the Toolkit is initialised with
                quarterly=True, otherwise "yearly".
            within_period (bool, optional): Whether to calculate the Pain Index from the daily drawdowns
                within each period (if period = 'yearly', a Pain Index per year) or from the drawdowns of the
                `period` returns over the full history (one Pain Index from the yearly returns of all years).
                Defaults to True.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Pain Ratio values.

        Notes:
        - The method retrieves historical data and calculates the Pain Ratio for each asset in
        the Toolkit instance.
        - A period without a drawdown has a Pain Index of 0 and no ratio (NaN).
        - The risk-free rate is the Toolkit's risk free rate, the 10-year US Treasury yield by default. Many
        textbook definitions (Sharpe, 1994; the Ken French factors) use a short-term Treasury bill instead, which
        the Toolkit uses when initialised with the 13-week Treasury bill as its risk free rate.
        - With growth, the method calculates the growth of the ratio values using
        the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["HD", "LOW", "TGT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.performance.get_pain_ratio()
        ```

        Which returns:

        | Date   |      HD |     LOW |     TGT |
        |:-------|--------:|--------:|--------:|
        | 2022   | -1.0217 | -1.1848 | -1.468  |
        | 2023   |  1.0889 |  1.3693 | -0.2573 |
        | 2024   |  1.7048 |  1.3511 | -0.5582 |
        | 2025   | -1.3355 | -0.4223 | -1.0867 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        returns = self._get_column(period, "Return", within_period=within_period)

        pain_index = get_pain_index(returns)

        period_returns = self._get_column(period, "Return", within_period=False)
        excess_return = performance_model.get_excess_return(
            period_returns, self._risk_free_rate_data[period][period_returns.columns]
        )

        pain_ratio = performance_model.get_pain_ratio(excess_return, pain_index)

        return finalize_dataset(
            dataset=pain_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_portfolio
    @handle_errors
    def get_m2_ratio(
        self,
        period: str | None = None,
        rolling: int | None = None,
        method: Literal["standard", "sortino"] = "standard",
        excess: str | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the M2 Ratio (Modigliani-Modigliani Measure): the return an asset would have earned at
        the volatility of the benchmark, which puts its risk-adjusted performance in percentage points
        that compare directly with the benchmark's return. The M2 Ratio helps investors and portfolio
        managers determine whether the portfolio is delivering returns that justify its level of risk.

        The formula is as follows:

            - M2 Ratio = Risk-Free Rate + [(Portfolio's Return — Risk-Free Rate) /
              Portfolio Standard Deviation] × Benchmark Standard Deviation

        This rescales the (dimensionless) Sharpe ratio back into return-space by asking
        what return the portfolio would have earned had it been leveraged or
        de-leveraged, via risk-free borrowing or lending, to match the benchmark's
        volatility exactly -- producing a number directly comparable to the benchmark's
        actual return. Requires a `benchmark_ticker` to be set on the Toolkit instance,
        since the benchmark's standard deviation is part of the formula.

        With the Sortino method it is the M2 for Sortino of Bacon (2008), which matches the
        downside risk of the benchmark instead of its total risk:

            - M2 for Sortino = Risk-Free Rate + Sortino Ratio × Benchmark Downside Deviation
            - Sortino Ratio = (Portfolio's Return — Risk-Free Rate) / Portfolio Downside Deviation

        Where the downside deviations are those of the daily excess returns within the period below 0
        (below the risk-free rate), scaled to the period. This equals Bacon's form
        Return + Sortino Ratio × (Benchmark Downside Deviation − Portfolio Downside Deviation) with
        the risk-free rate as the minimum acceptable return.

        With `excess` the result is the M2 excess return, the M2 measure above the benchmark's
        return (Bacon, 2008):

            - Arithmetic M2 Excess Return = M2 − Benchmark Return
            - Geometric M2 Excess Return = (1 + M2) / (1 + Benchmark Return) − 1

        See definition: https://en.wikipedia.org/wiki/Modigliani_risk-adjusted_performance

        For more information, see the following sources:

        - Modigliani, F., & Modigliani, L. (1997). "Risk-Adjusted Performance." Journal of
        Portfolio Management, 23(2), 45-54.
        - Bacon, C.R. (2008). "Practical Portfolio Performance Measurement and Attribution."
        2nd edition, John Wiley & Sons, pp. 68 and 102-103.

        Also known as: Modigliani-Modigliani measure, M2, risk-adjusted performance.

        Referenced against:
        - PerformanceAnalytics (R), MSquared: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/MSquared.R#L33
        - PerformanceAnalytics (R), MSquaredExcess: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/MSquaredExcess.R#L35

        Shares similar logic:
        - PerformanceAnalytics (R), Modigliani: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/Modigliani.R#L36

        Args:
            period (str, optional): The period to use for the calculation. Options are: "weekly",
                "monthly", "quarterly", "yearly". Defaults to "quarterly" if the Toolkit is initialised with
                quarterly=True, otherwise "yearly".
            rolling (int, optional): The rolling window size to use for the calculation. If set,
            the M2 ratio is calculated over a rolling window of this many periods across the full
            return history instead of per `period`. Only for method="standard". At least 2. Defaults to None.
            method (str, optional): The risk matched, as described above. Options are: "standard" (the
                benchmark's total risk), "sortino" (its downside risk). Defaults to "standard".
            excess (str | None, optional): The M2 excess return over the benchmark. Options are: None (the
                M2 measure itself), "arithmetic" (the arithmetic excess), "geometric" (the geometric excess).
                Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: M2 ratio values.

        Notes:
        - Daily M2 is not an option as the standard deviation for 1 day is close to zero. Therefore, it does
        not give any useful insights.
        - The method retrieves historical data and calculates the M2 for each asset in the Toolkit instance.
        - PerformanceAnalytics annualizes over the whole sample and uses population standard deviations;
        the Toolkit uses the return of each period and sample standard deviations. Its M2Sortino multiplies
        the per-observation Sortino ratio by the difference of the annualized downside deviations; the
        Toolkit keeps return, Sortino ratio and downside deviations on the scale of the period, which on
        PerformanceAnalytics' own example data gives 0.1023 instead of 0.1035.
        - The risk-free rate is the Toolkit's risk free rate, the 10-year US Treasury yield by default. Many
        textbook definitions (Sharpe, 1994; the Ken French factors) use a short-term Treasury bill instead, which
        the Toolkit uses when initialised with the 13-week Treasury bill as its risk free rate.
        - With growth, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["HD", "LOW"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2020-01-01",
            end_date="2025-12-31",
        )

        toolkit.performance.get_m2_ratio()
        ```

        Which returns:

        | Date   |      HD |     LOW |
        |:-------|--------:|--------:|
        | 2021   |  0.39   |  0.348  |
        | 2022   | -0.1634 | -0.1482 |
        | 2023   |  0.0932 |  0.0955 |
        | 2024   |  0.1084 |  0.0923 |
        | 2025   | -0.0718 |  0.0056 |
        """
        check_option("method", method, ("standard", "sortino"))
        check_option("excess", excess, (None, "arithmetic", "geometric"))
        if rolling and method == "sortino":
            raise ValueError(
                "The M2 for Sortino is calculated per period, not rolling."
            )

        period = period if period else "quarterly" if self._quarterly else "yearly"
        check_rolling(rolling, self._historical_data, period)
        period_returns = self._get_column(period, "Return", within_period=False)
        benchmark_period_returns = self._get_column(
            period, "Return", within_period=False, benchmark=True
        )
        risk_free_rate = self._risk_free_rate_data[period][period_returns.columns]

        if method == "sortino":
            m2_ratio = performance_model.get_m2_ratio(
                period_returns,
                risk_free_rate,
                self._get_downside_deviation(period, "Excess Return"),
                self._get_downside_deviation(period, "Excess Return", benchmark=True),
            )
        elif rolling:
            m2_ratio = performance_model.get_m2_ratio(
                period_returns,
                risk_free_rate,
                benchmark_returns=benchmark_period_returns,
                window=rolling,
            )
        else:
            daily_returns = self._get_column("daily", "Return", within_period=False)
            benchmark_daily_returns = self._get_column(
                "daily", "Return", within_period=False, benchmark=True
            )
            period_standard_deviation = get_volatility(daily_returns, period)
            benchmark_standard_deviation = get_volatility(
                benchmark_daily_returns, period
            )

            m2_ratio = performance_model.get_m2_ratio(
                period_returns,
                risk_free_rate,
                period_standard_deviation,
                benchmark_standard_deviation,
            )

        if excess:
            m2_ratio = performance_model.get_m2_excess_return(
                m2_ratio, benchmark_period_returns, excess
            )

        return finalize_dataset(
            dataset=m2_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_portfolio
    @handle_errors
    def get_tracking_error(
        self,
        period: str | None = None,
        rolling: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Tracking Error: the volatility of the difference between an asset's returns and
        those of its benchmark, how closely the asset follows the benchmark. A higher Tracking Error
        indicates greater divergence from the benchmark, while a lower Tracking Error suggests that the
        portfolio closely follows the benchmark.

        The formula is as follows:

            - Tracking Error (TE) = Standard Deviation of (Portfolio Returns — Benchmark Returns)

        By default one Tracking Error is reported per `period`, computed from the daily returns within that
        period, so it is a daily figure: multiply by SQRT(252) to compare it with the annualized tracking error
        funds report. With `rolling`, the returns of `period` itself are used instead.

        See definition: https://en.wikipedia.org/wiki/Tracking_error

        Also known as: active risk, benchmark deviation.

        Shares similar logic:
        - PerformanceAnalytics (R), TrackingError: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/TrackingError.R#L32

        Args:
            period (str, optional): The period to use for the calculation. Options are: "daily", "weekly",
                "monthly", "quarterly", "yearly". Defaults to "quarterly" if the Toolkit is initialised with
                quarterly=True, otherwise "yearly".
            rolling (int, optional): The rolling window size to use for the calculation. If set,
            Tracking Error is calculated over a rolling window of this many periods across the
            full return history instead of per `period`. At least 2. Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Tracking error values.

        Notes:
        - Daily Tracking Error is not an option as the standard deviation for 1 day is close to zero. Therefore, it does
        not give any useful insights.
        - The method retrieves historical data and calculates the TE for each asset in the Toolkit instance.
        - With growth, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["VOO", "RSP", "QQQ"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.performance.get_tracking_error()
        ```

        Which returns:

        | Date   |    VOO |    RSP |    QQQ |
        |:-------|-------:|-------:|-------:|
        | 2021   | 0.0002 | 0.0038 | 0.0059 |
        | 2022   | 0.0003 | 0.0034 | 0.0067 |
        | 2023   | 0.0002 | 0.0038 | 0.005  |
        | 2024   | 0.0003 | 0.0046 | 0.0047 |
        | 2025   | 0.001  | 0.0051 | 0.0041 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"
        check_rolling(rolling, self._historical_data, period)

        if rolling:
            returns = self._get_column(period, "Return", within_period=False)
            benchmark_returns = self._get_column(
                period, "Return", within_period=False, benchmark=True
            )

            tracking_error = performance_model.get_tracking_error(
                returns, benchmark_returns, window=rolling
            )
        else:
            returns = self._get_column(period, "Return", within_period=True)
            benchmark_returns = self._get_column(
                period, "Return", within_period=True, benchmark=True
            )

            tracking_error = performance_model.get_tracking_error(
                returns, benchmark_returns
            )

        return finalize_dataset(
            dataset=tracking_error,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_portfolio
    @handle_errors
    def get_information_ratio(
        self,
        period: str | None = None,
        rolling: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Information Ratio (IR): the return an asset earns above its benchmark per unit of
        Tracking Error, how consistently it beats the benchmark. It quantifies how much active return
        the portfolio generates per unit of the volatility of that active return, and is commonly used
        to evaluate the skill of a portfolio manager in generating returns beyond what would be expected
        based on the risk taken.

        - IR > 0: A positive Information Ratio indicates that the portfolio has generated
        excess returns compared to the benchmark, suggesting that the portfolio manager has added value.
        - IR = 0: An Information Ratio of zero implies that the portfolio's excess return
        is in line with the benchmark, meaning the portfolio manager has not added or lost value relative
        to the benchmark.
        - IR < 0: A negative Information Ratio suggests that the portfolio has underperformed
        the benchmark, potentially indicating that the portfolio manager has detracted value.

        The formula is as follows:

            - Information Ratio (IR) = Mean(Portfolio Return — Benchmark Return) / Tracking Error

        By default it is computed from the daily returns within each period and is not annualized: multiply by
        SQRT(252) to compare it with the annualized figures in the literature, where an IR of 0.5 is considered
        good. With `rolling`, the returns of `period` itself are used instead.

        See definition: https://en.wikipedia.org/wiki/Information_ratio

        Also known as: active return per risk.

        Shares similar logic:
        - PerformanceAnalytics (R), InformationRatio: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/InformationRatio.R#L32
        - quantstats, information_ratio: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L2884
        - ffn, calc_information_ratio: https://github.com/pmorissette/ffn/blob/v1.4.1/ffn/core.py#L1691

        Args:
            period (str, optional): The period to use for the calculation. Options are: "daily", "weekly",
                "monthly", "quarterly", "yearly". Defaults to "quarterly" if the Toolkit is initialised with
                quarterly=True, otherwise "yearly".
            rolling (int, optional): The rolling window size to use for the calculation. If set,
            the Information Ratio is calculated over a rolling window of this many periods across
            the full return history instead of per `period`. At least 2. Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Information ratio values.

        Notes:
        - Daily Information Ratio is not an option as the standard deviation for 1 day is close to zero.
        Therefore, it does not give any useful insights.
        - The method retrieves historical data and calculates the IR for each asset in the Toolkit instance.
        - With growth, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["XLK", "XLE", "XLV"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.performance.get_information_ratio()
        ```

        Which returns:

        | Date   |     XLK |     XLE |     XLV |
        |:-------|--------:|--------:|--------:|
        | 2021   |  0.0377 |  0.047  | -0.0204 |
        | 2022   | -0.0557 |  0.1406 |  0.0776 |
        | 2023   |  0.147  | -0.0655 | -0.1316 |
        | 2024   | -0.0053 | -0.0549 | -0.1084 |
        | 2025   |  0.042  | -0.0246 | -0.0114 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"
        check_rolling(rolling, self._historical_data, period)

        if rolling:
            returns = self._get_column(period, "Return", within_period=False)
            benchmark_returns = self._get_column(
                period, "Return", within_period=False, benchmark=True
            )

            information_ratio = performance_model.get_information_ratio(
                returns, benchmark_returns, window=rolling
            )
        else:
            returns = self._get_column(period, "Return", within_period=True)
            benchmark_returns = self._get_column(
                period, "Return", within_period=True, benchmark=True
            )

            information_ratio = performance_model.get_information_ratio(
                returns, benchmark_returns
            )

        return finalize_dataset(
            dataset=information_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_portfolio
    @handle_errors
    def get_upside_capture_ratio(
        self,
        period: str | None = None,
        rolling: int | None = None,
        method: Literal["geometric", "annualized", "arithmetic"] = "geometric",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Upside Capture Ratio: how much of the benchmark's return an asset captures in the
        periods the benchmark rises, above 1 when it gains more than the market in rising markets.

        The Upside Capture Ratio measures how well an investment portfolio or asset performs
        relative to a benchmark during periods in which the benchmark's return is positive. A
        ratio above 1 (or 100%) indicates the asset captured more of the benchmark's gains than
        the benchmark itself.

        The formula is as follows:

        - Upside Capture Ratio = Geometric Average Return in Up Periods / Geometric Average Benchmark Return in Up Periods
        - Geometric Average Return = (PRODUCT(1 + Return))^(1 / n) - 1

        Where the up periods are the observations (days within each period) with a positive benchmark
        return and n is their number, following Morningstar's definition. Geometric rather than arithmetic
        averages make the ratio reflect the compounded return captured, so it does not depend on how
        volatile the returns within the up periods were.

        Upside and downside capture (see `get_downside_capture_ratio`) are read together: an upside capture
        above the downside capture means the asset kept more of the benchmark's gains than of its losses, which
        is what defensive and low-volatility strategies aim for even with both ratios below 1. Both are computed
        from the daily returns within each period, while fund factsheets usually use monthly returns over three
        to five years, so the figures are not directly comparable with published ones.

        For more information, see the following sources:

        - Morningstar. "Upside/Downside Capture Ratio." Morningstar Investment Glossary and methodology documents.
        - Bacon, C.R. (2008). "Practical Portfolio Performance Measurement and Attribution."
        2nd edition, John Wiley & Sons.

        Referenced against:
        - empyrical, up_capture (annualized): https://github.com/quantopian/empyrical/blob/0.5.5/empyrical/stats.py#L1887
        - PerformanceAnalytics (R), UpDownRatios (arithmetic): https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/UpDownRatios.R#L44

        Shares similar logic:
        - empyrical, roll_up_capture: https://github.com/quantopian/empyrical/blob/0.5.5/empyrical/stats.py#L2020
        - empyrical, roll_down_capture: https://github.com/quantopian/empyrical/blob/0.5.5/empyrical/stats.py#L2045
        - empyrical, roll_up_down_capture: https://github.com/quantopian/empyrical/blob/0.5.5/empyrical/stats.py#L2070

        Args:
            period (str, optional): The period to use for the calculation. Options are: "daily", "weekly",
                "monthly", "quarterly", "yearly". Defaults to "quarterly" if the Toolkit is initialised with
                quarterly=True, otherwise "yearly".
            rolling (int, optional): The rolling window size to use for the calculation. If you select
                period = 'monthly' and set rolling to 36 you obtain the rolling three year Upside Capture
                Ratio of the monthly returns, Morningstar's figure. At least 2. Defaults to None, which uses the daily
                returns within each period.
            method (str, optional): How the returns are averaged. Options are:
                - "geometric": the geometric average return per observation (Morningstar).
                - "annualized": the geometric average annualized, (1 + average)^P - 1 with P the
                  observations in a year, the ratio of the annual returns empyrical's up_capture reports.
                - "arithmetic": the sum of the returns, as PerformanceAnalytics' UpDownRatios reports.
                Defaults to "geometric".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Upside Capture Ratio values.

        Notes:
        - The method retrieves historical data and calculates the Upside Capture Ratio for each
        asset in the Toolkit instance.
        - The annualized method agrees with empyrical's up_capture (and with rolling= its roll_up_capture) on the
        same returns. Annualizing compounds both averages over a year, which changes the value but not the
        ranking of tickers against the same benchmark. With the arithmetic method agrees with PerformanceAnalytics'
        UpDownRatios (the Capture method).
        - See `get_capture_ratios` for the overall capture ratio, the Up/Down Capture Ratio and the Up and Down
        Numbers and Percentages in one table.
        - With growth, the method calculates the growth of the ratio values using
        the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["QQQ", "SCHD", "USMV"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.performance.get_upside_capture_ratio()
        ```

        Which returns:

        | Date   |    QQQ |   SCHD |   USMV |
        |:-------|-------:|-------:|-------:|
        | 2021   | 1.1827 | 0.832  | 0.6961 |
        | 2022   | 1.2845 | 0.7129 | 0.7051 |
        | 2023   | 1.3244 | 0.765  | 0.6077 |
        | 2024   | 1.3372 | 0.5237 | 0.5323 |
        | 2025   | 1.2056 | 0.5019 | 0.4674 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"
        check_rolling(rolling, self._historical_data, period)
        returns = self._get_column(period, "Return", within_period=not rolling)
        benchmark_returns = self._get_column(
            period, "Return", within_period=not rolling, benchmark=True
        )
        periods_per_year = self._get_ratio_observations_per_year(
            period, rolling, returns.columns
        )

        if rolling:
            upside_capture_ratio = performance_model.get_capture_ratio(
                returns,
                benchmark_returns,
                window=rolling,
                side="up",
                method=method,
                periods_per_year=periods_per_year,
            )
        else:
            upside_capture_ratio = performance_model.get_upside_capture_ratio(
                returns,
                benchmark_returns,
                method=method,
                periods_per_year=periods_per_year,
            )

        return finalize_dataset(
            dataset=upside_capture_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=bool(rolling),
        )

    @handle_portfolio
    @handle_errors
    def get_downside_capture_ratio(
        self,
        period: str | None = None,
        rolling: int | None = None,
        method: Literal["geometric", "annualized", "arithmetic"] = "geometric",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Downside Capture Ratio: how much of the benchmark's loss an asset takes in the
        periods the benchmark falls, below 1 when it loses less than the market in falling markets.

        The Downside Capture Ratio measures how well an investment portfolio or asset performs
        relative to a benchmark during periods in which the benchmark's return is negative. A
        ratio below 1 (or 100%) indicates the asset lost less than the benchmark during those
        periods.

        The formula is as follows:

        - Downside Capture Ratio = Geometric Average Return in Down Periods / Geometric Average Benchmark Return in Down Periods
        - Geometric Average Return = (PRODUCT(1 + Return))^(1 / n) - 1

        Where the down periods are the observations (days within each period) with a negative benchmark
        return and n is their number, following Morningstar's definition. As with the Upside Capture Ratio,
        both ratios are computed from the daily returns within each period, while fund factsheets usually
        use monthly returns over three to five years.

        For more information, see the following sources:

        - Morningstar. "Upside/Downside Capture Ratio." Morningstar Investment Glossary and methodology documents.
        - Bacon, C.R. (2008). "Practical Portfolio Performance Measurement and Attribution."
        2nd edition, John Wiley & Sons.

        Referenced against:
        - empyrical, down_capture (annualized): https://github.com/quantopian/empyrical/blob/0.5.5/empyrical/stats.py#L1921

        Shares similar logic:
        - empyrical, roll_up_capture: https://github.com/quantopian/empyrical/blob/0.5.5/empyrical/stats.py#L2020
        - empyrical, roll_down_capture: https://github.com/quantopian/empyrical/blob/0.5.5/empyrical/stats.py#L2045
        - empyrical, roll_up_down_capture: https://github.com/quantopian/empyrical/blob/0.5.5/empyrical/stats.py#L2070
        - PerformanceAnalytics (R), UpDownRatios (arithmetic): https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/UpDownRatios.R#L44

        Args:
            period (str, optional): The period to use for the calculation. Options are: "daily", "weekly",
                "monthly", "quarterly", "yearly". Defaults to "quarterly" if the Toolkit is initialised with
                quarterly=True, otherwise "yearly".
            rolling (int, optional): The rolling window size to use for the calculation. If you select
                period = 'monthly' and set rolling to 36 you obtain the rolling three year Downside Capture
                Ratio of the monthly returns, Morningstar's figure. At least 2. Defaults to None, which uses the daily
                returns within each period.
            method (str, optional): How the returns are averaged. Options are:
                - "geometric": the geometric average return per observation (Morningstar).
                - "annualized": the geometric average annualized, (1 + average)^P - 1 with P the
                  observations in a year, the ratio of the annual returns empyrical's down_capture reports.
                - "arithmetic": the sum of the returns, as PerformanceAnalytics' UpDownRatios reports.
                Defaults to "geometric".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Downside Capture Ratio values.

        Notes:
        - The method retrieves historical data and calculates the Downside Capture Ratio for each
        asset in the Toolkit instance.
        - The annualized method agrees with empyrical's down_capture (and with rolling= its roll_down_capture) on the
        same returns. Annualizing compounds both averages over a year, which changes the value but not the
        ranking of tickers against the same benchmark. With the arithmetic method agrees with PerformanceAnalytics'
        UpDownRatios (the Capture method), except that PerformanceAnalytics also counts the days the benchmark
        did not move as down days.
        - See `get_capture_ratios` for the overall capture ratio, the Up/Down Capture Ratio and the Up and Down
        Numbers and Percentages in one table.
        - With growth, the method calculates the growth of the ratio values using
        the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["XLU", "SPLV", "ARKK"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.performance.get_downside_capture_ratio()
        ```

        Which returns:

        | Date   |    XLU |   SPLV |   ARKK |
        |:-------|-------:|-------:|-------:|
        | 2021   | 0.3911 | 0.5358 | 2.0537 |
        | 2022   | 0.4698 | 0.4991 | 2.5005 |
        | 2023   | 0.7024 | 0.6177 | 2.3265 |
        | 2024   | 0.2051 | 0.3127 | 2.5483 |
        | 2025   | 0.3459 | 0.3223 | 2.0919 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"
        check_rolling(rolling, self._historical_data, period)
        returns = self._get_column(period, "Return", within_period=not rolling)
        benchmark_returns = self._get_column(
            period, "Return", within_period=not rolling, benchmark=True
        )
        periods_per_year = self._get_ratio_observations_per_year(
            period, rolling, returns.columns
        )

        if rolling:
            downside_capture_ratio = performance_model.get_capture_ratio(
                returns,
                benchmark_returns,
                window=rolling,
                side="down",
                method=method,
                periods_per_year=periods_per_year,
            )
        else:
            downside_capture_ratio = performance_model.get_downside_capture_ratio(
                returns,
                benchmark_returns,
                method=method,
                periods_per_year=periods_per_year,
            )

        return finalize_dataset(
            dataset=downside_capture_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=bool(rolling),
        )

    @handle_errors
    def get_capture_ratios(
        self,
        period: str | None = None,
        method: Literal["geometric", "annualized", "arithmetic"] = "geometric",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the capture ratios and the up and down statistics of an investment portfolio or asset
        against its benchmark in one table: how much of the benchmark's return it captured overall, in
        rising and in falling markets, and how often it moved with or beat the benchmark in each.

        The formulas are as follows:

        - Capture Ratio = Average Return / Average Benchmark Return, over every day
        - Upside (Downside) Capture Ratio = the same over the days the benchmark rose (fell), see
          `get_upside_capture_ratio` and `get_downside_capture_ratio`
        - Up/Down Capture Ratio = Upside Capture Ratio / Downside Capture Ratio
        - Up Number = Number of Days with Return > 0 and Benchmark Return > 0 / Number of Days with
          Benchmark Return > 0
        - Down Number = Number of Days with Return < 0 and Benchmark Return < 0 / Number of Days with
          Benchmark Return < 0
        - Up Percent = Number of Days with Return > Benchmark Return and Benchmark Return > 0 / Number of
          Days with Benchmark Return > 0
        - Down Percent = Number of Days with Return > Benchmark Return and Benchmark Return < 0 / Number
          of Days with Benchmark Return < 0

        Where the averages are geometric by default (Morningstar), (PRODUCT(1 + Return))^(1 / n) - 1 over
        the same n days for the asset and the benchmark, all from the daily returns within each period. An
        Up/Down Capture Ratio above 1 means the asset kept more of the benchmark's gains than of its losses.
        A high Up Number with a low Down Number shows an asset that rises with the market but falls less
        often; the Up and Down Percent are the up and down market win rates (batting averages, see
        `get_win_rate`).

        For more information, see the following sources:

        - Bacon, C.R. (2008). "Practical Portfolio Performance Measurement and Attribution."
        2nd edition, John Wiley & Sons.
        - Morningstar. "Upside/Downside Capture Ratio." Morningstar Investment Glossary and methodology documents.

        Also known as: up/down capture, up-market and down-market capture, capture table.

        Referenced against:
        - empyrical, up_capture (annualized): https://github.com/quantopian/empyrical/blob/0.5.5/empyrical/stats.py#L1887
        - PerformanceAnalytics (R), UpDownRatios (arithmetic): https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/UpDownRatios.R#L44
        - empyrical, down_capture (annualized): https://github.com/quantopian/empyrical/blob/0.5.5/empyrical/stats.py#L1921
        - empyrical, capture (annualized): https://github.com/quantopian/empyrical/blob/0.5.5/empyrical/stats.py#L1531
        - PerformanceAnalytics (R), UpDownRatios (Up Number, Down Number): https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/UpDownRatios.R#L44
        - PerformanceAnalytics (R), UpDownRatios (Up Percent, Down Percent): https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/UpDownRatios.R#L44

        Shares similar logic:
        - quantstats, win_rate: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L564

        Args:
            period (str, optional): The period to use for the calculation. Options are: "daily", "weekly",
                "monthly", "quarterly", "yearly". Defaults to "quarterly" if the Toolkit is initialised with
                quarterly=True, otherwise "yearly".
            method (str, optional): How the capture ratios average the returns. Options are:
                - "geometric": the geometric average return per day (Morningstar).
                - "annualized": the geometric average annualized, (1 + average)^P - 1 with P the days in a
                  year (252, 365 for a ticker that trades on weekends), empyrical's capture ratios.
                - "arithmetic": the sum of the returns, PerformanceAnalytics' UpDownRatios.
                Defaults to "geometric".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The Capture Ratio, Upside Capture Ratio, Downside Capture Ratio, Up/Down Capture
            Ratio, Up Number, Down Number, Up Percent and Down Percent per period, with a Multi Index of
            (ticker, statistic) as the columns.

        Notes:
        - With the annualized method the Capture Ratio, Upside, Downside and Up/Down Capture Ratios equal
        empyrical's capture, up_capture, down_capture and up_down_capture on the same daily returns. With
        the arithmetic method the Upside and Downside Capture Ratios agree with PerformanceAnalytics' UpDownRatios,
        which also counts the days the benchmark did not move as down days for its Down Capture; its Up and
        Down Number and Percent equal the Toolkit's.
        - The Portfolio of a Portfolio's Toolkit is measured on its own daily return series, as a weighted
        average of capture ratios or counts is not the statistic of the portfolio.
        - With growth, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["QQQ", "USMV"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.performance.get_capture_ratios().xs("USMV", level=0, axis=1).T
        ```

        Which returns:

        |                        |   2021 |   2022 |   2023 |   2024 |   2025 |
        |:-----------------------|-------:|-------:|-------:|-------:|-------:|
        | Capture Ratio          | 0.7641 | 0.4938 | 0.4226 | 0.6578 | 0.4517 |
        | Upside Capture Ratio   | 0.6966 | 0.7055 | 0.6068 | 0.5319 | 0.4679 |
        | Downside Capture Ratio | 0.6706 | 0.6789 | 0.6784 | 0.4891 | 0.473  |
        | Up/Down Capture Ratio  | 1.0389 | 1.0392 | 0.8944 | 1.0874 | 0.9892 |
        | Up Number              | 0.8151 | 0.8899 | 0.8214 | 0.7365 | 0.7413 |
        | Down Number            | 0.8    | 0.8652 | 0.8073 | 0.7308 | 0.7383 |
        | Up Percent             | 0.3356 | 0.2018 | 0.2429 | 0.2703 | 0.2727 |
        | Down Percent           | 0.7048 | 0.7801 | 0.6697 | 0.7115 | 0.6822 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"
        returns = self._get_column(period, "Return", within_period=True)
        benchmark_returns = self._get_column(
            period, "Return", within_period=True, benchmark=True
        )
        periods_per_year = self._get_ratio_observations_per_year(
            period, None, returns.columns
        )

        upside_capture_ratio = performance_model.get_upside_capture_ratio(
            returns, benchmark_returns, method, periods_per_year
        )
        downside_capture_ratio = performance_model.get_downside_capture_ratio(
            returns, benchmark_returns, method, periods_per_year
        )

        capture_ratios = pd.concat(
            {
                "Capture Ratio": performance_model.get_capture_ratio(
                    returns, benchmark_returns, method, periods_per_year
                ),
                "Upside Capture Ratio": upside_capture_ratio,
                "Downside Capture Ratio": downside_capture_ratio,
                "Up/Down Capture Ratio": upside_capture_ratio / downside_capture_ratio,
                "Up Number": performance_model.get_up_down_number(
                    returns, benchmark_returns, market="up"
                ),
                "Down Number": performance_model.get_up_down_number(
                    returns, benchmark_returns, market="down"
                ),
                "Up Percent": performance_model.get_win_rate(
                    returns, benchmark_returns, market="up"
                ),
                "Down Percent": performance_model.get_win_rate(
                    returns, benchmark_returns, market="down"
                ),
            },
            axis=1,
        ).replace([np.inf, -np.inf], np.nan)

        # One block of statistics per ticker, as the market timing models show them.
        capture_ratios = capture_ratios.swaplevel(axis=1)[list(returns.columns)]

        return finalize_dataset(
            dataset=capture_ratios,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_portfolio
    @handle_errors
    def get_win_rate(
        self,
        period: str | None = None,
        market: str = "all",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Win Rate of an investment portfolio or asset against its benchmark: the share of
        days within each period in which the asset's return exceeds the benchmark's return. The Win Rate
        of `get_return_statistics` counts the days with a gain instead.

        The formula is as follows:

        - Win Rate = (Number of Days with Asset Return > Benchmark Return) /
          (Number of Days with both an Asset and a Benchmark Return), using the daily returns within each
          period (e.g. each year), as a decimal (0.55 = 55%)

        Days on which the asset or the benchmark has no return are left out of both counts rather than
        counted as losses.

        In up or down markets only the days the benchmark rose or fell are counted, the up and down
        market batting averages: a manager who beats a rising market by taking more risk shows a high up
        market and a low down market figure, while one who protects capital shows the reverse.

        - Up Market Win Rate = (Number of Days with Asset Return > Benchmark Return and Benchmark Return > 0) /
          (Number of Days with Benchmark Return > 0)

        For more information, see the following source:

        - Bacon, C.R. (2008). "Practical Portfolio Performance Measurement and Attribution."
        2nd edition, John Wiley & Sons.

        Also known as: batting average.

        Referenced against:
        - PerformanceAnalytics (R), UpDownRatios (Up Percent, Down Percent): https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/UpDownRatios.R#L44

        Shares similar logic:
        - quantstats, win_rate: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L564
        - empyrical, batting_average: https://github.com/stefan-jansen/empyrical-reloaded/blob/0.7.3/src/empyrical/stats.py#L1973

        Args:
            period (str, optional): The period to use for the calculation. Options are: "daily", "weekly",
                "monthly", "quarterly", "yearly". Defaults to "quarterly" if the Toolkit is initialised with
                quarterly=True, otherwise "yearly".
            market (str, optional): The days counted. Options are: "all" (every day), "up" (the benchmark
                rose), "down" (the benchmark fell). Defaults to "all".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Win Rate values.

        Notes:
        - The method retrieves historical data and calculates the Win Rate for each asset in the
        Toolkit instance.
        - The overall Win Rate agrees with empyrical's batting_average and the up and down market ones its "up market"
        and "down market" values, except that empyrical counts a day the benchmark did not move as an up market
        day, where the Toolkit counts it as neither (as PerformanceAnalytics' Up and Down Percent do).
        - With growth, the method calculates the growth of the ratio values using
        the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "GOOGL"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.performance.get_win_rate()
        ```

        Which returns:

        | Date   |   AAPL |   MSFT |   GOOGL |
        |:-------|-------:|-------:|--------:|
        | 2021   | 0.494  | 0.5139 |  0.5299 |
        | 2022   | 0.4781 | 0.4582 |  0.4502 |
        | 2023   | 0.576  | 0.512  |  0.524  |
        | 2024   | 0.5    | 0.4881 |  0.5317 |
        | 2025   | 0.472  | 0.484  |  0.564  |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"
        returns = self._get_column(period, "Return", within_period=True)
        benchmark_returns = self._get_column(
            period, "Return", within_period=True, benchmark=True
        )

        win_rate = performance_model.get_win_rate(
            returns, benchmark_returns, market=market
        )

        return finalize_dataset(
            dataset=win_rate,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_outperformance_probability(
        self,
        period: str = "monthly",
        windows: list[int] | None = None,
        relative: bool = True,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the probability of outperformance of an investment portfolio or asset: the share of all
        windows of a given number of periods (for example every 12 consecutive months) in which it returned
        more than the benchmark. Looking at windows of different lengths shows whether an asset that often
        lags over a month still tends to come out ahead over one or three years.

        The formula is as follows:

        - Window Return = PRODUCT(1 + Return over the window) - 1
        - Probability of Outperformance = Number of Windows with Window Return > Benchmark Window Return /
          Number of Windows in which they differ

        The windows end at every period from the first full window onwards, so consecutive windows overlap and
        a longer window has fewer of them. Measured on its own the window return is compared with 0 instead of
        the benchmark's, the share of windows with a gain: for 12 monthly returns ffn's "Win 12m %".

        The measure is defined in PerformanceAnalytics (Peterson and Carl) as table.ProbOutPerformance; the
        Toolkit applies it to each ticker against its benchmark.


        Also known as: rolling win rate, hit rate over rolling windows.

        Referenced against:
        - PerformanceAnalytics (R), table.ProbOutPerformance: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/table.ProbOutperformance.R#L32

        Shares similar logic:
        - ffn, PerformanceStats (twelve_month_win_perc): https://github.com/pmorissette/ffn/blob/v1.4.1/ffn/core.py#L123

        Args:
            period (str, optional): The period of the returns the windows are made of. Options are: "daily",
                "weekly", "monthly", "quarterly", "yearly". Defaults to "monthly".
            windows (list[int] | None, optional): The window lengths in periods. Defaults to None, which uses
                1, 3, 6, 9, 12, 18 and 36 (PerformanceAnalytics' defaults).
            relative (bool, optional): Whether to compare with the benchmark (True) or with a return of 0
                (False). Defaults to True.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The probability of outperformance per window length (rows) and ticker (columns).

        Notes:
        - Equals the probability column of PerformanceAnalytics' table.ProbOutPerformance on the same returns.
        PerformanceAnalytics reports the counts as well; a window in which both returns are equal counts as
        neither, there as here.
        - Measured on its own over a window of 12 months it agrees with ffn's twelve_month_win_perc, except that ffn
        also counts the window that starts at the first price, one window more than the monthly returns hold.
        - Each ticker is measured over its own history, so a later listing has fewer windows.
        - The Portfolio of a Portfolio's Toolkit is measured on its own daily return series, as a weighted
        average of probabilities is not that of the portfolio.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "KO"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2016-01-01",
            end_date="2025-12-31",
        )

        toolkit.performance.get_outperformance_probability()
        ```

        Which returns:

        |   Window |   AAPL |   MSFT |     KO |
        |---------:|-------:|-------:|-------:|
        |        1 | 0.5714 | 0.5882 | 0.4118 |
        |        3 | 0.6496 | 0.641  | 0.3932 |
        |        6 | 0.614  | 0.7105 | 0.3158 |
        |        9 | 0.6667 | 0.7838 | 0.3153 |
        |       12 | 0.7222 | 0.7963 | 0.3333 |
        |       18 | 0.8137 | 0.8333 | 0.3039 |
        |       36 | 0.8929 | 0.9286 | 0.0833 |
        """
        returns = self._get_column(period, "Return", within_period=False)
        benchmark_returns = (
            self._get_column(period, "Return", within_period=False, benchmark=True)
            if relative
            else None
        )

        outperformance_probability = performance_model.get_outperformance_probability(
            returns.loc[self._start_date : self._end_date],
            (
                benchmark_returns.loc[self._start_date : self._end_date]
                if benchmark_returns is not None
                else None
            ),
            windows=windows,
        )

        return apply_rounding(
            outperformance_probability,
            rounding if rounding is not None else self._rounding,
        )

    @handle_errors
    def get_bull_bear_beta(
        self,
        period: str | None = None,
        market: str = "excess_return",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the bull and bear market Beta and Alpha and the Timing Ratio of an investment portfolio or
        asset: the CAPM regression fitted separately over the days the market rose (bull) and fell (bear).

        The formulas are as follows:

        - Excess Return = Bull Alpha + Bull Beta x Benchmark Excess Return, over the days the market rose
        - Excess Return = Bear Alpha + Bear Beta x Benchmark Excess Return, over the days the market fell
        - Timing Ratio = Bull Beta / Bear Beta
        - Annualized Alpha = (1 + Alpha per Day)^P - 1

        Where the excess returns are the daily returns minus the risk free rate, the market "rose" when the
        benchmark excess return was positive (PerformanceAnalytics) or, with the return market, when the
        benchmark return was positive (empyrical), and P is the number of days in a year (252, 365 for a ticker
        that trades on weekends). A Timing Ratio above 1 means the asset moved more with the market when it rose
        than when it fell, the asymmetry a successful market timer shows; Henriksson and Merton (1981) test the
        same asymmetry with one regression (see the Henriksson-Merton model of `get_market_timing_model`).

        For more information, see the following sources:

        - Henriksson, R.D., & Merton, R.C. (1981). "On Market Timing and Investment Performance II: Statistical
        Procedures for Evaluating Forecasting Skills." Journal of Business, 54(4), 513-533.
        - Bacon, C.R. (2008). "Practical Portfolio Performance Measurement and Attribution."
        2nd edition, John Wiley & Sons.

        Also known as: up-market and down-market beta, conditional beta, up and down alpha.

        Referenced against:
        - PerformanceAnalytics (R), CAPM.beta.bull: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/CAPM.beta.R#L133
        - PerformanceAnalytics (R), CAPM.beta.bear: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/CAPM.beta.R#L184
        - PerformanceAnalytics (R), TimingRatio: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/CAPM.beta.R#L236
        - empyrical, up_alpha_beta: https://github.com/quantopian/empyrical/blob/0.5.5/empyrical/stats.py#L1986
        - empyrical, down_alpha_beta: https://github.com/quantopian/empyrical/blob/0.5.5/empyrical/stats.py#L2004

        Args:
            period (str, optional): The period to use for the calculation. Options are: "daily", "weekly",
                "monthly", "quarterly", "yearly". Defaults to "quarterly" if the Toolkit is initialised with
                quarterly=True, otherwise "yearly".
            market (str, optional): What splits the bull and bear days. Options are: "excess_return" (the
                sign of the benchmark excess return, PerformanceAnalytics), "return" (the sign of the
                benchmark return, empyrical). Defaults to "excess_return".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The Bull Alpha, Bull Beta, Bear Alpha, Bear Beta and Timing Ratio per period, with a
            Multi Index of (ticker, statistic) as the columns.

        Notes:
        - The Bull and Bear Betas and the Timing Ratio agree with PerformanceAnalytics' CAPM.beta.bull, CAPM.beta.bear
        and TimingRatio on the daily returns within each period, and the Alphas and Betas empyrical's up_alpha_beta
        and down_alpha_beta given the excess returns. empyrical splits the days on the sign of the benchmark return
        it is given, which the return market does with the benchmark's own return; given a risk free rate, empyrical
        computes its Beta from the returns themselves and its Alpha from the excess returns, where the Toolkit uses
        excess returns for both.
        - Unlike the upside or downside Beta of `get_beta`, which regresses the returns themselves, these
        regressions use excess returns, so the Alphas are the CAPM (Jensen's) Alphas of each market.
        - Each side needs at least 20 days with returns in a period (about a trading month), otherwise its values
          are NaN: the Alpha is compounded over a year, which amplifies the Alpha of a few days to extremes. Weekly
          and monthly periods have fewer days per side, so they give no values (an empty result, which is logged);
          quarterly and yearly periods give values.
        - The Portfolio of a Portfolio's Toolkit is regressed on its own daily return series, as the market
        timing models do.
        - With growth, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["XLU", "XLK"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.performance.get_bull_bear_beta().xs("XLK", level=0, axis=1)
        ```

        Which returns:

        |      |   Bull Alpha |   Bull Beta |   Bear Alpha |   Bear Beta |   Timing Ratio |
        |:-----|-------------:|------------:|-------------:|------------:|---------------:|
        | 2021 |      -0.1212 |      1.4017 |      -0.2456 |      1.1776 |         1.1903 |
        | 2022 |       0.0242 |      1.3213 |      -0.3301 |      1.2077 |         1.0941 |
        | 2023 |       0.0721 |      1.3097 |      -0.0304 |      1.1213 |         1.168  |
        | 2024 |      -0.0049 |      1.5541 |      -0.2769 |      1.5097 |         1.0294 |
        | 2025 |       0.3724 |      1.2526 |      -0.1605 |      1.3238 |         0.9462 |
        """
        check_option("market", market, ("excess_return", "return"))

        period = period if period else "quarterly" if self._quarterly else "yearly"
        excess_return = self._get_column(period, "Excess Return", within_period=True)
        benchmark_excess_return = self._get_column(
            period, "Excess Return", within_period=True, benchmark=True
        )
        benchmark_returns = self._get_column(
            period, "Return", within_period=True, benchmark=True
        )
        observations_per_year = self._get_ratio_observations_per_year(
            period, None, excess_return.columns
        )

        bull_bear_beta = {
            ticker: pd.DataFrame.from_dict(
                {
                    sub_period: performance_model.get_bull_bear_beta(
                        excess_return.loc[sub_period, ticker],
                        benchmark_excess_return.loc[sub_period],
                        (
                            benchmark_returns.loc[sub_period]
                            if market == "return"
                            else None
                        ),
                        observations_per_year[ticker],
                        minimum_observations=MINIMUM_BULL_BEAR_OBSERVATIONS,
                    )
                    for sub_period in excess_return.index.get_level_values(0).unique()
                },
                orient="index",
            )
            for ticker in excess_return.columns
        }
        bull_bear_beta = pd.concat(bull_bear_beta, axis=1)

        # The rows without values are dropped, so a period too short for either side would otherwise give an empty result without a reason.  # noqa: E501
        if not bull_bear_beta.empty and bull_bear_beta.isna().all().all():
            # Quarterly and yearly periods are already the longest, so only a longer date range adds days to them.  # noqa: E501
            advice = (
                "Use a longer date range."
                if period in ("quarterly", "yearly")
                else "Use quarterly or yearly periods."
            )
            logger.warning(
                "The Bull and Bear Betas need at least %s days with returns on each side of the "
                "market in a period, which no %s period holds, so there are no values. %s",
                MINIMUM_BULL_BEAR_OBSERVATIONS,
                period,
                advice,
            )

        return finalize_dataset(
            dataset=bull_bear_beta,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_portfolio
    @handle_errors
    def get_beta_fragility(
        self,
        period: str | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the beta fragility heuristic of an investment portfolio or asset, the stress-testing measure
        of Taleb, Canetti, Kinda, Loukoianova and Schmieder (2012): whether its response to the benchmark
        bends at the extremes, so that a linear Beta understates its losses in a large market fall.

        The formula is as follows:

        - H = (w_low x Return_low + w_high x Return_high) - Return_middle
        - w_low = (Benchmark_high - Benchmark_middle) / (Benchmark_high - Benchmark_low)
        - w_high = (Benchmark_middle - Benchmark_low) / (Benchmark_high - Benchmark_low)

        Where the daily returns within each period are sorted by the benchmark return, and low, middle and high
        are the days with the lowest, the median (position n // 2) and the highest benchmark return. H compares
        the asset's return on the median day with the straight line through its returns on the two extreme
        days. A negative H means the returns are concave in the market's (fragile: the losses accelerate in a
        large fall), a positive H that they are convex (antifragile).

        For more information, see the following source:

        - Taleb, N.N., Canetti, E., Kinda, T., Loukoianova, E., & Schmieder, C. (2012). "A New Heuristic Measure
        of Fragility and Tail Risks: Application to Stress Testing." IMF Working Paper WP/12/216.

        Also known as: fragility heuristic, beta fragility heuristic.

        Referenced against:
        - empyrical, beta_fragility_heuristic: https://github.com/quantopian/empyrical/blob/0.5.5/empyrical/stats.py#L1565

        Args:
            period (str, optional): The period to use for the calculation. Options are: "daily", "weekly",
                "monthly", "quarterly", "yearly". Defaults to "quarterly" if the Toolkit is initialised with
                quarterly=True, otherwise "yearly".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The beta fragility heuristic per period and ticker, a return.

        Notes:
        - Equals empyrical's beta_fragility_heuristic on the daily returns within each period. When several days
        share the median benchmark return, the Toolkit takes them in date order (a stable sort), while empyrical
        leaves their order to its sorting algorithm, so the two can pick a different middle day.
        - The heuristic rests on three days of each period, so it is noisy; compare it across tickers or with a
        longer period rather than reading a single value.
        - With growth, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["JPM", "KO", "NVDA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.performance.get_beta_fragility()
        ```

        Which returns:

        | Date   |     JPM |      KO |    NVDA |
        |:-------|--------:|--------:|--------:|
        | 2021   | -0.0132 |  0.0046 | -0.0032 |
        | 2022   | -0.019  |  0      |  0.0179 |
        | 2023   | -0.0052 |  0.0082 | -0.0109 |
        | 2024   |  0.0417 | -0.0025 | -0.0068 |
        | 2025   | -0.0313 | -0.0263 |  0.033  |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"
        returns = self._get_column(period, "Return", within_period=True)
        benchmark_returns = self._get_column(
            period, "Return", within_period=True, benchmark=True
        )

        beta_fragility = performance_model.get_beta_fragility_heuristic(
            returns, benchmark_returns
        )

        return finalize_dataset(
            dataset=beta_fragility,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_portfolio
    @handle_errors
    def get_stability_of_returns(
        self,
        period: str | None = None,
        within_period: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the stability of the returns of an investment portfolio or asset: how closely its growth
        follows a straight line on a logarithmic scale, a constant rate of growth.

        The formula is as follows:

        - Cumulative Log Return_t = SUM(LN(1 + Return)) up to day t
        - Stability = R Squared of the regression of Cumulative Log Return_t on t

        Where t counts the returns (0, 1, 2, ...). A value close to 1 means the asset grew (or fell) at a
        steady rate; a value close to 0 means its path wandered without a trend. The measure says nothing about
        the direction: a steady decline is as stable as a steady rise, so read it together with the return.

        The measure is defined in empyrical (Quantopian) as stability_of_timeseries and reported in pyfolio's
        tear sheets; the Toolkit applies it to the returns of each ticker and period.


        Also known as: stability of timeseries, R-squared of the equity curve.

        Referenced against:
        - empyrical, stability_of_timeseries: https://github.com/quantopian/empyrical/blob/0.5.5/empyrical/stats.py#L1471

        Shares similar logic:
        - pyfolio, stability_of_timeseries: https://github.com/quantopian/pyfolio/blob/0.9.2/pyfolio/timeseries.py#L368

        Args:
            period (str, optional): The period to use for the calculation. Options are: "daily", "weekly",
                "monthly", "quarterly", "yearly". Defaults to "quarterly" if the Toolkit is initialised with
                quarterly=True, otherwise "yearly".
            within_period (bool, optional): Whether to calculate the stability from the daily returns within each
                period or from the period returns of the entirety of all periods, one value per ticker.
                Defaults to True.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The stability per period and ticker, or with within_period=False a Series with one
            value per ticker.

        Notes:
        - Equals empyrical's (and pyfolio's) stability_of_timeseries on the same returns. Each ticker is
        measured over its own returns, missing returns left out.
        - With growth, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "KO"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.performance.get_stability_of_returns()
        ```

        Which returns:

        | Date   |   AAPL |   MSFT |     KO |
        |:-------|-------:|-------:|-------:|
        | 2021   | 0.7053 | 0.9381 | 0.7069 |
        | 2022   | 0.4067 | 0.7082 | 0.0116 |
        | 2023   | 0.6698 | 0.8243 | 0.1864 |
        | 2024   | 0.7797 | 0.2793 | 0.5094 |
        | 2025   | 0.3558 | 0.6004 | 0.2276 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"
        returns = self._get_column(period, "Return", within_period=within_period)

        stability = performance_model.get_stability_of_returns(returns)

        return finalize_dataset(
            dataset=stability,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            apply_slice=within_period,
            dropna=True,
        )

    @handle_portfolio
    @handle_errors
    def get_r_squared(
        self,
        period: str | None = None,
        rolling: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the R-squared of an investment portfolio or asset against its benchmark: the share of the
        variance of its returns that the market explains in the CAPM regression on the benchmark.

        The formula is as follows:

        - R Squared = Covariance(Returns, Benchmark Returns)^2 / (Variance(Returns) x Variance(Benchmark Returns))

        With one regressor this is the squared correlation with the benchmark. An R-squared close to 1 means the
        asset moves almost entirely with the market, so its Beta and Alpha describe it well; a low R-squared
        means most of its risk is its own (see `get_risk_decomposition`), and a Beta or Treynor ratio says little
        about it. Funds that hug their index have an R-squared above 0.95.

        For more information, see the following source:

        - Bacon, C.R. (2008). "Practical Portfolio Performance Measurement and Attribution."
        2nd edition, John Wiley & Sons.

        Also known as: coefficient of determination, R2.

        Referenced against:
        - quantstats, r_squared: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L2828

        Args:
            period (str, optional): The period to use for the calculation. Options are: "daily", "weekly",
                "monthly", "quarterly", "yearly". Defaults to "quarterly" if the Toolkit is initialised with
                quarterly=True, otherwise "yearly".
            rolling (int, optional): The rolling window size to use for the calculation. If you select
                period = 'monthly' and set rolling to 36 you obtain the rolling three year R-squared of the
                monthly returns. At least 2. Defaults to None, which uses the daily returns within each period.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: R-squared values.

        Notes:
        - Equals quantstats' r_squared on the same returns. Each ticker is paired with the benchmark on the days
        both have a return.
        - The R Squared of `get_market_timing_model` includes its timing term,
        so it is at least this one.
        - With growth, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["SPY", "QQQ", "GLD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.performance.get_r_squared()
        ```

        Which returns:

        | Date   |    QQQ |    GLD |
        |:-------|-------:|-------:|
        | 2021   | 0.7625 | 0.0365 |
        | 2022   | 0.9351 | 0.0377 |
        | 2023   | 0.8404 | 0      |
        | 2024   | 0.8917 | 0.0707 |
        | 2025   | 0.9449 | 0.0007 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"
        check_rolling(rolling, self._historical_data, period)
        returns = self._get_column(period, "Return", within_period=not rolling)
        benchmark_returns = self._get_column(
            period, "Return", within_period=not rolling, benchmark=True
        )

        if rolling:
            r_squared = performance_model.get_r_squared(
                returns, benchmark_returns, window=rolling
            )
        else:
            r_squared = performance_model.get_r_squared(returns, benchmark_returns)

        return finalize_dataset(
            dataset=r_squared,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_portfolio
    @handle_errors
    def get_risk_return_ratio(
        self,
        period: str | None = None,
        rolling: int | None = None,
        annualize: bool = False,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the risk-return ratio of an investment portfolio or asset: its mean return per unit of
        volatility, the Sharpe ratio without the risk free rate.

        The formula is as follows:

        - Risk-Return Ratio = Mean(Returns) / Standard Deviation(Returns)

        Computed from the daily returns within each period (or the period returns over a rolling window), per
        observation; annualizing multiplies it by the square root of the observations in a year (252
        daily returns, 365 for a ticker that trades on weekends, or 52, 12, 4 or 1 for rolling weekly, monthly,
        quarterly or yearly returns). Leaving out the risk free rate makes it comparable across currencies and
        rate environments, but it overstates the reward of a low-risk asset that earns little more than cash;
        `get_sharpe_ratio` subtracts the risk free rate.

        The ratio under this name is defined in quantstats (Ran Aroussi) and ffn (Philippe Morissette).


        Also known as: return to risk ratio, return over volatility, information ratio against cash.

        Referenced against:
        - quantstats, smart_sharpe (with get_autocorrelation_penalty): https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L1000

        Shares similar logic:
        - quantstats, rolling_sharpe: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L1030
        - PerformanceAnalytics (R), SharpeRatio: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/SharpeRatio.R#L80
        - empyrical, sharpe_ratio: https://github.com/quantopian/empyrical/blob/0.5.5/empyrical/stats.py#L652
        - quantstats, sharpe: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L938
        - quantstats, risk_return_ratio: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L2603
        - ffn, calc_risk_return_ratio: https://github.com/pmorissette/ffn/blob/v1.4.1/ffn/core.py#L1592

        Args:
            period (str, optional): The period to use for the calculation. Options are: "daily", "weekly",
                "monthly", "quarterly", "yearly". Defaults to "quarterly" if the Toolkit is initialised with
                quarterly=True, otherwise "yearly".
            rolling (int, optional): The rolling window size to use for the calculation. If you select
                period = 'monthly' and set rolling to 12 you obtain the rolling 12-month ratio. At least 2. Defaults to None.
            annualize (bool, optional): Whether to annualize the ratio, multiplying it by the square root of the
                observations in a year. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Risk-return ratio values.

        Notes:
        - Equals quantstats' risk_return_ratio (not annualized) and, annualized, ffn's
        calc_risk_return_ratio on the same returns. Returns without variation give NaN.
        - With growth, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "KO"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.performance.get_risk_return_ratio(annualize=True)
        ```

        Which returns:

        | Date   |    AAPL |    MSFT |      KO |
        |:-------|--------:|--------:|--------:|
        | 2021   |  1.4198 |  2.2294 |  1.0697 |
        | 2022   | -0.6845 | -0.7585 |  0.613  |
        | 2023   |  2.083  |  1.9698 | -0.2722 |
        | 2024   |  1.2937 |  0.7093 |  0.7286 |
        | 2025   |  0.4286 |  0.7212 |  0.921  |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"
        check_rolling(rolling, self._historical_data, period)
        returns = self._get_column(period, "Return", within_period=not rolling)

        # The Sharpe ratio of the returns themselves, without a risk free rate.
        if rolling:
            risk_return_ratio = performance_model.get_sharpe_ratio(
                returns, window=rolling
            )
        else:
            risk_return_ratio = performance_model.get_sharpe_ratio(returns)

        if annualize:
            risk_return_ratio = risk_return_ratio * np.sqrt(
                self._get_ratio_observations_per_year(
                    period, rolling, risk_return_ratio.columns
                ).to_numpy()
            )

        return finalize_dataset(
            dataset=risk_return_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_errors
    def get_risk_decomposition(
        self,
        period: str | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the systematic, specific and total risk of an investment portfolio or asset: its volatility
        split into the part that comes from the market, which diversification cannot remove, and the part that
        is its own.

        The formulas are as follows:

        - Systematic Risk = Beta x Volatility of the Benchmark Excess Returns
        - Specific Risk = Volatility of the Residuals
        - Total Risk = SQRT(Systematic Risk^2 + Specific Risk^2)

        Where Beta and the Residuals come from the CAPM regression of the daily excess returns on the
        benchmark's within each period, Excess Return = Alpha + Beta x Benchmark Excess Return + Residual. The
        volatilities are sample standard deviations of the daily values scaled to the period (times the square
        root of 252 trading days for a year, 365 calendar days for a ticker that trades on weekends), so the
        yearly figures are annualized. With the same degrees of freedom the regression splits the variance
        exactly, so the Total Risk equals the volatility of the excess returns. A specific risk that dominates
        the total means the asset's moves are mostly its own, as a low R-squared (see `get_r_squared`) shows.

        For more information, see the following source:

        - Bacon, C.R. (2008). "Practical Portfolio Performance Measurement and Attribution."
        2nd edition, John Wiley & Sons, pp. 71-72.

        Also known as: market and idiosyncratic risk, diversifiable and non-diversifiable risk, residual risk.

        Referenced against:
        - PerformanceAnalytics (R), SystematicRisk: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/SystematicRisk.R#L36
        - PerformanceAnalytics (R), SpecificRisk: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/SpecificRisk.R#L27

        Shares similar logic:
        - PerformanceAnalytics (R), TotalRisk: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/TotalRisk.R#L30

        Args:
            period (str, optional): The period to use for the calculation. Options are: "daily", "weekly",
                "monthly", "quarterly", "yearly". Defaults to "quarterly" if the Toolkit is initialised with
                quarterly=True, otherwise "yearly".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The Systematic Risk, Specific Risk and Total Risk per period, with a Multi Index of
            (ticker, risk) as the columns.

        Notes:
        - The Systematic Risk agrees with PerformanceAnalytics' SystematicRisk on the daily returns of a year. Its
        SpecificRisk divides the squared residuals by n instead of n - 1, so it is SQRT((n - 1) / n) times the
        Toolkit's (0.2% lower for a year of daily returns), and its TotalRisk combines the two.
        - The specific risk here is that of the CAPM regression on the benchmark; the idiosyncratic volatility
        of the risk module uses the residuals of the Fama-French factor model instead.
        - The Portfolio of a Portfolio's Toolkit is measured on its own daily return series, as the risks of
        a portfolio do not add up as a weighted average.
        - With growth, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "XOM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.performance.get_risk_decomposition().xs("XOM", level=0, axis=1)
        ```

        Which returns:

        |      |   Systematic Risk |   Specific Risk |   Total Risk |
        |:-----|------------------:|----------------:|-------------:|
        | 2021 |            0.1274 |          0.2663 |       0.2952 |
        | 2022 |            0.1293 |          0.3262 |       0.3509 |
        | 2023 |            0.0742 |          0.2395 |       0.2508 |
        | 2024 |            0.0233 |          0.191  |       0.1924 |
        | 2025 |            0.1007 |          0.2113 |       0.2341 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"
        excess_return = self._get_column(period, "Excess Return", within_period=True)
        benchmark_excess_return = self._get_column(
            period, "Excess Return", within_period=True, benchmark=True
        )
        # The daily returns in a period of each ticker's calendar scale the daily risks to it
        # (a Series, as the returns are a DataFrame).
        days_per_period = (
            pd.Series(
                get_days_per_period(
                    self._get_column("daily", "Return", within_period=False), period
                )
            )
            if period != "daily"
            else pd.Series(1.0, index=excess_return.columns)
        )

        risk_decomposition = {
            ticker: pd.DataFrame.from_dict(
                {
                    sub_period: performance_model.get_risk_decomposition(
                        excess_return.loc[sub_period, ticker],
                        benchmark_excess_return.loc[sub_period],
                        days_per_period[ticker],
                    )
                    for sub_period in excess_return.index.get_level_values(0).unique()
                },
                orient="index",
            )
            for ticker in excess_return.columns
        }

        return finalize_dataset(
            dataset=pd.concat(risk_decomposition, axis=1),
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_errors
    def get_capital_market_line(
        self,
        period: str | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Capital Market Line and Security Market Line of the benchmark and where an investment
        portfolio or asset stands against them (Sharpe, 1964), with the regression epsilon of Bacon (2008).

        The formulas are as follows:

        - Capital Market Line Slope = Mean(Benchmark Excess Return) / Standard Deviation(Benchmark Return)
        - Capital Market Line Return = Mean(Risk Free Rate) + Capital Market Line Slope x Standard Deviation(Return)
        - Security Market Line Slope = Mean(Benchmark Excess Return)
        - Regression Epsilon = Excess Return of the Period - Alpha - Beta x Benchmark Excess Return of the Period

        The first three are per day, from the daily returns within each period. The CML slope is the
        benchmark's Sharpe ratio, the extra return the market paid per unit of volatility; the CML return is
        what a mix of the benchmark and the risk free asset with the asset's volatility earned per day, so an
        asset whose mean daily return lies below it was not rewarded for its total risk. The SML slope is the
        market risk premium per day, the extra return the CAPM gives per unit of Beta. The Regression Epsilon
        is the part of the period's excess return the CAPM regression (Alpha and Beta per day, fitted on the
        daily excess returns) does not predict.

        For more information, see the following sources:

        - Sharpe, W.F. (1964). "Capital Asset Prices: A Theory of Market Equilibrium under Conditions of
        Risk." Journal of Finance, 19(3), 425-442.
        - Bacon, C.R. (2008). "Practical Portfolio Performance Measurement and Attribution."
        2nd edition, John Wiley & Sons, p. 71.

        Also known as: CML, SML, capital allocation line of the market, regression epsilon.

        Referenced against:
        - PerformanceAnalytics (R), CAPM.CML.slope: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/CAPM.utils.R#L6
        - PerformanceAnalytics (R), CAPM.SML.slope: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/CAPM.utils.R#L152
        - PerformanceAnalytics (R), CAPM.epsilon: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/CAPM.epsilon.R#L33

        Args:
            period (str, optional): The period to use for the calculation. Options are: "daily", "weekly",
                "monthly", "quarterly", "yearly". Defaults to "quarterly" if the Toolkit is initialised with
                quarterly=True, otherwise "yearly".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The Capital Market Line Slope, Capital Market Line Return, Security Market Line Slope
            and Regression Epsilon per period, with a Multi Index of (ticker, statistic) as the columns. The
            slopes belong to the benchmark and repeat for every ticker, measured over the days it has a return.

        Notes:
        - The Capital Market Line Slope agrees with PerformanceAnalytics' CAPM.CML.slope. PerformanceAnalytics
        reports the reciprocal of the market risk premium as its CAPM.SML.slope (the slope with Beta on the
        vertical axis) and its CAPM.CML multiplies the slope by the asset's mean return instead of its volatility;
        the Toolkit follows Sharpe (1964).
        - The Regression Epsilon agrees with PerformanceAnalytics' CAPM.epsilon with a risk free rate of 0, where it
        uses the annualized returns as the period's. With a risk free rate PerformanceAnalytics adds it to the
        asset's return instead of subtracting it; the Toolkit uses excess returns throughout.
        - The Portfolio of a Portfolio's Toolkit is regressed on its own daily return series, as the market
        timing models do.
        - With growth, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "KO"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.performance.get_capital_market_line().xs("KO", level=0, axis=1).T
        ```

        Which returns:

        |                            |    2021 |    2022 |    2023 |   2024 |   2025 |
        |:---------------------------|--------:|--------:|--------:|-------:|-------:|
        | Capital Market Line Slope  |  0.1274 | -0.0523 |  0.0983 | 0.0947 | 0.0457 |
        | Capital Market Line Return |  0.0012 | -0.0005 |  0.001  | 0.0009 | 0.0007 |
        | Security Market Line Slope |  0.001  | -0.0008 |  0.0008 | 0.0007 | 0.0006 |
        | Regression Epsilon         | -0.0104 |  0.179  | -0.1686 | 0.0357 | 0.1063 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"
        returns = self._get_column(period, "Return", within_period=True)
        excess_return = self._get_column(period, "Excess Return", within_period=True)
        benchmark_returns = self._get_column(
            period, "Return", within_period=True, benchmark=True
        )
        benchmark_excess_return = self._get_column(
            period, "Excess Return", within_period=True, benchmark=True
        )

        # The period excess returns compound the daily returns within each period, the same
        # returns the regression uses, so the first period has one as well, less the risk free
        # rate of each ticker's own dates, the benchmark measured against the same rate.
        risk_free_rate = self._risk_free_rate_data[period]
        period_returns = (1 + returns).groupby(level=0).prod(min_count=1) - 1
        period_excess_return = period_returns - risk_free_rate[period_returns.columns]
        benchmark_period_excess_return = risk_free_rate[period_returns.columns].rsub(
            (1 + benchmark_returns).groupby(level=0).prod(min_count=1) - 1,
            axis=0,
        )

        capital_market_line = {}
        for ticker in returns.columns:
            ticker_results = {}

            for sub_period in returns.index.get_level_values(0).unique():
                ticker_results[sub_period] = performance_model.get_capital_market_line(
                    returns.loc[sub_period, ticker],
                    benchmark_returns.loc[sub_period],
                    returns.loc[sub_period, ticker]
                    - excess_return.loc[sub_period, ticker],
                )
                ticker_results[sub_period]["Regression Epsilon"] = (
                    performance_model.get_regression_epsilon(
                        excess_return.loc[sub_period, ticker],
                        benchmark_excess_return.loc[sub_period],
                        period_excess_return[ticker].get(sub_period, np.nan),
                        benchmark_period_excess_return[ticker].get(sub_period, np.nan),
                    )
                )

            capital_market_line[ticker] = pd.DataFrame.from_dict(
                ticker_results, orient="index"
            )

        return finalize_dataset(
            dataset=pd.concat(capital_market_line, axis=1),
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_portfolio
    @handle_errors
    def get_kappa_ratio(
        self,
        period: str | None = None,
        order: int = 3,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Kappa Ratio: the excess return an asset earns per unit of the lower partial moment
        of a chosen order, a generalization of the Sortino ratio that weighs large losses more as the
        order rises.

        The Kappa Ratio is a generalization of the Sortino Ratio that penalizes downside risk
        using a higher-order lower partial moment. The Sortino Ratio is the special case of the
        Kappa Ratio with an order of 2.

        Note that this already subtracts the Risk Free Rate.

        The formula is as follows:

        - Daily Excess Return = Daily Return - Daily Risk Free Rate, with Daily Risk Free Rate =
          (1 + Annual Risk Free Rate)^(1/252) - 1 (compounded over the calendar days since the previous return,
          on 365 days a year, when a ticker trades on weekends)
        - Lower Partial Moment (LPM) = MEAN(MAX(-Daily Excess Return, 0)^Order), averaged over all days in the
          period (days with a non-negative excess return count as 0)
        - Kappa Ratio = MEAN(Daily Excess Return) / LPM^(1/Order), per period and not annualized, with Order
          defaulting to 3

        Shares similar logic:
        - PerformanceAnalytics (R), Kappa: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/Kappa.R#L52

        Args:
            period (str, optional): The period to use for the calculation. Options are: "daily", "weekly",
                "monthly", "quarterly", "yearly". Defaults to "quarterly" if the Toolkit is initialised with
                quarterly=True, otherwise "yearly".
            order (int, optional): The order of the lower partial moment used in the denominator.
            Defaults to 3.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Kappa Ratio values.

        Notes:
        - Daily Kappa Ratio is not an option as the standard deviation for 1 day is close to zero.
        Therefore, it does not give any useful insights.
        - The method retrieves historical data and calculates the Kappa Ratio for each asset in
        the Toolkit instance.
        - The risk-free rate is the Toolkit's risk free rate, the 10-year US Treasury yield by default. Many
        textbook definitions (Sharpe, 1994; the Ken French factors) use a short-term Treasury bill instead, which
        the Toolkit uses when initialised with the 13-week Treasury bill as its risk free rate.
        - With growth, the method calculates the growth of the ratio values using
        the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["LLY", "MRK", "ABBV"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.performance.get_kappa_ratio()
        ```

        Which returns:

        | Date   |    LLY |     MRK |    ABBV |
        |:-------|-------:|--------:|--------:|
        | 2021   | 0.1248 |  0.0091 |  0.0811 |
        | 2022   | 0.0844 |  0.1426 |  0.0552 |
        | 2023   | 0.1351 | -0.004  | -0.0059 |
        | 2024   | 0.0652 | -0.0191 |  0.0298 |
        | 2025   | 0.0513 |  0.0193 |  0.064  |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"
        excess_return = self._get_column(period, "Excess Return", within_period=True)

        kappa_ratio = performance_model.get_kappa_ratio(excess_return, order)

        return finalize_dataset(
            dataset=kappa_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_portfolio
    @handle_errors
    def get_omega_ratio(
        self,
        period: str | None = None,
        within_period: bool = True,
        rolling: int | None = None,
        minimum_acceptable_return: float = 0.0,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Omega Ratio: the probability-weighted gains above a threshold divided by the
        probability-weighted losses below it, a ratio of upside to downside that uses the whole
        distribution of the returns.

        The Omega Ratio is a risk-return measure that divides the sum of gains above a minimum
        acceptable return (MAR) by the sum of losses below it, capturing the full shape of the
        return distribution rather than only its first two moments (unlike the Sharpe Ratio).

        The formula is as follows:

        - Omega Ratio = SUM(Gains above MAR) / SUM(Losses below MAR)

        See definition: https://en.wikipedia.org/wiki/Omega_ratio

        Shares similar logic:
        - PerformanceAnalytics (R), Omega: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/Omega.R#L72
        - empyrical, omega_ratio: https://github.com/quantopian/empyrical/blob/0.5.5/empyrical/stats.py#L599
        - quantstats, omega: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L1508

        Args:
            period (str, optional): The period to use for the calculation. Options are: "daily", "weekly",
                "monthly", "quarterly", "yearly". Defaults to "quarterly" if the Toolkit is initialised with
                quarterly=True, otherwise "yearly".
            within_period (bool, optional): Whether to calculate the Omega Ratio within the specified
            period or for the entire period. Thus whether to look at the Omega Ratio within a specific
            year (if period = 'yearly') or look at the entirety of all years. Defaults to True.
            rolling (int, optional): The rolling window size to use for the calculation. If set, the
            Omega Ratio is calculated over a rolling window of this many periods across the full
            return history instead of per `period`. At least 2. Defaults to None.
            minimum_acceptable_return (float, optional): The minimum acceptable return (MAR) used as
            the threshold between gains and losses. Defaults to 0.0.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Omega Ratio values.

        Notes:
        - The method retrieves historical data and calculates the Omega Ratio for each asset in
        the Toolkit instance.
        - With growth, the method calculates the growth of the ratio values using
        the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["BLK", "SCHW", "BX"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.performance.get_omega_ratio()
        ```

        Which returns:

        |      |    BLK |   SCHW |     BX |
        |:-----|-------:|-------:|-------:|
        | 2021 | 1.2446 | 1.3257 | 1.5958 |
        | 2022 | 0.9272 | 1.0318 | 0.8945 |
        | 2023 | 1.1525 | 0.9625 | 1.3792 |
        | 2024 | 1.2732 | 1.094  | 1.2109 |
        | 2025 | 1.0728 | 1.2867 | 0.9944 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"
        check_rolling(rolling, self._historical_data, period)

        if rolling:
            returns = self._get_column(period, "Return", within_period=False)
            omega_ratio = performance_model.get_omega_ratio(
                returns, minimum_acceptable_return, window=rolling
            )
        else:
            returns = self._get_column(period, "Return", within_period=within_period)

            omega_ratio = performance_model.get_omega_ratio(
                returns, minimum_acceptable_return
            )

        return finalize_dataset(
            dataset=omega_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            apply_slice=bool(rolling) or within_period,
        )

    def _get_morningstar_risk_adjusted_return(
        self,
        gamma: float,
        period: str,
        within_period: bool,
        rolling: int | None,
    ) -> pd.DataFrame | pd.Series:
        """
        The Morningstar Risk-Adjusted Return of every ticker from its monthly geometric excess
        returns, per period, over the whole range or over a rolling window of months.
        """
        if period not in ("yearly", "quarterly"):
            raise InvalidArgumentError(
                "The Morningstar Risk-Adjusted Return groups monthly returns, so the period "
                "must be yearly or quarterly."
            )

        excess_returns = performance_model.get_geometric_excess_return(
            self._get_column("monthly", "Return", within_period=False),
            self._risk_free_rate_data["monthly"][self._tickers_with_returns],
        )

        if rolling:
            return performance_model.get_morningstar_risk_adjusted_return(
                excess_returns, gamma, window=rolling
            )

        if within_period:
            groups = excess_returns.index.asfreq("Y" if period == "yearly" else "Q")
            return excess_returns.groupby(groups).apply(
                lambda group: performance_model.get_morningstar_risk_adjusted_return(
                    group, gamma
                )
            )

        return performance_model.get_morningstar_risk_adjusted_return(
            excess_returns, gamma
        )

    @handle_errors
    def get_morningstar_risk_adjusted_return(
        self,
        gamma: float = 2.0,
        period: str = "yearly",
        within_period: bool = True,
        rolling: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Morningstar Risk-Adjusted Return (MRAR): the return that penalizes the variation
        of the returns as a risk-averse investor would, the measure Morningstar ranks funds on within
        their category for its star rating.

        MRAR is the annualized certainty equivalent of the monthly geometric excess returns for
        an investor with risk aversion gamma: the return such an investor would accept with
        certainty instead of the uncertain returns. It equals the annualized geometric mean
        excess return when every month returns the same and falls further below it the more
        the returns vary, penalizing downside variation most. Morningstar uses a gamma of 2 and
        rates funds on 36, 60 and 120 months (three, five and ten years), which rolling windows
        of 36, 60 and 120 months give, weighting them 20%, 30% and 50% in the overall rating.
        For US dollar funds its risk-free rate is the three-month Treasury bill (auction
        average), which the 13-week Treasury bill rate of the Toolkit follows. Morningstar also adjusts for
        tax-advantaged dividends where appropriate, which these total returns don't.

        The formula is as follows:

        - Geometric Excess Return = (1 + Monthly Total Return) / (1 + Monthly Risk Free Rate) - 1
        - MRAR(gamma) = [1/T * SUM((1 + Geometric Excess Return t)^-gamma)]^(-12/gamma) - 1

        with T the number of months. With a gamma of 0 it is the annualized geometric mean of
        the excess returns, Morningstar Return.

        For more information, see the following:

        - Morningstar (2023). "The Morningstar Rating for Funds." Morningstar Methodology, April 2023.
          https://s25.q4cdn.com/634000362/files/doc_downloads/1/1157231.pdf
        - Morningstar (2009). "MRAR Illustrated."
          https://www.morningstar.com/content/dam/marketing/apac/au/pdfs/Legal/MRARIllustrated_Factsheet.pdf

        Args:
            gamma (float, optional): The risk aversion, 0 or higher. Defaults to 2.0, which
                Morningstar uses for its ratings.
            period (str, optional): The period to group the monthly returns by. Options are: "quarterly",
                "yearly". Defaults to "yearly".
            within_period (bool, optional): Whether to calculate the MRAR of the months within
                each period or of all months in the range. Defaults to True.
            rolling (int, optional): The number of months in a rolling window, e.g. 36 for the
                three year MRAR at every month end. Overrides period and within_period.
                At least 2. Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the rounding of the Toolkit (4 unless set
                otherwise).
            growth (bool, optional): Whether to calculate the growth of the values. Defaults to
                False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The MRAR per period (rows) and ticker (columns) as decimals, or one
            value per ticker over the whole range with within_period=False.

        Notes:
        - The returns are monthly total returns from the adjusted close, so dividends count.
          Morningstar also deducts sales loads, which an exchange-traded fund does not have.
        - Morningstar uses a three-month Treasury bill as the risk-free rate for US funds,
          which `Toolkit(..., risk_free_rate="13w")` matches; the Toolkit's default is the
          10-year Treasury yield, each converted to a monthly return.
        - A month's risk-free return is the month's average yield, compounded to a month.
        - A period is annualized from the months it has: the first month of the range has no
          return (it has no earlier close), so the first year counts eleven months, which is
          why 2021 in the example is above the funds' calendar-year return.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["VOO", "QQQ", "SCHD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
            risk_free_rate="13w",
        )

        toolkit.performance.get_morningstar_risk_adjusted_return()
        ```

        Which returns:

        | Date   |     VOO |     QQQ |    SCHD |
        |:-------|--------:|--------:|--------:|
        | 2021   |  0.3176 |  0.278  |  0.3209 |
        | 2022   | -0.235  | -0.3836 | -0.0848 |
        | 2023   |  0.1799 |  0.4301 | -0.0227 |
        | 2024   |  0.1789 |  0.1822 |  0.0479 |
        | 2025   |  0.1198 |  0.1351 | -0.0098 |
        """
        check_rolling(rolling, self._historical_data, "monthly")
        mrar = self._get_morningstar_risk_adjusted_return(
            gamma, period, within_period, rolling
        )

        return finalize_dataset(
            dataset=mrar,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            apply_slice=bool(rolling) or within_period,
        )

    @handle_errors
    def get_morningstar_risk(
        self,
        gamma: float = 2.0,
        period: str = "yearly",
        within_period: bool = True,
        rolling: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate Morningstar Risk, the return a risk-averse investor would give up to remove
        the variation of the returns: Morningstar Return (the annualized geometric mean excess
        return) minus the Morningstar Risk-Adjusted Return. It is zero for returns that never
        vary and grows with their variation, most with variation on the downside. Morningstar
        rates funds Low to High on it within their category, over 36, 60 and 120 months
        (rolling windows of 36, 60 and 120).

        The formula is as follows:

        - Morningstar Risk = MRAR(0) - MRAR(gamma)

        with MRAR the Morningstar Risk-Adjusted Return (see get_morningstar_risk_adjusted_return)
        and MRAR(0) Morningstar Return.

        For more information, see the following:

        - Morningstar (2023). "The Morningstar Rating for Funds." Morningstar Methodology, April 2023.
          https://s25.q4cdn.com/634000362/files/doc_downloads/1/1157231.pdf

        Args:
            gamma (float, optional): The risk aversion, 0 or higher. Defaults to 2.0, which
                Morningstar uses for its ratings.
            period (str, optional): The period to group the monthly returns by. Options are: "quarterly",
                "yearly". Defaults to "yearly".
            within_period (bool, optional): Whether to calculate the risk of the months within
                each period or of all months in the range. Defaults to True.
            rolling (int, optional): The number of months in a rolling window, e.g. 36 for the
                three year figure at every month end. Overrides period and within_period.
                At least 2. Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the rounding of the Toolkit (4 unless set
                otherwise).
            growth (bool, optional): Whether to calculate the growth of the values. Defaults to
                False.
            lag (int | list[int], optional): The lag to use for the growth calculation.
                Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Morningstar Risk per period (rows) and ticker (columns) as decimals,
            or one value per ticker over the whole range with within_period=False.

        Notes:
        - The returns and the risk-free rate are those of get_morningstar_risk_adjusted_return;
          `Toolkit(..., risk_free_rate="13w")` matches Morningstar's three-month Treasury bill.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["VOO", "QQQ", "SCHD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
            risk_free_rate="13w",
        )

        toolkit.performance.get_morningstar_risk()
        ```

        Which returns:

        | Date   |    VOO |    QQQ |   SCHD |
        |:-------|-------:|-------:|-------:|
        | 2021   | 0.0148 | 0.0205 | 0.0215 |
        | 2022   | 0.0375 | 0.0449 | 0.0338 |
        | 2023   | 0.023  | 0.0444 | 0.0183 |
        | 2024   | 0.0122 | 0.0147 | 0.0162 |
        | 2025   | 0.0125 | 0.0256 | 0.0126 |
        """
        check_rolling(rolling, self._historical_data, "monthly")
        morningstar_return = self._get_morningstar_risk_adjusted_return(
            0, period, within_period, rolling
        )
        mrar = self._get_morningstar_risk_adjusted_return(
            gamma, period, within_period, rolling
        )

        return finalize_dataset(
            dataset=morningstar_return - mrar,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            apply_slice=bool(rolling) or within_period,
        )

    @handle_portfolio
    @handle_errors
    def get_gain_to_pain_ratio(
        self,
        period: str | None = None,
        within_period: bool = True,
        minimum_acceptable_return: float = 0.0,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Gain-to-Pain Ratio: the sum of the returns divided by the sum of the absolute
        losses, how much an asset gained for the losses it went through.

        The Gain-to-Pain Ratio, popularized by Jack Schwager, divides the sum of all returns by
        the sum of the absolute value of all losses, summarizing the entire return history into
        a single measure of return earned per unit of pain endured.

        The formula is as follows:

        - Gain-to-Pain Ratio = SUM(Returns) / SUM(|Losses|)

        Measured against a minimum acceptable return (MAR) it is the Omega-Sharpe ratio of Kazemi,
        Schneeweis and Gupta (2004), which is the Omega Ratio (see `get_omega_ratio`) minus 1:

        - Omega-Sharpe Ratio = SUM(Returns − MAR) / SUM(MAX(MAR − Returns, 0)) = Omega Ratio − 1

        For more information, see the following sources:

        - Schwager, J.D. (2012). "Hedge Fund Market Wizards: How Winning Traders Win." John Wiley &
        Sons.
        - Kazemi, H., Schneeweis, T., & Gupta, R. (2004). "Omega as a Performance Measure."
        Journal of Performance Measurement, 8(3), 16-25.

        Also known as: Omega-Sharpe ratio.

        Referenced against:
        - PerformanceAnalytics (R), OmegaSharpeRatio: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/OmegaSharpeRatio.R#L37
        - quantstats, gain_to_pain_ratio: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L1584

        Args:
            period (str, optional): The period to use for the calculation. Options are: "daily", "weekly",
                "monthly", "quarterly", "yearly". Defaults to "quarterly" if the Toolkit is initialised with
                quarterly=True, otherwise "yearly".
            within_period (bool, optional): Whether to calculate the Gain-to-Pain Ratio within the
            specified period or for the entire period. Thus whether to look at the Gain-to-Pain Ratio
            within a specific year (if period = 'yearly') or look at the entirety of all years.
            Defaults to True.
            minimum_acceptable_return (float, optional): The minimum acceptable return (MAR) per
            daily return (per period return with within_period=False), which turns the ratio into
            the Omega-Sharpe ratio. Defaults to 0.0.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Gain-to-Pain Ratio values.

        Notes:
        - The method retrieves historical data and calculates the Gain-to-Pain Ratio for each asset
        in the Toolkit instance.
        - Equals quantstats' gain_to_pain_ratio and, with a minimum acceptable return, PerformanceAnalytics'
        OmegaSharpeRatio on the same returns.
        - With growth, the method calculates the growth of the ratio values using
        the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["GCUSD", "SIUSD", "HGUSD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.performance.get_gain_to_pain_ratio()
        ```

        Which returns:

        |      |   GCUSD |   SIUSD |   HGUSD |
        |:-----|--------:|--------:|--------:|
        | 2021 | -0.0617 | -0.0653 |  0.1797 |
        | 2022 |  0.0115 |  0.0413 | -0.0663 |
        | 2023 |  0.1812 |  0.0233 |  0.0337 |
        | 2024 |  0.3038 |  0.1403 |  0.0446 |
        | 2025 |  0.5152 |  0.5701 |  0.2573 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        returns = self._get_column(period, "Return", within_period=within_period)

        gain_to_pain_ratio = performance_model.get_gain_to_pain_ratio(
            returns, minimum_acceptable_return
        )

        return finalize_dataset(
            dataset=gain_to_pain_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            apply_slice=within_period,
        )

    @handle_portfolio
    @handle_errors
    def get_d_ratio(
        self,
        period: str | None = None,
        within_period: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the d ratio of an investment portfolio or asset's returns, the size of its losses
        relative to its gains, each weighted by how often they occur.

        The formula is as follows:

        - d Ratio = (n_d × SUM(|Returns| where Returns < 0)) / (n_u × SUM(Returns where Returns > 0))

        Where n_d is the number of negative and n_u the number of positive daily returns within the
        period. It is the inverse of the gain-loss (Bernardo-Ledoit) ratio weighted by the frequencies
        of the losses and gains. Lower is better: 0 means there was no loss at all, and a value
        above 1 means the losses weigh more than the gains.

        For more information, see the following source:

        - Bacon, C.R. (2008). "Practical Portfolio Performance Measurement and Attribution."
        2nd edition, John Wiley & Sons, p. 95.

        Referenced against:
        - PerformanceAnalytics (R), DRatio: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/DRatio.R#L38

        Args:
            period (str, optional): The period to use for the calculation. Options are: "daily", "weekly",
                "monthly", "quarterly", "yearly". Defaults to "quarterly" if the Toolkit is initialised with
                quarterly=True, otherwise "yearly".
            within_period (bool, optional): Whether to calculate the ratio from the daily returns
            within each period or from the period returns of the entirety of all periods, one value
            per ticker. Defaults to True.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: d ratio values, or with within_period=False a Series with one value per ticker.

        Notes:
        - The method retrieves historical data and calculates the d ratio for each asset in the Toolkit instance.
        - Equals PerformanceAnalytics' DRatio on the same returns. Without a positive return the ratio is NaN,
        where PerformanceAnalytics reports infinity.
        - With growth, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["KO", "PEP", "MDLZ"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.performance.get_d_ratio()
        ```

        Which returns:

        | Date   |     KO |    PEP |   MDLZ |
        |:-------|-------:|-------:|-------:|
        | 2021   | 0.6675 | 0.5798 | 0.6805 |
        | 2022   | 0.7072 | 0.8231 | 0.8447 |
        | 2023   | 0.9899 | 1.0095 | 0.7533 |
        | 2024   | 0.7724 | 1.1237 | 1.3704 |
        | 2025   | 0.8631 | 1.0868 | 1.038  |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        returns = self._get_column(period, "Return", within_period=within_period)

        d_ratio = performance_model.get_d_ratio(returns)

        return finalize_dataset(
            dataset=d_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            apply_slice=within_period,
            dropna=True,
        )

    @handle_portfolio
    @handle_errors
    def get_prospect_ratio(
        self,
        period: str | None = None,
        within_period: bool = True,
        minimum_acceptable_return: float = 0.0,
        loss_aversion: float = 2.25,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Prospect Ratio of an investment portfolio or asset's returns, a Sortino-type
        ratio in which losses weigh more heavily than gains, following the loss aversion of prospect
        theory: people feel a loss about 2.25 times as strongly as a gain of the same size.

        The formula is as follows:

        - Prospect Ratio = ((1 / N) × SUM(MAX(Returns, 0) + λ × MIN(Returns, 0)) − MAR) / Downside Deviation
        - Downside Deviation = SQRT( (1 / N) × SUM( MIN(Returns − MAR, 0)^2 ) )

        Where N is the number of daily returns within the period, MAR the minimum acceptable return
        per day and λ the loss aversion (2.25, Tversky and Kahneman, 1992). Like the Sortino ratio it
        is a per-observation ratio, not annualized.

        For more information, see the following sources:

        - Watanabe, Y. (2006). "Is Sharpe Ratio Still Effective?" Journal of Performance Measurement,
        11(1), 55-66.
        - Tversky, A., & Kahneman, D. (1992). "Advances in Prospect Theory: Cumulative Representation
        of Uncertainty." Journal of Risk and Uncertainty, 5(4), 297-323.
        - Bacon, C.R. (2008). "Practical Portfolio Performance Measurement and Attribution."
        2nd edition, John Wiley & Sons, p. 100.

        Referenced against:
        - PerformanceAnalytics (R), ProspectRatio: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/ProspectRatio.R#L32

        Args:
            period (str, optional): The period to use for the calculation. Options are: "daily", "weekly",
                "monthly", "quarterly", "yearly". Defaults to "quarterly" if the Toolkit is initialised with
                quarterly=True, otherwise "yearly".
            within_period (bool, optional): Whether to calculate the ratio from the daily returns
            within each period or from the period returns of the entirety of all periods, one value
            per ticker. Defaults to True.
            minimum_acceptable_return (float, optional): The minimum acceptable return (MAR) per daily
            return (per period return with within_period=False). Defaults to 0.0.
            loss_aversion (float, optional): The weight λ of the losses relative to the gains.
            Defaults to 2.25.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Prospect Ratio values, or with within_period=False a Series with one value per ticker.

        Notes:
        - The method retrieves historical data and calculates the Prospect Ratio for each asset in the Toolkit
        instance.
        - Equals PerformanceAnalytics' ProspectRatio for a MAR of 0 (the default). For another MAR the
        PerformanceAnalytics function subtracts MAR / N instead of the MAR from the average return, unlike the
        formula in its documentation and in Bacon (2008), which the Toolkit follows.
        - Without a return below the MAR the downside deviation is 0 and the ratio is undefined (NaN).
        - With growth, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["JPM", "BAC", "WFC"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.performance.get_prospect_ratio()
        ```

        Which returns:

        | Date   |     JPM |     BAC |     WFC |
        |:-------|--------:|--------:|--------:|
        | 2021   | -0.5512 | -0.4938 | -0.475  |
        | 2022   | -0.7413 | -0.7963 | -0.715  |
        | 2023   | -0.4453 | -0.6298 | -0.5457 |
        | 2024   | -0.3686 | -0.5124 | -0.4513 |
        | 2025   | -0.4087 | -0.4111 | -0.4329 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        returns = self._get_column(period, "Return", within_period=within_period)

        prospect_ratio = performance_model.get_prospect_ratio(
            returns, minimum_acceptable_return, loss_aversion
        )

        return finalize_dataset(
            dataset=prospect_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            apply_slice=within_period,
            dropna=True,
        )

    @handle_portfolio
    @handle_errors
    def get_omega_excess_return(
        self,
        period: str | None = None,
        minimum_acceptable_return: float = 0.0,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Omega Excess Return of an investment portfolio or asset, its return minus a
        charge for its downside risk, priced at the downside risk of the benchmark.

        The formula is as follows:

        - Omega Excess Return = Return − 3 × β_S × σ_DB^2 = Return − 3 × σ_D × σ_DB
        - β_S = σ_D / σ_DB

        Where Return is the return of the period, σ_D and σ_DB are the downside deviations of the daily
        returns of the asset and of the benchmark within the period below the minimum acceptable
        return, scaled to the period (times SQRT(252) for a year, SQRT(365) for a ticker that trades on
        weekends), and β_S is the style beta. The higher the downside risk of the asset relative to the
        benchmark, the more of its return is given up.

        For more information, see the following source:

        - Bacon, C.R. (2008). "Practical Portfolio Performance Measurement and Attribution."
        2nd edition, John Wiley & Sons, p. 103.

        Referenced against:
        - PerformanceAnalytics (R), OmegaExcessReturn: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/OmegaExcessReturn.R#L36

        Args:
            period (str, optional): The period to use for the calculation. Options are: "daily", "weekly",
                "monthly", "quarterly", "yearly". Defaults to "quarterly" if the Toolkit is initialised with
                quarterly=True, otherwise "yearly".
            minimum_acceptable_return (float, optional): The minimum acceptable return (MAR) per daily
            return that the downside deviations are measured below. Defaults to 0.0.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Omega Excess Return values.

        Notes:
        - The method retrieves historical data and calculates the Omega Excess Return for each asset in the
        Toolkit instance. It requires a benchmark_ticker.
        - PerformanceAnalytics' OmegaExcessReturn uses the annualized return and downside deviations over the
        whole sample; the Toolkit uses those of each period, with the same formula.
        - With growth, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["XOM", "CVX", "COP"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.performance.get_omega_excess_return()
        ```

        Which returns:

        | Date   |     XOM |     CVX |     COP |
        |:-------|--------:|--------:|--------:|
        | 2022   |  0.7502 |  0.468  |  0.563  |
        | 2023   | -0.1068 | -0.18   | -0.0297 |
        | 2024   |  0.0781 | -0.024  | -0.1731 |
        | 2025   |  0.0928 |  0.0292 | -0.1214 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        omega_excess_return = performance_model.get_omega_excess_return(
            self._get_column(period, "Return", within_period=False),
            self._get_downside_deviation(period, "Return", minimum_acceptable_return),
            self._get_downside_deviation(
                period, "Return", minimum_acceptable_return, benchmark=True
            ),
        )

        return finalize_dataset(
            dataset=omega_excess_return,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_errors
    def get_kelly_criterion(
        self,
        period: str | None = None,
        within_period: bool = True,
        rolling: int | None = None,
        method: Literal["discrete", "continuous"] = "discrete",
        fraction: float = 1.0,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Kelly Criterion, the share of capital to put at risk that maximizes the
        expected logarithmic growth of wealth, from the returns of an investment portfolio or
        asset.

        The discrete form is Kelly's (1956) bet: a probability of winning and a ratio of the
        average win to the average loss, calculated from the returns.
        The continuous form is Thorp's (2006) for returns that are close to normally
        distributed: the average excess return over the risk free rate divided by the
        variance of the excess returns. Since investors rarely bet the full Kelly amount,
        whose wealth swings widely, `fraction` gives a fractional Kelly such as half Kelly.

        The formulas are as follows:

        - Discrete: Kelly = Win Rate - (1 - Win Rate) / Payoff Ratio
        - Continuous: Kelly = Average Excess Return / Variance of the Excess Returns
        - Win Rate = Number of Positive Returns / Number of Non-Zero Returns
        - Payoff Ratio = Average Positive Return / |Average Negative Return|

        A value of 0.25 means putting a quarter of the capital at risk, a value above 1 means
        borrowing to invest and a negative value means the history shows a negative edge.

        For more information, see the following sources:

        - Kelly, J.L. (1956). "A New Interpretation of Information Rate." Bell System
        Technical Journal, 35(4), 917-926.
        - Thorp, E.O. (2006). "The Kelly Criterion in Blackjack, Sports Betting, and the
        Stock Market." Handbook of Asset and Liability Management, Volume 1, 385-428.
        - MacLean, L.C., Thorp, E.O. & Ziemba, W.T. (2011). "The Kelly Capital Growth
        Investment Criterion." World Scientific.

        Shares similar logic:
        - quantstats, kelly_criterion (discrete): https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L2786
        - PerformanceAnalytics (R), KellyRatio (continuous): https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/KellyRatio.R#L39
        - PerformanceAnalytics (R), Return.excess: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/Return.excess.R#L42

        Args:
            period (str, optional): The period to use for the calculation. Options are: "daily", "weekly",
                "monthly", "quarterly", "yearly". Defaults to "quarterly" if the Toolkit is initialised with
                quarterly=True, otherwise "yearly".
            within_period (bool, optional): Whether to calculate the Kelly Criterion from the daily
                returns within each period (e.g. each year if period = 'yearly') or from the returns
                of the periods themselves over the whole history. Defaults to True.
            rolling (int, optional): The rolling window size to use for the calculation. If set, the
                Kelly Criterion is calculated over a rolling window of this many periods across the
                full return history instead of per `period`. At least 2. Defaults to None.
            method (str, optional): The Kelly Criterion. Options are: "discrete" (Kelly's bet from the win
                rate and payoff ratio), "continuous" (average excess return over its variance). Defaults to
                "discrete".
            fraction (float, optional): The share of the full Kelly bet, e.g. 0.5 for half Kelly.
                Defaults to 1.0.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Kelly Criterion values.

        Notes:
        - A zero return counts as neither a win nor a loss. The discrete form is NaN without wins
        or without losses, the continuous form without variance.
        - The continuous form does not depend on the return frequency, since the average and the
        variance of the excess returns scale alike.
        - The Portfolio of a Portfolio's Toolkit is measured on its own daily return series, as a weighted
        average of each asset's Kelly Criterion is not the Kelly Criterion of the portfolio.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "KO"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.performance.get_kelly_criterion()
        ```

        Which returns:

        |      |    AAPL |    MSFT |      KO |
        |:-----|--------:|--------:|--------:|
        | 2021 |  0.1112 |  0.165  |  0.0913 |
        | 2022 | -0.0559 | -0.061  |  0.0559 |
        | 2023 |  0.1637 |  0.154  | -0.0245 |
        | 2024 |  0.1135 |  0.0601 |  0.0606 |
        | 2025 |  0.0434 |  0.0697 |  0.0716 |
        """
        check_option("method", method, ("discrete", "continuous"))

        period = period if period else "quarterly" if self._quarterly else "yearly"
        check_rolling(rolling, self._historical_data, period)

        if rolling:
            returns = self._get_column(period, "Return", within_period=False)

            if method == "continuous":
                returns = performance_model.get_excess_return(
                    returns, self._risk_free_rate_data[period][returns.columns]
                )

            kelly_criterion = performance_model.get_kelly_criterion(
                returns, method, fraction, window=rolling
            )
        else:
            column = "Excess Return" if method == "continuous" else "Return"

            if method == "continuous" and not within_period:
                returns = performance_model.get_excess_return(
                    self._get_column(period, "Return", within_period=False),
                    self._risk_free_rate_data[period][self._tickers_with_returns],
                )
            else:
                returns = self._get_column(period, column, within_period=within_period)

            kelly_criterion = performance_model.get_kelly_criterion(
                returns, method, fraction
            )

        return finalize_dataset(
            dataset=kelly_criterion,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            apply_slice=bool(rolling) or within_period,
        )

    @handle_errors
    def get_return_statistics(
        self,
        period: str | None = None,
        within_period: bool = False,
        confidence_level: float = 0.95,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the statistics traders report about the returns of an investment portfolio or
        asset: how often it wins, how large its wins and losses are, how long its winning and
        losing streaks run, how its gains relate to its drawdown and tails, its best and worst
        returns and outliers, and how precisely its mean return is known. These statistics are
        absolute: a win is a positive return, unlike the Win Rate of `get_win_rate`, which counts
        the days an asset beats its benchmark.

        The formulas are as follows:

        - Win Rate = Number of Positive Returns / Number of Non-Zero Returns
        - Average Win = Average of the Positive Returns
        - Average Loss = Average of the Negative Returns
        - Payoff Ratio = Average Win / |Average Loss|
        - Profit Factor = Sum of the Positive Returns / |Sum of the Negative Returns|
        - Expectancy = Win Rate x Average Win + (1 - Win Rate) x Average Loss
        - Max Consecutive Wins (Losses) = The longest run of positive (negative) returns
        - Recovery Factor = Cumulative Return / |Maximum Drawdown|
        - Common Sense Ratio = Profit Factor x Tail Ratio (95th / 5th percentile)
        - Best (Worst) Return = The highest (lowest) return
        - Exposure = Number of Non-Zero Returns / Number of Returns
        - CPC Index = Profit Factor x Win Rate x Payoff Ratio
        - Profit Ratio = (Average Win / Number of Positive Returns) / (|Average Loss| / Number of Negative Returns)
        - Outlier Win Ratio = 99th Percentile of the Returns / Average Win
        - Outlier Loss Ratio = 1st Percentile of the Returns / Average Loss
        - Risk of Ruin = ((1 - Win Rate) / (1 + Win Rate))^n
        - Risk-Adjusted Return = Annualized Return / Exposure
        - Mean Return = Average of all Returns
        - Standard Error = Standard Deviation of the Returns / SQRT(n)
        - Lower (Upper) Confidence Limit = Mean Return -(+) t(1 - (1 - c) / 2, n - 1) x Standard Error

        Where n is the number of returns, c the confidence level and t the quantile of Student's t distribution.
        The Annualized Return compounds the returns, PRODUCT(1 + Return)^(P / n) - 1 with P the returns in a
        year (252 daily returns, 365 for a ticker that trades on weekends such as a cryptocurrency, or 52, 12,
        4 or 1 for weekly, monthly, quarterly or yearly returns). The confidence limits show how far the true
        mean return can lie from the measured one: when they straddle 0, the history does not tell a positive
        mean return from luck.

        The Recovery Factor compounds the returns into the cumulative return and keeps its sign,
        so a history that lost money does not appear to recover.

        The Expectancy and the Payoff Ratio follow Tharp (2006). The Common Sense Ratio, the CPC Index, the
        Profit Ratio, the Outlier Win and Loss Ratios, the Risk of Ruin, the Exposure and the Risk-Adjusted
        Return are defined in quantstats (Ran Aroussi); the Toolkit applies them to the returns of each
        ticker and period.

        For more information, see the following source:

        - Tharp, V.K. (2006). "Trade Your Way to Financial Freedom." 2nd ed., McGraw-Hill.

        Referenced against:
        - quantstats, avg_return (Expectancy): https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L640
        - quantstats, best: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L362
        - quantstats, worst: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L395
        - quantstats, exposure: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L514
        - quantstats, cpc_index: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L2417
        - quantstats, profit_ratio: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L2312
        - quantstats, outlier_win_ratio: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L2484
        - quantstats, outlier_loss_ratio: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L2523
        - quantstats, risk_of_ruin: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L1968
        - PerformanceAnalytics (R), mean.stderr: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/mean.utils.R#L137
        - PerformanceAnalytics (R), mean.LCL: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/mean.utils.R#L169
        - PerformanceAnalytics (R), mean.UCL: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/mean.utils.R#L208

        Shares similar logic:
        - quantstats, win_rate: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L564
        - quantstats, payoff_ratio: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L2242
        - quantstats, profit_factor: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L2370
        - quantstats, consecutive_wins: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L428
        - quantstats, consecutive_losses: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L471
        - quantstats, recovery_factor: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L2562
        - quantstats, common_sense_ratio: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L2451
        - quantstats, rar: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L1698
        - ffn, PerformanceStats (best_day to worst_year): https://github.com/pmorissette/ffn/blob/v1.4.1/ffn/core.py#L123

        Args:
            period (str, optional): The period of the returns over the whole history, or with
                within_period=True the period the daily returns are grouped by. Options are: "daily",
                "weekly", "monthly", "quarterly", "yearly". Defaults to "daily",
                or with within_period=True to "quarterly" if the Toolkit is initialised with
                quarterly=True, otherwise "yearly".
            within_period (bool, optional): Whether to calculate the statistics from the daily returns
                within each period (e.g. each year if period = 'yearly') instead of from the returns of
                `period` over the whole history. Defaults to False.
            confidence_level (float, optional): The confidence level of the Lower and Upper Confidence
                Limits of the mean return. Defaults to 0.95.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Over the whole history, the statistics as the index and the tickers as the
            columns. Within periods, the periods as the index and (statistic, ticker) as the columns.

        Notes:
        - A zero return is neither a win nor a loss: it is left out of the Win Rate and ends a
        streak. Each ticker is measured over its own returns.
        - The Expectancy agrees with quantstats' avg_return (the mean of the non-zero returns), Best and Worst
        Return its best and worst (and ffn's best and worst day, month or year with the matching period),
        and the CPC Index, Outlier Win and Loss Ratios, Risk of Ruin and the Standard Error and Confidence
        Limits agree with quantstats' cpc_index, outlier_win_ratio, outlier_loss_ratio and risk_of_ruin and
        PerformanceAnalytics' mean.stderr, mean.LCL and mean.UCL. quantstats counts a zero return as a win in
        its Profit Ratio and Outlier Win Ratio, the Toolkit as neither win nor loss, so those two differ when a
        return is exactly 0. quantstats rounds the Exposure up to two decimals, the Toolkit does not, so its
        Risk-Adjusted Return (quantstats' rar, the annualized return over the Exposure) differs by that rounding.
        - The Risk of Ruin raises a ratio below 1 to the number of returns, so over a long daily history it is
        practically 0; it is meant for a series of trades or a short history.
        - A statistic without the returns it needs, such as the Profit Factor or the Payoff Ratio
        without losses, is NaN, as for the Omega Ratio.
        - The Portfolio of a Portfolio's Toolkit is measured on its own daily return series, as a weighted
        average of streak lengths or payoff ratios is not the statistic of the portfolio.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "KO"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.performance.get_return_statistics()
        ```

        Which returns:

        |                        |    AAPL |    MSFT |      KO |
        |:-----------------------|--------:|--------:|--------:|
        | Win Rate               |  0.5316 |  0.5243 |  0.5326 |
        | Average Win            |  0.0125 |  0.012  |  0.0073 |
        | Average Loss           | -0.0126 | -0.0115 | -0.0075 |
        | Payoff Ratio           |  0.9962 |  1.0395 |  0.9771 |
        | Profit Factor          |  1.1306 |  1.1459 |  1.1133 |
        | Expectancy             |  0.0008 |  0.0008 |  0.0004 |
        | Max Consecutive Wins   | 11      |  9      |  9      |
        | Max Consecutive Losses |  8      |  8      |  7      |
        | Recovery Factor        |  3.4713 |  3.5406 |  3.1268 |
        | Common Sense Ratio     |  1.0544 |  1.0959 |  1.1902 |
        | Best Return            |  0.1533 |  0.1013 |  0.0473 |
        | Worst Return           | -0.0925 | -0.0772 | -0.0696 |
        | Exposure               |  0.9976 |  0.9992 |  0.9912 |
        | CPC Index              |  0.5987 |  0.6245 |  0.5793 |
        | Profit Ratio           |  0.8779 |  0.943  |  0.8575 |
        | Outlier Win Ratio      |  3.4485 |  3.4075 |  3.5048 |
        | Outlier Loss Ratio     |  3.3735 |  3.478  |  3.4129 |
        | Risk of Ruin           |  0      |  0      |  0      |
        | Risk-Adjusted Return   |  0.1676 |  0.1839 |  0.0915 |
        | Mean Return            |  0.0008 |  0.0008 |  0.0004 |
        | Standard Error         |  0.0005 |  0.0005 |  0.0003 |
        | Lower Confidence Limit | -0.0002 | -0.0001 | -0.0002 |
        | Upper Confidence Limit |  0.0017 |  0.0017 |  0.0009 |
        """
        if not period:
            # Within periods the daily returns of each year (or quarter) are measured.
            period = (
                ("quarterly" if self._quarterly else "yearly")
                if within_period
                else "daily"
            )

        returns = self._get_column(period, "Return", within_period=within_period)

        # Without returns every statistic is undefined (and the Risk of Ruin would read 1).
        if returns.dropna(how="all").empty:
            logger.warning(
                "There are no %s returns between %s and %s, so the return statistics are empty.",  # noqa: E501
                period,
                self._start_date,
                self._end_date,
            )
            return pd.DataFrame()

        # The returns in a year: the daily returns within each period, or the returns of
        # `period` over the whole history (the daily returns of each ticker's own calendar).
        periods_per_year = self._get_ratio_observations_per_year(
            period, rolling=None if within_period else 1, columns=returns.columns
        )

        return_statistics = performance_model.get_return_statistics_table(
            returns,
            periods_per_year=periods_per_year,
            confidence_level=confidence_level,
        )

        rounding = rounding if rounding is not None else self._rounding

        # A Toolkit without rounding (rounding=None) keeps the full precision.
        return (
            return_statistics.round(rounding)
            if rounding is not None
            else return_statistics
        )

    @handle_portfolio
    @handle_errors
    def get_compound_growth_rate(
        self, rounding: int | None = None, show_columns: str | list[str] | None = None
    ):
        """
        Calculate the Compound Growth Rate (CGR): the constant rate per period at which an investment
        would have grown from its first to its last value, for yearly, quarterly, monthly, weekly or
        daily periods.

        The CGR is a measure that provides the mean growth rate of an investment over a specified period of time.
        It is a useful measure for comparing the performance of investments over different time periods or across
        different asset classes. The CGR is calculated by taking the ratio of the final value to the initial value,
        raising it to the inverse of the number of periods, and then subtracting one.

        The formula is as follows:

            - Years = (Last Date - First Date) / 365.25 days
            - CAGR = (Final Value / Initial Value) ^ (1 / Years) - 1
            - Period Rate = (1 + CAGR) ^ (1 / Periods per Year) - 1

        The CAGR is measured on calendar time: each asset from its own first to its own last
        close between the start and end date, with the years in between counted in calendar
        days. The rate thereby does not depend on how often an asset trades, so a stock (which
        trades on weekdays) and a cryptocurrency (which trades every day) in the same Toolkit
        are measured over the same years, and an asset without a close on the first or last
        date of the range is measured over the dates it has. The quarterly, monthly, weekly and
        daily rates follow from the CAGR with 4 quarters, 12 months, 52 weeks and 252 trading
        days to a year, so the rates are consistent with each other: (1 + CDGR) ^ 252 equals
        1 + CAGR.

        Also known as: CAGR, compound annual growth rate, annualized return.

        Shares similar logic:
        - quantstats, cagr: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L1636
        - ffn, calc_cagr: https://github.com/pmorissette/ffn/blob/v1.4.1/ffn/core.py#L1563

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: A DataFrame containing the CGR for each period. The DataFrame has the periods
            as the index and the CGR values as the column.

        Notes:
        - When verifying the calculation, note that rounding applies and it could be slightly off because of that
        This is mostly noticeable when looking at the Compound Daily Growth Rate. Adjust the rounding with the
        rounding parameter accordingly to get a more precise figure.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["BTCUSD", "ETHUSD", "^GSPC"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.performance.get_compound_growth_rate()
        ```

        Which returns:

        |                                       |   BTCUSD |   ETHUSD |   ^GSPC |   Benchmark |
        |:--------------------------------------|---------:|---------:|--------:|------------:|
        | Compound Annual Growth Rate (CAGR)    |   0.2438 |   0.3236 |  0.1312 |      0.1469 |
        | Compound Quarterly Growth Rate (CQGR) |   0.0561 |   0.0726 |  0.0313 |      0.0348 |
        | Compound Monthly Growth Rate (CMGR)   |   0.0183 |   0.0236 |  0.0103 |      0.0115 |
        | Compound Weekly Growth Rate (CWGR)    |   0.0042 |   0.0054 |  0.0024 |      0.0026 |
        | Compound Daily Growth Rate (CDGR)     |   0.0009 |   0.0011 |  0.0005 |      0.0005 |
        """
        compound_growth_rates = {}

        prices = (
            self._historical_data["daily"]
            .loc[:, "Adj Close"]
            .loc[self._start_date : self._end_date]
        )

        # On calendar time and per asset: the rows of the daily data are the union of every ticker's trading days, so counting rows would stretch the years of a stock by the weekends a cryptocurrency in the same Toolkit trades on, and a ticker without a close on the first or last row would get no rate at all.  # noqa: E501
        compound_annual_growth_rate = performance_model.get_compound_annual_growth_rate(
            prices
        )

        # The shorter periods are the CAGR converted geometrically, so that (1 + CDGR) ^ 252 equals 1 + CAGR.  # noqa: E501
        for period, label in [
            ("yearly", "Compound Annual Growth Rate (CAGR)"),
            ("quarterly", "Compound Quarterly Growth Rate (CQGR)"),
            ("monthly", "Compound Monthly Growth Rate (CMGR)"),
            ("weekly", "Compound Weekly Growth Rate (CWGR)"),
            ("daily", "Compound Daily Growth Rate (CDGR)"),
        ]:
            compound_growth_rates[label] = convert_annualized_rate_to_period(
                compound_annual_growth_rate, period
            )

        compound_growth_rate = pd.DataFrame(compound_growth_rates).T

        compound_growth_rate = apply_rounding(
            compound_growth_rate, rounding if rounding is not None else self._rounding
        )

        return compound_growth_rate

    @handle_portfolio
    @handle_errors
    def get_returns(
        self,
        period: str | None = None,
        cumulative: bool = False,
        logarithmic: bool = False,
        unsmoothed: bool = False,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the returns of each period: how much the value of each asset changed over each week,
        month, quarter or year, compounded from the daily returns.

        The period Return is obtained by compounding the daily returns within each
        period. If cumulative is set to True, the period returns within the selected date
        range are compounded further into a cumulative return (growth of 1) instead.

        The formula is as follows:

        - Period Return = ((1 + Daily Return 1) * (1 + Daily Return 2) * ... * (1 + Daily Return N)) - 1
        - Cumulative Return = (1 + Period Return 1) * (1 + Period Return 2) * ... * (1 + Period Return N)

        The compounding starts at 1 at the beginning of the date range, so the first value
        is 1 plus the return of the first period.

        The first daily return in the range is that of the second trading day: the prices start
        at the start_date, so the move from the previous close into the first day is not part
        of it. For a year-to-date or calendar-year return, set start_date to the last trading
        day of the previous year (e.g. start_date="2025-12-31" for the return of 2026 so far),
        so the first period starts from that close.

        A period without any daily return for a ticker (before it listed, or a gap in its
        history) is NaN rather than 0, and the cumulative return carries the previous value
        past such a period.

        Logarithmic returns are log (continuously compounded) returns, which add up over
        time where simple returns compound:

        - Log Return = ln(1 + Period Return)
        - Cumulative Log Return = ln(Cumulative Return), the sum of the log returns

        With unsmoothing the period returns are unsmoothed with Geltner's filter first, for
        assets whose reported returns are smoothed (appraisal-based real estate, private equity,
        illiquid funds), which shows up as positive autocorrelation and understates the volatility:

        - Unsmoothed Return_t = (Period Return_t - ρ1 x Period Return_t-1) / (1 - ρ1)

        Where ρ1 is the first order autocorrelation of each ticker's period returns over the date
        range; the first period has no predecessor and is NaN. See `get_smoothing_index` to measure
        the smoothing.

        For more information, see the following sources:

        - Geltner, D. (1991). "Smoothing in Appraisal-Based Returns." Journal of Real Estate Finance
        and Economics, 4(3), 327-345.
        - Geltner, D. (1993). "Estimating Market Values from Appraised Values without Assuming an
        Efficient Market." Journal of Real Estate Research, 8(3), 325-345.

        Also known as: periodic return, log return, continuously compounded return, unsmoothed
        return, Geltner return, de-smoothed return.

        Referenced against:
        - PerformanceAnalytics (R), Return.Geltner: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/Return.Geltner.R#L43

        Shares similar logic:
        - ffn, to_log_returns: https://github.com/pmorissette/ffn/blob/v1.4.1/ffn/core.py#L1223
        - pandas-ta-classic, log_return: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/performance/log_return.py#L12
        - empyrical, aggregate_returns: https://github.com/quantopian/empyrical/blob/0.5.5/empyrical/stats.py#L316
        - PerformanceAnalytics (R), Return.cumulative: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/Return.cumulative.R#L27

        Args:
            period (str, optional): The data frequency for returns. Options are: "daily", "weekly", "monthly",
                "quarterly", "yearly". Defaults to "quarterly" if the Toolkit is initialised with quarterly=True,
                otherwise "yearly".
            cumulative (bool, optional): Whether to return the cumulative return over time
            instead of the discrete return per period. Defaults to False.
            logarithmic (bool, optional): Whether to return log returns, ln(1 + return), or with
            cumulative=True the cumulative log return. Defaults to False.
            unsmoothed (bool, optional): Whether to unsmooth the period returns with Geltner's
            first order autocorrelation filter. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the Return values over time. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.Series: Return values with time as the index.

        Notes:
        - The method retrieves the daily historical return data and calculates the Return for
        the specified `period` for each asset in the Toolkit instance.
        - The daily log returns agree with pandas-ta-classic's log_return and ffn's to_log_returns of the adjusted
        close, the unsmoothed returns PerformanceAnalytics' Return.Geltner of the same returns.
        - With growth, the method calculates the growth of Return values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["ASML.AS", "SAP.DE", "MC.PA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.performance.get_returns()
        ```

        Which returns:

        | Date   |   ASML.AS |   SAP.DE |   MC.PA |   Benchmark |
        |:-------|----------:|---------:|--------:|------------:|
        | 2021   |    0.748  |   0.2018 |  0.4345 |      0.3058 |
        | 2022   |   -0.2789 |  -0.2049 | -0.0478 |     -0.1818 |
        | 2023   |    0.3664 |   0.4709 |  0.0958 |      0.2615 |
        | 2024   |    0.0035 |   0.7229 | -0.1179 |      0.2493 |
        | 2025   |    0.3709 |  -0.115  |  0.0393 |      0.1778 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        check_period(period, ["daily", "weekly", "monthly", "quarterly", "yearly"])

        returns = self._historical_data["daily"]["Return"]

        # Compounded only after slicing to the date range, so the cumulative return
        # starts at 1 before the first period and that period keeps its own return.
        period_returns = (
            returns
            if period == "daily"
            else performance_model.get_returns(returns, period)
        ).loc[self._start_date : self._end_date]

        if unsmoothed:
            period_returns = performance_model.get_geltner_returns(period_returns)

        if cumulative:
            period_returns = (1 + period_returns).cumprod()

        if logarithmic:
            period_returns = (
                np.log(period_returns)
                if cumulative
                else performance_model.get_log_returns(period_returns)
            )

        return finalize_dataset(
            dataset=period_returns,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_portfolio
    @handle_errors
    def get_excess_return(
        self,
        period: str | None = None,
        cumulative: bool = False,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the excess return of each period: how much each asset returned above the risk free
        rate over each week, month, quarter or year, compounded from the daily returns.

        The Excess Return is defined as the period Return minus the risk free rate earned over
        that period. If cumulative is set to True, the excess returns within the selected date
        range are compounded further into a cumulative excess return (growth of 1) instead.

        The formula is as follows:

        - Period Return = ((1 + Daily Return 1) * (1 + Daily Return 2) * ... * (1 + Daily Return N)) - 1
        - Daily Risk Free Rate = (1 + Annual Risk Free Rate)^(1 / 252) - 1, or (1 + Annual Risk Free Rate)^(Calendar
          Days / 365) - 1 when a ticker trades on weekends, with the Treasury yield of each day
        - Period Risk Free Rate = ((1 + Daily Risk Free Rate 1) * ... * (1 + Daily Risk Free Rate N)) - 1, over the same
          N days as the Period Return, so a period the data covers in part (such as the current year) earns the
          risk free rate of the days it covers
        - Excess Return = Period Return - Period Risk Free Rate
        - Cumulative Excess Return = (1 + Excess Return 1) * (1 + Excess Return 2) * ... * (1 + Excess Return N)

        The compounding starts at 1 at the beginning of the date range, so the first value
        is 1 plus the excess return of the first period.

        A period without any daily return for a ticker (before it listed, or a gap in its
        history) is NaN rather than 0, and the cumulative excess return carries the previous
        value past such a period.

        Also known as: return minus the risk-free rate.

        Shares similar logic:
        - empyrical, aggregate_returns: https://github.com/quantopian/empyrical/blob/0.5.5/empyrical/stats.py#L316
        - PerformanceAnalytics (R), Return.cumulative: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/Return.cumulative.R#L27
        - PerformanceAnalytics (R), Return.excess: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/Return.excess.R#L42

        Args:
            period (str, optional): The data frequency for returns. Options are: "weekly", "monthly", "quarterly",
                "yearly". Defaults to "quarterly" if the Toolkit is initialised with quarterly=True, otherwise
                "yearly".
            cumulative (bool, optional): Whether to return the cumulative excess return over time
            instead of the discrete excess return per period. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the Excess Return values over time.
            Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.Series: Excess Return values with time as the index.

        Notes:
        - The method retrieves the daily historical return data and calculates the Excess Return for
        the specified `period` for each asset in the Toolkit instance.
        - The risk-free rate is the Toolkit's risk free rate, the 10-year US Treasury yield by default. Many
        textbook definitions (Sharpe, 1994; the Ken French factors) use a short-term Treasury bill instead, which
        the Toolkit uses when initialised with the 13-week Treasury bill as its risk free rate.
        - With growth, the method calculates the growth of Excess Return values using the
        specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["MCD", "SBUX", "CMG"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.performance.get_excess_return()
        ```

        Which returns:

        | Date   |     MCD |    SBUX |     CMG |   Benchmark |
        |:-------|--------:|--------:|--------:|------------:|
        | 2021   |  0.2901 |  0.139  |  0.3114 |      0.2908 |
        | 2022   | -0.0244 | -0.1612 | -0.2359 |     -0.2112 |
        | 2023   |  0.1111 | -0.0514 |  0.6088 |      0.2222 |
        | 2024   | -0.0411 | -0.0672 |  0.2759 |      0.2064 |
        | 2025   |  0.036  | -0.0956 | -0.4293 |      0.1343 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        check_period(period, ["weekly", "monthly", "quarterly", "yearly"])

        returns = self._historical_data["daily"]["Return"]
        period_returns = performance_model.get_returns(returns, period)
        risk_free_rate = self._risk_free_rate_data[period][period_returns.columns]

        # Compounded only after slicing to the date range, see get_returns.
        excess_return = performance_model.get_excess_return(
            period_returns, risk_free_rate
        ).loc[self._start_date : self._end_date]

        if cumulative:
            excess_return = (1 + excess_return).cumprod()

        return finalize_dataset(
            dataset=excess_return,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_errors
    def get_correlation_matrix(
        self,
        period: str | None = None,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the full pairwise Correlation Matrix across all assets (and the
        benchmark) in the Toolkit instance, based on the returns at the frequency given
        by `period`.

        Unlike `get_beta`, which relates a single asset to the benchmark, this computes
        the correlation between every pair of assets at once. This is a prerequisite for
        portfolio variance calculations and any mean-variance optimization work.

        The `period` sets the return frequency, and with it the number of observations: yearly returns over
        five years give only a handful of points per pair, which makes the correlations close to meaningless,
        so monthly or weekly returns over the same span are the usual choice for diversification and portfolio
        work. Correlations between risky assets tend to rise in sell-offs, so a matrix estimated over a calm
        stretch understates how much assets fall together when it matters.

        The formula is as follows:

        - Correlation (A, B) = Covariance (Return A, Return B) / (Standard Deviation A * Standard Deviation B),
          the Pearson correlation of the period returns (e.g. monthly) within the Toolkit's date range, using
          the overlapping (pairwise complete) observations of each pair

        Args:
            period (str, optional): The data frequency for returns. Options are: "daily", "weekly", "monthly",
                "quarterly", "yearly". Defaults to "quarterly" if the Toolkit is initialised with quarterly=True,
                otherwise "yearly"; fewer than three returns in the date range are logged as a warning.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The N x N Correlation Matrix, with assets as both the index and
            the columns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["TLT", "GLD", "EFA", "VNQ"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.performance.get_correlation_matrix(period="monthly")
        ```

        Which returns:

        |           |    TLT |    GLD |    EFA |    VNQ |   Benchmark |
        |:----------|-------:|-------:|-------:|-------:|------------:|
        | TLT       | 1      | 0.3747 | 0.6185 | 0.6626 |      0.5646 |
        | GLD       | 0.3747 | 1      | 0.3918 | 0.2266 |      0.1283 |
        | EFA       | 0.6185 | 0.3918 | 1      | 0.781  |      0.7999 |
        | VNQ       | 0.6626 | 0.2266 | 0.781  | 1      |      0.8282 |
        | Benchmark | 0.5646 | 0.1283 | 0.7999 | 0.8282 |      1      |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        returns = self._get_matrix_returns(period)

        correlation_matrix = performance_model.get_correlation_matrix(returns)

        return apply_rounding(
            correlation_matrix, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_covariance_matrix(
        self,
        period: str | None = None,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the full pairwise Covariance Matrix across all assets (and the
        benchmark) in the Toolkit instance, based on the returns at the frequency given
        by `period`.

        Unlike `get_covariance`, which relates a single asset to the benchmark, this
        computes the covariance between every pair of assets at once. This is a
        prerequisite for portfolio variance calculations and any mean-variance
        optimization work.

        The formula is as follows:

        - Covariance (A, B) = SUM((Return A - Mean Return A) * (Return B - Mean Return B)) / (N - 1), the sample
          covariance (with n - 1 in the denominator) of the period returns (e.g. monthly) within the Toolkit's date
          range, per period and not annualized, using the overlapping (pairwise complete) observations of each pair

        Args:
            period (str, optional): The data frequency for returns. Options are: "daily", "weekly", "monthly",
                "quarterly", "yearly". Defaults to "quarterly" if the Toolkit is initialised with quarterly=True,
                otherwise "yearly"; fewer than three returns in the date range are logged as a warning.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The N x N Covariance Matrix, with assets as both the index and
            the columns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["^GSPC", "EURUSD", "GCUSD", "CLUSD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.performance.get_covariance_matrix(period="monthly")
        ```

        Which returns:

        |           |   ^GSPC |   EURUSD |   GCUSD |   CLUSD |   Benchmark |
        |:----------|--------:|---------:|--------:|--------:|------------:|
        | ^GSPC     |  0.0019 |   0.0003 |  0.0003 |  0.0003 |      0.0019 |
        | EURUSD    |  0.0003 |   0.0005 |  0.0005 | -0.0001 |      0.0003 |
        | GCUSD     |  0.0003 |   0.0005 |  0.0018 | -0.0007 |      0.0003 |
        | CLUSD     |  0.0003 |  -0.0001 | -0.0007 |  0.0067 |      0.0003 |
        | Benchmark |  0.0019 |   0.0003 |  0.0003 |  0.0003 |      0.0019 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        returns = self._get_matrix_returns(period)

        covariance_matrix = performance_model.get_covariance_matrix(returns)

        return apply_rounding(
            covariance_matrix, rounding if rounding is not None else self._rounding
        )

    @handle_portfolio
    @handle_errors
    def get_appraisal_ratio(
        self,
        period: str | None = None,
        rolling: int | None = None,
        method: Literal["appraisal", "modified", "alternative"] = "appraisal",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Appraisal Ratio, i.e. Jensen's Alpha divided by the idiosyncratic
        (residual, unsystematic) standard deviation left over from the CAPM regression
        that produced that Alpha.

        Jensen's Alpha (see `get_alpha` with the Jensen method) measures how much return a manager
        generated above what CAPM would predict given the asset's Beta. However, a large
        Alpha achieved with wildly noisy, unpredictable residual returns is far less
        attractive than the same Alpha achieved consistently. The Appraisal Ratio
        normalizes Alpha by that noise (the "specific risk" not explained by market
        exposure), giving a Sharpe-ratio-like measure of stock-picking or timing skill per
        unit of idiosyncratic risk taken.

        The formula is as follows:

        - Appraisal Ratio = Jensen's Alpha / Residual Standard Deviation

        Where the residual standard deviation is the standard deviation of the pointwise
        CAPM regression residuals (Asset Excess Return − Beta * Benchmark Excess Return),
        reusing the exact same CAPM regression formula as `get_alpha` with the Jensen method.

        Bacon (2008) scales Jensen's Alpha by two other risk measures as well, which `method`
        selects:

        - Modified Jensen's Alpha = Jensen's Alpha / Beta
        - Alternative Jensen's Alpha = Jensen's Alpha / (Beta × Benchmark Volatility)

        Where Beta × Benchmark Volatility is the systematic risk, with the benchmark volatility the
        standard deviation of its daily returns within the period scaled to the period, or of its
        returns over the rolling window.

        See definition: https://en.wikipedia.org/wiki/Information_ratio

        For more information, see the following sources:

        - Treynor, J.L., & Black, F. (1973). "How to Use Security Analysis to Improve Portfolio
        Selection." Journal of Business, 46(1), 66-86.
        - Bacon, C.R. (2008). "Practical Portfolio Performance Measurement and Attribution."
        2nd edition, John Wiley & Sons, p. 77.

        Also known as: Treynor-Black Appraisal Ratio.

        Referenced against:
        - PerformanceAnalytics (R), AppraisalRatio (modified, alternative): https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/AppraisalRatio.R#L45

        Shares similar logic:
        - PerformanceAnalytics (R), CAPM.jensenAlpha: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/CAPM.jensenAlpha.R#L34
        - empyrical, alpha: https://github.com/quantopian/empyrical/blob/0.5.5/empyrical/stats.py#L1154
        - empyrical, roll_alpha_beta: https://github.com/quantopian/empyrical/blob/0.5.5/empyrical/stats.py#L1058
        - PerformanceAnalytics (R), Return.excess: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/Return.excess.R#L42
        - PerformanceAnalytics (R), CAPM.beta: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/CAPM.beta.R#L89
        - empyrical, beta: https://github.com/quantopian/empyrical/blob/0.5.5/empyrical/stats.py#L1314
        - PerformanceAnalytics (R), AppraisalRatio: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/AppraisalRatio.R#L45

        Args:
            period (str, optional): The period to use for the calculation. Options are: "daily", "weekly",
                "monthly", "quarterly", "yearly". Defaults to "quarterly" if the Toolkit is initialised with
                quarterly=True, otherwise "yearly".
            rolling (int, optional): The rolling window size to use for the calculation. If set, Beta and
            the residual standard deviation are estimated over a rolling window of this many periods
            across the full return history instead of per `period`, the residuals of each window from
            the regression over that window (see `performance_model.get_appraisal_ratio`).
            At least 3. Defaults to None.
            method (str, optional): The ratio, as described above. Options are: "appraisal" (Jensen's Alpha
                per unit of residual risk), "modified" (Jensen's Alpha per unit of Beta), "alternative"
                (Jensen's Alpha per unit of systematic risk). Defaults to "appraisal".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Appraisal Ratio values.

        Notes:
        - Daily Appraisal Ratio is not an option as the standard deviation for 1 day is close to
        zero. Therefore, it does not give any useful insights.
        - The method retrieves historical data and calculates Jensen's Alpha and the CAPM
        regression residuals for each asset in the Toolkit instance, reusing the same Beta and
        CAPM formula as `get_alpha` with the Jensen method.
        - PerformanceAnalytics' AppraisalRatio annualizes the returns over the whole sample and takes the
        population standard deviation of the residuals; the Toolkit uses the return, Beta and residuals of
        each period. The three methods follow the same formulas.
        - With growth, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["ADBE", "CRM", "ORCL"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2020-01-01",
            end_date="2025-12-31",
        )

        toolkit.performance.get_appraisal_ratio()
        ```

        Which returns:

        | Date   |    ADBE |     CRM |    ORCL |
        |:-------|--------:|--------:|--------:|
        | 2021   | -1.1713 | -0.8079 |  0.5274 |
        | 2022   | -0.4645 | -0.6352 |  0.5195 |
        | 2023   |  1.576  |  2.7166 |  0.1372 |
        | 2024   | -1.6888 | -0.1942 |  0.9947 |
        | 2025   | -1.415  | -1.4877 | -0.1032 |
        """
        check_option("method", method, ("appraisal", "modified", "alternative"))

        period = period if period else "quarterly" if self._quarterly else "yearly"
        check_rolling(rolling, self._historical_data, period, 3)

        if rolling:
            returns = self._get_column(period, "Return", within_period=False)
            benchmark_returns = self._get_column(
                period, "Return", within_period=False, benchmark=True
            )

            beta = performance_model.get_beta(
                returns, benchmark_returns, window=rolling
            )

            risk_free_rate = self._risk_free_rate_data[period]
            excess_return = performance_model.get_excess_return(
                returns, risk_free_rate[returns.columns]
            )
            benchmark_excess_return = performance_model.get_excess_return(
                benchmark_returns, risk_free_rate[self._benchmark_name]
            )
        else:
            returns = self._get_column(period, "Return", within_period=True)
            benchmark_returns = self._get_column(
                period, "Return", within_period=True, benchmark=True
            )

            beta = performance_model.get_beta(returns, benchmark_returns)

            within_excess_return = self._get_column(
                period, "Excess Return", within_period=True
            )
            within_benchmark_excess_return = self._get_column(
                period, "Excess Return", within_period=True, benchmark=True
            )

            capm_residuals = performance_model.get_capm_residuals(
                within_excess_return, beta, within_benchmark_excess_return
            )

        period_returns = self._get_column(period, "Return", within_period=False)

        risk_free_rate = self._risk_free_rate_data[period]
        benchmark_period_returns = self._get_column(
            period, "Return", within_period=False, benchmark=True
        )

        jensens_alpha = performance_model.get_jensens_alpha(
            period_returns, risk_free_rate, beta, benchmark_period_returns
        )

        if method in ("modified", "alternative"):
            return finalize_dataset(
                dataset=performance_model.get_modified_jensens_alpha(
                    jensens_alpha,
                    beta,
                    (
                        self._get_benchmark_volatility(period, rolling)
                        if method == "alternative"
                        else None
                    ),
                ),
                start_date=self._start_date,
                end_date=self._end_date,
                default_rounding=self._rounding,
                growth=growth,
                lag=lag,
                rounding=rounding,
                standardize=standardize,
                axis="rows",
                row_slice=True,
                dropna=True,
            )

        # A rolling Alpha is divided by the residual standard deviation of the regression
        # over the same window, so each ratio only uses the returns of its own window.
        appraisal_ratio = (
            performance_model.get_appraisal_ratio(
                jensens_alpha,
                excess_returns=excess_return,
                benchmark_excess_returns=benchmark_excess_return,
                window=rolling,
            )
            if rolling
            else performance_model.get_appraisal_ratio(jensens_alpha, capm_residuals)
        )

        return finalize_dataset(
            dataset=appraisal_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_portfolio
    @handle_errors
    def get_fama_decomposition(
        self,
        period: str | None = None,
        rolling: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Fama (1972) decomposition: Jensen's Alpha split into Selectivity, the return above
        what a fully diversified portfolio with the same total risk would have earned, and
        Diversification, the return the risk left undiversified should have earned.

        Jensen's Alpha alone conflates two very different sources of excess return:
        genuine stock/timing selection skill, and simply carrying more total risk than the
        market by holding an under-diversified portfolio (which, in a CAPM world, should
        be compensated with extra return even absent any skill). Fama's decomposition
        separates the two by comparing the portfolio's actual return against two different
        CAPM-implied return benchmarks: one using the portfolio's actual Beta (systematic
        risk only), and one using the portfolio's actual *total* risk ratio
        (Sigma_Portfolio / Sigma_Market) in place of Beta.

        The formulas are as follows:

        - Selectivity = (Asset Return − Risk-Free Rate) − (Sigma_Portfolio / Sigma_Market) *
          (Benchmark Return − Risk-Free Rate)
        - Diversification = [Risk-Free Rate + (Sigma_Portfolio / Sigma_Market) * (Benchmark Return − Risk-Free Rate)] −
          [Risk-Free Rate + Beta * (Benchmark Return − Risk-Free Rate)]

        Selectivity is the return earned above what would be required for a fully
        diversified portfolio carrying the same total risk, i.e. genuine security
        selection or timing skill. Diversification is the extra return the manager left on
        the table (if positive, it is a cost) by taking on unsystematic risk that a fully
        diversified portfolio of the same total risk would not have. Selectivity plus
        Diversification equals Jensen's Alpha (see `get_alpha` with the Jensen method).

        Also known as: Fama's Net Selectivity, Fama performance decomposition.

        Shares similar logic:
        - PerformanceAnalytics (R), NetSelectivity: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/NetSelectivity.R#L35
        - PerformanceAnalytics (R), Selectivity: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/Selectivity.R#L32
        - empyrical, roll_alpha_beta: https://github.com/quantopian/empyrical/blob/0.5.5/empyrical/stats.py#L1058
        - PerformanceAnalytics (R), CAPM.beta: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/CAPM.beta.R#L89
        - empyrical, beta: https://github.com/quantopian/empyrical/blob/0.5.5/empyrical/stats.py#L1314

        Args:
            period (str, optional): The period to use for the calculation. Options are: "daily", "weekly",
                "monthly", "quarterly", "yearly". Defaults to "quarterly" if the Toolkit is initialised with
                quarterly=True, otherwise "yearly".
            rolling (int, optional): The rolling window size to use for the calculation. If set, Beta and
            the standard deviations of the asset and the benchmark are estimated from the returns per
            `period` over a rolling window of this many periods across the full return history,
            instead of from the daily returns within each `period`. At least 2. Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Selectivity and Diversification values, with a Multi Index of
            (ticker, component) as the columns.

        Notes:
        - Daily Fama Decomposition is not an option as the standard deviation for 1 day is close
        to zero. Therefore, it does not give any useful insights.
        - The method retrieves historical data and calculates Beta, the asset's and benchmark's
        standard deviation, and the Selectivity and Diversification components for each asset in
        the Toolkit instance.
        - With growth, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["QQQ", "IWM", "XLE"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2020-01-01",
            end_date="2025-12-31",
        )

        toolkit.performance.get_fama_decomposition().xs("Selectivity", level=1, axis=1)
        ```

        Which returns:

        | Date   |     QQQ |     IWM |     XLE |
        |:-------|--------:|--------:|--------:|
        | 2021   | -0.123  | -0.3306 | -0.1031 |
        | 2022   | -0.0751 |  0.0105 |  0.9386 |
        | 2023   |  0.2054 | -0.2123 | -0.4352 |
        | 2024   | -0.0824 | -0.2696 | -0.2803 |
        | 2025   |  0.0022 | -0.0738 | -0.1335 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"
        check_rolling(rolling, self._historical_data, period)

        if rolling:
            returns = self._get_column(period, "Return", within_period=False)
            benchmark_returns = self._get_column(
                period, "Return", within_period=False, benchmark=True
            )

            beta = performance_model.get_beta(
                returns, benchmark_returns, window=rolling
            )
        else:
            returns = self._get_column(period, "Return", within_period=True)
            benchmark_returns = self._get_column(
                period, "Return", within_period=True, benchmark=True
            )

            beta = performance_model.get_beta(returns, benchmark_returns)

        period_returns = self._get_column(period, "Return", within_period=False)
        risk_free_rate = self._risk_free_rate_data[period]
        benchmark_period_returns = self._get_column(
            period, "Return", within_period=False, benchmark=True
        )

        if rolling:
            # The standard deviations of the same period returns and rolling window as the
            # rolling Beta, so that Total Risk Beta = Asset Standard Deviation / Benchmark
            # Standard Deviation is at least |Beta| (the correlation is at most 1).
            # Each on its own dates, and the asset on the dates it shares with the benchmark (as Beta), so a stock next to a cryptocurrency keeps full windows on daily data.  # noqa: E501
            asset_standard_deviation = performance_model.get_standard_deviation(
                returns, benchmark_returns, window=rolling
            )
            benchmark_standard_deviation = performance_model.get_standard_deviation(
                benchmark_returns, window=rolling
            )
        else:
            daily_returns = self._get_column("daily", "Return", within_period=False)
            daily_benchmark_returns = self._get_column(
                "daily", "Return", within_period=False, benchmark=True
            )

            asset_standard_deviation = get_volatility(daily_returns, period)
            benchmark_standard_deviation = get_volatility(
                daily_benchmark_returns, period
            )

        selectivity, diversification = performance_model.get_fama_decomposition(
            period_returns,
            risk_free_rate,
            beta,
            benchmark_period_returns,
            asset_standard_deviation,
            benchmark_standard_deviation,
        )

        fama_decomposition = pd.concat(
            {"Selectivity": selectivity, "Diversification": diversification}, axis=1
        ).swaplevel(0, 1, axis=1)

        fama_decomposition = fama_decomposition.reindex(
            self._tickers_with_returns, level=0, axis=1
        )

        return finalize_dataset(
            dataset=fama_decomposition,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_portfolio
    @handle_errors
    def get_starr_ratio(
        self,
        period: str | None = None,
        within_period: bool = True,
        alpha: float = 0.05,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the STARR (Stable Tail Adjusted Return Ratio): the excess return an asset earns per
        unit of Conditional Value at Risk, the average loss in the tail.

        The Sharpe ratio penalizes upside and downside volatility equally via the
        standard deviation. The STARR ratio instead scales the mean excess return by the
        Conditional Value at Risk (CVaR / Expected Shortfall), a coherent tail-risk
        measure that only looks at the average magnitude of losses beyond the `alpha`
        quantile. This makes STARR more appropriate than the Sharpe ratio for return
        distributions with fat left tails.

        The formula is as follows:

        - STARR Ratio = Excess Return / |CVaR(alpha)|

        See definition: https://en.wikipedia.org/wiki/Expected_shortfall

        Also known as: Stable Tail Adjusted Return Ratio, Conditional Sharpe Ratio.

        Shares similar logic:
        - PerformanceAnalytics (R), Return.excess: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/Return.excess.R#L42

        Args:
            period (str, optional): The period to use for the calculation. Options are: "daily", "weekly",
                "monthly", "quarterly", "yearly". Defaults to "quarterly" if the Toolkit is initialised with
                quarterly=True, otherwise "yearly".
            within_period (bool, optional): Whether to calculate the CVaR within the specified
            period or for the entire period. Thus whether to look at the CVaR within a specific
            year (if period = 'yearly') or look at the entirety of all years. Defaults to True.
            alpha (float, optional): The confidence level used for the CVaR calculation (e.g. 0.05
            for the worst 5% of outcomes). Defaults to 0.05.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: STARR Ratio values.

        Notes:
        - The method retrieves historical data and calculates the STARR Ratio for each asset in
        the Toolkit instance.
        - Periods with very few return observations (e.g. a partial period at the very start of
        the selected date range) can produce a degenerate (e.g. zero or ±infinite) CVaR, since
        CVaR is not a meaningful statistic with only one or two data points. This mirrors the
        analogous caveat for the Sharpe Ratio needing enough observations for its standard
        deviation to be meaningful.
        - With growth, the method calculates the growth of the ratio values using
        the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["CAT", "DE", "HON"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2020-01-01",
            end_date="2025-12-31",
        )

        toolkit.performance.get_starr_ratio()
        ```

        Which returns:

        | Date   |    CAT |      DE |     HON |
        |:-------|-------:|--------:|--------:|
        | 2021   | 0.287  |  0.4695 | -0.0375 |
        | 2022   | 0.2127 |  0.2898 |  0.0391 |
        | 2023   | 0.3702 | -0.1498 | -0.0629 |
        | 2024   | 0.3589 |  0.0623 |  0.1468 |
        | 2025   | 0.8327 |  0.1244 | -0.1883 |
        """
        validate_alpha(alpha)
        period = period if period else "quarterly" if self._quarterly else "yearly"

        returns = self._get_column(period, "Return", within_period=within_period)
        period_returns = self._get_column(period, "Return", within_period=False)
        risk_free_rate = self._risk_free_rate_data[period][period_returns.columns]

        excess_return = performance_model.get_excess_return(
            period_returns, risk_free_rate
        )

        starr_ratio = performance_model.get_starr_ratio(excess_return, returns, alpha)

        return finalize_dataset(
            dataset=starr_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_portfolio
    @handle_errors
    def get_rachev_ratio(
        self,
        period: str | None = None,
        within_period: bool = True,
        alpha: float = 0.05,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Rachev Ratio (R-Ratio): the average of the best returns in the right tail divided
        by the average loss in the left tail, how large the extreme gains are compared to the extreme
        losses.

        The Rachev ratio compares the "quality" of the best outcomes to the "quality" of
        the worst outcomes by taking the ratio of the right-tail Expected Shortfall (the
        average of the best `alpha` fraction of returns) to the left-tail Expected
        Shortfall (the average magnitude of the worst `alpha` fraction of returns). A
        ratio above 1 indicates that the average size of extreme gains outweighs the
        average size of extreme losses.

        The formula is as follows:

        - Rachev Ratio = ES_right(alpha) / ES_left(alpha)

        Also known as: R-Ratio.

        Shares similar logic:
        - PerformanceAnalytics (R), RachevRatio: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/RachevRatio.R#L30

        Args:
            period (str, optional): The period to use for the calculation. Options are: "daily", "weekly",
                "monthly", "quarterly", "yearly". Defaults to "quarterly" if the Toolkit is initialised with
                quarterly=True, otherwise "yearly".
            within_period (bool, optional): Whether to calculate the Rachev Ratio within the
            specified period or for the entire period. Thus whether to look at the return
            distribution within a specific year (if period = 'yearly') or look at the entirety of
            all years. Defaults to True.
            alpha (float, optional): The confidence level used for both tails (e.g. 0.05 for the
            best/worst 5% of outcomes). Defaults to 0.05.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Rachev Ratio values.

        Notes:
        - The method retrieves historical data and calculates the Rachev Ratio for each asset in
        the Toolkit instance.
        - With growth, the method calculates the growth of the ratio values using
        the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["TSLA", "AMD", "PLTR"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.performance.get_rachev_ratio()
        ```

        Which returns:

        |      |   TSLA |    AMD |   PLTR |
        |:-----|-------:|-------:|-------:|
        | 2021 | 1.2069 | 1.1556 | 1.2789 |
        | 2022 | 0.9469 | 0.9396 | 0.9301 |
        | 2023 | 1.1169 | 1.3483 | 1.6236 |
        | 2024 | 1.3081 | 0.8908 | 2.0305 |
        | 2025 | 1.0925 | 1.4024 | 1.0683 |
        """
        validate_alpha(alpha)
        period = period if period else "quarterly" if self._quarterly else "yearly"

        returns = self._get_column(period, "Return", within_period=within_period)

        rachev_ratio = performance_model.get_rachev_ratio(returns, alpha)

        return finalize_dataset(
            dataset=rachev_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            apply_slice=within_period,
            dropna=True,
        )

    @handle_portfolio
    @handle_errors
    def get_market_timing_model(
        self,
        period: str | None = None,
        method: Literal["treynor_mazuy", "henriksson_merton"] = "treynor_mazuy",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate a market timing model for each asset: whether the manager raised the exposure to the
        market before it rose and lowered it before it fell, with the Treynor-Mazuy or the
        Henriksson-Merton model.

        Jensen's Alpha and Beta from a plain CAPM regression cannot distinguish
        stock-picking skill (selectivity) from market-timing skill (shifting exposure
        ahead of market moves). Both models add a timing term in the benchmark excess
        return to the regression.

        The Treynor-Mazuy model adds a quadratic term: a manager
        who successfully increases (decreases) market exposure ahead of up (down) markets
        will show a return profile that curves upward as a function of the benchmark
        return, captured by a positive quadratic coefficient (Gamma). The formula is as follows:

        - Excess Return = Alpha + Beta * Benchmark Excess Return + Gamma * Benchmark Excess Return^2 + Residuals

        Gamma > 0 indicates positive market-timing ability; Gamma <= 0 indicates no
        timing ability.

        The Henriksson-Merton model models timing as a piecewise
        (rather than quadratic) change in Beta: a "down-market" Beta and an "up-market" Beta.
        The formula is as follows:

        - Excess Return = Alpha + Beta * Benchmark Excess Return +
          Up Market Beta * max(Benchmark Excess Return, 0) + Residuals

        Beta is the "down-market" Beta (the portfolio's market exposure when the
        benchmark excess return is negative), and Beta + Up Market Beta is the
        "up-market" Beta. Up Market Beta > 0 indicates positive market-timing ability;
        Up Market Beta <= 0 indicates no timing ability.

        For more information, see the following sources:

        - Treynor, J., & Mazuy, K. (1966). "Can Mutual Funds Outguess the Market?" Harvard
        Business Review, 44(4), 131-136.
        - Henriksson, R.D., & Merton, R.C. (1981). "On Market Timing and Investment
        Performance II: Statistical Procedures for Evaluating Forecasting Skills." Journal of
        Business, 54(4), 513-533.

        Also known as: Treynor-Mazuy quadratic timing model, TM model, Henriksson-Merton
        piecewise timing model, HM model, market timing regression.

        Shares similar logic:
        - PerformanceAnalytics (R), MarketTiming (TM): https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/MarketTiming.R#L52
        - PerformanceAnalytics (R), MarketTiming (HM): https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/MarketTiming.R#L52

        Args:
            period (str, optional): The period to use for the calculation. Options are: "daily", "weekly",
                "monthly", "quarterly", "yearly". Defaults to "quarterly" if the Toolkit is initialised with
                quarterly=True, otherwise "yearly".
            method (str, optional): The timing model. Options are: "treynor_mazuy" (the quadratic timing term,
                Gamma), "henriksson_merton" (the piecewise timing term, Up Market Beta). Defaults to
                "treynor_mazuy".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Alpha, Beta, Gamma (Treynor-Mazuy) or Up Market Beta
            (Henriksson-Merton) and R Squared values, with a Multi Index of
            (ticker, parameter) as the columns.

        Notes:
        - Daily and weekly results are not an option as there would be too few
        observations within each period to run a meaningful regression.
        - The method retrieves historical data and performs the regression for each asset
        in the Toolkit instance, within each period.
        - With growth, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["ARKK", "JEPI"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.performance.get_market_timing_model().xs("ARKK", level=0, axis=1)
        ```

        Which returns:

        |      |   Alpha |   Beta |    Gamma |   R Squared |
        |:-----|--------:|-------:|---------:|------------:|
        | 2021 | -0.0034 | 1.6646 |  12.4821 |      0.2937 |
        | 2022 | -0.0035 | 2.1567 |   6.9512 |      0.5959 |
        | 2023 |  0.0015 | 2.2614 | -14.8854 |      0.529  |
        | 2024 | -0.001  | 2.1368 |  -3.1566 |      0.5658 |
        | 2025 |  0.0006 | 1.8994 |  -1.3437 |      0.6767 |

        With the Henriksson-Merton model:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["BRK-B", "FBCG"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.performance.get_market_timing_model(method="henriksson_merton").xs("FBCG", level=0, axis=1)
        ```

        Which returns:

        |      |   Alpha |   Beta |   Up Market Beta |   R Squared |
        |:-----|--------:|-------:|-----------------:|------------:|
        | 2021 | -0.0004 | 1.4217 |          -0.064  |      0.7204 |
        | 2022 | -0.0009 | 1.4208 |           0.0312 |      0.885  |
        | 2023 |  0.0004 | 1.359  |           0.0602 |      0.8342 |
        | 2024 |  0      | 1.4674 |           0.0496 |      0.8536 |
        | 2025 |  0.0002 | 1.3907 |          -0.0759 |      0.8893 |
        """
        check_option("method", method, ("treynor_mazuy", "henriksson_merton"))

        timing_model = (
            performance_model.get_treynor_mazuy_model
            if method == "treynor_mazuy"
            else performance_model.get_henriksson_merton_model
        )

        period = period if period else "quarterly" if self._quarterly else "yearly"
        excess_return = self._get_column(period, "Excess Return", within_period=True)
        benchmark_excess_return = self._get_column(
            period, "Excess Return", within_period=True, benchmark=True
        )

        logger.info("Calculating the %s Market Timing Model", method)

        regression_results: dict[str, pd.DataFrame] = {}

        for ticker in self._tickers_with_returns:
            ticker_results: dict = {}

            for sub_period in excess_return.index.get_level_values(0).unique():
                asset_excess = excess_return.loc[sub_period, ticker]
                benchmark_excess = benchmark_excess_return.loc[sub_period]

                result, _ = timing_model(asset_excess, benchmark_excess)
                ticker_results[sub_period] = result

            regression_results[ticker] = pd.DataFrame.from_dict(
                ticker_results, orient="index"
            )

        market_timing_model = pd.concat(regression_results, axis=1)

        return finalize_dataset(
            dataset=market_timing_model,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_errors
    def get_factor_analysis(
        self,
        metrics: str | list[str] | pd.DataFrame = "all",
        period: str = "monthly",
        horizons: list[int] | int | None = None,
        quantiles: int = 5,
        periods_apart: int = 1,
        detail: str | None = None,
        column: str | None = None,
        group: str | dict | None = None,
        demeaned: bool = False,
        binning: str = "quantile",
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Calculate the factor analysis (predictive power) of the performance metrics, or of a factor of your own:
        whether the tickers that score high on a metric go on to return more than those that score low. Per period, the
        information coefficient (IC) is the Spearman rank correlation between the metric of the tickers in the period
        and their return over the following horizon: 1 means the metric ranks the later returns perfectly, -1 in
        reverse and 0 that it says nothing about them. The IC measures skill in Grinold and Kahn's fundamental law of
        active management, IR ≈ IC × √breadth: a small but consistent IC across many tickers and periods gives a high
        information ratio.

        The summary has a row per metric and horizon with the mean IC, its standard deviation, the
        IC information ratio (mean divided by standard deviation), the t-statistic of the mean with
        its p-value, the skewness and excess kurtosis of the IC, the share of periods with a
        positive IC, the number of periods, the top minus bottom spread (the mean forward return of
        the highest quantile minus that of the lowest), the annualized alpha and the beta of the
        factor-weighted long-short portfolio against the equally weighted tickers, the mean turnover
        of the top and bottom quantiles and the mean rank autocorrelation of the metric from one
        period to the next. With detail, it returns one part of the analysis per period, quantile or
        group for a single metric.

        The formula is as follows:

        - IC(t) = Spearman rank correlation of f(i, t) and R(i, t, t + h) over the tickers i,
          with f the factor and R the forward return over the horizon h
        - Demeaned return: R(i, t) − mean R(j, t) over the tickers j (of the group of i)
        - Portfolio weight: w(i, t) = (f(i, t) − mean f(t)) / Σ_j |f(j, t) − mean f(t)|
        - Portfolio return: R_F(t) = Σ_i w(i, t) × R(i, t, t + h)
        - Alpha and beta: R_F(t) = α + β × R_U(t) + ε(t), with R_U(t) = mean R(i, t, t + h)
        - Annualized Alpha = (1 + α) ^ (periods per year / h) − 1
        - Per period: (1 + mean return) ^ (1 / h) − 1 and standard error / √h
        - Standard error of a quantile = standard deviation over the periods / √periods

        Each metric is measured within each period (e.g. the volatility of a month's daily
        returns) and compared with the returns from the close of the period's last trading day
        onwards, using the dividend and split adjusted close, so a metric never overlaps the
        returns it is compared with. Each ticker keeps its own history: a ticker without a
        value or a return in a period is left out of that period only.

        A factor of your own is a DataFrame with the dates as index and the tickers as
        columns, applied from each date on, or with the tickers as index and periods as
        columns, applied from the end of each period. Make sure each value was known on the
        date it is given for, or the analysis looks ahead.

        Shares similar logic:
        - alphalens, factor_information_coefficient: https://github.com/stefan-jansen/alphalens-reloaded/blob/v0.4.4/src/alphalens/performance.py#L28
        - alphalens, factor_returns: https://github.com/stefan-jansen/alphalens-reloaded/blob/v0.4.4/src/alphalens/performance.py#L210
        - alphalens, mean_return_by_quantile: https://github.com/stefan-jansen/alphalens-reloaded/blob/v0.4.4/src/alphalens/performance.py#L461

        Args:
            metrics (str | list[str] | pd.DataFrame): A performance method name with a value per ticker and
                period, e.g. "get_sharpe_ratio", a list of them, "all", or a factor of your own
                as a DataFrame. "all" holds the returns, excess return, beta, Sharpe, Sortino,
                Treynor and information ratios, tracking error, Jensen's alpha, Calmar and
                Omega ratios and the win rate. Defaults to "all".
            period (str, optional): The frequency at which the metrics are measured and
                evaluated. Options are: "weekly", "monthly", "quarterly", "yearly". Each metric is calculated
                within each period from that period's daily returns, so use monthly or longer
                for enough returns per period. Defaults to "monthly".
            horizons (list[int] | int | None, optional): The number of periods the forward
                returns cover. Defaults to None, which uses about a period, a quarter and a
                year ahead (1, 3 and 12 months for monthly).
            quantiles (int, optional): The number of quantiles the tickers are sorted into per
                period. Defaults to 5.
            periods_apart (int, optional): The number of periods between the dates the
                quantile turnover and the rank autocorrelation compare. Defaults to 1.
            detail (str | None, optional): Returns one part of the analysis of a single metric
                instead of the summary: "information_coefficient" (per period and horizon),
                "information_coefficient_by_month" or "information_coefficient_by_year" (the mean IC
                per calendar month or year), "group_information_coefficient" (the IC statistics
                within each group, needs group), "quantile_returns" (the mean forward return per
                quantile and horizon), "quantile_returns_by_date" (per period),
                "quantile_return_statistics" (the mean, its standard error and both per single
                period), "quantile_statistics" (the range, mean and count of the factor values in
                each quantile), "quantile_turnover" (per period and quantile),
                "rank_autocorrelation" (per period), "factor_weights" (the weights of the
                factor-weighted long-short portfolio per period and ticker), "factor_returns" (its
                return per period and horizon), "cumulative_returns" (the growth of 1 in it and in
                each quantile) or "event_returns" (the mean price path of each quantile from the
                longest horizon before to the longest horizon after the periods it is formed).
                Defaults to None.
            column (str | None, optional): The output to use of a metric with several.
                Defaults to None, which uses the first.
            group (str | dict | None, optional): The group of each ticker for the group options:
                "sector" or "industry" (or another company profile item) from the company profiles
                of Financial Modeling Prep, or a dictionary of tickers and their groups. Defaults to
                None.
            demeaned (bool, optional): Whether to compare the forward returns with the mean of
                the tickers in each period, or with group with the mean of the ticker's group:
                a group-neutral analysis, in which the IC is group-adjusted and the
                factor-weighted portfolio neutral within each group. Without group the IC
                stays the same, as subtracting one mean from every ticker keeps their ranks.
                Defaults to False.
            binning (str, optional): How the tickers are sorted into quantiles per period. Options are:
                "quantile" (equally sized), "equal_width" (bins of equal width between the lowest
                and highest value), "zero_aware" (half of the quantiles for the negative and half
                for the zero and positive values, for a signal centred on zero; needs an even number
                of quantiles), "group" (equally sized within each group, needs group). Defaults to
                "quantile".
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            **kwargs: Parameters passed to the metrics that take them, e.g. alpha=0.05.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The summary per metric and horizon, or with detail that part of the
            analysis of the metric.

        Notes:
            - The IC is reported as measured: short-term reversal, where last month's
              winners go on to lag, shows as a negative IC for the monthly return. Read the
              sign as the direction of the relationship.
            - The cross-section is the tickers of the Toolkit, so use at least ten and
              preferably twenty or more for a meaningful rank correlation and quantiles.
            - A horizon longer than one period makes consecutive returns overlap, which
              inflates the t-statistic; judge those horizons by their mean IC and IC
              information ratio.
            - The groups from the company profiles are today's sector or industry, applied
              to every period.
            - The factor-weighted portfolio is long-short: its weights sum to zero and their
              absolute values to one. A horizon longer than one period makes its returns
              overlap too, so its alpha and beta are averages rather than tests.
            - The cumulative returns split the capital over as many portfolios as the horizon
              has periods, started on consecutive periods and each rebalanced once per
              horizon; a value shows when its holding period ends, so the series starts at 1.
            - The IC is that of Grinold, R.C. and Kahn, R.N. (2000). "Active Portfolio
              Management." 2nd ed. McGraw-Hill; the factor-weighted portfolio, the demeaned
              and group-neutral returns that of Qian, E., Hua, R. and Sorensen, E. (2007).
              "Quantitative Equity Portfolio Management." Chapman & Hall/CRC; the alpha that
              of Jensen, M.C. (1968). "The Performance of Mutual Funds in the Period
              1945-1964." Journal of Finance 23(2); the event returns follow MacKinlay, A.C.
              (1997). "Event Studies in Economics and Finance." Journal of Economic Literature
              35(1).
            - The top minus bottom spread of alphalens (Stefan Jansen's alphalens-reloaded)
              is half of the one here (a gross exposure of one rather than two), it
              drops a period and ticker without every horizon's return where each horizon
              here uses its own, and its event returns average the single-period returns at
              each offset where these are the price paths rebased to the period itself.

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

        # Momentum: the return over the previous twelve months, without the latest month.
        prices = toolkit.get_historical_data()["Adj Close"]
        momentum = (prices.shift(21) / prices.shift(252) - 1).drop(columns="Benchmark")

        toolkit.performance.get_factor_analysis(momentum, horizons=[1, 12])
        ```

        Which returns:

        |                                |   Mean IC |   IC Standard Deviation |   IC Information Ratio |   t-Statistic |   P-Value |   IC Skewness |   IC Kurtosis |   Positive Share |   Observations |   Top Minus Bottom |   Annualized Alpha |   Beta |   Quantile Turnover |   Rank Autocorrelation |
        |:-------------------------------|----------:|------------------------:|-----------------------:|--------------:|----------:|--------------:|--------------:|-----------------:|---------------:|-------------------:|-------------------:|-------:|--------------------:|-----------------------:|
        | ('Custom Factor', '1 Month')   |    0.018  |                  0.374  |                 0.0481 |        0.4979 |    0.6196 |       -0.2886 |       -0.6254 |           0.4953 |            107 |             0.004  |             0.0648 | 0.0264 |              0.2391 |                 0.8821 |
        | ('Custom Factor', '12 Months') |    0.1119 |                  0.4193 |                 0.2668 |        2.6144 |    0.0104 |       -0.4186 |       -0.2835 |           0.5729 |             96 |             0.0194 |             0.0152 | 0.1641 |              0.2391 |                 0.8821 |
        """
        if isinstance(metrics, pd.DataFrame):
            factors = {
                "Custom Factor": factor_model.to_daily_factor(
                    metrics, self._tickers, column
                )
            }
        else:
            factors = factor_model.get_daily_factors(
                self,
                factor_model.select_metrics(metrics, FACTOR_COLLECTIONS),
                kwargs,
                {"period": period, "within_period": True},
                self._tickers,
                column,
            )
        result = factor_model.get_factor_analysis(
            factors,
            self._historical_data["daily"]["Adj Close"],
            period,
            horizons,
            quantiles,
            periods_apart,
            detail,
            self._start_date,
            self._end_date,
            returns=self._historical_data["daily"].get("Return"),
            binning=binning,
            demeaned=demeaned,
            groups=factor_model.get_groups(group, self._profile_source, self._tickers),
        )

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

    @handle_portfolio
    @handle_errors
    def get_sharpe_ratio_standard_error(
        self,
        period: str | None = None,
        method: str = "iid",
        lags: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the standard error of the Sharpe ratio: how far the measured Sharpe ratio may
        lie from the true one. A Sharpe ratio measured over a year of daily returns is an
        estimate, and roughly 95% of the time the true ratio lies within two standard errors of
        it.

        The formulas are as follows, with SR the Sharpe ratio of one observation (the Sharpe ratio
        of `get_sharpe_ratio` without annualizing) and n the number of observations:

            - Independent normal returns: SE = SQRT( (1 + SR^2 / 2) / n ), for normally and independently distributed
              returns (Lo, 2002, equation 9)
            - Non-normal returns: SE = SQRT( (1 − γ₃·SR + ((γ₄ − 1) / 4)·SR^2) / (n − 1) ), for
              independent but skewed and fat-tailed returns, with γ₃ the skewness and γ₄ the
              non-excess (raw) kurtosis (Mertens, 2002; the standard error inside the
              Probabilistic Sharpe Ratio of Bailey and López de Prado, 2012)
            - GMM (serially correlated returns): SE = SQRT( ∂g/∂θ · Σ · ∂g/∂θ' / n ), for returns that may
              also be serially correlated and heteroskedastic (Lo, 2002, Appendix). With θ = (μ, σ²) and
              g(θ) = μ / σ, ∂g/∂θ = (1 / σ, −μ / (2σ³)) and Σ is the Newey and West (1987)
              long-run covariance of the moment conditions (r_t − μ, (r_t − μ)² − σ²).

        Negative skewness, fat tails and positive autocorrelation all widen the standard error, so
        the same Sharpe ratio is then less certain. The standard error is on the scale of the
        Sharpe ratio of one observation; multiply by SQRT(252) to compare it with a Sharpe ratio
        annualized by SQRT(252) (exact for the independent normal and non-normal errors, a rescaling of the
        same estimate for the GMM error).

        For more information about the method, see the following papers:

        - Lo, A.W. (2002). "The Statistics of Sharpe Ratios." Financial Analysts Journal, 58(4),
        36-52.
        - Mertens, E. (2002). "Comments on Variance of the IID Estimator in Lo (2002)." Working
        Paper, University of Basel.
        - Bailey, D.H., & López de Prado, M. (2012). "The Sharpe Ratio Efficient Frontier."
        Journal of Risk, 15(2), 3-44.
        - Newey, W.K., & West, K.D. (1987). "A Simple, Positive Semi-Definite, Heteroskedasticity
        and Autocorrelation Consistent Covariance Matrix." Econometrica, 55(3), 703-708.

        Also known as: Sharpe ratio confidence interval, Sharpe ratio estimation error.

        Args:
            period (str, optional): The period to use for the calculation. Options are: "daily", "weekly",
                "monthly", "quarterly", "yearly". Defaults to "quarterly" if the Toolkit is initialised with
                quarterly=True, otherwise "yearly".
            method (str, optional): The standard error, as described above. Options are: "iid" (independent
                normal returns), "non_normal" (independent skewed and fat-tailed returns), "gmm" (serially
                correlated and heteroskedastic returns). Defaults to "iid".
            lags (int, optional): The number of lags in the Newey-West covariance, only used
                when method="gmm". Defaults to None, which uses the Newey and West (1994) rule
                floor(4 * (n / 100)^(2/9)), 4 for a year of daily returns.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the standard errors. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The standard error of the Sharpe ratio per period.

        Notes:
        - The Sharpe ratio is computed from the daily excess returns within each period, as in
        `get_sharpe_ratio`; the daily period needs intraday data (initialise the Toolkit with an
        intraday period).
        - The non-normal error uses the bias-corrected sample skewness and kurtosis, as
        the Probabilistic Sharpe Ratio of `get_sharpe_ratio` does; the GMM error uses the moments of
        GMM (σ² with 1 / n).
        - With growth, the method calculates the growth of the values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["SPY", "TLT", "GLD", "BTCUSD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
            benchmark_ticker=None,
        )

        toolkit.performance.get_sharpe_ratio_standard_error(method="gmm")
        ```

        Which returns:

        | Date   |    SPY |    TLT |    GLD |   BTCUSD |
        |:-------|-------:|-------:|-------:|---------:|
        | 2021   | 0.0586 | 0.0584 | 0.0634 |   0.0506 |
        | 2022   | 0.0602 | 0.0567 | 0.0624 |   0.0497 |
        | 2023   | 0.0635 | 0.0564 | 0.0609 |   0.0533 |
        | 2024   | 0.0645 | 0.0629 | 0.0617 |   0.0506 |
        | 2025   | 0.0554 | 0.0611 | 0.0623 |   0.047  |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        standard_error = performance_model.get_sharpe_ratio_standard_error(
            self._get_column(period, "Excess Return", within_period=True),
            method=method,
            lags=lags,
        )

        return finalize_dataset(
            dataset=standard_error,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_portfolio
    @handle_errors
    def get_sharpe_ratio_difference_test(
        self,
        period: str | None = None,
        method: str = "memmel",
        lags: int | None = None,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Test whether the Sharpe ratio of each asset differs from that of the benchmark: is the difference in
        risk-adjusted return more than the estimation error of two Sharpe ratios measured on the same dates?

        Two Sharpe ratios measured over a year of daily returns are both estimates, and because the two assets
        move together their errors are correlated, which a comparison of two separate standard errors ignores.
        Jobson and Korkie (1981) derived the standard error of the difference for jointly normal returns, which
        Memmel (2003) corrected. Ledoit and Wolf (2008) showed that this test is unreliable when returns have fat
        tails or are serially correlated, as daily returns usually are, and replaced its standard error with a
        heteroskedasticity and autocorrelation consistent (HAC) one by the delta method.

        The formulas are as follows, with n the number of dates both have an excess return:

        - Memmel: z = (SR_a − SR_b) / √((2 − 2ρ + (SR_a² + SR_b² − 2·SR_a·SR_b·ρ²) / 2) / n)
        - Ledoit-Wolf: z = Δ / √(∇f' Ψ ∇f / n), with Δ = f(v) = μ_a / √(γ_a − μ_a²) − μ_b / √(γ_b − μ_b²)
        - ∇f = (γ_a / c_a, −γ_b / c_b, −μ_a / (2·c_a), μ_b / (2·c_b)), with c = (γ − μ²)^(3/2)
        - Ψ = n / (n − 4) · (Γ(0) + Σ_{j=1..L} (1 − j / (L + 1))·(Γ(j) + Γ(j)')), Γ(j) = Σ_t y(t)·y(t − j)' / n
        - y(t) = (r_a(t) − μ_a, r_b(t) − μ_b, r_a(t)² − γ_a, r_b(t)² − γ_b)
        - P-Value = 2 · (1 − Φ(|z|))

        Where SR is the Sharpe ratio of one observation (mean over standard deviation with n - 1 in the denominator,
        of the daily excess returns, as `get_sharpe_ratio` before annualizing), ρ the correlation of the two excess
        returns, μ and γ the means of the excess returns and of their squares, Ψ the Bartlett (Newey-West) estimate of
        their long-run covariance with L lags and Φ the standard normal distribution. A low p-value (e.g. below 0.05)
        means the Sharpe ratios differ significantly; a positive Sharpe Difference that the asset's Sharpe ratio is
        the higher one.

        For more information about the method, see the following papers:

        - Jobson, J.D. and Korkie, B.M. (1981). "Performance Hypothesis Testing with the Sharpe and Treynor
        Measures." Journal of Finance, 36(4), 889-908.
        - Memmel, C. (2003). "Performance Hypothesis Testing with the Sharpe Ratio." Finance Letters, 1, 21-23.
        - Ledoit, O. and Wolf, M. (2008). "Robust Performance Hypothesis Testing with the Sharpe Ratio." Journal
        of Empirical Finance, 15(5), 850-859.
        - Newey, W.K., & West, K.D. (1987). "A Simple, Positive Semi-Definite, Heteroskedasticity and
        Autocorrelation Consistent Covariance Matrix." Econometrica, 55(3), 703-708.

        Also known as: Jobson-Korkie test, Memmel test, Ledoit-Wolf test, Sharpe ratio equality test, Sharpe ratio
        significance test.

        Args:
            period (str, optional): The period to use for the calculation. Options are: "daily", "weekly",
                "monthly", "quarterly", "yearly". Defaults to "quarterly" if the Toolkit is initialised with
                quarterly=True, otherwise "yearly".
            method (str, optional): The test. Options are: "memmel" (the Jobson-Korkie test with Memmel's
                correction), "ledoit_wolf" (the HAC test of Ledoit and Wolf). Defaults to "memmel".
            lags (int, optional): The number of lags L in the Newey-West covariance, only used when
                method="ledoit_wolf". Defaults to None, which uses the Newey and West (1994) rule
                floor(4 * (n / 100)^(2/9)), 4 for a year of daily returns.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per period, for each ticker the Sharpe Difference (its Sharpe ratio minus that of the
                benchmark, of one observation), the Standard Error of the difference, the Z-Statistic and the
                P-Value.

        Notes:
        - The tests use the daily excess returns within each period, as `get_sharpe_ratio` does; the daily
        period needs intraday data (initialise the Toolkit with an intraday period). A period with fewer than five
        shared dates is NaN.
        - The Ledoit-Wolf Sharpe Difference uses standard deviations with 1 / n, as the delta method of Ledoit
        and Wolf (2008) does, so it is a factor √((n − 1) / n) from the Memmel one.
        - Ledoit and Wolf (2008) use a prewhitened Quadratic Spectral kernel and also propose a studentized bootstrap;
        the Toolkit uses the Bartlett kernel of the GMM standard error of `get_sharpe_ratio_standard_error`, which
        gives the same test with another weighting of the autocovariances. RiskLabAI's sharpe_difference_test
        calibrates the p-value with that bootstrap instead.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            "MSFT",
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.performance.get_sharpe_ratio_difference_test(method="ledoit_wolf")["MSFT"]
        ```

        Which returns:

        |      |   Sharpe Difference |   Standard Error |   Z-Statistic |   P-Value |
        |:-----|--------------------:|-----------------:|--------------:|----------:|
        | 2021 |              0.0088 |           0.0509 |        0.1722 |    0.8633 |
        | 2022 |             -0.0007 |           0.0326 |       -0.0218 |    0.9826 |
        | 2023 |              0.016  |           0.0556 |        0.2881 |    0.7733 |
        | 2024 |             -0.0632 |           0.046  |       -1.3738 |    0.1695 |
        | 2025 |             -0.0111 |           0.0478 |       -0.2329 |    0.8158 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        difference_test = performance_model.get_sharpe_ratio_difference_test(
            self._get_column(period, "Excess Return", within_period=True),
            self._get_column(
                period, "Excess Return", within_period=True, benchmark=True
            ),
            method=method,
            lags=lags,
        )

        # A period cut to a single date by the start or end date has no test.
        return apply_rounding(
            difference_test.loc[self._start_date : self._end_date].dropna(how="all"),
            rounding if rounding is not None else self._rounding,
        )

    @handle_portfolio
    @handle_errors
    def get_minimum_track_record_length(
        self,
        period: str | None = None,
        rolling: int | None = None,
        benchmark_sharpe_ratio: float = 0.0,
        confidence_level: float = 0.95,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Minimum Track Record Length (MinTRL): how many return observations are
        needed before the measured Sharpe ratio is, with the given confidence, above the
        benchmark Sharpe ratio. A short track record with a high Sharpe ratio can still be luck;
        the MinTRL says how long the record has to be for it not to be, taking the skewness and
        fat tails of the returns into account.

        The formula is as follows:

            - MinTRL = 1 + (1 − γ₃·SR + ((γ₄ − 1) / 4)·SR^2) · (Z_α / (SR − SR*))^2

        Where SR is the Sharpe ratio of one observation (not annualized), SR* the benchmark
        Sharpe ratio on the same scale, γ₃ the skewness and γ₄ the non-excess (raw) kurtosis
        of the excess returns (3 for a normal distribution) and Z_α the standard normal quantile
        at `confidence_level` (1.645 for 95%). With n = MinTRL observations, the Probabilistic
        Sharpe Ratio, the Probabilistic Sharpe Ratio of `get_sharpe_ratio`, equals `confidence_level`.
        Negative skewness and fat tails lengthen the record needed. When SR does not exceed SR*,
        no track record is long enough and the value is left out (NaN).

        By default one MinTRL is reported per `period`, computed from the daily excess returns
        within that period, so it counts trading days (divide by 252 for years). For a rolling
        period, `period` sets the frequency of the returns and the measure is computed over a
        rolling window of `rolling` such returns, so it counts periods (e.g. months).

        For more information about the method, see the following paper:

        - Bailey, D.H., & López de Prado, M. (2012). "The Sharpe Ratio Efficient Frontier."
        Journal of Risk, 15(2), 3-44.

        Also known as: MinTRL, minimum track record, required track record length.

        Referenced against:
        - quantstats, smart_sharpe (with get_autocorrelation_penalty): https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L1000

        Shares similar logic:
        - PerformanceAnalytics (R), MinTrackRecord: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/MinTRL.R#L49
        - PerformanceAnalytics (R), Return.excess: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/Return.excess.R#L42
        - quantstats, rolling_sharpe: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L1030
        - PerformanceAnalytics (R), SharpeRatio: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/SharpeRatio.R#L80
        - empyrical, sharpe_ratio: https://github.com/quantopian/empyrical/blob/0.5.5/empyrical/stats.py#L652
        - quantstats, sharpe: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L938

        Args:
            period (str, optional): The period to use for the calculation. Options are: "daily", "weekly",
                "monthly", "quarterly", "yearly". Defaults to "quarterly" if the Toolkit is initialised with
                quarterly=True, otherwise "yearly".
            rolling (int, optional): The rolling window size, in observations of `period`.
                At least 2. Defaults to None, one value per period.
            benchmark_sharpe_ratio (float, optional): The benchmark Sharpe ratio SR*, on the scale
                of one observation (e.g. 1 / SQRT(252) for an annualized benchmark of 1 on daily
                returns). Defaults to 0.0, i.e. any skill at all.
            confidence_level (float, optional): The confidence level. Defaults to 0.95.
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
            pd.DataFrame: The minimum number of return observations per period.

        Notes:
        - The skewness and kurtosis are the bias-corrected sample estimates, as in
        the Probabilistic Sharpe Ratio of `get_sharpe_ratio`.
        - the daily period without a rolling window needs intraday data (initialise the Toolkit with an
          intraday period).
        - With growth, the method calculates the growth of the values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["SPY", "TLT", "GLD", "BTCUSD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
            benchmark_ticker=None,
        )

        toolkit.performance.get_minimum_track_record_length()
        ```

        Which returns:

        | Date   |      SPY |    TLT |      GLD |   BTCUSD |
        |:-------|---------:|-------:|---------:|---------:|
        | 2021   |  176.667 |    nan |  nan     | 1102.59  |
        | 2023   |  282.689 | 755917 | 1474.87  |  178.459 |
        | 2024   |  320.09  |    nan |  380.932 |  344.545 |
        | 2025   | 1227.15  | 195160 |  134.903 |  nan     |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"
        check_rolling(rolling, self._historical_data, period)

        if rolling:
            excess_return = performance_model.get_excess_return(
                self._get_column(period, "Return", within_period=False),
                self._risk_free_rate_data[period][self._tickers_with_returns],
            )
            sharpe_ratio = performance_model.get_sharpe_ratio(
                excess_return, window=rolling
            )
            skewness = get_skewness(excess_return, window=rolling)
            kurtosis = get_kurtosis(excess_return, fisher=False, window=rolling)
        else:
            excess_return = self._get_column(
                period, "Excess Return", within_period=True
            )
            sharpe_ratio = performance_model.get_sharpe_ratio(excess_return)
            skewness = get_skewness(excess_return)
            kurtosis = get_kurtosis(excess_return, fisher=False)

        minimum_track_record_length = performance_model.get_minimum_track_record_length(
            sharpe_ratio=sharpe_ratio,
            benchmark_sharpe_ratio=benchmark_sharpe_ratio,
            skewness=skewness,
            kurtosis=kurtosis,
            confidence_level=confidence_level,
        )

        # Without a Sharpe ratio above the benchmark's no track record is long enough, so
        # every value is NaN and the result would otherwise be empty without a reason.
        if (
            minimum_track_record_length.isna().all().all()
            and sharpe_ratio.notna().any().any()
        ):
            logger.warning(
                "The Sharpe ratio does not exceed the benchmark Sharpe ratio of %s in any period, "  # noqa: E501
                "so no track record is long enough and the Minimum Track Record Length is empty.",  # noqa: E501
                benchmark_sharpe_ratio,
            )

        return finalize_dataset(
            dataset=minimum_track_record_length,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_portfolio
    @handle_errors
    def get_coskewness(
        self,
        period: str | None = None,
        method: Literal["standardized", "residual"] = "standardized",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the co-skewness of each asset with the benchmark: how the asset's return moves
        with the squared, i.e. large, moves of the market in either direction. An asset with a
        negative co-skewness does worse when the market makes large moves and adds to the
        downside skewness of a market portfolio, so investors demand a higher return for holding
        it (Kraus and Litzenberger, 1976; Harvey and Siddique, 2000). A positive co-skewness
        works as a hedge against large market moves.

        The formulas are as follows:

            - Standardized: Co-Skewness = E[(r_i − μ_i)(r_m − μ_m)^2] / (σ_i · σ_m^2)
            - Residual: β_SKD = E[ε_i · ε_m^2] / (SQRT(E[ε_i^2]) · E[ε_m^2]) (Harvey and
              Siddique, 2000)

        Where r_i is the asset return, r_m the benchmark return, ε_m = r_m − μ_m and ε_i the
        residual of the regression r_i = α + β·r_m + ε_i. The residual version only looks at the
        part of the asset's return the market does not explain, so the market's own skewness does
        not enter through the beta. For the benchmark itself the standardized co-skewness is the
        benchmark's skewness.

        By default one value is reported per `period`, computed from the daily returns within
        that period on the dates both the asset and the benchmark have a return. The expectations
        are population moments (1 / n), so σ_i and σ_m use 1 / n as well and the measure does not
        depend on the scale of the returns.

        For more information about the method, see the following papers:

        - Harvey, C.R., & Siddique, A. (2000). "Conditional Skewness in Asset Pricing Tests."
        Journal of Finance, 55(3), 1263-1295.
        - Kraus, A., & Litzenberger, R.H. (1976). "Skewness Preference and the Valuation of Risk
        Assets." Journal of Finance, 31(4), 1085-1100.

        Also known as: coskewness, systematic skewness, co-skewness beta.

        Shares similar logic:
        - PerformanceAnalytics (R), CoSkewness: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/CoMoments.R#L391

        Args:
            period (str, optional): The period to use for the calculation. Options are: "daily", "weekly",
                "monthly", "quarterly", "yearly". Defaults to "quarterly" if the Toolkit is initialised with
                quarterly=True, otherwise "yearly".
            method (str, optional): The co-moment, as described above. Options are: "standardized" (the
                standardized co-moment), "residual" (the residual co-moment). Defaults to "standardized".
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
            pd.DataFrame: The co-skewness of each asset with the benchmark per period.

        Notes:
        - The returns are the total returns (Adj Close), as for `get_beta`; the risk free rate,
        nearly constant within a period, drops out with the means.
        - The daily period needs intraday data (initialise the Toolkit with an intraday period).
        - With growth, the method calculates the growth of the values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "TLT", "GLD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.performance.get_coskewness()
        ```

        Which returns:

        | Date   |    AAPL |    MSFT |     TLT |     GLD |
        |:-------|--------:|--------:|--------:|--------:|
        | 2021   | -0.1202 | -0.1723 | -0.069  | -0.0972 |
        | 2022   |  0.1063 |  0.09   |  0.1369 |  0.0849 |
        | 2023   |  0.1241 |  0.085  |  0.0816 |  0.059  |
        | 2024   | -0.423  | -0.6011 | -0.0005 | -0.418  |
        | 2025   |  1.4001 |  1.4254 |  0.2683 |  0.6826 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        coskewness = performance_model.get_coskewness(
            self._get_column(period, "Return", within_period=True),
            self._get_column(period, "Return", within_period=True, benchmark=True),
            method=method,
        )

        return finalize_dataset(
            dataset=coskewness,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_portfolio
    @handle_errors
    def get_cokurtosis(
        self,
        period: str | None = None,
        method: Literal["standardized", "residual"] = "standardized",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the co-kurtosis of each asset with the benchmark: how the asset's return moves
        with the cubed moves of the market, i.e. how much it shares the market's extreme moves.
        An asset with a high co-kurtosis adds to the fat tails of a market portfolio, and
        Dittmar (2002) shows that investors who dislike kurtosis demand a higher return for
        holding it.

        The formulas are as follows:

            - Standardized: Co-Kurtosis = E[(r_i − μ_i)(r_m − μ_m)^3] / (σ_i · σ_m^3)
            - Residual: E[ε_i · ε_m^3] / (SQRT(E[ε_i^2]) · E[ε_m^2]^(3/2))

        Where r_i is the asset return, r_m the benchmark return, ε_m = r_m − μ_m and ε_i the
        residual of the regression r_i = α + β·r_m + ε_i, the residual form Harvey and Siddique
        (2000) use for co-skewness. For the benchmark itself the standardized co-kurtosis is the
        benchmark's non-excess kurtosis (3 for normal returns); for an asset with a correlation ρ
        to normally distributed market returns it is 3ρ.

        By default one value is reported per `period`, computed from the daily returns within
        that period on the dates both the asset and the benchmark have a return. The expectations
        are population moments (1 / n), so σ_i and σ_m use 1 / n as well.

        For more information about the method, see the following papers:

        - Dittmar, R.F. (2002). "Nonlinear Pricing Kernels, Kurtosis Preference, and Evidence
        from the Cross Section of Equity Returns." Journal of Finance, 57(1), 369-403.
        - Harvey, C.R., & Siddique, A. (2000). "Conditional Skewness in Asset Pricing Tests."
        Journal of Finance, 55(3), 1263-1295.

        Also known as: cokurtosis, systematic kurtosis, co-kurtosis beta.

        Shares similar logic:
        - PerformanceAnalytics (R), CoKurtosis: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/CoMoments.R#L466

        Args:
            period (str, optional): The period to use for the calculation. Options are: "daily", "weekly",
                "monthly", "quarterly", "yearly". Defaults to "quarterly" if the Toolkit is initialised with
                quarterly=True, otherwise "yearly".
            method (str, optional): The co-moment, as described above. Options are: "standardized" (the
                standardized co-moment), "residual" (the residual co-moment). Defaults to "standardized".
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
            pd.DataFrame: The co-kurtosis of each asset with the benchmark per period.

        Notes:
        - The returns are the total returns (Adj Close), as for `get_beta`; the risk free rate,
        nearly constant within a period, drops out with the means.
        - The daily period needs intraday data (initialise the Toolkit with an intraday period).
        - With growth, the method calculates the growth of the values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "TLT", "GLD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.performance.get_cokurtosis()
        ```

        Which returns:

        | Date   |    AAPL |    MSFT |     TLT |    GLD |
        |:-------|--------:|--------:|--------:|-------:|
        | 2021   |  2.3149 |  2.4305 | -0.3616 | 0.761  |
        | 2022   |  3.0324 |  2.9407 |  0.6437 | 1.1731 |
        | 2023   |  2.0735 |  1.5943 |  0.3763 | 0.0201 |
        | 2024   |  2.5064 |  3.5367 | -0.2675 | 1.2651 |
        | 2025   | 22.5236 | 18.9645 |  1.749  | 7.2913 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        cokurtosis = performance_model.get_cokurtosis(
            self._get_column(period, "Return", within_period=True),
            self._get_column(period, "Return", within_period=True, benchmark=True),
            method=method,
        )

        return finalize_dataset(
            dataset=cokurtosis,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_portfolio
    @handle_errors
    def get_serenity_index(
        self,
        period: str | None = None,
        alpha: float = 0.05,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Serenity Index (Serenity Ratio) of an investment portfolio or asset, KeyQuant's
        excess return per unit of drawdown risk: the Ulcer Performance Index, penalized further when the
        worst drawdowns are deep compared to the volatility of the returns.

        The formula is as follows:

        - Ulcer Index = SQRT(MEAN(Drawdown^2)), with each day's drawdown measured from the highest value since
          the start of the period
        - Pitfall Indicator = |Conditional Drawdown at Risk(alpha)| / Volatility
        - Serenity Index = (Period Return - Risk-Free Rate) / (Ulcer Index x Pitfall Indicator)

        The Conditional Drawdown at Risk is the average of the worst `alpha` share of the daily drawdowns
        within the period and the Volatility the standard deviation of the daily returns within the period
        scaled to the period (times the square root of the trading days in it), so the Pitfall Indicator
        counts how many standard deviations the worst drawdowns reach. Two assets with the same Ulcer
        Performance Index get a lower Serenity Index when one of them has the deeper tail of drawdowns.

        For more information, see the following sources:

        - KeyQuant (2017). "Assessing Performance Through Risk: Part 1." KeyQuant white paper (the Serenity
        Ratio and the Pitfall Indicator).
        - Martin, P.G., & McCann, B.B. (1989). "The Investor's Guide to Fidelity Funds." John Wiley & Sons (the
        Ulcer Index).

        Also known as: Serenity Ratio.

        Shares similar logic:
        - PerformanceAnalytics (R), Return.excess: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/Return.excess.R#L42
        - quantstats, serenity_index: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L1906

        Args:
            period (str, optional): The period to use for the calculation. Options are: "daily", "weekly",
                "monthly", "quarterly", "yearly". Defaults to "quarterly" if the Toolkit is initialised with
                quarterly=True, otherwise "yearly".
            alpha (float, optional): The share of the worst drawdowns the Conditional Drawdown at Risk averages.
                Defaults to 0.05.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Serenity Index values.

        Notes:
        - quantstats' serenity_index divides the sum of the daily returns (not the compounded excess return)
        by its Ulcer Index (the root of the summed squared drawdowns over n - 1 observations) times a Pitfall
        Indicator built from the gaussian expected shortfall of the drawdown series and the daily (not period)
        standard deviation, over the whole history. The Toolkit follows KeyQuant's definition with its own
        Ulcer Index, historical Conditional Drawdown at Risk and volatility per period, so the values differ
        from quantstats; with quantstats' components the formula gives its value (to 1e-12).
        - A period without a drawdown leaves the index undefined (NaN).
        - With growth, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "KO"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.performance.get_serenity_index()
        ```

        Which returns:

        | Date   |    AAPL |    MSFT |      KO |
        |:-------|--------:|--------:|--------:|
        | 2022   | -2.3292 | -1.5965 |  1.6427 |
        | 2023   | 12.9384 | 21.254  | -1.0582 |
        | 2024   |  6.842  |  1.8838 |  0.7925 |
        | 2025   |  0.5514 |  2.0173 |  4.7907 |
        """
        validate_alpha(alpha)
        period = period if period else "quarterly" if self._quarterly else "yearly"

        returns = self._get_column(period, "Return", within_period=True)
        period_returns = self._get_column(period, "Return", within_period=False)
        excess_return = performance_model.get_excess_return(
            period_returns, self._risk_free_rate_data[period][period_returns.columns]
        )

        serenity_index = performance_model.get_serenity_index(
            excess_return,
            get_ui(returns, rolling=None),
            get_conditional_drawdown_at_risk(returns, alpha),
            get_volatility(
                self._get_column("daily", "Return", within_period=False), period
            ),
        )

        return finalize_dataset(
            dataset=serenity_index,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_portfolio
    @handle_errors
    def get_risk_measure_ratio(
        self,
        period: str | None = None,
        risk_measure: str = "cvar",
        within_period: bool = True,
        alpha: float = 0.05,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the mean excess return per unit of any risk measure of the risk module: the Sharpe ratio
        with a tail, deviation or drawdown measure in place of the standard deviation.

        The formula is as follows:

        - Risk Measure Ratio = Mean(Daily Return - Daily Risk Free Rate) / Risk Measure(Daily Returns)

        The risk measures are those of `toolkit.risk`, as a positive number:

        - Value at Risk: the historical Value at Risk, the `alpha` quantile of the returns.
        - Conditional Value at Risk: the historical Conditional Value at Risk (Expected Shortfall), the mean of the
          returns at or below it (the Conditional Sharpe or STARR ratio).
        - Entropic Value at Risk: the historical Entropic Value at Risk.
        - Entropic Risk Measure: the Entropic Risk Measure with a risk aversion of 1.
        - Worst Realization: the Worst Realization, the lowest return.
        - Mean Absolute Deviation: the Mean Absolute Deviation from the mean.
        - Lower Partial Moment: the first Lower Partial Moment below the mean, MEAN(MAX(Mean - Return, 0)).
        - Gini Mean Difference: the Gini Mean Difference.
        - Maximum Drawdown: the Maximum Drawdown (a Calmar ratio).
        - Average Drawdown: the Average Drawdown over every day.
        - Drawdown at Risk, Conditional Drawdown at Risk and Entropic Drawdown at Risk: the three drawdown
          measures at `alpha`.
        - Ulcer Index: the Ulcer Index, the root mean square drawdown.

        The ratio is computed from the daily returns within each period (e.g. each year), with the mean and
        the risk measure per day and not annualized, as skfolio's ratio measures are. Over the whole history
        the returns of `period` over the whole history give one ratio per ticker. The drawdowns compound the
        returns and start from a wealth of 1 at the start of each period.

        For more information, see the following sources:

        - Cajas, D. (2021). "Riskfolio-Lib: Portfolio Optimization in Python", the risk-adjusted return
        ratios.
        - Gregoriou, G.N., & Gueyie, J.P. (2003). "Risk-Adjusted Performance of Funds of Hedge Funds Using a
        Modified Sharpe Ratio." Journal of Wealth Management, 6(3), 77-83.

        Also known as: return over risk, risk-adjusted return ratio, generalized Sharpe ratio, conditional Sharpe
        ratio, VaR ratio, CVaR ratio, EVaR ratio, CDaR ratio, MAD ratio, Gini mean difference ratio.

        Referenced against:
        - skfolio, RatioMeasure: https://skfolio.org/generated/skfolio.measures.RatioMeasure.html

        Shares similar logic:
        - Riskfolio-Lib, Sharpe: https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html#RiskFunctions.Sharpe
        - PerformanceAnalytics (R), SharpeRatio (VaR and ES): https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/SharpeRatio.R#L80
        - PerformanceAnalytics (R), Return.excess: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/Return.excess.R#L42

        Args:
            period (str, optional): The period to use for the calculation. Options are: "daily", "weekly",
                "monthly", "quarterly", "yearly". Defaults to "quarterly" if the Toolkit is initialised with
                quarterly=True, otherwise "yearly".
            risk_measure (str, optional): The risk measure, as described above. Options are: "var", "cvar",
                "evar", "erm", "worst", "mad", "lpm", "gmd", "max_drawdown", "average_drawdown", "dar", "cdar",
                "edar", "ulcer". Defaults to "cvar".
            within_period (bool, optional): Whether to calculate the ratio from the daily returns within each
                period, or from the returns of `period` over the whole history (one value per ticker).
                Defaults to True.
            alpha (float, optional): The tail probability of the VaR, CVaR, EVaR, ERM, DaR, CDaR and EDaR.
                Defaults to 0.05.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame | pd.Series: The ratio per period and ticker, or per ticker with within_period=False.

        Notes:
        - The ratio agrees with skfolio's ratio measures where the risk measures are estimated alike: with the mean
        absolute deviation, the first lower partial moment, the worst realization, the Gini mean difference,
        the Entropic Value at Risk, the Entropic Risk Measure and, on the drawdowns, the maximum and average
        drawdown, the Ulcer Index and the Entropic Drawdown at Risk. skfolio measures the drawdowns without
        compounding by default, the Toolkit compounds them. The historical Value at Risk,
        Conditional Value at Risk and their drawdown versions use the interpolated quantile of the risk module,
        where skfolio takes the order statistic and the exact (Rockafellar-Uryasev) Conditional Value at Risk,
        so these differ slightly from skfolio on short samples.
        - `get_starr_ratio` divides the period excess return by the CVaR scaled to the period, `get_calmar_ratio`
        and `get_ulcer_performance_index` the period excess return by the drawdown measure; this ratio keeps the
        mean and the risk measure per day, so every risk measure is on the same scale.
        - A risk measure of 0 (e.g. no drawdown) leaves the ratio undefined (NaN).
        - With growth, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "KO"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.performance.get_risk_measure_ratio(risk_measure="cdar")
        ```

        Which returns:

        |      |    AAPL |    MSFT |      KO |
        |:-----|--------:|--------:|--------:|
        | 2021 |  0.0085 |  0.025  |  0.0072 |
        | 2022 | -0.0039 | -0.0036 |  0.0024 |
        | 2023 |  0.0119 |  0.0153 | -0.0019 |
        | 2024 |  0.0073 |  0.003  |  0.0015 |
        | 2025 |  0.0017 |  0.0031 |  0.0052 |
        """
        validate_alpha(alpha)
        period = period if period else "quarterly" if self._quarterly else "yearly"

        if within_period:
            returns = self._get_column(period, "Return", within_period=True)
            excess_returns = self._get_column(
                period, "Excess Return", within_period=True
            )
        else:
            returns = self._get_column(period, "Return", within_period=False)
            excess_returns = performance_model.get_excess_return(
                returns, self._risk_free_rate_data[period][returns.columns]
            )

        risk_measure_ratio = performance_model.get_risk_measure_ratio(
            excess_returns, returns, risk_measure, alpha
        )

        return finalize_dataset(
            dataset=risk_measure_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            apply_slice=within_period,
            dropna=True,
        )

    @handle_portfolio
    @handle_errors
    def get_probabilistic_momentum(
        self,
        period: str | None = None,
        rolling: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Probabilistic Momentum of Varadi (2014): the confidence that an asset outperforms the
        benchmark, the probability under Student's t distribution that the mean of the differences between
        their returns is above 0.

        The formula is as follows:

        - Information Ratio = Mean(Asset Return - Benchmark Return) / Standard Deviation(Asset Return - Benchmark Return)
        - Probabilistic Momentum = T(Information Ratio x SQRT(N), N - 1)

        Where N is the number of return differences and T the cumulative Student's t distribution with N - 1
        degrees of freedom. It is computed from the daily returns within each period, or with `rolling` from
        the returns of `period` over every rolling window of that many periods (e.g. daily returns and
        a rolling window of 60 for Varadi's 60-day lookback). A value above 0.5 means the asset did better than the
        benchmark, and the more returns and the steadier the difference, the closer it gets to 1. Varadi
        switches between two assets only when the confidence passes a threshold such as 0.6, rather than on
        the sign of the return difference alone, which cuts the number of trades a momentum strategy makes on
        noise.

        For more information, see the following source:

        - Varadi, D. (2014). "Are Simple Momentum Strategies Too Dumb? Introducing Probabilistic Momentum." CSS
        Analytics, 28 January 2014.

        Also known as: probabilistic momentum, momentum confidence.

        Referenced against:
        - ffn, calc_prob_mom: https://github.com/pmorissette/ffn/blob/v1.4.1/ffn/core.py#L1739

        Shares similar logic:
        - PerformanceAnalytics (R), InformationRatio: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/InformationRatio.R#L32
        - quantstats, information_ratio: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L2884
        - ffn, calc_information_ratio: https://github.com/pmorissette/ffn/blob/v1.4.1/ffn/core.py#L1691

        Args:
            period (str, optional): The period to use for the calculation. Options are: "daily", "weekly",
                "monthly", "quarterly", "yearly". Defaults to "quarterly" if the Toolkit is initialised with
                quarterly=True, otherwise "yearly".
            rolling (int | None, optional): The number of periods in each rolling window. At least 2. Defaults to None, which
                uses the daily returns within each period.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The probability, between 0 and 1, per period (or date) and ticker.

        Notes:
        - It agrees with ffn's calc_prob_mom of the ticker against the benchmark over the same returns.
        - The returns are compared on the dates both the ticker and the benchmark have a return, as in
        `get_information_ratio`.
        - With growth, the method calculates the growth of the values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "KO"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.performance.get_probabilistic_momentum()
        ```

        Which returns:

        | Date   |   AAPL |   MSFT |     KO |
        |:-------|-------:|-------:|-------:|
        | 2021   | 0.6635 | 0.9006 | 0.2077 |
        | 2022   | 0.3459 | 0.2999 | 0.927  |
        | 2023   | 0.8956 | 0.8939 | 0.0311 |
        | 2024   | 0.6292 | 0.255  | 0.217  |
        | 2025   | 0.4214 | 0.4813 | 0.4662 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"
        check_rolling(rolling, self._historical_data, period)

        returns = self._get_column(period, "Return", within_period=not rolling)
        benchmark_returns = self._get_column(
            period, "Return", within_period=not rolling, benchmark=True
        )

        if rolling:
            information_ratio = performance_model.get_information_ratio(
                returns, benchmark_returns, window=rolling
            )
            observations = information_ratio.notna() * float(rolling)
        else:
            information_ratio = performance_model.get_information_ratio(
                returns, benchmark_returns
            )
            observations = (
                returns.sub(benchmark_returns, axis=0).groupby(level=0).count()
            )

        probabilistic_momentum = performance_model.get_probabilistic_momentum(
            information_ratio, observations.reindex_like(information_ratio)
        )

        return finalize_dataset(
            dataset=probabilistic_momentum,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_errors
    def get_trailing_returns(
        self, rounding: int | None = None, show_columns: str | list[str] | None = None
    ):
        """
        Calculate the trailing returns of every ticker up to its last close: month to date, the last 3 and 6
        months, year to date, the last year, the last 3, 5 and 10 years (annualized) and since the start of
        the data (annualized), the table of returns fund factsheets show.

        The formula is as follows:

        - Trailing Return = Last Close / Close at the Start of the Window - 1
        - Annualized Trailing Return = (1 + Trailing Return) ^ (365.25 / Calendar Days in the Window) - 1

        The close at the start of a window is the last close on or before the date the window lies before the
        last close (e.g. 3 months earlier). When the last close falls on the last business day of its month,
        the windows run from month end to month end (a history up to 30 June has its 3 month return from the
        close of 31 March). Month to date and year to date start from the last close of the previous month
        and year. The returns of 3 years and longer are annualized over the calendar days of the window, and
        the return since inception is the Compound Annual Growth Rate over the date range (see
        `get_compound_growth_rate`). The returns use the dividend-adjusted close, so they are total returns.


        Also known as: trailing performance, period returns, MTD, YTD, 1Y, 3Y, 5Y, 10Y returns.

        Referenced against:
        - ffn, PerformanceStats: https://github.com/pmorissette/ffn/blob/v1.4.1/ffn/core.py#L123

        Shares similar logic:
        - ffn, calc_mtd: https://github.com/pmorissette/ffn/blob/v1.4.1/ffn/core.py#L1426
        - ffn, calc_ytd: https://github.com/pmorissette/ffn/blob/v1.4.1/ffn/core.py#L1437
        - PerformanceAnalytics (R), table.TrailingPeriods: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/table.RollingPeriods.R#L36

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The trailing returns as the index and the tickers (and the Benchmark) as the columns.

        Notes:
        - A window that starts before the first close in the date range is NaN, so set start_date at least
        ten years back for the 10 year return.
        - The month to date, 3 and 6 month, year to date and 1 year returns agree with ffn's mtd, three_month,
        six_month, ytd and one_year. ffn starts its 3, 5 and 10 year returns from the first close on or after
        the start of the window and does not use the month end rule for them; the Toolkit starts every window
        the same way, so these can differ by a day of returns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "KO"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2015-01-01",
            end_date="2025-12-31",
        )

        toolkit.performance.get_trailing_returns()
        ```

        Which returns:

        |                              |    AAPL |    MSFT |      KO |   Benchmark |
        |:-----------------------------|--------:|--------:|--------:|------------:|
        | Month to Date                | -0.0251 | -0.0171 | -0.0372 |      0.0008 |
        | 3 Months                     |  0.0687 | -0.0645 |  0.0615 |      0.0267 |
        | 6 Months                     |  0.3278 | -0.0243 |  0.0027 |      0.11   |
        | Year to Date                 |  0.0905 |  0.1558 |  0.156  |      0.1772 |
        | 1 Year                       |  0.0905 |  0.1558 |  0.156  |      0.1772 |
        | 3 Years (Annualized)         |  0.2851 |  0.273  |  0.0634 |      0.2284 |
        | 5 Years (Annualized)         |  0.1605 |  0.1778 |  0.0818 |      0.1434 |
        | 10 Years (Annualized)        |  0.2761 |  0.2583 |  0.0834 |      0.1471 |
        | Since Inception (Annualized) |  0.2459 |  0.2548 |  0.0807 |      0.1343 |
        """
        prices = (
            self._historical_data["daily"]
            .loc[:, "Adj Close"]
            .loc[self._start_date : self._end_date]
        )

        trailing_returns = performance_model.get_trailing_returns(prices)

        return apply_rounding(
            trailing_returns, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_calendar_returns(
        self, rounding: int | None = None, show_columns: str | list[str] | None = None
    ):
        """
        Calculate the calendar returns of every ticker: the return of every month with the years as the rows
        and the months as the columns, and the return of the whole year, the monthly returns table of fund
        factsheets.

        The formula is as follows:

        - Month Return = ((1 + Daily Return 1) x (1 + Daily Return 2) x ... x (1 + Daily Return N)) - 1
        - Year Return = ((1 + Month Return 1) x ... x (1 + Month Return 12)) - 1

        A month without any return (before the start date or a listing, or after the end date) is NaN, and the
        Year compounds the months it has, so a partial first or last year is the return over the months
        covered. The returns use the dividend-adjusted close, so they are total returns.


        Also known as: monthly returns table, return table, calendar year returns.

        Shares similar logic:
        - quantstats, monthly_returns: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L3190
        - ffn, PerformanceStats (return_table): https://github.com/pmorissette/ffn/blob/v1.4.1/ffn/core.py#L123
        - PerformanceAnalytics (R), table.CalendarReturns: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/table.CalendarReturns.R#L41

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The years as the index and (ticker, month) as the columns, the months Jan to Dec
            followed by Year.

        Notes:
        - quantstats fills a month without returns with 0 and ffn's return table does too; the Toolkit leaves
        it NaN. The months and years otherwise agree with quantstats' monthly_returns (compounded) and the Year
        column equals its EOY, and the monthly and yearly `get_returns`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.performance.get_calendar_returns()["AAPL"].T
        ```

        Which returns:

        |      |    2021 |    2022 |    2023 |    2024 |    2025 |
        |:-----|--------:|--------:|--------:|--------:|--------:|
        | Jan  |  0.0197 | -0.0157 |  0.1105 | -0.0422 | -0.0576 |
        | Feb  | -0.0797 | -0.0541 |  0.0232 | -0.0185 |  0.0259 |
        | Mar  |  0.0073 |  0.0575 |  0.1186 | -0.0513 | -0.0815 |
        | Apr  |  0.0762 | -0.0971 |  0.029  | -0.0067 | -0.0434 |
        | May  | -0.0505 | -0.0545 |  0.0461 |  0.1302 | -0.0536 |
        | Jun  |  0.0991 | -0.0814 |  0.0943 |  0.0956 |  0.0215 |
        | Jul  |  0.065  |  0.1886 |  0.0128 |  0.0544 |  0.0117 |
        | Aug  |  0.0425 | -0.0312 | -0.0424 |  0.0324 |  0.1196 |
        | Sep  | -0.068  | -0.121  | -0.0887 |  0.0175 |  0.0969 |
        | Oct  |  0.0587 |  0.1096 | -0.0026 | -0.0304 |  0.0618 |
        | Nov  |  0.1051 | -0.033  |  0.1138 |  0.0517 |  0.0324 |
        | Dec  |  0.0742 | -0.1223 |  0.0136 |  0.0552 | -0.0251 |
        | Year |  0.3806 | -0.264  |  0.4901 |  0.3071 |  0.0905 |
        """
        returns = (
            self._historical_data["daily"]
            .loc[:, "Return"]
            .loc[self._start_date : self._end_date]
        )

        calendar_returns = performance_model.get_calendar_returns(returns)

        return apply_rounding(
            calendar_returns, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_bootstrap_statistics(
        self,
        n_bootstrap: int = 1000,
        method: str = "resample",
        seed: int | None = 0,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate how much the performance statistics of the history could have differed by chance: the
        distribution of the cumulative and annualized return, the volatility, the Sharpe ratio and the
        maximum drawdown over simulated histories drawn from the daily returns (a bootstrap).

        Every simulation draws a history of as many daily returns as the date range holds:

        - Resampling draws the returns with replacement (Efron's bootstrap), so a day can appear
          several times or not at all and every statistic varies across resamples, as pyfolio does.
        - Shuffling puts the same returns in a random order (a permutation), as quantstats' Monte Carlo
          does: the cumulative return, the volatility and the Sharpe ratio stay those of the history and only
          the path, so the maximum drawdown, varies. It shows how much of a drawdown was the order of the
          returns rather than their size.

        For every simulation:

        - Cumulative Return = PRODUCT(1 + Return) - 1
        - Annualized Return = (1 + Cumulative Return) ^ (P / N) - 1
        - Annualized Volatility = STDEV(Return) x SQRT(P)
        - Sharpe Ratio = MEAN(Excess Return) / STDEV(Excess Return) x SQRT(P)
        - Maximum Drawdown = MIN(Wealth / Running Peak - 1), the wealth starting at 1

        Where N is the number of daily returns and P the daily returns in a year (252, or 365 for a ticker that
        trades on weekends such as a cryptocurrency). The excess return of each day (the return minus the
        daily risk-free rate) is drawn together with the return. Each statistic is summarized by the mean, the
        median, the standard deviation and the 5th, 25th, 75th and 95th percentiles over the resamples; the
        5th to 95th percentile is a 90% bootstrap interval.

        For more information, see the following source:

        - Efron, B. (1979). "Bootstrap Methods: Another Look at the Jackknife." The Annals of Statistics, 7(1),
        1-26.

        Also known as: Monte Carlo simulation, bootstrapped performance statistics, resampled returns.

        Shares similar logic:
        - pyfolio, perf_stats_bootstrap: https://github.com/quantopian/pyfolio/blob/0.9.2/pyfolio/timeseries.py#L742
        - quantstats, montecarlo: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L3428
        - ffn, resample_returns: https://github.com/pmorissette/ffn/blob/v1.4.1/ffn/core.py#L3293

        Args:
            n_bootstrap (int, optional): The number of bootstrap resamples (simulated histories). Defaults to 1000.
            method (str, optional): The bootstrap. Options are: "resample" (draws with replacement),
                "shuffle" (a random order without replacement). Defaults to "resample".
            seed (int | None, optional): The seed of the random numbers, so that results can be reproduced;
                None draws new ones every time. Defaults to 0.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The statistics as the index and (ticker, summary) as the columns, the summaries being
            Mean, Median, Std, 5%, 25%, 75% and 95%.

        Notes:
        - Every ticker draws from its own daily returns with the same seed, so a ticker's result does not
        depend on the other tickers in the Toolkit.
        - The simple bootstrap treats the returns as independent: it keeps their distribution but not their
        autocorrelation or volatility clustering.
        - pyfolio reports the mean, median and 5th and 95th percentiles of the same statistics (its standard deviation
        with n in the denominator). quantstats keeps the history itself as one of the paths, counts 252 returns to a
        year and computes the Sharpe ratio of its paths from their percentage changes, which leaves out the first
        return; on the same paths the statistics otherwise agree.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.performance.get_bootstrap_statistics()["AAPL"]
        ```

        Which returns:

        |                       |    Mean |   Median |    Std |      5% |     25% |     75% |     95% |
        |:----------------------|--------:|---------:|-------:|--------:|--------:|--------:|--------:|
        | Cumulative Return     |  1.5276 |   1.1649 | 1.5579 | -0.1818 |  0.4564 |  2.1464 |  4.4302 |
        | Annualized Return     |  0.1738 |   0.1679 | 0.135  | -0.0395 |  0.0785 |  0.259  |  0.405  |
        | Annualized Volatility |  0.278  |   0.2777 | 0.0113 |  0.2603 |  0.2703 |  0.2857 |  0.2961 |
        | Sharpe Ratio          |  0.571  |   0.5736 | 0.4096 | -0.1303 |  0.2915 |  0.8507 |  1.2197 |
        | Maximum Drawdown      | -0.3857 |  -0.373  | 0.1035 | -0.5737 | -0.4507 | -0.3084 | -0.2412 |
        """
        returns = self._get_column("daily", "Return", within_period=False).loc[
            self._start_date : self._end_date
        ]
        excess_returns = returns - self._risk_free_rate_data["daily"][
            returns.columns
        ].reindex(returns.index)
        # A Series, as the returns are a DataFrame.
        observations_per_year = pd.Series(get_observations_per_year(returns))

        bootstrap_statistics = pd.concat(
            {
                ticker: performance_model.get_bootstrap_statistics(
                    returns[ticker],
                    excess_returns[ticker],
                    float(observations_per_year[ticker]),
                    n_bootstrap=n_bootstrap,
                    method=method,
                    seed=seed,
                )
                for ticker in returns.columns
            },
            axis=1,
        )

        return apply_rounding(
            bootstrap_statistics, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_forecast_cone(
        self,
        horizon: int = 252,
        standard_deviations: list[float] | None = None,
        simulations: int = 1000,
        seed: int | None = 0,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate a forecast cone of the growth of 1 over the coming trading days, bootstrapped from the daily
        returns: the mean path and the bounds a number of standard deviations above and below it, against
        which the performance after the date range can be judged.

        Every simulation draws `horizon` daily returns with replacement from the returns in the date range
        and compounds them into a path of wealth that starts at 1:

        - Wealth_t = PRODUCT(1 + Simulated Return_1 ... Simulated Return_t)
        - Mean_t = MEAN(Wealth_t) over the simulations
        - Upper (Lower) k_t = Mean_t + (-) k x STDEV(Wealth_t)

        A strategy whose live performance leaves the cone (e.g. below the lower 2 standard deviation bound)
        behaves differently from its history. The cone does not assume normally distributed returns, but it
        does assume that the coming returns are drawn independently from the same distribution as the past
        ones.

        The bootstrap forecast cone is the one pyfolio (Quantopian) draws in its tear sheets.

        For more information, see the following source:

        - Efron, B. (1979). "Bootstrap Methods: Another Look at the Jackknife." The Annals of Statistics, 7(1),
        1-26.

        Also known as: bootstrap cone, probability cone, Bayesian cone (pyfolio's tear sheet).

        Referenced against:
        - pyfolio, summarize_paths: https://github.com/quantopian/pyfolio/blob/0.9.2/pyfolio/timeseries.py#L1111

        Shares similar logic:
        - pyfolio, forecast_cone_bootstrap: https://github.com/quantopian/pyfolio/blob/0.9.2/pyfolio/timeseries.py#L1147

        Args:
            horizon (int, optional): The number of trading days to forecast. Defaults to 252.
            standard_deviations (list[float] | None, optional): The numbers of standard deviations of the
                bounds. Defaults to None, which uses 1, 1.5 and 2.
            simulations (int, optional): The number of simulated paths. Defaults to 1000.
            seed (int | None, optional): The seed of the random numbers, so that results can be reproduced;
                None draws new ones every time. Defaults to 0.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The trading days ahead (1 to horizon) as the index and (ticker, bound) as the columns,
            the bounds being Lower 2, Lower 1.5, Lower 1, Mean, Upper 1, Upper 1.5 and Upper 2 by default.

        Notes:
        - It agrees with pyfolio's forecast_cone_bootstrap on the same simulated paths, apart from the standard
        deviation over the simulations, which pyfolio takes with n and the Toolkit with n - 1 in the denominator.
        - Every ticker draws from its own daily returns with the same seed, so a ticker's result does not
        depend on the other tickers in the Toolkit.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.performance.get_forecast_cone()["AAPL"].iloc[[0, 20, 62, 125, 251]]
        ```

        Which returns:

        |     |   Lower 2 |   Lower 1.5 |   Lower 1 |   Mean |   Upper 1 |   Upper 1.5 |   Upper 2 |
        |----:|----------:|------------:|----------:|-------:|----------:|------------:|----------:|
        |   1 |    0.9675 |      0.9757 |    0.9839 | 1.0003 |    1.0168 |      1.025  |    1.0332 |
        |  21 |    0.8545 |      0.8947 |    0.9349 | 1.0152 |    1.0955 |      1.1357 |    1.1759 |
        |  63 |    0.7578 |      0.8304 |    0.903  | 1.0483 |    1.1935 |      1.2661 |    1.3387 |
        | 126 |    0.6692 |      0.7766 |    0.8839 | 1.0986 |    1.3132 |      1.4206 |    1.5279 |
        | 252 |    0.5356 |      0.7019 |    0.8683 | 1.201  |    1.5336 |      1.7    |    1.8663 |
        """
        returns = self._get_column("daily", "Return", within_period=False).loc[
            self._start_date : self._end_date
        ]

        forecast_cone = pd.concat(
            {
                ticker: performance_model.get_forecast_cone(
                    returns[ticker],
                    horizon=horizon,
                    standard_deviations=standard_deviations,
                    simulations=simulations,
                    seed=seed,
                )
                for ticker in returns.columns
            },
            axis=1,
        )

        return apply_rounding(
            forecast_cone, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_probability_of_backtest_overfitting(
        self,
        blocks: int = 16,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Probability of Backtest Overfitting (PBO) of picking the ticker with the best Sharpe
        ratio: how often the ticker that looks best on one half of the history ends up in the bottom half of
        the tickers on the other half, with the Combinatorially Symmetric Cross-Validation (CSCV) of Bailey,
        Borwein, López de Prado and Zhu (2017).

        The tickers are the candidate strategies, for instance funds to choose from or the variants of a
        strategy (such as the Portfolio of every weighting). The daily returns in the date range are split
        into `blocks` blocks of equal length and for every way to pick half of the blocks as the in-sample
        history, with the other half out of sample:

        - The ticker with the highest in-sample Sharpe ratio (mean / standard deviation of the daily returns)
          is selected.
        - ω = Rank of its out-of-sample Sharpe ratio among the N tickers / (N + 1)
        - λ = ln(ω / (1 - ω))
        - PBO = Share of the combinations with λ <= 0

        A PBO close to 0 means the ticker that looked best kept doing well out of sample; around 0.5 or above,
        selecting on the past Sharpe ratio adds nothing and the best backtest is likely overfit. The table also
        gives the Average Out-of-Sample Rank (the mean ω, 0.5 for a random choice), the Probability of Loss (the
        share of combinations in which the selected ticker has a negative out-of-sample Sharpe ratio), the
        Performance Degradation (the slope of the regression of the selected ticker's out-of-sample Sharpe ratio
        on its in-sample Sharpe ratio, negative when a high in-sample Sharpe ratio is followed by a low one) and
        the number of Combinations, C(blocks, blocks / 2).

        For more information, see the following source:

        - Bailey, D.H., Borwein, J.M., López de Prado, M., & Zhu, Q.J. (2017). "The Probability of Backtest
        Overfitting." Journal of Computational Finance, 20(4), 39-69.

        Also known as: PBO, CSCV, combinatorially symmetric cross-validation.

        Referenced against:
        - ffn, calc_prob_backtest_overfitting: https://github.com/pmorissette/ffn/blob/v1.4.1/ffn/core.py#L3165

        Shares similar logic:
        - RiskLabAI, probability_of_backtest_overfitting: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/backtest/probability_of_backtest_overfitting.py#L80

        Args:
            blocks (int, optional): The number of blocks the history is split into, even and at least 2. 16 gives
                12,870 combinations, as in the paper. Defaults to 16.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.Series: The Probability of Backtest Overfitting, Average Out-of-Sample Rank, Probability of Loss,
            Performance Degradation and Combinations.

        Notes:
        - It needs at least two tickers; the Benchmark is not one of the candidates. The more tickers, the more
        informative the PBO: with two, the selected one is either the best or the worst out of sample.
        - The Probability of Backtest Overfitting and the Average Out-of-Sample Rank agree with ffn's
        calc_prob_backtest_overfitting with the same number of blocks. The daily returns beyond a multiple of
        `blocks` are left out from the end, as in ffn.
        - The Sharpe ratios skip missing returns, so a ticker that trades on other days than the rest (such as
        a cryptocurrency next to stocks) is measured on its own days within each block.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "KO"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2016-01-01",
            end_date="2025-12-31",
        )

        toolkit.performance.get_probability_of_backtest_overfitting()
        ```

        Which returns:

        |                                     |      Value |
        |:------------------------------------|-----------:|
        | Probability of Backtest Overfitting |     0.2035 |
        | Average Out-of-Sample Rank          |     0.7193 |
        | Probability of Loss                 |     0.0037 |
        | Performance Degradation             |    -0.5346 |
        | Combinations                        | 12870      |
        """
        returns = self._get_column("daily", "Return", within_period=False).loc[
            self._start_date : self._end_date
        ]
        # The first date of the range has no return; dates no ticker traded hold no return either. The Portfolio
        # combines the tickers, so it is no candidate strategy of its own.
        returns = returns.drop(columns="Portfolio", errors="ignore").dropna(how="all")

        probability_of_backtest_overfitting = (
            performance_model.get_probability_of_backtest_overfitting(returns, blocks)
        )

        rounding = rounding if rounding is not None else self._rounding

        return (
            probability_of_backtest_overfitting.round(rounding)
            if rounding is not None
            else probability_of_backtest_overfitting
        )

    @handle_errors
    def get_smoothing_index(
        self,
        period: str = "monthly",
        order: int = 2,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Smoothing Index of Getmansky, Lo and Makarov (2004), how much the returns look smoothed:
        the reported return of an illiquid asset (a hedge fund, private equity, real estate or a stale-priced
        bond fund) carries part of the returns of earlier periods, as if it were a weighted average of the
        true returns.

        The formula is as follows:

        - Observed Return_t = θ0 x True Return_t + θ1 x True Return_t-1 + θ2 x True Return_t-2
        - θ0 + θ1 + θ2 = 1
        - Smoothing Index ξ = θ0² + θ1² + θ2²

        The weights θ are those of a moving average model (MA(2) by default) fitted by exact maximum likelihood
        to the de-meaned returns of `period` over the date range, normalized to add up to 1. A Smoothing Index
        of 1 (θ0 = 1) means the returns show no smoothing; the lower it is, the more of each return is carried
        over from earlier periods, down to 1 / 3 when the three periods weigh the same. Smoothed returns
        understate the volatility and the correlation with the market and so overstate the Sharpe ratio.

        For more information, see the following source:

        - Getmansky, M., Lo, A.W., & Makarov, I. (2004). "An Econometric Model of Serial Correlation and
        Illiquidity in Hedge Fund Returns." Journal of Financial Economics, 74(3), 529-609.

        Also known as: Getmansky-Lo-Makarov smoothing index, return smoothing, serial correlation of returns.

        Referenced against:
        - statsmodels, ARIMA (order (0, 0, k), without a mean): https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html

        Shares similar logic:
        - PerformanceAnalytics (R), SmoothingIndex: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/SmoothingIndex.R#L96

        Args:
            period (str, optional): The period of the returns. Options are: "weekly", "monthly", "quarterly",
                "yearly". The paper uses monthly returns. Defaults to "monthly".
            order (int, optional): The number of earlier periods a return carries, k in MA(k). Defaults to 2.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The weights Theta 0 to Theta k and the Smoothing Index as the index, the tickers as the
            columns.

        Notes:
        - Negative moving average coefficients are set to 0 before the weights are normalized, as
        PerformanceAnalytics does by default, since a negative weight on a past return is no smoothing.
        - PerformanceAnalytics fits the model without a mean to the returns as given, where Getmansky, Lo and
        Makarov de-mean them first, as the Toolkit does. On the same (raw) returns the Toolkit's maximum
        likelihood estimate equals that of statsmodels' ARIMA to 3e-8 and PerformanceAnalytics' recorded
        examples to 6e-5, the tolerance of R's optimizer.
        - Liquid stocks have a Smoothing Index close to 1. `get_returns` with unsmoothing removes first order
        smoothing from the returns (Geltner).

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "VNQ", "HYG", "PSP"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2016-01-01",
            end_date="2025-12-31",
        )

        toolkit.performance.get_smoothing_index()
        ```

        Which returns:

        |                 |   AAPL |   VNQ |    HYG |   PSP |
        |:----------------|-------:|------:|-------:|------:|
        | Theta 0         | 0.9646 |     1 | 0.9708 |     1 |
        | Theta 1         | 0.0354 |     0 | 0      |     0 |
        | Theta 2         | 0      |     0 | 0.0292 |     0 |
        | Smoothing Index | 0.9318 |     1 | 0.9433 |     1 |
        """
        validate_period(period)

        returns = (
            self._get_column(period, "Return", within_period=False)
            .loc[self._start_date : self._end_date]
            .dropna(how="all")
        )

        smoothing_index = pd.DataFrame(
            {
                ticker: performance_model.get_smoothing_index(
                    returns[ticker] - returns[ticker].mean(), order
                )
                for ticker in returns.columns
            }
        )

        return apply_rounding(
            smoothing_index, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_variability_weighted_return(
        self,
        period: str | None = None,
        within_period: bool = True,
        tau: float = 2.0,
        maximum_deviation: float = 0.2,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Variability-Weighted Return (VWR) of an investment portfolio or asset: the
        annualized return reduced by how far the value strays from a steady exponential path with
        the same overall growth. Two assets with the same return score alike only when they got
        there alike; the one that swung more keeps less of its return, and none of it once the
        standard deviation of its deviations from the steady path reaches `maximum_deviation`.
        Unlike the Sharpe ratio it stays a return, and it judges the path of the value rather
        than the spread of the single returns.

        The formula is as follows:

        - r̄ = Average of ln(1 + Return), the average log return
        - Annualized Return = exp(r̄ x P) - 1
        - Deviation_i = V_i / (V_0 x exp(r̄ x i)) - 1, for i = 1..N, with V the growth of 1
        - VWR = Annualized Return x (1 - (Std(Deviation) / maximum_deviation) ^ tau)

        Where P is the number of returns in a year (252 daily returns, 365 for a ticker that
        trades on weekends such as a cryptocurrency, or 52, 12, 4 or 1 for weekly, monthly,
        quarterly or yearly returns), N the number of returns and Std the sample standard
        deviation (with n - 1 in the denominator). The Annualized Return compounds continuously from the average log
        return, which equals the compounded annual growth over the returns.


        Also known as: Variability-Weighted Return, VWR, log return Sharpe alternative.

        Shares similar logic:
        - backtrader, VWR: https://github.com/backtrader/backtrader/blob/1.9.74.123/backtrader/analyzers/vwr.py#L32

        Args:
            period (str, optional): The period to use for the calculation. Options are: "daily", "weekly",
                "monthly", "quarterly", "yearly". Defaults to "quarterly" if the Toolkit is initialised with
                quarterly=True, otherwise "yearly".
            within_period (bool, optional): Whether to calculate the VWR from the daily returns within each
                period (e.g. each year if period = 'yearly') or from the returns of the periods themselves over
                the whole history. Defaults to True.
            tau (float, optional): The exponent of the penalty for variability: higher values spare small
                deviations and punish large ones. Defaults to 2.0.
            maximum_deviation (float, optional): The standard deviation of the deviations from the steady path
                at which the return is reduced to zero, as a decimal. Defaults to 0.2.
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
            pd.DataFrame | pd.Series: The VWR per period and ticker as a decimal, or over the whole history
            one value per ticker.

        Notes:
        - backtrader documents tau = 2 and a maximum deviation of 0.20 (as the crystalbull.com article), but its
        parameters set tau = 0.20 and sdev_max = 2.0, which makes the penalty (s / 2)^0.2 and leaves far less of
        the return than the documented definition. The defaults here are the documented ones; a tau of 0.2
        and a maximum deviation of 2.0 give backtrader's default output, which backtrader reports in percent.
        - With a maximum deviation that the deviations exceed, the VWR changes sign: a gain turns into a
        negative value, so compare it only between assets of similar variability or raise `maximum_deviation`.
        - The Portfolio of a Portfolio's Toolkit is measured on its own daily return series, as a weighted
        average of the holdings' VWR is not the VWR of the portfolio.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "KO"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.performance.get_variability_weighted_return()
        ```

        Which returns:

        |      |    AAPL |    MSFT |      KO |
        |:-----|--------:|--------:|--------:|
        | 2021 |  0.3558 |  0.5444 |  0.1549 |
        | 2022 | -0.2281 | -0.2632 |  0.1007 |
        | 2023 |  0.4289 |  0.5329 | -0.0434 |
        | 2024 |  0.2772 |  0.1237 |  0.0816 |
        | 2025 |  0.0731 |  0.1284 |  0.1487 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        returns = self._get_column(period, "Return", within_period=within_period)
        periods_per_year = self._get_ratio_observations_per_year(
            period, rolling=None if within_period else 1, columns=returns.columns
        )

        variability_weighted_return = performance_model.get_variability_weighted_return(
            (
                returns
                if within_period
                else returns.loc[self._start_date : self._end_date]
            ),
            periods_per_year,
            tau,
            maximum_deviation,
        )

        return finalize_dataset(
            dataset=variability_weighted_return,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            apply_slice=within_period,
        )

    @handle_errors
    def get_return_concentration(
        self,
        period: str | None = None,
        within_period: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the concentration of the returns of an investment portfolio or asset, as López de Prado
        measures a strategy's bets: how much of the gains come from a few returns, how much of the losses
        do, and how much the returns cluster in a few months. Each is a normalized Herfindahl-Hirschman Index
        between 0, where every return contributes the same, and 1, where one return holds everything. A
        strategy whose gains rest on a handful of days is more fragile than one that earns them steadily,
        even with the same total.

        The formula is as follows:

        - h = (SUM w_j² - 1 / N) / (1 - 1 / N), with w_j = x_j / SUM x over N values
        - Positive Concentration = h of the returns of 0 or more
        - Negative Concentration = h of the returns below 0
        - Time Concentration = h of the number of returns in each calendar month

        For more information, see the following sources:

        - López de Prado, M. (2018). "Advances in Financial Machine Learning." Wiley, section 14.5.1
        (Snippet 14.3).
        - Herfindahl, O.C. (1950). "Concentration in the Steel Industry." Columbia University.

        Also known as: bets concentration, HHI of returns, Herfindahl-Hirschman Index of returns.

        Args:
            period (str, optional): The period of the returns over the whole history, or with
                within_period=True the period the daily returns are grouped by. Options are: "daily",
                "weekly", "monthly", "quarterly", "yearly". Defaults to "daily",
                or with within_period=True to "quarterly" if the Toolkit is initialised with
                quarterly=True, otherwise "yearly".
            within_period (bool, optional): Whether to calculate the concentration from the daily returns
                within each period (e.g. each year if period = 'yearly') instead of from the returns of
                `period` over the whole history. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Over the whole history, the concentrations as the index and the tickers as the
            columns. Within periods, the periods as the index and (concentration, ticker) as the columns.

        Notes:
        - A return of exactly 0 counts with the positive returns, as in López de Prado's Snippet 14.3.
        - Every return is a bet, so the Time Concentration of an asset held every day is close to 0 (the
        months differ only in their number of trading days); it tells more about a strategy's returns, where
        the days out of the market are left out of the returns. With weekly or longer returns over the whole
        history, few months hold more than one return.
        - A concentration needs at least three values and is NaN otherwise.
        - The Portfolio of a Portfolio's Toolkit is measured on its own daily return series.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "KO"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.performance.get_return_concentration()
        ```

        Which returns:

        |                        |   AAPL |   MSFT |     KO |
        |:-----------------------|-------:|-------:|-------:|
        | Positive Concentration | 0.0015 | 0.0014 | 0.0012 |
        | Negative Concentration | 0.0015 | 0.0015 | 0.0015 |
        | Time Concentration     | 0.0001 | 0.0001 | 0.0001 |
        """
        if not period:
            period = (
                ("quarterly" if self._quarterly else "yearly")
                if within_period
                else "daily"
            )

        returns = self._get_column(period, "Return", within_period=within_period)
        if not within_period:
            returns = returns.loc[self._start_date : self._end_date]

        concentration = performance_model.get_backtest_statistics_table(
            returns, performance_model.get_return_concentration
        )

        return apply_rounding(
            concentration, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_strategy_risk(
        self,
        period: str | None = None,
        within_period: bool = False,
        target_sharpe_ratio: float = 2.0,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the strategy risk of López de Prado for an investment portfolio or asset: its returns read as
        bets that either win the average gain or lose the average loss, the Sharpe ratio such a strategy has,
        the precision (share of winning bets) and the number of bets a year it needs to reach a target Sharpe
        ratio, and the probability that its true precision falls short of the one needed, the probability of
        failure. Unlike the volatility, which measures the risk of the asset, this measures the risk that the
        strategy does not deliver its target: a small drop of the precision can take a strategy with a high
        target from success to failure.

        The formula is as follows:

        - Precision p = Number of Returns above 0 / Number of Returns
        - Average Gain π+ = Average of the Returns above 0, Average Loss π- = Average of the Returns of 0 or below
        - Binomial Sharpe Ratio = ((π+ - π-) x p + π-) / ((π+ - π-) x SQRT(p x (1 - p))) x SQRT(n)
        - Implied Precision p* = (-b + SQRT(b² - 4ac)) / (2a), with a = (n + θ²) x (π+ - π-)²,
          b = (2 x n x π- - θ² x (π+ - π-)) x (π+ - π-) and c = n x π-²
        - Implied Frequency n* = θ² x (π+ - π-)² x p x (1 - p) / ((π+ - π-) x p + π-)²
        - Probability of Failure = N((p* - p) / SQRT(p x (1 - p) / N))

        Where n is the number of bets a year (252 daily returns, 365 for a ticker that trades on weekends such as
        a cryptocurrency, or 52, 12, 4 or 1 for weekly, monthly, quarterly or yearly returns), θ the target Sharpe
        ratio, N the number of returns and N(.) the standard normal distribution function. The Implied Precision
        is the precision that reaches θ at the measured gains, losses and frequency, the Implied Frequency the
        number of bets a year that reaches θ at the measured precision, gains and losses.

        For more information, see the following sources:

        - López de Prado, M. (2018). "Advances in Financial Machine Learning." Wiley, chapter 15 (Snippets 15.1
        to 15.5).
        - Bailey, D.H. and López de Prado, M. (2014). "The Sharpe Ratio Efficient Frontier." Journal of Risk,
        15(2), 3-44.

        Also known as: implied precision, implied betting frequency, binomial Sharpe ratio, probability of
        strategy failure.

        Shares similar logic:
        - RiskLabAI, implied_precision: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/backtest/strategy_risk.py#L82
        - RiskLabAI, bin_frequency: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/backtest/strategy_risk.py#L134
        - RiskLabAI, binomial_sharpe_ratio: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/backtest/strategy_risk.py#L193
        - RiskLabAI, failure_probability: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/backtest/strategy_risk.py#L281

        Args:
            period (str, optional): The period of the returns over the whole history, or with
                within_period=True the period the daily returns are grouped by. Options are: "daily",
                "weekly", "monthly", "quarterly", "yearly". Defaults to "daily",
                or with within_period=True to "quarterly" if the Toolkit is initialised with
                quarterly=True, otherwise "yearly".
            within_period (bool, optional): Whether to calculate the strategy risk from the daily returns
                within each period (e.g. each year if period = 'yearly') instead of from the returns of
                `period` over the whole history. Defaults to False.
            target_sharpe_ratio (float, optional): The annualized Sharpe ratio the strategy should reach.
                Defaults to 2.0, as López de Prado.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Over the whole history, the Precision, Average Gain, Average Loss, Bets per Year,
            Binomial Sharpe Ratio, Implied Precision, Implied Frequency and Probability of Failure as the index
            and the tickers as the columns. Within periods, the periods as the index and (measure, ticker) as the
            columns.

        Notes:
        - A return of exactly 0 counts as a losing bet, as in López de Prado's Snippet 15.5, so every bet either
        wins or loses.
        - The probability of failure compares the implied precision with the distribution of the measured
        precision over the N bets, whose standard error is SQRT(p x (1 - p) / N), the spread of its bootstrap
        distribution. López de Prado's Snippet 15.5 approximates it with a normal distribution with a standard
        deviation of p x (1 - p), which does not shrink with the number of bets; RiskLabAI uses the standard error
        but returns N((p - p*) / SE), the probability of success, and solves the implied precision with the loss
        as a positive number.
        - The Implied Precision is NaN when no precision reaches the target with these gains and losses, the
        Implied Frequency when the average bet loses money (or makes money for a negative target).
        - The Binomial Sharpe Ratio has no risk free rate, as López de Prado's.
        - The Portfolio of a Portfolio's Toolkit is measured on its own daily return series.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "KO"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.performance.get_strategy_risk()
        ```

        Which returns:

        |                        |      AAPL |     MSFT |        KO |
        |:-----------------------|----------:|---------:|----------:|
        | Precision              |    0.5303 |   0.5239 |    0.5279 |
        | Average Gain           |    0.0125 |   0.012  |    0.0073 |
        | Average Loss           |   -0.0125 |  -0.0115 |   -0.0074 |
        | Bets per Year          |  252      | 252      |  252      |
        | Binomial Sharpe Ratio  |    0.9746 |   1.0813 |    0.8521 |
        | Implied Precision      |    0.5622 |   0.5525 |    0.5636 |
        | Implied Frequency      | 1061.26   | 862.048  | 1388.41   |
        | Probability of Failure |    0.9881 |   0.9788 |    0.9943 |
        """
        if not period:
            period = (
                ("quarterly" if self._quarterly else "yearly")
                if within_period
                else "daily"
            )

        returns = self._get_column(period, "Return", within_period=within_period)
        if not within_period:
            returns = returns.loc[self._start_date : self._end_date]

        if returns.dropna(how="all").empty:
            logger.warning(
                "There are no %s returns between %s and %s, so the strategy risk is empty.",  # noqa: E501
                period,
                self._start_date,
                self._end_date,
            )
            return pd.DataFrame()

        # The bets in a year: the daily returns within each period, or the returns of `period`
        # over the whole history (the daily returns of each ticker's own calendar).
        bets_per_year = self._get_ratio_observations_per_year(
            period, rolling=None if within_period else 1, columns=returns.columns
        )

        strategy_risk = performance_model.get_backtest_statistics_table(
            returns,
            performance_model.get_strategy_risk,
            frequency=bets_per_year,
            target_sharpe_ratio=target_sharpe_ratio,
        )

        return apply_rounding(
            strategy_risk, rounding if rounding is not None else self._rounding
        )

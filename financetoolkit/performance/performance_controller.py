"""Performance Module"""

__docformat__ = "google"

import warnings
from collections.abc import Callable
from functools import cached_property

import numpy as np
import pandas as pd

from financetoolkit.helpers import LazyDict, handle_portfolio
from financetoolkit.performance import performance_model
from financetoolkit.performance.helpers import determine_within_dataset
from financetoolkit.risk.helpers import (
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
    get_rolling_kurtosis,
    get_rolling_skewness,
    get_skewness,
    get_ui,
    get_volatility,
)
from financetoolkit.utilities import factor_model
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import handle_errors
from financetoolkit.utilities.logger_model import get_logger
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
        "get_jensens_alpha",
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
    def _tickers_without_portfolio(self) -> list[str]:
        """The tickers other than the Portfolio."""
        return [ticker for ticker in self._tickers if ticker != "Portfolio"]

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

        daily_historical_data = self._historical_data["daily"]
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
                float(PERIODS_PER_YEAR["daily"]), index=self._tickers_without_portfolio
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
            raise ValueError(
                "Intraday data is required for within-period daily calculations. "
                "Initialise the Toolkit with an intraday_period to use period='daily' here."
            )

        return self._within_historical_data[period]

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
                self._benchmark_name if benchmark else self._tickers_without_portfolio
            ]

        return self._data_cache[key]

    @handle_errors
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
        Calculates and collects all performance metrics.

        Args:
            period (str, optional): The period to use for the calculation. Defaults to "quarterly" if the
                Toolkit is initialised with quarterly=True, otherwise "yearly".
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
            "Jensen's Alpha": self.get_jensens_alpha(
                period=period,
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
        method: str = "ols",
        lags: int = 1,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Beta, a measurement that assess the systematic risk of a stock or investment.

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

        Beta is read against 1, the benchmark's own beta: above 1 the asset tends to amplify market moves,
        below 1 to dampen them. Each period's Beta comes from the daily returns within that period, so a yearly
        figure rests on about 250 observations and can swing with a single turbulent month; a rolling window
        (for example period='monthly' with rolling=36) gives a steadier estimate for cost of capital work. A Beta
        close to zero can come from a low correlation with the market rather than from low volatility, so it is
        worth checking both before calling a stock defensive.

        See definition: https://en.wikipedia.org/wiki/Beta_(finance)

        Also known as: market sensitivity, systematic risk.

        Args:
            period (str, optional): The period to use for the calculation. Defaults to "quarterly" if the Toolkit is
                initialised with quarterly=True, otherwise "yearly".
            rolling (int, optional): The rolling window size to use for the calculation. If you select
                period = 'monthly' and set rolling to 12 you obtain the rolling 12-month Beta. Defaults to None.
            method (str, optional): The estimator of the Beta. Defaults to "ols". Choose from:
                - "ols": the covariance over the benchmark variance (the formula above).
                - "blume": the adjusted Beta of Bloomberg, 0.67 x OLS Beta + 0.33 (Blume, 1971, 1975).
                - "vasicek": the OLS Beta shrunk towards the mean Beta of the tickers by the precision of
                  each estimate (Vasicek, 1973).
                - "dimson": the sum of the slopes on the lagged, same and leading benchmark returns
                  (Dimson, 1979).
                - "scholes_williams": the lagged, same and leading slopes over the benchmark's
                  autocorrelation (Scholes and Williams, 1977).
                - "upside" or "downside": the OLS Beta on the days the benchmark rose or fell.
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
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.
        - The OLS Beta of a stock that trades thinly or reacts to the market a day late understates its market
        risk, because part of the reaction falls on the next day. Dimson and Scholes-Williams add the slopes on
        the neighbouring benchmark returns to correct for that; their leads and lags come from the full history,
        so the first day of a period uses the benchmark return of the day before.
        - Blume found that Betas move towards 1 between estimation windows by about a third. Vasicek shrinks
        each Beta in proportion to its uncertainty: the prior is the mean and sample variance (ddof=1) of the
        OLS Betas of the tickers in the same period, so it depends on which tickers are passed. A single
        ticker gets a prior mean of 1 (the market's Beta) with a variance of 0.25.
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
            beta = performance_model.get_rolling_beta(
                returns, benchmark_returns, rolling
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
            period (str, optional): The period whose returns are explained: "daily", "weekly", "monthly",
                "quarterly" or "yearly". Defaults to "daily".
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
        - The market cap is that of `get_market_cap` with show_daily=True: the traded split-adjusted Close
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
        CAPM, or the Capital Asset Pricing Model, is a financial model used to estimate the expected return
        on an investment, such as a stock or portfolio of stocks. It provides a framework for evaluating the
        risk and return trade-off of an asset or portfolio in relation to the overall market. CAPM is based
        on the following key components:

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

        Also known as: CAPM, expected return model.

        Args:
            period (str, optional): The period to use for the calculation. Defaults to "quarterly" if the Toolkit is
                initialised with quarterly=True, otherwise "yearly".
            rolling (int, optional): The rolling window size to use for the Beta component of the
            calculation. If set, Beta is estimated over a rolling window of this many periods across
            the full return history instead of per `period`. Defaults to None.
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
        - The risk-free rate is the Toolkit's risk_free_rate, the 10-year US Treasury yield by default. Many
        textbook definitions (Sharpe, 1994; the Ken French factors) use a short-term Treasury bill instead, which
        the Toolkit uses when initialised with risk_free_rate="13w".
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

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

        if rolling:
            returns = self._get_column(period, "Return", within_period=False)
            benchmark_returns = self._get_column(
                period, "Return", within_period=False, benchmark=True
            )

            beta = performance_model.get_rolling_beta(
                returns, benchmark_returns, rolling
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
        the factor dataset's RF) on the factors, see `performance_model.get_rolling_factor_regression`.

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
                ticker: performance_model.get_rolling_factor_regression(
                    returns[ticker] - factor_dataset["RF"],
                    factor_dataset[factors],
                    rolling,
                )
                for ticker in self._tickers_without_portfolio
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
        Calculates factor exposures for each asset.

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
            period (str, optional): The period to use for the calculation. Defaults to "quarterly" if the Toolkit is
                initialised with quarterly=True, otherwise "yearly".
            factors_to_calculate (list of str, optional): List of factors to calculate scores and residuals for.
                Defaults to ["Mkt-RF", "SMB", "HML", "RMW", "CMA"].
            region (str, optional): The region whose factors to use: "United States", "Developed",
                "Developed ex US", "Europe", "Japan", "Asia Pacific ex Japan" or "North America" (the
                factors of Ken French's library for that region's stocks). Every region's factors are
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
                raise ValueError(
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
        for ticker in self._tickers_without_portfolio:
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
            .reindex(self._tickers_without_portfolio, level=0, axis=1)
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
        Calculates factor correlations between each factor. This is useful to understand how correlated
        each factor is to each other. This is based off the Fama and French 5 Factor model which includes:

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
            period (str, optional): The period to use for the calculation. Defaults to "quarterly" if the Toolkit is
                initialised with quarterly=True, otherwise "yearly".
            factors_to_calculate (list of str, optional): List of factors to calculate scores and residuals for.
                Defaults to ["Mkt-RF", "SMB", "HML", "RMW", "CMA"].
            exclude_risk_free (bool, optional): Whether to exclude the risk-free rate from the results. Defaults to True.
            region (str, optional): The region whose factors to use: "United States", "Developed",
                "Developed ex US", "Europe", "Japan", "Asia Pacific ex Japan" or "North America" (the
                factors of Ken French's library for that region's stocks). Every region's factors are
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
                raise ValueError(
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
        Calculate Fama and French 5 Factor model scores and residuals for a set of financial assets.

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

        Also known as: Fama-French model, three-factor model, five-factor model, FF3, FF5.

        Args:
            period (str, optional): The period for the calculation (e.g., "weekly", "monthly", "quarterly", "yearly").
                Defaults to "quarterly" if the Toolkit is initialised with quarterly=True, otherwise "yearly".
            method (str, optional): The regression method to use for the calculation. Defaults to 'multi'.
            factors_to_calculate (list of str, optional): List of factors to calculate scores and residuals for.
                Defaults to ["Mkt-RF", "SMB", "HML", "RMW", "CMA"].
            include_daily_residuals (bool, optional): Whether to also return the pointwise (daily)
                regression residuals as a second DataFrame. Defaults to False.
            region (str, optional): The region whose factors to use: "United States", "Developed",
                "Developed ex US", "Europe", "Japan", "Asia Pacific ex Japan" or "North America" (the
                factors of Ken French's library for that region's stocks). Every region's factors are
                U.S. dollar returns with the U.S. one-month Treasury bill as the risk-free rate (RF),
                so a ticker quoted in another currency is best analysed with its U.S. dollar returns;
                otherwise the exchange rate's moves end up in the factor loadings, mostly the market
                beta. Defaults to "United States".
            rolling (int | None, optional): The number of trading days of a rolling regression: the multi
                linear regression is fitted on the last `rolling` daily observations at every date instead of
                once per period, so the factor loadings show how the exposures move over time (e.g. 126 for
                the six months pyfolio uses). Only with method="multi". Defaults to None.
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
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

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
        if method not in ["simple", "multi"]:
            raise ValueError(
                f"Method {method} is not a valid method. Please select from: simple or multi (default). "
                "This refers to a simple linear regression (regression of each factor on each asset return series) "
                "or a multi linear regression (regression of all factors on each asset return series) respectively"
            )

        factors_to_calculate = (
            factors_to_calculate
            if factors_to_calculate
            else ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
        )

        for factor in factors_to_calculate:
            if factor not in ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]:
                raise ValueError(
                    f"Factor {factor} is not a valid factor. Please select from: Mkt-RF, SMB, HML, RMW, CMA."
                )

        if rolling:
            if method != "multi":
                raise ValueError("A rolling regression requires method='multi'.")

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
        for ticker in self._tickers_without_portfolio:
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
        Calculate Carhart Four Factor model scores for a set of financial assets.

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

        Args:
            period (str, optional): The period for the calculation (e.g., "weekly", "monthly", "quarterly", "yearly").
                Defaults to "quarterly" if the Toolkit is initialised with quarterly=True, otherwise "yearly".
            region (str, optional): The region whose factors to use: "United States", "Developed",
                "Developed ex US", "Europe", "Japan", "Asia Pacific ex Japan" or "North America" (the
                factors of Ken French's library for that region's stocks). Every region's factors are
                U.S. dollar returns with the U.S. one-month Treasury bill as the risk-free rate (RF),
                so a ticker quoted in another currency is best analysed with its U.S. dollar returns;
                otherwise the exchange rate's moves end up in the factor loadings, mostly the market
                beta. Defaults to "United States".
            rolling (int | None, optional): The number of trading days of a rolling regression: the regression
                is fitted on the last `rolling` daily observations at every date instead of once per period, so
                the factor loadings show how the exposures move over time (e.g. 126 for the six months pyfolio
                uses). Defaults to None.
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
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

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
        for ticker in self._tickers_without_portfolio:
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
        window: int | None = None,
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

        A window estimates the weights over a rolling number of periods instead, which shows how
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
            period (str, optional): The return frequency (daily, weekly, monthly, quarterly or
                yearly). Defaults to "monthly".
            window (int | None, optional): The number of periods to estimate the weights over on a
                rolling basis, which adds the Out-of-Sample Selection Return. Defaults to None,
                which uses every period.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the rounding of the Toolkit.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per ticker the weight of every style, the R Squared, the Selection Return
            and the Tracking Error, or with a window the same and the Out-of-Sample Selection
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
        if period not in PERIODS_PER_YEAR:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly or yearly."
            )
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
            for ticker in self._tickers_without_portfolio
            if ticker not in style_tickers
        ]
        if not funds:
            raise ValueError(
                "Every ticker is a style ticker, so there is nothing to analyze."
            )

        returns = returns.loc[
            self._start_date : self._end_date, [*funds, *style_tickers]
        ]
        # When the prices start within the range, their first period has no prior close and so no return for any ticker; leaving it in would give the first window one observation fewer than `window`.  # noqa: E501
        if (first_return := returns.first_valid_index()) is not None:
            returns = returns.loc[first_return:]
        periods_per_year = PERIODS_PER_YEAR[period]

        # A fund with too few returns (a recent listing, say) comes back as NaN in the model, so it does not empty the result of every other fund.  # noqa: E501
        if window is None:
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
                    fund: performance_model.get_rolling_style_analysis(
                        returns[fund], returns[style_tickers], window, periods_per_year
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
        method: str = "standard",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Alpha, in a general sense, represents the excess return an investment generates relative to
        a benchmark or a risk-adjusted return. It can be positive (indicating the investment
        outperformed the benchmark) or negative (indicating underperformance).

        The formula is as follows:

            - Alpha = Asset's Actual Return — Benchmark's Actual Return

        With method="annualized" it is the Active Premium (or annualized active return) of Bacon
        (2008) instead, one value per ticker over the whole date range:

            - Active Premium = Annualized Return of the Asset − Annualized Return of the Benchmark
            - Annualized Return = PRODUCT(1 + Daily Return) ^ (Observations per Year / N) − 1

        Where N is the number of daily returns and the observations per year are 252 (365 for a
        ticker that trades on weekends, such as a cryptocurrency). The benchmark is annualized over
        the dates from the first to the last return of each ticker, so a ticker listed later is
        compared over its own history.

        See definition: https://en.wikipedia.org/wiki/Alpha_(finance)

        For more information, see the following sources:

        - Bacon, C.R. (2008). "Practical Portfolio Performance Measurement and Attribution."
        2nd edition, John Wiley & Sons.
        - PerformanceAnalytics: ActivePremium (ActiveReturn) (https://github.com/braverock/PerformanceAnalytics).

        Also known as: excess return, outperformance, active return, active premium.

        Args:
            period (str, optional): The period to use for the calculation. Defaults to "quarterly" if the Toolkit is
                initialised with quarterly=True, otherwise "yearly".
            show_full_results (bool, optional): Whether to show the actual and benchmark returns next to the
                Alpha. Defaults to False.
            rolling (int, optional): The rolling window size to use for the calculation. If set,
            Alpha is calculated as the rolling mean excess return over this many periods across
            the full return history instead of per `period`. Defaults to None.
            method (str, optional): "standard" for the difference of the returns of each period, or
            "annualized" for the Active Premium, the difference of the annualized returns over the
            whole date range, one value per ticker (`period` and `rolling` are then not used).
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
            pd.DataFrame: Alpha values. With method="annualized" a Series with the Active Premium per
            ticker, or with show_full_results=True a DataFrame with the annualized return of each
            ticker and of the benchmark next to it.

        Notes:
        - The method retrieves historical data and calculates the Alpha for each asset in the Toolkit instance.
        - method="annualized" agrees with PerformanceAnalytics' ActivePremium on the daily returns (252 observations
        a year). PerformanceAnalytics compares on the dates both have a return; the Toolkit annualizes each on
        its own calendar within the ticker's history, so a cryptocurrency keeps its weekend returns.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

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
        if method not in ("standard", "annualized"):
            raise ValueError("Method must be standard or annualized.")

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
                # One number per ticker over the whole date range, with no dates to slice.
                apply_slice=False,
            )

        period = period if period else "quarterly" if self._quarterly else "yearly"
        returns = self._get_column(period, "Return", within_period=False)
        benchmark_returns = self._get_column(
            period, "Return", within_period=False, benchmark=True
        )

        if rolling:
            alpha = performance_model.get_rolling_alpha(
                returns, benchmark_returns, rolling
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
                [
                    f"Actual Return {ticker}"
                    for ticker in self._tickers_without_portfolio
                ]
                + ["Benchmark Returns"]
                + [f"Alpha {ticker}" for ticker in self._tickers_without_portfolio]
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
    def get_jensens_alpha(
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
        Calculate Jensen's Alpha, a measure of an asset's performance relative to its expected return
        based on the Capital Asset Pricing Model (CAPM).

        Jensen's Alpha is used to assess whether an investment has outperformed or underperformed
        its expected return given its systematic risk, as represented by the asset's Beta.

        The formula is as follows:

            - Jensen's Alpha = Asset's Actual Return — [Risk-Free Rate + Beta * (Benchmark Return — Risk-Free Rate)]

        See definition: https://en.wikipedia.org/wiki/Jensen%27s_alpha

        Also known as: Jensen alpha, risk-adjusted excess return.

        Args:
            period (str, optional): The period to use for the calculation. Defaults to "quarterly" if the Toolkit is
                initialised with quarterly=True, otherwise "yearly".
            rolling (int, optional): The rolling window size to use for the Beta component of the
            calculation. If set, Beta is estimated over a rolling window of this many periods across
            the full return history instead of per `period`. Defaults to None.
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
            pd.DataFrame: Jensen's Alpha values.

        Notes:
        - Daily Jensen's Alpha is not an option as the standard deviation for 1 day is close to zero. Therefore, it does
        not give any useful insights.
        - The method retrieves historical data and calculates the CAPM for each asset in the Toolkit instance.
        - The risk-free rate is the Toolkit's risk_free_rate, the 10-year US Treasury yield by default. Many
        textbook definitions (Sharpe, 1994; the Ken French factors) use a short-term Treasury bill instead, which
        the Toolkit uses when initialised with risk_free_rate="13w".
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["UNH", "JNJ", "PFE"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2020-01-01",
            end_date="2025-12-31",
        )

        toolkit.performance.get_jensens_alpha()
        ```

        Which returns:

        | Date   |     UNH |     JNJ |     PFE |
        |:-------|--------:|--------:|--------:|
        | 2021   |  0.2601 | -0.0092 |  0.6494 |
        | 2022   |  0.1608 |  0.0946 | -0.0263 |
        | 2023   | -0.095  | -0.2012 | -0.5535 |
        | 2024   | -0.0914 | -0.0806 | -0.1165 |
        | 2025   | -0.4154 |  0.4249 | -0.0975 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        if rolling:
            returns = self._get_column(period, "Return", within_period=False)
            benchmark_returns = self._get_column(
                period, "Return", within_period=False, benchmark=True
            )

            beta = performance_model.get_rolling_beta(
                returns, benchmark_returns, rolling
            )
        else:
            returns = self._get_column(period, "Return", within_period=True)
            benchmark_returns = self._get_column(
                period, "Return", within_period=True, benchmark=True
            )

            beta = performance_model.get_beta(returns, benchmark_returns)

        period_returns = self._get_column(period, "Return", within_period=False)

        risk_free_rate = self._risk_free_rate_data[period]
        benchmark_returns = self._get_column(
            period, "Return", within_period=False, benchmark=True
        )

        jensens_alpha = performance_model.get_jensens_alpha(
            period_returns, risk_free_rate, beta, benchmark_returns
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
        The Treynor Ratio, also known as Treynor's Measure or the Reward-to-Volatility Ratio, is a
        financial metric used to assess the risk-adjusted performance of an investment portfolio or
        asset. It measures the excess return generated by the portfolio per unit of systematic or market
        risk, often represented by Beta. The Treynor Ratio is a valuable tool for evaluating the
        performance of investments in relation to their market risk exposure.

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
        - PerformanceAnalytics: TreynorRatio (https://github.com/braverock/PerformanceAnalytics).

        Also known as: reward-to-volatility ratio.

        Args:
            period (str, optional): The period to use for the calculation. Defaults to "quarterly" if the Toolkit is
                initialised with quarterly=True, otherwise "yearly".
            rolling (int, optional): The rolling window size to use for the Beta component of the
            calculation. If set, Beta is estimated over a rolling window of this many periods across
            the full return history instead of per `period`. Defaults to None.
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
        - The risk-free rate is the Toolkit's risk_free_rate, the 10-year US Treasury yield by default. Many
        textbook definitions (Sharpe, 1994; the Ken French factors) use a short-term Treasury bill instead, which
        the Toolkit uses when initialised with risk_free_rate="13w".
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

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

        if rolling:
            returns = self._get_column(period, "Return", within_period=False)
            benchmark_returns = self._get_column(
                period, "Return", within_period=False, benchmark=True
            )

            beta = performance_model.get_rolling_beta(
                returns, benchmark_returns, rolling
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
        method: str = "standard",
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
        Calculate the Sharpe ratio, a measure of risk-adjusted return that evaluates the excess return
        of an investment portfolio or asset per unit of risk taken.

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
        risk_free_rate="13w" (the 13-week Treasury bill; "5y", "10y" and "30y" are the other
        options) to use a short rate instead.

        By default the result is **not annualized**: it is a per-observation Sharpe ratio, so a
        value computed from daily returns is roughly SQRT(252) smaller than the annualized figure
        usually quoted in the literature (SQRT(52), SQRT(12) and SQRT(4) for weekly, monthly
        and quarterly returns respectively). Set annualize=True to multiply by that factor, to
        compare against published annualized Sharpe ratios.

        The plain Sharpe ratio only looks at the mean and standard deviation of returns, implicitly
        assuming Gaussian, i.i.d. returns and ignoring how much uncertainty surrounds the estimate
        itself. The `method` parameter selects one of five corrections for that, each keeping the
        same excess returns and therefore the same underlying Sharpe ratio as its starting point:

        - `"smart"` — the Smart Sharpe ratio of quantstats, which divides the Sharpe ratio by an
        autocorrelation penalty so that serially correlated returns (whose standard deviation
        understates the risk over several periods) no longer flatter it:

            - Smart Sharpe Ratio = SR / SQRT( 1 + 2 · Σ_{k=1}^{n−1} ((n − k) / n) · |ρ|^k )

        where n is the number of excess returns and ρ their first order autocorrelation (see
        `performance_model.get_autocorrelation_penalty`). It is a simplification of `"lo"`, which
        uses the autocorrelation at every lag rather than powers of the first one, and unlike
        `"lo"` it is not annualized unless annualize=True.

        - `"lo"` — the annualized Sharpe ratio of Lo (2002), which corrects the annualization for
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

        - `"adjusted"` — the Adjusted Sharpe Ratio (ASR, Pezier & White, 2006) penalizes (or
        rewards) the Sharpe ratio for negative skewness and excess kurtosis using a
        Cornish-Fisher-style expansion, so that two strategies with the same Sharpe ratio but
        different tail shapes are no longer scored identically:

            - ASR = SR * [1 + (S / 6) * SR − ((K − 3) / 24) * SR^2]

        - `"probabilistic"` — the Probabilistic Sharpe Ratio (PSR) is the probability that the true
        (population) Sharpe ratio exceeds `benchmark_sharpe_ratio`, folding the skewness and
        (non-excess) kurtosis of the underlying returns into the standard error of the Sharpe ratio
        so that a short, lumpy sample no longer looks more convincing than it is:

            - PSR(SR*) = Φ( (SR̂ − SR*) · sqrt(n − 1) / sqrt(1 − γ₃·SR̂ + ((γ₄ − 1) / 4)·SR̂²) )

        - `"deflated"` — the Deflated Sharpe Ratio (DSR) is the Probabilistic Sharpe Ratio corrected
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
        *rolling* Sharpe ratio series (see `get_rolling_sharpe_ratio`) computed over a
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

        Args:
            period (str, optional): The period to use for the calculation. Defaults to "quarterly" if the Toolkit is
                initialised with quarterly=True, otherwise "yearly".
            rolling (int, optional): The rolling period to use for the calculation. If you select
            period = 'monthly' and set rolling to 12 you obtain the rolling 12-month Sharpe Ratio.
            Defaults to None, which calculates the Sharpe ratio within each period.
            method (str, optional): Which Sharpe ratio to calculate, one of "standard", "smart", "lo",
            "adjusted", "probabilistic" or "deflated", as described above. Defaults to "standard".
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
            pd.DataFrame: Sharpe ratio values. For method="probabilistic" and method="deflated"
            these are probabilities between 0 and 1 rather than ratios.

        Notes:
        - period="daily" needs intraday data (initialise the Toolkit with an intraday_period): the ratio per day
        is then computed from the intraday bars within each day, with the daily risk free rate spread over the
        bars of a day. Without intraday data a single day holds one return, so the result is empty and logs why.
        - The method retrieves historical data and calculates the Sharpe ratio for each asset in the Toolkit instance.
        - The risk-free rate is the Toolkit's risk_free_rate, the 10-year US Treasury yield by default; choose
        risk_free_rate="13w" when initialising the Toolkit for the short-term rate Sharpe (1994) uses.
        - The "adjusted", "probabilistic" and "deflated" variants use the **non-excess (raw)** kurtosis
        convention, i.e. a Normal distribution has a kurtosis of 3, not 0. Internally this calls
        `risk_model.get_kurtosis(..., fisher=False)`.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.
        - `get_sharpe_ratio(method="probabilistic")` returns the probability that each Sharpe ratio is genuine,
        i.e. above `benchmark_sharpe_ratio` once the sample length, skewness and kurtosis are accounted for.
        - method="smart" agrees with quantstats' smart_sharpe (stats.smart_sharpe,
        https://github.com/ranaroussi/quantstats) on the same excess returns; quantstats returns a penalty of 1
        where the autocorrelation is undefined (fewer than two returns, or returns without variation), the Toolkit NaN.

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
        if method not in (
            "standard",
            "smart",
            "lo",
            "adjusted",
            "probabilistic",
            "deflated",
        ):
            raise ValueError(
                "Method must be standard, smart, lo, adjusted, probabilistic, or deflated."
            )

        period = period if period else "quarterly" if self._quarterly else "yearly"

        if rolling:
            period_returns = self._get_column(period, "Return", within_period=False)
            full_excess_return = performance_model.get_excess_return(
                period_returns,
                self._risk_free_rate_data[period][period_returns.columns],
            )

        if rolling:
            excess_return = full_excess_return
            sharpe_ratio = performance_model.get_rolling_sharpe_ratio(
                excess_return, rolling
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
                performance_model.get_rolling_autocorrelation_penalty(
                    excess_return, rolling
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
                        performance_model.get_rolling_lo_sharpe_ratio(
                            excess_return[[column]],
                            rolling,
                            observations_per_year[column],
                            lags,
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
                skewness = get_rolling_skewness(excess_return, rolling)
                kurtosis = get_rolling_kurtosis(excess_return, rolling, fisher=False)
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
                        column: performance_model.get_rolling_sharpe_ratio(
                            trials_returns[column], int(window[column])
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
        method: str = "standard",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        annualize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """

        The Sortino Ratio is a financial metric used to assess the risk-adjusted performance of an
        investment portfolio or asset by considering only the downside risk. It measures the excess
        return generated by the portfolio per unit of downside risk, specifically, the standard
        deviation of negative returns. The Sortino Ratio is particularly useful for investors who
        are primarily concerned with minimizing the downside risk of their investments.

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
        risk_free_rate, the 10-year US Treasury yield by default; initialise the Toolkit with
        risk_free_rate="13w" for a short-term Treasury bill rate instead.

        A period needs at least two observations for a Sortino ratio, with a single one the ratio
        would only show its sign, so such periods are left out.

        As with the Sharpe Ratio, the result is **not annualized** by default: it is a
        per-observation ratio, roughly SQRT(252), SQRT(52), SQRT(12) or SQRT(4) smaller than the
        annualized figure for daily, weekly, monthly or quarterly returns respectively. Set
        annualize=True to multiply by that factor.

        The `method` parameter selects a variant of the same ratio:

        - `"smart"` — the Smart Sortino ratio of quantstats, the Sortino ratio divided by the
        autocorrelation penalty that also gives the Smart Sharpe ratio, so that serially
        correlated returns no longer flatter it (see `performance_model.get_autocorrelation_penalty`):

            - Smart Sortino Ratio = Sortino Ratio / SQRT( 1 + 2 · Σ_{k=1}^{n−1} ((n − k) / n) · |ρ|^k )

        - `"adjusted"` — the Sortino ratio divided by SQRT(2), as Jack Schwager proposed to make it
        comparable with the Sharpe ratio: for symmetric returns the downside deviation is about
        1 / SQRT(2) times the standard deviation, so the plain Sortino ratio is about SQRT(2)
        times the Sharpe ratio.

        - `"probabilistic"` and `"probabilistic_adjusted"` — the probability that the true Sortino
        (or adjusted Sortino) ratio is above 0, with the standard error of the Probabilistic Sharpe
        Ratio of Bailey and López de Prado (2012) applied to it, as quantstats does:

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
        - quantstats: stats.sortino, stats.smart_sortino, stats.adjusted_sortino,
        stats.probabilistic_sortino_ratio and stats.probabilistic_adjusted_sortino_ratio, the
        reference for the variants (https://github.com/ranaroussi/quantstats).

        Args:
            period (str, optional): The period to use for the calculation. Defaults to "quarterly" if the Toolkit is
                initialised with quarterly=True, otherwise "yearly".
            rolling (int, optional): The rolling window size to use for the calculation. If set,
            the Sortino ratio is calculated over a rolling window of this many periods across the
            full return history instead of per `period`. Defaults to None.
            method (str, optional): Which Sortino ratio to calculate, one of "standard", "smart",
            "adjusted", "probabilistic" or "probabilistic_adjusted", as described above.
            Defaults to "standard".
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
            pd.DataFrame: Sortino ratio values. For method="probabilistic" and
            "probabilistic_adjusted" these are probabilities between 0 and 1.

        Notes:
        - period="daily" needs intraday data (initialise the Toolkit with an intraday_period): the ratio per day
        is then computed from the intraday bars within each day, with the daily risk free rate spread over the
        bars of a day. Without intraday data a single day holds one return, so the result is empty and logs why.
        - The method retrieves historical data and calculates the Sortino ratio for each asset in the Toolkit instance.
        - The risk-free rate is the Toolkit's risk_free_rate, the 10-year US Treasury yield by default. Many
        textbook definitions (Sharpe, 1994; the Ken French factors) use a short-term Treasury bill instead, which
        the Toolkit uses when initialised with risk_free_rate="13w".
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.
        - The variants agree with quantstats' sortino, smart_sortino, adjusted_sortino and the probabilistic Sortino
        ratios (with annualize=False) on the same excess returns. quantstats returns a penalty of 1 where the
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
        if method not in (
            "standard",
            "smart",
            "adjusted",
            "probabilistic",
            "probabilistic_adjusted",
        ):
            raise ValueError(
                "Method must be standard, smart, adjusted, probabilistic, or "
                "probabilistic_adjusted."
            )

        period = period if period else "quarterly" if self._quarterly else "yearly"

        if rolling:
            period_returns = self._get_column(period, "Return", within_period=False)
            excess_return = performance_model.get_excess_return(
                period_returns,
                self._risk_free_rate_data[period][period_returns.columns],
            )
            sortino_ratio = performance_model.get_rolling_sortino_ratio(
                excess_return, rolling
            )
        else:
            excess_return = self._get_column(
                period, "Excess Return", within_period=True
            )

            sortino_ratio = performance_model.get_sortino_ratio(excess_return)

        if method == "smart":
            sortino_ratio = sortino_ratio / (
                performance_model.get_rolling_autocorrelation_penalty(
                    excess_return, rolling
                )
                if rolling
                else performance_model.get_autocorrelation_penalty(excess_return)
            )
        elif method in ("adjusted", "probabilistic_adjusted"):
            sortino_ratio = sortino_ratio / np.sqrt(2)

        if method in ("probabilistic", "probabilistic_adjusted"):
            if rolling:
                skewness = get_rolling_skewness(excess_return, rolling)
                kurtosis = get_rolling_kurtosis(excess_return, rolling, fisher=False)
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
        threshold: float = 0.0,
        rolling: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate Roy's Safety-First Ratio of an investment portfolio or asset's returns.

        Roy (1952) argues that an investor first of all wants to avoid a disaster, a return
        below a threshold, and so prefers the portfolio with the lowest probability of falling
        below it. For normally distributed returns that probability is N(-Safety-First Ratio),
        so the best portfolio is the one with the highest ratio: the number of standard
        deviations the expected return lies above the threshold.

        The formula is as follows:

        - Safety-First Ratio = (Mean(Return) - Threshold per Observation) / Standard Deviation(Return)
        - Threshold per Observation = (1 + Threshold)^(1 / Observations per Year) - 1

        The threshold is an annual return, compounded down to the returns the ratio is computed
        from: by default the **daily** returns falling inside each `period`, 252 a year for a
        ticker that trades on trading days and 365 for one that also trades on weekends (such
        as a cryptocurrency). With `rolling`, `period` instead sets the frequency of the returns
        and the ratio is computed over a rolling window of `rolling` such returns, with the
        threshold compounded to that frequency. The standard deviation is the sample one
        (ddof=1). With the risk-free rate as the threshold the ratio is the Sharpe Ratio
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
            period (str, optional): The period to use for the calculation. Defaults to "quarterly" if the Toolkit is
                initialised with quarterly=True, otherwise "yearly".
            threshold (float, optional): The minimum acceptable (shortfall) return as an annual decimal return,
                e.g. 0.02 for 2% a year. Defaults to 0.0.
            rolling (int | None, optional): The rolling window size to use for the calculation. If set, the
                ratio is calculated over a rolling window of this many periods across the full return history
                instead of per `period`. Defaults to None.
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
        - period="daily" needs intraday data (initialise the Toolkit with an intraday_period): the ratio per day
        is then computed from the intraday bars within each day, with the threshold compounded over the bars
        of a year. Without intraday data a single day holds one return, so the result is empty and logs why.
        - The method retrieves historical data and calculates the ratio for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["KO", "PEP", "MDLZ"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.performance.get_safety_first_ratio(threshold=0.02)
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
        threshold_per_observation = (1 + threshold) ** (1 / observations_per_year) - 1

        safety_first_ratio = (
            performance_model.get_rolling_safety_first_ratio(
                returns, rolling, threshold_per_observation
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
        Calculate the Upside Potential Ratio of an investment portfolio or asset's returns.

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

        Args:
            period (str, optional): The period to use for the calculation. Defaults to "quarterly" if the Toolkit is
                initialised with quarterly=True, otherwise "yearly".
            rolling (int | None, optional): The rolling window size to use for the calculation. If set, the
                ratio is calculated over a rolling window of this many periods across the full return history
                instead of per `period`. Defaults to None.
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
        - period="daily" needs intraday data (initialise the Toolkit with an intraday_period): the ratio per day
        is then computed from the intraday bars within each day, with the daily risk free rate spread over the
        bars of a day. Without intraday data a single day holds one return, so the result is empty and logs why.
        - The method retrieves historical data and calculates the ratio for each asset in the Toolkit instance.
        - The risk-free rate is the Toolkit's risk_free_rate, the 10-year US Treasury yield by default. Many
        textbook definitions (Sharpe, 1994; the Ken French factors) use a short-term Treasury bill instead, which
        the Toolkit uses when initialised with risk_free_rate="13w".
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

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

        if rolling:
            period_returns = self._get_column(period, "Return", within_period=False)
            excess_return = performance_model.get_excess_return(
                period_returns,
                self._risk_free_rate_data[period][period_returns.columns],
            )
            upside_potential_ratio = (
                performance_model.get_rolling_upside_potential_ratio(
                    excess_return, rolling
                )
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
        Calculate the Ulcer Performance Index (UPI), alternatively called Martin ratio, a measure of risk-adjusted
        return that evaluates the excess return of an investment portfolio or asset per unit of risk taken.

        The formula is as follows:

        - Ulcer Performance Index = (Period Return - Risk-Free Rate) / Ulcer Index
        - Ulcer Index = SQRT(MEAN(Drawdown^2)), with each day's drawdown measured from the highest value since
        the start of the period (Martin's definition), or of the trailing `rolling` days within the period

        It measures the excess return earned per unit of drawdown pain: the depth and the duration of the declines
        both weigh in, where the Sharpe Ratio treats upward and downward swings alike. Similar to the Sharpe Ratio,
        a higher UPI is better than a lower one (since investors prefer more return for less risk).

        Also known as: UPI, Martin ratio.

        Args:
            period (str, optional): The period to use for the calculation. Defaults to "quarterly" if the Toolkit is
                initialised with quarterly=True, otherwise "yearly".
            rolling (int | None, optional): The number of trailing days whose highest value each day's drawdown
                is measured from. Defaults to None, which measures it from the highest value since the start of
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
        - The risk-free rate is the Toolkit's risk_free_rate, the 10-year US Treasury yield by default. Many
        textbook definitions (Sharpe, 1994; the Ken French factors) use a short-term Treasury bill instead, which
        the Toolkit uses when initialised with risk_free_rate="13w".
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

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
        method: str = "standard",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Calmar Ratio of an investment portfolio or asset's returns.

        The Calmar Ratio is a risk-adjusted return metric that divides the return of an investment
        portfolio or asset over each period (a calendar-year return for period='yearly', a quarterly
        return for period='quarterly') by its Maximum Drawdown, providing insight into the return
        achieved per unit of the worst historical loss of value.

        The formula is as follows:

        - Calmar Ratio = Return / |Maximum Drawdown|

        With method="annualized" it is the definition of the reference libraries instead, one value
        per ticker over the whole date range:

        - Calmar Ratio = Annualized Return / |Maximum Drawdown|
        - Annualized Return = PRODUCT(1 + Daily Return) ^ (Observations per Year / N) − 1

        Where both are measured over the daily returns of the whole date range, N is their number and
        the observations per year are 252 (365 for a ticker that trades on weekends, such as a
        cryptocurrency). Young (1991) measured it over the last 36 months.

        See definition: https://en.wikipedia.org/wiki/Calmar_ratio

        For more information, see the following sources:

        - Young, T.W. (1991). "Calmar Ratio: A Smoother Tool." Futures, 20(1), 40.
        - PerformanceAnalytics: CalmarRatio (https://github.com/braverock/PerformanceAnalytics).
        - empyrical: calmar_ratio (https://github.com/stefan-jansen/empyrical-reloaded).
        - quantstats: stats.calmar (https://github.com/ranaroussi/quantstats).
        - ffn: calc_calmar_ratio (https://github.com/pmorissette/ffn).

        Also known as: Drawdown ratio.

        Args:
            period (str, optional): The period to use for the calculation. Defaults to "quarterly" if the Toolkit is
                initialised with quarterly=True, otherwise "yearly".
            within_period (bool, optional): Whether to calculate the Maximum Drawdown within the
            specified period or for the entire period. Thus whether to look at the Maximum Drawdown
            within a specific year (if period = 'yearly') or look at the entirety of all years.
            Defaults to True.
            method (str, optional): "standard" for the ratio per period, or "annualized" for the
            annualized return over the whole date range divided by the Maximum Drawdown over it,
            one value per ticker (`period` and `within_period` are then not used). Defaults to
            "standard".
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
            pd.DataFrame: Calmar Ratio values, or with method="annualized" a Series with one value
            per ticker.

        Notes:
        - The method retrieves historical data and calculates the Calmar Ratio for each asset in
        the Toolkit instance.
        - method="annualized" agrees with PerformanceAnalytics' CalmarRatio, empyrical's calmar_ratio and quantstats'
        calmar on the daily returns (252 observations a year). ffn (calc_calmar_ratio) counts the years of the
        annualized return in calendar days, which differs slightly.
        - If `growth` is set to True, the method calculates the growth of the ratio values using
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
        if method not in ("standard", "annualized"):
            raise ValueError("Method must be standard or annualized.")

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
                # One number per ticker over the whole date range, with no dates to slice.
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
        method: str = "standard",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Sterling Ratio of an investment portfolio or asset's returns.

        The Sterling Ratio is a risk-adjusted return metric that divides the return of an investment
        portfolio or asset over each period (a calendar-year return for period='yearly', a quarterly
        return for period='quarterly') by its Average Drawdown plus a fixed adjustment
        (conventionally 10%), providing insight into the return achieved relative to the typical
        depth of its drawdowns rather than only the single worst one (as with the Calmar Ratio).

        The formula is as follows:

        - Sterling Ratio = Return / (|Average Drawdown| + Adjustment)

        Where the Average Drawdown is the mean drawdown from the running peak over all observations
        on which the asset is below that peak (by default every such day within the period), not the
        average of the largest drawdowns. This is the average drawdown form of Bacon (2008).

        With method="annualized" it is the definition of PerformanceAnalytics (SterlingRatio, after
        Kestner, 1996) instead, one value per ticker over the whole date range:

        - Sterling Ratio = Annualized Return / (|Maximum Drawdown| + Adjustment)
        - Annualized Return = PRODUCT(1 + Daily Return) ^ (Observations per Year / N) − 1

        Where both are measured over the daily returns of the whole date range, N is their number and
        the observations per year are 252 (365 for a ticker that trades on weekends).

        For more information, see the following sources:

        - Kestner, L.N. (1996). "Getting a Handle on True Performance." Futures, 25(1), 44-46.
        - Bacon, C.R. (2008). "Practical Portfolio Performance Measurement and Attribution."
        2nd edition, John Wiley & Sons, p. 89.
        - PerformanceAnalytics: SterlingRatio (https://github.com/braverock/PerformanceAnalytics).

        Also known as: Sterling-Calmar ratio.

        Args:
            period (str, optional): The period to use for the calculation. Defaults to "quarterly" if the Toolkit is
                initialised with quarterly=True, otherwise "yearly".
            within_period (bool, optional): Whether to calculate the Average Drawdown within the
            specified period or for the entire period. Thus whether to look at the Average Drawdown
            within a specific year (if period = 'yearly') or look at the entirety of all years.
            Defaults to True.
            adjustment (float, optional): The fixed adjustment added to the Average Drawdown,
            conventionally 0.1 (10%). Defaults to 0.1.
            method (str, optional): "standard" for the ratio per period with the Average Drawdown,
            or "annualized" for the annualized return over the whole date range divided by the
            Maximum Drawdown over it plus the adjustment, one value per ticker (`period` and
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
            pd.DataFrame: Sterling Ratio values, or with method="annualized" a Series with one value
            per ticker.

        Notes:
        - The method retrieves historical data and calculates the Sterling Ratio for each asset in
        the Toolkit instance.
        - method="annualized" agrees with PerformanceAnalytics' SterlingRatio on the daily returns (252
        observations a year, excess = `adjustment`).
        - If `growth` is set to True, the method calculates the growth of the ratio values using
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
        | 2021   |  1.2842 |  1.4182 | 1.2009 |
        | 2022   | -0.3726 |  0.1186 | 0.4819 |
        | 2023   | -0.8979 | -0.0838 | 0.1342 |
        | 2024   |  1.307  |  1.1541 | 1.5685 |
        | 2025   |  0.9868 |  0.9407 | 0.6777 |
        """
        if method not in ("standard", "annualized"):
            raise ValueError("Method must be standard or annualized.")

        if method == "annualized":
            daily_returns, annualized_return = self._get_whole_sample_returns()

            return finalize_dataset(
                dataset=performance_model.get_sterling_ratio(
                    annualized_return, get_max_drawdown(daily_returns), adjustment
                ),
                start_date=self._start_date,
                end_date=self._end_date,
                default_rounding=self._rounding,
                rounding=rounding,
                # One number per ticker over the whole date range, with no dates to slice.
                apply_slice=False,
            )

        period = period if period else "quarterly" if self._quarterly else "yearly"

        returns = self._get_column(period, "Return", within_period=within_period)

        average_drawdown = performance_model.get_average_drawdown(returns)

        period_returns = self._get_column(period, "Return", within_period=False)

        sterling_ratio = performance_model.get_sterling_ratio(
            period_returns, average_drawdown, adjustment
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
        Calculate the Burke Ratio of an investment portfolio or asset's returns.

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
        - PerformanceAnalytics: BurkeRatio (https://github.com/braverock/PerformanceAnalytics).

        Args:
            period (str, optional): The period to use for the calculation. Defaults to "quarterly" if the Toolkit is
                initialised with quarterly=True, otherwise "yearly".
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
        - The risk-free rate is the Toolkit's risk_free_rate, the 10-year US Treasury yield by default. Many
        textbook definitions (Sharpe, 1994; the Ken French factors) use a short-term Treasury bill instead, which
        the Toolkit uses when initialised with risk_free_rate="13w".
        - If `growth` is set to True, the method calculates the growth of the ratio values using
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
        Calculate the Pain Ratio of an investment portfolio or asset's returns.

        The Pain Ratio divides the excess return (return minus the risk-free rate) of an
        investment portfolio or asset by its Pain Index, the mean depth of its drawdowns. Where
        the Ulcer Performance Index squares the drawdowns, which makes the deep ones dominate,
        the Pain Index weighs every observation under water by its depth alone.

        The formula is as follows:

        - Pain Ratio = (Period Return - Risk-Free Rate) / Pain Index
        - Pain Index = SUM(|Drawdown_t|) / N
        - Drawdown_t = Wealth_t / Peak_t - 1, with Peak_t the highest wealth since the first return

        With within_period=True (the default) the observations are the daily returns within each
        period: N is the number of days in the period, and the wealth and its peak restart from 1
        before the first return of every period, so a loss on its first day is a drawdown as
        well. With within_period=False the observations are the `period` returns (such as the
        yearly returns) of the full history: N is the number of those periods, and every period's
        excess return is divided by that one Pain Index, as the Calmar, Sterling and Burke ratios
        do. Observations at a new high count as a drawdown of 0. The risk-free rate is compounded over the days of the
        period, so a partial period (such as the current year) earns the rate of the days it
        covers. Bacon (2013) divides the annualized return minus the risk-free rate; for
        period="yearly" that is the same for every full year, and for other periods the ratio
        uses the period's own (not annualized) excess return, as the Ulcer Performance Index
        and the Burke Ratio do.

        For more information, see the following source:

        - Bacon, C.R. (2013). "Practical Risk-Adjusted Performance Measurement." John Wiley &
        Sons, chapter 5 (the Pain Index and Pain Ratio of Zephyr Associates).

        Args:
            period (str, optional): The period to use for the calculation. Defaults to "quarterly" if the Toolkit is
                initialised with quarterly=True, otherwise "yearly".
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
        - The risk-free rate is the Toolkit's risk_free_rate, the 10-year US Treasury yield by default. Many
        textbook definitions (Sharpe, 1994; the Ken French factors) use a short-term Treasury bill instead, which
        the Toolkit uses when initialised with risk_free_rate="13w".
        - If `growth` is set to True, the method calculates the growth of the ratio values using
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
        method: str = "standard",
        excess: str | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        The M2 Ratio, also known as the Modigliani-Modigliani Measure, is a financial
        metric used to evaluate the risk-adjusted performance of an investment portfolio
        or strategy. It assesses the excess return generated by the portfolio relative
        to a risk-free investment, taking into account the portfolio's volatility or risk.
        The M2 Ratio helps investors and portfolio managers determine whether the portfolio
        is delivering returns that justify its level of risk.

        The formula is as follows:

            - M2 Ratio = Risk-Free Rate + [(Portfolio's Return — Risk-Free Rate) /
              Portfolio Standard Deviation] × Benchmark Standard Deviation

        This rescales the (dimensionless) Sharpe ratio back into return-space by asking
        what return the portfolio would have earned had it been leveraged or
        de-leveraged, via risk-free borrowing or lending, to match the benchmark's
        volatility exactly -- producing a number directly comparable to the benchmark's
        actual return. Requires a `benchmark_ticker` to be set on the Toolkit instance,
        since the benchmark's standard deviation is part of the formula.

        With method="sortino" it is the M2 for Sortino of Bacon (2008), which matches the
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
        - PerformanceAnalytics: MSquared, M2Sortino and MSquaredExcess
        (https://github.com/braverock/PerformanceAnalytics).

        Also known as: Modigliani-Modigliani measure, M2, risk-adjusted performance.

        Args:
            period (str, optional): The period to use for the calculation. Defaults to "quarterly" if the Toolkit is
                initialised with quarterly=True, otherwise "yearly".
            rolling (int, optional): The rolling window size to use for the calculation. If set,
            the M2 ratio is calculated over a rolling window of this many periods across the full
            return history instead of per `period`. Only for method="standard". Defaults to None.
            method (str, optional): "standard" to match the benchmark's total risk, or "sortino" to
            match its downside risk, as described above. Defaults to "standard".
            excess (str | None, optional): None for the M2 measure itself, or "arithmetic" or
            "geometric" for the M2 excess return over the benchmark. Defaults to None.
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
        - The risk-free rate is the Toolkit's risk_free_rate, the 10-year US Treasury yield by default. Many
        textbook definitions (Sharpe, 1994; the Ken French factors) use a short-term Treasury bill instead, which
        the Toolkit uses when initialised with risk_free_rate="13w".
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

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
        if method not in ("standard", "sortino"):
            raise ValueError("Method must be standard or sortino.")
        if excess not in (None, "arithmetic", "geometric"):
            raise ValueError("Excess must be None, arithmetic or geometric.")
        if rolling and method == "sortino":
            raise ValueError(
                "The M2 for Sortino is calculated per period, not rolling."
            )

        period = period if period else "quarterly" if self._quarterly else "yearly"
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
            m2_ratio = performance_model.get_rolling_m2_ratio(
                period_returns, risk_free_rate, benchmark_period_returns, rolling
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
        Tracking Error is a financial metric that quantifies the volatility or dispersion of the
        difference between the returns of an investment portfolio or asset and the returns of a
        benchmark index. It measures how closely the portfolio tracks its benchmark and provides
        insights into the consistency of the portfolio's performance relative to the benchmark.
        A higher Tracking Error indicates greater divergence from the benchmark, while a lower
        Tracking Error suggests that the portfolio closely follows the benchmark.

        The formula is as follows:

            - Tracking Error (TE) = Standard Deviation of (Portfolio Returns — Benchmark Returns)

        By default one Tracking Error is reported per `period`, computed from the daily returns within that
        period, so it is a daily figure: multiply by SQRT(252) to compare it with the annualized tracking error
        funds report. With `rolling`, the returns of `period` itself are used instead.

        See definition: https://en.wikipedia.org/wiki/Tracking_error

        Also known as: active risk, benchmark deviation.

        Args:
            period (str, optional): The period to use for the calculation. Defaults to "quarterly" if the Toolkit is
                initialised with quarterly=True, otherwise "yearly".
            rolling (int, optional): The rolling window size to use for the calculation. If set,
            Tracking Error is calculated over a rolling window of this many periods across the
            full return history instead of per `period`. Defaults to None.
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
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

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

        if rolling:
            returns = self._get_column(period, "Return", within_period=False)
            benchmark_returns = self._get_column(
                period, "Return", within_period=False, benchmark=True
            )

            tracking_error = performance_model.get_rolling_tracking_error(
                returns, benchmark_returns, rolling
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
        The Information Ratio (IR) is a financial metric that assesses the risk-adjusted
        performance of a portfolio or investment strategy relative to a benchmark index. It
        quantifies how much active return (the return above the benchmark) the portfolio
        generates per unit of tracking error, the volatility of that active return.
        The Information Ratio is commonly used by portfolio managers, financial analysts, and
        investors to evaluate the skill of a portfolio manager in generating returns beyond
        what would be expected based on the risk taken.

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

        Args:
            period (str, optional): The period to use for the calculation. Defaults to "quarterly" if the Toolkit is
                initialised with quarterly=True, otherwise "yearly".
            rolling (int, optional): The rolling window size to use for the calculation. If set,
            the Information Ratio is calculated over a rolling window of this many periods across
            the full return history instead of per `period`. Defaults to None.
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
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

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

        if rolling:
            returns = self._get_column(period, "Return", within_period=False)
            benchmark_returns = self._get_column(
                period, "Return", within_period=False, benchmark=True
            )

            information_ratio = performance_model.get_rolling_information_ratio(
                returns, benchmark_returns, rolling
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
        method: str = "geometric",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Upside Capture Ratio of an investment portfolio or asset's returns.

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
        - empyrical: up_capture and roll_up_capture (https://github.com/stefan-jansen/empyrical-reloaded).
        - PerformanceAnalytics: UpDownRatios (https://github.com/braverock/PerformanceAnalytics).

        Args:
            period (str, optional): The period to use for the calculation. Defaults to "quarterly" if the Toolkit is
                initialised with quarterly=True, otherwise "yearly".
            rolling (int, optional): The rolling window size to use for the calculation. If you select
                period = 'monthly' and set rolling to 36 you obtain the rolling three year Upside Capture
                Ratio of the monthly returns, Morningstar's figure. Defaults to None, which uses the daily
                returns within each period.
            method (str, optional): How the returns are averaged. Defaults to "geometric". Choose from:
                - "geometric": the geometric average return per observation (Morningstar).
                - "annualized": the geometric average annualized, (1 + average)^P - 1 with P the
                  observations in a year, the ratio of the annual returns empyrical's up_capture reports.
                - "arithmetic": the sum of the returns, as PerformanceAnalytics' UpDownRatios reports.
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
        - method="annualized" agrees with empyrical's up_capture (and with rolling= its roll_up_capture) on the
        same returns. Annualizing compounds both averages over a year, which changes the value but not the
        ranking of tickers against the same benchmark. method="arithmetic" agrees with PerformanceAnalytics'
        UpDownRatios (method="Capture").
        - See `get_capture_ratios` for the overall capture ratio, the Up/Down Capture Ratio and the Up and Down
        Numbers and Percentages in one table.
        - If `growth` is set to True, the method calculates the growth of the ratio values using
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
        returns = self._get_column(period, "Return", within_period=not rolling)
        benchmark_returns = self._get_column(
            period, "Return", within_period=not rolling, benchmark=True
        )
        periods_per_year = self._get_ratio_observations_per_year(
            period, rolling, returns.columns
        )

        if rolling:
            upside_capture_ratio = performance_model.get_rolling_capture_ratio(
                returns,
                benchmark_returns,
                rolling,
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
            # The rolling ratio starts after the first full window.
            dropna=bool(rolling),
        )

    @handle_portfolio
    @handle_errors
    def get_downside_capture_ratio(
        self,
        period: str | None = None,
        rolling: int | None = None,
        method: str = "geometric",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Downside Capture Ratio of an investment portfolio or asset's returns.

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
        - empyrical: down_capture and roll_down_capture (https://github.com/stefan-jansen/empyrical-reloaded).
        - PerformanceAnalytics: UpDownRatios (https://github.com/braverock/PerformanceAnalytics).

        Args:
            period (str, optional): The period to use for the calculation. Defaults to "quarterly" if the Toolkit is
                initialised with quarterly=True, otherwise "yearly".
            rolling (int, optional): The rolling window size to use for the calculation. If you select
                period = 'monthly' and set rolling to 36 you obtain the rolling three year Downside Capture
                Ratio of the monthly returns, Morningstar's figure. Defaults to None, which uses the daily
                returns within each period.
            method (str, optional): How the returns are averaged. Defaults to "geometric". Choose from:
                - "geometric": the geometric average return per observation (Morningstar).
                - "annualized": the geometric average annualized, (1 + average)^P - 1 with P the
                  observations in a year, the ratio of the annual returns empyrical's down_capture reports.
                - "arithmetic": the sum of the returns, as PerformanceAnalytics' UpDownRatios reports.
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
        - method="annualized" agrees with empyrical's down_capture (and with rolling= its roll_down_capture) on the
        same returns. Annualizing compounds both averages over a year, which changes the value but not the
        ranking of tickers against the same benchmark. method="arithmetic" agrees with PerformanceAnalytics'
        UpDownRatios (method="Capture"), except that PerformanceAnalytics also counts the days the benchmark
        did not move as down days.
        - See `get_capture_ratios` for the overall capture ratio, the Up/Down Capture Ratio and the Up and Down
        Numbers and Percentages in one table.
        - If `growth` is set to True, the method calculates the growth of the ratio values using
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
        returns = self._get_column(period, "Return", within_period=not rolling)
        benchmark_returns = self._get_column(
            period, "Return", within_period=not rolling, benchmark=True
        )
        periods_per_year = self._get_ratio_observations_per_year(
            period, rolling, returns.columns
        )

        if rolling:
            downside_capture_ratio = performance_model.get_rolling_capture_ratio(
                returns,
                benchmark_returns,
                rolling,
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
            # The rolling ratio starts after the first full window.
            dropna=bool(rolling),
        )

    @handle_errors
    def get_capture_ratios(
        self,
        period: str | None = None,
        method: str = "geometric",
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
        - empyrical: capture, up_capture, down_capture and up_down_capture
        (https://github.com/stefan-jansen/empyrical-reloaded).
        - PerformanceAnalytics: UpDownRatios, table.UpDownRatios and table.CaptureRatios
        (https://github.com/braverock/PerformanceAnalytics).

        Also known as: up/down capture, up-market and down-market capture, capture table.

        Args:
            period (str, optional): The period to use for the calculation. Defaults to "quarterly" if the Toolkit is
                initialised with quarterly=True, otherwise "yearly".
            method (str, optional): How the capture ratios average the returns. Defaults to "geometric".
                Choose from:
                - "geometric": the geometric average return per day (Morningstar).
                - "annualized": the geometric average annualized, (1 + average)^P - 1 with P the days in a
                  year (252, 365 for a ticker that trades on weekends), empyrical's capture ratios.
                - "arithmetic": the sum of the returns, PerformanceAnalytics' UpDownRatios.
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
        - With method="annualized" the Capture Ratio, Upside, Downside and Up/Down Capture Ratios equal
        empyrical's capture, up_capture, down_capture and up_down_capture on the same daily returns. With
        method="arithmetic" the Upside and Downside Capture Ratios agree with PerformanceAnalytics' UpDownRatios,
        which also counts the days the benchmark did not move as down days for its Down Capture; its Up and
        Down Number and Percent equal the Toolkit's.
        - There is no Portfolio column: a weighted average of capture ratios or counts is not the statistic
        of the portfolio.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

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
        Calculate the Win Rate of an investment portfolio or asset's returns.

        The Win Rate is the share of days within each period in which the asset's return exceeds the
        benchmark's return.

        The formula is as follows:

        - Win Rate = (Number of Days with Asset Return > Benchmark Return) /
          (Number of Days with both an Asset and a Benchmark Return), using the daily returns within each
          period (e.g. each year), as a decimal (0.55 = 55%)

        Days on which the asset or the benchmark has no return are left out of both counts rather than
        counted as losses.

        With market="up" or "down" only the days the benchmark rose or fell are counted, the up and down
        market batting averages: a manager who beats a rising market by taking more risk shows a high up
        market and a low down market figure, while one who protects capital shows the reverse.

        - Up Market Win Rate = (Number of Days with Asset Return > Benchmark Return and Benchmark Return > 0) /
          (Number of Days with Benchmark Return > 0)

        For more information, see the following sources:

        - Bacon, C.R. (2008). "Practical Portfolio Performance Measurement and Attribution."
        2nd edition, John Wiley & Sons.
        - empyrical: batting_average (https://github.com/stefan-jansen/empyrical-reloaded).
        - PerformanceAnalytics: UpDownRatios (method="Percent") (https://github.com/braverock/PerformanceAnalytics).

        Also known as: batting average.

        Args:
            period (str, optional): The period to use for the calculation. Defaults to "quarterly" if the Toolkit is
                initialised with quarterly=True, otherwise "yearly".
            market (str, optional): The days counted: "all", "up" (the benchmark rose) or "down" (the
                benchmark fell). Defaults to "all".
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
        - If `growth` is set to True, the method calculates the growth of the ratio values using
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
        a longer window has fewer of them. With relative=False the window return is compared with 0 instead of
        the benchmark's, the share of windows with a gain: for 12 monthly returns ffn's "Win 12m %".

        The measure is defined in PerformanceAnalytics (Peterson and Carl) as table.ProbOutPerformance; the
        Toolkit applies it to each ticker against its benchmark.

        For more information, see the following sources:

        - PerformanceAnalytics: table.ProbOutPerformance (https://github.com/braverock/PerformanceAnalytics).
        - ffn: PerformanceStats.twelve_month_win_perc (https://github.com/pmorissette/ffn).

        Also known as: rolling win rate, hit rate over rolling windows.

        Args:
            period (str, optional): The period of the returns the windows are made of. Defaults to "monthly".
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
        - With relative=False and windows=[12] it agrees with ffn's twelve_month_win_perc, except that ffn also counts
        the window that starts at the first price, one window more than the monthly returns hold.
        - Each ticker is measured over its own history, so a later listing has fewer windows.
        - There is no Portfolio column: a weighted average of probabilities is not that of the portfolio.

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
        benchmark excess return was positive (PerformanceAnalytics) or, with market="return", when the
        benchmark return was positive (empyrical), and P is the number of days in a year (252, 365 for a ticker
        that trades on weekends). A Timing Ratio above 1 means the asset moved more with the market when it rose
        than when it fell, the asymmetry a successful market timer shows; Henriksson and Merton (1981) test the
        same asymmetry with one regression (see `get_henriksson_merton_model`).

        For more information, see the following sources:

        - Henriksson, R.D., & Merton, R.C. (1981). "On Market Timing and Investment Performance II: Statistical
        Procedures for Evaluating Forecasting Skills." Journal of Business, 54(4), 513-533.
        - Bacon, C.R. (2008). "Practical Portfolio Performance Measurement and Attribution."
        2nd edition, John Wiley & Sons.
        - PerformanceAnalytics: CAPM.beta.bull, CAPM.beta.bear and TimingRatio
        (https://github.com/braverock/PerformanceAnalytics).
        - empyrical: up_alpha_beta and down_alpha_beta (https://github.com/stefan-jansen/empyrical-reloaded).

        Also known as: up-market and down-market beta, conditional beta, up and down alpha.

        Args:
            period (str, optional): The period to use for the calculation. Defaults to "quarterly" if the Toolkit is
                initialised with quarterly=True, otherwise "yearly".
            market (str, optional): What splits the bull and bear days: "excess_return" for the sign of the
                benchmark excess return (PerformanceAnalytics) or "return" for the sign of the benchmark return
                (empyrical). Defaults to "excess_return".
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
        it is given, which market="return" does with the benchmark's own return; given a risk free rate, empyrical
        computes its Beta from the returns themselves and its Alpha from the excess returns, where the Toolkit uses
        excess returns for both.
        - Unlike `get_beta` with method="upside" or "downside", which regresses the returns themselves, these
        regressions use excess returns, so the Alphas are the CAPM (Jensen's) Alphas of each market.
        - Each side needs at least three days with returns in a period, otherwise its values are NaN.
        - There is no Portfolio column, as for the market timing models.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

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
        if market not in ("excess_return", "return"):
            raise ValueError("The market must be excess_return or return.")

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
                    )
                    for sub_period in excess_return.index.get_level_values(0).unique()
                },
                orient="index",
            )
            for ticker in excess_return.columns
        }

        return finalize_dataset(
            dataset=pd.concat(bull_bear_beta, axis=1),
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

        For more information, see the following sources:

        - Taleb, N.N., Canetti, E., Kinda, T., Loukoianova, E., & Schmieder, C. (2012). "A New Heuristic Measure
        of Fragility and Tail Risks: Application to Stress Testing." IMF Working Paper WP/12/216.
        - empyrical: beta_fragility_heuristic (https://github.com/stefan-jansen/empyrical-reloaded).

        Also known as: fragility heuristic, beta fragility heuristic.

        Args:
            period (str, optional): The period to use for the calculation. Defaults to "quarterly" if the Toolkit is
                initialised with quarterly=True, otherwise "yearly".
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
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

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

        For more information, see the following sources:

        - empyrical: stability_of_timeseries (https://github.com/stefan-jansen/empyrical-reloaded).
        - pyfolio: timeseries.stability_of_timeseries (https://github.com/stefan-jansen/pyfolio-reloaded).

        Also known as: stability of timeseries, R-squared of the equity curve.

        Args:
            period (str, optional): The period to use for the calculation. Defaults to "quarterly" if the Toolkit is
                initialised with quarterly=True, otherwise "yearly".
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
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

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
            # Over the whole period the stability is one number per ticker, with no dates to slice.
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

        For more information, see the following sources:

        - Bacon, C.R. (2008). "Practical Portfolio Performance Measurement and Attribution."
        2nd edition, John Wiley & Sons.
        - quantstats: stats.r_squared (https://github.com/ranaroussi/quantstats).

        Also known as: coefficient of determination, R2.

        Args:
            period (str, optional): The period to use for the calculation. Defaults to "quarterly" if the Toolkit is
                initialised with quarterly=True, otherwise "yearly".
            rolling (int, optional): The rolling window size to use for the calculation. If you select
                period = 'monthly' and set rolling to 36 you obtain the rolling three year R-squared of the
                monthly returns. Defaults to None, which uses the daily returns within each period.
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
        - The R Squared of `get_henriksson_merton_model` and `get_treynor_mazuy_model` includes their timing term,
        so it is at least this one.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

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
        returns = self._get_column(period, "Return", within_period=not rolling)
        benchmark_returns = self._get_column(
            period, "Return", within_period=not rolling, benchmark=True
        )

        if rolling:
            r_squared = performance_model.get_rolling_r_squared(
                returns, benchmark_returns, rolling
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
        observation; set annualize=True to multiply it by the square root of the observations in a year (252
        daily returns, 365 for a ticker that trades on weekends, or 52, 12, 4 or 1 for rolling weekly, monthly,
        quarterly or yearly returns). Leaving out the risk free rate makes it comparable across currencies and
        rate environments, but it overstates the reward of a low-risk asset that earns little more than cash;
        `get_sharpe_ratio` subtracts the risk free rate.

        The ratio under this name is defined in quantstats (Ran Aroussi) and ffn (Philippe Morissette).

        For more information, see the following sources:

        - quantstats: stats.risk_return_ratio (https://github.com/ranaroussi/quantstats).
        - ffn: calc_risk_return_ratio (https://github.com/pmorissette/ffn).

        Also known as: return to risk ratio, return over volatility, information ratio against cash.

        Args:
            period (str, optional): The period to use for the calculation. Defaults to "quarterly" if the Toolkit is
                initialised with quarterly=True, otherwise "yearly".
            rolling (int, optional): The rolling window size to use for the calculation. If you select
                period = 'monthly' and set rolling to 12 you obtain the rolling 12-month ratio. Defaults to None.
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
        - Equals quantstats' risk_return_ratio (not annualized) and, with annualize=True, ffn's
        calc_risk_return_ratio on the same returns. Returns without variation give NaN.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

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
        returns = self._get_column(period, "Return", within_period=not rolling)

        # The Sharpe ratio of the returns themselves, without a risk free rate.
        if rolling:
            risk_return_ratio = performance_model.get_rolling_sharpe_ratio(
                returns, rolling
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

        For more information, see the following sources:

        - Bacon, C.R. (2008). "Practical Portfolio Performance Measurement and Attribution."
        2nd edition, John Wiley & Sons, pp. 71-72.
        - PerformanceAnalytics: SystematicRisk, SpecificRisk, TotalRisk and table.SpecificRisk
        (https://github.com/braverock/PerformanceAnalytics).

        Also known as: market and idiosyncratic risk, diversifiable and non-diversifiable risk, residual risk.

        Args:
            period (str, optional): The period to use for the calculation. Defaults to "quarterly" if the Toolkit is
                initialised with quarterly=True, otherwise "yearly".
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
        - There is no Portfolio column: the risks of a portfolio do not add up as a weighted average.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

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
        - PerformanceAnalytics: CAPM.CML.slope, CAPM.CML, CAPM.SML.slope and CAPM.epsilon
        (https://github.com/braverock/PerformanceAnalytics).

        Also known as: CML, SML, capital allocation line of the market, regression epsilon.

        Args:
            period (str, optional): The period to use for the calculation. Defaults to "quarterly" if the Toolkit is
                initialised with quarterly=True, otherwise "yearly".
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
        - There is no Portfolio column, as for the market timing models.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

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
        Calculate the Kappa Ratio of an investment portfolio or asset's returns.

        The Kappa Ratio is a generalization of the Sortino Ratio that penalizes downside risk
        using a higher-order lower partial moment. The Sortino Ratio is the special case of the
        Kappa Ratio with order=2.

        Note that this already subtracts the Risk Free Rate.

        The formula is as follows:

        - Daily Excess Return = Daily Return - Daily Risk Free Rate, with Daily Risk Free Rate =
          (1 + Annual Risk Free Rate)^(1/252) - 1 (compounded over the calendar days since the previous return,
          on 365 days a year, when a ticker trades on weekends)
        - Lower Partial Moment (LPM) = MEAN(MAX(-Daily Excess Return, 0)^Order), averaged over all days in the
          period (days with a non-negative excess return count as 0)
        - Kappa Ratio = MEAN(Daily Excess Return) / LPM^(1/Order), per period and not annualized, with Order
          defaulting to 3

        Args:
            period (str, optional): The period to use for the calculation. Defaults to "quarterly" if the Toolkit is
                initialised with quarterly=True, otherwise "yearly".
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
        - The risk-free rate is the Toolkit's risk_free_rate, the 10-year US Treasury yield by default. Many
        textbook definitions (Sharpe, 1994; the Ken French factors) use a short-term Treasury bill instead, which
        the Toolkit uses when initialised with risk_free_rate="13w".
        - If `growth` is set to True, the method calculates the growth of the ratio values using
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
        Calculate the Omega Ratio of an investment portfolio or asset's returns.

        The Omega Ratio is a risk-return measure that divides the sum of gains above a minimum
        acceptable return (MAR) by the sum of losses below it, capturing the full shape of the
        return distribution rather than only its first two moments (unlike the Sharpe Ratio).

        The formula is as follows:

        - Omega Ratio = SUM(Gains above MAR) / SUM(Losses below MAR)

        See definition: https://en.wikipedia.org/wiki/Omega_ratio

        Args:
            period (str, optional): The period to use for the calculation. Defaults to "quarterly" if the Toolkit is
                initialised with quarterly=True, otherwise "yearly".
            within_period (bool, optional): Whether to calculate the Omega Ratio within the specified
            period or for the entire period. Thus whether to look at the Omega Ratio within a specific
            year (if period = 'yearly') or look at the entirety of all years. Defaults to True.
            rolling (int, optional): The rolling window size to use for the calculation. If set, the
            Omega Ratio is calculated over a rolling window of this many periods across the full
            return history instead of per `period`. Defaults to None.
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
        - If `growth` is set to True, the method calculates the growth of the ratio values using
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

        if rolling:
            returns = self._get_column(period, "Return", within_period=False)
            omega_ratio = performance_model.get_rolling_omega_ratio(
                returns, rolling, minimum_acceptable_return
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
            # Over the whole period the ratio is one number per ticker, with no dates to slice.
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
            raise ValueError(
                "The Morningstar Risk-Adjusted Return groups monthly returns, so the period "
                "must be yearly or quarterly."
            )

        excess_returns = performance_model.get_geometric_excess_return(
            self._get_column("monthly", "Return", within_period=False),
            self._risk_free_rate_data["monthly"][self._tickers_without_portfolio],
        )

        if rolling:
            return performance_model.get_rolling_morningstar_risk_adjusted_return(
                excess_returns, rolling, gamma
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
        Calculate the Morningstar Risk-Adjusted Return (MRAR), the measure Morningstar ranks funds
        on within their category for its star rating.

        MRAR is the annualized certainty equivalent of the monthly geometric excess returns for
        an investor with risk aversion gamma: the return such an investor would accept with
        certainty instead of the uncertain returns. It equals the annualized geometric mean
        excess return when every month returns the same and falls further below it the more
        the returns vary, penalizing downside variation most. Morningstar uses a gamma of 2 and
        rates funds on 36, 60 and 120 months (three, five and ten years), which rolling=36,
        rolling=60 and rolling=120 give, weighting them 20%, 30% and 50% in the overall rating.
        For US dollar funds its risk-free rate is the three-month Treasury bill (auction
        average), which risk_free_rate="13w" follows. Morningstar also adjusts for
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
            period (str, optional): The period to group the monthly returns by, "yearly" or
                "quarterly". Defaults to "yearly".
            within_period (bool, optional): Whether to calculate the MRAR of the months within
                each period or of all months in the range. Defaults to True.
            rolling (int, optional): The number of months in a rolling window, e.g. 36 for the
                three year MRAR at every month end. Overrides period and within_period.
                Defaults to None.
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
            # Over the whole range the measure is one number per ticker, with no dates to slice.
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
        (rolling=36, 60 and 120).

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
            period (str, optional): The period to group the monthly returns by, "yearly" or
                "quarterly". Defaults to "yearly".
            within_period (bool, optional): Whether to calculate the risk of the months within
                each period or of all months in the range. Defaults to True.
            rolling (int, optional): The number of months in a rolling window, e.g. 36 for the
                three year figure at every month end. Overrides period and within_period.
                Defaults to None.
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
        Calculate the Gain-to-Pain Ratio of an investment portfolio or asset's returns.

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
        - PerformanceAnalytics: OmegaSharpeRatio (https://github.com/braverock/PerformanceAnalytics).
        - quantstats: stats.gain_to_pain_ratio (https://github.com/ranaroussi/quantstats).

        Also known as: Omega-Sharpe ratio.

        Args:
            period (str, optional): The period to use for the calculation. Defaults to "quarterly" if the Toolkit is
                initialised with quarterly=True, otherwise "yearly".
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
        - If `growth` is set to True, the method calculates the growth of the ratio values using
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
            # Over the whole period the ratio is one number per ticker, with no dates to slice.
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

        For more information, see the following sources:

        - Bacon, C.R. (2008). "Practical Portfolio Performance Measurement and Attribution."
        2nd edition, John Wiley & Sons, p. 95.
        - PerformanceAnalytics: DRatio (https://github.com/braverock/PerformanceAnalytics).

        Args:
            period (str, optional): The period to use for the calculation. Defaults to "quarterly" if the Toolkit is
                initialised with quarterly=True, otherwise "yearly".
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
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

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
            # Over the whole period the ratio is one number per ticker, with no dates to slice.
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
        - PerformanceAnalytics: ProspectRatio (https://github.com/braverock/PerformanceAnalytics).

        Args:
            period (str, optional): The period to use for the calculation. Defaults to "quarterly" if the Toolkit is
                initialised with quarterly=True, otherwise "yearly".
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
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

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
            # Over the whole period the ratio is one number per ticker, with no dates to slice.
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

        For more information, see the following sources:

        - Bacon, C.R. (2008). "Practical Portfolio Performance Measurement and Attribution."
        2nd edition, John Wiley & Sons, p. 103.
        - PerformanceAnalytics: OmegaExcessReturn (https://github.com/braverock/PerformanceAnalytics).

        Args:
            period (str, optional): The period to use for the calculation. Defaults to "quarterly" if the Toolkit is
                initialised with quarterly=True, otherwise "yearly".
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
        method: str = "discrete",
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

        Args:
            period (str, optional): The period to use for the calculation. Defaults to "quarterly" if the Toolkit is
                initialised with quarterly=True, otherwise "yearly".
            within_period (bool, optional): Whether to calculate the Kelly Criterion from the daily
                returns within each period (e.g. each year if period = 'yearly') or from the returns
                of the periods themselves over the whole history. Defaults to True.
            rolling (int, optional): The rolling window size to use for the calculation. If set, the
                Kelly Criterion is calculated over a rolling window of this many periods across the
                full return history instead of per `period`. Defaults to None.
            method (str, optional): "discrete" (Kelly's bet from the win rate and payoff ratio) or
                "continuous" (average excess return over its variance). Defaults to "discrete".
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
        - There is no Portfolio row: a weighted average of each asset's Kelly Criterion is not the
        Kelly Criterion of the portfolio.

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
        if method not in ("discrete", "continuous"):
            raise ValueError("Please choose a method of 'discrete' or 'continuous'.")

        period = period if period else "quarterly" if self._quarterly else "yearly"

        if rolling:
            returns = self._get_column(period, "Return", within_period=False)

            if method == "continuous":
                returns = performance_model.get_excess_return(
                    returns, self._risk_free_rate_data[period][returns.columns]
                )

            kelly_criterion = performance_model.get_rolling_kelly_criterion(
                returns, rolling, method, fraction
            )
        else:
            column = "Excess Return" if method == "continuous" else "Return"

            if method == "continuous" and not within_period:
                returns = performance_model.get_excess_return(
                    self._get_column(period, "Return", within_period=False),
                    self._risk_free_rate_data[period][self._tickers_without_portfolio],
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
            # Over the whole period the criterion is one number per ticker, with no dates to slice.
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
        returns and outliers, and how precisely its mean return is known.

        Unlike the Win Rate of `get_win_rate`, which counts the periods an asset beats its
        benchmark, these statistics are absolute: a win is a positive return.

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

        For more information, see the following sources:

        - Tharp, V.K. (2006). "Trade Your Way to Financial Freedom." 2nd ed., McGraw-Hill.
        - quantstats: stats.avg_return, best, worst, exposure, cpc_index, profit_ratio, outlier_win_ratio,
        outlier_loss_ratio, risk_of_ruin and rar (https://github.com/ranaroussi/quantstats).
        - ffn: PerformanceStats best_day, worst_day, best_month, worst_month, best_year and worst_year
        (https://github.com/pmorissette/ffn).
        - PerformanceAnalytics: mean.stderr, mean.LCL and mean.UCL (https://github.com/braverock/PerformanceAnalytics).

        Args:
            period (str, optional): The period of the returns over the whole history, or with
                within_period=True the period the daily returns are grouped by. Defaults to "daily",
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
        - There is no Portfolio column: a weighted average of streak lengths or payoff ratios is
        not the statistic of the portfolio.

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
        This function calculates the Compound Growth Rate (CGR) for different periods: yearly, quarterly, monthly,
        weekly, and daily.

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
        log: bool = False,
        unsmoothed: bool = False,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Return of an investment portfolio or asset for a given period
        based on the daily historical returns.

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

        With log=True the returns are log (continuously compounded) returns, which add up over
        time where simple returns compound:

        - Log Return = ln(1 + Period Return)
        - Cumulative Log Return = ln(Cumulative Return), the sum of the log returns

        With unsmoothed=True the period returns are unsmoothed with Geltner's filter first, for
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
        - ffn: to_log_returns and PerformanceStats.log_returns (https://github.com/pmorissette/ffn).
        - pandas-ta-classic: log_return (https://github.com/xgboosted/pandas-ta-classic).
        - PerformanceAnalytics: Return.Geltner (https://github.com/braverock/PerformanceAnalytics).

        Also known as: periodic return, log return, continuously compounded return, unsmoothed
        return, Geltner return, de-smoothed return.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "quarterly" if the Toolkit is initialised with quarterly=True, otherwise "yearly".
            cumulative (bool, optional): Whether to return the cumulative return over time
            instead of the discrete return per period. Defaults to False.
            log (bool, optional): Whether to return log returns, ln(1 + return), or with
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
        - If `growth` is set to True, the method calculates the growth of Return values using the specified `lag`.

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

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )

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

        if log:
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
        Calculate the Excess Return of an investment portfolio or asset for a given period
        based on the daily historical returns.

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

        Args:
            period (str, optional): The data frequency for returns (weekly, monthly, quarterly, or yearly). Defaults
                to "quarterly" if the Toolkit is initialised with quarterly=True, otherwise "yearly".
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
        - The risk-free rate is the Toolkit's risk_free_rate, the 10-year US Treasury yield by default. Many
        textbook definitions (Sharpe, 1994; the Ken French factors) use a short-term Treasury bill instead, which
        the Toolkit uses when initialised with risk_free_rate="13w".
        - If `growth` is set to True, the method calculates the growth of Excess Return values using the
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

        if period not in ["weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError("Period must be weekly, monthly, quarterly, or yearly.")

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
            period (str, optional): The data frequency for returns (weekly, monthly, quarterly, or yearly). Defaults
                to "quarterly" if the Toolkit is initialised with quarterly=True, otherwise "yearly".
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

        returns = self._historical_data[period]["Return"].loc[
            self._start_date : self._end_date
        ]

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
          covariance (ddof=1) of the period returns (e.g. monthly) within the Toolkit's date range, per period
          and not annualized, using the overlapping (pairwise complete) observations of each pair

        Args:
            period (str, optional): The data frequency for returns (weekly, monthly, quarterly, or yearly). Defaults
                to "quarterly" if the Toolkit is initialised with quarterly=True, otherwise "yearly".
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

        returns = self._historical_data[period]["Return"].loc[
            self._start_date : self._end_date
        ]

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
        method: str = "appraisal",
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

        Jensen's Alpha (see `get_jensens_alpha`) measures how much return a manager
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
        reusing the exact same CAPM regression formula as `get_jensens_alpha`.

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
        - PerformanceAnalytics: AppraisalRatio (https://github.com/braverock/PerformanceAnalytics).

        Also known as: Treynor-Black Appraisal Ratio.

        Args:
            period (str, optional): The period to use for the calculation. Defaults to "quarterly" if the Toolkit is
                initialised with quarterly=True, otherwise "yearly".
            rolling (int, optional): The rolling window size to use for the calculation. If set, Beta and
            the residual standard deviation are estimated over a rolling window of this many periods
            across the full return history instead of per `period`, the residuals of each window from
            the regression over that window (see `performance_model.get_rolling_appraisal_ratio`).
            Defaults to None.
            method (str, optional): "appraisal" for Jensen's Alpha per unit of residual risk,
            "modified" for Jensen's Alpha per unit of Beta or "alternative" for Jensen's Alpha per
            unit of systematic risk, as described above. Defaults to "appraisal".
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
        CAPM formula as `get_jensens_alpha`.
        - PerformanceAnalytics' AppraisalRatio annualizes the returns over the whole sample and takes the
        population standard deviation of the residuals; the Toolkit uses the return, Beta and residuals of
        each period. The three methods follow the same formulas.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

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
        if method not in ("appraisal", "modified", "alternative"):
            raise ValueError("Method must be appraisal, modified, or alternative.")

        period = period if period else "quarterly" if self._quarterly else "yearly"

        if rolling:
            returns = self._get_column(period, "Return", within_period=False)
            benchmark_returns = self._get_column(
                period, "Return", within_period=False, benchmark=True
            )

            beta = performance_model.get_rolling_beta(
                returns, benchmark_returns, rolling
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
            performance_model.get_rolling_appraisal_ratio(
                jensens_alpha, excess_return, benchmark_excess_return, rolling
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
        Calculate the Fama (1972) decomposition of total excess return into Selectivity
        and Diversification.

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
        Diversification equals Jensen's Alpha (see `get_jensens_alpha`).

        Also known as: Fama's Net Selectivity, Fama performance decomposition.

        Args:
            period (str, optional): The period to use for the calculation. Defaults to "quarterly" if the Toolkit is
                initialised with quarterly=True, otherwise "yearly".
            rolling (int, optional): The rolling window size to use for the calculation. If set, Beta and
            the standard deviations of the asset and the benchmark are estimated from the returns per
            `period` over a rolling window of this many periods across the full return history,
            instead of from the daily returns within each `period`. Defaults to None.
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
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

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

        if rolling:
            returns = self._get_column(period, "Return", within_period=False)
            benchmark_returns = self._get_column(
                period, "Return", within_period=False, benchmark=True
            )

            beta = performance_model.get_rolling_beta(
                returns, benchmark_returns, rolling
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
            asset_standard_deviation = performance_model.get_rolling_standard_deviation(
                returns, rolling, benchmark_returns
            )
            benchmark_standard_deviation = (
                performance_model.get_rolling_standard_deviation(
                    benchmark_returns, rolling
                )
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
            self._tickers_without_portfolio, level=0, axis=1
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
        Calculate the STARR (Stable Tail Adjusted Return Ratio) of an investment
        portfolio or asset's returns.

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

        Args:
            period (str, optional): The period to use for the calculation. Defaults to "quarterly" if the Toolkit is
                initialised with quarterly=True, otherwise "yearly".
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
        - If `growth` is set to True, the method calculates the growth of the ratio values using
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
        Calculate the Rachev Ratio (R-Ratio) of an investment portfolio or asset's
        returns.

        The Rachev ratio compares the "quality" of the best outcomes to the "quality" of
        the worst outcomes by taking the ratio of the right-tail Expected Shortfall (the
        average of the best `alpha` fraction of returns) to the left-tail Expected
        Shortfall (the average magnitude of the worst `alpha` fraction of returns). A
        ratio above 1 indicates that the average size of extreme gains outweighs the
        average size of extreme losses.

        The formula is as follows:

        - Rachev Ratio = ES_right(alpha) / ES_left(alpha)

        Also known as: R-Ratio.

        Args:
            period (str, optional): The period to use for the calculation. Defaults to "quarterly" if the Toolkit is
                initialised with quarterly=True, otherwise "yearly".
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
        - If `growth` is set to True, the method calculates the growth of the ratio values using
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
            # Over the whole period the ratio is one number per ticker, with no dates to slice.
            apply_slice=within_period,
            dropna=True,
        )

    @handle_portfolio
    @handle_errors
    def get_treynor_mazuy_model(
        self,
        period: str | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Treynor-Mazuy market timing model for each asset in the Toolkit
        instance.

        Jensen's Alpha and Beta from a plain CAPM regression cannot distinguish
        stock-picking skill (selectivity) from market-timing skill (shifting exposure
        ahead of market moves). The Treynor-Mazuy model adds a quadratic term in the
        benchmark excess return to the regression: a manager who successfully increases
        (decreases) market exposure ahead of up (down) markets will show a return profile
        that curves upward as a function of the benchmark return, captured by a positive
        quadratic coefficient (Gamma).

        The formula is as follows:

        - Excess Return = Alpha + Beta * Benchmark Excess Return + Gamma * Benchmark Excess Return^2 + Residuals

        Gamma > 0 indicates positive market-timing ability; Gamma <= 0 indicates no
        timing ability.

        Also known as: Treynor-Mazuy quadratic timing model, TM model.

        Args:
            period (str, optional): The period to use for the calculation. Defaults to "quarterly" if the Toolkit is
                initialised with quarterly=True, otherwise "yearly".
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
            pd.DataFrame: Alpha, Beta, Gamma and R Squared values, with a Multi Index of
            (ticker, parameter) as the columns.

        Notes:
        - Daily and weekly Treynor-Mazuy results are not an option as there would be too few
        observations within each period to run a meaningful regression.
        - The method retrieves historical data and performs a quadratic regression for each asset
        in the Toolkit instance, within each period.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["ARKK", "JEPI"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.performance.get_treynor_mazuy_model().xs("ARKK", level=0, axis=1)
        ```

        Which returns:

        |      |   Alpha |   Beta |    Gamma |   R Squared |
        |:-----|--------:|-------:|---------:|------------:|
        | 2021 | -0.0034 | 1.6646 |  12.4821 |      0.2937 |
        | 2022 | -0.0035 | 2.1567 |   6.9512 |      0.5959 |
        | 2023 |  0.0015 | 2.2614 | -14.8854 |      0.529  |
        | 2024 | -0.001  | 2.1368 |  -3.1566 |      0.5658 |
        | 2025 |  0.0006 | 1.8994 |  -1.3437 |      0.6767 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"
        excess_return = self._get_column(period, "Excess Return", within_period=True)
        benchmark_excess_return = self._get_column(
            period, "Excess Return", within_period=True, benchmark=True
        )

        logger.info("Calculating Treynor-Mazuy Market Timing Model")

        regression_results: dict[str, pd.DataFrame] = {}

        for ticker in self._tickers_without_portfolio:
            ticker_results: dict = {}

            for sub_period in excess_return.index.get_level_values(0).unique():
                asset_excess = excess_return.loc[sub_period, ticker]
                benchmark_excess = benchmark_excess_return.loc[sub_period]

                result, _ = performance_model.get_treynor_mazuy_model(
                    asset_excess, benchmark_excess
                )
                ticker_results[sub_period] = result

            regression_results[ticker] = pd.DataFrame.from_dict(
                ticker_results, orient="index"
            )

        treynor_mazuy_model = pd.concat(regression_results, axis=1)

        return finalize_dataset(
            dataset=treynor_mazuy_model,
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
    def get_henriksson_merton_model(
        self,
        period: str | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Henriksson-Merton market timing model for each asset in the
        Toolkit instance.

        Like the Treynor-Mazuy model (see `get_treynor_mazuy_model`), this separates
        market-timing skill from selectivity, but models timing as a piecewise (rather
        than quadratic) change in Beta: a "down-market" Beta and an "up-market" Beta.

        The formula is as follows:

        - Excess Return = Alpha + Beta * Benchmark Excess Return +
          Up Market Beta * max(Benchmark Excess Return, 0) + Residuals

        Beta is the "down-market" Beta (the portfolio's market exposure when the
        benchmark excess return is negative), and Beta + Up Market Beta is the
        "up-market" Beta. Up Market Beta > 0 indicates positive market-timing ability;
        Up Market Beta <= 0 indicates no timing ability.

        Also known as: Henriksson-Merton piecewise timing model, HM model.

        Args:
            period (str, optional): The period to use for the calculation. Defaults to "quarterly" if the Toolkit is
                initialised with quarterly=True, otherwise "yearly".
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
            pd.DataFrame: Alpha, Beta, Up Market Beta and R Squared values, with a Multi Index of
            (ticker, parameter) as the columns.

        Notes:
        - Daily and weekly Henriksson-Merton results are not an option as there would be too few
        observations within each period to run a meaningful regression.
        - The method retrieves historical data and performs a piecewise regression for each asset
        in the Toolkit instance, within each period.
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["BRK-B", "FBCG"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.performance.get_henriksson_merton_model().xs("FBCG", level=0, axis=1)
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
        period = period if period else "quarterly" if self._quarterly else "yearly"
        excess_return = self._get_column(period, "Excess Return", within_period=True)
        benchmark_excess_return = self._get_column(
            period, "Excess Return", within_period=True, benchmark=True
        )

        logger.info("Calculating Henriksson-Merton Market Timing Model")

        regression_results: dict[str, pd.DataFrame] = {}

        for ticker in self._tickers_without_portfolio:
            ticker_results: dict = {}

            for sub_period in excess_return.index.get_level_values(0).unique():
                asset_excess = excess_return.loc[sub_period, ticker]
                benchmark_excess = benchmark_excess_return.loc[sub_period]

                result, _ = performance_model.get_henriksson_merton_model(
                    asset_excess, benchmark_excess
                )
                ticker_results[sub_period] = result

            regression_results[ticker] = pd.DataFrame.from_dict(
                ticker_results, orient="index"
            )

        henriksson_merton_model = pd.concat(regression_results, axis=1)

        return finalize_dataset(
            dataset=henriksson_merton_model,
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
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Factor analysis (predictive power) of the performance metrics, or of a factor of your own: whether the tickers that score
        high on a metric go on to return more than those that score low. Per period, the
        information coefficient (IC) is the Spearman rank correlation between the metric of the
        tickers in the period and their return over the following horizon: 1 means the metric
        ranks the later returns perfectly, -1 in reverse and 0 that it says nothing about them.
        The IC measures skill in Grinold and Kahn's fundamental law of active management,
        IR ≈ IC × √breadth: a small but consistent IC across many tickers and periods gives a
        high information ratio.

        The summary has a row per metric and horizon with the mean IC, its standard deviation,
        the IC information ratio (mean divided by standard deviation), the t-statistic of the
        mean with its p-value, the share of periods with a positive IC, the number of periods,
        the top minus bottom spread (the mean forward return of the highest quantile minus
        that of the lowest), the mean turnover of the top and bottom quantiles and the mean
        rank autocorrelation of the metric from one period to the next. With detail, it
        returns one of those parts per period (or per quantile) for a single metric.

        Each metric is measured within each period (e.g. the volatility of a month's daily
        returns) and compared with the returns from the close of the period's last trading day
        onwards, using the dividend and split adjusted close, so a metric never overlaps the
        returns it is compared with. Each ticker keeps its own history: a ticker without a
        value or a return in a period is left out of that period only.

        A factor of your own is a DataFrame with the dates as index and the tickers as
        columns, applied from each date on, or with the tickers as index and periods as
        columns, applied from the end of each period. Make sure each value was known on the
        date it is given for, or the analysis looks ahead.

        Args:
            metrics (str | list[str] | pd.DataFrame): A performance method name with a value per ticker and
                period, e.g. "get_sharpe_ratio", a list of them, "all", or a factor of your own
                as a DataFrame. "all" holds the returns, excess return, beta, Sharpe, Sortino,
                Treynor and information ratios, tracking error, Jensen's alpha, Calmar and
                Omega ratios and the win rate. Defaults to "all".
            period (str, optional): The frequency at which the metrics are measured and
                evaluated: weekly, monthly, quarterly or yearly. Each metric is calculated
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
                "quantile_returns" (the mean forward return per quantile and horizon),
                "quantile_turnover" (per period and quantile) or "rank_autocorrelation" (per
                period). Defaults to None.
            column (str | None, optional): The output to use of a metric with several.
                Defaults to None, which uses the first.
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

        # Momentum: the return over the previous twelve months, without the latest month.
        prices = toolkit.get_historical_data()["Adj Close"]
        momentum = (prices.shift(21) / prices.shift(252) - 1).drop(columns="Benchmark")

        toolkit.performance.get_factor_analysis(momentum, horizons=[1, 12])
        ```

        Which returns:

        |                                |   Mean IC |   IC Standard Deviation |   IC Information Ratio |   t-Statistic |   p-Value |   Positive Share |   Observations |   Top Minus Bottom |   Quantile Turnover |   Rank Autocorrelation |
        |:-------------------------------|----------:|------------------------:|-----------------------:|--------------:|----------:|-----------------:|---------------:|-------------------:|--------------------:|-----------------------:|
        | ('Custom Factor', '1 Month')   |    0.018  |                  0.3745 |                 0.0481 |        0.498  |    0.6195 |           0.4953 |            107 |             0.004  |              0.2391 |                 0.8822 |
        | ('Custom Factor', '12 Months') |    0.1122 |                  0.419  |                 0.2678 |        2.6242 |    0.0101 |           0.5729 |             96 |             0.0195 |              0.2391 |                 0.8822 |
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

            - "iid": SE = SQRT( (1 + SR^2 / 2) / n ), for normally and independently distributed
              returns (Lo, 2002, equation 9)
            - "non_normal": SE = SQRT( (1 − γ₃·SR + ((γ₄ − 1) / 4)·SR^2) / (n − 1) ), for
              independent but skewed and fat-tailed returns, with γ₃ the skewness and γ₄ the
              non-excess (raw) kurtosis (Mertens, 2002; the standard error inside the
              Probabilistic Sharpe Ratio of Bailey and López de Prado, 2012)
            - "gmm": SE = SQRT( ∂g/∂θ · Σ · ∂g/∂θ' / n ), for returns that may also be serially
              correlated and heteroskedastic (Lo, 2002, Appendix). With θ = (μ, σ²) and
              g(θ) = μ / σ, ∂g/∂θ = (1 / σ, −μ / (2σ³)) and Σ is the Newey and West (1987)
              long-run covariance of the moment conditions (r_t − μ, (r_t − μ)² − σ²).

        Negative skewness, fat tails and positive autocorrelation all widen the standard error, so
        the same Sharpe ratio is then less certain. The standard error is on the scale of the
        Sharpe ratio of one observation; multiply by SQRT(252) to compare it with a Sharpe ratio
        annualized by SQRT(252) (exact for the "iid" and "non_normal" errors, a rescaling of the
        same estimate for "gmm").

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
            period (str, optional): The period to use for the calculation. Defaults to "quarterly" if the Toolkit is
                initialised with quarterly=True, otherwise "yearly".
            method (str, optional): "iid", "non_normal" or "gmm", as described above. Defaults
                to "iid".
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
        `get_sharpe_ratio`; period="daily" needs intraday data (initialise the Toolkit with an
        intraday_period).
        - The "non_normal" error uses the bias-corrected sample skewness and kurtosis, as
        `get_sharpe_ratio(method="probabilistic")` does; the "gmm" error uses the moments of
        GMM (σ² with 1 / n).
        - If `growth` is set to True, the method calculates the growth of the values using the specified `lag`.

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
        Sharpe Ratio, `get_sharpe_ratio(method="probabilistic")`, equals `confidence_level`.
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

        Args:
            period (str, optional): The period to use for the calculation. Defaults to "quarterly" if the Toolkit is
                initialised with quarterly=True, otherwise "yearly".
            rolling (int, optional): The rolling window size, in observations of `period`.
                Defaults to None, one value per period.
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
        `get_sharpe_ratio(method="probabilistic")`.
        - period="daily" without rolling needs intraday data (initialise the Toolkit with an
        intraday_period).
        - If `growth` is set to True, the method calculates the growth of the values using the specified `lag`.

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

        if rolling:
            excess_return = performance_model.get_excess_return(
                self._get_column(period, "Return", within_period=False),
                self._risk_free_rate_data[period][self._tickers_without_portfolio],
            )
            sharpe_ratio = performance_model.get_rolling_sharpe_ratio(
                excess_return, rolling
            )
            skewness = get_rolling_skewness(excess_return, rolling)
            kurtosis = get_rolling_kurtosis(excess_return, rolling, fisher=False)
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
        method: str = "standardized",
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

            - "standardized": Co-Skewness = E[(r_i − μ_i)(r_m − μ_m)^2] / (σ_i · σ_m^2)
            - "residual": β_SKD = E[ε_i · ε_m^2] / (SQRT(E[ε_i^2]) · E[ε_m^2]) (Harvey and
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

        Args:
            period (str, optional): The period to use for the calculation. Defaults to "quarterly" if the Toolkit is
                initialised with quarterly=True, otherwise "yearly".
            method (str, optional): "standardized" or "residual", as described above. Defaults to
                "standardized".
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
        - period="daily" needs intraday data (initialise the Toolkit with an intraday_period).
        - If `growth` is set to True, the method calculates the growth of the values using the specified `lag`.

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
        method: str = "standardized",
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

            - "standardized": Co-Kurtosis = E[(r_i − μ_i)(r_m − μ_m)^3] / (σ_i · σ_m^3)
            - "residual": E[ε_i · ε_m^3] / (SQRT(E[ε_i^2]) · E[ε_m^2]^(3/2))

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

        Args:
            period (str, optional): The period to use for the calculation. Defaults to "quarterly" if the Toolkit is
                initialised with quarterly=True, otherwise "yearly".
            method (str, optional): "standardized" or "residual", as described above. Defaults to
                "standardized".
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
        - period="daily" needs intraday data (initialise the Toolkit with an intraday_period).
        - If `growth` is set to True, the method calculates the growth of the values using the specified `lag`.

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
        - quantstats: stats.serenity_index (https://github.com/ranaroussi/quantstats).

        Also known as: Serenity Ratio.

        Args:
            period (str, optional): The period to use for the calculation. Defaults to "quarterly" if the Toolkit is
                initialised with quarterly=True, otherwise "yearly".
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
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

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

        The risk measures (`risk_measure`) are those of `toolkit.risk`, as a positive number:

        - "var": the historical Value at Risk, the `alpha` quantile of the returns.
        - "cvar": the historical Conditional Value at Risk (Expected Shortfall), the mean of the returns at or
          below it (the Conditional Sharpe or STARR ratio).
        - "evar": the historical Entropic Value at Risk.
        - "erm": the Entropic Risk Measure with a risk aversion of 1.
        - "worst": the Worst Realization, the lowest return.
        - "mad": the Mean Absolute Deviation from the mean.
        - "lpm": the first Lower Partial Moment below the mean, MEAN(MAX(Mean - Return, 0)).
        - "gmd": the Gini Mean Difference.
        - "max_drawdown": the Maximum Drawdown (a Calmar ratio).
        - "average_drawdown": the Average Drawdown over every day.
        - "dar", "cdar" and "edar": the Drawdown at Risk, Conditional Drawdown at Risk and Entropic Drawdown
          at Risk at `alpha`.
        - "ulcer": the Ulcer Index, the root mean square drawdown.

        The ratio is computed from the daily returns within each period (e.g. each year), with the mean and
        the risk measure per day and not annualized, as skfolio's ratio measures are. With within_period=False
        the returns of `period` over the whole history give one ratio per ticker. The drawdowns compound the
        returns and start from a wealth of 1 at the start of each period.

        For more information, see the following sources:

        - Cajas, D. (2021). "Riskfolio-Lib: Portfolio Optimization in Python", the risk-adjusted return ratios
        (https://riskfolio-lib.readthedocs.io).
        - Gregoriou, G.N., & Gueyie, J.P. (2003). "Risk-Adjusted Performance of Funds of Hedge Funds Using a
        Modified Sharpe Ratio." Journal of Wealth Management, 6(3), 77-83.
        - skfolio: RatioMeasure (https://skfolio.org/generated/skfolio.measures.RatioMeasure.html).
        - Riskfolio-Lib: RiskFunctions.Sharpe (https://github.com/dcajasn/Riskfolio-Lib).
        - PerformanceAnalytics: SharpeRatio with FUN="VaR" or "ES" (https://github.com/braverock/PerformanceAnalytics).

        Also known as: return over risk, risk-adjusted return ratio, generalized Sharpe ratio, conditional Sharpe
        ratio, VaR ratio, CVaR ratio, EVaR ratio, CDaR ratio, MAD ratio, Gini mean difference ratio.

        Args:
            period (str, optional): The period to use for the calculation. Defaults to "quarterly" if the Toolkit is
                initialised with quarterly=True, otherwise "yearly".
            risk_measure (str, optional): The risk measure, one of "var", "cvar", "evar", "erm", "worst", "mad",
                "lpm", "gmd", "max_drawdown", "average_drawdown", "dar", "cdar", "edar" or "ulcer". Defaults to
                "cvar".
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
        compounding by default (compounded=False), the Toolkit compounds them. The historical Value at Risk,
        Conditional Value at Risk and their drawdown versions use the interpolated quantile of the risk module,
        where skfolio takes the order statistic and the exact (Rockafellar-Uryasev) Conditional Value at Risk,
        so these differ slightly from skfolio on short samples.
        - `get_starr_ratio` divides the period excess return by the CVaR scaled to the period, `get_calmar_ratio`
        and `get_ulcer_performance_index` the period excess return by the drawdown measure; this ratio keeps the
        mean and the risk measure per day, so every risk measure is on the same scale.
        - A risk measure of 0 (e.g. no drawdown) leaves the ratio undefined (NaN).
        - If `growth` is set to True, the method calculates the growth of the ratio values using the specified `lag`.

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
            # Over the whole period the ratio is one number per ticker, with no dates to slice.
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
        the returns of `period` over every rolling window of that many periods (e.g. period="daily" and
        rolling=60 for Varadi's 60-day lookback). A value above 0.5 means the asset did better than the
        benchmark, and the more returns and the steadier the difference, the closer it gets to 1. Varadi
        switches between two assets only when the confidence passes a threshold such as 0.6, rather than on
        the sign of the return difference alone, which cuts the number of trades a momentum strategy makes on
        noise.

        For more information, see the following sources:

        - Varadi, D. (2014). "Are Simple Momentum Strategies Too Dumb? Introducing Probabilistic Momentum." CSS
        Analytics, 28 January 2014.
        - ffn: calc_prob_mom (https://github.com/pmorissette/ffn).

        Also known as: probabilistic momentum, momentum confidence.

        Args:
            period (str, optional): The period to use for the calculation. Defaults to "quarterly" if the Toolkit is
                initialised with quarterly=True, otherwise "yearly".
            rolling (int | None, optional): The number of periods in each rolling window. Defaults to None, which
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
        - If `growth` is set to True, the method calculates the growth of the values using the specified `lag`.

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

        returns = self._get_column(period, "Return", within_period=not rolling)
        benchmark_returns = self._get_column(
            period, "Return", within_period=not rolling, benchmark=True
        )

        if rolling:
            information_ratio = performance_model.get_rolling_information_ratio(
                returns, benchmark_returns, rolling
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

        For more information, see the following sources:

        - ffn: calc_mtd, calc_ytd, calc_cagr and PerformanceStats (mtd, three_month, six_month, ytd, one_year,
        three_year, five_year, ten_year, incep) (https://github.com/pmorissette/ffn).
        - PerformanceAnalytics: table.TrailingPeriods (https://github.com/braverock/PerformanceAnalytics).

        Also known as: trailing performance, period returns, MTD, YTD, 1Y, 3Y, 5Y, 10Y returns.

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

        For more information, see the following sources:

        - quantstats: stats.monthly_returns (https://github.com/ranaroussi/quantstats).
        - ffn: PerformanceStats.return_table (https://github.com/pmorissette/ffn).
        - PerformanceAnalytics: table.CalendarReturns (https://github.com/braverock/PerformanceAnalytics).

        Also known as: monthly returns table, return table, calendar year returns.

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
        column equals its EOY, and `get_returns` with period="monthly" and "yearly".

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
        simulations: int = 1000,
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

        - method="resample" draws the returns with replacement (Efron's bootstrap), so a day can appear
          several times or not at all and every statistic varies across simulations, as pyfolio does.
        - method="shuffle" puts the same returns in a random order (a permutation), as quantstats' Monte Carlo
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
        median, the standard deviation and the 5th, 25th, 75th and 95th percentiles over the simulations; the
        5th to 95th percentile is a 90% bootstrap interval.

        For more information, see the following sources:

        - Efron, B. (1979). "Bootstrap Methods: Another Look at the Jackknife." The Annals of Statistics, 7(1),
        1-26.
        - pyfolio: timeseries.perf_stats_bootstrap, calc_bootstrap and calc_distribution_stats
        (https://github.com/quantopian/pyfolio).
        - quantstats: stats.montecarlo, montecarlo_cagr, montecarlo_drawdown and montecarlo_sharpe
        (https://github.com/ranaroussi/quantstats).
        - ffn: resample_returns (https://github.com/pmorissette/ffn).

        Also known as: Monte Carlo simulation, bootstrapped performance statistics, resampled returns.

        Args:
            simulations (int, optional): The number of simulated histories. Defaults to 1000.
            method (str, optional): "resample" (with replacement) or "shuffle" (without replacement). Defaults
                to "resample".
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
        - pyfolio reports the mean, median and 5th and 95th percentiles of the same statistics (its standard
        deviation with ddof=0). quantstats keeps the history itself as one of the paths, counts 252 returns to
        a year and computes the Sharpe ratio of its paths from their percentage changes, which leaves out the
        first return; on the same paths the statistics otherwise agree.

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
                    simulations=simulations,
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

        For more information, see the following sources:

        - pyfolio: timeseries.forecast_cone_bootstrap, simulate_paths and summarize_paths
        (https://github.com/quantopian/pyfolio).
        - Efron, B. (1979). "Bootstrap Methods: Another Look at the Jackknife." The Annals of Statistics, 7(1),
        1-26.

        Also known as: bootstrap cone, probability cone, Bayesian cone (pyfolio's tear sheet).

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
        deviation over the simulations, which pyfolio takes with ddof=0 and the Toolkit with ddof=1.
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

        For more information, see the following sources:

        - Bailey, D.H., Borwein, J.M., López de Prado, M., & Zhu, Q.J. (2017). "The Probability of Backtest
        Overfitting." Journal of Computational Finance, 20(4), 39-69.
        - ffn: calc_prob_backtest_overfitting (https://github.com/pmorissette/ffn).

        Also known as: PBO, CSCV, combinatorially symmetric cross-validation.

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
        # The first date of the range has no return; dates no ticker traded hold no return either.
        returns = returns.dropna(how="all")

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

        For more information, see the following sources:

        - Getmansky, M., Lo, A.W., & Makarov, I. (2004). "An Econometric Model of Serial Correlation and
        Illiquidity in Hedge Fund Returns." Journal of Financial Economics, 74(3), 529-609.
        - PerformanceAnalytics: SmoothingIndex (https://github.com/braverock/PerformanceAnalytics).

        Also known as: Getmansky-Lo-Makarov smoothing index, return smoothing, serial correlation of returns.

        Args:
            period (str, optional): The period of the returns, weekly, monthly, quarterly or yearly. The paper
                uses monthly returns. Defaults to "monthly".
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
        - Liquid stocks have a Smoothing Index close to 1. `get_returns(unsmoothed=True)` removes first order
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

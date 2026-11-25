"""Risk Module"""

__docformat__ = "google"

from collections.abc import Callable
from functools import cached_property

import pandas as pd

from financetoolkit.helpers import LazyDict, handle_portfolio
from financetoolkit.risk.dependence.dependence_controller import DependenceRisk
from financetoolkit.risk.distribution.distribution_controller import DistributionRisk
from financetoolkit.risk.drawdown.drawdown_controller import DrawdownRisk
from financetoolkit.risk.helpers import (
    check_rolling,
    determine_within_historical_data,
    get_risk_free_rate_per_ticker,
    mask_non_trading_days,
)
from financetoolkit.risk.liquidity.liquidity_controller import LiquidityRisk
from financetoolkit.risk.value_at_risk.value_at_risk_controller import ValueAtRiskRisk
from financetoolkit.risk.volatility.volatility_controller import VolatilityRisk
from financetoolkit.utilities import factor_model
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import (
    InvalidArgumentError,
    check_period,
    handle_errors,
)
from financetoolkit.utilities.logger_model import get_logger, log_each_message_once
from financetoolkit.utilities.sources_model import track_sources
from financetoolkit.utilities.statistics_model import apply_rounding, finalize_dataset

logger = get_logger()


@filter_shown_columns
@track_sources
class RiskBase:
    """
    The data every risk topic shares: the tickers, returns and prices.
    """

    def __init__(
        self,
        tickers: list[str] | Callable[[], list[str]],
        historical_data: dict[str, pd.DataFrame] | None = None,
        risk_free_rate_data: dict[str, pd.Series] | None = None,
        intraday_period: str | None = None,
        quarterly: bool = False,
        rounding: int | None = 4,
        start_date: str | None = None,
        end_date: str | None = None,
        profile: pd.DataFrame | Callable[[], pd.DataFrame] | None = None,
    ):
        """
        Initializes the Risk Controller Class.

        The Toolkit passes the tickers as a function without arguments and the period
        dictionaries as a LazyDict (financetoolkit.helpers), so a metric retrieves the data of
        the period it is calculated over when it is first needed, and nothing else.

        Args:
            tickers (list[str] | Callable[[], list[str]]): The tickers to use for the Toolkit instance.
            historical_data (dict[str, pd.DataFrame] | None, optional): The historical data per period.
                Defaults to None, which is treated as an empty dictionary.
            risk_free_rate_data (dict[str, pd.Series] | None, optional): The annualized risk free rate
                per period frequency, used for the Excess Volatility calculations. Defaults to None.
            intraday_period (str | None, optional): The intraday period used for within-period calculations.
                Defaults to None.
            quarterly (bool, optional): Whether to use quarterly data. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            start_date (str | None, optional): The start date to use for the calculations. Defaults to None.
            end_date (str | None, optional): The end date to use for the calculations. Defaults to None.
            profile (pd.DataFrame | Callable | None, optional): The company profiles, with
                the tickers as columns and the profile items (such as "Sector" and
                "Industry") as rows, or a function that retrieves them when first needed,
                for the groups of get_factor_analysis. Defaults to None, which leaves
                group="sector" and group="industry" unavailable.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT"],
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
        | MSFT      | -0.2188 |
        | Benchmark | -0.1279 |
        """
        self._historical_data = historical_data if historical_data is not None else {}
        self._annualized_risk_free_rate_data = (
            risk_free_rate_data if risk_free_rate_data is not None else {}
        )
        # The risk free rate is quoted as an annualized yield, so it is converted to the rate each ticker earns per period of its returns, on its own calendar (see `_get_risk_free_rate`).  # noqa: E501
        self._risk_free_rate_data = LazyDict(self._get_risk_free_rate)
        self._tickers_source = tickers
        self._profile_source = profile
        self._quarterly = quarterly
        self._rounding: int | None = rounding
        self._start_date: str | None = start_date
        self._end_date: str | None = end_date
        self._portfolio_weights: dict | None = None

        # The returns within each period, built when a metric first asks for a period. Missing values stay missing: filling them with 0 would turn a period without data (e.g. before a listing, or the lone boundary observation at the start of the range) into 0% returns, i.e. a VaR, skewness or drawdown of exactly 0 instead of no value.  # noqa: E501
        self._within_historical_data = LazyDict(
            lambda period: determine_within_historical_data(
                period, self._historical_data, intraday_period
            )
        )

        # Memo for the return slices handed to the calculations, keyed by (period,
        # within_period): nearly every method starts from that slice and the MCP server
        # collects many metrics per session. Neither frame is mutated after this point.
        self._data_cache: dict[tuple[str, bool], pd.DataFrame] = {}

    @cached_property
    def _tickers(self) -> list[str]:
        """The tickers, retrieved when first needed."""
        return (
            self._tickers_source()
            if callable(self._tickers_source)
            else self._tickers_source
        )

    def _get_risk_free_rate(self, period: str) -> pd.DataFrame:
        """
        The risk free return per period of each ticker (and the Benchmark) on its own
        calendar, see `financetoolkit.risk.helpers.get_risk_free_rate_per_ticker`.
        """
        return get_risk_free_rate_per_ticker(
            self._annualized_risk_free_rate_data, self._historical_data, period
        )

    @handle_portfolio
    def _get_returns(self, period: str, within_period: bool) -> pd.DataFrame:
        """
        The Return slice a calculation runs on, memoized per (period, within_period)
        pair (see `_data_cache` in `__init__`).

        The whole-series slice drops the rows without any return: a resampled series
        starts with a NaN return (there is no previous period to compare against).
        Rows where only some tickers miss a return (e.g. a ticker that listed later)
        are kept, so every ticker uses its own full history; the calculations skip the
        missing returns per column. The within-period frames keep their missing values
        as well, so a period without observations for a ticker gives NaN rather than 0.
        """
        key = (period, within_period)

        if key not in self._data_cache:
            self._data_cache[key] = (
                self._within_historical_data[period]["Return"]
                if within_period
                else self._historical_data[period]["Return"].dropna(how="all")
            )

        returns = self._data_cache[key]

        if not within_period:
            # A measure over the whole range compares returns with each other, so a single
            # return (e.g. one yearly return in a range within one year) gives a drawdown of
            # 0 and an undefined volatility, which reads as "no risk" without this note.
            short_history = returns.columns[returns.count() < 2]  # noqa: PLR2004

            if len(short_history):
                logger.warning(
                    "Fewer than two %s returns between %s and %s for %s, so measures over the whole range "  # noqa: E501
                    "are 0 or NaN. Use a higher frequency period, or within_period=True for a value per period.",  # noqa: E501
                    period,
                    self._start_date,
                    self._end_date,
                    ", ".join(map(str, short_history)),
                )

        return returns

    def _get_measure_returns(
        self,
        period: str,
        within_period: bool,
        rolling: int | None,
        growth: bool,
        minimum: int = 2,
    ) -> pd.DataFrame:
        """
        The returns a measure of the return distribution runs on, after validating the
        parameters it shares with the other distribution measures: the `period` returns of
        the full history for a rolling window, else those of `_get_returns`. `minimum` is the
        smallest rolling window the measure has a value for.
        """
        check_period(period, ["daily", "weekly", "monthly", "quarterly", "yearly"])
        if (
            period == "daily"
            and within_period
            and self._historical_data["intraday"].empty
        ):
            raise InvalidArgumentError(
                "Intraday data is required for within-period daily calculations. Either set "
                "within_period=False or initialise the Toolkit with an intraday_period."
            )
        if growth and not within_period and not rolling:
            raise InvalidArgumentError(
                "Growth calculations need one value per period, while "
                "within_period=False produces a single value for the entire period. "
                "Set within_period=True to calculate growth."
            )

        check_rolling(rolling, self._historical_data, period, minimum)
        if rolling:
            return self._historical_data[period]["Return"]

        return self._get_returns(period, within_period)

    def _finalize_measure(
        self,
        result: pd.Series | pd.DataFrame,
        sliced: bool,
        rounding: int | None,
        growth: bool,
        lag: int | list[int],
        standardize: bool,
    ) -> pd.Series | pd.DataFrame:
        """
        Cut a measure to the date range when it has a value per period or date (`sliced`,
        with within_period=True or rolling) and apply the growth, rounding and
        standardization, as every distribution measure does.
        """
        if sliced:
            result = result.loc[self._start_date : self._end_date]

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
            apply_slice=False,
        )

    def _get_prices(
        self, column: str, period: str = "daily", within_period: bool = False
    ) -> pd.DataFrame:
        """
        The `column` prices on the dates each ticker traded, see
        `helpers.mask_non_trading_days`.
        """
        frame = (
            self._within_historical_data[period]
            if within_period
            else self._historical_data[period]
        )

        return mask_non_trading_days(frame, column)

    def _get_price_column(self, period: str, column: str) -> pd.DataFrame:
        # Reads the plain period-frequency history rather than the "within period" multi-index, so period="daily" simply means daily observations here and (unlike every within-period method above) needs no intraday data.  # noqa: E501
        check_period(period, ["daily", "weekly", "monthly", "quarterly", "yearly"])

        # Only rows without any value are dropped: the pairwise calculations (copulas, tail dependence, CoVaR) align each pair themselves, so a joint dropna would cut every ticker's sample down to the shortest history.  # noqa: E501
        return self._historical_data[period][column].dropna(how="all")


# ruff: noqa: E501

# The risk metrics per collection that get_factor_analysis compares across tickers, each
# measured within every evaluation period. The default holds the well-known risk anomalies:
# low volatility, the largest drawdown, skewness (lottery-like stocks) and illiquidity. The
# GARCH models are left out as fitting one per ticker and period is slow, and the pair and
# portfolio measures as they have no value per ticker.
FACTOR_COLLECTIONS = {
    "anomalies": [
        "get_volatility",
        "get_maximum_drawdown",
        "get_skewness",
        "get_amihud_illiquidity",
    ],
}

FACTOR_COLLECTIONS["all"] = FACTOR_COLLECTIONS["anomalies"] + [
    "get_downside_deviation",
    "get_kurtosis",
    "get_value_at_risk",
    "get_conditional_value_at_risk",
    "get_tail_ratio",
    "get_roll_spread",
]


@filter_shown_columns
@track_sources
class Risk(
    ValueAtRiskRisk,
    DrawdownRisk,
    VolatilityRisk,
    DependenceRisk,
    DistributionRisk,
    LiquidityRisk,
    RiskBase,
):
    """
    The Risk module is meant to calculate important risk metrics such
    as Value at Risk (VaR), Conditional Value at Risk (cVaR), Maximum
    Drawdown, Correlations, GARCH, EWMA and more.
    """

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
        Collect all risk metrics of every asset in one table: its volatility, tail risk, drawdowns and
        distribution side by side.

        Args:
            period (str, optional): The data frequency for returns. Options are: "daily", "weekly", "monthly",
                "quarterly", "yearly". Defaults to "quarterly" if the Toolkit is initialised with quarterly=True, otherwise "yearly".
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
            pd.Series or pd.DataFrame: Risk metrics calculated based on the specified parameters.

        Notes:
        - The method calculates various risk metrics for each asset in the Toolkit instance.
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

        metrics = toolkit.risk.collect_all_metrics().xs("AAPL", level=1, axis=1)

        metrics[["Value at Risk", "Maximum Drawdown", "Ulcer Index", "Skewness", "Kurtosis"]]
        ```

        Which returns:

        |      |   Value at Risk |   Maximum Drawdown |   Ulcer Index |   Skewness |   Kurtosis |
        |:-----|----------------:|-------------------:|--------------:|-----------:|-----------:|
        | 2021 |         -0.0256 |            -0.1861 |        0.0366 |    -0.0748 |     3.4006 |
        | 2022 |         -0.0373 |            -0.3032 |        0.0664 |     0.3168 |     4.0553 |
        | 2023 |         -0.0175 |            -0.1498 |        0.0324 |    -0.0686 |     4.4756 |
        | 2024 |         -0.0217 |            -0.1539 |        0.0343 |     0.4447 |     6.6112 |
        | 2025 |         -0.0321 |            -0.3107 |        0.0502 |     1.1306 |    17.0892 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        check_period(period, ["daily", "weekly", "monthly", "quarterly", "yearly"])
        if period == "daily" and self._historical_data["intraday"].empty:
            raise InvalidArgumentError(
                "Intraday data is required for daily calculations. Please initialise the "
                "Toolkit with an intraday_period or use a weekly or longer period."
            )

        rounding = rounding if rounding is not None else self._rounding

        risk_metrics = {
            "Value at Risk": self.get_value_at_risk(
                period=period,
                within_period=True,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
            "Conditional Value at Risk": self.get_conditional_value_at_risk(
                period=period,
                within_period=True,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
            "Entropic Value at Risk": self.get_entropic_value_at_risk(
                period=period,
                within_period=True,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
            "Conditional Drawdown at Risk": self.get_conditional_drawdown_at_risk(
                period=period,
                within_period=True,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
            "Tail Ratio": self.get_tail_ratio(
                period=period,
                within_period=True,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
            "Maximum Drawdown": self.get_maximum_drawdown(
                period=period,
                within_period=True,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
            "Maximum Drawdown Duration": self.get_maximum_drawdown_duration(
                period=period,
                within_period=True,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
            "Maximum Drawdown Recovery Time": self.get_maximum_drawdown_recovery_time(
                period=period,
                within_period=True,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
            "Ulcer Index": self.get_ulcer_index(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
            "Pain Index": self.get_pain_index(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
            # GARCH stays a whole-series fit: it models the conditional variance of the
            # per-period returns, and a per-period refit would also break on the lone
            # boundary observation the first period of a date range typically holds.
            "GARCH": self.get_garch(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
            "Skewness": self.get_skewness(
                period=period,
                within_period=True,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
            "Kurtosis": self.get_kurtosis(
                period=period,
                within_period=True,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
            "Downside Deviation": self.get_downside_deviation(
                period=period,
                within_period=True,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            ),
        }

        if period != "daily":
            risk_metrics["Variance"] = self.get_variance(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            )
            risk_metrics["Volatility"] = self.get_volatility(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            )
            risk_metrics["Excess Volatility"] = self.get_excess_volatility(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            )
            risk_metrics["Mean Absolute Deviation"] = self.get_mean_absolute_deviation(
                period=period,
                rounding=rounding,
                growth=growth,
                lag=lag,
                standardize=standardize,
            )
            risk_metrics["Coefficient of Variation"] = (
                self.get_coefficient_of_variation(
                    period=period,
                    rounding=rounding,
                    growth=growth,
                    lag=lag,
                    standardize=standardize,
                )
            )

        risk_metrics = pd.concat(risk_metrics, axis=1)

        if len(self._tickers) == 1:
            risk_metrics = risk_metrics.xs(self._tickers[0], level=1, axis=1)

        return risk_metrics

    @handle_errors
    def get_factor_analysis(
        self,
        metrics: str | list[str] = "anomalies",
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
        Calculate the factor analysis (predictive power) of the risk metrics: whether the tickers that
        score high on a metric go on to return more than those that score low.

        Per period, the
        information coefficient (IC) is the Spearman rank correlation between the metric of the
        tickers in the period and their return over the following horizon: 1 means the metric
        ranks the later returns perfectly, -1 in reverse and 0 that it says nothing about them.
        The IC measures skill in Grinold and Kahn's fundamental law of active management,
        IR ≈ IC × √breadth: a small but consistent IC across many tickers and periods gives a
        high information ratio.

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

        Shares similar logic:
        - alphalens, factor_information_coefficient: https://github.com/stefan-jansen/alphalens-reloaded/blob/v0.4.4/src/alphalens/performance.py#L28
        - alphalens, factor_returns: https://github.com/stefan-jansen/alphalens-reloaded/blob/v0.4.4/src/alphalens/performance.py#L210
        - alphalens, mean_return_by_quantile: https://github.com/stefan-jansen/alphalens-reloaded/blob/v0.4.4/src/alphalens/performance.py#L461

        Args:
            metrics (str | list[str], optional): A risk method name with a value per ticker and period,
                e.g. "get_volatility", a list of them, or a collection. Options for a collection are:
                "anomalies" (the volatility, maximum drawdown, skewness and Amihud illiquidity), "all"
                (every risk metric with a value per ticker and period). Defaults to "anomalies". "all" adds the downside deviation, kurtosis, value at risk,
                conditional value at risk, tail ratio and Roll spread.
            period (str, optional): The frequency at which the metrics are measured and evaluated.
                Options are: "weekly", "monthly", "quarterly", "yearly". Each metric is calculated
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
                instead of the summary. Options are: "information_coefficient" (per period and horizon),
                "information_coefficient_by_month" (the mean IC per calendar month),
                "information_coefficient_by_year" (the mean IC per calendar year),
                "group_information_coefficient" (the IC statistics within each group, needs group),
                "quantile_returns" (the mean forward return per quantile and horizon),
                "quantile_returns_by_date" (per period), "quantile_return_statistics" (the mean, its
                standard error and both per single period), "quantile_statistics" (the range, mean and
                count of the factor values in each quantile), "quantile_turnover" (per period and
                quantile), "rank_autocorrelation" (per period), "factor_weights" (the weights of the
                factor-weighted long-short portfolio per period and ticker), "factor_returns" (its
                return per period and horizon), "cumulative_returns" (the growth of 1 in it and in
                each quantile), "event_returns" (the mean price path of each quantile from the
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
            - The IC is reported as measured: the low-volatility anomaly, where the least
              volatile stocks go on to return more, shows as a negative IC for the volatility.
              Read the sign as the direction of the relationship.
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

        toolkit.risk.get_factor_analysis(horizons=[1, 12])
        ```

        Which returns:

        |                                         |   Mean IC |   IC Standard Deviation |   IC Information Ratio |   t-Statistic |   P-Value |   IC Skewness |   IC Kurtosis |   Positive Share |   Observations |   Top Minus Bottom |   Annualized Alpha |    Beta |   Quantile Turnover |   Rank Autocorrelation |
        |:----------------------------------------|----------:|------------------------:|-----------------------:|--------------:|----------:|--------------:|--------------:|-----------------:|---------------:|-------------------:|-------------------:|--------:|--------------------:|-----------------------:|
        | ('get_volatility', '1 Month')           |    0.0963 |                  0.3676 |                 0.262  |        2.8586 |    0.005  |       -0.2056 |       -0.405  |           0.6218 |            119 |             0.0185 |             0.0337 |  0.407  |              0.4688 |                 0.5848 |
        | ('get_volatility', '12 Months')         |    0.2759 |                  0.3703 |                 0.7451 |        7.7428 |    0      |       -0.7586 |       -0.2658 |           0.7685 |            108 |             0.3154 |            -0.0757 |  1.0993 |              0.4688 |                 0.5848 |
        | ('get_maximum_drawdown', '1 Month')     |   -0.0257 |                  0.3315 |                -0.0774 |       -0.8443 |    0.4002 |       -0.0332 |       -0.4476 |           0.4622 |            119 |            -0.0041 |             0.0118 | -0.3113 |              0.6275 |                 0.2719 |
        | ('get_maximum_drawdown', '12 Months')   |   -0.1759 |                  0.3542 |                -0.4965 |       -5.1603 |    0      |        0.496  |       -0.4167 |           0.3241 |            108 |            -0.2094 |             0.0475 | -0.6263 |              0.6275 |                 0.2719 |
        | ('get_skewness', '1 Month')             |    0.0136 |                  0.2943 |                 0.0464 |        0.5056 |    0.6141 |       -0.0773 |        0.1094 |           0.5462 |            119 |            -0.0007 |            -0.0055 |  0.0598 |              0.7521 |                 0.0593 |
        | ('get_skewness', '12 Months')           |    0.0081 |                  0.2996 |                 0.0271 |        0.2815 |    0.7789 |        0.1058 |       -0.547  |           0.5093 |            108 |             0.0142 |            -0.0111 |  0.0936 |              0.7521 |                 0.0593 |
        | ('get_amihud_illiquidity', '1 Month')   |   -0.0285 |                  0.3896 |                -0.0731 |       -0.7971 |    0.427  |       -0.0875 |       -0.9255 |           0.479  |            119 |            -0.0059 |             0.0216 | -0.1902 |              0.1905 |                 0.8996 |
        | ('get_amihud_illiquidity', '12 Months') |   -0.1164 |                  0.3789 |                -0.3072 |       -3.1927 |    0.0019 |        0.68   |       -0.2884 |           0.3611 |            108 |            -0.0796 |             0.1036 | -0.457  |              0.1905 |                 0.8996 |
        """
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

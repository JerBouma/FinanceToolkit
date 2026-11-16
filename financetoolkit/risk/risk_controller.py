"""Risk Module"""

__docformat__ = "google"

import pandas as pd

from financetoolkit.risk.dependence.dependence_controller import DependenceRisk
from financetoolkit.risk.distribution.distribution_controller import DistributionRisk
from financetoolkit.risk.drawdown.drawdown_controller import DrawdownRisk
from financetoolkit.risk.liquidity.liquidity_controller import LiquidityRisk
from financetoolkit.risk.value_at_risk.value_at_risk_controller import ValueAtRiskRisk
from financetoolkit.risk.volatility.volatility_controller import VolatilityRisk
from financetoolkit.utilities import factor_model
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import handle_errors
from financetoolkit.utilities.sources_model import track_sources
from financetoolkit.utilities.statistics_model import (
    apply_rounding,
)

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
):
    """
    The Risk module is meant to calculate important risk metrics such
    as Value at Risk (VaR), Conditional Value at Risk (cVaR), Maximum
    Drawdown, Correlations, GARCH, EWMA and more.
    """

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
        Calculates and collects all risk metrics.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "quarterly" if the Toolkit is initialised with quarterly=True, otherwise "yearly".
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

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )
        if period == "daily" and self._historical_data["intraday"].empty:
            raise ValueError("Intraday data is required for daily calculations.")

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
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Factor analysis (predictive power) of the risk metrics: whether the tickers that score
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

        Args:
            metrics (str | list[str], optional): A risk method name with a value per ticker and period,
                e.g. "get_volatility", a list of them, or a collection: "anomalies" or "all".
                Defaults to "anomalies", the volatility, maximum drawdown, skewness and Amihud
                illiquidity. "all" adds the downside deviation, kurtosis, value at risk,
                conditional value at risk, tail ratio and Roll spread.
            period (str, optional): The frequency at which the metrics are measured and evaluated
                (weekly, monthly, quarterly or yearly). Each metric is calculated
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
            - The IC is reported as measured: the low-volatility anomaly, where the least
              volatile stocks go on to return more, shows as a negative IC for the volatility.
              Read the sign as the direction of the relationship.
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

        toolkit.risk.get_factor_analysis(horizons=[1, 12])
        ```

        Which returns:
        |                                         |   Mean IC |   IC Standard Deviation |   IC Information Ratio |   t-Statistic |   p-Value |   Positive Share |   Observations |   Top Minus Bottom |   Quantile Turnover |   Rank Autocorrelation |
        |:----------------------------------------|----------:|------------------------:|-----------------------:|--------------:|----------:|-----------------:|---------------:|-------------------:|--------------------:|-----------------------:|
        | ('get_volatility', '1 Month')           |    0.0967 |                  0.368  |                 0.2628 |        2.8673 |    0.0049 |           0.6218 |            119 |             0.0184 |              0.4702 |                 0.586  |
        | ('get_volatility', '12 Months')         |    0.2766 |                  0.3693 |                 0.7489 |        7.7824 |    0      |           0.7685 |            108 |             0.3166 |              0.4702 |                 0.586  |
        | ('get_maximum_drawdown', '1 Month')     |   -0.0257 |                  0.3311 |                -0.0775 |       -0.8451 |    0.3997 |           0.4538 |            119 |            -0.0038 |              0.6275 |                 0.2711 |
        | ('get_maximum_drawdown', '12 Months')   |   -0.1745 |                  0.3544 |                -0.4923 |       -5.1158 |    0      |           0.3241 |            108 |            -0.2068 |              0.6275 |                 0.2711 |
        | ('get_skewness', '1 Month')             |    0.0137 |                  0.2949 |                 0.0464 |        0.5066 |    0.6134 |           0.5378 |            119 |            -0.0005 |              0.7521 |                 0.0583 |
        | ('get_skewness', '12 Months')           |    0.0107 |                  0.2985 |                 0.0358 |        0.3717 |    0.7109 |           0.5185 |            108 |             0.0142 |              0.7521 |                 0.0583 |
        | ('get_amihud_illiquidity', '1 Month')   |   -0.0285 |                  0.3897 |                -0.0732 |       -0.7985 |    0.4262 |           0.479  |            119 |            -0.0059 |              0.1852 |                 0.9003 |
        | ('get_amihud_illiquidity', '12 Months') |   -0.1158 |                  0.3791 |                -0.3054 |       -3.1735 |    0.002  |           0.3611 |            108 |            -0.08   |              0.1852 |                 0.9003 |
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
        )

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

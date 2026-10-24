"""Risk Module"""

__docformat__ = "google"

import itertools
import warnings

import numpy as np
import pandas as pd

from financetoolkit.helpers import handle_portfolio
from financetoolkit.risk import (
    backtesting_model,
    copula_model,
    covar_model,
    cvar_model,
    evar_model,
    garch_model,
    market_liquidity_model,
    realized_volatility_model,
    risk_model,
    var_model,
)
from financetoolkit.risk.helpers import determine_within_historical_data
from financetoolkit.utilities.error_model import handle_errors
from financetoolkit.utilities.statistics_model import (
    apply_rounding,
    convert_annualized_rate_to_period,
    finalize_dataset,
)

# Division by zero is normal in these calculations, not a bug.
warnings.filterwarnings("ignore", category=RuntimeWarning)

MINIMUM_TICKERS_FOR_ALL_PAIRS = 2

# pylint: disable=too-many-instance-attributes,too-few-public-methods,too-many-lines,too-many-locals
# pylint: disable=too-many-boolean-expressions


def _as_scalar(value: float | np.ndarray | pd.Series | pd.DataFrame) -> float:
    """
    Reduces a Value at Risk or Conditional Value at Risk estimate to a single float.

    A rolling apply only accepts a scalar return value, but the Student-t and Extreme
    Value Theory variants return a length one array or Series when given a Series, which
    would otherwise raise a TypeError before it reaches the backtest.

    Args:
        value (float | np.ndarray | pd.Series | pd.DataFrame): the estimate to reduce.

    Returns:
        float: the estimate as a single float.
    """
    return float(np.asarray(value).reshape(-1)[0])


def _get_out_of_sample_estimates(
    returns: pd.DataFrame,
    measure: str,
    distribution: str,
    alpha: float,
    window_size: int,
) -> pd.DataFrame:
    """
    Builds the rolling, out-of-sample Value at Risk or Conditional Value at Risk path
    of every ticker, as used by the VaR and Expected Shortfall backtests.

    Each ticker is rolled over its own non-missing returns, so a ticker with a shorter
    history (or a missing day) does not shorten the sample of the others, and the window
    always holds `window_size` actual observations. The path is shifted by one
    observation so that every estimate only uses the returns before the one it is
    compared against.

    Args:
        returns (pd.DataFrame): The returns, one column per ticker.
        measure (str): Either "var" or "cvar".
        distribution (str): One of "historic", "gaussian", "studentt" or "evt".
        alpha (float): The tail probability (e.g., 0.05 for the 95% confidence level).
        window_size (int): The number of observations in each rolling window.

    Returns:
        pd.DataFrame: The out-of-sample estimates, on the index of `returns`.
    """
    parametric_estimators = {
        ("var", "gaussian"): var_model.get_var_gaussian,
        ("var", "studentt"): var_model.get_var_studentt,
        ("var", "evt"): var_model.get_var_evt,
        ("cvar", "gaussian"): cvar_model.get_cvar_gaussian,
        ("cvar", "studentt"): cvar_model.get_cvar_studentt,
        ("cvar", "evt"): cvar_model.get_cvar_evt,
    }

    estimates = {}

    for column in returns.columns:
        column_returns = returns[column].dropna()

        if distribution == "historic":
            estimate = (
                var_model.get_rolling_var_historic(column_returns, alpha, window_size)
                if measure == "var"
                else cvar_model.get_rolling_cvar_historic(
                    column_returns, alpha, window_size
                )
            )
        else:
            estimator = parametric_estimators[(measure, distribution)]
            estimate = column_returns.rolling(window=window_size).apply(
                lambda window, estimator=estimator: _as_scalar(
                    estimator(window, alpha)
                ),
                raw=False,
            )

        # A trailing rolling window includes the current observation, so without the shift each estimate would be compared against a return it was computed from.  # noqa: E501
        estimates[column] = estimate.shift(1).reindex(returns.index)

    return pd.DataFrame(estimates, index=returns.index, columns=returns.columns)


class Risk:
    """
    The Risk module is meant to calculate important risk metrics such
    as Value at Risk (VaR), Conditional Value at Risk (cVaR), Maximum
    Drawdown, Correlations, GARCH, EWMA and more.
    """

    def __init__(
        self,
        tickers: str | list[str],
        historical_data: dict[str, pd.DataFrame] | None = None,
        risk_free_rate_data: dict[str, pd.Series] | None = None,
        intraday_period: str | None = None,
        quarterly: bool = False,
        rounding: int | None = 4,
        start_date: str | None = None,
        end_date: str | None = None,
    ):
        """
        Initializes the Risk Controller Class.

        Args:
            tickers (str | list[str]): The tickers to use for the Toolkit instance.
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
        # The risk free rate is quoted as an annualized yield, so it is converted to the matching frequency. Without this, a daily return would have a full year of risk free rate subtracted from it.  # noqa: E501
        self._risk_free_rate_data: dict[str, pd.Series] = {}
        for frequency, rate in (risk_free_rate_data or {}).items():
            period_rate = convert_annualized_rate_to_period(rate, frequency)
            # The conversion is also typed for scalars; a Series in gives a Series out.
            if not isinstance(period_rate, pd.Series):
                raise TypeError(f"Expected a risk free rate series for {frequency}.")
            self._risk_free_rate_data[frequency] = period_rate
        self._tickers = tickers
        self._quarterly = quarterly
        self._rounding: int | None = rounding
        self._start_date: str | None = start_date
        self._end_date: str | None = end_date
        self._portfolio_weights: dict | None = None

        # Within Return Calculations. Missing values stay missing: filling them with 0 would turn a period without data (e.g. before a listing, or the lone boundary observation at the start of the range) into 0% returns, i.e. a VaR, skewness or drawdown of exactly 0 instead of no value.  # noqa: E501
        daily_historical_data = self._historical_data["daily"].copy()
        intraday_historical_data = self._historical_data["intraday"].copy()

        daily_historical_data.index = pd.DatetimeIndex(
            daily_historical_data.to_timestamp().index
        )

        if not self._historical_data["intraday"].empty:
            intraday_historical_data.index = pd.DatetimeIndex(
                intraday_historical_data.to_timestamp().index
            )

        self._within_historical_data = determine_within_historical_data(
            daily_historical_data=daily_historical_data,
            intraday_historical_data=intraday_historical_data,
            intraday_period=intraday_period,
        )

        # Memo for the return slices handed to the calculations, keyed by (period,
        # within_period): nearly every method starts from that slice and the MCP server
        # collects many metrics per session. Neither frame is mutated after this point.
        self._data_cache: dict[tuple[str, bool], pd.DataFrame] = {}

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

        return self._data_cache[key]

    @handle_errors
    def collect_all_metrics(
        self,
        period: str | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ):
        """
        Calculates and collects all risk metrics.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "quarterly" if the Toolkit is initialised with quarterly=True, otherwise "yearly".
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the ratios. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

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

    @handle_portfolio
    @handle_errors
    def get_value_at_risk(
        self,
        period: str | None = None,
        alpha: float = 0.05,
        within_period: bool = False,
        rolling: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        distribution: str = "historic",
        threshold_percentile: float = 0.95,
    ):
        """
        Calculate the Value at Risk (VaR) of an investment portfolio or asset's returns.

        Value at Risk (VaR) is a risk management metric that quantifies the maximum potential loss
        an investment portfolio or asset may experience over a specified time horizon and confidence level.
        It provides insights into the downside risk associated with an investment and helps investors make
        informed decisions about risk tolerance.

        The VaR is calculated as the quantile of the return distribution, representing the loss threshold
        that is only exceeded with tail probability alpha (e.g., in 5% of the periods for alpha=0.05, the
        95% confidence level).

        Value at Risk is a quantile: it says how bad a period gets at the chosen confidence level, not how
        bad the periods beyond it are, so two assets with the same VaR can have very different tails (see
        the Conditional Value at Risk for that). Historical VaR depends heavily on the window: one calm year
        gives a much smaller figure than a window that includes a crisis, so compare assets over the same
        period and confidence level. Scaling a daily VaR to a longer horizon with the square root of time
        assumes independent returns, which tends to understate risk when volatility clusters.

        Also known as: VaR, maximum expected loss, portfolio loss risk.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "daily".
            alpha (float, optional): The tail probability for the VaR calculation (e.g., 0.05 for the 95%
            confidence level). Defaults to 0.05.
            within_period (bool, optional): Whether to calculate VaR within the specified period or for the entire
            period. Thus whether to look at the VaR within a specific year (if period = 'yearly') or look at the entirety
            of all years. Defaults to False. Note that this requires intraday data when period = 'daily', since a
            single day only nests observations when intraday data was fetched.
            rolling (int, optional): The rolling window size to use for the calculation. If set, VaR is
            calculated over a rolling window of this many periods across the full return history instead
            of per `period` (e.g. a rolling 60-day VaR). Only available for
            `distribution="historic"`; see `get_var_backtest` for a rolling, out-of-sample VaR path
            under the parametric distributions. Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the VaR values over time. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            distribution (str): The distribution to use for the VaR calculations (historic, gaussian,
            cornish-fisher, studentt or evt). Defaults to "historic". The parametric distributions use the
            sample standard deviation; "cornish-fisher" adjusts the gaussian quantile for the skewness and
            excess kurtosis of the returns (see `var_model.get_var_cornish_fisher`) and "studentt" uses the
            quantile of a Student-T distribution fitted by maximum likelihood.
            threshold_percentile (float, optional): Only used when `distribution` is "evt". The percentile
            of losses above which the Generalized Pareto Distribution is fitted; `alpha` can be at most
            1 - `threshold_percentile`. Defaults to 0.95.

        Returns:
            pd.Series: VaR values with time as the index.

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates VaR for each
        asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of VaR values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AMZN", "TSLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_value_at_risk(period="yearly", within_period=True)
        ```

        Which returns:

        |      |    AMZN |    TSLA |   Benchmark |
        |:-----|--------:|--------:|------------:|
        | 2021 | -0.0253 | -0.05   |     -0.0126 |
        | 2022 | -0.0518 | -0.0714 |     -0.0262 |
        | 2023 | -0.0273 | -0.0505 |     -0.0139 |
        | 2024 | -0.0248 | -0.0522 |     -0.0132 |
        | 2025 | -0.0305 | -0.0553 |     -0.0165 |
        """
        period = period if period else "daily"

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )
        if (
            period == "daily"
            and within_period
            and self._historical_data["intraday"].empty
        ):
            raise ValueError(
                "Intraday data is required for within-period daily calculations. Either set "
                "within_period=False or initialise the Toolkit with an intraday_period."
            )
        if growth and not within_period and not rolling:
            raise ValueError(
                "Growth calculations need one value per period, while "
                "within_period=False produces a single value for the entire period. "
                "Set within_period=True to calculate growth."
            )

        if rolling and distribution != "historic":
            raise ValueError(
                "The rolling parameter is only available for distribution='historic'. "
                "Use get_var_backtest to obtain a rolling, out-of-sample VaR path "
                "under one of the parametric distributions instead."
            )

        if rolling:
            returns = self._historical_data[period]["Return"]
            value_at_risk = var_model.get_rolling_var_historic(returns, alpha, rolling)
        else:
            returns = self._get_returns(period, within_period)

            if distribution == "historic":
                value_at_risk = var_model.get_var_historic(returns, alpha)
            elif distribution == "gaussian":
                value_at_risk = var_model.get_var_gaussian(returns, alpha)
            elif distribution == "cornish-fisher":
                value_at_risk = var_model.get_var_cornish_fisher(returns, alpha)
            elif distribution == "studentt":
                value_at_risk = var_model.get_var_studentt(returns, alpha)
            elif distribution == "evt":
                value_at_risk = var_model.get_var_evt(
                    returns, alpha, threshold_percentile
                )
            else:
                raise ValueError(
                    "Distribution must be historic, gaussian, cornish-fisher, studentt "
                    "or evt."
                )

        if rolling or within_period:
            value_at_risk = value_at_risk.loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=value_at_risk,
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

    @handle_portfolio
    @handle_errors
    def get_conditional_value_at_risk(
        self,
        period: str | None = None,
        alpha: float = 0.05,
        within_period: bool = False,
        rolling: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
        distribution: str = "historic",
        threshold_percentile: float = 0.95,
    ):
        """
        Calculate the Conditional Value at Risk (CVaR) of an investment portfolio or asset's returns.

        Conditional Value at Risk (CVaR) is a risk management metric that quantifies the loss in the worst % of cases of
        an investment portfolio or asset may experience over a specified time horizon and confidence level.
        It provides insights into the downside risk associated with an investment and helps investors make
        informed decisions about risk tolerance.

        The CVaR is calculated as the expected loss given that the loss threshold (VaR) with a given confidence
        level (e.g., 5% for alpha=0.05) is exceeded.

        The Conditional Value at Risk (expected shortfall) is the measure banks report for market risk under
        the Basel framework, at 97.5% rather than 95% confidence, because unlike VaR it accounts for the size
        of the losses in the tail and never penalises diversification (it is subadditive). With daily data
        the estimate rests on few observations (one year at 95% leaves about 12 days in the tail), so a
        single extreme day can move it a lot; a longer window or a parametric distribution gives a steadier
        figure.

        Also known as: CVaR, expected shortfall, ES, tail risk.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "daily".
            alpha (float, optional): The tail probability for the CVaR calculation (e.g., 0.05 for the 95%
            confidence level). Defaults to 0.05.
            within_period (bool, optional): Whether to calculate CVaR within the specified period or for the entire
            period. Thus whether to look at the CVaR within a specific year (if period = 'yearly') or look at the entirety
            of all years. Defaults to False. Note that this requires intraday data when period = 'daily', since a
            single day only nests observations when intraday data was fetched.
            rolling (int, optional): The rolling window size to use for the calculation. If set, CVaR is
            calculated over a rolling window of this many periods across the full return history instead
            of per `period` (e.g. a rolling 60-day CVaR). Only available for
            `distribution="historic"`; see `get_acerbi_szekely_test` for a rolling, out-of-sample CVaR
            path under the parametric distributions. Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the CVaR values over time. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            distribution (str): The distribution to use for the CVaR calculations (historic, gaussian, studentt, laplace,
            logistic, cornish-fisher or evt). Defaults to "historic".
            threshold_percentile (float, optional): Only used when `distribution` is "evt". The percentile
            of losses above which the Generalized Pareto Distribution is fitted; `alpha` can be at most
            1 - `threshold_percentile`. Defaults to 0.95.

        Returns:
            pd.Series: CVaR values with time as the index.

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates CVaR for each
        asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of CVaR values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AMZN", "TSLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_conditional_value_at_risk(period="yearly", within_period=True)
        ```

        Which returns:

        |      |    AMZN |    TSLA |   Benchmark |
        |:-----|--------:|--------:|------------:|
        | 2021 | -0.0327 | -0.0683 |     -0.0183 |
        | 2022 | -0.0685 | -0.0914 |     -0.0335 |
        | 2023 | -0.0413 | -0.071  |     -0.0157 |
        | 2024 | -0.0372 | -0.0776 |     -0.0189 |
        | 2025 | -0.047  | -0.0821 |     -0.0275 |
        """
        period = period if period else "daily"

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )
        if (
            period == "daily"
            and within_period
            and self._historical_data["intraday"].empty
        ):
            raise ValueError(
                "Intraday data is required for within-period daily calculations. Either set "
                "within_period=False or initialise the Toolkit with an intraday_period."
            )
        if growth and not within_period and not rolling:
            raise ValueError(
                "Growth calculations need one value per period, while "
                "within_period=False produces a single value for the entire period. "
                "Set within_period=True to calculate growth."
            )

        if rolling and distribution != "historic":
            raise ValueError(
                "The rolling parameter is only available for distribution='historic'. "
                "Use get_acerbi_szekely_test to obtain a rolling, out-of-sample CVaR "
                "path under one of the parametric distributions instead."
            )

        if rolling:
            returns = self._historical_data[period]["Return"]
            conditional_value_at_risk = cvar_model.get_rolling_cvar_historic(
                returns, alpha, rolling
            )
        else:
            returns = self._get_returns(period, within_period)

            if distribution == "historic":
                conditional_value_at_risk = cvar_model.get_cvar_historic(returns, alpha)
            elif distribution == "gaussian":
                conditional_value_at_risk = cvar_model.get_cvar_gaussian(returns, alpha)
            elif distribution == "studentt":
                conditional_value_at_risk = cvar_model.get_cvar_studentt(returns, alpha)
            elif distribution == "laplace":
                conditional_value_at_risk = cvar_model.get_cvar_laplace(returns, alpha)
            elif distribution == "logistic":
                conditional_value_at_risk = cvar_model.get_cvar_logistic(returns, alpha)
            elif distribution == "cornish-fisher":
                conditional_value_at_risk = cvar_model.get_cvar_cornish_fisher(
                    returns, alpha
                )
            elif distribution == "evt":
                conditional_value_at_risk = cvar_model.get_cvar_evt(
                    returns, alpha, threshold_percentile
                )
            else:
                raise ValueError(
                    "Distribution must be historic, gaussian, studentt, laplace, "
                    "logistic, cornish-fisher or evt."
                )

        if rolling or within_period:
            conditional_value_at_risk = conditional_value_at_risk.loc[
                self._start_date : self._end_date
            ]

        return finalize_dataset(
            dataset=conditional_value_at_risk,
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

    @handle_portfolio
    @handle_errors
    def get_entropic_value_at_risk(
        self,
        period: str | None = None,
        alpha: float = 0.05,
        within_period: bool = False,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ):
        """
        Calculate the Entropic Value at Risk (EVaR) of an investment portfolio or asset's returns.

        Entropic Value at Risk (EVaR) is a risk management metric that quantifies upper bound for the value
        at risk (VaR) and the conditional value at risk (CVaR) over a specified time horizon and confidence
        level. EVaR is obtained from the Chernoff inequality. It provides insights into the downside risk
        associated with an investment and helps investors make informed decisions about risk tolerance.

        The EVaR is calculated under the assumption that returns are normally distributed, for which the
        Chernoff bound has the closed form Mean - Standard Deviation * SQRT(-2 * ln(alpha)), where alpha is
        the tail probability (0.05 for the 95% confidence level) and the standard deviation is the sample
        standard deviation. It is an upper bound of the gaussian VaR and CVaR at the same confidence level.
        Since it only uses the mean and the standard deviation of the returns, it does not reflect their
        skewness or fat tails, and at high confidence levels it can be less severe than the historical CVaR
        of an asset with heavy tails.

        For more information about the method, see the following paper:

        - Ahmadi-Javid, A. (2012). "Entropic Value-at-Risk: A New Coherent Risk Measure." Journal of
        Optimization Theory and Applications, 155(3), 1105-1123.

        Also known as: EVaR.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "daily".
            alpha (float, optional): The tail probability for the EVaR calculation (e.g., 0.05 for the 95%
            confidence level). Defaults to 0.05.
            within_period (bool, optional): Whether to calculate EVaR within the specified period or for the entire
            period. Thus whether to look at the CVaR within a specific year (if period = 'yearly') or look at the entirety
            of all years. Defaults to False. Note that this requires intraday data when period = 'daily', since a
            single day only nests observations when intraday data was fetched.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the CVaR values over time. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.Series: EVaR values with time as the index.

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates EVaR for each
        asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of EVaR values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AMZN", "TSLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_entropic_value_at_risk(period="yearly", within_period=True)
        ```

        Which returns:

        |      |    AMZN |    TSLA |   Benchmark |
        |:-----|--------:|--------:|------------:|
        | 2021 | -0.0368 | -0.0824 |     -0.0188 |
        | 2022 | -0.0795 | -0.1061 |     -0.0381 |
        | 2023 | -0.0483 | -0.08   |     -0.0192 |
        | 2024 | -0.0417 | -0.0952 |     -0.0185 |
        | 2025 | -0.0526 | -0.0967 |     -0.0293 |
        """
        period = period if period else "daily"

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )
        if (
            period == "daily"
            and within_period
            and self._historical_data["intraday"].empty
        ):
            raise ValueError(
                "Intraday data is required for within-period daily calculations. Either set "
                "within_period=False or initialise the Toolkit with an intraday_period."
            )
        if growth and not within_period:
            raise ValueError(
                "Growth calculations need one value per period, while "
                "within_period=False produces a single value for the entire period. "
                "Set within_period=True to calculate growth."
            )

        returns = self._get_returns(period, within_period)

        entropic_value_at_risk = evar_model.get_evar_gaussian(returns, alpha)

        if within_period:
            entropic_value_at_risk = entropic_value_at_risk.loc[
                self._start_date : self._end_date
            ]

        return finalize_dataset(
            dataset=entropic_value_at_risk,
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

    @handle_portfolio
    @handle_errors
    def get_conditional_drawdown_at_risk(
        self,
        period: str | None = None,
        alpha: float = 0.05,
        within_period: bool = False,
        rolling: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ):
        """
        Calculate the Conditional Drawdown at Risk (CDaR) of an investment portfolio or asset's returns.

        Conditional Drawdown at Risk (CDaR) extends the concept of Value at Risk and Conditional Value at
        Risk to the drawdown series instead of the return series. It is calculated as the average of the
        worst drawdowns that exceed the Drawdown at Risk (DaR), i.e. the alpha-quantile of the drawdown
        distribution, giving insight into the depth of the most severe drawdowns an investment portfolio or
        asset could experience.

        Also known as: CDaR.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "daily".
            alpha (float, optional): The tail probability for the CDaR calculation (e.g., 0.05 for the 95%
            confidence level). Defaults to 0.05.
            within_period (bool, optional): Whether to calculate CDaR within the specified period or for the entire
            period. Thus whether to look at the CDaR within a specific year (if period = 'yearly') or look at the entirety
            of all years. Defaults to False. Note that this requires intraday data when period = 'daily', since a
            single day only nests observations when intraday data was fetched.
            rolling (int, optional): The rolling window size to use for the calculation. If set, CDaR is
            calculated over a rolling window of this many periods across the full return history instead
            of per `period`. Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the CDaR values over time. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.Series: CDaR values with time as the index.

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates CDaR for each
        asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of CDaR values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AMZN", "TSLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_conditional_drawdown_at_risk(period="yearly", within_period=True)
        ```

        Which returns:

        |      |    AMZN |    TSLA |   Benchmark |
        |:-----|--------:|--------:|------------:|
        | 2021 | -0.1324 | -0.3407 |     -0.0413 |
        | 2022 | -0.4989 | -0.6603 |     -0.2343 |
        | 2023 | -0.1756 | -0.2867 |     -0.0806 |
        | 2024 | -0.1612 | -0.3664 |     -0.0576 |
        | 2025 | -0.2721 | -0.4558 |     -0.1434 |
        """
        period = period if period else "daily"

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )
        if (
            period == "daily"
            and within_period
            and self._historical_data["intraday"].empty
        ):
            raise ValueError(
                "Intraday data is required for within-period daily calculations. Either set "
                "within_period=False or initialise the Toolkit with an intraday_period."
            )
        if growth and not within_period and not rolling:
            raise ValueError(
                "Growth calculations need one value per period, while "
                "within_period=False produces a single value for the entire period. "
                "Set within_period=True to calculate growth."
            )

        if rolling:
            returns = self._historical_data[period]["Return"]
            conditional_drawdown_at_risk = (
                risk_model.get_rolling_conditional_drawdown_at_risk(
                    returns, alpha, rolling
                )
            )
        else:
            returns = self._get_returns(period, within_period)

            conditional_drawdown_at_risk = risk_model.get_conditional_drawdown_at_risk(
                returns, alpha
            )

        if rolling or within_period:
            conditional_drawdown_at_risk = conditional_drawdown_at_risk.loc[
                self._start_date : self._end_date
            ]

        return finalize_dataset(
            dataset=conditional_drawdown_at_risk,
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

    @handle_portfolio
    @handle_errors
    def get_tail_ratio(
        self,
        period: str | None = None,
        alpha: float = 0.05,
        within_period: bool = False,
        rolling: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ):
        """
        Calculate the Tail Ratio of an investment portfolio or asset's returns.

        The Tail Ratio compares the size of the right (gain) tail to the left (loss) tail of the
        return distribution, calculated as the absolute value of the (1 - alpha)-th percentile of
        returns divided by the absolute value of the alpha-th percentile of returns. A Tail Ratio
        above 1 indicates that best-case gains outsize worst-case losses.

        Also known as: gain-to-pain tail ratio.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "daily".
            alpha (float, optional): The percentile used to define each tail (e.g., 0.05 uses the 5th and
            95th percentile). Defaults to 0.05.
            within_period (bool, optional): Whether to calculate the Tail Ratio within the specified period or
            for the entire period. Thus whether to look at the Tail Ratio within a specific year (if period =
            'yearly') or look at the entirety of all years. Defaults to False.
            Note that this requires intraday data when period = 'daily', since a
            single day only nests observations when intraday data was fetched.
            rolling (int, optional): The rolling window size to use for the calculation. If set, the Tail
            Ratio is calculated over a rolling window of this many periods across the full return history
            instead of per `period`. Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the Tail Ratio values over time.
            Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.Series: Tail Ratio values with time as the index.

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates the Tail
        Ratio for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of Tail Ratio values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AMZN", "TSLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_tail_ratio(period="yearly", within_period=True)
        ```

        Which returns:

        |      |   AMZN |   TSLA |   Benchmark |
        |:-----|-------:|-------:|------------:|
        | 2021 | 0.8574 | 1.017  |      1.1304 |
        | 2022 | 0.8415 | 1.0049 |      0.9524 |
        | 2023 | 1.2609 | 1.1958 |      1.0466 |
        | 2024 | 1.1246 | 1.1942 |      0.9091 |
        | 2025 | 0.9359 | 1.0702 |      0.93   |
        """
        period = period if period else "daily"

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )
        if (
            period == "daily"
            and within_period
            and self._historical_data["intraday"].empty
        ):
            raise ValueError(
                "Intraday data is required for within-period daily calculations. Either set "
                "within_period=False or initialise the Toolkit with an intraday_period."
            )
        if growth and not within_period and not rolling:
            raise ValueError(
                "Growth calculations need one value per period, while "
                "within_period=False produces a single value for the entire period. "
                "Set within_period=True to calculate growth."
            )

        if rolling:
            returns = self._historical_data[period]["Return"]
            tail_ratio = risk_model.get_rolling_tail_ratio(returns, alpha, rolling)
        else:
            returns = self._get_returns(period, within_period)

            tail_ratio = risk_model.get_tail_ratio(returns, alpha)

        if rolling or within_period:
            tail_ratio = tail_ratio.loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=tail_ratio,
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

    @handle_portfolio
    @handle_errors
    def get_maximum_drawdown(
        self,
        period: str | None = None,
        within_period: bool = False,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ):
        """
        Calculate the Maximum Drawdown (MDD) of an investment portfolio or asset's returns.

        Maximum Drawdown (MDD) is a risk management metric that quantifies the largest historical loss of
        an investment portfolio or asset experienced over a specified time horizon. It provides insights into
        the downside risk associated with an investment and helps investors make informed decisions about
        risk tolerance.

        The Maximum Drawdown is a single-path statistic: it depends entirely on the window chosen, so a
        figure over a window that includes a market crash and one over a window that starts after it are
        not comparable. It says how deep the worst fall was, not how long it lasted, which is why it is
        usually read together with the drawdown duration and recovery time (see
        `get_maximum_drawdown_duration` and `get_maximum_drawdown_recovery_time`), and it underlies
        return-to-risk ratios such as the Calmar ratio.

        Also known as: max drawdown, peak-to-trough decline.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "daily".
            within_period (bool, optional): Whether to calculate the Maximum Drawdown within the specified period
            or for the entire period. Thus whether to look at the Maximum Drawdown within a specific year
            (if period = 'yearly') or look at the entirety of all years. Defaults to False.
            Note that this requires intraday data when period = 'daily', since a
            single day only nests observations when intraday data was fetched.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the Maximum Drawdown values over time.
            Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.Series: Maximum Drawdown values with time as the index.

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates MMD for each
        asset in the Toolkit instance.
        - Without `within_period`, the drawdown is measured on the returns of the chosen period, so
        period="yearly" only compares year-end values: it is 0 when every year in the range closed at a
        new high, even if prices fell within a year. Use period="daily" for the deepest decline, or
        within_period=True for the deepest decline within each year.
        - If `growth` is set to True, the method calculates the growth of MMD values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AMZN", "TSLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_maximum_drawdown(period="yearly", within_period=True)
        ```

        Which returns:

        |      |    AMZN |    TSLA |   Benchmark |
        |:-----|--------:|--------:|------------:|
        | 2021 | -0.1456 | -0.3625 |     -0.051  |
        | 2022 | -0.5197 | -0.7272 |     -0.245  |
        | 2023 | -0.1965 | -0.3273 |     -0.0999 |
        | 2024 | -0.1949 | -0.4281 |     -0.0841 |
        | 2025 | -0.3088 | -0.4821 |     -0.1877 |
        """
        period = period if period else "daily"

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )
        if (
            period == "daily"
            and within_period
            and self._historical_data["intraday"].empty
        ):
            raise ValueError(
                "Intraday data is required for within-period daily calculations. Either set "
                "within_period=False or initialise the Toolkit with an intraday_period."
            )
        if growth and not within_period:
            raise ValueError(
                "Growth calculations need one value per period, while "
                "within_period=False produces a single value for the entire period. "
                "Set within_period=True to calculate growth."
            )

        returns = self._get_returns(period, within_period)

        maximum_drawdown = risk_model.get_max_drawdown(returns)

        if within_period:
            maximum_drawdown = maximum_drawdown.loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=maximum_drawdown,
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

    @handle_portfolio
    @handle_errors
    def get_maximum_drawdown_duration(
        self,
        period: str | None = None,
        within_period: bool = False,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ):
        """
        Calculate the Maximum Drawdown Duration of an investment portfolio or asset's returns.

        The Maximum Drawdown Duration is the number of periods between the peak and the lowest point
        of the largest drawdown, giving insight into how long the worst loss of value took to unfold.

        Also known as: drawdown length.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "daily".
            within_period (bool, optional): Whether to calculate the duration within the specified period or
            for the entire period. Thus whether to look at the duration within a specific year (if period =
            'yearly') or look at the entirety of all years. Defaults to False.
            Note that this requires intraday data when period = 'daily', since a
            single day only nests observations when intraday data was fetched.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the duration values over time. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.Series: Maximum Drawdown Duration values, in number of periods, with time as the index.

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates the Maximum
        Drawdown Duration for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the duration values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AMZN", "TSLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_maximum_drawdown_duration(period="yearly", within_period=True)
        ```

        Which returns:

        |      |   AMZN |   TSLA |   Benchmark |
        |:-----|-------:|-------:|------------:|
        | 2021 |     30 |     28 |          21 |
        | 2022 |    248 |    247 |         195 |
        | 2023 |     25 |     73 |          63 |
        | 2024 |     23 |     77 |          14 |
        | 2025 |     52 |     57 |          34 |
        """
        period = period if period else "daily"

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )
        if (
            period == "daily"
            and within_period
            and self._historical_data["intraday"].empty
        ):
            raise ValueError(
                "Intraday data is required for within-period daily calculations. Either set "
                "within_period=False or initialise the Toolkit with an intraday_period."
            )
        if growth and not within_period:
            raise ValueError(
                "Growth calculations need one value per period, while "
                "within_period=False produces a single value for the entire period. "
                "Set within_period=True to calculate growth."
            )

        returns = self._get_returns(period, within_period)

        maximum_drawdown_duration = risk_model.get_max_drawdown_duration(returns)

        if within_period:
            maximum_drawdown_duration = maximum_drawdown_duration.loc[
                self._start_date : self._end_date
            ]

        return finalize_dataset(
            dataset=maximum_drawdown_duration,
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

    @handle_portfolio
    @handle_errors
    def get_maximum_drawdown_recovery_time(
        self,
        period: str | None = None,
        within_period: bool = False,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ):
        """
        Calculate the Maximum Drawdown Recovery Time of an investment portfolio or asset's returns.

        The Maximum Drawdown Recovery Time is the number of periods it takes for the cumulative return
        to reach a new high after the lowest point of the largest drawdown. If the drawdown has not yet been
        recovered from within the selected period, this returns NaN.

        Also known as: time to recovery, drawdown recovery.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "daily".
            within_period (bool, optional): Whether to calculate the recovery time within the specified period
            or for the entire period. Thus whether to look at the recovery time within a specific year (if
            period = 'yearly') or look at the entirety of all years. Defaults to False.
            Note that this requires intraday data when period = 'daily', since a
            single day only nests observations when intraday data was fetched.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the recovery time values over time.
            Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.Series: Maximum Drawdown Recovery Time values, in number of periods, with time as the index.

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates the Maximum
        Drawdown Recovery Time for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the recovery time values using the
        specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AMZN", "TSLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_maximum_drawdown_recovery_time(period="yearly", within_period=True)
        ```

        Which returns:

        |      |   AMZN |   TSLA |   Benchmark |
        |:-----|-------:|-------:|------------:|
        | 2021 |    nan |    159 |          12 |
        | 2022 |    nan |    nan |         nan |
        | 2023 |     46 |    nan |          23 |
        | 2024 |     66 |     51 |          32 |
        | 2025 |    135 |    114 |          54 |
        """
        period = period if period else "daily"

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )
        if (
            period == "daily"
            and within_period
            and self._historical_data["intraday"].empty
        ):
            raise ValueError(
                "Intraday data is required for within-period daily calculations. Either set "
                "within_period=False or initialise the Toolkit with an intraday_period."
            )
        if growth and not within_period:
            raise ValueError(
                "Growth calculations need one value per period, while "
                "within_period=False produces a single value for the entire period. "
                "Set within_period=True to calculate growth."
            )

        returns = self._get_returns(period, within_period)

        maximum_drawdown_recovery_time = risk_model.get_max_drawdown_recovery_time(
            returns
        )

        if within_period:
            maximum_drawdown_recovery_time = maximum_drawdown_recovery_time.loc[
                self._start_date : self._end_date
            ]

        return finalize_dataset(
            dataset=maximum_drawdown_recovery_time,
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

    @handle_portfolio
    @handle_errors
    def get_ulcer_index(
        self,
        period: str | None = None,
        rolling: int | None = 14,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ):
        """
        The Ulcer Index is a financial metric used to assess the risk and volatility of an
        investment portfolio or asset. Developed by Peter Martin in the 1980s, the Ulcer Index
        is particularly useful for evaluating the downside risk and drawdowns associated with investments.

        The Ulcer Index differs from traditional volatility measures like standard deviation or variance
        because it focuses on the depth and duration of drawdowns rather than the dispersion of
        returns.

        The formula is as follows:

        Ulcer Index = SQRT(SUM[((Pn - Highest High) / Highest High)^2] / n)

        Martin measures every drawdown from the highest value so far (`rolling=None`); the default
        trailing 14-period high is the variant StockCharts popularised. While fewer than `rolling`
        observations are available, the high is taken over the observations so far, so periods shorter
        than the window still have a value. The result is a decimal (0.05 is a root mean squared
        drawdown of 5%), where Martin quotes percentages.

        The Ulcer Index punishes both the depth and the length of drawdowns, so an asset that falls 10% and
        stays there for months scores worse than one that drops 10% and recovers within a week, which a
        volatility figure would not show. It is the risk measure behind the Martin ratio (excess return
        divided by the Ulcer Index) and is read best relative to peers over the same window, since the
        default 14-period lookback gives lower values than a since-inception high-water mark.

        Also known as: UI, drawdown risk.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "quarterly" if the Toolkit is initialised with quarterly=True, otherwise "yearly".
            rolling (int | None, optional): The trailing lookback window used as the high-water
                mark reference for each day's drawdown. Pass None for an expanding
                (since-inception) high-water mark instead -- Martin's "Highest High"; a fixed
                int window is the common trailing-high variant (e.g. a 14-day trailing high).
                Defaults to 14.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the UI values over time. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.Series: UI values with time as the index.

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates UI for each
        asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of UI values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AMZN", "TSLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_ulcer_index()
        ```

        Which returns:

        |      |   AMZN |   TSLA |   Benchmark |
        |:-----|-------:|-------:|------------:|
        | 2021 | 0.0418 | 0.0833 |      0.013  |
        | 2022 | 0.106  | 0.1365 |      0.0481 |
        | 2023 | 0.0472 | 0.0827 |      0.0198 |
        | 2024 | 0.0388 | 0.0908 |      0.0164 |
        | 2025 | 0.0555 | 0.0982 |      0.027  |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )
        if period == "daily" and self._historical_data["intraday"].empty:
            raise ValueError("Intraday data is required for daily calculations.")

        returns = self._within_historical_data[period]["Return"]

        ulcer_index = risk_model.get_ui(returns, rolling)

        ulcer_index = ulcer_index.loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=ulcer_index,
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

    @handle_portfolio
    @handle_errors
    def get_garch(
        self,
        period: str | None = None,
        time_steps: int | None = None,
        optimization_t: int | None = None,
        within_period: bool = False,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ):
        """
        Calculates the conditional variance (sigma_t^2) of the returns with a GARCH(1, 1) model fitted to
        them. Take the square root for the conditional volatility, and see `get_garch_forecast` for the
        variance forecast beyond the last observation.

        GARCH (Generalized autoregressive conditional heteroskedasticity) is stochastic model for time series, which is
        for instance used to model volatility clusters, stock return and inflation. It is a generalisation of the ARCH
        models.

        Also known as: GARCH, volatility clustering, conditional heteroscedasticity.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "daily".
            time_steps (int, optional): Time steps to calculate GARCH for.
            optimization_t (int, optional): Time steps to optimize GARCH for. It is only used if no weights are given.
            within_period (bool, optional): Whether to calculate GARCH within the specified period or for the entire
            period. Thus whether to look at the GARCH within a specific year (if period = 'yearly') or look at the
            entirety of all years. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the GARCH values over time. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.DataFrame | pd.Series: GARCH values

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates GARCH for each
        asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of GARCH values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AMZN", "TSLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_garch(period="quarterly")
        ```

        Which returns:

        | Date   |   AMZN |   TSLA |   Benchmark |
        |:-------|-------:|-------:|------------:|
        | 2021Q2 | 0.0254 | 0.103  |      0.0066 |
        | 2021Q3 | 0.0254 | 0.0854 |      0.0065 |
        | 2021Q4 | 0.0254 | 0.0888 |      0.0064 |
        | 2022Q1 | 0.0254 | 0.1084 |      0.0063 |
        | 2022Q2 | 0.0254 | 0.0854 |      0.0063 |
        | 2022Q3 | 0.0254 | 0.11   |      0.0062 |
        | 2022Q4 | 0.0254 | 0.0911 |      0.0061 |
        | 2023Q1 | 0.0254 | 0.1356 |      0.006  |
        | 2023Q2 | 0.0254 | 0.1675 |      0.0059 |
        | 2023Q3 | 0.0254 | 0.0973 |      0.0058 |
        | 2023Q4 | 0.0254 | 0.0856 |      0.0058 |
        | 2024Q1 | 0.0254 | 0.0853 |      0.0057 |
        | 2024Q2 | 0.0254 | 0.1003 |      0.0056 |
        | 2024Q3 | 0.0254 | 0.0881 |      0.0055 |
        | 2024Q4 | 0.0254 | 0.1035 |      0.0054 |
        | 2025Q1 | 0.0254 | 0.1372 |      0.0054 |
        | 2025Q2 | 0.0254 | 0.1078 |      0.0053 |
        | 2025Q3 | 0.0254 | 0.0942 |      0.0052 |
        | 2025Q4 | 0.0254 | 0.1134 |      0.0051 |
        """
        period = period if period else "daily"

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )
        if (
            period == "daily"
            and within_period
            and self._historical_data["intraday"].empty
        ):
            raise ValueError(
                "Intraday data is required for within-period daily calculations. Either set "
                "within_period=False or initialise the Toolkit with an intraday_period."
            )

        returns = self._get_returns(period, within_period).dropna().replace(0, 1e-100)

        garch_sigma_2 = garch_model.get_garch(
            returns=returns,
            weights=None,
            time_steps=time_steps,
            optimization_t=optimization_t,
        )

        garch_sigma_2 = garch_sigma_2.loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=garch_sigma_2,
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

    @handle_portfolio
    @handle_errors
    def get_garch_forecast(
        self,
        period: str | None = None,
        time_steps: int = 10,
        within_period: bool = False,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ):
        """
        Calculates sigma_2 forecasts.

        GARCH (Generalized autoregressive conditional heteroskedasticity) is stochastic model for time series, which is
        for instance used to model volatility clusters, stock return and inflation. It is a generalisation of the ARCH
        models.

        The forecasting with GARCH is done with the following formula:

        - sigma_l ** 2 + (sigma_t ** 2 - sigma_l ** 2) * (alpha + beta) ** (t - 1)

        For more information about the method, see the following book:

        - Finance Compact Plus Band 1, by Yvonne Seler Zimmerman and Heinz Zimmerman; ISBN: 978-3-907291-31-1

        Also known as: volatility forecast, predicted volatility.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "daily".
            time_steps (int, optional): Time steps to calculate GARCH and to forecast sigma_2 values for.
            within_period (bool, optional): Whether to calculate GARCH within each specified period or all
            at once. Thus whether to look at the GARCH within each specific year (if period = 'yearly') or
            look at the entirety of all years. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            growth (bool, optional): Whether to calculate the growth of the GARCH values over time. Defaults to
            False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.DataFrame | pd.Series: sigma_2 forecast values

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates the sigma_2
        forecast for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the forecasted sigma_2 values using
        the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AMZN", "TSLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_garch_forecast(period="quarterly")
        ```

        Which returns:

        |        |   AMZN |   TSLA |   Benchmark |
        |:-------|-------:|-------:|------------:|
        | 2026Q1 | 0.0254 | 0.0853 |      0.0051 |
        | 2026Q2 | 0.0254 | 0.1003 |      0.005  |
        | 2026Q3 | 0.0254 | 0.1029 |      0.0049 |
        | 2026Q4 | 0.0254 | 0.1034 |      0.0049 |
        | 2027Q1 | 0.0254 | 0.1034 |      0.0048 |
        | 2027Q2 | 0.0254 | 0.1035 |      0.0047 |
        | 2027Q3 | 0.0254 | 0.1035 |      0.0047 |
        | 2027Q4 | 0.0254 | 0.1035 |      0.0046 |
        | 2028Q1 | 0.0254 | 0.1035 |      0.0045 |
        | 2028Q2 | 0.0254 | 0.1035 |      0.0045 |
        """
        period = period if period else "daily"

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )
        if (
            period == "daily"
            and within_period
            and self._historical_data["intraday"].empty
        ):
            raise ValueError(
                "Intraday data is required for within-period daily calculations. Either set "
                "within_period=False or initialise the Toolkit with an intraday_period."
            )

        returns = self._get_returns(period, within_period).dropna().replace(0, 1e-100)

        sigma_2_forecast = garch_model.get_garch_forecast(
            returns, None, time_steps
        ).dropna()

        # The forecast steps continue the period of the returns (business days for daily).
        period_symbol = {
            "daily": "B",
            "weekly": "W",
            "monthly": "ME",
            "quarterly": "QE",
        }.get(period, "YE")
        # Business days are kept as daily periods, as business-day periods are deprecated.
        period_index = pd.PeriodIndex(
            pd.date_range(
                start=returns.index[-1].to_timestamp(),
                periods=time_steps + 1,
                freq=period_symbol,
            ),
            freq="D" if period_symbol == "B" else None,
        )

        sigma_2_forecast.index = period_index[1:]

        return finalize_dataset(
            dataset=sigma_2_forecast,
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

    @handle_portfolio
    @handle_errors
    def get_garch_parameters(
        self,
        period: str | None = None,
        optimization_t: int | None = None,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Fits a GARCH(1, 1) model to the historical returns and returns the estimated
        Omega, Alpha and Beta parameters for each asset.

        GARCH (Generalized autoregressive conditional heteroskedasticity) is a stochastic model
        for time series, used to model volatility clustering. A GARCH(1, 1) model expresses the
        conditional variance sigma_t^2 as:

        - sigma_t^2 = Omega + Alpha * u_(t-1)^2 + Beta * sigma_(t-1)^2

        With the constraints Omega, Alpha, Beta > 0 and Alpha + Beta < 1. The parameters are
        estimated by maximizing the GARCH log-likelihood with L-BFGS-B, started from a
        domain-informed initial guess and from 32 points of a Sobol sequence, and the best fit is
        finished with Nelder-Mead. The search runs on the parameters scaled to their bounds, as
        Omega (of the order of a daily variance) and Alpha and Beta differ by orders of magnitude.

        The variance recursion starts from a backcast, an exponentially weighted average
        (decay 0.94) of the first 75 squared returns, which stands in for both the squared
        return and the variance before the first observation, and every observation enters the
        likelihood. This is the start the arch package uses, so the estimates agree with it.

        Unlike `get_garch` and `get_garch_forecast`, which return a (forecasted) conditional variance
        path, this method returns the fitted parameters themselves. This is useful when the parameters
        are needed directly, for example to seed a separate volatility simulation.

        For daily equity returns, Alpha is typically between 0.05 and 0.15 and Alpha + Beta (the
        persistence) between 0.95 and 0.99, meaning a volatility shock fades over weeks rather than
        days. The long-run variance the model reverts to is Omega / (1 - Alpha - Beta). A fit far
        outside these ranges is a reason to check the data and the window before using the parameters.

        For more information about the method, see the following book:

        - Finance Compact Plus Band 1, by Yvonne Seler Zimmerman and Heinz Zimmerman; ISBN: 978-3-907291-31-1

        Also known as: GARCH weights, GARCH coefficients, conditional variance parameters.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "daily".
            optimization_t (int, optional): Time steps of the returns series to use for the optimization.
            Defaults to the full length of the returns series.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: Omega, Alpha and Beta values per asset.

        Notes:
        - The method retrieves historical return data based on the specified `period` and fits a
        GARCH(1, 1) model for each asset in the Toolkit instance.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AMZN", "TSLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_garch_parameters(period="daily", rounding=6)
        ```

        Which returns:

        |       |     AMZN |     TSLA |   Benchmark |
        |:------|---------:|---------:|------------:|
        | Omega | 5.9e-05  | 3.3e-05  |    4e-06    |
        | Alpha | 0.199892 | 0.034278 |    0.10995  |
        | Beta  | 0.69455  | 0.943306 |    0.857926 |
        """
        period = period if period else "daily"

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )

        returns = self._historical_data[period]["Return"].dropna().replace(0, 1e-100)

        parameters = pd.DataFrame(
            {
                ticker: garch_model.get_garch_weights(
                    returns[ticker].to_numpy(), t=optimization_t
                )
                for ticker in returns.columns
            },
            index=["Omega", "Alpha", "Beta"],
        )

        return apply_rounding(
            parameters, rounding if rounding is not None else self._rounding
        )

    @handle_portfolio
    @handle_errors
    def get_gjr_garch(
        self,
        period: str | None = None,
        time_steps: int | None = None,
        optimization_t: int | None = None,
        within_period: bool = False,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ):
        """
        Calculates the conditional variance (sigma_t^2) of the returns with a GJR-GARCH(1, 1, 1) model
        fitted to them. Take the square root for the conditional volatility, and see
        `get_gjr_garch_forecast` for the variance forecast.

        GJR-GARCH extends GARCH with a leverage term that lets negative shocks (bad
        news) raise volatility by more than positive shocks of the same size, a well
        documented asymmetry in equity returns that symmetric GARCH cannot capture.

        For more information about the method, see the following paper:

        - Glosten, L.R., Jagannathan, R., and Runkle, D.E. (1993). "On the Relation
        between the Expected Value and the Volatility of the Nominal Excess Return on
        Stocks." The Journal of Finance, 48(5), 1779-1801.

        Also known as: GJR-GARCH, threshold GARCH, TGARCH.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "daily".
            time_steps (int, optional): Time steps to calculate GJR-GARCH for.
            optimization_t (int, optional): Time steps to optimize GJR-GARCH for. It is only used if no
            weights are given.
            within_period (bool, optional): Whether to calculate GJR-GARCH within the specified period or
            for the entire period. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the GJR-GARCH values over time.
            Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.DataFrame | pd.Series: GJR-GARCH values

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates
        GJR-GARCH for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of GJR-GARCH values using the
        specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AMZN", "TSLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_gjr_garch(period="quarterly")
        ```

        Which returns:

        | Date   |   AMZN |   TSLA |   Benchmark |
        |:-------|-------:|-------:|------------:|
        | 2021Q2 | 0.0263 | 0.1079 |      0.0066 |
        | 2021Q3 | 0.0242 | 0.0913 |      0.0065 |
        | 2021Q4 | 0.0229 | 0.0867 |      0.0064 |
        | 2022Q1 | 0.0216 | 0.0853 |      0.0063 |
        | 2022Q2 | 0.0208 | 0.085  |      0.0063 |
        | 2022Q3 | 0.0386 | 0.1369 |      0.0062 |
        | 2022Q4 | 0.0332 | 0.0994 |      0.0061 |
        | 2023Q1 | 0.0393 | 0.1951 |      0.006  |
        | 2023Q2 | 0.0338 | 0.1157 |      0.006  |
        | 2023Q3 | 0.0297 | 0.0935 |      0.0059 |
        | 2023Q4 | 0.0267 | 0.088  |      0.0059 |
        | 2024Q1 | 0.0245 | 0.0857 |      0.0058 |
        | 2024Q2 | 0.0228 | 0.1167 |      0.0058 |
        | 2024Q3 | 0.0216 | 0.0938 |      0.0057 |
        | 2024Q4 | 0.0209 | 0.0873 |      0.0057 |
        | 2025Q1 | 0.0201 | 0.0855 |      0.0056 |
        | 2025Q2 | 0.0223 | 0.1325 |      0.0056 |
        | 2025Q3 | 0.0212 | 0.0982 |      0.0056 |
        | 2025Q4 | 0.0204 | 0.0886 |      0.0055 |
        """
        period = period if period else "daily"

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )
        if (
            period == "daily"
            and within_period
            and self._historical_data["intraday"].empty
        ):
            raise ValueError(
                "Intraday data is required for within-period daily calculations. Either set "
                "within_period=False or initialise the Toolkit with an intraday_period."
            )

        returns = self._get_returns(period, within_period).dropna().replace(0, 1e-100)

        gjr_garch_sigma_2 = garch_model.get_gjr_garch(
            returns=returns,
            weights=None,
            time_steps=time_steps,
            optimization_t=optimization_t,
        )

        gjr_garch_sigma_2 = gjr_garch_sigma_2.loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=gjr_garch_sigma_2,
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

    @handle_portfolio
    @handle_errors
    def get_gjr_garch_forecast(
        self,
        period: str | None = None,
        time_steps: int = 10,
        within_period: bool = False,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ):
        """
        Calculates sigma_2 forecasts based on the GJR-GARCH model.

        For more information about the method, see the following paper:

        - Glosten, L.R., Jagannathan, R., and Runkle, D.E. (1993). "On the Relation
        between the Expected Value and the Volatility of the Nominal Excess Return on
        Stocks." The Journal of Finance, 48(5), 1779-1801.

        Also known as: volatility forecast, predicted volatility.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "daily".
            time_steps (int, optional): Time steps to calculate GJR-GARCH and to forecast sigma_2 values for.
            within_period (bool, optional): Whether to calculate GJR-GARCH within each specified period or
            all at once. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.
            growth (bool, optional): Whether to calculate the growth of the GJR-GARCH values over time.
            Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.DataFrame | pd.Series: sigma_2 forecast values

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates the
        sigma_2 forecast for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the forecasted sigma_2 values using
        the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AMZN", "TSLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_gjr_garch_forecast(period="quarterly")
        ```

        Which returns:

        |        |   AMZN |   TSLA |   Benchmark |
        |:-------|-------:|-------:|------------:|
        | 2026Q1 | 0.0198 | 0.0859 |      0.0055 |
        | 2026Q2 | 0.0209 | 0.101  |      0.0055 |
        | 2026Q3 | 0.0217 | 0.1081 |      0.0055 |
        | 2026Q4 | 0.0224 | 0.1113 |      0.0054 |
        | 2027Q1 | 0.023  | 0.1129 |      0.0054 |
        | 2027Q2 | 0.0235 | 0.1136 |      0.0054 |
        | 2027Q3 | 0.0239 | 0.1139 |      0.0054 |
        | 2027Q4 | 0.0242 | 0.114  |      0.0053 |
        | 2028Q1 | 0.0244 | 0.1141 |      0.0053 |
        | 2028Q2 | 0.0246 | 0.1142 |      0.0053 |
        """
        period = period if period else "daily"

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )
        if (
            period == "daily"
            and within_period
            and self._historical_data["intraday"].empty
        ):
            raise ValueError(
                "Intraday data is required for within-period daily calculations. Either set "
                "within_period=False or initialise the Toolkit with an intraday_period."
            )

        returns = self._get_returns(period, within_period).dropna().replace(0, 1e-100)

        sigma_2_forecast = garch_model.get_gjr_garch_forecast(
            returns, None, time_steps
        ).dropna()

        # The forecast steps continue the period of the returns (business days for daily).
        period_symbol = {
            "daily": "B",
            "weekly": "W",
            "monthly": "ME",
            "quarterly": "QE",
        }.get(period, "YE")
        # Business days are kept as daily periods, as business-day periods are deprecated.
        period_index = pd.PeriodIndex(
            pd.date_range(
                start=returns.index[-1].to_timestamp(),
                periods=time_steps + 1,
                freq=period_symbol,
            ),
            freq="D" if period_symbol == "B" else None,
        )

        sigma_2_forecast.index = period_index[1:]

        return finalize_dataset(
            dataset=sigma_2_forecast,
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

    @handle_portfolio
    @handle_errors
    def get_gjr_garch_parameters(
        self,
        period: str | None = None,
        optimization_t: int | None = None,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Fits a GJR-GARCH(1, 1, 1) model to the historical returns and returns the
        estimated Omega, Alpha, Gamma and Beta parameters for each asset.

        A positive Gamma indicates the presence of a leverage effect (negative shocks
        raise volatility by more than positive ones of the same size), which symmetric
        GARCH cannot represent.

        The variance recursion starts from a backcast, an exponentially weighted average
        (decay 0.94) of the first 75 squared returns, with the leverage term at its expectation
        (Gamma / 2) for the period before the first observation, and every observation enters
        the likelihood. This is the start the arch package uses, so the estimates agree with it.

        For more information about the method, see the following paper:

        - Glosten, L.R., Jagannathan, R., and Runkle, D.E. (1993). "On the Relation
        between the Expected Value and the Volatility of the Nominal Excess Return on
        Stocks." The Journal of Finance, 48(5), 1779-1801.

        Also known as: GJR-GARCH weights, GJR-GARCH coefficients, leverage parameters.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "daily".
            optimization_t (int, optional): Time steps of the returns series to use for the optimization.
            Defaults to the full length of the returns series.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.

        Returns:
            pd.DataFrame: Omega, Alpha, Gamma and Beta values per asset.

        Notes:
        - The method retrieves historical return data based on the specified `period` and fits a
        GJR-GARCH(1, 1, 1) model for each asset in the Toolkit instance.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AMZN", "TSLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_gjr_garch_parameters(period="daily", rounding=6)
        ```

        Which returns:

        |       |     AMZN |     TSLA |   Benchmark |
        |:------|---------:|---------:|------------:|
        | Omega | 2.9e-05  | 3.8e-05  |    3e-06    |
        | Alpha | 0.032278 | 0.016799 |    0        |
        | Gamma | 0.141638 | 0.042582 |    0.200974 |
        | Beta  | 0.843795 | 0.937679 |    0.874133 |
        """
        period = period if period else "daily"

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )

        returns = self._historical_data[period]["Return"].dropna().replace(0, 1e-100)

        parameters = pd.DataFrame(
            {
                ticker: garch_model.get_gjr_garch_weights(
                    returns[ticker].to_numpy(), t=optimization_t
                )
                for ticker in returns.columns
            },
            index=["Omega", "Alpha", "Gamma", "Beta"],
        )

        return apply_rounding(
            parameters, rounding if rounding is not None else self._rounding
        )

    @handle_portfolio
    @handle_errors
    def get_egarch(
        self,
        period: str | None = None,
        time_steps: int | None = None,
        optimization_t: int | None = None,
        within_period: bool = False,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ):
        """
        Calculates the conditional variance (sigma_t^2) of the returns with an EGARCH(1, 1) model fitted
        to them. Take the square root for the conditional volatility, and see `get_egarch_forecast` for
        the variance forecast.

        EGARCH models the log of the conditional variance, which avoids having to
        constrain the parameters to keep the variance positive and, like GJR-GARCH,
        lets negative and positive shocks of the same size have a different impact on
        volatility (the leverage effect).

        For more information about the method, see the following paper:

        - Nelson, D.B. (1991). "Conditional Heteroskedasticity in Asset Returns: A New
        Approach." Econometrica, 59(2), 347-370.

        Also known as: exponential GARCH, log-GARCH.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "daily".
            time_steps (int, optional): Time steps to calculate EGARCH for.
            optimization_t (int, optional): Time steps to optimize EGARCH for. It is only used if no
            weights are given.
            within_period (bool, optional): Whether to calculate EGARCH within the specified period or
            for the entire period. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the EGARCH values over time.
            Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.DataFrame | pd.Series: EGARCH values

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates
        EGARCH for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of EGARCH values using the
        specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AMZN", "TSLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_egarch(period="quarterly")
        ```

        Which returns:

        | Date   |   AMZN |   TSLA |   Benchmark |
        |:-------|-------:|-------:|------------:|
        | 2021Q2 | 0.0229 | 0.0923 |      0.0077 |
        | 2021Q3 | 0.0158 | 0.1671 |      0.0049 |
        | 2021Q4 | 0.0244 | 0.1704 |      0.0131 |
        | 2022Q1 | 0.0411 | 0.0597 |      0.0062 |
        | 2022Q2 | 0.0737 | 0.1035 |      0.0116 |
        | 2022Q3 | 0.0642 | 0.2104 |      0.009  |
        | 2022Q4 | 0.0868 | 0.1936 |      0.0145 |
        | 2023Q1 | 0.0975 | 0.3935 |      0.0105 |
        | 2023Q2 | 0.0675 | 0.0902 |      0.0078 |
        | 2023Q3 | 0.0321 | 0.0321 |      0.0048 |
        | 2023Q4 | 0.0565 | 0.0654 |      0.0108 |
        | 2024Q1 | 0.0348 | 0.133  |      0.0046 |
        | 2024Q2 | 0.0167 | 0.2705 |      0.0017 |
        | 2024Q3 | 0.0149 | 0.3389 |      0.002  |
        | 2024Q4 | 0.0239 | 0.2276 |      0.0015 |
        | 2025Q1 | 0.0094 | 0.0473 |      0.0033 |
        | 2025Q2 | 0.0077 | 0.0964 |      0.0076 |
        | 2025Q3 | 0.0013 | 0.0458 |      0.0033 |
        | 2025Q4 | 0.0025 | 0.0022 |      0.0017 |
        """
        period = period if period else "daily"

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )
        if (
            period == "daily"
            and within_period
            and self._historical_data["intraday"].empty
        ):
            raise ValueError(
                "Intraday data is required for within-period daily calculations. Either set "
                "within_period=False or initialise the Toolkit with an intraday_period."
            )

        returns = self._get_returns(period, within_period).dropna().replace(0, 1e-100)

        egarch_sigma_2 = garch_model.get_egarch(
            returns=returns,
            weights=None,
            time_steps=time_steps,
            optimization_t=optimization_t,
        )

        egarch_sigma_2 = egarch_sigma_2.loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=egarch_sigma_2,
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

    @handle_portfolio
    @handle_errors
    def get_egarch_forecast(
        self,
        period: str | None = None,
        time_steps: int = 10,
        within_period: bool = False,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ):
        """
        Calculates sigma_2 forecasts based on the EGARCH model.

        For more information about the method, see the following paper:

        - Nelson, D.B. (1991). "Conditional Heteroskedasticity in Asset Returns: A New
        Approach." Econometrica, 59(2), 347-370.

        Also known as: volatility forecast, predicted volatility.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "daily".
            time_steps (int, optional): Time steps to calculate EGARCH and to forecast sigma_2 values for.
            within_period (bool, optional): Whether to calculate EGARCH within each specified period or
            all at once. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.
            growth (bool, optional): Whether to calculate the growth of the EGARCH values over time.
            Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.DataFrame | pd.Series: sigma_2 forecast values

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates the
        sigma_2 forecast for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the forecasted sigma_2 values using
        the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AMZN", "TSLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_egarch_forecast(period="quarterly")
        ```

        Which returns:

        |        |   AMZN |   TSLA |   Benchmark |
        |:-------|-------:|-------:|------------:|
        | 2026Q1 | 0.0012 | 0.0028 |      0.0034 |
        | 2026Q2 | 0.001  | 0.0026 |      0.0055 |
        | 2026Q3 | 0.0009 | 0.0024 |      0.007  |
        | 2026Q4 | 0.0008 | 0.0022 |      0.0079 |
        | 2027Q1 | 0.0007 | 0.002  |      0.0084 |
        | 2027Q2 | 0.0006 | 0.0018 |      0.0086 |
        | 2027Q3 | 0.0005 | 0.0017 |      0.0087 |
        | 2027Q4 | 0.0004 | 0.0016 |      0.0088 |
        | 2028Q1 | 0.0004 | 0.0014 |      0.0089 |
        | 2028Q2 | 0.0003 | 0.0013 |      0.0089 |
        """
        period = period if period else "daily"

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )
        if (
            period == "daily"
            and within_period
            and self._historical_data["intraday"].empty
        ):
            raise ValueError(
                "Intraday data is required for within-period daily calculations. Either set "
                "within_period=False or initialise the Toolkit with an intraday_period."
            )

        returns = self._get_returns(period, within_period).dropna().replace(0, 1e-100)

        sigma_2_forecast = garch_model.get_egarch_forecast(
            returns, None, time_steps
        ).dropna()

        # The forecast steps continue the period of the returns (business days for daily).
        period_symbol = {
            "daily": "B",
            "weekly": "W",
            "monthly": "ME",
            "quarterly": "QE",
        }.get(period, "YE")
        # Business days are kept as daily periods, as business-day periods are deprecated.
        period_index = pd.PeriodIndex(
            pd.date_range(
                start=returns.index[-1].to_timestamp(),
                periods=time_steps + 1,
                freq=period_symbol,
            ),
            freq="D" if period_symbol == "B" else None,
        )

        sigma_2_forecast.index = period_index[1:]

        return finalize_dataset(
            dataset=sigma_2_forecast,
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

    @handle_portfolio
    @handle_errors
    def get_egarch_parameters(
        self,
        period: str | None = None,
        optimization_t: int | None = None,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Fits an EGARCH(1, 1) model to the historical returns and returns the estimated
        Omega, Alpha, Gamma and Beta parameters for each asset.

        A negative Gamma indicates the presence of a leverage effect (negative shocks
        raise volatility by more than positive ones of the same size).

        For more information about the method, see the following paper:

        - Nelson, D.B. (1991). "Conditional Heteroskedasticity in Asset Returns: A New
        Approach." Econometrica, 59(2), 347-370.

        Also known as: EGARCH weights, EGARCH coefficients, leverage parameters.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "daily".
            optimization_t (int, optional): Time steps of the returns series to use for the optimization.
            Defaults to the full length of the returns series.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.

        Returns:
            pd.DataFrame: Omega, Alpha, Gamma and Beta values per asset.

        Notes:
        - The method retrieves historical return data based on the specified `period` and fits an
        EGARCH(1, 1) model for each asset in the Toolkit instance.
        - EGARCH's log-variance parameterization is less constrained than plain GARCH, which makes it
        prone to unstable, boundary-hugging fits on short return histories, which include quarterly
        data over a few years. Prefer daily or weekly returns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AMZN", "TSLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_egarch_parameters(period="daily", rounding=6)
        ```

        Which returns:

        |       |      AMZN |      TSLA |   Benchmark |
        |:------|----------:|----------:|------------:|
        | Omega | -0.382664 | -0.225452 |   -0.309737 |
        | Alpha |  0.193573 |  0.081906 |    0.059855 |
        | Gamma | -0.080739 | -0.03535  |   -0.181222 |
        | Beta  |  0.949074 |  0.964858 |    0.96578  |
        """
        period = period if period else "daily"

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )

        returns = self._historical_data[period]["Return"].dropna().replace(0, 1e-100)

        parameters = pd.DataFrame(
            {
                ticker: garch_model.get_egarch_weights(
                    returns[ticker].to_numpy(), t=optimization_t
                )
                for ticker in returns.columns
            },
            index=["Omega", "Alpha", "Gamma", "Beta"],
        )

        return apply_rounding(
            parameters, rounding if rounding is not None else self._rounding
        )

    def _get_price_column(self, period: str, column: str) -> pd.DataFrame:
        # Reads the plain period-frequency history rather than the "within period" multi-index, so period="daily" simply means daily observations here and (unlike every within-period method above) needs no intraday data.  # noqa: E501
        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )

        # Only rows without any value are dropped: the pairwise calculations (copulas, tail dependence, CoVaR) align each pair themselves, so a joint dropna would cut every ticker's sample down to the shortest history.  # noqa: E501
        return self._historical_data[period][column].dropna(how="all")

    def _resolve_ticker_pairs(
        self, ticker_a: str | None, ticker_b: str | None
    ) -> list[tuple[str, str]]:
        if ticker_a is None and ticker_b is None:
            if len(self._tickers) < MINIMUM_TICKERS_FOR_ALL_PAIRS:
                raise ValueError(
                    "At least 2 tickers are required in the Toolkit instance to "
                    "calibrate a copula between every pair."
                )

            return list(itertools.combinations(self._tickers, 2))

        if ticker_a is None or ticker_b is None:
            raise ValueError(
                "Either provide both ticker_a and ticker_b, or neither (to use "
                "every pair of tickers in the Toolkit instance)."
            )

        return [(ticker_a, ticker_b)]

    @handle_errors
    def get_tail_dependence_coefficient(
        self,
        ticker_a: str,
        ticker_b: str,
        period: str | None = None,
        column: str = "Return",
        q: float = 0.95,
        method: str = "empirical",
        dof: float = 4.0,
        rounding: int | None = None,
    ) -> pd.Series:
        """
        Calculate the Upper and Lower Tail Dependence Coefficients between `ticker_a`
        and `ticker_b`.

        Correlation only captures the *average* co-movement between two assets -- it
        says nothing about whether they are more likely to crash together than an
        equivalent gaussian relationship would imply. The Tail Dependence Coefficient
        answers that specific question directly: the probability that one asset is in
        extreme distress, given that the other one already is.

        The empirical estimate at q = 0.95 rests on the few days on which both assets are
        in their worst (or best) 5% at the same time, so it needs several years of daily
        data to be stable. A gaussian dependence of any correlation below 1 implies zero
        tail dependence, so a Lower Tail Dependence Coefficient well above zero means that
        diversification between the two assets is weakest exactly when it is needed.

        For more information about the method, see the following papers:

        - Embrechts, P., McNeil, A., & Straumann, D. (1999). "Correlation: Pitfalls
        and Alternatives." RISK Magazine, 12, 69-71.
        - Poon, S.H., Rockinger, M., & Tawn, J. (2004). "Extreme Value Dependence in
        Financial Markets: Diagnostics, Models, and Financial Implications." Review of
        Financial Studies, 17(2), 581-610.

        Also known as: tail dependence, extremal dependence coefficient.

        Args:
            ticker_a (str): The first asset.
            ticker_b (str): The second asset.
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily", since a dependence estimate needs far more observations than a
                lower frequency provides -- at "yearly" a decade of history is only ten
                observations.
            column (str, optional): The historical data column to use. Defaults to "Return", since
            tail dependence between return series is the standard risk management application.
            q (float, optional): The threshold quantile used for the "empirical" method, in (0.5, 1).
            Defaults to 0.95.
            method (str, optional): The estimation method, one of "empirical", "gaussian" or
            "student-t". Defaults to "empirical".
            dof (float, optional): The degrees of freedom of the Student-T copula, only used when
            `method="student-t"`. Defaults to 4.0.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.

        Returns:
            pd.Series: The Lower and Upper Tail Dependence Coefficients, the linear (Pearson)
            correlation between the two assets, and the number of observations used.

        Notes:
        - The method retrieves historical data based on the specified `period` for the two given
        assets and estimates the tail dependence coefficient between them.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AMZN", "TSLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_tail_dependence_coefficient("AMZN", "TSLA", period="weekly")
        ```

        Which returns:

        |                       |    Value |
        |:----------------------|---------:|
        | Lower Tail Dependence |   0.0769 |
        | Upper Tail Dependence |   0.1538 |
        | Correlation           |   0.4393 |
        | Observations          | 260      |
        """
        period = period if period else "daily"
        returns = self._get_price_column(period, column)

        result = copula_model.get_tail_dependence_coefficient(
            returns[ticker_a], returns[ticker_b], q=q, method=method, dof=dof
        )

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_copula_parameters(
        self,
        ticker_a: str | None = None,
        ticker_b: str | None = None,
        copula: str = "gaussian",
        period: str | None = None,
        column: str = "Return",
        rounding: int | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Calibrate a bivariate copula between `ticker_a` and `ticker_b` on the
        pseudo-observations (ranks) of their returns: the Clayton, Gumbel and Frank
        parameters by maximum likelihood, the gaussian correlation from the normal
        scores, and the Student-T correlation from Kendall's tau with the degrees of
        freedom by maximum likelihood (Demarta & McNeil, 2005).

        When `ticker_a`/`ticker_b` are not given, every unique pair among the
        Toolkit's tickers is calibrated instead.

        A copula separates the dependence structure between two assets from
        their individual (marginal) return distributions (Sklar's theorem),
        letting the two be modeled independently. Five families are supported,
        each capturing a different shape of dependence:

        - "gaussian": zero tail dependence -- crashes and rallies are no more
        likely to happen together than the correlation alone implies. Included
        mainly as a baseline, since real asset returns typically show more
        joint tail risk than this.
        - "student-t": symmetric, nonzero tail dependence in both tails.
        - "clayton": nonzero *lower* tail dependence only -- assets crash
        together more than they rally together. The most common choice for
        equity return pairs.
        - "gumbel": nonzero *upper* tail dependence only -- assets rally
        together more than they crash together.
        - "frank": zero tail dependence in both tails, but (unlike gaussian) can
        represent negative dependence and is symmetric around independence.

        See `get_best_fitting_copula` to compare all five by AIC on the same
        pair of assets, and see `get_copula_simulation` to draw joint scenarios
        from the fitted copula.

        For more information about the method, see:

        - Sklar, A. (1959). Publications de l'Institut de Statistique de
        l'Universite de Paris, 8, 229-231.
        - McNeil, A.J., Frey, R., & Embrechts, P. (2015). "Quantitative Risk
        Management: Concepts, Techniques and Tools." Princeton University Press.
        - Demarta, S., & McNeil, A.J. (2005). "The T Copula and Related
        Copulas." International Statistical Review, 73(1), 111-129.

        Also known as: copula calibration, copula fit, dependence modeling.

        Args:
            ticker_a (str, optional): The first asset. Defaults to None, meaning every unique pair of
            tickers in the Toolkit instance is calibrated (requires `ticker_b` to also be None).
            ticker_b (str, optional): The second asset. Defaults to None, see `ticker_a`.
            copula (str, optional): The copula family to fit, one of "gaussian",
            "student-t", "clayton", "gumbel" or "frank". Defaults to "gaussian".
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily", since a dependence estimate needs far more observations than a
                lower frequency provides -- at "yearly" a decade of history is only ten
                observations.
            column (str, optional): The historical data column to use. Defaults to "Return".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.

        Raises:
            ValueError: If `copula` is not one of "gaussian", "student-t", "clayton", "gumbel" or "frank",
            if only one of `ticker_a`/`ticker_b` is given, if fewer than 2 tickers are available to pair
            up when neither is given, or if fewer than 10 paired, non-missing observations are available.

        Returns:
            pd.Series | pd.DataFrame: The fitted copula parameter(s), the Lower and Upper Tail
            Dependence implied by them, the Log-Likelihood, the AIC and the number of observations used
            -- a Series for a single given pair, or a DataFrame indexed by every (Ticker A, Ticker B)
            pair when neither ticker is given.

        Notes:
        - The method retrieves historical data based on the specified `period` and calibrates the
        requested copula between the given pair, or every pair, of tickers.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AMZN", "TSLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_copula_parameters("AMZN", "TSLA", copula="clayton", period="weekly")
        ```

        Which returns:

        |                       |    Value |
        |:----------------------|---------:|
        | Theta                 |   0.6155 |
        | Lower Tail Dependence |   0.3243 |
        | Upper Tail Dependence |   0      |
        | Log-Likelihood        |  21.68   |
        | AIC                   | -41.3599 |
        | Observations          | 260      |
        """
        period = period if period else "daily"
        returns = self._get_price_column(period, column)

        fit_functions = {
            "gaussian": copula_model.get_gaussian_copula_parameters,
            "student-t": copula_model.get_student_t_copula_parameters,
            "clayton": copula_model.get_clayton_copula_parameters,
            "gumbel": copula_model.get_gumbel_copula_parameters,
            "frank": copula_model.get_frank_copula_parameters,
        }

        if copula not in fit_functions:
            raise ValueError(f"copula must be one of {', '.join(fit_functions)}.")

        ticker_pairs = self._resolve_ticker_pairs(ticker_a, ticker_b)

        if len(ticker_pairs) == 1:
            ticker_a, ticker_b = ticker_pairs[0]
            result = pd.Series(
                fit_functions[copula](returns[ticker_a], returns[ticker_b])
            )

            return apply_rounding(
                result, rounding if rounding is not None else self._rounding
            )

        results = {
            (pair_a, pair_b): fit_functions[copula](returns[pair_a], returns[pair_b])
            for pair_a, pair_b in ticker_pairs
        }

        result_df = pd.DataFrame(results).T
        result_df.index.names = ["Ticker A", "Ticker B"]

        return apply_rounding(
            result_df, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_copula_simulation(
        self,
        ticker_a: str | None = None,
        ticker_b: str | None = None,
        copula: str = "gaussian",
        period: str | None = None,
        column: str = "Return",
        n_simulations: int = 10_000,
        random_state: int = 42,
        empirical_margins: bool = True,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Calibrate a bivariate copula (see `get_copula_parameters`) between
        `ticker_a` and `ticker_b`, and simulate joint scenarios from it.

        When `ticker_a`/`ticker_b` are not given, every unique pair among the
        Toolkit's tickers is calibrated and simulated instead, as columns
        grouped under a top-level (Ticker A, Ticker B) column per pair.

        This is what a copula-based portfolio Monte Carlo simulation needs: draws
        that preserve the calibrated dependence structure -- including tail
        dependence, for the "student-t", "clayton" and "gumbel" families -- rather
        than the (typically understated) joint crash risk a plain multivariate
        gaussian simulation would produce.

        Also known as: copula Monte Carlo, dependence simulation, scenario
        generation.

        Args:
            ticker_a (str, optional): The first asset. Defaults to None, meaning every unique pair of
            tickers in the Toolkit instance is used (requires `ticker_b` to also be None).
            ticker_b (str, optional): The second asset. Defaults to None, see `ticker_a`.
            copula (str, optional): The copula family to fit and simulate from, one of "gaussian",
            "student-t", "clayton", "gumbel" or "frank". Defaults to "gaussian".
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily", since a dependence estimate needs far more observations than a
                lower frequency provides -- at "yearly" a decade of history is only ten
                observations.
            column (str, optional): The historical data column to use. Defaults to "Return".
            n_simulations (int, optional): The number of joint draws to simulate. Defaults to 10,000.
            random_state (int, optional): The seed for the random number generator. Defaults to 42.
            empirical_margins (bool, optional): Whether to map the simulated pseudo-observations back
            to realistic returns via each asset's own empirical (historical) quantile function.
            Defaults to True. When False, the raw pseudo-observations (each in (0, 1)) are returned
            instead.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.

        Raises:
            ValueError: If `copula` is not one of "gaussian", "student-t", "clayton", "gumbel" or "frank",
            if only one of `ticker_a`/`ticker_b` is given, if fewer than 2 tickers are available to pair
            up when neither is given, or if fewer than 10 paired, non-missing observations are available.

        Returns:
            pd.DataFrame: `n_simulations` simulated joint draws, one column per asset (or "U"/"V" when
            `empirical_margins=False`) for a single given pair, or with columns grouped by
            (Ticker A, Ticker B) pair when neither ticker is given.

        Notes:
        - The method retrieves historical data based on the specified `period`, calibrates the
        requested copula, and simulates from it, for the given pair or every pair of tickers.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AMZN", "TSLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_copula_simulation(
            "AMZN", "TSLA", copula="clayton", period="weekly", n_simulations=5000
        ).describe()
        ```

        Which returns:

        |       |          AMZN |          TSLA |
        |:------|--------------:|--------------:|
        | count | 5000          | 5000          |
        | mean  |    0.00206652 |    0.00380104 |
        | std   |    0.0435259  |    0.0792412  |
        | min   |   -0.1382     |   -0.1796     |
        | 25%   |   -0.022325   |   -0.049125   |
        | 50%   |    0.0004     |   -0.0001     |
        | 75%   |    0.0297     |    0.052725   |
        | max   |    0.1385     |    0.3331     |
        """
        period = period if period else "daily"
        returns = self._get_price_column(period, column)

        fit_functions = {
            "gaussian": copula_model.get_gaussian_copula_parameters,
            "student-t": copula_model.get_student_t_copula_parameters,
            "clayton": copula_model.get_clayton_copula_parameters,
            "gumbel": copula_model.get_gumbel_copula_parameters,
            "frank": copula_model.get_frank_copula_parameters,
        }
        simulate_functions = {
            "gaussian": lambda parameters: copula_model.get_gaussian_copula_simulation(
                parameters["Rho"], n_simulations, random_state
            ),
            "student-t": lambda parameters: copula_model.get_student_t_copula_simulation(
                parameters["Rho"],
                parameters["Degrees of Freedom"],
                n_simulations,
                random_state,
            ),
            "clayton": lambda parameters: copula_model.get_clayton_copula_simulation(
                parameters["Theta"], n_simulations, random_state
            ),
            "gumbel": lambda parameters: copula_model.get_gumbel_copula_simulation(
                parameters["Theta"], n_simulations, random_state
            ),
            "frank": lambda parameters: copula_model.get_frank_copula_simulation(
                parameters["Theta"], n_simulations, random_state
            ),
        }

        if copula not in fit_functions:
            raise ValueError(f"copula must be one of {', '.join(fit_functions)}.")

        ticker_pairs = self._resolve_ticker_pairs(ticker_a, ticker_b)

        if len(ticker_pairs) == 1:
            ticker_a, ticker_b = ticker_pairs[0]
            simulation = self._simulate_copula_pair(
                ticker_a,
                ticker_b,
                returns,
                fit_functions[copula],
                simulate_functions[copula],
                empirical_margins,
            )

            return apply_rounding(
                simulation, rounding if rounding is not None else self._rounding
            )

        simulation = pd.concat(
            {
                (pair_a, pair_b): self._simulate_copula_pair(
                    pair_a,
                    pair_b,
                    returns,
                    fit_functions[copula],
                    simulate_functions[copula],
                    empirical_margins,
                )
                for pair_a, pair_b in ticker_pairs
            },
            axis=1,
        )
        simulation.columns.names = ["Ticker A", "Ticker B", None]

        return apply_rounding(
            simulation, rounding if rounding is not None else self._rounding
        )

    @staticmethod
    def _simulate_copula_pair(
        pair_a: str,
        pair_b: str,
        returns: pd.DataFrame,
        fit_function,
        simulate_function,
        empirical_margins: bool,
    ) -> pd.DataFrame:
        fitted_parameters = fit_function(returns[pair_a], returns[pair_b])
        simulation = simulate_function(fitted_parameters)

        if empirical_margins:
            simulation = pd.DataFrame(
                {
                    pair_a: np.quantile(returns[pair_a].dropna(), simulation["U"]),
                    pair_b: np.quantile(returns[pair_b].dropna(), simulation["V"]),
                }
            )

        return simulation

    @handle_errors
    def get_best_fitting_copula(
        self,
        ticker_a: str | None = None,
        ticker_b: str | None = None,
        period: str | None = None,
        column: str = "Return",
        show_full_results: bool = False,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Calibrate all five supported copula families (see `get_copula_parameters`)
        between `ticker_a` and `ticker_b`, and compare them by AIC (Akaike
        Information Criterion) -- the lower the AIC, the better the fit relative
        to its number of parameters, so the top row is the best-fitting family.

        When `ticker_a`/`ticker_b` are not given, every unique pair among the
        Toolkit's tickers is compared instead.

        Also known as: copula selection, copula comparison.

        Args:
            ticker_a (str, optional): The first asset. Defaults to None, meaning every unique pair of
            tickers in the Toolkit instance is compared (requires `ticker_b` to also be None).
            ticker_b (str, optional): The second asset. Defaults to None, see `ticker_a`.
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily", since a dependence estimate needs far more observations than a
                lower frequency provides -- at "yearly" a decade of history is only ten
                observations.
            column (str, optional): The historical data column to use. Defaults to "Return".
            show_full_results (bool, optional): Only relevant when neither ticker is given. When False
            (the default), returns a square ticker-by-ticker grid of just the winning copula family
            per pair. When True, returns one row per pair instead, with the winning family's fitted
            parameter(s), Lower and Upper Tail Dependence, Log-Likelihood, AIC and the number of
            observations used. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.

        Raises:
            ValueError: If only one of `ticker_a`/`ticker_b` is given, or if fewer than 2 tickers are
            available to pair up when neither is given.

        Returns:
            pd.DataFrame: For a single given pair, one row per copula family, sorted by AIC (best fit
            first), with each family's fitted parameter(s), Lower and Upper Tail Dependence,
            Log-Likelihood, AIC and the number of observations used, indexed by "Copula". When neither
            ticker is given, a square ticker-by-ticker grid of the winning copula family per pair (or,
            with `show_full_results=True`, one row per pair instead, indexed by (Ticker A, Ticker B),
            with a "Best Copula" column and that family's fitted stats).

        Notes:
        - The method retrieves historical data based on the specified `period` and calibrates every
        supported copula family for the given pair, or every pair, of tickers.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AMZN", "TSLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        copulas = toolkit.risk.get_best_fitting_copula("AMZN", "TSLA", period="weekly")

        copulas[["Lower Tail Dependence", "Upper Tail Dependence", "Log-Likelihood", "AIC"]]
        ```

        Which returns:

        | Copula    |   Lower Tail Dependence |   Upper Tail Dependence |   Log-Likelihood |      AIC |
        |:----------|------------------------:|------------------------:|-----------------:|---------:|
        | Frank     |                  0      |                  0      |          32.3868 | -62.7736 |
        | Gaussian  |                  0      |                  0      |          30.2817 | -58.5634 |
        | Student-T |                  0.0198 |                  0.0198 |          30.5851 | -57.1701 |
        | Gumbel    |                  0      |                  0.3659 |          27.8687 | -53.7375 |
        | Clayton   |                  0.3243 |                  0      |          21.68   | -41.3599 |
        """
        period = period if period else "daily"
        returns = self._get_price_column(period, column)

        fit_functions = {
            "Gaussian": copula_model.get_gaussian_copula_parameters,
            "Student-T": copula_model.get_student_t_copula_parameters,
            "Clayton": copula_model.get_clayton_copula_parameters,
            "Gumbel": copula_model.get_gumbel_copula_parameters,
            "Frank": copula_model.get_frank_copula_parameters,
        }

        ticker_pairs = self._resolve_ticker_pairs(ticker_a, ticker_b)

        if len(ticker_pairs) == 1:
            ticker_a, ticker_b = ticker_pairs[0]
            comparison_df = self._compare_copula_pair(
                ticker_a, ticker_b, returns, fit_functions
            )

            return apply_rounding(
                comparison_df, rounding if rounding is not None else self._rounding
            )

        # One row per pair: only the lowest-AIC family, which sorts first, is kept.
        comparison_df = pd.concat(
            {
                (pair_a, pair_b): self._compare_copula_pair(
                    pair_a, pair_b, returns, fit_functions
                ).iloc[[0]]
                for pair_a, pair_b in ticker_pairs
            }
        )
        comparison_df.index.names = ["Ticker A", "Ticker B", "Best Copula"]
        comparison_df = comparison_df.reset_index(level="Best Copula")

        if show_full_results:
            return apply_rounding(
                comparison_df, rounding if rounding is not None else self._rounding
            )

        tickers = sorted({ticker for pair in ticker_pairs for ticker in pair})
        grid = pd.DataFrame(index=tickers, columns=tickers, dtype=object)

        for (pair_a, pair_b), best_copula in comparison_df["Best Copula"].items():
            grid.loc[pair_a, pair_b] = best_copula
            grid.loc[pair_b, pair_a] = best_copula

        grid.index.name = "Ticker A"
        grid.columns.name = "Ticker B"

        return grid

    @staticmethod
    def _compare_copula_pair(
        pair_a: str, pair_b: str, returns: pd.DataFrame, fit_functions: dict
    ) -> pd.DataFrame:
        comparison = {
            copula_name: fit_function(returns[pair_a], returns[pair_b])
            for copula_name, fit_function in fit_functions.items()
        }

        pair_df = pd.DataFrame(comparison).T.sort_values("AIC")
        pair_df.index.name = "Copula"

        return pair_df

    @handle_errors
    def get_covar(
        self,
        ticker: str,
        conditioning_ticker: str,
        period: str | None = None,
        column: str = "Return",
        alpha: float = 0.05,
        rounding: int | None = None,
    ) -> pd.Series:
        """
        Calculate the (Delta-)CoVaR of `ticker` conditional on `conditioning_ticker`
        being in its own distress state.

        Ordinary Value at Risk treats each asset in isolation, which misses systemic
        risk -- the fact that one asset's distress can spill over and worsen
        another's risk. CoVaR directly measures that spillover: it is the VaR of
        `ticker`, conditional on `conditioning_ticker` itself being at its own
        `alpha`-VaR, estimated via a linear Quantile Regression of `ticker`'s returns
        on `conditioning_ticker`'s returns at quantile `alpha`. The Delta-CoVaR
        isolates the marginal, distress-specific contribution by subtracting the same
        construction evaluated in the "normal" (median) state instead.

        CoVaR was introduced to measure systemic risk: in its original application `ticker`
        is a financial-system index (or a bank's peers) and `conditioning_ticker` a single
        institution, so the Delta-CoVaR reads as that institution's contribution to
        system-wide tail risk. Because the quantile regression is fitted at the `alpha`
        quantile, it needs several years of daily data, and the result is a statement about
        co-movement in the tails, not about causality.

        For more information about the method, see the following paper:

        - Adrian, T., & Brunnermeier, M.K. (2016). "CoVaR." American Economic Review,
        106(7), 1705-1741.

        Also known as: Conditional Value at Risk (systemic risk sense), Delta-CoVaR.

        Args:
            ticker (str): The asset whose conditional VaR is being measured.
            conditioning_ticker (str): The asset (or e.g. a benchmark/index) whose distress
            `ticker` is conditioned on.
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily", since a dependence estimate needs far more observations than a
                lower frequency provides -- at "yearly" a decade of history is only ten
                observations.
            column (str, optional): The historical data column to use. Defaults to "Return".
            alpha (float, optional): The tail probability for both the tail quantile regression and
            the VaR of `conditioning_ticker` (e.g., 0.05 for the 95% confidence level). Defaults to 0.05.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.

        Returns:
            pd.Series: The CoVaR, the Delta-CoVaR, the tail (alpha-quantile) Quantile Regression
            slope and intercept, and the number of observations used.

        Notes:
        - The method retrieves historical data based on the specified `period` for the two given
        assets and estimates the CoVaR of `ticker` conditional on `conditioning_ticker`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AMZN", "TSLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_covar("AMZN", "TSLA", period="weekly")
        ```

        Which returns:

        |                               |    Value |
        |:------------------------------|---------:|
        | CoVaR                         |  -0.1041 |
        | Delta-CoVaR                   |  -0.0423 |
        | Quantile Regression Slope     |   0.3256 |
        | Quantile Regression Intercept |  -0.062  |
        | Observations                  | 260      |
        """
        period = period if period else "daily"
        returns = self._get_price_column(period, column)

        result = covar_model.get_covar(
            returns[ticker], returns[conditioning_ticker], alpha=alpha
        )

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

    def _get_portfolio_weights(
        self, returns: pd.DataFrame, weights: dict[str, float] | pd.Series | None
    ) -> pd.Series:
        if weights is None:
            constituent_tickers = [
                ticker
                for ticker in returns.columns
                if ticker not in ("Portfolio", "Benchmark")
            ]
            if not constituent_tickers:
                raise ValueError(
                    "No tickers available to build equal weights from. Please "
                    "provide the `weights` argument explicitly."
                )
            return pd.Series(1 / len(constituent_tickers), index=constituent_tickers)

        weights_series = (
            pd.Series(weights) if isinstance(weights, dict) else weights.copy()
        )
        missing_tickers = [
            ticker for ticker in weights_series.index if ticker not in returns.columns
        ]
        if missing_tickers:
            raise ValueError(
                f"weights contains tickers not present in the Toolkit: {missing_tickers}"
            )

        return weights_series

    @handle_errors
    def get_marginal_value_at_risk(
        self,
        weights: dict[str, float] | None = None,
        period: str | None = None,
        column: str = "Return",
        alpha: float = 0.05,
        distribution: str = "historic",
        rounding: int | None = None,
    ) -> pd.Series:
        """
        Calculate the Marginal Value at Risk (Marginal VaR) of each asset in a
        portfolio.

        Ordinary VaR (see `get_value_at_risk`) treats each asset in isolation.
        Marginal VaR instead measures how much the *portfolio's* VaR would change for
        an infinitesimal increase in a given asset's portfolio weight — i.e. the
        sensitivity of portfolio risk to each holding, not the risk of the holding on
        its own:

        - Portfolio Return = SUM(weight_i * Return_i)
        - Beta_i = Cov(Return_i, Portfolio Return) / Var(Portfolio Return)
        - Marginal VaR_i = Beta_i * Portfolio VaR

        An asset with Beta_i > 1 contributes disproportionately to portfolio risk,
        while Beta_i < 1 (and especially Beta_i < 0) indicates a diversifying holding.

        For more information about the method, see the following sources:

        - Garman, M.B. (1997). "Taking VaR to Pieces." Risk, 10(10), 70-71.
        - Litterman, R. (1996). "Hot Spots and Hedges." Goldman Sachs Risk Management
        Series.
        - Jorion, P. (2006). "Value at Risk: The New Benchmark for Managing Financial
        Risk." 3rd ed., McGraw-Hill, Chapter 7.

        Also known as: Marginal VaR, MVaR.

        Args:
            weights (dict[str, float] | None, optional): Portfolio weights
            keyed by ticker. Normalized internally to sum to 1. Defaults to None, which
            uses equal weights across every ticker in the Toolkit instance (excluding
            the "Portfolio" and "Benchmark" pseudo-tickers, if present).
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily", since a whole-series estimate needs far more observations than a single
                quarter or year holds.
            column (str, optional): The historical data column to use. Defaults to "Return".
            alpha (float, optional): The tail probability (e.g., 0.05 for the 95% confidence level).
            Defaults to 0.05.
            distribution (str, optional): The distribution to use for the underlying portfolio
            VaR calculation (historic, gaussian, cornish-fisher or studentt). Defaults to
            "historic".
            rounding (int | None, optional): The number of decimals to round the results to.
            Defaults to None.

        Returns:
            pd.Series: The Marginal VaR of each asset.

        Notes:
        - The method retrieves historical return data based on the specified `period` for
        every ticker in the Toolkit instance and computes each asset's Marginal VaR against
        the weighted portfolio built from `weights`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AMZN", "TSLA", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_marginal_value_at_risk(weights={"AMZN": 0.5, "TSLA": 0.3, "MSFT": 0.2})
        ```

        Which returns:

        |      |   Marginal VaR |
        |:-----|---------------:|
        | AMZN |        -0.0297 |
        | TSLA |        -0.05   |
        | MSFT |        -0.0182 |
        """
        period = period if period else "daily"
        returns = self._get_price_column(period, column)
        weights_series = self._get_portfolio_weights(returns, weights)

        # The portfolio return needs every constituent on the same dates, so only here are the rows with any missing return dropped.  # noqa: E501
        marginal_var = var_model.get_marginal_var(
            returns[weights_series.index].dropna(),
            weights_series,
            alpha,
            distribution,
        )
        marginal_var.name = "Marginal VaR"

        return apply_rounding(
            marginal_var, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_component_value_at_risk(
        self,
        weights: dict[str, float] | None = None,
        period: str | None = None,
        column: str = "Return",
        alpha: float = 0.05,
        distribution: str = "historic",
        rounding: int | None = None,
    ) -> pd.Series:
        """
        Calculate the Component Value at Risk (Component VaR) of each asset in a
        portfolio.

        Component VaR allocates total portfolio VaR across its constituent assets,
        such that the allocations sum exactly back to the portfolio VaR — an "Euler"
        (or "fully consistent") risk decomposition:

        - Component VaR_i = weight_i * Marginal VaR_i    (see `get_marginal_value_at_risk`)
        - SUM(Component VaR_i) = Portfolio VaR

        Where Marginal VaR measures each asset's risk *sensitivity*, Component VaR
        measures its actual *contribution* in the same units as portfolio VaR, making
        it directly usable to identify which holdings account for the largest share of
        portfolio risk.

        Component VaR is the basis of risk budgeting: a holding with a 20% weight that
        accounts for 45% of the portfolio VaR stands out immediately, and a component with
        the opposite sign of the portfolio VaR marks a holding that hedges the rest. The
        components always add up to the portfolio VaR, but the beta-based allocation is
        exact only for normally (elliptically) distributed returns and an approximation for
        the historical distribution used by default.

        For more information about the method, see the following sources:

        - Garman, M.B. (1997). "Taking VaR to Pieces." Risk, 10(10), 70-71.
        - Litterman, R. (1996). "Hot Spots and Hedges." Goldman Sachs Risk Management
        Series.
        - Jorion, P. (2006). "Value at Risk: The New Benchmark for Managing Financial
        Risk." 3rd ed., McGraw-Hill, Chapter 7.

        Also known as: Component VaR, risk contribution.

        Args:
            weights (dict[str, float] | None, optional): Portfolio weights
            keyed by ticker. Normalized internally to sum to 1. Defaults to None, which
            uses equal weights across every ticker in the Toolkit instance (excluding
            the "Portfolio" and "Benchmark" pseudo-tickers, if present).
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily", since a whole-series estimate needs far more observations than a single
                quarter or year holds.
            column (str, optional): The historical data column to use. Defaults to "Return".
            alpha (float, optional): The tail probability (e.g., 0.05 for the 95% confidence level).
            Defaults to 0.05.
            distribution (str, optional): The distribution to use for the underlying portfolio
            VaR calculation (historic, gaussian, cornish-fisher or studentt). Defaults to
            "historic".
            rounding (int | None, optional): The number of decimals to round the results to.
            Defaults to None.

        Returns:
            pd.Series: The Component VaR of each asset, plus a "Portfolio" entry equal to
            their sum (the total portfolio VaR), for validation.

        Notes:
        - The method retrieves historical return data based on the specified `period` for
        every ticker in the Toolkit instance and computes each asset's Component VaR against
        the weighted portfolio built from `weights`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AMZN", "TSLA", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_component_value_at_risk(weights={"AMZN": 0.5, "TSLA": 0.3, "MSFT": 0.2})
        ```

        Which returns:

        |           |   Component VaR |
        |:----------|----------------:|
        | AMZN      |         -0.0149 |
        | TSLA      |         -0.015  |
        | MSFT      |         -0.0036 |
        | Portfolio |         -0.0335 |
        """
        period = period if period else "daily"
        returns = self._get_price_column(period, column)
        weights_series = self._get_portfolio_weights(returns, weights)

        # The portfolio return needs every constituent on the same dates, so only here are the rows with any missing return dropped.  # noqa: E501
        component_var = var_model.get_component_var(
            returns[weights_series.index].dropna(),
            weights_series,
            alpha,
            distribution,
        )
        component_var.loc["Portfolio"] = component_var.sum()
        component_var.name = "Component VaR"

        return apply_rounding(
            component_var, rounding if rounding is not None else self._rounding
        )

    @handle_portfolio
    @handle_errors
    def get_var_backtest(
        self,
        period: str | None = None,
        distribution: str = "historic",
        alpha: float = 0.05,
        window_size: int = 252,
        test: str = "both",
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Backtest a Value at Risk model against realized returns.

        A Value at Risk estimate is only useful if it is actually well-calibrated
        against reality. This method builds a rolling, out-of-sample VaR path (each
        VaR estimate uses only the `window_size` returns preceding it, never the
        return it is compared against) and tests it with Kupiec's Proportion of
        Failures test (is the overall breach rate consistent with `alpha`?) and/or
        Christoffersen's independence test (are breaches spread out over time, or do
        they cluster together?).

        Regulators backtest VaR on the number of exceptions over the last 250 trading days
        (the Basel traffic-light approach at 99% confidence), and Kupiec's test formalises
        the same question. With few exceptions the test has little power, so a model that
        passes is not necessarily well calibrated. Clustered exceptions, which
        Christoffersen's test detects, are the usual failure of historical VaR when
        volatility rises.

        For more information about the methods, see the following papers:

        - Kupiec, P.H. (1995). "Techniques for Verifying the Accuracy of Risk
        Measurement Models." The Journal of Derivatives, 3(2), 73-84.
        - Christoffersen, P.F. (1998). "Evaluating Interval Forecasts." International
        Economic Review, 39(4), 841-862.

        Also known as: VaR validation, VaR backtest, POF test.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, quarterly, or yearly).
            Defaults to "daily", since `window_size` is expressed in return observations of this frequency
            (252 only means "about one year" when `period` is daily).
            distribution (str, optional): The distribution to use for the rolling VaR estimates, one of
            "historic", "gaussian", "studentt" or "evt". Defaults to "historic".
            alpha (float, optional): The tail probability for the VaR estimates (e.g., 0.05 for the 95%
            confidence level). Defaults to 0.05.
            window_size (int, optional): The rolling window size (in number of return observations) used to
            estimate each VaR value. Defaults to 252 (approximately one trading year of daily returns).
            test (str, optional): Which test(s) to run, one of "kupiec", "christoffersen" or "both". Defaults
            to "both".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: The requested test statistic(s) and their p-value(s) per asset.

        Notes:
        - The rolling VaR path is calculated over the full return history, not the `within_period` slices
        used elsewhere in this module, since a meaningful rolling window generally needs more history than
        a single sub-period provides. Each asset is rolled over its own available returns, so a shorter
        history of one asset does not shorten the others.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AMZN", "TSLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_var_backtest(window_size=252)
        ```

        Which returns:

        |                          |   AMZN |   TSLA |   Benchmark |
        |:-------------------------|-------:|-------:|------------:|
        | Kupiec Statistic         | 1.9421 | 0.1736 |      0.312  |
        | P-Value                  | 0.1634 | 0.677  |      0.5765 |
        | Christoffersen Statistic | 0.5605 | 5.9293 |      4.7906 |
        | P-Value                  | 0.4541 | 0.0149 |      0.0286 |
        """
        period = period if period else "daily"

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )
        if distribution not in ["historic", "gaussian", "studentt", "evt"]:
            raise ValueError(
                "Distribution must be historic, gaussian, studentt, or evt."
            )
        if test not in ["kupiec", "christoffersen", "both"]:
            raise ValueError("Test must be kupiec, christoffersen, or both.")

        returns = self._historical_data[period]["Return"].dropna(how="all")
        rolling_var = _get_out_of_sample_estimates(
            returns, "var", distribution, alpha, window_size
        )

        results = []
        if test in ("kupiec", "both"):
            results.append(
                backtesting_model.get_kupiec_test(returns, rolling_var, alpha)
            )
        if test in ("christoffersen", "both"):
            results.append(
                backtesting_model.get_christoffersen_test(returns, rolling_var)
            )

        result = pd.concat(results, axis=0) if len(results) > 1 else results[0]

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

    @handle_portfolio
    @handle_errors
    def get_acerbi_szekely_test(
        self,
        period: str | None = None,
        distribution: str = "historic",
        alpha: float = 0.05,
        window_size: int = 252,
        n_bootstrap: int = 1000,
        random_state: int = 42,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Backtest a Conditional Value at Risk (Expected Shortfall) model against
        realized returns, via the Acerbi-Szekely (2014) Z2 statistic.

        `get_var_backtest` above only backtests the VaR estimate itself -- it checks
        how often (and how independently) the VaR threshold is breached, but says
        nothing about the *severity* of the losses on those breach days, which is
        exactly the extra information a CVaR (Expected Shortfall) estimate is
        supposed to add over VaR. This method builds a rolling, out-of-sample CVaR
        (and VaR) path and compares the actual loss on each breach day to the CVaR
        that was supposed to describe the average loss on such days.

        For more information about the method, see the following paper:

        - Acerbi, C., & Szekely, B. (2014). "Back-Testing Expected Shortfall." RISK
        Magazine, 27(11), 76-81.

        Also known as: Acerbi-Szekely test, ES backtest, Z2 test.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, quarterly, or yearly).
            Defaults to "daily", since `window_size` is expressed in return observations of this frequency.
            distribution (str, optional): The distribution to use for the rolling VaR/CVaR estimates,
            one of "historic", "gaussian", "studentt" or "evt". Defaults to "historic".
            alpha (float, optional): The tail probability for the VaR/CVaR estimates (e.g., 0.05 for the 95%
            confidence level). Defaults to 0.05.
            window_size (int, optional): The rolling window size (in number of return observations) used
            to estimate each VaR/CVaR value. Defaults to 252 (approximately one trading year of daily
            returns).
            n_bootstrap (int, optional): The number of bootstrap resamples used to estimate the Standard
            Error of Z2. Defaults to 1000.
            random_state (int, optional): The seed for the bootstrap random number generator. Defaults to
            42.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: The Z2 statistic, its bootstrap Standard Error, its one-sided p-value (a low
            value means the model underestimates the tail losses), and the number of breaches observed
            (returns strictly below the VaR), per asset.

        Notes:
        - The rolling VaR/CVaR path is calculated over the full return history, not the `within_period`
        slices used elsewhere in this module, since a meaningful rolling window generally needs more
        history than a single sub-period provides. Each asset is rolled over its own available returns.
        - The paper simulates the null distribution of Z2 from the model's predictive distributions; a
        rolling path does not carry those, so the p-value is approximated with a paired bootstrap of the
        observed (return, VaR, CVaR) triples and a normal approximation (see
        `backtesting_model.get_acerbi_szekely_test`).

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AMZN", "TSLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_acerbi_szekely_test(window_size=100)
        ```

        Which returns:

        |                          |    AMZN |    TSLA |   Benchmark |
        |:-------------------------|--------:|--------:|------------:|
        | Acerbi-Szekely Statistic |  0.2146 |  0.2444 |      0.3931 |
        | Standard Error           |  0.1615 |  0.1543 |      0.1554 |
        | P-Value                  |  0.0919 |  0.0566 |      0.0057 |
        | Breaches                 | 65      | 70      |     78      |
        """
        period = period if period else "daily"

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )
        if distribution not in ["historic", "gaussian", "studentt", "evt"]:
            raise ValueError(
                "Distribution must be historic, gaussian, studentt, or evt."
            )

        returns = self._historical_data[period]["Return"].dropna(how="all")
        rolling_var = _get_out_of_sample_estimates(
            returns, "var", distribution, alpha, window_size
        )
        rolling_cvar = _get_out_of_sample_estimates(
            returns, "cvar", distribution, alpha, window_size
        )

        result = backtesting_model.get_acerbi_szekely_test(
            returns, rolling_var, rolling_cvar, alpha, n_bootstrap, random_state
        )

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

    @handle_portfolio
    @handle_errors
    def get_skewness(
        self,
        period: str | None = None,
        within_period: bool = False,
        rolling: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ):
        """
        Calculate the Skewness of an investment portfolio or asset's returns.

        Skewness is a statistical measure used in finance to assess the asymmetry in the distribution of
        returns for an investment portfolio or asset over a defined period. It offers valuable insights
        into the shape of the return distribution, indicating whether returns are skewed towards the
        positive or negative side of the mean. Skewness is a crucial tool for investors and analysts seeking
        to understand the potential risk and return characteristics of an investment, aiding in the assessment
        of the distribution's tails and potential outliers. It provides a means to gauge the level of
        skew in returns, enabling more informed investment decisions and risk management strategies.

        Also known as: return distribution asymmetry, tail skew.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "daily".
            within_period (bool, optional): Whether to calculate the Skewness within the specified period or for the
            entire period. Thus whether to look at the Skewness within a specific year (if period = 'yearly') or look
            at the entirety of all years. Defaults to False.
            Note that this requires intraday data when period = 'daily', since a
            single day only nests observations when intraday data was fetched.
            rolling (int, optional): The rolling window size to use for the calculation. If set, Skewness is
            calculated over a rolling window of this many periods across the full return history instead of
            per `period`. Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the Skewness values over time.
            Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.Series: Skewness values with time as the index.

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates Skew for each
        asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of Skewness values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["MSFT", "AAPL", "TSLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_skewness(period="yearly", within_period=True)
        ```

        Which returns:

        |      |    MSFT |    AAPL |    TSLA |   Benchmark |
        |:-----|--------:|--------:|--------:|------------:|
        | 2021 | -0.0195 | -0.0748 |  0.667  |     -0.3334 |
        | 2022 |  0.1478 |  0.3168 | -0.026  |      0.0454 |
        | 2023 |  0.3773 | -0.0686 |  0.0021 |     -0.0206 |
        | 2024 | -0.888  |  0.4447 |  0.8937 |     -0.5242 |
        | 2025 |  1.3116 |  1.1306 |  0.4045 |      1.5069 |
        """
        period = period if period else "daily"

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )
        if (
            period == "daily"
            and within_period
            and self._historical_data["intraday"].empty
        ):
            raise ValueError(
                "Intraday data is required for within-period daily calculations. Either set "
                "within_period=False or initialise the Toolkit with an intraday_period."
            )
        if growth and not within_period and not rolling:
            raise ValueError(
                "Growth calculations need one value per period, while "
                "within_period=False produces a single value for the entire period. "
                "Set within_period=True to calculate growth."
            )

        if rolling:
            returns = self._historical_data[period]["Return"]
            skewness = risk_model.get_rolling_skewness(returns, rolling)
        else:
            returns = self._get_returns(period, within_period)

            skewness = risk_model.get_skewness(returns)

        if rolling or within_period:
            skewness = skewness.loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=skewness,
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

    @handle_portfolio
    @handle_errors
    def get_kurtosis(
        self,
        period: str | None = None,
        within_period: bool = False,
        fisher: bool = False,
        rolling: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ):
        """
        Calculate the Kurtosis of an investment portfolio or asset's returns.

        Kurtosis is a statistical measure used in finance to evaluate the shape of the probability
        distribution of returns for an investment portfolio or asset over a defined time period.
        It assesses the "tailedness" of the return distribution, indicating whether returns have
        fatter or thinner tails compared to a normal distribution. Kurtosis plays a critical role
        in risk assessment by revealing the potential presence of extreme outliers or the likelihood
        of heavy tails in the return data. This information aids investors and analysts in
        understanding the degree of risk associated with an investment and assists in making
        more informed decisions regarding risk tolerance. In essence, kurtosis serves as a valuable
        tool for comprehending the distribution characteristics of returns, offering insights
        into the potential for rare but significant events in the financial markets.

        By default this is Pearson's kurtosis, which is 3 for a normal distribution; with `fisher=True`
        it is the excess kurtosis, which is 0. Daily stock returns typically show an excess kurtosis of
        several points, and within a single year one exceptional day (an earnings shock, a market-wide
        gap) can push it above 10, so a high value often reflects one event rather than a persistently
        fat-tailed distribution.

        Also known as: tail heaviness, fat tails, leptokurtosis.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "daily".
            within_period (bool, optional): Whether to calculate the Kurtosis within the specified period or for the
            entire period. Thus whether to look at the Kurtosis within a specific year (if period = 'yearly') or look
            at the entirety of all years. Defaults to False.
            Note that this requires intraday data when period = 'daily', since a
            single day only nests observations when intraday data was fetched.
            fisher (bool, optional): Whether to use Fisher's definition of kurtosis (kurtosis = 0.0
            for a normal distribution). Defaults to False.
            rolling (int, optional): The rolling window size to use for the calculation. If set, Kurtosis is
            calculated over a rolling window of this many periods across the full return history instead of
            per `period`. Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the Kurtosis values over time.
            Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.Series: Kurtosis values with time as the index.

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates Kurtosis for each
        asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of Kurtosis values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["MSFT", "AAPL", "TSLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_kurtosis(period="yearly", within_period=True)
        ```

        Which returns:

        |      |    MSFT |    AAPL |   TSLA |   Benchmark |
        |:-----|--------:|--------:|-------:|------------:|
        | 2021 |  3.377  |  3.4006 | 7.4582 |      3.7144 |
        | 2022 |  3.896  |  4.0553 | 3.386  |      3.3635 |
        | 2023 |  4.4636 |  4.4756 | 4.3336 |      2.843  |
        | 2024 |  5.1739 |  6.6112 | 7.5225 |      4.7961 |
        | 2025 | 13.1151 | 17.0892 | 7.7019 |     26.576  |
        """
        period = period if period else "daily"

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )
        if (
            period == "daily"
            and within_period
            and self._historical_data["intraday"].empty
        ):
            raise ValueError(
                "Intraday data is required for within-period daily calculations. Either set "
                "within_period=False or initialise the Toolkit with an intraday_period."
            )
        if growth and not within_period and not rolling:
            raise ValueError(
                "Growth calculations need one value per period, while "
                "within_period=False produces a single value for the entire period. "
                "Set within_period=True to calculate growth."
            )

        if rolling:
            returns = self._historical_data[period]["Return"]
            kurtosis = risk_model.get_rolling_kurtosis(returns, rolling, fisher=fisher)
        else:
            returns = self._get_returns(period, within_period)

            kurtosis = risk_model.get_kurtosis(returns, fisher=fisher)

        if rolling or within_period:
            kurtosis = kurtosis.loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=kurtosis,
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

    @handle_portfolio
    @handle_errors
    def get_hill_estimator(
        self,
        period: str | None = None,
        within_period: bool = False,
        k: int | float = 0.1,
        tail: str = "left",
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Hill Estimator of the tail index of returns, per asset.

        Unlike the (finite-sample) Skewness and Kurtosis above, the Hill Estimator is
        a semi-parametric estimate of how heavy the tail of the return distribution
        actually is, under the assumption that the tail follows a Pareto-type power
        law. Smaller values of the tail index indicate a heavier tail (more extreme
        outliers are likely) -- as a rule of thumb, a tail index below 4 implies the
        Kurtosis is theoretically infinite, and below 2 implies the Variance itself is
        theoretically infinite.

        For daily stock returns the tail index typically lies between about 2 and 5, while
        a normal distribution has an infinite tail index. The estimate depends on `k`:
        practitioners plot it against a range of `k` (a Hill plot) and read it where it is
        stable, since a small `k` is noisy and a large `k` pulls in the center of the
        distribution.

        For more information about the method, see the following paper:

        - Hill, B.M. (1975). "A Simple General Approach to Inference About the Tail of
        a Distribution." The Annals of Statistics, 3(5), 1163-1174.

        Also known as: Hill tail index estimator, Hill's estimator.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "daily".
            within_period (bool, optional): Whether to calculate the estimator within the specified period
            or for the entire period. Defaults to False.
            Note that this requires intraday data when period = 'daily', since a
            single day only nests observations when intraday data was fetched.
            k (int | float, optional): The number of upper order statistics to use. If a float in (0, 1)
            it is interpreted as the fraction of the strictly positive observations to use. Defaults to
            0.1 (the top 10%).
            tail (str, optional): Which tail to estimate, one of "left" (the loss tail) or "right" (the
            gain tail). Defaults to "left".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.

        Returns:
            pd.DataFrame: The Hill tail index, the Hill shape parameter, its Standard Error and the
            number of order statistics used, per asset.

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates the
        Hill Estimator for each asset in the Toolkit instance.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AMZN", "TSLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_hill_estimator(period="weekly", within_period=False)
        ```

        Which returns:

        |                       |    AMZN |    TSLA |   Benchmark |
        |:----------------------|--------:|--------:|------------:|
        | Hill Tail Index       |  3.2425 |  9.5123 |      2.7487 |
        | Hill Shape (xi)       |  0.3084 |  0.1051 |      0.3638 |
        | Standard Error        |  0.936  |  2.746  |      0.8288 |
        | Observations Used (k) | 12      | 12      |     11      |
        """
        period = period if period else "daily"

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )
        if (
            period == "daily"
            and within_period
            and self._historical_data["intraday"].empty
        ):
            raise ValueError(
                "Intraday data is required for within-period daily calculations. Either set "
                "within_period=False or initialise the Toolkit with an intraday_period."
            )

        returns = self._get_returns(period, within_period)

        result = risk_model.get_hill_estimator(returns, k=k, tail=tail)

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

    @handle_portfolio
    @handle_errors
    def get_variance(
        self,
        period: str | None = None,
        rolling: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ):
        """
        Calculate the Variance of an investment portfolio or asset's returns for a
        given period based on the daily historical returns.

        Variance measures the spread or dispersion of returns around the mean. A higher
        Variance indicates more variability in the returns, while a lower Variance suggests
        that the returns are closer to the mean.

        The daily Variance is scaled to the given period by multiplying it with the number
        of trading days within that period (e.g. 252 / 52 for weekly).

        Also known as: dispersion, spread.

        Args:
            period (str, optional): The data frequency for returns (weekly, monthly, quarterly, or yearly). Defaults
                to "quarterly" if the Toolkit is initialised with quarterly=True, otherwise "yearly".
            rolling (int, optional): The rolling window size to use for the calculation. If set,
            Variance is calculated over a rolling window of this many periods (e.g. period='monthly'
            and rolling=6 gives the rolling 6-month Variance) instead of one value per `period`.
            Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the Variance values over time. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.Series: Variance values with time as the index.

        Notes:
        - The method retrieves the daily historical return data and calculates the Variance for
        the specified `period` for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of Variance values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AMZN", "TSLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_variance(period="yearly")
        ```

        Which returns:

        | Date   |   AMZN |   TSLA |   Benchmark |
        |:-------|-------:|-------:|------------:|
        | 2021   | 0.0578 | 0.3    |      0.0167 |
        | 2022   | 0.2508 | 0.4446 |      0.0588 |
        | 2023   | 0.109  | 0.2922 |      0.0171 |
        | 2024   | 0.0789 | 0.4032 |      0.0158 |
        | 2025   | 0.1184 | 0.4031 |      0.038  |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        if period not in ["weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError("Period must be weekly, monthly, quarterly, or yearly.")

        if rolling:
            period_returns = self._historical_data[period]["Return"]
            variance = risk_model.get_rolling_variance(period_returns, period, rolling)
        else:
            returns = self._historical_data["daily"]["Return"]
            variance = risk_model.get_variance(returns, period)

        variance = variance.loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=variance,
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

    @handle_portfolio
    @handle_errors
    def get_volatility(
        self,
        period: str | None = None,
        rolling: int | None = None,
        method: str = "close_to_close",
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ):
        """
        Calculate the Volatility of an investment portfolio or asset's returns for a
        given period based on the daily historical prices.

        Volatility measures the amount of dispersion or variability in returns. It is the
        square root of the Variance. A higher Volatility indicates greater variability, while
        a lower Volatility suggests that returns are closer to the mean.

        By default this is the close-to-close Volatility, i.e. the standard deviation of the
        daily returns. The `method` parameter selects one of four range-based estimators
        instead, each of which uses more of the day's price action than just the close and is
        therefore more statistically efficient (i.e. needs fewer observations to reach the
        same precision), at the cost of additional assumptions about how prices move:

        - `"parkinson"` — uses the daily trading range (High vs Low) rather than the
        close-to-close return, assuming prices follow a continuous geometric Brownian motion
        with no drift and no overnight jumps.
        - `"garman_klass"` — extends Parkinson by also incorporating the Open and Close, which
        allows it to account for the opening jump and makes it more efficient still (assuming,
        as Parkinson does, no drift and no overnight jumps beyond the modeled open).
        - `"rogers_satchell"` — drift-independent, meaning it remains unbiased even when the
        underlying asset has a non-zero expected return over the period, at the cost of still
        assuming no overnight jumps.
        - `"yang_zhang"` — a weighted combination of the overnight (close-to-open) Variance,
        the open-to-close Variance and the Rogers-Satchell Variance. It is both
        drift-independent and accounts for overnight jumps, which makes it the most
        statistically efficient of the range-based estimators implemented here.

        In every case the daily Volatility is scaled to the given period by multiplying the
        underlying Variance with the number of trading days within that period (e.g. 252 / 52
        for weekly).

        Also known as: standard deviation of returns. The range-based estimators are also known
        as Parkinson's range-based or high-low Volatility, Garman-Klass range-based Volatility,
        Rogers-Satchell drift-independent Volatility and Yang-Zhang drift-independent
        overnight-aware Volatility.

        For more information about the range-based estimators, see the following papers:

        - Parkinson, M. (1980). "The Extreme Value Method for Estimating the Variance of the
        Rate of Return." Journal of Business, 53(1), 61-65.
        - Garman, M.B., & Klass, M.J. (1980). "On the Estimation of Security Price
        Volatilities from Historical Data." Journal of Business, 53(1), 67-78.
        - Rogers, L.C.G., & Satchell, S.E. (1991). "Estimating Variance from High, Low and
        Close Prices." Annals of Applied Probability, 1(4), 504-512.
        - Yang, D., & Zhang, Q. (2000). "Drift-Independent Volatility Estimation Based on
        High, Low, Open, and Close Prices." Journal of Business, 73(3), 477-491.

        Args:
            period (str, optional): The data frequency for returns (weekly, monthly, quarterly, or yearly). Defaults
                to "quarterly" if the Toolkit is initialised with quarterly=True, otherwise "yearly".
            rolling (int, optional): The rolling window size to use for the calculation. If set,
            Volatility is calculated over a rolling window of this many periods (e.g. period='monthly'
            and rolling=6 gives the rolling 6-month Volatility) instead of one value per `period`.
            Only available for method="close_to_close". Defaults to None.
            method (str, optional): Which Volatility estimator to use, one of "close_to_close",
            "parkinson", "garman_klass", "rogers_satchell" or "yang_zhang", as described above.
            Defaults to "close_to_close".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the Volatility values over time.
            Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.Series: Volatility values with time as the index.

        Notes:
        - The method retrieves the daily historical price data and calculates the Volatility for
        the specified `period` for each asset in the Toolkit instance. The close-to-close estimator
        only needs the daily returns, Parkinson needs the High and Low, and the remaining
        estimators need the Open, High, Low and Close.
        - If `growth` is set to True, the method calculates the growth of Volatility values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AMZN", "TSLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_volatility(period="yearly")
        ```

        Which returns:

        | Date   |   AMZN |   TSLA |   Benchmark |
        |:-------|-------:|-------:|------------:|
        | 2021   | 0.2404 | 0.5477 |      0.1293 |
        | 2022   | 0.5008 | 0.6668 |      0.2425 |
        | 2023   | 0.3302 | 0.5406 |      0.1307 |
        | 2024   | 0.2809 | 0.635  |      0.1257 |
        | 2025   | 0.3442 | 0.6349 |      0.1948 |

        And, using the daily trading range instead of only the closes:

        ```python
        toolkit.risk.get_volatility(period="yearly", method="parkinson")
        ```

        Which returns:

        | Date   |   AMZN |   TSLA |   Benchmark |
        |:-------|-------:|-------:|------------:|
        | 2021   | 0.2089 | 0.4563 |      0.1073 |
        | 2022   | 0.3717 | 0.5547 |      0.1916 |
        | 2023   | 0.2617 | 0.438  |      0.1101 |
        | 2024   | 0.219  | 0.4357 |      0.0989 |
        | 2025   | 0.267  | 0.5062 |      0.152  |
        """
        if method not in (
            "close_to_close",
            "parkinson",
            "garman_klass",
            "rogers_satchell",
            "yang_zhang",
        ):
            raise ValueError(
                "Method must be close_to_close, parkinson, garman_klass, "
                "rogers_satchell, or yang_zhang."
            )

        period = period if period else "quarterly" if self._quarterly else "yearly"

        if period not in ["weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError("Period must be weekly, monthly, quarterly, or yearly.")

        if rolling and method != "close_to_close":
            raise ValueError(
                "The rolling parameter is only available for method='close_to_close'."
            )

        if method == "close_to_close":
            if rolling:
                period_returns = self._historical_data[period]["Return"]
                volatility = risk_model.get_rolling_volatility(
                    period_returns, period, rolling
                )
            else:
                returns = self._historical_data["daily"]["Return"]
                volatility = risk_model.get_volatility(returns, period)
        elif method == "parkinson":
            volatility = realized_volatility_model.get_parkinson_volatility(
                self._historical_data["daily"]["High"],
                self._historical_data["daily"]["Low"],
                period,
            )
        else:
            estimators = {
                "garman_klass": realized_volatility_model.get_garman_klass_volatility,
                "rogers_satchell": realized_volatility_model.get_rogers_satchell_volatility,
                "yang_zhang": realized_volatility_model.get_yang_zhang_volatility,
            }

            volatility = estimators[method](
                self._historical_data["daily"]["Open"],
                self._historical_data["daily"]["High"],
                self._historical_data["daily"]["Low"],
                self._historical_data["daily"]["Close"],
                period,
            )

        volatility = volatility.loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=volatility,
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

    @handle_portfolio
    @handle_errors
    def get_har_rv_forecast(
        self,
        estimator: str = "squared_return",
        weekly_window: int = 5,
        monthly_window: int = 22,
        horizon: int = 1,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ):
        """
        Calculate the Corsi (2009) Heterogeneous Autoregressive Realized Volatility
        (HAR-RV) forecast of future daily Realized Variance, per asset.

        Volatility clustering happens across multiple, overlapping time horizons at
        once. HAR-RV captures this cheaply -- without the numerical optimization a
        GARCH-family fit requires (see `get_garch`) -- by regressing future daily
        Realized Variance on trailing daily, weekly and monthly average Realized
        Variance components. The daily Realized Variance itself can be constructed in
        several ways via `estimator`: the simplest is the squared daily return, while
        the OHLC range-based estimators (see `get_volatility`, which exposes the same
        estimators via its own `method` parameter) use the daily
        (pre-period-aggregation) term behind each of those estimators instead,
        which is more statistically efficient since it uses the daily trading range
        rather than only the close-to-close move.

        For more information about the method, see the following paper:

        - Corsi, F. (2009). "A Simple Approximate Long-Memory Model of Realized
        Volatility." Journal of Financial Econometrics, 7(2), 174-196.

        Also known as: HAR-RV model, Corsi's HAR model, Heterogeneous Autoregressive model.

        Args:
            estimator (str, optional): How to construct the daily Realized Variance input, one of
            "squared_return", "parkinson", "garman_klass" or "rogers_satchell". Defaults to
            "squared_return".
            weekly_window (int, optional): The trailing window (in trading days) for the weekly RV
            component. Defaults to 5.
            monthly_window (int, optional): The trailing window (in trading days) for the monthly RV
            component. Defaults to 22.
            horizon (int, optional): The number of days ahead to forecast. Defaults to 1.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the HAR-RV forecast values over
            time. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.Series: The HAR-RV forecast of Realized Variance with time as the index.

        Notes:
        - The method retrieves daily historical price data and calculates the HAR-RV forecast for each
        asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the HAR-RV forecast values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AMZN", "TSLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_har_rv_forecast(estimator="squared_return").tail()
        ```

        Which returns:

        | Date       |   AMZN |   TSLA |   Benchmark |
        |:-----------|-------:|-------:|------------:|
        | 2025-12-24 | 0.0003 | 0.0011 |      0.0001 |
        | 2025-12-26 | 0.0003 | 0.0011 |      0.0001 |
        | 2025-12-29 | 0.0003 | 0.0011 |      0.0001 |
        | 2025-12-30 | 0.0003 | 0.0011 |      0.0001 |
        | 2025-12-31 | 0.0003 | 0.0011 |      0.0001 |

        The last row is the forecast for the first trading day after 2025-12-31, which has not
        been observed yet.
        """
        if estimator not in (
            "squared_return",
            "parkinson",
            "garman_klass",
            "rogers_satchell",
        ):
            raise ValueError(
                "estimator must be 'squared_return', 'parkinson', 'garman_klass', "
                "or 'rogers_satchell'."
            )

        if estimator == "squared_return":
            # The forecast drops missing values per ticker, so only rows without any return go.
            realized_variance = (
                self._historical_data["daily"]["Return"].dropna(how="all") ** 2
            )
        else:
            open_prices = self._historical_data["daily"]["Open"]
            high_prices = self._historical_data["daily"]["High"]
            low_prices = self._historical_data["daily"]["Low"]
            close_prices = self._historical_data["daily"]["Close"]

            if estimator == "parkinson":
                realized_variance = np.log(high_prices / low_prices) ** 2 / (
                    4 * np.log(2)
                )
            elif estimator == "garman_klass":
                realized_variance = (
                    0.5 * np.log(high_prices / low_prices) ** 2
                    - (2 * np.log(2) - 1) * np.log(close_prices / open_prices) ** 2
                )
            else:
                realized_variance = np.log(high_prices / close_prices) * np.log(
                    high_prices / open_prices
                ) + np.log(low_prices / close_prices) * np.log(low_prices / open_prices)

        forecast = realized_volatility_model.get_har_rv_forecast(
            realized_variance, weekly_window, monthly_window, horizon
        )

        forecast = forecast.loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=forecast,
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

    @handle_portfolio
    @handle_errors
    def get_amihud_illiquidity(
        self,
        period: str | None = None,
        within_period: bool = False,
        scale: float = 1_000_000,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ):
        """
        Calculate the Amihud (2002) Illiquidity ratio, per asset.

        The Amihud ratio measures the average price impact of trading -- how much the
        price moves per dollar of trading volume. A high value means that even a small
        amount of trading moves the price a lot (the asset is illiquid), while a low
        value means the asset can absorb a large amount of trading with little price
        impact (the asset is liquid).

        The Amihud ratio is mainly a cross-sectional measure: it ranks stocks by how much a
        given dollar volume moves the price and is used as a liquidity factor in asset
        pricing. Its level depends on the scale, the frequency and the price level of the
        market, so compare stocks within the same market and period, and expect the ratio
        of large caps to be several orders of magnitude below that of small caps.

        For more information about the method, see the following paper:

        - Amihud, Y. (2002). "Illiquidity and Stock Returns: Cross-Section and
        Time-Series Effects." Journal of Financial Markets, 5(1), 31-56.

        Also known as: Amihud illiquidity ratio, ILLIQ, price impact ratio.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "daily".
            within_period (bool, optional): Whether to calculate the ratio within the specified period or
            for the entire period. Defaults to False. Note that this requires intraday data when period = 'daily', since a
            single day only nests observations when intraday data was fetched.
            scale (float, optional): A multiplier applied to the resulting ratio purely for readability.
            Defaults to 1,000,000.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the Amihud Illiquidity values over
            time. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.Series: Amihud Illiquidity values with time as the index.

        Notes:
        - The method retrieves historical return, Close price and Volume data based on the specified
        `period` and calculates the Amihud Illiquidity ratio for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the Amihud Illiquidity values
        using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AMZN", "TSLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_amihud_illiquidity(period="quarterly", scale=1e12)
        ```

        Which returns:

        |           |   Value |
        |:----------|--------:|
        | AMZN      |  0.2447 |
        | TSLA      |  0.1704 |
        | Benchmark |  0.0306 |

        Note that a large `scale` is used here since these are liquid, large-cap
        stocks with very high dollar trading volume relative to their typical daily
        price move -- the default `scale` of 1,000,000 (as used in Amihud's original
        1980s/1990s-era paper) would round these to 0.0 at the default precision.
        """
        period = period if period else "daily"

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )
        if (
            period == "daily"
            and within_period
            and self._historical_data["intraday"].empty
        ):
            raise ValueError(
                "Intraday data is required for within-period daily calculations. Either set "
                "within_period=False or initialise the Toolkit with an intraday_period."
            )
        if growth and not within_period:
            raise ValueError(
                "Growth calculations need one value per period, while "
                "within_period=False produces a single value for the entire period. "
                "Set within_period=True to calculate growth."
            )

        source_data = (
            self._within_historical_data[period]
            if within_period
            else self._historical_data[period]
        )

        # Missing returns are skipped per ticker, so a ticker with a shorter history does not cut the others.
        returns = source_data["Return"].dropna(how="all")
        dollar_volume = source_data["Close"] * source_data["Volume"]

        illiquidity = market_liquidity_model.get_amihud_illiquidity(
            returns, dollar_volume, scale
        )

        if within_period:
            illiquidity = illiquidity.loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=illiquidity,
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

    @handle_portfolio
    @handle_errors
    def get_roll_spread(
        self,
        period: str | None = None,
        within_period: bool = False,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Roll (1984) implied bid-ask spread, per asset.

        Roll's model shows that, under a stylized microstructure model in which the
        true (efficient) price follows a random walk and observed trade prices
        randomly bounce between the bid and the ask, the effective spread can be
        backed out purely from the serial covariance of consecutive price changes,
        without needing any actual quote data. If the estimated covariance is zero or
        positive, no valid estimate can be backed out and NaN is returned instead.

        The Roll measure is used when no quote data is available, typically for less liquid
        stocks or historical studies. On daily closes of liquid large caps it mostly picks
        up price noise rather than the bid-ask bounce, and in trending periods the
        autocovariance is positive and no estimate exists, so treat it as a rough, relative
        measure.

        For more information about the method, see the following paper:

        - Roll, R. (1984). "A Simple Implicit Measure of the Effective Bid-Ask Spread
        in an Efficient Market." The Journal of Finance, 39(4), 1127-1139.

        Also known as: Roll's implied spread, Roll measure.

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily".
            within_period (bool, optional): Whether to calculate the spread within the specified period or
            for the entire period. Defaults to False. Note that this requires intraday data when period = 'daily', since a
            single day only nests observations when intraday data was fetched.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.

        Returns:
            pd.DataFrame: The Roll Spread (in price units), the Relative Roll Spread (the Roll Spread divided
            by the mean price, as a decimal), the underlying lag-1 autocovariance, and whether that
            autocovariance was negative (i.e. whether a valid estimate could be backed out), per asset.

        Notes:
        - The method retrieves historical Close price data based on the specified `period` and calculates
        the Roll Spread for each asset in the Toolkit instance.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AMZN", "TSLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_roll_spread()
        ```

        Which returns:

        |                |      AMZN |     TSLA |   Benchmark |
        |:---------------|----------:|---------:|------------:|
        | Roll Spread    |  0.917291 |  3.64757 |    1.97906  |
        | Autocovariance | -0.210356 | -3.32619 |   -0.979167 |
        | Valid Estimate |  1        |  1       |    1        |
        """
        period = period if period else "daily"

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )
        if (
            period == "daily"
            and within_period
            and self._historical_data["intraday"].empty
        ):
            raise ValueError(
                "Intraday data is required for within-period daily calculations. Either set "
                "within_period=False or initialise the Toolkit with an intraday_period."
            )

        close_prices = (
            self._within_historical_data[period]["Close"]
            if within_period
            else self._historical_data[period]["Close"]
        )

        result = market_liquidity_model.get_roll_spread(close_prices)

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

    @handle_portfolio
    @handle_errors
    def get_excess_volatility(
        self,
        period: str | None = None,
        rolling: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ):
        """
        Calculate the Excess Volatility of an investment portfolio or asset's returns for a
        given period based on the daily historical returns.

        Excess Volatility is the Volatility of the Excess Return, i.e. the daily return minus
        the risk free rate, scaled to the given period in the same way as the Volatility.

        Also known as: standard deviation of excess returns.

        Args:
            period (str, optional): The data frequency for returns (weekly, monthly, quarterly, or yearly). Defaults
                to "quarterly" if the Toolkit is initialised with quarterly=True, otherwise "yearly".
            rolling (int, optional): The rolling window size to use for the calculation. If set,
            Excess Volatility is calculated over a rolling window of this many periods (e.g.
            period='monthly' and rolling=6 gives the rolling 6-month Excess Volatility) instead of
            one value per `period`. Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the Excess Volatility values
            over time. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.Series: Excess Volatility values with time as the index.

        Notes:
        - The method retrieves the daily historical return data and calculates the Excess Volatility for
        the specified `period` for each asset in the Toolkit instance.
        - The risk-free rate is often represented by the return of a risk-free investment, such as a Treasury bond.
        - If `growth` is set to True, the method calculates the growth of Excess Volatility values using
        the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AMZN", "TSLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_excess_volatility(period="yearly")
        ```

        Which returns:

        | Date   |   AMZN |   TSLA |   Benchmark |
        |:-------|-------:|-------:|------------:|
        | 2021   | 0.2404 | 0.5477 |      0.1293 |
        | 2022   | 0.5008 | 0.6668 |      0.2425 |
        | 2023   | 0.3302 | 0.5406 |      0.1307 |
        | 2024   | 0.2809 | 0.635  |      0.1257 |
        | 2025   | 0.3442 | 0.6349 |      0.1948 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        if period not in ["weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError("Period must be weekly, monthly, quarterly, or yearly.")

        if rolling:
            period_returns = self._historical_data[period]["Return"]
            risk_free_rate = self._risk_free_rate_data[period]
            excess_volatility = risk_model.get_rolling_excess_volatility(
                period_returns, risk_free_rate, period, rolling
            )
        else:
            returns = self._historical_data["daily"]["Return"]
            risk_free_rate = self._risk_free_rate_data["daily"]

            excess_volatility = risk_model.get_excess_volatility(
                returns, risk_free_rate, period
            )

        excess_volatility = excess_volatility.loc[self._start_date : self._end_date]

        return finalize_dataset(
            dataset=excess_volatility,
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

    @handle_portfolio
    @handle_errors
    def get_downside_deviation(
        self,
        period: str | None = None,
        minimum_acceptable_return: float = 0.0,
        within_period: bool = False,
        rolling: int | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ):
        """
        Calculate the Downside Deviation of an investment portfolio or asset's returns.

        The Downside Deviation, also known as semi-deviation, is the standard deviation of only the
        returns that fall below a minimum acceptable return (MAR), isolating the volatility of negative
        outcomes from the volatility of the overall return distribution. It underlies risk-adjusted
        return measures such as the Sortino Ratio and the Omega Ratio.

        Also known as: semi-deviation, downside risk, downside volatility.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "daily".
            minimum_acceptable_return (float, optional): The minimum acceptable return (MAR) used as the
            threshold below which returns are considered downside. Defaults to 0.0.
            within_period (bool, optional): Whether to calculate the Downside Deviation within the specified
            period or for the entire period. Thus whether to look at the Downside Deviation within a specific
            year (if period = 'yearly') or look at the entirety of all years. Defaults to False.
            Note that this requires intraday data when period = 'daily', since a
            single day only nests observations when intraday data was fetched.
            rolling (int, optional): The rolling window size to use for the calculation. If set, the Downside
            Deviation is calculated over a rolling window of this many periods across the full return history
            instead of per `period`. Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the Downside Deviation values over
            time. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.Series: Downside Deviation values with time as the index.

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates the
        Downside Deviation for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the Downside Deviation values using
        the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AMZN", "TSLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_downside_deviation(period="yearly", within_period=True)
        ```

        Which returns:

        |      |   AMZN |   TSLA |   Benchmark |
        |:-----|-------:|-------:|------------:|
        | 2021 | 0.0108 | 0.0221 |      0.0054 |
        | 2022 | 0.0229 | 0.0318 |      0.011  |
        | 2023 | 0.013  | 0.022  |      0.0053 |
        | 2024 | 0.0117 | 0.0243 |      0.0055 |
        | 2025 | 0.0146 | 0.0268 |      0.0081 |
        """
        period = period if period else "daily"

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )
        if (
            period == "daily"
            and within_period
            and self._historical_data["intraday"].empty
        ):
            raise ValueError(
                "Intraday data is required for within-period daily calculations. Either set "
                "within_period=False or initialise the Toolkit with an intraday_period."
            )
        if growth and not within_period and not rolling:
            raise ValueError(
                "Growth calculations need one value per period, while "
                "within_period=False produces a single value for the entire period. "
                "Set within_period=True to calculate growth."
            )

        if rolling:
            returns = self._historical_data[period]["Return"]
            downside_deviation = risk_model.get_rolling_downside_deviation(
                returns, rolling, minimum_acceptable_return
            )
        else:
            returns = self._get_returns(period, within_period)

            downside_deviation = risk_model.get_downside_deviation(
                returns, minimum_acceptable_return
            )

        if rolling or within_period:
            downside_deviation = downside_deviation.loc[
                self._start_date : self._end_date
            ]

        return finalize_dataset(
            dataset=downside_deviation,
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

    @handle_portfolio
    @handle_errors
    def get_mean_absolute_deviation(
        self,
        period: str | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ):
        """
        Calculate the Mean Absolute Deviation (MAD) of an investment portfolio or asset's
        returns for a given period based on the daily historical returns.

        MAD measures the average absolute distance of each return from the mean return. Unlike
        Variance and Volatility, it does not square the deviations, making it less sensitive to
        outliers.

        Args:
            period (str, optional): The data frequency for returns (weekly, monthly, quarterly, or yearly). Defaults
                to "quarterly" if the Toolkit is initialised with quarterly=True, otherwise "yearly".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the MAD values over time. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.Series: Mean Absolute Deviation values with time as the index.

        Notes:
        - The method retrieves the daily historical return data and calculates the MAD for
        the specified `period` for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of MAD values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AMZN", "TSLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_mean_absolute_deviation(period="yearly")
        ```

        Which returns:

        | Date   |   AMZN |   TSLA |   Benchmark |
        |:-------|-------:|-------:|------------:|
        | 2021   | 0.0114 | 0.0246 |      0.0061 |
        | 2022   | 0.0235 | 0.032  |      0.0119 |
        | 2023   | 0.0156 | 0.0255 |      0.0064 |
        | 2024   | 0.0132 | 0.0286 |      0.0058 |
        | 2025   | 0.015  | 0.0292 |      0.0074 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        if period not in ["weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError("Period must be weekly, monthly, quarterly, or yearly.")

        returns = self._historical_data["daily"]["Return"]
        mean_absolute_deviation = risk_model.get_mean_absolute_deviation(
            returns, period
        )

        mean_absolute_deviation = mean_absolute_deviation.loc[
            self._start_date : self._end_date
        ]
        mean_absolute_deviation = mean_absolute_deviation.dropna(how="all", axis=0)

        return finalize_dataset(
            dataset=mean_absolute_deviation,
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

    @handle_portfolio
    @handle_errors
    def get_coefficient_of_variation(
        self,
        period: str | None = None,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ):
        """
        Calculate the Coefficient of Variation (CV) of an investment portfolio or asset's
        returns for a given period based on the daily historical returns.

        The Coefficient of Variation is the ratio of the standard deviation to the mean of
        returns, which normalizes dispersion relative to the average return. This makes it
        useful for comparing the relative volatility of assets with different average returns,
        which a raw standard deviation cannot do.

        The Coefficient of Variation is only meaningful when the mean return is clearly positive. In a
        period with an average daily return close to zero it explodes, and in a down period it turns
        negative, so for returns it is best read over periods with a positive mean, or replaced by a
        Sharpe ratio, which is its inverse with the risk-free rate subtracted.

        Also known as: relative standard deviation.

        Args:
            period (str, optional): The data frequency for returns (weekly, monthly, quarterly, or yearly). Defaults
                to "quarterly" if the Toolkit is initialised with quarterly=True, otherwise "yearly".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the CV values over time. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.Series: Coefficient of Variation values with time as the index.

        Notes:
        - The method retrieves the daily historical return data and calculates the CV for
        the specified `period` for each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of CV values using the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AMZN", "TSLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_coefficient_of_variation(period="yearly")
        ```

        Which returns:

        | Date   |     AMZN |     TSLA |   Benchmark |
        |:-------|---------:|---------:|------------:|
        | 2021   |  51.2239 |  16.7257 |      7.4324 |
        | 2022   | -14.1455 | -12.7544 |    -22.3928 |
        | 2023   |   8.0356 |  10.0506 |      8.5443 |
        | 2024   |  10.9398 |  14.7623 |      8.6564 |
        | 2025   |  49.9543 |  32.8037 |     16.8499 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        if period not in ["weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError("Period must be weekly, monthly, quarterly, or yearly.")

        returns = self._historical_data["daily"]["Return"]
        coefficient_of_variation = risk_model.get_coefficient_of_variation(
            returns, period
        )

        coefficient_of_variation = coefficient_of_variation.loc[
            self._start_date : self._end_date
        ]
        coefficient_of_variation = coefficient_of_variation.dropna(how="all", axis=0)

        return finalize_dataset(
            dataset=coefficient_of_variation,
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

    @handle_portfolio
    @handle_errors
    def get_ewma_volatility(
        self,
        lambda_: float = 0.94,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        standardize: bool = False,
    ):
        """
        Calculate the exponentially weighted moving average (EWMA) Volatility of an
        investment portfolio or asset's daily returns, following the RiskMetrics
        methodology.

        Unlike a fixed-window rolling Volatility, EWMA Volatility weights recent
        observations more heavily than older ones, so it reacts faster to changes in the
        underlying volatility regime. It is a simpler, more interpretable alternative to a
        full GARCH fit.

        The formula is as follows:

        - EWMA Variance(t) = lambda * EWMA Variance(t-1) + (1 - lambda) * Return(t-1) ** 2

        With the RiskMetrics decay factor of 0.94, about 70% of the weight sits on the last 20 trading
        days, so EWMA Volatility reacts within days to a market shock and falls back almost as quickly.
        The values are daily; multiply them by the square root of 252 for an annualized figure comparable
        with `get_volatility(period="yearly")`.

        Also known as: RiskMetrics volatility, exponentially weighted volatility.

        Args:
            lambda_ (float, optional): The decay factor. Higher values weight the past
            more heavily (slower to react), lower values weight recent returns more
            heavily (faster to react). RiskMetrics uses 0.94 for daily data. Defaults to 0.94.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the EWMA Volatility values over
            time. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.Series: Daily EWMA Volatility values with time as the index.

        Notes:
        - The method retrieves the daily historical return data and calculates the EWMA Volatility for
        each asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of the EWMA Volatility values using
        the specified `lag`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AMZN", "TSLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_ewma_volatility()
        ```

        Which returns:

        | Date       |   AMZN |   TSLA |   Benchmark |
        |:-----------|-------:|-------:|------------:|
        | 2025-12-17 | 0.018  | 0.0276 |      0.007  |
        | 2025-12-18 | 0.0175 | 0.029  |      0.0073 |
        | 2025-12-19 | 0.018  | 0.0294 |      0.0073 |
        | 2025-12-22 | 0.0175 | 0.0285 |      0.0075 |
        | 2025-12-23 | 0.017  | 0.0279 |      0.0074 |
        | 2025-12-24 | 0.017  | 0.0271 |      0.0073 |
        | 2025-12-26 | 0.0164 | 0.0263 |      0.0071 |
        | 2025-12-29 | 0.0159 | 0.026  |      0.0069 |
        | 2025-12-30 | 0.0155 | 0.0264 |      0.0067 |
        | 2025-12-31 | 0.015  | 0.0258 |      0.0065 |
        """
        returns = self._historical_data["daily"]["Return"]
        ewma_volatility = risk_model.get_ewma_volatility(returns, lambda_)

        ewma_volatility = ewma_volatility.loc[self._start_date : self._end_date]
        ewma_volatility = ewma_volatility.dropna(how="all", axis=0)

        return finalize_dataset(
            dataset=ewma_volatility,
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

    @handle_errors
    def get_autocorrelation(
        self,
        lags: int = 10,
        rounding: int | None = None,
    ):
        """
        Calculate the Autocorrelation Function (ACF) of each asset's daily returns for a
        range of lags.

        The ACF measures the correlation between a return series and a lagged version of
        itself. A significant ACF at a given lag indicates that returns are not fully
        independent over time, which is relevant for assessing return predictability and
        volatility clustering (as opposed to a trading-signal use case, which is why this
        lives in the Risk module rather than Technicals).

        Args:
            lags (int, optional): The number of lags to calculate the ACF for. Defaults to 10.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.

        Returns:
            pd.DataFrame: The ACF value for each lag (rows) and each asset (columns).

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AMZN", "TSLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_autocorrelation()
        ```

        Which returns:

        |   Lag |    AMZN |    TSLA |   Benchmark |
        |------:|--------:|--------:|------------:|
        |     1 | -0.0125 | -0.026  |     -0.031  |
        |     2 | -0.0064 |  0.0128 |     -0.0069 |
        |     3 | -0.0113 | -0.0109 |     -0.0626 |
        |     4 |  0.0112 |  0.0189 |     -0.0344 |
        |     5 | -0.0144 | -0.0213 |     -0.0075 |
        |     6 |  0.0048 |  0.045  |     -0.0104 |
        |     7 | -0.0523 |  0.0309 |     -0.0109 |
        |     8 | -0.0309 |  0.0198 |     -0.0063 |
        |     9 | -0.0103 |  0.063  |      0.0399 |
        |    10 | -0.0083 | -0.0499 |     -0.0111 |
        """
        returns = self._historical_data["daily"]["Return"]

        autocorrelation = returns.apply(
            lambda column: risk_model.get_autocorrelation(column, lags=lags)
        )
        autocorrelation.index.name = "Lag"

        return apply_rounding(
            autocorrelation, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_hurst_exponent(
        self,
        max_lag: int = 20,
        rounding: int | None = None,
    ):
        """
        Calculate the Hurst Exponent of each asset's daily (adjusted close) price series, a
        measure of long-term memory that indicates whether the price is mean-reverting,
        trending, or a random walk.

        The Hurst Exponent (H) is interpreted as follows:

        - H < 0.5: the series is mean-reverting (anti-persistent).
        - H = 0.5: the series is a random walk (no memory).
        - H > 0.5: the series is trending (persistent).

        For liquid stocks the estimate is usually close to 0.5, and with lags of at most
        `max_lag` days it describes short-term behaviour only; a value of 0.45 or 0.55 over a
        few years is within estimation noise rather than evidence of a tradable pattern.

        Args:
            max_lag (int, optional): The maximum lag, included, to use when estimating the exponent:
                the regression runs over the lags 2 up to and including `max_lag`. Defaults to 20.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to 4.

        Returns:
            pd.Series: The estimated Hurst Exponent for each asset.

        Notes:
        - The exponent is estimated with the generalized Hurst (structure function) method of
        Di Matteo, Aste & Dacorogna (2005), see `risk_model.get_hurst_exponent`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AMZN", "TSLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.risk.get_hurst_exponent()
        ```

        Which returns:

        |           |   Value |
        |:----------|--------:|
        | AMZN      |  0.459  |
        | TSLA      |  0.5198 |
        | Benchmark |  0.4361 |
        """
        # The estimator regresses the dispersion of lagged differences on the lag, which only identifies self-affinity on a level series. Feeding it returns, which are already differenced, collapses the exponent towards zero.  # noqa: E501
        prices = self._historical_data["daily"]["Adj Close"]

        hurst_exponent = prices.apply(
            lambda column: risk_model.get_hurst_exponent(column, max_lag=max_lag)
        )

        return apply_rounding(
            hurst_exponent, rounding if rounding is not None else self._rounding
        )

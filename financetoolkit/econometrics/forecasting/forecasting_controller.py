"""Forecasting Module"""

__docformat__ = "google"

from typing import Literal

import numpy as np
import pandas as pd

from financetoolkit.econometrics.forecasting import (
    forecast_evaluation_model,
    time_series_model,
)
from financetoolkit.helpers import handle_portfolio
from financetoolkit.risk.volatility import garch_model
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import (
    InvalidArgumentError,
    check_option,
    check_period,
    handle_errors,
    validate_alpha,
    validate_window,
)
from financetoolkit.utilities.logger_model import get_logger
from financetoolkit.utilities.sources_model import track_sources
from financetoolkit.utilities.statistics_model import PERIODS_PER_YEAR, apply_rounding

# ruff: noqa: E501

logger = get_logger()

PERIODS = ["daily", "weekly", "monthly", "quarterly", "yearly"]

# The GARCH-family variance models the model confidence set can compare: those of the returns alone.
VOLATILITY_GARCH_METHODS = [
    method
    for method in garch_model.GARCH_FAMILY_VARIANCE_PARAMETERS
    if method != "realized"
]

# A GARCH fit needs at least 100 returns (as rugarch's ugarchfit), so its window is two years with that minimum.  # noqa: E501
MINIMUM_GARCH_WINDOW = 100


def _garch_window_size(period: str) -> int:
    """Two years of observations with a minimum of 100, the default window of a GARCH fit."""
    return max(2 * PERIODS_PER_YEAR[period], MINIMUM_GARCH_WINDOW)


@filter_shown_columns
@track_sources
class ForecastingEconometrics:
    """
    Forecasts and their evaluation: ARIMA, VAR and VECM forecasts, impulse responses, variance decomposition, RMSE, MAE, out-of-sample validation, the Diebold-Mariano, Berkowitz, Hong-Li and directional accuracy tests and the model confidence set.
    """

    @handle_portfolio
    @handle_errors
    def get_diebold_mariano_test(
        self,
        period: str | None = None,
        method_a: str = "ewma",
        method_b: str = "rolling",
        window: int = 22,
        lambda_: float = 0.94,
        loss: str = "squared",
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Diebold-Mariano (1995) test per asset: whether one of two volatility
        forecasting methods predicts the realized (squared return) Variance more accurately.

        Rather than comparing two different assets, this compares two different ways
        of forecasting the *same* asset's next-day Variance -- e.g. the simpler
        exponentially weighted (EWMA, see `Risk.get_ewma_volatility`) approach against a
        plain rolling-window Standard Deviation -- to determine whether one is
        significantly more accurate than the other for a given asset.

        The squared return is a noisy stand-in for the true Variance. Patton (2011) shows
        that the squared-error loss still ranks two volatility forecasts correctly with such
        a proxy, while the absolute-error loss can rank them the wrong way round, so the
        squared-error loss is the safer choice. A negative statistic favors the first method.

        The formula is as follows:

        - Realized Variance: rₜ²
        - EWMA forecast: σₜ² = λ·σₜ₋₁² + (1 - λ)·rₜ₋₁², with λ = `lambda_` (0.94 by default), so
          the forecast for t uses returns up to t - 1
        - Rolling forecast: the sample variance (ddof=1) of the `window` (22 by default)
          returns up to t - 1
        - dₜ = L(rₜ² - f_a,ₜ) - L(rₜ² - f_b,ₜ), with L(e) = e² (squared loss) or |e| (absolute loss)
        - DM = mean(d) / √(γ₀ / n)·√((n - 1) / n), with γ₀ = (1 / n)·Σ(dₜ - mean(d))² and the Harvey,
          Leybourne and Newbold (1997) correction for a one-step horizon
        - P-Value = 2·P(T > |DM|) for a Student-t distribution with n - 1 degrees of freedom

        For more information about the method, see the following papers:

        - Diebold, F.X., & Mariano, R.S. (1995). "Comparing Predictive Accuracy."
        Journal of Business & Economic Statistics, 13(3), 253-263.
        - Patton, A.J. (2011). "Volatility Forecast Comparison Using Imperfect Volatility
        Proxies." Journal of Econometrics, 160(1), 246-256.

        Also known as: DM test, forecast comparison test.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
            Defaults to "daily", since `window` is expressed in return observations of this
            frequency.
            method_a (str, optional): The first volatility forecasting method. Options are: "ewma"
            (RiskMetrics EWMA), "rolling" (rolling sample variance). Defaults to "ewma".
            method_b (str, optional): The second (competing) volatility forecasting method. Options are:
            "ewma" (RiskMetrics EWMA), "rolling" (rolling sample variance). Defaults to "rolling".
            window (int, optional): The rolling window size used by the rolling method, at least 2.
            Defaults to 22 (approximately one trading month).
            lambda_ (float, optional): The decay factor used by the EWMA method. Defaults to 0.94.
            loss (str, optional): The loss function to compare forecast errors with. Options are:
            "squared" (squared error), "absolute" (absolute error). Defaults to "squared".
            include_benchmark (bool, optional): Whether to include "Benchmark" among the
            assets tested. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The Diebold-Mariano statistic, its p-value, the mean loss differential (negative
            favors `method_a`) and the number of observations used, per asset.

        Notes:
        - Both forecasts for period t use only the returns up to, and including, period t - 1, so this
        is a genuinely out-of-sample forecast comparison rather than an in-sample fit comparison.
        - Each asset uses its own return history, so an asset with a shorter history does not shorten
        the comparison of the others.
        - Like `Risk.get_var_backtest`, this is calculated over the full return history rather than the
        `within_period` slices used elsewhere in this module, since the rolling/EWMA forecasts need more
        history than a single sub-period provides.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["XOM", "CVX", "SHEL", "BP"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_diebold_mariano_test(method_a="ewma", method_b="rolling")
        ```

        Which returns:

        |                           |       XOM |       CVX |      SHEL |        BP |
        |:--------------------------|----------:|----------:|----------:|----------:|
        | Diebold-Mariano Statistic |   -3.9115 |   -4.9425 |   -1.8262 |   -1.3358 |
        | P-Value                   |    0.0001 |    0      |    0.0681 |    0.1818 |
        | Mean Loss Differential    |    0      |    0      |    0      |    0      |
        | Observations              | 1232      | 1232      | 1232      | 1232      |
        """
        validate_window("window", window, 2)
        if method_a not in ("ewma", "rolling") or method_b not in ("ewma", "rolling"):
            raise InvalidArgumentError(
                "method_a and method_b must be 'ewma' or 'rolling'."
            )

        period = period if period else "daily"

        check_period(period, ["daily", "weekly", "monthly", "quarterly", "yearly"])

        returns = self._filter_benchmark(
            self._historical_data[period]["Return"]
            .loc[self._start_date : self._end_date]
            .dropna(how="all"),
            include_benchmark=include_benchmark,
        )

        actual = returns**2
        forecast_a = forecast_evaluation_model.get_volatility_forecast(
            returns, method_a, window, lambda_
        )
        forecast_b = forecast_evaluation_model.get_volatility_forecast(
            returns, method_b, window, lambda_
        )

        result = forecast_evaluation_model.get_diebold_mariano_test(
            actual, forecast_a, forecast_b, loss=loss, horizon=1
        )

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_arima_forecast(
        self,
        period: str | None = None,
        column: str = "Adj Close",
        p: int = 1,
        d: int = 1,
        q: int = 1,
        forecast_steps: int = 5,
        add_constant: bool = True,
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Fit an ARIMA(p, d, q) model to every ticker's price series and forecast it: the future
        path of each series from its own past values and past forecast errors.

        The model is fitted with statsmodels' ARIMA.

        Also known as: Box-Jenkins model, autoregressive integrated moving average.

        An ARIMA(p, d, q) model differences the series `d` times to remove a
        (stochastic) trend, then fits an autoregressive-moving-average model to the
        result -- see `time_series_model.get_arima_forecast` for the full formula,
        estimation method (exact Maximum Likelihood via the Kalman filter) and its
        practical caveats.

        The formula is as follows:

        - y'ₜ = ΔᵈYₜ, the `column` level (Adj Close by default) differenced d times (d = 1 by default)
        - y'ₜ - μ = Σ(i=1..p) φᵢ·(y'ₜ₋ᵢ - μ) + εₜ + Σ(j=1..q) θⱼ·εₜ₋ⱼ, with p = q = 1 by default and μ the
          mean (d = 0) or drift (d ≥ 1) of y'ₜ when a constant is added, and μ = 0 otherwise
        - Fitted by exact Maximum Likelihood via the Kalman filter (statsmodels ARIMA)
        - Forecast: the conditional mean E[Y_T+h | Y_1..Y_T] for h = 1..`forecast_steps`, in levels

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily".
            column (str, optional): The historical data column to fit. Defaults to
            "Adj Close".
            p (int, optional): The autoregressive order. Defaults to 1.
            d (int, optional): The number of times to difference the series. Defaults
            to 1 (the typical choice for a non-stationary price level series).
            q (int, optional): The moving-average order. Defaults to 1.
            forecast_steps (int, optional): The number of periods ahead to forecast.
            Defaults to 5.
            add_constant (bool, optional): Whether to estimate the mean of the
            differenced series (its drift for `d >= 1`). Defaults to True -- see
            `time_series_model.get_arima_forecast` for when to set this to False.
            include_benchmark (bool, optional): Whether to include "Benchmark" among
            the tickers forecast. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the
            results to. Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The `forecast_steps`-ahead forecast, one column per ticker,
            indexed `1, ..., forecast_steps` (periods ahead, not a continuation of the
            historical date index). Each ticker's ARIMA model is fit independently on
            its own history; a ticker without enough data has an empty (NaN) column.

        Notes:
        - The method retrieves historical price data based on the specified `period`
        for every ticker in the Toolkit instance and fits an ARIMA model to each.
        - See `time_series_model.get_arima_forecast`'s `Notes` for this estimator's
        practical limitations (in particular, keep `p + q` small).
        - Requires statsmodels (`pip install financetoolkit[econometrics]`).

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["GCUSD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_arima_forecast(forecast_steps=10)
        ```

        Which returns:

        |   Step |   GCUSD |
        |-------:|--------:|
        |      1 | 4348.34 |
        |      2 | 4348.38 |
        |      3 | 4350.82 |
        |      4 | 4352.47 |
        |      5 | 4354.38 |
        |      6 | 4356.2  |
        |      7 | 4358.05 |
        |      8 | 4359.89 |
        |      9 | 4361.73 |
        |     10 | 4363.57 |
        """
        period = period if period else "daily"
        # Each ARIMA fit drops the missing values of its own series, so a series observed
        # at other dates does not cut the others down to the shared rows.
        prices = self._get_series_column(period, column)

        tickers = self._get_tickers(include_benchmark=include_benchmark)

        forecasts = {}
        for ticker in tickers:
            # Each ticker is fitted on its own history; one without enough data of its
            # own gets an empty column instead of failing the other tickers.
            try:
                forecasts[ticker] = time_series_model.get_arima_forecast(
                    prices[ticker],
                    p=p,
                    d=d,
                    q=q,
                    forecast_steps=forecast_steps,
                    add_constant=add_constant,
                )["forecast"]
            except ValueError as error:
                logger.warning("No ARIMA forecast for %s: %s", ticker, error)

        if not forecasts:
            raise ValueError("None of the tickers has enough data for an ARIMA fit.")

        result = pd.DataFrame(forecasts).reindex(columns=tickers)
        result.index.name = "Step"

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_var_forecast(
        self,
        period: str | None = None,
        column: str = "Return",
        lags: int = 1,
        forecast_steps: int = 5,
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Fit a Vector Autoregression (VAR) across every ticker and forecast it: how the tickers
        move together, each driven by the past values of all of them.

        A VAR models several series jointly: each ticker's value is regressed on `lags` past
        values of all tickers, including its own. Each asset's future can therefore depend on
        its own history and on the history of the others, without deciding in advance which
        one drives which.

        The formula is as follows:

        - Y_t = c + SUM_{l=1}^{lags}(Phi_l * Y_(t-l)) + e_t, with Y_t every ticker's `column` (Return by
          default) and `lags` = 1 by default
        - Each equation is an OLS regression on a constant and the `lags` lagged values of all tickers
        - Forecast: Yhat_(T+h) = c + SUM_{l=1}^{lags}(Phi_l * Yhat_(T+h-l)), with Yhat_s = Y_s for s <= T

        Where `Y_t` holds the values of all tickers at time `t` and each `Phi_l` is a matrix of
        coefficients. Because every equation has the same regressors, the model is estimated
        exactly by a separate OLS regression per ticker. Forecasts are produced by running the
        fitted equations forward one step at a time, feeding each forecast back in as the
        most recent lag. The coefficients show how shocks spread between assets;
        `get_impulse_response_function` and `get_variance_decomposition` summarise that, and
        `get_granger_causality` tests whether one ticker helps predict another. A VAR assumes
        stationary series, so returns are generally more suitable than prices (see
        `get_unit_root_test`).

        For more information about the method, see the following paper:

        - Sims, C.A. (1980). "Macroeconomics and Reality." Econometrica, 48(1), 1-48.

        Also known as: VAR model, vector autoregressive model.

        Referenced against:
        - statsmodels, VAR: https://www.statsmodels.org/stable/generated/statsmodels.tsa.vector_ar.var_model.VAR.html

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily".
            column (str, optional): The historical data column to model. Defaults to
            "Return".
            lags (int, optional): The VAR order. Defaults to 1.
            forecast_steps (int, optional): The number of periods ahead to forecast.
            Defaults to 5.
            include_benchmark (bool, optional): Whether to include "Benchmark" among
            the tickers modeled jointly. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the
            results to. Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The `forecast_steps`-ahead forecast, one column per ticker,
            indexed `1, ..., forecast_steps` (periods ahead).

        Notes:
        - The method retrieves historical data based on the specified `period` for every
        ticker in the Toolkit instance and fits the VAR model jointly across all of them.
        - Unlike `get_vecm_forecast`, this operates on `column` directly (typically
        "Return", a stationary series) rather than price levels -- a VAR in levels is
        misspecified for non-stationary series (see
        `cointegration_model.get_johansen_cointegration`); use `get_vecm_forecast`
        instead if the assets' price levels are cointegrated.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["^VIX", "^VIX3M"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_var_forecast(column="Adj Close", forecast_steps=20)
        ```

        Which returns:

        |   Step |    ^VIX |   ^VIX3M |
        |-------:|--------:|---------:|
        |      1 | 15.2712 |  18.3667 |
        |      2 | 15.5578 |  18.5342 |
        |      3 | 15.8138 |  18.6846 |
        |      4 | 16.0425 |  18.82   |
        |      5 | 16.2471 |  18.9418 |
        |      6 | 16.4301 |  19.0517 |
        |      7 | 16.5941 |  19.1509 |
        |      8 | 16.7411 |  19.2406 |
        |      9 | 16.873  |  19.322  |
        |     10 | 16.9915 |  19.3958 |
        |     11 | 17.0981 |  19.4629 |
        |     12 | 17.1942 |  19.5241 |
        |     13 | 17.2808 |  19.58   |
        |     14 | 17.359  |  19.6311 |
        |     15 | 17.4298 |  19.6781 |
        |     16 | 17.494  |  19.7213 |
        |     17 | 17.5523 |  19.7611 |
        |     18 | 17.6053 |  19.7979 |
        |     19 | 17.6536 |  19.8321 |
        |     20 | 17.6978 |  19.8639 |
        """
        period = period if period else "daily"
        tickers = self._get_tickers(include_benchmark=include_benchmark)
        returns = self._get_series_column(period, column, tickers)

        result = time_series_model.get_var_forecast(
            returns[tickers], lags=lags, forecast_steps=forecast_steps
        )

        return result["forecast"].pipe(
            apply_rounding, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_impulse_response_function(
        self,
        period: str | None = None,
        column: str = "Return",
        lags: int = 1,
        periods: int = 10,
        orthogonalized: bool = True,
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Impulse Response Function (IRF) of a Vector Autoregression across every
        ticker: how a shock to each ticker propagates through the whole system over the periods
        ahead.

        The shock is one standard deviation of the orthogonalized innovation by default and a
        unit shock to the reduced-form residual otherwise.

        Also known as: IRF.

        A natural companion to `get_var_forecast`: rather than forecasting the levels
        forward, this traces out each ticker's dynamic response to a shock in every
        ticker (including itself) -- see `time_series_model.get_impulse_response_function`
        for the full formula, the Cholesky-orthogonalization used to identify the
        shocks, and why the ordering of tickers is an identifying assumption.

        The formula is as follows:

        - Ψ₀ = I and Ψₙ = Σ(l=1..min(n, p)) Φₗ·Ψₙ₋ₗ, the moving-average matrices of the VAR(p) fitted on
          every ticker's `column` (Return by default), p = `lags` (1 by default)
        - Σᵤ = Û'Û / (T - k·p - 1) = P·P', with P the lower-triangular Cholesky factor
        - Θₙ = Ψₙ·P (orthogonalized, a one-standard-deviation shock) or Θₙ = Ψₙ
          (not orthogonalized, a unit reduced-form shock)
        - Response of ticker i, n periods after a shock to ticker j: Θₙ[i, j], for n = 0..`periods`

        Referenced against:
        - statsmodels, VARResults.irf: https://www.statsmodels.org/stable/generated/statsmodels.tsa.vector_ar.var_model.VARResults.irf.html

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily".
            column (str, optional): The historical data column to model. Defaults to
            "Return".
            lags (int, optional): The VAR order. Defaults to 1.
            periods (int, optional): The number of periods ahead to trace the response
            out to. Defaults to 10.
            orthogonalized (bool, optional): Whether to orthogonalize the shocks via a
            Cholesky decomposition of the residual covariance matrix -- see
            `time_series_model.get_impulse_response_function`. Defaults to True.
            include_benchmark (bool, optional): Whether to include "Benchmark" among
            the tickers modeled jointly. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the
            results to. Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Each shock ticker's response table, one column per response
            ticker, concatenated side by side under a top-level column per shock
            ticker, indexed `0, ..., periods` (horizon, `0` = impact response).

        Notes:
        - The ordering of the tickers modeled determines which ticker is treated as
        contemporaneously prior to the others. That ordering follows the Toolkit
        instance's own ticker order and is not configurable here -- reorder the
        Toolkit instance's tickers themselves to change the Cholesky identification
        order -- see `time_series_model.get_impulse_response_function`'s `Notes`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["^VIX", "^OVX"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_impulse_response_function(column="Adj Close", periods=20)
        ```

        Which returns:

        |   Horizon |   ('^VIX', '^VIX') |   ('^VIX', '^OVX') |   ('^OVX', '^VIX') |   ('^OVX', '^OVX') |
        |----------:|-------------------:|-------------------:|-------------------:|-------------------:|
        |         0 |             1.7975 |             1.1106 |             0      |             2.4048 |
        |         1 |             1.6806 |             1.1595 |             0.022  |             2.2148 |
        |         2 |             1.5725 |             1.1956 |             0.0407 |             2.0416 |
        |         3 |             1.4723 |             1.2207 |             0.0565 |             1.8834 |
        |         4 |             1.3794 |             1.2361 |             0.0698 |             1.7389 |
        |         5 |             1.2933 |             1.2433 |             0.0807 |             1.6069 |
        |         6 |             1.2133 |             1.2434 |             0.0897 |             1.4861 |
        |         7 |             1.1389 |             1.2374 |             0.097  |             1.3755 |
        |         8 |             1.0698 |             1.2262 |             0.1027 |             1.2743 |
        |         9 |             1.0054 |             1.2106 |             0.1071 |             1.1814 |
        |        10 |             0.9454 |             1.1914 |             0.1104 |             1.0963 |
        |        11 |             0.8895 |             1.1691 |             0.1126 |             1.0181 |
        |        12 |             0.8374 |             1.1444 |             0.114  |             0.9462 |
        |        13 |             0.7887 |             1.1176 |             0.1146 |             0.8801 |
        |        14 |             0.7432 |             1.0893 |             0.1145 |             0.8193 |
        |        15 |             0.7006 |             1.0597 |             0.1139 |             0.7633 |
        |        16 |             0.6608 |             1.0292 |             0.1129 |             0.7117 |
        |        17 |             0.6236 |             0.9981 |             0.1114 |             0.664  |
        |        18 |             0.5886 |             0.9667 |             0.1096 |             0.6201 |
        |        19 |             0.5559 |             0.9351 |             0.1075 |             0.5794 |
        |        20 |             0.5252 |             0.9034 |             0.1052 |             0.5418 |
        """
        period = period if period else "daily"
        tickers = self._get_tickers(include_benchmark=include_benchmark)
        returns = self._get_series_column(period, column, tickers)

        var_result = time_series_model.get_var_forecast(
            returns[tickers], lags=lags, forecast_steps=1
        )
        result = time_series_model.get_impulse_response_function(
            var_result, periods=periods, orthogonalized=orthogonalized
        )

        return apply_rounding(
            time_series_model.irf_summary_table(result),
            rounding if rounding is not None else self._rounding,
        )

    @handle_errors
    def get_variance_decomposition(
        self,
        period: str | None = None,
        column: str = "Return",
        lags: int = 1,
        periods: int = 10,
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the (orthogonalized) Forecast Error Variance Decomposition (FEVD) of a Vector
        Autoregression across every ticker: which share of each ticker's forecast error variance
        comes from the shock of each ticker.

        The shares are calculated for every horizon h = 1, ..., periods.

        Also known as: FEVD, variance decomposition.

        The other natural companion to `get_var_forecast` (alongside
        `get_impulse_response_function`, which the FEVD is built from) -- see
        `time_series_model.get_variance_decomposition` for the full formula. A large
        own-shock share at short horizons that decays as the horizon grows is the
        classic signature of a ticker that is initially self-driven but increasingly
        explained by the rest of the system over time.

        The formula is as follows:

        - Θₙ = Ψₙ·P, the orthogonalized moving-average matrices of `get_impulse_response_function`, from
          a VAR(p) on every ticker's `column` (Return by default), p = `lags` (1 by default)
        - Share of ticker j's shock in ticker i's h-step forecast error variance:
          ωᵢⱼ(h) = Σ(n=0..h-1) Θₙ[i, j]² / Σ(m=1..k) Σ(n=0..h-1) Θₙ[i, m]², for h = 1..`periods`

        Referenced against:
        - statsmodels, VARResults.fevd: https://www.statsmodels.org/stable/generated/statsmodels.tsa.vector_ar.var_model.VARResults.fevd.html

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily".
            column (str, optional): The historical data column to model. Defaults to
            "Return".
            lags (int, optional): The VAR order. Defaults to 1.
            periods (int, optional): The forecast horizon to decompose out to.
            Defaults to 10.
            include_benchmark (bool, optional): Whether to include "Benchmark" among
            the tickers modeled jointly. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the
            results to. Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Each response ticker's variance-share table (one column per
            shock ticker, rows summing to 1), concatenated side by side under a
            top-level column per response ticker, indexed `1, ..., periods` (horizon).

        Notes:
        - See `get_impulse_response_function`'s `Notes` on the Cholesky ordering --
        the same identifying assumption applies here.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["^VIX", "^GVZ"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_variance_decomposition(column="Adj Close", periods=20)
        ```

        Which returns:

        |   Horizon |   ('^VIX', '^VIX') |   ('^VIX', '^GVZ') |   ('^GVZ', '^VIX') |   ('^GVZ', '^GVZ') |
        |----------:|-------------------:|-------------------:|-------------------:|-------------------:|
        |         1 |             1      |             0      |             0.1832 |             0.8168 |
        |         2 |             1      |             0      |             0.1925 |             0.8075 |
        |         3 |             0.9999 |             0.0001 |             0.2016 |             0.7984 |
        |         4 |             0.9998 |             0.0002 |             0.2104 |             0.7896 |
        |         5 |             0.9997 |             0.0003 |             0.219  |             0.781  |
        |         6 |             0.9996 |             0.0004 |             0.2273 |             0.7727 |
        |         7 |             0.9995 |             0.0005 |             0.2354 |             0.7646 |
        |         8 |             0.9993 |             0.0007 |             0.2432 |             0.7568 |
        |         9 |             0.9991 |             0.0009 |             0.2506 |             0.7494 |
        |        10 |             0.9989 |             0.0011 |             0.2578 |             0.7422 |
        |        11 |             0.9987 |             0.0013 |             0.2647 |             0.7353 |
        |        12 |             0.9985 |             0.0015 |             0.2714 |             0.7286 |
        |        13 |             0.9983 |             0.0017 |             0.2777 |             0.7223 |
        |        14 |             0.9981 |             0.0019 |             0.2837 |             0.7163 |
        |        15 |             0.9978 |             0.0022 |             0.2895 |             0.7105 |
        |        16 |             0.9976 |             0.0024 |             0.2949 |             0.7051 |
        |        17 |             0.9974 |             0.0026 |             0.3002 |             0.6998 |
        |        18 |             0.9971 |             0.0029 |             0.3051 |             0.6949 |
        |        19 |             0.9969 |             0.0031 |             0.3098 |             0.6902 |
        |        20 |             0.9967 |             0.0033 |             0.3142 |             0.6858 |
        """
        period = period if period else "daily"
        tickers = self._get_tickers(include_benchmark=include_benchmark)
        returns = self._get_series_column(period, column, tickers)

        var_result = time_series_model.get_var_forecast(
            returns[tickers], lags=lags, forecast_steps=1
        )
        result = time_series_model.get_variance_decomposition(
            var_result, periods=periods
        )

        return apply_rounding(
            time_series_model.variance_decomposition_summary_table(result),
            rounding if rounding is not None else self._rounding,
        )

    @handle_errors
    def get_vecm_forecast(
        self,
        period: str | None = None,
        column: str = "Adj Close",
        lags: int = 1,
        forecast_steps: int = 5,
        significance: float = 0.05,
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Fit a Vector Error Correction Model (VECM) across every (cointegrated) ticker and forecast
        it: how the prices move back towards their long-run equilibrium while following their
        short-run dynamics.

        The model is fitted with statsmodels' VECM.

        Also known as: VECM, error correction model (for the multivariate/cointegrated
        case).

        A VECM keeps a VAR's short-run dynamics while ALSO letting each asset's price
        change react to how far the system currently sits from its long-run
        equilibrium (the cointegrating relationship(s) among the tickers, taken from
        `cointegration_model.get_johansen_cointegration`) -- see
        `time_series_model.get_vecm_forecast` for the full formula, estimation method
        and verification notes.

        The formula is as follows:

        - Rank r = the number of leading Johansen trace statistics (with a constant and `lags` lagged differences) that
          exceed their critical value at `significance` (5% by default); r = 0 raises an error
        - ΔYₜ = α·β'·Yₜ₋₁ + Σ(i=1..k) Γᵢ·ΔYₜ₋ᵢ + c + εₜ, with Yₜ every ticker's `column` level (Adj Close by
          default), k = `lags` (1 by default) and c a constant outside the cointegrating relation
          (statsmodels VECM with an unrestricted constant), fitted by Maximum Likelihood
        - Forecast: the fitted recursion run forward `forecast_steps` periods, in levels

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily".
            column (str, optional): The historical data column to model. Defaults to
            "Adj Close" -- a VECM needs price LEVELS (non-stationary, cointegrated
            series), not returns, the same input `get_johansen_cointegration` expects.
            lags (int, optional): The number of lagged first differences to
            include as short-run dynamics. Defaults to 1.
            forecast_steps (int, optional): The number of periods ahead to forecast.
            Defaults to 5.
            significance (float, optional): The significance level (one of 0.01, 0.05,
            0.10) at which the Johansen trace test determines the cointegrating rank.
            Defaults to 0.05.
            include_benchmark (bool, optional): Whether to include "Benchmark" among
            the tickers modeled jointly. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the
            results to. Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The `forecast_steps`-ahead forecast, on the price-level
            scale, one column per ticker, indexed `1, ..., forecast_steps` (periods
            ahead).

        Raises:
            ValueError: If the Johansen test does not reject a cointegrating rank of 0
            for the Toolkit instance's tickers at `significance` -- a VECM is not
            appropriate for non-cointegrated assets; use `get_var_forecast` on their
            returns instead.

        Notes:
        - The method retrieves historical price data based on the specified `period`
        for every ticker in the Toolkit instance and fits the VECM jointly across all
        of them.
        - Requires statsmodels (`pip install financetoolkit[econometrics]`).

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["GOOGL", "GOOG"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        # The two share classes of Alphabet are cointegrated: their prices cannot drift apart.
        toolkit.econometrics.get_vecm_forecast()
        ```

        Which returns:

        |   Step |   GOOGL |    GOOG |
        |-------:|--------:|--------:|
        |      1 | 312.569 | 313.352 |
        |      2 | 312.75  | 313.553 |
        |      3 | 312.929 | 313.747 |
        |      4 | 313.109 | 313.941 |
        |      5 | 313.289 | 314.135 |
        """
        period = period if period else "daily"
        tickers = self._get_tickers(include_benchmark=include_benchmark)
        prices = self._get_series_column(period, column, tickers)

        result = time_series_model.get_vecm_forecast(
            prices[tickers],
            lags=lags,
            forecast_steps=forecast_steps,
            significance=significance,
        )

        return result["forecast"].pipe(
            apply_rounding, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_forecast_error(
        self,
        period: str | None = None,
        column: str = "Return",
        include_benchmark: bool = False,
        method: Literal["rmse", "mae"] = "rmse",
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series:
        """
        Calculate the forecast error between every pair of tickers: how far apart the two series
        are in a typical period, as the Root Mean Squared Error or the Mean Absolute Error.

        This controller method compares two ASSETS' series directly, treating one as a naive
        "forecast" proxy for the other -- a simple, tracking-error style measure of
        how closely two series move together in absolute deviation terms (e.g. a
        portfolio versus a benchmark, or one asset as a naive stand-in forecast for
        a similar one). For evaluating an actual FORECASTING MODEL (ARIMA/VAR)
        rather than one asset as a naive proxy for another, use
        `get_out_of_sample_validation` instead.

        The formula for the RMSE is as follows:

        - RMSE = √((1 / n)·Σ(aₜ - bₜ)²), with aₜ and bₜ the two tickers' `column` (Return by default) on
          the n dates where both have data

        The formula for the MAE is as follows:

        - MAE = (1 / n)·Σ|aₜ - bₜ|, with aₜ and bₜ the two tickers' `column` (Return by default) on the
          n dates where both have data

        Both are expressed in the units of the series itself (e.g. a return difference) and
        answer "how far apart are the two series in a typical period"; a lower value means the
        two tickers track each other more closely. The RMSE squares the differences and is
        therefore dominated by a few large gaps, while in the MAE every difference counts in
        proportion to its size, so the MAE is robust to occasional outliers.

        For a fund against its index the RMSE comes close to the tracking error but is not
        the same: the tracking error (`Performance.get_tracking_error`) is the Standard
        Deviation of the return differences, which leaves out their average, while the RMSE
        includes it. Both are per period, so a daily RMSE is usually multiplied by the
        square root of 252 to annualize it.

        Also known as: RMSE, RMSD (Root Mean Squared Deviation), MAE, MAD (Mean Absolute
        Deviation), forecast accuracy.

        Referenced against:
        - statsmodels, rmse: https://www.statsmodels.org/stable/generated/statsmodels.tools.eval_measures.rmse.html
        - statsmodels, meanabs: https://www.statsmodels.org/stable/generated/statsmodels.tools.eval_measures.meanabs.html

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily".
            column (str, optional): The historical data column to compare. Defaults to
            "Return".
            include_benchmark (bool, optional): Whether to include "Benchmark" among the
            tickers paired up. Defaults to False.
            method (str, optional): The error measure. Options are: "rmse" (Root Mean Squared
            Error), "mae" (Mean Absolute Error). Defaults to "rmse".
            rounding (int | None, optional): The number of decimals to round the
            results to. Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.Series: The RMSE or the MAE (named "RMSE" or "MAE"), indexed by unordered
            `(Ticker A, Ticker B)` pair.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["IVV", "VOO"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_forecast_error(include_benchmark=True)
        ```

        Which returns:

        |                      |   RMSE |
        |:---------------------|-------:|
        | ('IVV', 'VOO')       | 0.0004 |
        | ('IVV', 'Benchmark') | 0.0009 |
        | ('VOO', 'Benchmark') | 0.001  |
        """
        check_option("method", method, ("rmse", "mae"))

        period = period if period else "daily"
        returns = self._get_series_column(period, column)

        error = (
            forecast_evaluation_model.get_mae
            if method == "mae"
            else forecast_evaluation_model.get_rmse
        )
        values = {
            (ticker_a, ticker_b): error(returns[ticker_a], returns[ticker_b])
            for ticker_a, ticker_b in self._all_unordered_pairs(
                include_benchmark=include_benchmark
            )
        }

        name = method.upper()

        if not values:
            raise ValueError(
                f"The {name} compares two series, a forecast and the actual values, so it "
                "needs at least two tickers (or include_benchmark=True for the benchmark)."
            )

        result = pd.Series(values, name=name)
        result.index = result.index.set_names(["Ticker A", "Ticker B"])

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_out_of_sample_validation(
        self,
        period: str | None = None,
        column: str = "Adj Close",
        model: str = "arima",
        train_fraction: float = 0.8,
        p: int = 1,
        d: int = 1,
        q: int = 1,
        add_constant: bool = True,
        lags: int = 1,
        other_tickers: list[str] | None = None,
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the out-of-sample forecast error of an ARIMA or VAR forecast of every ticker:
        its RMSE and MAE on a holdout the model was not fitted on.

        The model is fitted only on a training portion of the history, forecasts the held-out
        remainder, and the forecast is scored against what actually happened.

        Also known as: hold-out validation, train/test split validation.

        See `forecast_evaluation_model.get_out_of_sample_validation` for the general
        harness this wraps. Since that function takes a raw Python callable (not
        serializable for e.g. the MCP-facing tool layer), this controller method
        instead hardcodes the choice between the two forecasting models:

        - ARIMA: `time_series_model.get_arima_forecast` is fit on each
          ticker's own training-period series (`p`, `d`, `q`, `add_constant`
          control the model, same as `get_arima_forecast`).
        - VAR: `time_series_model.get_var_forecast` is fit on the
          training-period series of each ticker together with `other_tickers`
          (`lags` controls the VAR order, defaulting to every other ticker in the
          Toolkit instance if not given); only that ticker's own forecast column
          is scored against its holdout.

        The formula is as follows:

        - Training sample: the first int(`train_fraction`·n) observations (80% by default) of the
          `column` level (Adj Close by default); the remaining H observations are the holdout
        - Ŷₜ: the H-step forecast of the model fitted on the training sample only, an ARIMA(p, d, q)
          (ARIMA(1, 1, 1) by default) or a VAR(`lags`) on the ticker and `other_tickers`
        - RMSE = √((1 / H)·Σ(Yₜ - Ŷₜ)²) and MAE = (1 / H)·Σ|Yₜ - Ŷₜ|, over the holdout

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily".
            column (str, optional): The historical data column to validate. Defaults
            to "Adj Close".
            model (str, optional): The forecasting model. Options are: "arima" (ARIMA), "var"
            (Vector Autoregression). Defaults to "arima".
            train_fraction (float, optional): The fraction of observations used for
            training; the remainder is the holdout. Defaults to 0.8.
            p (int, optional): The ARIMA autoregressive order (`model="arima"` only).
            Defaults to 1.
            d (int, optional): The ARIMA differencing order (`model="arima"` only).
            Defaults to 1.
            q (int, optional): The ARIMA moving-average order (`model="arima"` only).
            Defaults to 1.
            add_constant (bool, optional): Whether the ARIMA model estimates a
            free intercept (`model="arima"` only). Defaults to True.
            lags (int, optional): The VAR order (`model="var"` only). Defaults to 1.
            other_tickers (list[str] | None, optional): The other assets to include in
            the VAR system alongside the ticker being validated (`model="var"` only).
            Defaults to None, meaning every other ticker in the Toolkit instance.
            include_benchmark (bool, optional): Whether to include "Benchmark" among
            the tickers validated (and, for `model="var"`, among the default
            `other_tickers`). Defaults to False.
            rounding (int | None, optional): The number of decimals to round the
            results to. Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: `RMSE`, `MAE` and `Holdout Observations`, one column per
            ticker. With `model="arima"` each ticker uses its own history and a ticker
            without enough data has an empty (NaN) column.

        Raises:
            ValueError: If `model` is not "arima" or "var", or (for `model="var"`)
            fewer than 2 tickers are available to form a VAR system.

        Notes:
        - Requires statsmodels for the ARIMA model, the default; the VAR model does not
        (`pip install financetoolkit[econometrics]`).

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["BTCUSD", "ETHUSD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_out_of_sample_validation()
        ```

        Which returns:

        |                      |   BTCUSD |   ETHUSD |
        |:---------------------|---------:|---------:|
        | RMSE                 | 11424.9  |  1003.11 |
        | MAE                  |  9896.95 |   842.37 |
        | Holdout Observations |   366    |   366    |
        """
        period = period if period else "daily"
        # Every series at its own observations: the ARIMA validation drops the missing
        # values per series and the VAR branch aligns its own system with dropna below.
        prices = self._get_series_column(period, column)

        tickers = self._get_tickers(include_benchmark=include_benchmark)

        check_option("model", model, ("arima", "var"))

        columns = {}
        for ticker in tickers:
            if model == "arima":
                # As in get_arima_forecast, a ticker without enough data of its own
                # gets an empty column instead of failing the other tickers.
                try:
                    columns[ticker] = (
                        forecast_evaluation_model.get_out_of_sample_validation(
                            prices[ticker],
                            time_series_model.get_arima_forecast,
                            train_fraction=train_fraction,
                            p=p,
                            d=d,
                            q=q,
                            add_constant=add_constant,
                        )
                    )
                except ValueError as error:
                    logger.warning("No ARIMA validation for %s: %s", ticker, error)
                continue

            ticker_other_tickers = (
                other_tickers
                if other_tickers is not None
                else [
                    other_ticker for other_ticker in tickers if other_ticker != ticker
                ]
            )

            if not ticker_other_tickers:
                raise InvalidArgumentError(
                    "other_tickers must be provided (and non-empty) when model='var' "
                    "-- a VAR needs at least one other series to model jointly with "
                    "each ticker."
                )

            system = prices[[ticker, *ticker_other_tickers]].dropna()
            split = int(len(system) * train_fraction)
            train, holdout = system.iloc[:split], system.iloc[split:]

            var_result = time_series_model.get_var_forecast(
                train, lags=lags, forecast_steps=len(holdout)
            )
            forecast = pd.Series(
                var_result["forecast"][ticker].to_numpy(), index=holdout.index
            )

            columns[ticker] = pd.Series(
                {
                    "RMSE": forecast_evaluation_model.get_rmse(
                        holdout[ticker], forecast
                    ),
                    "MAE": forecast_evaluation_model.get_mae(holdout[ticker], forecast),
                    "Holdout Observations": len(holdout),
                }
            )

        if not columns:
            raise ValueError("None of the tickers has enough data for an ARIMA fit.")

        result = pd.DataFrame(columns).reindex(columns=tickers)

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

    @handle_portfolio
    @handle_errors
    def get_berkowitz_test(
        self,
        period: str | None = None,
        model: str = "garch",
        alpha: float | None = None,
        window: int | None = None,
        lambda_: float = 0.94,
        refit_every: int = 21,
        garch_method: str = "garch",
        garch_distribution: str = "normal",
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate Berkowitz's likelihood ratio test of the one-period-ahead density forecasts of a return model:
        whether the forecast distributions match the returns in mean, variance and independence.

        A Value at Risk backtest only looks at one quantile; a density forecast says how likely every return
        is. The test transforms each realized return with the distribution function of its forecast (the
        probability integral transform, PIT), which is uniform and independent over time when the forecasts
        are right, takes the normal quantile of the PITs and tests whether those are independent standard
        normal: a mean away from 0 means the forecasts are biased, a standard deviation above 1 that they
        understate the risk and an autocorrelation that they miss its dynamics. With a tail probability, only
        the tail below that quantile is tested, the part a Value at Risk at that level depends on.

        The formula is as follows (with the default GARCH model and daily returns, for which the window is 504):

        - PIT_t = F((Return_t - Mu) / Volatility_t), from a GARCH(1, 1) with normal innovations fitted on the 504 returns
          before t, refitted every 21 returns
        - z_t = Φ^-1(PIT_t) and z_t - μ = ρ (z_(t-1) - μ) + ε_t, with ε_t ~ N(0, σ^2)
        - Berkowitz Statistic = -2 (ln L(0, 1, 0) - ln L(μ_hat, σ_hat, ρ_hat)), compared with a χ² distribution with 3
          degrees of freedom
        - With alpha: every z_t above q = Φ^-1(alpha) is censored at q, the AR(1) term is left out and the statistic
          -2 (ln L(0, 1) - ln L(μ_hat, σ_hat)) is compared with a χ² distribution with 2 degrees of freedom

        Where L is the exact Gaussian AR(1) likelihood (see `forecast_evaluation_model.get_berkowitz_test`) and F
        the distribution function of the standardized innovations with their fitted Shape and Skew.
        The Gaussian model uses a normal distribution with the mean and standard deviation (ddof=1) of the
        `window` returns before t and the EWMA model a normal distribution with mean zero and the RiskMetrics
        variance with decay `lambda_`.

        For more information about the method, see the following papers:

        - Berkowitz, J. (2001). "Testing Density Forecasts, With Applications to Risk Management." Journal of
        Business & Economic Statistics, 19(4), 465-474.
        - Diebold, F.X., Gunther, T.A. and Tay, A.S. (1998). "Evaluating Density Forecasts with Applications to
        Financial Risk Management." International Economic Review, 39(4), 863-883.

        Also known as: density forecast test, PIT test, Berkowitz LR test.

        Referenced against:
        - statsmodels, ARIMA: https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html

        Shares similar logic:
        - rugarch (R), BerkowitzTest: https://github.com/cran/rugarch/blob/1.5-6/R/rugarch-tests.R#L327

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly or yearly).
            Defaults to "daily".
            model (str, optional): The density forecast. Options are: "garch" (GARCH-family model),
            "gaussian" (rolling normal), "ewma" (RiskMetrics EWMA). Defaults to "garch".
            alpha (float | None, optional): The tail probability of the tail test. Defaults to None, which tests
            the whole density.
            window (int | None, optional): The number of returns of every window or GARCH fit, at least 100
            for the GARCH model (a GARCH fit needs at least 100 returns) and 2 otherwise. Defaults to None,
            which uses two years of observations with a minimum of 100 for "garch" (504 daily returns) and
            a year with a minimum of 20 for "gaussian".
            lambda_ (float, optional): The decay factor of model="ewma". Defaults to 0.94.
            refit_every (int, optional): The number of returns between two GARCH fits. Defaults to 21.
            garch_method (str, optional): The GARCH variance model. Options are: "garch" (GARCH(1, 1)), "gjr"
            (GJR-GARCH(1, 1, 1)), "egarch" (EGARCH(1, 1)), "igarch" (IGARCH(1, 1)), "aparch" (APARCH(1, 1)),
            "tgarch" (TGARCH), "ngarch" (NGARCH), "avgarch" (AVGARCH), "nagarch" (NAGARCH), "component"
            (component GARCH), "figarch" (FIGARCH(1, d, 1)). Defaults to "garch".
            garch_distribution (str, optional): The GARCH innovation distribution. Options are: "normal"
            (normal), "t" (Student-t), "skewt" (skewed Student-t of Hansen), "skewt_fernandez_steel" (skewed
            Student-t of Fernández and Steel), "ged" (generalized error distribution), "johnson_su" (Johnson
            SU). Defaults to "normal".
            include_benchmark (bool, optional): Whether to include "Benchmark" among the assets tested. Defaults
            to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The Berkowitz statistic, its p-value, the fitted Mean, Standard Deviation and
            Autocorrelation (NaN for the tail test) of z, the number of observations and, with `alpha`, of
            tail observations, per asset.

        Notes:
        - Every forecast only uses the returns before it, and each asset uses its own return history.
        - Unlike rugarch's BerkowitzTest, which fits the AR(1) by least squares, conditional on the first
        observation, the statistic here maximizes the exact likelihood of the paper, which differs by about 1%
        of the statistic on 800 observations.
        - A GARCH fit takes one to five seconds per asset and every refit a fraction of a second.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["QQQ", "TLT", "GLD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2019-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_berkowitz_test()
        ```

        Which returns:

        |                     |       QQQ |       TLT |       GLD |
        |:--------------------|----------:|----------:|----------:|
        | Berkowitz Statistic |    8.5086 |    3.9236 |    3.8163 |
        | P-Value             |    0.0366 |    0.2698 |    0.282  |
        | Mean                |   -0.0443 |   -0.0153 |    0.0396 |
        | Standard Deviation  |    1.0494 |    0.9641 |    1.0273 |
        | Autocorrelation     |   -0.0028 |   -0.0176 |    0.002  |
        | Observations        | 1255      | 1255      | 1255      |
        """
        if alpha is not None:
            validate_alpha(alpha)
        period = period if period else "daily"

        check_period(period, PERIODS)
        check_option("model", model, ("garch", "gaussian", "ewma"))

        validate_window(
            "window", window, MINIMUM_GARCH_WINDOW if model == "garch" else 2
        )
        window = (
            window
            if window is not None
            else (
                _garch_window_size(period)
                if model == "garch"
                else max(PERIODS_PER_YEAR[period], 20)
            )
        )
        returns = self._filter_benchmark(
            self._historical_data[period]["Return"]
            .loc[self._start_date : self._end_date]
            .dropna(how="all"),
            include_benchmark=include_benchmark,
        )
        # A ticker whose forecasts cannot be evaluated (a constant series has no volatility to
        # fit) is left out instead of failing the other tickers.
        tests = {}
        for ticker in returns.columns:
            try:
                pit = forecast_evaluation_model.get_probability_integral_transform(
                    returns[ticker],
                    model,
                    window,
                    lambda_,
                    refit_every,
                    garch_method,
                    garch_distribution,
                )
                tests[ticker] = forecast_evaluation_model.get_berkowitz_test(pit, alpha)
            except InvalidArgumentError:
                # A mistake in the call is not a reason to skip a ticker.
                raise
            except ValueError as error:
                logger.warning("No Berkowitz test for %s: %s", ticker, error)

        if not tests:
            raise ValueError(
                "None of the tickers has returns to evaluate forecasts on."
            )

        result = pd.DataFrame(tests)

        if alpha is None:
            result = result.drop(index="Tail Observations")

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

    @handle_portfolio
    @handle_errors
    def get_hong_li_test(
        self,
        period: str | None = None,
        model: str = "garch",
        lags: int = 4,
        window: int | None = None,
        lambda_: float = 0.94,
        refit_every: int = 21,
        garch_method: str = "garch",
        garch_distribution: str = "normal",
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the nonparametric density forecast test of Hong and Li: whether the one-period-ahead forecast
        densities of a return model have the right shape and capture the dependence of the returns.

        Correct density forecasts turn the realized returns into probability integral transforms (PITs) that
        are independent and uniform on (0, 1), so every pair of a PIT and the PIT j periods before it is spread
        evenly over the unit square. The test estimates the joint density of those pairs with a kernel and
        measures how far it is from flat, for every lag: a forecast density of the wrong shape (too thin
        tails, a missed skew) and dependence the model misses (volatility clustering it does not follow) both
        move it away. Where `get_berkowitz_test` checks the mean, variance and first autocorrelation of the
        normal quantiles of the PITs, this test looks at the whole joint density.

        The formula is as follows (with the default GARCH model and daily returns, for which the window is 504):

        - PIT_t = F((Return_t - Mu) / Volatility_t), from a GARCH(1, 1) with normal innovations fitted on the 504 returns
          before t, refitted every 21 returns (see `get_berkowitz_test`)
        - g_j(x_1, x_2) = Σ_t K_h(x_1, PIT_t) K_h(x_2, PIT_(t-j)) / (n - j), with K_h the boundary-modified quartic
          kernel and h = S n^(-1/6), S the standard deviation of the PITs
        - M(j) = ∫∫ (g_j(x_1, x_2) - 1)^2 dx_1 dx_2 over the unit square
        - Q(j) = h ((n - j) M(j) - A_h) / √V_0 for j = 1, ..., `lags`
        - W = Σ_j Q(j) / √`lags`, the portmanteau statistic

        Where A_h and V_0 are the centring and scaling constants of Hong and Li (2005), see
        `forecast_evaluation_model.get_hong_li_test`. Only large values reject. The statistics are standard normal
        under the null only as the sample grows, and slowly, so the p-values come from their distribution
        simulated under the null (independent uniform PITs) for the same number of PITs and lags, see
        `forecast_evaluation_model.get_hong_li_p_value`. The Gaussian and EWMA models give the density
        forecasts of `get_berkowitz_test`.

        For more information about the method, see the following papers:

        - Hong, Y. and Li, H. (2005). "Nonparametric Specification Testing for Continuous-Time Models with
        Applications to Term Structure of Interest Rates." The Review of Financial Studies, 18(1), 37-84.
        - Diebold, F.X., Gunther, T.A. and Tay, A.S. (1998). "Evaluating Density Forecasts with Applications to
        Financial Risk Management." International Economic Review, 39(4), 863-883.

        Also known as: Hong-Li test, nonparametric density forecast test, PIT independence and uniformity
        test, HL portmanteau test.

        Shares similar logic:
        - rugarch (R), HLTest: https://github.com/cran/rugarch/blob/1.5-6/R/rugarch-tests.R#L607

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly or yearly).
            Defaults to "daily".
            model (str, optional): The density forecast. Options are: "garch" (GARCH-family model),
            "gaussian" (rolling normal), "ewma" (RiskMetrics EWMA). Defaults to "garch".
            lags (int, optional): The number of lags of the portmanteau statistic. Defaults to 4.
            window (int | None, optional): The number of returns of every window or GARCH fit, at least 100
            for the GARCH model (a GARCH fit needs at least 100 returns) and 2 otherwise. Defaults to None,
            which uses two years of observations with a minimum of 100 for "garch" (504 daily returns) and
            a year with a minimum of 20 for "gaussian".
            lambda_ (float, optional): The decay factor of model="ewma". Defaults to 0.94.
            refit_every (int, optional): The number of returns between two GARCH fits. Defaults to 21.
            garch_method (str, optional): The GARCH variance model. Options are: "garch" (GARCH(1, 1)), "gjr"
            (GJR-GARCH(1, 1, 1)), "egarch" (EGARCH(1, 1)), "igarch" (IGARCH(1, 1)), "aparch" (APARCH(1, 1)),
            "tgarch" (TGARCH), "ngarch" (NGARCH), "avgarch" (AVGARCH), "nagarch" (NAGARCH), "component"
            (component GARCH), "figarch" (FIGARCH(1, d, 1)). Defaults to "garch".
            garch_distribution (str, optional): The GARCH innovation distribution. Options are: "normal"
            (normal), "t" (Student-t), "skewt" (skewed Student-t of Hansen), "skewt_fernandez_steel" (skewed
            Student-t of Fernández and Steel), "ged" (generalized error distribution), "johnson_su" (Johnson
            SU). Defaults to "normal".
            include_benchmark (bool, optional): Whether to include "Benchmark" among the assets tested. Defaults
            to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The statistic Q(j) and its p-value for every lag, the portmanteau W and its p-value,
            the bandwidth and the number of observations, per asset.

        Notes:
        - Every forecast only uses the returns before it, and each asset uses its own return history.
        - The statistic follows rugarch's HLTest as its documentation defines it. Unlike HLTest, which compares
        the statistics with the standard normal distribution, the p-values come from the simulated null
        distribution.
        - The asymptotic normal distribution is reached slowly: on simulated independent uniform PITs every Q(j)
        has a mean of about 0 and a standard deviation of about 1.2 at 500 to 2,000 observations, and as the
        Q(j) share the estimate of the marginal density of the PITs (a correlation of about 0.45), W has one of
        about 1.8 with 4 lags. Compared with the standard normal distribution W would reject a correct model at
        5% about 16% to 20% of the time; with the simulated null distribution about 4.4% (1,000 PITs). The
        simulation treats the PITs as those of a known model: estimating the model parameters does not change
        the asymptotic distribution (Hong and Li, 2005), and the GARCH parameters here come from the returns
        before each forecast.
        - A GARCH fit takes one to five seconds per asset and every refit a fraction of a second.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["QQQ", "TLT", "GLD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2019-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_hong_li_test()
        ```

        Which returns:

        |                |       QQQ |       TLT |       GLD |
        |:---------------|----------:|----------:|----------:|
        | Q(1) Statistic |    7.1786 |    4.2855 |    4.451  |
        | Q(1) P-Value   |    0      |    0.0016 |    0.0012 |
        | Q(2) Statistic |    4.7805 |    2.5799 |    3.4082 |
        | Q(2) P-Value   |    0.0006 |    0.0273 |    0.0075 |
        | Q(3) Statistic |    3.666  |    2.833  |    2.463  |
        | Q(3) P-Value   |    0.0049 |    0.0187 |    0.0323 |
        | Q(4) Statistic |    4.8672 |    2.2805 |    5.7016 |
        | Q(4) P-Value   |    0.0006 |    0.0418 |    0.0001 |
        | W Statistic    |   10.2461 |    5.9894 |    8.0119 |
        | W P-Value      |    0.0001 |    0.005  |    0.0007 |
        | Bandwidth      |    0.0855 |    0.0855 |    0.0847 |
        | Observations   | 1255      | 1255      | 1255      |
        """
        period = period if period else "daily"

        check_period(period, PERIODS)
        check_option("model", model, ("garch", "gaussian", "ewma"))

        validate_window(
            "window", window, MINIMUM_GARCH_WINDOW if model == "garch" else 2
        )
        window = (
            window
            if window is not None
            else (
                _garch_window_size(period)
                if model == "garch"
                else max(PERIODS_PER_YEAR[period], 20)
            )
        )
        returns = self._filter_benchmark(
            self._historical_data[period]["Return"]
            .loc[self._start_date : self._end_date]
            .dropna(how="all"),
            include_benchmark=include_benchmark,
        )
        pit = forecast_evaluation_model.get_probability_integral_transform(
            returns,
            model,
            window,
            lambda_,
            refit_every,
            garch_method,
            garch_distribution,
        )
        result = forecast_evaluation_model.get_hong_li_test(pit, lags)

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

    @handle_portfolio
    @handle_errors
    def get_directional_accuracy_test(
        self,
        period: str | None = None,
        method: Literal["pesaran_timmermann", "anatolyev_gerko"] = "pesaran_timmermann",
        forecast: str = "momentum",
        window: int | None = None,
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate a directional accuracy test: whether a return forecast predicts the direction of the next
        return better than chance, with the Pesaran-Timmermann or the Anatolyev-Gerko test.

        A forecast that is right about the sign more than half of the time is not necessarily useful: in a rising
        market a forecast that is always positive is right on most days. The Pesaran-Timmermann test compares the
        share of correct signs with the share expected if the forecast and the returns were independent. The
        Anatolyev-Gerko test weighs each sign by the size of the return: it is the excess return of going long
        when the forecast is positive and short when it is negative over that of the same share of long positions
        held at random, so it asks whether the forecast would have made money.

        The formula is as follows (with the default momentum forecast and daily returns, for which the window is 252):

        - Forecast_t = mean(Return_(t-252), ..., Return_(t-1))
        - P = (1 / n) Σ 1{Forecast_t · Return_t > 0}, P* = P_y P_x + (1 - P_y) (1 - P_x)
        - Pesaran-Timmermann Statistic = (P - P*) / √(V(P) - V(P*))
        - Anatolyev-Gerko Statistic = (A - B) / √V, with A = (1 / n) Σ sign(Forecast_t) Return_t and
          B = mean(sign(Forecast)) · mean(Return)
        - P-Value = P(N(0, 1) > Statistic)

        Where P_y and P_x are the shares of positive returns and forecasts, V(P) = P* (1 - P*) / n,
        V(P*) = (2 P_y - 1)^2 P_x (1 - P_x) / n + (2 P_x - 1)^2 P_y (1 - P_y) / n + 4 P_y P_x (1 - P_y) (1 - P_x) / n^2
        and V = 4 p (1 - p) Σ (Return_t - mean(Return))^2 / n^2 with p = (1 + mean(sign(Forecast))) / 2.
        The autoregressive forecast predicts each return with a + b · Return_(t-1), the least-squares AR(1) over the
        `window` pairs of returns before it. Both statistics are standard normal under the null hypothesis of
        no directional predictability; the p-value is one-sided.

        For more information about the methods, see the following papers:

        - Pesaran, M.H. and Timmermann, A. (1992). "A Simple Nonparametric Test of Predictive Performance."
        Journal of Business & Economic Statistics, 10(4), 461-465.
        - Anatolyev, S. and Gerko, A. (2005). "A Trading Approach to Testing for Predictability." Journal of
        Business & Economic Statistics, 23(4), 455-461.
        - Moskowitz, T.J., Ooi, Y.H. and Pedersen, L.H. (2012). "Time Series Momentum." Journal of Financial
        Economics, 104(2), 228-250.

        Also known as: directional accuracy test, market timing test, sign test, PT test, AG test, DAC test.

        Referenced against:
        - rugarch (R), DACTest: https://github.com/cran/rugarch/blob/1.5-6/R/rugarch-tests.R#L384

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly or yearly).
            Defaults to "daily".
            method (str, optional): The test. Options are: "pesaran_timmermann" (Pesaran-Timmermann),
            "anatolyev_gerko" (Anatolyev-Gerko). Defaults to "pesaran_timmermann".
            forecast (str, optional): The return forecast. Options are: "momentum" (the mean of the previous
            `window` returns), "autoregressive" (the rolling AR(1)). Defaults to "momentum".
            window (int | None, optional): The number of returns of every forecast window, at least 2.
            Defaults to None, which uses a year of observations with a minimum of 20 (252 daily, 52 weekly and 20 monthly,
            quarterly or yearly returns).
            include_benchmark (bool, optional): Whether to include "Benchmark" among the assets tested. Defaults
            to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The statistic, its one-sided p-value, the Directional Accuracy (the share of correctly
            predicted signs) and the number of observations, per asset.

        Notes:
        - Every forecast only uses the returns before it, and each asset uses its own return history.
        - A return or forecast of exactly zero counts as a wrong sign in P and as no position in A.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["QQQ", "TLT", "GLD", "USO"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2010-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_directional_accuracy_test(period="monthly", window=12)
        ```

        Which returns:

        |                              |      QQQ |      TLT |      GLD |      USO |
        |:-----------------------------|---------:|---------:|---------:|---------:|
        | Pesaran-Timmermann Statistic |   0.7683 |  -1.195  |   1.307  |  -0.9591 |
        | P-Value                      |   0.2211 |   0.8839 |   0.0956 |   0.8312 |
        | Directional Accuracy         |   0.6313 |   0.4581 |   0.5475 |   0.4637 |
        | Observations                 | 179      | 179      | 179      | 179      |
        """
        period = period if period else "daily"

        check_period(period, PERIODS)

        validate_window("window", window, 2)
        window = window if window is not None else max(PERIODS_PER_YEAR[period], 20)
        returns = self._filter_benchmark(
            self._historical_data[period]["Return"]
            .loc[self._start_date : self._end_date]
            .dropna(how="all"),
            include_benchmark=include_benchmark,
        )
        forecasts = forecast_evaluation_model.get_return_forecast(
            returns, forecast, window
        )
        result = forecast_evaluation_model.get_directional_accuracy_test(
            returns, forecasts, method
        )

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

    @handle_portfolio
    @handle_errors
    def get_model_confidence_set(
        self,
        period: str | None = None,
        models: list[str] | None = None,
        loss: str = "qlike",
        statistic: str = "max",
        alpha: float = 0.1,
        n_bootstrap: int = 1000,
        block_size: int | None = None,
        window: int = 22,
        lambda_: float = 0.94,
        garch_window: int | None = None,
        refit_every: int = 21,
        seed: int = 0,
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Model Confidence Set (MCS) of Hansen, Lunde and Nason: the volatility forecasting models
        that cannot be told apart from the best one.

        The Diebold-Mariano test compares two forecasts; the MCS compares any number of them. Every model forecasts
        the next period's variance of each asset from the returns before it, and its loss against the squared return
        is recorded. Starting from all models, the MCS tests whether they are equally good and, if not, eliminates
        the worst and tests again, until the remaining models cannot be told apart. The models left over form a set
        that contains the best model with a probability of at least 1 - alpha; the MCS p-value of a model says at
        which size it would leave the set.

        The formula is as follows (with the default max statistic, QLIKE loss and daily returns):

        - Loss_i,t = Return_t^2 / Forecast_i,t + ln(Forecast_i,t), or (Return_t^2 - Forecast_i,t)^2 with the squared loss
        - d_i = mean(Loss_i) - (1 / m) Σ_j mean(Loss_j) and t_i = d_i / √Var(d_i)
        - T = max_i t_i, with P-Value = (1 / B) Σ_b 1{T*_b >= T} over B stationary bootstrap resamples
        - while the p-value is below alpha, the model with the largest t_i is eliminated and the test repeated
        - MCS P-Value of a model = the largest p-value of the steps up to its elimination (1 for the last model)

        Where m is the number of models still in the set, Var(d_i) the bootstrap variance of d_i and T*_b the
        statistic of the b-th resample centered at the full sample (see
        `forecast_evaluation_model.get_model_confidence_set`, also for the range statistic, the largest |t_ij|
        of all pairs). The forecasts of the models are:

        - EWMA: the RiskMetrics variance λ·σₜ₋₁² + (1 - λ)·rₜ₋₁² with λ = `lambda_`
        - Rolling: the sample variance (ddof=1) of the `window` returns before t
        - GARCH, GJR-GARCH and the other models of `Risk.get_garch`: the one-step variance of the GARCH-family model
          with a constant mean and normal innovations (quasi-maximum likelihood), refitted every `refit_every`
          returns on the `garch_window` returns before t

        Every model is compared on the dates all of them have a forecast. The squared-error loss and QLIKE rank
        volatility forecasts correctly with the squared return as a noisy proxy of the true variance (Patton
        2011); QLIKE (written without the terms that do not depend on the forecast) is less sensitive to the
        largest returns.

        For more information about the method, see the following papers:

        - Hansen, P.R., Lunde, A. and Nason, J.M. (2011). "The Model Confidence Set." Econometrica, 79(2),
        453-497.
        - Patton, A.J. (2011). "Volatility Forecast Comparison Using Imperfect Volatility Proxies." Journal of
        Econometrics, 160(1), 246-256.

        Also known as: MCS, model confidence set, Hansen-Lunde-Nason test, forecast model selection.

        Referenced against:
        - arch, MCS: https://bashtage.github.io/arch/multiple-comparison/generated/arch.bootstrap.MCS.html
        - rugarch (R), mcsTest: https://github.com/cran/rugarch/blob/1.5-6/R/rugarch-tests.R#L919

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly or yearly).
            Defaults to "daily".
            models (list[str] | None, optional): The forecasting models to compare, at least two. Options are:
            "ewma" (RiskMetrics EWMA), "rolling" (rolling sample variance), "garch", "gjr", "egarch", "igarch",
            "aparch", "tgarch", "ngarch", "avgarch", "nagarch", "component", "figarch" (the GARCH-family models
            of `Risk.get_garch`). Defaults to None, which compares "ewma", "rolling", "garch" and "gjr".
            loss (str, optional): The loss of a forecast. Options are: "qlike" (QLIKE), "squared" (squared
            error). Defaults to "qlike". The squared
            losses of daily variances are of the order of 1e-7, so they need a larger `rounding` to show.
            statistic (str, optional): The test statistic. Options are: "max" (the largest t_i), "range" (the
            largest |t_ij|). Defaults to "max".
            alpha (float, optional): The size of the test. Defaults to 0.1, the 90% model confidence set.
            n_bootstrap (int, optional): The number of bootstrap resamples. Defaults to 1000.
            block_size (int | None, optional): The average block length of the stationary bootstrap. Defaults to
            None, which uses the cube root of the number of periods, rounded.
            window (int, optional): The window of the "rolling" model, at least 2. Defaults to 22 (about a
            month).
            lambda_ (float, optional): The decay factor of the "ewma" model. Defaults to 0.94.
            garch_window (int | None, optional): The number of returns of every GARCH fit, at least 100, as a
            GARCH fit on fewer returns is not reliable. Defaults to None, which uses two years of observations
            with a minimum of 100 (504 daily returns).
            refit_every (int, optional): The number of returns between two GARCH fits. Defaults to 21.
            seed (int, optional): The seed of the bootstrap. Defaults to 0.
            include_benchmark (bool, optional): Whether to include "Benchmark" among the assets. Defaults to
            False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per asset and model the Average Loss, the MCS P-Value, whether the model is In
            Confidence Set and its Rank (1 for the last model standing).

        Notes:
        - Every forecast only uses the returns before it, and each asset uses its own return history.
        - A GARCH fit takes one to five seconds per asset and every refit a fraction of a second.
        - In samples of a few hundred periods the range statistic eliminates a model too often (about 14% of
        the time at an alpha of 0.1 with four equally good models over 250 periods, against 8% for the max
        statistic), a property of the range statistic with the stationary bootstrap that arch's MCS shares;
        prefer the max statistic for short samples.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["QQQ", "TLT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2019-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_model_confidence_set()
        ```

        Which returns:

        |                    |   Average Loss |   MCS P-Value | In Confidence Set   |   Rank |
        |:-------------------|---------------:|--------------:|:--------------------|-------:|
        | ('QQQ', 'gjr')     |        -7.7142 |         1     | True                |      1 |
        | ('QQQ', 'garch')   |        -7.68   |         0.115 | True                |      2 |
        | ('QQQ', 'ewma')    |        -7.67   |         0.115 | True                |      3 |
        | ('QQQ', 'rolling') |        -7.6442 |         0.088 | False               |      4 |
        | ('TLT', 'ewma')    |        -8.2518 |         1     | True                |      1 |
        | ('TLT', 'gjr')     |        -8.2426 |         0.598 | True                |      2 |
        | ('TLT', 'garch')   |        -8.2414 |         0.598 | True                |      3 |
        | ('TLT', 'rolling') |        -8.2155 |         0.095 | False               |      4 |
        """
        validate_alpha(alpha)
        period = period if period else "daily"
        models = models if models else ["ewma", "rolling", "garch", "gjr"]
        models = [models] if isinstance(models, str) else list(models)

        check_period(period, PERIODS)
        check_option("loss", loss, ("squared", "qlike"))
        if len(set(models)) < 2:  # noqa: PLR2004
            raise ValueError(
                "The model confidence set compares at least two different models."
            )
        unknown = [
            model
            for model in models
            if model not in ("ewma", "rolling")
            and model not in VOLATILITY_GARCH_METHODS
        ]
        if unknown:
            raise InvalidArgumentError(
                f"Unknown models {', '.join(unknown)}: use ewma, rolling or the methods of "
                f"Risk.get_garch ({', '.join(VOLATILITY_GARCH_METHODS)})."
            )

        validate_window("window", window, 2)
        validate_window("garch_window", garch_window, MINIMUM_GARCH_WINDOW)
        garch_window = (
            garch_window if garch_window is not None else _garch_window_size(period)
        )
        returns = self._filter_benchmark(
            self._historical_data[period]["Return"]
            .loc[self._start_date : self._end_date]
            .dropna(how="all"),
            include_benchmark=include_benchmark,
        )

        results = {}
        for ticker in returns.columns:
            ticker_returns = returns[ticker].dropna()
            # A constant price has no variance to forecast, and every loss would be 0 / 0.
            if (
                len(ticker_returns) > 1
                and (ticker_returns == ticker_returns.iloc[0]).all()
            ):
                logger.warning(
                    "%s has constant returns, so its volatility forecasts have no losses to compare "
                    "and it has no model confidence set.",
                    ticker,
                )
                continue
            realized = ticker_returns**2
            forecasts = {
                model: forecast_evaluation_model.get_volatility_forecast(
                    ticker_returns,
                    model,
                    window,
                    lambda_,
                    garch_window,
                    refit_every,
                )
                for model in dict.fromkeys(models)
            }
            # A forecast of zero variance (before a model has returns to start from) has no
            # QLIKE loss (0 / 0 plus the logarithm of 0); those dates are dropped with the
            # warm-up of the other models.
            with np.errstate(divide="ignore", invalid="ignore"):
                losses = pd.DataFrame(
                    {
                        model: (
                            (realized - forecast) ** 2
                            if loss == "squared"
                            else realized / forecast + np.log(forecast)
                        )
                        for model, forecast in forecasts.items()
                    }
                ).dropna()

            if len(losses) < 2:  # noqa: PLR2004
                logger.warning(
                    "%s has too few returns for the forecasts of every model, so it has no model confidence set.",
                    ticker,
                )
                continue

            results[ticker] = forecast_evaluation_model.get_model_confidence_set(
                losses, alpha, statistic, n_bootstrap, block_size, seed
            )

        if not results:
            raise ValueError(
                "None of the assets has enough returns for the forecasts of every model."
            )

        result = pd.concat(results, names=["Ticker", "Model"])

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

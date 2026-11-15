"""Forecasting Module"""

__docformat__ = "google"

import pandas as pd

from financetoolkit.econometrics.base.base_controller import EconometricsBase
from financetoolkit.econometrics.forecasting import (
    forecast_evaluation_model,
    time_series_model,
)
from financetoolkit.helpers import handle_portfolio
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import handle_errors
from financetoolkit.utilities.logger_model import get_logger
from financetoolkit.utilities.sources_model import track_sources
from financetoolkit.utilities.statistics_model import apply_rounding

# ruff: noqa: E501

logger = get_logger()


@filter_shown_columns
@track_sources
class ForecastingEconometrics(EconometricsBase):
    """
    Forecasts and their evaluation: ARIMA, VAR and VECM forecasts, impulse responses, variance decomposition, RMSE, MAE, out-of-sample validation and the Diebold-Mariano test.
    """

    @handle_portfolio
    @handle_errors
    def get_diebold_mariano_test(
        self,
        period: str | None = None,
        method_a: str = "ewma",
        method_b: str = "rolling",
        window_size: int = 22,
        lambda_: float = 0.94,
        loss: str = "squared",
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Compare the forecast accuracy of two volatility forecasting methods against
        realized (squared return) Variance, per asset, via the Diebold-Mariano (1995)
        test.

        Rather than comparing two different assets, this compares two different ways
        of forecasting the *same* asset's next-day Variance -- e.g. the simpler
        exponentially weighted (EWMA, see `Risk.get_ewma_volatility`) approach against a
        plain rolling-window Standard Deviation -- to determine whether one is
        significantly more accurate than the other for a given asset.

        The squared return is a noisy stand-in for the true Variance. Patton (2011) shows
        that the squared-error loss still ranks two volatility forecasts correctly with such
        a proxy, while the absolute-error loss can rank them the wrong way round, so
        `loss="squared"` is the safer choice. A negative statistic favors `method_a`.

        The formula is as follows:

        - Realized Variance: rₜ²
        - EWMA forecast ("ewma"): σₜ² = λ·σₜ₋₁² + (1 - λ)·rₜ₋₁², with λ = `lambda_` (0.94 by default), so
          the forecast for t uses returns up to t - 1
        - Rolling forecast ("rolling"): the sample variance (ddof=1) of the `window_size` (22 by default)
          returns up to t - 1
        - dₜ = L(rₜ² - f_a,ₜ) - L(rₜ² - f_b,ₜ), with L(e) = e² (loss="squared") or |e| (loss="absolute")
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
            Defaults to "daily", since `window_size` is expressed in return observations of this
            frequency.
            method_a (str, optional): The first volatility forecasting method, one of "ewma" or "rolling".
            Defaults to "ewma".
            method_b (str, optional): The second (competing) volatility forecasting method, one of "ewma"
            or "rolling". Defaults to "rolling".
            window_size (int, optional): The rolling window size used by the "rolling" method. Defaults to
            22 (approximately one trading month).
            lambda_ (float, optional): The decay factor used by the "ewma" method. Defaults to 0.94.
            loss (str, optional): The loss function to compare forecast errors with, one of "squared" or
            "absolute". Defaults to "squared".
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
        if method_a not in ("ewma", "rolling") or method_b not in ("ewma", "rolling"):
            raise ValueError("method_a and method_b must be 'ewma' or 'rolling'.")

        period = period if period else "daily"

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )

        returns = self._filter_benchmark(
            self._historical_data[period]["Return"]
            .loc[self._start_date : self._end_date]
            .dropna(how="all"),
            include_benchmark=include_benchmark,
        )

        actual = returns**2
        forecast_a = forecast_evaluation_model.get_volatility_forecast(
            returns, method_a, window_size, lambda_
        )
        forecast_b = forecast_evaluation_model.get_volatility_forecast(
            returns, method_b, window_size, lambda_
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
        include_constant: bool = True,
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Fit an ARIMA(p, d, q) model to every ticker's price series in the Toolkit
        instance and forecast `forecast_steps` periods ahead.

        Also known as: Box-Jenkins model, autoregressive integrated moving average.

        An ARIMA(p, d, q) model differences the series `d` times to remove a
        (stochastic) trend, then fits an autoregressive-moving-average model to the
        result -- see `time_series_model.get_arima_forecast` for the full formula,
        estimation method (exact Maximum Likelihood via the Kalman filter) and its
        practical caveats.

        The formula is as follows:

        - y'ₜ = ΔᵈYₜ, the `column` level (Adj Close by default) differenced d times (d = 1 by default)
        - y'ₜ - μ = Σ(i=1..p) φᵢ·(y'ₜ₋ᵢ - μ) + εₜ + Σ(j=1..q) θⱼ·εₜ₋ⱼ, with p = q = 1 by default and μ the
          mean (d = 0) or drift (d ≥ 1) of y'ₜ when include_constant=True, and μ = 0 otherwise
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
            include_constant (bool, optional): Whether to estimate the mean of the
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
                    include_constant=include_constant,
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
        Fit a Vector Autoregression (VAR) across every ticker in the Toolkit instance and
        forecast `forecast_steps` periods ahead.

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
        `get_augmented_dickey_fuller`).

        For more information about the method, see the following paper:

        - Sims, C.A. (1980). "Macroeconomics and Reality." Econometrica, 48(1), 1-48.

        Also known as: VAR model, vector autoregressive model.

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
        Fit a Vector Autoregression (VAR) across every ticker in the Toolkit instance
        and trace out the Impulse Response Function (IRF) -- how a shock to each
        ticker propagates through the whole system over `periods` periods ahead. The
        shock is one standard deviation of the orthogonalized innovation when
        `orthogonalized=True` (the default) and a unit shock to the reduced-form
        residual when `orthogonalized=False`.

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
        - Θₙ = Ψₙ·P (orthogonalized=True, a one-standard-deviation shock) or Θₙ = Ψₙ
          (orthogonalized=False, a unit reduced-form shock)
        - Response of ticker i, n periods after a shock to ticker j: Θₙ[i, j], for n = 0..`periods`

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
        Fit a Vector Autoregression (VAR) across every ticker in the Toolkit instance
        and compute the (orthogonalized) Forecast Error Variance Decomposition (FEVD)
        -- what fraction of each ticker's `h`-step-ahead forecast error variance is
        attributable to each ticker's own structural shock, for `h = 1, ..., periods`.

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
        k_ar_diff: int = 1,
        forecast_steps: int = 5,
        significance: float = 0.05,
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Fit a Vector Error Correction Model (VECM) across every (cointegrated)
        ticker in the Toolkit instance and forecast `forecast_steps` periods ahead.

        Also known as: VECM, error correction model (for the multivariate/cointegrated
        case).

        A VECM keeps a VAR's short-run dynamics while ALSO letting each asset's price
        change react to how far the system currently sits from its long-run
        equilibrium (the cointegrating relationship(s) among the tickers, taken from
        `cointegration_model.get_johansen_cointegration`) -- see
        `time_series_model.get_vecm_forecast` for the full formula, estimation method
        and verification notes.

        The formula is as follows:

        - Rank r = the number of leading Johansen trace statistics (det_order=0, `k_ar_diff` lags) that
          exceed their critical value at `significance` (5% by default); r = 0 raises an error
        - ΔYₜ = α·β'·Yₜ₋₁ + Σ(i=1..k) Γᵢ·ΔYₜ₋ᵢ + c + εₜ, with Yₜ every ticker's `column` level (Adj Close by
          default), k = `k_ar_diff` (1 by default) and c a constant outside the cointegrating relation
          (statsmodels VECM with deterministic="co"), fitted by Maximum Likelihood
        - Forecast: the fitted recursion run forward `forecast_steps` periods, in levels

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily".
            column (str, optional): The historical data column to model. Defaults to
            "Adj Close" -- a VECM needs price LEVELS (non-stationary, cointegrated
            series), not returns, the same input `get_johansen_cointegration` expects.
            k_ar_diff (int, optional): The number of lagged first differences to
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
            k_ar_diff=k_ar_diff,
            forecast_steps=forecast_steps,
            significance=significance,
        )

        return result["forecast"].pipe(
            apply_rounding, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_rmse(
        self,
        period: str | None = None,
        column: str = "Return",
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series:
        """
        Calculate the Root Mean Squared Error (RMSE) between every unordered pair
        of tickers' series in the Toolkit instance.

        Also known as: RMSD (Root Mean Squared Deviation).

        This controller method compares two ASSETS' series directly, treating one as a naive
        "forecast" proxy for the other -- a simple, tracking-error style measure of
        how closely two series move together in absolute deviation terms (e.g. a
        portfolio versus a benchmark, or one asset as a naive stand-in forecast for
        a similar one). For evaluating an actual FORECASTING MODEL (ARIMA/VAR)
        rather than one asset as a naive proxy for another, use
        `get_out_of_sample_validation` instead.

        For a fund against its index the RMSE comes close to the tracking error but is not
        the same: the tracking error (`Performance.get_tracking_error`) is the Standard
        Deviation of the return differences, which leaves out their average, while the RMSE
        includes it. The RMSE is per period, so a daily figure is usually multiplied by the
        square root of 252 to annualize it.

        The formula is as follows:

        - RMSE = √((1 / n)·Σ(aₜ - bₜ)²), with aₜ and bₜ the two tickers' `column` (Return by default) on
          the n dates where both have data

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily".
            column (str, optional): The historical data column to compare. Defaults to
            "Return".
            include_benchmark (bool, optional): Whether to include "Benchmark" among the
            tickers paired up. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the
            results to. Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.Series: The RMSE, indexed by unordered `(Ticker A, Ticker B)` pair.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["IVV", "VOO"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_rmse(include_benchmark=True)
        ```

        Which returns:

        |                      |   RMSE |
        |:---------------------|-------:|
        | ('IVV', 'VOO')       | 0.0004 |
        | ('IVV', 'Benchmark') | 0.0009 |
        | ('VOO', 'Benchmark') | 0.001  |
        """
        period = period if period else "daily"
        returns = self._get_series_column(period, column)

        values = {
            (ticker_a, ticker_b): forecast_evaluation_model.get_rmse(
                returns[ticker_a], returns[ticker_b]
            )
            for ticker_a, ticker_b in self._all_unordered_pairs(
                include_benchmark=include_benchmark
            )
        }

        if not values:
            raise ValueError(
                "The RMSE compares two series, a forecast and the actual values, so it "
                "needs at least two tickers (or include_benchmark=True for the benchmark)."
            )

        result = pd.Series(values, name="RMSE")
        result.index = result.index.set_names(["Ticker A", "Ticker B"])

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_mae(
        self,
        period: str | None = None,
        column: str = "Return",
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series:
        """
        Calculate the Mean Absolute Error (MAE) between every unordered pair of tickers'
        series in the Toolkit instance.

        MAE is the average absolute difference between two series.

        The formula is as follows:

        - MAE = (1 / n)·Σ|aₜ - bₜ|, with aₜ and bₜ the two tickers' `column` (Return by default) on the
          n dates where both have data

        It is expressed in the units of the series itself (e.g. a return difference) and
        answers "how far apart are the two series in a typical period". Every difference counts
        in proportion to its size, so unlike `get_rmse`, which squares the differences and is
        therefore dominated by a few large gaps, MAE is robust to occasional outliers. A
        lower value means the two tickers track each other more closely.

        This method compares two assets directly, for example how closely a fund tracks its
        index or how two related stocks move together, rather than an asset against a model's
        forecast; see `get_rmse` for the reasoning, and `get_out_of_sample_validation` for
        evaluating actual forecasts.

        Also known as: MAD (Mean Absolute Deviation).

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily".
            column (str, optional): The historical data column to compare. Defaults to
            "Return".
            include_benchmark (bool, optional): Whether to include "Benchmark" among the
            tickers paired up. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the
            results to. Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.Series: The MAE, indexed by unordered `(Ticker A, Ticker B)` pair.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["GLD", "IAU", "GLDM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_mae()
        ```

        Which returns:

        |                 |    MAE |
        |:----------------|-------:|
        | ('GLD', 'IAU')  | 0.0002 |
        | ('GLD', 'GLDM') | 0.0003 |
        | ('IAU', 'GLDM') | 0.0003 |
        """
        period = period if period else "daily"
        returns = self._get_series_column(period, column)

        values = {
            (ticker_a, ticker_b): forecast_evaluation_model.get_mae(
                returns[ticker_a], returns[ticker_b]
            )
            for ticker_a, ticker_b in self._all_unordered_pairs(
                include_benchmark=include_benchmark
            )
        }

        if not values:
            raise ValueError(
                "The MAE compares two series, a forecast and the actual values, so it "
                "needs at least two tickers (or include_benchmark=True for the benchmark)."
            )

        result = pd.Series(values, name="MAE")
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
        include_constant: bool = True,
        lags: int = 1,
        other_tickers: list[str] | None = None,
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Out-of-sample validate an ARIMA or VAR forecast of every ticker in the
        Toolkit instance -- fit only on a training portion of the history,
        forecast the held-out remainder, and score the forecast against what
        actually happened.

        Also known as: hold-out validation, train/test split validation.

        See `forecast_evaluation_model.get_out_of_sample_validation` for the general
        harness this wraps. Since that function takes a raw Python callable (not
        serializable for e.g. the MCP-facing tool layer), this controller method
        instead hardcodes the choice between the two Part-1 forecasting models via the
        `model` string:

        - `model="arima"`: `time_series_model.get_arima_forecast` is fit on each
          ticker's own training-period series (`p`, `d`, `q`, `include_constant`
          control the model, same as `get_arima_forecast`).
        - `model="var"`: `time_series_model.get_var_forecast` is fit on the
          training-period series of each ticker together with `other_tickers`
          (`lags` controls the VAR order, defaulting to every other ticker in the
          Toolkit instance if not given); only that ticker's own forecast column
          is scored against its holdout.

        The formula is as follows:

        - Training sample: the first int(`train_fraction`·n) observations (80% by default) of the
          `column` level (Adj Close by default); the remaining H observations are the holdout
        - Ŷₜ: the H-step forecast of the model fitted on the training sample only, an ARIMA(p, d, q)
          (model="arima", ARIMA(1, 1, 1) by default) or a VAR(`lags`) on the ticker and `other_tickers`
          (model="var")
        - RMSE = √((1 / H)·Σ(Yₜ - Ŷₜ)²) and MAE = (1 / H)·Σ|Yₜ - Ŷₜ|, over the holdout

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily".
            column (str, optional): The historical data column to validate. Defaults
            to "Adj Close".
            model (str, optional): Either "arima" or "var". Defaults to "arima".
            train_fraction (float, optional): The fraction of observations used for
            training; the remainder is the holdout. Defaults to 0.8.
            p (int, optional): The ARIMA autoregressive order (`model="arima"` only).
            Defaults to 1.
            d (int, optional): The ARIMA differencing order (`model="arima"` only).
            Defaults to 1.
            q (int, optional): The ARIMA moving-average order (`model="arima"` only).
            Defaults to 1.
            include_constant (bool, optional): Whether the ARIMA model estimates a
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

        if model not in ("arima", "var"):
            raise ValueError(f"model must be 'arima' or 'var', received '{model}'.")

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
                            include_constant=include_constant,
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
                raise ValueError(
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

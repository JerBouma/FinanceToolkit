"""Forecast Evaluation Model"""

__docformat__ = "google"

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

from financetoolkit.portfolio.construction.optimization import estimation_model
from financetoolkit.risk import risk_model
from financetoolkit.risk.volatility import garch_model
from financetoolkit.utilities.statistics_model import newey_west_long_run_variance

MINIMUM_OBSERVATIONS_FOR_DM_TEST = 3

# The exact AR(1) likelihood of the Berkowitz test has three parameters.
MINIMUM_OBSERVATIONS_FOR_BERKOWITZ_TEST = 10

# A probability integral transform of exactly 0 or 1 has an infinite normal quantile.
PIT_BOUND = 1e-10

# The Pesaran-Timmermann and Anatolyev-Gerko statistics need a few signs of each kind.
MINIMUM_OBSERVATIONS_FOR_DIRECTIONAL_TEST = 3

# The model confidence set compares at least two models.
MINIMUM_MODELS_FOR_MCS = 2


def get_diebold_mariano_test(
    actual: pd.Series | pd.DataFrame,
    forecast_a: pd.Series | pd.DataFrame,
    forecast_b: pd.Series | pd.DataFrame,
    loss: str = "squared",
    horizon: int = 1,
    small_sample_correction: bool = True,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Diebold-Mariano (1995) test for comparing the forecast accuracy of
    two competing forecasts (e.g. two different VaR, Volatility or GARCH models)
    against the same realized values.

    Neither `statsmodels` nor `linearmodels` ships a Diebold-Mariano implementation --
    along with `unitroot_model.get_phillips_perron_test` and
    `diagnostics_model.get_variance_ratio_test`, this is one of the few tests in this
    module that remains hand-built.

    The test compares the loss differential between the two forecasts' errors:

    - e_a_t = actual_t - forecast_a_t,  e_b_t = actual_t - forecast_b_t
    - d_t = L(e_a_t) - L(e_b_t)

    Where `L` is a loss function (squared or absolute error). If both forecasts are
    equally accurate, `d_t` should average out to zero; if `forecast_a` is
    systematically more accurate, `d_t` should be negative on average (and vice
    versa). The test statistic standardizes the mean loss differential by its long-run
    variance, which accounts for the serial correlation that `d_t` has for
    multi-step-ahead (`horizon > 1`) forecasts, up to lag `horizon - 1`:

    - LRV(d_t) = gamma_0 + 2 * SUM_{k=1}^{horizon - 1} gamma_k
    - DM = mean(d_t) / SQRT(LRV(d_t) / n)

    Where gamma_k is the k-th sample autocovariance of `d_t` (divided by n). These are
    the equal (rectangular) weights of Diebold & Mariano (1995), for which Harvey,
    Leybourne & Newbold (1997) derive their correction. When this estimate is zero or
    negative, which can happen in a small sample, the Newey-West (Bartlett) weights
    1 - k / horizon of the same lags are used instead, as they are never negative.

    Which is asymptotically standard normal under the null of equal forecast
    accuracy, allowing for a two-sided p-value. When `small_sample_correction=True`
    (the default), the Harvey, Leybourne & Newbold (1997) finite-sample correction is
    applied -- rescaling the statistic and comparing it to a Student-T(n - 1)
    distribution instead of the normal -- since the plain asymptotic DM test is known
    to over-reject (find spurious significance) in small samples.

    A significant result (low p-value) indicates one forecast is significantly more
    accurate than the other; a negative statistic favors `forecast_a`, a positive
    statistic favors `forecast_b`.

    For more information about the method, see the following papers:

    - Diebold, F.X., & Mariano, R.S. (1995). "Comparing Predictive Accuracy." Journal
    of Business & Economic Statistics, 13(3), 253-263.
    - Harvey, D., Leybourne, S., & Newbold, P. (1997). "Testing the Equality of
    Prediction Mean Squared Errors." International Journal of Forecasting, 13(2),
    281-291.

    Also known as: DM test, forecast comparison test.

    Args:
        actual (pd.Series | pd.DataFrame): A Series or Dataframe of realized values.
        forecast_a (pd.Series | pd.DataFrame): A Series or Dataframe of the first
        forecast, aligned to the same index as `actual`.
        forecast_b (pd.Series | pd.DataFrame): A Series or Dataframe of the second
        (competing) forecast, aligned to the same index as `actual`.
        loss (str, optional): The loss function to compare forecast errors with, one
        of "squared" or "absolute". Defaults to "squared".
        horizon (int, optional): The forecast horizon (in periods), which sets the number
        of autocovariances in the long-run variance (`horizon - 1`) since an
        `h`-step-ahead forecast error is autocorrelated up to lag `h - 1`. Defaults to 1
        (one-step-ahead forecasts, no autocorrelation correction needed).
        small_sample_correction (bool, optional): Whether to apply the Harvey,
        Leybourne & Newbold (1997) finite-sample correction. Defaults to True.

    Returns:
        pd.Series | pd.DataFrame: The Diebold-Mariano statistic, its p-value, the mean
        loss differential (negative favors `forecast_a`) and the number of observations
        used.

    Raises:
        ValueError: If `loss` is not one of "squared" or "absolute".
    """
    if loss not in ("squared", "absolute"):
        raise ValueError("loss must be 'squared' or 'absolute'.")

    if isinstance(actual, pd.DataFrame):
        return pd.DataFrame(
            {
                column: get_diebold_mariano_test(
                    actual[column],
                    forecast_a[column],
                    forecast_b[column],
                    loss,
                    horizon,
                    small_sample_correction,
                )
                for column in actual.columns
            }
        )
    if isinstance(actual, pd.Series):
        aligned = pd.concat(
            [actual, forecast_a, forecast_b], axis=1, join="inner"
        ).dropna()
        aligned.columns = ["actual", "forecast_a", "forecast_b"]
        n = len(aligned)

        if n <= MINIMUM_OBSERVATIONS_FOR_DM_TEST:
            return pd.Series(
                {
                    "Diebold-Mariano Statistic": np.nan,
                    "P-Value": np.nan,
                    "Mean Loss Differential": np.nan,
                    "Observations": n,
                }
            )

        error_a = aligned["actual"] - aligned["forecast_a"]
        error_b = aligned["actual"] - aligned["forecast_b"]

        if loss == "squared":
            loss_a, loss_b = error_a**2, error_b**2
        else:
            loss_a, loss_b = error_a.abs(), error_b.abs()

        d = (loss_a - loss_b).to_numpy()
        d_bar = d.mean()

        # Diebold & Mariano (1995) weigh the first h - 1 autocovariances fully (a rectangular
        # window): an h-step forecast error is MA(h - 1), so these are exactly the nonzero
        # ones, and Harvey, Leybourne & Newbold (1997) derive their correction for this
        # estimator. It can be negative in a small sample; the Bartlett (Newey-West) weights
        # of the same lags are positive semi-definite and take over then.
        lags = max(horizon - 1, 0)
        demeaned = d - d_bar
        long_run_variance = float(np.sum(demeaned**2) / n) + 2 * sum(
            float(np.sum(demeaned[lag:] * demeaned[:-lag]) / n)
            for lag in range(1, lags + 1)
        )
        if long_run_variance <= 0:
            long_run_variance = newey_west_long_run_variance(demeaned, lags)

        if long_run_variance <= 0:
            return pd.Series(
                {
                    "Diebold-Mariano Statistic": np.nan,
                    "P-Value": np.nan,
                    "Mean Loss Differential": d_bar,
                    "Observations": n,
                }
            )

        dm_statistic = d_bar / np.sqrt(long_run_variance / n)

        if small_sample_correction:
            correction = np.sqrt(
                (n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n
            )
            dm_statistic *= correction
            p_value = 2 * stats.t.sf(abs(dm_statistic), n - 1)
        else:
            p_value = 2 * stats.norm.sf(abs(dm_statistic))

        return pd.Series(
            {
                "Diebold-Mariano Statistic": dm_statistic,
                "P-Value": p_value,
                "Mean Loss Differential": d_bar,
                "Observations": n,
            }
        )

    raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")


def get_volatility_forecast(
    returns: pd.Series | pd.DataFrame,
    method: str,
    window_size: int,
    lambda_: float,
    garch_window_size: int = 504,
    refit_every: int = 21,
) -> pd.Series | pd.DataFrame:
    """
    One-period-ahead Variance forecast built from the returns up to the prior period
    only, so it can be compared to the realized squared return out-of-sample -- the
    forecast half of the Diebold-Mariano test in `get_diebold_mariano_test`.

    - "ewma": σₜ² = λ·σₜ₋₁² + (1 - λ)·rₜ₋₁², the RiskMetrics recursion of
      `risk_model.get_ewma_volatility`, which already uses returns up to t - 1
    - "rolling": the sample variance (ddof=1) of the `window_size` returns up to t - 1
    - a GARCH-family method ("garch", "gjr", ...): the one-step variance of the model with a
      constant mean and normal innovations (quasi-maximum likelihood), refitted every
      `refit_every` returns on the `garch_window_size` returns before it (see
      `garch_model.get_garch_family_rolling_forecast`)

    Args:
        returns (pd.Series | pd.DataFrame): The asset's return series, or one column per asset.
        Each column is forecast on its own non-missing observations.
        method (str): The forecasting method, "ewma", "rolling" or a GARCH-family method.
        window_size (int): The rolling window size used by the "rolling" method.
        lambda_ (float): The decay factor used by the "ewma" method.
        garch_window_size (int, optional): The number of returns of every GARCH fit.
            Defaults to 504.
        refit_every (int, optional): The number of returns between two GARCH fits.
            Defaults to 21.

    Returns:
        pd.Series | pd.DataFrame: The Variance forecast for each period, shaped like `returns`.
    """
    if isinstance(returns, pd.DataFrame):
        return pd.DataFrame(
            {
                column: get_volatility_forecast(
                    returns[column],
                    method,
                    window_size,
                    lambda_,
                    garch_window_size,
                    refit_every,
                )
                for column in returns.columns
            }
        )

    clean_returns = returns.dropna()

    if method == "ewma":
        return risk_model.get_ewma_volatility(clean_returns, lambda_) ** 2

    if method in garch_model.GARCH_FAMILY_VARIANCE_PARAMETERS:
        if len(clean_returns) < garch_window_size:
            return pd.Series(np.nan, index=clean_returns.index)
        # The forecast made at t - 1 is dated by its last return, so it is shifted to t.
        return (
            garch_model.get_garch_family_rolling_forecast(
                clean_returns,
                garch_window_size,
                refit_every,
                method,
                "normal",
                "constant",
            )["Volatility"]
            ** 2
        ).shift(1)

    # The window ending at t includes rₜ itself, so it is shifted to end at t - 1.
    return clean_returns.rolling(window=window_size).var().shift(1)


def get_rmse(
    actual: pd.Series | pd.DataFrame, forecast: pd.Series | pd.DataFrame
) -> float | pd.Series:
    """
    Calculate the Root Mean Squared Error (RMSE) between realized and forecast values.

    Also known as: RMSD (Root Mean Squared Deviation).

    The most common scale-dependent forecast accuracy metric -- the square root of the
    average squared forecast error:

    - RMSE = SQRT(mean((actual_t - forecast_t)^2))

    Squaring the errors before averaging means RMSE penalizes large errors
    disproportionately more than small ones (compare `get_mae`, which weighs every
    error linearly) -- prefer RMSE when large misses are especially costly, and MAE
    when a single big outlier forecast shouldn't dominate the comparison.

    Args:
        actual (pd.Series | pd.DataFrame): The realized values.
        forecast (pd.Series | pd.DataFrame): The forecast values, aligned to the same
        index as `actual`.

    Returns:
        float | pd.Series: The RMSE. A single float for `pd.Series` inputs, or a
        `pd.Series` of one RMSE per column for `pd.DataFrame` inputs.

    Raises:
        TypeError: If `actual` is not a `pd.Series` or `pd.DataFrame`.

    Notes:
        Referenced against statsmodels' `rmse`.

    As an example:

    ```python
    import pandas as pd
    from financetoolkit.econometrics.forecasting import forecast_evaluation_model

    actual = pd.Series([10.0, 12.0, 11.0, 13.0])
    forecast = pd.Series([11.0, 11.0, 13.0, 12.0])

    forecast_evaluation_model.get_rmse(actual, forecast)
    ```

    Which returns:

    ```
    1.3228756555322954
    ```
    """
    if isinstance(actual, pd.DataFrame):
        return pd.Series(
            {
                column: get_rmse(actual[column], forecast[column])
                for column in actual.columns
            }
        )
    if isinstance(actual, pd.Series):
        aligned = pd.concat([actual, forecast], axis=1, join="inner").dropna()
        aligned.columns = ["actual", "forecast"]
        errors = aligned["actual"].to_numpy() - aligned["forecast"].to_numpy()
        return float(np.sqrt(np.mean(errors**2)))

    raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")


def get_mae(
    actual: pd.Series | pd.DataFrame, forecast: pd.Series | pd.DataFrame
) -> float | pd.Series:
    """
    Calculate the Mean Absolute Error (MAE) between realized and forecast values.

    Also known as: MAD (Mean Absolute Deviation).

    A scale-dependent forecast accuracy metric -- the average absolute forecast error:

    - MAE = mean(|actual_t - forecast_t|)

    Unlike `get_rmse`, every error contributes linearly regardless of size, so MAE is
    more robust to (i.e. less dominated by) a handful of large forecast misses -- it
    answers "how far off is a typical forecast", whereas RMSE answers a
    large-error-weighted version of the same question.

    Args:
        actual (pd.Series | pd.DataFrame): The realized values.
        forecast (pd.Series | pd.DataFrame): The forecast values, aligned to the same
        index as `actual`.

    Returns:
        float | pd.Series: The MAE. A single float for `pd.Series` inputs, or a
        `pd.Series` of one MAE per column for `pd.DataFrame` inputs.

    Raises:
        TypeError: If `actual` is not a `pd.Series` or `pd.DataFrame`.

    Notes:
        Referenced against statsmodels' `meanabs`.

    As an example:

    ```python
    import pandas as pd
    from financetoolkit.econometrics.forecasting import forecast_evaluation_model

    actual = pd.Series([10.0, 12.0, 11.0, 13.0])
    forecast = pd.Series([11.0, 11.0, 13.0, 12.0])

    forecast_evaluation_model.get_mae(actual, forecast)
    ```

    Which returns:

    ```
    1.25
    ```
    """
    if isinstance(actual, pd.DataFrame):
        return pd.Series(
            {
                column: get_mae(actual[column], forecast[column])
                for column in actual.columns
            }
        )
    if isinstance(actual, pd.Series):
        aligned = pd.concat([actual, forecast], axis=1, join="inner").dropna()
        aligned.columns = ["actual", "forecast"]
        errors = aligned["actual"].to_numpy() - aligned["forecast"].to_numpy()
        return float(np.mean(np.abs(errors)))

    raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")


def get_out_of_sample_validation(
    series: pd.Series,
    forecast_function: Callable,
    train_fraction: float = 0.8,
    **forecast_kwargs,
) -> pd.Series:
    """
    Split `series` into a training and holdout portion, forecast the holdout with
    `forecast_function` fit only on the training portion, and score the result with
    `get_rmse`/`get_mae`.

    Also known as: hold-out validation, train/test split validation.

    A model can fit its own training sample well while still being a poor forecaster
    -- out-of-sample validation guards against that by refitting on data the model
    (and the fit itself) never saw, and comparing its forecast of the held-out tail
    against what actually happened:

    1. Split `series` (chronologically -- no shuffling, this is time series data) into
       the first `train_fraction` of observations (training) and the remaining
       `1 - train_fraction` (holdout).
    2. Call `forecast_function(train_series, forecast_steps=len(holdout),
       **forecast_kwargs)` -- designed to accept `time_series_model.get_arima_forecast`
       or `time_series_model.get_var_forecast` directly (or any function with a
       compatible signature that returns either a result dict with a `"forecast"`
       key, such as those two return, or a plain `pd.Series`/single-column
       `pd.DataFrame`/array-like of forecast values).
    3. Compare that forecast against the actual holdout values via `get_rmse`/`get_mae`.

    Args:
        series (pd.Series): The full series to validate against, chronologically
        ordered.
        forecast_function (Callable): A function called as
        `forecast_function(train_series, forecast_steps=h, **forecast_kwargs)`,
        returning `h` forecast values (directly, or via a `"forecast"` dict key).
        train_fraction (float, optional): The fraction of (non-missing) observations
        used for training; the remainder is the holdout. Defaults to 0.8.
        **forecast_kwargs: Additional keyword arguments passed through to
        `forecast_function` (e.g. `p=1, d=1, q=1` for `get_arima_forecast`).

    Returns:
        pd.Series: `RMSE`, `MAE` and `Holdout Observations` (the number of holdout
        points the forecast was scored against).

    Raises:
        TypeError: If `series` is not a `pd.Series`.
        ValueError: If `train_fraction` is not strictly between 0 and 1, the split
        leaves an empty training or holdout portion, `forecast_function` returns a
        multi-column forecast, or the number of forecast values returned does not
        match the holdout length.

    As an example:

    ```python
    import numpy as np
    import pandas as pd
    from financetoolkit.econometrics.forecasting import forecast_evaluation_model, time_series_model

    rng = np.random.default_rng(1)
    e = rng.standard_normal(200)
    y = np.zeros(200)
    for t in range(1, 200):
        y[t] = 0.6 * y[t - 1] + e[t]

    series = pd.Series(y)

    forecast_evaluation_model.get_out_of_sample_validation(
        series, time_series_model.get_arima_forecast, train_fraction=0.9, p=1, d=0, q=0
    )
    ```
    """
    if not isinstance(series, pd.Series):
        raise TypeError(
            f"series must be a pd.Series, received {type(series).__name__}."
        )
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be strictly between 0 and 1.")

    clean_series = series.dropna()
    split = int(len(clean_series) * train_fraction)

    if split < 1 or split >= len(clean_series):
        raise ValueError(
            "train_fraction leaves no training observations or no holdout "
            "observations -- choose a value strictly between 0 and 1 that splits the "
            f"{len(clean_series)} available observations into two non-empty parts."
        )

    train = clean_series.iloc[:split]
    holdout = clean_series.iloc[split:]

    raw_forecast = forecast_function(
        train, forecast_steps=len(holdout), **forecast_kwargs
    )
    forecast_values = (
        raw_forecast["forecast"] if isinstance(raw_forecast, dict) else raw_forecast
    )

    if isinstance(forecast_values, pd.DataFrame):
        if forecast_values.shape[1] != 1:
            raise ValueError(
                "forecast_function returned a multi-column forecast -- "
                "get_out_of_sample_validation compares a single series and expects a "
                "one-dimensional forecast (a pd.Series, or a single-column pd.DataFrame)."
            )
        forecast_values = forecast_values.iloc[:, 0]

    forecast_array = np.asarray(forecast_values, dtype=float).reshape(-1)
    if len(forecast_array) != len(holdout):
        raise ValueError(
            f"forecast_function returned {len(forecast_array)} forecast value(s), "
            f"expected {len(holdout)} to match the holdout length."
        )

    aligned_forecast = pd.Series(forecast_array, index=holdout.index)

    return pd.Series(
        {
            "RMSE": get_rmse(holdout, aligned_forecast),
            "MAE": get_mae(holdout, aligned_forecast),
            "Holdout Observations": len(holdout),
        }
    )


def get_probability_integral_transform(
    returns: pd.Series | pd.DataFrame,
    model: str = "garch",
    window_size: int = 504,
    lambda_: float = 0.94,
    refit_every: int = 21,
    garch_method: str = "garch",
    garch_distribution: str = "normal",
) -> pd.Series | pd.DataFrame:
    """
    Calculate the probability integral transform (PIT) of every return under a one-period-ahead
    density forecast built from the returns before it, the input of density forecast tests such
    as `get_berkowitz_test`.

    The PIT is the forecast distribution function evaluated at the realized return. If the
    density forecasts are correct, the PITs are independent and uniformly distributed on (0, 1)
    (Diebold, Gunther and Tay 1998):

    - PIT_t = F_t(Return_t), with F_t the forecast distribution of Return_t given the returns before t

    The density forecasts are:

    - "gaussian": a normal distribution with the mean and standard deviation (ddof=1) of the
      `window_size` returns before t
    - "ewma": a normal distribution with mean zero and the RiskMetrics variance
      σₜ² = λ·σₜ₋₁² + (1 - λ)·rₜ₋₁² (see `risk_model.get_ewma_volatility`)
    - "garch": the GARCH-family model `garch_method` with a constant mean and innovations
      `garch_distribution`, refitted every `refit_every` returns on the `window_size` returns
      before t, so F_t(r) = F((r - Mu) / Volatility_t) with F the distribution function of the
      standardized innovations (see `garch_model.get_garch_family_rolling_forecast` and
      `garch_model.get_standardized_cdf`)

    For more information, see the following paper:

    - Diebold, F.X., Gunther, T.A. and Tay, A.S. (1998). "Evaluating Density Forecasts with
    Applications to Financial Risk Management." International Economic Review, 39(4), 863-883.

    Args:
        returns (pd.Series | pd.DataFrame): The returns, one column per asset. Each column is
            forecast on its own non-missing observations.
        model (str, optional): "gaussian", "ewma" or "garch". Defaults to "garch".
        window_size (int, optional): The number of returns of every window or GARCH fit.
            Defaults to 504.
        lambda_ (float, optional): The decay factor of "ewma". Defaults to 0.94.
        refit_every (int, optional): The number of returns between two GARCH fits. Defaults
            to 21.
        garch_method (str, optional): The GARCH variance model. Defaults to "garch".
        garch_distribution (str, optional): The GARCH innovation distribution. Defaults to
            "normal".

    Returns:
        pd.Series | pd.DataFrame: The PIT of every return with a forecast, NaN before the first.
    """
    if isinstance(returns, pd.DataFrame):
        return pd.DataFrame(
            {
                column: get_probability_integral_transform(
                    returns[column],
                    model,
                    window_size,
                    lambda_,
                    refit_every,
                    garch_method,
                    garch_distribution,
                )
                for column in returns.columns
            },
            index=returns.index,
        )

    clean_returns = returns.dropna()

    if model == "gaussian":
        window = clean_returns.rolling(window=window_size)
        return pd.Series(
            special.ndtr(
                (clean_returns - window.mean().shift(1)) / window.std(ddof=1).shift(1)
            ),
            index=clean_returns.index,
        )
    if model == "ewma":
        return pd.Series(
            special.ndtr(
                clean_returns / risk_model.get_ewma_volatility(clean_returns, lambda_)
            ),
            index=clean_returns.index,
        )
    if model != "garch":
        raise ValueError("The model must be gaussian, ewma or garch.")

    if len(clean_returns) < window_size:
        return pd.Series(np.nan, index=clean_returns.index)

    # The forecast made at t - 1 is dated by its last return, so it is shifted to t.
    forecast = garch_model.get_garch_family_rolling_forecast(
        clean_returns,
        window_size,
        refit_every,
        garch_method,
        garch_distribution,
        "constant",
    ).shift(1)
    forecast = forecast.reindex(clean_returns.index)
    standardized = (clean_returns - forecast["Mean"]) / forecast["Volatility"]
    parameter_names = list(forecast.columns[2:])
    pit = pd.Series(np.nan, index=clean_returns.index)

    # The distribution parameters change at a refit only, so the dates of each set are evaluated together.  # noqa: E501
    available = forecast[forecast["Volatility"].notna()]
    parameters = (
        list(available[parameter_names].itertuples(index=False, name=None))
        if parameter_names
        else [()] * len(available)
    )
    for values in set(parameters):
        dates = available.index[[row == values for row in parameters]]
        pit.loc[dates] = garch_model.get_standardized_cdf(
            standardized.loc[dates].to_numpy(), garch_distribution, values
        )

    return pit


def get_berkowitz_test(
    pit: pd.Series | pd.DataFrame, alpha: float | None = None
) -> pd.Series | pd.DataFrame:
    """
    Calculate Berkowitz's likelihood ratio test of density forecasts.

    If the density forecasts are correct, the probability integral transforms (PIT, see
    `get_probability_integral_transform`) are independent and uniform, so their normal
    quantiles z_t = Φ^-1(PIT_t) are independent standard normal. The test fits a Gaussian
    AR(1) to z by exact maximum likelihood and compares it with the standard normal, which
    tests the mean, the variance and the independence of the forecasts together:

    - z_t - μ = ρ (z_(t-1) - μ) + ε_t, with ε_t ~ N(0, σ^2)
    - ln L(μ, σ, ρ) = -ln(2π) T / 2 - ln(σ^2 / (1 - ρ^2)) / 2 - (z_1 - μ)^2 (1 - ρ^2) / (2 σ^2) - ln(σ^2) (T - 1) / 2
      - Σ_(t=2)^T (z_t - μ - ρ (z_(t-1) - μ))^2 / (2 σ^2)
    - Statistic = -2 (ln L(0, 1, 0) - ln L(μ_hat, σ_hat, ρ_hat)), compared with a χ² distribution with 3 degrees of
      freedom

    With `alpha`, the tail test of the paper only looks at the forecasts of the tail: every
    z_t above the cut-off q = Φ^-1(alpha) is censored at q, so a model is judged on the size
    and frequency of its tail events only, as a Value at Risk at `alpha` is:

    - ln L(μ, σ) = Σ_(z_t < q) (ln φ((z_t - μ) / σ) - ln σ) + Σ_(z_t >= q) ln(1 - Φ((q - μ) / σ))
    - Statistic = -2 (ln L(0, 1) - ln L(μ_hat, σ_hat)), compared with a χ² distribution with 2 degrees of freedom

    PITs are kept within [1e-10, 1 - 1e-10], so a return far outside its forecast gives a large
    but finite z.

    For more information about the method, see the following paper:

    - Berkowitz, J. (2001). "Testing Density Forecasts, With Applications to Risk Management."
    Journal of Business & Economic Statistics, 19(4), 465-474.

    Referenced against `BerkowitzTest` of R's rugarch (https://cran.r-project.org/package=rugarch),
    which fits the AR(1) by least squares on the demeaned z (the likelihood conditional on the
    first observation, with the residual standard deviation of the regression) where the test
    here maximizes the exact likelihood of the paper; its tail test is the same.

    Args:
        pit (pd.Series | pd.DataFrame): The probability integral transforms, one column per
            asset.
        alpha (float | None, optional): The tail probability of the tail test. Defaults to
            None, which tests the whole density.

    Returns:
        pd.Series | pd.DataFrame: The Berkowitz statistic, its p-value, the fitted Mean,
        Standard Deviation and Autocorrelation (NaN for the tail test) of z and the number of
        observations (and of tail observations).
    """
    if isinstance(pit, pd.DataFrame):
        return pd.DataFrame(
            {column: get_berkowitz_test(pit[column], alpha) for column in pit.columns}
        )
    if not isinstance(pit, pd.Series):
        raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")

    z = special.ndtri(
        np.clip(pit.dropna().to_numpy(dtype=float), PIT_BOUND, 1 - PIT_BOUND)
    )
    n = len(z)
    empty = {
        "Berkowitz Statistic": np.nan,
        "P-Value": np.nan,
        "Mean": np.nan,
        "Standard Deviation": np.nan,
        "Autocorrelation": np.nan,
        "Observations": n,
        "Tail Observations": np.nan,
    }

    if n < MINIMUM_OBSERVATIONS_FOR_BERKOWITZ_TEST:
        return pd.Series(empty)

    if alpha is not None:
        cutoff = float(special.ndtri(alpha))
        tail = z[z < cutoff]
        censored = int(np.sum(z >= cutoff))

        def tail_log_likelihood(mean: float, deviation: float) -> float:
            return float(
                np.sum(stats.norm.logpdf(tail, mean, deviation))
                + censored * stats.norm.logsf(cutoff, mean, deviation)
            )

        if len(tail) < 2:  # noqa: PLR2004
            return pd.Series({**empty, "Tail Observations": len(tail)})

        start = [float(tail.mean()), float(max(tail.std(), 0.1))]
        fit = optimize.minimize(
            lambda x: -tail_log_likelihood(x[0], np.exp(x[1])),
            [start[0], np.log(start[1])],
            method="Nelder-Mead",
            options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 10000},
        )
        mean, deviation = float(fit.x[0]), float(np.exp(fit.x[1]))
        statistic = max(
            -2 * (tail_log_likelihood(0.0, 1.0) - tail_log_likelihood(mean, deviation)),
            0.0,
        )

        return pd.Series(
            {
                "Berkowitz Statistic": statistic,
                "P-Value": stats.chi2.sf(statistic, 2),
                "Mean": mean,
                "Standard Deviation": deviation,
                "Autocorrelation": np.nan,
                "Observations": n,
                "Tail Observations": len(tail),
            }
        )

    def log_likelihood(mean: float, deviation: float, autocorrelation: float) -> float:
        variance = deviation**2
        first_variance = variance / (1 - autocorrelation**2)
        innovations = (z[1:] - mean) - autocorrelation * (z[:-1] - mean)
        return float(
            -0.5 * n * np.log(2 * np.pi)
            - 0.5 * np.log(first_variance)
            - (z[0] - mean) ** 2 / (2 * first_variance)
            - 0.5 * (n - 1) * np.log(variance)
            - np.sum(innovations**2) / (2 * variance)
        )

    # The least-squares AR(1), the likelihood conditional on the first observation, starts the search.  # noqa: E501
    slope, intercept = np.polyfit(z[:-1], z[1:], 1)
    slope = float(np.clip(slope, -0.99, 0.99))
    residuals = z[1:] - intercept - slope * z[:-1]
    start = [
        intercept / (1 - slope),
        np.log(max(float(residuals.std()), 1e-6)),
        np.arctanh(slope),
    ]
    fit = optimize.minimize(
        lambda x: -log_likelihood(x[0], np.exp(x[1]), np.tanh(x[2])),
        start,
        method="Nelder-Mead",
        options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20000},
    )
    mean, deviation, autocorrelation = (
        float(fit.x[0]),
        float(np.exp(fit.x[1])),
        float(np.tanh(fit.x[2])),
    )
    statistic = max(
        -2
        * (
            log_likelihood(0.0, 1.0, 0.0)
            - log_likelihood(mean, deviation, autocorrelation)
        ),
        0.0,
    )

    return pd.Series(
        {
            "Berkowitz Statistic": statistic,
            "P-Value": stats.chi2.sf(statistic, 3),
            "Mean": mean,
            "Standard Deviation": deviation,
            "Autocorrelation": autocorrelation,
            "Observations": n,
            "Tail Observations": np.nan,
        }
    )


def get_return_forecast(
    returns: pd.Series | pd.DataFrame,
    method: str = "momentum",
    window_size: int = 252,
) -> pd.Series | pd.DataFrame:
    """
    One-period-ahead return forecast built from the returns up to the prior period only, the
    forecast half of the directional accuracy tests in `get_directional_accuracy_test`.

    - "momentum": the mean of the `window_size` returns up to t - 1, whose sign is the time
      series momentum signal of Moskowitz, Ooi and Pedersen (2012)
    - "autoregressive": a + b·rₜ₋₁, with a and b the least-squares AR(1) of the returns on
      their previous return over the `window_size` pairs up to t - 1

    For more information, see the following paper:

    - Moskowitz, T.J., Ooi, Y.H. and Pedersen, L.H. (2012). "Time Series Momentum." Journal of
    Financial Economics, 104(2), 228-250.

    Args:
        returns (pd.Series | pd.DataFrame): The returns, one column per asset. Each column is
            forecast on its own non-missing observations.
        method (str, optional): "momentum" or "autoregressive". Defaults to "momentum".
        window_size (int, optional): The number of returns (or pairs of returns) of every
            window. Defaults to 252.

    Returns:
        pd.Series | pd.DataFrame: The forecast of every return, NaN before the first full window.
    """
    if isinstance(returns, pd.DataFrame):
        return pd.DataFrame(
            {
                column: get_return_forecast(returns[column], method, window_size)
                for column in returns.columns
            },
            index=returns.index,
        )

    clean_returns = returns.dropna()

    if method == "momentum":
        return clean_returns.rolling(window=window_size).mean().shift(1)
    if method != "autoregressive":
        raise ValueError("The method must be momentum or autoregressive.")

    previous = clean_returns.shift(1)
    slope = (
        clean_returns.rolling(window=window_size).cov(previous)
        / previous.rolling(window=window_size).var()
    )
    intercept = (
        clean_returns.rolling(window=window_size).mean()
        - slope * previous.rolling(window=window_size).mean()
    )

    # The fit over the pairs up to t - 1 forecasts rₜ from rₜ₋₁.
    return intercept.shift(1) + slope.shift(1) * previous


def get_directional_accuracy_test(
    actual: pd.Series | pd.DataFrame,
    forecast: pd.Series | pd.DataFrame,
    method: str = "pesaran_timmermann",
) -> pd.Series | pd.DataFrame:
    """
    Calculate a test of whether a forecast predicts the direction (the sign) of a series better
    than chance.

    The Pesaran-Timmermann test compares the share of correctly predicted signs with the share
    expected if the forecast and the actual signs were independent:

    - P = (1 / n) Σ 1{forecast_t · actual_t > 0}, P_y = (1 / n) Σ 1{actual_t > 0}, P_x = (1 / n) Σ 1{forecast_t > 0}
    - P* = P_y P_x + (1 - P_y) (1 - P_x)
    - V(P) = P* (1 - P*) / n
    - V(P*) = (2 P_y - 1)^2 P_x (1 - P_x) / n + (2 P_x - 1)^2 P_y (1 - P_y) / n + 4 P_y P_x (1 - P_y) (1 - P_x) / n^2
    - Statistic = (P - P*) / √(V(P) - V(P*))

    The Anatolyev-Gerko test weighs every correct or wrong sign by the size of the move: it is
    the excess return of a strategy that goes long when the forecast is positive and short when
    it is negative, over the return of a strategy with the same share of long positions held at
    random:

    - A = (1 / n) Σ sign(forecast_t) · actual_t
    - B = ((1 / n) Σ sign(forecast_t)) · ((1 / n) Σ actual_t)
    - p = (1 + (1 / n) Σ sign(forecast_t)) / 2
    - V = 4 p (1 - p) Σ (actual_t - mean(actual))^2 / n^2
    - Statistic = (A - B) / √V

    Both statistics are standard normal under the null hypothesis of no directional
    predictability, and the p-value is one-sided, P(N(0, 1) > Statistic), since only a forecast
    that is right more often than chance has value.

    For more information about the methods, see the following papers:

    - Pesaran, M.H. and Timmermann, A. (1992). "A Simple Nonparametric Test of Predictive
    Performance." Journal of Business & Economic Statistics, 10(4), 461-465.
    - Anatolyev, S. and Gerko, A. (2005). "A Trading Approach to Testing for Predictability."
    Journal of Business & Economic Statistics, 23(4), 455-461.

    Referenced against `DACTest` of R's rugarch (https://cran.r-project.org/package=rugarch).

    Args:
        actual (pd.Series | pd.DataFrame): The realized values, such as returns.
        forecast (pd.Series | pd.DataFrame): The forecasts, aligned to the same index.
        method (str, optional): "pesaran_timmermann" or "anatolyev_gerko". Defaults to
            "pesaran_timmermann".

    Returns:
        pd.Series | pd.DataFrame: The statistic, its one-sided p-value, the directional
        accuracy (the share of correctly predicted signs) and the number of observations.
    """
    if isinstance(actual, pd.DataFrame):
        return pd.DataFrame(
            {
                column: get_directional_accuracy_test(
                    actual[column], forecast[column], method
                )
                for column in actual.columns
            }
        )
    if not isinstance(actual, pd.Series):
        raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")
    if method not in ("pesaran_timmermann", "anatolyev_gerko"):
        raise ValueError("The method must be pesaran_timmermann or anatolyev_gerko.")

    name = (
        "Pesaran-Timmermann Statistic"
        if method == "pesaran_timmermann"
        else "Anatolyev-Gerko Statistic"
    )
    aligned = pd.concat([actual, forecast], axis=1, join="inner").dropna()
    y = aligned.iloc[:, 0].to_numpy(dtype=float)
    x = aligned.iloc[:, 1].to_numpy(dtype=float)
    n = len(y)
    accuracy = float(np.mean(x * y > 0)) if n else np.nan

    statistic = np.nan
    if n >= MINIMUM_OBSERVATIONS_FOR_DIRECTIONAL_TEST:
        with np.errstate(divide="ignore", invalid="ignore"):
            if method == "pesaran_timmermann":
                share_actual, share_forecast = np.mean(y > 0), np.mean(x > 0)
                expected = share_actual * share_forecast + (1 - share_actual) * (
                    1 - share_forecast
                )
                variance = expected * (1 - expected) / n - (
                    (2 * share_actual - 1) ** 2
                    * share_forecast
                    * (1 - share_forecast)
                    / n
                    + (2 * share_forecast - 1) ** 2
                    * share_actual
                    * (1 - share_actual)
                    / n
                    + 4
                    * share_actual
                    * share_forecast
                    * (1 - share_actual)
                    * (1 - share_forecast)
                    / n**2
                )
                statistic = (accuracy - expected) / np.sqrt(variance)
            else:
                signs = np.sign(x)
                excess = np.mean(signs * y) - np.mean(signs) * np.mean(y)
                share_long = 0.5 * (1 + np.mean(signs))
                variance = (
                    4
                    * share_long
                    * (1 - share_long)
                    * np.sum((y - y.mean()) ** 2)
                    / n**2
                )
                statistic = excess / np.sqrt(variance)
        statistic = float(statistic) if np.isfinite(statistic) else np.nan

    return pd.Series(
        {
            name: statistic,
            "P-Value": stats.norm.sf(statistic),
            "Directional Accuracy": accuracy,
            "Observations": n,
        }
    )


def get_model_confidence_set(
    losses: pd.DataFrame,
    alpha: float = 0.1,
    statistic: str = "max",
    n_bootstrap: int = 1000,
    block_size: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Calculate the Model Confidence Set (MCS) of Hansen, Lunde and Nason: the set of models that
    contains the best model (the one with the lowest expected loss) with a probability of at
    least 1 - alpha, found by eliminating the worst model until the remaining ones are equally
    good.

    Where the Diebold-Mariano test compares two forecasts, the MCS compares any number at once
    and keeps every model the data cannot tell apart from the best. Starting from all m models,
    with L_i,t the loss of model i at t and bars for averages over t:

    - d_ij = mean(L_i) - mean(L_j) and d_i = mean(L_i) - (1 / m) Σ_j mean(L_j)
    - t_ij = d_ij / √Var(d_ij) and t_i = d_i / √Var(d_i)
    - statistic="max": T = max_i t_i; statistic="range": T = max_(i, j) |t_ij|
    - P-Value = (1 / B) Σ_b 1{T*_b >= T}, with T*_b the statistic of the b-th bootstrap resample, centered at
      the full sample: t*_i = (d*_i - d_i) / √Var(d_i) and t*_ij = (d*_ij - d_ij) / √Var(d_ij)
    - when the p-value is below alpha, the model with the largest t_i ("max") or max_j t_ij ("range") is
      eliminated and the test is repeated on the remaining models

    The variances are the mean squared deviation of the bootstrap averages from the full
    sample, Var(d_i) = (1 / B) Σ_b (d*_i - d_i)^2, from B resamples of the periods drawn with the
    stationary bootstrap of Politis and Romano (1994) (blocks of a random length with mean
    `block_size`, see `estimation_model.get_bootstrap_indices`), which keeps the dependence of the
    losses over time. The same resamples serve every step. The MCS p-value of a model is the
    largest p-value of the steps up to its elimination, and the last model has an MCS p-value of
    1; the models with an MCS p-value of at least alpha form the confidence set.

    For more information about the method, see the following papers:

    - Hansen, P.R., Lunde, A. and Nason, J.M. (2011). "The Model Confidence Set." Econometrica,
    79(2), 453-497.
    - Politis, D.N. and Romano, J.P. (1994). "The Stationary Bootstrap." Journal of the
    American Statistical Association, 89(428), 1303-1313.

    Referenced against `MCS` of the arch package (https://arch.readthedocs.io), whose methods "max"
    and "R" are the statistics "max" and "range" here, and `mcsTest` of R's rugarch
    (https://cran.r-project.org/package=rugarch). Like arch, the variance of each t_ij of
    "range" is that of the bootstrap averages with 1 added to its zero diagonal.

    Args:
        losses (pd.DataFrame): The loss of every model (a column) at every period (a row), such
            as squared forecast errors. Periods with a missing loss are left out.
        alpha (float, optional): The size of the test: the set contains the best model with a
            probability of at least 1 - alpha. Defaults to 0.1.
        statistic (str, optional): "max" or "range". Defaults to "max".
        n_bootstrap (int, optional): The number of bootstrap resamples B. Defaults to 1000.
        block_size (int | None, optional): The average block length of the stationary
            bootstrap. Defaults to None, which uses the cube root of the number of periods,
            rounded.
        random_state (int, optional): The seed of the bootstrap. Defaults to 42.

    Returns:
        pd.DataFrame: Per model the Average Loss, the MCS P-Value, whether it is In Confidence
        Set and its Rank (1 for the last model standing, m for the first eliminated).
    """
    if statistic not in ("max", "range"):
        raise ValueError("The statistic must be max or range.")

    losses = losses.dropna()
    values = losses.to_numpy(dtype=float)
    n, m = values.shape

    if m < MINIMUM_MODELS_FOR_MCS or n < MINIMUM_MODELS_FOR_MCS:
        raise ValueError(
            "The model confidence set needs at least two models and two periods of losses."
        )

    block_size = block_size if block_size else max(int(round(n ** (1 / 3))), 1)
    rng = np.random.default_rng(random_state)
    counts = np.array(
        [
            np.bincount(
                estimation_model.get_bootstrap_indices(
                    n, "stationary", block_size, rng
                ),
                minlength=n,
            )
            for _ in range(n_bootstrap)
        ]
    )
    average = values.mean(axis=0)
    bootstrap_average = counts @ values / n

    remaining = list(range(m))
    eliminated = []

    while len(remaining) > 1:
        model_average = average[remaining]
        bootstrap_models = bootstrap_average[:, remaining]

        if statistic == "max":
            difference = model_average - model_average.mean()
            bootstrap_difference = (
                bootstrap_models - bootstrap_models.mean(axis=1, keepdims=True)
            ) - difference
            deviation = np.sqrt(np.mean(bootstrap_difference**2, axis=0))
            deviation = np.where(deviation > 0, deviation, np.finfo(float).eps)
            t_values = difference / deviation
            observed = t_values.max()
            simulated = (bootstrap_difference / deviation).max(axis=1)
            worst = int(np.argmax(t_values))
        else:
            difference = model_average[:, None] - model_average[None, :]
            bootstrap_difference = (
                bootstrap_models[:, :, None] - bootstrap_models[:, None, :]
            ) - difference
            deviation = np.sqrt(
                np.mean(bootstrap_difference**2, axis=0) + np.eye(len(remaining))
            )
            t_values = difference / deviation
            observed = np.abs(t_values).max()
            simulated = np.abs(bootstrap_difference / deviation).max(axis=(1, 2))
            worst = int(np.argmax(t_values.max(axis=1)))

        eliminated.append((remaining[worst], float(np.mean(simulated >= observed))))
        remaining.pop(worst)

    eliminated.append((remaining[0], 1.0))
    p_values = np.maximum.accumulate([p_value for _, p_value in eliminated])
    order = [model for model, _ in eliminated]

    result = pd.DataFrame(
        {
            "Average Loss": average[order],
            "MCS P-Value": p_values,
            "In Confidence Set": p_values >= alpha,
            "Rank": np.arange(m, 0, -1),
        },
        index=losses.columns[order],
    )

    return result.iloc[::-1]

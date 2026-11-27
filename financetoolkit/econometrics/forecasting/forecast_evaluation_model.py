"""Forecast Evaluation Model"""

__docformat__ = "google"

import functools
from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import integrate, optimize, special, stats

from financetoolkit.portfolio.construction.optimization import estimation_model
from financetoolkit.risk import risk_model
from financetoolkit.risk.volatility import garch_model
from financetoolkit.utilities.error_model import InvalidArgumentError, check_option
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

# The Hong-Li density estimate is integrated with an 8-point Gauss-Legendre rule on panels a fifth of
# the bandwidth wide (five panels in each boundary strip); 4 and 16 points per panel agree with it to
# 1e-6 of the statistic.
HONG_LI_NODES_PER_PANEL = 8
HONG_LI_PANELS_PER_BANDWIDTH = 5

# The Hong-Li test needs a few pairs per lag to estimate their joint density.
MINIMUM_OBSERVATIONS_FOR_HONG_LI_TEST = 20

# A bandwidth within rounding of zero (1e-14 per observation) means PITs without variation.
HONG_LI_ROUNDING = 1e-14

# The joint cumulants of the Hong-Li statistics Q(1), ..., Q(10) under the null, per number of
# observations: the mean κ₁, variance κ₂ and third cumulant κ₃ of one Q(j), the covariance κ₁₁
# of two, and the joint third cumulants κ₂₁ = E[(Q(i) - κ₁)² (Q(j) - κ₁)] and κ₁₁₁ (three
# different lags), in the order κ₁, κ₂, κ₁₁, κ₃, κ₂₁, κ₁₁₁. Each row averages 3,000 seeded
# simulations of independent uniform PITs (1,500 at 3,000 observations and 1,000 at 5,000)
# over the 10 lags and their pairs and triples, smoothed by a quadratic in ln n over the grid,
# weighted by the square root of the simulations (see `get_hong_li_p_value`). The simulations
# integrate with 4 rather than 8 Gauss-Legendre nodes per panel, which moves a statistic by
# less than 1e-4.
HONG_LI_NULL_GRID = np.array(
    [50, 100, 175, 250, 350, 500, 750, 1000, 1500, 2000, 3000, 5000]
)
HONG_LI_NULL_CUMULANTS = np.array(
    [
        [-0.0412, 1.5386, 0.8223, 1.7922, 1.2043, 1.1058],
        [-0.0238, 1.5496, 0.8036, 1.5328, 1.0492, 0.9896],
        [-0.0124, 1.5490, 0.7810, 1.3421, 0.9291, 0.8941],
        [-0.0065, 1.5442, 0.7630, 1.2294, 0.8549, 0.8323],
        [-0.0017, 1.5365, 0.7435, 1.1293, 0.7866, 0.7735],
        [0.0024, 1.5251, 0.7202, 1.0298, 0.7159, 0.7106],
        [0.0058, 1.5078, 0.6902, 0.9250, 0.6378, 0.6382],
        [0.0075, 1.4929, 0.6668, 0.8560, 0.5839, 0.5864],
        [0.0087, 1.4681, 0.6308, 0.7663, 0.5098, 0.5127],
        [0.0089, 1.4478, 0.6031, 0.7080, 0.4587, 0.4598],
        [0.0080, 1.4154, 0.5609, 0.6334, 0.3887, 0.3847],
        [0.0051, 1.3682, 0.5028, 0.5519, 0.3039, 0.2888],
    ]
)


def get_diebold_mariano_test(
    actual: pd.Series | pd.DataFrame,
    forecast_a: pd.Series | pd.DataFrame,
    forecast_b: pd.Series | pd.DataFrame,
    loss: str = "squared",
    horizon: int = 1,
    small_sample_correction: bool = True,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Diebold-Mariano (1995) test: whether one of two competing forecasts (e.g.
    two different VaR, Volatility or GARCH models) is more accurate than the other.

    The test compares the loss differential between the two forecasts' errors:

    - e_a_t = actual_t - forecast_a_t,  e_b_t = actual_t - forecast_b_t
    - d_t = L(e_a_t) - L(e_b_t)

    Where `L` is a loss function (squared or absolute error). If both forecasts are
    equally accurate, `d_t` should average out to zero; if `forecast_a` is
    systematically more accurate, `d_t` should be negative on average (and vice
    versa). The test statistic standardizes the mean loss differential by its long-run
    variance, which accounts for the serial correlation that `d_t` has for
    multi-step-ahead forecasts (a horizon above 1), up to lag `horizon - 1`:

    - LRV(d_t) = gamma_0 + 2 * SUM_{k=1}^{horizon - 1} gamma_k
    - DM = mean(d_t) / SQRT(LRV(d_t) / n)

    Where gamma_k is the k-th sample autocovariance of `d_t` (divided by n). These are
    the equal (rectangular) weights of Diebold & Mariano (1995), for which Harvey,
    Leybourne & Newbold (1997) derive their correction. When this estimate is zero or
    negative, which can happen in a small sample, the Newey-West (Bartlett) weights
    1 - k / horizon of the same lags are used instead, as they are never negative.

    Which is asymptotically standard normal under the null of equal forecast
    accuracy, allowing for a two-sided p-value. With the small-sample correction (the
    default), the Harvey, Leybourne & Newbold (1997) finite-sample correction is
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
        loss (str, optional): The loss function to compare forecast errors with. Options
        are: "squared" (squared error), "absolute" (absolute error). Defaults to "squared".
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
    check_option("loss", loss, ("squared", "absolute"))

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
    Calculate a one-period-ahead Variance forecast from the returns up to the prior period
    only, so it can be compared to the realized squared return out-of-sample.

    The forecast is the forecast half of the Diebold-Mariano test in
    `get_diebold_mariano_test`. The methods are:

    - EWMA: σₜ² = λ·σₜ₋₁² + (1 - λ)·rₜ₋₁², the RiskMetrics recursion of
      `risk_model.get_ewma_volatility`, which already uses returns up to t - 1
    - Rolling: the sample variance (ddof=1) of the `window_size` returns up to t - 1
    - A GARCH-family model: the one-step variance of the model with a
      constant mean and normal innovations (quasi-maximum likelihood), refitted every
      `refit_every` returns on the `garch_window_size` returns before it (see
      `garch_model.get_garch_family_rolling_forecast`)

    Args:
        returns (pd.Series | pd.DataFrame): The asset's return series, or one column per asset.
        Each column is forecast on its own non-missing observations.
        method (str): The forecasting method. Options are: "ewma" (RiskMetrics EWMA),
        "rolling" (rolling sample variance), or a GARCH-family model: "garch", "gjr",
        "egarch", "igarch", "aparch", "tgarch", "ngarch", "avgarch", "nagarch",
        "component", "figarch".
        window_size (int): The rolling window size used by the rolling method.
        lambda_ (float): The decay factor used by the EWMA method.
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
    Calculate the Root Mean Squared Error (RMSE): how far forecasts are from the realized
    values, with large misses weighing more.

    Also known as: RMSD (Root Mean Squared Deviation).

    The most common scale-dependent forecast accuracy metric -- the square root of the
    average squared forecast error:

    - RMSE = SQRT(mean((actual_t - forecast_t)^2))

    Squaring the errors before averaging means RMSE penalizes large errors
    disproportionately more than small ones (compare `get_mae`, which weighs every
    error linearly) -- prefer RMSE when large misses are especially costly, and MAE
    when a single big outlier forecast shouldn't dominate the comparison.

    Referenced against:
    - statsmodels, rmse: https://www.statsmodels.org/stable/generated/statsmodels.tools.eval_measures.rmse.html

    Args:
        actual (pd.Series | pd.DataFrame): The realized values.
        forecast (pd.Series | pd.DataFrame): The forecast values, aligned to the same
        index as `actual`.

    Returns:
        float | pd.Series: The RMSE. A single float for `pd.Series` inputs, or a
        `pd.Series` of one RMSE per column for `pd.DataFrame` inputs.

    Raises:
        TypeError: If `actual` is not a `pd.Series` or `pd.DataFrame`.

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
    Calculate the Mean Absolute Error (MAE): how far a typical forecast is from the realized
    value.

    Also known as: MAD (Mean Absolute Deviation).

    A scale-dependent forecast accuracy metric -- the average absolute forecast error:

    - MAE = mean(|actual_t - forecast_t|)

    Unlike `get_rmse`, every error contributes linearly regardless of size, so MAE is
    more robust to (i.e. less dominated by) a handful of large forecast misses -- it
    answers "how far off is a typical forecast", whereas RMSE answers a
    large-error-weighted version of the same question.

    Referenced against:
    - statsmodels, meanabs: https://www.statsmodels.org/stable/generated/statsmodels.tools.eval_measures.meanabs.html

    Args:
        actual (pd.Series | pd.DataFrame): The realized values.
        forecast (pd.Series | pd.DataFrame): The forecast values, aligned to the same
        index as `actual`.

    Returns:
        float | pd.Series: The MAE. A single float for `pd.Series` inputs, or a
        `pd.Series` of one MAE per column for `pd.DataFrame` inputs.

    Raises:
        TypeError: If `actual` is not a `pd.Series` or `pd.DataFrame`.

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
    Calculate the out-of-sample forecast error of a model: its RMSE and MAE on a holdout it
    was not fitted on.

    The series is split into a training and holdout portion, the holdout is forecast with
    `forecast_function` fit only on the training portion, and the result is scored with
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
    Calculate the probability integral transform (PIT) of every return: where it falls in the
    one-period-ahead density forecast built from the returns before it.

    The PITs are the input of density forecast tests such as `get_berkowitz_test`.

    The PIT is the forecast distribution function evaluated at the realized return. If the
    density forecasts are correct, the PITs are independent and uniformly distributed on (0, 1)
    (Diebold, Gunther and Tay 1998):

    - PIT_t = F_t(Return_t), with F_t the forecast distribution of Return_t given the returns before t

    The density forecasts are:

    - Gaussian: a normal distribution with the mean and standard deviation (ddof=1) of the
      `window_size` returns before t
    - EWMA: a normal distribution with mean zero and the RiskMetrics variance
      σₜ² = λ·σₜ₋₁² + (1 - λ)·rₜ₋₁² (see `risk_model.get_ewma_volatility`)
    - GARCH: the chosen GARCH-family model with a constant mean and the chosen innovation
      distribution, refitted every `refit_every` returns on the `window_size` returns
      before t, so F_t(r) = F((r - Mu) / Volatility_t) with F the distribution function of the
      standardized innovations (see `garch_model.get_garch_family_rolling_forecast` and
      `garch_model.get_standardized_cdf`)

    For more information, see the following paper:

    - Diebold, F.X., Gunther, T.A. and Tay, A.S. (1998). "Evaluating Density Forecasts with
    Applications to Financial Risk Management." International Economic Review, 39(4), 863-883.

    Args:
        returns (pd.Series | pd.DataFrame): The returns, one column per asset. Each column is
            forecast on its own non-missing observations.
        model (str, optional): The density forecast. Options are: "gaussian" (rolling normal),
            "ewma" (RiskMetrics EWMA), "garch" (GARCH-family model). Defaults to "garch".
        window_size (int, optional): The number of returns of every window or GARCH fit.
            Defaults to 504.
        lambda_ (float, optional): The decay factor of the EWMA forecast. Defaults to 0.94.
        refit_every (int, optional): The number of returns between two GARCH fits. Defaults
            to 21.
        garch_method (str, optional): The GARCH variance model. Options are: "garch"
            (GARCH(1, 1)), "gjr" (GJR-GARCH(1, 1, 1)), "egarch" (EGARCH(1, 1)), "igarch"
            (IGARCH(1, 1)), "aparch" (APARCH(1, 1)), "tgarch" (TGARCH), "ngarch" (NGARCH),
            "avgarch" (AVGARCH), "nagarch" (NAGARCH), "component" (component GARCH),
            "figarch" (FIGARCH(1, d, 1)). Defaults to "garch".
        garch_distribution (str, optional): The GARCH innovation distribution. Options are:
            "normal" (normal), "t" (Student-t), "skewt" (skewed Student-t of Hansen),
            "skewt_fernandez_steel" (skewed Student-t of Fernández and Steel), "ged"
            (generalized error distribution), "johnson_su" (Johnson SU). Defaults to
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
        raise InvalidArgumentError(
            f"model must be one of 'gaussian', 'ewma' or 'garch', "
            f"not {model!r}. Please pass one of these as model."
        )

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
    Calculate Berkowitz's likelihood ratio test of density forecasts: whether the forecast
    distributions match the returns in mean, variance and independence.

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

    With a tail probability, the tail test of the paper only looks at the forecasts of the tail: every
    z_t above the cut-off q = Φ^-1(alpha) is censored at q, so a model is judged on the size
    and frequency of its tail events only, as a Value at Risk at `alpha` is:

    - ln L(μ, σ) = Σ_(z_t < q) (ln φ((z_t - μ) / σ) - ln σ) + Σ_(z_t >= q) ln(1 - Φ((q - μ) / σ))
    - Statistic = -2 (ln L(0, 1) - ln L(μ_hat, σ_hat)), compared with a χ² distribution with 2 degrees of freedom

    PITs are kept within [1e-10, 1 - 1e-10], so a return far outside its forecast gives a large
    but finite z.

    For more information about the method, see the following paper:

    - Berkowitz, J. (2001). "Testing Density Forecasts, With Applications to Risk Management."
    Journal of Business & Economic Statistics, 19(4), 465-474.

    Referenced against:
    - statsmodels, ARIMA: https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html

    Shares similar logic:
    - rugarch (R), BerkowitzTest: https://github.com/cran/rugarch/blob/1.5-6/R/rugarch-tests.R#L327

    Args:
        pit (pd.Series | pd.DataFrame): The probability integral transforms, one column per
            asset.
        alpha (float | None, optional): The tail probability of the tail test. Defaults to
            None, which tests the whole density.

    Returns:
        pd.Series | pd.DataFrame: The Berkowitz statistic, its p-value, the fitted Mean,
        Standard Deviation and Autocorrelation (NaN for the tail test) of z and the number of
        observations (and of tail observations).

    Notes:
    - Unlike rugarch's BerkowitzTest, which fits the AR(1) by least squares on the demeaned z
      (the likelihood conditional on the first observation, with the residual standard
      deviation of the regression), the test here maximizes the exact likelihood of the
      paper; the tail test is the same.
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


def _quartic_kernel(u: np.ndarray) -> np.ndarray:
    """The quartic kernel k(u) = 15 / 16 (1 - u^2)^2 on [-1, 1]."""
    return np.where(np.abs(u) <= 1, 15 / 16 * (1 - u * u) ** 2, 0.0)


def _quartic_kernel_mass(b: np.ndarray | float) -> np.ndarray | float:
    """∫_(-1)^b k(u) du = 15 / 16 (b - 2 b^3 / 3 + b^5 / 5) + 1 / 2 for b in [-1, 1]."""
    b = np.clip(b, -1.0, 1.0)
    return 15 / 16 * (b - 2 * b**3 / 3 + b**5 / 5) + 0.5


@functools.cache
def _hong_li_constants() -> tuple[float, float, float]:
    """
    ∫ k^2 du, ∫_0^1 ∫_(-1)^b k_b^2(u) du db with k_b = k / ∫_(-1)^b k and V_0 of Hong and Li
    (2005) for the quartic kernel.
    """
    squared = integrate.quad(lambda u: float(_quartic_kernel(u) ** 2), -1, 1)[0]
    boundary = integrate.quad(
        lambda b: integrate.quad(lambda u: float(_quartic_kernel(u) ** 2), -1, b)[0]
        / float(_quartic_kernel_mass(b)) ** 2,
        0,
        1,
    )[0]

    def convolution(u: float) -> float:
        return integrate.quad(
            lambda v: float(_quartic_kernel(u + v) * _quartic_kernel(v)), -1, 1
        )[0]

    variance = 2 * integrate.quad(lambda u: convolution(u) ** 2, -1, 1)[0] ** 2

    return squared, boundary, variance


def _boundary_kernel(x: np.ndarray, y: np.ndarray, bandwidth: float) -> np.ndarray:
    """
    The boundary-modified kernel K_h(x, y) of Hong and Li (2005) for every x (rows) and y
    (columns): h^-1 k((x - y) / h), divided by ∫_(-x/h)^1 k(u) du for x in [0, h) and by
    ∫_(-1)^((1-x)/h) k(u) du for x in (1 - h, 1], so it integrates to 1 over y in [0, 1].
    """
    kernel = _quartic_kernel((x[:, None] - y[None, :]) / bandwidth) / bandwidth
    mass = np.where(
        x < bandwidth,
        _quartic_kernel_mass(x / bandwidth),
        np.where(x > 1 - bandwidth, _quartic_kernel_mass((1 - x) / bandwidth), 1.0),
    )

    return kernel / mass[:, None]


def get_hong_li_p_value(statistic: float, observations: int, lags: int = 1) -> float:
    """
    Calculate the p-value of a Hong-Li statistic: how likely a value at least this large is
    when the density forecasts are correct.

    The statistic is a Q(j) (one lag) or the portmanteau W over several lags, and its
    distribution under the null is simulated on independent uniform PITs.

    Under the null the PITs are independent and uniform, whatever the model, so the
    distribution of the statistics only depends on the number of observations n and of lags
    p. It is far from the standard normal distribution of the asymptotic theory: on
    1,000 observations every Q(j) has a standard deviation of about 1.2 and a skewness of
    about 0.4, and two Q(j) are correlated by about 0.45 because they share the kernel
    estimate of the marginal density of the same PITs, so W has a standard deviation of
    about 1.8 (p = 4); with the standard normal distribution W rejects at 5% about 17% to
    20% of the time. The Q(j) are exchangeable (the same distribution for every lag and the
    same dependence between every pair), so their joint cumulants up to the third, simulated
    for a grid of n with 10 lags (HONG_LI_NULL_CUMULANTS), give those of W for every p:

    - E[W] = √p κ₁
    - Var(W) = κ₂ + (p - 1) κ₁₁
    - κ₃(W) = (κ₃ + 3 (p - 1) κ₂₁ + (p - 1)(p - 2) κ₁₁₁) / √p
    - P-Value = P(X > W), with X the Pearson type III (gamma) distribution with that mean,
      variance and skewness κ₃(W) / Var(W)^1.5

    Where κ₁ is the mean of a Q(j), κ₂ its variance, κ₁₁ the covariance of two, κ₃ the third
    cumulant of one, κ₂₁ the joint cumulant E[(Q(i) - κ₁)² (Q(j) - κ₁)] and κ₁₁₁ that of three
    different lags. The cumulants are interpolated linearly in ln n between the grid points
    and held at the ends of the grid (50 and 5,000 observations): below 50 the
    approximation is rough, and above 5,000 the statistics move slowly towards the standard
    normal distribution, so the cumulants of 5,000 observations give p-values that are on
    the conservative side.

    Args:
        statistic (float): The statistic Q(j) or W.
        observations (int): The number of PITs n.
        lags (int, optional): The number of lags p of W, 1 for a Q(j). Defaults to 1.

    Returns:
        float: The p-value, the probability of a statistic at least as large under the null.
    """
    if not np.isfinite(statistic):
        return np.nan

    log_observations = np.log(
        np.clip(observations, HONG_LI_NULL_GRID[0], HONG_LI_NULL_GRID[-1])
    )
    mean, variance, covariance, third, pair, triple = (
        float(np.interp(log_observations, np.log(HONG_LI_NULL_GRID), cumulant))
        for cumulant in HONG_LI_NULL_CUMULANTS.T
    )
    portmanteau_variance = variance + (lags - 1) * covariance
    portmanteau_third = (
        third + 3 * (lags - 1) * pair + (lags - 1) * (lags - 2) * triple
    ) / np.sqrt(lags)

    return float(
        stats.pearson3.sf(
            statistic,
            portmanteau_third / portmanteau_variance**1.5,
            loc=np.sqrt(lags) * mean,
            scale=np.sqrt(portmanteau_variance),
        )
    )


def get_hong_li_test(
    pit: pd.Series | pd.DataFrame, lags: int = 4
) -> pd.Series | pd.DataFrame:
    """
    Calculate the nonparametric density forecast test of Hong and Li (2005): whether the
    forecast densities have the right shape and capture the dependence of the returns.

    If the density forecasts are correct, the probability integral transforms (PIT, see
    `get_probability_integral_transform`) are independent and uniform on (0, 1), so the joint
    density of every pair (PIT_t, PIT_(t-j)) is 1 on the unit square. The test estimates that
    density with a kernel and measures its distance from 1, lag by lag: it detects a wrong
    shape of the forecast density and dependence the model misses together, where the
    Berkowitz test only looks at the mean, variance and first autocorrelation of the normal
    quantiles.

    The formula is as follows, with U_t the n PITs, k the quartic kernel 15 / 16 (1 - u^2)^2 on
    [-1, 1], K_h its boundary-modified version (see the notes) and h = S_U n^(-1/6) the bandwidth,
    with S_U the standard deviation of U (ddof=1):

    - g_j(x_1, x_2) = Σ_(t=j+1..n) K_h(x_1, U_t) K_h(x_2, U_(t-j)) / (n - j)
    - M(j) = ∫_0^1 ∫_0^1 (g_j(x_1, x_2) - 1)^2 dx_1 dx_2
    - A_h = [(h^-1 - 2) ∫ k^2(u) du + 2 ∫_0^1 ∫_(-1)^b k_b^2(u) du db]^2 - 1, with k_b = k / ∫_(-1)^b k(v) dv
    - V_0 = 2 [∫_(-1)^1 (∫_(-1)^1 k(u + v) k(v) dv)^2 du]^2
    - Q(j) = h ((n - j) M(j) - A_h) / √V_0, standard normal under the null as n grows
    - W = Σ_(j=1..p) Q(j) / √p, the portmanteau statistic over p = `lags` lags, standard normal
      under the null as n grows

    Only large values reject (the upper tail). The standard normal distribution is reached
    slowly, so the p-values come from the distribution of the statistics simulated under the
    null for the same number of observations and lags (see `get_hong_li_p_value`): on
    simulated independent uniform PITs W (p = 4) then rejects at 5% about 4.3% of the time at
    500 observations and 4.4% at 1,000, where the standard normal distribution gives 18% and
    17%.

    For more information about the method, see the following papers:

    - Hong, Y. and Li, H. (2005). "Nonparametric Specification Testing for Continuous-Time
    Models with Applications to Term Structure of Interest Rates." The Review of Financial
    Studies, 18(1), 37-84.
    - Diebold, F.X., Gunther, T.A. and Tay, A.S. (1998). "Evaluating Density Forecasts with
    Applications to Financial Risk Management." International Economic Review, 39(4), 863-883.

    Shares similar logic:
    - rugarch (R), HLTest: https://github.com/cran/rugarch/blob/1.5-6/R/rugarch-tests.R#L607

    Args:
        pit (pd.Series | pd.DataFrame): The probability integral transforms, one column per
            asset.
        lags (int, optional): The number of lags p. Defaults to 4.

    Returns:
        pd.Series | pd.DataFrame: The statistic Q(j) and its p-value for every lag, the
        portmanteau W and its p-value, the bandwidth and the number of observations; NaN
        with fewer than 20 observations or PITs without variation.

    Notes:
        The statistic follows rugarch's HLTest as its documentation defines it. (n - j) M(j)
        has an expected value of about A_h under the null, so A_h enters multiplied by h,
        the scale of the statistic; without it the statistic centres far below zero (around
        -60 on 1,000 independent uniform draws). Simulated independent uniforms give every
        Q(j) a mean of about 0 and a standard deviation of about 1.2 at 500 to 2,000
        observations, and W one of about 1.8 (p = 4), which the simulated null distribution
        of the p-values takes into account. Unlike rugarch's HLTest, which compares the
        statistics with the standard normal distribution, the p-values here come from the
        simulated null distribution. The integral is a Gauss-Legendre rule on panels of a
        fifth of the bandwidth.
    """
    if lags < 1:
        raise ValueError(f"The test needs at least 1 lag, not {lags}.")

    if isinstance(pit, pd.DataFrame):
        return pd.DataFrame(
            {column: get_hong_li_test(pit[column], lags) for column in pit.columns}
        )
    if not isinstance(pit, pd.Series):
        raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")

    values = pit.dropna().to_numpy(dtype=float)
    n = len(values)
    names = [f"Q({lag})" for lag in range(1, lags + 1)]

    nan_result = pd.Series(
        {
            **{
                f"{name} {kind}": np.nan
                for name in names
                for kind in ("Statistic", "P-Value")
            },
            "W Statistic": np.nan,
            "W P-Value": np.nan,
            "Bandwidth": np.nan,
            "Observations": n,
        }
    )

    if n < max(MINIMUM_OBSERVATIONS_FOR_HONG_LI_TEST, lags + 2):
        return nan_result

    bandwidth = float(np.std(values, ddof=1) * n ** (-1 / 6))

    # PITs without variation give a bandwidth of 0 (or one of rounding noise), for which the
    # kernel and the integration panels do not exist.
    if not bandwidth > n * HONG_LI_ROUNDING:
        return nan_result
    squared, boundary, variance = _hong_li_constants()
    centring = ((1 / bandwidth - 2) * squared + 2 * boundary) ** 2 - 1

    # Composite Gauss-Legendre nodes on [0, 1], with panel edges at h and 1 - h where the kernel's normalization changes.  # noqa: E501
    panel = bandwidth / HONG_LI_PANELS_PER_BANDWIDTH
    edges = np.unique(
        np.concatenate(
            [
                np.linspace(0, bandwidth, HONG_LI_PANELS_PER_BANDWIDTH + 1),
                np.linspace(
                    bandwidth,
                    1 - bandwidth,
                    max(int(np.ceil((1 - 2 * bandwidth) / panel)), 1) + 1,
                ),
                np.linspace(1 - bandwidth, 1, HONG_LI_PANELS_PER_BANDWIDTH + 1),
            ]
        )
    )
    unit_nodes, unit_weights = np.polynomial.legendre.leggauss(HONG_LI_NODES_PER_PANEL)
    half = np.diff(edges)[:, None] / 2
    nodes = (half * unit_nodes + (edges[:-1, None] + half)).ravel()
    weights = (half * unit_weights).ravel()

    kernel = _boundary_kernel(nodes, values, bandwidth)
    result = {}
    statistics = []

    for lag, name in zip(range(1, lags + 1), names, strict=True):
        pairs = n - lag
        density = kernel[:, lag:] @ kernel[:, :-lag].T / pairs
        distance = float(weights @ (density - 1) ** 2 @ weights)
        statistic = bandwidth * (pairs * distance - centring) / np.sqrt(variance)
        statistics.append(statistic)
        result[f"{name} Statistic"] = statistic
        result[f"{name} P-Value"] = get_hong_li_p_value(statistic, n, 1)

    portmanteau = float(np.sum(statistics) / np.sqrt(lags))
    result["W Statistic"] = portmanteau
    result["W P-Value"] = get_hong_li_p_value(portmanteau, n, lags)
    result["Bandwidth"] = bandwidth
    result["Observations"] = n

    return pd.Series(result)


def get_return_forecast(
    returns: pd.Series | pd.DataFrame,
    method: str = "momentum",
    window_size: int = 252,
) -> pd.Series | pd.DataFrame:
    """
    Calculate a one-period-ahead return forecast from the returns up to the prior period only.

    The forecast is the forecast half of the directional accuracy tests in
    `get_directional_accuracy_test`. The methods are:

    - Momentum: the mean of the `window_size` returns up to t - 1, whose sign is the time
      series momentum signal of Moskowitz, Ooi and Pedersen (2012)
    - Autoregressive: a + b·rₜ₋₁, with a and b the least-squares AR(1) of the returns on
      their previous return over the `window_size` pairs up to t - 1

    For more information, see the following paper:

    - Moskowitz, T.J., Ooi, Y.H. and Pedersen, L.H. (2012). "Time Series Momentum." Journal of
    Financial Economics, 104(2), 228-250.

    Args:
        returns (pd.Series | pd.DataFrame): The returns, one column per asset. Each column is
            forecast on its own non-missing observations.
        method (str, optional): The forecasting method. Options are: "momentum" (time series
            momentum), "autoregressive" (AR(1)). Defaults to "momentum".
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
        raise InvalidArgumentError(
            f"method must be one of 'momentum' or 'autoregressive', "
            f"not {method!r}. Please pass one of these as method."
        )

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

    Referenced against:
    - rugarch (R), DACTest: https://github.com/cran/rugarch/blob/1.5-6/R/rugarch-tests.R#L384

    Args:
        actual (pd.Series | pd.DataFrame): The realized values, such as returns.
        forecast (pd.Series | pd.DataFrame): The forecasts, aligned to the same index.
        method (str, optional): The test. Options are: "pesaran_timmermann"
            (Pesaran-Timmermann), "anatolyev_gerko" (Anatolyev-Gerko). Defaults to
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
    check_option("method", method, ("pesaran_timmermann", "anatolyev_gerko"))

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
    seed: int = 0,
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
    - Max statistic: T = max_i t_i; range statistic: T = max_(i, j) |t_ij|
    - P-Value = (1 / B) Σ_b 1{T*_b >= T}, with T*_b the statistic of the b-th bootstrap resample, centered at
      the full sample: t*_i = (d*_i - d_i) / √Var(d_i) and t*_ij = (d*_ij - d_ij) / √Var(d_ij)
    - when the p-value is below alpha, the model with the largest t_i (max) or max_j t_ij (range) is
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

    Like arch, the variance of each t_ij of the range statistic is that of the bootstrap
    averages with 1 added to its zero diagonal.

    In samples of a few hundred periods the range statistic eliminates a model too often:
    with four models of equal expected loss over 250 periods (500 resamples) it eliminated at
    least one about 14% of the time at alpha = 0.1 and 8% at alpha = 0.05, and arch's MCS about
    12% and 6%, where the max statistic stays close to alpha (8% and 5%). Prefer the max
    statistic for short samples.

    Referenced against:
    - arch, MCS: https://bashtage.github.io/arch/multiple-comparison/generated/arch.bootstrap.MCS.html
    - rugarch (R), mcsTest: https://github.com/cran/rugarch/blob/1.5-6/R/rugarch-tests.R#L919

    Args:
        losses (pd.DataFrame): The loss of every model (a column) at every period (a row), such
            as squared forecast errors. Periods with a missing loss are left out.
        alpha (float, optional): The size of the test: the set contains the best model with a
            probability of at least 1 - alpha. Defaults to 0.1.
        statistic (str, optional): The elimination statistic. Options are: "max" (the largest
            t_i, arch's "max"), "range" (the largest |t_ij|, arch's "R"). Defaults to "max".
        n_bootstrap (int, optional): The number of bootstrap resamples B. Defaults to 1000.
        block_size (int | None, optional): The average block length of the stationary
            bootstrap. Defaults to None, which uses the cube root of the number of periods,
            rounded.
        seed (int, optional): The seed of the bootstrap. Defaults to 0.

    Returns:
        pd.DataFrame: Per model the Average Loss, the MCS P-Value, whether it is In Confidence
        Set and its Rank (1 for the last model standing, m for the first eliminated).
    """
    check_option("statistic", statistic, ("max", "range"))

    losses = losses.dropna()
    values = losses.to_numpy(dtype=float)
    n, m = values.shape

    if m < MINIMUM_MODELS_FOR_MCS or n < MINIMUM_MODELS_FOR_MCS:
        raise ValueError(
            "The model confidence set needs at least two models and two periods of losses."
        )

    block_size = block_size if block_size else max(int(round(n ** (1 / 3))), 1)
    rng = np.random.default_rng(seed)
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

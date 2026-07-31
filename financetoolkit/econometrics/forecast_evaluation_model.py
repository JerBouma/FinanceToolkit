"""Forecast Evaluation Model"""

__docformat__ = "google"

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tools.eval_measures import meanabs, rmse

MINIMUM_OBSERVATIONS_FOR_DM_TEST = 3


def _newey_west_long_run_variance(residuals: np.ndarray, lags: int) -> float:
    """
    Newey-West (1987) HAC long-run variance estimator with Bartlett kernel weights,
    for an already-demeaned series. See `unitroot_model._newey_west_long_run_variance`
    for the same estimator used to correct the KPSS and Phillips-Perron test
    statistics for serial correlation -- this is the loss-differential analogue,
    needed here because `get_diebold_mariano_test`'s loss differential series is
    itself typically serially correlated for multi-step-ahead (`h > 1`) forecasts.

    - lambda^2 = gamma_0 + 2 * SUM_{l=1}^{lags} (1 - l / (lags + 1)) * gamma_l
    """
    number_of_observations = len(residuals)
    variance = float(np.sum(residuals**2) / number_of_observations)

    for lag in range(1, lags + 1):
        weight = 1 - lag / (lags + 1)
        autocovariance = (
            np.sum(residuals[lag:] * residuals[:-lag]) / number_of_observations
        )
        variance += 2 * weight * autocovariance

    return variance


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
    versa). The test statistic standardizes the mean loss differential by a
    Newey-West long-run variance estimate (which accounts for the serial correlation
    that `d_t` typically has for multi-step-ahead, `horizon > 1`, forecasts):

    - DM = mean(d_t) / SQRT(NeweyWest_LRV(d_t) / n)

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
        horizon (int, optional): The forecast horizon (in periods), used to set the
        Newey-West truncation lag (`horizon - 1`) since an `h`-step-ahead forecast
        error is expected to be autocorrelated up to lag `h - 1`. Defaults to 1
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

        lags = max(horizon - 1, 0)
        long_run_variance = _newey_west_long_run_variance(d - d_bar, lags)

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


def get_rmse(
    actual: pd.Series | pd.DataFrame, forecast: pd.Series | pd.DataFrame
) -> float | pd.Series:
    """
    Calculate the Root Mean Squared Error (RMSE) between realized and forecast values,
    via `statsmodels.tools.eval_measures.rmse`.

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

    As an example:

    ```python
    import pandas as pd
    from financetoolkit.econometrics import forecast_evaluation_model

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
        return float(rmse(aligned["actual"].to_numpy(), aligned["forecast"].to_numpy()))

    raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")


def get_mae(
    actual: pd.Series | pd.DataFrame, forecast: pd.Series | pd.DataFrame
) -> float | pd.Series:
    """
    Calculate the Mean Absolute Error (MAE) between realized and forecast values, via
    `statsmodels.tools.eval_measures.meanabs`.

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

    As an example:

    ```python
    import pandas as pd
    from financetoolkit.econometrics import forecast_evaluation_model

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
        return float(
            meanabs(aligned["actual"].to_numpy(), aligned["forecast"].to_numpy())
        )

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
       compatible signature that returns either an object with a `.forecast`
       attribute, such as `ARIMAResult`/`VARResult`, or a plain `pd.Series`/single-
       column `pd.DataFrame`/array-like of forecast values).
    3. Compare that forecast against the actual holdout values via `get_rmse`/`get_mae`.

    Args:
        series (pd.Series): The full series to validate against, chronologically
        ordered.
        forecast_function (Callable): A function called as
        `forecast_function(train_series, forecast_steps=h, **forecast_kwargs)`,
        returning `h` forecast values (directly, or via a `.forecast` attribute).
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
    from financetoolkit.econometrics import forecast_evaluation_model, time_series_model

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
    forecast_values = getattr(raw_forecast, "forecast", raw_forecast)

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

"""Unit Root Model"""

__docformat__ = "google"

import numpy as np
import pandas as pd

# Asymptotic critical values for the (A)DF tau statistic, by regression type.
# Source: MacKinnon, J.G. (1996). "Numerical Distribution Functions for Unit Root
# and Cointegration Tests." Journal of Applied Econometrics, 11(6), 601-618 (updating
# the original tables in Fuller, W.A. (1976). Introduction to Statistical Time Series).
# These are the standard textbook asymptotic values, not the (more precise but far more
# involved) finite-sample response-surface p-value approximation from the same paper.
ADF_CRITICAL_VALUES: dict[str, dict[float, float]] = {
    "n": {0.01: -2.56, 0.05: -1.94, 0.10: -1.62},
    "c": {0.01: -3.43, 0.05: -2.86, 0.10: -2.57},
    "ct": {0.01: -3.96, 0.05: -3.41, 0.10: -3.13},
}

MINIMUM_USABLE_OBSERVATIONS_BUFFER = 3


def get_augmented_dickey_fuller(
    series: pd.Series, max_lag: int | None = None, regression: str = "c"
) -> pd.Series:
    """
    Calculate the Augmented Dickey-Fuller (ADF) test for a unit root.

    The test regresses the first difference of the series on its own lagged level and
    `p` lags of its own first difference:

    - dy_t = gamma * y_(t-1) + SUM(delta_i * dy_(t-i)) + [constant] + [trend] + e_t

    The null hypothesis is that the series has a unit root (is a random walk, i.e. not
    mean-reverting); the alternative is that it is stationary. The test statistic is
    the t-statistic on `gamma`; unlike a regular t-statistic it does not follow a
    Student-T distribution, so it must be compared against Dickey-Fuller specific
    critical values rather than a standard significance table.

    The number of lags `p` is chosen automatically (up to `max_lag`) by minimizing the
    Akaike Information Criterion (AIC) across candidate lag lengths, unless `max_lag`
    is given explicitly.

    For more information about the method, see the following paper:

    - Dickey, D.A. and Fuller, W.A. (1979). "Distribution of the Estimators for
    Autoregressive Time Series with a Unit Root." Journal of the American Statistical
    Association, 74(366a), 427-431.

    Also known as: ADF test, unit root test, stationarity test.

    Args:
        series (pd.Series): A Series of values (e.g. prices, or a spread between two prices).
        max_lag (int, optional): The maximum number of lagged differences to consider. Defaults to the
        Schwert (1989) rule of thumb, ceil(12 * (n / 100) ** 0.25).
        regression (str, optional): Which deterministic terms to include, one of "n" (none), "c"
        (constant) or "ct" (constant and trend). Defaults to "c".

    Returns:
        pd.Series: The ADF statistic, the number of lags used, the number of observations used, the
        1%/5%/10% critical values, and whether the unit root is rejected at the 5% level.
    """
    if regression not in ["n", "c", "ct"]:
        raise ValueError(
            "regression must be 'n' (no constant), 'c' (constant), or 'ct' (constant and trend)."
        )

    values = series.dropna().to_numpy()
    n = len(values)

    if max_lag is None:
        max_lag = int(np.ceil(12 * (n / 100) ** 0.25))

    delta_y = np.diff(values)

    best_aic = np.inf
    best_result = None

    for lag in range(max_lag + 1):
        usable = len(delta_y) - lag
        minimum_usable = lag + MINIMUM_USABLE_OBSERVATIONS_BUFFER

        if usable <= minimum_usable:
            continue

        y_dependent = delta_y[lag:]
        y_lagged_level = values[lag : lag + usable]

        regressors = [y_lagged_level]
        for i in range(1, lag + 1):
            regressors.append(delta_y[lag - i : lag - i + usable])

        if regression in ("c", "ct"):
            regressors.append(np.ones(usable))
        if regression == "ct":
            regressors.append(np.arange(usable, dtype=float))

        x = np.column_stack(regressors)

        coefficients, _, _, _ = np.linalg.lstsq(x, y_dependent, rcond=None)
        fitted_values = x @ coefficients
        residuals = y_dependent - fitted_values

        number_of_parameters = x.shape[1]
        residual_sum_of_squares = np.sum(residuals**2)
        akaike_information_criterion = usable * np.log(
            residual_sum_of_squares / usable
        ) + 2 * (number_of_parameters + 1)

        if akaike_information_criterion < best_aic:
            sigma_squared = residual_sum_of_squares / (usable - number_of_parameters)
            xtx_inverse = np.linalg.inv(x.T @ x)
            standard_errors = np.sqrt(sigma_squared * np.diag(xtx_inverse))
            tau_statistic = coefficients[0] / standard_errors[0]

            best_aic = akaike_information_criterion
            best_result = (tau_statistic, lag, usable)

    if best_result is None:
        return pd.Series(
            {
                "ADF Statistic": np.nan,
                "Lags Used": np.nan,
                "Observations": np.nan,
                "Critical Value 1%": np.nan,
                "Critical Value 5%": np.nan,
                "Critical Value 10%": np.nan,
                "Reject Unit Root (5%)": False,
            }
        )

    tau_statistic, selected_lag, number_of_observations = best_result
    critical_values = ADF_CRITICAL_VALUES[regression]

    return pd.Series(
        {
            "ADF Statistic": tau_statistic,
            "Lags Used": selected_lag,
            "Observations": number_of_observations,
            "Critical Value 1%": critical_values[0.01],
            "Critical Value 5%": critical_values[0.05],
            "Critical Value 10%": critical_values[0.10],
            "Reject Unit Root (5%)": bool(tau_statistic < critical_values[0.05]),
        }
    )

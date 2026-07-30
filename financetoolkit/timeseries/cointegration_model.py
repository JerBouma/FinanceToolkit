"""Cointegration Model"""

__docformat__ = "google"

import numpy as np
import pandas as pd

from financetoolkit.timeseries import unitroot_model

# Asymptotic critical values for the Engle-Granger residual-based cointegration test,
# for the two-variable case (one dependent series regressed on one independent series
# plus a constant). Source: MacKinnon, J.G. (1991). "Critical Values for Cointegration
# Tests." In Engle, R.F. and Granger, C.W.J. (eds), Long-Run Economic Relationships.
EG_CRITICAL_VALUES: dict[float, float] = {0.01: -3.90, 0.05: -3.34, 0.10: -3.04}


def get_engle_granger_cointegration(
    series_a: pd.Series, series_b: pd.Series, max_lag: int | None = None
) -> pd.Series:
    """
    Calculate the Engle-Granger test for cointegration between two series.

    Two individually non-stationary series (e.g. two stock prices, each following a
    random walk) are cointegrated if some linear combination of them is stationary,
    i.e. they share a long-run equilibrium relationship even though each wanders on
    its own in the short run. This is the classic statistical foundation for
    pairs-trading: if two series are cointegrated, deviations of the spread from its
    equilibrium level tend to revert, making the spread itself tradeable.

    The test proceeds in two steps:

    1. Regress `series_a` on `series_b` (plus a constant): `series_a = alpha + beta * series_b + residuals`.
    2. Run an Augmented Dickey-Fuller test (with no constant, since OLS residuals
       already have zero mean) on the regression residuals.

    The null hypothesis is that the two series are NOT cointegrated (the residuals
    have a unit root). Because the residuals come from an estimated regression rather
    than being observed directly, the test statistic must be compared against
    Engle-Granger specific critical values, which are more negative than plain ADF
    critical values.

    For more information about the method, see the following paper:

    - Engle, R.F. and Granger, C.W.J. (1987). "Co-integration and Error Correction:
    Representation, Estimation, and Testing." Econometrica, 55(2), 251-276.

    Also known as: EG test, residual-based cointegration test.

    Args:
        series_a (pd.Series): The dependent series.
        series_b (pd.Series): The independent series.
        max_lag (int, optional): The maximum number of lagged differences to consider in the
        underlying ADF test on the residuals. Defaults to the Schwert (1989) rule of thumb.

    Returns:
        pd.Series: The Engle-Granger statistic, the number of lags used, the number of
        observations used, the 1%/5%/10% critical values, and whether cointegration is found
        at the 5% level.
    """
    aligned = pd.concat([series_a, series_b], axis=1, join="inner").dropna()
    dependent = aligned.iloc[:, 0].to_numpy()
    independent = aligned.iloc[:, 1].to_numpy()

    design_matrix = np.column_stack([independent, np.ones(len(independent))])
    coefficients, _, _, _ = np.linalg.lstsq(design_matrix, dependent, rcond=None)
    residuals = dependent - design_matrix @ coefficients

    adf_result = unitroot_model.get_augmented_dickey_fuller(
        pd.Series(residuals), max_lag=max_lag, regression="n"
    )

    return pd.Series(
        {
            "Engle-Granger Statistic": adf_result["ADF Statistic"],
            "Lags Used": adf_result["Lags Used"],
            "Observations": adf_result["Observations"],
            "Critical Value 1%": EG_CRITICAL_VALUES[0.01],
            "Critical Value 5%": EG_CRITICAL_VALUES[0.05],
            "Critical Value 10%": EG_CRITICAL_VALUES[0.10],
            "Cointegrated (5%)": bool(
                adf_result["ADF Statistic"] < EG_CRITICAL_VALUES[0.05]
            ),
        }
    )

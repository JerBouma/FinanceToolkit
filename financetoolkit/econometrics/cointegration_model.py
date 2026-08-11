"""Cointegration Model"""

__docformat__ = "google"

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

JOHANSEN_MINIMUM_NUMBER_OF_SERIES = 2

# The conventional 5% significance level used for the "Cointegrated (5%)" flag.
SIGNIFICANCE_LEVEL = 0.05


def get_engle_granger_cointegration(
    series_a: pd.Series, series_b: pd.Series, max_lag: int | None = None
) -> pd.Series:
    """
    Calculate the Engle-Granger test for cointegration between two series, via
    `statsmodels.tsa.stattools.coint`.

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
        underlying ADF test on the residuals. Defaults to `statsmodels`' automatic selection.

    Returns:
        pd.Series: The Engle-Granger statistic, its p-value, the 1%/5%/10% critical values,
        and whether cointegration is found at the 5% level.
    """
    aligned = pd.concat([series_a, series_b], axis=1, join="inner").dropna()
    dependent = aligned.iloc[:, 0].to_numpy()
    independent = aligned.iloc[:, 1].to_numpy()

    try:
        eg_statistic, p_value, critical_values = coint(
            dependent, independent, trend="c", maxlag=max_lag, autolag="aic"
        )
    except (ValueError, np.linalg.LinAlgError):
        return pd.Series(
            {
                "Engle-Granger Statistic": np.nan,
                "P-Value": np.nan,
                "Critical Value 1%": np.nan,
                "Critical Value 5%": np.nan,
                "Critical Value 10%": np.nan,
                "Cointegrated (5%)": False,
            }
        )

    return pd.Series(
        {
            "Engle-Granger Statistic": eg_statistic,
            "P-Value": p_value,
            "Critical Value 1%": critical_values[0],
            "Critical Value 5%": critical_values[1],
            "Critical Value 10%": critical_values[2],
            "Cointegrated (5%)": bool(p_value < SIGNIFICANCE_LEVEL),
        }
    )


def get_johansen_cointegration(
    data: pd.DataFrame, det_order: int = 0, k_ar_diff: int = 1
) -> pd.DataFrame:
    """
    Calculate the Johansen test for cointegration among two or more series, via
    `statsmodels.tsa.vector_ar.vecm.coint_johansen`.

    The Engle-Granger test (see `cointegration_model.get_engle_granger_cointegration`)
    only handles two series and imposes an arbitrary normalization (which series is
    "dependent"). Johansen's test generalizes this to `N >= 2` series at once by
    testing the rank of the long-run coefficient matrix `Pi` in the Vector Error
    Correction Model (VECM) representation of a VAR(p):

    - Delta_Y_t = Pi * Y_(t-1) + SUM(Gamma_i * Delta_Y_(t-i)) + mu + e_t

    `Pi`'s rank `r` equals the number of independent cointegrating relationships among
    the `N` series: `r = 0` means no cointegration (each series wanders independently,
    or their differences do), `r = N` means the whole system is already stationary in
    levels, and `0 < r < N` means `r` stationary linear combinations exist among `N`
    individually non-stationary series.

    Two statistics are computed for every candidate rank `r = 0, ..., N-1`, each
    compared against its own Johansen-specific asymptotic critical values (a standard
    t/F table does not apply):

    - Trace statistic: tests H0: rank <= r against a general alternative.
    - Maximum-eigenvalue statistic: tests H0: rank = r against H1: rank = r + 1.

    Reading down either column and finding the first row that is NOT rejected gives
    the estimated cointegration rank.

    For more information about the method, see the following papers:

    - Johansen, S. (1988). "Statistical Analysis of Cointegration Vectors." Journal of
    Economic Dynamics and Control, 12(2-3), 231-254.
    - Johansen, S. (1991). "Estimation and Hypothesis Testing of Cointegration Vectors
    in Gaussian Vector Autoregressive Models." Econometrica, 59(6), 1551-1580.

    Also known as: Johansen test, Johansen procedure, VECM rank test.

    Args:
        data (pd.DataFrame): A DataFrame with one column per series (e.g. price levels of
        several assets), at least 2 columns.
        det_order (int, optional): Which deterministic term to include: -1 (none), 0 (a
        constant, restricted to lie in the cointegrating relation -- no separate linear
        trend) or 1 (a linear trend restricted to the cointegrating relation, alongside
        an unrestricted constant in the short-run dynamics). These are Johansen's cases
        1, 2 and 4 respectively, matching the critical-value tables `statsmodels`
        returns. Defaults to 0.
        k_ar_diff (int, optional): The number of lagged first differences to include as
        short-run dynamics (equivalent to fitting a levels-VAR of order `k_ar_diff + 1`).
        Defaults to 1.

    Returns:
        pd.DataFrame: One row per candidate rank `r = 0, ..., N-1` (indexed `"r <= 0"`,
        `"r <= 1"`, ...), with the corresponding eigenvalue, trace statistic, max-eigenvalue
        statistic, their 90%/95%/99% critical values, and whether each is rejected at the 5%
        level.

    Raises:
        ValueError: If `data` has fewer than 2 columns, or if `k_ar_diff` is negative.
    """
    if det_order not in (-1, 0, 1):
        raise ValueError(
            "det_order must be one of -1 (none), 0 (constant) or 1 (constant and trend)."
        )

    if k_ar_diff < 0:
        raise ValueError("k_ar_diff must be a non-negative integer.")

    aligned = data.dropna()
    number_of_series = aligned.shape[1]

    if number_of_series < JOHANSEN_MINIMUM_NUMBER_OF_SERIES:
        raise ValueError(
            "data must have at least 2 columns (series) -- Johansen cointegration "
            "requires at least 2 series."
        )

    nan_row = {
        "Eigenvalue": np.nan,
        "Trace Statistic": np.nan,
        "Trace Critical Value 90%": np.nan,
        "Trace Critical Value 95%": np.nan,
        "Trace Critical Value 99%": np.nan,
        "Reject (Trace, 5%)": False,
        "Max-Eigenvalue Statistic": np.nan,
        "Max-Eig Critical Value 90%": np.nan,
        "Max-Eig Critical Value 95%": np.nan,
        "Max-Eig Critical Value 99%": np.nan,
        "Reject (Max-Eig, 5%)": False,
    }
    index = [f"r <= {rank}" for rank in range(number_of_series)]

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = coint_johansen(aligned.to_numpy(), det_order, k_ar_diff)
    except (ValueError, np.linalg.LinAlgError):
        return pd.DataFrame([nan_row] * number_of_series, index=index)

    eigenvalues = np.real(result.eig)
    trace_statistics = np.real(result.lr1)
    max_eigenvalue_statistics = np.real(result.lr2)

    rows = []
    for rank in range(number_of_series):
        rows.append(
            {
                "Eigenvalue": eigenvalues[rank],
                "Trace Statistic": trace_statistics[rank],
                "Trace Critical Value 90%": result.cvt[rank, 0],
                "Trace Critical Value 95%": result.cvt[rank, 1],
                "Trace Critical Value 99%": result.cvt[rank, 2],
                "Reject (Trace, 5%)": bool(
                    trace_statistics[rank] > result.cvt[rank, 1]
                ),
                "Max-Eigenvalue Statistic": max_eigenvalue_statistics[rank],
                "Max-Eig Critical Value 90%": result.cvm[rank, 0],
                "Max-Eig Critical Value 95%": result.cvm[rank, 1],
                "Max-Eig Critical Value 99%": result.cvm[rank, 2],
                "Reject (Max-Eig, 5%)": bool(
                    max_eigenvalue_statistics[rank] > result.cvm[rank, 1]
                ),
            }
        )

    return pd.DataFrame(rows, index=index)
